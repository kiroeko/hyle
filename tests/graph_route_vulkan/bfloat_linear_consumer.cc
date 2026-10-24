// Actual Vulkan BF16 constant orientation, projection precision and replay.
// Expected products are independent FP64 dots of the original BF16 payload.
#include <hyle/compute/model_ops.h>
#include <hyle/compute/vulkan_memory.h>
#include <hyle/graph/executable.h>
#include <hyle/graph/result.h>
#include <hyle/graph/vulkan.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <utility>
#include <vector>

namespace {
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace g = hyle::graph;
using hyle::Result;
using hyle::Status;
constexpr std::size_t columns = 128, depth = 256;
constexpr auto bf16 = c::DataType::kBFloat16;
constexpr auto f32 = c::DataType::kFloat32;

Status require(bool condition, const char *message) {
  return condition ? Status::Ok() : Status::Internal(message);
}
double decode(std::uint16_t value) {
  return std::bit_cast<float>(std::uint32_t(value) << 16);
}
// Search adjacent representable BF16 values and select nearest/even. This
// finite-fixture oracle does not call product conversion or matrix kernels.
std::uint16_t encode(double value) {
  const auto sign = std::signbit(value) ? std::uint16_t{0x8000} : std::uint16_t{0};
  const auto magnitude = std::abs(value);
  std::uint16_t lower = 0, upper = 0x7F7F;
  while (lower < upper) {
    const auto middle = static_cast<std::uint16_t>((unsigned(lower) + unsigned(upper) + 1u) / 2u);
    if (decode(middle) <= magnitude) lower = middle;
    else upper = static_cast<std::uint16_t>(middle - 1u);
  }
  if (lower < 0x7F7F) {
    const double below = magnitude - decode(lower), above = decode(static_cast<std::uint16_t>(lower + 1u)) - magnitude;
    if (above < below || (above == below && (lower & 1u))) ++lower;
  }
  return static_cast<std::uint16_t>(sign | lower);
}
Result<c::ReadLease> upload(const v::Context &context, const v::MemoryResource &device,
                           const c::TensorDesc &description,
                           std::span<const std::uint16_t> values) {
  HYLE_ASSIGN_OR_RETURN(auto tensor, context.upload(description, std::as_bytes(values)));
  return device.import_readonly(tensor);
}
template<class T>
Result<std::vector<T>> read(const v::Context &context, const v::MemoryResource &device,
                            const c::ReadLease &lease, const c::TensorDesc &description) {
  HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto storage, lease.view());
  HYLE_RETURN_IF_ERROR(require(storage.resource() == device.ref() && lease.size() == bytes && bytes % sizeof(T) == 0,
      "linear output lost its exact device resource or logical byte extent"));
  HYLE_ASSIGN_OR_RETURN(auto tensor, device.binding(lease, description));
  std::vector<T> result(bytes / sizeof(T));
  HYLE_RETURN_IF_ERROR(context.download(tensor, std::as_writable_bytes(std::span(result))));
  return result;
}
Status readonly(c::ReadLease &lease) {
  const auto generation = lease.generation();
  auto writer = lease.try_write();
  return require(!writer && lease.valid() && lease.generation() == generation,
      "immutable BF16 weight/input or generated output acquired writer permission");
}
std::vector<std::uint16_t> weights() {
  std::vector<std::uint16_t> result(columns * depth);
  for (std::size_t column = 4; column < columns; ++column)
    for (std::size_t k = 2; k + 2 < depth; ++k)
      result[column * depth + k] = encode(.23 * std::sin(double(column * 17 + k * 3 + 1) * .11));
  // Two sparse near-tie columns must retain their small F32 score difference.
  result[0] = encode(1.0);
  result[depth] = encode(1.0); result[depth + 1] = encode(1.0);
  // Exact BF16 products outside F16's finite/normal range reject a silent
  // BF16-to-F16 approximation. Other columns do not read these coordinates.
  result[2 * depth + depth - 1] = encode(std::ldexp(1.0, 30));
  result[3 * depth + depth - 2] = encode(std::ldexp(1.0, -30));
  return result;
}
std::vector<std::uint16_t> inputs(std::size_t rows, std::size_t phase) {
  std::vector<std::uint16_t> result(rows * depth);
  const double fraction = phase ? .0029296875 : .001953125;
  for (std::size_t row = 0; row < rows; ++row) {
    result[row * depth] = encode(1.0); result[row * depth + 1] = encode(fraction);
    for (std::size_t k = 2; k + 2 < depth; ++k)
      result[row * depth + k] = encode(.19 * std::cos(double(row * 13 + k * 5 + phase * 7 + 3) * .07));
    result[row * depth + depth - 1] = encode(std::ldexp(1.0, -30));
    result[row * depth + depth - 2] = encode(std::ldexp(1.0, 30));
  }
  return result;
}
std::vector<double> dot(std::span<const std::uint16_t> activation,
                        std::span<const std::uint16_t> stored, std::size_t rows) {
  std::vector<double> result(rows * columns);
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t column = 0; column < columns; ++column) {
      double sum = 0;
      for (std::size_t k = 0; k < depth; ++k)
        sum += decode(activation[row * depth + k]) * decode(stored[column * depth + k]);
      result[row * columns + column] = sum;
    }
  return result;
}
Status verify(std::span<const std::uint16_t> ordinary,
              std::span<const std::uint16_t> dense,
              std::span<const float> scores, std::span<const double> reference,
              std::size_t rows, std::size_t phase) {
  HYLE_RETURN_IF_ERROR(require(ordinary.size() == reference.size() && dense.size() == reference.size() &&
      scores.size() == reference.size(), "linear output element count differs from the mathematical product"));
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double rounded = decode(encode(reference[i]));
    const double ordinary_value = decode(ordinary[i]), dense_value = decode(dense[i]);
    const double low_bound = .008 * std::abs(rounded) + 2e-6;
    const double score_bound = 6e-6 * std::abs(reference[i]) + 2e-6;
    HYLE_RETURN_IF_ERROR(require(std::isfinite(ordinary_value) && std::isfinite(dense_value) && std::isfinite(scores[i]) &&
        std::abs(ordinary_value - rounded) <= low_bound && std::abs(dense_value - rounded) <= low_bound &&
        std::abs(double(scores[i]) - reference[i]) <= score_bound,
        "BF16 product/rounding or F32 score differs from independent decoded-BF16 FP64 dot"));
    HYLE_RETURN_IF_ERROR(require(std::abs(ordinary_value - dense_value) <= 2 * low_bound,
        "same-dtype dense constant packing differs from the natural permuted constant product"));
  }
  const double fraction = phase ? .0029296875 : .001953125;
  for (std::size_t row = 0; row < rows; ++row) {
    const auto offset = row * columns;
    HYLE_RETURN_IF_ERROR(require(scores[offset] == 1.F && double(scores[offset + 1]) == 1.0 + fraction &&
        scores[offset + 1] > scores[offset] && decode(ordinary[offset]) == 1.0 && decode(ordinary[offset + 1]) == 1.0,
        "F32 near-tie scores were rounded through the ordinary BF16 output contract"));
    HYLE_RETURN_IF_ERROR(require(scores[offset + 2] == 1.F && scores[offset + 3] == 1.F &&
        decode(ordinary[offset + 2]) == 1.0 && decode(ordinary[offset + 3]) == 1.0,
        "BF16 matrix route narrowed its operand exponent range"));
  }
  return Status::Ok();
}

Status run() {
  HYLE_ASSIGN_OR_RETURN(auto context, v::Context::create());
  HYLE_RETURN_IF_ERROR(require(!context.environment().software_device, "BF16 linear consumer requires an actual Vulkan device"));
  HYLE_ASSIGN_OR_RETURN(auto device, v::MemoryResource::create(context, v::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto source, g::vulkan::execution_source(context));
  HYLE_ASSIGN_OR_RETURN(auto stored_description, c::TensorDesc::dense(bf16,
      std::array<std::int64_t, 2>{columns, depth}));
  HYLE_ASSIGN_OR_RETURN(auto dense_description, c::TensorDesc::dense(bf16,
      std::array<std::int64_t, 2>{depth, columns}));
  const auto stored_payload = weights();
  std::vector<std::uint16_t> dense_payload(stored_payload.size());
  for (std::size_t column = 0; column < columns; ++column)
    for (std::size_t k = 0; k < depth; ++k)
      dense_payload[k * columns + column] = stored_payload[column * depth + k];
  HYLE_ASSIGN_OR_RETURN(auto stored_owner, upload(context, device, stored_description, stored_payload));
  HYLE_ASSIGN_OR_RETURN(auto dense_owner, upload(context, device, dense_description, dense_payload));
  HYLE_RETURN_IF_ERROR(readonly(stored_owner)); HYLE_RETURN_IF_ERROR(readonly(dense_owner));
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto rows_dimension, builder.dimension({1, 15, 1}));
  HYLE_ASSIGN_OR_RETURN(auto input_spec, g::TensorSpec::dense(bf16,
      std::array{g::Dimension(rows_dimension), g::Dimension::fixed(depth)}));
  HYLE_ASSIGN_OR_RETURN(auto ordinary_spec, g::TensorSpec::dense(bf16,
      std::array{g::Dimension(rows_dimension), g::Dimension::fixed(columns)}));
  HYLE_ASSIGN_OR_RETURN(auto score_spec, g::TensorSpec::dense(f32,
      std::array{g::Dimension(rows_dimension), g::Dimension::fixed(columns)}));
  HYLE_ASSIGN_OR_RETURN(auto input, builder.input(input_spec));
  HYLE_ASSIGN_OR_RETURN(auto stored, builder.constant(stored_description));
  HYLE_ASSIGN_OR_RETURN(auto dense_stored, builder.constant(dense_description));
  HYLE_ASSIGN_OR_RETURN(auto permutation, c::View::permute(std::array<std::int32_t, 2>{1, 0}));
  HYLE_ASSIGN_OR_RETURN(auto logical_weight, builder.add(permutation, stored));
  const c::MatMul matrix{.accumulation_type = f32};
  HYLE_ASSIGN_OR_RETURN(auto ordinary_node, builder.add_node(c::Operation(matrix),
      std::array{input, logical_weight}, std::array{ordinary_spec}));
  HYLE_ASSIGN_OR_RETURN(auto ordinary, builder.result(ordinary_node));
  HYLE_ASSIGN_OR_RETURN(auto dense_node, builder.add_node(c::Operation(matrix),
      std::array{input, dense_stored}, std::array{ordinary_spec}));
  HYLE_ASSIGN_OR_RETURN(auto dense_ordinary, builder.result(dense_node));
  // This existing precision contract takes the original [N,K] constant;
  // ordinary MatMul above takes the logical [K,N] view and returns BF16.
  HYLE_ASSIGN_OR_RETURN(auto score_node, builder.add_node(c::model_ops::dense_linear_f32(),
      std::array{input, stored}, std::array{score_spec}));
  HYLE_ASSIGN_OR_RETURN(auto score, builder.result(score_node));
  for (const auto output : {ordinary, dense_ordinary, score}) HYLE_RETURN_IF_ERROR(builder.publish(output));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  HYLE_ASSIGN_OR_RETURN(auto stored_view, stored_owner.tensor(stored_description));
  HYLE_ASSIGN_OR_RETURN(auto dense_view, dense_owner.tensor(dense_description));
  const std::array constants{g::ConstantBinding{stored, stored_view, g::ConstantPolicy::kBorrow},
      g::ConstantBinding{dense_stored, dense_view, g::ConstantPolicy::kBorrow}};
  const std::array one{g::ShapeBinding{rows_dimension, 1}}, fifteen{g::ShapeBinding{rows_dimension, 15}};
  const std::array shapes{one, fifteen};
  const std::array variants{g::ExactShapeVariant{one}, g::ExactShapeVariant{fifteen}};
  const std::array domains{g::StorageDomain{device.ref()}};
  const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}}};
  g::CompileOptions options;
  options.shape_policy = g::CompileShapePolicy::kFiniteExact;
  options.exact_variants = variants; options.constants = constants;
  HYLE_ASSIGN_OR_RETURN(auto compiled, g::compile(graph, {domains, targets}, options));
  HYLE_RETURN_IF_ERROR(require(compiled.executable.exact_variant_count() == 2,
      "linear finite family lost its T1/T15 domain"));
  std::size_t products_checked = 0;
  for (std::size_t branch = 0; branch < shapes.size(); ++branch) {
    const std::size_t rows = branch ? 15 : 1;
    HYLE_ASSIGN_OR_RETURN(auto input_description, c::TensorDesc::dense(bf16,
        std::array<std::int64_t, 2>{static_cast<std::int64_t>(rows), depth}));
    HYLE_ASSIGN_OR_RETURN(auto ordinary_description, c::TensorDesc::dense(bf16,
        std::array<std::int64_t, 2>{static_cast<std::int64_t>(rows), columns}));
    HYLE_ASSIGN_OR_RETURN(auto score_description, c::TensorDesc::dense(f32,
        std::array<std::int64_t, 2>{static_cast<std::int64_t>(rows), columns}));
    HYLE_ASSIGN_OR_RETURN(auto invocation,
        compiled.executable.create_invocation({.shape = shapes[branch], .resources = g::ResourceOptions{}}));
    for (std::size_t phase = 0; phase < 2; ++phase) {
      const auto payload = inputs(rows, phase);
      const auto reference = dot(payload, stored_payload, rows);
      HYLE_ASSIGN_OR_RETURN(auto owner, upload(context, device, input_description, payload));
      HYLE_RETURN_IF_ERROR(readonly(owner));
      HYLE_RETURN_IF_ERROR(invocation.bind(input, input_description, owner));
      HYLE_RETURN_IF_ERROR(invocation.prepare_replay({}));
      {
        HYLE_ASSIGN_OR_RETURN(auto completion, invocation.submit(std::span<const g::QueueBinding>{}));
        HYLE_RETURN_IF_ERROR(completion.wait());
        HYLE_ASSIGN_OR_RETURN(auto ordinary_result, invocation.result(ordinary));
        HYLE_ASSIGN_OR_RETURN(auto dense_result, invocation.result(dense_ordinary));
        HYLE_ASSIGN_OR_RETURN(auto score_result, invocation.result(score));
        HYLE_ASSIGN_OR_RETURN(auto ordinary_desc, ordinary_result.description());
        HYLE_ASSIGN_OR_RETURN(auto dense_desc, dense_result.description());
        HYLE_ASSIGN_OR_RETURN(auto score_desc, score_result.description());
        HYLE_RETURN_IF_ERROR(require(ordinary_desc == ordinary_description && dense_desc == ordinary_description &&
            score_desc == score_description, "MatMul storage dtype or F32 score output contract changed"));
        HYLE_ASSIGN_OR_RETURN(auto ordinary_lease, ordinary_result.storage_lease());
        HYLE_ASSIGN_OR_RETURN(auto dense_lease, dense_result.storage_lease());
        HYLE_ASSIGN_OR_RETURN(auto score_lease, score_result.storage_lease());
        HYLE_RETURN_IF_ERROR(readonly(ordinary_lease)); HYLE_RETURN_IF_ERROR(readonly(dense_lease));
        HYLE_RETURN_IF_ERROR(readonly(score_lease));
        HYLE_ASSIGN_OR_RETURN(auto ordinary_values, read<std::uint16_t>(context, device, ordinary_lease, ordinary_description));
        HYLE_ASSIGN_OR_RETURN(auto dense_values, read<std::uint16_t>(context, device, dense_lease, ordinary_description));
        HYLE_ASSIGN_OR_RETURN(auto score_values, read<float>(context, device, score_lease, score_description));
        HYLE_RETURN_IF_ERROR(verify(ordinary_values, dense_values, score_values, reference, rows, phase));
        products_checked += reference.size() * 3;
      }
      HYLE_RETURN_IF_ERROR(invocation.unbind(input));
      HYLE_ASSIGN_OR_RETURN(auto unchanged_input, read<std::uint16_t>(context, device, owner, input_description));
      HYLE_ASSIGN_OR_RETURN(auto unchanged_stored, read<std::uint16_t>(context, device, stored_owner, stored_description));
      HYLE_ASSIGN_OR_RETURN(auto unchanged_dense, read<std::uint16_t>(context, device, dense_owner, dense_description));
      HYLE_RETURN_IF_ERROR(require(unchanged_input == payload && unchanged_stored == stored_payload && unchanged_dense == dense_payload,
          "linear execution mutated original BF16 activation or immutable constant payload"));
      HYLE_RETURN_IF_ERROR(readonly(stored_owner)); HYLE_RETURN_IF_ERROR(readonly(dense_owner));
    }
  }
  std::printf("PASS Vulkan BF16 linear: stored[N,K] permute and dense[K,N] constants, T1/T15 two-input replay, BF16/F32 outputs, independent FP64 dots, near-tie precision and immutable bytes (%zu checked output elements); device=%s\n",
      products_checked, context.environment().device_name.c_str());
  return Status::Ok();
}
} // namespace

int main() {
  const auto status = run();
  if (status.ok()) return 0;
  std::fprintf(stderr, "Vulkan BF16 linear consumer: %.*s\n", int(status.message().size()), status.message().data());
  return 1;
}
