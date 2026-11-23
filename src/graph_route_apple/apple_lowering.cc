#include "hyle/compute/detail/apple_plan.h"
#include "../compute_route_apple/apple_bfloat_projection.h"
#include "../compute_route_apple/apple_carrier.h"
#include "../compute_route_apple/apple_sequence.h"
#include "../compute_route_apple/apple_static_fp8.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/workload_semantics.h"
#include "hyle/compute/model_ops.h"
#include "hyle/graph/apple.h"
#include "apple_lowering_internal.h"
#include "native_region.h"
#include "../compute_route_apple/apple_runtime_internal.h"
#include "../compute_route_apple/apple_tensor.h"
#include "../graph/native_analysis.h"
#include "../graph/artifact_packed_region.h"
#include "../graph/static_fp8_region.h"

#include "hyle/obs/trace.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <mlx/ops.h>
#include <mlx/allocator.h>

namespace hyle::graph::apple {
namespace mx = mlx::core;
namespace ca = compute::apple;

namespace {
struct Node {
  std::uint32_t original_ordinal = 0;
  compute::Operation operation;
  std::vector<std::size_t> inputs;
  std::vector<std::size_t> outputs;
  bool control_boundary = false;
  bool preserve_outputs = false;
  ca::detail::RegionKind carrier = ca::detail::RegionKind::kMlx;
  bool indexed_bfloat_projection = false;
  std::shared_ptr<const ca::detail::PreparedBFloatProjection> packed_projection;
};

struct RowCandidate final {
  std::uint32_t node = 0;
  std::size_t bank = SIZE_MAX;
  ca::detail::RowCacheAccessFacts expected;
  bool traced = false, matched = true;
};
using RowCandidates = std::vector<RowCandidate>;

struct Plan {
  compute::WorkloadContext workload;
  std::vector<Node> nodes;
  std::vector<std::size_t> inputs;
  std::vector<std::size_t> constants;
  std::vector<std::size_t> outputs;
  std::vector<compute::TensorDesc> descriptions;
  std::vector<mx::array> values;
  // One table for ready constant-derived F32 operands and original eligible
  // Cast outputs. Constant SSA slots deduplicate the same source; Cast output
  // slots only alias that array. Logical input arrays remain in values.
  std::vector<std::optional<mx::array>> prepared_casts;
  detail::PreparedConstantStorage prepared_storage;
  CustomLowering custom;
  mutable std::atomic<bool> custom_failed{false};
  bool contiguous_outputs = false;
  // Only during the synchronous cold compile. Its scope clears this pointer
  // before returning a Program; managed replay/retrace cannot mutate proofs.
  RowCandidates *row_candidates = nullptr;
};

struct ColdRowAccess final {
  std::shared_ptr<Plan> plan;
  RowCandidates candidates;
  ~ColdRowAccess() { plan->row_candidates = nullptr; }
};

Result<ca::InputSpec> native_spec(const compute::TensorDesc &description) {
  return ca::input_spec(description);
}

Status match(const compute::TensorDesc &description, const mx::array &value) {
  HYLE_ASSIGN_OR_RETURN(auto spec, native_spec(description));
  if (spec.shape != value.shape() || spec.dtype != value.dtype())
    return Status::InvalidArgument(
        "Apple Graph native array shape/dtype disagrees with schema");
  return Status::Ok();
}

const ConstantBinding *find_constant(const RegionCompileRequest &boundary, std::size_t ordinal) {
  const auto found = std::ranges::find_if(boundary.constants,
      [&](const ConstantBinding &value) { return value.value.ordinal() == ordinal; });
  return found == boundary.constants.end() ? nullptr : &*found;
}

// A canonical constant may already be a boundary view, or the reshape may
// still be inside this region. Both keep the exact original immutable bytes.
const ConstantBinding *find_reshaped_constant(const RegionCompileRequest &boundary,
    const Plan &plan, std::size_t ordinal, const compute::TensorDesc &canonical) {
  if (const auto *binding = find_constant(boundary, ordinal);
      binding && (binding->tensor.description() == plan.descriptions[ordinal] ||
                  binding->tensor.description() == canonical))
    return binding;
  for (const auto &prior : plan.nodes) {
    const auto *view = prior.operation.get_if<compute::View>();
    if (view && view->kind() == compute::ViewKind::kReshape &&
        prior.inputs.size() == 1 && prior.outputs.size() == 1 && prior.outputs[0] == ordinal &&
        plan.descriptions[prior.inputs[0]] == canonical)
      return find_constant(boundary, prior.inputs[0]);
  }
  return nullptr;
}

Status prepare_indexed_bfloat_projection(const std::shared_ptr<ca::detail::Runtime> &runtime,
    const RegionCompileRequest &boundary, const Plan &plan, Node &node,
    std::span<const compute::TensorDesc> inputs) {
  using compute::DataType;
  using compute::TensorDesc;
  if (inputs[0].rank() != 3 || inputs[1].rank() != 3 || inputs[2].rank() != 2)
    return Status::Ok();
  const auto weights = inputs[0].dimensions(), activation = inputs[1].dimensions(),
             indices = inputs[2].dimensions();
  if (!inputs[0].is_plain() || !inputs[0].is_dense() || inputs[0].data_type() != DataType::kBFloat16 ||
      !inputs[1].is_dense() || !inputs[1].is_plain() || inputs[1].data_type() != DataType::kFloat32 ||
      !inputs[2].is_dense() || !inputs[2].is_plain() || inputs[2].data_type() != DataType::kInt32 ||
      weights[1] != 1 || activation[0] != 1 || activation[1] != 1 || activation[2] != weights[2] ||
      indices[0] != 1 || !ca::supports_indexed_bfloat_projection(weights[0], weights[2], indices[1]))
    return Status::Ok();

  HYLE_ASSIGN_OR_RETURN(auto weight_description,
      TensorDesc::dense(DataType::kBFloat16, std::array{weights[0], weights[2]}));
  HYLE_ASSIGN_OR_RETURN(auto index_description, TensorDesc::dense(DataType::kInt32, std::array{indices[1]}));
  const auto *weight = find_reshaped_constant(boundary, plan, node.inputs[0], weight_description);
  const auto *ids = find_reshaped_constant(boundary, plan, node.inputs[2], index_description);
  if (!weight || !weight->owner || !weight->lease || !ids || !ids->owner || !ids->lease)
    return Status::Ok();

  // This is a cold read of the immutable row map. Regular GroupedMatMul IDs
  // (e.g. expert choices produced at runtime) keep their checked provider.
  std::vector<std::int32_t> mapping(static_cast<std::size_t>(indices[1]));
  const auto bytes = std::as_writable_bytes(std::span(mapping));
  HYLE_ASSIGN_OR_RETURN(auto source, ids->tensor.storage().subview(0, bytes.size()));
  HYLE_RETURN_IF_ERROR(source.read(bytes));
  if (std::ranges::any_of(mapping, [&](auto id) { return id < 0 || id >= weights[0]; }))
    return Status::OutOfRange("Apple indexed BF16 projection row is outside the immutable bank");

  HYLE_ASSIGN_OR_RETURN(auto original, compute::ConstTensorView::create(weight_description, weight->tensor.storage()));
  HYLE_ASSIGN_OR_RETURN(node.packed_projection, ca::detail::prepare_bfloat_projection(
      runtime, original, weight->owner, weight->lease, boundary.resource_budget));
  node.indexed_bfloat_projection = true;
  node.carrier = ca::detail::RegionKind::kMlx;
  return Status::Ok();
}

Status prepare_packed_projection(const std::shared_ptr<ca::detail::Runtime> &runtime,
    const RegionCompileRequest &boundary, const Plan &plan, Node &node,
    std::span<const compute::TensorDesc> inputs) {
  if (inputs.size() == 3 && node.operation.custom_as<compute::model_ops::GroupedMatMul>())
    return prepare_indexed_bfloat_projection(runtime, boundary, plan, node, inputs);
  if (inputs.size() != 2 || node.outputs.size() != 1) return Status::Ok();
  const bool scores = node.carrier == ca::detail::RegionKind::kGgml &&
      ca::detail::packed_bfloat_projection_eligible(node.operation, inputs[0], inputs[1]);
  const bool matmul = boundary.workload.region == compute::workload_semantics::kTargetTextForward &&
      node.carrier == ca::detail::RegionKind::kMlx &&
      plan.descriptions[node.outputs[0]].data_type() == compute::DataType::kBFloat16 &&
      ca::detail::packed_bfloat_matmul_eligible(node.operation, inputs[0], inputs[1]);
  if (!scores && !matmul) return Status::Ok();
  auto description = inputs[1];
  if (matmul) {
    HYLE_ASSIGN_OR_RETURN(description, inputs[1].transposed());
    if (!description.is_dense()) return Status::Ok();
  }
  const auto ordinal = node.inputs[1];
  auto weight = find_constant(boundary, ordinal);
  // A transpose can be an incoming constant alias or a view within the native
  // region. Both refer to the same original contiguous immutable rows.
  if (!weight && matmul) {
    for (const auto &prior : plan.nodes) {
      const auto *view = prior.operation.get_if<compute::View>();
      constexpr std::array<std::int8_t, 2> transpose{1, 0};
      if (!view || view->kind() != compute::ViewKind::kPermute ||
          !std::ranges::equal(view->permutation(), transpose) ||
          prior.inputs.size() != 1 || prior.outputs.size() != 1 ||
          prior.outputs[0] != ordinal || plan.descriptions[prior.inputs[0]] != description)
        continue;
      weight = find_constant(boundary, prior.inputs[0]);
      break;
    }
  }
  if (!weight || (weight->tensor.description() != description && weight->tensor.description() != inputs[1]))
    return Status::Ok();
  HYLE_ASSIGN_OR_RETURN(auto original, compute::ConstTensorView::create(description, weight->tensor.storage()));
  HYLE_ASSIGN_OR_RETURN(node.packed_projection, ca::detail::prepare_bfloat_projection(
      runtime, original, weight->owner, weight->lease, boundary.resource_budget));
  if (node.packed_projection) node.carrier = ca::detail::RegionKind::kMlx;
  return Status::Ok();
}

// A new readonly descriptor retains the actual native backing and its charge.
// It never owns/frees that native buffer a second time. Escaped aliases keep
// the charge even after the original Plan/Program has been destroyed.
struct PreparedConstantOwner final {
  compute::ResourceBudget::Reservation charge, private_metadata;
  mx::array source;
};

Status prepare_constant_casts(const std::shared_ptr<ca::detail::Runtime> &runtime,
                              const Graph &graph, Plan &plan,
                              const RegionCompileRequest *boundary) {
  const auto prepare = [&](std::size_t source) -> Status {
    if (!plan.prepared_casts.empty() && plan.prepared_casts[source])
      return Status::Ok();
    const auto position = std::ranges::find(plan.constants, source);
    if (position == plan.constants.end()) return Status::Ok();
    const auto &from = plan.descriptions[source];
    if (!from.is_plain() || !from.is_dense() ||
        (from.data_type() != compute::DataType::kFloat16 &&
         from.data_type() != compute::DataType::kBFloat16))
      return Status::Ok();
    HYLE_ASSIGN_OR_RETURN(auto elements, from.element_count());
    if (elements > (SIZE_MAX - plan.prepared_storage.bytes) / sizeof(float))
      return Status::ResourceExhausted("Apple prepared constant byte count overflow");
    const auto bytes = static_cast<std::size_t>(elements) * sizeof(float);
    compute::ResourceBudget::Reservation charge, private_metadata, scratch;
    mem::MemoryResourceRef charged_resource;
    if (boundary && boundary->resource_budget.valid()) {
      const auto binding = std::ranges::find_if(boundary->constants,
          [&](const ConstantBinding &value) { return value.value.ordinal() == source; });
      if (binding == boundary->constants.end())
        return Status::Internal("Apple prepared constant has no original resource binding");
      charged_resource = binding->tensor.storage().resource();
      HYLE_ASSIGN_OR_RETURN(auto source_bytes, from.required_bytes());
      // The isolated warmup output and final derived output do not overlap:
      // ContextAccess::compile drains/destroys its local outputs first. Keep
      // one F32 payload reservation throughout; only its extra input is scratch.
      HYLE_ASSIGN_OR_RETURN(charge, boundary->resource_budget.reserve(
          charged_resource, bytes, compute::ResourceBudgetCategory::kConstant));
      HYLE_ASSIGN_OR_RETURN(scratch, boundary->resource_budget.reserve(
          charged_resource, source_bytes,
          compute::ResourceBudgetCategory::kWorkspace));
      HYLE_ASSIGN_OR_RETURN(private_metadata,
          boundary->resource_budget.reserve_unattributed(std::nullopt));
    }
    const std::array examples{plan.values[static_cast<std::size_t>(position - plan.constants.begin())]};
    const std::array specs{ca::InputSpec{examples[0].shape(), examples[0].dtype()}};
    HYLE_ASSIGN_OR_RETURN(auto cast_program,
        ca::detail::ContextAccess::compile(runtime,
            [workload = plan.workload](const auto &values, mx::Stream stream) {
              return ca::lower(compute::Operation{compute::Cast{compute::DataType::kFloat32}},
                               values, stream, workload);
            }, specs, plan.workload));
    scratch.reset(); // The original cold isolated inputs/outputs have retired.
    HYLE_ASSIGN_OR_RETURN(auto invocation, cast_program.create_invocation());
    HYLE_RETURN_IF_ERROR(invocation.bind(examples));
    HYLE_ASSIGN_OR_RETURN(auto work, invocation.submit());
    HYLE_ASSIGN_OR_RETURN(auto result, work.wait());
    const auto &native = result.get()->arrays()[0];
    auto allocator = mem::system_allocator();
    if (boundary && boundary->resource_budget.valid()) {
      HYLE_ASSIGN_OR_RETURN(allocator, boundary->resource_budget.metadata_allocator());
    }
    auto owner = std::allocate_shared<PreparedConstantOwner>(mem::StdAllocator<PreparedConstantOwner>(allocator), PreparedConstantOwner{
        std::move(charge), std::move(private_metadata), native});
    mx::array base(mx::Shape{1}, mx::uint8, nullptr, {});
    base.set_data(native.buffer(),
        [retained = std::move(owner)](mx::allocator::Buffer) {});
    base.set_status(mx::array::available);
    mx::array ready(native.shape(), native.dtype(), nullptr, {});
    ready.copy_shared_buffer(base, native.strides(), native.flags(),
                             native.data_size(), native.offset());
    ready.set_status(mx::array::available);
    if (plan.prepared_casts.empty()) plan.prepared_casts.resize(plan.descriptions.size());
    plan.prepared_casts[source] = std::move(ready);
    ++plan.prepared_storage.count;
    plan.prepared_storage.bytes += bytes;
    return Status::Ok();
  };
  for (const auto &node : plan.nodes) {
    if (const auto op = node.operation.custom_as<compute::model_ops::MlxQuantized>();
        op && op->format == compute::model_ops::MlxWeightFormat::kAffine &&
        op->kind != compute::model_ops::MlxQuantizedKind::kEmbedding) {
      HYLE_RETURN_IF_ERROR(prepare(node.inputs[2]));
      HYLE_RETURN_IF_ERROR(prepare(node.inputs[3]));
    }
    const auto *cast = node.operation.get_if<compute::Cast>();
    if (!cast || cast->output_type != compute::DataType::kFloat32 ||
        node.inputs.size() != 1 || node.outputs.size() != 1)
      continue;
    const auto source = node.inputs[0], output = node.outputs[0];
    const auto &from = plan.descriptions[source], &to = plan.descriptions[output];
    if (!from.is_plain() || !from.is_dense() || from.rank() != 2 ||
        from.data_type() != compute::DataType::kBFloat16 ||
        !to.is_plain() || !to.is_dense())
      continue;
    bool published = false;
    for (std::size_t port = 0; port < graph.output_count(); ++port) {
      auto value = graph.output_id(port).value();
      for (;;) {
        if (value.ordinal() == output) { published = true; break; }
        if (value.kind() != ValueKind::kNodeOutput) break;
        const auto producer = graph.producer(value).value();
        if (graph.operation_kind(producer).value() != compute::OperationKind::kView) break;
        value = graph.node_input(producer, 0).value();
      }
      if (published) break;
    }
    if (published) continue;
    HYLE_RETURN_IF_ERROR(prepare(source));
    if (!plan.prepared_casts.empty() && plan.prepared_casts[source])
      plan.prepared_casts[output] = plan.prepared_casts[source];
  }
  return Status::Ok();
}

struct BFloatL2 final {
  std::size_t source, epsilon;
};

std::optional<BFloatL2> find_bfloat_l2(const Plan &plan,
    std::span<const Node *const> producers, std::size_t port) noexcept {
  using namespace compute;
  const auto from = [&](std::size_t id) { return producers[id]; };
  const auto bf16 = [&](std::size_t id) { return plan.descriptions[id].data_type() == DataType::kBFloat16; };
  const auto binary = [](const Node *node, BinaryKind kind) {
    const auto *op = node ? node->operation.get_if<Binary>() : nullptr;
    return op && op->kind == kind && node->inputs.size() == 2;
  };
  const auto *cast = from(port);
  const auto *conversion = cast ? cast->operation.get_if<Cast>() : nullptr;
  if (!conversion || conversion->output_type != DataType::kFloat32 || cast->inputs.size() != 1 || !bf16(cast->inputs[0]))
    return {};
  const auto *normalize = from(cast->inputs[0]);
  if (!binary(normalize, BinaryKind::kMultiply))
    return {};
  const auto source = normalize->inputs[0], scale = normalize->inputs[1];
  const auto &desc = plan.descriptions[source];
  if (!desc.is_plain() || !bf16(source) || !bf16(scale) || desc.rank() != 4 ||
      desc.dimensions()[0] != 1 || desc.dimensions()[1]<1 || desc.dimensions()[1]>4 ||
      desc.dimensions()[2]<1 || desc.dimensions()[2]>32 || desc.dimensions()[3] != 128)
    return {};
  const auto *root = from(scale);
  const auto *rsqrt = root ? root->operation.get_if<Unary>() : nullptr;
  if (!rsqrt || rsqrt->kind != UnaryKind::kRsqrt || root->inputs.size() != 1 || !bf16(root->inputs[0])) return {};
  const auto *add = from(root->inputs[0]);
  if (!binary(add, BinaryKind::kAdd))
    return {};
  const auto sum_id = add->inputs[0], epsilon = add->inputs[1];
  const auto count = plan.descriptions[epsilon].element_count();
  if (!bf16(sum_id) || !bf16(epsilon) || !count || *count != 1) return {};
  const auto *reduce = from(sum_id);
  const auto *sum = reduce ? reduce->operation.get_if<Reduce>() : nullptr;
  if (!sum || sum->kind != ReductionKind::kSum || !sum->keep_dimensions ||
      sum->accumulation_type != DataType::kFloat32 || sum->axes.is_all() || sum->axes.values().size() != 1 ||
      (sum->axes.values()[0] != -1 && sum->axes.values()[0] != 3) || reduce->inputs.size() != 1 ||
      !bf16(reduce->inputs[0])) return {};
  const auto *square = from(reduce->inputs[0]);
  if (!binary(square, BinaryKind::kMultiply) || square->inputs[0] != source || square->inputs[1] != source)
    return {};
  return BFloatL2{source, epsilon};
}

struct ConvChannels final {
  std::size_t activation;
  std::int64_t start, size;
};

std::optional<ConvChannels> find_conv_channels(std::span<const Node *const> producers,
    std::size_t id) noexcept {
  const auto peel_copy = [&](std::size_t value) {
    const auto *node = producers[value];
    return node && node->operation.get_if<compute::Copy>() && node->inputs.size() == 1 ? node->inputs[0] : value;
  };
  const auto *reshape = producers[peel_copy(id)];
  const auto *view = reshape?reshape->operation.get_if<compute::View>():nullptr;
  if (!view || view->kind() != compute::ViewKind::kReshape || reshape->inputs.size() != 1) return {};
  const auto *slice = producers[peel_copy(reshape->inputs[0])];
  view = slice?slice->operation.get_if<compute::View>():nullptr;
  if (!view || view->kind() != compute::ViewKind::kSlice || (view->axis() != 2 && view->axis() != -1) ||
      view->step() != 1 || slice->inputs.size() != 1) return {};
  return ConvChannels{slice->inputs[0], view->start(), view->count()};
}

void prepare_conv_gdn_inputs(const Plan &plan, std::span<const Node *const> producers, const Node &node,
    BFloatL2 q, BFloatL2 k, const std::vector<std::optional<mx::array>> &values,
    std::vector<mx::array> &arguments, mx::Stream stream) {
  using namespace compute;
  const auto *v_cast = producers[node.inputs[2]];
  const auto *vc = v_cast?v_cast->operation.get_if<Cast>():nullptr;
  if (!vc || vc->output_type != DataType::kFloat32 || v_cast->inputs.size() != 1) return;
  const auto q_channel = find_conv_channels(producers, q.source);
  const auto k_channel = find_conv_channels(producers, k.source);
  const auto v_channel = find_conv_channels(producers, v_cast->inputs[0]);
  if (!q_channel || !k_channel || !v_channel ||
      q_channel->activation != k_channel->activation || q_channel->activation != v_channel->activation)
    return;
  const auto *activation = producers[q_channel->activation];
  const auto *silu = activation?activation->operation.get_if<Unary>():nullptr;
  if (!silu || silu->kind != UnaryKind::kSilu || activation->inputs.size() != 1 ||
      plan.descriptions[q_channel->activation].data_type() != DataType::kBFloat16)
    return;
  const auto *activation_cast = producers[activation->inputs[0]];
  const auto *ac = activation_cast?activation_cast->operation.get_if<Cast>():nullptr;
  if (!ac || ac->output_type != DataType::kBFloat16 || activation_cast->inputs.size() != 1) return;
  const auto *conv = producers[activation_cast->inputs[0]];
  if (!conv || !conv->operation.custom_as<model_ops::CausalConv1d>() || conv->inputs.size() != 2) return;
  const auto *input_cast = producers[conv->inputs[0]];
  const auto *ic = input_cast?input_cast->operation.get_if<Cast>():nullptr;
  if (!ic || ic->output_type != DataType::kFloat32 || input_cast->inputs.size() != 1 ||
      !values[input_cast->inputs[0]] || !values[conv->inputs[1]]) return;
  const auto &input = *values[input_cast->inputs[0]], &weights = *values[conv->inputs[1]];
  const auto &keys = arguments[0].shape(), &val = arguments[2].shape();
  if (keys.size() != 4 || val.size() != 4 || val[0] != 1 || val[1] != keys[1] || val[3] != 128 ||
      val[2] <= 0 || val[2]>32 || input.ndim() != 3 || input.dtype() != mx::bfloat16 ||
      input.shape(0) != 1 || input.shape(1) != keys[1]+3 || weights.dtype() != mx::float32 ||
      weights.ndim() != 2 || weights.shape(1) != 4 || weights.shape(0) != input.shape(2))
    return;
  const auto valid = [&](const ConvChannels &slice, std::int64_t count) {
    return slice.start >= 0 && slice.size == count &&
           slice.start <= input.shape(2) - count;
  };
  if (!valid(*q_channel, keys[2]*128) || !valid(*k_channel, keys[2]*128) || !valid(*v_channel, val[2]*128))
    return;
  const std::array starts{static_cast<int>(q_channel->start), static_cast<int>(k_channel->start),
      static_cast<int>(v_channel->start)};
  auto lowered = ca::detail::lower_short_conv_gdn_inputs(input, weights,
      *values[q.epsilon], *values[k.epsilon], keys, val, starts, stream);
  for (std::size_t i = 0; i < 3; ++i) arguments[i] = std::move(lowered[i]);
}

void prepare_gdn_inputs(const Plan &plan, std::span<const Node *const> producers,
    const Node &node, const std::vector<std::optional<mx::array>> &values,
    std::vector<mx::array> &arguments, mx::Stream stream) {
  if (producers.empty() || !node.operation.get_if<compute::GatedDeltaNet>() || arguments.size() != 6)
    return;
  const auto q = find_bfloat_l2(plan, producers, node.inputs[0]);
  const auto k = find_bfloat_l2(plan, producers, node.inputs[1]);
  if (!q || !k || !values[q->source] || !values[k->source] || !values[q->epsilon] || !values[k->epsilon])
    return;
  const auto &raw_q = *values[q->source], &raw_k = *values[k->source];
  if (raw_q.shape() != raw_k.shape() || raw_q.shape() != arguments[0].shape() ||
      raw_k.shape() != arguments[1].shape() || stream.device != mx::Device::gpu)
    return;
  prepare_conv_gdn_inputs(plan, producers, node, *q, *k, values, arguments, stream);
}

std::optional<mx::array> prepare_gated_projection(
    std::span<const Node *const> producers, std::span<const std::uint8_t> uses,
    const Node &node, const std::vector<std::optional<mx::array>> &values, mx::Stream stream) {
  const auto *multiply = node.operation.get_if<compute::Binary>();
  if (!multiply || multiply->kind != compute::BinaryKind::kMultiply || node.inputs.size() != 2)
    return std::nullopt;
  const auto unique_producer = [&](std::size_t id) -> const Node * {
    if (uses[id] != 1) return nullptr;
    const auto *producer = producers[id];
    return producer && !producer->control_boundary && !producer->preserve_outputs ? producer : nullptr;
  };
  const auto *activate = unique_producer(node.inputs[0]);
  const auto *silu = activate ? activate->operation.get_if<compute::Unary>() : nullptr;
  if (!silu || silu->kind != compute::UnaryKind::kSilu || activate->inputs.size() != 1)
    return std::nullopt;
  const auto *gate = unique_producer(activate->inputs[0]);
  const auto *up = unique_producer(node.inputs[1]);
  const auto plain_projection = [](const Node *producer) {
    const auto *operation = producer ? producer->operation.get_if<compute::MatMul>() : nullptr;
    return operation && operation->accumulation_type == compute::DataType::kFloat32 &&
        producer->inputs.size() == 2;
  };
  if (!plain_projection(gate) || !plain_projection(up) || gate->inputs[0] != up->inputs[0])
    return std::nullopt;
  if (values[activate->inputs[0]]->dtype() != mx::bfloat16 ||
      values[node.inputs[0]]->dtype() != mx::bfloat16 ||
      values[node.inputs[1]]->dtype() != mx::bfloat16)
    return std::nullopt;
  const auto &x = *values[gate->inputs[0]];
  if (auto short_projection = ca::detail::lower_short_bfloat_gated_projection(x,
      *values[gate->inputs[1]], *values[up->inputs[1]], stream))
    return short_projection;
  return ca::detail::lower_bfloat_silu_product(
      *values[activate->inputs[0]], *values[node.inputs[1]], stream);
}

Result<std::vector<mx::array>> trace(const Plan &plan,
                                     std::span<const Node> nodes,
                                     std::span<const std::size_t> input_ids,
                                     std::span<const std::size_t> output_ids,
                                     const std::vector<mx::array> &inputs,
                                     mx::Stream stream) {
  // This storage exists only during cold MLX tracing. Hot runs invoke its
  // compiled artifact; they never allocate or interpret this HYLE value table.
  std::vector<std::optional<mx::array>> values(plan.descriptions.size());
  for (std::size_t i = 0; i < inputs.size(); ++i)
    values[input_ids[i]] = inputs[i];
  for (std::size_t i = 0; i < plan.constants.size(); ++i)
    values[plan.constants[i]] = plan.values[i];
  std::vector<const Node *> producers;
  std::vector<std::uint8_t> uses(plan.descriptions.size());
  producers.resize(plan.descriptions.size());
  const auto use = [&](std::size_t id) { if (uses[id] < 2) ++uses[id]; };
  for (const auto &producer : nodes) {
    for (auto id : producer.outputs) producers[id] = &producer;
    for (auto id : producer.inputs) use(id);
  }
  for (auto id : output_ids) use(id);
  std::vector<mx::array> arguments;
  std::vector<compute::TensorDesc> descriptions;
  for (const auto &node : nodes) {
    if (node.outputs.size() == 1 && !plan.prepared_casts.empty() &&
        plan.prepared_casts[node.outputs[0]]) {
      values[node.outputs[0]] = *plan.prepared_casts[node.outputs[0]];
      continue; // Captured ready array: MLX compile_replace reuses this leaf.
    }
    arguments.clear();
    descriptions.clear();
    for (auto index : node.inputs) {
      arguments.push_back(*values[index]);
      descriptions.push_back(plan.descriptions[index]);
    }
    prepare_gdn_inputs(plan, producers, node, values, arguments, stream);
    ca::detail::PreparedMlxAffineMetadata prepared;
    const auto affine = node.operation.custom_as<compute::model_ops::MlxQuantized>();
    if (affine && affine->format == compute::model_ops::MlxWeightFormat::kAffine &&
        affine->kind != compute::model_ops::MlxQuantizedKind::kEmbedding &&
        !plan.prepared_casts.empty()) {
      prepared.scales = plan.prepared_casts[node.inputs[2]];
      prepared.biases = plan.prepared_casts[node.inputs[3]];
    }
    ca::detail::RowCacheAccessFacts row_access;
    auto fused = [&]() -> std::optional<mx::array> {
      if (node.indexed_bfloat_projection)
        return ca::detail::lower_indexed_bfloat_projection(node.packed_projection.get(),
            arguments[0], arguments[1], arguments[2], stream);
      if (!node.packed_projection)
        return prepare_gated_projection(producers, uses, node, values, stream);
      const bool matmul = node.operation.get_if<compute::MatMul>() != nullptr;
      const auto weights = matmul ? mx::transpose(arguments[1], stream) : arguments[1];
      return ca::detail::lower_packed_bfloat_projection(*node.packed_projection, weights,
          arguments[0], matmul ? mx::bfloat16 : mx::float32, stream);
    }();
    auto outputs =
        fused ? Result<std::vector<mx::array>>(std::vector{std::move(*fused)}) :
        prepared.scales || prepared.biases
            ? ca::detail::lower_tensor(node.operation, arguments, stream, &prepared)
        : ca::detail::is_mlx_gguf_view(node.operation, descriptions)
            ? ca::detail::lower_mlx_gguf_view(node.operation, descriptions, arguments, stream)
        : ca::detail::is_mlx_exact_gguf_gather(node.operation, descriptions)
            ? ca::detail::lower_mlx_exact_gguf_gather(node.operation, descriptions, arguments, stream)
        :
        ca::detail::is_mlx_float_operation(node.operation, descriptions)
            ? ca::detail::lower_mlx_float(node.operation, descriptions, arguments, stream,
                plan.row_candidates ? &row_access : nullptr)
        : ca::lower(node.operation, arguments, stream, plan.workload);
    if (!outputs && outputs.error().IsUnimplemented() && plan.custom &&
        node.operation.kind() == compute::OperationKind::kCustom) {
      outputs = plan.custom(node.operation, arguments, stream);
      if (!outputs) plan.custom_failed.store(true, std::memory_order_release);
    }
    if (!outputs)
      return std::unexpected(std::move(outputs).error());
    if (plan.row_candidates) {
      const auto position = std::ranges::lower_bound(*plan.row_candidates,
          node.original_ordinal, {}, &RowCandidate::node);
      if (position != plan.row_candidates->end() && position->node == node.original_ordinal) {
        position->traced = true;
        position->matched = position->matched && row_access == position->expected;
      }
    }
    if (outputs->size() != node.outputs.size())
      return std::unexpected(Status::InvalidArgument(
          "Apple custom lowering output arity mismatch"));
    for (std::size_t i = 0; i < outputs->size(); ++i) {
      const auto index = node.outputs[i];
      HYLE_RETURN_IF_ERROR(match(plan.descriptions[index], (*outputs)[i]));
      values[index] = std::move((*outputs)[i]);
    }
  }
  std::vector<mx::array> result;
  result.reserve(output_ids.size());
  for (auto index : output_ids) {
    auto value = std::move(*values[index]);
    result.push_back(plan.contiguous_outputs ? mx::contiguous(value, false, stream) : std::move(value));
  }
  return result;
}

Result<ca::Program> compile_regions(const std::shared_ptr<ca::detail::Runtime> &runtime,
                                    std::shared_ptr<const Plan> plan,
                                    std::span<const std::size_t> descriptor_values) {
  ca::detail::RegionPlan native;
  native.workload = plan->workload;
  native.values.resize(plan->descriptions.size());
  native.inputs = plan->inputs;
  native.outputs = plan->outputs;
  native.constants = plan->constants;
  native.constant_values = plan->values;
  for (std::size_t id = 0; id < plan->descriptions.size(); ++id)
    if (plan->descriptions[id].valid()) {
      HYLE_ASSIGN_OR_RETURN(native.values[id],
                            native_spec(plan->descriptions[id]));
    }
  std::vector<std::size_t> last_use(native.values.size());
  std::vector<bool> constant(native.values.size(), false);
  for (auto id : plan->constants)
    constant[id] = true;
  for (std::size_t i = 0; i < plan->nodes.size(); ++i)
    for (auto id : plan->nodes[i].inputs)
      last_use[id] = i;
  for (auto id : plan->outputs)
    last_use[id] = plan->nodes.size();

  // Track only the indices output of the original TopK. SSA identity and
  // reshape preserve its range; external IDs, values, casts and arithmetic do
  // not establish this proof. This is cold O(nodes + values), not a hot scan.
  std::vector<std::int64_t> index_extents(plan->descriptions.size(), 0);
  for (const auto &node : plan->nodes) {
    if (const auto *top = node.operation.get_if<compute::TopK>();
        top && node.inputs.size() == 1 && node.outputs.size() == 2) {
      const auto &source = plan->descriptions[node.inputs[0]];
      const auto rank = static_cast<std::int32_t>(source.rank());
      const auto axis = top->axis < 0 ? top->axis + rank : top->axis;
      if (axis >= 0 && axis < rank && top->count > 0 &&
          source.dimensions()[axis] > 0 &&
          top->count <= static_cast<std::uint64_t>(source.dimensions()[axis]) &&
          plan->descriptions[node.outputs[1]].data_type() == compute::DataType::kInt32)
        index_extents[node.outputs[1]] = source.dimensions()[axis];
    } else if (const auto *view = node.operation.get_if<compute::View>();
        view && view->kind() == compute::ViewKind::kReshape &&
        node.inputs.size() == 1 && node.outputs.size() == 1 &&
        plan->descriptions[node.inputs[0]].data_type() == compute::DataType::kInt32 &&
        plan->descriptions[node.outputs[0]].data_type() == compute::DataType::kInt32) {
      index_extents[node.outputs[0]] = index_extents[node.inputs[0]];
    }
  }

  for (std::size_t begin = 0; begin < plan->nodes.size();) {
    ca::detail::Region region;
    const auto &first = plan->nodes[begin];
    const bool row_indexer=first.operation.custom_as<compute::model_ops::RowCacheIndexerScores>().has_value();
    if (row_indexer || first.carrier != ca::detail::RegionKind::kMlx) {
      region.kind = first.carrier;
      region.operation = first.operation;
      region.inputs = first.inputs;
      region.outputs = first.outputs;
      if (first.operation.custom_as<compute::model_ops::GroupedMatMul>() &&
          first.inputs.size() == 3 && plan->descriptions[first.inputs[0]].rank() == 3 &&
          index_extents[first.inputs[2]] > 0 &&
          index_extents[first.inputs[2]] == plan->descriptions[first.inputs[0]].dimensions()[0]) {
        region.proven_index_port = 2;
        region.proven_index_extent = index_extents[first.inputs[2]];
      }
      for (auto id : first.inputs)
        region.native_inputs.push_back(plan->descriptions[id]);
      if(row_indexer) {
        region.mlx=[plan,begin,inputs=region.inputs,outputs=region.outputs](const auto&values,mx::Stream stream){
          return trace(*plan,std::span<const Node>(plan->nodes).subspan(begin,1),inputs,outputs,values,stream);
        };
      }
      native.regions.push_back(std::move(region));
      ++begin;
      continue;
    }
    auto end = begin + 1;
    while (!first.control_boundary && end < plan->nodes.size() &&
           !plan->nodes[end].control_boundary &&
           !plan->nodes[end].operation.custom_as<compute::model_ops::RowCacheIndexerScores>() &&
           plan->nodes[end].carrier == ca::detail::RegionKind::kMlx)
      ++end;
    std::vector<bool> internal(native.values.size(), false);
    std::vector<bool> argument(native.values.size(), false);
    for (auto i = begin; i < end; ++i) {
      for (auto id : plan->nodes[i].inputs)
        if (!internal[id] && !constant[id] && !argument[id]) {
          region.inputs.push_back(id);
          argument[id] = true;
        }
      for (auto id : plan->nodes[i].outputs) {
        internal[id] = true;
        if (last_use[id] >= end || plan->nodes[i].preserve_outputs)
          region.outputs.push_back(id);
      }
    }
    // Only cross-region/public values become MLX outputs. Publishing every
    // intermediate here would prevent the original compiler's fusion.
    region.mlx = [plan, begin, end, inputs = region.inputs,
                  outputs = region.outputs](const auto &values,
                                            mx::Stream stream) {
      return trace(
          *plan, std::span<const Node>(plan->nodes).subspan(begin, end - begin),
          inputs, outputs, values, stream);
    };
    native.regions.push_back(std::move(region));
    begin = end;
  }
  return ca::detail::ContextAccess::compile(runtime, std::move(native), descriptor_values);
}
} // namespace

static Result<ca::Program> compile_impl(const std::shared_ptr<ca::detail::Runtime> &runtime,
                                        const Graph &graph,
                                        std::span<const mx::array> constants,
                                        CompileOptions options,
                                        const RegionCompileRequest *boundary = nullptr,
                                        bool *custom_failed = nullptr,
                                        detail::PreparedRowAccesses *row_accesses = nullptr,
                                        detail::PreparedConstantStorage *storage = nullptr,
                                        bool device_prefix = false) noexcept {
  HYLE_TRACE_ZONE_NAMED("hyle.graph.apple.compile");
  if (custom_failed) *custom_failed = false;
  if (row_accesses) row_accesses->clear();
  if (storage) *storage = {};
  if (!options.workload.valid())
    return std::unexpected(Status::InvalidArgument("invalid Apple workload"));
  // A pure producer region can publish boundary values into a zero-output
  // effect Graph; the public whole-Graph entry still needs published outputs.
  if (!runtime || !graph || (boundary ? boundary->outputs.empty() : graph.output_count() == 0))
    return std::unexpected(Status::InvalidArgument(
        "Apple Graph needs a Context and published outputs"));
  if (graph.dimension_count() != 0)
    return std::unexpected(
        Status::Unimplemented("Apple Graph admits fixed, pure regions only"));
  if (constants.size() != (boundary ? boundary->constants.size() : graph.constant_count()))
    return std::unexpected(
        Status::InvalidArgument("Apple Graph constant count mismatch"));
  try {
    auto *report = options.report;
    if (report != nullptr) {
      report->environment = runtime->environment;
      report->implementation_policy = "One MLX GPU region; fixed typed "
                                      "lowering plus same-runtime custom nodes";
      report->numerical_policy =
          "Typed F32/F16/BF16 tensor, F32 gate/norm/reduce/softmax and stable "
          "TopK; "
          "I32/I64/Bool storage, checked Gather/ordered Scatter/Scan and "
          "batched MatMul, exact-position RoPE, normalized Hadamard, GQA SDPA "
          "and affine Q4/group64; "
          "each node retains its "
          "schema contract; MLX private intermediates unknown";
      report->boundary_policy =
          "Caller inputs/constants must be ready; no HYLE shadow output copy; "
          "native result leases retain MLX outputs; private GPU algorithm "
          "status is read after completion; internal fusion/copies "
          "unknown";
      report->concurrency_policy =
          "Independent Invocation/results; one Context owner executes FIFO "
          "and checks native completion before releasing admitted resources";
      report->declared_constant_count = constants.size();
    }
    auto plan = std::make_shared<Plan>();
    ColdRowAccess row_trace{plan, {}};
    struct CustomOutcome final {
      std::shared_ptr<const Plan> plan;
      bool *failed;
      ~CustomOutcome() {
        if (failed) *failed = plan->custom_failed.load(std::memory_order_acquire);
      }
    } outcome{custom_failed ? plan : nullptr, custom_failed};
    plan->workload = options.workload;
    plan->custom = std::move(options.custom);
    plan->contiguous_outputs = options.contiguous_outputs;
    plan->descriptions.resize(graph.value_count());
    std::vector<ca::InputSpec> specs;
    specs.reserve(graph.input_count());
    auto describe = [&](ValueId id) -> Status {
      HYLE_ASSIGN_OR_RETURN(const auto &spec, graph.value_spec(id));
      HYLE_ASSIGN_OR_RETURN(auto description, spec.concrete_description());
      HYLE_ASSIGN_OR_RETURN(auto native, native_spec(description));
      plan->descriptions[id.ordinal()] = description;
      return Status::Ok();
    };
    const auto input_count = boundary ? boundary->inputs.size() : graph.input_count();
    for (std::size_t i = 0; i < input_count; ++i) {
      HYLE_ASSIGN_OR_RETURN(auto id, boundary ? Result<ValueId>(boundary->inputs[i]) : graph.input_id(i));
      if (id.kind() == ValueKind::kConstant) continue;
      HYLE_RETURN_IF_ERROR(describe(id));
      HYLE_ASSIGN_OR_RETURN(auto spec,
                            native_spec(plan->descriptions[id.ordinal()]));
      specs.push_back(std::move(spec));
      plan->inputs.push_back(id.ordinal());
    }
    for (std::size_t i = 0; i < constants.size(); ++i) {
      HYLE_ASSIGN_OR_RETURN(auto id, boundary ? Result<ValueId>(boundary->constants[i].value) : graph.constant_id(i));
      HYLE_RETURN_IF_ERROR(describe(id));
      if (constants[i].status() != mx::array::available)
        return std::unexpected(Status::FailedPrecondition(
            "Apple Graph constants must be successfully ready"));
      HYLE_RETURN_IF_ERROR(
          match(plan->descriptions[id.ordinal()], constants[i]));
      plan->constants.push_back(id.ordinal());
      plan->values.push_back(constants[i]);
      if (report != nullptr) {
        HYLE_ASSIGN_OR_RETURN(
            auto bytes, plan->descriptions[id.ordinal()].required_bytes());
        if (bytes > SIZE_MAX - report->logical_constant_bytes)
          return std::unexpected(Status::ResourceExhausted(
              "Apple report logical constant bytes overflow"));
        report->logical_constant_bytes += bytes;
      }
    }
    HYLE_ASSIGN_OR_RETURN(auto analysis, boundary
        ? graph::detail::analyze_native_fragment(graph, boundary->nodes)
        : graph::detail::analyze_native_pure(graph));
    std::vector<bool> live(graph.value_count(), false);
    const auto output_count = boundary ? boundary->outputs.size() : graph.output_count();
    for (std::size_t i = 0; i < output_count; ++i) {
      HYLE_ASSIGN_OR_RETURN(auto id, boundary ? Result<ValueId>(boundary->outputs[i]) : graph.output_id(i));
      HYLE_RETURN_IF_ERROR(describe(id));
      plan->outputs.push_back(id.ordinal());
      live[id.ordinal()] = true;
    }
    // GraphBuilder guarantees topological order. Reverse reachability removes
    // dead pure nodes without splitting the mature runtime's fusion region.
    std::vector<std::optional<Node>> nodes(graph.node_count());
    using Packed = graph::detail::ArtifactPackedPattern;
    using Static = graph::detail::StaticFloat8Pattern;
    std::vector<std::optional<Packed::Pattern>> packed;
    std::vector<std::optional<Static::Pattern>> static_fp8;
    std::vector<std::uint8_t> absorbed;
    if (boundary && boundary->route == detail::kNativeCtPackedRoute) {
      packed.resize(graph.node_count());
      absorbed.resize(graph.node_count(), 0);
      Packed::Uses uses(mem::system_allocator());
      HYLE_RETURN_IF_ERROR(uses.initialize(graph));
      std::vector<std::uint8_t> allowed(graph.node_count(), 0);
      for (auto id : boundary->nodes) allowed[id.ordinal()] = 1;
      bool selected = false;
      for (auto id : boundary->nodes) {
        Packed::Pattern pattern;
        if (!Packed::pattern(graph, id, pattern) ||
            pattern.descriptor.format == compute::model_ops::WeightArtifactFormat::kNf4DoubleQuant ||
            !Packed::closed(graph, pattern, uses, allowed))
          continue;
        const auto terminal = graph.producer(pattern.output).value().ordinal();
        packed[terminal] = pattern;
        selected = true;
        for (auto node : pattern.span()) absorbed[node.ordinal()] = node.ordinal() != terminal;
      }
      if (!selected)
        return std::unexpected(Status::Unimplemented("Apple packed route has no closed artifact recipe"));
    }
    if (boundary && boundary->route == detail::kNativeStaticFloat8Route) {
      static_fp8.resize(graph.node_count());
      absorbed.resize(graph.node_count(), 0);
      Static::Uses uses(mem::system_allocator());
      HYLE_RETURN_IF_ERROR(uses.initialize(graph));
      std::vector<std::uint8_t> allowed(graph.node_count(), 0);
      for (auto id : boundary->nodes) allowed[id.ordinal()] = 1;
      bool selected = false;
      for (auto id : boundary->nodes) {
        Static::Pattern pattern;
        if (!Static::pattern(graph, id, pattern) || !Static::closed(graph, pattern, uses, allowed))
          continue;
        const auto terminal = graph.producer(pattern.output).value().ordinal();
        static_fp8[terminal] = pattern;
        selected = true;
        for (auto node : pattern.span()) absorbed[node.ordinal()] = node.ordinal() != terminal;
      }
      if (!selected)
        return std::unexpected(Status::Unimplemented("Apple static FP8 route has no closed stored-scale recipe"));
    }
    for (std::size_t i = graph.node_count(); i > 0; --i) {
      HYLE_ASSIGN_OR_RETURN(auto id, graph.node_id(i - 1));
      if (boundary && std::ranges::find(boundary->nodes, id) == boundary->nodes.end())
        continue;
      if (!absorbed.empty() && absorbed[i - 1]) continue;
      HYLE_ASSIGN_OR_RETURN(const auto &operation, graph.operation(id));
      HYLE_ASSIGN_OR_RETURN(auto output_count, graph.node_output_count(id));
      bool needed = analysis.needed[i - 1] != 0;
      Node node;
      node.original_ordinal = static_cast<std::uint32_t>(i - 1);
      node.operation = operation;
      node.control_boundary = analysis.boundary[i - 1] != 0;
      node.preserve_outputs = analysis.control_predecessor[i - 1] != 0;
      if (!static_fp8.empty() && static_fp8[i - 1]) {
        const auto &pattern = *static_fp8[i - 1];
        node.operation = pattern.composite();
        HYLE_RETURN_IF_ERROR(describe(pattern.output));
        node.outputs.push_back(pattern.output.ordinal());
        for (auto source : pattern.span()) {
          node.control_boundary |= analysis.boundary[source.ordinal()] != 0;
          node.preserve_outputs |= analysis.control_predecessor[source.ordinal()] != 0;
        }
        for (auto value : pattern.inputs) {
          HYLE_RETURN_IF_ERROR(describe(value));
          node.inputs.push_back(value.ordinal());
          live[value.ordinal()] = true;
        }
        nodes[i - 1] = std::move(node);
        continue;
      }
      if (!packed.empty() && packed[i - 1]) {
        const auto &pattern = *packed[i - 1];
        node.operation = pattern.composite();
        HYLE_RETURN_IF_ERROR(describe(pattern.output));
        node.outputs.push_back(pattern.output.ordinal());
        for (auto source : pattern.span()) {
          node.control_boundary |= analysis.boundary[source.ordinal()] != 0;
          node.preserve_outputs |= analysis.control_predecessor[source.ordinal()] != 0;
        }
        for (auto value : std::span(pattern.inputs).first(pattern.input_count)) {
          HYLE_RETURN_IF_ERROR(describe(value));
          node.inputs.push_back(value.ordinal());
          live[value.ordinal()] = true;
        }
        nodes[i - 1] = std::move(node);
        continue;
      }
      for (std::size_t port = 0; port < output_count; ++port) {
        HYLE_ASSIGN_OR_RETURN(auto value, graph.result(id, port));
        node.outputs.push_back(value.ordinal());
        needed = needed || live[value.ordinal()];
        HYLE_RETURN_IF_ERROR(describe(value));
      }
      if (!needed)
        continue;
      HYLE_ASSIGN_OR_RETURN(auto input_count, graph.node_input_count(id));
      for (std::size_t port = 0; port < input_count; ++port) {
        HYLE_ASSIGN_OR_RETURN(auto value, graph.node_input(id, port));
        HYLE_RETURN_IF_ERROR(describe(value));
        node.inputs.push_back(value.ordinal());
        live[value.ordinal()] = true;
      }
      nodes[i - 1] = std::move(node);
    }
    for (std::size_t index = 0; index < nodes.size(); ++index) {
      auto &node = nodes[index];
      if (!node)
        continue;
      if (report != nullptr)
        report->nodes.push_back(CompileNodeReport{
            .ordinal = static_cast<std::uint32_t>(index),
            .kind = node->operation.kind(),
            .traits = compute::operation_traits(node->operation),
        });
      std::vector<compute::TensorDesc> descriptions;
      descriptions.reserve(node->inputs.size());
      for (const auto input : node->inputs)
        descriptions.push_back(plan->descriptions[input]);
      HYLE_ASSIGN_OR_RETURN(node->carrier,
          ca::detail::analyze_native_carrier(node->operation, descriptions));
      if (boundary) HYLE_RETURN_IF_ERROR(prepare_packed_projection(runtime, *boundary, *plan, *node, descriptions));
      plan->nodes.push_back(std::move(*node));
    }
    // Retain only constants reachable by the live native nodes or published
    // boundary. Declaration validation/reporting above remains unchanged.
    std::vector<bool> used_constant(plan->descriptions.size(), false);
    for (const auto &node : plan->nodes)
      for (const auto input : node.inputs) used_constant[input] = true;
    for (const auto output : plan->outputs) used_constant[output] = true;
    std::size_t retained = 0;
    for (std::size_t index = 0; index < plan->constants.size(); ++index) {
      if (!used_constant[plan->constants[index]]) continue;
      plan->constants[retained] = plan->constants[index];
      if (retained != index) plan->values[retained] = std::move(plan->values[index]);
      ++retained;
    }
    plan->constants.resize(retained);
    plan->values.erase(plan->values.begin() + static_cast<std::ptrdiff_t>(retained), plan->values.end());
    HYLE_RETURN_IF_ERROR(prepare_constant_casts(runtime, graph, *plan, boundary));
    if (report) {
      report->prepared_constant_count = plan->prepared_storage.count;
      report->prepared_constant_bytes = plan->prepared_storage.bytes;
    }
    if (storage) *storage = plan->prepared_storage;
    if (report != nullptr) {
      for (const auto input : plan->inputs)
        report->inputs.push_back(plan->descriptions[input]);
      for (const auto output : plan->outputs)
        report->outputs.push_back(plan->descriptions[output]);
    }
    for (const auto &node : plan->nodes) {
      const bool has_gguf = std::ranges::any_of(node.inputs, [&](auto id) {
        return compute::gguf_quantized_type(
                   plan->descriptions[id].storage_format())
            .has_value();
      });
      std::vector<compute::TensorDesc> descriptions;
      for (auto id : node.inputs) descriptions.push_back(plan->descriptions[id]);
      if (has_gguf && node.carrier != ca::detail::RegionKind::kGgml &&
          !ca::detail::is_mlx_gguf_view(node.operation, descriptions) &&
          !ca::detail::is_mlx_exact_gguf_gather(node.operation, descriptions))
        return std::unexpected(Status::Unimplemented(
            "Apple Graph GGUF carrier requires a qualified native model operation: " +
            std::string(compute::operation_name(node.operation))));
    }
    // Cold bank descriptors do not grant projected-read permissions. Direct
    // and Graph use the original math's bank role, including contiguous scorer
    // and unchecked expert consumers. Internal produced banks remain real work.
    std::vector<bool> boundary_bank(plan->descriptions.size(), false);
    for (auto value : plan->inputs) boundary_bank[value] = true;
    for (auto value : plan->constants) boundary_bank[value] = true;
    std::vector<std::size_t> descriptor_values, descriptor_inputs;
    for (const auto &node : plan->nodes) {
      if (node.carrier != ca::detail::RegionKind::kMlx) continue;
      const auto bank_port = ca::detail::mlx_row_bank_input(node.operation);
      if (!bank_port || *bank_port >= node.inputs.size()) continue;
      const auto bank = node.inputs[*bank_port];
      if (!boundary_bank[bank]) continue;
      boundary_bank[bank] = false;
      descriptor_values.push_back(bank);
    }
    for (std::size_t port = 0; port < plan->inputs.size(); ++port)
      if (std::ranges::find(descriptor_values, plan->inputs[port]) != descriptor_values.end())
        descriptor_inputs.push_back(port);
    // Nominate projected readers from the same native access analyzer.
    // Only the actual successful trace below can publish a selected proof.
    // A bank must be a boundary input with one exact decoder reader: aliases,
    // outputs, and additional readers retain the ordinary full-input contract.
    std::unordered_map<std::size_t, std::size_t> banks;
    for (const auto &node : plan->nodes) {
      if (node.carrier != ca::detail::RegionKind::kMlx) continue;
      std::vector<compute::TensorDesc> descriptions;
      descriptions.reserve(node.inputs.size());
      for (auto id : node.inputs) descriptions.push_back(plan->descriptions[id]);
      auto analyzed = ca::detail::analyze_mlx_row_access(node.operation, descriptions);
      if (!analyzed) {
        if (analyzed.error().IsUnimplemented()) continue;
        return std::unexpected(analyzed.error());
      }
      if (!*analyzed) continue;
      const auto fact = **analyzed;
      const auto bank = fact.projected_input == UINT32_MAX ? SIZE_MAX
          : node.inputs[fact.projected_input];
      auto &candidate = row_trace.candidates.emplace_back(
          RowCandidate{node.original_ordinal, bank, fact});
      if (bank != SIZE_MAX) {
        candidate.matched = std::ranges::find(plan->inputs, bank) != plan->inputs.end() &&
            std::ranges::find(plan->outputs, bank) == plan->outputs.end();
        const auto [position, inserted] = banks.emplace(bank, row_trace.candidates.size() - 1);
        if (!inserted) {
          candidate.matched = false;
          row_trace.candidates[position->second].matched = false;
        }
      }
    }
    for (const auto &node : plan->nodes)
      for (std::size_t port = 0; port < node.inputs.size(); ++port) {
        const auto position = banks.find(node.inputs[port]);
        if (position == banks.end()) continue;
        auto &candidate = row_trace.candidates[position->second];
        if (candidate.node != node.original_ordinal || candidate.expected.projected_input != port)
          candidate.matched = false;
      }
    plan->row_candidates = &row_trace.candidates;
    const bool has_native_model_region =
        std::ranges::any_of(plan->nodes, [&](const auto &node) {
          return node.carrier == ca::detail::RegionKind::kGgml;
        });
    const bool has_control = std::ranges::any_of(plan->nodes,
        [](const auto &node) { return node.control_boundary; });
    const bool has_attention_blocks=std::ranges::any_of(plan->nodes,
        [](const auto &node){return node.carrier == ca::detail::RegionKind::kAttentionBlocks;});
    if (has_native_model_region || has_control || has_attention_blocks) {
      if (report != nullptr) {
        report->implementation_policy = has_native_model_region
            ? "Original ggml Metal model regions and maximal MLX compiled regions; fixed cold binding"
            : has_attention_blocks ? "Original MLX compiled regions and private checked attention blocks; fixed cold binding"
            : "Original MLX compiled regions split at explicit control endpoints; fixed cold binding";
        if (has_native_model_region)
          report->numerical_policy +=
              "; provider-owned GGUF MatMul/grouped expert/checked embedding and mHC/recurrent "
              "F32 regions, explicit non-mutating initial/final state";
        report->boundary_policy +=
            "; checked host completion at runtime boundaries; native MTLBuffer "
            "import; strided inputs use original MLX contiguous copy";
      }
    }
    auto program = [&]() -> Result<ca::Program> {
      if (has_native_model_region || has_control || has_attention_blocks)
        return compile_regions(runtime, plan, descriptor_values);
      const auto region = [plan](const auto &inputs, mx::Stream stream) {
        return trace(*plan, plan->nodes, plan->inputs, plan->outputs, inputs, stream);
      };
      // Internal MLX edges can carry the original pending arrays and their
      // completion. Keep ordinary direct callers and published results on
      // their selected path, and leave copy and continuation policy intact.
      bool internal_producer = boundary && !plan->custom;
      for (std::size_t i = 0; internal_producer && i < graph.output_count(); ++i) {
        HYLE_ASSIGN_OR_RETURN(auto output, graph.output_id(i));
        internal_producer = std::ranges::find(boundary->outputs, output) == boundary->outputs.end();
      }
      if ((device_prefix || internal_producer) && !plan->custom)
        return ca::detail::ContextAccess::trace_device_prefix(runtime, region, specs, options.workload, descriptor_inputs);
      if (device_prefix)
        return ca::detail::ContextAccess::compile_device_prefix(runtime, region, specs, options.workload, descriptor_inputs);
      return ca::detail::ContextAccess::compile(runtime, region, specs, options.workload, descriptor_inputs);
    }();
    if (!program) return std::unexpected(program.error());
    if (row_accesses && boundary) {
      for (const auto &candidate : row_trace.candidates) {
        if (!candidate.matched || !candidate.traced) continue;
        auto port = UINT32_MAX;
        if (candidate.bank != SIZE_MAX) {
          const auto position = std::ranges::find_if(boundary->inputs,
              [&](ValueId value) { return value.ordinal() == candidate.bank; });
          if (position == boundary->inputs.end()) continue;
          port = static_cast<std::uint32_t>(position - boundary->inputs.begin());
        }
        row_accesses->push_back({candidate.node, port, candidate.expected});
      }
    }
    return program;
  } catch (const std::bad_alloc &) {
    return std::unexpected(
        Status::ResourceExhausted("Apple Graph preparation allocation failed"));
  } catch (const std::exception &error) {
    return std::unexpected(Status::Internal(error.what()));
  } catch (...) {
    return std::unexpected(Status::Internal("Apple Graph preparation failed"));
  }
}
Result<ca::Program> compile(const ca::Context &context, const Graph &graph,
                            std::span<const mx::array> constants,
                            CompileOptions options) noexcept {
  auto *report = options.report;
  if (report != nullptr)
    *report = {};
  auto program = [&]() -> Result<ca::Program> {
    if (!context) return std::unexpected(Status::InvalidArgument("Apple Graph requires a Context"));
    const auto &runtime = ca::detail::ContextAccess::runtime(context);
    if (graph.dimension_count() == 0 && options.shape.empty())
      return compile_impl(runtime, graph, constants, std::move(options));
    HYLE_ASSIGN_OR_RETURN(auto exact, specialize_exact(graph, options.shape));
    options.shape = {};
    return compile_impl(runtime, exact.graph, constants, std::move(options));
  }();
  if (report != nullptr) {
    if (!program) {
      *report = {};
      report->status = program.error();
    } else {
      report->status = Status::Ok();
    }
  }
  return program;
}

Result<ca::Program> detail::prepare_region(const std::shared_ptr<ca::detail::Runtime> &runtime,
    const RegionCompileRequest &request, std::span<const mx::array> constants,
    CustomLowering custom, bool *custom_failed, PreparedRowAccesses *row_accesses,
    PreparedConstantStorage *storage, bool device_prefix) noexcept {
  if (custom_failed) *custom_failed = false;
  if (storage) *storage = {};
  if (!request.graph) return std::unexpected(Status::InvalidArgument("Apple region has no graph"));
  return compile_impl(runtime, *request.graph, constants,
      {.custom = std::move(custom), .workload = request.workload}, &request,
      custom_failed, row_accesses, storage, device_prefix);
}

} // namespace hyle::graph::apple
