#include "hyle/compute/apple.h"
#include "hyle/compute/detail/apple_plan.h"

#include <array>
#include <cstdint>

#include <mlx/ops.h>

#include "apple_internal.h"
#include "apple_tensor.h"
#include "hyle/compute/route_rule.h"

namespace hyle::compute::apple {
namespace mx = mlx::core;
namespace {
inline constexpr RouteRule kAffineQ4Rule{
    .route = NativeRoute::kApple,
    .operation = OperationKind::kQuantizedMatMul,
    .input0_type = DataType::kFloat32,
    .input1_type = DataType::kFloat32,
    .output0_type = DataType::kFloat32,
    .input1_format = kAffineQ4Format,
    .quantization = {.presence = RouteQuantizationPresence::kQuantizedMatMul,
                     .mode = QuantizedWeightMode::kUnsignedAffine4,
                     .group_size = 64,
                     .weight_format = kAffineQ4Format}};

Result<TensorDesc> float_description(const mx::array &input) noexcept {
  if (input.dtype() != mx::float32 || input.ndim() > kMaxTensorRank)
    return std::unexpected(Status::Unimplemented(
        "Apple typed operations require F32 within the tensor rank limit"));
  std::array<std::int64_t, kMaxTensorRank> shape{};
  for (std::size_t axis = 0; axis < input.ndim(); ++axis) {
    if (input.shape(static_cast<int>(axis)) <= 0)
      return std::unexpected(Status::InvalidArgument(
          "Apple typed operations require nonempty dimensions"));
    shape[axis] = input.shape(static_cast<int>(axis));
  }
  // Native MLX owns any stride handling. This description validates logical
  // operation semantics; it does not import a HYLE buffer or claim zero copy.
  return TensorDesc::dense(DataType::kFloat32, {shape.data(), input.ndim()});
}
} // namespace

Result<TensorDesc> affine_q4_description(std::int64_t output_channels,
                                         std::int64_t input_channels) noexcept {
  if (output_channels <= 0 || input_channels <= 0 || input_channels % 64 != 0)
    return std::unexpected(Status::InvalidArgument(
        "Apple affine Q4 needs positive N and K divisible by 64"));
  const auto format = StorageFormat::blocked(kAffineQ4Format, 8, 4, 4);
  const std::array shape{output_channels, input_channels};
  return TensorDesc::blocked_dense(DataType::kFloat32, shape, *format);
}

Result<std::vector<mx::array>> lower(const Operation &operation,
                                     std::span<const mx::array> inputs,
                                     mx::Stream stream,
                                     const WorkloadContext &workload) noexcept {
  if (!workload.valid())
    return std::unexpected(Status::InvalidArgument("invalid Apple workload"));
  try {
    const auto *matmul = operation.get_if<MatMul>();
    const auto *qmm = operation.get_if<QuantizedMatMul>();
    if (matmul == nullptr && qmm == nullptr)
      return detail::lower_tensor(operation, inputs, stream);
    if (!operation_traits(operation).inputs.accepts(inputs.size()))
      return std::unexpected(Status::InvalidArgument(
          "Apple typed operation input count mismatch"));
    if (stream.device.type != mx::Device::gpu)
      return std::unexpected(Status::InvalidArgument(
          "Apple typed lowering requires a GPU stream"));
    if (matmul && (matmul->accumulation_type != DataType::kFloat32 ||
                   inputs[0].ndim() != 2 || inputs[1].ndim() != 2))
      return std::unexpected(Status::Unimplemented(
          "Apple typed MatMul admits 2-D F32 accumulation"));
    if (qmm && (qmm->accumulation_type != DataType::kFloat32 ||
                inputs[0].ndim() != 2 || inputs[1].ndim() != 2 ||
                inputs[1].dtype() != mx::uint32))
      return std::unexpected(Status::Unimplemented(
          "Apple Q4 admits 2-D affine group64 F32 with uint32 packed weights"));

    std::array<TensorDesc, 4> descriptions{};
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      auto description =
          qmm != nullptr && i == 1
              ? affine_q4_description(inputs[i].shape(0),
                                      std::int64_t{inputs[i].shape(1)} * 8)
              : float_description(inputs[i]);
      if (!description)
        return std::unexpected(std::move(description).error());
      descriptions[i] = *description;
    }
    auto output =
        infer_single_output(operation, {descriptions.data(), inputs.size()});
    if (!output)
      return std::unexpected(std::move(output).error());

    if (qmm) {
      const std::array outputs{*output};
      const RouteMatchInput route_input{
          .platform = kBuildRoutePlatform,
          .route = NativeRoute::kApple,
          .model = workload.model.description(),
          .phase = workload.phase,
          .region = workload.region,
          .operation = operation,
          .inputs = {descriptions.data(), inputs.size()},
          .outputs = outputs};
      if (!match_route_rule(kAffineQ4Rule, route_input).matched())
        return std::unexpected(Status::Unimplemented(
            "Apple Q4 requires affine group64 public storage"));
    }

    if (matmul)
      return std::vector<mx::array>{mx::matmul(inputs[0], inputs[1], stream)};
    return std::vector<mx::array>{
        mx::quantized_matmul(inputs[0], inputs[1], inputs[2], inputs[3], true,
                             64, 4, "affine", stream)};
  } catch (...) {
    return std::unexpected(detail::native_exception());
  }
}

Result<Program>
Context::prepare(const Operation &operation,
                 std::span<const mx::array> examples,
                 const WorkloadContext &workload) const noexcept {
  try {
    if (operation.kind() == OperationKind::kGatedDeltaNet) {
      std::vector<InputSpec> specs;
      specs.reserve(examples.size());
      for (const auto &input : examples) {
        if (input.status() != mx::array::available)
          return std::unexpected(Status::FailedPrecondition("Apple examples must be ready arrays"));
        specs.push_back({input.shape(), input.dtype()});
      }
      return prepare(operation, specs, workload);
    }
    return compile(
        [operation, workload](const auto &inputs, mx::Stream stream) {
          return lower(operation, inputs, stream, workload);
        },
        examples, workload);
  } catch (...) {
    return std::unexpected(detail::native_exception());
  }
}

Result<Program>
Context::prepare(const Operation &operation,
                 std::span<const InputSpec> inputs,
                 const WorkloadContext &workload) const noexcept {
  try {
    if (operation.kind() == OperationKind::kGatedDeltaNet) {
      detail::RegionPlan plan;
      plan.workload = workload;
      detail::Region region;
      region.kind = detail::RegionKind::kGatedDeltaNet;
      for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].dtype != mx::float32)
          return std::unexpected(Status::Unimplemented("Apple original recurrent region currently admits F32 inputs"));
        std::vector<std::int64_t> shape(inputs[i].shape.begin(), inputs[i].shape.end());
        HYLE_ASSIGN_OR_RETURN(auto desc, TensorDesc::dense(DataType::kFloat32, shape));
        region.recurrent_inputs.push_back(desc);
        region.inputs.push_back(i);
        plan.inputs.push_back(i);
        plan.values.push_back(inputs[i]);
      }
      std::array<TensorDesc, 2> outputs;
      HYLE_RETURN_IF_ERROR(infer_output_descriptions(operation, region.recurrent_inputs, outputs));
      for (const auto &output : outputs) {
        const auto id = plan.values.size();
        region.outputs.push_back(id);
        plan.outputs.push_back(id);
        InputSpec spec;
        for (auto d : output.dimensions())
          spec.shape.push_back(static_cast<int>(d));
        plan.values.push_back(std::move(spec));
      }
      plan.regions.push_back(std::move(region));
      return detail::ContextAccess::compile(*this, std::move(plan));
    }
    return compile(
        [operation, workload](const auto &values, mx::Stream stream) {
          return lower(operation, values, stream, workload);
        },
        inputs, workload);
  } catch (...) {
    return std::unexpected(detail::native_exception());
  }
}
} // namespace hyle::compute::apple
