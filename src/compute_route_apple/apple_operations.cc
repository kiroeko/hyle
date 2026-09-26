#include "hyle/compute/apple.h"
#include "hyle/compute/detail/apple_plan.h"
#include "hyle/compute/model_ops.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

#include <mlx/ops.h>

#include "apple_internal.h"
#include "apple_tensor.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/route_policy.h"

namespace hyle::compute::apple {
namespace mx = mlx::core;
namespace {
inline constexpr std::array kAffineQ4Facts{
    RouteFactPredicate::number(
        operation_route_fact::kQuantizedWeightMode,
        static_cast<std::uint64_t>(QuantizedWeightMode::kUnsignedAffine4)),
    RouteFactPredicate::number(operation_route_fact::kQuantizationGroupSize,
                               64),
    RouteFactPredicate::number(operation_route_fact::kAccumulationDataType,
                               static_cast<std::uint64_t>(DataType::kFloat32))};
inline constexpr std::array kAffineQ4Ports{
    RouteTensorPredicate{.direction = RoutePortDirection::kInput,
                         .index = 0,
                         .data_type = DataType::kFloat32,
                         .storage_kind = RouteStorageKind::kPlain,
                         .minimum_rank = 2,
                         .maximum_rank = 2},
    RouteTensorPredicate{.direction = RoutePortDirection::kInput,
                         .index = 1,
                         .data_type = DataType::kFloat32,
                         .storage_kind = RouteStorageKind::kBlocked,
                         .storage_format = kAffineQ4Format,
                         .minimum_rank = 2,
                         .maximum_rank = 2},
    RouteTensorPredicate{.direction = RoutePortDirection::kInput,
                         .index = 2,
                         .data_type = DataType::kFloat32,
                         .storage_kind = RouteStorageKind::kPlain,
                         .minimum_rank = 2,
                         .maximum_rank = 2},
    RouteTensorPredicate{.direction = RoutePortDirection::kInput,
                         .index = 3,
                         .data_type = DataType::kFloat32,
                         .storage_kind = RouteStorageKind::kPlain,
                         .minimum_rank = 2,
                         .maximum_rank = 2},
    RouteTensorPredicate{.direction = RoutePortDirection::kOutput,
                         .index = 0,
                         .data_type = DataType::kFloat32,
                         .storage_kind = RouteStorageKind::kPlain,
                         .minimum_rank = 2,
                         .maximum_rank = 2}};
inline constexpr RouteSelector kAffineQ4Rule{
    .route = NativeRoute::kApple,
    .operation = OperationKind::kQuantizedMatMul,
    .operation_facts = kAffineQ4Facts,
    .ports = kAffineQ4Ports};

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

bool fixed_ggml_model_operation(const Operation &operation) noexcept {
  return operation.get_if<GatedDeltaNet>() ||
         operation.custom_as<model_ops::RecurrentGatedDeltaNet>() ||
         operation.custom_as<model_ops::GroupedMatMul>() ||
         operation.custom_as<model_ops::BFloatLinearF32>() ||
         operation.custom_as<model_ops::HyperConnectionCombine>() ||
         operation.custom_as<model_ops::HyperConnectionPre>() ||
         operation.custom_as<model_ops::HyperConnectionPost>() ||
         operation.custom_as<model_ops::LightningIndexer>();
}

bool ggml_model_operation(const Operation &operation,
                          std::span<const TensorDesc> inputs) noexcept {
  return fixed_ggml_model_operation(operation) ||
         (operation.get_if<MatMul>() && inputs.size() == 2 &&
          gguf_quantized_type(inputs[1].storage_format()).has_value()) ||
         (operation.get_if<Gather>() && inputs.size() == 2 &&
          gguf_quantized_type(inputs[0].storage_format()).has_value());
}

Result<DataType> logical_type(mx::Dtype type) noexcept {
  if (type == mx::float32)
    return DataType::kFloat32;
  if (type == mx::float16)
    return DataType::kFloat16;
  if (type == mx::bfloat16)
    return DataType::kBFloat16;
  if (type == mx::int32)
    return DataType::kInt32;
  if (type == mx::int64)
    return DataType::kInt64;
  if (type == mx::bool_)
    return DataType::kBool;
  return std::unexpected(Status::Unimplemented(
      "Apple input admits F32/F16/BF16/I32/I64/Bool logical storage"));
}

Result<TensorDesc> logical_description(const InputSpec &spec) noexcept {
  if (spec.logical_description.valid()) {
    HYLE_ASSIGN_OR_RETURN(auto expected, input_spec(spec.logical_description));
    if (expected.shape != spec.shape || expected.dtype != spec.dtype)
      return std::unexpected(Status::InvalidArgument(
          "Apple input carrier disagrees with its logical description"));
    return spec.logical_description;
  }
  HYLE_ASSIGN_OR_RETURN(auto type, logical_type(spec.dtype));
  std::vector<std::int64_t> shape(spec.shape.begin(), spec.shape.end());
  return TensorDesc::dense(type, shape);
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

Result<InputSpec> input_spec(const TensorDesc &description) noexcept {
  if (!description.valid())
    return std::unexpected(
        Status::InvalidArgument("Apple input requires a valid description"));
  InputSpec result;
  result.logical_description = description;
  if (gguf_quantized_type(description.storage_format())) {
    HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
    if (bytes > static_cast<std::size_t>((std::numeric_limits<int>::max)()))
      return std::unexpected(
          Status::OutOfRange("Apple packed input exceeds native shape range"));
    result.shape = {static_cast<int>(bytes)};
    result.dtype = mx::uint8;
    return result;
  }
  switch (description.data_type()) {
  case DataType::kFloat32:
    result.dtype = mx::float32;
    break;
  case DataType::kFloat16:
    result.dtype = mx::float16;
    break;
  case DataType::kBFloat16:
    result.dtype = mx::bfloat16;
    break;
  case DataType::kInt32:
    result.dtype = mx::int32;
    break;
  case DataType::kInt64:
    result.dtype = mx::int64;
    break;
  case DataType::kBool:
    result.dtype = mx::bool_;
    break;
  default:
    return std::unexpected(Status::Unimplemented(
        "Apple input requires F32/F16/BF16/I32/I64/Bool logical storage"));
  }
  for (auto dimension : description.dimensions()) {
    if (dimension > (std::numeric_limits<int>::max)())
      return std::unexpected(
          Status::OutOfRange("Apple input dimension exceeds native range"));
    result.shape.push_back(static_cast<int>(dimension));
  }
  if (!description.is_plain()) {
    if (!description.is_dense() ||
        description.data_type() != DataType::kFloat32 ||
        description.rank() != 2 ||
        description.storage_format().id() != kAffineQ4Format)
      return std::unexpected(
          Status::Unimplemented("Apple input storage encoding is not admitted"));
    HYLE_ASSIGN_OR_RETURN(
        auto expected,
        affine_q4_description(description.dimensions()[0],
                              description.dimensions()[1]));
    if (description != expected)
      return std::unexpected(Status::InvalidArgument(
          "Apple affine Q4 layout disagrees with native packing"));
    result.shape.back() /= 8;
    result.dtype = mx::uint32;
  }
  return result;
}

Result<std::vector<mx::array>> lower(const Operation &operation,
                                     std::span<const mx::array> inputs,
                                     mx::Stream stream,
                                     const WorkloadContext &workload) noexcept {
  if (!workload.valid())
    return std::unexpected(Status::InvalidArgument("invalid Apple workload"));
  try {
    const auto *qmm = operation.get_if<QuantizedMatMul>();
    if (qmm == nullptr)
      return detail::lower_tensor(operation, inputs, stream);
    if (!operation_traits(operation).inputs.accepts(inputs.size()))
      return std::unexpected(Status::InvalidArgument(
          "Apple typed operation input count mismatch"));
    if (stream.device.type != mx::Device::gpu)
      return std::unexpected(Status::InvalidArgument(
          "Apple typed lowering requires a GPU stream"));
    if (qmm->accumulation_type != DataType::kFloat32 || inputs[0].ndim() != 2 ||
        inputs[1].ndim() != 2 || inputs[1].dtype() != mx::uint32)
      return std::unexpected(Status::Unimplemented(
          "Apple Q4 admits 2-D affine group64 F32 with uint32 packed weights"));

    std::array<TensorDesc, 4> descriptions{};
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      auto description =
          i == 1 ? affine_q4_description(inputs[i].shape(0),
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

    const std::array outputs{*output};
    const RouteMatchInput route_input{
        .platform = kBuildRoutePlatform,
        .route = NativeRoute::kApple,
        .profile = workload.profile.description(),
        .phase = workload.phase,
        .region = workload.region,
        .operation = operation,
        .inputs = {descriptions.data(), inputs.size()},
        .outputs = outputs};
    if (!match_route_selector(kAffineQ4Rule, route_input).matched())
      return std::unexpected(Status::Unimplemented(
          "Apple Q4 requires affine group64 public storage"));

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
    if (fixed_ggml_model_operation(operation)) {
      std::vector<InputSpec> specs;
      specs.reserve(examples.size());
      for (const auto &input : examples) {
        if (input.status() != mx::array::available)
          return std::unexpected(Status::FailedPrecondition(
              "Apple examples must be ready arrays"));
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
Context::prepare(const Operation &operation, std::span<const InputSpec> inputs,
                 const WorkloadContext &workload) const noexcept {
  try {
    std::vector<TensorDesc> descriptions;
    descriptions.reserve(inputs.size());
    for (const auto &input : inputs) {
      HYLE_ASSIGN_OR_RETURN(auto description, logical_description(input));
      descriptions.push_back(std::move(description));
    }
    if (ggml_model_operation(operation, descriptions)) {
      detail::RegionPlan plan;
      plan.workload = workload;
      detail::Region region;
      region.kind = detail::RegionKind::kGgml;
      region.operation = operation;
      for (std::size_t i = 0; i < inputs.size(); ++i) {
        region.native_inputs.push_back(descriptions[i]);
        region.inputs.push_back(i);
        plan.inputs.push_back(i);
        plan.values.push_back(inputs[i]);
      }
      std::vector<TensorDesc> outputs(operation_traits(operation).outputs);
      HYLE_RETURN_IF_ERROR(
          infer_output_descriptions(operation, region.native_inputs, outputs));
      for (const auto &output : outputs) {
        const auto id = plan.values.size();
        region.outputs.push_back(id);
        plan.outputs.push_back(id);
        HYLE_ASSIGN_OR_RETURN(auto spec, input_spec(output));
        plan.values.push_back(std::move(spec));
      }
      plan.regions.push_back(std::move(region));
      return detail::ContextAccess::compile(*this, std::move(plan));
    }
    if (std::ranges::any_of(descriptions, [](const auto &description) {
          return gguf_quantized_type(description.storage_format()).has_value();
        }))
      return std::unexpected(Status::Unimplemented(
          "Apple GGUF carrier is only admitted by qualified native model operations"));
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
