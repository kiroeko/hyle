#include "hyle/compute/operation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>

#include "hyle/math/scalar.h"
#include "view_alias.h"

namespace hyle::compute {
namespace {

bool is_floating(DataType type) noexcept {
  switch (type) {
  case DataType::kFloat8E4M3FN:
  case DataType::kFloat8E5M2:
  case DataType::kFloat16:
  case DataType::kBFloat16:
  case DataType::kFloat32:
  case DataType::kFloat64:
    return true;
  default:
    return false;
  }
}

bool is_integer(DataType type) noexcept {
  switch (type) {
  case DataType::kInt8:
  case DataType::kUInt8:
  case DataType::kInt16:
  case DataType::kUInt16:
  case DataType::kInt32:
  case DataType::kUInt32:
  case DataType::kInt64:
  case DataType::kUInt64:
    return true;
  default:
    return false;
  }
}

bool is_index(DataType type) noexcept {
  return type == DataType::kInt32 || type == DataType::kInt64;
}

bool is_numeric(DataType type) noexcept {
  return is_floating(type) || is_integer(type);
}

bool is_quantized_matmul_float(DataType type) noexcept {
  return type == DataType::kFloat16 || type == DataType::kBFloat16 ||
         type == DataType::kFloat32 || type == DataType::kFloat64;
}

template <class Enum> bool enum_in_range(Enum value, Enum last) noexcept {
  using Underlying = std::underlying_type_t<Enum>;
  return static_cast<Underlying>(value) <= static_cast<Underlying>(last);
}

Status validate_operation_modes(const Operation &operation) noexcept {
  switch (operation.kind()) {
  case OperationKind::kQuantizedMatMul: {
    const auto &matmul = *operation.get_if<QuantizedMatMul>();
    if (!enum_in_range(matmul.mode, QuantizedWeightMode::kUnsignedAffine4) ||
        matmul.group_size == 0 ||
        !is_quantized_matmul_float(matmul.accumulation_type)) {
      return Status::InvalidArgument(
          "QuantizedMatMul requires a valid Q4 mode, nonzero group and "
          "explicit floating accumulation type");
    }
    break;
  }
  case OperationKind::kView:
    if (!enum_in_range(operation.get_if<View>()->kind(),
                       ViewKind::kBroadcast)) {
      return Status::InvalidArgument("View kind is invalid");
    }
    break;
  case OperationKind::kUnary:
    if (!enum_in_range(operation.get_if<Unary>()->kind, UnaryKind::kSoftplus)) {
      return Status::InvalidArgument("Unary kind is invalid");
    }
    break;
  case OperationKind::kBinary:
    if (!enum_in_range(operation.get_if<Binary>()->kind,
                       BinaryKind::kMaximum)) {
      return Status::InvalidArgument("Binary kind is invalid");
    }
    break;
  case OperationKind::kReduce:
    if (!enum_in_range(operation.get_if<Reduce>()->kind,
                       ReductionKind::kMaximum)) {
      return Status::InvalidArgument("Reduce kind is invalid");
    }
    break;
  case OperationKind::kGatedActivation: {
    const GatedActivation &gated = *operation.get_if<GatedActivation>();
    if (!enum_in_range(gated.activation, GatedActivationKind::kGeluTanh) ||
        !enum_in_range(gated.inputs, GatedInputKind::kSeparate)) {
      return Status::InvalidArgument("GatedActivation mode is invalid");
    }
    break;
  }
  case OperationKind::kGather: {
    const Gather &gather = *operation.get_if<Gather>();
    if (!enum_in_range(gather.bounds, IndexBoundsPolicy::kUnchecked)) {
      return Status::InvalidArgument("Gather bounds policy is invalid");
    }
    break;
  }
  case OperationKind::kScatter: {
    const Scatter &scatter = *operation.get_if<Scatter>();
    if (!enum_in_range(scatter.kind, ScatterKind::kAdd) ||
        !enum_in_range(scatter.bounds, IndexBoundsPolicy::kUnchecked)) {
      return Status::InvalidArgument("Scatter mode is invalid");
    }
    break;
  }
  case OperationKind::kRoPE:
    if (!enum_in_range(operation.get_if<RoPE>()->layout,
                       RopeLayout::kHalfSplit)) {
      return Status::InvalidArgument("RoPE layout is invalid");
    }
    break;
  case OperationKind::kSoftmax: {
    const Softmax &softmax = *operation.get_if<Softmax>();
    if (!enum_in_range(softmax.kind, SoftmaxKind::kLogSoftmax) ||
        !enum_in_range(softmax.mask, MaskKind::kBoolean)) {
      return Status::InvalidArgument("Softmax mode is invalid");
    }
    break;
  }
  case OperationKind::kScaledDotProductAttention: {
    const ScaledDotProductAttention &attention =
        *operation.get_if<ScaledDotProductAttention>();
    if (!enum_in_range(attention.layout,
                       AttentionLayout::kBatchHeadSequenceDepth) ||
        !enum_in_range(attention.mask, MaskKind::kBoolean)) {
      return Status::InvalidArgument("SDPA mode is invalid");
    }
    break;
  }
  case OperationKind::kUnknown:
  case OperationKind::kCopy:
  case OperationKind::kFill:
  case OperationKind::kCast:
  case OperationKind::kSelect:
  case OperationKind::kRmsNorm:
  case OperationKind::kLayerNorm:
  case OperationKind::kMatMul:
  case OperationKind::kConcat:
  case OperationKind::kScan:
  case OperationKind::kTopK:
  case OperationKind::kCustom:
    break;
  }
  return Status::Ok();
}

Status validate_description(const TensorDesc &description) noexcept {
  if (!description.valid()) {
    return Status::InvalidArgument("operation tensor description is invalid");
  }
  if (data_type_size(description.data_type()) == 0) {
    return Status::InvalidArgument("operation tensor data type is invalid");
  }
  return Status::Ok();
}

Status validate_accumulation(DataType type) noexcept {
  if (type == DataType::kUnknown)
    return Status::Ok();
  if (!is_numeric(type)) {
    return Status::InvalidArgument(
        "operation accumulation type must be numeric");
  }
  return Status::Ok();
}

Result<std::size_t> normalize_axis(std::int32_t axis,
                                   std::size_t rank) noexcept {
  if (rank == 0) {
    return std::unexpected(
        Status::InvalidArgument("operation axis requires rank >= 1"));
  }
  const std::int64_t signed_rank = static_cast<std::int64_t>(rank);
  const std::int64_t normalized =
      axis < 0 ? signed_rank + static_cast<std::int64_t>(axis)
               : static_cast<std::int64_t>(axis);
  if (normalized < 0 || normalized >= signed_rank) {
    return std::unexpected(
        Status::OutOfRange("operation axis is outside tensor rank"));
  }
  return static_cast<std::size_t>(normalized);
}

Result<TensorDesc> dense_like(const TensorDesc &input, DataType type) noexcept {
  return TensorDesc::dense(type, input.dimensions());
}

Result<TensorShape>
broadcast_shape(std::span<const std::int64_t> left,
                std::span<const std::int64_t> right) noexcept {
  const std::size_t rank = (std::max)(left.size(), right.size());
  if (rank > kMaxTensorRank) {
    return std::unexpected(
        Status::InvalidArgument("broadcast result rank exceeds limit"));
  }
  std::array<std::int64_t, kMaxTensorRank> dimensions{};
  for (std::size_t reverse = 0; reverse < rank; ++reverse) {
    const std::int64_t left_extent =
        reverse < left.size() ? left[left.size() - 1 - reverse] : 1;
    const std::int64_t right_extent =
        reverse < right.size() ? right[right.size() - 1 - reverse] : 1;
    std::int64_t extent = 0;
    if (left_extent == right_extent) {
      extent = left_extent;
    } else if (left_extent == 1) {
      extent = right_extent;
    } else if (right_extent == 1) {
      extent = left_extent;
    } else {
      return std::unexpected(
          Status::InvalidArgument("tensor dimensions cannot be broadcast"));
    }
    dimensions[rank - 1 - reverse] = extent;
  }
  return TensorShape::create(
      std::span<const std::int64_t>(dimensions.data(), rank));
}

Status validate_broadcast_to(std::span<const std::int64_t> source,
                             std::span<const std::int64_t> target) noexcept {
  auto result = broadcast_shape(source, target);
  if (!result)
    return std::move(result).error();
  if (!std::ranges::equal(result->dimensions(), target)) {
    return Status::InvalidArgument(
        "tensor is not broadcast-compatible with target shape");
  }
  return Status::Ok();
}

Result<TensorDesc> infer_view(const View &view,
                              const TensorDesc &input) noexcept {
  switch (view.kind()) {
  case ViewKind::kReshape: {
    if (!view.shape().valid()) {
      return std::unexpected(
          Status::InvalidArgument("reshape target shape is invalid"));
    }
    if (!input.is_plain() || !input.is_dense()) {
      return std::unexpected(Status::Unimplemented(
          "generic zero-copy reshape requires a dense plain input"));
    }
    auto source_count = input.element_count();
    if (!source_count)
      return std::unexpected(std::move(source_count).error());
    auto target_count = view.shape().element_count();
    if (!target_count)
      return std::unexpected(std::move(target_count).error());
    if (*source_count != *target_count) {
      return std::unexpected(Status::InvalidArgument(
          "reshape source and target element counts differ"));
    }
    return TensorDesc::dense(input.data_type(), view.shape().dimensions());
  }
  case ViewKind::kSlice: {
    if (!input.is_plain()) {
      return std::unexpected(Status::Unimplemented(
          "generic zero-copy slice requires a plain input"));
    }
    auto axis = normalize_axis(view.axis(), input.rank());
    if (!axis)
      return std::unexpected(std::move(axis).error());
    if (view.start() < 0 || view.count() < 0 || view.step() <= 0) {
      return std::unexpected(Status::InvalidArgument(
          "slice start/count must be non-negative and step positive"));
    }
    const std::int64_t extent = input.dimensions()[*axis];
    if (view.count() != 0) {
      if (view.start() >= extent) {
        return std::unexpected(
            Status::OutOfRange("slice starts outside its input dimension"));
      }
      const std::int64_t available_steps =
          (extent - 1 - view.start()) / view.step();
      if (view.count() - 1 > available_steps) {
        return std::unexpected(
            Status::OutOfRange("slice extends outside its input dimension"));
      }
    } else if (view.start() > extent) {
      return std::unexpected(
          Status::OutOfRange("empty slice starts outside its input dimension"));
    }

    std::array<std::int64_t, kMaxTensorRank> dimensions{};
    std::array<std::int64_t, kMaxTensorRank> strides{};
    std::copy(input.dimensions().begin(), input.dimensions().end(),
              dimensions.begin());
    std::copy(input.byte_strides().begin(), input.byte_strides().end(),
              strides.begin());
    dimensions[*axis] = view.count();
    auto stepped_stride = math::try_multiply(strides[*axis], view.step());
    if (!stepped_stride) {
      return std::unexpected(
          Status::OutOfRange("slice byte stride overflows int64"));
    }
    strides[*axis] = *stepped_stride;
    return TensorDesc::strided_bytes(
        input.data_type(),
        std::span<const std::int64_t>(dimensions.data(), input.rank()),
        std::span<const std::int64_t>(strides.data(), input.rank()));
  }
  case ViewKind::kPermute: {
    if (view.permutation().size() != input.rank()) {
      return std::unexpected(
          Status::InvalidArgument("permutation rank does not match its input"));
    }
    std::array<std::size_t, kMaxTensorRank> normalized{};
    std::array<bool, kMaxTensorRank> seen{};
    for (std::size_t index = 0; index < input.rank(); ++index) {
      auto axis = normalize_axis(view.permutation()[index], input.rank());
      if (!axis)
        return std::unexpected(std::move(axis).error());
      if (seen[*axis]) {
        return std::unexpected(
            Status::InvalidArgument("permutation contains duplicate axes"));
      }
      seen[*axis] = true;
      normalized[index] = *axis;
    }
    return input.permuted(
        std::span<const std::size_t>(normalized.data(), input.rank()));
  }
  case ViewKind::kBroadcast: {
    if (!view.shape().valid()) {
      return std::unexpected(
          Status::InvalidArgument("broadcast target shape is invalid"));
    }
    if (!input.is_plain()) {
      return std::unexpected(Status::Unimplemented(
          "generic zero-copy broadcast requires a plain input"));
    }
    const auto target = view.shape().dimensions();
    if (target.size() < input.rank()) {
      return std::unexpected(Status::InvalidArgument(
          "broadcast target rank is smaller than input rank"));
    }
    std::array<std::int64_t, kMaxTensorRank> strides{};
    const std::size_t prefix = target.size() - input.rank();
    for (std::size_t axis = 0; axis < target.size(); ++axis) {
      if (axis < prefix) {
        strides[axis] = 0;
        continue;
      }
      const std::size_t input_axis = axis - prefix;
      const std::int64_t source_extent = input.dimensions()[input_axis];
      if (source_extent == target[axis]) {
        strides[axis] = input.byte_strides()[input_axis];
      } else if (source_extent == 1) {
        strides[axis] = 0;
      } else {
        return std::unexpected(Status::InvalidArgument(
            "input dimension cannot broadcast to target"));
      }
    }
    return TensorDesc::strided_bytes(
        input.data_type(), target,
        std::span<const std::int64_t>(strides.data(), target.size()));
  }
  }
  return std::unexpected(Status::InvalidArgument("view kind is invalid"));
}

Result<TensorDesc> infer_binary(const TensorDesc &left,
                                const TensorDesc &right) noexcept {
  if (left.data_type() != right.data_type()) {
    return std::unexpected(
        Status::InvalidArgument("binary input data types differ"));
  }
  auto shape = broadcast_shape(left.dimensions(), right.dimensions());
  if (!shape)
    return std::unexpected(std::move(shape).error());
  return TensorDesc::dense(left.data_type(), shape->dimensions());
}

Status validate_vector_parameter(const TensorDesc &parameter,
                                 const TensorDesc &input,
                                 std::string_view name) noexcept {
  const bool promoted_parameter = (input.data_type() == DataType::kFloat16 ||
                                   input.data_type() == DataType::kBFloat16) &&
                                  parameter.data_type() == DataType::kFloat32;
  if ((!promoted_parameter && parameter.data_type() != input.data_type()) ||
      parameter.rank() != 1 ||
      parameter.dimensions()[0] != input.dimensions().back()) {
    return Status::InvalidArgument(name);
  }
  return Status::Ok();
}

Result<TensorDesc> infer_normalization(const TensorDesc &input,
                                       std::span<const TensorDesc> parameters,
                                       bool allow_bias) noexcept {
  if (!is_floating(input.data_type()) || input.rank() == 0) {
    return std::unexpected(Status::InvalidArgument(
        "normalization requires a floating tensor with rank >= 1"));
  }
  if (!allow_bias && parameters.size() > 1) {
    return std::unexpected(
        Status::InvalidArgument("RMSNorm accepts at most one scale tensor"));
  }
  if (allow_bias && parameters.size() > 2) {
    return std::unexpected(Status::InvalidArgument(
        "LayerNorm accepts at most scale and bias tensors"));
  }
  if (!parameters.empty()) {
    Status status = validate_vector_parameter(
        parameters[0], input,
        "normalization scale must match the last input dimension");
    if (!status.ok())
      return std::unexpected(std::move(status));
  }
  if (parameters.size() == 2) {
    Status status = validate_vector_parameter(
        parameters[1], input,
        "LayerNorm bias must match the last input dimension");
    if (!status.ok())
      return std::unexpected(std::move(status));
  }
  return dense_like(input, input.data_type());
}

Result<TensorDesc>
infer_quantized_matmul(const QuantizedMatMul &operation,
                       std::span<const TensorDesc> inputs) noexcept {
  const TensorDesc &activation = inputs[0];
  const TensorDesc &weight = inputs[1];
  if (!is_quantized_matmul_float(activation.data_type()) ||
      activation.rank() < 2 || weight.rank() != 2 || weight.is_plain()) {
    return std::unexpected(Status::InvalidArgument(
        "QuantizedMatMul requires floating rank >= 2 activations and "
        "packed logical [N,K] weights"));
  }
  for (const TensorDesc &input : inputs) {
    if (input.data_type() != activation.data_type()) {
      return std::unexpected(Status::InvalidArgument(
          "QuantizedMatMul inputs must share one floating logical type"));
    }
  }
  const std::int64_t channels = weight.dimensions()[0];
  const std::int64_t reduction = weight.dimensions()[1];
  if (reduction <= 0 || reduction != activation.dimensions().back() ||
      reduction % operation.group_size != 0) {
    return std::unexpected(Status::InvalidArgument(
        "QuantizedMatMul K must match, be positive and divisible by group"));
  }
  const std::int64_t groups = reduction / operation.group_size;
  for (const TensorDesc &parameter : inputs.subspan(2)) {
    if (parameter.rank() != 2 || parameter.dimensions()[0] != channels ||
        parameter.dimensions()[1] != groups) {
      return std::unexpected(Status::InvalidArgument(
          "QuantizedMatMul scales and biases must have logical shape "
          "[N,K/group]"));
    }
  }
  std::array<std::int64_t, kMaxTensorRank> shape{};
  std::copy(activation.dimensions().begin(), activation.dimensions().end(),
            shape.begin());
  shape[activation.rank() - 1] = channels;
  return TensorDesc::dense(
      activation.data_type(),
      std::span<const std::int64_t>(shape.data(), activation.rank()));
}

Result<TensorDesc> infer_matmul(const TensorDesc &left,
                                const TensorDesc &right) noexcept {
  if (left.data_type() != right.data_type() || !is_numeric(left.data_type())) {
    return std::unexpected(Status::InvalidArgument(
        "MatMul inputs must have the same numeric logical data type"));
  }
  if (left.rank() < 2 || right.rank() < 2) {
    return std::unexpected(
        Status::InvalidArgument("MatMul inputs require rank >= 2"));
  }
  const auto left_dimensions = left.dimensions();
  const auto right_dimensions = right.dimensions();
  if (left_dimensions[left.rank() - 1] != right_dimensions[right.rank() - 2]) {
    return std::unexpected(
        Status::InvalidArgument("MatMul inner dimensions do not match"));
  }

  const auto left_batch = left_dimensions.first(left.rank() - 2);
  const auto right_batch = right_dimensions.first(right.rank() - 2);
  auto batch_shape = broadcast_shape(left_batch, right_batch);
  if (!batch_shape)
    return std::unexpected(std::move(batch_shape).error());
  if (batch_shape->rank() + 2 > kMaxTensorRank) {
    return std::unexpected(
        Status::InvalidArgument("MatMul output rank exceeds limit"));
  }
  std::array<std::int64_t, kMaxTensorRank> output{};
  std::copy(batch_shape->dimensions().begin(), batch_shape->dimensions().end(),
            output.begin());
  output[batch_shape->rank()] = left_dimensions[left.rank() - 2];
  output[batch_shape->rank() + 1] = right_dimensions[right.rank() - 1];
  return TensorDesc::dense(
      left.data_type(),
      std::span<const std::int64_t>(output.data(), batch_shape->rank() + 2));
}

Result<TensorShape> gather_shape(const TensorDesc &source,
                                 const TensorDesc &indices,
                                 std::size_t axis) noexcept {
  const std::size_t rank = source.rank() - 1 + indices.rank();
  if (rank > kMaxTensorRank) {
    return std::unexpected(
        Status::InvalidArgument("Gather output rank exceeds limit"));
  }
  std::array<std::int64_t, kMaxTensorRank> output{};
  std::size_t cursor = 0;
  for (std::size_t index = 0; index < axis; ++index) {
    output[cursor++] = source.dimensions()[index];
  }
  for (const std::int64_t extent : indices.dimensions()) {
    output[cursor++] = extent;
  }
  for (std::size_t index = axis + 1; index < source.rank(); ++index) {
    output[cursor++] = source.dimensions()[index];
  }
  return TensorShape::create(
      std::span<const std::int64_t>(output.data(), cursor));
}

Status validate_rope(const RoPE &rope,
                     std::span<const TensorDesc> inputs) noexcept {
  const TensorDesc &data = inputs[0];
  const TensorDesc &positions = inputs[1];
  const TensorDesc &frequencies = inputs[2];
  if (!is_floating(data.data_type()) || !is_floating(frequencies.data_type()) ||
      !is_index(positions.data_type())) {
    return Status::InvalidArgument(
        "RoPE requires floating data/frequencies and i32/i64 positions");
  }
  if (data.rank() < 2 || rope.rotary_dimensions == 0 ||
      (rope.rotary_dimensions & 1u) != 0 ||
      rope.rotary_dimensions >
          static_cast<std::size_t>(data.dimensions().back())) {
    return Status::InvalidArgument(
        "RoPE rotary dimension must be positive, even and fit the last axis");
  }
  auto sequence_axis = normalize_axis(rope.sequence_axis, data.rank());
  if (!sequence_axis)
    return std::move(sequence_axis).error();
  if (*sequence_axis == data.rank() - 1) {
    return Status::InvalidArgument(
        "RoPE sequence axis cannot be the rotary depth axis");
  }
  const std::int64_t sequence_extent = data.dimensions()[*sequence_axis];
  if (positions.rank() == 0 ||
      (positions.rank() == 1 && positions.dimensions()[0] != sequence_extent)) {
    return Status::InvalidArgument(
        "RoPE positions must include the complete sequence axis");
  }
  if (positions.rank() > 1) {
    if (positions.rank() != *sequence_axis + 1) {
      return Status::InvalidArgument(
          "RoPE batched positions must match the data prefix through sequence");
    }
    for (std::size_t axis = 0; axis < positions.rank(); ++axis) {
      if (positions.dimensions()[axis] != data.dimensions()[axis]) {
        return Status::InvalidArgument(
            "RoPE batched position shape does not match data prefix");
      }
    }
  }
  if (frequencies.rank() == 0 ||
      frequencies.dimensions().back() !=
          static_cast<std::int64_t>(rope.rotary_dimensions / 2)) {
    return Status::InvalidArgument(
        "RoPE frequency last dimension must equal rotary_dimensions / 2");
  }
  if (frequencies.rank() > 2 ||
      (frequencies.rank() == 2 &&
       frequencies.dimensions()[0] != sequence_extent)) {
    return Status::InvalidArgument(
        "RoPE frequencies must be [pairs] or [sequence,pairs]");
  }
  return Status::Ok();
}

Status validate_mask(const TensorDesc &mask, MaskKind kind,
                     std::span<const std::int64_t> target) noexcept {
  if (kind == MaskKind::kBoolean) {
    if (mask.data_type() != DataType::kBool) {
      return Status::InvalidArgument("boolean mask requires Bool data type");
    }
  } else if (kind == MaskKind::kAdditive) {
    if (!is_floating(mask.data_type())) {
      return Status::InvalidArgument(
          "additive mask requires a floating data type");
    }
  } else {
    return Status::InvalidArgument("mask input supplied for no-mask operation");
  }
  return validate_broadcast_to(mask.dimensions(), target);
}

Result<TensorDesc>
infer_attention(const ScaledDotProductAttention &attention,
                std::span<const TensorDesc> inputs) noexcept {
  const TensorDesc &query = inputs[0];
  const TensorDesc &key = inputs[1];
  const TensorDesc &value = inputs[2];
  if (query.rank() != 4 || key.rank() != 4 || value.rank() != 4 ||
      query.data_type() != key.data_type() ||
      query.data_type() != value.data_type() ||
      !is_floating(query.data_type())) {
    return std::unexpected(Status::InvalidArgument(
        "SDPA requires three rank-4 tensors with one floating data type"));
  }
  if (!std::isfinite(attention.scale)) {
    return std::unexpected(
        Status::InvalidArgument("SDPA scale must be finite"));
  }
  Status accumulation = validate_accumulation(attention.accumulation_type);
  if (!accumulation.ok())
    return std::unexpected(std::move(accumulation));

  std::size_t sequence_axis = 1;
  std::size_t head_axis = 2;
  if (attention.layout == AttentionLayout::kBatchHeadSequenceDepth) {
    sequence_axis = 2;
    head_axis = 1;
  }
  const auto query_heads = query.dimensions()[head_axis];
  const auto kv_heads = key.dimensions()[head_axis];
  if (query_heads <= 0 || kv_heads <= 0 || query_heads % kv_heads != 0 ||
      query.dimensions()[0] != key.dimensions()[0] ||
      query.dimensions()[0] != value.dimensions()[0] ||
      kv_heads != value.dimensions()[head_axis] ||
      key.dimensions()[sequence_axis] != value.dimensions()[sequence_axis] ||
      query.dimensions()[3] != key.dimensions()[3]) {
    return std::unexpected(Status::InvalidArgument(
        "SDPA batch/head/sequence/depth shapes disagree"));
  }

  std::array<std::int64_t, 4> output{
      query.dimensions()[0], query.dimensions()[1], query.dimensions()[2],
      value.dimensions()[3]};
  std::array<std::int64_t, 4> scores{
      query.dimensions()[0], query.dimensions()[head_axis],
      query.dimensions()[sequence_axis], key.dimensions()[sequence_axis]};
  if (attention.mask != MaskKind::kNone) {
    Status mask = validate_mask(inputs[3], attention.mask, scores);
    if (!mask.ok())
      return std::unexpected(std::move(mask));
  }
  return TensorDesc::dense(query.data_type(), output);
}

} // namespace

Result<Axes> Axes::from(std::span<const std::int32_t> axes) noexcept {
  if (axes.empty()) {
    return std::unexpected(
        Status::InvalidArgument("explicit axis list cannot be empty"));
  }
  if (axes.size() > kMaxTensorRank) {
    return std::unexpected(
        Status::InvalidArgument("axis list exceeds kMaxTensorRank"));
  }
  Axes result;
  result.all_ = false;
  result.count_ = static_cast<std::uint8_t>(axes.size());
  for (std::size_t index = 0; index < axes.size(); ++index) {
    if (axes[index] < -static_cast<std::int32_t>(kMaxTensorRank) ||
        axes[index] >= static_cast<std::int32_t>(kMaxTensorRank)) {
      return std::unexpected(
          Status::OutOfRange("axis cannot address a supported tensor rank"));
    }
    result.values_[index] = static_cast<std::int8_t>(axes[index]);
  }
  return result;
}

Result<View> View::reshape(TensorShape shape) noexcept {
  if (!shape.valid()) {
    return std::unexpected(
        Status::InvalidArgument("reshape target shape is invalid"));
  }
  View result;
  result.kind_ = ViewKind::kReshape;
  result.shape_ = shape;
  return result;
}

Result<View> View::slice(std::int32_t axis, std::int64_t start,
                         std::int64_t count, std::int64_t step) noexcept {
  if (axis < -static_cast<std::int32_t>(kMaxTensorRank) ||
      axis >= static_cast<std::int32_t>(kMaxTensorRank) || start < 0 ||
      count < 0 || step <= 0) {
    return std::unexpected(Status::InvalidArgument(
        "slice axis/start/count/step is outside its static contract"));
  }
  View result;
  result.kind_ = ViewKind::kSlice;
  result.axis_ = axis;
  result.start_ = start;
  result.count_ = count;
  result.step_ = step;
  return result;
}

Result<View> View::permute(std::span<const std::int32_t> axes) noexcept {
  if (axes.size() > kMaxTensorRank) {
    return std::unexpected(
        Status::InvalidArgument("permutation rank exceeds limit"));
  }
  View result;
  result.kind_ = ViewKind::kPermute;
  result.permutation_count_ = static_cast<std::uint8_t>(axes.size());
  for (std::size_t index = 0; index < axes.size(); ++index) {
    if (axes[index] < -static_cast<std::int32_t>(kMaxTensorRank) ||
        axes[index] >= static_cast<std::int32_t>(kMaxTensorRank)) {
      return std::unexpected(
          Status::OutOfRange("permutation axis exceeds supported rank"));
    }
    result.permutation_[index] = static_cast<std::int8_t>(axes[index]);
  }
  return result;
}

Result<View> View::broadcast_to(TensorShape shape) noexcept {
  if (!shape.valid()) {
    return std::unexpected(
        Status::InvalidArgument("broadcast target shape is invalid"));
  }
  View result;
  result.kind_ = ViewKind::kBroadcast;
  result.shape_ = shape;
  return result;
}

std::string_view operation_name(OperationKind operation) noexcept {
  switch (operation) {
  case OperationKind::kView:
    return "view";
  case OperationKind::kCopy:
    return "copy";
  case OperationKind::kFill:
    return "fill";
  case OperationKind::kCast:
    return "cast";
  case OperationKind::kUnary:
    return "unary";
  case OperationKind::kBinary:
    return "binary";
  case OperationKind::kSelect:
    return "select";
  case OperationKind::kReduce:
    return "reduce";
  case OperationKind::kScan:
    return "scan";
  case OperationKind::kRmsNorm:
    return "rms-norm";
  case OperationKind::kLayerNorm:
    return "layer-norm";
  case OperationKind::kMatMul:
    return "matmul";
  case OperationKind::kQuantizedMatMul:
    return "quantized-matmul";
  case OperationKind::kGatedActivation:
    return "gated-activation";
  case OperationKind::kGather:
    return "gather";
  case OperationKind::kScatter:
    return "scatter";
  case OperationKind::kConcat:
    return "concat";
  case OperationKind::kRoPE:
    return "rope";
  case OperationKind::kSoftmax:
    return "softmax";
  case OperationKind::kTopK:
    return "top-k";
  case OperationKind::kScaledDotProductAttention:
    return "scaled-dot-product-attention";
  case OperationKind::kCustom:
    return "custom";
  case OperationKind::kUnknown:
    break;
  }
  return "unknown";
}

std::string_view operation_name(const Operation &operation) noexcept {
  if (operation.kind() == OperationKind::kCustom) {
    const auto *value =
        std::get_if<detail::CustomOperationValue>(&operation.value_);
    return value != nullptr && value->vtable != nullptr
               ? value->vtable->name
               : std::string_view("custom");
  }
  return operation_name(operation.kind());
}

OperationTraits operation_traits(const Operation &operation) noexcept {
  OperationTraits result;
  switch (operation.kind()) {
  case OperationKind::kScan:
    return {{1, 1},
            1,
            AliasPolicy::kMayAliasInput0,
            EffectKind::kPure,
            PrecisionPolicy::kAccumulateInOutputType,
            DataType::kUnknown,
            true};
  case OperationKind::kView:
    return {{1, 1},
            1,
            AliasPolicy::kMustAliasInput0,
            EffectKind::kPure,
            PrecisionPolicy::kExact,
            DataType::kUnknown,
            false};
  case OperationKind::kCopy:
    return {{1, 1},
            1,
            AliasPolicy::kNone,
            EffectKind::kPure,
            PrecisionPolicy::kExact,
            DataType::kUnknown,
            true};
  case OperationKind::kFill:
    return {{0, 0},
            1,
            AliasPolicy::kNone,
            EffectKind::kPure,
            PrecisionPolicy::kExact,
            DataType::kUnknown,
            true};
  case OperationKind::kCast:
    return {{1, 1},
            1,
            AliasPolicy::kNone,
            EffectKind::kPure,
            PrecisionPolicy::kExact,
            DataType::kUnknown,
            true};
  case OperationKind::kUnary:
    return {{1, 1},
            1,
            AliasPolicy::kMayAliasInput0,
            EffectKind::kPure,
            PrecisionPolicy::kExact,
            DataType::kUnknown,
            true};
  case OperationKind::kBinary:
    return {{2, 2},
            1,
            AliasPolicy::kMayAliasAnyInput,
            EffectKind::kPure,
            PrecisionPolicy::kExact,
            DataType::kUnknown,
            true};
  case OperationKind::kSelect:
    return {{3, 3},
            1,
            AliasPolicy::kMayAliasAnyInput,
            EffectKind::kPure,
            PrecisionPolicy::kExact,
            DataType::kUnknown,
            true};
  case OperationKind::kReduce: {
    const auto *reduce = operation.get_if<Reduce>();
    const DataType accumulation =
        reduce != nullptr ? reduce->accumulation_type : DataType::kUnknown;
    return {{1, 1},
            1,
            AliasPolicy::kNone,
            EffectKind::kPure,
            accumulation == DataType::kUnknown
                ? PrecisionPolicy::kAccumulateInOutputType
                : PrecisionPolicy::kAccumulateInDeclaredType,
            accumulation,
            true};
  }
  case OperationKind::kRmsNorm: {
    const auto *norm = operation.get_if<RmsNorm>();
    return {{1, 2},
            1,
            AliasPolicy::kMayAliasInput0,
            EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType,
            norm != nullptr ? norm->accumulation_type : DataType::kUnknown,
            true};
  }
  case OperationKind::kLayerNorm: {
    const auto *norm = operation.get_if<LayerNorm>();
    return {{1, 3},
            1,
            AliasPolicy::kMayAliasInput0,
            EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType,
            norm != nullptr ? norm->accumulation_type : DataType::kUnknown,
            true};
  }
  case OperationKind::kMatMul: {
    const auto *matmul = operation.get_if<MatMul>();
    return {{2, 2},
            1,
            AliasPolicy::kNone,
            EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType,
            matmul != nullptr ? matmul->accumulation_type : DataType::kUnknown,
            true};
  }
  case OperationKind::kQuantizedMatMul: {
    const auto &matmul = *operation.get_if<QuantizedMatMul>();
    const std::uint16_t count =
        matmul.mode == QuantizedWeightMode::kUnsignedAffine4 ? 4 : 3;
    return {{count, count},
            1,
            AliasPolicy::kNone,
            EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType,
            matmul.accumulation_type,
            true};
  }
  case OperationKind::kGatedActivation: {
    const auto *gated = operation.get_if<GatedActivation>();
    const std::uint16_t count =
        gated != nullptr && gated->inputs == GatedInputKind::kSeparate ? 2 : 1;
    return {{count, count},
            1,
            AliasPolicy::kNone,
            EffectKind::kPure,
            PrecisionPolicy::kExact,
            DataType::kUnknown,
            true};
  }
  case OperationKind::kGather:
    return {{2, 2},
            1,
            AliasPolicy::kNone,
            EffectKind::kPure,
            PrecisionPolicy::kExact,
            DataType::kUnknown,
            true};
  case OperationKind::kScatter:
    return {{3, 3},
            1,
            AliasPolicy::kMayAliasInput0,
            EffectKind::kPure,
            PrecisionPolicy::kExact,
            DataType::kUnknown,
            true};
  case OperationKind::kConcat:
    return {{1, InputArity::kVariadic},
            1,
            AliasPolicy::kNone,
            EffectKind::kPure,
            PrecisionPolicy::kExact,
            DataType::kUnknown,
            true};
  case OperationKind::kRoPE:
    return {{3, 3},
            1,
            AliasPolicy::kMayAliasInput0,
            EffectKind::kPure,
            PrecisionPolicy::kExact,
            DataType::kUnknown,
            true};
  case OperationKind::kSoftmax: {
    const auto *softmax = operation.get_if<Softmax>();
    const std::uint16_t count =
        softmax != nullptr && softmax->mask != MaskKind::kNone ? 2 : 1;
    return {{count, count},
            1,
            AliasPolicy::kMayAliasInput0,
            EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType,
            softmax != nullptr ? softmax->accumulation_type
                               : DataType::kUnknown,
            true};
  }
  case OperationKind::kTopK:
    return {{1, 1},
            2,
            AliasPolicy::kNone,
            EffectKind::kPure,
            PrecisionPolicy::kExact,
            DataType::kUnknown,
            true};
  case OperationKind::kScaledDotProductAttention: {
    const auto *attention = operation.get_if<ScaledDotProductAttention>();
    const std::uint16_t count =
        attention != nullptr && attention->mask != MaskKind::kNone ? 4 : 3;
    return {{count, count},
            1,
            AliasPolicy::kNone,
            EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType,
            attention != nullptr ? attention->accumulation_type
                                 : DataType::kUnknown,
            true};
  }
  case OperationKind::kCustom: {
    const auto *value =
        std::get_if<detail::CustomOperationValue>(&operation.value_);
    if (value != nullptr && value->vtable != nullptr &&
        value->vtable->traits != nullptr) {
      return value->vtable->traits(value->descriptor.data());
    }
    return {};
  }
  case OperationKind::kUnknown:
    return {};
  }
  return result;
}

Result<ResourceAccess> operation_resource_access(const Operation &operation,
                                                 std::size_t index) noexcept {
  if (!operation.valid())
    return std::unexpected(Status::InvalidArgument("operation is invalid"));
  const OperationTraits traits = operation_traits(operation);
  if (traits.effect != EffectKind::kResourceAccess ||
      traits.resource_count == 0) {
    return std::unexpected(
        Status::InvalidArgument("operation does not declare resource ports"));
  }
  if (index >= traits.resource_count) {
    return std::unexpected(
        Status::OutOfRange("operation resource index is out of range"));
  }
  const auto *value =
      std::get_if<detail::CustomOperationValue>(&operation.value_);
  if (value == nullptr || value->vtable == nullptr ||
      value->vtable->resource_access == nullptr) {
    return std::unexpected(
        Status::InvalidArgument("resource operation has no access schema"));
  }
  auto access = value->vtable->resource_access(value->descriptor.data(), index);
  if (!access)
    return std::unexpected(std::move(access).error());
  switch (*access) {
  case ResourceAccess::kRead:
  case ResourceAccess::kWrite:
  case ResourceAccess::kReadWrite:
    return *access;
  }
  return std::unexpected(
      Status::InvalidArgument("operation resource access mode is invalid"));
}

Status validate_operation_effects(const Operation &operation) noexcept {
  if (!operation.valid())
    return Status::InvalidArgument("operation is invalid");
  const OperationTraits traits = operation_traits(operation);
  switch (traits.effect) {
  case EffectKind::kPure:
  case EffectKind::kExternalSideEffect:
    if (traits.resource_count != 0) {
      return Status::InvalidArgument(
          "only resource-access operations may declare resource ports");
    }
    return Status::Ok();
  case EffectKind::kResourceAccess:
    if (traits.resource_count == 0) {
      return Status::InvalidArgument(
          "resource-access operation must declare at least one resource port");
    }
    for (std::size_t index = 0; index < traits.resource_count; ++index) {
      auto access = operation_resource_access(operation, index);
      if (!access)
        return std::move(access).error();
    }
    return Status::Ok();
  }
  return Status::InvalidArgument("operation effect kind is invalid");
}

Status infer_output_descriptions(const Operation &operation,
                                 std::span<const TensorDesc> inputs,
                                 std::span<TensorDesc> outputs) noexcept {
  HYLE_RETURN_IF_ERROR(validate_operation_effects(operation));
  Status mode_status = validate_operation_modes(operation);
  if (!mode_status.ok())
    return mode_status;
  const OperationTraits traits = operation_traits(operation);
  if (!traits.inputs.accepts(inputs.size())) {
    return Status::InvalidArgument(
        "operation input count does not match its typed contract");
  }
  if (outputs.size() != traits.outputs) {
    return Status::InvalidArgument(
        "operation output span must have the exact output count");
  }
  for (const TensorDesc &input : inputs) {
    Status status = validate_description(input);
    if (!status.ok())
      return status;
  }

  if (operation.kind() == OperationKind::kCustom) {
    const auto *value =
        std::get_if<detail::CustomOperationValue>(&operation.value_);
    if (value == nullptr || value->vtable == nullptr ||
        value->vtable->infer == nullptr) {
      return Status::Internal("custom operation vtable is invalid");
    }
    return value->vtable->infer(value->descriptor.data(), inputs, outputs);
  }

  std::array<TensorDesc, 2> inferred{};
  Result<TensorDesc> first = std::unexpected(
      Status::Internal("operation inference route was not selected"));

  switch (operation.kind()) {
  case OperationKind::kView:
    first = infer_view(*operation.get_if<View>(), inputs[0]);
    break;
  case OperationKind::kCopy:
    if (!inputs[0].is_plain()) {
      first = std::unexpected(Status::Unimplemented(
          "Copy v1 handles plain dense/strided storage; use Repack for "
          "blocked or opaque storage"));
    } else {
      first = dense_like(inputs[0], inputs[0].data_type());
    }
    break;
  case OperationKind::kFill: {
    const Fill &fill = *operation.get_if<Fill>();
    if (!fill.output_shape.valid() || data_type_size(fill.output_type) == 0 ||
        !fill.value.valid() || data_type_size(fill.value.data_type()) == 0 ||
        fill.output_type != fill.value.data_type()) {
      first = std::unexpected(Status::InvalidArgument(
          "Fill requires a valid shape and exactly typed scalar"));
    } else {
      first =
          TensorDesc::dense(fill.output_type, fill.output_shape.dimensions());
    }
    break;
  }
  case OperationKind::kCast: {
    const Cast &cast = *operation.get_if<Cast>();
    if (data_type_size(cast.output_type) == 0 || !inputs[0].is_plain()) {
      first = std::unexpected(Status::InvalidArgument(
          "Cast requires a valid destination type and plain input"));
    } else {
      first = dense_like(inputs[0], cast.output_type);
    }
    break;
  }
  case OperationKind::kUnary: {
    const UnaryKind kind = operation.get_if<Unary>()->kind;
    const DataType type = inputs[0].data_type();
    const bool broad_numeric = kind == UnaryKind::kAbs ||
                               kind == UnaryKind::kNeg ||
                               kind == UnaryKind::kRelu;
    if ((!broad_numeric && !is_floating(type)) ||
        (broad_numeric && !is_numeric(type))) {
      first = std::unexpected(Status::InvalidArgument(
          "Unary mode does not support the input data type"));
    } else {
      first = dense_like(inputs[0], type);
    }
    break;
  }
  case OperationKind::kBinary:
    if (!is_numeric(inputs[0].data_type())) {
      first = std::unexpected(
          Status::InvalidArgument("Binary requires numeric inputs"));
    } else {
      first = infer_binary(inputs[0], inputs[1]);
    }
    break;
  case OperationKind::kSelect: {
    if (inputs[0].data_type() != DataType::kBool ||
        inputs[1].data_type() != inputs[2].data_type()) {
      first = std::unexpected(Status::InvalidArgument(
          "Select requires Bool condition and equal branch types"));
      break;
    }
    auto branch_shape =
        broadcast_shape(inputs[1].dimensions(), inputs[2].dimensions());
    if (!branch_shape) {
      first = std::unexpected(std::move(branch_shape).error());
      break;
    }
    Status condition = validate_broadcast_to(inputs[0].dimensions(),
                                             branch_shape->dimensions());
    if (!condition.ok()) {
      first = std::unexpected(std::move(condition));
      break;
    }
    first =
        TensorDesc::dense(inputs[1].data_type(), branch_shape->dimensions());
    break;
  }
  case OperationKind::kScan: {
    const DataType type = inputs[0].data_type();
    if (!inputs[0].is_plain() ||
        (type != DataType::kInt32 && type != DataType::kFloat32)) {
      first = std::unexpected(Status::InvalidArgument(
          "Scan requires plain int32 or float32 input"));
      break;
    }
    auto axis =
        normalize_axis(operation.get_if<Scan>()->axis, inputs[0].rank());
    first = axis ? dense_like(inputs[0], type)
                 : std::unexpected(std::move(axis).error());
    break;
  }
  case OperationKind::kReduce: {
    const Reduce &reduce = *operation.get_if<Reduce>();
    if (!is_numeric(inputs[0].data_type())) {
      first = std::unexpected(
          Status::InvalidArgument("Reduce requires a numeric input"));
      break;
    }
    Status accumulation = validate_accumulation(reduce.accumulation_type);
    if (!accumulation.ok()) {
      first = std::unexpected(std::move(accumulation));
      break;
    }
    std::array<bool, kMaxTensorRank> selected{};
    bool axes_valid = true;
    if (reduce.axes.is_all()) {
      std::fill_n(selected.begin(), inputs[0].rank(), true);
    } else {
      for (const std::int8_t requested : reduce.axes.values()) {
        auto axis = normalize_axis(requested, inputs[0].rank());
        if (!axis) {
          first = std::unexpected(std::move(axis).error());
          axes_valid = false;
          break;
        }
        if (selected[*axis]) {
          first = std::unexpected(
              Status::InvalidArgument("Reduce axis list contains a duplicate"));
          axes_valid = false;
          break;
        }
        selected[*axis] = true;
      }
    }
    if (!axes_valid)
      break;
    std::array<std::int64_t, kMaxTensorRank> shape{};
    std::size_t output_rank = 0;
    for (std::size_t axis = 0; axis < inputs[0].rank(); ++axis) {
      if (selected[axis]) {
        if (reduce.keep_dimensions)
          shape[output_rank++] = 1;
      } else {
        shape[output_rank++] = inputs[0].dimensions()[axis];
      }
    }
    first = TensorDesc::dense(
        inputs[0].data_type(),
        std::span<const std::int64_t>(shape.data(), output_rank));
    break;
  }
  case OperationKind::kRmsNorm: {
    const RmsNorm &norm = *operation.get_if<RmsNorm>();
    if (!std::isfinite(norm.epsilon) || norm.epsilon < 0.0F) {
      first = std::unexpected(Status::InvalidArgument(
          "RMSNorm epsilon must be finite and non-negative"));
      break;
    }
    Status accumulation = validate_accumulation(norm.accumulation_type);
    if (!accumulation.ok()) {
      first = std::unexpected(std::move(accumulation));
      break;
    }
    first = infer_normalization(inputs[0], inputs.subspan(1), false);
    break;
  }
  case OperationKind::kLayerNorm: {
    const LayerNorm &norm = *operation.get_if<LayerNorm>();
    if (!std::isfinite(norm.epsilon) || norm.epsilon < 0.0F) {
      first = std::unexpected(Status::InvalidArgument(
          "LayerNorm epsilon must be finite and non-negative"));
      break;
    }
    Status accumulation = validate_accumulation(norm.accumulation_type);
    if (!accumulation.ok()) {
      first = std::unexpected(std::move(accumulation));
      break;
    }
    first = infer_normalization(inputs[0], inputs.subspan(1), true);
    break;
  }
  case OperationKind::kMatMul: {
    Status accumulation =
        validate_accumulation(operation.get_if<MatMul>()->accumulation_type);
    if (!accumulation.ok()) {
      first = std::unexpected(std::move(accumulation));
    } else {
      first = infer_matmul(inputs[0], inputs[1]);
    }
    break;
  }
  case OperationKind::kQuantizedMatMul:
    first = infer_quantized_matmul(*operation.get_if<QuantizedMatMul>(), inputs);
    break;
  case OperationKind::kGatedActivation: {
    const GatedActivation &gated = *operation.get_if<GatedActivation>();
    if (!is_floating(inputs[0].data_type()) || inputs[0].rank() == 0) {
      first = std::unexpected(Status::InvalidArgument(
          "GatedActivation requires floating rank >= 1 inputs"));
      break;
    }
    if (gated.inputs == GatedInputKind::kSeparate) {
      if (inputs[0].data_type() != inputs[1].data_type() ||
          !std::ranges::equal(inputs[0].dimensions(), inputs[1].dimensions())) {
        first = std::unexpected(Status::InvalidArgument(
            "separate gate/value tensors must have equal type and shape"));
      } else {
        first = dense_like(inputs[0], inputs[0].data_type());
      }
    } else {
      if ((inputs[0].dimensions().back() & 1) != 0) {
        first = std::unexpected(Status::InvalidArgument(
            "packed gate/value last dimension must be even"));
        break;
      }
      std::array<std::int64_t, kMaxTensorRank> shape{};
      std::copy(inputs[0].dimensions().begin(), inputs[0].dimensions().end(),
                shape.begin());
      shape[inputs[0].rank() - 1] /= 2;
      first = TensorDesc::dense(
          inputs[0].data_type(),
          std::span<const std::int64_t>(shape.data(), inputs[0].rank()));
    }
    break;
  }
  case OperationKind::kGather: {
    if (!is_index(inputs[1].data_type())) {
      first = std::unexpected(
          Status::InvalidArgument("Gather indices must use Int32 or Int64"));
      break;
    }
    auto axis =
        normalize_axis(operation.get_if<Gather>()->axis, inputs[0].rank());
    if (!axis) {
      first = std::unexpected(std::move(axis).error());
      break;
    }
    auto shape = gather_shape(inputs[0], inputs[1], *axis);
    if (!shape) {
      first = std::unexpected(std::move(shape).error());
    } else {
      first = TensorDesc::dense(inputs[0].data_type(), shape->dimensions());
    }
    break;
  }
  case OperationKind::kScatter: {
    if (!is_index(inputs[1].data_type()) ||
        inputs[0].data_type() != inputs[2].data_type()) {
      first = std::unexpected(Status::InvalidArgument(
          "Scatter requires i32/i64 indices and matching source/update type"));
      break;
    }
    auto axis =
        normalize_axis(operation.get_if<Scatter>()->axis, inputs[0].rank());
    if (!axis) {
      first = std::unexpected(std::move(axis).error());
      break;
    }
    auto expected = gather_shape(inputs[0], inputs[1], *axis);
    if (!expected) {
      first = std::unexpected(std::move(expected).error());
      break;
    }
    if (!std::ranges::equal(expected->dimensions(), inputs[2].dimensions())) {
      first = std::unexpected(Status::InvalidArgument(
          "Scatter update shape does not match indexed source shape"));
      break;
    }
    first = dense_like(inputs[0], inputs[0].data_type());
    break;
  }
  case OperationKind::kConcat: {
    const auto axis =
        normalize_axis(operation.get_if<Concat>()->axis, inputs[0].rank());
    if (!axis) {
      first = std::unexpected(std::move(axis).error());
      break;
    }
    std::array<std::int64_t, kMaxTensorRank> shape{};
    std::copy(inputs[0].dimensions().begin(), inputs[0].dimensions().end(),
              shape.begin());
    bool inputs_valid = true;
    for (std::size_t index = 1; index < inputs.size(); ++index) {
      if (inputs[index].data_type() != inputs[0].data_type() ||
          inputs[index].rank() != inputs[0].rank()) {
        first = std::unexpected(Status::InvalidArgument(
            "Concat inputs must have equal data type and rank"));
        inputs_valid = false;
        break;
      }
      bool mismatch = false;
      for (std::size_t dimension = 0; dimension < inputs[0].rank();
           ++dimension) {
        if (dimension != *axis &&
            inputs[index].dimensions()[dimension] != shape[dimension]) {
          mismatch = true;
          break;
        }
      }
      if (mismatch) {
        first = std::unexpected(
            Status::InvalidArgument("Concat non-axis dimensions must match"));
        inputs_valid = false;
        break;
      }
      auto sum = math::try_add(shape[*axis], inputs[index].dimensions()[*axis]);
      if (!sum) {
        first = std::unexpected(
            Status::OutOfRange("Concat axis dimension overflows int64"));
        inputs_valid = false;
        break;
      }
      shape[*axis] = *sum;
    }
    if (!inputs_valid)
      break;
    first = TensorDesc::dense(
        inputs[0].data_type(),
        std::span<const std::int64_t>(shape.data(), inputs[0].rank()));
    break;
  }
  case OperationKind::kRoPE: {
    Status status = validate_rope(*operation.get_if<RoPE>(), inputs);
    if (!status.ok()) {
      first = std::unexpected(std::move(status));
    } else {
      first = dense_like(inputs[0], inputs[0].data_type());
    }
    break;
  }
  case OperationKind::kSoftmax: {
    const Softmax &softmax = *operation.get_if<Softmax>();
    if (!is_floating(inputs[0].data_type()) || inputs[0].rank() == 0 ||
        !std::isfinite(softmax.scale)) {
      first = std::unexpected(Status::InvalidArgument(
          "Softmax requires floating rank >= 1 input and finite scale"));
      break;
    }
    auto axis = normalize_axis(softmax.axis, inputs[0].rank());
    if (!axis) {
      first = std::unexpected(std::move(axis).error());
      break;
    }
    Status accumulation = validate_accumulation(softmax.accumulation_type);
    if (!accumulation.ok()) {
      first = std::unexpected(std::move(accumulation));
      break;
    }
    if (softmax.mask != MaskKind::kNone) {
      Status mask =
          validate_mask(inputs[1], softmax.mask, inputs[0].dimensions());
      if (!mask.ok()) {
        first = std::unexpected(std::move(mask));
        break;
      }
    }
    first = dense_like(inputs[0], inputs[0].data_type());
    break;
  }
  case OperationKind::kTopK: {
    const TopK &top_k = *operation.get_if<TopK>();
    if (!is_numeric(inputs[0].data_type()) || inputs[0].rank() == 0 ||
        top_k.count == 0) {
      first = std::unexpected(Status::InvalidArgument(
          "TopK requires numeric rank >= 1 input and positive count"));
      break;
    }
    auto axis = normalize_axis(top_k.axis, inputs[0].rank());
    if (!axis) {
      first = std::unexpected(std::move(axis).error());
      break;
    }
    if (inputs[0].dimensions()[*axis] >
        static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
      first = std::unexpected(Status::OutOfRange(
          "TopK selected dimension cannot be represented by i32 indices"));
      break;
    }
    auto count = math::try_integer_cast<std::int64_t>(top_k.count);
    if (!count || *count > inputs[0].dimensions()[*axis]) {
      first = std::unexpected(
          Status::OutOfRange("TopK count exceeds selected dimension"));
      break;
    }
    std::array<std::int64_t, kMaxTensorRank> shape{};
    std::copy(inputs[0].dimensions().begin(), inputs[0].dimensions().end(),
              shape.begin());
    shape[*axis] = *count;
    first = TensorDesc::dense(
        inputs[0].data_type(),
        std::span<const std::int64_t>(shape.data(), inputs[0].rank()));
    if (first) {
      inferred[1] = TensorDesc::dense(DataType::kInt32,
                                      std::span<const std::int64_t>(
                                          shape.data(), inputs[0].rank()))
                        .value();
    }
    break;
  }
  case OperationKind::kScaledDotProductAttention:
    first =
        infer_attention(*operation.get_if<ScaledDotProductAttention>(), inputs);
    break;
  case OperationKind::kUnknown:
  case OperationKind::kCustom:
    return Status::Internal("operation inference route is invalid");
  }

  if (!first)
    return std::move(first).error();
  inferred[0] = *first;
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    outputs[index] = inferred[index];
  }
  return Status::Ok();
}

Result<TensorDesc>
infer_single_output(const Operation &operation,
                    std::span<const TensorDesc> inputs) noexcept {
  if (operation_traits(operation).outputs != 1) {
    return std::unexpected(
        Status::FailedPrecondition("operation does not have one output"));
  }
  TensorDesc output;
  Status status = infer_output_descriptions(operation, inputs,
                                            std::span<TensorDesc>(&output, 1));
  if (!status.ok())
    return std::unexpected(std::move(status));
  return output;
}

Result<std::size_t>
detail::view_alias_offset(const View &view, const TensorDesc &input,
                          const TensorDesc &output) noexcept {
  HYLE_ASSIGN_OR_RETURN(const auto output_bytes, output.required_bytes());
  HYLE_ASSIGN_OR_RETURN(const auto input_bytes, input.required_bytes());
  std::size_t offset = 0;
  // Empty views have no reachable address, including an empty slice at the
  // end of its input. Preserve the existing direct-provider convention.
  if (view.kind() == ViewKind::kSlice && output_bytes != 0) {
    HYLE_ASSIGN_OR_RETURN(const auto axis,
                          normalize_axis(view.axis(), input.rank()));
    const auto start = math::try_integer_cast<std::size_t>(view.start());
    const auto stride =
        math::try_integer_cast<std::size_t>(input.byte_strides()[axis]);
    if (!start || !stride)
      return std::unexpected(
          Status::OutOfRange("view storage offset exceeds size_t"));
    const auto product = math::try_multiply(*start, *stride);
    if (!product)
      return std::unexpected(
          Status::OutOfRange("view storage offset overflows size_t"));
    offset = *product;
  }
  if (offset > input_bytes || output_bytes > input_bytes - offset)
    return std::unexpected(
        Status::OutOfRange("view storage span exceeds its input"));
  return offset;
}

} // namespace hyle::compute
