#include "apple_tensor.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <mlx/fast.h>
#include <mlx/ops.h>

#include "apple_internal.h"
#include "apple_checks.h"
#include "apple_mature_indexing.h"
#include "apple_mature_reduction.h"
#include "hyle/compute/model_ops.h"

namespace hyle::compute::apple::detail {
namespace mx = mlx::core;
namespace {
Result<DataType> logical_type(mx::Dtype type) {
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
  if (type == mx::uint8)
    return DataType::kUInt8;
  if (type == mx::uint32)
    return DataType::kUInt32;
  return std::unexpected(Status::Unimplemented(
      "Apple tensor lowering admits F32/F16/BF16/I32/I64/Bool storage"));
}
Result<mx::Dtype> native_type(DataType type) {
  switch (type) {
  case DataType::kFloat32:
    return mx::float32;
  case DataType::kFloat16:
    return mx::float16;
  case DataType::kBFloat16:
    return mx::bfloat16;
  case DataType::kInt32:
    return mx::int32;
  case DataType::kInt64:
    return mx::int64;
  case DataType::kBool:
    return mx::bool_;
  default:
    return std::unexpected(Status::Unimplemented(
        "Apple tensor output requires F32/F16/BF16/I32/I64/Bool storage"));
  }
}
bool floating(mx::Dtype type) {
  return type == mx::float32 || type == mx::float16 || type == mx::bfloat16;
}
Result<TensorDesc> description(const mx::array &input) {
  HYLE_ASSIGN_OR_RETURN(auto type, logical_type(input.dtype()));
  if (input.ndim() > kMaxTensorRank)
    return std::unexpected(
        Status::Unimplemented("Apple tensor rank exceeds the schema limit"));
  std::array<std::int64_t, kMaxTensorRank> shape{};
  std::copy(input.shape().begin(), input.shape().end(), shape.begin());
  // These are logical native arrays, not imported Mem storage. MLX owns
  // strides and any layout materialization inside its compiled region.
  return TensorDesc::dense(type, {shape.data(), input.ndim()});
}
Result<mx::Shape> native_shape(const TensorDesc &description) {
  mx::Shape shape;
  shape.reserve(description.rank());
  for (auto dimension : description.dimensions()) {
    if (dimension > (std::numeric_limits<int>::max)())
      return std::unexpected(
          Status::OutOfRange("Apple dimension exceeds the native int range"));
    shape.push_back(static_cast<int>(dimension));
  }
  return shape;
}
mx::array scalar(ScalarValue value, mx::Stream stream) {
  const auto bits = value.bits();
  switch (value.data_type()) {
  case DataType::kFloat32:
    return mx::array(std::bit_cast<float>(static_cast<std::uint32_t>(bits)));
  case DataType::kFloat16:
  case DataType::kBFloat16:
    return mx::view(mx::array(static_cast<std::uint16_t>(bits)),
                    value.data_type() == DataType::kFloat16 ? mx::float16
                                                            : mx::bfloat16,
                    stream);
  case DataType::kInt32:
    return mx::array(
        std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(bits)));
  case DataType::kInt64:
    return mx::array(std::bit_cast<std::int64_t>(bits));
  case DataType::kBool:
    return mx::array(bits != 0);
  default:
    throw std::invalid_argument("Apple scalar type is not admitted");
  }
}
mx::array unary(UnaryKind kind, const mx::array &x, mx::Stream stream) {
  const mx::array zero(0.0F), one(1.0F);
  switch (kind) {
  case UnaryKind::kAbs:
    return mx::abs(x, stream);
  case UnaryKind::kNeg:
    return mx::negative(x, stream);
  case UnaryKind::kExp:
    return mx::exp(x, stream);
  case UnaryKind::kLog:
    return mx::log(x, stream);
  case UnaryKind::kSqrt:
    return mx::sqrt(x, stream);
  case UnaryKind::kRsqrt:
    return mx::rsqrt(x, stream);
  case UnaryKind::kTanh:
    return mx::tanh(x, stream);
  case UnaryKind::kSigmoid:
    return mx::sigmoid(x, stream);
  case UnaryKind::kRelu:
    return mx::where(mx::isnan(x, stream), x,
                     mx::where(mx::greater(x, zero, stream), x, zero, stream),
                     stream);
  case UnaryKind::kSoftplus:
    return mx::logaddexp(x, zero, stream);
  case UnaryKind::kSilu:
  case UnaryKind::kGelu:
  case UnaryKind::kGeluTanh: {
    auto result = x;
    if (kind == UnaryKind::kSilu) {
      result = mx::multiply(x, mx::sigmoid(x, stream), stream);
    } else {
      auto inner = mx::multiply(x, mx::array(0.70710678118654752440F), stream);
      if (kind == UnaryKind::kGelu) {
        inner = mx::erf(inner, stream);
      } else {
        inner = mx::multiply(
            mx::multiply(x, mx::array(0.79788456080286535588F), stream),
            mx::add(one,
                    mx::multiply(mx::array(0.044715F), mx::square(x, stream),
                                 stream),
                    stream),
            stream);
        inner = mx::tanh(inner, stream);
      }
      result = mx::multiply(mx::multiply(mx::array(0.5F), x, stream),
                            mx::add(one, inner, stream), stream);
    }
    // The limiting activation value is +0, not (-Inf)*0 -> NaN.
    return mx::where(
        mx::equal(x, mx::array(-std::numeric_limits<float>::infinity()),
                  stream),
        zero, result, stream);
  }
  }
  throw std::invalid_argument("Apple unary mode is invalid");
}
mx::array binary(BinaryKind kind, const mx::array &left, const mx::array &right,
                 mx::Stream stream) {
  switch (kind) {
  case BinaryKind::kAdd:
    return mx::add(left, right, stream);
  case BinaryKind::kSubtract:
    return mx::subtract(left, right, stream);
  case BinaryKind::kMultiply:
    return mx::multiply(left, right, stream);
  case BinaryKind::kDivide:
    return mx::divide(left, right, stream);
  case BinaryKind::kMinimum:
  case BinaryKind::kMaximum: {
    auto value = kind == BinaryKind::kMinimum
                     ? mx::minimum(left, right, stream)
                     : mx::maximum(left, right, stream);
    // HYLE's elementwise min/max have fmin/fmax semantics: one NaN leaves
    // the other operand. Reductions have a separate NaN contract.
    return mx::where(mx::isnan(left, stream), right,
                     mx::where(mx::isnan(right, stream), left, value, stream),
                     stream);
  }
  }
  throw std::invalid_argument("Apple binary mode is invalid");
}
bool admitted(OperationKind kind) {
  switch (kind) {
  case OperationKind::kView:
  case OperationKind::kCopy:
  case OperationKind::kFill:
  case OperationKind::kCast:
  case OperationKind::kUnary:
  case OperationKind::kBinary:
  case OperationKind::kSelect:
  case OperationKind::kConcat:
  case OperationKind::kGatedActivation:
  case OperationKind::kRmsNorm:
  case OperationKind::kLayerNorm:
  case OperationKind::kReduce:
  case OperationKind::kSoftmax:
  case OperationKind::kTopK:
  case OperationKind::kGather:
  case OperationKind::kScatter:
  case OperationKind::kScan:
  case OperationKind::kMatMul:
  case OperationKind::kRoPE:
  case OperationKind::kHadamard:
  case OperationKind::kScaledDotProductAttention:
    return true;
  default:
    return false;
  }
}

mx::array lower_rope(const RoPE &rope, std::span<const mx::array> inputs,
                     mx::Stream stream) {
  const auto &data = inputs[0];
  const auto rank = data.ndim();
  const auto sequence_axis = rope.sequence_axis < 0
                                 ? rope.sequence_axis + static_cast<int>(rank)
                                 : rope.sequence_axis;
  const auto pairs = static_cast<int>(rope.rotary_dimensions / 2);

  mx::Shape position_shape(rank, 1);
  if (inputs[1].ndim() == 1) {
    position_shape[sequence_axis] = data.shape(sequence_axis);
  } else {
    for (std::size_t axis = 0; axis < inputs[1].ndim(); ++axis)
      position_shape[axis] = inputs[1].shape(static_cast<int>(axis));
  }
  auto positions = mx::reshape(mx::astype(inputs[1], mx::float32, stream),
                               position_shape, stream);

  mx::Shape frequency_shape(rank, 1);
  frequency_shape.back() = pairs;
  if (inputs[2].ndim() == 2)
    frequency_shape[sequence_axis] = data.shape(sequence_axis);
  auto frequencies = mx::reshape(mx::astype(inputs[2], mx::float32, stream),
                                 frequency_shape, stream);
  auto angles = mx::multiply(positions, frequencies, stream);
  auto cosines = mx::astype(mx::cos(angles, stream), data.dtype(), stream);
  auto sines = mx::astype(mx::sin(angles, stream), data.dtype(), stream);

  mx::Shape start(rank, 0), stop = data.shape(), strides(rank, 1);
  mx::array first = data;
  mx::array second = data;
  if (rope.layout == RopeLayout::kInterleaved) {
    stop.back() = static_cast<int>(rope.rotary_dimensions);
    strides.back() = 2;
    first = mx::slice(data, start, stop, strides, stream);
    start.back() = 1;
    second = mx::slice(data, start, stop, strides, stream);
  } else {
    stop.back() = pairs;
    first = mx::slice(data, start, stop, stream);
    start.back() = pairs;
    stop.back() = static_cast<int>(rope.rotary_dimensions);
    second = mx::slice(data, start, stop, stream);
  }
  auto rotated_first =
      mx::subtract(mx::multiply(first, cosines, stream),
                   mx::multiply(second, sines, stream), stream);
  auto rotated_second = mx::add(mx::multiply(first, sines, stream),
                                mx::multiply(second, cosines, stream), stream);
  mx::array rotated = rotated_first;
  if (rope.layout == RopeLayout::kInterleaved) {
    auto interleaved =
        mx::concatenate({mx::expand_dims(rotated_first, -1, stream),
                         mx::expand_dims(rotated_second, -1, stream)},
                        -1, stream);
    auto rotary_shape = data.shape();
    rotary_shape.back() = static_cast<int>(rope.rotary_dimensions);
    rotated = mx::reshape(interleaved, rotary_shape, stream);
  } else {
    rotated = mx::concatenate({rotated_first, rotated_second}, -1, stream);
  }
  if (rope.rotary_dimensions < static_cast<std::size_t>(data.shape(-1))) {
    start = mx::Shape(rank, 0);
    start.back() = static_cast<int>(rope.rotary_dimensions);
    rotated = mx::concatenate(
        {rotated, mx::slice(data, start, data.shape(), stream)}, -1, stream);
  }
  return rotated;
}

mx::array lower_attention(const ScaledDotProductAttention &attention,
                          std::span<const mx::array> inputs,
                          mx::Stream stream) {
  auto query = inputs[0];
  auto key = inputs[1];
  auto value = inputs[2];
  const bool sequence_first =
      attention.layout == AttentionLayout::kBatchSequenceHeadDepth;
  if (sequence_first) {
    query = mx::transpose(query, {0, 2, 1, 3}, stream);
    key = mx::transpose(key, {0, 2, 1, 3}, stream);
    value = mx::transpose(value, {0, 2, 1, 3}, stream);
  }
  std::optional<mx::array> mask;
  std::string mask_mode;
  if (attention.mask != MaskKind::kNone) {
    mask = inputs[3];
    mask_mode = "array";
  } else if (attention.causal) {
    mask_mode = "causal";
  }
  auto result = mx::fast::scaled_dot_product_attention(
      query, key, value, attention.scale, mask_mode, mask, std::nullopt, false,
      stream);
  return sequence_first ? mx::transpose(result, {0, 2, 1, 3}, stream) : result;
}
} // namespace

Result<std::vector<mx::array>> lower_tensor(const Operation &operation,
                                            std::span<const mx::array> inputs,
                                            mx::Stream stream) noexcept {
  try {
    const auto causal_conv = operation.custom_as<model_ops::CausalConv1d>();
    const auto mlx_quantized = operation.custom_as<model_ops::MlxQuantized>();
    const auto sink_attention = operation.custom_as<model_ops::SinkAttention>();
    if (!causal_conv && !sink_attention && !mlx_quantized &&
        !admitted(operation.kind()))
      return std::unexpected(Status::Unimplemented(
          "Apple typed operation has no admitted MLX lowering"));
    if (!operation_traits(operation).inputs.accepts(inputs.size()))
      return std::unexpected(Status::InvalidArgument(
          "Apple typed operation input count mismatch"));
    if (stream.device.type != mx::Device::gpu)
      return std::unexpected(Status::InvalidArgument(
          "Apple typed lowering requires a GPU stream"));
    std::vector<TensorDesc> descriptions;
    descriptions.reserve(inputs.size());
    for (const auto &input : inputs) {
      HYLE_ASSIGN_OR_RETURN(auto desc, description(input));
      descriptions.push_back(desc);
    }
    std::vector<TensorDesc> outputs(operation_traits(operation).outputs);
    HYLE_RETURN_IF_ERROR(
        infer_output_descriptions(operation, descriptions, outputs));
    const auto &output = outputs[0];
    HYLE_ASSIGN_OR_RETURN(auto type, native_type(output.data_type()));
    HYLE_ASSIGN_OR_RETURN(auto shape, native_shape(output));
    if (mlx_quantized) {
      const auto &op = *mlx_quantized;
      const bool affine = op.format == model_ops::MlxWeightFormat::kAffine4;
      const bool embedding = op.kind == model_ops::MlxQuantizedKind::kEmbedding;
      const bool grouped = op.kind == model_ops::MlxQuantizedKind::kGrouped;
      const int bits = op.format == model_ops::MlxWeightFormat::kMxfp8 ? 8 : 4;
      const std::string mode = affine ? "affine" : bits == 4 ? "mxfp4" : "mxfp8";
      const std::size_t w = embedding ? 0 : 1;
      const auto weights = inputs[w];
      const auto scales = inputs[w + 1];
      const std::optional<mx::array> bias = affine
          ? std::optional<mx::array>{inputs[w + 2]} : std::nullopt;
      std::optional<mx::array> status;
      std::optional<mx::array> ids;
      if (embedding || grouped) {
        const auto raw = inputs.back();
        auto valid = mx::logical_and(mx::greater_equal(raw, mx::array(0), stream),
            mx::less(raw, mx::array(weights.shape(0)), stream), stream);
        ids = mx::where(valid, raw, mx::array(0), stream);
        status = mx::where(mx::all(valid, false, stream), mx::array(0),
                           mx::array(kIndexOutOfRange), stream);
      }
      mx::array result = weights;
      if (embedding) {
        auto selected_w = mx::take(weights, *ids, 0, stream);
        auto selected_s = mx::take(scales, *ids, 0, stream);
        auto selected_b = mx::take(*bias, *ids, 0, stream);
        result = mx::dequantize(selected_w, selected_s, selected_b,
             static_cast<int>(op.group_size), bits, mode, std::nullopt, type, stream);
      } else {
        // The original GPU kernels accumulate in F32. An explicit widened
        // result for mixed affine ports is part of the declared contract.
        auto activation = mx::astype(inputs[0], type, stream);
        auto scale = affine ? mx::astype(scales, type, stream) : scales;
        auto biases = bias ? std::optional<mx::array>{mx::astype(*bias, type, stream)}
                           : std::nullopt;
        if (grouped) {
          const auto count64 = std::int64_t{activation.shape(0)} * activation.shape(1);
          if (count64 > (std::numeric_limits<int>::max)())
            return std::unexpected(Status::OutOfRange("MLX selected expert batch exceeds native range"));
          const int count = static_cast<int>(count64);
          auto rows = mx::reshape(activation, {count, 1, activation.shape(2)}, stream);
          auto selected = mx::reshape(*ids, {count}, stream);
          result = mx::gather_qmm(rows, weights, scale, biases, std::nullopt,
              selected, true, static_cast<int>(op.group_size), bits, mode, false, stream);
          result = mx::reshape(result, shape, stream);
        } else {
          result = mx::quantized_matmul(activation, weights, scale, biases, true,
                       static_cast<int>(op.group_size), bits, mode, stream);
        }
      }
      result = mx::astype(result, type, stream);
      if (status) {
        HYLE_ASSIGN_OR_RETURN(auto checked,
            with_device_status(std::move(result), std::move(*status), stream));
        result = std::move(checked);
      }
      return std::vector{std::move(result)};
    }
    if (sink_attention) {
      auto query = mx::transpose(inputs[0], {0, 2, 1, 3}, stream);
      auto key = mx::transpose(inputs[1], {0, 2, 1, 3}, stream);
      auto value = mx::transpose(inputs[2], {0, 2, 1, 3}, stream);
      auto result = mx::fast::scaled_dot_product_attention(query, key, value,
          sink_attention->scale, "array", inputs[3], inputs[4], false, stream);
      return std::vector{mx::transpose(result, {0, 2, 1, 3}, stream)};
    }
    if (causal_conv) {
      if (!floating(inputs[0].dtype()) ||
          inputs[1].dtype() != inputs[0].dtype())
        return std::unexpected(Status::Unimplemented(
            "Apple MLX CausalConv1d requires matching floating inputs"));
      const auto channels = inputs[0].shape(-1);
      auto weights = mx::expand_dims(inputs[1], -1, stream);
      return std::vector{
          mx::conv1d(inputs[0], weights, 1, 0, 1, channels, stream)};
    }
    if (operation.kind() == OperationKind::kGather ||
        operation.kind() == OperationKind::kScan ||
        operation.kind() == OperationKind::kScatter ||
        operation.kind() == OperationKind::kTopK)
      return lower_mature_indexing(operation, inputs, shape, stream);
    if (const auto *fill = operation.get_if<Fill>())
      return std::vector{
          mx::full(shape, scalar(fill->value, stream), type, stream)};
    if (const auto *view = operation.get_if<View>()) {
      switch (view->kind()) {
      case ViewKind::kReshape:
        return std::vector{mx::reshape(inputs[0], shape, stream)};
      case ViewKind::kBroadcast:
        return std::vector{mx::broadcast_to(inputs[0], shape, stream)};
      case ViewKind::kPermute: {
        std::vector<int> axes;
        for (auto axis : view->permutation())
          axes.push_back(axis < 0 ? axis + static_cast<int>(inputs[0].ndim())
                                  : axis);
        return std::vector{mx::transpose(inputs[0], axes, stream)};
      }
      case ViewKind::kSlice: {
        const auto axis =
            view->axis() < 0 ? view->axis() + static_cast<int>(inputs[0].ndim())
                             : view->axis();
        mx::Shape start(inputs[0].ndim(), 0), stop = inputs[0].shape(),
                                              step(inputs[0].ndim(), 1);
        start[axis] = static_cast<int>(view->start());
        stop[axis] = static_cast<int>(
            view->count() == 0
                ? view->start()
                : view->start() + (view->count() - 1) * view->step() + 1);
        step[axis] = static_cast<int>(std::min<std::int64_t>(
            view->step(), (std::numeric_limits<int>::max)()));
        return std::vector{mx::slice(inputs[0], start, stop, step, stream)};
      }
      }
    }
    if (operation.kind() == OperationKind::kCopy)
      return std::vector{mx::copy(inputs[0], stream)};
    if (operation.kind() == OperationKind::kSelect) {
      if (!floating(type))
        return std::vector{mx::where(inputs[0], inputs[1], inputs[2], stream)};
      // Selection is storage-exact. A compiled floating selection can
      // canonicalize BF16 NaNs; same-width native views keep payload/sign bits
      // out of floating constant conversion without introducing arithmetic.
      const auto bits = type == mx::float32 ? mx::uint32 : mx::uint16;
      return std::vector{
          mx::view(mx::where(inputs[0], mx::view(inputs[1], bits, stream),
                             mx::view(inputs[2], bits, stream), stream),
                   type, stream)};
    }
    if (const auto *concat = operation.get_if<Concat>())
      return std::vector{
          mx::concatenate(std::vector<mx::array>(inputs.begin(), inputs.end()),
                          concat->axis, stream)};
    if (!floating(inputs[0].dtype()) || !floating(type))
      return std::unexpected(Status::Unimplemented(
          "Apple arithmetic and Cast require F32/F16/BF16"));
    if (operation.kind() == OperationKind::kReduce ||
        operation.kind() == OperationKind::kSoftmax)
      return lower_mature_reduction(operation, inputs, shape, stream);
    if (operation.kind() == OperationKind::kCast)
      return std::vector{mx::astype(inputs[0], type, stream)};
    if (const auto *matmul = operation.get_if<MatMul>()) {
      if (matmul->accumulation_type != DataType::kFloat32 ||
          !floating(inputs[1].dtype()))
        return std::unexpected(Status::Unimplemented(
            "Apple MLX MatMul requires floating inputs and F32 accumulation"));
      return std::vector{mx::matmul(inputs[0], inputs[1], stream)};
    }
    if (const auto *rope = operation.get_if<RoPE>())
      return std::vector{lower_rope(*rope, inputs, stream)};
    if (const auto *hadamard = operation.get_if<Hadamard>()) {
      if (hadamard->accumulation_type != DataType::kFloat32)
        return std::unexpected(Status::Unimplemented(
            "Apple MLX Hadamard requires F32 accumulation"));
      const auto scale =
          1.0F / std::sqrt(static_cast<float>(inputs[0].shape(-1)));
      return std::vector{mx::hadamard_transform(inputs[0], scale, stream)};
    }
    if (const auto *attention = operation.get_if<ScaledDotProductAttention>()) {
      if (attention->accumulation_type != DataType::kFloat32 ||
          (attention->causal && attention->mask != MaskKind::kNone))
        return std::unexpected(Status::Unimplemented(
            "Apple MLX SDPA requires F32 accumulation and one mask mode"));
      return std::vector{lower_attention(*attention, inputs, stream)};
    }
    auto first = mx::astype(inputs[0], mx::float32, stream);
    if (const auto *value = operation.get_if<Unary>())
      return std::vector{
          mx::astype(unary(value->kind, first, stream), type, stream)};
    if (const auto *value = operation.get_if<Binary>())
      return std::vector{
          mx::astype(binary(value->kind, first,
                            mx::astype(inputs[1], mx::float32, stream), stream),
                     type, stream)};
    if (const auto *gated = operation.get_if<GatedActivation>()) {
      auto up = first;
      if (gated->inputs == GatedInputKind::kSeparate) {
        up = mx::astype(inputs[1], mx::float32, stream);
      } else {
        auto halves = mx::split(first, 2, -1, stream);
        first = std::move(halves[0]);
        up = std::move(halves[1]);
      }
      const auto kind = gated->activation == GatedActivationKind::kSilu
                            ? UnaryKind::kSilu
                        : gated->activation == GatedActivationKind::kGelu
                            ? UnaryKind::kGelu
                            : UnaryKind::kGeluTanh;
      return std::vector{mx::astype(
          mx::multiply(unary(kind, first, stream), up, stream), type, stream)};
    }
    const auto *rms = operation.get_if<RmsNorm>();
    const auto *layer = operation.get_if<LayerNorm>();
    if (rms == nullptr && layer == nullptr)
      return std::unexpected(
          Status::Internal("Apple typed lowering not bound"));
    if ((rms ? rms->accumulation_type : layer->accumulation_type) !=
        DataType::kFloat32)
      return std::unexpected(Status::Unimplemented(
          "Apple normalization requires explicit F32 accumulation"));
    HYLE_ASSIGN_OR_RETURN(auto count, output.element_count());
    if (count == 0)
      return std::vector{mx::zeros(shape, type, stream)};
    std::optional<mx::array> weight, bias;
    if (inputs.size() > 1)
      weight = mx::astype(inputs[1], mx::float32, stream);
    if (inputs.size() > 2)
      bias = mx::astype(inputs[2], mx::float32, stream);
    auto result =
        rms ? mx::fast::rms_norm(first, weight, rms->epsilon, stream)
            : mx::fast::layer_norm(first, weight, bias, layer->epsilon, stream);
    return std::vector{mx::astype(result, type, stream)};
  } catch (...) {
    return std::unexpected(native_exception());
  }
}
} // namespace hyle::compute::apple::detail
