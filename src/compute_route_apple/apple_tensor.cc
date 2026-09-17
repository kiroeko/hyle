#include "apple_tensor.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

#include <mlx/fast.h>
#include <mlx/ops.h>

#include "apple_indexing.h"
#include "apple_internal.h"
#include "apple_reduction.h"

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
    return true;
  default:
    return false;
  }
}
} // namespace

Result<std::vector<mx::array>> lower_tensor(const Operation &operation,
                                            std::span<const mx::array> inputs,
                                            mx::Stream stream) noexcept {
  try {
    if (!admitted(operation.kind()))
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
    if (operation.kind() == OperationKind::kGather ||
        operation.kind() == OperationKind::kScatter ||
        operation.kind() == OperationKind::kScan)
      return lower_indexing(operation, inputs, shape, stream);
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
        operation.kind() == OperationKind::kSoftmax ||
        operation.kind() == OperationKind::kTopK)
      return lower_reduction(operation, inputs, shape, stream);
    if (operation.kind() == OperationKind::kCast)
      return std::vector{mx::astype(inputs[0], type, stream)};
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
