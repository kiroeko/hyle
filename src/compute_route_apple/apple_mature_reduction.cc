#include "apple_mature_reduction.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>

#include <mlx/ops.h>

namespace hyle::compute::apple::detail {
namespace mx = mlx::core;
namespace {
std::vector<int> native_axes(const Reduce &operation, std::size_t rank) {
  std::vector<int> axes;
  if (operation.axes.is_all()) {
    axes.resize(rank);
    std::iota(axes.begin(), axes.end(), 0);
  } else {
    for (auto axis : operation.axes.values())
      axes.push_back(axis < 0 ? axis + static_cast<int>(rank) : axis);
  }
  return axes;
}

float empty_identity(ReductionKind kind) {
  switch (kind) {
  case ReductionKind::kSum:
    return 0.0F;
  case ReductionKind::kMean:
    return std::numeric_limits<float>::quiet_NaN();
  case ReductionKind::kMinimum:
    return std::numeric_limits<float>::infinity();
  case ReductionKind::kMaximum:
    return -std::numeric_limits<float>::infinity();
  }
  return 0.0F; // The common schema rejects unknown reduction kinds.
}

mx::array reduce(const Reduce &operation, const mx::array &input,
                 const mx::Shape &output_shape, mx::Stream stream) {
  if (std::find(output_shape.begin(), output_shape.end(), 0) !=
      output_shape.end())
    return mx::zeros(output_shape, input.dtype(), stream);
  if (input.size() == 0)
    return mx::full(output_shape, empty_identity(operation.kind), input.dtype(),
                    stream);

  const auto axes = native_axes(operation, input.ndim());
  const auto keep = operation.keep_dimensions;
  const auto x = mx::astype(input, mx::float32, stream);
  if (operation.kind == ReductionKind::kSum)
    return mx::astype(mx::sum(x, axes, keep, stream), input.dtype(), stream);
  if (operation.kind == ReductionKind::kMean)
    return mx::astype(mx::mean(x, axes, keep, stream), input.dtype(), stream);

  const bool minimum = operation.kind == ReductionKind::kMinimum;
  auto value = minimum ? mx::min(x, axes, keep, stream)
                       : mx::max(x, axes, keep, stream);
  // Native Min/Max propagate NaNs, but equal signed zeros have no preferred
  // sign. Original integer views and boolean reduction preserve that contract
  // without a custom reduction kernel or another scan for NaNs.
  const mx::array positive_zero(std::uint32_t{0});
  const mx::array negative_zero(std::uint32_t{0x80000000});
  const auto preferred = mx::any(
      mx::equal(mx::view(x, mx::uint32, stream),
                minimum ? negative_zero : positive_zero, stream),
      axes, keep, stream);
  const auto zero_bits =
      mx::where(preferred, minimum ? negative_zero : positive_zero,
                 minimum ? positive_zero : negative_zero, stream);
  value = mx::where(mx::equal(value, mx::array(0.0F), stream),
                     mx::view(zero_bits, mx::float32, stream), value, stream);
  return mx::astype(value, input.dtype(), stream);
}

mx::array probability(const Softmax &operation,
                      std::span<const mx::array> inputs,
                      mx::Stream stream) {
  const auto &input = inputs[0];
  if (input.size() == 0)
    return mx::zeros(input.shape(), input.dtype(), stream);
  auto x = mx::multiply(mx::astype(input, mx::float32, stream),
                         mx::array(operation.scale), stream);
  if (operation.mask == MaskKind::kBoolean)
    x = mx::where(inputs[1], x,
                    mx::array(-std::numeric_limits<float>::infinity()), stream);
  else if (operation.mask == MaskKind::kAdditive)
    x = mx::add(x, mx::astype(inputs[1], mx::float32, stream), stream);

  if (operation.kind == SoftmaxKind::kSoftmax)
    return mx::astype(mx::softmax(x, operation.axis, true, stream),
                       input.dtype(), stream);

  // MLX nn.log_softmax uses this shifted sequence: subtracting logsumexp(x)
  // directly loses the normalizer when the common offset is large.
  const auto shifted =
      mx::subtract(x, mx::max(x, operation.axis, true, stream), stream);
  return mx::astype(
      mx::subtract(shifted,
                    mx::logsumexp(shifted, operation.axis, true, stream), stream),
      input.dtype(), stream);
}
} // namespace

Result<std::vector<mx::array>> lower_mature_reduction(
    const Operation &operation, std::span<const mx::array> inputs,
    const mx::Shape &output_shape, mx::Stream stream) {
  if (const auto *value = operation.get_if<Reduce>()) {
    if (value->accumulation_type != DataType::kFloat32 &&
        !(inputs[0].dtype() == mx::float32 &&
          value->accumulation_type == DataType::kUnknown))
      return std::unexpected(Status::Unimplemented(
          "Apple half reductions require explicit F32 accumulation"));
    return std::vector{reduce(*value, inputs[0], output_shape, stream)};
  }
  if (const auto *value = operation.get_if<Softmax>()) {
    if (value->accumulation_type != DataType::kFloat32)
      return std::unexpected(Status::Unimplemented(
          "Apple softmax requires explicit F32 accumulation"));
    if (value->mask == MaskKind::kAdditive &&
        inputs[1].dtype() != inputs[0].dtype())
      return std::unexpected(Status::Unimplemented(
          "Apple additive softmax mask must match the input storage type"));
    return std::vector{probability(*value, inputs, stream)};
  }
  return std::unexpected(
      Status::Internal("Apple mature reduction operation is not bound"));
}
} // namespace hyle::compute::apple::detail
