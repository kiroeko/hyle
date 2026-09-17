#include "apple_reduction.h"

#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>

#include <mlx/ops.h>

namespace hyle::compute::apple::detail {
namespace mx = mlx::core;
namespace {
constexpr std::uint32_t sign_bit = 0x80000000U;
constexpr std::uint32_t magnitude_bits = 0x7fffffffU;
constexpr std::uint32_t infinity_bits = 0x7f800000U;

// IEEE non-NaN values become unsigned ascending keys, including -0 < +0.
// Native integer reductions avoid Metal float min/max's equal-zero choice.
mx::array ordered_key(const mx::array &bits, mx::Stream stream) {
  return mx::where(mx::less(bits, mx::array(sign_bit), stream),
                   mx::bitwise_xor(bits, mx::array(sign_bit), stream),
                   mx::bitwise_invert(bits, stream), stream);
}
mx::array from_key(const mx::array &key, mx::Stream stream) {
  return mx::view(mx::where(mx::less(key, mx::array(sign_bit), stream),
                            mx::bitwise_invert(key, stream),
                            mx::bitwise_xor(key, mx::array(sign_bit), stream),
                            stream),
                  mx::float32, stream);
}
mx::array magnitude(const mx::array &bits, mx::Stream stream) {
  return mx::bitwise_and(bits, mx::array(magnitude_bits), stream);
}
mx::array nan_bits(const mx::array &bits, mx::Stream stream) {
  return mx::greater(magnitude(bits, stream), mx::array(infinity_bits), stream);
}
std::vector<int> axes(const Reduce &operation, int rank) {
  std::vector<int> result;
  if (operation.axes.is_all()) {
    result.resize(rank);
    std::iota(result.begin(), result.end(), 0);
  } else {
    for (int axis : operation.axes.values())
      result.push_back(axis < 0 ? axis + rank : axis);
  }
  return result;
}
float identity(ReductionKind kind) {
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
  return std::numeric_limits<float>::quiet_NaN();
}
mx::array reduce(const Reduce &operation, const mx::array &first,
                 mx::Stream stream) {
  const auto selected = axes(operation, static_cast<int>(first.ndim()));
  const auto keep = operation.keep_dimensions;
  if (operation.kind == ReductionKind::kSum)
    return mx::sum(first, selected, keep, stream);
  if (operation.kind == ReductionKind::kMean)
    return mx::mean(first, selected, keep, stream);
  const auto raw = mx::view(first, mx::uint32, stream);
  const bool minimum = operation.kind == ReductionKind::kMinimum;
  // Both sentinels decode to NaN and win over every non-NaN key. There is
  // one native collective, not a separate scan for NaN and signed zero.
  const auto key =
      mx::where(nan_bits(raw, stream), mx::array(minimum ? 0U : 0xffffffffU),
                ordered_key(raw, stream), stream);
  return from_key(minimum ? mx::min(key, selected, keep, stream)
                          : mx::max(key, selected, keep, stream),
                  stream);
}
mx::array softmax(const Softmax &operation, std::span<const mx::array> inputs,
                  mx::array first, mx::Stream stream) {
  if (operation.scale != 1.0F)
    first = mx::multiply(first, mx::array(operation.scale), stream);
  if (operation.mask == MaskKind::kBoolean)
    first =
        mx::where(inputs[1], first,
                  mx::array(-std::numeric_limits<float>::infinity()), stream);
  else if (operation.mask == MaskKind::kAdditive)
    first = mx::add(first, mx::astype(inputs[1], mx::float32, stream), stream);
  if (operation.kind == SoftmaxKind::kSoftmax)
    return mx::softmax(first, operation.axis, true, stream);
  // Shift before computing the normalizer, so a large common offset cannot
  // cancel log(sum(exp(x-max))). Invalid rows propagate NaN through the sum.
  const auto shifted =
      mx::subtract(first, mx::max(first, operation.axis, true, stream), stream);
  const auto normalizer = mx::log(
      mx::sum(mx::exp(shifted, stream), operation.axis, true, stream), stream);
  return mx::subtract(shifted, normalizer, stream);
}
std::vector<mx::array> topk(const TopK &operation, const mx::array &input,
                            const mx::array &first, mx::Stream stream) {
  const auto raw = mx::view(first, mx::uint32, stream);
  auto key = mx::where(mx::equal(magnitude(raw, stream), mx::array(0U), stream),
                       mx::array(sign_bit), ordered_key(raw, stream), stream);
  if (operation.largest)
    key = mx::bitwise_invert(key, stream);
  key = mx::where(nan_bits(raw, stream), mx::array(0xffffffffU), key, stream);
  const int axis = operation.axis < 0
                       ? operation.axis + static_cast<int>(input.ndim())
                       : operation.axis;
  auto indices = [&] {
    if (operation.count == 1)
      return mx::argmin(key, axis, true, stream);
    // The pinned MLX GPU argsort is stable; argpartition uses the same full
    // merge sort. Charge its actual O(N log N) work and native scratch.
    auto order = mx::argsort(key, axis, stream);
    mx::Shape start(input.ndim(), 0), stop = input.shape();
    stop[axis] = static_cast<int>(operation.count);
    return mx::slice(order, start, stop, stream);
  }();
  const auto storage = input.dtype() == mx::float32 ? mx::uint32 : mx::uint16;
  auto values = mx::view(mx::take_along_axis(mx::view(input, storage, stream),
                                             indices, axis, stream),
                         input.dtype(), stream);
  return {std::move(values), mx::astype(indices, mx::int32, stream)};
}
} // namespace

Result<std::vector<mx::array>>
lower_reduction(const Operation &operation, std::span<const mx::array> inputs,
                const mx::Shape &output_shape, mx::Stream stream) {
  const auto *reduction = operation.get_if<Reduce>();
  const auto *probability = operation.get_if<Softmax>();
  const auto *selection = operation.get_if<TopK>();
  const auto type = inputs[0].dtype();
  if (reduction && reduction->accumulation_type != DataType::kFloat32 &&
      !(type == mx::float32 &&
        reduction->accumulation_type == DataType::kUnknown))
    return std::unexpected(Status::Unimplemented(
        "Apple Reduce requires F32 accumulation; half must declare it"));
  if (probability && probability->accumulation_type != DataType::kFloat32)
    return std::unexpected(Status::Unimplemented(
        "Apple Softmax requires explicit F32 accumulation"));
  if (probability && probability->mask == MaskKind::kAdditive &&
      inputs[1].dtype() != type)
    return std::unexpected(Status::Unimplemented(
        "Apple additive Softmax mask must match the input storage type"));
  if (inputs[0].size() == 0) {
    if (selection)
      return std::vector{mx::zeros(output_shape, type, stream),
                         mx::zeros(output_shape, mx::int32, stream)};
    return std::vector{mx::full(
        output_shape, mx::array(reduction ? identity(reduction->kind) : 0.0F),
        type, stream)};
  }
  const auto first = mx::astype(inputs[0], mx::float32, stream);
  if (selection)
    return topk(*selection, inputs[0], first, stream);
  const auto result = reduction ? reduce(*reduction, first, stream)
                                : softmax(*probability, inputs, first, stream);
  return std::vector{mx::astype(result, type, stream)};
}
} // namespace hyle::compute::apple::detail
