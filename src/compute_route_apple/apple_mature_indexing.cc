#include "apple_mature_indexing.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>

#include <mlx/ops.h>

#include "apple_checks.h"

namespace hyle::compute::apple::detail {
namespace mx = mlx::core;
namespace {
int native_axis(std::int32_t axis, const mx::array &input) {
  return axis < 0 ? axis + static_cast<int>(input.ndim()) : axis;
}

Result<std::vector<mx::array>> gather(const Gather &operation,
                                    std::span<const mx::array> inputs,
                                    mx::Stream stream) {
  const auto axis = native_axis(operation.axis, inputs[0]);
  const auto extent = inputs[0].shape(axis);
  if (extent == 0)
    return std::unexpected(
        Status::OutOfRange("Apple Gather cannot index an empty source axis"));
  if (operation.bounds == IndexBoundsPolicy::kUnchecked)
    return std::vector{mx::take(inputs[0], inputs[1], axis, stream)};

  const auto index_type = inputs[1].dtype();
  auto invalid = mx::logical_or(
      mx::less(inputs[1], mx::array(0, index_type), stream),
      mx::greater_equal(inputs[1], mx::array(extent, index_type), stream),
      stream);
  // Native take must never see an invalid index, even though the checked
  // invocation will fail. Only its status scalar reaches the host.
  auto safe = mx::where(invalid, mx::array(0, index_type), inputs[1], stream);
  auto output = mx::take(inputs[0], safe, axis, stream);
  auto status = mx::where(mx::any(invalid, stream), mx::array(kIndexOutOfRange),
                          mx::array(0), stream);
  HYLE_ASSIGN_OR_RETURN(
      auto checked,
      with_device_status(std::move(output), std::move(status), stream));
  return std::vector{std::move(checked)};
}

Result<std::vector<mx::array>> scan(const Scan &operation,
                                  const mx::array &input, mx::Stream stream) {
  const auto axis = native_axis(operation.axis, input);
  if (input.dtype() == mx::float32)
    return std::vector{
        mx::cumsum(input, axis, false, !operation.exclusive, stream)};

  // The native axis length fits int, so every sum of I32 values fits I64.
  // Check the inclusive total as well, including for an exclusive result.
  auto prefix = mx::cumsum(input, axis, false, true, mx::int64, stream);
  constexpr std::int64_t minimum = (std::numeric_limits<std::int32_t>::min)();
  constexpr std::int64_t maximum = (std::numeric_limits<std::int32_t>::max)();
  auto invalid = mx::any(
      mx::logical_or(mx::less(prefix, mx::array(minimum), stream),
                     mx::greater(prefix, mx::array(maximum), stream), stream),
      stream);
  auto output = prefix;
  if (operation.exclusive) {
    mx::Shape start(input.ndim(), 0), stop = input.shape();
    --stop[axis];
    auto zero_shape = input.shape();
    zero_shape[axis] = 1;
    output = mx::concatenate(
        {mx::zeros(zero_shape, mx::int64, stream),
         mx::slice(prefix, start, stop, stream)},
        axis, stream);
  }
  auto status = mx::where(invalid, mx::array(kScanOverflow), mx::array(0), stream);
  HYLE_ASSIGN_OR_RETURN(
      auto checked,
      with_device_status(mx::astype(output, mx::int32, stream),
                         std::move(status), stream));
  return std::vector{std::move(checked)};
}

std::vector<mx::array> top_k(const TopK &operation, const mx::array &input,
                            mx::Stream stream) {
  const auto axis = native_axis(operation.axis, input);
  // Original MLX argsort is stable and sorts NaNs last. Negate only widened
  // keys: I32_MIN remains representable and output values keep storage bits.
  auto keys = mx::astype(input, input.dtype() == mx::int32 ? mx::int64
                                                         : mx::float32,
                         stream);
  if (operation.largest)
    keys = mx::negative(keys, stream);
  auto order = mx::argsort(keys, axis, stream);
  mx::Shape start(input.ndim(), 0), stop = input.shape();
  stop[axis] = static_cast<int>(operation.count);
  order = mx::slice(order, start, stop, stream);
  return {mx::take_along_axis(input, order, axis, stream),
          mx::astype(order, mx::int32, stream)};
}
} // namespace

Result<std::vector<mx::array>> lower_mature_indexing(
    const Operation &operation, std::span<const mx::array> inputs,
    const mx::Shape &output_shape, mx::Stream stream) {
  const auto type = inputs[0].dtype();
  if (operation.kind() == OperationKind::kTopK && type != mx::float32 &&
      type != mx::float16 && type != mx::bfloat16 && type != mx::int32)
    return std::unexpected(
        Status::Unimplemented("Apple TopK requires F32/F16/BF16/I32"));
  if (operation.kind() == OperationKind::kScan && type != mx::float32 &&
      type != mx::int32)
    return std::unexpected(
        Status::Unimplemented("Apple Scan requires F32/I32"));
  if (std::find(output_shape.begin(), output_shape.end(), 0) !=
      output_shape.end()) {
    std::vector outputs{mx::zeros(output_shape, type, stream)};
    if (operation.kind() == OperationKind::kTopK)
      outputs.push_back(mx::zeros(output_shape, mx::int32, stream));
    return outputs;
  }
  if (operation.kind() == OperationKind::kGather ||
      operation.kind() == OperationKind::kScan) {
    constexpr auto limit =
        static_cast<std::size_t>((std::numeric_limits<std::int32_t>::max)());
    std::size_t count = 1;
    for (auto dimension : output_shape) {
      if (count > limit / static_cast<std::size_t>(dimension))
        return std::unexpected(Status::OutOfRange(
            "Apple indexed output exceeds the admitted I32 element range"));
      count *= static_cast<std::size_t>(dimension);
    }
    if (operation.kind() == OperationKind::kGather && inputs[1].size() > limit)
      return std::unexpected(Status::OutOfRange(
          "Apple Gather indices exceed the admitted I32 element range"));
  }
  if (const auto *value = operation.get_if<Gather>())
    return gather(*value, inputs, stream);
  if (const auto *value = operation.get_if<Scan>())
    return scan(*value, inputs[0], stream);
  if (const auto *value = operation.get_if<TopK>())
    return top_k(*value, inputs[0], stream);
  return std::unexpected(Status::Internal("Apple indexed operation not bound"));
}
} // namespace hyle::compute::apple::detail
