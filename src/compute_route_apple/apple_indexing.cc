#include "apple_indexing.h"

#include <cstdint>
#include <limits>

#include <mlx/ops.h>

#include "apple_checks.h"

namespace hyle::compute::apple::detail {
namespace mx = mlx::core;
namespace {
bool empty(const mx::Shape &shape) {
  for (int dimension : shape)
    if (dimension == 0)
      return true;
  return false;
}
mx::array error_code(const mx::array &bad, std::int32_t code,
                     mx::Stream stream) {
  return mx::where(mx::any(bad, false, stream), mx::array(code), mx::array(0),
                   stream);
}
Result<mx::array> gather(const Gather &operation,
                         std::span<const mx::array> inputs,
                         const mx::Shape &shape, mx::Stream stream) {
  const auto &source = inputs[0];
  auto indices = inputs[1];
  if (empty(shape))
    return mx::zeros(shape, source.dtype(), stream);
  const int axis = operation.axis < 0
                       ? operation.axis + static_cast<int>(source.ndim())
                       : operation.axis;
  const int extent = source.shape(axis);
  const bool checked = operation.bounds == IndexBoundsPolicy::kChecked;
  if (extent == 0) {
    if (!checked)
      return std::unexpected(Status::OutOfRange(
          "Apple unchecked Gather cannot read an empty selected axis"));
    return with_device_status(mx::zeros(shape, source.dtype(), stream),
                              mx::array(kIndexOutOfRange), stream);
  }
  auto status = mx::array(0);
  if (checked) {
    const auto zero = mx::array(0, indices.dtype());
    const auto limit = mx::array(extent, indices.dtype());
    status = error_code(
        mx::logical_or(mx::less(indices, zero, stream),
                       mx::greater_equal(indices, limit, stream), stream),
        kIndexOutOfRange, stream);
    // Native Take wraps negative indices and does not report HYLE bounds
    // errors. Keep every native access safe, then suppress the failed lease.
    indices = mx::minimum(mx::maximum(indices, zero, stream),
                          mx::array(extent - 1, indices.dtype()), stream);
  }
  const auto storage =
      source.dtype() == mx::float32 ? mx::uint32
      : (source.dtype() == mx::float16 || source.dtype() == mx::bfloat16)
          ? mx::uint16
          : source.dtype();
  auto output =
      mx::take(mx::view(source, storage, stream), indices, axis, stream);
  output = mx::view(output, source.dtype(), stream);
  if (checked)
    return with_device_status(std::move(output), std::move(status), stream);
  return output;
}
Result<mx::array> scan(const Scan &operation, const mx::array &input,
                       mx::Stream stream) {
  if (input.size() == 0)
    return mx::zeros(input.shape(), input.dtype(), stream);
  if (input.dtype() == mx::float32)
    return mx::cumsum(input, operation.axis, false, !operation.exclusive,
                      stream);
  // At most INT32_MAX I32 values are accumulated, so every prefix fits I64.
  auto prefix = mx::cumsum(mx::astype(input, mx::int64, stream), operation.axis,
                           false, true, stream);
  const auto bad = mx::logical_or(
      mx::less(
          prefix,
          mx::array(std::int64_t{(std::numeric_limits<std::int32_t>::min)()}),
          stream),
      mx::greater(
          prefix,
          mx::array(std::int64_t{(std::numeric_limits<std::int32_t>::max)()}),
          stream),
      stream);
  const auto status = error_code(bad, kScanOverflow, stream);
  if (operation.exclusive) {
    const int axis = operation.axis < 0
                         ? operation.axis + static_cast<int>(input.ndim())
                         : operation.axis;
    mx::Shape zero_shape = input.shape(), start(input.ndim(), 0),
              stop = input.shape();
    zero_shape[axis] = 1;
    --stop[axis];
    prefix = mx::concatenate({mx::zeros(zero_shape, mx::int64, stream),
                              mx::slice(prefix, start, stop, stream)},
                             axis, stream);
  }
  return with_device_status(mx::astype(prefix, mx::int32, stream), status,
                            stream);
}
} // namespace

Result<std::vector<mx::array>> lower_indexing(const Operation &operation,
                                              std::span<const mx::array> inputs,
                                              const mx::Shape &output_shape,
                                              mx::Stream stream) {
  constexpr auto maximum = (std::numeric_limits<std::int32_t>::max)();
  if (!empty(output_shape)) {
    std::int64_t count = 1;
    for (int dimension : output_shape) {
      if (count > maximum / dimension)
        return std::unexpected(Status::Unimplemented(
            "Apple indexed output exceeds the INT32_MAX element domain"));
      count *= dimension;
    }
    if ((operation.kind() == OperationKind::kGather ||
         operation.kind() == OperationKind::kScatter) &&
        inputs[1].size() > maximum)
      return std::unexpected(Status::Unimplemented(
          "Apple indices exceed the INT32_MAX element domain"));
    if (operation.kind() == OperationKind::kScatter &&
        inputs[2].size() > maximum)
      return std::unexpected(Status::Unimplemented(
          "Apple Scatter updates exceed the INT32_MAX element domain"));
  }
  if (const auto *scatter = operation.get_if<Scatter>()) {
    HYLE_ASSIGN_OR_RETURN(auto result, lower_scatter(*scatter, inputs, stream));
    return std::vector{std::move(result)};
  }
  auto result =
      operation.kind() == OperationKind::kGather
          ? gather(*operation.get_if<Gather>(), inputs, output_shape, stream)
          : scan(*operation.get_if<Scan>(), inputs[0], stream);
  if (!result)
    return std::unexpected(std::move(result).error());
  return std::vector{std::move(*result)};
}
} // namespace hyle::compute::apple::detail
