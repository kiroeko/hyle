#include "apple_indexing.h"

#include <optional>
#include <utility>

#include <mlx/fast.h>
#include <mlx/ops.h>

#include "apple_checks.h"

namespace hyle::compute::apple::detail {
namespace mx = mlx::core;
namespace {
// MLX owns compilation, layout materialization, temporaries and submission.
// Each output element has one writer. One shared search per threadgroup
// avoids repeating the binary search for every feature in an indexed row.
constexpr auto kOrderedScatter = R"metal(
  const uint lane = thread_position_in_grid.x;
  const int target = int(thread_position_in_grid.y);
  threadgroup uint2 range;
  if (thread_index_in_threadgroup == 0) {
    uint low = 0, high = Count;
    while (low < high) {
      const uint mid = low + (high - low) / 2;
      if (keys[order[mid]] < target) low = mid + 1;
      else high = mid;
    }
    range.x = low;
    high = Count;
    while (low < high) {
      const uint mid = low + (high - low) / 2;
      if (keys[order[mid]] <= target) low = mid + 1;
      else high = mid;
    }
    range.y = low;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint outer = lane / Inner, inner = lane % Inner;
  const uint destination = (outer * Extent + uint(target)) * Inner + inner;
  Storage value = base[destination];
  if constexpr (Adding) {
    for (uint cursor = range.x; cursor < range.y; ++cursor) {
      const uint offset = (outer * Count + order[cursor]) * Inner + inner;
      const float sum = float(as_type<Value>(value)) +
                        float(as_type<Value>(updates[offset]));
      value = as_type<Storage>(Value(sum));
    }
  } else if (range.x != range.y) {
    const uint offset = (outer * Count + order[range.y - 1]) * Inner + inner;
    value = updates[offset];
  }
  output[destination] = value;
)metal";
} // namespace

Result<mx::array> lower_scatter(const Scatter &operation,
                                std::span<const mx::array> inputs,
                                mx::Stream stream) {
  const auto &source = inputs[0];
  const auto type = source.dtype();
  const bool floating =
      type == mx::float32 || type == mx::float16 || type == mx::bfloat16;
  const bool adding = operation.kind == ScatterKind::kAdd;
  if (adding && !floating)
    return std::unexpected(Status::Unimplemented(
        "Apple Scatter add requires F32/F16/BF16 storage"));
  if (source.size() == 0)
    return mx::zeros(source.shape(), type, stream);
  const auto storage = type == mx::float32 ? mx::uint32
                       : floating          ? mx::uint16
                                           : type;
  const auto base = mx::view(source, storage, stream);
  if (inputs[1].size() == 0)
    return mx::view(mx::copy(base, stream), type, stream);
  const int axis = operation.axis < 0
                       ? operation.axis + static_cast<int>(source.ndim())
                       : operation.axis;
  const int extent = source.shape(axis);
  const int count = static_cast<int>(inputs[1].size());
  auto indices = mx::reshape(inputs[1], {count}, stream);
  const bool checked = operation.bounds == IndexBoundsPolicy::kChecked;
  auto status = mx::array(0);
  if (checked) {
    auto bad = mx::logical_or(
        mx::less(indices, mx::array(0, indices.dtype()), stream),
        mx::greater_equal(indices, mx::array(extent, indices.dtype()), stream),
        stream);
    status = mx::where(mx::any(bad, false, stream), mx::array(kIndexOutOfRange),
                       mx::array(0), stream);
    indices =
        mx::minimum(mx::maximum(indices, mx::array(0, indices.dtype()), stream),
                    mx::array(extent - 1, indices.dtype()), stream);
  }
  auto keys = mx::astype(indices, mx::int32, stream);
  auto order = count == 1 ? mx::zeros({1}, mx::uint32, stream)
                          : mx::argsort(keys, 0, stream);
  int inner = 1;
  for (std::size_t dimension = axis + 1; dimension < source.ndim(); ++dimension)
    inner *= source.shape(dimension);
  const int lanes = static_cast<int>(source.size()) / extent;
  const auto kernel = mx::fast::metal_kernel(
      "hyle_ordered_scatter", {"base", "keys", "order", "updates"}, {"output"},
      kOrderedScatter);
  auto output =
      kernel({mx::reshape(base, {-1}, stream), keys, order,
              mx::reshape(mx::view(inputs[2], storage, stream), {-1}, stream)},
             {source.shape()}, {storage}, {lanes, extent, 1}, {256, 1, 1},
             {{"Storage", storage},
              {"Value", type},
              {"Adding", adding},
              {"Inner", inner},
              {"Extent", extent},
              {"Count", count}},
             std::nullopt, false, stream)[0];
  output = mx::view(output, type, stream);
  if (checked)
    return with_device_status(std::move(output), std::move(status), stream);
  return output;
}
} // namespace hyle::compute::apple::detail
