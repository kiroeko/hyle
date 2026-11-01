#include "cuda_indexed_copy.h"

#include <cuda_runtime.h>

#include "device_status_kernel.h"

namespace hyle::compute::cuda::detail {
namespace {
constexpr unsigned kThreads = 256;
constexpr std::size_t kTileBytes = 16U * 1024U;
constexpr unsigned kMaxBlocks = 65535;
constexpr std::size_t kVectorBytes = sizeof(uint4);
static_assert(kTileBytes % kVectorBytes == 0 && kVectorBytes == 16);

__global__ void check_indices(const std::int32_t *indices,
                             IndexedCopyLayout layout, unsigned *status) {
  for (std::size_t row = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       row < layout.row_count; row += std::size_t(gridDim.x) * blockDim.x) {
    const auto selected = indices[row];
    if (selected < 0 || static_cast<std::uint32_t>(selected) >= layout.slot_count)
      atomicCAS(status, 0U, static_cast<unsigned>(DeviceStatusCode::kIndexOutOfRange));
  }
}

__global__ void copy_rows(unsigned char *capacity, const unsigned char *rows,
                         const std::int32_t *indices, IndexedCopyLayout layout,
                         const unsigned *status) {
  // The separate preceding check has finished for ALL IDs. A bad index or an
  // earlier ordered-source error therefore prevents every payload write.
  if (*status != 0)
    return;
  for (std::size_t row = blockIdx.y; row < layout.row_count; row += gridDim.y) {
    const auto selected = indices[row];
    // The original caller holds these controls immutable through retirement.
    // Retain a defensive bounds guard without reading a destination byte.
    if (selected < 0 || static_cast<std::uint32_t>(selected) >= layout.slot_count)
      return;
    auto *destination = capacity + std::size_t(selected) * layout.slot_stride;
    const auto *source = rows + row * layout.row_bytes;
    const auto tiles = (layout.row_bytes - 1) / kTileBytes + 1;
    const bool vectorized =
        (reinterpret_cast<std::uintptr_t>(destination) | reinterpret_cast<std::uintptr_t>(source)) % kVectorBytes == 0;
    // Every original row/backing remains disjoint. Several CTAs may copy its
    // nonoverlapping tiles; the native sequence still owns one complete use.
    for (std::size_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
      const auto offset = tile * kTileBytes;
      const auto remaining = layout.row_bytes - offset;
      const auto bytes = remaining < kTileBytes ? remaining : kTileBytes;
      if (vectorized) {
        auto *vectors_out = reinterpret_cast<uint4 *>(destination + offset);
        const auto *vectors_in = reinterpret_cast<const uint4 *>(source + offset);
        const auto vectors = bytes / kVectorBytes;
        for (std::size_t index = threadIdx.x; index < vectors; index += blockDim.x)
          vectors_out[index] = vectors_in[index];
        const auto tail = vectors * kVectorBytes;
        for (std::size_t byte = tail + threadIdx.x; byte < bytes; byte += blockDim.x)
          destination[offset + byte] = source[offset + byte];
      } else {
        for (std::size_t byte = threadIdx.x; byte < bytes; byte += blockDim.x)
          destination[offset + byte] = source[offset + byte];
      }
    }
  }
}
} // namespace

const void *indexed_copy_check_kernel() noexcept {
  return reinterpret_cast<const void *>(check_indices);
}
const void *indexed_copy_rows_kernel() noexcept {
  return reinterpret_cast<const void *>(copy_rows);
}
cudaError_t launch_indexed_copy(const void *check_kernel, const void *copy_kernel,
    void *capacity, const void *rows, const std::int32_t *indices,
    IndexedCopyLayout layout, unsigned *status, cudaStream_t stream) noexcept {
  // The typed host preparation checks these extents too. Keep this private
  // launch seam safe independently, before an index/status kernel is queued.
  if (!layout.slot_count || !layout.row_count || layout.row_count > layout.slot_count ||
      !layout.row_bytes || layout.slot_stride < layout.row_bytes ||
      layout.row_count > SIZE_MAX / layout.row_bytes ||
      std::size_t(layout.slot_count - 1) > (SIZE_MAX - layout.row_bytes) / layout.slot_stride)
    return cudaErrorInvalidValue;
  const auto required = (std::size_t(layout.row_count) - 1) / kThreads + 1;
  const auto check_blocks = static_cast<unsigned>(required < kMaxBlocks ? required : kMaxBlocks);
  void *check_arguments[]{&indices, &layout, &status};
  auto result = cudaLaunchKernel(check_kernel, dim3(check_blocks), dim3(kThreads),
                                check_arguments, 0, stream);
  if (result != cudaSuccess)
    return result;
  const auto tiles = (layout.row_bytes - 1) / kTileBytes + 1;
  const auto tile_blocks = static_cast<unsigned>(tiles < kMaxBlocks ? tiles : kMaxBlocks);
  const auto maximum_rows = kMaxBlocks / tile_blocks;
  const auto row_blocks = layout.row_count < maximum_rows ? layout.row_count : maximum_rows;
  void *copy_arguments[]{&capacity, &rows, &indices, &layout, &status};
  return cudaLaunchKernel(copy_kernel, dim3(tile_blocks, row_blocks), dim3(kThreads),
                          copy_arguments, 0, stream);
}
} // namespace hyle::compute::cuda::detail
