#include "cuda_indexed_copy.h"

#include <cuda_runtime.h>

#include "device_status_kernel.h"

namespace hyle::compute::cuda::detail {
namespace {
constexpr unsigned kThreads = 256;

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
  for (std::size_t row = blockIdx.x; row < layout.row_count; row += gridDim.x) {
    const auto selected = indices[row];
    // The original caller holds these controls immutable through retirement.
    // Retain a defensive bounds guard without reading a destination byte.
    if (selected < 0 || static_cast<std::uint32_t>(selected) >= layout.slot_count)
      return;
    auto *destination = capacity + std::size_t(selected) * layout.slot_stride;
    const auto *source = rows + row * layout.row_bytes;
    for (std::size_t byte = threadIdx.x; byte < layout.row_bytes; byte += blockDim.x)
      destination[byte] = source[byte];
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
  const auto required = (std::size_t(layout.row_count) - 1) / kThreads + 1;
  const auto check_blocks = static_cast<unsigned>(required < 65535U ? required : 65535U);
  void *check_arguments[]{&indices, &layout, &status};
  auto result = cudaLaunchKernel(check_kernel, dim3(check_blocks), dim3(kThreads),
                                check_arguments, 0, stream);
  if (result != cudaSuccess)
    return result;
  const auto copy_blocks = layout.row_count < 65535U ? layout.row_count : 65535U;
  void *copy_arguments[]{&capacity, &rows, &indices, &layout, &status};
  return cudaLaunchKernel(copy_kernel, dim3(copy_blocks), dim3(kThreads),
                          copy_arguments, 0, stream);
}
} // namespace hyle::compute::cuda::detail
