#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace hyle::compute::cuda::detail {
// CUDA-only wire geometry keeps C++23 host ownership headers out of NVCC.
struct IndexedCopyLayout final {
  std::uint32_t slot_count = 0, row_count = 0;
  std::size_t row_bytes = 0, slot_stride = 0;
};
const void *indexed_copy_check_kernel() noexcept;
const void *indexed_copy_rows_kernel() noexcept;
cudaError_t launch_indexed_copy(const void *check_kernel, const void *copy_kernel,
    void *capacity, const void *rows, const std::int32_t *indices,
    IndexedCopyLayout, unsigned *status, cudaStream_t stream) noexcept;
} // namespace hyle::compute::cuda::detail
