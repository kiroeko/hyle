#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

namespace hyle::compute::ggml_cuda::internal {

// Invalid entries become a safe native address; the original IDs stay intact.
cudaError_t check_indices(const std::int32_t *source, std::int32_t *safe,
                          std::size_t count, std::int64_t extent,
                          unsigned *status, cudaStream_t stream) noexcept;

// Both original slot IDs and the finite selector are checked. A prior error
// or invalid control writes -1, which the original indexed kernels reject
// before any source read or Store destination write.
cudaError_t select_indexed_source(const std::int32_t *sources,
    const std::int32_t *selector, std::int32_t *selected, std::int64_t slots,
    unsigned *status, cudaStream_t stream) noexcept;

// A checked region publishes none of its outputs after any invalid entry.
cudaError_t publish_checked(const void *source, void *destination,
                            std::size_t bytes, const unsigned *status,
                            cudaStream_t stream) noexcept;

} // namespace hyle::compute::ggml_cuda::internal
