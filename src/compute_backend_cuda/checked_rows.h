#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

#include "hyle/compute/visibility.h"

namespace hyle::compute::cuda::detail {

// Same-build native helpers inside the caller's already admitted CUDA stream.
// Offsets are immutable I32[2] for one original packed sequence. Validation
// preserves an earlier device error and atomically records the first failure.
// This interface owns no storage, permissions, stream or completion.
HYLE_COMPUTE_BACKEND_CUDA_API cudaError_t check_packed_r1_offsets_expected(
    const std::int32_t *offsets, std::int64_t capacity,
    std::int32_t expected_rows, unsigned *status, cudaStream_t stream) noexcept;

// The caller retains a disjoint private result prefix and the complete public
// destination through original retirement. Nonzero status performs no payload
// access. Otherwise copy only [0,active_bytes) and write positive-zero bytes to
// [active_bytes,total_bytes). A null source is valid only for an empty prefix.
HYLE_COMPUTE_BACKEND_CUDA_API cudaError_t publish_checked_prefix(
    const void *source, void *destination, std::size_t active_bytes,
    std::size_t total_bytes, const unsigned *status, cudaStream_t stream) noexcept;

// Combined publication for immutable I32[2] offsets and a CPU-bounded L. Each
// CTA independently validates [0,L] and the inherited original device status
// before any payload access. Invalid controls publish no bytes and retain the
// first error. Capacity and row_bytes are positive; 0 <= L <= capacity. The
// public destination, active private source, offsets and status are disjoint.
HYLE_COMPUTE_BACKEND_CUDA_API cudaError_t publish_packed_r1_prefix(
    const void *source, void *destination, std::size_t row_bytes,
    std::int64_t capacity, std::int32_t expected_rows,
    const std::int32_t *offsets, unsigned *status, cudaStream_t stream) noexcept;

} // namespace hyle::compute::cuda::detail
