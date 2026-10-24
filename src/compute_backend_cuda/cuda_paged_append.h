#pragma once

#include <cuda_runtime_api.h>

#include "hyle/compute/detail/paged_append_wire.h"

namespace hyle::compute::cuda::detail {

const void *paged_append_kernel() noexcept;
cudaError_t launch_paged_append(
    const void *kernel, void *arena, const void *rows, const void *metadata,
    compute::detail::PagedAppendNativeSpec spec, cudaStream_t stream) noexcept;

} // namespace hyle::compute::cuda::detail
