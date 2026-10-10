#pragma once

#include "hyle/compute/backends/cuda.h"

namespace hyle::compute::cuda::detail {

// Cold exact-context admission for composition resources; device ordinal alone
// is insufficient for CUDA asynchronous transfer.
HYLE_COMPUTE_BACKEND_CUDA_API Status validate_composition_resources(
    const Context &context, const MemoryResource &host,
    const MemoryResource &device) noexcept;

} // namespace hyle::compute::cuda::detail
