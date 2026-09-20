#pragma once

#include "hyle/compute/cuda.h"

namespace hyle::compute::cuda::internal {
// Same default roster for Direct and Graph; Graph attaches its region table
// to the ggml implementation without duplicating route ordering or resources.
HYLE_COMPUTE_ROUTE_CUDA_API Result<Context>
create_context(DefaultContextOptions options, Provider ggml) noexcept;
} // namespace hyle::compute::cuda::internal
