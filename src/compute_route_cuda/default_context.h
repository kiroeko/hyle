#pragma once

#include "hyle/compute/cuda.h"

namespace hyle::compute::cuda::internal {
enum class DefaultScope : std::uint8_t { kDirect, kGraph };
// Same default roster for Direct and Graph; Graph attaches its region table
// to the ggml implementation without duplicating route ordering or resources.
HYLE_COMPUTE_ROUTE_CUDA_API Result<Context>
create_context(DefaultContextOptions options, Provider ggml,
               DefaultScope scope = DefaultScope::kDirect) noexcept;
} // namespace hyle::compute::cuda::internal
