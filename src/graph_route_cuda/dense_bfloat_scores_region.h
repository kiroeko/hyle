#pragma once

#include "hyle/compute/provider.h"

namespace hyle::graph::cuda::internal {

// Decorates only the original native CUDA provider's Graph capability. Its
// identity, configuration ownership and execution/context tables are retained.
compute::Provider dense_bfloat_scores_provider() noexcept;

} // namespace hyle::graph::cuda::internal
