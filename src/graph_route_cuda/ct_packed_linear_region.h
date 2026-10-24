#pragma once

#include "hyle/compute/provider.h"

namespace hyle::graph::cuda::internal {

// Same original artifact Provider, with a private Graph contraction capability.
// Configuration ownership and all non-region hooks remain with that Provider.
compute::Provider ct_packed_linear_provider() noexcept;

} // namespace hyle::graph::cuda::internal
