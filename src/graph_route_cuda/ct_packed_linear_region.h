#pragma once

#include "hyle/graph/region_provider.h"

namespace hyle::graph::cuda::internal {

// Cold packed recipes share the existing native CUDA provider and its context.
const RegionProviderVTable &ct_packed_linear_regions() noexcept;

} // namespace hyle::graph::cuda::internal
