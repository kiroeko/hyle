#include "ct_packed_linear_region.h"

#include "../graph/artifact_packed_region.h"
#include "../graph/static_fp8_region.h"
#include "../graph/mapped_pattern_region.h"
#include "hyle/compute/providers/cuda.h"

namespace hyle::graph::cuda::internal {
namespace {
const compute::detail::ProviderVTable &original() noexcept {
  return *compute::detail::ProviderAccess::vtable(compute::cuda_native::provider());
}
}
const RegionProviderVTable &ct_packed_linear_regions() noexcept {
  return graph::detail::MappedPatternRegions<&original,
      graph::detail::ArtifactPackedPattern, graph::detail::StaticFloat8Pattern>::kRegion;
}
} // namespace hyle::graph::cuda::internal
