#include "ct_packed_linear_region.h"

#include "../graph/artifact_packed_region.h"
#include "../graph/static_fp8_region.h"
#include "../graph/mapped_pattern_region.h"
#include "hyle/compute/providers/native.h"

namespace hyle::graph::cpu::internal {
namespace {
const compute::detail::ProviderVTable &original() noexcept {
  return *compute::detail::ProviderAccess::vtable(compute::native::provider());
}
}
compute::Provider ct_packed_regions(compute::Provider provider) noexcept {
  return graph::detail::MappedPatternRegions<&original,
      graph::detail::ArtifactPackedPattern, graph::detail::StaticFloat8Pattern>::decorate(std::move(provider));
}
} // namespace hyle::graph::cpu::internal
