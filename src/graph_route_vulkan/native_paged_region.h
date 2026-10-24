#pragma once

#include "hyle/graph/region_provider.h"

namespace hyle::graph::vulkan::detail {
bool native_paged_operation(const compute::Operation &) noexcept;
Status propose_native_paged(void *, const RegionDiscoveryRequest &,
    std::span<const std::uint8_t> needed, RegionProposalWriter &) noexcept;
Status native_paged_resources(void *, const RegionCompileRequest &,
    std::span<RegionInputRequirement>, std::span<RegionOutputRequirement>,
    RegionResourceProposal *) noexcept;
compute::ProviderPrepareOutcome compile_native_paged(void *,
    const RegionCompileRequest &, mem::AllocatorRef, ProviderRegionPrepared *) noexcept;
} // namespace hyle::graph::vulkan::detail
