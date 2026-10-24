#pragma once

#include "hyle/graph/region_provider.h"

namespace hyle::graph::vulkan::detail {
inline constexpr std::uint64_t kNativeRaggedRoute = 3;
bool native_ragged_operation(const compute::Operation &) noexcept;
Status propose_native_ragged(const RegionDiscoveryRequest &,
    std::span<const std::uint8_t> needed, RegionProposalWriter &) noexcept;
Status native_ragged_resources(void *, const RegionCompileRequest &,
    std::span<RegionInputRequirement>, std::span<RegionOutputRequirement>,
    RegionResourceProposal *) noexcept;
compute::ProviderPrepareOutcome compile_native_ragged(void *,
    const RegionCompileRequest &, mem::AllocatorRef, ProviderRegionPrepared *) noexcept;
} // namespace hyle::graph::vulkan::detail
