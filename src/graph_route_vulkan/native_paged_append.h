#pragma once
#include "hyle/graph/region_provider.h"

namespace hyle::graph::vulkan::detail {
inline constexpr std::uint64_t kNativeAppendRoute = 6;
bool native_append_operation(const compute::Operation &) noexcept;
Status propose_native_append(const RegionDiscoveryRequest &, RegionProposalWriter &) noexcept;
Status native_append_resources(void *, const RegionCompileRequest &,
    std::span<RegionInputRequirement>, std::span<RegionOutputRequirement>,
    RegionResourceProposal *) noexcept;
compute::ProviderPrepareOutcome compile_native_append(void *, const RegionCompileRequest &,
    mem::AllocatorRef, ProviderRegionPrepared *) noexcept;
} // namespace hyle::graph::vulkan::detail
