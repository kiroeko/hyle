#pragma once

#include "hyle/graph/region_provider.h"

namespace hyle::graph::apple::detail {
inline constexpr std::uint64_t kNativeSlotStoreRoute = 7;
bool native_slot_store_operation(const compute::Operation &) noexcept;
Status propose_native_slot_store(void *, const RegionDiscoveryRequest &,
    std::span<const std::uint8_t>, RegionProposalWriter &) noexcept;
Status native_slot_store_resources(void *, const RegionCompileRequest &,
    std::span<RegionInputRequirement>, std::span<RegionOutputRequirement>,
    RegionResourceProposal *) noexcept;
compute::ProviderPrepareOutcome compile_native_slot_store(void *,
    const RegionCompileRequest &, mem::AllocatorRef, ProviderRegionPrepared *) noexcept;
} // namespace hyle::graph::apple::detail
