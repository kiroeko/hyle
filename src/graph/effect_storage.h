#pragma once

#include "hyle/graph/region_provider.h"

namespace hyle::graph::detail {

// Borrow only original cold node-placement facts. The artifact retains its
// selected resource, never these spans. Tensor source locality is independent
// of this caller-owned effect bank and may require an ordinary Graph copy.
struct EffectStoragePlacement final {
  std::span<const NodePlacement> nodes;
  std::span<const StorageDomain> domains;
  ExecutionTargetId target;
  mem::MemoryResourceRef fallback;

  Result<std::uint32_t> preference(NodeId node,
      mem::MemoryResourceRef resource) const noexcept {
    const NodePlacement *required = nullptr, *preferred = nullptr;
    for (const auto &placement : nodes) {
      if (placement.node != node || placement.target != target || !placement.default_storage.valid()) continue;
      if (placement.strength == PlacementStrength::kRequired) required = &placement;
      else preferred = &placement;
    }
    const auto *selected = required ? required : preferred;
    if (!selected) return resource == fallback ? 0u : 1u;
    const auto storage = selected->default_storage;
    if (storage.ordinal() >= domains.size())
      return std::unexpected(Status::InvalidArgument("effect node storage domain is not configured"));
    if (domains[storage.ordinal()].resource == resource) return 0u;
    if (required)
      return std::unexpected(Status::Unimplemented("effect bank differs from required node storage"));
    return 1u;
  }
};

} // namespace hyle::graph::detail
