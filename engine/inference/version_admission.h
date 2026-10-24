#pragma once

#include <cstddef>
#include <ranges>
#include <stdexcept>

namespace hyle::inference {

// Borrowed current facts, not another owner inventory. In the present engine
// these are the actual Session and its canonical Snapshot identity. A selected
// {nullptr, nullptr} requests one new head; an empty resident owns no version.
template<class Session, class Version> struct VersionHolder final {
  const Session *session = nullptr;
  const Version *canonical = nullptr;
};
template<class Session, class Version>
VersionHolder(const Session *, const Version *) -> VersionHolder<Session, Version>;

struct VersionCapacity final {
  std::size_t live = 0, limit = 0;
};
struct VersionAdmission final {
  // Proven replaceable bases, not a debit or a promise of native retirement.
  // When spare capacity already fits, no owner scan is needed and this is 0.
  std::size_t retiring = 0;
  bool fits = false;
};

// All ranges/getters are inspected synchronously and may be traversed more than
// once. Real selected Sessions must be unique; canonical identities may alias.
// The caller supplies its complete resident/other-retainer facts, verifies the
// original Model/Session/retirement gates, and rereads live after each eviction.
// No inference from N=1 or prefix-off grants exclusive ownership. The original
// callers may keep a separately proven sole-owner fast path outside this query.
template<std::ranges::forward_range Selected, std::ranges::forward_range Residents,
         class Retained>
  requires std::ranges::sized_range<Selected>
VersionAdmission version_admission(VersionCapacity capacity, Selected &&selected,
                                   Residents &&residents, Retained &&retained) {
  if (!capacity.limit || capacity.live > capacity.limit)
    throw std::runtime_error("retained version capacity is invalid");
  const auto requested = static_cast<std::size_t>(std::ranges::size(selected));
  if (requested <= capacity.limit - capacity.live) return {0, true};
  std::size_t retiring = 0, index = 0;
  for (const auto holder : selected) {
    const auto *canonical = holder.canonical;
    if (!canonical) { ++index; continue; }
    bool seen = false;
    std::size_t prior = 0;
    for (const auto other : selected) {
      if (prior++ == index) break;
      if (other.canonical == canonical) { seen = true; break; }
    }
    ++index;
    if (seen || retained(canonical)) continue;
    bool external = false;
    for (const auto resident : residents) {
      if (resident.canonical != canonical) continue;
      bool chosen = false;
      for (const auto candidate : selected)
        if (candidate.session == resident.session) { chosen = true; break; }
      if (!chosen) { external = true; break; }
    }
    if (!external) ++retiring;
  }
  if (retiring > capacity.live)
    throw std::runtime_error("replaceable version inventory exceeds live versions");
  return {retiring, requested <= capacity.limit - (capacity.live - retiring)};
}

} // namespace hyle::inference
