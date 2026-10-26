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

struct VersionRoomObservation final {
  VersionCapacity capacity;
  std::size_t evictable = 0;
  bool fits = false;
};
enum class VersionRoomResult { kReady, kRetained };

// Serialized directory eviction only: observe rereads actual live versions,
// candidate entries and the complete admission query. Removing an alias is not
// proof that a version or any physical bytes were freed. Native retirement and
// its errors remain with the resource owner, outside this loop.
//
// Each successful evict must consume candidate inventory. The fixed initial
// bound therefore limits attempts, even when aliases keep every version live.
// The callbacks borrow all ownership; valid traversal allocates nothing. False
// eviction stops admission, and exceptions propagate without retry or rollback.
template<class Observe, class Evict>
VersionRoomResult make_version_room(Observe &&observe, Evict &&evict) {
  VersionRoomObservation current = observe();
  if (!current.capacity.limit || current.capacity.live > current.capacity.limit)
    throw std::logic_error("version room requires valid current capacity");
  if (current.fits) return VersionRoomResult::kReady;
  const auto attempts = current.evictable;
  for (std::size_t attempt = 0; attempt < attempts; ++attempt) {
    if (!current.evictable || !evict()) return VersionRoomResult::kRetained;
    const VersionRoomObservation next = observe();
    if (next.capacity.limit != current.capacity.limit ||
        next.capacity.live > current.capacity.live || next.evictable >= current.evictable)
      throw std::logic_error("version room eviction changed capacity or failed to reduce its inventory");
    current = next;
    if (current.fits) return VersionRoomResult::kReady;
  }
  return VersionRoomResult::kRetained;
}

} // namespace hyle::inference
