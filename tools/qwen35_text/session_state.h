#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "hyle/compute/paged_arena.h"

namespace qwen35::cache {
namespace c = hyle::compute;

inline std::size_t add(std::size_t a, std::size_t b) {
  if (b > (std::numeric_limits<std::size_t>::max)() - a)
    throw std::runtime_error("cache backing byte count overflows");
  return a + b;
}
inline std::size_t multiply(std::size_t a, std::size_t b) {
  if (a && b > (std::numeric_limits<std::size_t>::max)() / a)
    throw std::runtime_error("cache backing byte count overflows");
  return a * b;
}

// Target and draft inventories use the same page geometry but independent
// arena blocks and policy references; this policy is not a Compute owner.
struct Plan final {
  static constexpr std::uint32_t page_rows = 16;
  std::size_t resident_sessions = 0, active_sequences = 1, recurrent_payload_bytes = 0;
  std::uint32_t max_requests = 0, max_query_rows = 0, max_kv_rows = 0;
  std::uint32_t max_pages = 0, max_page_refs = 0, state_slots = 0;
  std::uint32_t snapshot_slots = 0, candidate_boundaries = 1, hidden_slots = 0;
  std::size_t bank_bytes = 0, hidden_bytes = 0, backing_bytes = 0;
  bool draft_cache = false;
};
inline Plan plan(std::size_t context, std::size_t residents,
                 std::size_t query_rows_per_request, std::size_t recurrent_payload,
                  std::size_t active_sequences = 1,
                   std::size_t candidate_boundaries = 1,
                   bool draft_cache = false) {
  if (!context || !residents || !query_rows_per_request || query_rows_per_request > context ||
      !active_sequences || active_sequences > residents ||
      (candidate_boundaries != 1 && candidate_boundaries != 2) ||
      (draft_cache && candidate_boundaries != 2))
    throw std::runtime_error("cache plan requires valid context/query rows, one to N active sequences and one or two candidate boundaries; draft cache requires two");
  const auto refs = add(context, Plan::page_rows - 1) / Plan::page_rows;
  // One batch retains at most R candidates beside N committed versions.
  // Untouched pages are shared; each candidate can retain one old partial
  // COW tail beyond the N complete-context capacities. Publication/admission
  // must still preserve the N committed-version limit after the batch.
  const auto pages = add(multiply(refs, residents), active_sequences);
  const auto snapshots = add(residents, active_sequences);
  // Verify can materialize both post-token alternatives without creating a
  // second immutable Snapshot. Complete MTP heads may retain carry and prior
  // hidden rows together, so reserve two rows per committed/candidate version.
  const auto states = add(residents, multiply(candidate_boundaries, active_sequences));
  const auto hidden_slots = candidate_boundaries == 2 ? multiply(2, snapshots) : 0;
  const auto hidden_bytes = hidden_slots ? add(multiply(4096, hidden_slots), 1) : 0;
  const auto page_refs = multiply(refs, active_sequences);
  const auto query_rows = multiply(query_rows_per_request, active_sequences);
  const auto kv_rows = multiply(context, active_sequences);
  constexpr auto max_index = std::size_t((std::numeric_limits<std::int32_t>::max)());
  if (active_sequences >= max_index || pages > max_index || page_refs > max_index ||
      query_rows > max_index || kv_rows > max_index ||
      states > std::size_t((std::numeric_limits<std::uint32_t>::max)()) ||
      hidden_slots > std::size_t((std::numeric_limits<std::uint32_t>::max)()))
    throw std::runtime_error("cache batch/page/state capacity exceeds the supported index range");
  Plan result;
  result.resident_sessions = residents;
  result.active_sequences = active_sequences;
  result.recurrent_payload_bytes = recurrent_payload;
  result.max_requests = static_cast<std::uint32_t>(active_sequences);
  result.max_query_rows = static_cast<std::uint32_t>(query_rows);
  result.max_kv_rows = static_cast<std::uint32_t>(kv_rows);
  result.max_pages = static_cast<std::uint32_t>(pages);
  result.max_page_refs = static_cast<std::uint32_t>(page_refs);
  result.state_slots = static_cast<std::uint32_t>(states);
  result.snapshot_slots = static_cast<std::uint32_t>(snapshots);
  result.candidate_boundaries = static_cast<std::uint32_t>(candidate_boundaries);
  result.hidden_slots = static_cast<std::uint32_t>(hidden_slots);
  result.hidden_bytes = hidden_bytes;
  result.draft_cache = draft_cache;
  result.bank_bytes = add(multiply(multiply(pages, Plan::page_rows), 1024), 1);
  // Each real arena/pool includes its own final byte. There are 12 KV arena
  // backings and 36 independent conv/GDN pools, each charged exactly once.
  // The optional carry/prior hidden pool has its own final byte as well.
  // Draft K/V adds two independent same-geometry arena backings, not aliases.
  result.backing_bytes = add(add(multiply(result.bank_bytes, draft_cache ? 14 : 12),
      add(multiply(recurrent_payload, result.state_slots), 36)), hidden_bytes);
  return result;
}

struct PageInventory;
class PageRef final {
public:
  PageRef() noexcept = default;
  PageRef(const PageRef &) noexcept;
  PageRef &operator=(const PageRef &) noexcept;
  PageRef(PageRef &&other) noexcept
      : owner_(std::exchange(other.owner_, nullptr)), page_(other.page_) {}
  PageRef &operator=(PageRef &&other) noexcept {
    if (this != &other) { PageRef next(std::move(other)); swap(next); }
    return *this;
  }
  ~PageRef();
  std::uint32_t page() const noexcept { return page_; }
private:
  friend struct PageInventory;
  PageRef(PageInventory *owner, std::uint32_t page) noexcept : owner_(owner), page_(page) {}
  void swap(PageRef &other) noexcept { std::swap(owner_, other.owner_); std::swap(page_, other.page_); }
  PageInventory *owner_ = nullptr;
  std::uint32_t page_ = 0;
};
struct PageReservation final {
  // Writers die before the policy reference can make this ID eligible again.
  PageRef reference;
  std::array<c::PagedPageWriter, 6> keys, values;
};
struct PageInventory final {
  std::array<c::PagedArena, 6> keys, values;
  std::vector<std::uint32_t> owners, free_ids;
  std::size_t active_layers = 6;
  void prepare(std::uint32_t pages, std::size_t layers = 6) {
    if (layers != 1 && layers != 6)
      throw std::runtime_error("Qwen page inventory requires one draft or six target layers");
    owners.assign(pages, 0); free_ids.reserve(pages);
    for (auto page = pages; page != 0; --page) free_ids.push_back(page - 1);
    active_layers = layers;
  }
  hyle::Result<PageReservation> acquire() noexcept {
    if (active_layers != 1 && active_layers != 6)
      return std::unexpected(hyle::Status::InvalidArgument("Qwen page inventory layer count is invalid"));
    auto attempts = free_ids.size();
    while (attempts--) {
      const auto page = free_ids.back(); free_ids.pop_back();
      PageReservation result;
      hyle::Status status;
      for (std::size_t i = 0; i < active_layers && status.ok(); ++i) {
        auto key = keys[i].acquire(page);
        if (!key) { status = key.error(); break; }
        result.keys[i] = std::move(*key);
        auto value = values[i].acquire(page);
        if (!value) { status = value.error(); break; }
        result.values[i] = std::move(*value);
      }
      if (status.ok()) {
        owners[page] = 1; result.reference = PageRef(this, page); return result;
      }
      // Policy eligibility never proves native retirement. The real acquire
      // can defer an ID while old Graph readers still retain its blocks.
      if (status.IsUnavailable()) { free_ids.insert(free_ids.begin(), page); continue; }
      free_ids.push_back(page); return std::unexpected(std::move(status));
    }
    return std::unexpected(hyle::Status::ResourceExhausted("shared Qwen KV pages are retained; release a prefix/session or enlarge the cache"));
  }
  void retain(std::uint32_t page) noexcept { ++owners[page]; }
  void release(std::uint32_t page) noexcept {
    if (--owners[page] == 0) free_ids.push_back(page);
  }
};
inline PageRef::PageRef(const PageRef &other) noexcept : owner_(other.owner_), page_(other.page_) {
  if (owner_) owner_->retain(page_);
}
inline PageRef &PageRef::operator=(const PageRef &other) noexcept {
  if (this != &other) { PageRef next(other); swap(next); }
  return *this;
}
inline PageRef::~PageRef() { if (owner_) owner_->release(page_); }

struct Recurrent final { c::ReadLease history, recurrent; };
struct StatePools final { c::LeasePool history, recurrent; };
// Counts distinct immutable versions, not shared_ptr aliases or users. An
// explicitly retained prefix consumes the same finite capacity as an old head.
struct VersionCount final { std::uint32_t live = 0, capacity = 0; };
class VersionRef final {
public:
  VersionRef() noexcept = default;
  VersionRef(const VersionRef &) = delete;
  VersionRef &operator=(const VersionRef &) = delete;
  VersionRef(VersionRef &&other) noexcept : owner_(std::exchange(other.owner_, nullptr)) {}
  VersionRef &operator=(VersionRef &&other) noexcept {
    if (this != &other) { VersionRef next(std::move(other)); std::swap(owner_, next.owner_); }
    return *this;
  }
  ~VersionRef() { if (owner_) --owner_->live; }
  static hyle::Result<VersionRef> acquire(VersionCount &owner) noexcept {
    if (owner.live == owner.capacity)
      return std::unexpected(hyle::Status::ResourceExhausted("Qwen cache retains all snapshot versions; release a prefix/checkpoint or enlarge the cache"));
    ++owner.live; VersionRef result; result.owner_ = &owner; return result;
  }
private:
  VersionCount *owner_ = nullptr;
};
struct DraftState final {
  // Reverse destruction releases the actual ready frames before page policy.
  std::vector<PageRef> pages;
  c::PagedArenaRead key, value;
  std::size_t frontier = 0;
  // Committed MTP heads have frontier == target length - lag. Lag1 also
  // requires a valid catchup_token and Snapshot.prior_hidden; lag0 has no token.
  std::uint32_t lag = 0;
  std::int32_t catchup_token = -1;
};
// Snapshot is immutable once published. Tokens/positions and every layer's
// state advance by replacing one shared pointer. Model outlives all snapshots.
struct Snapshot final {
  // Reverse destruction releases actual frames before policy references.
  VersionRef version;
  std::vector<PageRef> pages;
  std::array<c::PagedArenaRead, 6> keys, values;
  std::array<Recurrent, 24> state;
  c::ReadLease carry_hidden, prior_hidden;
  std::optional<DraftState> draft;
  const void *model = nullptr;
  const void *math = nullptr;
  std::size_t length = 0;
  std::int32_t next_id = -1; // Valid only at a complete nonempty evaluated boundary.
};
using SnapshotRef = std::shared_ptr<const Snapshot>;
} // namespace qwen35::cache
