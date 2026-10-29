#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "hyle/compute/lease_slots.h"
#include "hyle/compute/paged_frame.h"
#include "hyle/core/assert.h"
#include "hyle/container/small_vector.h"
#include "model_geometry.h"
#include "../../inference/sampling_cursor.h"
#include "../../inference/reusable_output_storage.h"
#include "../../inference/paged_sequence.h"

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

// State pools use this alignment explicitly at allocation sites. Capacity
// includes the gaps between simultaneously live rows, not only their payloads.
inline constexpr std::size_t state_pool_alignment = 256;
inline std::size_t state_pool_padding(std::size_t payload) {
  return (state_pool_alignment - payload % state_pool_alignment) % state_pool_alignment;
}
inline std::size_t state_pool_backing_bytes(std::size_t payload, std::size_t slots) {
  if (!payload || !slots)
    throw std::runtime_error("state pool requires positive payload and slot counts");
  return add(add(multiply(payload, slots), multiply(state_pool_padding(payload), slots - 1)), 1);
}

struct KvRowBytes final {
  std::size_t key = 0, value = 0;
  friend constexpr bool operator==(const KvRowBytes &, const KvRowBytes &) noexcept = default;
};

// Original physical source facts, including an available but unused draft.
// One bank's carrier never implies another bank's dtype, scale or row extent.
struct CacheRowBytes final {
  KvRowBytes target, draft;
  friend constexpr bool operator==(const CacheRowBytes &, const CacheRowBytes &) noexcept = default;
};

// Target and draft inventories use the same page geometry but independent
// typed arena blocks and byte extents; this policy is not a Compute owner.
struct Plan final {
  static constexpr std::uint32_t page_rows = 16;
  CacheRowBytes row_bytes;
  std::size_t resident_sessions = 0, active_sequences = 1, recurrent_payload_bytes = 0;
  std::uint32_t max_requests = 0, max_query_rows = 0, max_kv_rows = 0;
  std::uint32_t max_pages = 0, max_page_refs = 0, state_slots = 0;
  std::uint32_t snapshot_slots = 0, candidate_boundaries = 1, hidden_slots = 0;
  std::uint32_t in_flight_batches = 1;
  std::size_t key_bank_bytes = 0, value_bank_bytes = 0; // Per target layer.
  std::size_t draft_key_bank_bytes = 0, draft_value_bank_bytes = 0; // Zero without a draft plan.
  std::size_t hidden_row_bytes = 0, hidden_bytes = 0, backing_bytes = 0;
  std::size_t attention_layers = 0, recurrent_layers = 0;
  bool draft_cache = false;
};
inline Plan plan(const ModelGeometry &geometry, CacheRowBytes row_bytes,
                 std::size_t context, std::size_t residents,
                 std::size_t query_rows_per_request, std::size_t recurrent_payload,
                  std::size_t active_sequences = 1,
                   std::size_t candidate_boundaries = 1,
                   bool draft_cache = false, bool frontier_hidden = false,
                   std::size_t in_flight_batches = 1) {
  geometry.validate();
  if (!row_bytes.target.key || !row_bytes.target.value ||
      ((!row_bytes.draft.key) != (!row_bytes.draft.value)) ||
      (draft_cache && (!row_bytes.draft.key || !row_bytes.draft.value)))
    throw std::runtime_error("cache plan requires actual target K/V row extents and an independent complete draft pair when used");
  if (context > static_cast<std::size_t>(geometry.context_limit) || (draft_cache && geometry.mtp_layers != 1))
    throw std::runtime_error("cache context or NextN selection exceeds source geometry");
  if (!context || !residents || !query_rows_per_request || query_rows_per_request > context ||
      !active_sequences || active_sequences > residents || !in_flight_batches ||
      !candidate_boundaries || candidate_boundaries > context)
    throw std::runtime_error("cache plan requires valid context/query rows, one to N active sequences, positive in-flight batch depth and candidate boundaries within context");
  const auto refs = add(context, Plan::page_rows - 1) / Plan::page_rows;
  // Depth counts simultaneously retained batches, independently of each
  // batch's R sequences and K verifier boundaries. Every in-flight sequence
  // retains one candidate version and can require one extra partial COW tail
  // beside the N committed complete-context capacities. This is a capacity
  // bound, not permission to reuse an in-flight page or publish a candidate.
  const auto pending_sequences = multiply(in_flight_batches, active_sequences);
  const auto pages = add(multiply(refs, residents), pending_sequences);
  const auto snapshots = add(residents, pending_sequences);
  // Each verifier prefix has independent material state, while its episode
  // still creates one unpublished version. Committed heads may retain carry
  // and prior together; each candidate retains one target row per prefix.
  const auto candidate_states = multiply(candidate_boundaries, pending_sequences);
  const auto states = add(residents, candidate_states);
  const auto hidden_slots = draft_cache || candidate_boundaries > 1
      ? add(multiply(2, residents), candidate_states)
      : frontier_hidden ? snapshots : 0;
  const auto hidden_bytes = hidden_slots ? state_pool_backing_bytes(geometry.hidden_bytes(), hidden_slots) : 0;
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
  result.row_bytes = row_bytes;
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
  result.in_flight_batches = static_cast<std::uint32_t>(in_flight_batches);
  result.hidden_slots = static_cast<std::uint32_t>(hidden_slots);
  result.hidden_bytes = hidden_bytes;
  result.draft_cache = draft_cache;
  result.hidden_row_bytes = geometry.hidden_bytes();
  result.attention_layers = geometry.attention_layers(); result.recurrent_layers = geometry.recurrent_layers();
  const auto bank_rows = multiply(pages, Plan::page_rows);
  result.key_bank_bytes = add(multiply(bank_rows, row_bytes.target.key), 1);
  result.value_bank_bytes = add(multiply(bank_rows, row_bytes.target.value), 1);
  if (draft_cache) {
    result.draft_key_bank_bytes = add(multiply(bank_rows, row_bytes.draft.key), 1);
    result.draft_value_bank_bytes = add(multiply(bank_rows, row_bytes.draft.value), 1);
  }
  // Each real arena contributes its own final byte. An available unused draft
  // remains in row_bytes but has no arena or backing charge in this plan.
  const auto arena_bytes = add(multiply(add(result.key_bank_bytes, result.value_bank_bytes),
      result.attention_layers), add(result.draft_key_bank_bytes, result.draft_value_bank_bytes));
  std::size_t recurrent_padding = 0;
  if (result.recurrent_layers) {
    const auto history_bytes = multiply(multiply(static_cast<std::size_t>(geometry.conv_kernel - 1),
        static_cast<std::size_t>(geometry.gdn_qkv_width())), 2);
    const auto recurrent_bytes = multiply(multiply(static_cast<std::size_t>(geometry.gdn_value_heads),
        static_cast<std::size_t>(geometry.gdn_key_depth)),
        multiply(static_cast<std::size_t>(geometry.gdn_value_depth), 4));
    recurrent_padding = multiply(multiply(add(state_pool_padding(history_bytes),
        state_pool_padding(recurrent_bytes)), states - 1), result.recurrent_layers);
  }
  result.backing_bytes = add(add(arena_bytes,
      add(add(multiply(recurrent_payload, result.state_slots), multiply(2, result.recurrent_layers)),
          recurrent_padding)), hidden_bytes);
  return result;
}

struct Recurrent final { c::ReadLease history, recurrent; };
struct StatePools final {
  c::LeasePool history, recurrent;
  // Coordinate layouts retain their exact original pool domains. Model setup
  // assigns them only after both pools exist; default layouts are invalid.
  c::LeaseSlotLayout history_layout, recurrent_layout;
  // Shared by the Model's R active members, never a second state per each of
  // N resident sessions. Destroy cached writers before their original pools.
  hyle::inference::ReusableOutputStorage history_reuse, recurrent_reuse;

  hyle::Result<c::WriteLease> acquire_history(std::size_t bytes) noexcept {
    return acquire(history, history_reuse, bytes);
  }
  hyle::Result<c::WriteLease> acquire_recurrent(std::size_t bytes) noexcept {
    return acquire(recurrent, recurrent_reuse, bytes);
  }
private:
  static hyle::Result<c::WriteLease> acquire(c::LeasePool &pool,
      hyle::inference::ReusableOutputStorage &reuse, std::size_t bytes) noexcept {
    if (bytes == reuse.bytes())
      if (auto writer = reuse.take()) return std::move(*writer);
    return pool.allocate(bytes, state_pool_alignment);
  }
};
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
  hyle::inference::PagedSequenceState paged;
  std::size_t frontier = 0;
  // Committed MTP heads have frontier == target length - lag. Lag1 also
  // requires a valid catchup_token and State.prior_hidden; lag0 has no token.
  std::uint32_t lag = 0;
  std::int32_t catchup_token = -1;
};
using hyle::inference::SamplingCursor;
struct State;
using SnapshotRef = std::shared_ptr<const State>;
// Complete model boundary. A WorkingState owns mutable metadata exclusively;
// an explicit checkpoint transfers that object to an immutable SnapshotRef.
// Native payloads keep their original ready leases and retirement authority.
struct State final {
  // The complete cold inventory/control addresses outlive escaped prefixes
  // and every version/page reference below. Never retain the application Model.
  std::shared_ptr<const void> resource_owner;
  // Reverse destruction releases actual frames before policy references.
  VersionRef version;
  // Sampling-only wrappers share one evaluated model-state version. The
  // origin is always canonical (its own state_origin is empty), never a chain.
  SnapshotRef state_origin;
  hyle::inference::PagedSequenceState paged;
  hyle::container::SmallVector<Recurrent, 64> state;
  c::ReadLease carry_hidden, prior_hidden;
  std::optional<DraftState> draft;
  const void *model = nullptr;
  const void *math = nullptr;
  std::size_t length = 0;
  std::int32_t next_id = -1; // Valid only at a complete nonempty evaluated boundary.
  SamplingCursor sampling;
  const State *state_identity() const noexcept { return state_origin ? state_origin.get() : this; }

  hyle::Status can_seal() const noexcept {
    HYLE_RETURN_IF_ERROR(paged.can_seal());
    if (draft && draft->paged.frames.size() != 2)
      return hyle::Status::FailedPrecondition("Qwen MTP snapshot requires its two original draft banks");
    if (draft) HYLE_RETURN_IF_ERROR(draft->paged.can_seal());
    return hyle::Status::Ok();
  }
  void seal() noexcept {
    // The owner preflights the whole collection before publishing a Snapshot.
    paged.seal();
    if (draft) draft->paged.seal();
  }

  // Called only after the original Graph/native readers have withdrawn. Empty
  // metadata may be reused; no old payload, page policy or version stays pinned.
  // Release the resource owner last, just as in normal reverse destruction.
  void clear() noexcept {
    paged.clear(); state.clear();
    carry_hidden = {}; prior_hidden = {}; draft.reset();
    state_origin.reset(); version = {}; resource_owner.reset();
    model = nullptr; math = nullptr; length = 0; next_id = -1; sampling = {};
  }
};

// Serialized model-state owner, not another executor or native state pool.
// Borrowed views last only until commit/adopt/reset. Only checkpoint() can
// escape a retained state; a frozen object is never mutated or recycled.
class WorkingState final {
public:
  WorkingState() = default;
  WorkingState(const WorkingState &) = delete;
  WorkingState &operator=(const WorkingState &) = delete;
  const State *get() const noexcept { return current_ ? current_.get() : frozen_.get(); }
  const State *operator->() const noexcept { return get(); }
  const State &operator*() const noexcept { return *get(); }
  explicit operator bool() const noexcept { return get() != nullptr; }
  std::uint64_t epoch() const noexcept { return epoch_; }

  void preflight_commit() const {
    if (epoch_ == (std::numeric_limits<std::uint64_t>::max)())
      throw std::overflow_error("committed model-state epoch exhausted");
  }
  std::unique_ptr<State> acquire_candidate() {
    preflight_commit();
    return spare_ ? std::exchange(spare_, {}) : std::make_unique<State>();
  }
  void recycle_candidate(std::unique_ptr<State> candidate) noexcept {
    if (!candidate) return;
    candidate->clear(); spare_ = std::move(candidate);
  }
  // All fallible metadata/permission work and epoch preflight precede commit.
  // The old committed boundary survives until this whole-success point.
  void commit(std::unique_ptr<State> candidate) noexcept {
    commit(std::move(candidate), [](State &) noexcept {});
  }
  // Only the exclusively owned previous successful head may yield retired
  // payloads. Frozen checkpoints and failed candidates never enter this hook;
  // the caller has already withdrawn the original episode's native readers.
  template<class Recycle>
  void commit(std::unique_ptr<State> candidate, Recycle &&recycle) noexcept {
    commit(std::move(candidate), [](State *, State &) noexcept {}, std::forward<Recycle>(recycle));
  }
  // A prepared collection may transfer the exclusively owned working frames
  // from its old head and commit their checked deltas before the head swap.
  // Frozen heads have no mutable previous object and must be forked earlier.
  template<class Publish, class Recycle>
  void commit(std::unique_ptr<State> candidate, Publish &&publish, Recycle &&recycle) noexcept {
    static_assert(std::is_nothrow_invocable_v<Publish &, State *, State &>);
    static_assert(std::is_nothrow_invocable_v<Recycle &, State &>);
    assert(candidate && epoch_ != (std::numeric_limits<std::uint64_t>::max)());
    publish(current_.get(), *candidate);
    auto previous = std::move(current_);
    current_ = std::move(candidate); frozen_.reset(); ++epoch_;
    if (previous) recycle(*previous);
    recycle_candidate(std::move(previous));
  }
  void adopt(SnapshotRef checkpoint) {
    if (!checkpoint) throw std::invalid_argument("cannot adopt an empty model-state checkpoint");
    preflight_commit();
    auto previous = std::move(current_);
    frozen_ = std::move(checkpoint); ++epoch_;
    recycle_candidate(std::move(previous));
  }
  SnapshotRef checkpoint() {
    if (current_) {
      const auto ready = current_->can_seal();
      if (!ready.ok()) throw std::runtime_error("model-state frame collection cannot be frozen");
      // unique_ptr -> shared_ptr has no effect on allocation failure. The same
      // State and VersionRef survive; no tensor/frame payload is copied.
      std::shared_ptr<State> frozen(std::move(current_));
      // No allocation or failure after the shared control block exists.
      frozen->seal();
      frozen_ = std::move(frozen);
    }
    return frozen_;
  }
  void reset() noexcept { current_.reset(); frozen_.reset(); spare_.reset(); }
private:
  std::unique_ptr<State> current_, spare_;
  SnapshotRef frozen_;
  std::uint64_t epoch_ = 0;
};
} // namespace qwen35::cache
