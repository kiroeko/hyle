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
#include "cache_plan.h"
#include "../../inference/sampling_cursor.h"
#include "../../inference/reusable_output_storage.h"
#include "../../inference/paged_sequence.h"

namespace qwen35::cache {
namespace c = hyle::compute;

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
