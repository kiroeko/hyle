#pragma once

#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "hyle/core/compiler.h"
#include "hyle/time/clock.h"
#include "hyle/time/timer_entry.h"

// Intrusive radix timer queue for the real server/runtime workload: many arms,
// most timers cancelled before delivery, and one single-thread-owned queue per
// event-loop worker. It is the algorithmic shape used by Seastar timer_set:
// sort work is deferred until expiry rather than paid on every insertion.
//
//   arm       O(1): one xor + leading-zero bucket selection
//   cancel    O(1): intrusive unlink from the recoverable radix bucket
//   advance   O(number of buckets touched + entries delivered/rebucketed)
//   next      O(1), conservative after cancellation of the earliest timer
//
// Compared with a hierarchical fixed-tick wheel, this representation is both
// smaller (~1 KiB of bucket heads rather than ~8 KiB) and exact to one
// nanosecond: there is no tick rounding. Compared with an indexed heap,
// cancellation does not sift O(log n). The queue allocates nothing; callers own
// stable-address TimerEntry nodes.
//
// This is a data structure, not an executor. advance() hands due nodes to a
// caller-owned delivery list and never invokes callbacks. Delivery order across
// different radix buckets is unspecified; callers needing a total deadline order
// can sort the returned batch. It is deliberately not thread safe: cross-thread
// arm/cancel belongs in the owning event loop's command queue, avoiding locks on
// the common local path.

namespace hyle::time {

class TimerQueue {
 public:
  using EntryList = TimerList;

  TimerQueue() noexcept : TimerQueue(monotonic_ns()) {}
  explicit TimerQueue(std::uint64_t now_ns) noexcept : now_ns_(now_ns) {}
  explicit TimerQueue(MonotonicTimePoint now) noexcept
      : TimerQueue(to_monotonic_ns(now)) {}
  template <MonotonicClockSource Source>
  explicit TimerQueue(const Source& source) noexcept
      : TimerQueue(source.now()) {}
  ~TimerQueue() noexcept { detach_remaining(); }
  TimerQueue(const TimerQueue&) = delete;
  TimerQueue& operator=(const TimerQueue&) = delete;

  // Arm at an absolute monotonic deadline. An already-armed entry on this queue
  // is rescheduled. A deadline at/in the past becomes due on the next advance;
  // schedule itself never runs user code.
  void schedule(TimerEntry* entry, std::uint64_t deadline_ns) noexcept {
    HYLE_FATAL_ASSERT(entry != nullptr);
    if (entry->scheduled()) {
      HYLE_FATAL_ASSERT(entry->owned_by(this));
      (void)cancel(entry);
    }
    HYLE_FATAL_ASSERT(!entry->hook_.linked());
    entry->deadline_ns_ = deadline_ns;
    entry->set_owner(this);
    add_entry(entry);
  }

  void schedule(TimerEntry* entry, MonotonicTimePoint deadline) noexcept {
    schedule(entry, to_monotonic_ns(deadline));
  }

  void schedule_after(
      TimerEntry* entry, std::uint64_t timeout_ns,
      std::uint64_t now_ns = monotonic_ns()) noexcept {
    schedule(entry, saturating_add_ns(now_ns, timeout_ns));
  }

  template <MonotonicClockSource Source>
  HYLE_FORCE_INLINE void schedule_after(
      TimerEntry* entry, std::uint64_t timeout_ns,
      const Source& source) noexcept {
    schedule(entry, deadline_after_ns(source, timeout_ns));
  }

  template <class Rep, class Period>
  HYLE_FORCE_INLINE void schedule_after(
      TimerEntry* entry, std::chrono::duration<Rep, Period> timeout,
      MonotonicTimePoint now = monotonic_now()) noexcept {
    schedule(entry, saturating_add_ns(to_monotonic_ns(now),
                                     duration_ns_saturated(timeout)));
  }

  template <class Rep, class Period, MonotonicClockSource Source>
  HYLE_FORCE_INLINE void schedule_after(
      TimerEntry* entry, std::chrono::duration<Rep, Period> timeout,
      const Source& source) noexcept {
    schedule(entry, deadline_after(source, timeout));
  }

  // Returns whether an armed entry was detached. Passing an idle entry is an
  // idempotent no-op; passing an entry owned by another queue is a contract bug.
  bool cancel(TimerEntry* entry) noexcept {
    HYLE_FATAL_ASSERT(entry != nullptr);
    if (!entry->scheduled()) {
      HYLE_FATAL_ASSERT(!entry->hook_.linked());
      return false;
    }
    HYLE_FATAL_ASSERT(entry->owned_by(this));
    const std::size_t bucket = bucket_for(entry->deadline_ns_);
    buckets_[bucket].remove(entry);
    clear_occupied_if_empty(bucket);
    entry->set_owner(nullptr);
    --count_;
    // Reset on the empty transition so the first subsequent arm publishes its
    // exact deadline instead of inheriting a stale early wake from the previous
    // generation. That branch saves a potentially expensive event-loop wake.
    if (count_ == 0) next_ns_ = UINT64_MAX;
    return true;
  }

  // Move every deadline <= now into `out`. Calls must use non-decreasing
  // monotonic timestamps; this is the same event-loop contract as Seastar's
  // timer_set and catches accidentally mixing realtime with monotonic time.
  std::size_t advance(std::uint64_t now_ns, EntryList& out) noexcept {
    HYLE_FATAL_ASSERT(now_ns >= now_ns_);
    const std::size_t inspect_bucket = bucket_for(now_ns);
    EntryList ready;

    if (inspect_bucket < kTimestampBits) {
      // Buckets with a larger leading-zero index are known to be no later than
      // `now` and can be spliced as complete lists.
      std::uint64_t known_due =
          inspect_bucket == kTimestampBits - 1
              ? 0
              : occupied_ & (~std::uint64_t{0} << (inspect_bucket + 1));
      while (known_due != 0) {
        const std::size_t bucket =
            static_cast<std::size_t>(std::countr_zero(known_due));
        ready.append(buckets_[bucket]);
        occupied_ &= ~(std::uint64_t{1} << bucket);
        known_due &= known_due - 1;
      }
      if (due_occupied_) {
        ready.append(buckets_[kDueBucket]);
        due_occupied_ = false;
      }
    }

    EntryList inspect;
    inspect.append(buckets_[inspect_bucket]);
    clear_occupied(inspect_bucket);
    now_ns_ = now_ns;
    next_ns_ = UINT64_MAX;

    // The boundary bucket can straddle `now`; deliver its due members and
    // reclassify the rest relative to the new radix origin.
    inspect.consume_all([&](TimerEntry* entry) noexcept {
      if (entry->deadline_ns_ <= now_ns) {
        ready.push_back(entry);
      } else {
        add_entry_no_count(entry);
      }
    });

    std::size_t fired = 0;
    ready.for_each([&](TimerEntry* entry) noexcept {
      entry->set_owner(nullptr);
      ++fired;
    });
    out.append(ready);
    HYLE_FATAL_ASSERT(fired <= count_);
    count_ -= fired;

    // Untouched buckets are farther away than any reinserted boundary entry.
    // Only when no reinsert supplied a minimum do we lazily scan the nearest
    // remaining bucket, matching timer_set's cancellation-friendly policy.
    if (next_ns_ == UINT64_MAX && count_ != 0) recompute_next();
    return fired;
  }

  std::size_t advance(MonotonicTimePoint now, EntryList& out) noexcept {
    return advance(to_monotonic_ns(now), out);
  }

  template <MonotonicClockSource Source>
  HYLE_FORCE_INLINE std::size_t advance(
      const Source& source, EntryList& out) noexcept {
    return advance(source.now(), out);
  }

  // Conservative absolute wakeup. Cancelling the current minimum may leave an
  // earlier stale value until the next advance; waking early is safe and avoids
  // an O(bucket-size) rescan on the cancellation hot path. Empty is UINT64_MAX.
  std::uint64_t next_deadline_ns() const noexcept {
    if (empty()) return UINT64_MAX;
    return next_ns_ < now_ns_ ? now_ns_ : next_ns_;
  }

  [[nodiscard]] std::optional<MonotonicTimePoint> next_deadline()
      const noexcept {
    if (empty()) return std::nullopt;
    return monotonic_time_point(next_deadline_ns());
  }

  // Cold shutdown path. Detaches every node without invoking callbacks; order
  // is unspecified because radix buckets are deliberately unsorted.
  std::size_t cancel_all(EntryList& out) noexcept {
    const std::size_t cancelled = count_;
    for (EntryList& bucket : buckets_) {
      bucket.for_each(
          [](TimerEntry* entry) noexcept { entry->set_owner(nullptr); });
      out.append(bucket);
    }
    occupied_ = 0;
    due_occupied_ = false;
    next_ns_ = UINT64_MAX;
    count_ = 0;
    return cancelled;
  }

  [[nodiscard]] bool empty() const noexcept { return count_ == 0; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }
  [[nodiscard]] std::uint64_t now_ns() const noexcept { return now_ns_; }

 private:
  static constexpr std::size_t kTimestampBits = 64;
  static constexpr std::size_t kDueBucket = kTimestampBits;
  static constexpr std::size_t kBucketCount = kTimestampBits + 1;

  std::size_t bucket_for(std::uint64_t deadline_ns) const noexcept {
    if (deadline_ns <= now_ns_) return kDueBucket;
    return static_cast<std::size_t>(
        std::countl_zero(deadline_ns ^ now_ns_));
  }

  void mark_occupied(std::size_t bucket) noexcept {
    if (bucket == kDueBucket) due_occupied_ = true;
    else occupied_ |= std::uint64_t{1} << bucket;
  }

  void clear_occupied(std::size_t bucket) noexcept {
    if (bucket == kDueBucket) due_occupied_ = false;
    else occupied_ &= ~(std::uint64_t{1} << bucket);
  }

  void clear_occupied_if_empty(std::size_t bucket) noexcept {
    if (buckets_[bucket].empty()) clear_occupied(bucket);
  }

  void add_entry(TimerEntry* entry) noexcept {
    add_entry_no_count(entry);
    ++count_;
  }

  void add_entry_no_count(TimerEntry* entry) noexcept {
    const std::size_t bucket = bucket_for(entry->deadline_ns_);
    // schedule() and the rebucket path both establish an unlinked node before
    // entering here; avoid repeating the linked-state branch on every arm.
    buckets_[bucket].push_back(entry);
    mark_occupied(bucket);
    if (entry->deadline_ns_ < next_ns_) next_ns_ = entry->deadline_ns_;
  }

  void recompute_next() noexcept {
    std::size_t nearest;
    if (due_occupied_) {
      nearest = kDueBucket;
    } else {
      HYLE_FATAL_ASSERT(occupied_ != 0);
      nearest = kTimestampBits - 1 -
                static_cast<std::size_t>(std::countl_zero(occupied_));
    }
    buckets_[nearest].for_each([&](const TimerEntry* entry) noexcept {
      if (entry->deadline_ns_ < next_ns_) next_ns_ = entry->deadline_ns_;
    });
  }

  // Normal owner-first teardown. Callers still own the entries, so there is no
  // delivery list to return and no callback to run; simply make every node idle
  // before the intrusive bucket members are destroyed. This is intentionally
  // O(n) cold-path work and adds nothing to arm/cancel/advance metadata.
  void detach_remaining() noexcept {
    std::size_t detached = 0;
    for (EntryList& bucket : buckets_) {
      while (TimerEntry* entry = bucket.pop_front()) {
        HYLE_FATAL_ASSERT(entry->owned_by(this));
        entry->set_owner(nullptr);
        ++detached;
      }
    }
    HYLE_FATAL_ASSERT(detached == count_);
    count_ = 0;
  }

  std::array<EntryList, kBucketCount> buckets_{};
  std::uint64_t occupied_ = 0;  // buckets [0, 63]
  std::uint64_t now_ns_ = 0;
  std::uint64_t next_ns_ = UINT64_MAX;
  std::size_t count_ = 0;
  bool due_occupied_ = false;   // bucket 64
};

}  // namespace hyle::time
