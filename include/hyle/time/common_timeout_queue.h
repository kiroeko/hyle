#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "hyle/macros.h"
#include "hyle/time/clock.h"
#include "hyle/time/timer_entry.h"

// Common-timeout queue: the timer structure for "millions of timers that all
// share ONE duration" -- e.g. every connection in a server armed with the same
// 30s idle timeout. This is libevent's `common_timeout` insight (event.c): a
// min-heap is the wrong tool here, because same-duration timeouts armed over a
// monotonic clock ALREADY arrive in deadline order, so there is nothing to sort.
//
//   arm    = O(1) tail-append   (deadline = now + fixed_duration is monotonic)
//   expire = O(1) head-pop       (earliest is always at the front)
//   cancel = O(1) intrusive unlink (no tombstone, no scan)
//   next   = O(1) read the front's deadline
//
// vs TimerQueue for THIS load: the radix queue is O(1) too but pays bucket
// selection/rebucketing; a single sorted list is one
// push_back and touches only two cache lines (tail + node). vs a min-heap: the
// heap needlessly does O(log n) sift on already-ordered inserts. So for the
// fixed-duration serving load this is the tighter algorithmic fit.
//
// SCOPE: ONE queue holds ONE duration. Multiple durations = multiple queues +
// a duration->queue router; that router is the CONSUMER's job (an io reactor,
// like libevent's common_timeout_queues[] array) and is deliberately NOT built
// in here (its shape -- how many durations, how to route -- must be driven by
// the real io use, not guessed). This is the reusable data-structure layer.
//
// NOT thread-safe by itself (per-worker/per-reactor owned, same as TimerQueue).
// Hands fired entries back; never runs callbacks (completion-spine contract).

namespace hyle::time {

class CommonTimeoutQueue {
 public:
  using EntryList = TimerList;

  // All entries in this queue share `duration_ns`. Unlike TimerQueue it needs
  // no origin: deadlines are absolute, so accepting an unused start timestamp
  // would be a false abstraction rather than useful polymorphism.
  explicit CommonTimeoutQueue(std::uint64_t duration_ns = 0) noexcept
      : duration_ns_(duration_ns) {}
  template <class Rep, class Period>
  explicit CommonTimeoutQueue(
      std::chrono::duration<Rep, Period> duration) noexcept
      : duration_ns_(duration_ns_saturated(duration)) {}
  ~CommonTimeoutQueue() { HYLE_FATAL_ASSERT(empty()); }
  CommonTimeoutQueue(const CommonTimeoutQueue&) = delete;
  CommonTimeoutQueue& operator=(const CommonTimeoutQueue&) = delete;

  // Normal fixed-duration arm. Since steady time never moves backwards, this
  // preserves the queue's ordering invariant by construction.
  void schedule(TimerEntry* e) noexcept {
    const bool scheduled = try_schedule_now(e);
    HYLE_FATAL_ASSERT(scheduled);
  }

  // Arm `e` to fire at absolute monotonic `deadline_ns`. O(1). Precondition (the
  // whole point of this structure): deadlines are armed in non-decreasing order,
  // which holds automatically when every entry uses the same duration over a
  // monotonic clock (deadline = monotonic_now + duration). We assert it rather
  // than silently mis-order: a violation means this entry does NOT belong to a
  // common-timeout queue (use TimerQueue for mixed durations). Reschedule of an
  // already-armed entry is supported (cancel + re-append).
  void schedule(TimerEntry* e, std::uint64_t deadline_ns) noexcept {
    HYLE_FATAL_ASSERT(e != nullptr);
    // hook is the cheapest complete state discriminator: every source-owned
    // entry is linked, and a fired-list entry is linked with owner == nullptr.
    // Only the uncommon linked case needs the exact-owner load.
    if (e->hook_.linked()) {
      HYLE_FATAL_ASSERT(e->owned_by(this));
      (void)cancel(e);
    }
    HYLE_FATAL_ASSERT(list_.empty() ||
                      list_.back()->deadline_ns_ <= deadline_ns);
    e->deadline_ns_ = deadline_ns;
    list_.push_back(e);
    e->set_owner(this);
    ++count_;
  }

  // Non-fatal form for FFI/system boundaries. On an ordering violation a new
  // entry stays unscheduled and a rescheduled entry preserves its prior arm.
  // Use TimerQueue for mixed/non-monotonic deadlines.
  bool try_schedule(TimerEntry* e, std::uint64_t deadline_ns) noexcept {
    HYLE_FATAL_ASSERT(e != nullptr);
    if (e->hook_.linked()) {
      HYLE_FATAL_ASSERT(e->owned_by(this));
      // Preserve the old arm on rejection. If e is not the tail, the current
      // tail remains the ordering bound. If e is the tail, temporarily unlink
      // it to reveal the preceding bound and restore it if the new deadline
      // would be invalid. This makes try_schedule genuinely transactional.
      if (list_.back() != e) {
        if (list_.back()->deadline_ns_ > deadline_ns) return false;
        (void)cancel(e);
      } else {
        const std::uint64_t old_deadline = e->deadline_ns_;
        (void)cancel(e);
        if (!list_.empty() && list_.back()->deadline_ns_ > deadline_ns) {
          e->deadline_ns_ = old_deadline;
          list_.push_back(e);
          e->set_owner(this);
          ++count_;
          return false;
        }
      }
    } else {
      // Monotonic-arrival invariant: the current tail (latest armed) must not
      // be later than this new deadline. Equal is fine and remains FIFO.
      if (!list_.empty() && list_.back()->deadline_ns_ > deadline_ns) {
        return false;
      }
    }
    e->deadline_ns_ = deadline_ns;
    list_.push_back(e);  // ordered => append stays sorted, O(1)
    e->set_owner(this);
    ++count_;
    return true;
  }

  // Canonical fixed-duration entry point: derive the absolute deadline from the
  // queue's declared duration with saturating arithmetic.
  bool try_schedule_after(TimerEntry* e, std::uint64_t now_ns) noexcept {
    return try_schedule(e, saturating_add_ns(now_ns, duration_ns_));
  }

  bool try_schedule_after(TimerEntry* e, MonotonicTimePoint now) noexcept {
    return try_schedule_after(e, to_monotonic_ns(now));
  }

  // Normal production entry point when the caller does not already have a
  // monotonic timestamp.
  bool try_schedule_now(TimerEntry* e) noexcept {
    return try_schedule_after(e, monotonic_ns());
  }

  // O(1) unlink. No tombstone. No-op if not scheduled.
  bool cancel(TimerEntry* e) noexcept {
    HYLE_FATAL_ASSERT(e != nullptr);
    if (!e->hook_.linked()) return false;
    HYLE_FATAL_ASSERT(e->owned_by(this));
    list_.remove(e);
    e->set_owner(nullptr);
    --count_;
    return true;
  }

  // Move every entry with deadline_ns <= now_ns onto `out` (FIFO / deadline
  // order, since the list is sorted). Stops at the first not-yet-due entry --
  // O(#fired), never scans the tail.
  std::size_t advance(std::uint64_t now_ns, EntryList& out) noexcept {
    const std::size_t moved = list_.move_prefix_to(
        out,
        [now_ns](const TimerEntry* entry) noexcept {
          return entry->deadline_ns_ <= now_ns;
        },
        [](TimerEntry* entry) noexcept { entry->set_owner(nullptr); });
    HYLE_FATAL_ASSERT(moved <= count_);
    count_ -= moved;
    return moved;
  }

  std::size_t advance(MonotonicTimePoint now, EntryList& out) noexcept {
    return advance(to_monotonic_ns(now), out);
  }

  // Earliest deadline (front), or UINT64_MAX if empty -- for park-until-next.
  std::uint64_t next_deadline_ns() const noexcept {
    if (list_.empty()) return UINT64_MAX;
    return list_.front()->deadline_ns_;
  }

  [[nodiscard]] std::optional<MonotonicTimePoint> next_deadline()
      const noexcept {
    if (empty()) return std::nullopt;
    return monotonic_time_point(next_deadline_ns());
  }

  bool empty() const noexcept { return count_ == 0; }
  std::size_t size() const noexcept { return count_; }
  std::uint64_t duration_ns() const noexcept { return duration_ns_; }
  Duration duration() const noexcept {
    constexpr auto max = static_cast<std::uint64_t>(Duration::max().count());
    return Duration(duration_ns_ > max
                        ? Duration::max().count()
                        : static_cast<Duration::rep>(duration_ns_));
  }

  // Explicit cold-path shutdown: detach every entry without firing it and
  // append in deadline/arm order to `out`.
  std::size_t cancel_all(EntryList& out) noexcept {
    const std::size_t cancelled = count_;
    while (TimerEntry* entry = list_.pop_front()) {
      entry->set_owner(nullptr);
      out.push_back(entry);
    }
    count_ = 0;
    return cancelled;
  }

 private:
  EntryList list_;               // sorted by deadline (append-ordered)
  std::size_t count_ = 0;
  std::uint64_t duration_ns_;    // the single duration this queue serves
};

}  // namespace hyle::time
