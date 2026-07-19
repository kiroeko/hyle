#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/macros.h"
#include "hyle/time/clock.h"
#include "hyle/time/timer_source.h"

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
// vs TimerWheel for THIS load: the wheel is O(1) too but pays an 8KB slot array +
// scattered writes + per-arm level/slot math; a single sorted list is one
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
// NOT thread-safe by itself (per-worker/per-reactor owned, same as TimerWheel).
// Hands fired entries back; never runs callbacks (completion-spine contract).

namespace hyle::time {

class CommonTimeoutQueue {
 public:
  using EntryList = TimerList;

  // All entries in this queue share `duration_ns`. `start_ns` is unused by the
  // structure (deadlines are absolute) but taken for signature symmetry with
  // TimerWheel so a consumer can construct either the same way. duration_ns is
  // documentation/validation only -- the queue keys purely on deadline_ns.
  explicit CommonTimeoutQueue(std::uint64_t /*start_ns*/ = 0,
                              std::uint64_t duration_ns = 0) noexcept
      : duration_ns_(duration_ns) {}
  ~CommonTimeoutQueue() { HYLE_FATAL_ASSERT(empty()); }
  CommonTimeoutQueue(const CommonTimeoutQueue&) = delete;
  CommonTimeoutQueue& operator=(const CommonTimeoutQueue&) = delete;

  // Arm `e` to fire at absolute monotonic `deadline_ns`. O(1). Precondition (the
  // whole point of this structure): deadlines are armed in non-decreasing order,
  // which holds automatically when every entry uses the same duration over a
  // monotonic clock (deadline = monotonic_now + duration). We assert it rather
  // than silently mis-order: a violation means this entry does NOT belong to a
  // common-timeout queue (use TimerWheel for mixed durations). Reschedule of an
  // already-armed entry is supported (cancel + re-append).
  void schedule(TimerEntry* e, std::uint64_t deadline_ns) noexcept {
    HYLE_FATAL_ASSERT(e != nullptr);
    if (e->scheduled()) {
      HYLE_FATAL_ASSERT(e->owned_by(this));
      cancel(e);
    }
    HYLE_FATAL_ASSERT(list_.empty() ||
                      list_.back()->deadline_ns <= deadline_ns);
    e->deadline_ns = deadline_ns;
    list_.push_back(e);
    e->set_owner(this);
    ++count_;
  }

  // Non-fatal form for FFI/system boundaries. On an ordering violation a new
  // entry stays unscheduled and a rescheduled entry preserves its prior arm.
  // Use TimerWheel for mixed/non-monotonic deadlines.
  bool try_schedule(TimerEntry* e, std::uint64_t deadline_ns) noexcept {
    HYLE_FATAL_ASSERT(e != nullptr);
    if (e->scheduled()) {
      HYLE_FATAL_ASSERT(e->owned_by(this));
      // Preserve the old arm on rejection. If e is not the tail, the current
      // tail remains the ordering bound. If e is the tail, temporarily unlink
      // it to reveal the preceding bound and restore it if the new deadline
      // would be invalid. This makes try_schedule genuinely transactional.
      if (list_.back() != e) {
        if (list_.back()->deadline_ns > deadline_ns) return false;
        cancel(e);
      } else {
        const std::uint64_t old_deadline = e->deadline_ns;
        cancel(e);
        if (!list_.empty() && list_.back()->deadline_ns > deadline_ns) {
          e->deadline_ns = old_deadline;
          list_.push_back(e);
          e->set_owner(this);
          ++count_;
          return false;
        }
      }
    } else {
      HYLE_FATAL_ASSERT(!e->hook.linked());
      // Monotonic-arrival invariant: the current tail (latest armed) must not
      // be later than this new deadline. Equal is fine and remains FIFO.
      if (!list_.empty() && list_.back()->deadline_ns > deadline_ns) {
        return false;
      }
    }
    e->deadline_ns = deadline_ns;
    list_.push_back(e);  // ordered => append keeps the list sorted, O(1)
    e->set_owner(this);
    ++count_;
    return true;
  }

  // Canonical fixed-duration entry point: derive the absolute deadline from the
  // queue's declared duration with saturating arithmetic.
  bool try_schedule_after(TimerEntry* e, std::uint64_t now_ns) noexcept {
    return try_schedule(e, saturating_add_ns(now_ns, duration_ns_));
  }

  // Normal production entry point when the caller does not already have a
  // monotonic timestamp.
  bool try_schedule_now(TimerEntry* e) noexcept {
    return try_schedule_after(e, monotonic_ns());
  }

  // O(1) unlink. No tombstone. No-op if not scheduled.
  void cancel(TimerEntry* e) noexcept {
    HYLE_FATAL_ASSERT(e != nullptr);
    if (!e->scheduled()) {
      HYLE_FATAL_ASSERT(!e->hook.linked());
      return;
    }
    HYLE_FATAL_ASSERT(e->owned_by(this));
    list_.remove(e);
    e->set_owner(nullptr);
    --count_;
  }

  // Move every entry with deadline_ns <= now_ns onto `out` (FIFO / deadline
  // order, since the list is sorted). Stops at the first not-yet-due entry --
  // O(#fired), never scans the tail.
  void advance(std::uint64_t now_ns, EntryList& out) noexcept {
    const std::size_t moved = list_.move_prefix_to(
        out,
        [now_ns](const TimerEntry* entry) noexcept {
          return entry->deadline_ns <= now_ns;
        },
        [](TimerEntry* entry) noexcept { entry->set_owner(nullptr); });
    HYLE_FATAL_ASSERT(moved <= count_);
    count_ -= moved;
  }

  // Earliest deadline (front), or UINT64_MAX if empty -- for park-until-next.
  std::uint64_t next_deadline_ns() const noexcept {
    if (list_.empty()) return UINT64_MAX;
    return list_.front()->deadline_ns;
  }

  bool empty() const noexcept { return count_ == 0; }
  std::size_t size() const noexcept { return count_; }
  std::uint64_t duration_ns() const noexcept { return duration_ns_; }

 private:
  EntryList list_;               // sorted by deadline (append-ordered)
  std::size_t count_ = 0;
  std::uint64_t duration_ns_;    // the single duration this queue serves
};

static_assert(TimerSource<CommonTimeoutQueue>,
              "CommonTimeoutQueue must model TimerSource");

}  // namespace hyle::time
