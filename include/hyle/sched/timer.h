#pragma once

#include "hyle/time/clock.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_SCHED_API
#elif defined(HYLE_SCHED_BUILD)
#  define HYLE_SCHED_API HYLE_EXPORT
#else
#  define HYLE_SCHED_API HYLE_IMPORT
#endif

namespace hyle::time {
class TimerEntry;  // fwd: defined in time/timer_entry.h (opaque here)
}

// Fiber sleep (sched-04b): a fiber suspends for a duration WITHOUT burning its
// worker. This is the first consumer of time::TimerQueue through the
// sched-04a Waker spine -- "timer expired" funnels into the same wake path as an
// io/gpu completion (sched-design §3). Each worker owns a TimerQueue; a sleeping
// fiber arms an entry on its worker's queue and suspend()s. The worker, when it
// would otherwise park, advances its queue and fires due timers -> Waker::wake()
// re-injects the fiber. Deadlines key on monotonic_ns (never the wall clock).
//
// COST MODEL (SOTA discipline): timers are touched ONLY on the park cold path.
// A worker with runnable fibers never reads the queue or a clock -- a workload
// with no sleeping fibers pays nothing for this feature's existence. Arm is O(1),
// the pre-park advance touches only due/rebucketed entries and occupied buckets.
//
// ConditionVariable::wait_for/wait_until also use this queue. Their notify-vs-
// timer race is atomic-arbitrated and the waiting fiber is temporarily pinned to
// the queue owner for that episode, so a notify winner cancels the loser timer on
// the correct worker without cross-thread queue mutation. Plain sleep is simpler:
// the timer is its sole waker.

namespace hyle::sched {

// Suspend the current fiber until the exact monotonic deadline is due, then
// resume. Must be called from inside a fiber running on a Scheduler worker
// (asserts otherwise). A non-positive duration becomes due on the next
// scheduler advance.
//
// A sleeping fiber is still OUTSTANDING work: Scheduler::drain()/shutdown() block
// until it wakes and finishes. A fiber sleeping N seconds therefore holds up
// teardown for N seconds -- cancel long sleeps before shutdown if that matters.
HYLE_SCHED_API void sleep_for(time::Duration duration) noexcept;

// Absolute-deadline variant. A past deadline becomes due on the next scheduler
// advance. The monotonic type cannot be confused with a wall-clock timestamp.
HYLE_SCHED_API void sleep_until(time::MonotonicTimePoint deadline) noexcept;

// ---- timed-io building blocks (io-06 / C2) ----------------------------------
// Lower-level timer arm/cancel that do NOT suspend -- for composing a timeout with
// a concurrent io op (recv/accept/connect + deadline, first-to-fire wins). Unlike
// sleep_*, the caller arms a timer AND submits an io op, then suspends ONCE; on
// resume it arbitrates which fired. Both operate on the CURRENT worker's queue and
// MUST be called from a fiber on that worker (shard-per-core: the timed io fiber is
// pinned, so arm/fire/cancel all happen single-threaded on one worker -- no
// cross-core cancel queue needed, which is why this can ship without sched-04d).
//
// arm_timeout: schedule `entry` to fire at `deadline`. With no fire callback
// callback it wakes THIS fiber (entry payload is filled automatically, same as
// sleep). With a fire callback set, the caller initializes the payload BEFORE this
// call; the callback owns wake arbitration. The entry MUST outlive the wait
// (put it on the coroutine/awaitable frame). Does NOT suspend.
HYLE_SCHED_API void arm_timeout(time::TimerEntry* entry,
                                time::MonotonicTimePoint deadline) noexcept;
// cancel_timeout: remove `entry` from this worker's queue if still armed (O(1),
// no-op if it already fired). Call when the io op won the race, so a stale timer
// doesn't fire into a reused fiber. Same-worker only.
HYLE_SCHED_API void cancel_timeout(time::TimerEntry* entry) noexcept;

}  // namespace hyle::sched
