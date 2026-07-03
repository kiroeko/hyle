#pragma once

#include <cstdint>

#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_SCHED_API
#elif defined(HYLE_SCHED_BUILD)
#  define HYLE_SCHED_API HYLE_EXPORT
#else
#  define HYLE_SCHED_API HYLE_IMPORT
#endif

// Fiber sleep (sched-04b): a fiber suspends for a duration WITHOUT burning its
// worker. This is the first consumer of the time-02 timing wheel through the
// sched-04a Waker spine -- "timer expired" funnels into the same wake path as an
// io/gpu completion (sched-design §3). Each worker owns a TimerWheel; a sleeping
// fiber arms an entry on its worker's wheel and suspend()s. The worker, when it
// would otherwise park, advances its wheel and fires due timers -> Waker::wake()
// re-injects the fiber. Deadlines key on monotonic_ns (never the wall clock).
//
// COST MODEL (SOTA discipline): timers are touched ONLY on the park cold path.
// A worker with runnable fibers never reads the wheel or a clock -- a workload
// with no sleeping fibers pays nothing for this feature's existence. Arm is O(1),
// the pre-park advance is O(#due) + O(6 levels) for the next-deadline scan.
//
// NOT here: wait_for/wait_until (timed CV waits). Those add a timeout-vs-notify
// race whose loser must CANCEL the other across workers (the fiber may resume on
// a different worker than the one holding its timer) -- that needs the per-worker
// cross-core cancel command queue (the time-02 headline divergence point), split
// out as sched-04d. Plain sleep has a single waker (the timer), so no such race.

namespace hyle::sched {

// Suspend the current fiber for AT LEAST `ns` nanoseconds (rounded up to the
// wheel's 1ms tick), then resume. Must be called from inside a fiber running on
// a Scheduler worker (asserts otherwise). ns == 0 resumes on the next tick.
//
// A sleeping fiber is still OUTSTANDING work: Scheduler::drain()/shutdown() block
// until it wakes and finishes. A fiber sleeping N seconds therefore holds up
// teardown for N seconds -- cancel long sleeps before shutdown if that matters.
HYLE_SCHED_API void sleep_for_ns(std::uint64_t ns) noexcept;

// Absolute-deadline variant: suspend until monotonic_ns() reaches at least
// `deadline_ns`. A deadline already in the past resumes on the next tick.
HYLE_SCHED_API void sleep_until_ns(std::uint64_t deadline_ns) noexcept;

}  // namespace hyle::sched
