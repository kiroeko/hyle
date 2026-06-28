#pragma once

#include "hyle/macros.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_SCHED_API
#elif defined(HYLE_SCHED_BUILD)
#  define HYLE_SCHED_API HYLE_EXPORT
#else
#  define HYLE_SCHED_API HYLE_IMPORT
#endif

// The completion spine (sched-design §3): the single seam through which every
// asynchronous source -- io_uring/IOCP completions, GPU events, timers -- wakes a
// fiber. A fiber that must block on an external event captures a Waker (which
// names {this scheduler, this fiber}), hands it to the source, and suspends();
// when the event fires the source calls Waker::wake() from ANY thread, which
// re-injects the fiber into the run loop. This is what makes "GPU finished / NIC
// got a packet / timer expired" all funnel back to one scheduler -- HYLE's thesis.
//
// sched-04a delivers the spine itself (suspend + wake, race-free). Deadline/timer
// wiring (Waker fired by a timer wheel) is sched-04b, gated on time-02.

namespace hyle::sched {

class Fiber;
class Scheduler;

// A one-shot handle to wake a specific suspended fiber. Trivially copyable
// ({scheduler, fiber} pointers); cheap to pass to a backend. wake() is safe to
// call from any thread, exactly once per suspend() episode. Get a FRESH Waker
// (current_waker()) for each suspend -- a Waker is bound to one wait episode.
class Waker {
 public:
  Waker() noexcept = default;
  Waker(Scheduler* sched, Fiber* fiber) noexcept : sched_(sched), fiber_(fiber) {}

  bool valid() const noexcept { return sched_ != nullptr && fiber_ != nullptr; }

  // Mark the associated fiber runnable again and hand it back to the scheduler.
  // May run before the fiber has finished suspending (wake-before-suspend); the
  // rendezvous inside makes that safe -- the worker re-injects post-suspend in
  // that case, and wake() becomes a no-op re-injection. Call at most once.
  HYLE_SCHED_API void wake() const noexcept;

 private:
  Scheduler* sched_ = nullptr;
  Fiber* fiber_ = nullptr;
};

// Build a Waker naming the scheduler + fiber currently running on this worker.
// Precondition: called from inside a fiber that is running under a Scheduler
// (i.e. on a worker thread); returns an invalid Waker otherwise.
HYLE_SCHED_API Waker current_waker() noexcept;

// Suspend the current fiber until its Waker fires. Unlike Fiber::yield() (which
// reschedules immediately), this takes the fiber OUT of the run loop: it will not
// run again until a Waker::wake() re-injects it. Call current_waker() FIRST and
// arrange for something to wake it, or the fiber blocks forever.
HYLE_SCHED_API void suspend() noexcept;

}  // namespace hyle::sched
