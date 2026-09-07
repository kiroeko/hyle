#pragma once

#include <utility>

#include "hyle/core/compiler.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_SCHED_API
#elif defined(HYLE_SCHED_BUILD)
#  define HYLE_SCHED_API HYLE_EXPORT
#else
#  define HYLE_SCHED_API HYLE_IMPORT
#endif

// The completion spine: the single seam through which every
// asynchronous source -- io_uring/IOCP completions, GPU events, timers -- wakes a
// fiber. A fiber that must block on an external event captures a Waker (which
// names {this scheduler, this fiber}), hands it to the source, and suspends();
// when the event fires the source calls Waker::wake() from ANY thread, which
// re-injects the fiber into the run loop. This is what makes "GPU finished / NIC
// got a packet / timer expired" all funnel back to one scheduler -- HYLE's thesis.
//
// suspend + wake form the race-free spine. Deadline/timer wiring uses the same
// Waker through the Time queue.

namespace hyle::sched {

class Fiber;
class Scheduler;
class Waker;

HYLE_SCHED_API Waker current_waker() noexcept;

// A one-shot handle to wake a specific suspended fiber. It remains exactly two
// pointers and is move-only so one wait episode has one completion owner.
// wake() is safe to call from any thread and consumes the handle. Get a FRESH
// Waker (current_waker()) for every suspend episode. Competing completion
// sources must first arbitrate one winner, then let only that winner wake.
class Waker {
 public:
  Waker() noexcept = default;
  Waker(Waker&& other) noexcept
      : sched_(std::exchange(other.sched_, nullptr)),
        fiber_(std::exchange(other.fiber_, nullptr)) {}
  Waker& operator=(Waker&& other) noexcept {
    if (this != &other) {
      sched_ = std::exchange(other.sched_, nullptr);
      fiber_ = std::exchange(other.fiber_, nullptr);
    }
    return *this;
  }
  Waker(const Waker&) = delete;
  Waker& operator=(const Waker&) = delete;

  bool valid() const noexcept { return sched_ != nullptr && fiber_ != nullptr; }

  // Mark the associated fiber runnable again and hand it back to the scheduler.
  // May run before the fiber has finished suspending (wake-before-suspend); the
  // rendezvous inside makes that safe -- the worker re-injects post-suspend in
  // that case. Precondition: valid(); this object is invalid after the call.
  HYLE_SCHED_API void wake() noexcept;

 private:
  friend HYLE_SCHED_API Waker current_waker() noexcept;
  Waker(Scheduler* sched, Fiber* fiber) noexcept : sched_(sched), fiber_(fiber) {}

  Scheduler* sched_ = nullptr;
  Fiber* fiber_ = nullptr;
};

// Build a Waker naming the scheduler + fiber currently running on this worker.
// Precondition: called from inside a fiber that is running under a Scheduler
// (i.e. on a worker thread); returns an invalid Waker otherwise.
// The index of the worker thread currently running this fiber, or -1 if not on a
// worker. Lets an integration owner pick THIS worker's per-core completion source so a
// submit and the reap of its completion happen on the same thread -- the per-core
// discipline that makes multi-worker io safe without locking the ring.
HYLE_SCHED_API int current_worker_index() noexcept;

// Suspend the current fiber until its Waker fires. Unlike Fiber::yield() (which
// reschedules immediately), this takes the fiber OUT of the run loop: it will not
// run again until a Waker::wake() re-injects it. Call current_waker() FIRST and
// arrange for something to wake it, or the fiber blocks forever.
HYLE_SCHED_API void suspend() noexcept;

}  // namespace hyle::sched
