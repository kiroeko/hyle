#pragma once

#include <cstddef>

#include "hyle/mem/allocator.h"
#include "hyle/sched/run_queue.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_SCHED_API
#elif defined(HYLE_SCHED_BUILD)
#  define HYLE_SCHED_API HYLE_EXPORT
#else
#  define HYLE_SCHED_API HYLE_IMPORT
#endif

namespace hyle::sched {

class Fiber;

// Single-worker driver over one RunQueue. sched-02 runs the loop on the CALLING
// thread (no OS thread yet -- that, plus multiple workers + work-stealing, is
// sched-03). Proves the local scheduling structure in isolation.
//
// Scheduling policy: a fiber that yields (still !done) goes back to the FIFO
// tail (fair round-robin); a finished fiber is dropped (ownership stays with the
// caller). LIFO-slot scheduling is exposed for completion wakeups (sched-04) but
// the run loop itself only reschedules yields onto FIFO.
//
// API macro on methods, not the class: the RunQueue member is non-POD, so
// exporting the class would trip MSVC C4251 (same pattern as Fiber/Parker).
class Worker {
 public:
  Worker(mem::Allocator alloc, std::size_t queue_capacity)
      : rq_(alloc, queue_capacity) {}

  // Enqueue a runnable fiber (FIFO). Returns false if the local ring is full.
  bool schedule(Fiber* f) noexcept { return rq_.schedule_fifo(f); }

  // Enqueue a fiber to run next (LIFO slot). For completion wakeups.
  bool schedule_next(Fiber* f) noexcept { return rq_.schedule_lifo(f); }

  // Drive until the queue is drained: take a fiber, resume it, and if it yielded
  // (not done) put it back on the FIFO tail. Returns the number of resume()
  // calls made (useful for tests/diagnostics).
  HYLE_SCHED_API std::size_t run();

 private:
  RunQueue rq_;
};

}  // namespace hyle::sched
