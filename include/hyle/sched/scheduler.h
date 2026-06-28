#pragma once

#include <cstddef>

#include "hyle/macros.h"
#include "hyle/mem/allocator.h"
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
class Waker;  // sched-04: re-injects a suspended fiber (needs impl_ via friend)

// M:N work-stealing scheduler: N OS-thread workers, each with its own FIFO local
// run-queue (container WsFifoQueue), a shared global injection queue (Vyukov
// MPMC), and random-victim stealing. A fiber that yields goes back on a local
// queue; an idle worker steals ~half from a random victim. (sched-design §2.2.)
//
// sched-03 step 1: idle workers BUSY-SPIN (yield). step 2 replaces that with
// spin-then-park + unpark relay (§2.3). Completion/waker wiring is sched-04.
//
// Pimpl: the implementation holds std::vector / threads / queues; keeping them
// out of this header avoids leaking std types across the DLL seam (C4251) and
// keeps the public surface a clean handle.
class HYLE_SCHED_API Scheduler {
 public:
  // Starts `num_workers` worker threads immediately. `queue_capacity` sizes each
  // local queue and the global injection queue (power of two).
  Scheduler(unsigned num_workers, mem::Allocator alloc,
            std::size_t queue_capacity);
  ~Scheduler();
  HYLE_DISABLE_COPY_AND_MOVE(Scheduler);

  // Submit a runnable fiber. Lands on the caller's local queue if called from a
  // worker, else on the global queue. The scheduler does NOT own the Fiber; the
  // caller must keep it alive until it has finished running.
  void spawn(Fiber* fiber);

  // Block the calling thread until every spawned fiber has finished.
  void drain();

  // drain(), then stop the workers and join them. Idempotent-safe to call once.
  void shutdown();

 private:
  friend class Waker;  // Waker::wake() re-injects via impl_ (completion spine)
  struct Impl;
  Impl* impl_;
};

}  // namespace hyle::sched
