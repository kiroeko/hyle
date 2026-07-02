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

  // Submit a fiber PINNED to worker `worker_idx`: it runs only on that worker and
  // is never stolen, on spawn and on every wake (shard-per-core, io-04). Use this
  // to keep a connection's fiber on the core that owns its io ring so submit +
  // reap + resume stay single-threaded. `worker_idx` must be < num_workers().
  void spawn_on(unsigned worker_idx, Fiber* fiber);

  // The number of worker threads (for choosing a spawn_on target / sizing an
  // IoContext's per-worker rings).
  unsigned num_workers() const noexcept;

  // Block the calling thread until every spawned fiber has finished.
  void drain();

  // drain(), then stop the workers and join them. Idempotent-safe to call once.
  void shutdown();

  // ---- io poll hook (io-04 integration) -------------------------------------
  // A worker calls this when its run queue drains, to service io completions
  // (proactor CQ). It is a raw function pointer + opaque ctx, NOT an io type:
  // sched/ must not depend on io/ (the dependency runs the other way -- io/ uses
  // sched's Waker). io/ registers its IoContext here; the worker invokes it with
  // its own index and a timeout budget. When no io is in use the hook stays null
  // and the worker never calls it -- zero cost.
  //
  // timeout_ns == 0 is the non-blocking drain (run-queue-drained hot path). A
  // non-zero timeout is the worker's PARK boundary: the hook may BLOCK up to that
  // long harvesting completions (a GETEVENTS io_uring_enter), which is a
  // correctness requirement -- io_uring only surfaces many completions when the
  // owning thread enters the kernel, so a worker with ops in flight MUST block in
  // the ring rather than the futex parker or those completions never arrive.
  //
  // RETURNS true iff io ops were in flight (so the hook serviced/blocked on the
  // ring): the worker then loops instead of parking. false means nothing was in
  // flight (the hook did at most a cheap non-blocking drain), so the worker parks
  // on its futex Parker as usual -- giving an immediate cross-thread wake when no
  // io is pending (a reap block can't be interrupted by parker.unpark()).
  using IoPollFn = bool (*)(void* ctx, unsigned worker_idx,
                            unsigned long long timeout_ns) noexcept;
  void set_io_poll(IoPollFn fn, void* ctx) noexcept;

 private:
  friend class Waker;  // Waker::wake() re-injects via impl_ (completion spine)
  struct Impl;
  Impl* impl_;
};

}  // namespace hyle::sched
