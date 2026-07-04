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

  // ---- graceful shutdown (sched-08, capability C4) --------------------------
  // Two-phase drain for a SERVER (long-lived fibers -- accept loops, connection
  // loops -- that never finish on their own, so plain drain() would block forever).
  //
  // PHASE 1: begin_shutdown() flips a flat "not accepting" flag and broadcasts it
  // to every worker (waking parked/reaping ones). New spawn()/spawn_on() after this
  // are REJECTED (the work isn't admitted -- like seastar's gate / tokio's closed
  // OwnedTasks). Long-lived fibers observe shutdown_requested() (below) at their
  // own yield points -- typically `when_any(shutdown_wait(), accept())` -- and
  // break their loop, which lets them finish and decrement the in-flight count.
  // This is a FLAT signal (one scheduler-wide flag broadcast), NOT a cancellation
  // tree (sched-07): every reference runtime -- seastar abort_source, tokio's
  // shutdown token -- uses a flat broadcast for shutdown; the tree is for the
  // orthogonal "cancel one task subtree" problem and is not needed here.
  //
  // PHASE 2: drain_for(timeout_ns) waits for the in-flight fibers to finish (the
  // count to reach zero). Returns true if drained, false on timeout. A stackful
  // fiber CANNOT be preempted, so on timeout the caller's recourse is cooperative:
  // request cancellation and hope fibers exit at their next yield point; a fiber
  // that still won't exit is left to the final shutdown() (its stack leaks, which
  // is acceptable at process exit -- tokio likewise abandons blocking threads on
  // timeout rather than force-killing). Then call shutdown() to stop+join workers.
  //
  // Whether a NEW spawn is accepted. False after begin_shutdown(). Lets a producer
  // (e.g. the accept loop) check before creating more work.
  bool accepting() const noexcept;
  // Enter phase 1: stop accepting new work + broadcast the shutdown signal so
  // long-lived fibers can wind down. Idempotent.
  void begin_shutdown() noexcept;
  // Wait up to `timeout_ns` for all in-flight fibers to finish (0 == wait forever,
  // same as drain()). Returns true if the count reached zero, false on timeout.
  bool drain_for(unsigned long long timeout_ns) noexcept;

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

  // Companion to set_io_poll: how the scheduler WAKES a worker that is blocked in
  // the io poll hook (a blocking reap) from another thread. When io is active an
  // idle worker blocks IN the proactor (a GETEVENTS reap / GQCSEx) with no poll
  // timeout -- truly asleep, no 1ms re-poll -- so a plain futex unpark can't reach
  // it. This hook breaks that block: io/ registers a fn that calls Proactor::wake()
  // on worker_idx's ring (writes its eventfd / posts its port). Same raw-fn-ptr +
  // ctx seam as set_io_poll (sched stays independent of io). Null when no io is in
  // use -> the worker parks on its futex Parker as before (zero cost).
  using IoWakeFn = void (*)(void* ctx, unsigned worker_idx) noexcept;
  void set_io_wake(IoWakeFn fn, void* ctx) noexcept;

 private:
  friend class Waker;  // Waker::wake() re-injects via impl_ (completion spine)
  struct Impl;
  Impl* impl_;
};

// True if the scheduler running the CURRENT fiber has begun shutting down (phase 1
// entered via begin_shutdown()). A long-lived fiber (accept/connection loop) checks
// this at its yield points and breaks its loop so it can finish -- the cooperative
// half of graceful shutdown (a stackful fiber can't be preempted, so it must poll).
// Reads a scheduler-wide flat flag via the worker's thread-local owner; returns
// false if not called from a scheduler worker fiber. Cheap (one relaxed load).
HYLE_SCHED_API bool shutdown_requested() noexcept;

}  // namespace hyle::sched
