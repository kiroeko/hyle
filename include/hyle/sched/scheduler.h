#pragma once

#include <cstddef>
#include <optional>

#include "hyle/error/status.h"
#include "hyle/core/compiler.h"
#include "hyle/mem/allocator.h"
#include "hyle/platform/cpu.h"
#include "hyle/platform/thread.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_SCHED_API
#elif defined(HYLE_SCHED_BUILD)
#  define HYLE_SCHED_API HYLE_EXPORT
#else
#  define HYLE_SCHED_API HYLE_IMPORT
#endif

namespace hyle::mem {
class FiberStackPool;  // sched-10: recycles spawn_task fiber stacks (avoid mmap churn)
}  // namespace hyle::mem

namespace hyle::sched {

class Fiber;
class Waker;  // sched-04: re-injects a suspended fiber (needs impl_ via friend)
HYLE_SCHED_API void wait_for_shutdown() noexcept;

struct SchedulerOptions {
  // A preference, not a hard filter: all process-available CPUs remain usable.
  // It only changes which CPUs lower-index workers are pinned to first.
  platform::CpuOrder worker_cpu_order = platform::CpuOrder::kOsDefault;
  // Optional OS scheduling preference. On macOS this maps to QoS and is the
  // portable way to request performance-oriented placement without fake pins.
  std::optional<platform::ThreadPriority> worker_priority;
};

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
  Scheduler(unsigned num_workers, mem::Allocator alloc,
            std::size_t queue_capacity, SchedulerOptions options);
  // Recoverable construction path for runtimes that treat allocation or
  // OS-thread creation failure as an operational error. On failure `status`
  // is non-OK and valid()==false; the object remains destructible but no other
  // operation may be called. The three-argument convenience constructor keeps
  // the native fail-fast contract and asserts construction succeeded.
  Scheduler(unsigned num_workers, mem::Allocator alloc,
            std::size_t queue_capacity, Status* status) noexcept;
  Scheduler(unsigned num_workers, mem::Allocator alloc,
            std::size_t queue_capacity, SchedulerOptions options,
            Status* status) noexcept;
  ~Scheduler();
  Scheduler(const Scheduler&) = delete;
  Scheduler& operator=(const Scheduler&) = delete;
  Scheduler(Scheduler&&) = delete;
  Scheduler& operator=(Scheduler&&) = delete;

  // Submit a runnable fiber. Lands on the caller's local queue if called from a
  // worker, else on the global queue. The scheduler does NOT own the Fiber; the
  // caller must keep it alive until it has finished running. Returns false when
  // graceful shutdown has closed admission, or when a worker recursively
  // submits after every bounded queue is full (returning avoids self-deadlock).
  // In either case the fiber was not queued/count-retained and remains entirely
  // owned by the caller. External producers apply backpressure instead.
  bool spawn(Fiber* fiber);

  // Submit a self-owned fire-and-forget callback. Unlike spawn(Fiber*), the
  // scheduler allocates the driver, obtains a guarded stack from its recycle
  // pool, and destroys both after the callback returns. This is the lightweight
  // product path for callbacks that do not need a result/JoinHandle; use
  // spawn_task for structured result and cancellation semantics.
  using DetachedFn = void (*)(void* ctx);
  Status spawn_detached(DetachedFn fn, void* ctx,
                        std::size_t stack_size) noexcept;

  // Queue a small non-suspending callback that runs directly on a worker stack.
  // This avoids allocating/switching a Fiber and is the scheduler counterpart
  // to Highway/marl lightweight work items. The callback MUST NOT call
  // yield/suspend/sleep, throw, or perform a blocking fiber wait; use
  // spawn_detached when it may suspend, and spawn_task when it also needs a
  // result/cancellation. Foreign producers apply bounded backpressure. A worker
  // recursively submitting after every inline shard is full receives
  // ResourceExhausted instead of spinning and deadlocking its own consumer.
  Status schedule(DetachedFn fn, void* ctx) noexcept;

  // Submit a fiber PINNED to worker `worker_idx`: it runs only on that worker and
  // is never stolen, on spawn and on every wake (shard-per-core, io-04). Use this
  // to keep a connection's fiber on the core that owns its io ring so submit +
  // reap + resume stay single-threaded. `worker_idx` must be < num_workers().
  bool spawn_on(unsigned worker_idx, Fiber* fiber);
  Status spawn_detached_on(unsigned worker_idx, DetachedFn fn, void* ctx,
                           std::size_t stack_size) noexcept;

  // The number of worker threads (for choosing a spawn_on target / sizing an
  // IoContext's per-worker rings).
  unsigned num_workers() const noexcept;
  bool valid() const noexcept { return impl_ != nullptr; }

  // Shared fiber-stack recycle pool for a given USABLE size (sched-10 spawn_task).
  // Returns a pool whose tier matches `usable_bytes` when it equals one of the
  // standard tiers (Small/Default/Large), else nullptr (caller allocates a
  // one-off stack). The pool is owned by the scheduler and outlives every fiber
  // it hands out (all spawn_task drivers are destroyed during shutdown() before
  // the scheduler dies), so a Fiber may safely release its stack back on destroy.
  // Thread-safe: acquire may run on any thread, release on the worker dispatch.
  mem::FiberStackPool* fiber_stack_pool(std::size_t usable_bytes) noexcept;

  // Block the calling thread until every spawned fiber has finished.
  void drain();

  // Drain finite work (admission stays open so running tasks may spawn
  // descendants), atomically close admission when the count reaches zero, then
  // stop and join workers. A concurrent external spawn is linearized at that
  // zero/close boundary: it is either counted and drained or rejected.
  // Long-lived/server work must call begin_shutdown() first so its fibers receive
  // shutdown_requested() and admission closes immediately. Concurrent calls
  // share one completion barrier and all return only after workers have joined.
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
  // request cancellation / close its I/O and call drain_for again. shutdown() never
  // abandons or force-kills a live stack: it will continue waiting for that fiber.
  // A process that cannot obtain cooperative completion must choose process-level
  // termination instead of calling shutdown() and expecting a bounded return.
  //
  // Whether a NEW spawn is accepted. False after begin_shutdown(). Lets a producer
  // (e.g. the accept loop) check before creating more work.
  bool accepting() const noexcept;
  // Enter phase 1: stop accepting new work + broadcast the shutdown signal so
  // long-lived fibers can wind down. Idempotent.
  void begin_shutdown() noexcept;
  // Suspend the current fiber owned by this scheduler until begin_shutdown().
  void wait_for_shutdown() noexcept;
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
  void init_(unsigned num_workers, mem::Allocator alloc,
             std::size_t queue_capacity, SchedulerOptions options,
             Status* status) noexcept;
  Status spawn_detached_(unsigned worker_idx, bool pinned, DetachedFn fn,
                         void* ctx, std::size_t stack_size) noexcept;
};

// True if the scheduler running the CURRENT fiber has begun shutting down (phase 1
// entered via begin_shutdown()). A long-lived fiber (accept/connection loop) checks
// this at its yield points and breaks its loop so it can finish -- the cooperative
// half of graceful shutdown (a stackful fiber can't be preempted, so it must poll).
// Reads a scheduler-wide flat flag via the worker's thread-local owner; returns
// false if not called from a scheduler worker fiber. Cheap (one relaxed load).
HYLE_SCHED_API bool shutdown_requested() noexcept;

// Suspend the current scheduler fiber until begin_shutdown() is called. Unlike
// polling shutdown_requested() between yields, this consumes no worker time and
// cannot miss the close edge. It is one arm for structured select patterns such
// as "accept/receive OR shutdown"; cancelling the other arm remains the caller's
// responsibility. Returns immediately if shutdown has already begun.
HYLE_SCHED_API void wait_for_shutdown() noexcept;

}  // namespace hyle::sched
