#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <new>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>

#include "hyle/error/status.h"
#include "hyle/core/compiler.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/fiber_stack.h"
#include "hyle/platform/cpu.h"
#include "hyle/platform/thread.h"
#include "hyle/time/clock.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_SCHED_API
#elif defined(HYLE_SCHED_BUILD)
#  define HYLE_SCHED_API HYLE_EXPORT
#else
#  define HYLE_SCHED_API HYLE_IMPORT
#endif

namespace hyle::mem {
class FiberStackPool;  // recycles structured-task stacks without repeated VM churn
}  // namespace hyle::mem

namespace hyle::sched {

class Fiber;
class Waker;  // re-injects a suspended fiber through the private runtime seam
namespace detail {
class SchedulerIntegration;
class SchedulerAccess;
}
HYLE_SCHED_API void wait_for_shutdown() noexcept;

enum class SchedulerIdlePolicy : std::uint8_t {
  kBalanced = 0,
  kLowLatency = 1,
  kPowerSaving = 2,
};

using SchedulerWorkerHook = void (*)(void* context,
                                     unsigned worker_index) noexcept;

struct SchedulerOptions {
  // Default is deliberately unpinned so an embedded scheduler composes with a
  // host runtime. Set worker_cpu_order to select and pin process-available CPUs,
  // or provide an explicit ordered CPU set. The set is copied synchronously by
  // Scheduler construction; its backing only needs to outlive the constructor.
  std::optional<platform::CpuOrder> worker_cpu_order;
  std::span<const unsigned> worker_cpu_set;
  // kRequired turns an invalid/unavailable requested name, affinity or priority
  // into construction failure. Best-effort is portable (notably on macOS).
  platform::ThreadConfigPolicy worker_config_policy =
      platform::ThreadConfigPolicy::kBestEffort;
  // Optional OS scheduling preference. On macOS this maps to QoS and is the
  // portable way to request performance-oriented placement without fake pins.
  std::optional<platform::ThreadPriority> worker_priority;
  // Semantic latency/energy choice. Balanced preserves the normal bounded
  // search; LowLatency spins longer before parking; PowerSaving parks directly.
  SchedulerIdlePolicy idle_policy = SchedulerIdlePolicy::kBalanced;
  // Cold lifecycle hooks for backend TLS, scratch arenas or profiler setup.
  // They run on each worker after scheduler TLS is installed and immediately
  // before it is removed. Hooks must not throw or block shutdown indefinitely.
  SchedulerWorkerHook on_worker_start = nullptr;
  SchedulerWorkerHook on_worker_stop = nullptr;
  void* worker_hook_context = nullptr;
};

// M:N work-stealing scheduler: N OS-thread workers, each with an owner FIFO and
// one bounded run-next slot, plus sharded external injection queues and
// random-victim stealing. Idle behavior is selected by SchedulerIdlePolicy and
// converges on park/unpark rather than burning a worker indefinitely.
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
  // schedule_task for a non-suspending typed result and spawn_task when the
  // structured task may suspend.
  using DetachedFn = void (*)(void* ctx);
  Status spawn_detached(DetachedFn fn, void* ctx,
                        std::size_t stack_size =
                            mem::kFiberStackDefault) noexcept;

  // Owning C++ convenience for capturing/move-only noexcept callables. It pays
  // one allocator-backed callable record; the raw fn+ctx overload above remains
  // the allocation-minimal route. The callable may use fiber suspension APIs.
  template <class Fn>
    requires std::invocable<std::decay_t<Fn>&> &&
             std::is_nothrow_invocable_v<std::decay_t<Fn>&> &&
             std::is_nothrow_constructible_v<std::decay_t<Fn>, Fn&&>
  Status spawn_detached(Fn&& fn,
                        std::size_t stack_size =
                            mem::kFiberStackDefault) noexcept {
    return submit_owned_callable_(false, 0, true, stack_size,
                                  std::forward<Fn>(fn));
  }

  // Queue a small non-suspending callback that runs directly on a worker stack.
  // This avoids allocating/switching a Fiber and is the scheduler counterpart
  // to Highway/marl lightweight work items. The callback MUST NOT call
  // yield/suspend/sleep, throw, or perform a blocking fiber wait; use
  // schedule_task when it needs a typed result but still cannot suspend,
  // spawn_detached when it may suspend without a result, and spawn_task when a
  // result-bearing task may suspend. Foreign producers apply bounded
  // backpressure. A worker recursively submitting after every inline shard is
  // full receives ResourceExhausted instead of spinning and deadlocking its own
  // consumer.
  Status schedule(DetachedFn fn, void* ctx) noexcept;

  // Natural one-shot lambda route for short, non-suspending work. The callable
  // record uses this Scheduler's injected allocator and destroys itself after
  // invocation. No type-erasure allocation is charged to the raw overload.
  template <class Fn>
    requires std::invocable<std::decay_t<Fn>&> &&
             std::is_nothrow_invocable_v<std::decay_t<Fn>&> &&
             std::is_nothrow_constructible_v<std::decay_t<Fn>, Fn&&>
  Status schedule(Fn&& fn) noexcept {
    return submit_owned_callable_(false, 0, false, 0,
                                  std::forward<Fn>(fn));
  }

  // Submit a fiber PINNED to worker `worker_idx`: it runs only on that worker and
  // is never stolen, on spawn and on every wake. Use this
  // to keep a connection's fiber on the core that owns its io ring so submit +
  // reap + resume stay single-threaded. `worker_idx` must be < num_workers().
  bool spawn_on(unsigned worker_idx, Fiber* fiber);
  Status spawn_detached_on(unsigned worker_idx, DetachedFn fn, void* ctx,
                           std::size_t stack_size =
                               mem::kFiberStackDefault) noexcept;
  template <class Fn>
    requires std::invocable<std::decay_t<Fn>&> &&
             std::is_nothrow_invocable_v<std::decay_t<Fn>&> &&
             std::is_nothrow_constructible_v<std::decay_t<Fn>, Fn&&>
  Status spawn_detached_on(
      unsigned worker_idx, Fn&& fn,
      std::size_t stack_size = mem::kFiberStackDefault) noexcept {
    return submit_owned_callable_(true, worker_idx, true, stack_size,
                                  std::forward<Fn>(fn));
  }

  // The number of worker threads (for choosing a spawn_on target / sizing an
  // IoContext's per-worker rings).
  unsigned num_workers() const noexcept;
  bool valid() const noexcept { return impl_ != nullptr; }

  // Shared fiber-stack recycle pool for a given USABLE size.
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

  // ---- graceful shutdown ----------------------------------------------------
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
  // tree: reference runtimes such as seastar abort_source and tokio's
  // shutdown token -- uses a flat broadcast for shutdown; the tree is for the
  // orthogonal "cancel one task subtree" problem and is not needed here.
  //
  // PHASE 2: drain_for(timeout) waits for the in-flight fibers to finish (the
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
  // Wait up to `timeout` for all in-flight fibers to finish. A non-positive
  // duration is an immediate check; use drain() for an unbounded wait.
  bool drain_for(time::Duration timeout) noexcept;
  // Absolute monotonic counterpart. A past deadline is an immediate check.
  bool drain_until(time::MonotonicTimePoint deadline) noexcept;

 private:
  friend class Waker;  // Waker::wake() re-injects via impl_ (completion spine)
  friend class detail::SchedulerIntegration;
  friend class detail::SchedulerAccess;
  using IoPollFn = bool (*)(void* context, unsigned worker_index,
                            unsigned long long timeout_ns) noexcept;
  using IoWakeFn = void (*)(void* context,
                            unsigned worker_index) noexcept;
  // Build-only IO integration. Public users compose through IoContext; these
  // single-slot worker-loop hooks are deliberately inaccessible as scheduler
  // policy and are not installed as an SDK header.
  void set_io_poll_(IoPollFn fn, void* context) noexcept;
  void set_io_wake_(IoWakeFn fn, void* context) noexcept;
  struct Impl;
  Impl* impl_;
  void init_(unsigned num_workers, mem::Allocator alloc,
             std::size_t queue_capacity, SchedulerOptions options,
             Status* status) noexcept;
  void* allocate_owned_callable_(std::size_t size,
                                 std::size_t alignment) noexcept;
  void deallocate_owned_callable_(void* pointer, std::size_t size,
                                  std::size_t alignment) noexcept;

  template <class Fn>
  struct OwnedCallable {
    Scheduler* scheduler;
    Fn fn;

    static void run(void* raw) noexcept {
      auto* self = static_cast<OwnedCallable*>(raw);
      Scheduler* scheduler = self->scheduler;
      std::invoke(self->fn);
      self->~OwnedCallable();
      scheduler->deallocate_owned_callable_(self, sizeof(OwnedCallable),
                                            alignof(OwnedCallable));
    }
  };

  template <class Fn>
  Status submit_owned_callable_(bool pinned, unsigned worker_idx,
                                bool detached, std::size_t stack_size,
                                Fn&& fn) noexcept {
    using Callable = std::decay_t<Fn>;
    using Record = OwnedCallable<Callable>;
    void* storage =
        allocate_owned_callable_(sizeof(Record), alignof(Record));
    if (storage == nullptr) {
      return Status::ResourceExhausted(
          "scheduler callable allocation failed");
    }
    auto* record = new (storage)
        Record{this, Callable(std::forward<Fn>(fn))};
    Status status;
    if (!detached) {
      status = schedule(&Record::run, record);
    } else if (pinned) {
      status = spawn_detached_on(worker_idx, &Record::run, record,
                                 stack_size);
    } else {
      status = spawn_detached(&Record::run, record, stack_size);
    }
    if (!status.ok()) {
      record->~Record();
      deallocate_owned_callable_(record, sizeof(Record), alignof(Record));
    }
    return status;
  }
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
