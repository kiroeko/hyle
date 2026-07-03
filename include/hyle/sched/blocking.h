#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

#include "hyle/error/status.h"
#include "hyle/macros.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/join.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_SCHED_API
#elif defined(HYLE_SCHED_BUILD)
#  define HYLE_SCHED_API HYLE_EXPORT
#else
#  define HYLE_SCHED_API HYLE_IMPORT
#endif

// sched-11: spawn_blocking -- run a BLOCKING syscall / CPU-bound closure on a
// dedicated offload thread pool, NOT on a scheduler worker (sched-design §2.4).
//
// WHY: an M:N scheduler's worker threads == core count; each drives many fibers.
// A fiber that makes a blocking syscall (a DB driver, synchronous file IO,
// getaddrinfo, a long CPU crunch) stalls its WHOLE worker -- every other fiber
// pinned to that core stops until the syscall returns. The fix every real runtime
// ships (tokio spawn_blocking / marl blocking call / seastar thread): move the
// blocking work to a SEPARATE pool of threads whose only job is to block, so the
// worker stays free to run the core's other fibers.
//
// DESIGN (tokio blocking::pool, adopted; sysmon NOT included -- tokio has none
// either, blocking offload is EXPLICIT, the caller decides what blocks):
//  - A pool of OS threads, LAZILY spawned up to `max_threads` (default 512, far
//    above core count: blocking work mostly WAITS, it costs a thread stack not a
//    core, so oversubscription is the point). Idle threads self-retire after a
//    keep-alive timeout so a burst doesn't leave 512 threads parked forever.
//  - One shared MPMC job queue + one condvar; a submit wakes one idle thread, or
//    spawns a new one if none idle and under the cap, else the job waits in queue.
//  - Result travels back through the SAME JoinState<T>/JoinHandle<T> as spawn_task
//    (sched-10): the pool thread runs the closure, stores the result, and the CV
//    inside JoinState wakes whoever is join()-ing -- fiber-aware if that joiner is
//    a fiber on a worker, thread-mode if it's a bare thread. No bespoke plumbing.
//
// The pool uses std::mutex/condition_variable for its OWN thread management (its
// threads are meant to block -- they are not fibers), distinct from the fiber-
// aware primitives that guard scheduler workers.

namespace hyle::sched {

namespace detail {

// Type-erased unit of blocking work. Uses raw fn-ptr + ctx (the module's type-
// erasure convention -- Fiber::EntryFn/FinishFn, Scheduler::IoPollFn -- NOT a
// vtable): the concrete job is one BlockingJobImpl<F,R> whose static run/cancel/
// destroy are wired in here at spawn. A vtable would be equally cheap on this cold
// path (one call per job, then a blocking syscall dwarfs it) but would be the only
// virtual type in the whole library's production code; fn-ptr keeps it consistent.
// Intrusive `next` links the MPMC queue with no per-job container allocation.
// `ctx` points at the owning BlockingJobImpl; the three fns are its statics.
struct BlockingJob {
  using RunFn = void (*)(void* ctx) noexcept;      // run the closure, publish result
  using CancelFn = void (*)(void* ctx) noexcept;   // publish Cancelled, don't run
  using DestroyFn = void (*)(void* ctx) noexcept;  // ~impl + free from its allocator

  BlockingJob* next = nullptr;
  void* ctx = nullptr;
  RunFn run_fn = nullptr;
  CancelFn cancel_fn = nullptr;
  DestroyFn destroy_fn = nullptr;

  void run() noexcept { run_fn(ctx); }
  void cancel() noexcept { cancel_fn(ctx); }
  void destroy() noexcept { destroy_fn(ctx); }
};

}  // namespace detail

// The offload thread pool. One per runtime is typical; construct with the OS
// allocator (its threads + jobs come from it). Move/copy disabled -- hand out
// references. Threads are joined on destruction (see shutdown()).
class HYLE_SCHED_API BlockingPool {
 public:
  // `max_threads` caps concurrently-live pool threads (default 512, tokio's
  // default). `keep_alive_ns` is how long an idle thread waits for work before
  // retiring (default 10s, tokio's default). alloc backs pool threads' bookkeeping.
  explicit BlockingPool(mem::Allocator alloc, unsigned max_threads = 512,
                        unsigned long long keep_alive_ns = 10ull * 1000 * 1000 * 1000);
  ~BlockingPool();
  HYLE_DISABLE_COPY_AND_MOVE(BlockingPool);

  // Stop accepting work, wake all threads, and join them. Jobs still RUNNING
  // finish; jobs still QUEUED (not yet started) are dropped -- their JoinHandle
  // resolves to Cancelled. Idempotent; also called by the destructor.
  void shutdown();

  // Diagnostics (racy snapshots).
  unsigned live_threads() const noexcept;

  // Submit a pre-built job (internal seam for spawn_blocking). Returns false if
  // the pool is shutting down (the caller must resolve the handle as Cancelled).
  bool submit(detail::BlockingJob* job) noexcept;

 private:
  struct Impl;
  Impl* impl_;
};

namespace detail {

// Concrete job: EMBEDS a BlockingJob header (not inheritance) whose fn-ptrs point
// at these statics. Owns the closure + the result slot. R = closure's return type.
template <class F, class R>
struct BlockingJobImpl {
  BlockingJob hdr;
  mem::Allocator alloc;
  F fn;
  JoinState<R>* state;

  BlockingJobImpl(mem::Allocator a, F&& f, JoinState<R>* s) noexcept
      : alloc(a), fn(std::move(f)), state(s) {
    hdr.ctx = this;
    hdr.run_fn = &run_impl;
    hdr.cancel_fn = &cancel_impl;
    hdr.destroy_fn = &destroy_impl;
  }

  static void run_impl(void* p) noexcept {
    auto* self = static_cast<BlockingJobImpl*>(p);
    R value = self->fn();                      // the blocking work (pool thread)
    self->state->set_value(std::move(value));  // publish + wake the joiner (CV)
  }
  static void cancel_impl(void* p) noexcept {
    static_cast<BlockingJobImpl*>(p)->state->set_error(
        Status::Cancelled("blocking pool shut down"));
  }
  static void destroy_impl(void* p) noexcept {
    auto* self = static_cast<BlockingJobImpl*>(p);
    mem::Allocator a = self->alloc;
    JoinState<R>* st = self->state;
    self->~BlockingJobImpl();
    a.deallocate(self, sizeof(BlockingJobImpl), alignof(BlockingJobImpl));
    st->release();  // drop the pool-side ref (join side keeps its own)
  }
};

// void specialization: no result value.
template <class F>
struct BlockingJobImpl<F, void> {
  BlockingJob hdr;
  mem::Allocator alloc;
  F fn;
  JoinState<void>* state;

  BlockingJobImpl(mem::Allocator a, F&& f, JoinState<void>* s) noexcept
      : alloc(a), fn(std::move(f)), state(s) {
    hdr.ctx = this;
    hdr.run_fn = &run_impl;
    hdr.cancel_fn = &cancel_impl;
    hdr.destroy_fn = &destroy_impl;
  }

  static void run_impl(void* p) noexcept {
    auto* self = static_cast<BlockingJobImpl*>(p);
    self->fn();
    self->state->set_done();
  }
  static void cancel_impl(void* p) noexcept {
    static_cast<BlockingJobImpl*>(p)->state->set_error(
        Status::Cancelled("blocking pool shut down"));
  }
  static void destroy_impl(void* p) noexcept {
    auto* self = static_cast<BlockingJobImpl*>(p);
    mem::Allocator a = self->alloc;
    JoinState<void>* st = self->state;
    self->~BlockingJobImpl();
    a.deallocate(self, sizeof(BlockingJobImpl), alignof(BlockingJobImpl));
    st->release();
  }
};

}  // namespace detail

// Run `fn` on the blocking pool; get a JoinHandle for its result. join() blocks
// (fiber-aware from a worker fiber, thread-mode from a bare thread) until the pool
// thread finishes. If the pool is shutting down the handle resolves to Cancelled.
//
// `fn` must be callable with no args; its return type R becomes JoinHandle<R>.
// The closure is MOVED onto the pool (it runs on another thread) -- capture by
// value, and remember its body runs OUTSIDE any fiber (no fiber-local APIs there).
template <class F, class R = std::invoke_result_t<F&>>
JoinHandle<R> spawn_blocking(BlockingPool& pool, mem::Allocator alloc,
                             F fn) noexcept {
  using detail::BlockingJobImpl;
  using detail::JoinState;

  auto* st = static_cast<JoinState<R>*>(
      alloc.allocate(sizeof(JoinState<R>), alignof(JoinState<R>)));
  HYLE_FATAL_ASSERT(st != nullptr);
  new (st) JoinState<R>{};
  st->alloc = alloc;

  auto* job = static_cast<BlockingJobImpl<F, R>*>(
      alloc.allocate(sizeof(BlockingJobImpl<F, R>), alignof(BlockingJobImpl<F, R>)));
  HYLE_FATAL_ASSERT(job != nullptr);
  new (job) BlockingJobImpl<F, R>(alloc, std::move(fn), st);

  if (!pool.submit(&job->hdr)) {
    // Pool shutting down: run nothing, resolve the handle as Cancelled, then free
    // the job (destroy drops the pool-side ref). cancel publishes the error.
    job->hdr.cancel();
    job->hdr.destroy();
  }
  return JoinHandle<R>(st);
}

}  // namespace hyle::sched
