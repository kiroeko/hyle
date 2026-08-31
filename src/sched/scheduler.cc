#include "hyle/sched/scheduler.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "hyle/container/bounded_queue.h"
#include "hyle/container/sharded_queue.h"
#include "hyle/container/ws_fifo_queue.h"
#include "hyle/core/cache.h"
#include "hyle/core/compiler.h"
#include "hyle/mem/fiber_stack.h"
#include "hyle/mem/std_allocator.h"
#include "hyle/obs/trace.h"
#include "hyle/platform/cpu.h"
#include "hyle/platform/thread.h"
#include "hyle/platform/wait.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/timer.h"
#include "hyle/sched/waker.h"
#include "hyle/time/clock.h"
#include "hyle/time/timer_queue.h"
#include "hyle/util/spin_lock.h"  // util::cpu_relax (pause hint)

#include "fiber_access.h"
#include "scheduler_access.h"

namespace hyle::sched {

namespace {

// The Scheduler running on this worker thread, for current_waker(). File-local
// (not an Impl member) so the free functions below can read it without naming
// the private nested Impl type.
thread_local Scheduler* tls_owner = nullptr;

// The current worker's timing queue, published by run_worker (file-local so the
// sleep_*_ns free functions can arm a timer without naming the private nested
// Impl::Worker type). Null when not on a worker fiber. Single-threaded access:
// only the owning worker thread reads/writes its own queue (arm happens inside a
// fiber running ON that worker; fire happens in that worker's loop).
thread_local time::TimerQueue* tls_timer_queue = nullptr;

// The current worker's index, published by run_worker (file-local so the
// current_worker_index() free function can read it without naming the private
// nested Impl). -1 when not on a worker fiber. io/ (plan A) reads this to pick
// this worker's per-core proactor for submit, so submit + reap share a thread.
thread_local int tls_worker_idx = -1;

// Per-worker xorshift64 for cheap random victim selection. It deliberately
// remains a scheduler-internal hint rather than consumer-visible randomness.
// The idle steal hot path benefits from its one-word state.
std::uint64_t xorshift(std::uint64_t& s) noexcept {
  s ^= s << 13;
  s ^= s >> 7;
  s ^= s << 17;
  return s;
}

struct DetachedDriver {
  mem::Allocator alloc;
  Scheduler::DetachedFn fn;
  void* ctx;
  Fiber fiber;

  DetachedDriver(mem::Allocator a, Scheduler::DetachedFn task, void* context,
                 mem::FiberStack stack, mem::FiberStackPool* pool) noexcept
      : alloc(a),
        fn(task),
        ctx(context),
        fiber(std::move(stack), &entry, this, pool) {
    detail::FiberAccess::set_on_finish(&fiber, &destroy, this);
  }

  static void entry(void* p) {
    auto* self = static_cast<DetachedDriver*>(p);
    self->fn(self->ctx);
  }

  static void destroy(void* p) noexcept {
    auto* self = static_cast<DetachedDriver*>(p);
    const mem::Allocator alloc = self->alloc;
    self->~DetachedDriver();
    alloc.deallocate(self, sizeof(DetachedDriver), alignof(DetachedDriver));
  }
};

struct InlineJob {
  Scheduler::DetachedFn fn;
  void* ctx;
  std::uint32_t accounting_idx = 0;
};
static_assert(std::is_trivially_copyable_v<InlineJob>);

// Allocation-free cold overflow for READY fibers. A wake/reschedule cannot
// report "queue full" or spin on the worker that must drain that same queue.
// The bounded lock-free rings remain the hot path; this intrusive FIFO is used
// only after every appropriate ring is full.
struct ReadyOverflow {
  std::mutex mu;
  std::atomic<bool> nonempty{false};
  Fiber* head = nullptr;
  Fiber* tail = nullptr;

  void push(Fiber* fiber) noexcept {
    std::lock_guard<std::mutex> lock(mu);
    HYLE_FATAL_ASSERT(detail::FiberAccess::next(fiber) == nullptr);
    if (tail == nullptr) {
      head = fiber;
    } else {
      detail::FiberAccess::set_next(tail, fiber);
    }
    tail = fiber;
    nonempty.store(true, std::memory_order_release);
  }

  Fiber* pop() noexcept {
    // Preserve the bounded-queue hot path: an unused overflow costs one atomic
    // read, not a mutex acquisition on every scheduler tick.
    if (!nonempty.load(std::memory_order_acquire)) return nullptr;
    std::lock_guard<std::mutex> lock(mu);
    Fiber* fiber = head;
    if (fiber == nullptr) return nullptr;
    head = detail::FiberAccess::next(fiber);
    if (head == nullptr) {
      tail = nullptr;
      nonempty.store(false, std::memory_order_release);
    }
    detail::FiberAccess::set_next(fiber, nullptr);
    return fiber;
  }
};

// Idle backoff while SEARCHING. The §2.3 ②③ anti-herd gate (num_searching_ +
// notify() seq_cst Dekker, see notify()/run_worker) is what kills the spawn-burst
// futex storm at the source: a producer skips waking parked workers while a
// searcher exists. This spin then just keeps the searcher warm to catch the next
// item before the (now rare) park, and bridges sub-µs producer gaps. cpu_relax,
// no syscall. (Prior comment said §2.3 ②③ were deferred -- they are now done.)
constexpr int kBalancedSpinPolls = 4096;
constexpr int kBalancedYieldsBeforePark = 4;
constexpr int kLowLatencySpinPolls = 32768;

// Cap on consecutive LIFO-slot polls before the slot's fiber is demoted to the
// local FIFO ring, so a hot ping-pong / RPC pair can't monopolize the worker and
// starve its ring (tokio MAX_LIFO_POLLS_PER_TICK = 3). See the internal
// lifo_slot field + find_task for the mechanism.
constexpr int kMaxLifoPolls = 3;

// Every Nth find_task, check the GLOBAL inject queue BEFORE the local ring, so a
// worker whose local queue is continuously refilled (e.g. a fiber that yields in a
// loop -> reschedule -> local.try_push_back) cannot starve tasks injected to the global
// queue indefinitely (complexity opportunity ③; starvation CONFIRMED by a probe:
// a 1-worker yield-hog did 17M yields while a globally-injected victim never ran in
// 500ms). tokio's global_queue_interval, default 61 -- a prime, to avoid harmonic
// resonance with power-of-two batch/queue sizes. Fixed (not EWMA-tuned): the tuning
// only matters for very-long-poll workloads, and a fixed prime already closes the
// starvation hole, which is the correctness point here.
constexpr std::uint32_t kGlobalQueueInterval = 61;

// Worker-stack callbacks cannot suspend, so draining a small bounded batch is
// safe and avoids a full pinned/local/global/steal scan after every two-word
// callback. The bound is a fairness budget, not an unbounded "inline queue
// wins" policy: after at most eight callbacks the worker must service fibers,
// timers and I/O again.
constexpr int kInlineBatch = 8;


}  // namespace

struct Scheduler::Impl {
  struct Worker {
    container::WsFifoQueue<Fiber*, mem::Allocator> local;
    platform::Parker parker;  // non-movable -> Worker has a stable allocation
    // Per-worker timing queue (sched-04b). Owned and touched ONLY by this worker
    // thread on its park cold path -> no lock needed. Sleeping fibers arm here;
    // the worker advances it before parking and fires due timers into the run
    // loop. Sharding the queue per-worker (vs one central) is the sched-side
    // choice the time-02 doc left open; per-worker keeps timer + woken fiber on
    // the same core (cache-hot re-inject) and needs no queue lock. Epoch = the
    // monotonic ns captured at construction.
    time::TimerQueue timers;
    // Private queue for fibers PINNED to this worker (shard-per-core, io-04). A
    // bounded MPMC ring: any thread (a Waker firing on another core, or spawn_on)
    // pushes here, but ONLY this worker pops it, and the work-stealing path never
    // touches it -- so a pinned fiber can never be stolen onto another core. Kept
    // separate from `local` (which IS stealable) precisely so stealing stays
    // untouched (no change to the anti-herd Dekker/CAS math). Checked before
    // `local` in find_task so a woken connection fiber resumes promptly.
    container::BoundedQueue<Fiber*, mem::Allocator> pinned;
    ReadyOverflow pinned_overflow;
    std::uint64_t rng;
    unsigned idx;             // own worker index: shard hint + pop-cursor seed
    std::size_t pop_cursor;   // round-robin cursor into the global ShardedQueue
    std::size_t inline_cursor;
    // LIFO fast-slot (tokio lifo_slot, complexity opportunity ①). A single "run
    // this next" fiber, filled when a wake fires ON this worker (reinject): the
    // just-woken fiber resumes immediately on THIS core, stack + buffers cache-hot,
    // instead of bouncing through the global queue + a notify() futex to some other
    // core. TOUCHED ONLY BY THIS WORKER THREAD (filled by reinject when it runs on
    // this worker, drained by find_task), so it is a plain field -- no atomics, and
    // the work-stealing path never sees it, so a slot fiber is naturally steal-
    // exempt (a ping-pong partner can't be stolen away mid-flight). lifo_polls caps
    // consecutive slot hits (kMaxLifoPolls) so the ring can't starve.
    Fiber* lifo_slot = nullptr;
    int lifo_polls = 0;
    // Per-worker find_task counter for the global-queue interval check (opportunity
    // ③). Every kGlobalQueueInterval-th call, find_task drains the global queue
    // first so a locally-firehosing worker can't starve global inject. u32 wrapping
    // is fine (only the modulo matters). Single-threaded: only this worker touches it.
    std::uint32_t tick = 0;
    // Callback accounting is sharded by worker so the producer's admission RMW
    // does not ping-pong one global cache line with all completing workers.
    // This field owns its cache line; otherwise producer writes would also
    // invalidate the worker-only cursors above it.
    alignas(core::kMaxDestructiveInterferenceSize)
        std::atomic<std::uint64_t> inline_outstanding{0};
    Worker(mem::Allocator a, std::size_t cap, std::uint64_t seed,
           unsigned i) noexcept
        : local(a, cap),
          timers(time::monotonic_ns()),
          pinned(a, cap),
          rng(seed),
          idx(i),
          pop_cursor(i),
          inline_cursor(i) {}
  };

  struct Ctx {
    Impl* self;
    unsigned idx;
  };

  // Global inject queue: SHARDED bounded MPMC (one ring per worker). The single
  // ring this used to be was a single-point CAS bottleneck under many concurrent
  // injectors (bench-02b: ~1.7x slower than moodycamel at 4P, diagnosed to the one
  // contended enqueue_pos_). Sharding gives each producer its own ring -> the
  // global queue scales with injectors (sharded bench: reversed to ~2.6-3x FASTER
  // than moodycamel, flat with producer count). Order across shards is not global
  // FIFO -- fine here: a work-stealing scheduler has no global order, it only needs
  // no-starvation + rough fairness. workers pop with their own round-robin cursor;
  // worker-context pushes hint to their own shard (locality), foreign pushes
  // round-robin via inject_rr_. Uses the OPT-IN runtime mem::Allocator variant (the
  // scheduler's public ctor injects an allocator; queues honor it).
  container::ShardedQueue<Fiber*, mem::Allocator> global;
  ReadyOverflow global_overflow;
  // Worker-stack callbacks need no owning heap record: queue the type-erased
  // function/context pair plus an internal accounting-shard tag by value. The
  // admitted count keeps shutdown/drain correct while removing one malloc/free
  // pair per tiny task.
  container::ShardedQueue<InlineJob, mem::Allocator> inline_jobs;
  mem::Allocator allocator;
  std::vector<Worker*, mem::StdAllocator<Worker*>> workers;
  std::vector<Ctx, mem::StdAllocator<Ctx>> ctxs;
  std::vector<platform::Thread, mem::StdAllocator<platform::Thread>> threads;
  std::atomic<bool> stop{false};
  // Fiber admission and its in-flight count share one linearizable state.
  // Separate `accepting` + a fiber count previously allowed this execution:
  //   spawn reads accepting=true; shutdown closes + observes outstanding=0;
  //   spawn increments and queues after shutdown has returned.
  // That stranded raw fibers and made spawn_task leak/hang forever. The top bit
  // closes admission; the remaining bits count admitted fibers. A CAS admission
  // is therefore ordered either wholly before close (and drain sees it) or wholly
  // after close (and is rejected).
  static constexpr std::uint64_t kAdmissionClosed = 1ull << 63;
  static constexpr std::uint64_t kOutstandingMask = ~kAdmissionClosed;
  std::atomic<std::uint64_t> lifecycle{0};
  // Worker-stack callbacks use per-worker counters (in Worker) instead of
  // contending on lifecycle's CAS loop or one shared completion cache line.
  // Admission uses open -> increment shard -> recheck: once close wins, a
  // racing callback either rejects and rolls back or was already counted and
  // is observed by drain().
  // Address-wait epoch used by external drainers. Completion of the final fiber
  // advances it and wakes waiters, avoiding a full CPU of yield-spin at teardown.
  std::atomic<std::uint32_t> drain_epoch{0};
  // Serialize the single stop+join owner. Idempotent shutdown must be a
  // completion barrier even when several external threads call it together.
  std::mutex stop_mu;
  std::condition_variable stop_cv;
  bool shutdown_started = false;
  bool shutdown_complete = false;
  struct ShutdownWaiter {
    ShutdownWaiter* next = nullptr;
    Waker waker;
    bool linked = false;
  };
  std::mutex shutdown_mu;
  ShutdownWaiter* shutdown_waiters = nullptr;

  // Shared fiber-stack recycle pools, one per standard tier (sched-10 spawn_task).
  // spawn_task acquires from the tier matching its stack size and the Fiber
  // releases back on destroy (in the worker dispatch), so a hot spawn/join loop
  // reuses stacks instead of paying reserve_guarded's 3 syscalls (VirtualAlloc
  // reserve + commit + VirtualProtect / mmap + mprotect) every time. The pools
  // live on the Impl, which outlives every driver (shutdown() drains all fibers
  // before ~Scheduler), so release-on-destroy is always into a live pool.
  // FiberStackPool holds a SpinLock + inline slot array (non-movable) -> heap it.
  mem::FiberStackPool pool_small{mem::kFiberStackSmall};
  mem::FiberStackPool pool_default{mem::kFiberStackDefault};
  mem::FiberStackPool pool_large{mem::kFiberStackLarge};

  mem::FiberStackPool* fiber_stack_pool(std::size_t usable) noexcept {
    if (usable == mem::kFiberStackSmall) return &pool_small;
    if (usable == mem::kFiberStackDefault) return &pool_default;
    if (usable == mem::kFiberStackLarge) return &pool_large;
    return nullptr;  // non-tier size: caller allocates a one-off stack
  }

  // IO poll hook. Registered by IoContext through the build-only integration
  // bridge; the worker loop calls it to reap completions. Null with no IO ->
  // never called, zero cost. ATOMIC because the registering thread (IoContext
  // ctor on the main thread) writes these while worker threads are already
  // spinning their loops and reading them -- a genuine cross-thread publish
  // (TSan caught the plain-field version). Write ctx first then fn with release;
  // read fn (acquire) then ctx, so a non-null fn implies its ctx is visible.
  std::atomic<Scheduler::IoPollFn> io_poll{nullptr};
  std::atomic<void*> io_poll_ctx{nullptr};
  // io wake hook (io-04 park merge): breaks a worker blocked in the io poll hook.
  // Same publish discipline as io_poll (write ctx then fn with release). io_poll
  // and io_wake are registered together by IoContext, but read independently.
  std::atomic<Scheduler::IoWakeFn> io_wake{nullptr};
  std::atomic<void*> io_wake_ctx{nullptr};
  Status init_status;

  std::mutex idle_mu;              // guards `sleepers` (cold path: park/unpark)
  std::vector<unsigned, mem::StdAllocator<unsigned>> sleepers;
  // Per-worker position in `sleepers` (or kNotSleeping), so remove_sleeper is O(1)
  // instead of an O(worker-count) scan (complexity opportunity ②). HYLE calls
  // remove_sleeper on EVERY park exit (unlike tokio, which only scans on the rare
  // self-wake-with-local-work path and thus tolerates the scan) -- and the scan
  // runs while holding idle_mu, so shrinking it also shortens that critical section
  // under park/unpark churn. All three mutators (register on park, unpark_one pop,
  // remove_sleeper) keep this in sync; a swap-remove updates the moved id's slot.
  // Guarded by idle_mu (same lock as `sleepers`) -- no new atomics.
  static constexpr std::size_t kNotSleeping = static_cast<std::size_t>(-1);
  std::vector<std::size_t, mem::StdAllocator<std::size_t>> sleeper_pos;
  // Count of workers currently SEARCHING for work (spinning/stealing, not yet
  // parked). The §2.3 ②③ anti-herd gate: a producer skips waking a parked worker
  // when a searcher already exists (it will find the new work), killing the
  // per-enqueue futex storm at the source. Lost-wakeup safety is the seq_cst
  // StoreLoad "Dekker" handshake between notify() and a worker about to park
  // (documented at both sites). tokio idle.rs / Go wakep / actives-thieves paper.
  std::atomic<int> num_searching_{0};
  // Round-robin shard selector for FOREIGN (non-worker) injects (reinject/spawn
  // off a worker thread) -- they have no worker idx, so spread them across shards.
  std::atomic<std::size_t> inject_rr_{0};
  unsigned n;
  Scheduler* owner;  // for tls_owner / current_waker()
  int spin_polls = kBalancedSpinPolls;
  int yields_before_park = kBalancedYieldsBeforePark;
  SchedulerWorkerHook on_worker_start = nullptr;
  SchedulerWorkerHook on_worker_stop = nullptr;
  void* worker_hook_context = nullptr;

  Impl(unsigned num_workers, mem::Allocator alloc, std::size_t cap,
       SchedulerOptions options,
       Scheduler* sched)
      : global(alloc, num_workers, cap),
        inline_jobs(alloc, num_workers, cap < 256 ? cap : 256),
        allocator(alloc),
        workers(mem::StdAllocator<Worker*>(alloc)),
        ctxs(mem::StdAllocator<Ctx>(alloc)),
        threads(mem::StdAllocator<platform::Thread>(alloc)),
        sleepers(mem::StdAllocator<unsigned>(alloc)),
        sleeper_pos(mem::StdAllocator<std::size_t>(alloc)),
        n(num_workers),
        owner(sched),
        on_worker_start(options.on_worker_start),
        on_worker_stop(options.on_worker_stop),
        worker_hook_context(options.worker_hook_context) {
    HYLE_FATAL_ASSERT(num_workers >= 1);
    switch (options.idle_policy) {
      case SchedulerIdlePolicy::kBalanced:
        spin_polls = kBalancedSpinPolls;
        yields_before_park = kBalancedYieldsBeforePark;
        break;
      case SchedulerIdlePolicy::kLowLatency:
        spin_polls = kLowLatencySpinPolls;
        yields_before_park = kBalancedYieldsBeforePark;
        break;
      case SchedulerIdlePolicy::kPowerSaving:
        spin_polls = 0;
        yields_before_park = 0;
        break;
    }
    sleeper_pos.assign(num_workers, kNotSleeping);  // O(1) sleeper-set bookkeeping
    workers.reserve(num_workers);
    ctxs.reserve(num_workers);
    threads.reserve(num_workers);
    // Default construction is deliberately unrestricted: an embedded runtime
    // must not silently replace its host's placement policy. Affinity ids are OS
    // ids, not dense worker ordinals. Linux cpusets can be sparse (for example
    // {4, 6}); Windows ids flatten processor groups. Only an explicit CPU set or
    // ordering request enters this path. If there are fewer CPUs than workers,
    // round-robin is an honest, deterministic oversubscription policy. Resolve
    // this allocator-backed temporary before constructing raw Worker records so
    // a vector allocation exception cannot strand those records.
    std::vector<unsigned, mem::StdAllocator<unsigned>> worker_cpus{
        mem::StdAllocator<unsigned>(alloc)};
    if (!options.worker_cpu_set.empty()) {
      worker_cpus.assign(options.worker_cpu_set.begin(),
                         options.worker_cpu_set.end());
    } else if (options.worker_cpu_order.has_value()) {
      worker_cpus.resize(num_workers);
      auto available = platform::available_cpu_ids(
          *options.worker_cpu_order, worker_cpus.data(), worker_cpus.size());
      if (!available) {
        if (options.worker_config_policy ==
            platform::ThreadConfigPolicy::kRequired) {
          init_status = std::move(available).error().WithContext(
              "scheduler worker CPU discovery");
          return;
        }
        worker_cpus.clear();
      } else {
        worker_cpus.resize(
            std::min<std::size_t>(*available, worker_cpus.size()));
        if (worker_cpus.empty() &&
            options.worker_config_policy ==
                platform::ThreadConfigPolicy::kRequired) {
          init_status = Status::Unavailable(
              "scheduler worker CPU discovery returned no available CPU");
          return;
        }
      }
    }
    for (unsigned i = 0; i < num_workers; ++i) {
      void* storage = allocator.allocate(sizeof(Worker), alignof(Worker));
      if (storage == nullptr) {
        init_status =
            Status::ResourceExhausted("scheduler worker allocation failed");
        return;
      }
      Worker* worker = new (storage) Worker(
          alloc, cap,
          0x9e3779b97f4a7c15ull ^ (static_cast<std::uint64_t>(i) + 1), i);
      workers.push_back(worker);
      ctxs.push_back(Ctx{this, i});
    }
    if (!global.valid() || !inline_jobs.valid()) {
      init_status =
          Status::ResourceExhausted("scheduler global queue allocation failed");
      return;
    }
    for (const auto& worker : workers) {
      if (!worker->local.valid() || !worker->pinned.valid()) {
        init_status =
            Status::ResourceExhausted("scheduler worker queue allocation failed");
        return;
      }
    }
    for (unsigned i = 0; i < num_workers; ++i) {
      platform::ThreadOpts opts;
      if (!worker_cpus.empty())
        opts.affinity = worker_cpus[i % worker_cpus.size()];
      opts.priority = options.worker_priority;
      opts.config_policy = options.worker_config_policy;
      char wname[24];
      std::snprintf(wname, sizeof(wname), "hyle-wkr-%u", i);
      opts.name = wname;  // string_view; spawn() copies it synchronously (make_start)
      auto r = platform::Thread::spawn(&Impl::worker_main, &ctxs[i], opts);
      if (!r.has_value()) {
        init_status = std::move(r).error().WithContext(
            "scheduler worker thread creation");
        stop.store(true, std::memory_order_release);
        for (std::size_t j = 0; j < threads.size(); ++j)
          wake_worker(static_cast<unsigned>(j));
        for (auto& thread : threads) {
          if (thread.joinable()) thread.join();
        }
        threads.clear();
        return;
      }
      threads.push_back(std::move(*r));
    }
  }

  ~Impl() {
    HYLE_FATAL_ASSERT(threads.empty() ||
                      std::all_of(threads.begin(), threads.end(),
                                  [](const platform::Thread& thread) {
                                    return !thread.joinable();
                                  }));
    for (Worker* worker : workers) {
      worker->~Worker();
      allocator.deallocate(worker, sizeof(Worker), alignof(Worker));
    }
  }

  static void worker_main(void* p) noexcept {
    auto* ctx = static_cast<Ctx*>(p);
    ctx->self->run_worker(ctx->idx);
  }

  Fiber* find_task(Worker& w, unsigned self_idx) noexcept {
    // Pinned fibers first: a fiber bound to THIS worker (shard-per-core) must run
    // here and is never stolen. Popped only by this worker; MPMC push from wakers.
    Fiber* pf = nullptr;
    if (w.pinned.try_pop(pf)) return pf;
    if ((pf = w.pinned_overflow.pop()) != nullptr) return pf;
    // Global-queue interval (opportunity ③): every kGlobalQueueInterval-th call,
    // drain the global inject queue BEFORE the LIFO slot / local ring. Without this
    // a worker whose local ring is continuously refilled (yield-loop fiber) never
    // reaches the global check below and starves injected tasks forever (probe-
    // confirmed). Checked AFTER pinned (io latency wins) but BEFORE local, only on
    // the interval tick, so the steady-state hot path is unchanged. tokio next_task.
    ++w.tick;
    if (w.tick % kGlobalQueueInterval == 0) {
      Fiber* gi = nullptr;
      if (global.try_pop(w.pop_cursor, gi)) return gi;
      if ((gi = global_overflow.pop()) != nullptr) return gi;
    }
    // LIFO fast-slot next (opportunity ①): a fiber just woken ON this worker
    // resumes immediately, cache-hot, while under its poll budget. The budget cap
    // (kMaxLifoPolls) stops a hot ping-pong pair from starving the local ring: once
    // spent, demote the slot fiber to the ring and serve FIFO (which resets the
    // budget). Mirrors tokio's lifo_slot + MAX_LIFO_POLLS_PER_TICK.
    // Single-threaded: only this worker touches lifo_slot.
    if (w.lifo_slot != nullptr) {
      if (w.lifo_polls < kMaxLifoPolls) {
        ++w.lifo_polls;
        Fiber* f = w.lifo_slot;
        w.lifo_slot = nullptr;
        return f;
      }
      // Budget spent: demote to the local ring so it stays reachable, then fall
      // through to FIFO. If the ring is full, leave it in the slot (still makes
      // progress -- it'll be taken once the budget resets after a FIFO draw).
      if (w.local.try_push_back(w.lifo_slot)) w.lifo_slot = nullptr;
    }
    Fiber* local = nullptr;
    if (w.local.try_pop_front(local)) {
      w.lifo_polls = 0;  // drawing from FIFO resets the LIFO budget
      return local;
    }
    Fiber* g = nullptr;
    if (global.try_pop(w.pop_cursor, g)) return g;
    if ((g = global_overflow.pop()) != nullptr) return g;
    for (unsigned a = 0; a < n; ++a) {
      unsigned v = static_cast<unsigned>(xorshift(w.rng) % n);
      if (v == self_idx) continue;
      Fiber* stolen = nullptr;
      if (workers[v]->local.try_steal_into(w.local, stolen)) return stolen;
    }
    // Nothing in the ring/global/steal: if a fiber is still stuck in the slot
    // (budget was spent but the ring was full so we couldn't demote), run it now
    // rather than falsely reporting "no work" and parking.
    if (w.lifo_slot != nullptr) {
      Fiber* f = w.lifo_slot;
      w.lifo_slot = nullptr;
      w.lifo_polls = 0;
      return f;
    }
    return nullptr;
  }

  void reschedule(Worker& w, Fiber* f) noexcept {
    // A pinned fiber (shard-per-core) must NOT go on the stealable local queue --
    // a plain cooperative yield would otherwise let another core steal it. Keep it
    // on its owner's private queue. (owner == w.idx here, since a pinned fiber only
    // ever runs on its owner, but route via owner to be explicit + future-proof.)
    if (detail::FiberAccess::owner_worker(f) >= 0) {
      unsigned owner_idx =
          static_cast<unsigned>(detail::FiberAccess::owner_worker(f));
      if (!workers[owner_idx]->pinned.try_push(f))
        workers[owner_idx]->pinned_overflow.push(f);
      return;  // owner is THIS running worker -> it'll pop next loop; no unpark
    }
    if (!w.local.try_push_back(f)) {
      // Spill through every global shard, then an allocation-free intrusive
      // overflow. The current worker must never spin waiting for queues which
      // only workers (including itself) can drain.
      bool queued = false;
      for (std::size_t i = 0; i < global.num_shards(); ++i) {
        if (global.try_push(w.idx + i, f)) {
          queued = true;
          break;
        }
      }
      if (!queued) global_overflow.push(f);
      notify();  // landed in global -> a parked worker may need to take it
    }
  }

  // Wake a parked worker ONLY when no worker is already searching (the §2.3 ②
  // anti-herd gate). The caller has already enqueued the task; the seq_cst fence
  // here pairs with the seq_cst fence a worker runs AFTER leaving the searching
  // state and BEFORE its final pre-park recheck (run_worker). The StoreLoad
  // "Dekker" theorem: across all seq_cst ops in one total order, two threads
  // cannot BOTH read the other's pre-fence value -- so either this notify sees
  // num_searching_ == 0 (and wakes a sleeper) OR the about-to-park worker's
  // recheck sees this newly enqueued task (and doesn't park). No lost wakeup.
  // A freshly-woken worker re-enters searching before this gate matters, so the
  // brief window only ever costs a few EXTRA wakeups, never a missed one.
  void notify() noexcept {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (num_searching_.load(std::memory_order_relaxed) > 0) return;  // a searcher will take it
    unpark_one();
  }

  // Wake worker `idx` whichever way it is sleeping. An idle worker is blocked in
  // EITHER its futex Parker (no io in flight) OR the io poll hook's blocking reap
  // (io in flight, park merge) -- and the waker can't cheaply tell which, and it
  // can race the transition. So hit BOTH channels: parker.unpark() latches a token
  // (harmless + consumed on the worker's next park if it was in reap), and
  // io_wake() writes the ring's wake eventfd (harmless + drained on the next reap
  // if it was in the futex). Idempotent; exactly one of the two actually unblocks.
  void wake_worker(unsigned idx) noexcept {
    workers[idx]->parker.unpark();
    if (IoWakeFn w = io_wake.load(std::memory_order_acquire); w != nullptr)
      w(io_wake_ctx.load(std::memory_order_relaxed), idx);
  }

  // Remove the sleeper at position `pos` in `sleepers` (caller holds idle_mu).
  // O(1) swap-remove: move the last id into the hole and fix ITS stored position,
  // then shrink. Returns the removed worker id. This is the single point that keeps
  // `sleepers` and `sleeper_pos` consistent for both pop paths.
  unsigned remove_at_locked(std::size_t pos) noexcept {
    unsigned idx = sleepers[pos];
    unsigned moved = sleepers.back();
    sleepers[pos] = moved;
    sleeper_pos[moved] = pos;   // the moved id now lives at `pos` (no-op if moved==idx)
    sleepers.pop_back();
    sleeper_pos[idx] = kNotSleeping;
    return idx;
  }

  void unpark_one() noexcept {
    unsigned idx;
    {
      std::lock_guard<std::mutex> lk(idle_mu);
      if (sleepers.empty()) return;
      idx = remove_at_locked(sleepers.size() - 1);  // pop the last-parked (LIFO)
    }
    wake_worker(idx);
  }

  // Wake ONE specific worker (shard-per-core pinned-wake path). Unconditional: the
  // Parker's 3-state machine latches the token, so if the target isn't parked yet
  // its next park() returns at once -- no lost wakeup, no sleepers-list bookkeeping
  // needed. Deliberately BYPASSES the num_searching_ anti-herd gate (notify()): a
  // pinned fiber can run ONLY on this worker, so "some other worker is searching"
  // is irrelevant -- that searcher can't take it. Waking the specific owner is
  // both necessary and sufficient; it never contributes to a wake storm because it
  // targets exactly one worker that has exactly one new task.
  void unpark_specific(unsigned idx) noexcept {
    wake_worker(idx);
  }

  void remove_sleeper(unsigned idx) noexcept {
    std::lock_guard<std::mutex> lk(idle_mu);
    // O(1) via the position index (opportunity ②): no scan. Idempotent -- a
    // notifier's unpark_one may have already popped us (pos == kNotSleeping).
    std::size_t pos = sleeper_pos[idx];
    if (pos != kNotSleeping) remove_at_locked(pos);
  }

  void finish_one() noexcept {
    const std::uint64_t old =
        lifecycle.fetch_sub(1, std::memory_order_acq_rel);
    HYLE_FATAL_ASSERT((old & kOutstandingMask) != 0);
    if ((old & kOutstandingMask) == 1) {
      drain_epoch.fetch_add(1, std::memory_order_release);
      platform::detail::wake_all(&drain_epoch);
    }
  }

  void finish_inline(unsigned accounting_idx) noexcept {
    HYLE_FATAL_ASSERT(accounting_idx < n);
    const std::uint64_t old =
        workers[accounting_idx]->inline_outstanding.fetch_sub(
            1, std::memory_order_acq_rel);
    HYLE_FATAL_ASSERT(old != 0);
    if (old == 1) {
      drain_epoch.fetch_add(1, std::memory_order_release);
      platform::detail::wake_all(&drain_epoch);
    }
  }

  bool has_inline_outstanding() const noexcept {
    for (const auto& worker : workers) {
      if (worker->inline_outstanding.load(std::memory_order_acquire) != 0)
        return true;
    }
    return false;
  }

  bool has_outstanding() const noexcept {
    return (lifecycle.load(std::memory_order_acquire) & kOutstandingMask) != 0 ||
           has_inline_outstanding();
  }

  enum class ScheduleResult : unsigned char { kQueued, kClosed, kFull };

  ScheduleResult schedule_with_hint(InlineJob job, std::size_t hint,
                                    bool on_worker) noexcept {
    if ((lifecycle.load(std::memory_order_acquire) & kAdmissionClosed) != 0)
      return ScheduleResult::kClosed;
    job.accounting_idx = static_cast<std::uint32_t>(
        (n & (n - 1)) == 0 ? hint & (n - 1) : hint % n);
    const std::uint64_t prior_inline =
        workers[job.accounting_idx]->inline_outstanding.fetch_add(
            1, std::memory_order_acq_rel);
    if ((lifecycle.load(std::memory_order_acquire) & kAdmissionClosed) != 0) {
      finish_inline(job.accounting_idx);
      return ScheduleResult::kClosed;
    }
    // Try every shard before applying backpressure. A worker must never spin
    // waiting for capacity in a queue that only workers can drain: with one
    // worker (or all workers recursively submitting), that is a self-deadlock.
    // Foreign producers may wait because workers remain available to consume.
    for (;;) {
      for (std::size_t i = 0; i < inline_jobs.num_shards(); ++i) {
        if (inline_jobs.try_push(hint + i, job)) {
          // Inline callbacks cannot suspend. The first outstanding callback in
          // each accounting shard wakes a worker; once workers have work, they
          // keep draining the queue until it is empty.
          // Avoiding a seq_cst notify fence for every item matters for the
          // common burst shape, especially on ARM. When the count returns to
          // zero, the next burst starts from prior_inline == 0 and wakes again.
          if (prior_inline == 0) notify();
          return ScheduleResult::kQueued;
        }
      }
      if (on_worker) {
        finish_inline(job.accounting_idx);
        return ScheduleResult::kFull;
      }
      platform::this_thread::yield();
    }
  }

  ScheduleResult schedule(InlineJob job) noexcept {
    const bool on_worker = t_current == this && t_worker_idx >= 0;
    const std::size_t hint =
        on_worker
            ? static_cast<std::size_t>(t_worker_idx)
            : inject_rr_.fetch_add(1, std::memory_order_relaxed);
    return schedule_with_hint(job, hint, on_worker);
  }

  ScheduleResult schedule_bulk_sibling(InlineJob job,
                                       unsigned ordinal) noexcept {
    HYLE_FATAL_ASSERT(t_current == this && t_worker_idx >= 0);
    HYLE_FATAL_ASSERT(ordinal + 1 < n);
    const std::size_t root = static_cast<std::size_t>(t_worker_idx);
    const std::size_t hint = (root + ordinal + 1) % n;
    return schedule_with_hint(job, hint, true);
  }

  bool run_inline(Worker& worker) noexcept {
    InlineJob job{};
    if (!inline_jobs.try_pop(worker.inline_cursor, job)) return false;
    job.fn(job.ctx);
    finish_inline(job.accounting_idx);
    return true;
  }

  bool run_inline_batch(Worker& worker) noexcept {
    bool ran = false;
    for (int i = 0; i < kInlineBatch; ++i) {
      if (!run_inline(worker)) break;
      ran = true;
    }
    return ran;
  }

  void dispatch(Worker& w, Fiber* f) noexcept {
    f->resume();
    if (f->done()) {
      // Capture the finish hook BEFORE decrementing outstanding: a self-owned
      // fiber (spawn_task's TaskDriver) registers a hook that frees itself. This
      // is the only safe free point -- control has returned from the fiber, so its
      // stack is no longer executing (unlike freeing from inside the fiber's own
      // entry, which UAFs the fcontext trampoline). Run the hook LAST; after it
      // `f` may be gone, so read nothing more from it.
      detail::FiberAccess::FinishFn fin = detail::FiberAccess::on_finish(f);
      void* fin_ctx = detail::FiberAccess::on_finish_context(f);
      if (fin != nullptr) fin(fin_ctx);
      // Count completion only AFTER the self-owned driver's finish hook has run.
      // Otherwise drain()/~Scheduler may return and destroy stack pools while the
      // last driver is still releasing its Fiber stack into one of those pools.
      finish_one();
    } else if (detail::FiberAccess::consume_wait_request(f)) {
      // The fiber suspended to WAIT on an external event (sched-04). Don't
      // reschedule -- it leaves the run loop until a Waker re-injects it. But a
      // wake() may have already fired (wake-before-suspend): the rendezvous tells
      // us if THIS side must do the re-injection now (wake got here first) or
      // leave it for wake() (we got here first). Re-injection is always post-
      // resume(), so the fiber is fully suspended -- never resumed on two threads.
      if (detail::FiberAccess::wake_rendezvous(f)) reschedule(w, f);
    } else {
      reschedule(w, f);  // ordinary cooperative yield
    }
  }

  // Hand a (suspended) fiber back to the run loop from ANY thread. Goes through
  // the global queue + notify so it's correct whether wake() runs on a worker or
  // a foreign thread. Does NOT touch `outstanding` -- a waiting fiber was never
  // decremented, it stayed counted while parked.
  void reinject(Fiber* f) noexcept {
    // A pinned fiber (shard-per-core) must wake back onto its OWNER worker, never
    // the global stealable pool -- so its io ring + fiber stay on one core. Push
    // to that worker's private queue and unpark IT specifically. The queue is a
    // bounded MPMC ring, so a Waker firing from any thread is safe.
    int owner_idx = detail::FiberAccess::owner_worker(f);
    if (owner_idx >= 0) {
      Worker& owner_worker = *workers[static_cast<unsigned>(owner_idx)];
      if (!owner_worker.pinned.try_push(f))
        owner_worker.pinned_overflow.push(f);
      // Same-core short-circuit: if this wake fires ON the owner worker's own thread
      // (the shard-per-core common case -- an io completion reaped by worker N wakes a
      // fiber pinned to N; run_worker's reap poll runs with t_worker_idx == N), the
      // worker is demonstrably awake (it's running this code), so it'll pop the pinned
      // queue on its next find_task. Skip the unpark -- else we'd fire a cross-thread
      // wake syscall (io_uring/epoll eventfd write, IOCP PostQueuedCompletionStatus)
      // AT OURSELVES every io wake (4/round-trip in echo). Mirrors reschedule's pinned
      // branch, which already relies on this "owner is THIS worker -> no unpark". A
      // foreign-thread wake (t_worker_idx != owner) still unparks the sleeping owner.
      if (!(t_current == this && t_worker_idx == owner_idx)) {
        unpark_specific(static_cast<unsigned>(owner_idx));
      }
      return;
    }
    // Foreign-thread inject (Waker may fire anywhere): round-robin across shards.
    // BUT if this wake is firing ON a worker thread (the common case: a fiber on
    // worker X wakes a peer via Waker::wake, e.g. ping-pong / channel send / RPC
    // reply), drop the woken fiber into THIS worker's LIFO fast-slot instead of the
    // global queue (opportunity ①). It then resumes on THIS core next loop, stack
    // + producer's just-written buffers cache-hot, and we skip both the global-shard
    // CAS and the notify() futex. Any prior slot occupant is evicted to the local
    // ring (never dropped). If the slot path can't take it (ring full on eviction),
    // fall through to the global queue. NOT used for pinned fibers (handled above)
    // nor cooperative yields (those go through reschedule, never here) -- only real
    // wakes, so tokio's "yields never use the LIFO slot" fairness rule is honored.
    if (t_current == this && t_worker_idx >= 0) {
      Worker& w = *workers[static_cast<unsigned>(t_worker_idx)];
      if (w.lifo_slot == nullptr) {
        w.lifo_slot = f;
        return;  // this worker pops it next find_task; no notify (we ARE awake)
      }
      // Slot occupied: evict the older occupant to our local ring, install f.
      if (w.local.try_push_back(w.lifo_slot)) {
        w.lifo_slot = f;
        return;
      }
      // Ring full -> can't evict; fall through to the global queue for f.
    }
    const std::size_t hint =
        inject_rr_.fetch_add(1, std::memory_order_relaxed);
    bool queued = false;
    for (std::size_t i = 0; i < global.num_shards(); ++i) {
      if (global.try_push(hint + i, f)) {
        queued = true;
        break;
      }
    }
    if (!queued) global_overflow.push(f);
    notify();
  }

  // ---- sched-04b timer integration ------------------------------------------

  // Advance THIS worker's queue to now and re-inject every fiber whose sleep
  // deadline passed. Called at the worker loop top; the queue.empty() gate keeps
  // it ~free when nothing is sleeping. Returns the number fired (informational --
  // the caller's find_task re-checks the queues regardless, so the count is not
  // load-bearing).
  std::size_t fire_due_timers(Worker& w) noexcept {
    // Cheap gate: with no sleeping fiber the queue is empty, so this is a single
    // size_t compare -- the "no timer workload pays nothing" property holds even
    // though we call this every loop iteration (needed so a perpetually-busy
    // worker still services its OWN pinned timers; per-worker queues aren't
    // stealable, unlike tokio's single global queue).
    if (w.timers.empty()) return 0;
    time::TimerQueue::EntryList due;
    w.timers.advance(time::monotonic_ns(), due);
    std::size_t n_fired = 0;
    due.consume_all([&](time::TimerEntry* e) noexcept {
      // io-06: a timed-io entry carries a fire callback that ARBITRATES against
      // a racing io completion (it may decline to wake if io already won) -- so the
      // timer side never blindly calls wake_rendezvous when io has already fired,
      // which would leave wake_state_ dirty and lost-wake the fiber's NEXT suspend.
      // A plain sleep entry has no callback: its data is the Fiber* and the timer is
      // its SOLE waker (no race), so wake it directly through the same rendezvous.
      if (e->fire_callback() != nullptr) {
        e->fire_callback()(e);  // timed-io arbiter: sets timer_fired and wakes
                                // only if io hasn't already won
      } else {
        auto* f = static_cast<Fiber*>(e->data());
        if (detail::FiberAccess::wake_rendezvous(f)) reinject(f);
      }
      ++n_fired;
    });
#if defined(HYLE_TRACE) && defined(HYLE_TRACE_TRACY)
    static thread_local std::uint64_t trace_timers_fired = 0;
    trace_timers_fired += n_fired;
    HYLE_TRACE_COUNTER("hyle.sched.timers_fired", trace_timers_fired);
#endif
    return n_fired;
  }

  // Absolute-deadline park bound for this worker: how long park_for should
  // sleep so it wakes for the nearest timer. UINT64_MAX (queue empty) -> caller
  // parks untimed. Clamped at 0 for an already-due timer (park returns at once).
  std::uint64_t park_timeout_ns(Worker& w) noexcept {
    std::uint64_t nd = w.timers.next_deadline_ns();
    if (nd == UINT64_MAX) return UINT64_MAX;
    std::uint64_t now = time::monotonic_ns();
    return nd > now ? nd - now : 0;
  }

  void run_worker(unsigned idx) noexcept {
#if defined(HYLE_TRACE) && defined(HYLE_TRACE_TRACY)
    char trace_thread_name[24];
    std::snprintf(trace_thread_name, sizeof(trace_thread_name),
                  "hyle.worker.%u", idx);
    HYLE_TRACE_THREAD_NAME(trace_thread_name);
#endif
    t_current = this;
    t_worker_idx = static_cast<int>(idx);
    tls_owner = owner;
    tls_worker_idx = static_cast<int>(idx);  // publish for current_worker_index()
    Worker& w = *workers[idx];
    tls_timer_queue = &w.timers;  // publish for sleep_*_ns (this worker only)
    if (on_worker_start != nullptr) on_worker_start(worker_hook_context, idx);

    while (!stop.load(std::memory_order_acquire)) {
      // Fire this worker's due sleep timers FIRST (cheap queue.empty() gate when
      // nothing is sleeping). Placed at the loop top -- not just the park path --
      // so a perpetually-busy worker still services its own pinned timers; a
      // fired timer re-injects its fiber, which the find_task below then runs.
      fire_due_timers(w);
      const bool ran_inline_at_top = run_inline_batch(w);
      Fiber* f = find_task(w, idx);
      if (f != nullptr) {
        dispatch(w, f);
        continue;
      }
      if (ran_inline_at_top) continue;
      // Run-queue DRAINED -> now reap io completions (io-04). Reaping HERE (on
      // drain) rather than every loop iteration is what lets submit batching work:
      // while the run queue had fibers, each ran and queued its SQE without a
      // reap in between, so many SQEs accumulate; this drain-time reap flushes
      // them in one enter and harvests the batch of completions, which re-inject
      // the next wave of fibers. (tokio's tick model: run a batch of tasks, then
      // service io -- not io between every task.) Null hook (no io) -> skipped,
      // zero cost. A reaped completion re-injects a fiber, so re-check find_task.
      if (IoPollFn poll = io_poll.load(std::memory_order_acquire); poll != nullptr) {
        poll(io_poll_ctx.load(std::memory_order_relaxed), idx, 0);
        if (run_inline_batch(w)) continue;
        f = find_task(w, idx);
        if (f != nullptr) {
          dispatch(w, f);
          continue;
        }
      }
      // No work after draining + reaping -> become a SEARCHER. While
      // num_searching_ > 0 producers skip waking parked workers (notify() gate),
      // so a spawn burst no longer triggers a futex per enqueue -- THIS searcher
      // absorbs them by finding the work during its spin. The spin also just
      // bridges the brief producer gap before the (now rarely needed) park.
      num_searching_.fetch_add(1, std::memory_order_relaxed);
      bool got = false;
      bool ran_inline = false;
      for (int spin = 0; spin < spin_polls && !got; ++spin) {
        util::cpu_relax();
        if (run_inline(w)) {
          ran_inline = true;
          got = true;
        } else {
          f = find_task(w, idx);
          if (f != nullptr) got = true;
        }
      }
      for (int y = 0; y < yields_before_park && !got; ++y) {
        platform::this_thread::yield();
        if (run_inline(w)) {
          ran_inline = true;
          got = true;
        } else {
          f = find_task(w, idx);
          if (f != nullptr) got = true;
        }
      }
      if (got) {
        // Leaving searching with work in hand. If we were the LAST searcher,
        // baton one parked worker to keep a searcher warm (tokio
        // transition_from_searching) so the next burst item is caught without a
        // cold park->unpark round-trip. fetch_sub returns the PRE value.
        if (num_searching_.fetch_sub(1, std::memory_order_relaxed) == 1) {
          unpark_one();
        }
        if (!ran_inline) dispatch(w, f);
        continue;
      }
      // Give up: register as a sleeper, THEN leave searching with a seq_cst RMW,
      // THEN recheck once. Order is load-bearing: the seq_cst fetch_sub pairs
      // with notify()'s seq_cst fence (Dekker) so a producer either sees
      // num_searching_ == 0 and wakes us (we are already registered) or our
      // recheck sees its task. Registering BEFORE the fetch_sub guarantees a
      // notify that observes 0 finds us in `sleepers`.
      {
        std::lock_guard<std::mutex> lk(idle_mu);
        sleeper_pos[idx] = sleepers.size();  // my slot = the tail I'm about to push
        sleepers.push_back(idx);
      }
      num_searching_.fetch_sub(1, std::memory_order_seq_cst);  // Dekker store+fence
      if (run_inline(w)) {
        remove_sleeper(idx);
        continue;
      }
      f = find_task(w, idx);                                   // Dekker recheck
      if (f != nullptr) {
        remove_sleeper(idx);
        dispatch(w, f);
        continue;
      }
      // Park until unparked OR the nearest sleep timer is due. An empty queue
      // gives UINT64_MAX -> untimed park (exactly the pre-04b behaviour, zero
      // cost when no fiber is sleeping). A pending timer bounds the park so the
      // worker wakes to fire it; park_for returning false (timeout) is fine
      // -- the loop top's fire_due_timers handles the expiry uniformly.
      std::uint64_t timeout = park_timeout_ns(w);
      if (IoPollFn poll = io_poll.load(std::memory_order_acquire);
          poll != nullptr) {
        // io registered: at the park boundary, hand the block to the PROACTOR
        // when ops are in flight -- a GETEVENTS io_uring_enter -- instead of the
        // futex parker. This is a CORRECTNESS requirement, not a latency tweak:
        // io_uring surfaces many completions only when the owning thread makes a
        // GETEVENTS enter (that enter runs the kernel's completion task-work). A
        // futex park + the drain-path non-blocking reap(0) never enters with
        // GETEVENTS, so once the run queue drains with ops in flight (notably
        // >ring-depth in flight) those completions never surface and their fibers
        // hang forever -- the worker just spins reap(0)==0 while drain() waits on
        // outstanding!=0 (root-caused 2026-07-03 via a blocking-reap probe that
        // made a 300-stream pipelined bench complete).
        //
        // The hook RETURNS whether ops were in flight. true -> it BLOCKED in the
        // ring (up to `timeout`) harvesting completions; loop to run whatever they
        // re-injected. false -> nothing in flight, so it did only a cheap poll;
        // fall through to the futex park (a cross-thread unpark interrupts it
        // immediately) -- so a scheduler that owns an IoContext but currently has
        // no io pending still wakes instantly on a foreign spawn/wake.
        //
        // PARK MERGE (io-04 perf): the block uses the FULL timeout (the timer
        // deadline, or INFINITE when no timer) -- NOT a 1ms re-poll cap. A truly
        // asleep worker no longer burns ~1000 io_uring_enter/sec busy-polling
        // (measured: 421 enters over 500ms idle before this). A foreign spawn/
        // steal breaks the block promptly via Scheduler::wake_worker -> the ring's
        // wake eventfd (Proactor::wake), so responsiveness is preserved without the
        // poll. This is why the integration wake hook is mandatory once the cap
        // is gone.
        if (poll(io_poll_ctx.load(std::memory_order_relaxed), idx, timeout)) {
          remove_sleeper(idx);
          continue;  // io was in flight -> serviced in the ring, re-check queues
        }
        // else: nothing in flight -> park below for an immediate cross-thread wake
      }
      if (timeout == UINT64_MAX) {
        w.parker.park();
      } else {
        w.parker.park_for(time::Duration(timeout));
      }
      remove_sleeper(idx);  // idempotent: a notifier may have popped us already
    }

    if (on_worker_stop != nullptr) on_worker_stop(worker_hook_context, idx);
    t_current = nullptr;
    t_worker_idx = -1;
    tls_timer_queue = nullptr;
    tls_worker_idx = -1;
    tls_owner = nullptr;
  }

  bool try_admit() noexcept {
    std::uint64_t state = lifecycle.load(std::memory_order_acquire);
    for (;;) {
      if ((state & kAdmissionClosed) != 0) return false;
      HYLE_FATAL_ASSERT((state & kOutstandingMask) != kOutstandingMask);
      if (lifecycle.compare_exchange_weak(state, state + 1,
                                          std::memory_order_acq_rel,
                                          std::memory_order_acquire))
        return true;
    }
  }

  bool spawn(Fiber* f) noexcept {
    HYLE_FATAL_ASSERT(f != nullptr);
    if (!try_admit()) return false;
    const bool on_worker = t_current == this && t_worker_idx >= 0;
    if (on_worker) {
      if (workers[static_cast<unsigned>(t_worker_idx)]->local.try_push_back(f)) {
        notify();
        return true;
      }
    }
    const std::size_t hint =
        on_worker ? static_cast<std::size_t>(t_worker_idx)
                  : inject_rr_.fetch_add(1, std::memory_order_relaxed);
    for (;;) {
      for (std::size_t i = 0; i < global.num_shards(); ++i) {
        if (global.try_push(hint + i, f)) {
          notify();
          return true;
        }
      }
      // A worker cannot wait for queues that only workers drain. In the
      // single-worker case this would deadlock immediately; multiple workers
      // can deadlock cyclically while recursively submitting to one another.
      if (on_worker) {
        finish_one();
        return false;
      }
      platform::this_thread::yield();
    }
  }

  // Spawn a fiber PINNED to worker `idx` (shard-per-core). It records its owner so
  // every subsequent wake also routes back here (reinject), then goes onto that
  // worker's private queue and unparks it specifically. Never stolen.
  bool spawn_on(unsigned idx, Fiber* f) noexcept {
    HYLE_FATAL_ASSERT(idx < n);
    HYLE_FATAL_ASSERT(f != nullptr);
    if (!try_admit()) return false;
    const int previous_owner = detail::FiberAccess::owner_worker(f);
    detail::FiberAccess::set_owner_worker(f, static_cast<int>(idx));
    while (!workers[idx]->pinned.try_push(f)) {
      if (t_current == this && t_worker_idx >= 0) {
        detail::FiberAccess::set_owner_worker(f, previous_owner);
        finish_one();
        return false;
      }
      platform::this_thread::yield();
    }
    unpark_specific(idx);
    return true;
  }

  void drain() noexcept {
    HYLE_FATAL_ASSERT(t_current != this &&
                      "drain from this scheduler's worker can deadlock");
    while (has_outstanding()) {
      const std::uint32_t epoch = drain_epoch.load(std::memory_order_acquire);
      if (!has_outstanding()) break;
      platform::detail::wait_on_address(&drain_epoch, epoch);
    }
  }

  // sched-08 phase 1: stop accepting new work + broadcast the signal so long-lived
  // fibers (which poll shutdown_requested()) can wind down. Waking every worker
  // (both channels) matters because a parked/reaping worker must re-check its state
  // and let its pinned long-lived fibers observe the flag. Idempotent.
  void begin_shutdown() noexcept {
    lifecycle.fetch_or(kAdmissionClosed, std::memory_order_acq_rel);
    ShutdownWaiter* waiters = nullptr;
    {
      std::lock_guard<std::mutex> lock(shutdown_mu);
      waiters = shutdown_waiters;
      shutdown_waiters = nullptr;
      for (ShutdownWaiter* p = waiters; p != nullptr; p = p->next)
        p->linked = false;
    }
    // Copy next before wake: wake may resume the owner and destroy its
    // stack-resident waiter immediately.
    while (waiters != nullptr) {
      ShutdownWaiter* next = waiters->next;
      const Waker waker = waiters->waker;
      waker.wake();
      waiters = next;
    }
    for (unsigned i = 0; i < n; ++i) wake_worker(i);
  }

  void wait_for_shutdown() noexcept {
    HYLE_FATAL_ASSERT(t_current == this && Fiber::current() != nullptr);
    ShutdownWaiter waiter;
    waiter.waker = current_waker();
    {
      std::lock_guard<std::mutex> lock(shutdown_mu);
      if ((lifecycle.load(std::memory_order_acquire) & kAdmissionClosed) != 0)
        return;
      waiter.next = shutdown_waiters;
      waiter.linked = true;
      shutdown_waiters = &waiter;
    }
    suspend();  // begin_shutdown owns the sole wake for this wait episode
    // Defensive unlink if a future alternate wake source is added.
    std::lock_guard<std::mutex> lock(shutdown_mu);
    if (!waiter.linked) return;
    ShutdownWaiter** link = &shutdown_waiters;
    while (*link != nullptr && *link != &waiter) link = &(*link)->next;
    HYLE_FATAL_ASSERT(*link == &waiter);
    *link = waiter.next;
    waiter.linked = false;
  }

  // Drain ordinary finite work while admission remains open, then atomically
  // close only when the count is zero. This preserves the useful shutdown()
  // contract that an already-running task may spawn descendants while draining.
  // A foreign producer racing the zero boundary is linearized by the same CAS:
  // either it increments first and we wait for it, or close wins and it rejects.
  void drain_and_close() noexcept {
    HYLE_FATAL_ASSERT(t_current != this &&
                      "shutdown from this scheduler's worker can deadlock");
    for (;;) {
      std::uint64_t state = lifecycle.load(std::memory_order_acquire);
      if ((state & kAdmissionClosed) != 0) {
        drain();
        return;
      }
      if ((state & kOutstandingMask) == 0 &&
          !has_inline_outstanding()) {
        if (lifecycle.compare_exchange_weak(
                state, kAdmissionClosed, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
          // An inline submitter may have incremented after our zero observation
          // but before close. Its mandatory recheck either rolls it back or it
          // was already admitted; drain catches both cases before return.
          drain();
          return;
        }
        continue;
      }
      const std::uint32_t epoch = drain_epoch.load(std::memory_order_acquire);
      if (has_outstanding())
        platform::detail::wait_on_address(&drain_epoch, epoch);
    }
  }

  // sched-08 phase 2: bounded wait for outstanding to hit zero. Unbounded waiting
  // remains the distinctly named drain() operation.
  bool drain_for(time::Duration timeout) noexcept {
    return drain_until(time::deadline_after(timeout));
  }

  bool drain_until(time::MonotonicTimePoint deadline) noexcept {
    HYLE_FATAL_ASSERT(t_current != this &&
                      "bounded drain from this scheduler's worker can deadlock");
    while (has_outstanding()) {
      const std::uint32_t epoch = drain_epoch.load(std::memory_order_acquire);
      if (!has_outstanding()) return true;
      const time::Duration remaining = time::remaining(deadline);
      if (remaining <= time::Duration::zero()) return false;
      platform::detail::wait_on_address_for(
          &drain_epoch, epoch, time::duration_ns_saturated(remaining));
    }
    return true;
  }

  void shutdown() {
    HYLE_FATAL_ASSERT(t_current != this &&
                      "shutdown from this scheduler's worker can deadlock");
    {
      std::unique_lock<std::mutex> lock(stop_mu);
      if (shutdown_started) {
        stop_cv.wait(lock, [&] { return shutdown_complete; });
        return;
      }
      shutdown_started = true;
    }
    // Ordinary shutdown drains finite task trees before closing. Call
    // begin_shutdown() explicitly for a server/long-lived workload that must
    // reject new work immediately and broadcast shutdown_requested().
    drain_and_close();
    stop.store(true, std::memory_order_release);
    // Wake every worker to observe `stop` -- via BOTH channels (wake_worker), since
    // an io-active worker is blocked in the reap, not the futex parker.
    for (unsigned i = 0; i < n; ++i) wake_worker(i);
    for (auto& t : threads) {
      if (t.joinable()) t.join();
    }
    // Every fiber has finished and no release() can race now. Return retained
    // guarded mappings at this real owner-level reclamation point instead of
    // keeping up to three full cache tiers until Scheduler destruction.
    pool_small.clear();
    pool_default.clear();
    pool_large.clear();
    {
      std::lock_guard<std::mutex> lock(stop_mu);
      shutdown_complete = true;
    }
    stop_cv.notify_all();
  }

  static thread_local Impl* t_current;
  static thread_local int t_worker_idx;
};

thread_local Scheduler::Impl* Scheduler::Impl::t_current = nullptr;
thread_local int Scheduler::Impl::t_worker_idx = -1;

Scheduler::Scheduler(unsigned num_workers, mem::Allocator alloc,
                     std::size_t queue_capacity)
    : impl_(nullptr) {
  Status status;
  init_(num_workers, alloc, queue_capacity, {}, &status);
  HYLE_FATAL_ASSERT(status.ok() &&
                    "scheduler construction failed; use the Status overload "
                    "when resource failure is recoverable");
}

Scheduler::Scheduler(unsigned num_workers, mem::Allocator alloc,
                     std::size_t queue_capacity, SchedulerOptions options)
    : impl_(nullptr) {
  Status status;
  init_(num_workers, alloc, queue_capacity, options, &status);
  HYLE_FATAL_ASSERT(status.ok() &&
                    "scheduler construction failed; use the Status overload "
                    "when resource failure is recoverable");
}

Scheduler::Scheduler(unsigned num_workers, mem::Allocator alloc,
                     std::size_t queue_capacity, Status* status) noexcept
    : impl_(nullptr) {
  HYLE_FATAL_ASSERT(status != nullptr);
  init_(num_workers, alloc, queue_capacity, {}, status);
}

Scheduler::Scheduler(unsigned num_workers, mem::Allocator alloc,
                     std::size_t queue_capacity, SchedulerOptions options,
                     Status* status) noexcept
    : impl_(nullptr) {
  HYLE_FATAL_ASSERT(status != nullptr);
  init_(num_workers, alloc, queue_capacity, options, status);
}

void Scheduler::init_(unsigned num_workers, mem::Allocator alloc,
                      std::size_t queue_capacity, SchedulerOptions options,
                      Status* status) noexcept {
  *status = Status{};
  if (!alloc.valid()) {
    *status = Status::InvalidArgument(
        "scheduler requires a complete allocator");
    return;
  }
  if (num_workers == 0 || num_workers > 1024) {
    *status = Status::InvalidArgument(
        "scheduler worker count must be in [1, 1024]");
    return;
  }
  if (options.worker_cpu_order.has_value() &&
      *options.worker_cpu_order != platform::CpuOrder::kOsDefault &&
      *options.worker_cpu_order != platform::CpuOrder::kPerformanceFirst &&
      *options.worker_cpu_order != platform::CpuOrder::kEfficiencyFirst) {
    *status = Status::InvalidArgument("scheduler worker CPU order is invalid");
    return;
  }
  if (options.worker_cpu_order.has_value() &&
      !options.worker_cpu_set.empty()) {
    *status = Status::InvalidArgument(
        "scheduler accepts either a worker CPU order or an explicit CPU set");
    return;
  }
  if (options.worker_config_policy !=
          platform::ThreadConfigPolicy::kBestEffort &&
      options.worker_config_policy !=
          platform::ThreadConfigPolicy::kRequired) {
    *status = Status::InvalidArgument(
        "scheduler worker configuration policy is invalid");
    return;
  }
  if (options.worker_priority &&
      (*options.worker_priority < platform::ThreadPriority::kLowest ||
       *options.worker_priority > platform::ThreadPriority::kHighest)) {
    *status = Status::InvalidArgument("scheduler worker priority is invalid");
    return;
  }
  if (options.idle_policy != SchedulerIdlePolicy::kBalanced &&
      options.idle_policy != SchedulerIdlePolicy::kLowLatency &&
      options.idle_policy != SchedulerIdlePolicy::kPowerSaving) {
    *status = Status::InvalidArgument("scheduler idle policy is invalid");
    return;
  }
  if (queue_capacity < 2 || queue_capacity > (std::size_t{1} << 31) ||
      !std::has_single_bit(queue_capacity)) {
    *status = Status::InvalidArgument(
        "scheduler queue capacity must be a power of two in [2, 2^31]");
    return;
  }
  try {
    void* storage = alloc.allocate(sizeof(Impl), alignof(Impl));
    if (storage == nullptr) {
      *status = Status::ResourceExhausted(
          "scheduler implementation allocation failed");
      return;
    }
    Impl* candidate = nullptr;
    try {
      candidate =
          new (storage) Impl(num_workers, alloc, queue_capacity, options, this);
    } catch (...) {
      alloc.deallocate(storage, sizeof(Impl), alignof(Impl));
      throw;
    }
    if (!candidate->init_status.ok()) {
      *status = candidate->init_status;
      candidate->~Impl();
      alloc.deallocate(candidate, sizeof(Impl), alignof(Impl));
      return;
    }
    impl_ = candidate;
  } catch (const std::bad_alloc&) {
    *status =
        Status::ResourceExhausted("scheduler bookkeeping allocation failed");
  } catch (const std::length_error&) {
    *status = Status::InvalidArgument("scheduler size exceeds platform limits");
  }
}

Scheduler::~Scheduler() {
  if (impl_ == nullptr) return;
  impl_->shutdown();
  const mem::Allocator allocator = impl_->allocator;
  impl_->~Impl();
  allocator.deallocate(impl_, sizeof(Impl), alignof(Impl));
}

bool Scheduler::spawn(Fiber* fiber) { return impl_->spawn(fiber); }
Status Scheduler::spawn_detached(DetachedFn fn, void* ctx,
                                 std::size_t stack_size) noexcept {
  return spawn_detached_(0, false, fn, ctx, stack_size);
}
Status Scheduler::schedule(DetachedFn fn, void* ctx) noexcept {
  if (fn == nullptr)
    return Status::InvalidArgument("scheduled task callback is null");
  const auto submitted = impl_->schedule(InlineJob{fn, ctx});
  if (submitted != Impl::ScheduleResult::kQueued) {
    if (submitted == Impl::ScheduleResult::kClosed)
      return Status::Cancelled("scheduler admission is closed");
    return Status::ResourceExhausted(
        "scheduler inline queue is full on a worker");
  }
  return Status{};
}

Status detail::SchedulerAccess::schedule_bulk_sibling(
    Scheduler& scheduler, unsigned ordinal, Scheduler::DetachedFn fn,
    void* context) noexcept {
  if (fn == nullptr)
    return Status::InvalidArgument("bulk sibling callback is null");
  const auto submitted =
      scheduler.impl_->schedule_bulk_sibling(InlineJob{fn, context}, ordinal);
  if (submitted != Scheduler::Impl::ScheduleResult::kQueued) {
    if (submitted == Scheduler::Impl::ScheduleResult::kClosed)
      return Status::Cancelled("scheduler admission is closed");
    return Status::ResourceExhausted(
        "scheduler inline queue is full during bulk admission");
  }
  return Status{};
}

void* Scheduler::allocate_owned_callable_(std::size_t size,
                                          std::size_t alignment) noexcept {
  return impl_ != nullptr ? impl_->allocator.allocate(size, alignment) : nullptr;
}

void Scheduler::deallocate_owned_callable_(void* pointer, std::size_t size,
                                           std::size_t alignment) noexcept {
  HYLE_FATAL_ASSERT(impl_ != nullptr);
  impl_->allocator.deallocate(pointer, size, alignment);
}

bool Scheduler::spawn_on(unsigned worker_idx, Fiber* fiber) {
  return impl_->spawn_on(worker_idx, fiber);
}
Status Scheduler::spawn_detached_on(unsigned worker_idx, DetachedFn fn,
                                    void* ctx,
                                    std::size_t stack_size) noexcept {
  return spawn_detached_(worker_idx, true, fn, ctx, stack_size);
}

Status Scheduler::spawn_detached_(unsigned worker_idx, bool pinned,
                                  DetachedFn fn, void* ctx,
                                  std::size_t stack_size) noexcept {
  if (fn == nullptr)
    return Status::InvalidArgument("detached task callback is null");
  if (stack_size < mem::kFiberStackSmall)
    return Status::InvalidArgument(
        "detached task stack is smaller than the safe tier");
  if (pinned && worker_idx >= impl_->n)
    return Status::InvalidArgument("detached task worker is out of range");

  mem::FiberStackPool* pool = impl_->fiber_stack_pool(stack_size);
  auto stack_result =
      pool != nullptr ? pool->acquire() : mem::FiberStack::allocate(stack_size);
  if (!stack_result) {
    return std::move(stack_result).error().WithContext(
        "detached task stack allocation");
  }
  mem::FiberStack stack = *std::move(stack_result);

  void* storage =
      impl_->allocator.allocate(sizeof(DetachedDriver), alignof(DetachedDriver));
  if (storage == nullptr) {
    if (pool != nullptr) {
      pool->release(std::move(stack));
    }
    return Status::ResourceExhausted("detached task record allocation failed");
  }
  auto* driver = new (storage)
      DetachedDriver(impl_->allocator, fn, ctx, std::move(stack), pool);
  const bool admitted =
      pinned ? impl_->spawn_on(worker_idx, &driver->fiber)
             : impl_->spawn(&driver->fiber);
  if (!admitted) {
    DetachedDriver::destroy(driver);
    return impl_->lifecycle.load(std::memory_order_acquire) &
                   Impl::kAdmissionClosed
               ? Status::Cancelled("scheduler admission is closed")
               : Status::ResourceExhausted(
                     "scheduler runnable queues are full on a worker");
  }
  return Status{};
}
unsigned Scheduler::num_workers() const noexcept { return impl_->n; }
mem::FiberStackPool* Scheduler::fiber_stack_pool(std::size_t usable_bytes) noexcept {
  return impl_->fiber_stack_pool(usable_bytes);
}
void Scheduler::drain() { impl_->drain(); }
void Scheduler::shutdown() { impl_->shutdown(); }
bool Scheduler::accepting() const noexcept {
  return (impl_->lifecycle.load(std::memory_order_acquire) &
          Impl::kAdmissionClosed) == 0;
}
void Scheduler::begin_shutdown() noexcept { impl_->begin_shutdown(); }
void Scheduler::wait_for_shutdown() noexcept { impl_->wait_for_shutdown(); }
bool Scheduler::drain_for(time::Duration timeout) noexcept {
  return impl_->drain_for(timeout);
}

bool Scheduler::drain_until(time::MonotonicTimePoint deadline) noexcept {
  return impl_->drain_until(deadline);
}

void Scheduler::set_io_poll_(IoPollFn fn, void* ctx) noexcept {
  // Publish ctx BEFORE fn (release) so a worker that reads a non-null fn is
  // guaranteed to see the matching ctx. Clearing (fn=null) is the reverse but
  // benign: a worker reading the stale non-null fn with a now-null ctx still runs
  // a valid (pre-teardown) hook; unregister happens only after the scheduler is
  // stopped (IoContext dtor), so no worker is looping then anyway.
  impl_->io_poll_ctx.store(ctx, std::memory_order_relaxed);
  impl_->io_poll.store(fn, std::memory_order_release);
}

void Scheduler::set_io_wake_(IoWakeFn fn, void* ctx) noexcept {
  // Same publish discipline as set_io_poll_ (ctx before fn, release).
  impl_->io_wake_ctx.store(ctx, std::memory_order_relaxed);
  impl_->io_wake.store(fn, std::memory_order_release);
}

// ---- completion spine (sched-04a) -------------------------------------------

void Waker::wake() const noexcept {
  // Rendezvous: whichever of {worker dispatch, this wake()} sees the other has
  // already arrived performs the single re-injection. If we arrived first, the
  // worker's dispatch will re-inject post-suspend.
  if (detail::FiberAccess::wake_rendezvous(fiber_))
    sched_->impl_->reinject(fiber_);
}

Waker current_waker() noexcept {
  return Waker{tls_owner, Fiber::current()};  // both null if not on a worker fiber
}

int current_worker_index() noexcept { return tls_worker_idx; }

bool detail::SchedulerAccess::is_current_scheduler(
    const Scheduler* scheduler) noexcept {
  return tls_owner == scheduler;
}

// sched-08: a long-lived fiber polls this to learn its scheduler has begun graceful
// shutdown, so it can break its loop. tls_owner is the Scheduler running this fiber
// (null off a worker -> not shutting down from this call's view). One relaxed-ish
// load through the public accepting() (acquire); cheap enough for a loop-top check.
bool shutdown_requested() noexcept {
  return tls_owner != nullptr && !tls_owner->accepting();
}

void wait_for_shutdown() noexcept {
  HYLE_FATAL_ASSERT(tls_owner != nullptr && Fiber::current() != nullptr);
  tls_owner->wait_for_shutdown();
}

void suspend() noexcept {
  Fiber* self = Fiber::current();
  HYLE_FATAL_ASSERT(self != nullptr);  // suspend() outside a fiber is a bug
  detail::FiberAccess::request_wait(
      self);  // tell dispatch: don't reschedule me, I'm waiting
  Fiber::yield();        // control returns to the worker; resumes when re-injected
}

// ---- fiber sleep (sched-04b) ------------------------------------------------

void sleep_until(time::MonotonicTimePoint deadline) noexcept {
  Fiber* self = Fiber::current();
  HYLE_FATAL_ASSERT(self != nullptr);  // sleep outside a fiber is a bug
  HYLE_FATAL_ASSERT(tls_timer_queue != nullptr);  // must be on a scheduler worker

  // The timer entry lives on THIS fiber's stack. Safe: the fiber suspends right
  // below and its stack is frozen until the timer (its sole waker) re-injects
  // it, so the entry outlives its time in the queue. data = the fiber to wake.
  // O(1) onto this worker's own queue (single-threaded: we ARE that worker).
  time::TimerEntry entry;
  entry.set_data(self);
  tls_timer_queue->schedule(&entry, deadline);

  // Suspend on the SAME spine as an io completion: request_wait + yield, and the
  // worker's dispatch rendezvous with the eventual timer fire (fire_due_timers
  // calls wake_rendezvous, exactly like Waker::wake). No competing waker exists
  // for a plain sleep, so the timer always fires and the entry always leaves the
  // queue -- no cancel needed (that's the fiber-mode wait_for/wait_until case,
  // sched-04d).
  detail::FiberAccess::request_wait(self);
  Fiber::yield();
}

void sleep_for(time::Duration duration) noexcept {
  sleep_until(time::deadline_after(duration));
}

// ---- timed-io building blocks (io-06) ---------------------------------------
// arm/cancel a timeout timer on THIS worker's queue WITHOUT suspending, so a
// timed io op can arm a deadline + submit an io op and then suspend once. Same
// single-threaded per-worker discipline as sleep (we ARE the owning worker); the
// timer wakes this fiber via the SAME reinject path (entry data = self, fired by
// fire_due_timers -> wake_rendezvous). No cross-core cancel queue: the timed io
// fiber is pinned (shard-per-core), so arm/fire/cancel are all on this worker.

void arm_timeout(time::TimerEntry* entry,
                 time::MonotonicTimePoint deadline) noexcept {
  Fiber* self = Fiber::current();
  HYLE_FATAL_ASSERT(self != nullptr);       // timed io outside a fiber is a bug
  HYLE_FATAL_ASSERT(tls_timer_queue != nullptr);  // must be on a scheduler worker
  if (entry->fire_callback() == nullptr) {
    entry->set_data(self);  // default: direct reinject, same as sleep
  } else {
    // Callback timers own their payload. It must be initialized BEFORE arm so a
    // sub-tick deadline can never expose a half-built callback record.
    HYLE_FATAL_ASSERT(entry->data() != nullptr);
  }
  tls_timer_queue->schedule(entry, deadline);
}

void cancel_timeout(time::TimerEntry* entry) noexcept {
  HYLE_FATAL_ASSERT(tls_timer_queue != nullptr);
  (void)tls_timer_queue->cancel(entry);  // O(1); no-op if already fired
}

}  // namespace hyle::sched
