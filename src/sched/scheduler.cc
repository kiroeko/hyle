#include "hyle/sched/scheduler.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <vector>

#include "hyle/container/bounded_queue.h"
#include "hyle/container/sharded_queue.h"
#include "hyle/container/ws_fifo_queue.h"
#include "hyle/macros.h"
#include "hyle/mem/fiber_stack.h"
#include "hyle/platform/thread.h"
#include "hyle/platform/wait.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/timer.h"
#include "hyle/sched/waker.h"
#include "hyle/time/clock.h"
#include "hyle/time/timer_wheel.h"
#include "hyle/util/spin_lock.h"  // util::cpu_relax (pause hint)

namespace hyle::sched {
namespace {

// The Scheduler running on this worker thread, for current_waker(). File-local
// (not an Impl member) so the free functions below can read it without naming
// the private nested Impl type.
thread_local Scheduler* tls_owner = nullptr;

// The current worker's timing wheel, published by run_worker (file-local so the
// sleep_*_ns free functions can arm a timer without naming the private nested
// Impl::Worker type). Null when not on a worker fiber. Single-threaded access:
// only the owning worker thread reads/writes its own wheel (arm happens inside a
// fiber running ON that worker; fire happens in that worker's loop).
thread_local time::TimerWheel* tls_wheel = nullptr;

// The current worker's index, published by run_worker (file-local so the
// current_worker_index() free function can read it without naming the private
// nested Impl). -1 when not on a worker fiber. io/ (plan A) reads this to pick
// this worker's per-core proactor for submit, so submit + reap share a thread.
thread_local int tls_worker_idx = -1;

// Per-worker xorshift64 for cheap random victim selection.
std::uint64_t xorshift(std::uint64_t& s) noexcept {
  s ^= s << 13;
  s ^= s >> 7;
  s ^= s << 17;
  return s;
}

// Idle backoff while SEARCHING. The §2.3 ②③ anti-herd gate (num_searching_ +
// notify() seq_cst Dekker, see notify()/run_worker) is what kills the spawn-burst
// futex storm at the source: a producer skips waking parked workers while a
// searcher exists. This spin then just keeps the searcher warm to catch the next
// item before the (now rare) park, and bridges sub-µs producer gaps. cpu_relax,
// no syscall. (Prior comment said §2.3 ②③ were deferred -- they are now done.)
constexpr int kSpinPolls = 4096;       // cpu_relax polls (sub-µs, no syscall)
constexpr int kYieldsBeforePark = 4;   // then cede the core a few times

// Cap on consecutive LIFO-slot polls before the slot's fiber is demoted to the
// local FIFO ring, so a hot ping-pong / RPC pair can't monopolize the worker and
// starve its ring (tokio MAX_LIFO_POLLS_PER_TICK = 3; the sched-02 RunQueue uses
// the same value). See the lifo_slot field + find_task for the mechanism.
constexpr int kMaxLifoPolls = 3;

// Every Nth find_task, check the GLOBAL inject queue BEFORE the local ring, so a
// worker whose local queue is continuously refilled (e.g. a fiber that yields in a
// loop -> reschedule -> local.push_back) cannot starve tasks injected to the global
// queue indefinitely (complexity opportunity ③; starvation CONFIRMED by a probe:
// a 1-worker yield-hog did 17M yields while a globally-injected victim never ran in
// 500ms). tokio's global_queue_interval, default 61 -- a prime, to avoid harmonic
// resonance with power-of-two batch/queue sizes. Fixed (not EWMA-tuned): the tuning
// only matters for very-long-poll workloads, and a fixed prime already closes the
// starvation hole, which is the correctness point here.
constexpr std::uint32_t kGlobalQueueInterval = 61;


}  // namespace

struct Scheduler::Impl {
  struct Worker {
    container::WsFifoQueue<Fiber*, mem::Allocator> local;
    platform::Parker parker;  // non-movable -> Worker lives in a unique_ptr
    // Per-worker timing wheel (sched-04b). Owned and touched ONLY by this worker
    // thread on its park cold path -> no lock needed. Sleeping fibers arm here;
    // the worker advances it before parking and fires due timers into the run
    // loop. Sharding the wheel per-worker (vs one central) is the sched-side
    // choice the time-02 doc left open; per-worker keeps timer + woken fiber on
    // the same core (cache-hot re-inject) and needs no wheel lock. Epoch = the
    // monotonic ns captured at construction.
    time::TimerWheel wheel;
    // Private queue for fibers PINNED to this worker (shard-per-core, io-04). A
    // bounded MPMC ring: any thread (a Waker firing on another core, or spawn_on)
    // pushes here, but ONLY this worker pops it, and the work-stealing path never
    // touches it -- so a pinned fiber can never be stolen onto another core. Kept
    // separate from `local` (which IS stealable) precisely so stealing stays
    // untouched (no change to the anti-herd Dekker/CAS math). Checked before
    // `local` in find_task so a woken connection fiber resumes promptly.
    container::BoundedQueue<Fiber*, mem::Allocator> pinned;
    std::uint64_t rng;
    unsigned idx;             // own worker index: shard hint + pop-cursor seed
    std::size_t pop_cursor;   // round-robin cursor into the global ShardedQueue
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
    Worker(mem::Allocator a, std::size_t cap, std::uint64_t seed, unsigned i)
        : local(a, cap),
          wheel(time::monotonic_ns()),
          pinned(a, cap),
          rng(seed),
          idx(i),
          pop_cursor(i) {}
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
  std::vector<std::unique_ptr<Worker>> workers;
  std::vector<Ctx> ctxs;
  std::vector<platform::Thread> threads;
  std::atomic<bool> stop{false};
  std::atomic<std::int64_t> outstanding{0};
  // sched-08 graceful shutdown, phase-1 flag. True normally; begin_shutdown() flips
  // it false to REJECT new spawns and signal long-lived fibers to wind down. Flat
  // (one scheduler-wide flag), broadcast to workers via wake -- no cancellation
  // tree. Distinct from `stop` (which ends the worker LOOP in phase 2): a worker
  // keeps running while !accepting so in-flight fibers can drain.
  std::atomic<bool> accepting{true};

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

  // io poll hook (io-04). Registered by an io::IoContext via set_io_poll; the
  // worker loop calls it to reap io completions. Null when no io is in use ->
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

  std::mutex idle_mu;              // guards `sleepers` (cold path: park/unpark)
  std::vector<unsigned> sleepers;  // ids of currently-parked workers (compact stack)
  // Per-worker position in `sleepers` (or kNotSleeping), so remove_sleeper is O(1)
  // instead of an O(worker-count) scan (complexity opportunity ②). HYLE calls
  // remove_sleeper on EVERY park exit (unlike tokio, which only scans on the rare
  // self-wake-with-local-work path and thus tolerates the scan) -- and the scan
  // runs while holding idle_mu, so shrinking it also shortens that critical section
  // under park/unpark churn. All three mutators (register on park, unpark_one pop,
  // remove_sleeper) keep this in sync; a swap-remove updates the moved id's slot.
  // Guarded by idle_mu (same lock as `sleepers`) -- no new atomics.
  static constexpr std::size_t kNotSleeping = static_cast<std::size_t>(-1);
  std::vector<std::size_t> sleeper_pos;  // size n; index into `sleepers` or kNotSleeping
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

  Impl(unsigned num_workers, mem::Allocator alloc, std::size_t cap,
       Scheduler* sched)
      : global(alloc, num_workers, cap), n(num_workers), owner(sched) {
    HYLE_FATAL_ASSERT(num_workers >= 1);
    sleeper_pos.assign(num_workers, kNotSleeping);  // O(1) sleeper-set bookkeeping
    workers.reserve(num_workers);
    ctxs.reserve(num_workers);
    threads.reserve(num_workers);
    for (unsigned i = 0; i < num_workers; ++i) {
      workers.push_back(std::make_unique<Worker>(
          alloc, cap, 0x9e3779b97f4a7c15ull ^ (static_cast<std::uint64_t>(i) + 1),
          i));
      ctxs.push_back(Ctx{this, i});
    }
    for (unsigned i = 0; i < num_workers; ++i) {
      // Pin worker i to logical core i (one-to-one when workers <= cores; the
      // platform layer best-effort-skips an out-of-range pin) and name it for
      // observability (top/htop, perf). affinity_cpu is honored only at spawn —
      // there is no post-hoc rebind API. platform-01 D3: sched per-core pinning.
      platform::ThreadOpts opts;
      opts.affinity_cpu = i;
      char wname[16];
      std::snprintf(wname, sizeof(wname), "hyle-wkr-%u", i);
      opts.name = wname;  // string_view; spawn() copies it synchronously (make_start)
      auto r = platform::Thread::spawn(&Impl::worker_main, &ctxs[i], opts);
      HYLE_FATAL_ASSERT(r.has_value());
      threads.push_back(std::move(*r));
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
    }
    // LIFO fast-slot next (opportunity ①): a fiber just woken ON this worker
    // resumes immediately, cache-hot, while under its poll budget. The budget cap
    // (kMaxLifoPolls) stops a hot ping-pong pair from starving the local ring: once
    // spent, demote the slot fiber to the ring and serve FIFO (which resets the
    // budget). Mirrors tokio's lifo_slot + MAX_LIFO_POLLS_PER_TICK and the sched-02
    // RunQueue::next logic. Single-threaded: only this worker touches lifo_slot.
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
      if (w.local.push_back(w.lifo_slot)) w.lifo_slot = nullptr;
    }
    auto r = w.local.pop();
    if (r.ok) {
      w.lifo_polls = 0;  // drawing from FIFO resets the LIFO budget
      return r.value;
    }
    Fiber* g = nullptr;
    if (global.try_pop(w.pop_cursor, g)) return g;
    for (unsigned a = 0; a < n; ++a) {
      unsigned v = static_cast<unsigned>(xorshift(w.rng) % n);
      if (v == self_idx) continue;
      auto s = workers[v]->local.steal_into(w.local);
      if (s.ok) return s.value;
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
    if (f->owner_worker() >= 0) {
      unsigned owner_idx = static_cast<unsigned>(f->owner_worker());
      while (!workers[owner_idx]->pinned.try_push(f)) platform::this_thread::yield();
      return;  // owner is THIS running worker -> it'll pop next loop; no unpark
    }
    if (!w.local.push_back(f)) {
      // Spill to OUR OWN shard (locality: our pop_cursor checks it first).
      while (!global.try_push(w.idx, f)) platform::this_thread::yield();
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

  void dispatch(Worker& w, Fiber* f) noexcept {
    f->resume();
    if (f->done()) {
      // Capture the finish hook BEFORE decrementing outstanding: a self-owned
      // fiber (spawn_task's TaskDriver) registers a hook that frees itself. This
      // is the only safe free point -- control has returned from the fiber, so its
      // stack is no longer executing (unlike freeing from inside the fiber's own
      // entry, which UAFs the fcontext trampoline). Run the hook LAST; after it
      // `f` may be gone, so read nothing more from it.
      Fiber::FinishFn fin = f->on_finish();
      void* fin_ctx = f->on_finish_ctx();
      outstanding.fetch_sub(1, std::memory_order_acq_rel);
      if (fin != nullptr) fin(fin_ctx);
    } else if (f->consume_wait_request()) {
      // The fiber suspended to WAIT on an external event (sched-04). Don't
      // reschedule -- it leaves the run loop until a Waker re-injects it. But a
      // wake() may have already fired (wake-before-suspend): the rendezvous tells
      // us if THIS side must do the re-injection now (wake got here first) or
      // leave it for wake() (we got here first). Re-injection is always post-
      // resume(), so the fiber is fully suspended -- never resumed on two threads.
      if (f->wake_rendezvous()) reschedule(w, f);
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
    int owner_idx = f->owner_worker();
    if (owner_idx >= 0) {
      while (!workers[static_cast<unsigned>(owner_idx)]->pinned.try_push(f))
        platform::this_thread::yield();
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
      if (w.local.push_back(w.lifo_slot)) {
        w.lifo_slot = f;
        return;
      }
      // Ring full -> can't evict; fall through to the global queue for f.
    }
    while (!global.try_push(inject_rr_.fetch_add(1, std::memory_order_relaxed), f))
      platform::this_thread::yield();
    notify();
  }

  // ---- sched-04b timer integration ------------------------------------------

  // Advance THIS worker's wheel to now and re-inject every fiber whose sleep
  // deadline passed. Called at the worker loop top; the wheel.empty() gate keeps
  // it ~free when nothing is sleeping. Returns the number fired (informational --
  // the caller's find_task re-checks the queues regardless, so the count is not
  // load-bearing).
  std::size_t fire_due_timers(Worker& w) noexcept {
    // Cheap gate: with no sleeping fiber the wheel is empty, so this is a single
    // size_t compare -- the "no timer workload pays nothing" property holds even
    // though we call this every loop iteration (needed so a perpetually-busy
    // worker still services its OWN pinned timers; per-worker wheels aren't
    // stealable, unlike tokio's single global wheel).
    if (w.wheel.empty()) return 0;
    time::TimerWheel::EntryList due;
    w.wheel.advance(time::monotonic_ns(), due);
    std::size_t n_fired = 0;
    while (time::TimerEntry* e = due.pop_front()) {
      // io-06: a timed-io entry carries an on_fire callback that ARBITRATES against
      // a racing io completion (it may decline to wake if io already won) -- so the
      // timer side never blindly calls wake_rendezvous when io has already fired,
      // which would leave wake_state_ dirty and lost-wake the fiber's NEXT suspend.
      // A plain sleep entry has no callback: its data is the Fiber* and the timer is
      // its SOLE waker (no race), so wake it directly through the same rendezvous.
      if (e->on_fire != nullptr) {
        e->on_fire(e);  // timed-io arbiter (net_timeout.h): sets timer_fired, wakes
                        // only if io hasn't already won
      } else {
        auto* f = static_cast<Fiber*>(e->data);
        if (f->wake_rendezvous()) reinject(f);
      }
      ++n_fired;
    }
    return n_fired;
  }

  // Absolute-deadline park bound for this worker: how long park_timeout should
  // sleep so it wakes for the nearest timer. UINT64_MAX (wheel empty) -> caller
  // parks untimed. Clamped at 0 for an already-due timer (park returns at once).
  std::uint64_t park_timeout_ns(Worker& w) noexcept {
    std::uint64_t nd = w.wheel.next_deadline_ns();
    if (nd == UINT64_MAX) return UINT64_MAX;
    std::uint64_t now = time::monotonic_ns();
    return nd > now ? nd - now : 0;
  }

  void run_worker(unsigned idx) noexcept {
    t_current = this;
    t_worker_idx = static_cast<int>(idx);
    tls_owner = owner;
    tls_worker_idx = static_cast<int>(idx);  // publish for current_worker_index()
    Worker& w = *workers[idx];
    tls_wheel = &w.wheel;  // publish for sleep_*_ns (file-local, this worker only)

    while (!stop.load(std::memory_order_acquire)) {
      // Fire this worker's due sleep timers FIRST (cheap wheel.empty() gate when
      // nothing is sleeping). Placed at the loop top -- not just the park path --
      // so a perpetually-busy worker still services its own pinned timers; a
      // fired timer re-injects its fiber, which the find_task below then runs.
      fire_due_timers(w);
      Fiber* f = find_task(w, idx);
      if (f != nullptr) {
        dispatch(w, f);
        continue;
      }
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
      for (int spin = 0; spin < kSpinPolls && !got; ++spin) {
        util::cpu_relax();
        f = find_task(w, idx);
        if (f != nullptr) got = true;
      }
      for (int y = 0; y < kYieldsBeforePark && !got; ++y) {
        platform::this_thread::yield();
        f = find_task(w, idx);
        if (f != nullptr) got = true;
      }
      if (got) {
        // Leaving searching with work in hand. If we were the LAST searcher,
        // baton one parked worker to keep a searcher warm (tokio
        // transition_from_searching) so the next burst item is caught without a
        // cold park->unpark round-trip. fetch_sub returns the PRE value.
        if (num_searching_.fetch_sub(1, std::memory_order_relaxed) == 1) {
          unpark_one();
        }
        dispatch(w, f);
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
      f = find_task(w, idx);                                   // Dekker recheck
      if (f != nullptr) {
        remove_sleeper(idx);
        dispatch(w, f);
        continue;
      }
      // Park until unparked OR the nearest sleep timer is due. An empty wheel
      // gives UINT64_MAX -> untimed park (exactly the pre-04b behaviour, zero
      // cost when no fiber is sleeping). A pending timer bounds the park so the
      // worker wakes to fire it; park_timeout returning false (timeout) is fine
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
        // poll. This is why set_io_wake is mandatory once the cap is gone.
        if (poll(io_poll_ctx.load(std::memory_order_relaxed), idx, timeout)) {
          remove_sleeper(idx);
          continue;  // io was in flight -> serviced in the ring, re-check queues
        }
        // else: nothing in flight -> park below for an immediate cross-thread wake
      }
      if (timeout == UINT64_MAX) {
        w.parker.park();
      } else {
        w.parker.park_timeout(timeout);
      }
      remove_sleeper(idx);  // idempotent: a notifier may have popped us already
    }

    t_current = nullptr;
    t_worker_idx = -1;
    tls_wheel = nullptr;
    tls_worker_idx = -1;
  }

  void spawn(Fiber* f) noexcept {
    // sched-08: reject new work once shutting down (phase 1). Not admitted -> not
    // counted, not queued (the caller owns the Fiber and should check accepting()
    // before building work; a spawn that races begin_shutdown is a safe no-op).
    if (!accepting.load(std::memory_order_acquire)) return;
    outstanding.fetch_add(1, std::memory_order_acq_rel);  // count before publish
    if (t_current == this && t_worker_idx >= 0) {
      if (workers[static_cast<unsigned>(t_worker_idx)]->local.push_back(f)) {
        notify();
        return;
      }
    }
    while (!global.try_push(inject_rr_.fetch_add(1, std::memory_order_relaxed), f))
      platform::this_thread::yield();
    notify();
  }

  // Spawn a fiber PINNED to worker `idx` (shard-per-core). It records its owner so
  // every subsequent wake also routes back here (reinject), then goes onto that
  // worker's private queue and unparks it specifically. Never stolen.
  void spawn_on(unsigned idx, Fiber* f) noexcept {
    HYLE_FATAL_ASSERT(idx < n);
    if (!accepting.load(std::memory_order_acquire)) return;  // sched-08: reject new work
    f->set_owner_worker(static_cast<int>(idx));
    outstanding.fetch_add(1, std::memory_order_acq_rel);  // count before publish
    while (!workers[idx]->pinned.try_push(f)) platform::this_thread::yield();
    unpark_specific(idx);
  }

  void drain() noexcept {
    while (outstanding.load(std::memory_order_acquire) != 0) {
      platform::this_thread::yield();
    }
  }

  // sched-08 phase 1: stop accepting new work + broadcast the signal so long-lived
  // fibers (which poll shutdown_requested()) can wind down. Waking every worker
  // (both channels) matters because a parked/reaping worker must re-check its state
  // and let its pinned long-lived fibers observe the flag. Idempotent.
  void begin_shutdown() noexcept {
    accepting.store(false, std::memory_order_release);
    for (unsigned i = 0; i < n; ++i) wake_worker(i);
  }

  // sched-08 phase 2: wait up to timeout_ns (0 == forever) for outstanding to hit
  // zero. Returns true if drained, false on timeout. Busy-yield like drain() -- a
  // shutdown drain is a one-shot teardown wait, not a hot path.
  bool drain_for(std::uint64_t timeout_ns) noexcept {
    if (timeout_ns == 0) {
      drain();
      return true;
    }
    const std::uint64_t deadline = time::deadline_after_ns(timeout_ns);
    while (outstanding.load(std::memory_order_acquire) != 0) {
      if (time::monotonic_ns() >= deadline) return false;
      platform::this_thread::yield();
    }
    return true;
  }

  void shutdown() {
    drain();
    stop.store(true, std::memory_order_release);
    // Wake every worker to observe `stop` -- via BOTH channels (wake_worker), since
    // an io-active worker is blocked in the reap, not the futex parker.
    for (unsigned i = 0; i < n; ++i) wake_worker(i);
    for (auto& t : threads) {
      if (t.joinable()) t.join();
    }
  }

  static thread_local Impl* t_current;
  static thread_local int t_worker_idx;
};

thread_local Scheduler::Impl* Scheduler::Impl::t_current = nullptr;
thread_local int Scheduler::Impl::t_worker_idx = -1;

Scheduler::Scheduler(unsigned num_workers, mem::Allocator alloc,
                     std::size_t queue_capacity)
    : impl_(new Impl(num_workers, alloc, queue_capacity, this)) {}

Scheduler::~Scheduler() {
  impl_->shutdown();
  delete impl_;
}

void Scheduler::spawn(Fiber* fiber) { impl_->spawn(fiber); }
void Scheduler::spawn_on(unsigned worker_idx, Fiber* fiber) {
  impl_->spawn_on(worker_idx, fiber);
}
unsigned Scheduler::num_workers() const noexcept { return impl_->n; }
mem::FiberStackPool* Scheduler::fiber_stack_pool(std::size_t usable_bytes) noexcept {
  return impl_->fiber_stack_pool(usable_bytes);
}
void Scheduler::drain() { impl_->drain(); }
void Scheduler::shutdown() { impl_->shutdown(); }
bool Scheduler::accepting() const noexcept {
  return impl_->accepting.load(std::memory_order_acquire);
}
void Scheduler::begin_shutdown() noexcept { impl_->begin_shutdown(); }
bool Scheduler::drain_for(unsigned long long timeout_ns) noexcept {
  return impl_->drain_for(timeout_ns);
}

void Scheduler::set_io_poll(IoPollFn fn, void* ctx) noexcept {
  // Publish ctx BEFORE fn (release) so a worker that reads a non-null fn is
  // guaranteed to see the matching ctx. Clearing (fn=null) is the reverse but
  // benign: a worker reading the stale non-null fn with a now-null ctx still runs
  // a valid (pre-teardown) hook; unregister happens only after the scheduler is
  // stopped (IoContext dtor), so no worker is looping then anyway.
  impl_->io_poll_ctx.store(ctx, std::memory_order_relaxed);
  impl_->io_poll.store(fn, std::memory_order_release);
}

void Scheduler::set_io_wake(IoWakeFn fn, void* ctx) noexcept {
  // Same publish discipline as set_io_poll (ctx before fn, release).
  impl_->io_wake_ctx.store(ctx, std::memory_order_relaxed);
  impl_->io_wake.store(fn, std::memory_order_release);
}

// ---- completion spine (sched-04a) -------------------------------------------

void Waker::wake() const noexcept {
  // Rendezvous: whichever of {worker dispatch, this wake()} sees the other has
  // already arrived performs the single re-injection. If we arrived first, the
  // worker's dispatch will re-inject post-suspend.
  if (fiber_->wake_rendezvous()) sched_->impl_->reinject(fiber_);
}

Waker current_waker() noexcept {
  return Waker{tls_owner, Fiber::current()};  // both null if not on a worker fiber
}

int current_worker_index() noexcept { return tls_worker_idx; }

// sched-08: a long-lived fiber polls this to learn its scheduler has begun graceful
// shutdown, so it can break its loop. tls_owner is the Scheduler running this fiber
// (null off a worker -> not shutting down from this call's view). One relaxed-ish
// load through the public accepting() (acquire); cheap enough for a loop-top check.
bool shutdown_requested() noexcept {
  return tls_owner != nullptr && !tls_owner->accepting();
}

void suspend() noexcept {
  Fiber* self = Fiber::current();
  HYLE_FATAL_ASSERT(self != nullptr);  // suspend() outside a fiber is a bug
  self->request_wait();  // tell dispatch: don't reschedule me, I'm waiting
  Fiber::yield();        // control returns to the worker; resumes when re-injected
}

// ---- fiber sleep (sched-04b) ------------------------------------------------

void sleep_until_ns(std::uint64_t deadline_ns) noexcept {
  Fiber* self = Fiber::current();
  HYLE_FATAL_ASSERT(self != nullptr);  // sleep outside a fiber is a bug
  HYLE_FATAL_ASSERT(tls_wheel != nullptr);  // must be on a scheduler worker

  // The timer entry lives on THIS fiber's stack. Safe: the fiber suspends right
  // below and its stack is frozen until the timer (its sole waker) re-injects
  // it, so the entry outlives its time in the wheel. data = the fiber to wake.
  // O(1) onto this worker's own wheel (single-threaded: we ARE that worker).
  time::TimerEntry entry;
  entry.data = self;
  tls_wheel->schedule(&entry, deadline_ns);

  // Suspend on the SAME spine as an io completion: request_wait + yield, and the
  // worker's dispatch rendezvous with the eventual timer fire (fire_due_timers
  // calls wake_rendezvous, exactly like Waker::wake). No competing waker exists
  // for a plain sleep, so the timer always fires and the entry always leaves the
  // wheel -- no cancel needed (that's the fiber-mode wait_for/wait_until case,
  // sched-04d).
  self->request_wait();
  Fiber::yield();
}

void sleep_for_ns(std::uint64_t ns) noexcept {
  sleep_until_ns(time::deadline_after_ns(ns));
}

// ---- timed-io building blocks (io-06) ---------------------------------------
// arm/cancel a timeout timer on THIS worker's wheel WITHOUT suspending, so a
// timed io op can arm a deadline + submit an io op and then suspend once. Same
// single-threaded per-worker discipline as sleep (we ARE the owning worker); the
// timer wakes this fiber via the SAME reinject path (entry.data = self, fired by
// fire_due_timers -> wake_rendezvous). No cross-core cancel queue: the timed io
// fiber is pinned (shard-per-core), so arm/fire/cancel are all on this worker.

void arm_timeout(time::TimerEntry* entry, std::uint64_t deadline_ns) noexcept {
  Fiber* self = Fiber::current();
  HYLE_FATAL_ASSERT(self != nullptr);       // timed io outside a fiber is a bug
  HYLE_FATAL_ASSERT(tls_wheel != nullptr);  // must be on a scheduler worker
  entry->data = self;  // fire_due_timers reinjects this fiber (same as sleep)
  tls_wheel->schedule(entry, deadline_ns);
}

void cancel_timeout(time::TimerEntry* entry) noexcept {
  HYLE_FATAL_ASSERT(tls_wheel != nullptr);
  tls_wheel->cancel(entry);  // O(1); no-op if already fired
}

}  // namespace hyle::sched
