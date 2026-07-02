#include "hyle/sched/scheduler.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "hyle/container/bounded_queue.h"
#include "hyle/container/sharded_queue.h"
#include "hyle/container/ws_fifo_queue.h"
#include "hyle/macros.h"
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

  // io poll hook (io-04). Registered by an io::IoContext via set_io_poll; the
  // worker loop calls it to reap io completions. Null when no io is in use ->
  // never called, zero cost. ATOMIC because the registering thread (IoContext
  // ctor on the main thread) writes these while worker threads are already
  // spinning their loops and reading them -- a genuine cross-thread publish
  // (TSan caught the plain-field version). Write ctx first then fn with release;
  // read fn (acquire) then ctx, so a non-null fn implies its ctx is visible.
  std::atomic<Scheduler::IoPollFn> io_poll{nullptr};
  std::atomic<void*> io_poll_ctx{nullptr};
  // Companion wake hook (io-04 park merge): posts a bare wake into a worker's ring
  // so a spawn/steal can break its blocking reap. Shares io_poll_ctx (same
  // IoContext). Atomic for the same cross-thread-publish reason as io_poll.
  std::atomic<Scheduler::IoWakeFn> io_wake{nullptr};

  std::mutex idle_mu;              // guards `sleepers` (cold path: park/unpark)
  std::vector<unsigned> sleepers;  // ids of currently-parked workers
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
      auto r = platform::Thread::spawn(&Impl::worker_main, &ctxs[i]);
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
    auto r = w.local.pop();
    if (r.ok) return r.value;
    Fiber* g = nullptr;
    if (global.try_pop(w.pop_cursor, g)) return g;
    for (unsigned a = 0; a < n; ++a) {
      unsigned v = static_cast<unsigned>(xorshift(w.rng) % n);
      if (v == self_idx) continue;
      auto s = workers[v]->local.steal_into(w.local);
      if (s.ok) return s.value;
    }
    return nullptr;
  }

  void reschedule(Worker& w, Fiber* f) noexcept {
    // A pinned fiber (shard-per-core) must NOT go on the stealable local queue --
    // a plain cooperative yield would otherwise let another core steal it. Keep it
    // on its owner's private queue. (owner == w.idx here, since a pinned fiber only
    // ever runs on its owner, but route via owner to be explicit + future-proof.)
    if (f->owner_worker() >= 0) {
      unsigned owner = static_cast<unsigned>(f->owner_worker());
      while (!workers[owner]->pinned.try_push(f)) platform::this_thread::yield();
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

  // Wake worker `idx` through BOTH wake channels (io-04 park merge). The futex
  // Parker token always fires (harmless + latched if the worker isn't parked).
  // When io is active the worker BLOCKS IN reap(), not the Parker, so we ALSO
  // post a bare wake into its ring -- that is what actually breaks the reap. One
  // of the two is always the effective wake depending on where the worker sits;
  // posting both is safe (the parker token is consumed on the next non-io park,
  // the ring wake is drained + skipped by net::complete's kWakeToken check).
  void wake_worker(unsigned idx) noexcept {
    workers[idx]->parker.unpark();
    if (IoWakeFn wk = io_wake.load(std::memory_order_acquire); wk != nullptr)
      wk(io_poll_ctx.load(std::memory_order_relaxed), idx);
  }

  void unpark_one() noexcept {
    unsigned idx;
    {
      std::lock_guard<std::mutex> lk(idle_mu);
      if (sleepers.empty()) return;
      idx = sleepers.back();
      sleepers.pop_back();
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
  void unpark_specific(unsigned idx) noexcept { wake_worker(idx); }

  void remove_sleeper(unsigned idx) noexcept {
    std::lock_guard<std::mutex> lk(idle_mu);
    for (std::size_t i = 0; i < sleepers.size(); ++i) {
      if (sleepers[i] == idx) {
        sleepers[i] = sleepers.back();
        sleepers.pop_back();
        return;
      }
    }
  }

  void dispatch(Worker& w, Fiber* f) noexcept {
    f->resume();
    if (f->done()) {
      outstanding.fetch_sub(1, std::memory_order_acq_rel);
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
    int owner = f->owner_worker();
    if (owner >= 0) {
      while (!workers[static_cast<unsigned>(owner)]->pinned.try_push(f))
        platform::this_thread::yield();
      unpark_specific(static_cast<unsigned>(owner));
      return;
    }
    // Foreign-thread inject (Waker may fire anywhere): round-robin across shards.
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
      // data is the Fiber* that armed this sleep. Firing == the sole wake of a
      // plain sleep: re-inject through the SAME rendezvous as an io completion,
      // so it composes with a concurrent wake() (none for pure sleep, but the
      // path stays uniform). The SleepNode (holding `e`) lives on the suspended
      // fiber's frozen stack and is valid until the fiber resumes.
      auto* f = static_cast<Fiber*>(e->data);
      if (f->wake_rendezvous()) reinject(f);
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
      // Reap io completions (io-04): non-blocking poll from the hot loop so a
      // ready completion re-injects its fiber before we look for work. Null hook
      // (no io in use) -> skipped entirely, zero cost.
      if (IoPollFn poll = io_poll.load(std::memory_order_acquire); poll != nullptr)
        poll(io_poll_ctx.load(std::memory_order_relaxed), idx, 0);
      Fiber* f = find_task(w, idx);
      if (f != nullptr) {
        dispatch(w, f);
        continue;
      }
      // No work on the first look -> become a SEARCHER. While num_searching_ > 0
      // producers skip waking parked workers (notify() gate), so a spawn burst
      // no longer triggers a futex per enqueue -- THIS searcher absorbs them by
      // finding the work during its spin. The spin also just bridges the brief
      // producer gap before the (now rarely needed) park.
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
      // worker wakes to fire it; timeout returning (expiry) is fine -- the loop
      // top's fire_due_timers handles it uniformly.
      std::uint64_t timeout = park_timeout_ns(w);
      // io-04 park merge: when an io hook is registered, BLOCK IN THE io reap
      // rather than the futex Parker. reap(timeout) flushes this worker's batched
      // SQEs (the submit-batch payoff -- one enter for the whole slice), then
      // waits for a completion OR the timer deadline. A completion wakes us with
      // zero latency; a spawn/steal wakes us because notify()/unpark posts a bare
      // kWakeToken into our ring (see unpark_specific). This unifies both wake
      // sources onto the one reap wait -- no 1ms poll cap. UINT64_MAX -> reap
      // blocks untimed (no sleeping timer). The routing of any reaped completions
      // happens inside the hook (net::complete), so the loop top's next poll +
      // find_task just picks up the woken fibers.
      if (IoPollFn poll = io_poll.load(std::memory_order_acquire);
          poll != nullptr) {
        poll(io_poll_ctx.load(std::memory_order_relaxed), idx, timeout);
      } else if (timeout == UINT64_MAX) {
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

  void shutdown() {
    drain();
    stop.store(true, std::memory_order_release);
    // Wake every worker to observe `stop`. wake_worker hits BOTH channels: the
    // futex Parker AND (if io active) a ring post -- an io-blocked worker sits in
    // reap(), so the ring post is what actually breaks it out.
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
void Scheduler::drain() { impl_->drain(); }
void Scheduler::shutdown() { impl_->shutdown(); }

void Scheduler::set_io_poll(IoPollFn fn, void* ctx) noexcept {
  // Publish ctx BEFORE fn (release) so a worker that reads a non-null fn is
  // guaranteed to see the matching ctx. Clearing (fn=null) is the reverse but
  // benign: a worker reading the stale non-null fn with a now-null ctx still runs
  // a valid (pre-teardown) hook; unregister happens only after the scheduler is
  // stopped (IoContext dtor), so no worker is looping then anyway.
  impl_->io_poll_ctx.store(ctx, std::memory_order_relaxed);
  impl_->io_poll.store(fn, std::memory_order_release);
}

void Scheduler::set_io_wake(IoWakeFn fn, void* /*ctx*/) noexcept {
  // Shares io_poll_ctx (same IoContext registered both hooks), so only the fn
  // pointer is stored here. Release so a worker seeing a non-null io_poll also
  // sees this (they are set together, io_wake last).
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
  // wheel -- no cancel needed (that's the wait_for/wait_until case, sched-04c).
  self->request_wait();
  Fiber::yield();
}

void sleep_for_ns(std::uint64_t ns) noexcept {
  sleep_until_ns(time::monotonic_ns() + ns);
}

}  // namespace hyle::sched
