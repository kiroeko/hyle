#include "hyle/sched/scheduler.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "hyle/container/sharded_queue.h"
#include "hyle/container/ws_fifo_queue.h"
#include "hyle/macros.h"
#include "hyle/platform/thread.h"
#include "hyle/platform/wait.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/waker.h"
#include "hyle/util/spin_lock.h"  // util::cpu_relax (pause hint)

namespace hyle::sched {
namespace {

// The Scheduler running on this worker thread, for current_waker(). File-local
// (not an Impl member) so the free functions below can read it without naming
// the private nested Impl type.
thread_local Scheduler* tls_owner = nullptr;

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
    container::WsFifoQueue<Fiber*> local;
    platform::Parker parker;  // non-movable -> Worker lives in a unique_ptr
    std::uint64_t rng;
    unsigned idx;             // own worker index: shard hint + pop-cursor seed
    std::size_t pop_cursor;   // round-robin cursor into the global ShardedQueue
    Worker(mem::Allocator a, std::size_t cap, std::uint64_t seed, unsigned i)
        : local(a, cap), rng(seed), idx(i), pop_cursor(i) {}
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

  void unpark_one() noexcept {
    unsigned idx;
    {
      std::lock_guard<std::mutex> lk(idle_mu);
      if (sleepers.empty()) return;
      idx = sleepers.back();
      sleepers.pop_back();
    }
    workers[idx]->parker.unpark();
  }

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
    // Foreign-thread inject (Waker may fire anywhere): round-robin across shards.
    while (!global.try_push(inject_rr_.fetch_add(1, std::memory_order_relaxed), f))
      platform::this_thread::yield();
    notify();
  }

  void run_worker(unsigned idx) noexcept {
    t_current = this;
    t_worker_idx = static_cast<int>(idx);
    tls_owner = owner;
    Worker& w = *workers[idx];

    while (!stop.load(std::memory_order_acquire)) {
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
      w.parker.park();
      remove_sleeper(idx);  // idempotent: a notifier may have popped us already
    }

    t_current = nullptr;
    t_worker_idx = -1;
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

  void drain() noexcept {
    while (outstanding.load(std::memory_order_acquire) != 0) {
      platform::this_thread::yield();
    }
  }

  void shutdown() {
    drain();
    stop.store(true, std::memory_order_release);
    for (auto& w : workers) w->parker.unpark();  // wake parked ones to see stop
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
void Scheduler::drain() { impl_->drain(); }
void Scheduler::shutdown() { impl_->shutdown(); }

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

void suspend() noexcept {
  Fiber* self = Fiber::current();
  HYLE_FATAL_ASSERT(self != nullptr);  // suspend() outside a fiber is a bug
  self->request_wait();  // tell dispatch: don't reschedule me, I'm waiting
  Fiber::yield();        // control returns to the worker; resumes when re-injected
}

}  // namespace hyle::sched
