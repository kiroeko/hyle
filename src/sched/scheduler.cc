#include "hyle/sched/scheduler.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "hyle/container/bounded_queue.h"
#include "hyle/container/ws_fifo_queue.h"
#include "hyle/macros.h"
#include "hyle/platform/thread.h"
#include "hyle/platform/wait.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/waker.h"

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

// Bounded spin sweeps before an idle worker parks. Park is a syscall and work
// often arrives within microseconds, so a little spinning avoids park/unpark
// churn (sched-design §2.3 ①; HYLE/Go-style, not current tokio).
constexpr int kSpinSweeps = 4;

}  // namespace

struct Scheduler::Impl {
  struct Worker {
    container::WsFifoQueue<Fiber*> local;
    platform::Parker parker;  // non-movable -> Worker lives in a unique_ptr
    std::uint64_t rng;
    Worker(mem::Allocator a, std::size_t cap, std::uint64_t seed)
        : local(a, cap), rng(seed) {}
  };

  struct Ctx {
    Impl* self;
    unsigned idx;
  };

  container::BoundedQueue<Fiber*> global;
  std::vector<std::unique_ptr<Worker>> workers;
  std::vector<Ctx> ctxs;
  std::vector<platform::Thread> threads;
  std::atomic<bool> stop{false};
  std::atomic<std::int64_t> outstanding{0};

  std::mutex idle_mu;              // guards `sleepers` (cold path: park/unpark)
  std::vector<unsigned> sleepers;  // ids of currently-parked workers
  unsigned n;
  Scheduler* owner;  // for tls_owner / current_waker()

  Impl(unsigned num_workers, mem::Allocator alloc, std::size_t cap,
       Scheduler* sched)
      : global(alloc, cap), n(num_workers), owner(sched) {
    HYLE_FATAL_ASSERT(num_workers >= 1);
    workers.reserve(num_workers);
    ctxs.reserve(num_workers);
    threads.reserve(num_workers);
    for (unsigned i = 0; i < num_workers; ++i) {
      workers.push_back(std::make_unique<Worker>(
          alloc, cap, 0x9e3779b97f4a7c15ull ^ (static_cast<std::uint64_t>(i) + 1)));
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
    if (global.try_pop(g)) return g;
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
      while (!global.try_push(f)) platform::this_thread::yield();
      notify();  // landed in global -> a parked worker may need to take it
    }
  }

  // Wake one parked worker, UNCONDITIONALLY on every enqueue. Correctness rests
  // on this being unconditional: gating it on a "is anyone searching?" read
  // (the §2.3 ② anti-herd optimization) reintroduces a StoreLoad race that can
  // drop a wakeup (worker decrements its searching flag, a producer reads it as
  // 0... reversed and you hang). The herd-suppression gate + baton relay (§2.3
  // ②③) are DEFERRED: they need a seq_cst Dekker handshake and model-checking;
  // shipping a subtly-wrong relay is worse than a few extra wakeups. TODO sched.
  void notify() noexcept { unpark_one(); }

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
    while (!global.try_push(f)) platform::this_thread::yield();
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
      // Idle: spin a few sweeps (work often arrives fast) before parking.
      bool got = false;
      for (int sweep = 0; sweep < kSpinSweeps; ++sweep) {
        platform::this_thread::yield();
        f = find_task(w, idx);
        if (f != nullptr) {
          got = true;
          break;
        }
      }
      if (got) {
        dispatch(w, f);
        continue;
      }
      // Prepare to park: register as a sleeper, then RE-CHECK once more. The
      // register-before-recheck plus an UNCONDITIONAL notify (see notify())
      // close the lost-wakeup window: any producer that enqueues after our
      // recheck also runs unpark_one after we are registered, so it pops and
      // wakes us; Parker's remembered token covers the unpark-before-park race.
      {
        std::lock_guard<std::mutex> lk(idle_mu);
        sleepers.push_back(idx);
      }
      f = find_task(w, idx);
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
    while (!global.try_push(f)) platform::this_thread::yield();
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
