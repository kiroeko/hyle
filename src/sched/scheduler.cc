#include "hyle/sched/scheduler.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "hyle/container/bounded_queue.h"
#include "hyle/container/ws_fifo_queue.h"
#include "hyle/macros.h"
#include "hyle/platform/thread.h"
#include "hyle/sched/fiber.h"

namespace hyle::sched {
namespace {

// Per-worker xorshift64 for cheap random victim selection.
std::uint64_t xorshift(std::uint64_t& s) noexcept {
  s ^= s << 13;
  s ^= s >> 7;
  s ^= s << 17;
  return s;
}

}  // namespace

struct Scheduler::Impl {
  struct Worker {
    container::WsFifoQueue<Fiber*> local;
    std::uint64_t rng;
    Worker(mem::Allocator a, std::size_t cap, std::uint64_t seed)
        : local(a, cap), rng(seed) {}
  };

  // Trampoline context: stable address (workers vector reserved up front).
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
  unsigned n;

  Impl(unsigned num_workers, mem::Allocator alloc, std::size_t cap)
      : global(alloc, cap), n(num_workers) {
    HYLE_FATAL_ASSERT(num_workers >= 1);
    workers.reserve(num_workers);
    ctxs.reserve(num_workers);
    threads.reserve(num_workers);
    for (unsigned i = 0; i < num_workers; ++i) {
      workers.push_back(std::make_unique<Worker>(
          alloc, cap, 0x9e3779b97f4a7c15ull ^ (static_cast<std::uint64_t>(i) + 1)));
      ctxs.push_back(Ctx{this, i});
    }
    // Start workers only after all state is constructed.
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

  // Find the next runnable fiber: local -> global -> steal. nullptr if none.
  Fiber* find_task(Worker& w, unsigned self_idx) noexcept {
    auto r = w.local.pop();
    if (r.ok) return r.value;
    Fiber* g = nullptr;
    if (global.try_pop(g)) return g;
    // Steal ~half from up to n random victims into our own local queue.
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
    }
  }

  void run_worker(unsigned idx) noexcept {
    t_current = this;
    t_worker_idx = static_cast<int>(idx);
    Worker& w = *workers[idx];
    while (!stop.load(std::memory_order_acquire)) {
      Fiber* f = find_task(w, idx);
      if (f == nullptr) {
        platform::this_thread::yield();  // step1: busy-spin (park is step2)
        continue;
      }
      f->resume();
      if (f->done()) {
        outstanding.fetch_sub(1, std::memory_order_acq_rel);
      } else {
        reschedule(w, f);  // voluntary yield -> back on a queue
      }
    }
    t_current = nullptr;
    t_worker_idx = -1;
  }

  void spawn(Fiber* f) noexcept {
    outstanding.fetch_add(1, std::memory_order_acq_rel);  // count before publish
    if (t_current == this && t_worker_idx >= 0) {
      if (workers[static_cast<unsigned>(t_worker_idx)]->local.push_back(f)) return;
    }
    while (!global.try_push(f)) platform::this_thread::yield();
  }

  void drain() noexcept {
    while (outstanding.load(std::memory_order_acquire) != 0) {
      platform::this_thread::yield();
    }
  }

  void shutdown() {
    drain();
    stop.store(true, std::memory_order_release);
    for (auto& t : threads) {
      if (t.joinable()) t.join();
    }
  }

  // Which scheduler/worker the current OS thread is running (for spawn routing).
  static thread_local Impl* t_current;
  static thread_local int t_worker_idx;
};

thread_local Scheduler::Impl* Scheduler::Impl::t_current = nullptr;
thread_local int Scheduler::Impl::t_worker_idx = -1;

Scheduler::Scheduler(unsigned num_workers, mem::Allocator alloc,
                     std::size_t queue_capacity)
    : impl_(new Impl(num_workers, alloc, queue_capacity)) {}

Scheduler::~Scheduler() {
  impl_->shutdown();  // safe if already shut down (threads not joinable)
  delete impl_;
}

void Scheduler::spawn(Fiber* fiber) { impl_->spawn(fiber); }
void Scheduler::drain() { impl_->drain(); }
void Scheduler::shutdown() { impl_->shutdown(); }

}  // namespace hyle::sched
