// SOTA parity: HYLE fiber sync vs marl (the fiber-sync library this design
// converged from). Built only with -DHYLE_BENCH_MARL=ON.
//
// Two head-to-head workloads, same shape on both schedulers (4 workers):
//  1. CONTENDED Mutex: 8 tasks x K (lock / ++counter / unlock). HYLE's Mutex is
//     a futex (Drepper 3-state, parks the worker on contention); marl::mutex IS
//     std::mutex (also futex-backed). Same family -> expect parity.
//  2. WaitGroup fork-join: N tasks each done(), one waiter. Both have a near-
//     identical WaitGroup API.
//
// HONEST framing: marl::mutex == std::mutex; HYLE's Mutex is its own futex over
// platform::wait_on_address. Both park the worker (not the fiber) on contention
// -- a mutex guards short sections, where parking beats yielding. #1 checks
// futex-vs-futex parity; #2 is dominated by HYLE's per-fiber 64KB stack alloc
// (no fiber-stack pool yet) vs marl's pooled fibers, NOT by the WaitGroup sync.

#include <nanobench.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "marl/mutex.h"
#include "marl/scheduler.h"
#include "marl/waitgroup.h"

#include "hyle/mem/allocator.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/sync.h"

using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;

namespace {
constexpr int kWorkers = 4;
constexpr int kMutexTasks = 8;
constexpr int kMutexIters = 2000;  // per task
constexpr int kWgTasks = 1024;

// HYLE contended-mutex fiber body.
struct HyleHammer {
  hyle::sched::Mutex m;
  std::uint64_t counter = 0;
  int iters;
};
void hyle_hammer_body(void* p) {
  auto* h = static_cast<HyleHammer*>(p);
  for (int i = 0; i < h->iters; ++i) {
    h->m.lock();
    ++h->counter;
    h->m.unlock();
  }
}
struct HyleWg {
  hyle::sched::WaitGroup* wg;
};
void hyle_wg_done(void* p) { static_cast<HyleWg*>(p)->wg->done(); }
void hyle_wg_wait(void* p) { static_cast<HyleWg*>(p)->wg->wait(); }
}  // namespace

int main() {
  // ---- 1. contended Mutex: HYLE fiber Mutex vs marl mutex(=std::mutex) ----
  {
    const std::uint64_t batch =
        static_cast<std::uint64_t>(kMutexTasks) * kMutexIters;
    ankerl::nanobench::Bench b;
    b.title("contended mutex (8 tasks/4 workers, lock+inc+unlock)")
        .relative(true)
        .batch(batch)
        .minEpochIterations(20);

    // marl: persistent scheduler, per-epoch schedule 8 tasks + WaitGroup join.
    {
      marl::Scheduler::Config cfg;
      cfg.setWorkerThreadCount(kWorkers);
      marl::Scheduler sched(cfg);
      sched.bind();
      b.run("marl mutex (=std::mutex)", [&] {
        marl::mutex m;
        std::uint64_t counter = 0;
        marl::WaitGroup wg(kMutexTasks);
        for (int t = 0; t < kMutexTasks; ++t) {
          marl::schedule([&m, &counter, &wg] {
            for (int i = 0; i < kMutexIters; ++i) {
              m.lock();
              ++counter;
              m.unlock();
            }
            wg.done();
          });
        }
        wg.wait();
        ankerl::nanobench::doNotOptimizeAway(counter);
      });
      marl::Scheduler::unbind();
    }

    // HYLE: persistent scheduler, per-epoch spawn 8 fibers + drain.
    {
      hyle::sched::Scheduler sched(kWorkers, system_allocator(), 1024);
      b.run("HYLE Mutex (futex)", [&] {
        HyleHammer h;
        h.iters = kMutexIters;
        std::vector<std::unique_ptr<Fiber>> fibers;
        fibers.reserve(kMutexTasks);
        for (int t = 0; t < kMutexTasks; ++t)
          fibers.push_back(std::make_unique<Fiber>(
              system_allocator(), kDefaultStackSize, &hyle_hammer_body, &h));
        for (auto& f : fibers) sched.spawn(f.get());
        sched.drain();
        ankerl::nanobench::doNotOptimizeAway(h.counter);
      });
      sched.shutdown();
    }
  }

  // ---- 2. WaitGroup fork-join: N tasks + one waiter ----
  {
    ankerl::nanobench::Bench b;
    b.title("WaitGroup fork-join (1024 tasks/4 workers)")
        .relative(true)
        .batch(kWgTasks)
        .minEpochIterations(20);

    {
      marl::Scheduler::Config cfg;
      cfg.setWorkerThreadCount(kWorkers);
      marl::Scheduler sched(cfg);
      sched.bind();
      b.run("marl WaitGroup", [&] {
        marl::WaitGroup wg(kWgTasks);
        for (int i = 0; i < kWgTasks; ++i)
          marl::schedule([wg] { wg.done(); });
        wg.wait();
      });
      marl::Scheduler::unbind();
    }

    {
      hyle::sched::Scheduler sched(kWorkers, system_allocator(), 2048);
      b.run("HYLE WaitGroup", [&] {
        hyle::sched::WaitGroup wg;
        wg.add(kWgTasks);
        HyleWg arg{&wg};
        std::vector<std::unique_ptr<Fiber>> fibers;
        fibers.reserve(kWgTasks + 1);
        fibers.push_back(std::make_unique<Fiber>(
            system_allocator(), kDefaultStackSize, &hyle_wg_wait, &arg));
        for (int i = 0; i < kWgTasks; ++i)
          fibers.push_back(std::make_unique<Fiber>(
              system_allocator(), kDefaultStackSize, &hyle_wg_done, &arg));
        for (auto& f : fibers) sched.spawn(f.get());
        sched.drain();
      });
      sched.shutdown();
    }
  }

  return 0;
}
