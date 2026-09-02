// SOTA parity: HYLE fiber sync vs marl (the fiber-sync library this design
// converged from). Built only with -DHYLE_BENCH_MARL=ON.
//
// Two head-to-head workloads, same shape on both schedulers (4 workers):
//  1. CONTENDED Mutex: 8 tasks x K (lock / ++counter / unlock). HYLE uses the
//     platform fast lock (SRWLOCK on Windows; compact address-wait lock on
//     POSIX); marl::mutex IS std::mutex. Same short-critical-section contract.
//  2. WaitGroup fork-join: N tasks each done(), one waiter. Both have a near-
//     identical WaitGroup API.
//
// HONEST framing: marl::mutex == std::mutex; both park the worker (not the fiber)
// on contention -- a mutex guards short sections, where parking beats yielding.
// #2 and #3 call marl::Scheduler::enqueue directly, deliberately leaving the
// benchmark caller unbound: marl::bind would turn the caller into an additional
// executable fiber worker and make a configured "4 workers" a 5-vs-4 shape.
// They compare marl's fire-and-forget schedule with
// HYLE's equal-semantics spawn_detached; an extra HYLE spawn_task row prices the
// richer coroutine + JoinHandle structured-concurrency layer separately.

#include <nanobench.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "marl/mutex.h"
#include "marl/scheduler.h"
#include "marl/task.h"
#include "marl/waitgroup.h"

#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/join.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/sync.h"

using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::JoinHandle;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Task;

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
  hyle::sched::WaitGroup* done = nullptr;
};
void hyle_hammer_body(void* p) {
  auto* h = static_cast<HyleHammer*>(p);
  for (int i = 0; i < h->iters; ++i) {
    h->m.lock();
    ++h->counter;
    h->m.unlock();
  }
  if (h->done != nullptr) h->done->done();
}
Task<void> hyle_wg_task(hyle::mem::AllocatorRef, hyle::sched::WaitGroup* wg) {
  wg->done();
  co_return;
}
Task<void> hyle_noop_task(hyle::mem::AllocatorRef) { co_return; }
void hyle_wg_done(void* p) {
  static_cast<hyle::sched::WaitGroup*>(p)->done();
}
void hyle_noop(void*) {}
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
      b.run("marl mutex (=std::mutex)", [&] {
        marl::mutex m;
        std::uint64_t counter = 0;
        marl::WaitGroup wg(kMutexTasks);
        for (int t = 0; t < kMutexTasks; ++t) {
          sched.enqueue(marl::Task([&m, &counter, &wg] {
            for (int i = 0; i < kMutexIters; ++i) {
              m.lock();
              ++counter;
              m.unlock();
            }
            wg.done();
          }));
        }
        wg.wait();
        ankerl::nanobench::doNotOptimizeAway(counter);
      });
    }

    // HYLE: persistent scheduler, same worker-stack callback shape as marl.
    // Mutex protects a short critical section and may briefly park a worker;
    // no fiber-only wait/yield/sleep API is used here.
    {
      hyle::sched::Scheduler sched(kWorkers, system_allocator(), 1024);
      b.run("HYLE Mutex (platform fast lock)", [&] {
        HyleHammer h;
        h.iters = kMutexIters;
        hyle::sched::WaitGroup done(kMutexTasks);
        h.done = &done;
        for (int t = 0; t < kMutexTasks; ++t) {
          const auto status = sched.schedule(&hyle_hammer_body, &h);
          ankerl::nanobench::doNotOptimizeAway(status.ok());
        }
        done.wait();
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
      b.run("marl WaitGroup", [&] {
        marl::WaitGroup wg(kWgTasks);
        for (int i = 0; i < kWgTasks; ++i)
          sched.enqueue(marl::Task([wg] { wg.done(); }));
        wg.wait();
      });
    }

    {
      hyle::sched::Scheduler sched(kWorkers, system_allocator(), 2048);
      b.run("HYLE WaitGroup + schedule (worker-stack)", [&] {
        hyle::sched::WaitGroup wg;
        wg.add(kWgTasks);
        for (int i = 0; i < kWgTasks; ++i) {
          const auto status = sched.schedule(&hyle_wg_done, &wg);
          ankerl::nanobench::doNotOptimizeAway(status.ok());
        }
        wg.wait();
      });
      sched.shutdown();
    }

    {
      hyle::sched::Scheduler sched(kWorkers, system_allocator(), 2048);
      b.run("HYLE WaitGroup + spawn_detached (pooled stacks)", [&] {
        hyle::sched::WaitGroup wg;
        wg.add(kWgTasks);
        for (int i = 0; i < kWgTasks; ++i) {
          const auto status = sched.spawn_detached(
              &hyle_wg_done, &wg, kDefaultStackSize);
          ankerl::nanobench::doNotOptimizeAway(status.ok());
        }
        wg.wait();
      });
      sched.shutdown();
    }

    {
      hyle::sched::Scheduler sched(kWorkers, system_allocator(), 2048);
      const auto alloc = system_allocator();
      b.run("HYLE WaitGroup + spawn_task (pooled stacks)", [&] {
        hyle::sched::WaitGroup wg;
        wg.add(kWgTasks);
        std::vector<JoinHandle<void>> tasks;
        tasks.reserve(kWgTasks);
        for (int i = 0; i < kWgTasks; ++i) {
          tasks.push_back(hyle::sched::spawn_task(
              sched, alloc, hyle_wg_task(alloc, &wg)));
        }
        wg.wait();
        for (auto& task : tasks) {
          auto result = task.join();
          ankerl::nanobench::doNotOptimizeAway(result.has_value());
        }
      });
      sched.shutdown();
    }
  }

  // ---- 3. pure spawn throughput: 2000 noop tasks (no per-task work) ----
  // Isolates spawn/dispatch cost vs marl (the SOTA scheduler), unlike the
  // scheduler_bench's vs-std::async (a thread-per-task strawman).
  {
    constexpr int kNoop = 2000;
    ankerl::nanobench::Bench b;
    b.title("spawn 2000 noop tasks").relative(true).batch(kNoop).minEpochIterations(20);

    {
      marl::Scheduler::Config cfg;
      cfg.setWorkerThreadCount(kWorkers);
      marl::Scheduler sched(cfg);
      b.run("marl schedule 2000 noop", [&] {
        marl::WaitGroup wg(kNoop);
        for (int i = 0; i < kNoop; ++i)
          sched.enqueue(marl::Task([wg] { wg.done(); }));
        wg.wait();
      });
    }

    {
      hyle::sched::Scheduler sched(kWorkers, system_allocator(), 4096);
      b.run("HYLE schedule 2000 noop (worker-stack)", [&] {
        hyle::sched::WaitGroup wg(kNoop);
        for (int i = 0; i < kNoop; ++i) {
          const auto status = sched.schedule(&hyle_wg_done, &wg);
          ankerl::nanobench::doNotOptimizeAway(status.ok());
        }
        wg.wait();
      });
      sched.shutdown();
    }

    // Same HYLE submit path with Scheduler's native completion barrier. Kept
    // separate because marl exposes no equivalent drain() primitive; the
    // WaitGroup row above is the strict equal-shape comparison.
    {
      hyle::sched::Scheduler sched(kWorkers, system_allocator(), 4096);
      b.run("HYLE schedule 2000 noop + drain", [&] {
        for (int i = 0; i < kNoop; ++i) {
          const auto status = sched.schedule(&hyle_noop, nullptr);
          ankerl::nanobench::doNotOptimizeAway(status.ok());
        }
        sched.drain();
      });
      sched.shutdown();
    }

    {
      hyle::sched::Scheduler sched(kWorkers, system_allocator(), 4096);
      b.run("HYLE spawn_detached 2000 noop (pooled)", [&] {
        hyle::sched::WaitGroup wg(kNoop);
        for (int i = 0; i < kNoop; ++i) {
          const auto status =
              sched.spawn_detached(&hyle_wg_done, &wg, kDefaultStackSize);
          ankerl::nanobench::doNotOptimizeAway(status.ok());
        }
        wg.wait();
      });
      sched.shutdown();
    }

    {
      hyle::sched::Scheduler sched(kWorkers, system_allocator(), 4096);
      const auto alloc = system_allocator();
      b.run("HYLE spawn_task 2000 noop (pooled)", [&] {
        std::vector<JoinHandle<void>> tasks;
        tasks.reserve(kNoop);
        for (int i = 0; i < kNoop; ++i) {
          tasks.push_back(hyle::sched::spawn_task(
              sched, alloc, hyle_noop_task(alloc)));
        }
        for (auto& task : tasks) {
          auto result = task.join();
          ankerl::nanobench::doNotOptimizeAway(result.has_value());
        }
      });
      sched.shutdown();
    }
  }

  return 0;
}
