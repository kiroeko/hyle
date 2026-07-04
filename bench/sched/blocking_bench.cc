// sched-11 spawn_blocking benchmark: the blocking-offload pool -- move a blocking
// syscall / CPU-bound closure OFF a scheduler worker onto a dedicated thread pool so
// the worker stays free for its other fibers. sched-11 shipped this with 7 tests but
// no perf number; this file adds it, with the marl SOTA comparison where a fair one
// exists.
//
// SOTA framing (honest): marl has blocking_call() -- the SAME goal (offload a blocking
// closure, don't stall the fiber's worker). But marl's blocking_call spawns a BRAND-NEW
// std::thread per call and joins it (marl/blockingcall.h OnNewThread), NO pool. HYLE's
// spawn_blocking uses a LAZY, REUSED pool (tokio model). So the head-to-head measures
// exactly the pool's payoff: repeated offloads amortize thread creation on HYLE but pay
// it every time on marl. Two rows here:
//   1. HYLE spawn_blocking round-trip (submit trivial closure + join), the pool cost.
//   2. vs marl blocking_call round-trip (gated -DHYLE_BENCH_MARL) -- pool vs new-thread.
// Plus a HYLE-only fan-out row: N concurrent blocking jobs, one joiner -- the pool's
// concurrency (up to max_threads) the single-shot row can't show.
//
// Run under scripts/bench-stat.py -n 15 for the statistical verdict.

#include <nanobench.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/sched/blocking.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/join.h"
#include "hyle/sched/scheduler.h"

#if defined(HYLE_BENCH_MARL)
#  include "marl/blockingcall.h"
#  include "marl/scheduler.h"
#endif

using hyle::mem::system_allocator;
using hyle::sched::BlockingPool;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::spawn_blocking;

namespace {
namespace nb = ankerl::nanobench;

// HYLE: a driver fiber submits `rounds` trivial blocking closures, joining each before
// the next (sequential round-trip: submit -> pool thread runs -> join wakes fiber).
// This measures the per-offload cost with a warm pool (thread creation amortized after
// the first few). The closure is trivial (return an int) so the number is the offload
// machinery, not the work.
struct HyleBlockingArg {
  BlockingPool* pool;
  int rounds;
  std::uint64_t sink;
};
void hyle_blocking_body(void* p) {
  auto* a = static_cast<HyleBlockingArg*>(p);
  std::uint64_t sink = 0;
  for (int i = 0; i < a->rounds; ++i) {
    auto h = spawn_blocking(*a->pool, system_allocator(), [i] { return i * 2; });
    auto r = h.join();
    if (r.has_value()) sink += static_cast<std::uint64_t>(*r);
  }
  a->sink = sink;
}

// HYLE fan-out: submit `n` blocking jobs up front, then join all -- they run
// concurrently on up to max_threads pool threads. Shows the pool servicing many
// in-flight offloads (the concurrency the sequential row can't).
struct HyleFanoutArg {
  BlockingPool* pool;
  int n;
  std::uint64_t sink;
};
void hyle_fanout_body(void* p) {
  auto* a = static_cast<HyleFanoutArg*>(p);
  std::vector<hyle::sched::JoinHandle<int>> handles;
  handles.reserve(a->n);
  for (int i = 0; i < a->n; ++i)
    handles.push_back(spawn_blocking(*a->pool, system_allocator(), [i] { return i; }));
  std::uint64_t sink = 0;
  for (auto& h : handles) {
    auto r = h.join();
    if (r.has_value()) sink += static_cast<std::uint64_t>(*r);
  }
  a->sink = sink;
}

}  // namespace

int main() {
  // ---- 1. HYLE spawn_blocking sequential round-trip (warm pool) ----
  // A driver fiber offloads `kRounds` trivial closures, joining each. batch=kRounds ->
  // ns per submit+run+join round-trip. The pool is created ONCE per epoch (outside the
  // inner loop) so thread-creation is amortized -- this is the steady-state offload
  // cost, the number that matters for a server offloading many blocking calls.
  {
    constexpr int kRounds = 2000;
    nb::Bench b;
    b.title("spawn_blocking round-trip (warm pool)").batch(kRounds).minEpochIterations(10);
    b.run("HYLE spawn_blocking submit+join", [&] {
      BlockingPool pool(system_allocator());
      Scheduler sched(2, system_allocator(), 256);
      HyleBlockingArg a{&pool, kRounds, 0};
      auto f = std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                       &hyle_blocking_body, &a);
      sched.spawn(f.get());
      sched.drain();
      nb::doNotOptimizeAway(a.sink);
    });
  }

  // ---- 2. HYLE fan-out: N concurrent blocking jobs, one joiner ----
  // Submit N up front (they run concurrently on the pool), then join all. Shows the
  // pool's concurrency. Swept over N. batch=N -> ns per concurrent offload.
  {
    nb::Bench b;
    b.title("spawn_blocking fan-out: N concurrent jobs").unit("job");
    for (int n : {8, 64, 256}) {
      char name[64];
      std::snprintf(name, sizeof(name), "HYLE %d concurrent blocking jobs", n);
      b.batch(n).minEpochIterations(5).run(name, [&] {
        BlockingPool pool(system_allocator());
        Scheduler sched(4, system_allocator(), 1024);
        HyleFanoutArg a{&pool, n, 0};
        auto f = std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                         &hyle_fanout_body, &a);
        sched.spawn(f.get());
        sched.drain();
        nb::doNotOptimizeAway(a.sink);
      });
    }
  }

#if defined(HYLE_BENCH_MARL)
  // ---- 3. SOTA: HYLE pooled offload vs marl blocking_call (new thread each) ----
  // Same shape: a task offloads kRounds trivial closures sequentially. marl's
  // blocking_call spawns+joins a fresh std::thread per call; HYLE reuses pool threads.
  // The ratio is the pool's payoff. marl needs its own scheduler bound on the task.
  {
    constexpr int kRounds = 500;  // fewer: marl pays a full thread create+join each
    nb::Bench b;
    b.title("blocking offload round-trip: HYLE pool vs marl new-thread")
        .relative(true)
        .batch(kRounds)
        .minEpochIterations(5);

    b.run("HYLE spawn_blocking (pooled)", [&] {
      BlockingPool pool(system_allocator());
      Scheduler sched(4, system_allocator(), 1024);
      HyleBlockingArg a{&pool, kRounds, 0};
      auto f = std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                       &hyle_blocking_body, &a);
      sched.spawn(f.get());
      sched.drain();
      nb::doNotOptimizeAway(a.sink);
    });

    b.run("marl blocking_call (new thread each)", [&] {
      marl::Scheduler::Config cfg;
      cfg.setWorkerThreadCount(4);
      marl::Scheduler marl_sched(cfg);
      marl_sched.bind();
      marl::WaitGroup done(1);
      std::uint64_t sink = 0;
      marl::schedule([&] {
        for (int i = 0; i < kRounds; ++i) {
          int r = marl::blocking_call([i] { return i * 2; });
          sink += static_cast<std::uint64_t>(r);
        }
        done.done();
      });
      done.wait();
      nb::doNotOptimizeAway(sink);
      marl::Scheduler::unbind();
    });
  }
#endif

  return 0;
}
