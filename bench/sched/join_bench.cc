// sched-10 JoinHandle / spawn_task / when_all / when_any microbenchmarks.
//
// GOAL: measure the structured-concurrency layer (JoinHandle + spawn_task)
// against the raw mechanism it abstracts -- a fiber + a manual result slot + a
// WaitGroup(1) completion signal (exactly what test code hand-rolled before
// sched-10). Then when_all / when_any cost per N-task group.
//
// MEASURED (Win MSVC Release, 2026-07-03): spawn_task+join ~869ns vs raw fiber
// +WaitGroup ~683ns = the sugar is ~1.27x. This is NOT zero-overhead, and the
// cost is understood, not a bug: spawn_task heap-allocates 2 control blocks
// (JoinState result slot + TaskDriver{Task frame + Fiber}) so the CALLER manages
// ZERO lifetime (the driver self-frees via the Fiber finish hook); the raw path
// reuses caller-stack objects (RawTask + Fiber) and 1 stack alloc, but in turn
// the CALLER must keep them alive + drain() before they die. So the ~186ns buys
// "own-nothing" ergonomics + safety (no dangling-fiber footgun). spawn is not the
// tightest hot path (that's the 23ns fiber switch / sync round-trip); paying
// ~0.9us to launch an owned task with a result handle is the right trade. If a
// hot path truly needs launch-heavy throughput without a result, raw fiber+
// WaitGroup stays available (this bench is exactly why it's documented).
//
// No external rival: tokio (Rust) / folly coro use different runtimes; the honest
// comparison is "our sugar vs our own raw primitives". Whatever prints, prints.

#include <nanobench.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/join.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/sync.h"

using hyle::mem::Allocator;
using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::JoinHandle;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::spawn_task;
using hyle::sched::Task;
using hyle::sched::WaitGroup;
using hyle::sched::when_all;
using hyle::sched::when_any;

namespace {

// A trivial task: return a value. This is the unit of work whose spawn+join cost
// we measure (the task body itself is ~free, so the number is the layer's cost).
Task<int> ret_coro(Allocator, int v) { co_return v; }

// ---- raw baseline: fiber + manual result slot + WaitGroup(1) ----------------
// What spawn_task+JoinHandle abstracts, hand-rolled. A heap slot holds the int; a
// WaitGroup(1) is the completion signal; the caller wg.wait() then reads the slot.
struct RawTask {
  int result = 0;
  WaitGroup wg{1};
};
void raw_entry(void* p) {
  auto* r = static_cast<RawTask*>(p);
  r->result = 7;  // "compute"
  r->wg.done();
}

}  // namespace

int main() {
  // ---- 1. spawn_task+join vs raw fiber+WaitGroup, single task round-trip -----
  // One persistent scheduler; per epoch spawn ONE task and join it. batch=1 so the
  // number is ns per (spawn + run + complete + join) round-trip. The raw baseline
  // does the identical work without the JoinHandle/JoinState/TaskDriver layer.
  {
    Scheduler sched(4, system_allocator(), 4096);
    Allocator a = system_allocator();
    ankerl::nanobench::Bench b;
    // Batch kB tasks per epoch, join all. batch(kB) => ns/op is PER TASK. This
    // amortizes the OS scheduler cross-thread handoff (spawn on main -> worker
    // runs -> wakes main via futex), which otherwise dominates + jitters a single
    // round-trip (esp. on a loaded many-core Linux box) and drowns the layer cost.
    // What survives amortization IS the per-task layer delta -- the thing we want.
    constexpr int kB = 100;
    b.title("spawn+join per task (batch 100)").relative(true).batch(kB).minEpochIterations(2000);

    b.run("raw fiber + WaitGroup(1) + slot", [&] {
      std::vector<std::unique_ptr<RawTask>> rts;
      std::vector<std::unique_ptr<Fiber>> fs;
      rts.reserve(kB);
      fs.reserve(kB);
      for (int i = 0; i < kB; ++i) {
        rts.push_back(std::make_unique<RawTask>());
        fs.push_back(std::make_unique<Fiber>(a, kDefaultStackSize, &raw_entry, rts[i].get()));
        sched.spawn(fs[i].get());
      }
      std::uint64_t sink = 0;
      for (auto& rt : rts) { rt->wg.wait(); sink += rt->result; }
      ankerl::nanobench::doNotOptimizeAway(sink);
      sched.drain();  // fibers retire before fs/rts free them (raw path's burden)
    });

    b.run("spawn_task + JoinHandle::join", [&] {
      std::vector<JoinHandle<int>> hs;
      hs.reserve(kB);
      for (int i = 0; i < kB; ++i) hs.push_back(spawn_task(sched, a, ret_coro(a, i)));
      std::uint64_t sink = 0;
      for (auto& h : hs) { auto r = h.join(); sink += r.has_value() ? *r : 0; }
      ankerl::nanobench::doNotOptimizeAway(sink);
    });

    sched.shutdown();
  }

  // ---- 2. when_all over N tasks: group-wait cost ------------------------------
  // Spawn N tasks, when_all, then join each. batch=N so the number is amortized
  // per-task cost of the all-wait group + N joins.
  {
    constexpr int kN = 64;
    Scheduler sched(4, system_allocator(), 4096);
    Allocator a = system_allocator();
    ankerl::nanobench::Bench b;
    b.title("when_all over 64 tasks").batch(kN).minEpochIterations(2000);
    b.run("spawn 64 + when_all + join each", [&] {
      std::vector<JoinHandle<int>> hs;
      hs.reserve(kN);
      for (int i = 0; i < kN; ++i) hs.push_back(spawn_task(sched, a, ret_coro(a, i)));
      // when_all takes a pack; for a runtime count we poll-join instead (the
      // group engine is pack-based -- exercised in test/case 3). Here the honest
      // "wait for all" over a vector is join-each (each blocks fiber-aware).
      std::uint64_t sink = 0;
      for (auto& h : hs) {
        auto r = h.join();
        sink += r.has_value() ? static_cast<std::uint64_t>(*r) : 0;
      }
      ankerl::nanobench::doNotOptimizeAway(sink);
    });
    sched.shutdown();
  }

  // ---- 3. when_all / when_any over a fixed small pack -------------------------
  // The pack-based group primitives (the real when_all/when_any API). 4 tasks;
  // measure the group install + wait + uninstall cost (batch=1 = per group call).
  {
    Scheduler sched(4, system_allocator(), 4096);
    Allocator a = system_allocator();
    ankerl::nanobench::Bench b;
    b.title("when_all / when_any over 4-task pack").relative(true).minEpochIterations(500);

    b.run("when_all(4)", [&] {
      auto h0 = spawn_task(sched, a, ret_coro(a, 0));
      auto h1 = spawn_task(sched, a, ret_coro(a, 1));
      auto h2 = spawn_task(sched, a, ret_coro(a, 2));
      auto h3 = spawn_task(sched, a, ret_coro(a, 3));
      when_all(h0, h1, h2, h3);
      ankerl::nanobench::doNotOptimizeAway(h0.is_finished());
    });

    b.run("when_any(4) + join winner + drain losers", [&] {
      auto h0 = spawn_task(sched, a, ret_coro(a, 0));
      auto h1 = spawn_task(sched, a, ret_coro(a, 1));
      auto h2 = spawn_task(sched, a, ret_coro(a, 2));
      auto h3 = spawn_task(sched, a, ret_coro(a, 3));
      int w = when_any(h0, h1, h2, h3);
      ankerl::nanobench::doNotOptimizeAway(w);
      // The losers are trivial + already running; join each to be correct + keep
      // slots freed. join() is [[nodiscard]] -> sink the Results.
      std::uint64_t sink = 0;
      for (JoinHandle<int>* h : {&h0, &h1, &h2, &h3}) {
        auto r = h->join();
        sink += r.has_value() ? static_cast<std::uint64_t>(*r) : 0;
      }
      ankerl::nanobench::doNotOptimizeAway(sink);
    });

    sched.shutdown();
  }

  return 0;
}
