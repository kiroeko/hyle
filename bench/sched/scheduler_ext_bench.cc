// Scheduler-layer gap benchmarks: the four sched behaviours the earlier benches
// left uncovered (scheduler_bench = raw primitives; queue_sota_bench already
// covers the work-stealing DATA STRUCTURE, i.e. WsFifoQueue::steal_into micro
// cost). These four measure SCHEDULER-LEVEL behaviour instead, so they are
// complementary, not duplicative:
//
//  1. WORK-STEALING SCALING (sched-03): a single thread injects a batch of
//     compute fibers into the global queue; 1/2/4/8 workers steal + run them in
//     parallel. Measures whether stealing actually spreads a point-injected load
//     across cores (the whole point of sched-03), reported as a scaling curve.
//  2. spawn_on AFFINITY vs spawn: the pinned route (private per-worker queue +
//     unpark_specific, bypassing the anti-herd notify() gate + never stolen) vs
//     the ordinary stealable route (global inject + conditional notify). The two
//     dispatch paths, head to head.
//  3. COROUTINE GLUE (sched-06): `co_await SuspendFiber` ping-pong vs the raw
//     suspend()/Waker::wake() ping-pong (the identical shape from
//     scheduler_bench). Isolates what the coroutine frame adds over the bare
//     fiber spine -- "our sugar vs our own raw primitive", no external rival.
//  4. FIBER SLEEP (sched-04b): the timing-wheel sleep spine. A single sleeper vs
//     N sleepers concurrent on ONE worker -- if fiber sleep does NOT burn its
//     worker, N concurrent sleeps of duration D finish in ~D, not N*D. This is
//     the property that distinguishes fiber sleep from std::this_thread::sleep_for
//     (which blocks a whole OS thread). Dominated by the OS timer floor (1ms
//     wheel tick, ~15ms on Windows), so the number is wall-clock, not pure
//     scheduler cost -- the CONTRAST between the two rows is the result.
//
// Env: Win MSVC Release + Linux GCC14 Release (native ext4), nanobench. Sub-us
// sched numbers are dominated by OS cross-thread wakeup jitter (spread 30-40%),
// so every ratio here MUST be re-checked with scripts/bench-stat.py across 15
// runs on both platforms -- a single-shot number is a coin flip (the lesson
// spawn/WaitGroup "2.17x/1.49x" taught: single-shot noise masquerading as a win).

#include <nanobench.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/timer.h"
#include "hyle/sched/waker.h"

using hyle::mem::Allocator;
using hyle::mem::system_allocator;
using hyle::sched::CaptureWaker;
using hyle::sched::current_waker;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::sleep_for_ns;
using hyle::sched::suspend;
using hyle::sched::SuspendFiber;
using hyle::sched::Task;
using hyle::sched::Waker;

namespace {

// ---- (1) work-stealing scaling: per-fiber CPU work, per-fiber result slot ----
// xorshift spin so the compiler can't fold the work away; each fiber writes its
// OWN slot (no cross-worker atomic sharing) so the scaling number reflects
// parallel compute + steal spread, not contention on a shared counter.
std::uint64_t spin(std::uint64_t x, int rounds) noexcept {
  for (int i = 0; i < rounds; ++i) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
  }
  return x;
}
struct WorkArg {
  std::uint64_t out;
  int rounds;
};
void work_body(void* p) noexcept {
  auto* a = static_cast<WorkArg*>(p);
  // Seed from the arg address so each fiber's work differs (and is non-constant).
  a->out = spin(reinterpret_cast<std::uintptr_t>(p) | 1u, a->rounds);
}

// Pure dispatch (no work): isolates the routing/wake path for the spawn_on bench.
void noop_body(void*) noexcept {}

// ---- (3) coroutine-glue ping-pong (mirrors scheduler_bench's PingPong) -------
// Two fibers bounce `cycles` suspend/wake round-trips. The RAW variant uses
// suspend()/current_waker() directly; the CORO variant uses co_await
// SuspendFiber{}/CaptureWaker{} -- identical handshake, one wrapped in coroutine
// syntax. Both run on ONE worker (true hand-off, not parallel). The published-
// Waker + ready-flag rendezvous is the proven waker_test pattern.
struct PingPong {
  std::atomic<bool> a_ready{false};
  std::atomic<bool> b_ready{false};
  Waker a_waker;
  Waker b_waker;
  int cycles;
};

// -- raw fiber spine --
void raw_a(void* p) {
  auto* s = static_cast<PingPong*>(p);
  for (int i = 0; i < s->cycles; ++i) {
    s->a_waker = current_waker();
    s->a_ready.store(true, std::memory_order_release);
    suspend();  // wait for B
    while (!s->b_ready.load(std::memory_order_acquire)) Fiber::yield();
    s->b_ready.store(false, std::memory_order_relaxed);
    s->b_waker.wake();
  }
}
void raw_b(void* p) {
  auto* s = static_cast<PingPong*>(p);
  for (int i = 0; i < s->cycles; ++i) {
    while (!s->a_ready.load(std::memory_order_acquire)) Fiber::yield();
    s->a_ready.store(false, std::memory_order_relaxed);
    s->a_waker.wake();
    s->b_waker = current_waker();
    s->b_ready.store(true, std::memory_order_release);
    suspend();  // wait for A
  }
}

// -- coroutine spine (same shape; co_await replaces the bare calls) --
Task<void> coro_a(Allocator, PingPong* s) {
  for (int i = 0; i < s->cycles; ++i) {
    s->a_waker = co_await CaptureWaker{};
    s->a_ready.store(true, std::memory_order_release);
    co_await SuspendFiber{};  // wait for B
    while (!s->b_ready.load(std::memory_order_acquire)) Fiber::yield();
    s->b_ready.store(false, std::memory_order_relaxed);
    s->b_waker.wake();
  }
}
Task<void> coro_b(Allocator, PingPong* s) {
  for (int i = 0; i < s->cycles; ++i) {
    while (!s->a_ready.load(std::memory_order_acquire)) Fiber::yield();
    s->a_ready.store(false, std::memory_order_relaxed);
    s->a_waker.wake();
    s->b_waker = co_await CaptureWaker{};
    s->b_ready.store(true, std::memory_order_release);
    co_await SuspendFiber{};  // wait for A
  }
}
// Drives a Task to completion. One resume() runs the whole coroutine: a co_await
// freezes the entire call (incl. resume()) on the fiber stack and SuspendFiber's
// await_suspend returns false so it CONTINUES on re-inject -- so resume() returns
// only once the coroutine has finished (see coro_test.cc).
struct CoroDrive {
  Task<void> task;
};
void coro_drive(void* p) { static_cast<CoroDrive*>(p)->task.resume(); }

// ---- (4) fiber sleep -----------------------------------------------------
struct SleepArg {
  std::uint64_t ns;
};
void sleep_body(void* p) { sleep_for_ns(static_cast<SleepArg*>(p)->ns); }

}  // namespace

int main() {
  // ---- 1. work-stealing scaling: 1 injector, N workers steal + run ----------
  // Main thread (a NON-worker) spawns kTasks compute fibers -> they all land on
  // the global inject queue (ShardedQueue, round-robin across shards); the N
  // workers pop their shard + steal from each other to run them in parallel.
  // relative(true) with the 1-worker row as the 100% baseline, so the 2/4/8 rows
  // print the scaling factor. batch(kTasks) => ns/op is per task. Each fiber does
  // `kRounds` xorshift rounds (~a couple us) so the number reflects parallel
  // COMPUTE spread, not the ~sub-us dispatch cost (which would hide scaling).
  // NOTE: scaling is capped by physical cores -- on a box with <8 cores the 8-
  // worker row won't be 8x. The shape (does stealing spread the load at all?) is
  // the result; the exact factor is machine-bound.
  {
    constexpr int kTasks = 512;
    constexpr int kRounds = 2000;  // ~a few us of work per fiber
    ankerl::nanobench::Bench b;
    b.title("work-stealing scaling (512 compute fibers, 1 injector)")
        .relative(true)
        .batch(kTasks)
        .minEpochIterations(20);
    for (unsigned w : {1u, 2u, 4u, 8u}) {
      Scheduler sched(w, system_allocator(), 1024);
      std::string name = "HYLE " + std::to_string(w) + " worker(s) steal+run";
      b.run(name, [&] {
        std::vector<WorkArg> args(kTasks);
        std::vector<std::unique_ptr<Fiber>> fibers;
        fibers.reserve(kTasks);
        for (int i = 0; i < kTasks; ++i) {
          args[i].rounds = kRounds;
          fibers.push_back(std::make_unique<Fiber>(
              system_allocator(), kDefaultStackSize, &work_body, &args[i]));
        }
        for (auto& f : fibers) sched.spawn(f.get());
        sched.drain();
        std::uint64_t sink = 0;
        for (auto& a : args) sink += a.out;
        ankerl::nanobench::doNotOptimizeAway(sink);
      });
      sched.shutdown();
    }
  }

  // ---- 2. spawn_on affinity vs ordinary spawn (4 workers, noop tasks) --------
  // Two dispatch ROUTES, head to head, same 4-worker scheduler:
  //  - spawn: lands on the global inject queue; wakes a worker only via the
  //    conditional notify() gate (skipped if a searcher already exists); the
  //    fiber is stealable.
  //  - spawn_on: lands on a specific worker's PRIVATE pinned queue; unparks that
  //    worker specifically (unconditional, bypasses the anti-herd gate); never
  //    stolen. Round-robin the target so all 4 workers get load.
  // noop bodies isolate the routing/wake cost (no per-task work). batch(kTasks).
  {
    constexpr int kTasks = 2000;
    constexpr unsigned kW = 4;
    ankerl::nanobench::Bench b;
    b.title("spawn vs spawn_on route (2000 noop tasks/4 workers)")
        .relative(true)
        .batch(kTasks)
        .minEpochIterations(20);

    Scheduler sched_spawn(kW, system_allocator(), 4096);
    b.run("HYLE spawn (global inject, stealable)", [&] {
      std::vector<std::unique_ptr<Fiber>> fibers;
      fibers.reserve(kTasks);
      for (int i = 0; i < kTasks; ++i)
        fibers.push_back(std::make_unique<Fiber>(
            system_allocator(), kDefaultStackSize, &noop_body, nullptr));
      for (auto& f : fibers) sched_spawn.spawn(f.get());
      sched_spawn.drain();
    });
    sched_spawn.shutdown();

    Scheduler sched_on(kW, system_allocator(), 4096);
    b.run("HYLE spawn_on (pinned, unpark_specific)", [&] {
      std::vector<std::unique_ptr<Fiber>> fibers;
      fibers.reserve(kTasks);
      for (int i = 0; i < kTasks; ++i)
        fibers.push_back(std::make_unique<Fiber>(
            system_allocator(), kDefaultStackSize, &noop_body, nullptr));
      for (int i = 0; i < kTasks; ++i)
        sched_on.spawn_on(static_cast<unsigned>(i) % kW, fibers[i].get());
      sched_on.drain();
    });
    sched_on.shutdown();
  }

  // ---- 3. coroutine glue: co_await ping-pong vs raw suspend/wake ping-pong ---
  // Same suspend/wake round-trip shape (scheduler_bench's PingPong), once with
  // bare suspend()/current_waker(), once with co_await SuspendFiber{}/
  // CaptureWaker{}. relative(true) so the coro row prints the overhead the
  // coroutine frame adds over the raw spine. 1 worker = genuine hand-off.
  // batch(kCycles) => ns per suspend->wake->resume round-trip.
  {
    constexpr int kCycles = 5000;
    ankerl::nanobench::Bench b;
    b.title("suspend/wake ping-pong: raw fiber vs co_await")
        .relative(true)
        .batch(kCycles)
        .minEpochIterations(20);

    b.run("raw suspend()+Waker::wake()", [&] {
      PingPong s;
      s.cycles = kCycles;
      Fiber fa(system_allocator(), kDefaultStackSize, &raw_a, &s);
      Fiber fb(system_allocator(), kDefaultStackSize, &raw_b, &s);
      Scheduler sched(1, system_allocator(), 256);
      sched.spawn(&fa);
      sched.spawn(&fb);
      sched.drain();
      ankerl::nanobench::doNotOptimizeAway(&s);
    });

    b.run("co_await SuspendFiber/CaptureWaker", [&] {
      PingPong s;
      s.cycles = kCycles;
      Allocator a = system_allocator();
      auto da = std::make_unique<CoroDrive>();
      auto db = std::make_unique<CoroDrive>();
      da->task = coro_a(a, &s);
      db->task = coro_b(a, &s);
      Fiber fa(a, kDefaultStackSize, &coro_drive, da.get());
      Fiber fb(a, kDefaultStackSize, &coro_drive, db.get());
      Scheduler sched(1, system_allocator(), 256);
      sched.spawn(&fa);
      sched.spawn(&fb);
      sched.drain();
      ankerl::nanobench::doNotOptimizeAway(&s);
    });
  }

  // ---- 4. fiber sleep: 1 sleeper vs N concurrent sleepers on ONE worker ------
  // The sched-04b timing-wheel spine. Both rows run on a SINGLE worker and sleep
  // the SAME duration D; if fiber sleep does not burn the worker, N concurrent
  // sleepers wake in ~D (one wheel tick fires them all), NOT N*D. So the two rows
  // being ~equal IS the result: proof that one worker carries many concurrent
  // sleepers without serializing them (the property std::this_thread::sleep_for
  // -- which blocks a whole OS thread -- cannot give). batch(1) = whole batch is
  // one op (the wall-clock to spawn the sleepers + wait for all to wake).
  // Dominated by the OS timer floor (1ms wheel tick; ~15ms timer granularity on
  // Windows), so these are wall-clock ms, not scheduler ns -- the CONTRAST (1 vs
  // N ~equal) is what matters, not the absolute value. minEpochIterations(1) +
  // few epochs since each epoch is a real sleep.
  {
    constexpr std::uint64_t kSleepNs = 2'000'000;  // 2ms (> 1ms wheel tick)
    constexpr int kN = 64;
    ankerl::nanobench::Bench b;
    b.title("fiber sleep: 1 vs 64 concurrent sleepers (1 worker)")
        .relative(true)
        .batch(1)
        .minEpochIterations(1);

    b.run("1 fiber sleep 2ms", [&] {
      SleepArg arg{kSleepNs};
      Fiber f(system_allocator(), kDefaultStackSize, &sleep_body, &arg);
      Scheduler sched(1, system_allocator(), 256);
      sched.spawn(&f);
      sched.drain();
    });

    b.run("64 fibers sleep 2ms (concurrent, 1 worker)", [&] {
      SleepArg arg{kSleepNs};
      std::vector<std::unique_ptr<Fiber>> fibers;
      fibers.reserve(kN);
      for (int i = 0; i < kN; ++i)
        fibers.push_back(std::make_unique<Fiber>(
            system_allocator(), kDefaultStackSize, &sleep_body, &arg));
      Scheduler sched(1, system_allocator(), 256);
      for (auto& f : fibers) sched.spawn(f.get());
      sched.drain();  // all 64 must wake before this returns
    });
  }

  return 0;
}
