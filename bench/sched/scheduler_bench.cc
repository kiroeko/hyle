// Scheduler microbenchmarks: HYLE's own core costs, measured honestly.
//  1. fiber context switch (yield) -- the primitive everything else rides on
//  2. spawn+run+complete throughput on the M:N scheduler
//  3. suspend/wake round-trip latency (the sched-04a completion spine)
// Plus a std::async baseline on (2): what the standard task-dispatch primitive
// costs, so the fiber number has a reference. Whatever prints, prints.

#include <nanobench.h>

#include <atomic>
#include <cstdint>
#include <future>
#include <memory>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/run_queue.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/waker.h"

using hyle::mem::system_allocator;
using hyle::sched::current_waker;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::RunQueue;
using hyle::sched::Scheduler;
using hyle::sched::suspend;
using hyle::sched::Waker;

namespace {

void yield_body(void* arg) {
  int n = *static_cast<int*>(arg);
  for (int i = 0; i < n; ++i) Fiber::yield();
}
void noop_body(void*) {}

// ---- suspend/wake completion-spine ping-pong (the safe harness) -------------
// Two long-lived fibers bounce N suspend/wake cycles between each other, all
// under one Scheduler, retired together by drain() -- so there is NO race between
// "fiber finished" and "worker still in dispatch()" (the hazard that kept this
// DEFERRED for a naive flag-spin+destroy harness). Each cycle is a full
// completion-spine round-trip: fiber A publishes its Waker + suspends (leaves the
// run loop); fiber B sees the Waker and calls wake() (re-injects A); then B
// suspends and A wakes it back. The published-Waker handshake (acquire/release on
// a flag) mirrors waker_test.cc's proven pattern. This measures the suspend ->
// wake -> resume latency that ops.h's thesis (GPU/NIC/timer all funnel here) rides
// on. No external rival -- it's HYLE's own spine; the number is the cost itself.
struct PingPong {
  std::atomic<bool> a_ready{false};
  std::atomic<bool> b_ready{false};
  Waker a_waker;
  Waker b_waker;
  int cycles;
};
void pingpong_a(void* p) {
  auto* s = static_cast<PingPong*>(p);
  for (int i = 0; i < s->cycles; ++i) {
    s->a_waker = current_waker();
    s->a_ready.store(true, std::memory_order_release);
    suspend();  // wait for B to wake us
    // Once resumed, wake B (it has published its waker by now, or will; B's
    // wake-before-suspend rendezvous makes an early wake safe).
    while (!s->b_ready.load(std::memory_order_acquire)) Fiber::yield();
    s->b_ready.store(false, std::memory_order_relaxed);
    s->b_waker.wake();
  }
}
void pingpong_b(void* p) {
  auto* s = static_cast<PingPong*>(p);
  for (int i = 0; i < s->cycles; ++i) {
    // Wake A (it published its waker first), then suspend waiting for A's wake.
    while (!s->a_ready.load(std::memory_order_acquire)) Fiber::yield();
    s->a_ready.store(false, std::memory_order_relaxed);
    s->a_waker.wake();
    s->b_waker = current_waker();
    s->b_ready.store(true, std::memory_order_release);
    suspend();  // wait for A to wake us back
  }
}

}  // namespace

int main() {
  // ---- 1. fiber context switch (yield round-trip) ----
  {
    constexpr int kY = 1000;
    ankerl::nanobench::Bench b;
    b.title("fiber context switch (resume/yield pair)").batch(kY).minEpochIterations(200);
    b.run("Fiber::yield", [&] {
      int n = kY;
      Fiber f(system_allocator(), kDefaultStackSize, &yield_body, &n);
      while (!f.done()) f.resume();
    });
  }

  // ---- 2. spawn+run+complete throughput: HYLE scheduler vs std::async ----
  // One persistent Scheduler (the realistic model: an app makes one), spawn+drain
  // a fresh batch of one-shot fibers per epoch. (Rapid Scheduler create/destroy
  // churn is a separate robustness question, not measured here.)
  {
    constexpr int kTasks = 2000;
    Scheduler sched(4, system_allocator(), 4096);
    ankerl::nanobench::Bench b;
    b.title("spawn 2000 trivial tasks").relative(true).batch(kTasks).minEpochIterations(50);
    b.run("std::async(launch::async)", [&] {
      std::vector<std::future<void>> fs;
      fs.reserve(kTasks);
      for (int i = 0; i < kTasks; ++i)
        fs.push_back(std::async(std::launch::async, [] {}));
      for (auto& f : fs) f.get();
    });
    b.run("HYLE fiber spawn+drain", [&] {
      std::vector<std::unique_ptr<Fiber>> fibers;
      fibers.reserve(kTasks);
      for (int i = 0; i < kTasks; ++i)
        fibers.push_back(std::make_unique<Fiber>(system_allocator(),
                                                 kDefaultStackSize, &noop_body,
                                                 nullptr));
      for (auto& f : fibers) sched.spawn(f.get());
      sched.drain();  // all fibers retired before `fibers` frees them
    });
    sched.shutdown();
  }

  // ---- 4. RunQueue schedule/next: the per-core LIFO-slot + FIFO ring ----------
  // The worker's hot inner loop: push a runnable fiber, pick the next. schedule_
  // fifo (ring push) + next (LIFO-slot-then-FIFO pick) are owner-only pointer ops.
  // No external rival (it's an internal scheduler structure); measured against
  // itself to expose the per-op cost and the LIFO-budget path. Fiber* are opaque
  // here (RunQueue never derefs them), so dummy non-null addresses are valid.
  {
    constexpr int kOps = 4096;
    // Distinct non-null dummy "fibers" -- RunQueue stores the pointers, never
    // dereferences them. Use an array of bytes and hand out distinct addresses.
    static char slots[64];
    ankerl::nanobench::Bench b;
    b.title("RunQueue schedule_fifo+next round-trip").relative(true).batch(kOps).minEpochIterations(2000);
    b.run("HYLE RunQueue fifo push+next", [&] {
      RunQueue rq(system_allocator(), 256);
      std::uint64_t sink = 0;
      for (int i = 0; i < kOps; ++i) {
        rq.schedule_fifo(reinterpret_cast<Fiber*>(&slots[i & 63]));
        Fiber* f = rq.next();
        sink += reinterpret_cast<std::uintptr_t>(f);
      }
      ankerl::nanobench::doNotOptimizeAway(sink);
    });
    // LIFO-slot path: schedule_lifo (run-next) then next -- the completion-wake
    // hot path (a just-woken fiber resumes immediately, stack hot).
    b.run("HYLE RunQueue lifo push+next", [&] {
      RunQueue rq(system_allocator(), 256);
      std::uint64_t sink = 0;
      for (int i = 0; i < kOps; ++i) {
        rq.schedule_lifo(reinterpret_cast<Fiber*>(&slots[i & 63]));
        Fiber* f = rq.next();
        sink += reinterpret_cast<std::uintptr_t>(f);
      }
      ankerl::nanobench::doNotOptimizeAway(sink);
    });
  }

  // ---- 5. suspend/wake completion-spine round-trip (the sched-04a spine) ------
  // Now benched via a SAFE long-lived ping-pong harness (see PingPong above): two
  // fibers exchange `kCycles` suspend/wake round-trips under one Scheduler,
  // retired by drain(). One epoch = kCycles full spine round-trips; batch=kCycles
  // so the number is ns per suspend->wake->resume. Single worker (1 thread) so the
  // two fibers genuinely hand off via suspend/wake, not run in parallel.
  {
    constexpr int kCycles = 5000;
    ankerl::nanobench::Bench b;
    b.title("suspend/wake completion-spine round-trip").batch(kCycles).minEpochIterations(20);
    b.run("HYLE suspend()+Waker::wake() ping-pong", [&] {
      PingPong s;
      s.cycles = kCycles;
      Fiber fa(system_allocator(), kDefaultStackSize, &pingpong_a, &s);
      Fiber fb(system_allocator(), kDefaultStackSize, &pingpong_b, &s);
      Scheduler sched(1, system_allocator(), 256);  // 1 worker: true hand-off
      sched.spawn(&fa);
      sched.spawn(&fb);
      sched.drain();  // both fibers retire here -- no flag-spin+destroy race
      ankerl::nanobench::doNotOptimizeAway(&s);
    });
  }

  return 0;
}
