// Scheduler microbenchmarks: HYLE's own core costs, measured honestly.
//  1. fiber context switch (yield) -- the primitive everything else rides on
//  2. spawn+run+complete throughput on the M:N scheduler
//  3. suspend/wake round-trip latency (the sched-04a completion spine)
// Plus a std::async baseline on (2): what the standard task-dispatch primitive
// costs, so the fiber number has a reference. Whatever prints, prints.

#include <nanobench.h>

#include <cstdint>
#include <future>
#include <memory>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/run_queue.h"
#include "hyle/sched/scheduler.h"

using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::RunQueue;
using hyle::sched::Scheduler;

namespace {

void yield_body(void* arg) {
  int n = *static_cast<int*>(arg);
  for (int i = 0; i < n; ++i) Fiber::yield();
}
void noop_body(void*) {}

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

  // ---- wake round-trip latency: DEFERRED (needs a safe harness) ----
  // A naive "spin on a flag the fiber sets, then destroy the fiber" races the
  // worker that is still finishing dispatch(f) after the fiber returned (the
  // tests avoid this via drain(), which waits for outstanding==0). Measuring
  // wake latency in isolation needs a harness that retires the fiber safely
  // (e.g. a long-lived ping-pong fiber pair) -- registered, not shipped racy.
  // Waker correctness itself is covered by waker_test.cc + WSL TSan.
  return 0;
}
