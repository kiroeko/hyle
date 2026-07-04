#include <gtest/gtest.h>

#include <atomic>
#include <memory>

#include "hyle/mem/allocator.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

// sched-08 graceful shutdown: two-phase drain for long-lived (server) fibers that
// never finish on their own. begin_shutdown() flips a flat "not accepting" flag +
// broadcasts it; a long-lived fiber polls shutdown_requested() at its yield points
// and breaks its loop; drain_for() waits (bounded) for the in-flight count to hit
// zero. No cancellation tree -- a flat scheduler-wide signal, per seastar/tokio.

using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::shutdown_requested;

namespace {

// A long-lived fiber: loops "forever" (like an accept/connection loop), yielding
// each iteration, until it observes shutdown_requested() -- then it breaks and
// finishes (decrementing the scheduler's in-flight count so drain can complete).
struct LongLived {
  std::atomic<bool>* exited;
  std::atomic<int>* iterations;
};
void longlived_entry(void* p) {
  auto* s = static_cast<LongLived*>(p);
  while (!shutdown_requested()) {
    s->iterations->fetch_add(1, std::memory_order_relaxed);
    Fiber::yield();  // a real server fiber would co_await accept/recv here
  }
  s->exited->store(true, std::memory_order_release);  // observed shutdown -> wind down
}

}  // namespace

// A long-lived fiber never finishes on its own; begin_shutdown() makes it observe
// the signal, break, and finish -- so drain_for() completes instead of hanging.
TEST(Shutdown, LongLivedFiberWindsDownOnSignal) {
  auto exited = std::make_unique<std::atomic<bool>>(false);
  auto iters = std::make_unique<std::atomic<int>>(0);
  LongLived body{exited.get(), iters.get()};
  Fiber f(system_allocator(), kDefaultStackSize, &longlived_entry, &body);

  Scheduler sched(2, system_allocator(), 256);
  sched.spawn(&f);

  // Let it spin a bit (prove it's genuinely long-lived: it would never finish).
  while (iters->load(std::memory_order_relaxed) < 5) {}
  EXPECT_FALSE(exited->load());  // still looping -- a plain drain() would hang here

  sched.begin_shutdown();                       // phase 1: broadcast the signal
  EXPECT_TRUE(sched.drain_for(2'000'000'000));  // phase 2: it winds down within 2s
  sched.shutdown();                             // stop + join workers

  EXPECT_TRUE(exited->load());  // the fiber observed shutdown and exited its loop
}

// After begin_shutdown(), new spawns are rejected (not admitted, not counted) so a
// producer can't keep feeding work into a draining scheduler.
TEST(Shutdown, SpawnRejectedAfterBeginShutdown) {
  Scheduler sched(2, system_allocator(), 256);
  EXPECT_TRUE(sched.accepting());

  sched.begin_shutdown();
  EXPECT_FALSE(sched.accepting());

  // This spawn must be a no-op: the fiber never runs. If it were admitted, ran
  // would flip and the count would be non-zero (drain_for would time out).
  std::atomic<bool> ran{false};
  struct B { std::atomic<bool>* ran; };
  B b{&ran};
  auto entry = [](void* p) { static_cast<B*>(p)->ran->store(true); };
  Fiber f(system_allocator(), kDefaultStackSize, entry, &b);
  sched.spawn(&f);

  EXPECT_TRUE(sched.drain_for(500'000'000));  // nothing admitted -> drains at once
  sched.shutdown();
  EXPECT_FALSE(ran.load());  // rejected spawn never executed
}

// drain_for returns false when work does NOT finish within the timeout (a fiber
// that ignores the shutdown signal). Proves the bounded-wait semantics -- the
// caller then escalates (cooperative cancel / abandon), never force-kills a
// stackful fiber.
TEST(Shutdown, DrainForTimesOutOnStuckFiber) {
  // A fiber that ignores shutdown_requested() and loops forever (yielding).
  auto keep_going = std::make_unique<std::atomic<bool>>(true);
  struct Stuck { std::atomic<bool>* go; };
  Stuck s{keep_going.get()};
  auto entry = [](void* p) {
    auto* st = static_cast<Stuck*>(p);
    while (st->go->load(std::memory_order_acquire)) Fiber::yield();
  };
  Fiber f(system_allocator(), kDefaultStackSize, entry, &s);

  Scheduler sched(2, system_allocator(), 256);
  sched.spawn(&f);
  sched.begin_shutdown();
  // The fiber ignores the signal, so drain can't complete -> timeout returns false.
  EXPECT_FALSE(sched.drain_for(100'000'000));  // 100ms budget, fiber won't exit

  // Release it so teardown can finish (real code would abandon it at process exit).
  keep_going->store(false, std::memory_order_release);
  sched.shutdown();
}
