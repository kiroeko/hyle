#include <gtest/gtest.h>

#include <atomic>

#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/join.h"
#include "hyle/sched/scheduler.h"

using hyle::mem::Allocator;
using hyle::mem::system_allocator;
using hyle::sched::is_cancel_requested;
using hyle::sched::JoinHandle;
using hyle::sched::Scheduler;
using hyle::sched::spawn_task;
using hyle::sched::SuspendFiber;
using hyle::sched::CaptureWaker;
using hyle::sched::Task;
using hyle::sched::Waker;
using hyle::sched::when_all;
using hyle::sched::when_any;

namespace {

Task<int> add_coro(Allocator, int a, int b) { co_return a + b; }

Task<void> bump_coro(Allocator, std::atomic<int>* c) {
  c->fetch_add(1, std::memory_order_relaxed);
  co_return;
}

// A coroutine that suspends until woken, publishing its Waker for a foreign wake.
struct WaitBox {
  std::atomic<bool> ready{false};
  Waker waker;
};
Task<int> wait_then(Allocator, WaitBox* box, int ret) {
  box->waker = co_await CaptureWaker{};
  box->ready.store(true, std::memory_order_release);
  co_await SuspendFiber{};
  co_return ret;
}

}  // namespace

// A spawn_task result can be joined for its value.
TEST(Join, SpawnTaskJoinReturnsValue) {
  Scheduler sched(2, system_allocator(), 256);
  auto h = spawn_task(sched, system_allocator(), add_coro(system_allocator(), 20, 22));
  auto r = h.join();
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(*r, 42);
  sched.shutdown();
}

// A void task joins to a valid empty Result.
TEST(Join, VoidTaskJoins) {
  std::atomic<int> c{0};
  Scheduler sched(2, system_allocator(), 256);
  auto h = spawn_task(sched, system_allocator(), bump_coro(system_allocator(), &c));
  auto r = h.join();
  EXPECT_TRUE(r.has_value());
  EXPECT_EQ(c.load(), 1);
  sched.shutdown();
}

// Detaching a handle lets the task run to completion unjoined (drain observes it).
TEST(Join, DetachRunsToCompletion) {
  std::atomic<int> c{0};
  {
    Scheduler sched(2, system_allocator(), 256);
    auto h = spawn_task(sched, system_allocator(), bump_coro(system_allocator(), &c));
    h.detach();
    sched.drain();  // task still counted in `outstanding`, so this waits for it
    sched.shutdown();
  }
  EXPECT_EQ(c.load(), 1);
}

// Letting the handle drop (no join, no explicit detach) is also detach.
TEST(Join, DropIsDetach) {
  std::atomic<int> c{0};
  {
    Scheduler sched(2, system_allocator(), 256);
    { auto h = spawn_task(sched, system_allocator(), bump_coro(system_allocator(), &c)); }
    sched.drain();
    sched.shutdown();
  }
  EXPECT_EQ(c.load(), 1);
}

// when_all waits for every task; each is joinable for its value afterwards.
TEST(Join, WhenAllWaitsForEvery) {
  Scheduler sched(4, system_allocator(), 256);
  auto a = spawn_task(sched, system_allocator(), add_coro(system_allocator(), 1, 2));
  auto b = spawn_task(sched, system_allocator(), add_coro(system_allocator(), 3, 4));
  auto d = spawn_task(sched, system_allocator(), add_coro(system_allocator(), 5, 6));
  when_all(a, b, d);
  EXPECT_TRUE(a.is_finished());
  EXPECT_TRUE(b.is_finished());
  EXPECT_TRUE(d.is_finished());
  EXPECT_EQ(*a.join(), 3);
  EXPECT_EQ(*b.join(), 7);
  EXPECT_EQ(*d.join(), 11);
  sched.shutdown();
}

// when_all over already-completed tasks (install-time pre-accounting path).
TEST(Join, WhenAllAllReadyBeforeWait) {
  Scheduler sched(4, system_allocator(), 256);
  auto a = spawn_task(sched, system_allocator(), add_coro(system_allocator(), 1, 1));
  auto b = spawn_task(sched, system_allocator(), add_coro(system_allocator(), 2, 2));
  // Give the tasks time to finish before we install the group.
  while (!a.is_finished() || !b.is_finished()) {}
  when_all(a, b);  // both already done -> pre-accounted, returns immediately
  EXPECT_EQ(*a.join(), 2);
  EXPECT_EQ(*b.join(), 4);
  sched.shutdown();
}

// when_any returns the index of a completed task; losers are left intact.
TEST(Join, WhenAnyReturnsAWinner) {
  Scheduler sched(4, system_allocator(), 256);
  WaitBox box;
  // One task blocks; two complete promptly. when_any must pick a finisher.
  auto slow = spawn_task(sched, system_allocator(), wait_then(system_allocator(), &box, 99));
  auto fast = spawn_task(sched, system_allocator(), add_coro(system_allocator(), 7, 7));
  int w = when_any(slow, fast);
  // The fast task (index 1) is the only one that can finish (slow is parked).
  EXPECT_EQ(w, 1);
  EXPECT_TRUE(fast.is_finished());
  EXPECT_FALSE(slow.is_finished());
  // Release the slow one so shutdown can drain it.
  while (!box.ready.load(std::memory_order_acquire)) {}
  box.waker.wake();
  EXPECT_EQ(*slow.join(), 99);
  EXPECT_EQ(*fast.join(), 14);
  sched.shutdown();
}

// abort() sets the cooperative flag; a task that checks is_cancel_requested()
// after being woken observes it and can return early (reported as Cancelled).
namespace {
Task<int> cancellable(Allocator, WaitBox* box) {
  box->waker = co_await CaptureWaker{};
  box->ready.store(true, std::memory_order_release);
  co_await SuspendFiber{};              // parked until woken
  if (is_cancel_requested()) co_return -1;  // observe cancellation, unwind
  co_return 1;
}
}  // namespace

TEST(Join, AbortIsCooperative) {
  Scheduler sched(2, system_allocator(), 256);
  WaitBox box;
  auto h = spawn_task(sched, system_allocator(), cancellable(system_allocator(), &box));
  while (!box.ready.load(std::memory_order_acquire)) {}
  h.abort();          // request cancellation (flag set)
  box.waker.wake();   // wake the parked fiber so it reaches the check
  auto r = h.join();
  // The task saw the flag and co_return -1 (still a value, not an error, since it
  // chose to return normally). The point: the flag was observable.
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(*r, -1);
  sched.shutdown();
}

// Concurrency stress: many tasks joined from the main thread. TSan gate for the
// refcount/CV/GroupSignal races.
TEST(Join, ManyTasksJoinAllComplete) {
  constexpr int kN = 200;
  Scheduler sched(8, system_allocator(), 1024);
  std::atomic<int> total{0};
  for (int i = 0; i < kN; ++i) {
    auto h = spawn_task(sched, system_allocator(),
                        add_coro(system_allocator(), i, 1));
    auto r = h.join();
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(*r, i + 1);
    total.fetch_add(1, std::memory_order_relaxed);
  }
  EXPECT_EQ(total.load(), kN);
  sched.shutdown();
}
