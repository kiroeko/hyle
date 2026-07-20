#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <thread>

#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/waker.h"

using hyle::mem::Allocator;
using hyle::mem::system_allocator;
using hyle::sched::CaptureWaker;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::SuspendFiber;
using hyle::sched::Task;
using hyle::sched::Waker;

namespace {

// A Task coroutine: FIRST param must be the allocator (frame allocated from it).
Task<int> add_coro(Allocator, int a, int b) { co_return a + b; }

// A coroutine that suspends its fiber once (via co_await), then returns. The
// Waker it captures is published so a foreign thread can wake it.
struct WaitBox {
  std::atomic<bool> waker_ready{false};
  Waker waker;
};

Task<int> waiting_coro(Allocator, WaitBox* box) {
  // Two-step: capture the Waker (fiber not yet suspended), publish it, THEN
  // co_await SuspendFiber -> leaves the run loop until wake().
  box->waker = co_await CaptureWaker{};
  box->waker_ready.store(true, std::memory_order_release);
  co_await SuspendFiber{};
  co_return 42;  // resumed -> return
}

// Fiber body that drives a Task to completion and records the result. Because
// co_await suspends the FIBER (fcontext), a single resume() here runs the whole
// coroutine incl. its awaits: the fiber freezes inside resume() at the await and
// thaws back into it on wake, so resume() returns only once the task is done.
struct DriveInt {
  Task<int> task;
  std::atomic<int> result{-1};
  std::atomic<bool> finished{false};
};
void drive_int_entry(void* p) {
  auto* d = static_cast<DriveInt*>(p);
  // One resume() runs the coroutine to completion: a co_await that fiber-yields
  // freezes this whole call (including resume()) on the fiber stack, and
  // SuspendFiber's await_suspend returns false so the coroutine CONTINUES when
  // the fiber is re-injected -- it never parks independently of its fiber. So
  // resume() returns only once the coroutine has actually finished.
  d->task.resume();
  d->result.store(d->task.result(), std::memory_order_release);
  d->finished.store(true, std::memory_order_release);
}

}  // namespace

// Frame allocated from mem::Allocator, plain co_return, no suspend.
TEST(Coro, TaskReturnsValueFrameFromAllocator) {
  auto d = std::make_unique<DriveInt>();
  d->task = add_coro(system_allocator(), 20, 22);
  Fiber f(kDefaultStackSize, &drive_int_entry, d.get());
  {
    Scheduler sched(2, system_allocator(), 256);
    sched.spawn(&f);
    sched.drain();
  }
  EXPECT_TRUE(d->finished.load());
  EXPECT_EQ(d->result.load(), 42);
}

// The core sched-06 claim: co_await inside a coroutine suspends the FIBER; a
// foreign wake resumes it; the coroutine continues and returns. If the fiber
// weren't truly suspended (or the wake lost), drain() would hang.
TEST(Coro, CoAwaitSuspendsFiberResumedByForeignWake) {
  auto box = std::make_unique<WaitBox>();
  auto d = std::make_unique<DriveInt>();
  d->task = waiting_coro(system_allocator(), box.get());
  Fiber f(kDefaultStackSize, &drive_int_entry, d.get());
  {
    Scheduler sched(2, system_allocator(), 256);
    sched.spawn(&f);
    // Wait until the coroutine captured + published its Waker (it is now about
    // to / already suspended on co_await SuspendFiber).
    while (!box->waker_ready.load(std::memory_order_acquire))
      std::this_thread::yield();
    EXPECT_TRUE(box->waker.valid());
    box->waker.wake();  // foreign-thread wake -> re-inject the fiber
    sched.drain();
  }
  EXPECT_TRUE(d->finished.load());
  EXPECT_EQ(d->result.load(), 42);
}

// Nested co_await: an outer coroutine awaits by calling a helper that itself
// suspends the fiber. Proves the allocator-backed coroutine-frame model composes (one fiber
// suspend covers the whole await chain).
namespace {
Task<int> inner_coro(Allocator, WaitBox* box) {
  box->waker = co_await CaptureWaker{};
  box->waker_ready.store(true, std::memory_order_release);
  co_await SuspendFiber{};
  co_return 7;
}
// Outer runs inner to completion inline (same fiber) then adds to it. inner's
// co_await suspends the shared fiber; outer resumes past it after wake.
Task<int> outer_coro(Allocator alloc, WaitBox* box) {
  Task<int> in = inner_coro(alloc, box);
  while (!in.done()) in.resume();  // drive inner; its co_await yields the fiber
  co_return in.result() + 35;      // 7 + 35 = 42
}
}  // namespace

TEST(Coro, NestedCoAwaitComposes) {
  auto box = std::make_unique<WaitBox>();
  auto d = std::make_unique<DriveInt>();
  d->task = outer_coro(system_allocator(), box.get());
  Fiber f(kDefaultStackSize, &drive_int_entry, d.get());
  {
    Scheduler sched(2, system_allocator(), 256);
    sched.spawn(&f);
    while (!box->waker_ready.load(std::memory_order_acquire))
      std::this_thread::yield();
    box->waker.wake();
    sched.drain();
  }
  EXPECT_TRUE(d->finished.load());
  EXPECT_EQ(d->result.load(), 42);
}
