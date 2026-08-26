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
void* fail_alloc(void*, std::size_t, std::size_t) noexcept { return nullptr; }
void fail_free(void*, void*, std::size_t, std::size_t) noexcept {}
void* fail_realloc(void*, void*, std::size_t, std::size_t,
                   std::size_t) noexcept {
  return nullptr;
}
Allocator failing_allocator() noexcept {
  return Allocator(&fail_alloc, &fail_free, &fail_realloc, nullptr);
}
}  // namespace

namespace {

Task<int> add_coro(Allocator, int a, int b) { co_return a + b; }

Task<void> bump_coro(Allocator, std::atomic<int>* c) {
  c->fetch_add(1, std::memory_order_relaxed);
  co_return;
}

struct MoveOnlyResult {
  explicit MoveOnlyResult(std::atomic<int>* destroyed) noexcept
      : destroyed(destroyed) {}
  MoveOnlyResult(MoveOnlyResult&& other) noexcept
      : destroyed(std::exchange(other.destroyed, nullptr)) {}
  MoveOnlyResult& operator=(MoveOnlyResult&&) = delete;
  MoveOnlyResult(const MoveOnlyResult&) = delete;
  MoveOnlyResult& operator=(const MoveOnlyResult&) = delete;
  ~MoveOnlyResult() {
    if (destroyed != nullptr) destroyed->fetch_add(1, std::memory_order_relaxed);
  }
  std::atomic<int>* destroyed;
};

Task<MoveOnlyResult> move_only_coro(Allocator, std::atomic<int>* destroyed) {
  co_return MoveOnlyResult(destroyed);
}

Task<int> throwing_coro(Allocator) {
  throw 42;
  co_return 0;
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

TEST(Join, UserExceptionBecomesInternalStatus) {
  Scheduler sched(1, system_allocator(), 64);
  auto h = spawn_task(sched, system_allocator(),
                      throwing_coro(system_allocator()));
  auto r = h.join();
  ASSERT_FALSE(r.has_value());
  EXPECT_TRUE(r.error().IsInternal());
  sched.shutdown();
}

TEST(Join, AllocationFailuresBecomeResourceExhausted) {
  Scheduler sched(1, system_allocator(), 64);

  // Coroutine-frame OOM is reported by the compiler's allocation-failure hook.
  auto no_frame = spawn_task(
      sched, system_allocator(), add_coro(failing_allocator(), 1, 2));
  ASSERT_TRUE(no_frame.valid());
  auto frame_result = no_frame.join();
  ASSERT_FALSE(frame_result.has_value());
  EXPECT_TRUE(frame_result.error().IsResourceExhausted());

  // Join-state OOM is an immediate completed error handle and participates in
  // grouping like any other already-completed task.
  auto no_state = spawn_task(
      sched, failing_allocator(), add_coro(system_allocator(), 1, 2));
  ASSERT_TRUE(no_state.valid());
  EXPECT_TRUE(no_state.is_finished());
  auto normal = spawn_task(
      sched, system_allocator(), add_coro(system_allocator(), 3, 4));
  EXPECT_EQ(when_any(no_state, normal), 0);
  auto state_result = no_state.join();
  ASSERT_FALSE(state_result.has_value());
  EXPECT_TRUE(state_result.error().IsResourceExhausted());
  EXPECT_EQ(*normal.join(), 7);

  auto unsafe_stack = spawn_task(
      sched, system_allocator(), add_coro(system_allocator(), 1, 2), 4096);
  auto stack_result = unsafe_stack.join();
  ASSERT_FALSE(stack_result.has_value());
  EXPECT_TRUE(stack_result.error().IsInvalidArgument());
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

// Regression: join used to require default construction + move assignment and
// then destroy the result slot twice (once in take, once in release).
TEST(Join, MoveOnlyResultIsDestroyedExactlyOnce) {
  std::atomic<int> destroyed{0};
  Scheduler sched(1, system_allocator(), 64);
  {
    auto h = spawn_task(
        sched, system_allocator(), move_only_coro(system_allocator(), &destroyed));
    auto r = h.join();
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(destroyed.load(std::memory_order_relaxed), 0);
  }
  EXPECT_EQ(destroyed.load(std::memory_order_relaxed), 1);
  sched.shutdown();
}

TEST(Join, SpawnAfterAdmissionClosedReturnsCancelled) {
  Scheduler sched(1, system_allocator(), 64);
  sched.begin_shutdown();
  auto h = spawn_task(sched, system_allocator(),
                      add_coro(system_allocator(), 20, 22));
  auto r = h.join();
  ASSERT_FALSE(r.has_value());
  EXPECT_TRUE(r.error().IsCancelled());
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
  // Exercise the shipping allocator. mimalloc's MI_TSAN instrumentation keeps
  // its free/reuse edges visible in sanitizer builds.
  Allocator al = system_allocator();
  Scheduler sched(8, al, 1024);
  std::atomic<int> total{0};
  for (int i = 0; i < kN; ++i) {
    auto h = spawn_task(sched, al, add_coro(al, i, 1));
    auto r = h.join();
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(*r, i + 1);
    total.fetch_add(1, std::memory_order_relaxed);
  }
  EXPECT_EQ(total.load(), kN);
  sched.shutdown();
}

// Regression: when_any where BOTH tasks complete near-simultaneously on different
// workers -- the loser's completion races the stack GroupSignal's destruction.
// Before the fix (JoinState::complete_ signalled the group AFTER unlocking its mu),
// a loser could read `group`, unlock, be delayed while the winner fired the group +
// the waiter ran clear_group() + returned (destroying the stack GroupSignal), then
// call g->on_complete() on freed memory -- a UAF. The existing WhenAnyReturnsAWinner
// test hides this: its loser is parked and NEVER completes. Here BOTH complete (two
// trivial add_coro), so the loser exercises the signal-the-group path, and many
// rounds on 8 workers widen the window. The shipping allocator is deliberately
// retained so the stress shape covers the production path as well as the race.
TEST(Join, WhenAnyLoserCompletesConcurrently) {
  Allocator al = system_allocator();
  Scheduler sched(8, al, 1024);
  constexpr int kRounds = 2000;
  for (int i = 0; i < kRounds; ++i) {
    auto a = spawn_task(sched, al, add_coro(al, i, 1));
    auto b = spawn_task(sched, al, add_coro(al, i, 2));
    int w = when_any(a, b);       // one wins; the OTHER still completes (the loser)
    EXPECT_TRUE(w == 0 || w == 1);
    // Join both so their states are fully retired before the next round (and to
    // confirm neither result was corrupted by the race).
    EXPECT_EQ(*a.join(), i + 1);
    EXPECT_EQ(*b.join(), i + 2);
  }
  sched.shutdown();
}
