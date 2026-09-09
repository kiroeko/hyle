#include <gtest/gtest.h>

#include <atomic>
#include <limits>
#include <span>
#include <thread>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/join.h"
#include "hyle/sched/scheduler.h"

using hyle::mem::AllocatorRef;
using hyle::mem::system_allocator;
using hyle::sched::is_cancel_requested;
using hyle::sched::JoinHandle;
using hyle::sched::Scheduler;
using hyle::sched::schedule_task;
using hyle::sched::spawn_task;
using hyle::sched::spawn_task_on;
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
AllocatorRef failing_allocator() noexcept {
  return AllocatorRef(&fail_alloc, &fail_free, &fail_realloc, nullptr);
}

struct AllocationCounter {
  AllocatorRef backing = system_allocator();
  std::atomic<std::size_t> attempts{0};
  std::atomic<std::size_t> allocations{0};
  std::atomic<std::size_t> frees{0};
  std::size_t fail_on = (std::numeric_limits<std::size_t>::max)();
};

void* counted_alloc(void* raw, std::size_t size,
                    std::size_t alignment) noexcept {
  auto* counter = static_cast<AllocationCounter*>(raw);
  const std::size_t attempt =
      counter->attempts.fetch_add(1, std::memory_order_relaxed);
  if (attempt == counter->fail_on) return nullptr;
  void* result = counter->backing.allocate(size, alignment);
  if (result != nullptr)
    counter->allocations.fetch_add(1, std::memory_order_relaxed);
  return result;
}

void counted_free(void* raw, void* ptr, std::size_t size,
                  std::size_t alignment) noexcept {
  auto* counter = static_cast<AllocationCounter*>(raw);
  if (ptr != nullptr)
    counter->frees.fetch_add(1, std::memory_order_relaxed);
  counter->backing.deallocate(ptr, size, alignment);
}

void* counted_realloc(void* raw, void* ptr, std::size_t old_size,
                      std::size_t new_size,
                      std::size_t alignment) noexcept {
  auto* counter = static_cast<AllocationCounter*>(raw);
  return counter->backing.reallocate(ptr, old_size, new_size, alignment);
}
}  // namespace

namespace {

Task<int> add_coro(AllocatorRef, int a, int b) { co_return a + b; }

Task<void> bump_coro(AllocatorRef, std::atomic<int>* c) {
  c->fetch_add(1, std::memory_order_relaxed);
  co_return;
}

Task<void> observe_worker_coro(AllocatorRef, std::atomic<int>* worker) {
  worker->store(hyle::sched::current_worker_index(),
                std::memory_order_release);
  co_return;
}

struct PinnedSaturationBox {
  Scheduler* scheduler = nullptr;
  std::atomic<int> ran{0};
  int accepted = 0;
  int full = 0;
};

Task<void> saturate_pinned_queue(AllocatorRef, PinnedSaturationBox* box) {
  for (int i = 0; i < 8; ++i) {
    auto child = spawn_task_on(
        *box->scheduler, 0, system_allocator(),
        bump_coro(system_allocator(), &box->ran));
    if (child.is_finished()) {
      auto result = child.join();
      if (!result && result.error().IsResourceExhausted()) ++box->full;
    } else {
      ++box->accepted;
      child.detach();
    }
  }
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

Task<MoveOnlyResult> move_only_coro(AllocatorRef, std::atomic<int>* destroyed) {
  co_return MoveOnlyResult(destroyed);
}

Task<int> throwing_coro(AllocatorRef) {
  throw 42;
  co_return 0;
}

// A coroutine that suspends until woken, publishing its Waker for a foreign wake.
struct WaitBox {
  std::atomic<bool> ready{false};
  Waker waker;
};
Task<int> wait_then(AllocatorRef, WaitBox* box, int ret) {
  box->waker = co_await CaptureWaker{};
  box->ready.store(true, std::memory_order_release);
  co_await SuspendFiber{};
  co_return ret;
}

struct PinnedWaitBox {
  std::atomic<bool> ready{false};
  Waker waker;
  std::atomic<int> first_worker{-1};
  std::atomic<int> after_yield_worker{-1};
  std::atomic<int> after_wake_worker{-1};
};

Task<int> pinned_wait_then(AllocatorRef, PinnedWaitBox* box, int ret) {
  box->first_worker.store(hyle::sched::current_worker_index(),
                          std::memory_order_relaxed);
  hyle::sched::Fiber::yield();
  box->after_yield_worker.store(hyle::sched::current_worker_index(),
                                std::memory_order_relaxed);
  box->waker = co_await CaptureWaker{};
  box->ready.store(true, std::memory_order_release);
  co_await SuspendFiber{};
  box->after_wake_worker.store(hyle::sched::current_worker_index(),
                               std::memory_order_relaxed);
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

TEST(Join, SpawnTaskOnReturnsTypedValueAndKeepsOwnerAcrossWake) {
  constexpr unsigned kTarget = 2;
  Scheduler sched(4, system_allocator(), 256);
  PinnedWaitBox box;
  auto handle = spawn_task_on(
      sched, kTarget, system_allocator(),
      pinned_wait_then(system_allocator(), &box, 42));

  while (!box.ready.load(std::memory_order_acquire)) std::this_thread::yield();
  ASSERT_TRUE(box.waker.valid());
  box.waker.wake();  // this test thread is foreign to the scheduler

  auto result = handle.join();
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, 42);
  EXPECT_EQ(box.first_worker.load(std::memory_order_relaxed),
            static_cast<int>(kTarget));
  EXPECT_EQ(box.after_yield_worker.load(std::memory_order_relaxed),
            static_cast<int>(kTarget));
  EXPECT_EQ(box.after_wake_worker.load(std::memory_order_relaxed),
            static_cast<int>(kTarget));
  sched.shutdown();
}

TEST(Join, SpawnTaskOnSupportsVoidAndMoveOnlyResults) {
  Scheduler sched(2, system_allocator(), 128);
  std::atomic<int> calls{0};
  auto void_handle = spawn_task_on(
      sched, 0, system_allocator(), bump_coro(system_allocator(), &calls));
  EXPECT_TRUE(void_handle.join().has_value());

  std::atomic<int> destroyed{0};
  {
    auto move_handle = spawn_task_on(
        sched, 1, system_allocator(),
        move_only_coro(system_allocator(), &destroyed));
    auto result = move_handle.join();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(destroyed.load(std::memory_order_relaxed), 0);
  }
  EXPECT_EQ(calls.load(std::memory_order_relaxed), 1);
  EXPECT_EQ(destroyed.load(std::memory_order_relaxed), 1);
  sched.shutdown();
}

TEST(Join, SpawnTaskOnRejectsInvalidWorkerBeforeOwnedAllocations) {
  AllocationCounter counter;
  AllocatorRef counted(&counted_alloc, &counted_free, &counted_realloc,
                       &counter);
  Scheduler sched(2, system_allocator(), 64);
  auto handle = spawn_task_on(
      sched, sched.num_workers(), counted,
      add_coro(system_allocator(), 20, 22));
  EXPECT_TRUE(handle.is_finished());
  auto result = handle.join();
  ASSERT_FALSE(result.has_value());
  EXPECT_TRUE(result.error().IsInvalidArgument());
  EXPECT_EQ(counter.allocations.load(std::memory_order_relaxed), 0u);
  EXPECT_EQ(counter.frees.load(std::memory_order_relaxed), 0u);
  sched.shutdown();
}

TEST(Join, SpawnTaskOnRejectsInvalidSchedulerWithoutDereferencingIt) {
  hyle::Status scheduler_status;
  Scheduler invalid(/*num_workers=*/0, system_allocator(), 64,
                    &scheduler_status);
  ASSERT_FALSE(scheduler_status.ok());
  ASSERT_FALSE(invalid.valid());
  auto handle = spawn_task_on(
      invalid, 0, system_allocator(),
      add_coro(system_allocator(), 20, 22));
  auto result = handle.join();
  ASSERT_FALSE(result.has_value());
  EXPECT_TRUE(result.error().IsInvalidArgument());
}

TEST(Join, SpawnTaskOnDriverOomReleasesTheJoinState) {
  AllocationCounter counter;
  counter.fail_on = 1;  // JoinState succeeds; TaskDriver allocation fails.
  AllocatorRef counted(&counted_alloc, &counted_free, &counted_realloc,
                       &counter);
  Scheduler sched(1, system_allocator(), 64);
  auto handle = spawn_task_on(
      sched, 0, counted, add_coro(system_allocator(), 20, 22));
  auto result = handle.join();
  ASSERT_FALSE(result.has_value());
  EXPECT_TRUE(result.error().IsResourceExhausted());
  EXPECT_EQ(counter.attempts.load(std::memory_order_relaxed), 2u);
  EXPECT_EQ(counter.allocations.load(std::memory_order_relaxed), 1u);
  EXPECT_EQ(counter.frees.load(std::memory_order_relaxed), 1u);
  sched.shutdown();
}

TEST(Join, SpawnTaskOnFrameAndJoinStateOomAreRecoverable) {
  Scheduler sched(1, system_allocator(), 64);

  auto no_frame = spawn_task_on(
      sched, 0, system_allocator(), add_coro(failing_allocator(), 1, 2));
  auto frame_result = no_frame.join();
  ASSERT_FALSE(frame_result.has_value());
  EXPECT_TRUE(frame_result.error().IsResourceExhausted());

  auto no_state = spawn_task_on(
      sched, 0, failing_allocator(), add_coro(system_allocator(), 1, 2));
  auto state_result = no_state.join();
  ASSERT_FALSE(state_result.has_value());
  EXPECT_TRUE(state_result.error().IsResourceExhausted());
  sched.shutdown();
}

TEST(Join, SpawnTaskOnStackFailureReleasesOwnedAllocations) {
  AllocationCounter counter;
  AllocatorRef counted(&counted_alloc, &counted_free, &counted_realloc,
                       &counter);
  Scheduler sched(1, system_allocator(), 64);
  {
    auto handle = spawn_task_on(
        sched, 0, counted, add_coro(system_allocator(), 1, 2),
        (std::numeric_limits<std::size_t>::max)());
    auto result = handle.join();
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().IsOutOfRange());
  }
  EXPECT_EQ(counter.allocations.load(std::memory_order_relaxed), 2u);
  EXPECT_EQ(counter.frees.load(std::memory_order_relaxed), 2u);
  sched.shutdown();
}

TEST(Join, SpawnTaskOnQueueFullReturnsAndReleasesRejectedTasks) {
  Scheduler sched(1, system_allocator(), 2);
  PinnedSaturationBox box{&sched};
  auto parent = spawn_task_on(
      sched, 0, system_allocator(),
      saturate_pinned_queue(system_allocator(), &box));
  ASSERT_TRUE(parent.join().has_value());
  sched.drain();
  EXPECT_EQ(box.accepted, 2);
  EXPECT_EQ(box.full, 6);
  EXPECT_EQ(box.ran.load(std::memory_order_relaxed), box.accepted);
  sched.shutdown();
}

TEST(Join, ScheduledTaskRunsOnWorkerStackAndReturnsTypedValue) {
  Scheduler scheduler(2, system_allocator(), 64);
  std::atomic<bool> used_worker_stack{false};
  auto handle = schedule_task(
      scheduler, system_allocator(), [&used_worker_stack]() noexcept {
        used_worker_stack.store(
            hyle::sched::Fiber::current() == nullptr &&
                hyle::sched::current_worker_index() >= 0,
            std::memory_order_release);
        return 42;
      });
  auto result = handle.join();
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, 42);
  EXPECT_TRUE(used_worker_stack.load(std::memory_order_acquire));
  scheduler.shutdown();
}

TEST(Join, ScheduledTaskUsesOneOwnedCell) {
  AllocationCounter counter;
  AllocatorRef counted(&counted_alloc, &counted_free, &counted_realloc,
                    &counter);
  Scheduler scheduler(1, system_allocator(), 64);
  {
    auto handle = schedule_task(
        scheduler, counted, [payload = 40]() noexcept {
          return payload + 2;
        });
    auto result = handle.join();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 42);
  }
  scheduler.drain();
  EXPECT_EQ(counter.allocations.load(std::memory_order_relaxed), 1u);
  EXPECT_EQ(counter.frees.load(std::memory_order_relaxed), 1u);
  scheduler.shutdown();
}

TEST(Join, ScheduledVoidTaskJoinsAndDetachStillDrains) {
  Scheduler scheduler(2, system_allocator(), 64);
  std::atomic<int> calls{0};
  auto joined = schedule_task(scheduler, system_allocator(), [&]() noexcept {
    calls.fetch_add(1, std::memory_order_relaxed);
  });
  EXPECT_TRUE(joined.join().has_value());
  {
    auto detached = schedule_task(
        scheduler, system_allocator(), [&]() noexcept {
          calls.fetch_add(1, std::memory_order_relaxed);
        });
    detached.detach();
  }
  scheduler.drain();
  EXPECT_EQ(calls.load(std::memory_order_relaxed), 2);
  scheduler.shutdown();
}

TEST(Join, ScheduledTaskFailureAndClosedAdmissionAreStructured) {
  Scheduler scheduler(1, system_allocator(), 64);
  auto exhausted = schedule_task(
      scheduler, failing_allocator(), []() noexcept { return 1; });
  ASSERT_FALSE(exhausted.join().has_value());

  scheduler.begin_shutdown();
  auto rejected = schedule_task(
      scheduler, system_allocator(), []() noexcept { return 2; });
  auto result = rejected.join();
  ASSERT_FALSE(result.has_value());
  EXPECT_TRUE(result.error().IsCancelled());
  scheduler.shutdown();
}

TEST(Join, ScheduledTaskSupportsDynamicTypedFanout) {
  constexpr int kTasks = 64;
  Scheduler scheduler(4, system_allocator(), 256);
  std::vector<JoinHandle<int>> handles;
  handles.reserve(kTasks);
  for (int value = 0; value < kTasks; ++value) {
    handles.push_back(schedule_task(
        scheduler, system_allocator(), [value]() noexcept {
          return value + 1;
        }));
  }
  when_all(std::span<JoinHandle<int>>(handles));
  int sum = 0;
  for (auto& handle : handles) sum += *handle.join();
  EXPECT_EQ(sum, kTasks * (kTasks + 1) / 2);
  scheduler.shutdown();
}

TEST(Join, ScheduledTaskCancelRequestIsVisibleDuringWorkerStackExecution) {
  Scheduler scheduler(1, system_allocator(), 64);
  std::atomic<bool> started{false};
  std::atomic<bool> observed{false};
  auto handle = schedule_task(
      scheduler, system_allocator(), [&]() noexcept {
        started.store(true, std::memory_order_release);
        while (!is_cancel_requested()) std::this_thread::yield();
        observed.store(true, std::memory_order_release);
        return 7;
      });
  while (!started.load(std::memory_order_acquire)) std::this_thread::yield();
  handle.request_cancel();
  auto result = handle.join();
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, 7);
  EXPECT_TRUE(observed.load(std::memory_order_acquire));
  scheduler.shutdown();
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

TEST(Join, SpawnTaskOnAfterAdmissionClosedReturnsCancelled) {
  Scheduler sched(2, system_allocator(), 64);
  sched.begin_shutdown();
  auto handle = spawn_task_on(
      sched, 1, system_allocator(),
      add_coro(system_allocator(), 20, 22));
  auto result = handle.join();
  ASSERT_FALSE(result.has_value());
  EXPECT_TRUE(result.error().IsCancelled());
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

TEST(Join, SpawnTaskOnDetachStillDrainsOnTheSelectedWorker) {
  constexpr unsigned kTarget = 1;
  std::atomic<int> observed{-1};
  {
    Scheduler sched(2, system_allocator(), 64);
    auto handle = spawn_task_on(
        sched, kTarget, system_allocator(),
        observe_worker_coro(system_allocator(), &observed));
    handle.detach();
    sched.drain();
    sched.shutdown();
  }
  EXPECT_EQ(observed.load(std::memory_order_acquire),
            static_cast<int>(kTarget));
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

TEST(Join, RuntimeSizedWhenAllSupportsDynamicHomogeneousFanout) {
  constexpr int kTasks = 32;
  Scheduler scheduler(4, system_allocator(), 128);
  std::vector<JoinHandle<int>> handles;
  handles.reserve(kTasks);
  for (int value = 0; value < kTasks; ++value) {
    handles.push_back(spawn_task(
        scheduler, system_allocator(),
        add_coro(system_allocator(), value, 1)));
  }
  when_all(std::span<JoinHandle<int>>(handles));
  int sum = 0;
  for (auto& handle : handles) sum += *handle.join();
  EXPECT_EQ(sum, kTasks * (kTasks + 1) / 2);

  std::span<JoinHandle<int>> empty;
  when_all(empty);
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

TEST(Join, RuntimeSizedWhenAnyReturnsDynamicWinner) {
  Scheduler scheduler(2, system_allocator(), 64);
  WaitBox box;
  std::vector<JoinHandle<int>> handles;
  handles.push_back(spawn_task(
      scheduler, system_allocator(),
      wait_then(system_allocator(), &box, 99)));
  handles.push_back(spawn_task(
      scheduler, system_allocator(),
      add_coro(system_allocator(), 20, 22)));
  EXPECT_EQ(when_any(std::span<JoinHandle<int>>(handles)), 1u);
  while (!box.ready.load(std::memory_order_acquire)) {}
  box.waker.wake();
  EXPECT_EQ(*handles[0].join(), 99);
  EXPECT_EQ(*handles[1].join(), 42);
}

// request_cancel() sets the cooperative flag; a task that checks
// is_cancel_requested()
// after being woken observes it and can return early (reported as Cancelled).
namespace {
Task<int> cancellable(AllocatorRef, WaitBox* box) {
  box->waker = co_await CaptureWaker{};
  box->ready.store(true, std::memory_order_release);
  co_await SuspendFiber{};              // parked until woken
  if (is_cancel_requested()) co_return -1;  // observe cancellation, unwind
  co_return 1;
}
}  // namespace

TEST(Join, CancelRequestDoesNotForgeWakeForParkedTask) {
  Scheduler sched(2, system_allocator(), 256);
  WaitBox box;
  auto h = spawn_task(sched, system_allocator(), cancellable(system_allocator(), &box));
  while (!box.ready.load(std::memory_order_acquire)) {}
  h.request_cancel(); // request cancellation (flag set)
  box.waker.wake();   // wake the parked fiber so it reaches the check
  auto r = h.join();
  // The task saw the flag and co_return -1 (still a value, not an error, since it
  // chose to return normally). The point: the flag was observable.
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(*r, -1);
  sched.shutdown();
}

TEST(Join, SpawnTaskOnCancelRequestKeepsPinnedWakeOwnership) {
  constexpr unsigned kTarget = 1;
  Scheduler sched(2, system_allocator(), 128);
  WaitBox box;
  auto handle = spawn_task_on(
      sched, kTarget, system_allocator(),
      cancellable(system_allocator(), &box));
  while (!box.ready.load(std::memory_order_acquire)) std::this_thread::yield();
  handle.request_cancel();
  box.waker.wake();
  auto result = handle.join();
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, -1);
  sched.shutdown();
}

TEST(Join, CancelRequestAfterCompletionDoesNotRewriteResult) {
  Scheduler scheduler(1, system_allocator(), 64);
  auto handle = schedule_task(
      scheduler, system_allocator(), []() noexcept { return 42; });
  scheduler.drain();
  ASSERT_TRUE(handle.is_finished());

  handle.request_cancel();
  auto result = handle.join();
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, 42);
  scheduler.shutdown();
}

TEST(Join, CancelRequestRacingCompletionHasOneValidOutcome) {
  Scheduler scheduler(4, system_allocator(), 256);
  for (int iteration = 0; iteration < 200; ++iteration) {
    auto handle = schedule_task(
        scheduler, system_allocator(), []() noexcept { return 7; });
    handle.request_cancel();
    auto result = handle.join();
    if (result) {
      EXPECT_EQ(*result, 7);
    } else {
      EXPECT_TRUE(result.error().IsCancelled());
    }
  }
  scheduler.shutdown();
}

// Concurrency stress: many tasks joined from the main thread. TSan gate for the
// refcount/CV/GroupSignal races.
TEST(Join, ManyTasksJoinAllComplete) {
  constexpr int kN = 200;
  // Exercise the shipping allocator. mimalloc's MI_TSAN instrumentation keeps
  // its free/reuse edges visible in sanitizer builds.
  AllocatorRef al = system_allocator();
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
  AllocatorRef al = system_allocator();
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
