#include "hyle/sched/priority.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <limits>
#include <thread>

#include "hyle/mem/allocator.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/join.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/task_scope.h"
#include "hyle/sched/timer.h"
#include "hyle/sched/waker.h"

namespace {

using hyle::mem::AllocatorRef;
using hyle::mem::system_allocator;
using hyle::sched::current_execution_priority;
using hyle::sched::current_worker_index;
using hyle::sched::ExecutionOptions;
using hyle::sched::ExecutionPriority;
using hyle::sched::Fiber;
using hyle::sched::FiberLaunchOptions;
using hyle::sched::Scheduler;
using hyle::sched::SchedulerOptions;
using hyle::sched::SchedulerPriorityMode;
using hyle::sched::Task;
using hyle::sched::TaskScope;
using hyle::sched::TaskScopeOptions;
using hyle::sched::Waker;

SchedulerOptions priority_options() {
  SchedulerOptions options;
  options.priority_mode = SchedulerPriorityMode::kThreeLanes;
  options.idle_policy = hyle::sched::SchedulerIdlePolicy::kPowerSaving;
  return options;
}

struct FailingAllocatorState {
  std::atomic<std::size_t> attempts{0};
  std::atomic<std::size_t> allocations{0};
  std::atomic<std::size_t> frees{0};
  std::size_t fail_on = (std::numeric_limits<std::size_t>::max)();
};

void* failing_allocate(void* raw, std::size_t size,
                       std::size_t alignment) noexcept {
  auto* state = static_cast<FailingAllocatorState*>(raw);
  const std::size_t attempt =
      state->attempts.fetch_add(1, std::memory_order_relaxed);
  if (attempt == state->fail_on) return nullptr;
  void* result = system_allocator().allocate(size, alignment);
  if (result != nullptr)
    state->allocations.fetch_add(1, std::memory_order_relaxed);
  return result;
}

void failing_deallocate(void* raw, void* pointer, std::size_t size,
                        std::size_t alignment) noexcept {
  if (pointer == nullptr) return;
  auto* state = static_cast<FailingAllocatorState*>(raw);
  state->frees.fetch_add(1, std::memory_order_relaxed);
  system_allocator().deallocate(pointer, size, alignment);
}

void* failing_reallocate(void* raw, void* pointer, std::size_t old_size,
                         std::size_t new_size, std::size_t alignment) noexcept {
  auto* state = static_cast<FailingAllocatorState*>(raw);
  const std::size_t attempt =
      state->attempts.fetch_add(1, std::memory_order_relaxed);
  if (attempt == state->fail_on) return nullptr;
  return system_allocator().reallocate(pointer, old_size, new_size, alignment);
}

AllocatorRef failing_allocator(FailingAllocatorState* state) noexcept {
  return AllocatorRef(&failing_allocate, &failing_deallocate,
                      &failing_reallocate, state);
}

struct OrderedLog {
  std::array<int, 16> values{};
  std::atomic<unsigned> size{0};

  void append(int value) noexcept {
    const unsigned index = size.fetch_add(1, std::memory_order_relaxed);
    if (index < values.size()) values[index] = value;
  }
};

struct BlockingRecord {
  std::atomic<bool> started{false};
  std::atomic<bool> release{false};
  OrderedLog* log = nullptr;
  int finish_value = 0;
};

void blocking_record(void* raw) noexcept {
  auto* record = static_cast<BlockingRecord*>(raw);
  record->started.store(true, std::memory_order_release);
  while (!record->release.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  if (record->log != nullptr) record->log->append(record->finish_value);
}

struct RecordValue {
  OrderedLog* log;
  int value;
  std::atomic<int>* observed_priority = nullptr;
};

void record_value(void* raw) noexcept {
  auto* record = static_cast<RecordValue*>(raw);
  if (record->observed_priority != nullptr) {
    record->observed_priority->store(
        static_cast<int>(current_execution_priority()),
        std::memory_order_release);
  }
  record->log->append(record->value);
}

struct YieldRecord {
  OrderedLog* log;
  std::atomic<int> first_priority{-1};
  std::atomic<int> second_priority{-1};
};

void yield_record(void* raw) noexcept {
  auto* record = static_cast<YieldRecord*>(raw);
  record->first_priority.store(static_cast<int>(current_execution_priority()),
                               std::memory_order_release);
  record->log->append(1);
  Fiber::yield();
  record->second_priority.store(static_cast<int>(current_execution_priority()),
                                std::memory_order_release);
  record->log->append(2);
}

struct WakeRecord {
  Waker waker;
  std::atomic<bool> armed{false};
  OrderedLog* log = nullptr;
  std::atomic<int> observed_priority{-1};
};

void wake_record(void* raw) noexcept {
  auto* record = static_cast<WakeRecord*>(raw);
  record->waker = hyle::sched::current_waker();
  record->armed.store(true, std::memory_order_release);
  hyle::sched::suspend();
  record->observed_priority.store(
      static_cast<int>(current_execution_priority()),
      std::memory_order_release);
  record->log->append(1);
}

struct InheritRecord {
  Scheduler* scheduler;
  std::atomic<int> parent{-1};
  std::atomic<int> child{-1};
  std::atomic<int> overridden{-1};
};

void record_child(void* raw) noexcept {
  auto* record = static_cast<InheritRecord*>(raw);
  record->child.store(static_cast<int>(current_execution_priority()),
                      std::memory_order_release);
}

void record_overridden(void* raw) noexcept {
  auto* record = static_cast<InheritRecord*>(raw);
  record->overridden.store(static_cast<int>(current_execution_priority()),
                           std::memory_order_release);
}

void submit_children(void* raw) noexcept {
  auto* record = static_cast<InheritRecord*>(raw);
  record->parent.store(static_cast<int>(current_execution_priority()),
                       std::memory_order_release);
  (void)record->scheduler->schedule(&record_child, record);
  (void)record->scheduler->schedule(&record_overridden, record,
                                    ExecutionOptions{ExecutionPriority::kLow});
}

struct PinnedRecord {
  std::atomic<int> worker{-1};
  std::atomic<int> priority{-1};
};

void record_pinned(void* raw) noexcept {
  auto* record = static_cast<PinnedRecord*>(raw);
  record->worker.store(current_worker_index(), std::memory_order_release);
  record->priority.store(static_cast<int>(current_execution_priority()),
                         std::memory_order_release);
}

struct SaturationRecord {
  Scheduler* scheduler;
  std::atomic<int> admitted{0};
  std::atomic<bool> rejected{false};
};

void no_op(void*) noexcept {}

void saturate_priority_lane(void* raw) noexcept {
  auto* record = static_cast<SaturationRecord*>(raw);
  for (int i = 0; i < 16; ++i) {
    hyle::Status status = record->scheduler->schedule(
        &no_op, nullptr, ExecutionOptions{ExecutionPriority::kHigh});
    if (!status.ok()) {
      record->rejected.store(status.IsResourceExhausted(),
                             std::memory_order_release);
      break;
    }
    record->admitted.fetch_add(1, std::memory_order_relaxed);
  }
}

struct TaskPriorityRecord {
  std::atomic<int> first{-1};
  std::atomic<int> second{-1};
};

Task<void> record_task_priority(AllocatorRef, TaskPriorityRecord* record) {
  record->first.store(static_cast<int>(current_execution_priority()),
                      std::memory_order_release);
  Fiber::yield();
  record->second.store(static_cast<int>(current_execution_priority()),
                       std::memory_order_release);
  co_return;
}

Task<int> return_task_priority(AllocatorRef) {
  Fiber::yield();
  co_return static_cast<int>(current_execution_priority());
}

struct TimerPriorityRecord {
  std::atomic<int> before{-1};
  std::atomic<int> after{-1};
};

void sleep_and_record_priority(void* raw) noexcept {
  auto* record = static_cast<TimerPriorityRecord*>(raw);
  record->before.store(static_cast<int>(current_execution_priority()),
                       std::memory_order_release);
  hyle::sched::sleep_for(std::chrono::milliseconds(1));
  record->after.store(static_cast<int>(current_execution_priority()),
                      std::memory_order_release);
}

struct CrossSchedulerRecord {
  Scheduler* target;
  std::atomic<int> source{-1};
  std::atomic<int> target_child{-1};
};

void record_cross_scheduler_child(void* raw) noexcept {
  auto* record = static_cast<CrossSchedulerRecord*>(raw);
  record->target_child.store(static_cast<int>(current_execution_priority()),
                             std::memory_order_release);
}

void submit_cross_scheduler_child(void* raw) noexcept {
  auto* record = static_cast<CrossSchedulerRecord*>(raw);
  record->source.store(static_cast<int>(current_execution_priority()),
                       std::memory_order_release);
  (void)record->target->schedule(&record_cross_scheduler_child, record);
}

struct ShutdownPriorityRecord {
  std::atomic<bool> started{false};
  std::atomic<int> resumed_priority{-1};
};

void wait_for_shutdown_and_record_priority(void* raw) noexcept {
  auto* record = static_cast<ShutdownPriorityRecord*>(raw);
  record->started.store(true, std::memory_order_release);
  hyle::sched::wait_for_shutdown();
  record->resumed_priority.store(
      static_cast<int>(current_execution_priority()),
      std::memory_order_release);
}

struct CancelPriorityRecord {
  std::atomic<bool> started{false};
  std::atomic<int> observed_priority{-1};
};

Task<hyle::Status> wait_for_cancel_and_record_priority(
    AllocatorRef, CancelPriorityRecord* record) {
  record->started.store(true, std::memory_order_release);
  while (!hyle::sched::is_cancel_requested()) Fiber::yield();
  record->observed_priority.store(
      static_cast<int>(current_execution_priority()),
      std::memory_order_release);
  co_return hyle::Status::Cancelled("priority child observed cancellation");
}

}  // namespace

TEST(SchedulerPriority, DefaultModeRejectsExplicitNonNormalPriority) {
  EXPECT_EQ(current_execution_priority(), ExecutionPriority::kNormal);
  Scheduler scheduler(1, system_allocator(), 64);
  EXPECT_FALSE(scheduler.execution_priorities_enabled());
  EXPECT_TRUE(
      scheduler
          .schedule(&no_op, nullptr, ExecutionOptions{ExecutionPriority::kHigh})
          .IsInvalidArgument());
  EXPECT_TRUE(
      scheduler
          .schedule(&no_op, nullptr,
                    ExecutionOptions{static_cast<ExecutionPriority>(255)})
          .IsInvalidArgument());
  scheduler.shutdown();
}

TEST(SchedulerPriority, ConstructionRejectsInvalidMode) {
  SchedulerOptions options;
  options.priority_mode = static_cast<SchedulerPriorityMode>(255);
  hyle::Status status;
  Scheduler scheduler(1, system_allocator(), 64, options, &status);
  EXPECT_FALSE(scheduler.valid());
  EXPECT_TRUE(status.IsInvalidArgument());
}

TEST(SchedulerPriority, SidecarAllocationFailureIsRecoverableAndLeakFree) {
  // Learn the complete normal-only allocation prefix instead of identifying an
  // implementation allocation by its size/alignment. The next allocation made
  // by the otherwise-identical priority construction is the optional sidecar.
  FailingAllocatorState normal_state;
  hyle::Status normal_status;
  {
    Scheduler normal(1, failing_allocator(&normal_state), 64, &normal_status);
    ASSERT_TRUE(normal_status.ok());
    normal.shutdown();
  }
  ASSERT_EQ(normal_state.allocations.load(std::memory_order_relaxed),
            normal_state.frees.load(std::memory_order_relaxed));

  FailingAllocatorState allocator_state;
  allocator_state.fail_on =
      normal_state.attempts.load(std::memory_order_relaxed);
  hyle::Status status;
  {
    Scheduler scheduler(1, failing_allocator(&allocator_state), 64,
                        priority_options(), &status);
    EXPECT_FALSE(scheduler.valid());
  }
  EXPECT_TRUE(status.IsResourceExhausted());
  EXPECT_EQ(status.message(), "scheduler priority sidecar allocation failed");
  EXPECT_EQ(allocator_state.allocations.load(std::memory_order_relaxed),
            allocator_state.frees.load(std::memory_order_relaxed));
}

TEST(SchedulerPriority, ChoosesHighestReadyLaneAtEachBoundary) {
  Scheduler scheduler(1, system_allocator(), 64, priority_options());
  ASSERT_TRUE(scheduler.execution_priorities_enabled());
  OrderedLog log;
  BlockingRecord blocker;
  blocker.log = &log;
  blocker.finish_value = 0;
  ASSERT_TRUE(scheduler.schedule(&blocking_record, &blocker).ok());
  while (!blocker.started.load(std::memory_order_acquire))
    std::this_thread::yield();

  RecordValue low{&log, 3};
  RecordValue normal{&log, 2};
  RecordValue high{&log, 1};
  ASSERT_TRUE(scheduler
                  .schedule(&record_value, &low,
                            ExecutionOptions{ExecutionPriority::kLow})
                  .ok());
  ASSERT_TRUE(scheduler.schedule(&record_value, &normal).ok());
  ASSERT_TRUE(scheduler
                  .schedule(&record_value, &high,
                            ExecutionOptions{ExecutionPriority::kHigh})
                  .ok());
  blocker.release.store(true, std::memory_order_release);
  scheduler.drain();

  ASSERT_EQ(log.size.load(std::memory_order_acquire), 4u);
  EXPECT_EQ(log.values[0], 0);
  EXPECT_EQ(log.values[1], 1);
  EXPECT_EQ(log.values[2], 2);
  EXPECT_EQ(log.values[3], 3);
  scheduler.shutdown();
}

TEST(SchedulerPriority, DoesNotPreemptAlreadyRunningLowWork) {
  Scheduler scheduler(1, system_allocator(), 64, priority_options());
  BlockingRecord low;
  ASSERT_TRUE(scheduler
                  .schedule(&blocking_record, &low,
                            ExecutionOptions{ExecutionPriority::kLow})
                  .ok());
  while (!low.started.load(std::memory_order_acquire))
    std::this_thread::yield();

  std::atomic<bool> high_ran{false};
  ASSERT_TRUE(scheduler
                  .schedule(
                      +[](void* raw) noexcept {
                        static_cast<std::atomic<bool>*>(raw)->store(
                            true, std::memory_order_release);
                      },
                      &high_ran, ExecutionOptions{ExecutionPriority::kHigh})
                  .ok());
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  EXPECT_FALSE(high_ran.load(std::memory_order_acquire));
  low.release.store(true, std::memory_order_release);
  scheduler.drain();
  EXPECT_TRUE(high_ran.load(std::memory_order_acquire));
  scheduler.shutdown();
}

TEST(SchedulerPriority, FiberYieldRetainsLane) {
  Scheduler scheduler(1, system_allocator(), 64, priority_options());
  OrderedLog log;
  BlockingRecord blocker;
  ASSERT_TRUE(scheduler.schedule(&blocking_record, &blocker).ok());
  while (!blocker.started.load(std::memory_order_acquire))
    std::this_thread::yield();

  YieldRecord yielded{&log};
  Fiber high(hyle::sched::kDefaultStackSize, &yield_record, &yielded);
  RecordValue normal{&log, 3};
  ASSERT_TRUE(
      scheduler.spawn(&high, ExecutionOptions{ExecutionPriority::kHigh}).ok());
  ASSERT_TRUE(scheduler.schedule(&record_value, &normal).ok());
  blocker.release.store(true, std::memory_order_release);
  scheduler.drain();

  ASSERT_EQ(log.size.load(std::memory_order_acquire), 3u);
  EXPECT_EQ(log.values[0], 1);
  EXPECT_EQ(log.values[1], 2);
  EXPECT_EQ(log.values[2], 3);
  EXPECT_EQ(yielded.first_priority.load(),
            static_cast<int>(ExecutionPriority::kHigh));
  EXPECT_EQ(yielded.second_priority.load(),
            static_cast<int>(ExecutionPriority::kHigh));
  scheduler.shutdown();
}

TEST(SchedulerPriority, ForeignWakeRetainsLaneAndRunsBeforeNormal) {
  Scheduler scheduler(1, system_allocator(), 64, priority_options());
  OrderedLog log;
  WakeRecord wake;
  wake.log = &log;
  Fiber high(hyle::sched::kDefaultStackSize, &wake_record, &wake);
  ASSERT_TRUE(
      scheduler.spawn(&high, ExecutionOptions{ExecutionPriority::kHigh}).ok());
  while (!wake.armed.load(std::memory_order_acquire)) std::this_thread::yield();

  BlockingRecord low;
  ASSERT_TRUE(scheduler
                  .schedule(&blocking_record, &low,
                            ExecutionOptions{ExecutionPriority::kLow})
                  .ok());
  while (!low.started.load(std::memory_order_acquire))
    std::this_thread::yield();
  RecordValue normal{&log, 2};
  ASSERT_TRUE(scheduler.schedule(&record_value, &normal).ok());
  wake.waker.wake();
  low.release.store(true, std::memory_order_release);
  scheduler.drain();

  ASSERT_EQ(log.size.load(std::memory_order_acquire), 2u);
  EXPECT_EQ(log.values[0], 1);
  EXPECT_EQ(log.values[1], 2);
  EXPECT_EQ(wake.observed_priority.load(),
            static_cast<int>(ExecutionPriority::kHigh));
  scheduler.shutdown();
}

TEST(SchedulerPriority, SameSchedulerChildrenInheritAndMayOverride) {
  Scheduler scheduler(1, system_allocator(), 64, priority_options());
  InheritRecord record{&scheduler};
  ASSERT_TRUE(scheduler
                  .schedule(&submit_children, &record,
                            ExecutionOptions{ExecutionPriority::kHigh})
                  .ok());
  scheduler.drain();
  EXPECT_EQ(record.parent.load(), static_cast<int>(ExecutionPriority::kHigh));
  EXPECT_EQ(record.child.load(), static_cast<int>(ExecutionPriority::kHigh));
  EXPECT_EQ(record.overridden.load(),
            static_cast<int>(ExecutionPriority::kLow));
  scheduler.shutdown();
}

TEST(SchedulerPriority, CrossSchedulerSubmissionDoesNotInherit) {
  Scheduler source(1, system_allocator(), 64, priority_options());
  Scheduler target(1, system_allocator(), 64, priority_options());
  CrossSchedulerRecord record{&target};
  ASSERT_TRUE(source
                  .schedule(&submit_cross_scheduler_child, &record,
                            ExecutionOptions{ExecutionPriority::kHigh})
                  .ok());
  source.drain();
  target.drain();
  EXPECT_EQ(record.source.load(), static_cast<int>(ExecutionPriority::kHigh));
  EXPECT_EQ(record.target_child.load(),
            static_cast<int>(ExecutionPriority::kNormal));
  source.shutdown();
  target.shutdown();
}

TEST(SchedulerPriority, TimerWakeRetainsFiberPriority) {
  Scheduler scheduler(1, system_allocator(), 64, priority_options());
  TimerPriorityRecord record;
  ASSERT_TRUE(
      scheduler
          .spawn_detached(&sleep_and_record_priority, &record,
                          FiberLaunchOptions{ExecutionPriority::kHigh,
                                             hyle::sched::kDefaultStackSize})
          .ok());
  scheduler.drain();
  EXPECT_EQ(record.before.load(), static_cast<int>(ExecutionPriority::kHigh));
  EXPECT_EQ(record.after.load(), static_cast<int>(ExecutionPriority::kHigh));
  scheduler.shutdown();
}

TEST(SchedulerPriority, TypedHelpersPreserveRequestedPriority) {
  Scheduler scheduler(1, system_allocator(), 64, priority_options());
  auto scheduled = hyle::sched::schedule_task(
      scheduler, system_allocator(),
      []() noexcept { return static_cast<int>(current_execution_priority()); },
      ExecutionOptions{ExecutionPriority::kHigh});
  auto spawned = hyle::sched::spawn_task(
      scheduler, system_allocator(), return_task_priority(system_allocator()),
      FiberLaunchOptions{ExecutionPriority::kLow,
                         hyle::sched::kDefaultStackSize});
  auto scheduled_result = scheduled.join();
  auto spawned_result = spawned.join();
  ASSERT_TRUE(scheduled_result.has_value());
  ASSERT_TRUE(spawned_result.has_value());
  EXPECT_EQ(*scheduled_result, static_cast<int>(ExecutionPriority::kHigh));
  EXPECT_EQ(*spawned_result, static_cast<int>(ExecutionPriority::kLow));
  scheduler.shutdown();
}

TEST(SchedulerPriority, OwningConveniencesAcceptPriorityOptions) {
  Scheduler scheduler(1, system_allocator(), 64, priority_options());
  std::atomic<int> scheduled{-1};
  std::atomic<int> detached{-1};
  ASSERT_TRUE(scheduler
                  .schedule(
                      [&scheduled]() noexcept {
                        scheduled.store(
                            static_cast<int>(current_execution_priority()),
                            std::memory_order_release);
                      },
                      ExecutionOptions{ExecutionPriority::kHigh})
                  .ok());
  ASSERT_TRUE(scheduler
                  .spawn_detached(
                      [&detached]() noexcept {
                        detached.store(
                            static_cast<int>(current_execution_priority()),
                            std::memory_order_release);
                      },
                      FiberLaunchOptions{ExecutionPriority::kLow,
                                         hyle::sched::kDefaultStackSize})
                  .ok());
  scheduler.drain();
  EXPECT_EQ(scheduled.load(), static_cast<int>(ExecutionPriority::kHigh));
  EXPECT_EQ(detached.load(), static_cast<int>(ExecutionPriority::kLow));
  scheduler.shutdown();
}

TEST(SchedulerPriority, PinnedPlacementAndPriorityAreOrthogonal) {
  Scheduler scheduler(2, system_allocator(), 64, priority_options());
  PinnedRecord record;
  ASSERT_TRUE(
      scheduler
          .spawn_detached_on(1, &record_pinned, &record,
                             FiberLaunchOptions{ExecutionPriority::kHigh,
                                                hyle::sched::kDefaultStackSize})
          .ok());
  scheduler.drain();
  EXPECT_EQ(record.worker.load(), 1);
  EXPECT_EQ(record.priority.load(), static_cast<int>(ExecutionPriority::kHigh));
  scheduler.shutdown();
}

TEST(SchedulerPriority, TaskScopeDefaultAndPerChildOverridePersistAcrossYield) {
  Scheduler scheduler(1, system_allocator(), 64, priority_options());
  auto created = TaskScope::create(scheduler, system_allocator(),
                                   TaskScopeOptions{ExecutionPriority::kHigh});
  ASSERT_TRUE(created.has_value()) << created.error().message();
  TaskScope scope = *std::move(created);
  TaskPriorityRecord inherited;
  TaskPriorityRecord overridden;
  TaskPriorityRecord added_inherited;
  TaskPriorityRecord added_overridden;
  auto high = scope.spawn(record_task_priority(system_allocator(), &inherited));
  auto low = scope.spawn(record_task_priority(system_allocator(), &overridden),
                         FiberLaunchOptions{ExecutionPriority::kLow,
                                            hyle::sched::kDefaultStackSize});
  ASSERT_TRUE(
      scope.add(record_task_priority(system_allocator(), &added_inherited))
          .ok());
  ASSERT_TRUE(
      scope
          .add(record_task_priority(system_allocator(), &added_overridden),
               FiberLaunchOptions{ExecutionPriority::kLow,
                                  hyle::sched::kDefaultStackSize})
          .ok());
  EXPECT_TRUE(scope.join().ok());
  EXPECT_TRUE(high.join().has_value());
  EXPECT_TRUE(low.join().has_value());
  for (std::atomic<int>* value :
       {&inherited.first, &inherited.second, &added_inherited.first,
        &added_inherited.second}) {
    EXPECT_EQ(value->load(), static_cast<int>(ExecutionPriority::kHigh));
  }
  for (std::atomic<int>* value :
       {&overridden.first, &overridden.second, &added_overridden.first,
        &added_overridden.second}) {
    EXPECT_EQ(value->load(), static_cast<int>(ExecutionPriority::kLow));
  }
  scheduler.shutdown();
}

TEST(SchedulerPriority, ShutdownWakeRetainsLaneAndClosesPriorityAdmission) {
  Scheduler scheduler(1, system_allocator(), 64, priority_options());
  ShutdownPriorityRecord record;
  Fiber high(hyle::sched::kDefaultStackSize,
             &wait_for_shutdown_and_record_priority, &record);
  ASSERT_TRUE(
      scheduler.spawn(&high, ExecutionOptions{ExecutionPriority::kHigh}).ok());
  while (!record.started.load(std::memory_order_acquire))
    std::this_thread::yield();

  scheduler.begin_shutdown();
  scheduler.drain();
  EXPECT_EQ(record.resumed_priority.load(std::memory_order_acquire),
            static_cast<int>(ExecutionPriority::kHigh));
  EXPECT_TRUE(
      scheduler
          .schedule(&no_op, nullptr,
                    ExecutionOptions{ExecutionPriority::kHigh})
          .IsCancelled());
  scheduler.shutdown();
}

TEST(SchedulerPriority, TaskScopeCancellationRetainsDefaultLane) {
  Scheduler scheduler(1, system_allocator(), 64, priority_options());
  auto created = TaskScope::create(scheduler, system_allocator(),
                                   TaskScopeOptions{ExecutionPriority::kHigh});
  ASSERT_TRUE(created.has_value()) << created.error().message();
  TaskScope scope = *std::move(created);
  CancelPriorityRecord record;
  auto child = scope.spawn(
      wait_for_cancel_and_record_priority(system_allocator(), &record));
  child.detach();
  while (!record.started.load(std::memory_order_acquire))
    std::this_thread::yield();

  EXPECT_TRUE(scope.cancel_and_join().IsCancelled());
  EXPECT_EQ(record.observed_priority.load(std::memory_order_acquire),
            static_cast<int>(ExecutionPriority::kHigh));
  scheduler.shutdown();
}

TEST(SchedulerPriority, ConcurrentProducersConserveEveryLaneItem) {
  Scheduler scheduler(4, system_allocator(), 64, priority_options());
  std::atomic<unsigned> completed{0};
  std::atomic<unsigned> rejected{0};
  constexpr unsigned kProducers = 4;
  constexpr unsigned kPerProducer = 128;
  std::array<std::thread, kProducers> producers;
  for (unsigned producer = 0; producer < kProducers; ++producer) {
    producers[producer] = std::thread([&, producer] {
      for (unsigned item = 0; item < kPerProducer; ++item) {
        const auto priority = static_cast<ExecutionPriority>(
            (producer + item) % 3);
        auto status = scheduler.schedule(
            +[](void* raw) noexcept {
              static_cast<std::atomic<unsigned>*>(raw)->fetch_add(
                  1, std::memory_order_relaxed);
            },
            &completed, ExecutionOptions{priority});
        if (!status.ok())
          rejected.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto& producer : producers) producer.join();
  scheduler.drain();

  EXPECT_EQ(rejected.load(std::memory_order_relaxed), 0u);
  EXPECT_EQ(completed.load(std::memory_order_relaxed),
            kProducers * kPerProducer);
  scheduler.shutdown();
}

TEST(SchedulerPriority, RecursiveLaneSaturationFailsWithoutSelfDeadlock) {
  Scheduler scheduler(1, system_allocator(), 2, priority_options());
  SaturationRecord record{&scheduler};
  ASSERT_TRUE(scheduler
                  .schedule(&saturate_priority_lane, &record,
                            ExecutionOptions{ExecutionPriority::kHigh})
                  .ok());
  scheduler.drain();
  EXPECT_GT(record.admitted.load(), 0);
  EXPECT_TRUE(record.rejected.load(std::memory_order_acquire));
  scheduler.shutdown();
}
