#include "hyle/sched/task_scope.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

#include "hyle/error/status.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/join.h"
#include "hyle/sched/scheduler.h"

#if defined(HYLE_TESTING)
namespace hyle::sched::detail {
using TaskScopeFinalPublishHookForTesting = void (*)(void*) noexcept;
HYLE_SCHED_API void set_task_scope_final_publish_hook_for_testing(
    TaskScopeFinalPublishHookForTesting hook, void* context) noexcept;
}  // namespace hyle::sched::detail
#endif

using hyle::Status;
using hyle::mem::AllocatorRef;
using hyle::mem::system_allocator;
using hyle::sched::CancelCallback;
using hyle::sched::Fiber;
using hyle::sched::is_cancel_requested;
using hyle::sched::Scheduler;
using hyle::sched::Task;
using hyle::sched::TaskScope;

namespace {

Task<void> increment(AllocatorRef, std::atomic<int>* value) {
  value->fetch_add(1, std::memory_order_relaxed);
  co_return;
}

Task<int> value_task(AllocatorRef, int value) { co_return value; }

Task<Status> status_task(AllocatorRef, Status status,
                         std::atomic<int>* completed = nullptr) {
  if (completed != nullptr) completed->fetch_add(1, std::memory_order_relaxed);
  co_return status;
}

Task<void> throwing_task(AllocatorRef) {
  throw 42;
  co_return;
}

Task<Status> wait_for_cancel(AllocatorRef, std::atomic<bool>* started,
                             std::atomic<bool>* observed) {
  started->store(true, std::memory_order_release);
  while (!is_cancel_requested()) Fiber::yield();
  observed->store(true, std::memory_order_release);
  co_return Status::Cancelled("child observed scope cancellation");
}

Task<void> wait_for_release(AllocatorRef, std::atomic<bool>* started,
                            std::atomic<bool>* release) {
  started->store(true, std::memory_order_release);
  while (!release->load(std::memory_order_acquire)) Fiber::yield();
  co_return;
}

#if defined(HYLE_TESTING)
Task<Status> status_wait_for_release(AllocatorRef, std::atomic<bool>* release) {
  while (!release->load(std::memory_order_acquire)) Fiber::yield();
  co_return Status{};
}
#endif

Task<Status> join_from_fiber(AllocatorRef allocator, Scheduler* scheduler,
                             std::atomic<int>* completed) {
  auto scope_result = TaskScope::create(*scheduler, allocator);
  if (!scope_result) co_return scope_result.error();
  TaskScope scope = *std::move(scope_result);
  for (int i = 0; i < 16; ++i) {
    auto child = scope.spawn(increment(allocator, completed));
    child.detach();
  }
  co_return scope.join();
}

Task<Status> cancel_and_join_from_fiber(AllocatorRef allocator,
                                        Scheduler* scheduler,
                                        std::atomic<bool>* observed) {
  auto scope_result = TaskScope::create(*scheduler, allocator);
  if (!scope_result) co_return scope_result.error();
  TaskScope scope = *std::move(scope_result);
  std::atomic<bool> started{false};
  auto child = scope.spawn(wait_for_cancel(allocator, &started, observed));
  child.detach();
  while (!started.load(std::memory_order_acquire)) Fiber::yield();
  co_return scope.cancel_and_join();
}

Task<void> spawn_descendant(AllocatorRef allocator, TaskScope* scope,
                            std::atomic<int>* completed) {
  completed->fetch_add(1, std::memory_order_relaxed);
  auto child = scope->spawn(increment(allocator, completed));
  child.detach();
  co_return;
}

Task<Status> saturate_scope_from_worker(AllocatorRef allocator,
                                        Scheduler* scheduler,
                                        std::atomic<int>* ran) {
  auto scope_result = TaskScope::create(*scheduler, allocator);
  if (!scope_result) co_return scope_result.error();
  TaskScope scope = *std::move(scope_result);
  for (int i = 0; i < 16; ++i) {
    auto child = scope.spawn(increment(allocator, ran));
    child.detach();
  }
  co_return scope.join();
}

struct MoveOnly {
  explicit MoveOnly(int input) noexcept : value(input) {}
  MoveOnly(MoveOnly&& other) noexcept : value(std::exchange(other.value, -1)) {}
  MoveOnly& operator=(MoveOnly&& other) noexcept {
    value = std::exchange(other.value, -1);
    return *this;
  }
  MoveOnly(const MoveOnly&) = delete;
  MoveOnly& operator=(const MoveOnly&) = delete;
  int value;
};

Task<MoveOnly> move_only_task(AllocatorRef, int value) {
  co_return MoveOnly(value);
}

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
  std::atomic<std::size_t> fail_on{(std::numeric_limits<std::size_t>::max)()};
};

void* counted_alloc(void* raw, std::size_t size,
                    std::size_t alignment) noexcept {
  auto* counter = static_cast<AllocationCounter*>(raw);
  const std::size_t attempt =
      counter->attempts.fetch_add(1, std::memory_order_relaxed);
  if (attempt == counter->fail_on.load(std::memory_order_relaxed))
    return nullptr;
  void* result = counter->backing.allocate(size, alignment);
  if (result != nullptr)
    counter->allocations.fetch_add(1, std::memory_order_relaxed);
  return result;
}

void counted_free(void* raw, void* ptr, std::size_t size,
                  std::size_t alignment) noexcept {
  auto* counter = static_cast<AllocationCounter*>(raw);
  if (ptr != nullptr) counter->frees.fetch_add(1, std::memory_order_relaxed);
  counter->backing.deallocate(ptr, size, alignment);
}

void* counted_realloc(void* raw, void* ptr, std::size_t old_size,
                      std::size_t new_size, std::size_t alignment) noexcept {
  auto* counter = static_cast<AllocationCounter*>(raw);
  return counter->backing.reallocate(ptr, old_size, new_size, alignment);
}

AllocatorRef counted_allocator(AllocationCounter* counter) noexcept {
  return AllocatorRef(&counted_alloc, &counted_free, &counted_realloc, counter);
}

void mark_cancelled(void* raw) noexcept {
  static_cast<std::atomic<bool>*>(raw)->store(true, std::memory_order_release);
}

#if defined(HYLE_TESTING)
struct FinalPublishProbe {
  std::atomic<bool> entered{false};
  std::atomic<bool> release{false};
};

void pause_final_publish(void* raw) noexcept {
  auto* probe = static_cast<FinalPublishProbe*>(raw);
  probe->entered.store(true, std::memory_order_release);
  while (!probe->release.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
}
#endif

}  // namespace

TEST(TaskScope, EmptyJoinClosesAndIsRepeatable) {
  Scheduler scheduler(1, system_allocator(), 64);
  auto created = TaskScope::create(scheduler, system_allocator());
  ASSERT_TRUE(created.has_value());
  TaskScope scope = *std::move(created);

  EXPECT_EQ(scope.remaining(), 0u);
  EXPECT_FALSE(scope.is_closed());
  EXPECT_TRUE(scope.join().ok());
  EXPECT_TRUE(scope.is_closed());
  EXPECT_TRUE(scope.join().ok());
  scheduler.shutdown();
}

TEST(TaskScope, DynamicChildrenDetachButScopeStillDrains) {
  Scheduler scheduler(4, system_allocator(), 1024);
  auto created = TaskScope::create(scheduler, system_allocator());
  ASSERT_TRUE(created.has_value());
  TaskScope scope = *std::move(created);
  std::atomic<int> completed{0};

  for (int i = 0; i < 257; ++i) {
    auto child = scope.spawn(increment(system_allocator(), &completed));
    ASSERT_TRUE(child.valid());
    child.detach();
  }
  EXPECT_TRUE(scope.join().ok());
  EXPECT_EQ(completed.load(std::memory_order_relaxed), 257);
  EXPECT_EQ(scope.remaining(), 0u);
  scheduler.shutdown();
}

#if defined(HYLE_TESTING)
TEST(TaskScope, FinalCompletionCannotPublishBeforeItsLastStateAccess) {
  Scheduler scheduler(1, system_allocator(), 64);
  auto created = TaskScope::create(scheduler, system_allocator());
  ASSERT_TRUE(created.has_value());
  TaskScope scope = *std::move(created);
  FinalPublishProbe probe;
  hyle::sched::detail::set_task_scope_final_publish_hook_for_testing(
      &pause_final_publish, &probe);

  std::atomic<bool> child_release{false};
  ASSERT_TRUE(
      scope.add(status_wait_for_release(system_allocator(), &child_release))
          .ok());
  scope.close();
  child_release.store(true, std::memory_order_release);
  while (!probe.entered.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  // completed == admitted is already externally visible here, but the final
  // publisher still owns wait_lock and has not made its last state access.
  Status joined;
  std::atomic<bool> join_started{false};
  std::atomic<bool> join_returned{false};
  std::thread joiner([&] {
    join_started.store(true, std::memory_order_release);
    joined = scope.join();
    join_returned.store(true, std::memory_order_release);
  });
  while (!join_started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  const bool returned_before_publisher_exit =
      join_returned.load(std::memory_order_acquire);
  probe.release.store(true, std::memory_order_release);
  joiner.join();
  hyle::sched::detail::set_task_scope_final_publish_hook_for_testing(nullptr,
                                                                     nullptr);

  EXPECT_FALSE(returned_before_publisher_exit);
  EXPECT_TRUE(joined.ok());
  scheduler.shutdown();
}
#endif

TEST(TaskScope, ScopeOwnedAddAggregatesStatusAndDrainsVoidChildren) {
  Scheduler scheduler(2, system_allocator(), 128);
  auto created = TaskScope::create(scheduler, system_allocator());
  ASSERT_TRUE(created.has_value());
  TaskScope scope = *std::move(created);
  std::atomic<int> status_completed{0};
  std::atomic<int> void_completed{0};

  EXPECT_TRUE(scope.add(status_task(system_allocator(), Status::NotFound(
                                                            "owned child"),
                                    &status_completed))
                  .ok());
  for (int i = 0; i < 32; ++i) {
    EXPECT_TRUE(
        scope.add(increment(system_allocator(), &void_completed)).ok());
  }

  Status joined = scope.join();
  EXPECT_TRUE(joined.IsNotFound());
  EXPECT_EQ(joined.message(), "owned child");
  EXPECT_EQ(status_completed.load(std::memory_order_relaxed), 1);
  EXPECT_EQ(void_completed.load(std::memory_order_relaxed), 32);
  scheduler.shutdown();
}

TEST(TaskScope, ScopeOwnedAddConvertsVoidExceptionToAggregateFailure) {
  Scheduler scheduler(1, system_allocator(), 64);
  auto created = TaskScope::create(scheduler, system_allocator());
  ASSERT_TRUE(created.has_value());
  TaskScope scope = *std::move(created);

  EXPECT_TRUE(scope.add(throwing_task(system_allocator())).ok());
  EXPECT_TRUE(scope.join().IsInternal());
  scheduler.shutdown();
}

TEST(TaskScope, ScopeOwnedAddPropagatesScopeCancellation) {
  Scheduler scheduler(2, system_allocator(), 128);
  auto created = TaskScope::create(scheduler, system_allocator());
  ASSERT_TRUE(created.has_value());
  TaskScope scope = *std::move(created);
  std::atomic<bool> started{false};
  std::atomic<bool> observed{false};

  EXPECT_TRUE(scope.add(wait_for_cancel(system_allocator(), &started, &observed))
                  .ok());
  while (!started.load(std::memory_order_acquire)) std::this_thread::yield();
  EXPECT_TRUE(scope.cancel_and_join().IsCancelled());
  EXPECT_TRUE(observed.load(std::memory_order_acquire));
  scheduler.shutdown();
}

TEST(TaskScope, CancellationTokenIsShareableWithExternalOperations) {
  Scheduler scheduler(1, system_allocator(), 64);
  auto created = TaskScope::create(scheduler, system_allocator());
  ASSERT_TRUE(created.has_value());
  TaskScope scope = *std::move(created);
  auto token = scope.cancellation_token();
  ASSERT_TRUE(token.valid());
  std::atomic<bool> callback_fired{false};
  CancelCallback callback(token, &mark_cancelled, &callback_fired);

  token.cancel();
  EXPECT_TRUE(token.is_cancelled());
  EXPECT_TRUE(scope.is_cancel_requested());
  EXPECT_TRUE(callback_fired.load(std::memory_order_acquire));
  EXPECT_TRUE(scope.join().ok());
  scheduler.shutdown();
}

TEST(TaskScope, ConcurrentScopeOwnedAddAndCloseHasNoHalfAdmission) {
  constexpr int kThreads = 4;
  constexpr int kAttempts = 128;
  Scheduler scheduler(4, system_allocator(), 4096);
  auto created = TaskScope::create(scheduler, system_allocator());
  ASSERT_TRUE(created.has_value());
  TaskScope scope = *std::move(created);
  std::atomic<bool> start{false};
  std::atomic<int> ran{0};
  std::atomic<int> accepted{0};
  std::atomic<int> rejected{0};
  std::vector<std::thread> submitters;

  for (int thread = 0; thread < kThreads; ++thread) {
    submitters.emplace_back([&] {
      while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
      for (int i = 0; i < kAttempts; ++i) {
        Status status = scope.add(increment(system_allocator(), &ran));
        if (status.ok()) {
          accepted.fetch_add(1, std::memory_order_relaxed);
        } else {
          EXPECT_TRUE(status.IsFailedPrecondition());
          rejected.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  start.store(true, std::memory_order_release);
  scope.close();
  for (auto& submitter : submitters) submitter.join();
  EXPECT_TRUE(scope.join().ok());
  EXPECT_EQ(accepted.load() + rejected.load(), kThreads * kAttempts);
  EXPECT_EQ(ran.load(), accepted.load());
  scheduler.shutdown();
}

TEST(TaskScope, ScopeOwnedAddReportsOnlyPreAdmissionRejectionDirectly) {
  Scheduler scheduler(1, system_allocator(), 64);

  {
    auto created = TaskScope::create(scheduler, system_allocator());
    ASSERT_TRUE(created.has_value());
    TaskScope scope = *std::move(created);
    std::atomic<int> ran{0};
    scope.close();
    Status rejected = scope.add(increment(system_allocator(), &ran));
    EXPECT_TRUE(rejected.IsFailedPrecondition());
    EXPECT_TRUE(scope.join().ok());
    EXPECT_EQ(ran.load(std::memory_order_relaxed), 0);
  }

  {
    auto created = TaskScope::create(scheduler, system_allocator());
    ASSERT_TRUE(created.has_value());
    TaskScope scope = *std::move(created);
    std::atomic<int> ran{0};
    Status rejected = scope.add(increment(failing_allocator(), &ran));
    EXPECT_TRUE(rejected.IsResourceExhausted());
    EXPECT_TRUE(scope.join().ok());
    EXPECT_EQ(ran.load(std::memory_order_relaxed), 0);
  }

  scheduler.shutdown();
}

TEST(TaskScope, ScopeOwnedPostAdmissionOomIsDeferredToJoin) {
  Scheduler scheduler(1, system_allocator(), 64);
  AllocationCounter counter;
  {
    AllocatorRef allocator = counted_allocator(&counter);
    auto created = TaskScope::create(scheduler, allocator);
    ASSERT_TRUE(created.has_value());
    TaskScope scope = *std::move(created);
    std::atomic<int> ran{0};
    const std::size_t next_attempt = counter.attempts.load();
    counter.fail_on.store(next_attempt);  // scope-owned driver allocation

    EXPECT_TRUE(scope.add(increment(system_allocator(), &ran)).ok());
    EXPECT_EQ(counter.attempts.load(), next_attempt + 1);
    EXPECT_TRUE(scope.join().IsResourceExhausted());
    EXPECT_EQ(ran.load(std::memory_order_relaxed), 0);
  }
  EXPECT_EQ(counter.allocations.load(), counter.frees.load());
  scheduler.shutdown();
}

TEST(TaskScope, ScopeOwnedStackAndSchedulerFailuresAreDeferredToJoin) {
  {
    Scheduler scheduler(1, system_allocator(), 64);
    auto created = TaskScope::create(scheduler, system_allocator());
    ASSERT_TRUE(created.has_value());
    TaskScope scope = *std::move(created);
    EXPECT_TRUE(scope.add(status_task(system_allocator(), Status{}),
                          (std::numeric_limits<std::size_t>::max)())
                    .ok());
    EXPECT_TRUE(scope.join().IsOutOfRange());
    scheduler.shutdown();
  }

  {
    Scheduler scheduler(1, system_allocator(), 64);
    auto created = TaskScope::create(scheduler, system_allocator());
    ASSERT_TRUE(created.has_value());
    TaskScope scope = *std::move(created);
    scheduler.begin_shutdown();
    EXPECT_TRUE(scope.add(status_task(system_allocator(), Status{})).ok());
    EXPECT_TRUE(scope.join().IsCancelled());
    scheduler.shutdown();
  }
}

TEST(TaskScope, TypedAndMoveOnlyResultsRemainInJoinHandles) {
  Scheduler scheduler(2, system_allocator(), 128);
  auto created = TaskScope::create(scheduler, system_allocator());
  ASSERT_TRUE(created.has_value());
  TaskScope scope = *std::move(created);

  auto integer = scope.spawn(value_task(system_allocator(), 42));
  auto move_only = scope.spawn(move_only_task(system_allocator(), 73));
  EXPECT_TRUE(scope.join().ok());
  ASSERT_TRUE(integer.is_finished());
  ASSERT_TRUE(move_only.is_finished());
  EXPECT_EQ(*integer.join(), 42);
  auto moved = move_only.join();
  ASSERT_TRUE(moved.has_value());
  EXPECT_EQ(moved->value, 73);
  scheduler.shutdown();
}

TEST(TaskScope, AggregatesFirstStatusFailureAndStillDrains) {
  Scheduler scheduler(2, system_allocator(), 128);
  auto created = TaskScope::create(scheduler, system_allocator());
  ASSERT_TRUE(created.has_value());
  TaskScope scope = *std::move(created);
  std::atomic<int> completed{0};

  auto first = scope.spawn(status_task(
      system_allocator(), Status::NotFound("first child"), &completed));
  auto first_result = first.join();
  ASSERT_TRUE(first_result.has_value());
  ASSERT_TRUE(first_result->IsNotFound());

  auto second = scope.spawn(status_task(
      system_allocator(), Status::Internal("later child"), &completed));
  second.detach();
  Status joined = scope.join();
  EXPECT_TRUE(joined.IsNotFound());
  EXPECT_EQ(joined.message(), "first child");
  EXPECT_EQ(completed.load(std::memory_order_relaxed), 2);
  scheduler.shutdown();
}

TEST(TaskScope, VoidExceptionBecomesAggregateInternalFailure) {
  Scheduler scheduler(1, system_allocator(), 64);
  auto created = TaskScope::create(scheduler, system_allocator());
  ASSERT_TRUE(created.has_value());
  TaskScope scope = *std::move(created);

  auto child = scope.spawn(throwing_task(system_allocator()));
  child.detach();
  EXPECT_TRUE(scope.join().IsInternal());
  scheduler.shutdown();
}

TEST(TaskScope, CancelAndJoinIsCooperativeAndDrains) {
  Scheduler scheduler(2, system_allocator(), 128);
  auto created = TaskScope::create(scheduler, system_allocator());
  ASSERT_TRUE(created.has_value());
  TaskScope scope = *std::move(created);
  std::atomic<bool> started{false};
  std::atomic<bool> observed{false};

  auto child =
      scope.spawn(wait_for_cancel(system_allocator(), &started, &observed));
  child.detach();
  while (!started.load(std::memory_order_acquire)) std::this_thread::yield();
  Status joined = scope.cancel_and_join();
  EXPECT_TRUE(joined.IsCancelled());
  EXPECT_TRUE(observed.load(std::memory_order_acquire));
  EXPECT_TRUE(scope.is_cancel_requested());
  EXPECT_EQ(scope.remaining(), 0u);
  scheduler.shutdown();
}

TEST(TaskScope, JoinHandleCancellationComposesWithScopeToken) {
  Scheduler scheduler(2, system_allocator(), 128);
  auto created = TaskScope::create(scheduler, system_allocator());
  ASSERT_TRUE(created.has_value());
  TaskScope scope = *std::move(created);
  std::atomic<bool> started{false};
  std::atomic<bool> observed{false};

  auto child =
      scope.spawn(wait_for_cancel(system_allocator(), &started, &observed));
  while (!started.load(std::memory_order_acquire)) std::this_thread::yield();
  child.request_cancel();
  auto result = child.join();
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->IsCancelled());
  EXPECT_TRUE(scope.join().IsCancelled());
  EXPECT_TRUE(observed.load(std::memory_order_acquire));
  scheduler.shutdown();
}

TEST(TaskScope, CancellationBeforeAdmissionIsInherited) {
  Scheduler scheduler(1, system_allocator(), 64);
  auto created = TaskScope::create(scheduler, system_allocator());
  ASSERT_TRUE(created.has_value());
  TaskScope scope = *std::move(created);
  std::atomic<int> ran{0};

  scope.request_cancel();
  auto child = scope.spawn(increment(system_allocator(), &ran));
  auto result = child.join();
  ASSERT_FALSE(result.has_value());
  EXPECT_TRUE(result.error().IsCancelled());
  EXPECT_TRUE(scope.join().IsCancelled());
  EXPECT_EQ(ran.load(std::memory_order_relaxed), 0);
  scheduler.shutdown();
}

TEST(TaskScope, CloseRejectsLateAdmissionWithoutPoisoningJoin) {
  Scheduler scheduler(1, system_allocator(), 64);
  auto created = TaskScope::create(scheduler, system_allocator());
  ASSERT_TRUE(created.has_value());
  TaskScope scope = *std::move(created);
  std::atomic<int> ran{0};

  scope.close();
  auto rejected = scope.spawn(increment(system_allocator(), &ran));
  auto result = rejected.join();
  ASSERT_FALSE(result.has_value());
  EXPECT_TRUE(result.error().IsFailedPrecondition());
  EXPECT_TRUE(scope.join().ok());
  EXPECT_EQ(ran.load(std::memory_order_relaxed), 0);
  scheduler.shutdown();
}

TEST(TaskScope, ConcurrentSpawnAndCloseHasNoHalfAdmission) {
  constexpr int kThreads = 4;
  constexpr int kAttempts = 128;
  Scheduler scheduler(4, system_allocator(), 4096);
  auto created = TaskScope::create(scheduler, system_allocator());
  ASSERT_TRUE(created.has_value());
  TaskScope scope = *std::move(created);
  std::atomic<bool> start{false};
  std::atomic<int> ran{0};
  std::atomic<int> accepted{0};
  std::atomic<int> rejected{0};
  std::vector<std::thread> submitters;

  for (int thread = 0; thread < kThreads; ++thread) {
    submitters.emplace_back([&] {
      while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
      for (int i = 0; i < kAttempts; ++i) {
        auto child = scope.spawn(increment(system_allocator(), &ran));
        auto result = child.join();
        if (result.has_value()) {
          accepted.fetch_add(1, std::memory_order_relaxed);
        } else {
          EXPECT_TRUE(result.error().IsFailedPrecondition());
          rejected.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  start.store(true, std::memory_order_release);
  scope.close();
  for (auto& submitter : submitters) submitter.join();
  EXPECT_TRUE(scope.join().ok());
  EXPECT_EQ(accepted.load() + rejected.load(), kThreads * kAttempts);
  EXPECT_EQ(ran.load(), accepted.load());
  scheduler.shutdown();
}

TEST(TaskScope, MultipleExternalJoinersReceiveSameError) {
  Scheduler scheduler(2, system_allocator(), 128);
  auto created = TaskScope::create(scheduler, system_allocator());
  ASSERT_TRUE(created.has_value());
  TaskScope scope = *std::move(created);
  std::atomic<bool> started{false};
  std::atomic<bool> release{false};

  auto child =
      scope.spawn(wait_for_release(system_allocator(), &started, &release));
  child.detach();
  while (!started.load(std::memory_order_acquire)) std::this_thread::yield();

  Status left;
  Status right;
  std::thread a([&] { left = scope.join(); });
  std::thread b([&] { right = scope.join(); });
  release.store(true, std::memory_order_release);
  a.join();
  b.join();
  EXPECT_TRUE(left.ok());
  EXPECT_TRUE(right.ok());
  scheduler.shutdown();
}

TEST(TaskScope, JoinFromSchedulerFiberIsFiberAware) {
  Scheduler scheduler(2, system_allocator(), 128);
  std::atomic<int> completed{0};
  auto parent = hyle::sched::spawn_task(
      scheduler, system_allocator(),
      join_from_fiber(system_allocator(), &scheduler, &completed));
  auto result = parent.join();
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->ok());
  EXPECT_EQ(completed.load(std::memory_order_relaxed), 16);
  scheduler.shutdown();
}

TEST(TaskScope, CancelAndJoinFromSchedulerFiberIsFiberAware) {
  Scheduler scheduler(2, system_allocator(), 128);
  std::atomic<bool> observed{false};
  auto parent = hyle::sched::spawn_task(
      scheduler, system_allocator(),
      cancel_and_join_from_fiber(system_allocator(), &scheduler, &observed));
  auto result = parent.join();
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->IsCancelled());
  EXPECT_TRUE(observed.load(std::memory_order_acquire));
  scheduler.shutdown();
}

TEST(TaskScope, AdmittedChildMayAddADescendantBeforeClose) {
  Scheduler scheduler(2, system_allocator(), 128);
  auto created = TaskScope::create(scheduler, system_allocator());
  ASSERT_TRUE(created.has_value());
  TaskScope scope = *std::move(created);
  std::atomic<int> completed{0};

  auto parent =
      scope.spawn(spawn_descendant(system_allocator(), &scope, &completed));
  ASSERT_TRUE(parent.join().has_value());
  EXPECT_TRUE(scope.join().ok());
  EXPECT_EQ(completed.load(std::memory_order_relaxed), 2);
  scheduler.shutdown();
}

TEST(TaskScope, CreationAndPreAdmissionFailuresAreTyped) {
  Scheduler scheduler(1, system_allocator(), 64);
  auto no_allocator = TaskScope::create(scheduler, AllocatorRef{});
  ASSERT_FALSE(no_allocator.has_value());
  EXPECT_TRUE(no_allocator.error().IsInvalidArgument());

  auto no_cancel_tree = TaskScope::create(scheduler, failing_allocator());
  ASSERT_FALSE(no_cancel_tree.has_value());
  EXPECT_TRUE(no_cancel_tree.error().IsResourceExhausted());

  auto created = TaskScope::create(scheduler, system_allocator());
  ASSERT_TRUE(created.has_value());
  TaskScope scope = *std::move(created);
  auto no_frame = scope.spawn(value_task(failing_allocator(), 42));
  auto frame_result = no_frame.join();
  ASSERT_FALSE(frame_result.has_value());
  EXPECT_TRUE(frame_result.error().IsResourceExhausted());
  EXPECT_TRUE(scope.join().ok());
  scheduler.shutdown();
}

TEST(TaskScope, ScopeStateJoinStateAndDriverOomCannotStrandJoin) {
  Scheduler scheduler(1, system_allocator(), 64);

  {
    AllocationCounter counter;
    counter.fail_on.store(2);  // cancel tree + node succeed; scope state fails
    auto no_state = TaskScope::create(scheduler, counted_allocator(&counter));
    ASSERT_FALSE(no_state.has_value());
    EXPECT_TRUE(no_state.error().IsResourceExhausted());
    EXPECT_EQ(counter.allocations.load(), counter.frees.load());
  }

  {
    AllocationCounter counter;
    AllocatorRef allocator = counted_allocator(&counter);
    auto created = TaskScope::create(scheduler, allocator);
    ASSERT_TRUE(created.has_value());
    TaskScope scope = *std::move(created);
    counter.fail_on.store(counter.attempts.load());  // join state
    auto child = scope.spawn(value_task(system_allocator(), 1));
    auto result = child.join();
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().IsResourceExhausted());
    EXPECT_TRUE(scope.join().IsResourceExhausted());
  }

  {
    AllocationCounter counter;
    AllocatorRef allocator = counted_allocator(&counter);
    auto created = TaskScope::create(scheduler, allocator);
    ASSERT_TRUE(created.has_value());
    TaskScope scope = *std::move(created);
    counter.fail_on.store(counter.attempts.load() + 1);  // driver
    auto child = scope.spawn(value_task(system_allocator(), 1));
    auto result = child.join();
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().IsResourceExhausted());
    EXPECT_TRUE(scope.join().IsResourceExhausted());
  }

  scheduler.shutdown();
}

TEST(TaskScope, StackAndSchedulerAdmissionFailureCannotStrandJoin) {
  {
    Scheduler scheduler(1, system_allocator(), 64);
    auto created = TaskScope::create(scheduler, system_allocator());
    ASSERT_TRUE(created.has_value());
    TaskScope scope = *std::move(created);
    auto child = scope.spawn(value_task(system_allocator(), 1),
                             (std::numeric_limits<std::size_t>::max)());
    auto result = child.join();
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().IsOutOfRange());
    Status joined = scope.join();
    EXPECT_TRUE(joined.IsOutOfRange());
    scheduler.shutdown();
  }

  {
    Scheduler scheduler(1, system_allocator(), 64);
    auto created = TaskScope::create(scheduler, system_allocator());
    ASSERT_TRUE(created.has_value());
    TaskScope scope = *std::move(created);
    scheduler.begin_shutdown();
    auto child = scope.spawn(value_task(system_allocator(), 1));
    auto result = child.join();
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().IsCancelled());
    EXPECT_TRUE(scope.join().IsCancelled());
    scheduler.shutdown();
  }
}

TEST(TaskScope, WorkerQueueSaturationIsAggregatedAndDoesNotLeakCount) {
  Scheduler scheduler(1, system_allocator(), 2);
  std::atomic<int> ran{0};
  auto parent = hyle::sched::spawn_task(
      scheduler, system_allocator(),
      saturate_scope_from_worker(system_allocator(), &scheduler, &ran));
  auto result = parent.join();
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->IsResourceExhausted());
  EXPECT_GT(ran.load(std::memory_order_relaxed), 0);
  EXPECT_LT(ran.load(std::memory_order_relaxed), 16);
  scheduler.shutdown();
}

TEST(TaskScope, MoveTransfersOwnerAndMovedFromCallsAreTyped) {
  Scheduler scheduler(1, system_allocator(), 64);
  auto created = TaskScope::create(scheduler, system_allocator());
  ASSERT_TRUE(created.has_value());
  TaskScope source = *std::move(created);
  TaskScope destination = std::move(source);
  EXPECT_FALSE(source.valid());
  EXPECT_TRUE(destination.valid());
  EXPECT_FALSE(source.cancellation_token().valid());
  EXPECT_TRUE(destination.cancellation_token().valid());

  auto rejected = source.spawn(value_task(system_allocator(), 1));
  auto rejected_result = rejected.join();
  ASSERT_FALSE(rejected_result.has_value());
  EXPECT_TRUE(rejected_result.error().IsInvalidArgument());
  EXPECT_TRUE(source.join().IsInvalidArgument());

  auto child = destination.spawn(value_task(system_allocator(), 42));
  EXPECT_TRUE(destination.join().ok());
  EXPECT_EQ(*child.join(), 42);
  scheduler.shutdown();
}

TEST(TaskScopeDeathTest, LiveChildCannotBeDetachedByDestruction) {
  EXPECT_DEATH(
      {
        Scheduler scheduler(1, system_allocator(), 64);
        std::atomic<bool> started{false};
        std::atomic<bool> release{false};
        auto created = TaskScope::create(scheduler, system_allocator());
        TaskScope scope = *std::move(created);
        auto child = scope.spawn(
            wait_for_release(system_allocator(), &started, &release));
        child.detach();
        while (!started.load(std::memory_order_acquire))
          std::this_thread::yield();
      },
      ".*");
}

TEST(TaskScopeDeathTest, UsedScopeRequiresExplicitJoin) {
  EXPECT_DEATH(
      {
        Scheduler scheduler(1, system_allocator(), 64);
        auto created = TaskScope::create(scheduler, system_allocator());
        TaskScope scope = *std::move(created);
        auto child = scope.spawn(value_task(system_allocator(), 1));
        (void)child.join();
      },
      ".*");
}
