#pragma once

#include <atomic>
#include <cstddef>
#include <new>
#include <optional>
#include <type_traits>
#include <utility>

#include "hyle/error/result.h"
#include "hyle/error/status.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/join.h"
#include "hyle/sched/scheduler.h"

namespace hyle::sched {

namespace detail {

struct TaskScopeState;

// One retained completion record per scoped child. It is embedded in the
// allocation that already owns the scoped driver: admission adds the scope
// reference, while the publisher keeps the initial reference until Event
// publication and its test-only tail are fully complete. Thus a waiter may
// tear down the TaskScope immediately after observing completion without
// freeing a record still touched by its publisher.
struct TaskScopeCompletion {
  using DisposeFn = void (*)(void*) noexcept;

  TaskScopeCompletion(void* owner_value, DisposeFn dispose_value) noexcept
      : owner(owner_value), dispose(dispose_value) {}
  TaskScopeCompletion(const TaskScopeCompletion&) = delete;
  TaskScopeCompletion& operator=(const TaskScopeCompletion&) = delete;

  // Admission is the only path that establishes the scope reference. The
  // record is still private to its driver here and the driver is not submitted
  // until after admission, so this is initialization rather than a concurrent
  // retain and does not need a locked read-modify-write.
  void retain_scope_before_publish() noexcept {
    references.store(2, std::memory_order_relaxed);
  }

  void release() noexcept {
    const unsigned previous =
        references.fetch_sub(1, std::memory_order_acq_rel);
    HYLE_FATAL_ASSERT(previous != 0);
    if (previous == 1) dispose(owner);
  }

  TaskScopeCompletion* next = nullptr;
  std::atomic<unsigned> references{1};  // publisher; scope adds one on admit
  Event completed{EventResetMode::kManual};
  void* owner = nullptr;
  DisposeFn dispose = nullptr;
};

// Compiled state operations keep synchronization and cancellation-tree layout
// out of the installed template header. Only scoped tasks call these seams;
// ordinary spawn_task/schedule paths do not carry a scope branch or field.
HYLE_SCHED_API mem::AllocatorRef task_scope_allocator(
    TaskScopeState* state) noexcept;
HYLE_SCHED_API Status task_scope_try_admit(
    TaskScopeState* state, TaskScopeCompletion* completion) noexcept;
HYLE_SCHED_API void task_scope_publish(TaskScopeState* state,
                                       TaskScopeCompletion* completion,
                                       const Status& status) noexcept;
HYLE_SCHED_API const CancelToken& task_scope_cancel_token(
    TaskScopeState* state) noexcept;

template <class T>
struct ScopedTaskDriver {
  TaskScopeCompletion completion;
  mem::AllocatorRef allocator;
  std::optional<Task<T>> task;
  JoinState<T>* join_state;
  TaskScopeState* scope_state;
  std::optional<Fiber> fiber;
  Status scope_status;

  ScopedTaskDriver(mem::AllocatorRef owner_allocator, Task<T> owner_task,
                   JoinState<T>* owner_join_state,
                   TaskScopeState* owner_scope_state, mem::FiberStack stack,
                   mem::FiberStackPool* pool) noexcept
      : completion(this, &ScopedTaskDriver::dispose),
        allocator(owner_allocator),
        task(std::move(owner_task)),
        join_state(owner_join_state),
        scope_state(owner_scope_state),
        fiber(std::in_place, std::move(stack), &ScopedTaskDriver::entry, this,
              pool) {}

  void arm_finish_hook() noexcept {
    fiber->set_on_finish(&ScopedTaskDriver::finish, this);
  }

  static void entry(void* raw) noexcept {
    auto* self = static_cast<ScopedTaskDriver*>(raw);
    JoinState<T>* join_state = self->join_state;
    Status scope_status;

    // Both cancellation sources are active only while user code runs. Tear
    // them down before publishing scope completion so join() may safely permit
    // destruction of the scope state and its token immediately after wake-up.
    {
      const CancelToken& scope_token =
          task_scope_cancel_token(self->scope_state);
      ScopedCancelFlag handle_cancel(&join_state->cancel_req);
      CancelScope scope_cancel(scope_token);
      if (join_state->cancel_req.load(std::memory_order_acquire) ||
          scope_token.is_cancelled()) {
        scope_status = Status::Cancelled("scoped task cancelled before start");
        join_state->set_error(scope_status);
      } else {
        self->task->resume();
        if ((join_state->cancel_req.load(std::memory_order_acquire) ||
             scope_token.is_cancelled()) &&
            !self->task->done()) {
          scope_status =
              Status::Cancelled("scoped task cancellation requested");
          join_state->set_error(scope_status);
        } else {
          try {
            if constexpr (std::is_void_v<T>) {
              self->task->result();
              join_state->set_done();
            } else {
              T value = self->task->result();
              if constexpr (std::is_same_v<T, Status>) {
                scope_status = value;
              }
              join_state->set_value(std::move(value));
            }
          } catch (...) {
            scope_status = Status::Internal("scoped task threw an exception");
            join_state->set_error(scope_status);
          }
        }
      }
    }

    self->scope_status = std::move(scope_status);
  }

  // Scheduler finish hook: only now is the Fiber off-stack. Release the Fiber
  // stack, coroutine frame and task-side JoinState reference before publishing
  // the record; scope.join() therefore includes all child cleanup.
  static void finish(void* raw) noexcept {
    auto* self = static_cast<ScopedTaskDriver*>(raw);
    JoinState<T>* join_state = self->join_state;
    self->fiber.reset();
    self->task.reset();
    join_state->release();
    task_scope_publish(self->scope_state, &self->completion,
                       self->scope_status);
    // May destroy `self` if scope teardown already released its reference.
    self->completion.release();
  }

  static void dispose(void* raw) noexcept {
    auto* self = static_cast<ScopedTaskDriver*>(raw);
    mem::AllocatorRef allocator = self->allocator;
    self->~ScopedTaskDriver();
    allocator.deallocate(self, sizeof(ScopedTaskDriver),
                         alignof(ScopedTaskDriver));
  }
};

}  // namespace detail

// Owns a runtime-sized set of Tasks submitted to one Scheduler. Admission is
// thread-safe and linearized with close(); join() closes admission and waits
// for every admitted child, whether its JoinHandle was retained or detached.
// Each child keeps its co-allocated completion cell until scope teardown. The
// Task, Fiber and stack are released before publication, but the cell storage
// remains; successful completions therefore do not contend on one scope-wide
// counter.
//
// TaskScope is deliberately separate from WaitGroup: it owns child lifetime,
// propagates one cooperative-cancellation source, and retains the first
// void/infrastructure or Task<Status> failure while still draining all work.
class HYLE_SCHED_API TaskScope {
 public:
  [[nodiscard]] static Result<TaskScope> create(
      Scheduler& scheduler, mem::AllocatorRef allocator) noexcept;

  TaskScope(TaskScope&& other) noexcept;
  TaskScope& operator=(TaskScope&& other) noexcept;
  TaskScope(const TaskScope&) = delete;
  TaskScope& operator=(const TaskScope&) = delete;
  ~TaskScope();

  // Admit and start a task on this scope's Scheduler. The returned handle owns
  // the typed result only; dropping it does not detach the task from the scope.
  // A closed/invalid scope or pre-admission validation failure returns an
  // already-failed handle and does not affect the scope's aggregate status.
  template <class T>
  [[nodiscard]] JoinHandle<T> spawn(
      Task<T> task, std::size_t stack_size = kDefaultStackSize) noexcept;

  // Add a child whose result is owned only by the scope. Prefer this route for
  // void/Status helpers when no per-child result or cancellation handle is
  // needed: it avoids manufacturing a JoinState/JoinHandle. A non-OK return
  // means ownership was not transferred (including driver/stack preparation or
  // closed admission). Once the completion record is admitted, Scheduler and
  // child failures are retained by join(), and add() returns OK.
  [[nodiscard]] Status add(Task<void> task,
                           std::size_t stack_size = kDefaultStackSize) noexcept;
  [[nodiscard]] Status add(Task<Status> task,
                           std::size_t stack_size = kDefaultStackSize) noexcept;

  // Stop future admission. Idempotent and safe to race with add()/spawn().
  void close() noexcept;

  // close + drain. Multiple waiters are supported and receive the same first
  // failure. Calling from a Scheduler fiber yields that fiber; calling from an
  // external thread parks the thread.
  [[nodiscard]] Status join() noexcept;

  // close + cooperative cancellation + drain. Cancellation does not preempt a
  // child; it must observe is_cancel_requested() or a cancellable HYLE wait.
  [[nodiscard]] Status cancel_and_join() noexcept;

  // Publish cancellation without closing admission. Later admitted children
  // inherit the already-cancelled scope token and finish as Cancelled unless
  // they have already completed normally.
  void request_cancel() noexcept;

  // A shareable copy of the scope cancellation source for explicit
  // cancellable IO/wait options. An invalid/moved-from scope returns an invalid
  // token; copying does not create a second cancellation tree.
  [[nodiscard]] CancelToken cancellation_token() const noexcept;

  // Control-plane snapshot. O(N) in admitted children because TaskScope avoids
  // a shared completion write on the worker hot path.
  [[nodiscard]] std::size_t remaining() const noexcept;
  [[nodiscard]] bool is_closed() const noexcept;
  [[nodiscard]] bool is_cancel_requested() const noexcept;
  [[nodiscard]] bool valid() const noexcept { return state_ != nullptr; }

 private:
  TaskScope(detail::TaskScopeState* state, Scheduler* scheduler) noexcept
      : state_(state), scheduler_(scheduler) {}

  Status wait_(bool cancel) noexcept;
  void reset_() noexcept;

  detail::TaskScopeState* state_ = nullptr;
  Scheduler* scheduler_ = nullptr;
};

template <class T>
JoinHandle<T> TaskScope::spawn(Task<T> task, std::size_t stack_size) noexcept {
  using Driver = detail::ScopedTaskDriver<T>;
  using JoinState = detail::JoinState<T>;

  if (state_ == nullptr || scheduler_ == nullptr) {
    return JoinHandle<T>(
        Status::InvalidArgument("TaskScope::spawn requires a valid scope"));
  }
  if (!task.valid()) {
    return JoinHandle<T>(
        Status::ResourceExhausted("scoped task coroutine allocation failed"));
  }
  if (stack_size < mem::kFiberStackSmall) {
    return JoinHandle<T>(Status::InvalidArgument(
        "scoped task fiber stack is smaller than the safe tier"));
  }

  const mem::AllocatorRef allocator = detail::task_scope_allocator(state_);

  auto* join_state = static_cast<JoinState*>(
      allocator.allocate(sizeof(JoinState), alignof(JoinState)));
  if (join_state == nullptr) {
    return JoinHandle<T>(
        Status::ResourceExhausted("scoped task join-state allocation failed"));
  }
  new (join_state) JoinState{};
  join_state->alloc = allocator;

  auto fail_before_admission = [join_state](Status status) noexcept {
    join_state->set_error(status);
    join_state->release();  // no driver owns the task-side reference
    return JoinHandle<T>(join_state);
  };

  auto* driver =
      static_cast<Driver*>(allocator.allocate(sizeof(Driver), alignof(Driver)));
  if (driver == nullptr) {
    return fail_before_admission(
        Status::ResourceExhausted("scoped task driver allocation failed"));
  }

  mem::FiberStackPool* pool = scheduler_->fiber_stack_pool(stack_size);
  auto stack_result =
      pool != nullptr ? pool->acquire() : mem::FiberStack::allocate(stack_size);
  if (!stack_result) {
    allocator.deallocate(driver, sizeof(Driver), alignof(Driver));
    return fail_before_admission(
        std::move(stack_result)
            .error()
            .WithContext("scoped task fiber stack allocation"));
  }

  mem::FiberStack stack = *std::move(stack_result);
  new (driver) Driver(allocator, std::move(task), join_state, state_,
                      std::move(stack), pool);
  Status admitted = detail::task_scope_try_admit(state_, &driver->completion);
  if (!admitted.ok()) {
    join_state->set_error(admitted);
    join_state->release();         // no worker owns the task-side reference
    driver->completion.release();  // record was never scope-owned
    return JoinHandle<T>(join_state);
  }

  driver->arm_finish_hook();
  if (!scheduler_->spawn(&*driver->fiber)) {
    Status status = scheduler_->accepting()
                        ? Status::ResourceExhausted(
                              "scheduler runnable queues are full on a worker")
                        : Status::Cancelled("scheduler is shutting down");
    join_state->set_error(status);
    driver->scope_status = status;
    Driver::finish(driver);
  }
  return JoinHandle<T>(join_state);
}

}  // namespace hyle::sched
