#pragma once

#include <atomic>
#include <cstddef>
#include <new>
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

// Compiled state operations keep synchronization and cancellation-tree layout
// out of the installed template header. Only scoped tasks call these seams;
// ordinary spawn_task/schedule paths do not carry a scope branch or field.
HYLE_SCHED_API Status task_scope_try_admit(
    TaskScopeState* state, mem::AllocatorRef* allocator) noexcept;
HYLE_SCHED_API void task_scope_complete(TaskScopeState* state,
                                        const Status& status) noexcept;
HYLE_SCHED_API const CancelToken& task_scope_cancel_token(
    TaskScopeState* state) noexcept;

template <class T>
struct ScopedTaskDriver {
  mem::AllocatorRef allocator;
  Task<T> task;
  JoinState<T>* join_state;
  TaskScopeState* scope_state;
  Fiber fiber;

  void arm_finish_hook() noexcept {
    fiber.set_on_finish(&ScopedTaskDriver::destroy, this);
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
        self->task.resume();
        if ((join_state->cancel_req.load(std::memory_order_acquire) ||
             scope_token.is_cancelled()) &&
            !self->task.done()) {
          scope_status =
              Status::Cancelled("scoped task cancellation requested");
          join_state->set_error(scope_status);
        } else {
          try {
            if constexpr (std::is_void_v<T>) {
              self->task.result();
              join_state->set_done();
            } else {
              T value = self->task.result();
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

    // JoinState is terminal before active-count publication: once scope.join()
    // returns, every retained JoinHandle is immediately joinable.
    task_scope_complete(self->scope_state, scope_status);
  }

  static void destroy(void* raw) noexcept {
    auto* self = static_cast<ScopedTaskDriver*>(raw);
    JoinState<T>* join_state = self->join_state;
    mem::AllocatorRef allocator = self->allocator;
    self->~ScopedTaskDriver();
    allocator.deallocate(self, sizeof(ScopedTaskDriver),
                         alignof(ScopedTaskDriver));
    join_state->release();
  }
};

}  // namespace detail

// Owns a runtime-sized set of Tasks submitted to one Scheduler. Admission is
// thread-safe and linearized with close(); join() closes admission and waits
// for every admitted child, whether its JoinHandle was retained or detached.
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
  // means ownership was not transferred. Once admitted, later infrastructure
  // and child failures are retained by join(), and add() returns OK.
  [[nodiscard]] Status add(
      Task<void> task,
      std::size_t stack_size = kDefaultStackSize) noexcept;
  [[nodiscard]] Status add(
      Task<Status> task,
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

  mem::AllocatorRef allocator;
  Status admitted = detail::task_scope_try_admit(state_, &allocator);
  if (!admitted.ok()) return JoinHandle<T>(std::move(admitted));

  auto fail_immediate = [this](Status status) noexcept {
    detail::task_scope_complete(state_, status);
    return JoinHandle<T>(std::move(status));
  };

  auto* join_state = static_cast<JoinState*>(
      allocator.allocate(sizeof(JoinState), alignof(JoinState)));
  if (join_state == nullptr) {
    return fail_immediate(
        Status::ResourceExhausted("scoped task join-state allocation failed"));
  }
  new (join_state) JoinState{};
  join_state->alloc = allocator;

  auto fail_with_state = [this, join_state](Status status) noexcept {
    join_state->set_error(status);
    detail::task_scope_complete(state_, status);
    join_state->release();  // no driver owns the task-side reference
    return JoinHandle<T>(join_state);
  };

  auto* driver =
      static_cast<Driver*>(allocator.allocate(sizeof(Driver), alignof(Driver)));
  if (driver == nullptr) {
    return fail_with_state(
        Status::ResourceExhausted("scoped task driver allocation failed"));
  }

  mem::FiberStackPool* pool = scheduler_->fiber_stack_pool(stack_size);
  auto stack_result =
      pool != nullptr ? pool->acquire() : mem::FiberStack::allocate(stack_size);
  if (!stack_result) {
    allocator.deallocate(driver, sizeof(Driver), alignof(Driver));
    return fail_with_state(
        std::move(stack_result)
            .error()
            .WithContext("scoped task fiber stack allocation"));
  }

  mem::FiberStack stack = *std::move(stack_result);
  new (driver) Driver{allocator, std::move(task), join_state, state_,
                      Fiber(std::move(stack), &Driver::entry, driver, pool)};
  driver->arm_finish_hook();
  if (!scheduler_->spawn(&driver->fiber)) {
    Status status = scheduler_->accepting()
                        ? Status::ResourceExhausted(
                              "scheduler runnable queues are full on a worker")
                        : Status::Cancelled("scheduler is shutting down");
    join_state->set_error(status);
    detail::task_scope_complete(state_, status);
    Driver::destroy(driver);
  }
  return JoinHandle<T>(join_state);
}

}  // namespace hyle::sched
