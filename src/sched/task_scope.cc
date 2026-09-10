#include "hyle/sched/task_scope.h"

#include <atomic>
#include <cstddef>
#include <new>
#include <optional>
#include <utility>

#include "fiber_access.h"
#include "hyle/core/compiler.h"
#include "hyle/sched/sync.h"
#include "hyle/util/spin_lock.h"

namespace hyle::sched {
namespace detail {

namespace {
#if defined(HYLE_TESTING)
using TaskScopeFinalPublishHookForTesting = void (*)(void*) noexcept;
std::atomic<TaskScopeFinalPublishHookForTesting> g_final_publish_hook{nullptr};
std::atomic<void*> g_final_publish_context{nullptr};

// TEST-ONLY: invokes the deterministic final-publication race seam.
void run_final_publish_hook_for_testing() noexcept {
  const auto hook = g_final_publish_hook.load(std::memory_order_acquire);
  if (hook != nullptr) {
    hook(g_final_publish_context.load(std::memory_order_relaxed));
  }
}
#endif

}  // namespace

struct TaskScopeState {
  enum class JoinPhase : unsigned char { kOpen, kJoining, kJoined };

  TaskScopeState(mem::AllocatorRef owner_allocator,
                 CancelToken owner_cancel) noexcept
      : allocator(owner_allocator), cancel(std::move(owner_cancel)) {}

  ~TaskScopeState() {
    TaskScopeCompletion* current = head;
    while (current != nullptr) {
      TaskScopeCompletion* next = current->next;
      current->release();  // release the scope-owned reference
      current = next;
    }
  }

  mem::AllocatorRef allocator;
  CancelToken cancel;
  // One short control lock is the admission/close linearization point and the
  // rare error aggregator. Successful child completion never touches it.
  util::SpinLock control_lock;
  TaskScopeCompletion* head = nullptr;
  TaskScopeCompletion* tail = nullptr;
  std::atomic<bool> closed{false};
  bool ever_admitted = false;
  JoinPhase join_phase = JoinPhase::kOpen;
  Status first_error;
};

mem::AllocatorRef task_scope_allocator(TaskScopeState* state) noexcept {
  HYLE_FATAL_ASSERT(state != nullptr);
  return state->allocator;
}

Status task_scope_try_admit(TaskScopeState* state,
                            TaskScopeCompletion* completion) noexcept {
  HYLE_FATAL_ASSERT(state != nullptr);
  HYLE_FATAL_ASSERT(completion != nullptr);
  HYLE_FATAL_ASSERT(completion->next == nullptr);
  HYLE_FATAL_ASSERT(completion->dispose != nullptr);

  state->control_lock.lock();
  if (state->closed.load(std::memory_order_relaxed)) {
    state->control_lock.unlock();
    return Status::FailedPrecondition("task scope is closed");
  }
  // The driver has not been submitted yet, so no publisher can race this
  // initialization of the scope-owned reference.
  completion->retain_scope_before_publish();
  if (state->tail == nullptr) {
    state->head = completion;
  } else {
    state->tail->next = completion;
  }
  state->tail = completion;
  state->ever_admitted = true;
  state->control_lock.unlock();
  return Status{};
}

void task_scope_publish(TaskScopeState* state, TaskScopeCompletion* completion,
                        const Status& status) noexcept {
  HYLE_FATAL_ASSERT(state != nullptr);
  HYLE_FATAL_ASSERT(completion != nullptr);
  if (!status.ok()) {
    state->control_lock.lock();
    state->first_error.Update(status);
    state->control_lock.unlock();
  }

  // Event is per-child, so successful publishers never bounce a scope-wide
  // cache line. The publisher reference remains live across signal() and the
  // test seam; the caller releases it only after this function returns.
  completion->completed.signal();
#if defined(HYLE_TESTING)
  run_final_publish_hook_for_testing();
#endif
}

#if defined(HYLE_TESTING)
// TEST-ONLY: pauses a record publisher after Event visibility but before it
// drops its publisher reference. The declaration stays out of installed API.
HYLE_SCHED_API void set_task_scope_final_publish_hook_for_testing(
    TaskScopeFinalPublishHookForTesting hook, void* context) noexcept {
  if (hook == nullptr) {
    g_final_publish_hook.store(nullptr, std::memory_order_release);
    g_final_publish_context.store(nullptr, std::memory_order_relaxed);
    return;
  }
  g_final_publish_context.store(context, std::memory_order_relaxed);
  g_final_publish_hook.store(hook, std::memory_order_release);
}
#endif

const CancelToken& task_scope_cancel_token(TaskScopeState* state) noexcept {
  HYLE_FATAL_ASSERT(state != nullptr);
  return state->cancel;
}

template <class T>
struct ScopeOwnedTaskDriver {
  static_assert(std::is_void_v<T> || std::is_same_v<T, Status>);

  TaskScopeCompletion completion;
  mem::AllocatorRef allocator;
  std::optional<Task<T>> task;
  TaskScopeState* scope_state;
  std::optional<Fiber> fiber;
  Status scope_status;

  ScopeOwnedTaskDriver(mem::AllocatorRef owner_allocator, Task<T> owner_task,
                       TaskScopeState* owner_scope_state, mem::FiberStack stack,
                       mem::FiberStackPool* pool) noexcept
      : completion(this, &ScopeOwnedTaskDriver::dispose),
        allocator(owner_allocator),
        task(std::move(owner_task)),
        scope_state(owner_scope_state),
        fiber(std::in_place, std::move(stack), &ScopeOwnedTaskDriver::entry,
              this, pool) {}

  void arm_finish_hook() noexcept {
    FiberAccess::set_on_finish(&*fiber, &ScopeOwnedTaskDriver::finish, this);
  }

  static void entry(void* raw) noexcept {
    auto* self = static_cast<ScopeOwnedTaskDriver*>(raw);
    Status scope_status;
    {
      const CancelToken& scope_token =
          task_scope_cancel_token(self->scope_state);
      CancelScope scope_cancel(scope_token);
      if (scope_token.is_cancelled()) {
        scope_status =
            Status::Cancelled("scope-owned task cancelled before start");
      } else {
        self->task->resume();
        if (scope_token.is_cancelled() && !self->task->done()) {
          scope_status =
              Status::Cancelled("scope-owned task cancellation requested");
        } else {
          try {
            if constexpr (std::is_void_v<T>) {
              self->task->result();
            } else {
              scope_status = self->task->result();
            }
          } catch (...) {
            scope_status =
                Status::Internal("scope-owned task threw an exception");
          }
        }
      }
    }
    self->scope_status = std::move(scope_status);
  }

  static void finish(void* raw) noexcept {
    auto* self = static_cast<ScopeOwnedTaskDriver*>(raw);
    self->fiber.reset();
    self->task.reset();
    task_scope_publish(self->scope_state, &self->completion,
                       self->scope_status);
    // May destroy `self` if the terminal owner already released its reference.
    self->completion.release();
  }

  static void dispose(void* raw) noexcept {
    auto* self = static_cast<ScopeOwnedTaskDriver*>(raw);
    mem::AllocatorRef allocator = self->allocator;
    self->~ScopeOwnedTaskDriver();
    allocator.deallocate(self, sizeof(ScopeOwnedTaskDriver),
                         alignof(ScopeOwnedTaskDriver));
  }
};

template <class T>
Status task_scope_add(TaskScopeState* state, Scheduler* scheduler, Task<T> task,
                      std::size_t stack_size) noexcept {
  using Driver = ScopeOwnedTaskDriver<T>;
  if (state == nullptr || scheduler == nullptr) {
    return Status::InvalidArgument("TaskScope::add requires a valid scope");
  }
  if (!task.valid()) {
    return Status::ResourceExhausted(
        "scope-owned task coroutine allocation failed");
  }
  if (stack_size < mem::kFiberStackSmall) {
    return Status::InvalidArgument(
        "scope-owned task fiber stack is smaller than the safe tier");
  }

  const mem::AllocatorRef allocator = task_scope_allocator(state);

  auto* driver =
      static_cast<Driver*>(allocator.allocate(sizeof(Driver), alignof(Driver)));
  if (driver == nullptr) {
    return Status::ResourceExhausted(
        "scope-owned task driver allocation failed");
  }

  mem::FiberStackPool* pool = scheduler->fiber_stack_pool(stack_size);
  auto stack_result =
      pool != nullptr ? pool->acquire() : mem::FiberStack::allocate(stack_size);
  if (!stack_result) {
    allocator.deallocate(driver, sizeof(Driver), alignof(Driver));
    return std::move(stack_result)
        .error()
        .WithContext("scope-owned task fiber stack allocation");
  }

  mem::FiberStack stack = *std::move(stack_result);
  new (driver)
      Driver(allocator, std::move(task), state, std::move(stack), pool);
  Status admitted = task_scope_try_admit(state, &driver->completion);
  if (!admitted.ok()) {
    driver->completion.release();  // record was never scope-owned
    return admitted;
  }

  driver->arm_finish_hook();
  if (!scheduler->spawn(&*driver->fiber)) {
    Status status = scheduler->accepting()
                        ? Status::ResourceExhausted(
                              "scheduler runnable queues are full on a worker")
                        : Status::Cancelled("scheduler is shutting down");
    driver->scope_status = std::move(status);
    Driver::finish(driver);
  }
  return Status{};
}

}  // namespace detail

Result<TaskScope> TaskScope::create(Scheduler& scheduler,
                                    mem::AllocatorRef allocator) noexcept {
  if (!scheduler.valid()) {
    return std::unexpected(
        Status::InvalidArgument("TaskScope requires a valid Scheduler"));
  }
  if (!allocator.valid()) {
    return std::unexpected(
        Status::InvalidArgument("TaskScope requires a complete allocator"));
  }

  auto cancel_result = CancelToken::try_create_root(allocator);
  if (!cancel_result) {
    return std::unexpected(
        std::move(cancel_result).error().WithContext("TaskScope cancellation"));
  }

  void* storage = allocator.allocate(sizeof(detail::TaskScopeState),
                                     alignof(detail::TaskScopeState));
  if (storage == nullptr) {
    return std::unexpected(
        Status::ResourceExhausted("TaskScope state allocation failed"));
  }
  auto* state = new (storage)
      detail::TaskScopeState(allocator, *std::move(cancel_result));
  return TaskScope(state, &scheduler);
}

TaskScope::TaskScope(TaskScope&& other) noexcept
    : state_(std::exchange(other.state_, nullptr)),
      scheduler_(std::exchange(other.scheduler_, nullptr)) {}

TaskScope& TaskScope::operator=(TaskScope&& other) noexcept {
  if (this != &other) {
    reset_();
    state_ = std::exchange(other.state_, nullptr);
    scheduler_ = std::exchange(other.scheduler_, nullptr);
  }
  return *this;
}

TaskScope::~TaskScope() { reset_(); }

void TaskScope::reset_() noexcept {
  if (state_ == nullptr) return;
  state_->control_lock.lock();
  // wait_ reaches kJoined only after it has consumed the stable list and
  // observed every completion. The owner invariant still makes both live and
  // completed-but-unjoined destruction fatal in release builds.
  HYLE_FATAL_ASSERT_MSG(
      !state_->ever_admitted ||
          state_->join_phase == detail::TaskScopeState::JoinPhase::kJoined,
      "TaskScope with admitted children must be joined before destruction");
  state_->control_lock.unlock();

  mem::AllocatorRef allocator = state_->allocator;
  state_->~TaskScopeState();
  allocator.deallocate(state_, sizeof(detail::TaskScopeState),
                       alignof(detail::TaskScopeState));
  state_ = nullptr;
  scheduler_ = nullptr;
}

void TaskScope::close() noexcept {
  if (state_ == nullptr) return;
  state_->control_lock.lock();
  state_->closed.store(true, std::memory_order_release);
  state_->control_lock.unlock();
}

Status TaskScope::join() noexcept { return wait_(false); }

Status TaskScope::cancel_and_join() noexcept { return wait_(true); }

Status TaskScope::add(Task<void> task, std::size_t stack_size) noexcept {
  return detail::task_scope_add(state_, scheduler_, std::move(task),
                                stack_size);
}

Status TaskScope::add(Task<Status> task, std::size_t stack_size) noexcept {
  return detail::task_scope_add(state_, scheduler_, std::move(task),
                                stack_size);
}

Status TaskScope::wait_(bool cancel) noexcept {
  if (state_ == nullptr) {
    return Status::InvalidArgument("TaskScope join requires a valid scope");
  }

  // close and terminal-owner claim share the admission linearization point.
  // Detaching the stable list lets this one owner release each completion
  // record as soon as it has observed it instead of retaining all records until
  // TaskScope destruction.
  state_->control_lock.lock();
  state_->closed.store(true, std::memory_order_release);
  if (state_->join_phase != detail::TaskScopeState::JoinPhase::kOpen) {
    state_->control_lock.unlock();
    return Status::FailedPrecondition(
        "TaskScope terminal join may be called only once");
  }
  state_->join_phase = detail::TaskScopeState::JoinPhase::kJoining;
  detail::TaskScopeCompletion* current = state_->head;
  state_->head = nullptr;
  state_->tail = nullptr;
  state_->control_lock.unlock();

  if (cancel) state_->cancel.cancel();

  while (current != nullptr) {
    detail::TaskScopeCompletion* next = current->next;
    if (!current->publisher_released()) current->completed.wait();
    current->release();  // release the scope-owned reference
    current = next;
  }

  state_->control_lock.lock();
  Status result = state_->first_error;
  state_->join_phase = detail::TaskScopeState::JoinPhase::kJoined;
  state_->control_lock.unlock();
  return result;
}

void TaskScope::request_cancel() noexcept {
  if (state_ != nullptr) state_->cancel.cancel();
}

CancelToken TaskScope::cancellation_token() const noexcept {
  return state_ != nullptr ? state_->cancel : CancelToken{};
}

std::size_t TaskScope::remaining() const noexcept {
  if (state_ == nullptr) return 0;
  std::size_t count = 0;
  state_->control_lock.lock();
  for (detail::TaskScopeCompletion* current = state_->head; current != nullptr;
       current = current->next) {
    if (!current->completed.is_signaled()) ++count;
  }
  state_->control_lock.unlock();
  return count;
}

bool TaskScope::is_closed() const noexcept {
  return state_ != nullptr && state_->closed.load(std::memory_order_acquire);
}

bool TaskScope::is_cancel_requested() const noexcept {
  return state_ != nullptr && state_->cancel.is_cancelled();
}

}  // namespace hyle::sched
