#include "hyle/sched/task_scope.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>

#include "hyle/core/cache.h"
#include "hyle/core/compiler.h"
#include "fiber_access.h"
#include "hyle/sched/sync.h"
#include "hyle/util/spin_lock.h"

namespace hyle::sched {
namespace detail {

namespace {

constexpr std::uint64_t kClosed = 1;
constexpr std::uint64_t kEverAdmitted = 2;
constexpr std::uint64_t kJoined = 4;
constexpr unsigned kCountShift = 3;
constexpr std::uint64_t kCountOne = std::uint64_t{1} << kCountShift;
constexpr std::uint64_t kMaxCount =
    (std::numeric_limits<std::uint64_t>::max)() >> kCountShift;

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
  TaskScopeState(mem::AllocatorRef owner_allocator,
                 CancelToken owner_cancel) noexcept
      : allocator(owner_allocator), cancel(std::move(owner_cancel)) {}

  mem::AllocatorRef allocator;
  CancelToken cancel;
  std::atomic<std::uint64_t> lifecycle{0};
  // Admission/close is producer-written while completion is worker-written.
  // Keeping their counters on different cache lines avoids making every short
  // child bounce the producer's CAS line back across the worker set.
  alignas(core::kDestructiveInterferenceSize)
      std::atomic<std::uint64_t> completed{0};
  util::SpinLock wait_lock;
  ConditionVariable drained;
  Status first_error;
};

Status task_scope_try_admit(TaskScopeState* state,
                            mem::AllocatorRef* allocator) noexcept {
  HYLE_FATAL_ASSERT(state != nullptr);
  HYLE_FATAL_ASSERT(allocator != nullptr);
  std::uint64_t current = state->lifecycle.load(std::memory_order_acquire);
  for (;;) {
    if ((current & kClosed) != 0) {
      return Status::FailedPrecondition("task scope is closed");
    }
    if ((current >> kCountShift) == kMaxCount) {
      return Status::ResourceExhausted("task scope child count overflow");
    }
    const std::uint64_t next = (current | kEverAdmitted) + kCountOne;
    if (state->lifecycle.compare_exchange_weak(current, next,
                                               std::memory_order_acq_rel,
                                               std::memory_order_acquire)) {
      *allocator = state->allocator;
      return Status{};
    }
  }
}

void task_scope_complete(TaskScopeState* state, const Status& status) noexcept {
  HYLE_FATAL_ASSERT(state != nullptr);
  if (!status.ok()) {
    state->wait_lock.lock();
    state->first_error.Update(status);
    const std::uint64_t completed =
        state->completed.fetch_add(1, std::memory_order_acq_rel) + 1;
    const std::uint64_t lifecycle =
        state->lifecycle.load(std::memory_order_acquire);
    HYLE_FATAL_ASSERT(completed <= (lifecycle >> kCountShift));
    const bool last = (lifecycle & kClosed) != 0 &&
                      completed == (lifecycle >> kCountShift);
    if (last) state->drained.notify_all();
    // Publishing completion while holding wait_lock prevents a joiner from
    // freeing state until this function has made its final state access.
    state->wait_lock.unlock();
    return;
  }

  std::uint64_t completed = state->completed.load(std::memory_order_relaxed);
  for (;;) {
    const std::uint64_t lifecycle =
        state->lifecycle.load(std::memory_order_acquire);
    const std::uint64_t admitted = lifecycle >> kCountShift;
    HYLE_FATAL_ASSERT(completed < admitted);

    if (completed + 1 == admitted) {
      // A currently-final publisher must retain wait_lock from before the
      // predicate becomes true through its final access to state. This both
      // prevents a joiner from freeing state after publication and closes the
      // lost-wakeup race where close occurs after the lifecycle load above.
      state->wait_lock.lock();
      completed = state->completed.load(std::memory_order_relaxed);
      const std::uint64_t locked_lifecycle =
          state->lifecycle.load(std::memory_order_acquire);
      const std::uint64_t locked_admitted = locked_lifecycle >> kCountShift;
      if (completed + 1 != locked_admitted) {
        state->wait_lock.unlock();
        continue;
      }
      state->completed.store(completed + 1, std::memory_order_release);
#if defined(HYLE_TESTING)
      // Pause after publication so the regression can prove that a joiner
      // cannot free state before the publisher's remaining notify/unlock.
      run_final_publish_hook_for_testing();
#endif
      if ((locked_lifecycle & kClosed) != 0) {
        state->drained.notify_all();
      }
      state->wait_lock.unlock();
      return;
    }

    if (state->completed.compare_exchange_weak(completed, completed + 1,
                                               std::memory_order_release,
                                               std::memory_order_relaxed)) {
      // The release publication is deliberately the last state access. If
      // close races immediately afterwards, join may now destroy state safely.
      return;
    }
  }
}

#if defined(HYLE_TESTING)
// TEST-ONLY: installs a deterministic pause in the final completion publisher.
// This declaration is kept in the test TU rather than the installed header.
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

  mem::AllocatorRef allocator;
  Task<T> task;
  TaskScopeState* scope_state;
  Fiber fiber;

  void arm_finish_hook() noexcept {
    FiberAccess::set_on_finish(&fiber, &ScopeOwnedTaskDriver::destroy, this);
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
        self->task.resume();
        if (scope_token.is_cancelled() && !self->task.done()) {
          scope_status =
              Status::Cancelled("scope-owned task cancellation requested");
        } else {
          try {
            if constexpr (std::is_void_v<T>) {
              self->task.result();
            } else {
              scope_status = self->task.result();
            }
          } catch (...) {
            scope_status =
                Status::Internal("scope-owned task threw an exception");
          }
        }
      }
    }
    task_scope_complete(self->scope_state, scope_status);
  }

  static void destroy(void* raw) noexcept {
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

  mem::AllocatorRef allocator;
  Status admitted = task_scope_try_admit(state, &allocator);
  if (!admitted.ok()) return admitted;

  auto complete_after_admission = [state](Status status) noexcept {
    task_scope_complete(state, status);
    // The scope owns every post-admission outcome. Returning OK makes it safe
    // for callers to propagate only true pre-admission rejection and still
    // reach the mandatory join/drain point.
    return Status{};
  };

  auto* driver = static_cast<Driver*>(
      allocator.allocate(sizeof(Driver), alignof(Driver)));
  if (driver == nullptr) {
    return complete_after_admission(Status::ResourceExhausted(
        "scope-owned task driver allocation failed"));
  }

  mem::FiberStackPool* pool = scheduler->fiber_stack_pool(stack_size);
  auto stack_result =
      pool != nullptr ? pool->acquire() : mem::FiberStack::allocate(stack_size);
  if (!stack_result) {
    allocator.deallocate(driver, sizeof(Driver), alignof(Driver));
    return complete_after_admission(
        std::move(stack_result)
            .error()
            .WithContext("scope-owned task fiber stack allocation"));
  }

  mem::FiberStack stack = *std::move(stack_result);
  new (driver) Driver{allocator, std::move(task), state,
                      Fiber(std::move(stack), &Driver::entry, driver, pool)};
  driver->arm_finish_hook();
  if (!scheduler->spawn(&driver->fiber)) {
    Status status = scheduler->accepting()
                        ? Status::ResourceExhausted(
                              "scheduler runnable queues are full on a worker")
                        : Status::Cancelled("scheduler is shutting down");
    task_scope_complete(state, status);
    Driver::destroy(driver);
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
  const std::uint64_t lifecycle =
      state_->lifecycle.load(std::memory_order_acquire);
  const std::uint64_t completed =
      state_->completed.load(std::memory_order_acquire);
  HYLE_FATAL_ASSERT_MSG((lifecycle >> detail::kCountShift) == completed,
                        "TaskScope destroyed with live children");
  HYLE_FATAL_ASSERT_MSG(
      (lifecycle & detail::kEverAdmitted) == 0 ||
          (lifecycle & detail::kJoined) != 0,
      "TaskScope with admitted children must be joined before destruction");

  mem::AllocatorRef allocator = state_->allocator;
  state_->~TaskScopeState();
  allocator.deallocate(state_, sizeof(detail::TaskScopeState),
                       alignof(detail::TaskScopeState));
  state_ = nullptr;
  scheduler_ = nullptr;
}

void TaskScope::close() noexcept {
  if (state_ == nullptr) return;
  state_->lifecycle.fetch_or(detail::kClosed, std::memory_order_acq_rel);
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
  close();
  if (cancel) state_->cancel.cancel();

  state_->wait_lock.lock();
  while (state_->completed.load(std::memory_order_acquire) !=
         (state_->lifecycle.load(std::memory_order_acquire) >>
          detail::kCountShift)) {
    state_->drained.wait(state_->wait_lock);
  }
  Status result = state_->first_error;
  state_->lifecycle.fetch_or(detail::kJoined, std::memory_order_release);
  state_->wait_lock.unlock();
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
  const std::uint64_t completed =
      state_->completed.load(std::memory_order_acquire);
  const std::uint64_t admitted =
      state_->lifecycle.load(std::memory_order_acquire) >> detail::kCountShift;
  HYLE_FATAL_ASSERT(completed <= admitted);
  return static_cast<std::size_t>(admitted - completed);
}

bool TaskScope::is_closed() const noexcept {
  return state_ != nullptr &&
         (state_->lifecycle.load(std::memory_order_acquire) &
          detail::kClosed) != 0;
}

bool TaskScope::is_cancel_requested() const noexcept {
  return state_ != nullptr && state_->cancel.is_cancelled();
}

}  // namespace hyle::sched
