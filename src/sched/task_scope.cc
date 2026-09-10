#include "hyle/sched/task_scope.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>

#include "hyle/core/compiler.h"
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

}  // namespace

struct TaskScopeState {
  TaskScopeState(mem::AllocatorRef owner_allocator,
                 CancelToken owner_cancel) noexcept
      : allocator(owner_allocator), cancel(std::move(owner_cancel)) {}

  mem::AllocatorRef allocator;
  CancelToken cancel;
  std::atomic<std::uint64_t> lifecycle{0};
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
    const std::uint64_t previous =
        state->lifecycle.fetch_sub(kCountOne, std::memory_order_acq_rel);
    HYLE_FATAL_ASSERT((previous >> kCountShift) != 0);
    const bool last = (previous >> kCountShift) == 1;
    state->wait_lock.unlock();
    if (last) state->drained.notify_all();
    return;
  }

  const std::uint64_t previous =
      state->lifecycle.fetch_sub(kCountOne, std::memory_order_acq_rel);
  HYLE_FATAL_ASSERT((previous >> kCountShift) != 0);
  if ((previous >> kCountShift) == 1) {
    // Pair with wait_() checking the predicate and publishing its waiter while
    // holding wait_lock. Only the final successful completion pays this lock.
    state->wait_lock.lock();
    state->wait_lock.unlock();
    state->drained.notify_all();
  }
}

const CancelToken& task_scope_cancel_token(TaskScopeState* state) noexcept {
  HYLE_FATAL_ASSERT(state != nullptr);
  return state->cancel;
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
  HYLE_FATAL_ASSERT_MSG((lifecycle >> detail::kCountShift) == 0,
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

Status TaskScope::wait_(bool cancel) noexcept {
  if (state_ == nullptr) {
    return Status::InvalidArgument("TaskScope join requires a valid scope");
  }
  close();
  if (cancel) state_->cancel.cancel();

  state_->wait_lock.lock();
  while ((state_->lifecycle.load(std::memory_order_acquire) >>
          detail::kCountShift) != 0) {
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

std::size_t TaskScope::remaining() const noexcept {
  return state_ == nullptr
             ? 0
             : static_cast<std::size_t>(
                   state_->lifecycle.load(std::memory_order_acquire) >>
                   detail::kCountShift);
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
