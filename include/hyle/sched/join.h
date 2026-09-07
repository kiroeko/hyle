#pragma once

#include <atomic>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <new>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>

#include "hyle/error/result.h"
#include "hyle/error/status.h"
#include "hyle/core/compiler.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/fiber_stack.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/sync.h"

// JoinHandle + when_all/when_any form the structured-concurrency layer.
// Reference implementations from Tokio, Folly, Seastar and Marl informed the
// following ownership decisions:
//
//  * JoinHandle does NOT own the Fiber -- only a refcounted result slot
//    (JoinState). marl proved a join can be as thin as WaitGroup(1)+slot; tokio's
//    6-bit state machine is complexity born of reusing the task cell for both the
//    future and the output -- HYLE's independent slot sidesteps it.
//  * Completion notification reuses ConditionVariable (dual-mode: a
//    joining fiber suspend()s, a bare thread parks) -- already TSan-proven. No
//    hand-rolled rendezvous state machine.
//  * Cancellation is COOPERATIVE. A stackful fiber cannot be preempted (folly
//    /fibers offers no external cancel, marl none) -- request_cancel() sets a
//    flag and the task checks is_cancel_requested() at its own suspend points
//    and unwinds via RAII. when_any's loser cancellation is the same mechanism.
//    In-flight OS operation cancellation remains the completion source owner's
//    job.
//  * schedule_task runs a non-suspending callable directly on a worker stack and
//    keeps its callable + JoinState in one allocation. spawn_task owns a Fiber
//    via a heap TaskDriver for work that really can suspend. In both routes,
//    dropping the JoinHandle detaches without abandoning scheduler ownership.
//  * when_all/when_any share ONE GroupSignal (a target-count latch each task
//    signals on completion). when_all targets N; when_any targets 1 + records the
//    winner. The critical UAF guard (folly WhenN-inl.h:59): the waiter UNINSTALLS
//    the group from every state before the group (a stack object) dies, so a
//    late-finishing loser never touches freed memory.

namespace hyle::sched {

namespace detail {

// Shared completion signal for when_all/when_any. Lives on the waiting fiber's
// stack. Each participating JoinState signals it on completion; the waiter blocks
// until `remaining` hits zero (all, or 1 for when_any). `winner` is the index of
// the first task to complete.
struct GroupSignal {
  std::atomic<long> remaining{0};
  std::atomic<int> winner{-1};
  Event completed{EventResetMode::kManual};

  // Called by a JoinState (under ITS lock) when its task completes while this
  // group is installed. Records the winner and fires when the target is met.
  void on_complete(int idx) noexcept {
    int no_winner = -1;
    winner.compare_exchange_strong(no_winner, idx,
                                   std::memory_order_acq_rel,
                                   std::memory_order_acquire);
    long pending = remaining.load(std::memory_order_acquire);
    while (pending > 0 &&
           !remaining.compare_exchange_weak(pending, pending - 1,
                                            std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
    }
    if (pending == 1) completed.signal();
  }
};

// The refcounted result slot. Shared by the running task (writes result +
// signals done) and the JoinHandle (reads it). Refcount starts at 2; last drop
// frees it. Completion is signalled through a ConditionVariable so a joiner
// (fiber or thread) blocks correctly regardless of join-vs-complete ordering.
template <class T>
struct JoinState {
  using DisposeFn = void (*)(JoinState*) noexcept;

  mem::AllocatorRef alloc;
  DisposeFn dispose_fn = &JoinState::dispose_default;
  std::atomic<int> refcount{2};
  util::SpinLock mu;              // guards done/status/storage/group
  ConditionVariable cv;          // a direct joiner waits here
  bool done = false;
  bool value_present = false;
  bool direct_waiter = false;
  std::atomic<bool> cancel_req{false};
  Status status;                 // OK on success, error/Cancelled otherwise
  GroupSignal* group = nullptr;  // set by when_all/when_any while grouped
  int group_idx = -1;
  alignas(T) unsigned char storage[sizeof(T)];  // result, live iff value_present

  void complete_(Status s, T* val) noexcept {
    mu.lock();
    status = std::move(s);
    if (status.ok() && val != nullptr) {
      new (storage) T(std::move(*val));
      value_present = true;
    }
    done = true;
    // Signal the group (if any) WHILE STILL HOLDING mu. Doing this after unlock
    // was a UAF: a loser could read `group`, unlock, then be delayed; meanwhile the
    // winner fires the group, the waiter wakes, calls clear_group() on every state
    // and returns -- destroying the stack GroupSignal -- and only then does the
    // delayed loser call g->on_complete() on freed memory. clear_group() takes THIS
    // state's mu, so holding mu across on_complete() forces the waiter's
    // clear_group to block until we are done touching `g`, keeping the group
    // alive. Group completion never acquires a JoinState lock, so there is no
    // reverse lock edge.
    if (group != nullptr) group->on_complete(group_idx);
    const bool notify_direct = direct_waiter;
    mu.unlock();
    if (notify_direct) cv.notify_all();
  }
  void set_value(T&& v) noexcept { complete_(Status(), &v); }
  void set_error(Status s) noexcept { complete_(std::move(s), nullptr); }

  Result<T> take() noexcept {
    mu.lock();
    if (!done) {
      direct_waiter = true;
      cv.wait(mu, [&] { return done; });
    }
    Status s = status;
    if (!s.ok()) {
      mu.unlock();
      return std::unexpected(std::move(s));
    }
    HYLE_FATAL_ASSERT(value_present);
    T* p = reinterpret_cast<T*>(storage);
    T out(std::move(*p));
    p->~T();
    value_present = false;
    mu.unlock();
    return out;
  }
  bool is_done() noexcept {
    mu.lock();
    bool d = done;
    mu.unlock();
    return d;
  }
  // Install `g` (index i) as this state's group. If already done, returns true so
  // the caller can pre-account the completion instead (avoids missing it).
  bool install_group(GroupSignal* g, int i) noexcept {
    mu.lock();
    bool already = done;
    if (!already) {
      group = g;
      group_idx = i;
    }
    mu.unlock();
    return already;
  }
  // Remove the group (called by the waiter before the group dies). Idempotent.
  void clear_group() noexcept {
    mu.lock();
    group = nullptr;
    mu.unlock();
  }

  void add_ref() noexcept { refcount.fetch_add(1, std::memory_order_relaxed); }
  void release() noexcept {
    if (refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      dispose_fn(this);
    }
  }

  static void dispose_default(JoinState* state) noexcept {
    if (state->value_present)
      reinterpret_cast<T*>(state->storage)->~T();  // detached value
    mem::AllocatorRef allocator = state->alloc;
    state->~JoinState();
    allocator.deallocate(state, sizeof(JoinState), alignof(JoinState));
  }
};

// void specialization: no result storage.
template <>
struct JoinState<void> {
  using DisposeFn = void (*)(JoinState*) noexcept;

  mem::AllocatorRef alloc;
  DisposeFn dispose_fn = &JoinState::dispose_default;
  std::atomic<int> refcount{2};
  util::SpinLock mu;
  ConditionVariable cv;
  bool done = false;
  bool direct_waiter = false;
  std::atomic<bool> cancel_req{false};
  Status status;
  GroupSignal* group = nullptr;
  int group_idx = -1;

  void complete_(Status s) noexcept {
    mu.lock();
    status = std::move(s);
    done = true;
    // Signal the group while holding mu -- see the T-specialization's complete_
    // for the full loser-vs-dead-GroupSignal UAF rationale.
    if (group != nullptr) group->on_complete(group_idx);
    const bool notify_direct = direct_waiter;
    mu.unlock();
    if (notify_direct) cv.notify_all();
  }
  void set_done() noexcept { complete_(Status()); }
  void set_error(Status s) noexcept { complete_(std::move(s)); }

  Status take() noexcept {
    mu.lock();
    if (!done) {
      direct_waiter = true;
      cv.wait(mu, [&] { return done; });
    }
    Status s = status;
    mu.unlock();
    return s;
  }
  bool is_done() noexcept {
    mu.lock();
    bool d = done;
    mu.unlock();
    return d;
  }
  bool install_group(GroupSignal* g, int i) noexcept {
    mu.lock();
    bool already = done;
    if (!already) {
      group = g;
      group_idx = i;
    }
    mu.unlock();
    return already;
  }
  void clear_group() noexcept {
    mu.lock();
    group = nullptr;
    mu.unlock();
  }
  void add_ref() noexcept { refcount.fetch_add(1, std::memory_order_relaxed); }
  void release() noexcept {
    if (refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      dispose_fn(this);
    }
  }

  static void dispose_default(JoinState* state) noexcept {
    mem::AllocatorRef allocator = state->alloc;
    state->~JoinState();
    allocator.deallocate(state, sizeof(JoinState), alignof(JoinState));
  }
};

}  // namespace detail

// A running task queries this at its own checkpoints; if true it should unwind
// (RAII) and return early. Cooperative -- a stackful fiber is never force-stopped.
HYLE_SCHED_API bool is_cancel_requested() noexcept;

// ---- JoinHandle<T> ----------------------------------------------------------
// Move-only handle to a spawned task's result. join() blocks (fiber-aware from a
// worker) until the task finishes, returning Result<T>. request_cancel() asks
// cooperative cancellation. Dropping detaches (task runs on, result discarded).
template <class T>
class JoinHandle {
 public:
  JoinHandle() noexcept = default;
  explicit JoinHandle(detail::JoinState<T>* s) noexcept : state_(s) {}
  explicit JoinHandle(Status immediate) noexcept
      : immediate_(std::move(immediate)) {}
  JoinHandle(JoinHandle&& o) noexcept
      : state_(std::exchange(o.state_, nullptr)),
        immediate_(std::move(o.immediate_)) {}
  JoinHandle& operator=(JoinHandle&& o) noexcept {
    if (this != &o) {
      if (state_) state_->release();
      state_ = std::exchange(o.state_, nullptr);
      immediate_ = std::move(o.immediate_);
    }
    return *this;
  }
  JoinHandle(const JoinHandle&) = delete;
  JoinHandle& operator=(const JoinHandle&) = delete;
  ~JoinHandle() {
    if (state_) state_->release();  // detach: task runs on, result discarded
  }

  Result<T> join() noexcept {
    if (state_ == nullptr) {
      HYLE_FATAL_ASSERT(!immediate_.ok());
      Status error = std::move(immediate_);
      immediate_ = Status{};
      return std::unexpected(std::move(error));
    }
    Result<T> result = state_->take();
    state_->release();
    state_ = nullptr;
    return result;
  }

  void request_cancel() noexcept {
    if (state_) state_->cancel_req.store(true, std::memory_order_release);
  }
  bool is_finished() const noexcept {
    return state_ != nullptr ? state_->is_done() : !immediate_.ok();
  }
  void detach() noexcept {
    if (state_) {
      state_->release();
      state_ = nullptr;
    }
    immediate_ = Status{};
  }
  bool valid() const noexcept { return state_ != nullptr || !immediate_.ok(); }

  // Internal: when_all/when_any reach the shared state without consuming.
  detail::JoinState<T>* state_for_group_() const noexcept {
    // A null state is meaningful only for an immediate construction failure.
    // Default-constructed, detached, or already-joined handles are programmer
    // errors and must not be silently counted as completed by when_all/when_any.
    HYLE_FATAL_ASSERT(state_ != nullptr || !immediate_.ok());
    return state_;
  }

 private:
  detail::JoinState<T>* state_ = nullptr;
  Status immediate_;
};

template <>
class JoinHandle<void> {
 public:
  JoinHandle() noexcept = default;
  explicit JoinHandle(detail::JoinState<void>* s) noexcept : state_(s) {}
  explicit JoinHandle(Status immediate) noexcept
      : immediate_(std::move(immediate)) {}
  JoinHandle(JoinHandle&& o) noexcept
      : state_(std::exchange(o.state_, nullptr)),
        immediate_(std::move(o.immediate_)) {}
  JoinHandle& operator=(JoinHandle&& o) noexcept {
    if (this != &o) {
      if (state_) state_->release();
      state_ = std::exchange(o.state_, nullptr);
      immediate_ = std::move(o.immediate_);
    }
    return *this;
  }
  JoinHandle(const JoinHandle&) = delete;
  JoinHandle& operator=(const JoinHandle&) = delete;
  ~JoinHandle() {
    if (state_) state_->release();
  }

  Result<void> join() noexcept {
    if (state_ == nullptr) {
      HYLE_FATAL_ASSERT(!immediate_.ok());
      Status error = std::move(immediate_);
      immediate_ = Status{};
      return std::unexpected(std::move(error));
    }
    Status s = state_->take();
    state_->release();
    state_ = nullptr;
    if (!s.ok()) return std::unexpected(std::move(s));
    return {};
  }
  void request_cancel() noexcept {
    if (state_) state_->cancel_req.store(true, std::memory_order_release);
  }
  bool is_finished() const noexcept {
    return state_ != nullptr ? state_->is_done() : !immediate_.ok();
  }
  void detach() noexcept {
    if (state_) {
      state_->release();
      state_ = nullptr;
    }
    immediate_ = Status{};
  }
  bool valid() const noexcept { return state_ != nullptr || !immediate_.ok(); }
  detail::JoinState<void>* state_for_group_() const noexcept {
    HYLE_FATAL_ASSERT(state_ != nullptr || !immediate_.ok());
    return state_;
  }

 private:
  detail::JoinState<void>* state_ = nullptr;
  Status immediate_;
};

// ---- schedule_task: typed worker-stack task --------------------------------
namespace detail {

template <class Fn>
using ScheduledCallable = std::decay_t<Fn>;

template <class Fn>
using ScheduledResult =
    std::remove_cv_t<std::invoke_result_t<ScheduledCallable<Fn>&>>;

template <class Fn>
concept WorkerStackTask =
    requires(ScheduledCallable<Fn>& callable) {
      { std::invoke(callable) } noexcept;
    } &&
    std::is_nothrow_constructible_v<ScheduledCallable<Fn>, Fn&&> &&
    std::is_nothrow_destructible_v<ScheduledCallable<Fn>> &&
    !std::is_reference_v<std::invoke_result_t<ScheduledCallable<Fn>&>> &&
    (std::is_void_v<ScheduledResult<Fn>> ||
     (std::is_nothrow_move_constructible_v<ScheduledResult<Fn>> &&
      std::is_nothrow_destructible_v<ScheduledResult<Fn>>));

// One allocation owns both the public completion state and the callable. The
// result may outlive callable execution, so the state supplies a typed disposer
// instead of forcing a second driver allocation or retaining captures until the
// JoinHandle is dropped.
template <class F, class R>
struct ScheduledTaskCell final : JoinState<R> {
  template <class Fn>
  ScheduledTaskCell(mem::AllocatorRef allocator, Fn&& callable) noexcept
      : fn(std::in_place, std::forward<Fn>(callable)) {
    this->alloc = allocator;
    this->dispose_fn = &ScheduledTaskCell::destroy;
  }

  static void run(void* raw) noexcept {
    auto* self = static_cast<ScheduledTaskCell*>(raw);
    JoinState<R>* state = self;
    if (state->cancel_req.load(std::memory_order_acquire)) {
      self->fn.reset();
      state->set_error(
          Status::Cancelled("scheduled task cancelled before start"));
    } else {
      ScopedCancelFlag cancel_scope(&state->cancel_req);
      R value = std::invoke(*self->fn);
      self->fn.reset();
      state->set_value(std::move(value));
    }
    state->release();
  }

  static void destroy(JoinState<R>* state) noexcept {
    auto* self = static_cast<ScheduledTaskCell*>(state);
    mem::AllocatorRef allocator = self->alloc;
    self->~ScheduledTaskCell();
    allocator.deallocate(self, sizeof(ScheduledTaskCell),
                         alignof(ScheduledTaskCell));
  }

  std::optional<F> fn;
};

template <class F>
struct ScheduledTaskCell<F, void> final : JoinState<void> {
  template <class Fn>
  ScheduledTaskCell(mem::AllocatorRef allocator, Fn&& callable) noexcept
      : fn(std::in_place, std::forward<Fn>(callable)) {
    this->alloc = allocator;
    this->dispose_fn = &ScheduledTaskCell::destroy;
  }

  static void run(void* raw) noexcept {
    auto* self = static_cast<ScheduledTaskCell*>(raw);
    JoinState<void>* state = self;
    if (state->cancel_req.load(std::memory_order_acquire)) {
      self->fn.reset();
      state->set_error(
          Status::Cancelled("scheduled task cancelled before start"));
    } else {
      ScopedCancelFlag cancel_scope(&state->cancel_req);
      std::invoke(*self->fn);
      self->fn.reset();
      state->set_done();
    }
    state->release();
  }

  static void destroy(JoinState<void>* state) noexcept {
    auto* self = static_cast<ScheduledTaskCell*>(state);
    mem::AllocatorRef allocator = self->alloc;
    self->~ScheduledTaskCell();
    allocator.deallocate(self, sizeof(ScheduledTaskCell),
                         alignof(ScheduledTaskCell));
  }

  std::optional<F> fn;
};

}  // namespace detail

// Schedule a short, non-suspending callable directly on a worker stack and
// return a typed JoinHandle. The callable, completion state and cooperative
// cancellation flag use one allocator-backed cell; no coroutine frame, guarded
// stack or context switch is created. The callable MUST be noexcept and MUST NOT
// yield, suspend, sleep or perform a blocking fiber wait. Use spawn_task for a
// coroutine that may suspend, or spawn_blocking for blocking work.
template <class Fn>
  requires detail::WorkerStackTask<Fn>
JoinHandle<detail::ScheduledResult<Fn>> schedule_task(
    Scheduler& scheduler, mem::AllocatorRef allocator, Fn&& fn) noexcept {
  using Callable = detail::ScheduledCallable<Fn>;
  using R = detail::ScheduledResult<Fn>;
  using Cell = detail::ScheduledTaskCell<Callable, R>;

  if (!allocator.valid()) {
    return JoinHandle<R>(
        Status::InvalidArgument("schedule_task requires a complete allocator"));
  }
  auto* cell = static_cast<Cell*>(
      allocator.allocate(sizeof(Cell), alignof(Cell)));
  if (cell == nullptr) {
    return JoinHandle<R>(
        Status::ResourceExhausted("scheduled task allocation failed"));
  }
  new (cell) Cell(allocator, std::forward<Fn>(fn));

  Status admitted = scheduler.schedule(&Cell::run, cell);
  if (!admitted.ok()) {
    cell->fn.reset();
    cell->set_error(std::move(admitted).WithContext("schedule_task"));
    cell->release();  // no worker owns the reserved task-side reference
  }
  return JoinHandle<R>(cell);
}

// ---- spawn_task: spawn a Task<T>, get a JoinHandle<T> -----------------------
namespace detail {

// A driver owns the Task + the Fiber that runs it. It must NOT free itself from
// inside entry (the fcontext trampoline reads the Fiber after entry returns -- a
// self-free there is a UAF, caught by TSan as a SEGV in the trampoline). Instead
// entry only produces the result; the scheduler's dispatch calls destroy() via
// the Fiber finish hook AFTER control has left the fiber (safe free point).
template <class T>
struct TaskDriver {
  mem::AllocatorRef alloc;
  Task<T> task;
  JoinState<T>* state;
  Fiber fiber;

  void arm_finish_hook() noexcept {
    fiber.set_on_finish(&TaskDriver::destroy, this);
  }

  static void entry(void* p) noexcept {
    auto* d = static_cast<TaskDriver*>(p);
    JoinState<T>* st = d->state;
    ScopedCancelFlag cancel_scope(&st->cancel_req);
    d->task.resume();  // one resume drives the whole coroutine (fiber-yields on await)
    if (st->cancel_req.load(std::memory_order_acquire) && !d->task.done()) {
      st->set_error(Status::Cancelled("task cancellation requested"));
    } else {
      try {
        st->set_value(d->task.result());
      } catch (...) {
        // The coroutine promise captures user exceptions. Never let its
        // result() rethrow escape this noexcept fiber trampoline.
        st->set_error(Status::Internal("task threw an exception"));
      }
    }
    // Do NOT free here -- entry runs ON the fiber's stack; the trampoline still
    // uses the Fiber after we return. dispatch() runs destroy() post-return.
  }

  // Finish hook: run by the worker's dispatch after the fiber is done() and off
  // its stack. Drops the task-side JoinState ref, then frees the driver (which
  // destroys the Fiber + its stack -- now safe, nothing runs on it).
  static void destroy(void* p) noexcept {
    auto* d = static_cast<TaskDriver*>(p);
    JoinState<T>* st = d->state;
    mem::AllocatorRef a = d->alloc;
    d->~TaskDriver();
    a.deallocate(d, sizeof(TaskDriver), alignof(TaskDriver));
    st->release();  // after driver is gone: last ref may free the state
  }
};

template <>
struct TaskDriver<void> {
  mem::AllocatorRef alloc;
  Task<void> task;
  JoinState<void>* state;
  Fiber fiber;

  void arm_finish_hook() noexcept {
    fiber.set_on_finish(&TaskDriver::destroy, this);
  }

  static void entry(void* p) noexcept {
    auto* d = static_cast<TaskDriver*>(p);
    JoinState<void>* st = d->state;
    ScopedCancelFlag cancel_scope(&st->cancel_req);
    d->task.resume();
    if (st->cancel_req.load(std::memory_order_acquire) && !d->task.done()) {
      st->set_error(Status::Cancelled("task cancellation requested"));
    } else {
      try {
        d->task.result();
        st->set_done();
      } catch (...) {
        st->set_error(Status::Internal("task threw an exception"));
      }
    }
  }

  static void destroy(void* p) noexcept {
    auto* d = static_cast<TaskDriver*>(p);
    JoinState<void>* st = d->state;
    mem::AllocatorRef a = d->alloc;
    d->~TaskDriver();
    a.deallocate(d, sizeof(TaskDriver), alignof(TaskDriver));
    st->release();
  }
};

}  // namespace detail

template <class T>
JoinHandle<T> spawn_task(Scheduler& sched, mem::AllocatorRef alloc, Task<T> task,
                         std::size_t stack_size = kDefaultStackSize) noexcept {
  using detail::JoinState;
  using detail::TaskDriver;
  if (!alloc.valid()) {
    return JoinHandle<T>(
        Status::InvalidArgument("spawn_task requires a complete allocator"));
  }
  if (stack_size < mem::kFiberStackSmall) {
    return JoinHandle<T>(
        Status::InvalidArgument("fiber stack is smaller than the safe tier"));
  }
  if (!task.valid()) {
    return JoinHandle<T>(
        Status::ResourceExhausted("coroutine frame allocation failed"));
  }
  auto* st = static_cast<JoinState<T>*>(
      alloc.allocate(sizeof(JoinState<T>), alignof(JoinState<T>)));
  if (st == nullptr) {
    return JoinHandle<T>(
        Status::ResourceExhausted("join state allocation failed"));
  }
  new (st) JoinState<T>{};
  st->alloc = alloc;

  auto* d = static_cast<TaskDriver<T>*>(
      alloc.allocate(sizeof(TaskDriver<T>), alignof(TaskDriver<T>)));
  if (d == nullptr) {
    st->set_error(Status::ResourceExhausted("task driver allocation failed"));
    st->release();  // no driver exists: drop its reserved task-side reference
    return JoinHandle<T>(st);
  }
  // Recycle the fiber stack through the scheduler's shared per-tier pool when the
  // size is a standard tier (the common case: kDefaultStackSize) -- a hot
  // spawn/join loop then reuses stacks instead of reserve_guarded'ing every time.
  // A non-tier size gets a one-off stack (pool == nullptr, released on destroy).
  mem::FiberStackPool* pool = sched.fiber_stack_pool(stack_size);
  auto stack_result =
      pool != nullptr ? pool->acquire() : mem::FiberStack::allocate(stack_size);
  if (!stack_result) {
    alloc.deallocate(d, sizeof(TaskDriver<T>), alignof(TaskDriver<T>));
    st->set_error(std::move(stack_result).error().WithContext(
        "task fiber stack allocation"));
    st->release();
    return JoinHandle<T>(st);
  }
  mem::FiberStack stack = *std::move(stack_result);
  new (d) TaskDriver<T>{alloc, std::move(task), st,
                        Fiber(std::move(stack), &TaskDriver<T>::entry, d, pool)};
  // dispatch() frees the driver via this hook AFTER the fiber finishes and is off
  // its stack (a fiber must never free its own stack from within entry). The
  // Fiber dtor returns the stack to `pool` (if any) at that safe point.
  d->arm_finish_hook();
  if (!sched.spawn(&d->fiber)) {
    // Admission closed or a recursively-submitting worker saturated every
    // bounded runnable queue. Resolve the handle instead of leaking the
    // self-owned driver and leaving join() blocked forever.
    st->set_error(
        sched.accepting()
            ? Status::ResourceExhausted(
                  "scheduler runnable queues are full on a worker")
            : Status::Cancelled("scheduler is shutting down"));
    TaskDriver<T>::destroy(d);
  }
  return JoinHandle<T>(st);
}

// ---- when_all / when_any (select) -------------------------------------------
// Both block (fiber-aware) on a stack GroupSignal that each task signals on
// completion. They do NOT consume the handles -- after when_all every handle is
// finished (join each for results); after when_any the winner index is returned
// and the caller may join the winner + request cancellation/join the losers.
// The waiter
// uninstalls the group from every state before returning so a late loser never
// touches the (stack) GroupSignal (folly WhenN UAF guard).

namespace detail {

// Shared engine: install `g` into each state, pre-accounting any already done;
// block until fired; then uninstall from all. Returns g.winner.
template <class... States>
int group_wait(GroupSignal& g, long target, States*... states) noexcept {
  constexpr int kN = static_cast<int>(sizeof...(States));
  g.remaining.store(target, std::memory_order_relaxed);
  // Install into each; collect indices of those already done to pre-account.
  int idx = 0;
  bool already[kN == 0 ? 1 : kN] = {};
  (void)std::initializer_list<int>{
      (already[idx] =
           (states == nullptr || states->install_group(&g, idx)),
       ++idx, 0)...};
  // Pre-account completions that happened before/at install (their on_complete
  // won't fire for us since group wasn't set at their completion).
  for (int i = 0; i < kN; ++i) {
    if (already[i]) g.on_complete(i);
  }
  // Block until the target is met.
  g.completed.wait();
  const int winner = g.winner.load(std::memory_order_acquire);
  // Uninstall from every state BEFORE g dies (critical: a not-yet-fired loser
  // would otherwise call on_complete on a dead stack object).
  (void)std::initializer_list<int>{
      (states != nullptr ? (states->clear_group(), 0) : 0)...};
  return winner;
}

template <class T>
int group_wait_span(GroupSignal& group, long target,
                    std::span<JoinHandle<T>> handles) noexcept {
  HYLE_FATAL_ASSERT(
      handles.size() <= static_cast<std::size_t>((std::numeric_limits<int>::max)()));
  group.remaining.store(target, std::memory_order_relaxed);
  for (std::size_t i = 0; i < handles.size(); ++i) {
    auto* state = handles[i].state_for_group_();
    if (state == nullptr || state->install_group(&group, static_cast<int>(i)))
      group.on_complete(static_cast<int>(i));
  }
  group.completed.wait();
  const int winner = group.winner.load(std::memory_order_acquire);
  for (JoinHandle<T>& handle : handles) {
    if (auto* state = handle.state_for_group_(); state != nullptr)
      state->clear_group();
  }
  return winner;
}

}  // namespace detail

// Wait until EVERY handle's task has finished. Handles are left intact (join
// each afterwards for results). No-op for an empty pack.
template <class... Ts>
void when_all(JoinHandle<Ts>&... handles) noexcept {
  if constexpr (sizeof...(Ts) > 0) {
    detail::GroupSignal g;
    detail::group_wait(g, static_cast<long>(sizeof...(Ts)),
                       handles.state_for_group_()...);
  }
}

// Runtime-sized homogeneous fan-out (batch decode shards, dynamically sized IO
// groups). This is the span counterpart to the heterogeneous parameter-pack
// overload and does not allocate or consume the handles.
template <class T>
void when_all(std::span<JoinHandle<T>> handles) noexcept {
  if (handles.empty()) return;
  detail::GroupSignal group;
  detail::group_wait_span(group, static_cast<long>(handles.size()), handles);
}

// Wait until the FIRST handle's task finishes; returns its index. Handles are
// left intact. request_cancel() only publishes a cooperative flag; if a loser
// owns an in-flight wait, cancel that operation through its CancelToken/owner
// before joining it. This is the select primitive underpinning C2 timeout io
// (io op vs timer, first wins).
template <class... Ts>
int when_any(JoinHandle<Ts>&... handles) noexcept {
  static_assert(sizeof...(Ts) > 0, "when_any needs at least one handle");
  detail::GroupSignal g;
  return detail::group_wait(g, 1, handles.state_for_group_()...);
}


template <class T>
std::size_t when_any(std::span<JoinHandle<T>> handles) noexcept {
  HYLE_FATAL_ASSERT(!handles.empty());
  detail::GroupSignal group;
  return static_cast<std::size_t>(
      detail::group_wait_span(group, 1, handles));
}

}  // namespace hyle::sched
