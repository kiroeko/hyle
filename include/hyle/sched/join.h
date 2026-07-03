#pragma once

#include <atomic>
#include <cstddef>
#include <new>
#include <utility>

#include "hyle/error/result.h"
#include "hyle/error/status.h"
#include "hyle/macros.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/sync.h"

// sched-10: JoinHandle + when_all/when_any(select) -- the structured-concurrency
// layer (capability C7, and the load-bearing prerequisite for C2 timeout io / C3
// cancellation). Reference-lib consensus (tokio/folly/seastar/marl, read in full
// 2026-07-03) drove every decision:
//
//  * JoinHandle does NOT own the Fiber -- only a refcounted result slot
//    (JoinState). marl proved a join can be as thin as WaitGroup(1)+slot; tokio's
//    6-bit state machine is complexity born of reusing the task cell for both the
//    future and the output -- HYLE's independent slot sidesteps it.
//  * Completion notification reuses the sched-05 ConditionVariable (dual-mode: a
//    joining fiber suspend()s, a bare thread parks) -- already TSan-proven. No
//    hand-rolled rendezvous state machine.
//  * Cancellation is COOPERATIVE. A stackful fiber cannot be preempted (folly
//    /fibers offers no external cancel, marl none) -- abort() sets a flag and the
//    task checks is_cancel_requested() at its own suspend points and unwinds via
//    RAII. when_any's loser cancellation is the same mechanism. (Real in-flight io
//    cancellation -- IORING_ASYNC_CANCEL/CancelIoEx -- is io-06/C2's job.)
//  * spawn_task OWNS the Fiber via a heap TaskDriver that self-destructs when the
//    fiber finishes (tokio drop==detach). This is the ONE place sched owns a
//    fiber; spawn(Fiber*)'s caller-keeps-it-alive contract is unchanged.
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
  util::SpinLock mu;
  ConditionVariable cv;
  long remaining = 0;  // completions still needed to fire
  int winner = -1;     // index of the first completer
  bool fired = false;

  // Called by a JoinState (under ITS lock) when its task completes while this
  // group is installed. Records the winner and fires when the target is met.
  void on_complete(int idx) noexcept {
    mu.lock();
    if (winner < 0) winner = idx;
    if (remaining > 0 && --remaining == 0) {
      fired = true;
      mu.unlock();
      cv.notify_all();
      return;
    }
    mu.unlock();
  }
};

// The refcounted result slot. Shared by the running task (writes result +
// signals done) and the JoinHandle (reads it). Refcount starts at 2; last drop
// frees it. Completion is signalled through a ConditionVariable so a joiner
// (fiber or thread) blocks correctly regardless of join-vs-complete ordering.
template <class T>
struct JoinState {
  mem::Allocator alloc;
  std::atomic<int> refcount{2};
  util::SpinLock mu;              // guards done/status/storage/group
  ConditionVariable cv;          // a direct joiner waits here
  bool done = false;
  std::atomic<bool> cancel_req{false};
  Status status;                 // OK on success, error/Cancelled otherwise
  GroupSignal* group = nullptr;  // set by when_all/when_any while grouped
  int group_idx = -1;
  alignas(T) unsigned char storage[sizeof(T)];  // result, live iff done&&status.ok()

  void complete_(Status s, T* val) noexcept {
    GroupSignal* g;
    int gi;
    mu.lock();
    status = std::move(s);
    if (status.ok() && val != nullptr) new (storage) T(std::move(*val));
    done = true;
    g = group;
    gi = group_idx;
    mu.unlock();
    if (g != nullptr) g->on_complete(gi);  // outside our lock: g takes its own
    cv.notify_all();
  }
  void set_value(T&& v) noexcept { complete_(Status(), &v); }
  void set_error(Status s) noexcept { complete_(std::move(s), nullptr); }

  Status take(T& out) noexcept {
    mu.lock();
    cv.wait(mu, [&] { return done; });
    Status s = status;
    if (s.ok()) {
      T* p = reinterpret_cast<T*>(storage);
      out = std::move(*p);
      p->~T();
    }
    mu.unlock();
    return s;
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
      if (done && status.ok()) reinterpret_cast<T*>(storage)->~T();  // detached value
      mem::Allocator a = alloc;
      this->~JoinState();
      a.deallocate(this, sizeof(JoinState), alignof(JoinState));
    }
  }
};

// void specialization: no result storage.
template <>
struct JoinState<void> {
  mem::Allocator alloc;
  std::atomic<int> refcount{2};
  util::SpinLock mu;
  ConditionVariable cv;
  bool done = false;
  std::atomic<bool> cancel_req{false};
  Status status;
  GroupSignal* group = nullptr;
  int group_idx = -1;

  void complete_(Status s) noexcept {
    GroupSignal* g;
    int gi;
    mu.lock();
    status = std::move(s);
    done = true;
    g = group;
    gi = group_idx;
    mu.unlock();
    if (g != nullptr) g->on_complete(gi);
    cv.notify_all();
  }
  void set_done() noexcept { complete_(Status()); }
  void set_error(Status s) noexcept { complete_(std::move(s)); }

  Status take() noexcept {
    mu.lock();
    cv.wait(mu, [&] { return done; });
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
      mem::Allocator a = alloc;
      this->~JoinState();
      a.deallocate(this, sizeof(JoinState), alignof(JoinState));
    }
  }
};

// The task's cooperative-cancel flag is published thread-locally while a task
// runs, so is_cancel_requested() (called from anywhere down the task's stack)
// can find it without threading a handle through every frame.
HYLE_SCHED_API void push_cancel_scope(std::atomic<bool>* flag) noexcept;
HYLE_SCHED_API void pop_cancel_scope() noexcept;

}  // namespace detail

// A running task queries this at its own checkpoints; if true it should unwind
// (RAII) and return early. Cooperative -- a stackful fiber is never force-stopped.
HYLE_SCHED_API bool is_cancel_requested() noexcept;

// ---- JoinHandle<T> ----------------------------------------------------------
// Move-only handle to a spawned task's result. join() blocks (fiber-aware from a
// worker) until the task finishes, returning Result<T>. abort() requests
// cooperative cancellation. Dropping detaches (task runs on, result discarded).
template <class T>
class JoinHandle {
 public:
  JoinHandle() noexcept = default;
  explicit JoinHandle(detail::JoinState<T>* s) noexcept : state_(s) {}
  JoinHandle(JoinHandle&& o) noexcept : state_(std::exchange(o.state_, nullptr)) {}
  JoinHandle& operator=(JoinHandle&& o) noexcept {
    if (this != &o) {
      if (state_) state_->release();
      state_ = std::exchange(o.state_, nullptr);
    }
    return *this;
  }
  HYLE_DISABLE_COPY(JoinHandle);
  ~JoinHandle() {
    if (state_) state_->release();  // detach: task runs on, result discarded
  }

  Result<T> join() noexcept {
    T out{};
    Status s = state_->take(out);
    state_->release();
    state_ = nullptr;
    if (!s.ok()) return std::unexpected(std::move(s));
    return out;
  }

  void abort() noexcept {
    if (state_) state_->cancel_req.store(true, std::memory_order_release);
  }
  bool is_finished() const noexcept { return state_ && state_->is_done(); }
  void detach() noexcept {
    if (state_) {
      state_->release();
      state_ = nullptr;
    }
  }
  bool valid() const noexcept { return state_ != nullptr; }

  // Internal: when_all/when_any reach the shared state without consuming.
  detail::JoinState<T>* state_for_group_() const noexcept { return state_; }

 private:
  detail::JoinState<T>* state_ = nullptr;
};

template <>
class JoinHandle<void> {
 public:
  JoinHandle() noexcept = default;
  explicit JoinHandle(detail::JoinState<void>* s) noexcept : state_(s) {}
  JoinHandle(JoinHandle&& o) noexcept : state_(std::exchange(o.state_, nullptr)) {}
  JoinHandle& operator=(JoinHandle&& o) noexcept {
    if (this != &o) {
      if (state_) state_->release();
      state_ = std::exchange(o.state_, nullptr);
    }
    return *this;
  }
  HYLE_DISABLE_COPY(JoinHandle);
  ~JoinHandle() {
    if (state_) state_->release();
  }

  Result<void> join() noexcept {
    Status s = state_->take();
    state_->release();
    state_ = nullptr;
    if (!s.ok()) return std::unexpected(std::move(s));
    return {};
  }
  void abort() noexcept {
    if (state_) state_->cancel_req.store(true, std::memory_order_release);
  }
  bool is_finished() const noexcept { return state_ && state_->is_done(); }
  void detach() noexcept {
    if (state_) {
      state_->release();
      state_ = nullptr;
    }
  }
  bool valid() const noexcept { return state_ != nullptr; }
  detail::JoinState<void>* state_for_group_() const noexcept { return state_; }

 private:
  detail::JoinState<void>* state_ = nullptr;
};

// ---- spawn_task: spawn a Task<T>, get a JoinHandle<T> -----------------------
namespace detail {

// A driver owns the Task + the Fiber that runs it. It must NOT free itself from
// inside entry (the fcontext trampoline reads the Fiber after entry returns -- a
// self-free there is a UAF, caught by TSan as a SEGV in the trampoline). Instead
// entry only produces the result; the scheduler's dispatch calls destroy() via
// the Fiber finish hook AFTER control has left the fiber (safe free point).
template <class T>
struct TaskDriver {
  mem::Allocator alloc;
  Task<T> task;
  JoinState<T>* state;
  Fiber fiber;

  static void entry(void* p) noexcept {
    auto* d = static_cast<TaskDriver*>(p);
    JoinState<T>* st = d->state;
    push_cancel_scope(&st->cancel_req);
    d->task.resume();  // one resume drives the whole coroutine (fiber-yields on await)
    pop_cancel_scope();
    if (st->cancel_req.load(std::memory_order_acquire) && !d->task.done()) {
      st->set_error(Status::Cancelled("task aborted"));
    } else {
      st->set_value(d->task.result());
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
    mem::Allocator a = d->alloc;
    d->~TaskDriver();
    a.deallocate(d, sizeof(TaskDriver), alignof(TaskDriver));
    st->release();  // after driver is gone: last ref may free the state
  }
};

template <>
struct TaskDriver<void> {
  mem::Allocator alloc;
  Task<void> task;
  JoinState<void>* state;
  Fiber fiber;

  static void entry(void* p) noexcept {
    auto* d = static_cast<TaskDriver*>(p);
    JoinState<void>* st = d->state;
    push_cancel_scope(&st->cancel_req);
    d->task.resume();
    pop_cancel_scope();
    if (st->cancel_req.load(std::memory_order_acquire) && !d->task.done()) {
      st->set_error(Status::Cancelled("task aborted"));
    } else {
      d->task.result();  // propagate exception path (no-op under no-except)
      st->set_done();
    }
  }

  static void destroy(void* p) noexcept {
    auto* d = static_cast<TaskDriver*>(p);
    JoinState<void>* st = d->state;
    mem::Allocator a = d->alloc;
    d->~TaskDriver();
    a.deallocate(d, sizeof(TaskDriver), alignof(TaskDriver));
    st->release();
  }
};

}  // namespace detail

template <class T>
JoinHandle<T> spawn_task(Scheduler& sched, mem::Allocator alloc, Task<T> task,
                         std::size_t stack_size = kDefaultStackSize) noexcept {
  using detail::JoinState;
  using detail::TaskDriver;
  auto* st = static_cast<JoinState<T>*>(
      alloc.allocate(sizeof(JoinState<T>), alignof(JoinState<T>)));
  HYLE_FATAL_ASSERT(st != nullptr);
  new (st) JoinState<T>{};
  st->alloc = alloc;

  auto* d = static_cast<TaskDriver<T>*>(
      alloc.allocate(sizeof(TaskDriver<T>), alignof(TaskDriver<T>)));
  HYLE_FATAL_ASSERT(d != nullptr);
  new (d) TaskDriver<T>{alloc, std::move(task), st,
                        Fiber(alloc, stack_size, &TaskDriver<T>::entry, d)};
  // dispatch() frees the driver via this hook AFTER the fiber finishes and is off
  // its stack (a fiber must never free its own stack from within entry).
  d->fiber.set_on_finish(&TaskDriver<T>::destroy, d);
  sched.spawn(&d->fiber);
  return JoinHandle<T>(st);
}

// ---- when_all / when_any (select) -------------------------------------------
// Both block (fiber-aware) on a stack GroupSignal that each task signals on
// completion. They do NOT consume the handles -- after when_all every handle is
// finished (join each for results); after when_any the winner index is returned
// and the caller may join the winner + abort()/join the losers. The waiter
// uninstalls the group from every state before returning so a late loser never
// touches the (stack) GroupSignal (folly WhenN UAF guard).

namespace detail {

// Shared engine: install `g` into each state, pre-accounting any already done;
// block until fired; then uninstall from all. Returns g.winner.
template <class... States>
int group_wait(GroupSignal& g, long target, States*... states) noexcept {
  detail::JoinState<void>* dummy = nullptr;  // only for pack-size deduction guard
  (void)dummy;
  constexpr int kN = static_cast<int>(sizeof...(States));
  g.remaining = target;
  // Install into each; collect indices of those already done to pre-account.
  int idx = 0;
  bool already[kN == 0 ? 1 : kN] = {};
  (void)std::initializer_list<int>{
      (already[idx] = states->install_group(&g, idx), ++idx, 0)...};
  // Pre-account completions that happened before/at install (their on_complete
  // won't fire for us since group wasn't set at their completion).
  for (int i = 0; i < kN; ++i) {
    if (already[i]) {
      g.mu.lock();
      if (g.winner < 0) g.winner = i;
      bool fire = (g.remaining > 0 && --g.remaining == 0);
      if (fire) g.fired = true;
      g.mu.unlock();
    }
  }
  // Block until the target is met.
  g.mu.lock();
  g.cv.wait(g.mu, [&] { return g.fired; });
  int winner = g.winner;
  g.mu.unlock();
  // Uninstall from every state BEFORE g dies (critical: a not-yet-fired loser
  // would otherwise call on_complete on a dead stack object).
  (void)std::initializer_list<int>{(states->clear_group(), 0)...};
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

// Wait until the FIRST handle's task finishes; returns its index. Handles are
// left intact -- the caller decides whether to abort()/join the losers. This is
// the select primitive underpinning C2 timeout io (io op vs timer, first wins).
template <class... Ts>
int when_any(JoinHandle<Ts>&... handles) noexcept {
  static_assert(sizeof...(Ts) > 0, "when_any needs at least one handle");
  detail::GroupSignal g;
  return detail::group_wait(g, 1, handles.state_for_group_()...);
}

}  // namespace hyle::sched
