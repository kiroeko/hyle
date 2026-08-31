#pragma once

#include <coroutine>
#include <cstddef>
#include <exception>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

#include "hyle/core/compiler.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/waker.h"

// Coroutine <-> fiber glue. A `co_await` inside a
// fiber suspends the FIBER (fcontext stack switch), not just the coroutine, so a
// deep call chain can await without every frame being a coroutine. The coroutine
// frame lives in the supplied Allocator; the fiber owns the execution stack, so
// fiber suspend/resume naturally freezes/thaws the calling stack around it.
//
// Two pieces:
//   - Task<T>: a coroutine whose frame is allocated through a HYLE mem::Allocator
//     (not the default operator new -- charter: explicit allocation everywhere).
//   - WakerAwaitable: `co_await` it to leave the run loop until a Waker fires.
//     await_suspend captures current_waker() (so the awaiter can hand it to an
//     io/gpu/timer source) and suspend()s the fiber; the source's wake() re-injects
//     the fiber, fiber resume returns from suspend(), and co_await continues.
//
// Coroutine frames are allocator-backed; execution is driven by a stackful
// fiber. Suspending that fiber freezes the caller stack while the coroutine
// frame remains stable in allocator storage.
//
// This is the ergonomic shell over the Waker spine; the raw fiber +
// Waker interface stays available for hot paths that don't want a coroutine
// frame.

namespace hyle::sched {

// Suspend the current fiber until its Waker fires (see waker.h). Declared here
// too so coroutine code needn't include waker.h separately.
HYLE_SCHED_API void suspend() noexcept;
HYLE_SCHED_API Waker current_waker() noexcept;

// ---- Task<T>: allocator-backed coroutine ------------------------------------
// A lazily-started coroutine returning T (or void). The promise allocates the
// coroutine frame from a mem::Allocator threaded through operator new/delete, so
// no hidden heap. Move-only handle; destroying it destroys the frame.
//
// Task remains the coroutine-frame/value primitive; spawn_task, JoinHandle and
// when_all/when_any provide runtime ownership and structured composition.
template <class T>
class Task;

namespace detail {

// Shared promise machinery: frame allocation via mem::Allocator, plus the
// final-suspend that hands control back so the driver can observe completion.
struct PromiseAllocBase {
  // Prefix bytes reserved before the frame to stash the Allocator, sized to
  // preserve the frame's alignment (>= sizeof(mem::Allocator), a multiple of
  // max_align_t so the returned pointer stays maximally aligned).
  static constexpr std::size_t kPrefix =
      (sizeof(mem::Allocator) + alignof(std::max_align_t) - 1) &
      ~(alignof(std::max_align_t) - 1);

  // operator new: the coroutine-frame ABI passes the coroutine's call args after
  // `size`. We require Task coroutines to take a mem::Allocator as their FIRST
  // parameter, so the frame is allocated from it (the standard "allocator as
  // first coroutine parameter" idiom). No default operator new is provided, so a
  // Task coroutine that forgets the allocator param fails to compile.
  static void* operator new(std::size_t n, mem::Allocator alloc,
                            auto&&...) noexcept {
    if (!alloc.valid() ||
        n > (std::numeric_limits<std::size_t>::max)() - kPrefix)
      return nullptr;
    void* base = alloc.allocate(n + kPrefix, alignof(std::max_align_t));
    if (base == nullptr) return nullptr;
    *static_cast<mem::Allocator*>(base) = alloc;
    return static_cast<char*>(base) + kPrefix;  // aligned: kPrefix % maxalign==0
  }
  static void operator delete(void* p, std::size_t n) noexcept {
    void* base = static_cast<char*>(p) - kPrefix;
    mem::Allocator alloc = *static_cast<mem::Allocator*>(base);
    alloc.deallocate(base, n + kPrefix, alignof(std::max_align_t));
  }
};

}  // namespace detail

template <class T>
class Task {
 public:
  static_assert(std::is_nothrow_move_constructible_v<T>,
                "Task<T> requires a noexcept move-constructible result");
  static_assert(alignof(T) <= alignof(std::max_align_t),
                "Task<T> does not support over-aligned result types");

  struct promise_type : detail::PromiseAllocBase {
    // Optional storage avoids imposing default-construction and move-assignment
    // on every task result. A coroutine result only needs to be move-constructible.
    std::optional<T> value_;
    std::exception_ptr exc_;
    std::coroutine_handle<> continuation_;  // who to resume on completion (unused in minimal)

    Task get_return_object() noexcept {
      return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
    }
    static Task get_return_object_on_allocation_failure() noexcept {
      return Task{};
    }
    // Lazy start: the coroutine is suspended at its open brace; the driver calls
    // start() to run it (so the caller controls when it begins on a fiber).
    std::suspend_always initial_suspend() noexcept { return {}; }
    // Keep the frame alive at completion so the driver can read result(); the
    // Task destructor destroys it. final_suspend never resumes.
    std::suspend_always final_suspend() noexcept { return {}; }
    void return_value(T v) noexcept { value_.emplace(std::move(v)); }
    void unhandled_exception() noexcept { exc_ = std::current_exception(); }
  };

  using handle_t = std::coroutine_handle<promise_type>;

  Task() noexcept = default;
  explicit Task(handle_t h) noexcept : h_(h) {}
  Task(Task&& o) noexcept : h_(std::exchange(o.h_, {})) {}
  Task& operator=(Task&& o) noexcept {
    if (this != &o) {
      if (h_) h_.destroy();
      h_ = std::exchange(o.h_, {});
    }
    return *this;
  }
  Task(const Task&) = delete;
  Task& operator=(const Task&) = delete;
  ~Task() {
    if (h_) h_.destroy();
  }

  // Run the coroutine until it next suspends or finishes. Call repeatedly (from
  // the fiber body) until done(). Safe to call after done() (no-op).
  void resume() {
    if (h_ && !h_.done()) h_.resume();
  }
  bool done() const noexcept { return !h_ || h_.done(); }
  bool valid() const noexcept { return static_cast<bool>(h_); }
  T result() {
    HYLE_FATAL_ASSERT(h_);
    if (h_.promise().exc_) std::rethrow_exception(h_.promise().exc_);
    HYLE_FATAL_ASSERT(h_.promise().value_.has_value());
    return std::move(*h_.promise().value_);
  }

  // ---- awaiter: co_await a sub-Task from within another Task ----------------
  // HYLE is STACKFUL: a co_await inside the sub-Task yields the whole FIBER (see
  // SuspendFiber), freezing this parent frame with it. So awaiting a sub-Task is
  // just "drive it to done on this fiber" -- each of its internal co_awaits parks
  // the fiber and resumes it, and when the sub-Task finishes we lift its result.
  // No continuation chaining is needed (the fiber suspend covers everything), which
  // is why this minimal awaiter suffices for composing nested Task pumps.
  bool await_ready() noexcept { return done(); }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    // Run the sub-Task to completion on THIS fiber. resume() returns either when it
    // finishes or when one of its co_awaits fiber-yields; on a yield the fiber (and
    // this whole coroutine) is frozen and control is already back at the worker, so
    // when we're resumed the loop continues the sub-Task. Loop until it's done.
    while (!h_.done()) h_.resume();
    return false;  // sub-Task done -> don't park the parent; go to await_resume
  }
  T await_resume() { return result(); }

 private:
  handle_t h_{};
};

// void specialization.
template <>
class Task<void> {
 public:
  struct promise_type : detail::PromiseAllocBase {
    std::exception_ptr exc_;
    Task get_return_object() noexcept {
      return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
    }
    static Task get_return_object_on_allocation_failure() noexcept {
      return Task{};
    }
    std::suspend_always initial_suspend() noexcept { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    void return_void() noexcept {}
    void unhandled_exception() noexcept { exc_ = std::current_exception(); }
  };
  using handle_t = std::coroutine_handle<promise_type>;

  Task() noexcept = default;
  explicit Task(handle_t h) noexcept : h_(h) {}
  Task(Task&& o) noexcept : h_(std::exchange(o.h_, {})) {}
  Task& operator=(Task&& o) noexcept {
    if (this != &o) {
      if (h_) h_.destroy();
      h_ = std::exchange(o.h_, {});
    }
    return *this;
  }
  Task(const Task&) = delete;
  Task& operator=(const Task&) = delete;
  ~Task() {
    if (h_) h_.destroy();
  }
  void resume() {
    if (h_ && !h_.done()) h_.resume();
  }
  bool done() const noexcept { return !h_ || h_.done(); }
  bool valid() const noexcept { return static_cast<bool>(h_); }
  void result() {
    HYLE_FATAL_ASSERT(h_);
    if (h_.promise().exc_) std::rethrow_exception(h_.promise().exc_);
  }

  // awaiter: co_await a sub-Task<void> (see Task<T>::await_* for the stackful
  // rationale -- the fiber suspend covers the sub-Task's internal awaits).
  bool await_ready() noexcept { return done(); }
  bool await_suspend(std::coroutine_handle<>) noexcept {
    while (!h_.done()) h_.resume();
    return false;
  }
  void await_resume() { result(); }

 private:
  handle_t h_{};
};

// ---- WakerAwaitable: co_await to suspend the fiber until woken ---------------
// `co_await suspend_fiber()` leaves the run loop; the Waker captured in
// await_suspend must be handed to something that will wake() it (io/gpu/timer),
// exactly like the raw suspend() path -- just wrapped in coroutine syntax. If you
// need the Waker to register it with a source, use co_await capture_waker()
// instead, which yields the Waker to your code before suspending.
struct SuspendFiber {
  bool await_ready() const noexcept { return false; }
  // await_suspend runs on the fiber's stack. suspend() sets the wait flag and
  // fiber-yields (fcontext switch back to the worker); it does NOT return until a
  // Waker::wake() re-injects this fiber. Once it returns the fiber (and the whole
  // coroutine stack frozen on it) is live again, so we return FALSE = "don't keep
  // the coroutine suspended, resume it now" -> control flows into await_resume
  // and the code after co_await. Returning void would suspend the coroutine a
  // SECOND time (handing control back to resume()'s caller) -- wrong, since the
  // wait already happened at the fiber level. This is the fiber<->coroutine
  // bridge: one fiber suspend covers the co_await; the coroutine never parks
  // independently of its fiber.
  bool await_suspend(std::coroutine_handle<>) const noexcept {
    suspend();     // fiber yields here; returns when re-injected by a wake
    return false;  // fiber is back -> continue the coroutine immediately
  }
  void await_resume() const noexcept {}
};

// Yields the current fiber's Waker to the awaiter, THEN suspends -- so the code
// can hand the Waker to an async source before leaving the run loop. Usage:
//   Waker w = co_await CaptureWaker{};   // w is valid; fiber NOT yet suspended
//   source.arm(op, w);                   // register the wake
//   co_await SuspendFiber{};             // now leave until wake()
// (Two-step so there is no window where the fiber suspends before the source
// knows its Waker.)
struct CaptureWaker {
  bool await_ready() const noexcept { return true; }  // never suspends
  void await_suspend(std::coroutine_handle<>) const noexcept {}
  Waker await_resume() const noexcept { return current_waker(); }
};

}  // namespace hyle::sched
