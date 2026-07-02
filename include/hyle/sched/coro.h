#pragma once

#include <coroutine>
#include <cstddef>
#include <exception>
#include <utility>

#include "hyle/macros.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/waker.h"

// sched-06: the coroutine <-> fiber glue (sched-design §4). A `co_await` inside a
// fiber suspends the FIBER (fcontext stack switch), not just the coroutine, so a
// deep call chain can await without every frame being a coroutine. The coroutine
// frame lives on the fiber's stack, so fiber suspend/resume naturally freezes/
// thaws the coroutine with it.
//
// Two pieces:
//   - Task<T>: a coroutine whose frame is allocated through a HYLE mem::Allocator
//     (not the default operator new -- charter: explicit allocation everywhere).
//   - WakerAwaitable: `co_await` it to leave the run loop until a Waker fires.
//     await_suspend captures current_waker() (so the awaiter can hand it to an
//     io/gpu/timer source) and suspend()s the fiber; the source's wake() re-injects
//     the fiber, fiber resume returns from suspend(), and co_await continues.
//
// This is the ergonomic shell over the sched-04a Waker spine; the raw fiber +
// Waker interface stays available for hot paths that don't want a coroutine
// frame (sched-design §4.3).

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
// Minimal by design (sched-06 is the glue, not a full task library): start(),
// done(), result(). Composition combinators (when_all/select) are later steps.
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
  struct promise_type : detail::PromiseAllocBase {
    T value_{};
    std::exception_ptr exc_;
    std::coroutine_handle<> continuation_;  // who to resume on completion (unused in minimal)

    Task get_return_object() noexcept {
      return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
    }
    // Lazy start: the coroutine is suspended at its open brace; the driver calls
    // start() to run it (so the caller controls when it begins on a fiber).
    std::suspend_always initial_suspend() noexcept { return {}; }
    // Keep the frame alive at completion so the driver can read result(); the
    // Task destructor destroys it. final_suspend never resumes.
    std::suspend_always final_suspend() noexcept { return {}; }
    void return_value(T v) noexcept { value_ = std::move(v); }
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
  HYLE_DISABLE_COPY(Task);
  ~Task() {
    if (h_) h_.destroy();
  }

  // Run the coroutine until it next suspends or finishes. Call repeatedly (from
  // the fiber body) until done(). Safe to call after done() (no-op).
  void resume() {
    if (h_ && !h_.done()) h_.resume();
  }
  bool done() const noexcept { return !h_ || h_.done(); }
  T result() {
    if (h_.promise().exc_) std::rethrow_exception(h_.promise().exc_);
    return std::move(h_.promise().value_);
  }

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
  HYLE_DISABLE_COPY(Task);
  ~Task() {
    if (h_) h_.destroy();
  }
  void resume() {
    if (h_ && !h_.done()) h_.resume();
  }
  bool done() const noexcept { return !h_ || h_.done(); }
  void result() {
    if (h_ && h_.promise().exc_) std::rethrow_exception(h_.promise().exc_);
  }

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
