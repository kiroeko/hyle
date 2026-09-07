#pragma once

#include <atomic>
#include <cstddef>
#if defined(HYLE_TRACE) && defined(HYLE_TRACE_TRACY)
#  include <memory>
#endif

#include "hyle/core/compiler.h"
#include "hyle/error/result.h"
#include "hyle/mem/fiber_stack.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_SCHED_API
#elif defined(HYLE_SCHED_BUILD)
#  define HYLE_SCHED_API HYLE_EXPORT
#else
#  define HYLE_SCHED_API HYLE_IMPORT
#endif

namespace hyle::sched {

class Fiber;

namespace detail {
struct FiberAccess;
template <class T>
struct TaskDriver;

// One active cooperative-cancellation source on a sched execution context.
// Fiber work stores the chain on the Fiber so migration is safe; worker-stack
// work stores it in thread-local state for the duration of the callback. The
// record lives in the owning RAII scope, so pushing a source allocates nothing.
// A linked stack is required: a request token must compose with
// JoinHandle::request_cancel instead of replacing it.
struct CancelFlagLink {
  std::atomic<bool>* flag = nullptr;
  CancelFlagLink* previous = nullptr;
  CancelFlagLink** owner_head = nullptr;
};

HYLE_SCHED_API bool push_cancel_flag(CancelFlagLink* link,
                                     std::atomic<bool>* flag) noexcept;
HYLE_SCHED_API void pop_cancel_flag(CancelFlagLink* link) noexcept;
HYLE_SCHED_API bool current_cancel_requested() noexcept;

class ScopedCancelFlag {
 public:
  explicit ScopedCancelFlag(std::atomic<bool>* flag) noexcept
      : active_(push_cancel_flag(&link_, flag)) {}
  ~ScopedCancelFlag() {
    if (active_) pop_cancel_flag(&link_);
  }
  ScopedCancelFlag(const ScopedCancelFlag&) = delete;
  ScopedCancelFlag& operator=(const ScopedCancelFlag&) = delete;

 private:
  CancelFlagLink link_{};
  bool active_ = false;
};
}  // namespace detail

// Tiered fiber stack sizes (usable bytes; a guard page is added below).
// Prefer these over raw literals so io / default / compute stay consistent.
inline constexpr std::size_t kSmallStackSize = mem::kFiberStackSmall;
inline constexpr std::size_t kDefaultStackSize = mem::kFiberStackDefault;
inline constexpr std::size_t kLargeStackSize = mem::kFiberStackLarge;

// Stackful fiber: a cooperative unit of execution with its own stack, able to
// suspend (yield) at ANY call depth and be resumed later. Built on the vendored
// fcontext assembly (see src/sched/fcontext.h). This is the scheduling unit the
// rest of sched/ manages.
//
// Stack storage is mem::FiberStack (platform VM + permanent low guard page).
// Stack grows down from top() toward base()/guard.
//
// Usage:
//   Fiber f(kDefaultStackSize, &my_entry, arg);
//   f.resume();            // runs my_entry until it yields or returns
//   ... Fiber::yield() ... // called from inside my_entry, returns to resume()
//   f.resume();            // continues after the yield
//   f.done();              // true once my_entry has returned
//
// A fiber is resumed by exactly one thread at a time. It may migrate between
// worker threads only while suspended. Ownership is movable before the first
// resume; moving a started fiber is forbidden because fcontext retains the
// original object address handed to its trampoline.
//
// API macro is on the methods, not the class: exporting the whole class would
// drag members across the seam (MSVC C4251). Method-level export keeps the layout
// client-side (same pattern as Parker).
class Fiber {
 public:
  // A raw fiber entry crosses the fcontext trampoline, so exceptions cannot
  // propagate to its resumer. The type rejects potentially-throwing functions
  // at compile time instead of terminating after the stack switch.
  using EntryFn = void (*)(void* arg) noexcept;

  // Recoverable construction for user-controlled inputs. Returns a structured
  // allocation/validation error and does not start the fiber.
  HYLE_SCHED_API static Result<Fiber> create(std::size_t stack_size,
                                             EntryFn entry,
                                             void* arg) noexcept;

  // Allocate a guarded stack of `stack_size` usable bytes and prepare `entry`.
  // Convenience contract for trusted sizes: invalid input/allocation failure is
  // fatal. Use create() when failure is recoverable. Does not start until resume.
  HYLE_SCHED_API Fiber(std::size_t stack_size, EntryFn entry, void* arg);

  // Take ownership of a pre-built stack (e.g. from FiberStackPool::acquire).
  // If `pool` is non-null, the stack is released back to that pool on destroy
  // (pool must outlive the fiber).
  HYLE_SCHED_API Fiber(mem::FiberStack stack, EntryFn entry, void* arg,
                       mem::FiberStackPool* pool = nullptr);

  HYLE_SCHED_API ~Fiber();

  HYLE_SCHED_API Fiber(Fiber&& other) noexcept;
  HYLE_SCHED_API Fiber& operator=(Fiber&& other) noexcept;
  Fiber(const Fiber&) = delete;
  Fiber& operator=(const Fiber&) = delete;

  // Enter (or continue) the fiber. Returns when the fiber yields or finishes.
  // Precondition: !done().
  HYLE_SCHED_API void resume();

  // True once the entry function has returned (fiber is spent).
  bool done() const noexcept { return finished_; }

  // Usable stack size in bytes (page-rounded; excludes the guard page).
  std::size_t stack_size() const noexcept {
    return stack_.valid() ? stack_.size() : 0;
  }
  bool has_stack_guard() const noexcept {
    return stack_.valid() && stack_.has_guard();
  }

  // Suspend the currently-running fiber, returning control to its resumer.
  // Must be called from within a fiber's call stack (asserts otherwise).
  HYLE_SCHED_API static void yield() noexcept;

  // The fiber currently running on this thread, or nullptr if none. Lets the
  // scheduler's suspend()/current_waker() name "this fiber" without the caller
  // threading a handle through.
  HYLE_SCHED_API static Fiber* current() noexcept;

  // Optional profiler track label. The pointer must remain valid for the
  // Fiber's lifetime (normally a string literal or runtime-owned name).
  const char* trace_name() const noexcept { return name_; }
  void set_trace_name(const char* name) noexcept { name_ = name; }

 private:
  // Scheduling/integration machinery is intentionally not ordinary user API.
  // Scheduler, synchronization and structured-task drivers own these state
  // transitions; ordinary users compose through their public operations.
  friend struct FiberLaunch;
  friend struct detail::FiberAccess;
  template <class T>
  friend struct detail::TaskDriver;
  friend bool detail::push_cancel_flag(detail::CancelFlagLink*,
                                       std::atomic<bool>*) noexcept;
  friend void detail::pop_cancel_flag(detail::CancelFlagLink*) noexcept;
  friend bool detail::current_cancel_requested() noexcept;

  // ---- wait/wake coordination ------------------------------------------------
  // A fiber that yields to WAIT on an external event (vs an ordinary cooperative
  // yield) sets this flag before yielding; the scheduler reads+clears it after
  // resume() returns and, if set, does NOT reschedule -- the fiber sits out of
  // the run loop until a Waker re-injects it.
  void request_wait() noexcept { wait_requested_ = true; }
  bool consume_wait_request() noexcept {
    bool r = wait_requested_;
    wait_requested_ = false;
    return r;
  }

  // ---- worker affinity -------------------------------------------------------
  // A pinned fiber runs ONLY on its owner worker: the scheduler routes it to that
  // worker's private queue on spawn AND on every wake, and the work-stealing path
  // never takes it. -1 = unpinned (ordinary stealable fiber). This is what lets a
  // connection's fiber stay on the core that owns its io ring (submit + reap +
  // resume all single-threaded, share-nothing). Set once before spawn_on.
  int owner_worker() const noexcept { return owner_worker_; }
  void set_owner_worker(int idx) noexcept { owner_worker_ = idx; }

  // ---- finish hook -----------------------------------------------------------
  // Optional callback the scheduler invokes AFTER this fiber has finished, from
  // the worker's dispatch -- i.e. once control has returned from the fiber and its
  // stack is no longer executing. This is the ONLY safe point to free a fiber that
  // owns itself (spawn_task's TaskDriver): a fiber must never free its own stack
  // from within its own entry (the fcontext trampoline still reads the Fiber after
  // entry returns). Raw fn-ptr + ctx, so sched stays ignorant of TaskDriver.
  // `ctx` is typically the owning object (the driver) to be destroyed; after the
  // hook runs the Fiber may itself be gone, so dispatch reads nothing more from it.
  using FinishFn = void (*)(void* ctx) noexcept;
  void set_on_finish(FinishFn fn, void* ctx) noexcept {
    on_finish_ = fn;
    on_finish_ctx_ = ctx;
  }
  FinishFn on_finish() const noexcept { return on_finish_; }
  void* on_finish_ctx() const noexcept { return on_finish_ctx_; }

  // Symmetric rendezvous closing the wake-before-suspend race. BOTH the worker
  // (after resume() returns, having seen a wait request) and Waker::wake() call
  // this; whichever observes the OTHER already arrived returns true and must do
  // the (single) re-injection. Because the worker only calls it AFTER resume()
  // returned, re-injection can never resume a fiber that is still on-stack.
  // Resets to 0 after a completed rendezvous, ready for the next wait episode.
  bool wake_rendezvous() noexcept {
    if (wake_state_.exchange(1, std::memory_order_acq_rel) == 1) {
      wake_state_.store(0, std::memory_order_release);
      return true;  // the other side got here first -> THIS side re-injects
    }
    return false;  // we arrived first -> the other side will re-inject
  }

 private:
  void release_stack_() noexcept;

  void* fctx_ = nullptr;        // fcontext_t: fiber's suspended context
  void* return_ctx_ = nullptr;  // fcontext_t: resumer's context to jump back to
  mem::FiberStack stack_;       // guarded usable region (owns VM)
  mem::FiberStackPool* pool_ = nullptr;  // optional recycle target (non-owning)
  EntryFn entry_ = nullptr;
  void* arg_ = nullptr;
  bool started_ = false;
  bool finished_ = false;
  bool wait_requested_ = false;       // set by request_wait(), read by scheduler
  int owner_worker_ = -1;             // shard-per-core pin (-1 = unpinned/stealable)
  FinishFn on_finish_ = nullptr;      // run by dispatch after done()
  void* on_finish_ctx_ = nullptr;     // ctx for on_finish_ (e.g. the TaskDriver)
  detail::CancelFlagLink* cancel_flags_ = nullptr;  // active cancel sources
  Fiber* ready_next_ = nullptr;       // scheduler cold overflow intrusive link
  const char* name_ = nullptr;        // obs-02: trace label (stable ptr), OFF-build unused
#if defined(HYLE_TRACE) && defined(HYLE_TRACE_TRACY)
  // Trace-only heap storage keeps the default name pointer stable across Fiber
  // moves. It is absent from normal builds, preserving the zero-cost law.
  std::unique_ptr<char[]> default_trace_name_;
#endif
  std::atomic<int> wake_state_{0};    // wake/suspend rendezvous (see wake_rendezvous)
  // ThreadSanitizer fiber handles (null/unused unless built with TSan). fcontext
  // swaps stacks behind TSan's back; without telling TSan which fiber is now
  // running, its per-thread shadow corrupts and SEGVs once fibers migrate across
  // worker threads. tsan_fiber_ is this fiber's handle; tsan_return_ is the
  // resumer's, captured each resume() so yield() can hand control back. See
  // fiber.cc (mirrors boost.context's BOOST_USE_TSAN integration).
  void* tsan_fiber_ = nullptr;
  void* tsan_return_ = nullptr;
};

}  // namespace hyle::sched
