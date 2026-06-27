#pragma once

#include <cstddef>

#include "hyle/macros.h"
#include "hyle/mem/allocator.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_SCHED_API
#elif defined(HYLE_SCHED_BUILD)
#  define HYLE_SCHED_API HYLE_EXPORT
#else
#  define HYLE_SCHED_API HYLE_IMPORT
#endif

namespace hyle::sched {

// Default fiber stack size. 64 KiB suits the common (io-ish) fiber: deep enough
// for ordinary call chains, small enough to keep millions affordable. sched-
// design §1 calls for *tiered* sizes later (small io stacks / large compute
// stacks); until that lands this is the one documented default, not a literal
// sprinkled at call sites.
inline constexpr std::size_t kDefaultStackSize = 64 * 1024;

// Stackful fiber: a cooperative unit of execution with its own stack, able to
// suspend (yield) at ANY call depth and be resumed later. Built on the vendored
// fcontext assembly (see src/sched/fcontext.h). This is the scheduling unit the
// rest of sched/ will manage; sched-01 only proves the switch works.
//
// Usage:
//   Fiber f(alloc, 64*1024, &my_entry, arg);
//   f.resume();            // runs my_entry until it yields or returns
//   ... Fiber::yield() ... // called from inside my_entry, returns to resume()
//   f.resume();            // continues after the yield
//   f.done();              // true once my_entry has returned
//
// Single-thread cooperative model: a fiber is resumed by exactly one thread at a
// time. Move-only ownership of the stack.
//
// API macro is on the methods, not the class: exporting the whole class would
// drag the mem::Allocator member across the seam (MSVC C4251). Method-level
// export keeps the layout client-side and clean (same pattern as Parker).
class Fiber {
 public:
  using EntryFn = void (*)(void* arg);

  // Allocates a `stack_size`-byte stack from `alloc` and prepares `entry` to run
  // on it. Does not start running until the first resume().
  HYLE_SCHED_API Fiber(mem::Allocator alloc, std::size_t stack_size,
                       EntryFn entry, void* arg);
  HYLE_SCHED_API ~Fiber();

  HYLE_SCHED_API Fiber(Fiber&& other) noexcept;
  HYLE_SCHED_API Fiber& operator=(Fiber&& other) noexcept;
  HYLE_DISABLE_COPY(Fiber);

  // Enter (or continue) the fiber. Returns when the fiber yields or finishes.
  // Precondition: !done().
  HYLE_SCHED_API void resume();

  // True once the entry function has returned (fiber is spent).
  bool done() const noexcept { return finished_; }

  // Suspend the currently-running fiber, returning control to its resumer.
  // Must be called from within a fiber's call stack (asserts otherwise).
  HYLE_SCHED_API static void yield() noexcept;

 private:
  // friend so the file-local fcontext trampoline can drive private state without
  // exposing transfer_t/fcontext_t in this public header.
  friend struct FiberLaunch;

  void* fctx_ = nullptr;        // fcontext_t: fiber's suspended context
  void* return_ctx_ = nullptr;  // fcontext_t: resumer's context to jump back to
  void* stack_ = nullptr;       // low address of the allocated stack
  std::size_t stack_size_ = 0;
  mem::Allocator alloc_;
  EntryFn entry_ = nullptr;
  void* arg_ = nullptr;
  bool finished_ = false;
};

}  // namespace hyle::sched
