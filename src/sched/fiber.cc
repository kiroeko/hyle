#include "hyle/sched/fiber.h"

#include <utility>

#include "fcontext.h"
#include "hyle/macros.h"
#include "hyle/obs/trace.h"  // obs-02: HYLE_TRACE_FIBER_* (compile-time gated, OFF=0 code)

// ---- ThreadSanitizer fiber integration --------------------------------------
// fcontext swaps the stack pointer invisibly to TSan. TSan keeps shadow state
// per logical context and assumes one known context per OS thread; once a fiber
// suspended on worker A resumes on worker B (work-stealing migration), TSan is
// applying A's stack bounds to code running under B and eventually SEGVs inside
// its own runtime. The fix (mandated by TSan, used by boost.context/folly) is to
// announce each switch via __tsan_switch_to_fiber. Compiles to nothing unless
// built with -fsanitize=thread, so the shipped/benchmarked build pays ZERO.
#if defined(__has_feature)
#  if __has_feature(thread_sanitizer)
#    define HYLE_TSAN 1
#  endif
#elif defined(__SANITIZE_THREAD__)
#  define HYLE_TSAN 1
#endif

#if defined(HYLE_TSAN)
#  include <sanitizer/tsan_interface.h>
#  define HYLE_TSAN_CREATE_FIBER() __tsan_create_fiber(0)
#  define HYLE_TSAN_DESTROY_FIBER(f) \
    do {                             \
      if (f) __tsan_destroy_fiber(f); \
    } while (0)
#  define HYLE_TSAN_CURRENT_FIBER() __tsan_get_current_fiber()
#  define HYLE_TSAN_SWITCH(f) __tsan_switch_to_fiber((f), 0)
#else
#  define HYLE_TSAN_CREATE_FIBER() nullptr
#  define HYLE_TSAN_DESTROY_FIBER(f) ((void)(f))
#  define HYLE_TSAN_CURRENT_FIBER() nullptr
#  define HYLE_TSAN_SWITCH(f) ((void)(f))
#endif

namespace hyle::sched {
namespace {

// The fiber currently running on this OS thread (so yield() can find itself).
// nullptr when no fiber is executing (we're on the resumer's stack).
thread_local Fiber* t_current = nullptr;

// Stacks are cache-line aligned (single source: HYLE_CACHE_LINE_SIZE, which also
// comfortably satisfies the 16-byte x64 stack ABI). A stack smaller than one OS
// page can't hold a guard page later and is almost certainly a caller mistake.
constexpr std::size_t kStackAlign = HYLE_CACHE_LINE_SIZE;
constexpr std::size_t kMinStackSize = 4096;  // one typical OS page

}  // namespace

// Bridges the C-ABI trampoline (which speaks transfer_t) to Fiber's private
// state. A struct so it can be a friend of Fiber without leaking fcontext types
// into the public header.
struct FiberLaunch {
  // Entry handed to make_fcontext. On first resume it receives the resumer's
  // context as t.fctx and `this` Fiber as t.data.
  static void trampoline(transfer_t t) noexcept {
    auto* self = static_cast<Fiber*>(t.data);
    self->return_ctx_ = t.fctx;  // where to jump back when we yield/finish
    self->entry_(self->arg_);    // run user code (may yield repeatedly)
    self->finished_ = true;
    // Hand TSan back to the resumer before the final stack switch (control never
    // returns here, so there is no finish-switch on this side).
    HYLE_TSAN_SWITCH(self->tsan_return_);
    jump_fcontext(self->return_ctx_, nullptr);
  }
};

Fiber::Fiber(mem::Allocator alloc, std::size_t stack_size, EntryFn entry,
             void* arg)
    : stack_size_(stack_size), alloc_(alloc), entry_(entry), arg_(arg) {
  HYLE_FATAL_ASSERT(alloc_.valid());
  HYLE_FATAL_ASSERT(entry_ != nullptr);
  HYLE_FATAL_ASSERT(stack_size_ >= kMinStackSize);
  stack_ = alloc_.allocate(stack_size_, kStackAlign);
  HYLE_FATAL_ASSERT(stack_ != nullptr);
  // Stack grows down: make_fcontext wants the TOP (highest address).
  void* top = static_cast<char*>(stack_) + stack_size_;
  fctx_ = make_fcontext(top, stack_size_, &FiberLaunch::trampoline);
  tsan_fiber_ = HYLE_TSAN_CREATE_FIBER();  // null in non-TSan builds
}

Fiber::~Fiber() {
  if (stack_ != nullptr) {
    alloc_.deallocate(stack_, stack_size_, kStackAlign);
  }
  HYLE_TSAN_DESTROY_FIBER(tsan_fiber_);
}

Fiber::Fiber(Fiber&& o) noexcept
    : fctx_(o.fctx_),
      return_ctx_(o.return_ctx_),
      stack_(o.stack_),
      stack_size_(o.stack_size_),
      alloc_(o.alloc_),
      entry_(o.entry_),
      arg_(o.arg_),
      finished_(o.finished_),
      tsan_fiber_(o.tsan_fiber_) {
  o.stack_ = nullptr;
  o.fctx_ = nullptr;
  o.tsan_fiber_ = nullptr;  // moved-from must not destroy the handle
}

Fiber& Fiber::operator=(Fiber&& o) noexcept {
  if (this != &o) {
    // Overwriting a live stack would leak it; a half-run fiber can't be dropped
    // mid-flight safely -> fail-fast (mirrors Thread's joinable assert).
    HYLE_FATAL_ASSERT(stack_ == nullptr);
    HYLE_TSAN_DESTROY_FIBER(tsan_fiber_);  // null on a moved-from target; safe
    fctx_ = o.fctx_;
    return_ctx_ = o.return_ctx_;
    stack_ = o.stack_;
    stack_size_ = o.stack_size_;
    alloc_ = o.alloc_;
    entry_ = o.entry_;
    arg_ = o.arg_;
    finished_ = o.finished_;
    tsan_fiber_ = o.tsan_fiber_;
    o.stack_ = nullptr;
    o.fctx_ = nullptr;
    o.tsan_fiber_ = nullptr;
  }
  return *this;
}

void Fiber::resume() {
  HYLE_FATAL_ASSERT(!finished_);
  Fiber* prev = t_current;
  t_current = this;  // this fiber is now the running one
  // Tell TSan we are leaving the resumer (worker) context for this fiber's. The
  // resumer handle is captured EACH resume so a fiber that migrated to another
  // worker hands control back to the right one on yield. Both are no-ops unless
  // built with TSan.
  tsan_return_ = HYLE_TSAN_CURRENT_FIBER();
  HYLE_TSAN_SWITCH(tsan_fiber_);
  // obs-02: mark that this OS thread is now running THIS fiber, so the profiler
  // attributes the work to the fiber track rather than the worker thread (an M:N
  // scheduler needs this). Compile-time gated -- expands to nothing (no read of
  // name_, no call) unless built with HYLE_TRACE. `name_` must be a stable pointer;
  // absent one, a shared literal groups anonymous fibers under one track.
  HYLE_TRACE_FIBER_ENTER(name_ != nullptr ? name_ : "hyle.fiber");
  // Switch into the fiber, handing it `this`. We come back here when it yields
  // (or finishes); t.fctx is the fiber's freshly-suspended context. yield()/the
  // trampoline switch TSan back to tsan_return_ before jumping here, so TSan is
  // already on the resumer context on return -- nothing to do.
  transfer_t t = jump_fcontext(fctx_, this);
  // Back on the resumer (worker) stack: the fiber yielded or finished. Leave the
  // fiber track so subsequent worker-thread work isn't misattributed to it.
  HYLE_TRACE_FIBER_LEAVE();
  t_current = prev;
  if (!finished_) fctx_ = t.fctx;
}

void Fiber::yield() noexcept {
  Fiber* self = t_current;
  HYLE_FATAL_ASSERT(self != nullptr);  // yield() called outside any fiber
  // Hand TSan back to the resumer before switching stacks. On the next resume(),
  // resume() switches TSan back to this fiber before its jump, so when we return
  // below TSan is already on this fiber.
  HYLE_TSAN_SWITCH(self->tsan_return_);
  // Switch back to the resumer. When resumed again, jump_fcontext returns here
  // with t.fctx = the (possibly new) resumer context to return to next time.
  transfer_t t = jump_fcontext(self->return_ctx_, nullptr);
  self->return_ctx_ = t.fctx;
}

Fiber* Fiber::current() noexcept { return t_current; }

}  // namespace hyle::sched
