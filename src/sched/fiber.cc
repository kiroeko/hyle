#include "hyle/sched/fiber.h"

#include <utility>

#include "fcontext.h"
#include "hyle/macros.h"

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
    // Final switch back to the resumer; control never returns here.
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
}

Fiber::~Fiber() {
  if (stack_ != nullptr) {
    alloc_.deallocate(stack_, stack_size_, kStackAlign);
  }
}

Fiber::Fiber(Fiber&& o) noexcept
    : fctx_(o.fctx_),
      return_ctx_(o.return_ctx_),
      stack_(o.stack_),
      stack_size_(o.stack_size_),
      alloc_(o.alloc_),
      entry_(o.entry_),
      arg_(o.arg_),
      finished_(o.finished_) {
  o.stack_ = nullptr;
  o.fctx_ = nullptr;
}

Fiber& Fiber::operator=(Fiber&& o) noexcept {
  if (this != &o) {
    // Overwriting a live stack would leak it; a half-run fiber can't be dropped
    // mid-flight safely -> fail-fast (mirrors Thread's joinable assert).
    HYLE_FATAL_ASSERT(stack_ == nullptr);
    fctx_ = o.fctx_;
    return_ctx_ = o.return_ctx_;
    stack_ = o.stack_;
    stack_size_ = o.stack_size_;
    alloc_ = o.alloc_;
    entry_ = o.entry_;
    arg_ = o.arg_;
    finished_ = o.finished_;
    o.stack_ = nullptr;
    o.fctx_ = nullptr;
  }
  return *this;
}

void Fiber::resume() {
  HYLE_FATAL_ASSERT(!finished_);
  Fiber* prev = t_current;
  t_current = this;  // this fiber is now the running one
  // Switch into the fiber, handing it `this`. We come back here when it yields
  // (or finishes); t.fctx is the fiber's freshly-suspended context.
  transfer_t t = jump_fcontext(fctx_, this);
  t_current = prev;
  if (!finished_) fctx_ = t.fctx;
}

void Fiber::yield() noexcept {
  Fiber* self = t_current;
  HYLE_FATAL_ASSERT(self != nullptr);  // yield() called outside any fiber
  // Switch back to the resumer. When resumed again, jump_fcontext returns here
  // with t.fctx = the (possibly new) resumer context to return to next time.
  transfer_t t = jump_fcontext(self->return_ctx_, nullptr);
  self->return_ctx_ = t.fctx;
}

Fiber* Fiber::current() noexcept { return t_current; }

}  // namespace hyle::sched
