#pragma once

#include "hyle/compute/completion.h"

namespace hyle::compute::detail {

// Internal lifetime attachment carried by an asynchronous Completion. The
// attachment owns one reference supplied by the caller. A backend settles it
// only after submitted work reaches a terminal state, even when the public
// Completion handle is abandoned without waiting.
struct CompletionLifetime final {
  using TerminalFn = void (*)(void *state, const Status &terminal) noexcept;
  using ReleaseFn = void (*)(void *state) noexcept;

  void *state = nullptr;
  TerminalFn terminal = nullptr;
  ReleaseFn release = nullptr;

  constexpr bool valid() const noexcept {
    return state != nullptr && terminal != nullptr && release != nullptr;
  }
};

// Borrowed, backend-private identity for one still-pending Completion. The
// source handle must outlive every Queue prefix that records a wait against
// this view. Backends compare the exact vtable before interpreting `state`;
// no vendor event type enters the common Compute headers.
struct CompletionDependency final {
  const CompletionVTable *vtable = nullptr;
  void *state = nullptr;
  const Status *terminal_hint = nullptr;

  constexpr bool pending() const noexcept {
    return vtable != nullptr && state != nullptr;
  }
};

// Cross-factor bridge for attaching an execution owner to a provider-created
// Completion.  This remains an implementation contract; ordinary users only
// consume Completion::poll/wait/request_cancel.
struct CompletionAccess final {
  static HYLE_COMPUTE_API Completion make(const CompletionVTable *vtable,
                                          void *state) noexcept;
  // True only while the handle still owns backend completion state. Inline
  // terminal completions need no execution-owner handoff.
  static bool has_async_state(const Completion &completion) noexcept {
    return completion.valid_ && completion.vtable_ != nullptr;
  }
  // Consumes one caller-owned reference in `lifetime` on every return path.
  // Its storage must be part of that retained owner and remain valid until the
  // release callback runs; Completion stores only this pointer on its async
  // path rather than adding three callback words to every public handle.
  static HYLE_COMPUTE_API Status attach_lifetime(
      Completion &completion, CompletionLifetime &lifetime) noexcept;
  // Preserve an already-observed backend failure; otherwise replace a future
  // successful terminal state with `terminal`.  Used when an ordered prefix
  // was accepted before a later sequence step was synchronously rejected.
  static HYLE_COMPUTE_API void append_terminal(Completion &completion,
                                               Status terminal) noexcept;
  static HYLE_COMPUTE_API CompletionDependency
  dependency(const Completion &completion) noexcept;
};

} // namespace hyle::compute::detail
