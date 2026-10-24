#pragma once

#include <atomic>
#include <cstdint>

#include "hyle/compute/completion.h"
#include "hyle/compute/detail/completion_access.h"
#include "hyle/mem/allocator.h"

namespace hyle::compute::detail {
struct RetainedCompletionState;

// The original posted driver's one reference, independent of the observation
// root. Exactly one successful handoff is required before destruction, including
// cancellation/posting failures (which hand off their real accepted prefix).
class RetainedCompletionHandoff final {
public:
  RetainedCompletionHandoff() noexcept = default;
  RetainedCompletionHandoff(const RetainedCompletionHandoff &) = delete;
  RetainedCompletionHandoff &operator=(const RetainedCompletionHandoff &) = delete;
  HYLE_COMPUTE_API RetainedCompletionHandoff(RetainedCompletionHandoff &&) noexcept;
  HYLE_COMPUTE_API RetainedCompletionHandoff &operator=(RetainedCompletionHandoff &&) noexcept;
  HYLE_COMPUTE_API ~RetainedCompletionHandoff();
  // Rejection consumes neither this ticket nor completion. Successful claim
  // empties the ticket before invoking any possibly synchronous callback.
  HYLE_COMPUTE_API Status handoff(Completion &&completion) noexcept;
private:
  friend class RetainedCompletion;
  explicit RetainedCompletionHandoff(RetainedCompletionState *state) noexcept : state_(state) {}
  RetainedCompletionState *state_ = nullptr;
};

struct RetainedCompletionPending final {
  Completion completion;
  RetainedCompletionHandoff ticket;
};

// One transferred reference to an existing progress notification owner. Hooks
// touch that owner's event/cancellation flag, not its invocation. It must not
// own this observer (directly or through the invocation). All hooks, including
// release, run outside the observer lock. wait must avoid lost notifications:
// return when revision != observed; notify wakes that same wait mechanism.
// Empty hooks use atomic wait/notify. A nonempty owner requires release, and a
// custom wait requires notify. No hook executes or observes native work.
struct RetainedCompletionControl final {
  void *state = nullptr;
  void (*request_cancel)(void *) noexcept = nullptr;
  void (*wait)(void *, const std::atomic<std::uint64_t> &,
               std::uint64_t observed) noexcept = nullptr;
  void (*notify)(void *) noexcept = nullptr;
  void (*release)(void *) noexcept = nullptr;
};

// One cold observation slot. Public/control observations are externally
// serialized, as with Completion; handoff may run concurrently with them.
// This owns the actual Completion, not another executor or Queue.
class RetainedCompletion final {
public:
  RetainedCompletion() noexcept = default;
  RetainedCompletion(const RetainedCompletion &) = delete;
  RetainedCompletion &operator=(const RetainedCompletion &) = delete;
  HYLE_COMPUTE_API RetainedCompletion(RetainedCompletion &&) noexcept;
  HYLE_COMPUTE_API RetainedCompletion &operator=(RetainedCompletion &&) noexcept;
  HYLE_COMPUTE_API ~RetainedCompletion();
  HYLE_COMPUTE_API static Result<RetainedCompletion> create(mem::AllocatorRef allocator) noexcept;
  HYLE_COMPUTE_API bool available() const noexcept;
  // The caller checks available before accepting native work. A public proxy
  // retains this slot until observation/abandonment; no hot allocation occurs.
  HYLE_COMPUTE_API Completion publish(Completion completion, bool deferred_retirement = false) noexcept;
  // Only expose the returned proxy after the original progress work is posted.
  // Consumes control's owner reference. The returned ticket holds the original
  // driver reference until handoff, independent of this root's reset/lifetime
  // and public proxy abandonment. No hot allocation occurs.
  HYLE_COMPUTE_API RetainedCompletionPending publish_pending(
      RetainedCompletionControl control = {},
      bool deferred_retirement = false) noexcept;
  // A compound adapter reserves this before acceptance and delivers it only
  // when all its accepted prefixes actually retire. A cached root error alone
  // cannot release lifetimes subsequently attached to the public proxy.
  HYLE_COMPUTE_API CompletionLifetime reserve_retirement() noexcept;
  // Observes the same native cell without settling the separate public proxy.
  // An empty/unpublished root reports its cached terminal, as wait does.
  HYLE_COMPUTE_API Result<CompletionPoll> poll() noexcept;
  HYLE_COMPUTE_API Status wait() noexcept;
  // F2 opt-in dependency bridge. Resolves only this known retained owner;
  // arbitrary backend/compound vtables remain opaque. The source owner stays
  // alive and its control/observations are serialized through native use.
  HYLE_COMPUTE_API static Result<CompletionDependency>
  native_dependency(const Completion &source) noexcept;
  HYLE_COMPUTE_API void reset() noexcept;
private:
  explicit RetainedCompletion(RetainedCompletionState *state) noexcept : state_(state) {}
  RetainedCompletionState *state_ = nullptr;
};
} // namespace hyle::compute::detail
