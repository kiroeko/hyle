#pragma once

#include "hyle/compute/completion.h"
#include "hyle/compute/detail/completion_access.h"

namespace hyle::compute {

// Stable, caller-owned attachment for one retained reference. No allocation,
// wait or execution is introduced. The node must remain at its address until
// ReleaseFn runs; the retained owner may contain the node itself.
//
// Thread-compatible: attachment/reuse/destruction require the caller's original
// synchronization. ReleaseFn may run on the runtime's retirement thread. An
// observation error or abandonment does not permit early release or node reuse.
class CompletionRetention final {
public:
  using ReleaseFn = void (*)(void *retained_state) noexcept;

  CompletionRetention() noexcept = default;
  CompletionRetention(const CompletionRetention &) = delete;
  CompletionRetention &operator=(const CompletionRetention &) = delete;
  CompletionRetention(CompletionRetention &&) = delete;
  CompletionRetention &operator=(CompletionRetention &&) = delete;
  // Requires an inactive node; it never cancels, waits or releases active work.
  HYLE_COMPUTE_API ~CompletionRetention();

  // Success consumes exactly one reference represented by retained_state.
  // Failure consumes none and leaves both node and Completion unchanged.
  // The original runtime releases only after actual retirement. An inline
  // Completion (including an error) may release synchronously during attach,
  // potentially destroying this node. Existing attachments are preserved.
  HYLE_COMPUTE_API Status attach(Completion &completion, void *retained_state,
                                 ReleaseFn release) noexcept;

private:
  detail::CompletionRetention retention_{};
};

} // namespace hyle::compute
