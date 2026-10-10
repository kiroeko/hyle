#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/graph/executable.h"

namespace hyle::graph::detail {

struct InvocationState;

// Same-build coarse submission cursor. One admitted run retains its original
// Invocation and fixed segment storage. A native composition owner drives it;
// no scheduler or new per-kernel interface enters ordinary Graph objects.
class SubmissionCursor final {
public:
  SubmissionCursor() noexcept = default;
  SubmissionCursor(const SubmissionCursor &) = delete;
  SubmissionCursor &operator=(const SubmissionCursor &) = delete;
  HYLE_GRAPH_API SubmissionCursor(SubmissionCursor &&other) noexcept;
  HYLE_GRAPH_API SubmissionCursor &operator=(SubmissionCursor &&other) noexcept;
  HYLE_GRAPH_API ~SubmissionCursor();

  HYLE_GRAPH_API static Result<SubmissionCursor>
  begin(Invocation &invocation, std::span<const QueueBinding> queues,
        bool retain_invocation = true) noexcept;
  // false means an explicit host dependency remains pending. advance never
  // waits for that dependency. A CPU/native managed region may itself perform
  // its declared synchronous work, so the caller must use the right endpoint.
  HYLE_GRAPH_API Result<bool> advance(bool wait_dependencies = false) noexcept;
  HYLE_GRAPH_API bool done() const noexcept;
  HYLE_GRAPH_API ExecutionTargetId next_target() const noexcept;
  const compute::Completion *pending_dependency() const noexcept {
    return pending_;
  }
  HYLE_GRAPH_API compute::Queue *pending_queue() const noexcept;
  HYLE_GRAPH_API void request_cancel() noexcept;
  // Stop unaccepted successors while preserving the actual failing endpoint.
  HYLE_GRAPH_API void reject_successors(Status terminal) noexcept;
  // Emits the actual final accepted prefix, including tail failure. This
  // retains all native owners even when its public handle is abandoned.
  HYLE_GRAPH_API Result<compute::Completion> finish() noexcept;

private:
  SubmissionCursor(InvocationState *state, bool retained) noexcept;
  void reset() noexcept;
  InvocationState *state_ = nullptr;
  std::size_t next_segment_ = 0;
  Status tail_error_{};
  const compute::Completion *pending_ = nullptr;
  std::size_t pending_segment_ = SIZE_MAX;
  bool finished_ = false;
  bool retained_ = false;
};

} // namespace hyle::graph::detail
