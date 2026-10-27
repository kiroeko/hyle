#pragma once

#include "hyle/compute/storage_owner.h"
#include "hyle/graph/executable.h"

namespace hyle::graph::detail {
struct SubmissionSourceState;
struct RegionAccessView;

// One exact original Graph submission: one proxy, Queue and whole verdict.
// Typed views share this identity, but keep their own actual resource proofs.
// Control calls are externally serialized; source observations use one mutex.
// Requires one original shared producer Queue: nonblocking prepared commands
// or complete actual command capture, retaining the original provider runtime.
class SubmissionSource final {
public:
  SubmissionSource() noexcept = default;
  SubmissionSource(const SubmissionSource &) = delete;
  SubmissionSource &operator=(const SubmissionSource &) = delete;
  HYLE_GRAPH_API SubmissionSource(SubmissionSource &&) noexcept;
  HYLE_GRAPH_API SubmissionSource &operator=(SubmissionSource &&) noexcept;
  HYLE_GRAPH_API ~SubmissionSource();

  // Cold, success-only transfer of the exact unobserved submit proxy. The
  // owner retains the allocator used for this metadata through actual native
  // retirement. Arbitrary Graph metadata keeps its original caller contract.
  HYLE_GRAPH_API static Result<SubmissionSource> create(Invocation &,
      compute::Completion &&, const compute::StorageOwner &metadata_owner) noexcept;
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_GRAPH_API Result<compute::CompletionPoll> poll() noexcept;
  HYLE_GRAPH_API Status wait() noexcept;
  // Forwards to the one original Completion under the observation mutex.
  // A request/capability error neither closes exports nor proves retirement
  // or changes the mathematical verdict returned by poll/wait.
  HYLE_GRAPH_API Status request_cancel() noexcept;
  // Explicitly closes this generation's view export after all views stopped
  // borrowing and the producer actually retired. This returns the control
  // operation's status: success is not mathematical success. poll/wait retain
  // the original whole verdict even after a failed generation is closed.
  // Pending/error observations alone never establish actual retirement.
  // Does not wait for pending work; final owner release has ordinary cleanup cost.
  HYLE_GRAPH_API Status detach() noexcept;
  // A short metadata-only borrow of this exact submitted producer's selected
  // region. No execution or payload access is authorized. The source mutex is
  // not held during the visitor; an adapter may capture concrete tickets there.
  HYLE_GRAPH_API Status visit(NodeId,
      Status (*)(void *, const RegionAccessView &) noexcept, void *) const noexcept;

private:
  friend class OutputSource;
  friend class StoreSource;
  friend class ResourceSource;
  explicit SubmissionSource(SubmissionSourceState *state) noexcept : state_(state) {}
  SubmissionSourceState *state_ = nullptr;
};
} // namespace hyle::graph::detail
