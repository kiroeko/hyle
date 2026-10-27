#pragma once

#include "hyle/graph/detail/submission_source.h"
#include "hyle/graph/detail/region_access.h"

namespace hyle::graph::detail {
struct OutputSourceState;
struct OutputBinding;
class StoreSource;

// One fixed caller-owned dense I32 output of a SubmissionSource. Its real
// storage owner and original permission are retained independently of other
// values; it never exposes pending bytes or creates a second observer.
class OutputSource final {
public:
  OutputSource() noexcept = default;
  OutputSource(const OutputSource &) = delete;
  OutputSource &operator=(const OutputSource &) = delete;
  HYLE_GRAPH_API OutputSource(OutputSource &&) noexcept;
  HYLE_GRAPH_API OutputSource &operator=(OutputSource &&) noexcept;
  HYLE_GRAPH_API ~OutputSource();

  // The original compiled StorageDomain must retain this same actual resource
  // owner. The pool proves this full writer's domain, not its other bytes.
  HYLE_GRAPH_API static Result<OutputSource> create(const SubmissionSource &,
      ValueId output, const compute::StorageOwner &, const compute::LeasePool &) noexcept;
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_GRAPH_API Result<compute::CompletionPoll> poll() noexcept;
  HYLE_GRAPH_API Status wait() noexcept;
  // Success AND actual producer retirement publish the original read lease.
  // Already admitted ordered readers may remain in flight. No payload copy.
  // Pending/error retains this view's borrow. Closing the common source and
  // returning its producer requires explicit SubmissionSource::detach().
  HYLE_GRAPH_API Status detach() noexcept;

private:
  friend struct OutputSourceAccess;
  friend HYLE_GRAPH_API Status bind_outputs(Invocation &, std::span<const OutputBinding>) noexcept;
  friend HYLE_GRAPH_API Status prepare_outputs(Invocation &, std::span<const OutputBinding>,
      std::span<const QueueBinding>) noexcept;
  friend HYLE_GRAPH_API Result<compute::Completion> submit_outputs(Invocation &,
      std::span<const OutputBinding>, std::span<const QueueBinding>) noexcept;
  explicit OutputSource(OutputSourceState *state) noexcept : state_(state) {}
  OutputSourceState *state_ = nullptr;
};

struct OutputBinding final {
  ValueId input{};
  const OutputSource *output = nullptr;
  const StoreSource *store = nullptr;
  ResourceProjection projection{};
  const Graph *graph = nullptr;
};

// Cold atomic bundle of distinct inputs from the same submission.
// Success retains all views, even when InputSources deduplicates their common
// dependency. A view binds one original consumer bundle once; the submission itself
// has no consumer restriction. Failure preserves bindings and view identities.
// Exactly one concrete view is supplied. Store entries derive their capacity
// input and bounded ready-ID input from projection; input must remain empty.
// Store graph is borrowed for this cold call to prove the actual checked
// Gather input/ID association, then discarded; no provider-name inference.
// Subsequent calls supply this same ordered bundle; the span is never retained.
HYLE_GRAPH_API Status bind_outputs(Invocation &, std::span<const OutputBinding>) noexcept;
// Prepared nonblocking Commands or complete existing capture reuse. The full
// final tuple is checked together; a miss never records pending values or
// falls back to managed Commands. Original metadata-only binding hooks apply.
HYLE_GRAPH_API Status prepare_outputs(Invocation &, std::span<const OutputBinding>,
    std::span<const QueueBinding>) noexcept;
HYLE_GRAPH_API Result<compute::Completion> submit_outputs(Invocation &,
    std::span<const OutputBinding>, std::span<const QueueBinding>) noexcept;
} // namespace hyle::graph::detail
