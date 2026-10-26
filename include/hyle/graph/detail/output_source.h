#pragma once

#include "hyle/compute/storage_owner.h"
#include "hyle/graph/executable.h"

namespace hyle::graph::detail {
struct OutputSourceState;

// One immutable output of one already-submitted original Invocation. This
// first adapter pins that Invocation through all source/consumer references;
// it is not a rotating execution pipeline. Control calls, including producer
// drain, are externally serialized. Native callbacks and source observations
// share the source's ordinary mutex.
class OutputSource final {
public:
  OutputSource() noexcept = default;
  OutputSource(const OutputSource &) = delete;
  OutputSource &operator=(const OutputSource &) = delete;
  HYLE_GRAPH_API OutputSource(OutputSource &&) noexcept;
  HYLE_GRAPH_API OutputSource &operator=(OutputSource &&) noexcept;
  HYLE_GRAPH_API ~OutputSource();

  // Cold, success-only transfer of the exact unobserved submit proxy. Only a fixed,
  // caller-owned dense I32 output with its original full writer qualifies.
  // Retains the actual resource/allocator owner; never mints a ready lease.
  // The original pool proves its domain, not access to its other bytes.
  // The original compiled StorageDomain must also retain this resource owner,
  // through the producer's eventual output/allocation destruction. External
  // pool roots and arbitrary Graph metadata keep their original contracts.
  HYLE_GRAPH_API static Result<OutputSource> create(Invocation &producer,
      ValueId output, compute::Completion &&submission,
      const compute::StorageOwner &, const compute::LeasePool &) noexcept;
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_GRAPH_API Result<compute::CompletionPoll> poll() noexcept;
  HYLE_GRAPH_API Status wait() noexcept;

private:
  friend HYLE_GRAPH_API Status bind_output(Invocation &, ValueId,
      const OutputSource &) noexcept;
  friend HYLE_GRAPH_API Status prepare_output(Invocation &, ValueId,
      const OutputSource &, std::span<const QueueBinding>) noexcept;
  friend HYLE_GRAPH_API Result<compute::Completion> submit_output(Invocation &,
      ValueId, const OutputSource &, std::span<const QueueBinding>) noexcept;
  explicit OutputSource(OutputSourceState *state) noexcept : state_(state) {}
  OutputSourceState *state_ = nullptr;
};

// One original consumer/input identity per source. Rejection preserves the
// ordinary bindings. Pending bytes are never exposed to the caller.
HYLE_GRAPH_API Status bind_output(Invocation &, ValueId,
    const OutputSource &) noexcept;
// Commands only: capture could execute user math on a pending input. Leaf
// preparation has its existing metadata-only contract. This first adapter
// rejects regions with binding callbacks rather than guessing their behavior.
HYLE_GRAPH_API Status prepare_output(Invocation &, ValueId,
    const OutputSource &, std::span<const QueueBinding>) noexcept;
HYLE_GRAPH_API Result<compute::Completion> submit_output(Invocation &, ValueId,
    const OutputSource &, std::span<const QueueBinding>) noexcept;
} // namespace hyle::graph::detail
