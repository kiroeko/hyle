#pragma once

#include "hyle/compute/storage_owner.h"
#include "hyle/graph/executable.h"

namespace hyle::graph::detail {
struct OutputSourceState;

// One immutable output of one already-submitted original Invocation. This
// pending adapter pins that Invocation. Successful detach preserves its
// original ready output independently, allowing a different producer writer;
// this is not a complete rotating inference pipeline. Control calls, including producer
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
  // Does not wait for a pending producer; idempotent after success. Requires
  // whole-producer success AND actual retirement. Pending/error keeps its pin.
  // Final owner release retains ordinary destructor/allocator cleanup cost.
  // Publishes only the original writer after its last write; already admitted
  // ordered readers may still be running. No payload copy or new execution.
  // The original Invocation retains a ready alias until reset/rebind. Reusing
  // it requires another genuine writer while this source remains immutable.
  HYLE_GRAPH_API Status detach() noexcept;

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
// Prepared nonblocking Commands or complete reuse of an already recorded
// physical tuple. A capture miss never records pending data or falls back to
// managed Commands. Leaf and region binding hooks keep their metadata/resource-
// only contract; they must not observe tensor contents or accept device work.
HYLE_GRAPH_API Status prepare_output(Invocation &, ValueId,
    const OutputSource &, std::span<const QueueBinding>) noexcept;
HYLE_GRAPH_API Result<compute::Completion> submit_output(Invocation &, ValueId,
    const OutputSource &, std::span<const QueueBinding>) noexcept;
} // namespace hyle::graph::detail
