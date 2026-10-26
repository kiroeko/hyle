#pragma once

#include "hyle/graph/executable.h"

namespace hyle::graph::detail {

// Cold Graph adapters may reuse an already recorded physical tuple. These
// queries grant neither input readiness nor permission to record pending data.
struct ReplayAccess final {
  // Pure inspection of existing records, their actual Queue/backend/private
  // arguments, and the current tuple with this one external input replaced.
  // Unsupported aliases or opaque native carriers return false. No binding,
  // provider preparation, resource allocation or recording is performed.
  HYLE_GRAPH_API static Result<bool> can_reuse_capture(const Invocation &,
      ValueId replacement_input, compute::ConstTensorView replacement) noexcept;
  // The ordinary replay preparation, restricted to existing complete capture
  // coverage. All ranges must match before any artifact is selected; a miss
  // rejects before capture begins and never falls back to ordinary Commands.
  HYLE_GRAPH_API static Status prepare_reuse(Invocation &,
      std::span<const QueueBinding>) noexcept;
};

} // namespace hyle::graph::detail
