#pragma once

#include "hyle/graph/executable.h"

namespace hyle::graph::detail {

struct ReplayInputBinding final {
  ValueId input{};
  compute::ConstTensorView tensor{};
};

// Cold Graph adapters may reuse an already recorded physical tuple. These
// queries grant neither input readiness nor permission to record pending data.
struct ReplayAccess final {
  // Pure inspection of existing records, their actual Queue/backend/private
  // arguments, and the final tuple with these external inputs replaced together.
  // Captured ranges must cover every managed/blocking command. Remaining gaps
  // may only contain prepared, nonblocking shared-Queue commands. Unsupported
  // aliases or opaque captured carriers return false. No binding, provider
  // preparation, resource allocation or recording is performed.
  HYLE_GRAPH_API static Result<bool> can_reuse_capture(const Invocation &,
      std::span<const ReplayInputBinding>) noexcept;
  // The ordinary replay preparation, restricted to existing recorded ranges
  // and qualified prepared gaps (Captured or Hybrid). All ranges must match
  // before any artifact is selected; a miss rejects before capture begins and
  // never records or falls back to ordinary Commands. Public RequireCapture
  // retains its independent requirement that every command be captured.
  HYLE_GRAPH_API static Status prepare_reuse(Invocation &,
      std::span<const QueueBinding>) noexcept;
};

} // namespace hyle::graph::detail
