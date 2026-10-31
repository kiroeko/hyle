#pragma once

#include "hyle/graph/executable.h"

namespace hyle::graph { class SourceBinding; }

namespace hyle::graph::detail {

struct ReplayInputBinding final {
  ValueId input{};
  compute::ConstTensorView tensor{};
};

// Cold Graph adapters may reuse an already recorded physical tuple. These
// queries grant neither input readiness nor permission to record pending data.
struct ReplayAccess final {
  // Necessary coverage only for one original plain OutputSource intent. Read
  // its fixed input coordinates without binding, observing completion or
  // acquiring permission. Every range must have one qualified retained
  // recording matching all coordinates of this input; other ports remain
  // unproved. False means missing coverage, not authorization to record.
  // The final complete-tuple and Source admission checks remain mandatory.
  // Source and consumer control follow the same serialization as bind_sources.
  HYLE_GRAPH_API static Result<bool> has_capture_candidate(const Invocation &,
      const SourceBinding &) noexcept;
  // Original Source replay preparation with a typed pending-capture miss.
  // False only means that its final physical tuple has no retained coverage;
  // other control/native/permission failures remain errors. Ready sources
  // retain the ordinary preparation path, which may record new commands.
  HYLE_GRAPH_API static Result<bool> try_prepare_sources(Invocation &,
      std::span<const QueueBinding>) noexcept;
  // Pure inspection of existing records, their actual Queue/backend/private
  // arguments, and the final tuple with these external inputs replaced together.
  // Captured ranges must cover every managed/blocking command. Remaining gaps
  // may only contain prepared, nonblocking shared-Queue commands. A prepared
  // full-span, zero-offset View may keep its exact original physical storage;
  // changed/partial/unknown aliases or opaque captured carriers return false.
  // No binding, provider
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

private:
  static Result<bool> candidate_with_input(const Invocation &,
      const ReplayInputBinding &) noexcept;
};

} // namespace hyle::graph::detail
