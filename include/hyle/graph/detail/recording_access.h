#pragma once

#include "hyle/graph/detail/region_access.h"

namespace hyle::graph::detail {

// Borrowed physical arguments for cold command recording. These views carry
// no ready/read/write permission and can never authorize a Graph submission.
struct RecordingStoreInputs final {
  compute::ConstTensorView capacity{}, indices{};
};

// Internal cold adapter only. The original Invocation keeps its ordinary
// resource/control gates, Store/Append owners and native capture machinery.
// While active, every submit and Source-export route rejects. The caller must
// recover its real transactions before end() withdraws the borrowed inputs;
// retained physical recordings then remain dormant until fresh preparation.
struct RecordingAccess final {
  HYLE_GRAPH_API static bool active(const Invocation &) noexcept;
  HYLE_GRAPH_API static Status begin(Invocation &) noexcept;
  HYLE_GRAPH_API static Status bind_input(Invocation &, ValueId,
      compute::ConstTensorView) noexcept;
  HYLE_GRAPH_API static Status bind_projection(Invocation &,
      const ResourceProjection &, RecordingStoreInputs) noexcept;
  HYLE_GRAPH_API static Status end(Invocation &) noexcept;

  // Borrow only the physical span of an existing caller output while the
  // original invocation is recording-only. This grants no lease permission.
  HYLE_GRAPH_API static Result<compute::ConstTensorView> output_view(
      Invocation &, const compute::LeasePool &, compute::TensorDesc) noexcept;
  // A separately prepared ordinary invocation may capture a complete ready
  // input before its original writer is reused by a predecessor. Verify the
  // actual current binding, withdraw it through Graph, and reclaim that same
  // full lease generation/span. No wait, publication, Source or owner is added.
  HYLE_GRAPH_API static Result<compute::WriteLease> reclaim_input_writer(
      Invocation &, ValueId, compute::TensorDesc, compute::ReadLease &) noexcept;

  // Implemented by the original Store adapter, using its exact selected slot.
  HYLE_GRAPH_API static Result<RecordingStoreInputs> store_output(
      Invocation &, NodeId) noexcept;
  HYLE_GRAPH_API static Status bind_store(Invocation &, NodeId,
      const RecordingStoreInputs &) noexcept;
};

} // namespace hyle::graph::detail
