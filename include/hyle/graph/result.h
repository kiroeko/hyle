#pragma once

#include <cstdint>

#include "hyle/compute/resource_leases.h"
#include "hyle/compute/tensor.h"
#include "hyle/graph/visibility.h"

namespace hyle::graph {
namespace detail { struct InvocationState; }
class Invocation;

// A readonly result retaining its original execution slot and owned backing.
// Copies prevent reuse of that slot. Raw external TensorView storage remains
// borrowed and must outlive its consumers; exporting an owned storage lease
// independently protects that backing against writer reuse.
class GraphOutput final {
public:
  GraphOutput() noexcept = default;
  HYLE_GRAPH_API GraphOutput(const GraphOutput &) noexcept;
  HYLE_GRAPH_API GraphOutput &operator=(const GraphOutput &) noexcept;
  HYLE_GRAPH_API GraphOutput(GraphOutput &&) noexcept;
  HYLE_GRAPH_API GraphOutput &operator=(GraphOutput &&) noexcept;
  HYLE_GRAPH_API ~GraphOutput();
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_GRAPH_API Result<compute::TensorDesc> description() const noexcept;
  // A memory view is available only when the actual native result has one.
  HYLE_GRAPH_API Result<compute::ConstTensorView> tensor() const noexcept;
  // Exports an actual external/native storage owner. Private workspace and
  // constants cannot become independent external leases.
  HYLE_GRAPH_API Result<compute::ReadLease> storage_lease() const noexcept;
  // Optional admission for the original full retained backing. Shared views
  // are charged once where the actual resource proves identity; prior aliases
  // and native uses keep that charge alive. This is completed-result retention,
  // not a reservation of future Provider allocations or a device-memory cap.
  // Failure leaves the original result/permissions intact.
  HYLE_GRAPH_API Result<compute::ReadLease>
  storage_lease(const compute::ResourceBudget &retention_budget) const noexcept;
  HYLE_GRAPH_API void reset() noexcept;

private:
  friend class Invocation;
  GraphOutput(detail::InvocationState *, std::uint32_t instance) noexcept;
  detail::InvocationState *state_ = nullptr;
  std::uint32_t instance_ = UINT32_MAX;
};

} // namespace hyle::graph
