#pragma once

#include "hyle/graph/detail/output_source.h"

namespace hyle::graph::detail {
struct StoreSourceState;

// Fixed consecutive original destination ordinals, not physical slot numbers.
// A nonempty range is selected before the producer submits.
struct StoreDestinationRange final {
  std::size_t first = 0, count = 0;
};

// Independent original Store destination permissions, not a new bank/pool or
// ready frame. The common SubmissionSource owns the only terminal contract.
class StoreSource final {
public:
  StoreSource() noexcept = default;
  StoreSource(const StoreSource &) = delete;
  StoreSource &operator=(const StoreSource &) = delete;
  HYLE_GRAPH_API StoreSource(StoreSource &&) noexcept;
  HYLE_GRAPH_API StoreSource &operator=(StoreSource &&) noexcept;
  HYLE_GRAPH_API ~StoreSource();
  // Before producer submission: reserve fixed metadata using the original
  // budget and an allocator kept alive by the actual StorageOwner. This does
  // not retain/seal destination writers or obstruct producer preparation.
  HYLE_GRAPH_API static Result<StoreSource> prepare(Invocation &, NodeId store,
      const compute::StorageOwner &, ResourceOptions = {}) noexcept;
  // Retain only this consecutive read footprint while still revalidating the
  // complete original Store transaction. Its I32 IDs are a genuine bounded
  // subview of the original ready carrier; no upload or payload copy occurs.
  HYLE_GRAPH_API static Result<StoreSource> prepare(Invocation &, NodeId store,
      const compute::StorageOwner &, StoreDestinationRange, ResourceOptions) noexcept;
  bool prepared() const noexcept { return state_ != nullptr; }
  HYLE_GRAPH_API bool valid() const noexcept;
  // After successful original submit: revalidate the exact Store transaction,
  // controls, coordinates and owner before sealing its destination tickets.
  // No allocation. A late rejection drops unescaped tickets but leaves sealed
  // original writers for ordinary retired take/publication/discard.
  HYLE_GRAPH_API Status capture(const SubmissionSource &) noexcept;
  // Number of destinations in this view, not the original ordinal upper bound.
  HYLE_GRAPH_API std::size_t destination_count() const noexcept;
  // The prepared imported Store supplies its exact original Gather projection.
  // Ready destinations must be the real complete publication of this selected
  // range, in original order. Nonconsecutive selections are not supported.
  HYLE_GRAPH_API Result<OutputBinding> reader_binding(Invocation &, const Graph &,
      NodeId store, std::span<const compute::ReadLease> ready = {}) const noexcept;
  // Only whole-producer success and actual retirement authorize publication.
  // destination remains an original Store ordinal, including for a range
  // view; an unselected ordinal rejects without consuming the writer.
  // The returned original writer must match this exact destination ticket;
  // already admitted readonly successors may still be running.
  HYLE_GRAPH_API Result<compute::ReadLease> publish(std::size_t destination,
      compute::WriteLease &returned_writer) const noexcept;
private:
  friend struct OutputSourceAccess;
  explicit StoreSource(StoreSourceState *state) noexcept : state_(state) {}
  StoreSourceState *state_ = nullptr;
};
} // namespace hyle::graph::detail
