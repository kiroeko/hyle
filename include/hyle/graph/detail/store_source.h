#pragma once

#include "hyle/graph/detail/submission_source.h"

namespace hyle::graph::detail {
struct StoreSourceState;

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
  bool prepared() const noexcept { return state_ != nullptr; }
  HYLE_GRAPH_API bool valid() const noexcept;
  // After successful original submit: revalidate the exact Store transaction,
  // controls, coordinates and owner before sealing its destination tickets.
  // No allocation. A late rejection drops unescaped tickets but leaves sealed
  // original writers for ordinary retired take/publication/discard.
  HYLE_GRAPH_API Status capture(const SubmissionSource &) noexcept;
  HYLE_GRAPH_API std::size_t destination_count() const noexcept;
  // Only whole-producer success and actual retirement authorize publication.
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
