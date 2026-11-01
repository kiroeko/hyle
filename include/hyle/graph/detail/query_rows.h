#pragma once

#include <cstdint>

#include "hyle/graph/region_provider.h"

namespace hyle::graph::batch::detail { struct QueryRowsAccess; }
namespace hyle::graph::detail {
struct InvocationState;

// Cold facts from one original, currently prepared Batch slot. The host row
// count and device offsets belong to that same owned layout/preparation.
// These borrowed coordinates grant no tensor or workspace permission.
struct QueryRowsFacts final {
  ValueId offsets_input;
  compute::ConstTensorView offsets;
  std::uint32_t rows = 0, capacity = 0;
  std::uint64_t layout_generation = 0, preparation_epoch = 0;
};

// Only the original Batch adapter can construct this capability. It borrows
// a stable slot in the SAME Invocation; it never retains another ReadLease.
// The original Graph configuration binding owns the actual offsets guard.
// inspect() is a cold/seal query, not a run callback: it rechecks the current
// producer token, layout, Queue and exact bound view before returning facts.
// Reset clears every configured consumer before the producer may be prepared
// again. Consumers must not export this handle or keep it beyond their slot.
class QueryRowsControl final {
 public:
  QueryRowsControl() noexcept = default;
  bool valid() const noexcept { return source_ && inspect_; }
  Result<QueryRowsFacts> inspect() const noexcept {
    if (!valid())
      return std::unexpected(Status::FailedPrecondition("query rows control is unbound"));
    return inspect_(source_, invocation_, epoch_, generation_, input_, offsets_);
  }

 private:
  friend struct batch::detail::QueryRowsAccess;
  using Inspect = Result<QueryRowsFacts> (*)(const void *, const InvocationState *,
      std::uint64_t, std::uint64_t, ValueId, compute::ConstTensorView) noexcept;
  QueryRowsControl(const void *source, const InvocationState *invocation,
      Inspect inspect, std::uint64_t epoch, std::uint64_t generation,
      ValueId input, compute::ConstTensorView offsets) noexcept
      : source_(source), invocation_(invocation), inspect_(inspect), epoch_(epoch),
        generation_(generation), input_(input), offsets_(offsets) {}
  const void *source_ = nullptr;
  const InvocationState *invocation_ = nullptr;
  Inspect inspect_ = nullptr;
  std::uint64_t epoch_ = 0, generation_ = 0;
  ValueId input_;
  compute::ConstTensorView offsets_;
};

// New private, versioned opt-in; the existing prepared-vtable ABI is intact.
// A selected artifact recognizes its compile-time row domain by the original
// Graph offsets ValueId. Matching a numeric capacity is not sufficient.
inline constexpr compute::ProviderExtensionId kRegionQueryRowsExtensionId{
    0x48594c45524f5753ULL, 0x5155455259435452ULL, 1};
struct RegionQueryRowsVTable final {
  // Pure validation/discovery; false means this artifact has no such domain.
  Result<bool> (*accepts)(const ProviderRegionPrepared &, void *invocation,
      ValueId offsets_input) noexcept = nullptr;
  // Cold host metadata only. The caller invalidates covering commands first.
  // Recheck control.inspect() here and at seal; retain no extra offsets lease.
  Status (*configure)(const ProviderRegionPrepared &, void *invocation,
      const QueryRowsControl &) noexcept = nullptr;
  // Infallible, no payload access/work/allocation. Permanently unbinds the
  // previous configuration even if the same layout/version is prepared again.
  void (*clear)(const ProviderRegionPrepared &, void *invocation) noexcept = nullptr;
};
} // namespace hyle::graph::detail
