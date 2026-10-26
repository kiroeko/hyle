#pragma once

#include <span>

#include "hyle/compute/resource_leases.h"

namespace hyle::compute {

// Coordinates relative to one LeaseSlotLayout, never a resource permission or
// globally unique identity. Keep the actual lease: a reused slot has a different
// generation and every native use still needs original LeaseUseBatch admission.
struct LeaseSlotAddress final {
  std::uint32_t index = 0;
  std::uint64_t generation = 0;
  friend bool operator==(LeaseSlotAddress, LeaseSlotAddress) noexcept = default;
};

// Fixed item geometry over an existing LeasePool's one backing. This retains
// the original domain, not a second allocation or an allocation authority. A
// pool move preserves it; dropping/closing the pool prevents new native use as
// before. Locate remains an address/permission-fact query after close, allowing
// retained aliases and retired recovery; it does not reopen native admission.
// Resource, Context and allocator lifetimes retain the original pool contract.
class LeaseSlotLayout final {
public:
  LeaseSlotLayout() noexcept = default;
  HYLE_COMPUTE_API LeaseSlotLayout(const LeaseSlotLayout &) noexcept;
  HYLE_COMPUTE_API LeaseSlotLayout &operator=(const LeaseSlotLayout &) noexcept;
  HYLE_COMPUTE_API LeaseSlotLayout(LeaseSlotLayout &&) noexcept;
  HYLE_COMPUTE_API LeaseSlotLayout &operator=(LeaseSlotLayout &&) noexcept;
  HYLE_COMPUTE_API ~LeaseSlotLayout();

  // The item is dense/plain and nonempty. Its bytes fit inside stride; every
  // slot is correctly aligned and the final item's end fits the actual pool.
  // Creation requires an open pool, consumes no lease and allocates no payload.
  HYLE_COMPUTE_API static Result<LeaseSlotLayout>
  create(const LeasePool &, TensorDesc item, std::uint32_t slots,
         std::size_t stride_bytes) noexcept;
  bool valid() const noexcept { return domain_ != nullptr; }
  TensorDesc description() const noexcept { return description_; }
  std::uint32_t slot_count() const noexcept { return slots_; }
  std::size_t stride_bytes() const noexcept { return stride_; }

  // Requires the actual original full, ordinary block in this exact domain:
  // one ready reader or an exclusive successful writer. A partial/oversized
  // block, appendable prefix, foreign import or padding offset is rejected.
  // Read aliases may share a source; writers with admitted uses are unavailable.
  HYLE_COMPUTE_API Result<LeaseSlotAddress> locate(const ReadLease &) const noexcept;
  HYLE_COMPUTE_API Result<LeaseSlotAddress> locate(const WriteLease &) const noexcept;

  // Validate then encode without allocation, consumption or partial outputs.
  // Destination objects are a contiguous span of original move-only writers;
  // original disjoint allocations and read/write mode prove no source/write
  // overlap or repeated destination. Sources may repeat. Outputs have matching
  // lengths and must not overlap each other. External synchronization keeps
  // handles stable for the call; the adapter retains/admit leases separately.
  HYLE_COMPUTE_API Status bind_indices(std::span<const ReadLease> sources,
      std::span<const WriteLease> destinations,
      std::span<LeaseSlotAddress> source_addresses,
      std::span<LeaseSlotAddress> destination_addresses) const noexcept;

private:
  friend struct detail::LeaseAccess;
  detail::LeaseDomain *domain_ = nullptr;
  TensorDesc description_{};
  std::size_t bytes_ = 0, stride_ = 0, extent_ = 0;
  std::uint32_t slots_ = 0;
};

} // namespace hyle::compute
