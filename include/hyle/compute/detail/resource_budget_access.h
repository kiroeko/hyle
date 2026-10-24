#pragma once

#include "hyle/compute/resource_budget.h"

namespace hyle::compute::detail {
struct SharedBackingRecord;
struct BackingIdentity final {
  const void *authority = nullptr;
  const void *value = nullptr;
};

// Accounting only: copies share one charge for a live original allocation in
// one budget scope. The native adapter must separately retain the payload and
// prevent reuse of its identity while any token remains. The authority names
// the adapter's identity namespace and must outlive all tokens.
class SharedBackingReservation final {
public:
  SharedBackingReservation() noexcept = default;
  HYLE_COMPUTE_API SharedBackingReservation(const SharedBackingReservation &) noexcept;
  HYLE_COMPUTE_API SharedBackingReservation &operator=(const SharedBackingReservation &) noexcept;
  HYLE_COMPUTE_API SharedBackingReservation(SharedBackingReservation &&) noexcept;
  HYLE_COMPUTE_API SharedBackingReservation &operator=(SharedBackingReservation &&) noexcept;
  HYLE_COMPUTE_API ~SharedBackingReservation();
  bool valid() const noexcept { return record_ != nullptr; }
  HYLE_COMPUTE_API bool same_scope(const ResourceBudget &) const noexcept;
  HYLE_COMPUTE_API void reset() noexcept;
private:
  friend struct ResourceBudgetAccess;
  explicit SharedBackingReservation(SharedBackingRecord *record) noexcept : record_(record) {}
  SharedBackingRecord *record_ = nullptr;
};

// Cold lifetime composition only: the accounting wrapper retains its budget,
// while the actual allocator object must be retained by its real owner.
struct ResourceBudgetAccess final {
  static HYLE_COMPUTE_API mem::AllocatorRef
  backing_allocator(const ResourceBudget &) noexcept;
  static HYLE_COMPUTE_API mem::AllocatorRef
  backing_allocator(const ResourceBudget::Reservation &) noexcept;
  static HYLE_COMPUTE_API bool
  same_scope(const ResourceBudget::Reservation &, const ResourceBudget &) noexcept;
  // The caller proves original allocation identity and the nominated exact
  // resource. Empty/unknown backing is rejected; aliases must agree on bytes.
  static HYLE_COMPUTE_API Result<SharedBackingReservation>
  reserve_backing(const ResourceBudget &, mem::MemoryResourceRef nominated,
                  BackingIdentity original_allocation, std::size_t bytes) noexcept;
  // Register a matching already-admitted allocation without a second quota
  // reservation. Success consumes existing; rejection leaves it untouched.
  static HYLE_COMPUTE_API Result<SharedBackingReservation>
  adopt_backing(const ResourceBudget &, mem::MemoryResourceRef nominated,
                BackingIdentity original_allocation, std::size_t bytes,
                ResourceBudget::Reservation &existing) noexcept;
};
} // namespace hyle::compute::detail
