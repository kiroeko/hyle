#pragma once

#include "hyle/compute/resource_budget.h"

namespace hyle::compute::detail {
struct SharedBackingRecord;
struct BackingIdentity final {
  const void *authority = nullptr;
  const void *value = nullptr;
};

// Accounting only: copies share one charge for a live original allocation in
// one budget scope. Active holders name the original live payload; its native
// adapter separately retains storage and prevents identity reuse. Teardown can
// withdraw that lookup authority into a retiring holder which preserves only
// quota through actual owner release. The authority namespace outlives both.
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

// Original storage teardown has withdrawn lookup authority, but its actual
// backing/provider owner has not finished releasing yet. Moving this token
// allocates nothing; its final reset releases only this retiring holder.
class RetiringBackingReservation final {
public:
  RetiringBackingReservation() noexcept = default;
  RetiringBackingReservation(const RetiringBackingReservation &) = delete;
  RetiringBackingReservation &operator=(const RetiringBackingReservation &) = delete;
  HYLE_COMPUTE_API RetiringBackingReservation(RetiringBackingReservation &&) noexcept;
  HYLE_COMPUTE_API RetiringBackingReservation &operator=(RetiringBackingReservation &&) noexcept;
  HYLE_COMPUTE_API ~RetiringBackingReservation();
  HYLE_COMPUTE_API void reset() noexcept;
private:
  friend struct ResourceBudgetAccess;
  explicit RetiringBackingReservation(SharedBackingRecord *record) noexcept : record_(record) {}
  SharedBackingRecord *record_ = nullptr;
};

// Cold lifetime composition only: the accounting wrapper retains its budget,
// while the actual allocator object must be retained by its real owner.
struct ResourceBudgetAccess final {
  // Transfer an existing active holder to actual storage retirement. The last
  // active holder unlinks its identity now; quota remains until every retiring
  // storage owner has finished. No new storage or fallible work is introduced.
  static HYLE_COMPUTE_API RetiringBackingReservation
  retire_backing(SharedBackingReservation &) noexcept;
  // Cold ownership identity only. Unwrap only this implementation's exact
  // self-retaining metadata allocator callbacks; unknown allocators remain
  // unchanged and still require their original external owner.
  static HYLE_COMPUTE_API mem::AllocatorRef
  original_allocator(mem::AllocatorRef) noexcept;
  // The actual allocator behind known budget adapters, not the callback that
  // must be used for freeing the budgeted allocation.
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
