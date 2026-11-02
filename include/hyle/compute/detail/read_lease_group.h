#pragma once

#include "hyle/compute/resource_leases.h"

namespace hyle::compute::detail {
struct ReadLeaseGroupData;

// Immutable permissions for original blocks that can no longer acquire a
// writer or append owner. Leaves own the original bounded ReadLease; joins
// share those leaves without copying or rescanning old page permissions.
// A group grants no access to holes between its independently leased blocks.
class ReadLeaseGroup final {
public:
  ReadLeaseGroup() noexcept = default;
  HYLE_COMPUTE_API ReadLeaseGroup(const ReadLeaseGroup &) noexcept;
  HYLE_COMPUTE_API ReadLeaseGroup &operator=(const ReadLeaseGroup &) noexcept;
  HYLE_COMPUTE_API ReadLeaseGroup(ReadLeaseGroup &&) noexcept;
  HYLE_COMPUTE_API ReadLeaseGroup &operator=(ReadLeaseGroup &&) noexcept;
  HYLE_COMPUTE_API ~ReadLeaseGroup();
  HYLE_COMPUTE_API static Result<ReadLeaseGroup> from_lease(const ReadLease &,
      mem::AllocatorRef = mem::system_allocator()) noexcept;
  // Both groups must use the same original domain, and every left range must
  // precede every right range. A balanced immutable tree bounds lookup and
  // reclamation depth even when a caller repeatedly appends one new page.
  HYLE_COMPUTE_API static Result<ReadLeaseGroup> join(const ReadLeaseGroup &left,
      const ReadLeaseGroup &right) noexcept;
  // Persistent bank-table edits. Only the new leaf is qualified; unrelated
  // original permissions remain in the shared tree. No caller storage is held.
  HYLE_COMPUTE_API Result<ReadLeaseGroup> with(const ReadLease &,
      mem::AllocatorRef = mem::system_allocator()) const noexcept;
  HYLE_COMPUTE_API Result<ReadLeaseGroup> without(std::size_t original_offset) const noexcept;
  HYLE_COMPUTE_API Result<ReadLease> at(std::size_t original_offset) const noexcept;
  bool valid() const noexcept { return data_ != nullptr; }
  HYLE_COMPUTE_API std::uint32_t size() const noexcept;
  HYLE_COMPUTE_API mem::AllocatorRef metadata_allocator() const noexcept;
  HYLE_COMPUTE_API bool contains(mem::ConstBufferView) const noexcept;
  HYLE_COMPUTE_API bool within(mem::ConstBufferView) const noexcept;
  HYLE_COMPUTE_API bool overlaps(mem::ConstBufferView) const noexcept;
  HYLE_COMPUTE_API bool subset_of(const ReadLeaseGroup &) const noexcept;
  // Borrow a genuine original guard; the caller retains this immutable group.
  HYLE_COMPUTE_API const ReadLease *find(mem::ConstBufferView) const noexcept;
  HYLE_COMPUTE_API const ReadLease *page(std::uint32_t physical_ordinal) const noexcept;
  bool same_owner(const ReadLeaseGroup &other) const noexcept { return data_ == other.data_; }
  HYLE_COMPUTE_API void reset() noexcept;
private:
  friend struct LeaseAccess;
  explicit ReadLeaseGroup(ReadLeaseGroupData *data) noexcept : data_(data) {}
  ReadLeaseGroupData *data_ = nullptr;
};
} // namespace hyle::compute::detail
