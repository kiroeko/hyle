#pragma once

#include "hyle/compute/resource_leases.h"

namespace hyle::compute::detail {

// One original allocation's monotone tail authority. ReadLease copies keep
// their original bounds; neither they nor this handle can mutate its committed
// prefix. Control of the same handle is externally synchronized, like WriteLease.
class AppendLease final {
public:
  AppendLease() noexcept = default;
  AppendLease(const AppendLease &) = delete;
  AppendLease &operator=(const AppendLease &) = delete;
  HYLE_COMPUTE_API AppendLease(AppendLease &&) noexcept;
  HYLE_COMPUTE_API AppendLease &operator=(AppendLease &&) noexcept;
  HYLE_COMPUTE_API ~AppendLease();
  bool valid() const noexcept { return block_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  HYLE_COMPUTE_API std::uint64_t generation() const noexcept;
  HYLE_COMPUTE_API std::uint64_t epoch() const noexcept;
  HYLE_COMPUTE_API std::size_t committed_bytes() const noexcept;
  HYLE_COMPUTE_API std::size_t reserved_bytes() const noexcept;
  HYLE_COMPUTE_API std::size_t capacity_bytes() const noexcept;
  // No allocation or native work. Requires all prior native uses to retire.
  HYLE_COMPUTE_API Status reserve(std::size_t bytes) noexcept;
  // Borrow the exact reserved suffix before admitting its native use.
  HYLE_COMPUTE_API Result<mem::BufferView> view() const noexcept;
  // Widen only after successful actual retirement. Older readers stay bounded.
  HYLE_COMPUTE_API Result<ReadLease> publish() noexcept;
  // Retired cleanup also works after domain close; it never advances prefix.
  HYLE_COMPUTE_API Status rollback() noexcept;
  HYLE_COMPUTE_API void reset() noexcept;
private:
  friend struct LeaseAccess;
  explicit AppendLease(LeaseBlock *block) noexcept : block_(block) {}
  LeaseBlock *block_ = nullptr;
};

struct AppendLeasePublication final {
  ReadLease prefix;
  AppendLease appender;
};

} // namespace hyle::compute::detail
