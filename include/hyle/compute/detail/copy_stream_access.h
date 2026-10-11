#pragma once

#include "hyle/compute/batch_transfer.h"
#include "hyle/compute/detail/copy_ranges.h"

namespace hyle::compute::detail {

// File-stream adapters keep a group mutable across bounded staging chunks.
// This transfers permission, never publishes bytes or broadens F2 output
// coverage. The adapter separately proves the entire group's initialized range.
struct CopyStreamAccess final {
  // Requires no retained submission or actual native use and successful
  // retirement of every accepted phase. Rejection leaves the binding intact.
  // An unsubmitted binding can also be withdrawn. Success invalidates that
  // binding and its prepared replay; the next chunk must bind and prepare again.
  static HYLE_COMPUTE_API Result<WriteLease> take_mutable(
      CopyBatchInvocation &invocation, CopyResourceId resource) noexcept;
  // Immutable native geometry for admission, without consuming a permission.
  static HYLE_COMPUTE_API Result<PhysicalCopyResource> physical(
      CopyBatchInvocation &, mem::ConstBufferView, bool writable) noexcept;
  // Nonblocking actual-retirement observation without publishing a write.
  static HYLE_COMPUTE_API Status observe(CopyBatchInvocation &) noexcept;
};

} // namespace hyle::compute::detail
