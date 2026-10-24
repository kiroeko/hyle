#pragma once

#include "hyle/compute/batch_transfer.h"
#include "hyle/compute/detail/copy_ranges.h"
#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/detail/context_ref.h"

namespace hyle::compute::detail {

// File-stream adapters keep a group mutable across bounded staging chunks.
// This transfers permission, never publishes bytes or broadens F2 output
// coverage. The adapter separately proves the entire group's initialized range.
// Addressable original mutable content, not a ready lease. Its writer/view
// stay borrowed from the move-owned source until every dependent native use
// retires. The receiver qualifies ordering before admitting any read.
struct CopyUpdateInput final {
  const WriteLease *writer = nullptr;
  mem::ConstBufferView view;
  PhysicalCopyResource physical;
  const void *root = nullptr;
  CopyPhaseId last_write;
};
struct CopyStreamAccess final {
  // Same cold preparation as public Context callers, with the exact retained
  // owner. No temporary Context handle may close that shared owner on release.
  static HYLE_COMPUTE_API Result<CopyBatchProgram> prepare(
      const ContextRef &, std::span<const CopyResourceSpec>, CopyBatchOptions) noexcept;
  // Borrow original completions for an opt-in actual-retirement observer.
  // Empty output queries capacity. A nonempty insufficient span is untouched.
  // Accepted phases and only their used producer bindings are included; no
  // poll/wait/reset is performed and the update remains the sole owner.
  static HYLE_COMPUTE_API Result<std::size_t> retirement_sources(
      CopyBatchUpdate &, std::span<Completion *>) noexcept;
  // Trusted update receivers only: no host reads or public ready conversion.
  static HYLE_COMPUTE_API Result<compute::detail::CopyUpdateInput> input(
      const CopyBatchUpdate &, CopyResourceId) noexcept;
  // Logical bytes written by the accepted original episode; no observation,
  // publication or permission transfer. Successful retirement is separate.
  static HYLE_COMPUTE_API Status covers(const CopyBatchUpdate &, CopyResourceId,
                                      std::size_t offset, std::size_t bytes) noexcept;
  // Borrow the latest real async prefix; empty final logical phases must not
  // erase preceding native work. Known failures always reject. Do not observe
  // or return the owner between resolving and recording this exact dependency.
  static HYLE_COMPUTE_API Result<compute::detail::CompletionDependency> dependency(
      const CopyBatchUpdate &) noexcept;
  // Caller has qualified the receiver endpoint and original native ordering.
  static HYLE_COMPUTE_API Status admit_read(LeaseUseBatch &, const CopyBatchUpdate &,
                                CopyResourceId) noexcept;

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
