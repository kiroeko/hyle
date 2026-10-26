#pragma once
#include "hyle/compute/apple_memory.h"
#if defined(HYLE_COMPUTE_APPLE)
#include "hyle/compute/detail/copy_stream_access.h"

namespace hyle::compute::apple::detail {
struct CopyStreamAccess final {
  static HYLE_COMPUTE_ROUTE_APPLE_API Result<CopyBatchProgram> prepare_program(
      std::shared_ptr<Runtime>, std::span<const CopyResourceSpec>, CopyBatchOptions) noexcept;
  // Original FIFO preflight also observes these completions. Share its lock
  // for nonblocking qualification, observation and retirement attachment.
  static HYLE_COMPUTE_ROUTE_APPLE_API void serialize(
      CopyBatchUpdate &, void (*)(void *) noexcept, void *) noexcept;
  static HYLE_COMPUTE_ROUTE_APPLE_API Result<std::size_t> retirement_sources(
      CopyBatchUpdate &, std::span<Completion *>) noexcept;
  // Trusted update receivers only: no host reads or public ready conversion.
  static HYLE_COMPUTE_ROUTE_APPLE_API Result<compute::detail::CopyUpdateInput> input(
      const CopyBatchUpdate &, CopyResourceId) noexcept;
  static HYLE_COMPUTE_ROUTE_APPLE_API Status covers(const CopyBatchUpdate &, CopyResourceId,
                                                 std::size_t offset, std::size_t bytes) noexcept;
  // Borrow the latest real async prefix; empty final logical phases must not
  // erase preceding native work. Known failures always reject. Do not observe
  // or return the owner between resolving and recording this exact dependency.
  static HYLE_COMPUTE_ROUTE_APPLE_API Result<compute::detail::CompletionDependency> dependency(
      const CopyBatchUpdate &) noexcept;
  // Original FIFO Job only: recheck every actual accepted recording before
  // reading pending destinations. Never observe public Completion handles or
  // wait; the receiver's existing source owner keeps this Update stable.
  static HYLE_COMPUTE_ROUTE_APPLE_API Status preflight(const CopyBatchUpdate &) noexcept;
  // Caller has qualified the receiver endpoint and original native ordering.
  static HYLE_COMPUTE_ROUTE_APPLE_API Status admit_read(LeaseUseBatch &, const CopyBatchUpdate &,
                                CopyResourceId) noexcept;

  static HYLE_COMPUTE_ROUTE_APPLE_API Result<WriteLease> take_mutable(CopyBatchInvocation &, CopyResourceId) noexcept;
  static HYLE_COMPUTE_ROUTE_APPLE_API Result<compute::detail::PhysicalCopyResource> physical(CopyBatchInvocation &, mem::ConstBufferView, bool) noexcept;
  static HYLE_COMPUTE_ROUTE_APPLE_API Status observe(CopyBatchInvocation &) noexcept;
};
} // namespace hyle::compute::apple::detail
#endif
