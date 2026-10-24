#pragma once
#include "hyle/compute/vulkan_memory.h"
#include "hyle/compute/detail/copy_stream_access.h"

namespace hyle::compute::vulkan::detail {
struct CopyStreamAccess final {
  static HYLE_COMPUTE_ROUTE_VULKAN_API Result<CopyBatchProgram> prepare_program(
      const std::shared_ptr<Runtime>&,std::span<const CopyResourceSpec>,CopyBatchOptions) noexcept;
  static HYLE_COMPUTE_ROUTE_VULKAN_API Result<std::size_t> retirement_sources(
      CopyBatchUpdate &, std::span<Completion *>) noexcept;
  // Trusted update receivers only: no host reads or public ready conversion.
  static HYLE_COMPUTE_ROUTE_VULKAN_API Result<compute::detail::CopyUpdateInput> input(
      const CopyBatchUpdate &, CopyResourceId) noexcept;
  static HYLE_COMPUTE_ROUTE_VULKAN_API Status covers(const CopyBatchUpdate &, CopyResourceId,
                                                  std::size_t offset, std::size_t bytes) noexcept;
  // Borrow the latest real async prefix; empty final logical phases must not
  // erase preceding native work. Known failures always reject. Do not observe
  // or return the owner between resolving and recording this exact dependency.
  static HYLE_COMPUTE_ROUTE_VULKAN_API Result<compute::detail::CompletionDependency> dependency(
      const CopyBatchUpdate &) noexcept;
  // Caller has qualified the receiver endpoint and original native ordering.
  static HYLE_COMPUTE_ROUTE_VULKAN_API Status admit_read(LeaseUseBatch &, const CopyBatchUpdate &,
                                CopyResourceId) noexcept;

  static HYLE_COMPUTE_ROUTE_VULKAN_API Result<WriteLease> take_mutable(CopyBatchInvocation &, CopyResourceId) noexcept;
  static HYLE_COMPUTE_ROUTE_VULKAN_API Result<compute::detail::PhysicalCopyResource> physical(CopyBatchInvocation &, mem::ConstBufferView, bool) noexcept;
  static HYLE_COMPUTE_ROUTE_VULKAN_API Status observe(CopyBatchInvocation &) noexcept;
};
} // namespace hyle::compute::vulkan::detail
