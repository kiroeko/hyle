#pragma once
#include "hyle/compute/vulkan_memory.h"
#include "hyle/compute/detail/copy_ranges.h"

namespace hyle::compute::vulkan::detail {
struct CopyStreamAccess final {
  static Result<WriteLease> take_mutable(CopyBatchInvocation &, CopyResourceId) noexcept;
  static Result<compute::detail::PhysicalCopyResource> physical(CopyBatchInvocation &, mem::ConstBufferView, bool) noexcept;
  static Status observe(CopyBatchInvocation &) noexcept;
};
} // namespace hyle::compute::vulkan::detail
