#pragma once

#include "hyle/compute/vulkan_memory.h"
#include "vulkan_internal.h"
#include "hyle/compute/detail/copy_ranges.h"

namespace hyle::compute::vulkan::detail {
struct CopySpan final {
  std::shared_ptr<Buffer> backing;
  compute::detail::PhysicalCopyResource physical;
  VkDeviceSize offset = 0;
};
struct MemoryAccess final {
  static HYLE_COMPUTE_ROUTE_VULKAN_API Result<std::shared_ptr<Buffer>>
  buffer(const MemoryResource &resource, const mem::Buffer &allocation) noexcept;
  static HYLE_COMPUTE_ROUTE_VULKAN_API const std::shared_ptr<Runtime> &
  runtime(const MemoryResource &resource) noexcept;
  static HYLE_COMPUTE_ROUTE_VULKAN_API Result<CopySpan>
  copy_span(const std::shared_ptr<Runtime> &runtime, mem::ConstBufferView view) noexcept;
};
} // namespace hyle::compute::vulkan::detail
