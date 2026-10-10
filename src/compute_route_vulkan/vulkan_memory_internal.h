#pragma once

#include "hyle/compute/vulkan_memory.h"
#include "vulkan_internal.h"

namespace hyle::compute::vulkan::detail {
struct MemoryAccess final {
  static HYLE_COMPUTE_ROUTE_VULKAN_API Result<std::shared_ptr<Buffer>>
  buffer(const MemoryResource &resource, const mem::Buffer &allocation) noexcept;
  static HYLE_COMPUTE_ROUTE_VULKAN_API const std::shared_ptr<Runtime> &
  runtime(const MemoryResource &resource) noexcept;
};
} // namespace hyle::compute::vulkan::detail
