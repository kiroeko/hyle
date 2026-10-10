#pragma once

#include "hyle/compute/vulkan.h"

namespace hyle::compute::vulkan::detail {
// Cold composition seam only. Outputs are uninitialized storage, not ready
// results. A coarse plan binds its internal references before any execution,
// then freezes the exact reference baseline. Ordinary native users keep the
// original exported-owner guard; this header is not installed as public API.
struct CompositionAccess final {
  static HYLE_COMPUTE_ROUTE_VULKAN_API Result<Tensor>
  output(Invocation &invocation, std::size_t index) noexcept;
  static HYLE_COMPUTE_ROUTE_VULKAN_API Status seal(Invocation &invocation) noexcept;
};
} // namespace hyle::compute::vulkan::detail
