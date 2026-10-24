#pragma once

#include "hyle/compute/vulkan_weights.h"

namespace hyle::compute::vulkan::detail {
// Metadata and ownership only. The public WeightBatch::binding still requires
// ready bytes; Graph orders this real producer before any reader may execute.
struct WeightBinding final {
  std::shared_ptr<WeightState> owner;
  mem::Buffer storage;
  compute::ConstTensorView tensor;
  mem::AllocatorRef allocator;
};
struct WeightAccess final {
  static HYLE_COMPUTE_ROUTE_VULKAN_API Result<WeightBinding>
  binding(const WeightBatch &, std::uint64_t) noexcept;
  static HYLE_COMPUTE_ROUTE_VULKAN_API Result<CompletionPoll>
  poll(WeightState *) noexcept;
  static HYLE_COMPUTE_ROUTE_VULKAN_API Status wait(WeightState *) noexcept;
};
} // namespace hyle::compute::vulkan::detail
