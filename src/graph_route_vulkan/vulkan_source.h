#pragma once

#include "hyle/graph/vulkan.h"
#include "../compute_route_vulkan/vulkan_carrier.h"

#include <memory>

namespace hyle::graph::vulkan::detail {
// Source-owned cold configuration. Native Programs retain the actual Runtime
// and resolved pipelines, never this callback or a borrowed public Context.
struct SourceContext final {
  std::shared_ptr<compute::vulkan::detail::Runtime> runtime;
  CustomLowering custom;
};
compute::vulkan::detail::ColdCustomResolver custom_resolver(
    const CustomLowering &) noexcept;

// Exact-source evidence only. These are uninitialized native output owners,
// not ready ReadLeases and not an alternate Program/Invocation.
HYLE_GRAPH_ROUTE_VULKAN_API Result<compute::vulkan::Tensor> output_backing(
    const ExecutionSource &, compute::vulkan::Invocation &, std::size_t) noexcept;
HYLE_GRAPH_ROUTE_VULKAN_API Status seal_outputs(
    const ExecutionSource &, compute::vulkan::Invocation &) noexcept;
} // namespace hyle::graph::vulkan::detail
