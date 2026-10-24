#pragma once

#include "hyle/graph/region_provider.h"

#include <memory>

namespace hyle::compute::vulkan::detail { struct Runtime; }

namespace hyle::graph::vulkan::detail {
const RegionProviderVTable *native_region_provider() noexcept;
// Shared by real copy and compute endpoints on this same native Runtime.
compute::detail::CompletionEndpoint region_ordered_endpoint(
    const std::shared_ptr<compute::vulkan::detail::Runtime> &) noexcept;
bool imports_region_ordered_endpoint(
    const std::shared_ptr<compute::vulkan::detail::Runtime> &,
    compute::detail::CompletionEndpoint) noexcept;
} // namespace hyle::graph::vulkan::detail
