#pragma once

#include "hyle/graph/region_provider.h"

#include <memory>

namespace hyle::compute::vulkan::detail { struct Runtime; struct Run; }

namespace hyle::graph::vulkan::detail {
const RegionProviderVTable *native_region_provider() noexcept;
// Shared by real copy and compute endpoints on this same native Runtime.
compute::detail::CompletionEndpoint region_ordered_endpoint(
    const std::shared_ptr<compute::vulkan::detail::Runtime> &) noexcept;
bool imports_region_ordered_endpoint(
    const std::shared_ptr<compute::vulkan::detail::Runtime> &,
    compute::detail::CompletionEndpoint) noexcept;
// Cold capacity of an original idle region slot. Original same-queue submits
// consume dependency descriptors immediately; Graph retains their Sources
// through actual completion. No event storage, Queue or ready permission is
// created here. A supplied Run is the exact original slot's observation gate;
// CopyRecording callers retain their own original pure idle check instead.
Status reserve_region_dependencies(
    const std::shared_ptr<compute::vulkan::detail::Runtime> &,
    std::size_t &capacity, std::size_t requested,
    const compute::vulkan::detail::Run * = nullptr) noexcept;
Status qualify_region_dependencies(std::size_t capacity,
    std::span<const compute::detail::CompletionDependency>) noexcept;
} // namespace hyle::graph::vulkan::detail
