#pragma once

#include "hyle/compute/vulkan.h"

#if defined(HYLE_COMPUTE_VULKAN)
#include <optional>

#include "hyle/compute/resource_budget.h"
#include "hyle/mem/buffer.h"

namespace hyle::compute::vulkan {
namespace detail { struct MemoryState; struct MemoryAccess; }

enum class MemoryKind : unsigned char { kDevice, kStaging };

struct AllocationOptions final {
  ResourceBudget resource_budget{};
  ResourceBudgetCategory category = ResourceBudgetCategory::kWorkspace;
};

struct NativeStorage final {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceSize offset_bytes = 0, capacity_bytes = 0;
  // Requested capacity differs from VkDeviceMemory requirements. Unknown for
  // external imports; never add this observation to managed scope usage.
  std::optional<std::size_t> allocation_bytes;
};

// Original route VkBuffer/VkDeviceMemory in the Context's exact Device/family.
// Mem handles are opaque allocation tokens, never CPU or VkBuffer pointers.
// Staging is explicitly host mapped; device storage is not advertised as host
// accessible even on a shared-memory GPU. Both kinds use original native owners.
// Resources/allocator remain borrowed through Buffer/quotas. Tensor allocations
// retain device and budget until their last original native owner disappears.
class MemoryResource final {
public:
  MemoryResource() noexcept = default;
  MemoryResource(const MemoryResource &) = delete;
  MemoryResource &operator=(const MemoryResource &) = delete;
  HYLE_COMPUTE_ROUTE_VULKAN_API MemoryResource(MemoryResource &&) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API MemoryResource &operator=(MemoryResource &&) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API ~MemoryResource();
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<MemoryResource>
  create(const Context &context, MemoryKind kind,
         mem::AllocatorRef allocator = mem::system_allocator()) noexcept;
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_COMPUTE_ROUTE_VULKAN_API mem::MemoryResourceRef ref() const noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Result<Tensor>
  allocate(const TensorDesc &description, AllocationOptions options = {}) const noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Result<NativeStorage>
  storage_info(const Tensor &tensor) const noexcept;
private:
  friend struct detail::MemoryAccess;
  explicit MemoryResource(std::unique_ptr<detail::MemoryState> state) noexcept;
  std::unique_ptr<detail::MemoryState> state_;
};
} // namespace hyle::compute::vulkan
#endif
