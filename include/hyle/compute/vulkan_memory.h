#pragma once

#include "hyle/compute/vulkan.h"

#if defined(HYLE_COMPUTE_VULKAN)
#include <optional>

#include "hyle/compute/resource_budget.h"
#include "hyle/compute/tensor.h"
#include "hyle/mem/buffer.h"

namespace hyle::compute::vulkan {
namespace detail { struct MemoryState; struct MemoryAccess; struct TransferState; struct TransferAccess; }

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
  bool transfer_eligible = false; // Original compute and transfer families.
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

// One cold staging allocation, command pool/buffer and fence. Host sources are
// borrowed only for upload's synchronous staging preparation. Returned native
// Completions protect accepted work; download data is exposed by read() only
// after successful terminal observation. All operations preserve exact bytes.
// Copy origins must be four-byte aligned; use an existing typed Copy/repack for
// other views. Reuse never grows capacity or reselects a Provider.
// Handles are thread-compatible. Resource/allocator outlive late work; Context
// collect/close retire abandoned prefixes using the original native runtime.
// Upload of eligible MemoryResource storage uses the original transfer queue
// and a semaphore handoff to compute. Ready remains compute ordered-prefix
// completion. Legacy/imported storage and downloads use the original compute
// queue; queue availability alone does not prove physical overlap.
class TransferSlot final {
public:
  TransferSlot() noexcept = default;
  TransferSlot(const TransferSlot &) = delete;
  TransferSlot &operator=(const TransferSlot &) = delete;
  HYLE_COMPUTE_ROUTE_VULKAN_API TransferSlot(TransferSlot &&) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API TransferSlot &operator=(TransferSlot &&) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API ~TransferSlot();
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<TransferSlot>
  create(const Context &context, const MemoryResource &staging,
         std::size_t capacity_bytes, AllocationOptions options = {}) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Result<Completion>
  upload(const Tensor &destination, ConstTensorView source) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Result<Completion> download(const Tensor &source) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status read(TensorView destination) const noexcept;
private:
  friend struct detail::TransferAccess;
  std::shared_ptr<detail::TransferState> state_;
};
} // namespace hyle::compute::vulkan
#endif
