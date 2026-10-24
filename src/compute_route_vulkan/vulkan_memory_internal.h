#pragma once

#include "hyle/compute/vulkan_memory.h"
#include "hyle/compute/paged_arena.h"
#include "vulkan_internal.h"
#include "hyle/compute/detail/copy_ranges.h"

namespace hyle::compute::vulkan::detail {
struct CopySpan final {
  std::shared_ptr<Buffer> backing;
  compute::detail::PhysicalCopyResource physical;
  VkDeviceSize offset = 0;
};
struct ConstantCopy final {
  compute::TensorView destination;
  compute::ConstTensorView source;
};
// Accounting identity only. The exporting original Buffer/view owner chain
// keeps this complete allocation alive; this is not a read/write permission.
struct RetainedBackingFacts final {
  const void *authority = nullptr;
  const void *canonical_owner = nullptr;
  std::size_t full_bytes = 0;
  std::size_t backing_offset = 0;
};
struct MemoryAccess final {
  static HYLE_COMPUTE_ROUTE_VULKAN_API Result<std::shared_ptr<Buffer>>
  buffer(const MemoryResource &resource, const mem::Buffer &allocation) noexcept;
  static HYLE_COMPUTE_ROUTE_VULKAN_API const std::shared_ptr<Runtime> &
  runtime(const MemoryResource &resource) noexcept;
  static HYLE_COMPUTE_ROUTE_VULKAN_API Status
  validate_resource(const std::shared_ptr<Runtime> &, mem::MemoryResourceRef) noexcept;
  // Native backing and ownership only: this does not assert initialization
  // or grant a ready lease. Used by cold generated ports before execution.
  static HYLE_COMPUTE_ROUTE_VULKAN_API Result<mem::Buffer>
  export_storage(const std::shared_ptr<Runtime> &, mem::MemoryResourceRef,
                 const Tensor &, mem::AllocatorRef) noexcept;
  // The caller nominates a typed root from the actual Program/Run storage
  // block, never by downcasting an opaque import owner. Verify this Tensor's
  // exact original owned root/alias chain before annotating an exported lease.
  static HYLE_COMPUTE_ROUTE_VULKAN_API Result<RetainedBackingFacts>
  retained_backing(const std::shared_ptr<Runtime> &, mem::MemoryResourceRef,
      const Tensor &, const std::shared_ptr<Buffer> &root,
      VkDeviceSize relative_offset) noexcept;
  // Cold metadata import from this actual resource. The native view retains
  // the original Buffer, not its enclosing result slot or Graph Invocation.
  static HYLE_COMPUTE_ROUTE_VULKAN_API Result<Tensor>
  tensor_view(const std::shared_ptr<Runtime> &, const ConstTensorView &,
              mem::AllocatorRef, ResourceBudget = {}) noexcept;
  static HYLE_COMPUTE_ROUTE_VULKAN_API Result<Tensor>
  lease_tensor(const std::shared_ptr<Runtime>&,mem::MemoryResourceRef,ReadLease,
      const TensorDesc&,LeaseImportOptions = {}) noexcept;
  static HYLE_COMPUTE_ROUTE_VULKAN_API Result<Tensor>
  paged_tensor(const std::shared_ptr<Runtime>&,mem::MemoryResourceRef,PagedArenaRead,
      LeaseImportOptions = {}) noexcept;
  static HYLE_COMPUTE_ROUTE_VULKAN_API Result<Tensor>
  paged_tensor(const std::shared_ptr<Runtime>&,mem::MemoryResourceRef,PagedArenaPendingRead,
      LeaseImportOptions = {}) noexcept;
  static HYLE_COMPUTE_ROUTE_VULKAN_API Result<CopySpan>
  copy_span(const std::shared_ptr<Runtime> &runtime, mem::ConstBufferView view) noexcept;
  // Cold exact-byte host/device constant transfers. Qualification never reads
  // payload or allocates. The selected batch retains all native endpoints,
  // reuses one max-capacity TransferSlot for both host directions, and the
  // existing CopyRecording for aligned same-Runtime device/device spans;
  // odd owned device carriers reuse that same TransferSlot without host copies.
  static HYLE_COMPUTE_ROUTE_VULKAN_API Result<bool>
  can_copy_constant(const std::shared_ptr<Runtime> &, const ConstantCopy &) noexcept;
  // Actual mapped payload requirement; aligned device/device uses no staging.
  // This payload bound excludes driver/recording metadata.
  static HYLE_COMPUTE_ROUTE_VULKAN_API Result<std::size_t>
  constant_copy_staging_bytes(const std::shared_ptr<Runtime> &, const ConstantCopy &) noexcept;
  static HYLE_COMPUTE_ROUTE_VULKAN_API Status
  copy_constants(const std::shared_ptr<Runtime> &, std::span<const ConstantCopy>,
                 mem::AllocatorRef) noexcept;
};
} // namespace hyle::compute::vulkan::detail
