#pragma once
#include "hyle/compute/vulkan_memory.h"
#if defined(HYLE_COMPUTE_VULKAN)
#include "hyle/compute/paged_arena.h"
namespace hyle::compute::vulkan::detail {
// Trusted fixed-batch adapters only; never a whole-arena ReadLease. The native
// view retains its original backing and the immutable exact page footprint.
struct PagedBatchAccess final {
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<Tensor> binding(
      const MemoryResource&,PagedArenaRead,LeaseImportOptions = {}) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<OwnedResult<Results>> submit(
      Invocation&,LeaseUseBatch) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API static Status wait_native(Invocation&) noexcept;
};
} // namespace hyle::compute::vulkan::detail
#endif
