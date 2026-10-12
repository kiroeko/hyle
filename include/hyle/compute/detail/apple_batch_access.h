#pragma once
#include "hyle/compute/apple_memory.h"
#if defined(HYLE_COMPUTE_APPLE)
#include "hyle/compute/paged_arena.h"
#include "hyle/compute/apple_batch.h"
namespace hyle::compute::apple::detail {
// Trusted fixed-batch adapters only. Capacity is an address view; the frame
// retains exact page permissions. Submit must admit every actual page guard.
struct PagedBatchAccess final {
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<mlx::core::array> binding(
      const MemoryResource&,PagedArenaRead,LeaseImportOptions = {}) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<OwnedResult<Results>> submit(
      Invocation&,LeaseUseBatch) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Status wait_native(Invocation&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Status reuse_gate(Invocation&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<MaterializedBatchInvocation> create(
      const Context&,Program,compute::MaterializedBatchProgram,const MemoryResource&,
      const MemoryResource&,MaterializedNativeBatchOptions)noexcept;
};
} // namespace hyle::compute::apple::detail
#endif
