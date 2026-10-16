#pragma once
#include "hyle/compute/vulkan_memory.h"
#if defined(HYLE_COMPUTE_VULKAN)
#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/paged_arena.h"
#include "hyle/compute/vulkan_batch.h"
namespace hyle::compute::vulkan::detail {
// Trusted fixed-batch adapters only; never a whole-arena ReadLease. The native
// view retains its original backing and the immutable exact page footprint.
struct PagedBatchAccess final {
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<Program> prepare_program(
      const Context&, const NativePagedBatchProgram&, BatchLayoutVersion,
      const WorkloadContext&) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<Tensor> binding(
      const MemoryResource&,PagedArenaRead,LeaseImportOptions = {}) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<Tensor> binding(
      const MemoryResource&,PagedArenaPendingRead,LeaseImportOptions = {}) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<OwnedResult<Results>> submit(
      Invocation&,LeaseUseBatch) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API static Status wait_native(Invocation&) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API static Status reuse_gate(Invocation&) noexcept;
  // Checked idle batch reuse only: release prior input imports, keep outputs.
  HYLE_COMPUTE_ROUTE_VULKAN_API static Status release_inputs(Invocation&) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API static Status qualify_source(const Context&,
      compute::detail::CompletionDependency)noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<MaterializedBatchInvocation> create(
      const Context&,Program,compute::MaterializedBatchProgram,const MemoryResource&,
      const MemoryResource&,MaterializedNativeBatchOptions)noexcept;
};
} // namespace hyle::compute::vulkan::detail
#endif
