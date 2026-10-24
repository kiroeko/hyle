#pragma once
#include "hyle/compute/vulkan_memory.h"
#if defined(HYLE_COMPUTE_VULKAN)
#include <cstdint>
#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/detail/generated_result.h"
#include "hyle/compute/paged_arena.h"
#include "hyle/compute/vulkan_batch.h"
namespace hyle::compute::vulkan::detail {
struct Plan;
// Trusted fixed-batch adapters only; never a whole-arena ReadLease. The native
// view retains its original backing and the immutable exact page footprint.
struct PagedBatchAccess final {
  HYLE_COMPUTE_ROUTE_VULKAN_API static bool cache_domain(const BatchCapacity&) noexcept;
  static Result<Program> prepare_ragged_program(const std::shared_ptr<Runtime>&, const Plan&,
      BatchLayoutVersion, bool causal, std::int32_t window_left) noexcept;
  // Original ragged mathematics with the Graph's concrete packed Q/output rows.
  // Only its inactive output suffix is zero-filled; history is not copied.
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<Program> prepare_ragged_graph(
      const std::shared_ptr<Runtime>&, BatchLayoutVersion, RaggedAttention,
      std::uint32_t query_rows, const WorkloadContext&) noexcept;
  static Status update_ragged(Invocation&, const BatchLayoutVersion&) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API static Status update_layout(
      Invocation&, const BatchLayoutVersion&) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<Program> prepare_program(
      const Context&, const NativePagedBatchProgram&, BatchLayoutVersion,
      const WorkloadContext&) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<Program> prepare_program(
      const std::shared_ptr<Runtime>&, const NativePagedBatchProgram&, BatchLayoutVersion,
      const WorkloadContext&) noexcept;
  // The same native page mathematics with a concrete Graph Q/output extent.
  // Only active layout rows are evaluated; its inactive output suffix is zero.
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<Program> prepare_paged_graph(
      const std::shared_ptr<Runtime>&, const NativePagedBatchProgram&, BatchLayoutVersion,
      std::uint32_t query_rows, const WorkloadContext&) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<Tensor> binding(
      const MemoryResource&,PagedArenaRead,LeaseImportOptions = {}) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<Tensor> binding(
      const MemoryResource&,PagedRowCacheRead,LeaseImportOptions = {}) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<Tensor> binding(
      const MemoryResource&,PagedArenaPendingRead,LeaseImportOptions = {}) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<OwnedResult<Results>> submit(
      Invocation&,LeaseUseBatch) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API static Status wait_native(Invocation&) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API static Status reuse_gate(Invocation&) noexcept;
  // Graph-only cold composition, after reuse_gate and all internal bindings.
  HYLE_COMPUTE_ROUTE_VULKAN_API static Status reseal_composition(Invocation&) noexcept;
  // Checked idle batch reuse only: release prior input imports, keep outputs.
  HYLE_COMPUTE_ROUTE_VULKAN_API static Status release_inputs(Invocation&) noexcept;
  // Original cold owner must exclude rebinding/submission through both calls.
  // Pure retirement/escape validation; closed/failed but retired cleanup is legal.
  HYLE_COMPUTE_ROUTE_VULKAN_API static Status can_detach_inputs(
      const Invocation&, const compute::detail::ResultStorageOwner&) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API static void detach_inputs(Invocation&) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API static Status qualify_source(const Context&,
      compute::detail::CompletionDependency)noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<MaterializedBatchInvocation> create(
      const Context&,Program,compute::MaterializedBatchProgram,const MemoryResource&,
      const MemoryResource&,MaterializedNativeBatchOptions)noexcept;
};
} // namespace hyle::compute::vulkan::detail
#endif
