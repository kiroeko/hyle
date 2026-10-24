#pragma once
#include "hyle/compute/apple_memory.h"
#if defined(HYLE_COMPUTE_APPLE)
#include <cstdint>
#include <functional>
#include <vector>
#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/detail/generated_result.h"
#include "hyle/compute/paged_arena.h"
#include "hyle/compute/apple_batch.h"
namespace hyle::compute::apple::detail {
// Trusted fixed-batch adapters only. Capacity is an address view; the frame
// retains exact page permissions. Submit must admit every actual page guard.
struct PagedBatchAccess final {
  static bool cache_domain(const BatchCapacity&) noexcept;
  // Native descriptor/layout validation only; no Program tracing or JIT.
  // The original two-pass kernels retain F32 partials for every input dtype.
  HYLE_COMPUTE_ROUTE_APPLE_API static Status validate_paged(
      const std::shared_ptr<Runtime>&, const BatchLayoutVersion&,
      bool causal, std::int32_t window_left) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Status validate_ragged(
      const std::shared_ptr<Runtime>&, const RaggedAttention&) noexcept;
  // The ordinary Graph freezes its concrete packed Q/output rows. Only its
  // inactive suffix is zero-filled; the expert Direct entry stays active-only.
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<Program> prepare_ragged_graph(
      std::shared_ptr<Runtime>, BatchLayoutVersion, RaggedAttention,
      std::uint32_t query_rows, const WorkloadContext&) noexcept;
  using RuntimeFunction = std::function<Result<std::vector<mlx::core::array>>(
      const BatchLayoutVersion&, const std::vector<mlx::core::array>&, mlx::core::Stream)>;
  // Only cold-qualified pure paged/ragged builders enter this managed bridge.
  static Result<Program> prepare_runtime_program(const Context&, BatchLayoutVersion,
      std::span<const InputSpec>, RuntimeFunction, const WorkloadContext&,
      bool causal = true, std::int32_t window_left = -1) noexcept;
  static Result<Program> prepare_runtime_program(std::shared_ptr<Runtime>, BatchLayoutVersion,
      std::span<const InputSpec>, RuntimeFunction, const WorkloadContext&,
      bool causal = true, std::int32_t window_left = -1) noexcept;
  static Status update_ragged(Invocation&, const BatchLayoutVersion&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Status update_layout(
      Invocation&, const BatchLayoutVersion&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<Program> prepare_program(
      const Context&, const NativePagedBatchProgram&, BatchLayoutVersion,
      const WorkloadContext&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<Program> prepare_program(
      std::shared_ptr<Runtime>, const NativePagedBatchProgram&, BatchLayoutVersion,
      const WorkloadContext&) noexcept;
  // Ordinary capacity Graphs keep their concrete Q/output extent while only
  // the owned layout's active rows enter attention. The suffix is zero-filled.
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<Program> prepare_paged_graph(
      std::shared_ptr<Runtime>, const NativePagedBatchProgram&, BatchLayoutVersion,
      std::uint32_t query_rows, const WorkloadContext&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<mlx::core::array> binding(
      const MemoryResource&,PagedArenaRead,LeaseImportOptions = {}) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<mlx::core::array> binding(
      const MemoryResource&,PagedRowCacheRead,LeaseImportOptions = {}) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<mlx::core::array> binding(
      const MemoryResource&,PagedArenaPendingRead,LeaseImportOptions = {}) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<mlx::core::array> binding(
      mem::MemoryResourceRef,PagedArenaRead,LeaseImportOptions = {}) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<mlx::core::array> binding(
      mem::MemoryResourceRef,PagedArenaPendingRead,LeaseImportOptions = {}) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<OwnedResult<Results>> submit(
      Invocation&,LeaseUseBatch) noexcept;
  // The original source/suffix owner retains this Update until native retirement.
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<OwnedResult<Results>> submit_pending(
      Invocation&,LeaseUseBatch,const CopyBatchUpdate&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Status wait_native(Invocation&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Status reuse_gate(Invocation&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static bool empty_cache_reads(const Invocation&) noexcept;
  // Checked idle batch reuse only: release prior input imports, keep the plan.
  HYLE_COMPUTE_ROUTE_APPLE_API static Status release_inputs(Invocation&) noexcept;
  // Original cold owner must exclude rebinding/submission through both calls.
  // Pure retirement/escape validation; closed/failed but retired cleanup is legal.
  HYLE_COMPUTE_ROUTE_APPLE_API static Status can_detach_inputs(
      const Invocation&, const compute::detail::ResultStorageOwner&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static void detach_inputs(Invocation&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Status qualify_source(const Context&,
      compute::detail::CompletionDependency)noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<MaterializedBatchInvocation> create(
      const Context&,Program,compute::MaterializedBatchProgram,const MemoryResource&,
      const MemoryResource&,MaterializedNativeBatchOptions)noexcept;
};
} // namespace hyle::compute::apple::detail
#endif
