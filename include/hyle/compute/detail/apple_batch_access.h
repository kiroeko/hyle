#pragma once
#include "hyle/compute/apple_memory.h"
#if defined(HYLE_COMPUTE_APPLE)
#include <cstdint>
#include <functional>
#include <vector>
#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/paged_arena.h"
#include "hyle/compute/apple_batch.h"
namespace hyle::compute::apple::detail {
// Trusted fixed-batch adapters only. Capacity is an address view; the frame
// retains exact page permissions. Submit must admit every actual page guard.
struct PagedBatchAccess final {
  using RuntimeFunction = std::function<Result<std::vector<mlx::core::array>>(
      const BatchLayoutVersion&, const std::vector<mlx::core::array>&, mlx::core::Stream)>;
  // Only cold-qualified pure paged/ragged builders enter this managed bridge.
  static Result<Program> prepare_runtime_program(const Context&, BatchLayoutVersion,
      std::span<const InputSpec>, RuntimeFunction, const WorkloadContext&,
      bool causal = true, std::int32_t window_left = -1) noexcept;
  static Status update_ragged(Invocation&, const BatchLayoutVersion&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Status update_layout(
      Invocation&, const BatchLayoutVersion&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<Program> prepare_program(
      const Context&, const NativePagedBatchProgram&, BatchLayoutVersion,
      const WorkloadContext&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<mlx::core::array> binding(
      const MemoryResource&,PagedArenaRead,LeaseImportOptions = {}) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<mlx::core::array> binding(
      const MemoryResource&,PagedArenaPendingRead,LeaseImportOptions = {}) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<OwnedResult<Results>> submit(
      Invocation&,LeaseUseBatch) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Status wait_native(Invocation&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Status reuse_gate(Invocation&) noexcept;
  // Checked idle batch reuse only: release prior input imports, keep the plan.
  HYLE_COMPUTE_ROUTE_APPLE_API static Status release_inputs(Invocation&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Status qualify_source(const Context&,
      compute::detail::CompletionDependency)noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<MaterializedBatchInvocation> create(
      const Context&,Program,compute::MaterializedBatchProgram,const MemoryResource&,
      const MemoryResource&,MaterializedNativeBatchOptions)noexcept;
};
} // namespace hyle::compute::apple::detail
#endif
