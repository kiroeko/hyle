#pragma once

#include <cstddef>

#include "hyle/compute/provider.h"
#include "hyle/compute/visibility.h"
#include "hyle/error/status.h"
#include "hyle/mem/allocator.h"

namespace hyle::compute::onednn::internal {

inline constexpr std::size_t kBrgemmWorkspaceAlignment = 64;

struct BrgemmShape final {
  std::size_t rows = 0;
  std::size_t inner = 0;
  std::size_t columns = 0;
};

struct BrgemmTaskRange final {
  std::size_t first_column = 0;
  std::size_t column_count = 0;
};

using BrgemmPairPostTaskFn = void (*)(void *context, std::size_t row_count,
                                      std::size_t leading_columns,
                                      BrgemmTaskRange range) noexcept;

// Opaque, allocator-owned oneDNN JIT artifact. This header is an uninstalled
// implementation seam shared by the direct Compute provider and Graph's
// oneDNN region factor; no dnnl type becomes part of either product API.
struct BrgemmPlan;

HYLE_COMPUTE_PROVIDER_ONEDNN_API std::size_t
recommended_brgemm_task_count(std::size_t columns,
                              std::size_t concurrency) noexcept;

HYLE_COMPUTE_PROVIDER_ONEDNN_API Status
create_brgemm_plan(BrgemmShape shape, std::size_t task_count,
                   mem::AllocatorRef allocator, BrgemmPlan **output) noexcept;

HYLE_COMPUTE_PROVIDER_ONEDNN_API void
destroy_brgemm_plan(BrgemmPlan *plan, mem::AllocatorRef allocator) noexcept;

HYLE_COMPUTE_PROVIDER_ONEDNN_API std::size_t
brgemm_task_count(const BrgemmPlan *plan) noexcept;

HYLE_COMPUTE_PROVIDER_ONEDNN_API std::size_t
brgemm_scratchpad_size(const BrgemmPlan *plan) noexcept;

// Invocation construction validates a mapped workspace once. Execute then
// receives the same stable base address and adds no per-task size/alignment
// checks or allocations.
HYLE_COMPUTE_PROVIDER_ONEDNN_API Status validate_brgemm_scratchpad(
    const BrgemmPlan *plan, const void *scratchpad, std::size_t size) noexcept;

HYLE_COMPUTE_PROVIDER_ONEDNN_API Status execute_brgemm(
    const BrgemmPlan *plan, ParallelExecutorRef executor, const float *left,
    const float *right, float *output, std::byte *scratchpad) noexcept;

// Executes two projections with a shared left matrix and one executor
// fan-out. Each task sets oneDNN's hardware context once, executes both
// kernels with the same scratch slice, releases the context, then optionally
// invokes the non-failing post callback for that output column range.
HYLE_COMPUTE_PROVIDER_ONEDNN_API Status execute_brgemm_pair(
    const BrgemmPlan *plan, ParallelExecutorRef executor, const float *left,
    const float *first_right, const float *second_right, float *first_output,
    float *second_output, std::byte *scratchpad,
    BrgemmPairPostTaskFn post_task = nullptr,
    void *post_context = nullptr) noexcept;

} // namespace hyle::compute::onednn::internal
