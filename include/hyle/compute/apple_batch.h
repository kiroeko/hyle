#pragma once
#include "hyle/compute/apple.h"

#if defined(HYLE_COMPUTE_APPLE)
#include "hyle/compute/batch_materialize.h"
namespace hyle::compute::apple {
// Explicit GPU materialization reference. Cold traces the fixed original
// typed recipe into one original MLX Program; run performs no recipe traversal
// or planning. Bind Q/K/V and same-version uploaded indices/mask in that order.
// Original native inputs/results/slots retain their ownership and close rules.
// MLX retains its immutable tracing callable and a value copy of the finite
// recipe; that native metadata is additional to the common logical byte bound.
// This is managed replay with separately unknown native storage, not capture
// or native paged. Common logical byte caps include GQA/gather/Copy costs.
HYLE_COMPUTE_ROUTE_APPLE_API Result<Program> prepare_materialized_batch(
    const Context&,const MaterializedBatchProgram&,const WorkloadContext& = {}) noexcept;
} // namespace hyle::compute::apple
#endif
