#pragma once
#include "hyle/compute/vulkan.h"

#if defined(HYLE_COMPUTE_VULKAN)
#include "hyle/compute/batch_materialize.h"
namespace hyle::compute::vulkan {
// Explicit GPU materialization reference over the original fixed native Plan.
// Cold preparation qualifies each typed step; run uses the original Program
// and independent native slots, with no new Queue/runtime or warm planning.
// Inputs: Q/K/V, same-version uploaded I32 indices and F32 additive mask.
// GQA/gather/F32/Copy byte caps are common logical upper bounds; native scratch
// and driver storage remain additional. Not a native-paged/capture claim.
HYLE_COMPUTE_ROUTE_VULKAN_API Result<Program> prepare_materialized_batch(
    const Context&,const MaterializedBatchProgram&,const WorkloadContext& = {}) noexcept;
} // namespace hyle::compute::vulkan
#endif
