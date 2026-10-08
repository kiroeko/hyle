#pragma once

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/operation.h"
#include "hyle/compute/provider.h"
#include "default_context.h"

namespace hyle::compute::cuda::internal {

// Exact C2 complete-candidate rules. The normal roster remains the fallback
// for other environments, ports, resources and shapes; execution is unchanged.
HYLE_COMPUTE_ROUTE_CUDA_API RoutePolicyId
qualified_policy(Operation, std::span<const TensorDesc>,
                 std::span<const TensorDesc>) noexcept;
HYLE_COMPUTE_ROUTE_CUDA_API bool
qualified_environment(const Environment &) noexcept;
HYLE_COMPUTE_ROUTE_CUDA_API bool qualified_caller() noexcept;
HYLE_COMPUTE_ROUTE_CUDA_API bool qualified_recipe() noexcept;
HYLE_COMPUTE_ROUTE_CUDA_API Provider
with_qualified_defaults(Provider, DefaultScope) noexcept;

} // namespace hyle::compute::cuda::internal
