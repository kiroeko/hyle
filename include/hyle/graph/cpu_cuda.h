#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/executable.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_GRAPH_ROUTE_CPU_CUDA_API
#elif defined(HYLE_GRAPH_ROUTE_CPU_CUDA_BUILD)
#define HYLE_GRAPH_ROUTE_CPU_CUDA_API HYLE_EXPORT
#else
#define HYLE_GRAPH_ROUTE_CPU_CUDA_API HYLE_IMPORT
#endif

namespace hyle::graph::cpu_cuda {

inline constexpr ExecutionTargetId kCudaTarget{0};
inline constexpr ExecutionTargetId kCpuTarget{1};
inline constexpr StorageDomainId kDeviceStorage{0};
inline constexpr StorageDomainId kHostStorage{1};

// Explicit execution endpoints. Resources and their CUDA Context are borrowed
// through every compiled artifact, invocation and completion. Host storage is
// exact-context CUDA pinned memory; pageable inputs need explicit staging.
struct Environment final {
  const compute::Context *cpu = nullptr;
  const compute::Context *cuda = nullptr;
  const compute::cuda::MemoryResource *host = nullptr;
  const compute::cuda::MemoryResource *device = nullptr;
  std::size_t host_workspace_budget_bytes = SIZE_MAX;
  std::size_t device_workspace_budget_bytes = SIZE_MAX;
};

// CUDA is the ordinary target; required CPU nodes use kCpuTarget in existing
// NodePlacement. Providers retain their original programs/regions and fixed
// cold rules. This never enables cross-device execution in ordinary compile().
HYLE_GRAPH_ROUTE_CPU_CUDA_API Result<CompiledGraph>
compile(const Graph &graph, Environment environment,
        CompileOptions options = {}) noexcept;

} // namespace hyle::graph::cpu_cuda
