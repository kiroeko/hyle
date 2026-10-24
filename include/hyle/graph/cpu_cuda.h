#pragma once

#include <cstddef>

#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/executable.h"
#include "hyle/visibility.h"

namespace hyle::graph::cpu_cuda {

inline constexpr ExecutionTargetId kCudaTarget{0};
inline constexpr ExecutionTargetId kCpuTarget{1};
inline constexpr StorageDomainId kDeviceStorage{0};
inline constexpr StorageDomainId kHostStorage{1};
inline constexpr StorageDomainId kStagingStorage{2};

// Explicit execution endpoints. Resources and their CUDA Context are borrowed
// through every compiled artifact, invocation and completion. CPU math uses
// ordinary host-accessible storage. Only crossing values use the exact-context
// CUDA pinned staging domain; its workspace limit is per Invocation.
struct Environment final {
  const compute::Context *cpu = nullptr;
  const compute::Context *cuda = nullptr;
  mem::MemoryResourceRef host = mem::system_resource();
  const compute::cuda::MemoryResource *staging = nullptr;
  const compute::cuda::MemoryResource *device = nullptr;
  std::size_t host_workspace_budget_bytes = SIZE_MAX;
  std::size_t device_workspace_budget_bytes = SIZE_MAX;
  std::size_t staging_workspace_budget_bytes = SIZE_MAX;
};

// CUDA is the ordinary target; required CPU nodes use kCpuTarget in existing
// NodePlacement. Providers retain their original programs/regions and fixed
// cold rules. Ordinary compile() admits the same sources by their actual
// capabilities; this helper assembles the CPU/CUDA resource environment.
HYLE_GRAPH_ROUTE_CPU_CUDA_API Result<CompiledGraph>
compile(const Graph &graph, Environment environment,
        CompileOptions options = {}) noexcept;

} // namespace hyle::graph::cpu_cuda
