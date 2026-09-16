#pragma once

#include "hyle/compute/cuda.h"
#include "hyle/graph/executable.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_GRAPH_ROUTE_CUDA_API
#elif defined(HYLE_GRAPH_ROUTE_CUDA_BUILD)
#define HYLE_GRAPH_ROUTE_CUDA_API HYLE_EXPORT
#else
#define HYLE_GRAPH_ROUTE_CUDA_API HYLE_IMPORT
#endif

namespace hyle::graph::cuda {

#if defined(HYLE_COMPUTE_CUDA)
// Uses the same fixed native implementation rules as CUDA Direct Compute.
HYLE_GRAPH_ROUTE_CUDA_API Result<compute::Context>
create_context(compute::cuda::DefaultContextOptions options = {}) noexcept;

// One GPU, one storage domain and one compute lane. The resource and constant
// backings are borrowed through the Executable, Invocations and Completions.
// Graph owns intermediate/workspace reuse; Providers retain their native
// artifacts. Advanced multi-lane use can supply graph::CompileEnvironment.
HYLE_GRAPH_ROUTE_CUDA_API Result<CompiledGraph>
compile(const Graph &graph, const compute::Context &context,
        mem::MemoryResourceRef device_resource,
        CompileOptions options = {}) noexcept;

// Cold binding/capture after binding every external tensor. PreferCapture
// reports the actual route through Invocation::replay_report(); RequireCapture
// rejects an uncapturable graph. Same-address input content may change between
// completed runs. Rebinding/reset invalidates preparation and captured
// pointers. The Queue is borrowed until preparation is replaced and all runs
// terminate.
HYLE_GRAPH_ROUTE_CUDA_API Status prepare_replay(
    Invocation &invocation, compute::Queue &queue,
    ReplayOptions options = {.mode = ReplayMode::kPreferCapture,
                             .host_wait = HostWaitPolicy::kReject}) noexcept;
#endif

} // namespace hyle::graph::cuda
