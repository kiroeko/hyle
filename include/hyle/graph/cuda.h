#pragma once

#include "hyle/compute/cuda.h"
#include "hyle/compute/cuda_weights.h"
#include "hyle/graph/executable.h"
#include "hyle/graph/weight_binding.h"
#include "hyle/visibility.h"

namespace hyle::graph::cuda {

#if defined(HYLE_COMPUTE_CUDA)
// Uses the same fixed native implementation rules as CUDA Direct Compute.
HYLE_GRAPH_ROUTE_CUDA_API Result<compute::Context>
create_context(compute::cuda::DefaultContextOptions options = {}) noexcept;

// Cold binding of an immutable selected upload to the original Invocation.
// Retains its real backing and orders pending bytes at their actual readers.
HYLE_GRAPH_ROUTE_CUDA_API Status bind(Invocation &, ValueId,
    const compute::cuda::WeightBatch &, std::uint64_t id) noexcept;
// All bindings use the same immutable upload. Rejection preserves every old
// binding and its preparation; no producer wait occurs during this cold call.
HYLE_GRAPH_ROUTE_CUDA_API Status bind(Invocation &,
    std::span<const graph::WeightBinding>, const compute::cuda::WeightBatch &) noexcept;

// One GPU, one storage domain and one compute lane. The resource and constant
// backings are borrowed through the Executable, Invocations and Completions.
// Graph owns intermediate/workspace reuse; Providers retain their native
// artifacts. Advanced multi-lane use can supply graph::CompileEnvironment.
HYLE_GRAPH_ROUTE_CUDA_API Result<CompiledGraph>
compile(const Graph &graph, const compute::Context &context,
        mem::MemoryResourceRef device_resource,
        CompileOptions options = {}) noexcept;
// Retains the actual allocation domain, including failed cold preparation.
HYLE_GRAPH_ROUTE_CUDA_API Result<CompiledGraph>
compile(const Graph &, const compute::Context &,
        const compute::StorageOwner &, CompileOptions = {}) noexcept;
HYLE_GRAPH_ROUTE_CUDA_API Result<CompiledGraph>
compile(const Graph &, CompileEnvironment, CompileOptions = {}) noexcept;

// Cold binding/capture after binding every external tensor. PreferCapture
// reports the actual route through Invocation::replay_report(); RequireCapture
// rejects an uncapturable graph. Same-address input content may change between
// completed runs. Rebinding/reset invalidates preparation; a later cold
// prepare may retain recordings whose full physical arguments are unchanged.
// The Queue is borrowed until preparation is replaced and all runs
// terminate.
HYLE_GRAPH_ROUTE_CUDA_API Status prepare_replay(
    Invocation &invocation, compute::Queue &queue,
    ReplayOptions options = {.mode = ReplayMode::kPreferCapture,
                             .host_wait = HostWaitPolicy::kReject}) noexcept;
#endif

} // namespace hyle::graph::cuda
