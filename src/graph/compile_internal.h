#pragma once

#include "hyle/graph/executable.h"

namespace hyle::graph::detail {

// Same-build native adapter admission. Ordinary graph::compile continues to
// require one Device. Only the typed CPU+CUDA adapter supplies these identities.
HYLE_GRAPH_API Result<CompiledGraph> compile_cpu_gpu(
    const Graph &graph, CompileEnvironment environment, CompileOptions options,
    compute::BackendId cpu_backend, compute::BackendId gpu_backend) noexcept;

} // namespace hyle::graph::detail
