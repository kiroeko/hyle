#pragma once

#include "hyle/graph/executable.h"
#include "hyle/container/small_vector.h"

namespace hyle::graph::detail {

// Shared cold resource-footprint admission, including cross-fragment aliases.
using EffectBindings = container::SmallVector<EffectResourceBinding, 0, mem::AllocatorRef>;
HYLE_GRAPH_API Status collect_effect_bindings(
    const Graph &, std::span<const EffectResourceBinding>,
    std::span<const ConstantBinding>, mem::AllocatorRef, EffectBindings &) noexcept;
HYLE_GRAPH_API Status validate_effect_tensor_alias(
    std::span<const EffectResourceBinding>, mem::ConstBufferView) noexcept;

// Same-build native adapter admission. Ordinary graph::compile continues to
// require one Device. Only the typed CPU+CUDA adapter supplies these identities.
HYLE_GRAPH_API Result<CompiledGraph> compile_cpu_gpu(
    const Graph &graph, CompileEnvironment environment, CompileOptions options,
    compute::BackendId cpu_backend, compute::BackendId gpu_backend) noexcept;

} // namespace hyle::graph::detail
