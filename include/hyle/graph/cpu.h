#pragma once

#include "hyle/compute/cpu.h"
#include "hyle/graph/executable.h"
#include "hyle/visibility.h"

namespace hyle::graph::cpu {

// CPU Graph with fixed mature region and leaf rules. Compile retains eligible
// ggml/oneDNN programs and uses the CPU leaf rules for remaining nodes.
// The options' allocator/executor are borrowed through all derived artifacts.
HYLE_GRAPH_ROUTE_CPU_API Result<compute::Context>
create_context(compute::cpu::DefaultContextOptions options = {}) noexcept;

// Cold assembly discovers the actual native capabilities on the retained
// Context, alongside its ordinary region and leaf implementations.
HYLE_GRAPH_ROUTE_CPU_API Result<CompiledGraph>
compile(const Graph &, const compute::Context &, CompileOptions = {}) noexcept;
HYLE_GRAPH_ROUTE_CPU_API Result<CompiledGraph>
compile(const Graph &, CompileEnvironment, CompileOptions = {}) noexcept;

} // namespace hyle::graph::cpu
