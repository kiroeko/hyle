#pragma once

#include "hyle/compute/cpu.h"
#include "hyle/graph/executable.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_GRAPH_ROUTE_CPU_API
#elif defined(HYLE_GRAPH_ROUTE_CPU_BUILD)
#define HYLE_GRAPH_ROUTE_CPU_API HYLE_EXPORT
#else
#define HYLE_GRAPH_ROUTE_CPU_API HYLE_IMPORT
#endif

namespace hyle::graph::cpu {

// CPU Graph with fixed mature region and leaf rules. Compile retains eligible
// ggml/oneDNN programs and uses the CPU leaf rules for remaining nodes.
// The options' allocator/executor are borrowed through all derived artifacts.
HYLE_GRAPH_ROUTE_CPU_API Result<compute::Context>
create_context(compute::cpu::DefaultContextOptions options = {}) noexcept;

} // namespace hyle::graph::cpu
