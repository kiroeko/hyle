#pragma once

#include "hyle/graph/detail/execution_source.h"
#include "hyle/graph/executable.h"
#include "hyle/graph/region_provider.h"

namespace hyle::graph::detail {

// Same-build cold assembly of an existing native capability. The source is
// the actual capability on the original Context; it is not a fabricated
// kRegionProviderExtensionId or a second provider/Context. Discovery and
// compilation still enter that source's admission gate.
//
// interface/context are compile-local callbacks/configuration on that source,
// for shared and native capabilities alike. They do not replace its identity
// or admission gate, and are borrowed only until compile_with_adapters returns.
// Every returned prepared artifact must retain its own configuration and
// native owners. No adapter callback or context is consulted during submit.
struct RegionAdapter final {
  GraphRegionSource source;
  // Only a real shared Compute capability has a provider slot. Native
  // sources use UINT32_MAX; they never acquire a fabricated provider identity.
  std::uint32_t provider_index = UINT32_MAX;
  ExecutionTargetId target{};
  const RegionProviderVTable *interface = nullptr;
  void *context = nullptr;
};

// Uses the ordinary compiler and its existing budgets, placement search,
// immutable plan and execution slots. Original region implementations remain
// candidates alongside these capability adapters. This is an internal module
// assembly input, not another execution API for inference engines.
HYLE_GRAPH_API Result<CompiledGraph> compile_with_adapters(
    const Graph &, CompileEnvironment, CompileOptions,
    std::span<const RegionAdapter>) noexcept;

} // namespace hyle::graph::detail
