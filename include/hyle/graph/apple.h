#pragma once

#include "hyle/compute/apple.h"
#include "hyle/graph/graph.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_GRAPH_ROUTE_APPLE_API
#elif defined(HYLE_GRAPH_ROUTE_APPLE_BUILD)
#define HYLE_GRAPH_ROUTE_APPLE_API HYLE_EXPORT
#else
#define HYLE_GRAPH_ROUTE_APPLE_API HYLE_IMPORT
#endif

#if defined(HYLE_COMPUTE_APPLE)
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <mlx/array.h>
#include <mlx/stream.h>

#include "hyle/compute/operation.h"
#include "hyle/error/result.h"
#include "hyle/graph/compile_report.h"

namespace hyle::graph::apple {

// Cold extension for a pure custom Operation schema, e.g. fast::metal_kernel
// in the same MLX runtime. Called only while tracing the region. Its outputs
// must match the schema and use the supplied stream; captured state is
// immutable.
using CustomLowering = std::function<Result<std::vector<mlx::core::array>>(
    const compute::Operation &, std::span<const mlx::core::array>,
    mlx::core::Stream)>;

// Independently owned cold diagnostics. No Graph, Program, MLX array, custom
// payload or Context is retained. Copies use the consumer's ordinary C++
// allocator/STL contract. An unattempted report has Unknown status; a rejected
// compilation contains only its failure Status, never stale success facts.
struct CompileReport final {
  Status status = Status::Unknown();
  compute::apple::Environment environment;
  std::vector<compute::TensorDesc> inputs;
  std::vector<compute::TensorDesc> outputs;
  std::vector<CompileNodeReport> nodes; // Live nodes in original ordinal order.
  std::size_t declared_constant_count = 0;
  // Sum of declared logical views, including repeated bindings. This is not
  // unique resident memory, RSS or a bound on MLX's private allocator.
  std::size_t logical_constant_bytes = 0;
  std::optional<std::size_t> private_storage_bytes; // MLX/driver: unknown.
  std::string implementation_policy;
  std::string numerical_policy;
  std::string boundary_policy;
  std::string concurrency_policy;
};

struct CompileOptions final {
  CustomLowering custom;
  // Borrowed only until compile returns; reset before every attempt. Metadata
  // allocation failure rejects compilation. Null adds no report construction.
  CompileReport *report = nullptr;
  compute::WorkloadContext workload;
};

// Cold typed DAG -> compiled MLX regions and admitted whole ggml Metal regions.
// Pure MLX graphs retain one compiled region. Constants are ready native owners
// in Graph constant order; the Program retains them. Bind inputs in Graph input
// order, read native outputs in published order. Graph may be destroyed after
// this call. Replays neither traverse the HYLE DAG nor call custom lowering.
//
// Admits fixed, dense, pure tensors; MLX Q4 constants use
// apple::kAffineQ4Format. Public GGUF block descriptions are admitted only as
// native ggml Metal MatMul/GroupedMatMul weights and bind as uint8 carriers.
// Symbolic dimensions, effects, explicit control edges and other encodings are
// rejected. Each provider manages its own region's intermediates/fusion;
// cross-provider boundaries use checked completion waits, no shadow output copy.
HYLE_GRAPH_ROUTE_APPLE_API Result<compute::apple::Program>
compile(const compute::apple::Context &context, const Graph &graph,
        std::span<const mlx::core::array> constants = {},
        CompileOptions options = {}) noexcept;

} // namespace hyle::graph::apple
#endif
