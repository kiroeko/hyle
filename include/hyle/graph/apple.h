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
#include <span>
#include <vector>

#include <mlx/array.h>
#include <mlx/stream.h>

#include "hyle/compute/operation.h"
#include "hyle/error/result.h"

namespace hyle::graph::apple {

// Cold extension for a pure custom Operation schema, e.g. fast::metal_kernel
// in the same MLX runtime. Called only while tracing the region. Its outputs
// must match the schema and use the supplied stream; captured state is
// immutable.
using CustomLowering = std::function<Result<std::vector<mlx::core::array>>(
    const compute::Operation &, std::span<const mlx::core::array>,
    mlx::core::Stream)>;

struct CompileOptions final {
  CustomLowering custom;
};

// Cold typed DAG -> one compiled MLX region. Constants are ready native owners
// in Graph constant order; the Program retains them. Bind inputs in Graph input
// order, read native outputs in published order. Graph may be destroyed after
// this call. Replays neither traverse the HYLE DAG nor call custom lowering.
//
// Admits fixed, dense, pure tensors; Q4 constants use apple::kAffineQ4Format.
// Symbolic dimensions, effects, explicit control edges and other encodings
// are rejected. Intermediates/fusion are managed by MLX; no shadow output copy.
HYLE_GRAPH_ROUTE_APPLE_API Result<compute::apple::Program>
compile(const compute::apple::Context &context, const Graph &graph,
        std::span<const mlx::core::array> constants = {},
        CompileOptions options = {}) noexcept;

} // namespace hyle::graph::apple
#endif
