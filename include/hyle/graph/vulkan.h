#pragma once

#include "hyle/compute/vulkan.h"
#include "hyle/graph/graph.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_GRAPH_ROUTE_VULKAN_API
#elif defined(HYLE_GRAPH_ROUTE_VULKAN_BUILD)
#define HYLE_GRAPH_ROUTE_VULKAN_API HYLE_EXPORT
#else
#define HYLE_GRAPH_ROUTE_VULKAN_API HYLE_IMPORT
#endif

#if defined(HYLE_COMPUTE_VULKAN)
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "hyle/compute/operation.h"
#include "hyle/error/result.h"

namespace hyle::graph::vulkan {

// One native shader may implement a complete fused custom fragment. Storage
// bindings follow schema input order, then its single generated output. The
// shader must honor those descriptions and the Context's enabled features.
struct CustomDispatch final {
  compute::vulkan::Kernel kernel;
  std::array<std::uint32_t, 3> groups{1, 1, 1};
  std::vector<std::byte> push_constants;
};

// Called once per live custom node during cold compilation. The resulting
// Program retains native pipelines; it never retains or invokes this callback.
using CustomLowering = std::function<Result<CustomDispatch>(
    const compute::vulkan::Context &, const compute::Operation &,
    std::span<const compute::TensorDesc>, const compute::TensorDesc &)>;

struct CompileOptions final {
  CustomLowering custom;
};

// Fixed pure F32 DAG -> native pipelines and an immutable recording recipe.
// Constants are ready immutable tensors in Graph constant order; inputs bind
// in Graph input order and generated outputs follow published order. The Graph
// may be destroyed after this call. Each Invocation owns independent commands,
// descriptors, fence, outputs and liveness-planned scratch; replay has no DAG
// traversal, shader compilation or implementation search.
//
// Built-ins admit the Compute route's MatMul and SiLU-gate domains. Custom
// schemas require the cold native lowering above. Symbolic/effect/control-edge
// graphs, aliasing schemas, multi-output nodes and input/constant-only
// published outputs are rejected explicitly. Native buffer readiness is a
// caller contract.
HYLE_GRAPH_ROUTE_VULKAN_API Result<compute::vulkan::Program>
compile(const compute::vulkan::Context &context, const Graph &graph,
        std::span<const compute::vulkan::Tensor> constants = {},
        CompileOptions options = {}) noexcept;

} // namespace hyle::graph::vulkan
#endif
