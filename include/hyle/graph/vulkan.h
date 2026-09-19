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
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "hyle/compute/operation.h"
#include "hyle/error/result.h"
#include "hyle/graph/compile_report.h"

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

// Cold, independently owned facts; never retains Graph/Program/Buffer/Kernel,
// Context or custom payload. Unknown status means no attempt. Failure resets
// all successful facts and preserves only the returned error Status.
struct CompileReport final {
  Status status = Status::Unknown();
  compute::vulkan::Environment environment;
  std::vector<compute::TensorDesc> inputs;
  std::vector<compute::TensorDesc> outputs;
  std::vector<CompileNodeReport> nodes; // Live nodes in original ordinal order.
  std::size_t native_dispatch_count = 0;
  std::size_t managed_region_count = 0;
  // HYLE-planned boundary/output and native scratch capacities for one slot.
  // Provider-private storage, native alignment and driver objects are
  // additional; separate slots never share these mutable blocks.
  std::vector<std::size_t> invocation_storage_block_bytes;
  // Deduplicated by HYLE Buffer owner after cold packing and unused-owner
  // removal, not by logical value. Borrowed owners may alias external memory.
  std::size_t retained_constant_owner_count = 0;
  std::size_t retained_constant_capacity_bytes = 0;
  std::size_t owned_constant_allocation_bytes = 0;
  bool has_borrowed_constants = false;
  std::optional<std::size_t>
      opaque_storage_bytes; // Provider/driver: unknown at compile.
  std::string implementation_policy;
  std::string numerical_policy;
  std::string boundary_policy;
  std::string concurrency_policy;
};

struct CompileOptions final {
  CustomLowering custom;
  // Borrowed only during this call, reset before every attempt. Metadata OOM
  // rejects compilation. Null adds no report construction or retained state.
  CompileReport *report = nullptr;
  compute::WorkloadContext workload;
};

// Fixed pure DAG in admitted Compute domains -> native pipelines and an
// immutable recording recipe.
// Constants are ready immutable tensors in Graph constant order; inputs bind
// in Graph input order and generated outputs follow published order. The Graph
// may be destroyed after this call. Each Invocation owns independent commands,
// descriptors, completion, outputs and liveness-planned scratch. HYLE replay
// performs no planning, shader compilation or implementation search; managed
// providers retain their own graph encoding and private resource management.
//
// Built-ins share the Compute route's qualified dtype/shape domains. Custom
// schemas require the cold native lowering above. Symbolic/effect/control-edge
// graphs, unsupported alias schemas and input/constant-only published outputs
// are rejected explicitly. Typed TopK has two outputs; custom nodes require one.
// Native buffer readiness is a caller contract.
HYLE_GRAPH_ROUTE_VULKAN_API Result<compute::vulkan::Program>
compile(const compute::vulkan::Context &context, const Graph &graph,
        std::span<const compute::vulkan::Tensor> constants = {},
        CompileOptions options = {}) noexcept;

} // namespace hyle::graph::vulkan
#endif
