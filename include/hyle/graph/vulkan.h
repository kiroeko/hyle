#pragma once

#include "hyle/compute/vulkan.h"
#include "hyle/graph/graph.h"
#include "hyle/graph/execution_source.h"
#include "hyle/graph/weight_binding.h"
#include "hyle/visibility.h"

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
#include "hyle/compute/vulkan_batch.h"
#include "hyle/compute/vulkan_weights.h"

namespace hyle::graph { class Invocation; struct QueueBinding; }

namespace hyle::graph::vulkan {
// Cold immutable ownership binding to the original Invocation. No upload wait
// occurs here; the original replay host-wait policy orders the native producer
// at its real readers. Batch/native owners are retained; resource and allocator
// objects remain borrowed through all retained storage and work.
HYLE_GRAPH_ROUTE_VULKAN_API Status bind(Invocation &, ValueId,
    const compute::vulkan::WeightBatch &, std::uint64_t id) noexcept;
// Resolve all banks and native storage owners before one atomic input commit.
HYLE_GRAPH_ROUTE_VULKAN_API Status bind(Invocation &,
    std::span<const graph::WeightBinding>, const compute::vulkan::WeightBatch &) noexcept;

// Typed pending page frames bind through batch::bind after cold batch::prepare
// reserves source capacity. This transfers the original native F2 controls to
// the same Graph submission; rejection before acceptance leaves update owned
// by the caller. Recovery requires actual source/suffix retirement and release
// of the original public outcome and result observations.
HYLE_GRAPH_ROUTE_VULKAN_API Result<compute::Completion> submit(Invocation &,
    std::span<const QueueBinding>, compute::vulkan::CopyBatchUpdate &&) noexcept;
HYLE_GRAPH_ROUTE_VULKAN_API Result<compute::vulkan::CopyBatchUpdate>
take_update(Invocation &) noexcept;

HYLE_GRAPH_ROUTE_VULKAN_API Result<compute::vulkan::MaterializedBatchInvocation>
prepare_materialized_batch(const compute::vulkan::Context&,compute::MaterializedBatchProgram,
    const compute::vulkan::MemoryResource&device,const compute::vulkan::MemoryResource&staging,
    compute::MaterializedNativeBatchOptions,const compute::WorkloadContext& = {})noexcept;

// One native shader may implement a complete fused custom fragment. Storage
// bindings follow schema input order, then its single generated output. The
// shader must honor those descriptions and the Context's enabled features.
struct CustomDispatch final {
  compute::vulkan::Kernel kernel;
  std::array<std::uint32_t, 3> groups{1, 1, 1};
  std::vector<std::byte> push_constants;
};

// Called only during cold compilation. Capture the actual Context/Kernel and
// any other required owner explicitly. The resulting native Program retains
// pipelines; it never retains or invokes this callback during replay.
using CustomLowering = std::function<Result<CustomDispatch>(
    const compute::Operation &,
    std::span<const compute::TensorDesc>, const compute::TensorDesc &)>;

struct SourceOptions final {
  // Retained by this configured Source. Captured borrows must remain valid
  // through every cold compile using it; no public Context address is borrowed
  // implicitly. Prepared native programs keep only the resulting pipelines.
  CustomLowering custom{};
  mem::AllocatorRef allocator = mem::system_allocator();
};
HYLE_GRAPH_ROUTE_VULKAN_API Result<ExecutionSource> execution_source(
    const compute::vulkan::Context &, SourceOptions = {}) noexcept;

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
  // Cold exact binding; no runtime JIT or general-shape claim.
  std::span<const ShapeBinding> shape{};
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
// schemas require the cold native lowering above. Symbolic graphs require exact
// bindings. Finite tables use the common Graph compile entry. Pure control endpoints separate
// native regions and retain predecessor outputs through original barriers.
// Effects, unsupported alias schemas and input/constant-only published outputs
// are rejected explicitly. TopK and qualified recurrent/categorical schemas may
// have two outputs; arbitrary custom lowering still requires one.
// Native buffer readiness is a caller contract.
HYLE_GRAPH_ROUTE_VULKAN_API Result<compute::vulkan::Program>
compile(const compute::vulkan::Context &context, const Graph &graph,
        std::span<const compute::vulkan::Tensor> constants = {},
        CompileOptions options = {}) noexcept;

} // namespace hyle::graph::vulkan
#endif
