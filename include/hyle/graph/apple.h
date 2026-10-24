#pragma once

#include "hyle/compute/apple.h"
#include "hyle/graph/graph.h"
#include "hyle/graph/weight_binding.h"
#include "hyle/visibility.h"

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
#include "hyle/graph/execution_source.h"
#include "hyle/compute/apple_batch.h"
#include "hyle/compute/apple_weights.h"

namespace hyle::graph { class Invocation; struct QueueBinding; }
namespace hyle::graph::apple {
// Cold immutable ownership binding to the original Invocation. Binding retains
// the original batch and metadata without waiting for its upload. The first
// actual reader follows the original replay HostWaitPolicy.
HYLE_GRAPH_ROUTE_APPLE_API Status bind(Invocation &, ValueId,
    const compute::apple::WeightBatch &, std::uint64_t id) noexcept;
// Resolve all banks and native owners before one atomic input commit.
HYLE_GRAPH_ROUTE_APPLE_API Status bind(Invocation &,
    std::span<const graph::WeightBinding>, const compute::apple::WeightBatch &) noexcept;

// Typed pending page updates use the original Invocation and source controls.
// Rejection before acceptance leaves the update with its caller; recovery
// requires actual source/suffix retirement and release of public observations.
HYLE_GRAPH_ROUTE_APPLE_API Result<compute::Completion> submit(Invocation &,
    std::span<const QueueBinding>, compute::apple::CopyBatchUpdate &&) noexcept;
HYLE_GRAPH_ROUTE_APPLE_API Result<compute::apple::CopyBatchUpdate>
take_update(Invocation &) noexcept;

HYLE_GRAPH_ROUTE_APPLE_API Result<compute::apple::MaterializedBatchInvocation>
prepare_materialized_batch(const compute::apple::Context&,compute::MaterializedBatchProgram,
    const compute::apple::MemoryResource&device,const compute::apple::MemoryResource&staging,
    compute::MaterializedNativeBatchOptions,const compute::WorkloadContext& = {})noexcept;

// Cold extension for a pure custom Operation schema, e.g. fast::metal_kernel
// in the same MLX runtime. Called only while tracing the region. Its outputs
// must match the schema and use the supplied stream; captured state is
// immutable.
using CustomLowering = std::function<Result<std::vector<mlx::core::array>>(
    const compute::Operation &, std::span<const mlx::core::array>,
    mlx::core::Stream)>;

struct SourceOptions final {
  // Retained by this configured Source. Captured borrows must remain valid
  // through its cold compiles; no public Context address is borrowed implicitly.
  // Original native programs retain the resulting compiled MLX callable.
  CustomLowering custom{};
  mem::AllocatorRef allocator = mem::system_allocator();
};
HYLE_GRAPH_ROUTE_APPLE_API Result<ExecutionSource> execution_source(
    const compute::apple::Context &, SourceOptions = {}) noexcept;

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
  // Unique ready F32 arrays derived from immutable constants during cold
  // preparation. Original logical bindings/bytes above remain unchanged.
  // Counts requested payload bytes, not MLX allocation/descriptor overhead.
  std::size_t prepared_constant_count = 0;
  std::size_t prepared_constant_bytes = 0;
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
  // Composition boundary: original MLX contiguous lowering stays inside the
  // prepared native region. No CPU layout loop or per-run recompilation.
  bool contiguous_outputs = false;
  // Cold exact binding; no runtime JIT or general-shape claim.
  std::span<const ShapeBinding> shape{};
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
// Symbolic dimensions require exact shape bindings. Common graph::compile with
// kFiniteExact prepares a finite table through the actual ExecutionSource.
// Pure control endpoints retain predecessor outputs and split
// managed regions with explicit completion waits. Effects and other unsupported
// encodings are rejected. Each provider manages its own intermediates/fusion;
// cross-provider boundaries use checked completion waits, no shadow output copy.
HYLE_GRAPH_ROUTE_APPLE_API Result<compute::apple::Program>
compile(const compute::apple::Context &context, const Graph &graph,
        std::span<const mlx::core::array> constants = {},
        CompileOptions options = {}) noexcept;

} // namespace hyle::graph::apple
#endif
