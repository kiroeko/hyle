#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/container/small_vector.h"
#include "hyle/error/result.h"
#include "hyle/error/status.h"
#include "hyle/macros.h"
#include "hyle/mem/allocator.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_GRAPH_API
#elif defined(HYLE_GRAPH_BUILD)
#  define HYLE_GRAPH_API HYLE_EXPORT
#else
#  define HYLE_GRAPH_API HYLE_IMPORT
#endif

// graph-01 (step-1): task-dependency DAG with DIRTY-FLAG INCREMENTAL RECOMPUTE.
// North-star ② (CAD/BIM/CAE parametric modeling + simulation): a change to ONE input
// re-runs ONLY the affected downstream nodes; clean nodes reuse their cached result.
// "drag a slider -> recompute only the geometry it touches"; "change one boundary
// condition -> re-solve only the dependent mesh cells". See docs/architecture/
// graph-design.md + docs/milestones/graph-01-incremental-recompute.md.
//
// WHY THIS IS FROM-SCRATCH, NOT PORTED FROM clay: clay's plugins/taskgraph is a real
// parallel DAG executor, but it is ONE-SHOT dataflow -- every execution re-runs the
// WHOLE graph from the sources (graph_executor.cc:41, decrease-input-degree topological
// sweep). It has ZERO dirty/incremental/cache machinery (grep dirty|incremental|
// recompute|invalidate over the whole plugin = 0 hits). Incremental recompute -- the
// exact north-star-② hot path -- is the part clay never had. HYLE inherits the DAG-
// modeling + topological-sweep IDEA (~1k lines of shape) and writes the increment
// engine (dirty propagation + result cache + recompute-only-dirty) new.
//
// THE LOAD-BEARING DECISIONS (step-1):
//  ① NODE = fn-ptr + ctx CLOSURE, not a value-typed op. A node's compute is a
//     `ComputeFn(void* ctx)` that reads its inputs and writes its output THROUGH the
//     user's own memory (captured in ctx). The graph does NOT own or type a dataflow
//     value system (clay's data_type/socket/AsyncValue, ~750 lines, is exactly the
//     application-layer coupling graph-design.md says to strip). The graph owns only
//     the TOPOLOGY (who depends on whom) + the DIRTY/CACHE state. This mirrors HYLE's
//     type-erasure convention everywhere (Fiber::EntryFn, BlockingJob fn-ptrs) -- no
//     std::function, no vtable. "Cached result" here means "this node's ComputeFn has
//     already run since its last dirtying and its outputs are valid in user memory";
//     reuse = skip the call. The graph is the change-tracker, the user owns the bytes.
//  ② EDGES POINT BACKWARD IN CREATION ORDER -> insertion order IS a topological order.
//     add_node returns a NodeId; a node may depend ONLY on already-created (smaller)
//     NodeIds. So iterating NodeId 0..N-1 is always a valid topological sweep -- NO
//     per-run topological sort, NO cycle detection needed (a cycle is unconstructable).
//     This fits CAD/CAE bottom-up graph construction (leaves/inputs first, then the
//     ops that consume them) and is the standard build-time-DAG simplification
//     (Bazel's action graph, Salsa/Adapton demand graphs are built the same way).
//  ③ DIRTY PROPAGATION IS EAGER ON mark_dirty, RECOMPUTE IS LAZY ON evaluate.
//     mark_dirty(n) floods the dirty flag DOWNSTREAM (n and everything transitively
//     depending on it) immediately -- cheap, just sets flags. Actual recompute happens
//     in evaluate(): a single forward sweep runs ComputeFn only for dirty nodes (their
//     inputs are guaranteed already-fresh because dependencies have smaller ids and were
//     visited first), clearing the flag. Clean nodes are skipped entirely = the
//     incremental win. This eager-mark / lazy-eval split is the demand-driven
//     incremental model (Salsa "durability", Adapton) minus the demand-tracking graph
//     rewiring (step-1 recomputes ALL dirty; "recompute only what a specific output
//     demands" is a later refinement, see milestone step-2/backlog).
//
// STEP-1 SCOPE: SERIAL evaluate on the calling thread (pure CPU, NO sched dependency).
// This proves the incremental-recompute SEMANTICS (the clay-zero-asset core) in
// isolation. PARALLEL evaluate -- independent dirty nodes at the same topological
// "level" dispatched onto the sched work-stealing pool, awaited via when_all -- is
// step-2 (graph-design.md §3: graph is a pure orchestration layer over sched+compute).
// A node's ComputeFn may itself launch a compute kernel; that is orthogonal to whether
// the GRAPH walks serially or in parallel.

namespace hyle::graph {

// Identifies a node within one TaskGraph. Stable for the graph's lifetime. The value
// is the creation index (see decision ②): dependencies must have strictly smaller ids.
enum class NodeId : std::uint32_t {};

// A node's compute closure: read inputs + write this node's output, all through `ctx`
// (user memory). Runs during evaluate() when the node is dirty. Must be no-throw (the
// whole library is no-throw; a failure is reported by RETURNING non-OK Status, not
// throwing). Returning an error aborts the evaluate() sweep and surfaces to the caller;
// the failed node stays dirty (so a later evaluate() retries it).
using ComputeFn = Status (*)(void* ctx) noexcept;

// A task-dependency DAG with incremental recompute. Build it once (add_node with
// dependencies), then interleave mark_dirty(input) + evaluate() as inputs change --
// each evaluate re-runs only the dirty subgraph. Move-only; owns its topology + dirty/
// cache state (all from `alloc`), owns NONE of the user memory the ComputeFns touch.
//
// NOT thread-safe: one owner drives build + mark_dirty + evaluate. (Parallel *node
// execution* within a single evaluate is step-2 and stays internally synchronized;
// concurrent external mutation of the same graph is out of scope, like every DAG lib.)
// NOTE on the export macro: TaskGraph holds non-POD members (container::SmallVector), so
// the HYLE_<MOD>_API macro goes on each out-of-line METHOD, NOT on the class (coding-
// conventions §4 / cheatsheet §4). Marking the whole class dllexport would force every
// SmallVector member to also be dll-interface (MSVC C4251) -- the members are an internal
// implementation detail, not part of the ABI a client links against. Inline methods
// (add_source) need no marker.
class TaskGraph {
 public:
  HYLE_GRAPH_API explicit TaskGraph(mem::Allocator alloc) noexcept;
  HYLE_GRAPH_API ~TaskGraph();
  HYLE_DISABLE_COPY(TaskGraph);
  HYLE_GRAPH_API TaskGraph(TaskGraph&& o) noexcept;
  HYLE_GRAPH_API TaskGraph& operator=(TaskGraph&& o) noexcept;

  // Add a node whose compute is `fn(ctx)`. `deps`/`num_deps` list the nodes it depends
  // on -- each MUST be an already-created id (decision ②: strictly < the new id), else
  // fatal assert. A freshly added node starts DIRTY (never computed). Returns its id.
  //   CONTRACT (caller bug -> fatal assert): fn non-null; every dep < next id.
  HYLE_GRAPH_API NodeId add_node(ComputeFn fn, void* ctx, const NodeId* deps,
                                 std::size_t num_deps) noexcept;

  // Convenience: a source node (no dependencies) -- a graph input / leaf.
  NodeId add_source(ComputeFn fn, void* ctx) noexcept {
    return add_node(fn, ctx, nullptr, 0);
  }

  // Mark `n` (and everything transitively depending on it) dirty: their ComputeFn will
  // re-run on the next evaluate(). Call this when the data a node reads has changed
  // (e.g. a user edited a parameter feeding a source node). Eager flag flood (decision
  // ③) -- O(affected subgraph), no compute. Idempotent.
  //   CONTRACT: n is a valid id of this graph.
  HYLE_GRAPH_API void mark_dirty(NodeId n) noexcept;

  // Run every DIRTY node once, in topological (creation) order, clearing its flag.
  // Clean nodes are SKIPPED (their cached output in user memory is reused) -- the
  // incremental win. A node's ComputeFn sees its dependencies already fresh (smaller
  // ids ran first). If any ComputeFn returns non-OK, the sweep STOPS and returns that
  // Status; the failed node stays dirty (retried next evaluate). All-clean = no-op.
  HYLE_GRAPH_API Status evaluate() noexcept;

  // Diagnostics.
  HYLE_GRAPH_API std::size_t num_nodes() const noexcept;
  HYLE_GRAPH_API bool is_dirty(NodeId n) const noexcept;  // n valid (fatal assert)

  // ---- internal seam for the parallel evaluator (graph-01 step-2) -----------
  // parallel.h's evaluate_parallel lives in the SAME module (hyle_graph) but a
  // separate, sched-gated TU (parallel_eval.cc) so this core header stays sched-free
  // and builds anywhere (mirrors compute's gpu_await split). It needs read access to
  // the topology + dirty state and to clear a node's dirty flag. These accessors are
  // the minimal seam -- NOT a public API (a consumer uses evaluate()/evaluate_parallel,
  // never these). Kept as methods (not friend) to avoid coupling this header to a
  // sched type. Trivial + inline; zero ABI weight.
  std::uint32_t node_count_() const noexcept {
    return static_cast<std::uint32_t>(fn_.size());
  }
  bool dirty_at_(std::uint32_t i) const noexcept { return dirty_[i]; }
  void clear_dirty_(std::uint32_t i) noexcept { dirty_[i] = false; }
  void set_dirty_(std::uint32_t i) noexcept { dirty_[i] = true; }
  Status run_node_(std::uint32_t i) const noexcept { return fn_[i](ctx_[i]); }
  // node i's dependency slice = dep_pool_[dep_begin_(i) .. dep_end_(i)).
  std::uint32_t dep_begin_(std::uint32_t i) const noexcept { return dep_off_[i]; }
  std::uint32_t dep_end_(std::uint32_t i) const noexcept { return dep_off_[i + 1]; }
  std::uint32_t dep_pool_at_(std::uint32_t k) const noexcept {
    return static_cast<std::uint32_t>(dep_pool_[k]);
  }

 private:
  // Storage is STRUCT-OF-ARRAYS with a CSR (compressed-sparse-row) dependency list --
  // not an array of Node structs. Reason: container::SmallVector requires a trivially-
  // copyable element (it relocates by memcpy), so a Node holding its own SmallVector of
  // deps is ill-formed. SoA sidesteps that AND is the better layout for the evaluate
  // sweep (the hot loop touches fn/ctx/dirty contiguously; deps are read once per dirty
  // node). All deps live flattened in `dep_pool_`; node i owns the slice
  // [dep_off_[i], dep_off_[i+1]). Since deps are known at add_node time and ids only
  // grow (decision ②), the CSR is built by plain append -- no back-patching.
  container::SmallVector<ComputeFn, 8, mem::Allocator> fn_;    // node i's compute
  container::SmallVector<void*, 8, mem::Allocator> ctx_;       // node i's closure ctx
  container::SmallVector<bool, 8, mem::Allocator> dirty_;      // node i needs recompute
  container::SmallVector<NodeId, 16, mem::Allocator> dep_pool_;      // flattened deps of all nodes
  container::SmallVector<std::uint32_t, 8, mem::Allocator> dep_off_;  // node i's deps = dep_pool_[dep_off_[i]..dep_off_[i+1])
  mem::Allocator alloc_;
};

}  // namespace hyle::graph
