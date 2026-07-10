#pragma once

#include "hyle/error/status.h"
#include "hyle/graph/taskgraph.h"
#include "hyle/mem/allocator.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_GRAPH_API
#elif defined(HYLE_GRAPH_BUILD)
#  define HYLE_GRAPH_API HYLE_EXPORT
#else
#  define HYLE_GRAPH_API HYLE_IMPORT
#endif

// graph-01 (step-2): PARALLEL incremental evaluate over the sched work-stealing pool.
// step-1's TaskGraph::evaluate() runs dirty nodes SERIALLY on the calling thread; this
// runs INDEPENDENT dirty nodes CONCURRENTLY, which is the whole point of a DAG executor
// (graph-design.md §3: graph is a pure orchestration layer -- it OWNS no threads, it
// dispatches nodes onto sched's workers). A CAD parametric recompute or a CAE mesh solve
// touches thousands of independent cells/elements per level; running them one-at-a-time
// wastes the machine.
//
// WHY THIS IS A SEPARATE, sched-GATED HEADER (mirrors compute's gpu_await.h split): the
// core taskgraph.h depends only on error/mem/container and builds ANYWHERE (no fiber asm
// needed). Parallel evaluate needs sched (spawn_task + when_all + fiber suspend), which
// only exists where fcontext asm exists. Keeping it here means a platform without
// fcontext still gets the serial core; the parallel path is additive.
//
// EXECUTION MODEL = LEVEL-BY-LEVEL FORK-JOIN (the simplest correct parallelization; a
// finer dataflow wavefront is a later refinement):
//   - A node's LEVEL = 1 + max(level of its deps); sources are level 0. Because deps
//     always have smaller ids (taskgraph decision ②), one forward id-sweep computes
//     every level. Two nodes at the SAME level provably have no edge between them (an
//     edge would force different levels), so they are INDEPENDENT and safe to run
//     concurrently.
//   - For each level 0,1,2,...: spawn_task every DIRTY node at that level (they fan out
//     across the workers), when_all them (a fiber-aware barrier), then advance. A node's
//     deps are all at strictly lower levels, so by the time its level runs they are
//     finished -- its inputs are fresh. Clean nodes are skipped (cache reuse), exactly
//     like the serial path.
//   - The level barrier is coarser than a pure dataflow wavefront (a node could start
//     the instant ITS OWN deps finish, not when the whole previous level does), but it
//     is simple, correct, and matches the CAD/CAE access pattern (wide, shallow levels
//     with thousands of same-level cells). Dataflow-wavefront is registered as a step-2
//     refinement, not built now.
//
// MUST BE CALLED FROM A FIBER running on `sched` (it spawns child tasks and when_all-
// blocks fiber-aware). The serial TaskGraph::evaluate() stays available as the no-sched
// baseline AND the correctness oracle (same result, one thread).

namespace hyle::sched {
class Scheduler;  // fwd (parallel.h references it by ref only; impl includes scheduler.h)
}  // namespace hyle::sched

namespace hyle::graph {

// Run every DIRTY node once, in topological-LEVEL order, with same-level nodes executed
// CONCURRENTLY on `sched`'s workers. Same nodes run / same skips / same cache reuse as
// TaskGraph::evaluate() -- only the schedule differs. Returns the first non-OK Status a
// node produced (after the level it occurred in fully joins), else OK. `alloc` backs the
// per-node task frames + the level bookkeeping. Must be called from a fiber on `sched`.
//
// OUTPUT-ALIASING CONTRACT (the price of concurrency): two nodes at the SAME topological
// level run in PARALLEL, so their ComputeFns must write DISJOINT user memory. Nodes with
// no dependency path between them CAN share a level (that's the whole point -- they're
// independent), so if two such nodes wrote the same output buffer they would data-race
// with no diagnostic. The graph guarantees ordering only along dependency EDGES; express
// any output-write ordering as an edge (make one node depend on the other) or write to
// disjoint memory. The serial evaluate() has no such constraint (one thread).
//
// ERROR-PATH SCHEDULE (differs slightly from serial): on a node failure, all ALREADY-
// SPAWNED same-level siblings still run to completion (they were dispatched before any
// join), then evaluate stops advancing to later levels. Serial evaluate() instead stops
// at the first failing id and never runs its later same-level siblings. Both leave the
// failed node dirty for retry; only the set of same-level siblings that ran on the error
// pass differs (benign, since same-level nodes are independent).
//   CONTRACT: called from a worker fiber of `sched`; same-level nodes write disjoint memory.
HYLE_GRAPH_API Status evaluate_parallel(TaskGraph& g, sched::Scheduler& sched,
                                        mem::Allocator alloc) noexcept;

}  // namespace hyle::graph
