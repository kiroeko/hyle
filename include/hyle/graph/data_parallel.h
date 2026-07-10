#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/error/status.h"
#include "hyle/mem/allocator.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_GRAPH_API
#elif defined(HYLE_GRAPH_BUILD)
#  define HYLE_GRAPH_API HYLE_EXPORT
#else
#  define HYLE_GRAPH_API HYLE_IMPORT
#endif

// graph-02 (step-1): the DATA-PARALLEL layer -- parallel_for / parallel_reduce over a
// range, run on HYLE's sched work-stealing pool. graph-01 orchestrates a DAG of DISTINCT
// tasks (node/edge); graph-02 slices ONE range and runs the SAME body on each block in
// parallel (a mesh of 1M cells across all cores). Both only ORCHESTRATE -- the block
// scheduling is sched work-stealing, the per-block compute is the user's body (typically a
// simd tight loop). See docs/architecture/graph-02-design.md.
//
// WHY NOT oneTBB/oneDPL: they bring their OWN task arena / work-stealing scheduler; running
// one on HYLE's cores = two schedulers oversubscribing + cache-thrashing (references.md:99).
// We borrow their DESIGN (blocked_range grainsize + recursive split, oneTBB's SOTA) and run
// it on HYLE sched. MUST be called from a fiber on the scheduler.
//
// THE 5 DESIGN DECISIONS (graph-02-design §3):
//  ① PARTITION = recursive binary split down to `grainsize`, NOT static N/W even split. Split
//     produces fine leaf tasks that HYLE work-stealing load-balances -- a slow block's
//     siblings get stolen (static split strands a slow block on one worker). Matches
//     oneTBB/Cilk; fits HYLE's work-stealing thesis (motivation: loads are not evenly divisible).
//  ② REDUCE combine MUST be associative (caller's contract); for a FIXED grainsize the fold
//     order is the fixed left->right bisection tree, so the result is deterministic (even for
//     float). The non-determinism step-2's `parallel_reduce_deterministic` removes is across
//     DIFFERENT grainsizes / auto-grain / worker counts (which change the tree shape, hence the
//     float rounding) -- a fixed pinned grain here is already reproducible. SOTA lib doesn't
//     pick the speed/precision trade for you.
//  ③ BODY TAKES A SUBRANGE [begin,end), NOT a per-element callback. Per-element indirect calls
//     kill vectorization; a subrange lets the body write a tight loop -> feed simd vec<T,N>.
//     This is the graph-02 <-> simd seam: blocking is graph-02's job, the in-block vector loop
//     is simd's. (oneTBB Body takes blocked_range for the same reason.)
//  ④/⑤ scan/sort + deterministic reduce are step-2/3; reuse graph-01's sched fork-join plumbing.

namespace hyle::sched {
class Scheduler;  // fwd (referenced by ref only; the .cc includes scheduler.h)
}  // namespace hyle::sched

namespace hyle::graph {

// A half-open integer index range [begin, end) with a grainsize: the unit of "small enough
// to run inline, stop splitting". is_divisible() == grainsize < size(). This is oneTBB's
// blocked_range shape (the SOTA partition primitive), specialized to std::size_t indices
// (the common data-parallel case: iterate a buffer/mesh by index; the body maps index->data).
struct BlockedRange {
  std::size_t begin;
  std::size_t end;
  std::size_t grainsize;  // >=1 explicit leaf size, OR kAutoGrain(0) = "layer picks it"

  // grainsize is stored AS-IS -- including 0 (kAutoGrain). Do NOT coerce 0->1 here: 0 is the
  // auto sentinel resolve_grain() must see (coercing it silently forced grain=1 = one 64KB
  // fiber per element = OOM/hang on large ranges -- a real bug this guard caused). The real
  // ">=1" floor is applied by resolve_grain (which returns >=1 for both explicit and auto).
  BlockedRange(std::size_t b, std::size_t e, std::size_t g = 1) noexcept
      : begin(b), end(e), grainsize(g) {}
  std::size_t size() const noexcept { return end - begin; }
  // Divisible if the resolved grain is smaller than the range. For a raw range carrying
  // kAutoGrain(0) this is a coarse "yes if size>0"; the .cc splits on the RESOLVED grain, not
  // this (kept for API parity with oneTBB blocked_range).
  bool is_divisible() const noexcept { return grainsize < size(); }
};

// `auto` grainsize: pass this to parallel_for/reduce to let the layer pick a grainsize that
// yields ~ several tasks per worker (enough to steal + balance, not so many the split/spawn
// overhead dominates). Concretely grainsize = ceil(size / (workers * kOversample)). Prefer
// this unless you have a measured reason to pin grainsize.
inline constexpr std::size_t kAutoGrain = 0;

// The per-block body: process the half-open [begin, end) sub-range. Runs on a sched worker.
// Must be no-throw; report failure by returning non-OK Status (the whole library is no-throw).
// The FIRST non-OK returned by any block aborts parallel_for and surfaces to the caller;
// other in-flight blocks still finish (they were already dispatched). `ctx` is the user's
// closure state (type-erased -- see the template wrappers below for the ergonomic form).
using ForBodyFn = Status (*)(void* ctx, std::size_t begin, std::size_t end) noexcept;

// parallel_for: split [0,n)/`range` recursively to grainsize and run `body(ctx, b, e)` on each
// leaf block, IN PARALLEL across sched workers (fork-join: split -> spawn one half, recurse the
// other, join). Returns the first non-OK block Status, else OK. `alloc` backs the task frames.
// If `range.grainsize == kAutoGrain`, an auto grainsize is chosen from the scheduler's worker
// count. Must be called from a fiber on `sched`. Same-block memory disjointness is the body's
// concern; distinct blocks are index-disjoint by construction.
//   CONTRACT: called from a worker fiber of `sched`; body non-null; range.begin <= range.end.
HYLE_GRAPH_API Status parallel_for(sched::Scheduler& sched, mem::Allocator alloc,
                                   BlockedRange range, ForBodyFn body, void* ctx) noexcept;

// The per-block reduce body: fold [begin,end) into a partial result written to `*out` (a slot
// of `elem_size` bytes the layer provides). combine(a_inout, b): fold b into a. identity fills
// a fresh slot. All operate on type-erased `elem_size`-byte values (the template wrapper below
// gives the typed form). combine MUST be ASSOCIATIVE. The fold is strictly LEFT-TO-RIGHT in
// index order (combine(left_partial, right_partial) at every tree node), so a non-commutative
// associative op (string concat, matrix product) is correct. For a FIXED grainsize the result
// is deterministic; float rounding only shifts if you change grainsize/worker count.
using ReduceBodyFn = Status (*)(void* ctx, std::size_t begin, std::size_t end,
                                void* out) noexcept;          // out := reduce([begin,end))
using CombineFn = void (*)(void* ctx, void* a_inout, const void* b) noexcept;  // a := combine(a,b)
using IdentityFn = void (*)(void* ctx, void* out) noexcept;   // out := identity

// parallel_reduce (FAST mode): split like parallel_for; each leaf reduces its block to a
// partial; partials are combined left->right up the fixed bisection tree (deterministic for a
// fixed grainsize). The final result lands in `*result` (an `elem_size`-byte slot the caller
// owns). Returns first non-OK, else OK. (step-2's deterministic variant fixes the tree across
// grainsize/worker-count changes, not this fixed-grain path.)
//   CONTRACT: called from a worker fiber of `sched`; body/combine/identity non-null;
//   result non-null; range.begin <= range.end; 0 < elem_size <= 256 (the partial-slot cap --
//   a reduce value bigger than 256 bytes should reduce into a caller-owned handle instead).
HYLE_GRAPH_API Status parallel_reduce(sched::Scheduler& sched, mem::Allocator alloc,
                                      BlockedRange range, std::size_t elem_size, ReduceBodyFn body,
                                      CombineFn combine, IdentityFn identity, void* ctx,
                                      void* result) noexcept;

}  // namespace hyle::graph
