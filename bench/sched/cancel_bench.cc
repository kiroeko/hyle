// sched-07 cancellation-tree microbenchmarks. sched-07 shipped the CancelToken tree
// (16 tests + TSan) but no perf number; this file adds it. The tree is a shard-local,
// (mostly) lock-free structure -- its only atomic is the per-node `cancelled` flag
// (for cross-worker visibility after fiber migration). The perf claims to PROVE:
//
//  1. is_cancelled() IS FREE ON THE HOT PATH. Every await point / HYLE_CHECK_CANCEL()
//     polls it, so it must be ~a plain load. Raced against a bare std::atomic<bool>
//     acquire load (the theoretical floor): the ratio should be ~1.0 (the token adds
//     only a null check over the raw atomic).
//
//  2. cancel() PROPAGATION IS O(N) IN THE SUBTREE. cancel_subtree walks the tree
//     iteratively (no recursion -> no stack blowup on deep fan-out). Build a tree of
//     N nodes, cancel the root, tear it down; batch=N -> ns-per-node. FLAT ns/node
//     across N = 64/256/1024 proves the whole build+cancel+teardown path is linear
//     (a rising line would betray a super-linear factor -- e.g. an accidental
//     re-walk). Both a WIDE tree (root with N children) and a DEEP chain (N deep) are
//     swept, since the iterative walk's sibling vs parent/child traversal differ.
//
//  3. derive() COST. Deriving a child = one allocation + O(1) front-insert; measured
//     as the per-child cost of fanning a request out to N sub-tasks.
//
// Run under scripts/bench-stat.py -n 15 for the statistical verdict.

#include <nanobench.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/sched/cancel.h"

using hyle::mem::system_allocator;
using hyle::sched::CancelToken;

namespace {
namespace nb = ankerl::nanobench;
}  // namespace

int main() {
  // ---- 1. is_cancelled() hot-path cost vs a bare atomic<bool> load ----
  // The floor is a plain acquire load; the token adds a null check over it. Ratio
  // ~1.0 proves polling the token at every await point is effectively free.
  {
    constexpr int kIters = 10000;
    CancelToken tok = CancelToken::root(system_allocator());
    std::atomic<bool> raw{false};
    nb::Bench b;
    b.title("CancelToken::is_cancelled() vs bare atomic<bool> load")
        .relative(true).batch(kIters).minEpochIterations(500);
    b.run("bare std::atomic<bool> acquire load", [&] {
      std::uint64_t sink = 0;
      for (int i = 0; i < kIters; ++i)
        sink += raw.load(std::memory_order_acquire) ? 1 : 0;
      nb::doNotOptimizeAway(sink);
    });
    b.run("CancelToken::is_cancelled()", [&] {
      std::uint64_t sink = 0;
      for (int i = 0; i < kIters; ++i) sink += tok.is_cancelled() ? 1 : 0;
      nb::doNotOptimizeAway(sink);
    });
  }

  // ---- 2a. cancel() propagation, WIDE tree: root with N direct children ----
  // Build root + N children, cancel root (walks all N), tear down. batch=N -> ns/node.
  // Flat across N => build+cancel+teardown is O(N) (no super-linear re-walk).
  {
    nb::Bench b;
    b.title("cancel propagation WIDE (root + N children): ns/node (flat => O(N))")
        .unit("node");
    for (int n : {64, 256, 1024}) {
      char name[64];
      std::snprintf(name, sizeof(name), "wide N=%d", n);
      b.batch(n).minEpochIterations(50).run(name, [&] {
        CancelToken root = CancelToken::root(system_allocator());
        std::vector<CancelToken> kids;
        kids.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) kids.push_back(root.derive());
        root.cancel();  // marks root + all N children (iterative walk)
        nb::doNotOptimizeAway(root.is_cancelled());
      });
    }
  }

  // ---- 2b. cancel() propagation, DEEP chain: a linear chain N deep ----
  // Each token derives the next -> a depth-N chain. cancel(root) walks parent->child
  // to the leaf. Exercises the iterative walk's descent path (vs 2a's sibling scan);
  // proves no recursion-depth blowup on a deep fan-out. Flat ns/node => O(N).
  {
    nb::Bench b;
    b.title("cancel propagation DEEP (chain N deep): ns/node (flat => O(N))")
        .unit("node");
    for (int n : {64, 256, 1024}) {
      char name[64];
      std::snprintf(name, sizeof(name), "deep N=%d", n);
      b.batch(n).minEpochIterations(50).run(name, [&] {
        CancelToken root = CancelToken::root(system_allocator());
        std::vector<CancelToken> chain;
        chain.reserve(static_cast<std::size_t>(n));
        CancelToken* cur = &root;
        for (int i = 0; i < n; ++i) {
          chain.push_back(cur->derive());
          cur = &chain.back();
        }
        root.cancel();  // walks the whole depth-N chain to the leaf
        nb::doNotOptimizeAway(chain.back().is_cancelled());
      });
    }
  }

  // ---- 3. derive() per-child cost (request fan-out to N sub-tasks) ----
  // Deriving a child = one allocation + O(1) front-insert. batch=N -> per-child cost.
  // Kept separate from cancel so this isolates the derive path (dominated by the
  // allocator). Teardown (token dtors unlink) is included -- the honest full cost.
  {
    nb::Bench b;
    b.title("derive() per child (fan-out to N sub-tasks)").unit("child");
    for (int n : {64, 256, 1024}) {
      char name[64];
      std::snprintf(name, sizeof(name), "derive N=%d", n);
      b.batch(n).minEpochIterations(50).run(name, [&] {
        CancelToken root = CancelToken::root(system_allocator());
        std::vector<CancelToken> kids;
        kids.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) kids.push_back(root.derive());
        nb::doNotOptimizeAway(kids.size());
      });
    }
  }

  return 0;
}
