#include "hyle/graph/parallel.h"

#include <cstdint>
#include <new>
#include <utility>

#include "hyle/container/small_vector.h"
#include "hyle/error/status.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/join.h"
#include "hyle/sched/scheduler.h"

// graph-01 step-2 implementation: level-by-level fork-join parallel evaluate. See
// parallel.h for the model. This TU is compiled only where fcontext (hence sched) exists.

namespace hyle::graph {

namespace {

// One node's parallel run. It writes its Status into a CALLER-OWNED slot (`*out`) and
// returns a trivial bool (ok?). WHY not `Task<Status>` (return the Status directly): the
// sched JoinState<T> stores the task's return value in a raw `storage` slot and both
// take() and release() run ~T() on it -- fine for a trivial T, but `Status` is refcounted
// (retain/release), and that value path double-handles the refcount -> a use-after-free
// on the heap message rep (ASan-proven). So node results travel through an explicit
// per-node slot (the GpuOp pattern, gpu_await.h:73), and the Task returns a POD. The slot
// lives in the `outcome` array below, which outlives the join.
sched::Task<bool> run_node_task(mem::Allocator, TaskGraph* g, std::uint32_t i,
                                Status* out) {
  *out = g->run_node_(i);
  co_return out->ok();
}

}  // namespace

Status evaluate_parallel(TaskGraph& g, sched::Scheduler& sched,
                         mem::Allocator alloc) noexcept {
  const std::uint32_t n = g.node_count_();
  if (n == 0) return Status();

  // 1. Compute each node's topological LEVEL = 1 + max(dep level); sources = 0. One
  //    forward id-sweep suffices (deps have smaller ids -> already leveled). level[] is
  //    trivially-copyable (uint32), so SmallVector is fine here.
  container::SmallVector<std::uint32_t, 64, mem::Allocator> level(alloc);
  std::uint32_t max_level = 0;
  for (std::uint32_t i = 0; i < n; ++i) {
    std::uint32_t lv = 0;
    for (std::uint32_t k = g.dep_begin_(i); k < g.dep_end_(i); ++k) {
      const std::uint32_t dl = level[g.dep_pool_at_(k)];
      if (dl + 1 > lv) lv = dl + 1;
    }
    level.push_back(lv);
    if (lv > max_level) max_level = lv;
  }

  // Per-level bookkeeping, sized to the whole node count (upper bound on one level's
  // width). handles[]: JoinHandle is MOVE-ONLY (not trivially copyable) so it can't live
  // in a SmallVector (memcpy-relocate) -- manual placement-new/destroy. ran_ids[] maps a
  // handle slot to its node id. outcome[]: each node's Status lands here (see run_node_task
  // -- avoids the refcounted-T-in-JoinState UAF); indexed by node id.
  using H = sched::JoinHandle<bool>;
  auto* handles = static_cast<H*>(alloc.allocate(sizeof(H) * n, alignof(H)));
  HYLE_FATAL_ASSERT(handles != nullptr);
  auto* ran_ids = static_cast<std::uint32_t*>(
      alloc.allocate(sizeof(std::uint32_t) * n, alignof(std::uint32_t)));
  HYLE_FATAL_ASSERT(ran_ids != nullptr);
  auto* outcome = static_cast<Status*>(alloc.allocate(sizeof(Status) * n, alignof(Status)));
  HYLE_FATAL_ASSERT(outcome != nullptr);
  for (std::uint32_t i = 0; i < n; ++i) new (&outcome[i]) Status();

  // 2. For each level, spawn every DIRTY node at that level (they fan out across
  //    workers), then join them all (fiber-aware barrier). A node's deps are at strictly
  //    lower levels -> already finished + fresh when this level runs. Clean nodes skipped.
  Status first_err;  // first non-OK across the whole evaluate (reported after joins)
  for (std::uint32_t lv = 0; lv <= max_level; ++lv) {
    std::uint32_t cnt = 0;
    for (std::uint32_t i = 0; i < n; ++i) {
      if (level[i] != lv || !g.dirty_at_(i)) continue;
      // ALL spawned before ANY join -> they run concurrently across workers.
      new (&handles[cnt])
          H(spawn_task(sched, alloc, run_node_task(alloc, &g, i, &outcome[i])));
      ran_ids[cnt] = i;
      ++cnt;
    }
    // Join every task spawned this level (fiber-aware block until each finishes), then
    // read its slot. Clear dirty only on success; on failure keep it dirty (retried next
    // evaluate) + record the first error -- but still join the rest so none dangles.
    for (std::uint32_t j = 0; j < cnt; ++j) {
      (void)handles[j].join();  // wait for completion; the Status is in outcome[id]
      const std::uint32_t id = ran_ids[j];
      if (outcome[id].ok()) {
        g.clear_dirty_(id);
      } else if (first_err.ok()) {
        first_err = outcome[id];
      }
      handles[j].~H();
    }
    if (!first_err.ok()) break;  // stop advancing levels once a node failed
  }

  for (std::uint32_t i = 0; i < n; ++i) outcome[i].~Status();
  alloc.deallocate(outcome, sizeof(Status) * n, alignof(Status));
  alloc.deallocate(ran_ids, sizeof(std::uint32_t) * n, alignof(std::uint32_t));
  alloc.deallocate(handles, sizeof(H) * n, alignof(H));
  return first_err;
}

}  // namespace hyle::graph
