#include "hyle/graph/parallel.h"

#include <cstdint>
#include <new>
#include <utility>

#include "hyle/container/small_vector.h"
#include "hyle/error/status.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/join.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/sync.h"  // Channel (wavefront completion signal)

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

// Wavefront node task: run the node, write its Status to the caller slot, then POST the
// node id to the completion channel so the driver fiber can release successors. Returns
// trivial bool (the handle is detached; the channel post is the real completion signal).
sched::Task<bool> wavefront_node_task(mem::Allocator, TaskGraph* g, std::uint32_t i,
                                      Status* out, sched::Channel<std::uint32_t>* done) {
  *out = g->run_node_(i);
  done->send(i);  // wake the driver; happens-before its outcome[i] read (channel mu)
  co_return out->ok();
}

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

Status evaluate_parallel_wavefront(TaskGraph& g, sched::Scheduler& sched,
                                   mem::Allocator alloc) noexcept {
  const std::uint32_t n = g.node_count_();
  if (n == 0) return Status();

  // remaining[i] = number of node i's DIRTY deps not yet finished. Only dirty deps GATE i
  // (a clean dep is already fresh). A dirty node with remaining==0 is ready to spawn NOW.
  // Also count how many dirty nodes there are total (the work to drain) + build the reverse
  // adjacency (successors) so a finishing node can find who to release -- the CSR only
  // stores forward deps. All of this is single-threaded (driver fiber); no atomics needed
  // because the counters are only ever touched here, between recv()s.
  container::SmallVector<std::uint32_t, 64, mem::Allocator> remaining(alloc);
  std::uint32_t dirty_total = 0;
  for (std::uint32_t i = 0; i < n; ++i) {
    std::uint32_t r = 0;
    if (g.dirty_at_(i)) {
      for (std::uint32_t k = g.dep_begin_(i); k < g.dep_end_(i); ++k) {
        if (g.dirty_at_(g.dep_pool_at_(k))) ++r;  // count dirty deps only
      }
      ++dirty_total;
    }
    remaining.push_back(r);
  }
  if (dirty_total == 0) return Status();  // all clean -> nothing to do

  // Reverse adjacency in CSR form: succ_off_[i]..succ_off_[i+1] indexes succ_[] = the ids
  // that depend on i. Built by counting then filling (standard 2-pass CSR transpose).
  container::SmallVector<std::uint32_t, 64, mem::Allocator> succ_off(alloc);
  const std::uint32_t total_edges = g.dep_pool_size_();
  container::SmallVector<std::uint32_t, 64, mem::Allocator> succ(alloc);
  succ_off.reserve(n + 1);
  for (std::uint32_t i = 0; i <= n; ++i) succ_off.push_back(0u);
  for (std::uint32_t i = 0; i < n; ++i)  // count: succ_off[dep+1]++ for every edge i->dep
    for (std::uint32_t k = g.dep_begin_(i); k < g.dep_end_(i); ++k) ++succ_off[g.dep_pool_at_(k) + 1];
  for (std::uint32_t i = 0; i < n; ++i) succ_off[i + 1] += succ_off[i];  // prefix sum
  for (std::uint32_t e = 0; e < total_edges; ++e) succ.push_back(0u);
  container::SmallVector<std::uint32_t, 64, mem::Allocator> fill(alloc);  // per-node write cursor
  for (std::uint32_t i = 0; i < n; ++i) fill.push_back(succ_off[i]);
  for (std::uint32_t i = 0; i < n; ++i)
    for (std::uint32_t k = g.dep_begin_(i); k < g.dep_end_(i); ++k) {
      const std::uint32_t dep = g.dep_pool_at_(k);
      succ[fill[dep]++] = i;  // dep's successor is i
    }

  // Per-node result slots (same UAF-avoidance as evaluate_parallel: node writes here,
  // driver reads after the channel recv gives a happens-before).
  auto* outcome = static_cast<Status*>(alloc.allocate(sizeof(Status) * n, alignof(Status)));
  HYLE_FATAL_ASSERT(outcome != nullptr);
  for (std::uint32_t i = 0; i < n; ++i) new (&outcome[i]) Status();

  // Completion channel: a finished node posts its id; the driver recv()s. Capacity = n
  // (a node posts once, at most n outstanding) so send() never blocks a worker.
  sched::Channel<std::uint32_t> done(alloc, n);

  // KEEP every spawned task's JoinHandle (do NOT detach) so we can join them ALL before
  // returning. WHY (red-team-found CRITICAL UAF): `done` is a STACK Channel, and
  // Channel::send does `mu_.unlock(); not_empty_.notify_one();` -- it touches `this` AFTER
  // the unlock. The driver's recv() can dequeue (on the unlock's release) and drive
  // outstanding to 0 while the sender is still between unlock and notify_one; if we returned
  // then, `done`'s dtor would run and the sender's notify_one() would hit freed memory. The
  // channel drain-to-zero is NOT a safe "all tasks done" signal. join() IS: a task's
  // set_value (which join awaits) runs AFTER co_return, which runs AFTER send() fully
  // returns (past notify_one). So join-all before the channel dies closes the race. The
  // JoinState is heap+refcounted (survives independently of the stack channel), same
  // safety model as evaluate_parallel. JoinHandle is move-only -> manual array (no SmallVector).
  using H = sched::JoinHandle<bool>;
  auto* handles = static_cast<H*>(alloc.allocate(sizeof(H) * n, alignof(H)));
  HYLE_FATAL_ASSERT(handles != nullptr);
  std::uint32_t spawned = 0;
  auto spawn = [&](std::uint32_t i) {
    new (&handles[spawned++])
        H(spawn_task(sched, alloc, wavefront_node_task(alloc, &g, i, &outcome[i], &done)));
  };

  // Seed: spawn every dirty node whose dirty-dep count is already 0 (sources / nodes whose
  // deps are all clean). `outstanding` = spawned-but-not-yet-drained. Loop until every
  // reachable dirty node has drained.
  std::uint32_t outstanding = 0;
  for (std::uint32_t i = 0; i < n; ++i)
    if (g.dirty_at_(i) && remaining[i] == 0) { spawn(i); ++outstanding; }

  Status first_err;
  while (outstanding > 0) {
    std::uint32_t fin = 0;
    (void)done.recv(fin);  // block (fiber-aware) until a node finishes
    --outstanding;
    if (outcome[fin].ok()) {
      g.clear_dirty_(fin);
      // Release dirty successors: decrement their remaining dirty-dep count; spawn on 0.
      for (std::uint32_t s = succ_off[fin]; s < succ_off[fin + 1]; ++s) {
        const std::uint32_t su = succ[s];
        if (!g.dirty_at_(su)) continue;  // clean successor doesn't run
        if (--remaining[su] == 0) { spawn(su); ++outstanding; }
      }
    } else if (first_err.ok()) {
      first_err = outcome[fin];  // record first error; its successors are never released
    }
  }

  // Join EVERY spawned task before `done` (stack) dies -- closes the send-after-free race
  // above. All tasks have already posted (outstanding==0 means all recv'd), so each join
  // returns immediately; this is a lifetime barrier, not a wait. Then destroy the handles.
  for (std::uint32_t j = 0; j < spawned; ++j) {
    (void)handles[j].join();
    handles[j].~H();
  }
  alloc.deallocate(handles, sizeof(H) * n, alignof(H));

  for (std::uint32_t i = 0; i < n; ++i) outcome[i].~Status();
  alloc.deallocate(outcome, sizeof(Status) * n, alignof(Status));
  return first_err;
}

}  // namespace hyle::graph
