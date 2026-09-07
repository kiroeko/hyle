#include "hyle/graph/parallel.h"

#include <atomic>
#include <cstdint>
#include <new>
#include <utility>

#include "hyle/container/small_vector.h"
#include "hyle/error/status.h"
#include "hyle/mem/region.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/join.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/waker.h" // current_waker + suspend (decentralized wavefront completion)
#include "hyle/util/spin_lock.h" // first_err guard

// graph-01 step-2 implementation: level-by-level fork-join parallel evaluate.
// See parallel.h for the model. This TU is compiled only where fcontext (hence
// sched) exists.

namespace hyle::graph {

namespace {

// One node's parallel run. It writes its Status into a CALLER-OWNED slot
// (`*out`) and returns a trivial bool (ok?). WHY not `Task<Status>` (return the
// Status directly): the sched JoinState<T> stores the task's return value in a
// raw `storage` slot and both take() and release() run ~T() on it -- fine for a
// trivial T, but `Status` is refcounted (retain/release), and that value path
// double-handles the refcount -> a use-after-free on the heap message rep
// (ASan-proven). So node results travel through an explicit per-node slot (the
// GpuOp pattern, gpu_await.h:73), and the Task returns a POD. The slot lives in
// the `outcome` array below, which outlives the join.
sched::Task<bool> run_node_task(mem::AllocatorRef, TaskGraph *g, std::uint32_t i,
                                Status *out) {
  *out = g->run_node_(i);
  co_return out->ok();
}

} // namespace

// ---- DECENTRALIZED wavefront (graph-01 step-3, refactored 2026-07-10)
// -------- Calibrated against taskflow: the OLD wavefront funneled every
// completion through one stack Channel to a single DRIVER fiber that serially
// decremented counters + spawned successors -- a serialization point on a wide
// DAG (HYLE's own target). taskflow instead DECENTRALIZES: the worker that
// FINISHES a node decrements its successors' atomic dep counters itself and
// spawns the newly-ready one onto ITS OWN local queue. This adopts that model.
// No driver, no channel; the top-level fiber just suspends until the last node
// wakes it.
//
// Shared state, HEAP-allocated + REFCOUNTED so it outlives every task
// regardless of who finishes last (the recurring lifetime hazard -- solved by
// construction, not luck). Each running task holds a ref; the waiting fiber
// holds one; last drop frees. `remaining[i]` is ATOMIC (workers concurrently
// decrement). `pending` counts spawned-not-finished nodes; when it hits 0 the
// last task wakes the waiter. The last finisher moves `waker` onto its stack
// before the wake, so wake() never touches this (possibly-freed) struct.
struct WfState {
  TaskGraph *g;
  mem::AllocatorRef alloc;
  sched::Scheduler *sched; // the scheduler tasks spawn onto
  Status *outcome;         // per-node result slot (n Status, placement-new'd)
  std::atomic<std::uint32_t>
      *remaining; // per-node dirty-dep counter (n atomics)
  std::atomic<std::uint8_t>
      *ran_ok;         // per-node: 1 iff this node ran and succeeded
  std::uint32_t *succ; // reverse-adjacency CSR (who depends on me) -- OWNED
  std::uint32_t
      *succ_off; // succ[succ_off[i]..succ_off[i+1]) = i's successors -- OWNED
  std::uint32_t n;
  std::uint32_t edges;                // len of succ[] (for dealloc)
  std::atomic<std::uint32_t> pending; // spawned-but-not-finished node count
  std::atomic<int> refcount;          // WfState lifetime (tasks + waiter)
  std::atomic<bool>
      had_err;           // any node failed (successors of a failure won't run)
  Status first_err;      // written once under had_err CAS (first failer)
  util::SpinLock err_mu; // guards first_err write
  sched::Waker waker;    // wakes the waiting fiber when pending hits 0

  void record(Status s) noexcept {
    bool expected = false;
    if (had_err.compare_exchange_strong(expected, true,
                                        std::memory_order_acq_rel)) {
      err_mu.lock();
      first_err = std::move(s);
      err_mu.unlock();
    }
  }

  // Balance one task's pending slot + WfState reference. Move the Waker before
  // drop_ref(): a future caller may legitimately make this the last reference,
  // and wake must not touch a freed WfState.  Today failed-admission rollback is
  // protected by either the seeding guard or the currently-running parent task,
  // but sharing the normal completion path keeps that invariant from becoming a
  // latent lost-wake if wf_spawn gains another caller.
  void finish_task_ref() noexcept {
    const bool last =
        pending.fetch_sub(1, std::memory_order_acq_rel) == 1;
    sched::Waker w;
    if (last) w = std::move(waker);
    drop_ref();
    if (last) w.wake();
  }

  void add_ref() noexcept { refcount.fetch_add(1, std::memory_order_relaxed); }
  void drop_ref() noexcept {
    if (refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      mem::AllocatorRef a = alloc;
      std::uint32_t nn = n, ne = edges;
      Status *oc = outcome;
      std::atomic<std::uint32_t> *rem = remaining;
      std::atomic<std::uint8_t> *rk = ran_ok;
      std::uint32_t *sc = succ;
      std::uint32_t *so = succ_off;
      for (std::uint32_t i = 0; i < nn; ++i)
        oc[i].~Status();
      this->~WfState();
      a.deallocate(so, sizeof(std::uint32_t) * (nn + 1),
                   alignof(std::uint32_t));
      a.deallocate(sc, sizeof(std::uint32_t) * (ne == 0 ? 1 : ne),
                   alignof(std::uint32_t));
      a.deallocate(rk, sizeof(std::atomic<std::uint8_t>) * nn,
                   alignof(std::atomic<std::uint8_t>));
      a.deallocate(rem, sizeof(std::atomic<std::uint32_t>) * nn,
                   alignof(std::atomic<std::uint32_t>));
      a.deallocate(oc, sizeof(Status) * nn, alignof(Status));
      a.deallocate(this, sizeof(WfState), alignof(WfState));
    }
  }
};

// forward decl (task spawns successor tasks of the same kind)
sched::Task<bool> wf_node_task(mem::AllocatorRef alloc, WfState *st,
                               std::uint32_t i);

// Spawn node `i` NOW (it is known-ready: 0 dirty deps left). Bumps pending +
// takes a ref, then spawns onto the CALLER's worker-local queue
// (decentralized). WHO calls this is the UNIQUE-ownership rule that prevents
// double-spawn: (a) the seed loop, for nodes whose INITIAL dirty-dep count is 0
// (sources / all-clean-dep nodes) -- computed single-threaded at setup, never
// re-read; (b) the completion path, for a successor whose `fetch_sub` returned
// 1 (i.e. THIS task drove its counter to 0 -- exactly one task can). A node is
// therefore spawned exactly once. (The old racy `remaining==0` RE-READ here let
// the seed loop and an early completion both observe 0 and double-spawn --
// ASan/flaky-proven.)
void wf_spawn(mem::AllocatorRef alloc, WfState *st, std::uint32_t i) {
  st->pending.fetch_add(1, std::memory_order_acq_rel);
  st->add_ref(); // the new task holds a ref
  auto handle = spawn_task(*st->sched, alloc, wf_node_task(alloc, st, i));
  // A failed admission is already a finished JoinHandle.  A successfully
  // admitted task can also finish before this check, so consume the handle only
  // when finished and distinguish those cases through join(): an unexpected
  // result means the task did not perform its WfState completion accounting, so
  // this path balances the pending/ref increments above.  A live admitted task
  // is detached normally.
  if (handle.is_finished()) {
    auto result = handle.join();
    if (!result) {
      st->record(std::move(result).error().WithContext(
          "wavefront node task admission"));
      st->finish_task_ref();
    }
  } else {
    handle.detach(); // lands on this worker's local queue
  }
}

// One node's run on a worker. Runs the node, then DECENTRALIZED-releases its
// successors: atomic-decrement each dirty successor's counter, spawn any that
// reach 0 onto THIS worker. Finally decrement `pending`; the task that drives
// it to 0 wakes the waiting fiber.
sched::Task<bool> wf_node_task(mem::AllocatorRef alloc, WfState *st,
                               std::uint32_t i) {
  Status s = st->g->run_node_(i);
  st->outcome[i] = s;
  if (s.ok()) {
    // Mark ran-and-ok in WfState (atomic); the actual g.clear_dirty_ is
    // DEFERRED to the waiter fiber post-resume. WHY: multiple workers finish
    // concurrently, so writing the graph's non-atomic dirty_[] from worker
    // threads is a data race (TSan-proven). Workers only touch WfState atomics;
    // the single waiter fiber applies clears after all done.
    st->ran_ok[i].store(1, std::memory_order_release);
    // Release dirty successors on THIS worker (taskflow decentralized model).
    // dirty_at_ is a READ of a flag nobody clears mid-flight now (clears are
    // deferred) -> concurrent reads are race-free.
    for (std::uint32_t e = st->succ_off[i]; e < st->succ_off[i + 1]; ++e) {
      const std::uint32_t su = st->succ[e];
      if (!st->g->dirty_at_(su))
        continue;
      if (st->remaining[su].fetch_sub(1, std::memory_order_acq_rel) == 1)
        wf_spawn(
            alloc, st,
            su); // THIS task drove su's counter to 0 -> unique owner, spawn
    }
  } else {
    // First failer records the error; a failed node's successors are never
    // released (their remaining never hits 0), so they stay dirty (deferred,
    // same as serial/level path).
    st->record(s);
  }
  // COMPLETION: if I am the last pending node, wake the waiter. Move the Waker
  // to MY stack FIRST -- after the wake the waiter may resume and drop the last
  // ref, freeing st; touching st->waker after that would be the exact
  // send-after-free the old channel design had.
  st->finish_task_ref();
  co_return s.ok();
}

Status evaluate_parallel(TaskGraph &g, sched::Scheduler &sched,
                         mem::AllocatorRef alloc) noexcept {
  const std::uint32_t n = g.node_count_();
  if (n == 0)
    return Status();

  // This function's level/handle/id/outcome storage has one exact phase
  // lifetime: every spawned handle is joined before return. A Arena expresses
  // that lifetime directly and amortizes the several arrays into growable
  // blocks. Coroutine frames still use `alloc`: their completion/deallocation
  // may occur on worker threads and is not a Arena contract.
  mem::Arena scratch(alloc);
  mem::AllocatorRef scratch_alloc = scratch.as_allocator();

  // 1. Compute each node's topological LEVEL = 1 + max(dep level); sources = 0.
  // One
  //    forward id-sweep suffices (deps have smaller ids -> already leveled).
  //    level[] is trivially-copyable (uint32), so SmallVector is fine here.
  container::SmallVector<std::uint32_t, 64, mem::AllocatorRef> level(
      scratch_alloc);
  std::uint32_t max_level = 0;
  for (std::uint32_t i = 0; i < n; ++i) {
    std::uint32_t lv = 0;
    for (std::uint32_t k = g.dep_begin_(i); k < g.dep_end_(i); ++k) {
      const std::uint32_t dl = level[g.dep_pool_at_(k)];
      if (dl + 1 > lv)
        lv = dl + 1;
    }
    level.push_back(lv);
    if (lv > max_level)
      max_level = lv;
  }

  // Per-level bookkeeping, sized to the whole node count (upper bound on one
  // level's width). handles[]: JoinHandle is MOVE-ONLY (not trivially copyable)
  // so it can't live in a SmallVector (memcpy-relocate) -- manual
  // placement-new/destroy. ran_ids[] maps a handle slot to its node id.
  // outcome[]: each node's Status lands here (see run_node_task
  // -- avoids the refcounted-T-in-JoinState UAF); indexed by node id.
  using H = sched::JoinHandle<bool>;
  auto *handles =
      static_cast<H *>(scratch_alloc.allocate(sizeof(H) * n, alignof(H)));
  HYLE_FATAL_ASSERT(handles != nullptr);
  auto *ran_ids = static_cast<std::uint32_t *>(scratch_alloc.allocate(
      sizeof(std::uint32_t) * n, alignof(std::uint32_t)));
  HYLE_FATAL_ASSERT(ran_ids != nullptr);
  auto *outcome = static_cast<Status *>(
      scratch_alloc.allocate(sizeof(Status) * n, alignof(Status)));
  HYLE_FATAL_ASSERT(outcome != nullptr);
  for (std::uint32_t i = 0; i < n; ++i)
    new (&outcome[i]) Status();

  // 2. For each level, spawn every DIRTY node at that level (they fan out
  // across
  //    workers), then join them all (fiber-aware barrier). A node's deps are at
  //    strictly lower levels -> already finished + fresh when this level runs.
  //    Clean nodes skipped.
  Status first_err; // first non-OK across the whole evaluate (reported after
                    // joins)
  for (std::uint32_t lv = 0; lv <= max_level; ++lv) {
    std::uint32_t cnt = 0;
    for (std::uint32_t i = 0; i < n; ++i) {
      if (level[i] != lv || !g.dirty_at_(i))
        continue;
      // ALL spawned before ANY join -> they run concurrently across workers.
      new (&handles[cnt])
          H(spawn_task(sched, alloc, run_node_task(alloc, &g, i, &outcome[i])));
      ran_ids[cnt] = i;
      ++cnt;
    }
    // Join every task spawned this level (fiber-aware block until each
    // finishes), then read its slot. Clear dirty only on success; on failure
    // keep it dirty (retried next evaluate) + record the first error -- but
    // still join the rest so none dangles.
    for (std::uint32_t j = 0; j < cnt; ++j) {
      auto joined = handles[j].join(); // waits for completion when admitted
      const std::uint32_t id = ran_ids[j];
      if (!joined) {
        // Allocation/admission failure means run_node_task never wrote outcome.
        // Keep the node dirty and surface the scheduler error instead of treating
        // the default-OK slot as a successful evaluation.
        if (first_err.ok()) {
          first_err = std::move(joined).error().WithContext(
              "parallel graph node task");
        }
      } else if (outcome[id].ok()) {
        g.clear_dirty_(id);
      } else if (first_err.ok()) {
        first_err = outcome[id];
      }
      handles[j].~H();
    }
    if (!first_err.ok())
      break; // stop advancing levels once a node failed
  }

  for (std::uint32_t i = 0; i < n; ++i)
    outcome[i].~Status();
  // Arena frees these arrays together on scope exit; individual deallocate is
  // intentionally unnecessary. Non-trivial elements were explicitly destroyed
  // above.
  return first_err;
}

Status evaluate_parallel_wavefront(TaskGraph &g, sched::Scheduler &sched,
                                   mem::AllocatorRef alloc) noexcept {
  const std::uint32_t n = g.node_count_();
  if (n == 0)
    return Status();

  // Reverse adjacency in CSR form (who depends on me):
  // succ_off[i]..succ_off[i+1] indexes succ[]. Built by count-then-fill (2-pass
  // transpose). HEAP-allocated + OWNED by WfState (freed in drop_ref) -- NOT
  // borrowed from the fiber stack. WHY: two back-to-back
  // evaluate_parallel_wavefront calls reuse the SAME stack addresses for their
  // CSR vectors; a task lagging past its wake could then read call-#1's succ at
  // an address call-#2 is already writing (TSan-flagged). Heap ownership tied
  // to the refcount removes that hazard entirely.
  const std::uint32_t total_edges = g.dep_pool_size_();
  auto *succ_off = static_cast<std::uint32_t *>(
      alloc.allocate(sizeof(std::uint32_t) * (n + 1), alignof(std::uint32_t)));
  HYLE_FATAL_ASSERT(succ_off != nullptr);
  auto *succ = static_cast<std::uint32_t *>(alloc.allocate(
      sizeof(std::uint32_t) * (total_edges == 0 ? 1 : total_edges),
      alignof(std::uint32_t)));
  HYLE_FATAL_ASSERT(succ != nullptr);
  for (std::uint32_t i = 0; i <= n; ++i)
    succ_off[i] = 0;
  for (std::uint32_t i = 0; i < n; ++i)
    for (std::uint32_t k = g.dep_begin_(i); k < g.dep_end_(i); ++k)
      ++succ_off[g.dep_pool_at_(k) + 1];
  for (std::uint32_t i = 0; i < n; ++i)
    succ_off[i + 1] += succ_off[i]; // prefix sum
  container::SmallVector<std::uint32_t, 64, mem::AllocatorRef> fill(
      alloc); // transient write cursor
  for (std::uint32_t i = 0; i < n; ++i)
    fill.push_back(succ_off[i]);
  for (std::uint32_t i = 0; i < n; ++i)
    for (std::uint32_t k = g.dep_begin_(i); k < g.dep_end_(i); ++k)
      succ[fill[g.dep_pool_at_(k)]++] = i; // dep's successor is i

  // Heap WfState (refcounted, outlives all tasks). OWNS
  // remaining[]/ran_ok[]/outcome[] + succ/succ_off (all freed in drop_ref).
  // remaining[i] = count of node i's DIRTY deps.
  auto *rem = static_cast<std::atomic<std::uint32_t> *>(
      alloc.allocate(sizeof(std::atomic<std::uint32_t>) * n,
                     alignof(std::atomic<std::uint32_t>)));
  HYLE_FATAL_ASSERT(rem != nullptr);
  auto *ran_ok = static_cast<std::atomic<std::uint8_t> *>(
      alloc.allocate(sizeof(std::atomic<std::uint8_t>) * n,
                     alignof(std::atomic<std::uint8_t>)));
  HYLE_FATAL_ASSERT(ran_ok != nullptr);
  auto *outcome = static_cast<Status *>(
      alloc.allocate(sizeof(Status) * n, alignof(Status)));
  HYLE_FATAL_ASSERT(outcome != nullptr);
  // Collect the SEED set (dirty nodes whose INITIAL dirty-dep count is 0) HERE,
  // single- threaded, before anything is spawnable. Spawning from this captured
  // list -- not a later re-read of remaining[] (which a running task may
  // already have decremented) -- is what keeps "spawned exactly once": seed
  // owns the initial-0 nodes, completions own the rest.
  container::SmallVector<std::uint32_t, 64, mem::AllocatorRef> seeds(alloc);
  std::uint32_t dirty_total = 0;
  for (std::uint32_t i = 0; i < n; ++i) {
    std::uint32_t r = 0;
    if (g.dirty_at_(i)) {
      for (std::uint32_t k = g.dep_begin_(i); k < g.dep_end_(i); ++k)
        if (g.dirty_at_(g.dep_pool_at_(k)))
          ++r;
      ++dirty_total;
      if (r == 0)
        seeds.push_back(i); // ready immediately
    }
    new (&rem[i]) std::atomic<std::uint32_t>(r);
    new (&ran_ok[i]) std::atomic<std::uint8_t>(0);
    new (&outcome[i]) Status();
  }
  if (dirty_total ==
      0) { // all clean -> nothing to run; tear the buffers back down
    for (std::uint32_t i = 0; i < n; ++i) {
      rem[i].~atomic();
      ran_ok[i].~atomic();
      outcome[i].~Status();
    }
    alloc.deallocate(succ_off, sizeof(std::uint32_t) * (n + 1),
                     alignof(std::uint32_t));
    alloc.deallocate(
        succ, sizeof(std::uint32_t) * (total_edges == 0 ? 1 : total_edges),
        alignof(std::uint32_t));
    alloc.deallocate(ran_ok, sizeof(std::atomic<std::uint8_t>) * n,
                     alignof(std::atomic<std::uint8_t>));
    alloc.deallocate(rem, sizeof(std::atomic<std::uint32_t>) * n,
                     alignof(std::atomic<std::uint32_t>));
    alloc.deallocate(outcome, sizeof(Status) * n, alignof(Status));
    return Status();
  }

  auto *st =
      static_cast<WfState *>(alloc.allocate(sizeof(WfState), alignof(WfState)));
  HYLE_FATAL_ASSERT(st != nullptr);
  new (st) WfState{};
  st->g = &g;
  st->alloc = alloc;
  st->sched = &sched;
  st->outcome = outcome;
  st->remaining = rem;
  st->ran_ok = ran_ok;
  st->succ = succ;
  st->succ_off = succ_off;
  st->n = n;
  st->edges = total_edges;
  // pending starts at 1 = a SEEDING GUARD held by this waiter (NOT a real
  // task). WHY: seeds are spawned one-at-a-time below; without the guard, seed
  // #1 could finish (pending 1->0) and fire a PREMATURE wake before the waiter
  // spawns seed #2 -- the waiter would then resume and run the deferred
  // dirty-clears while later nodes hadn't run yet, leaving them wrongly dirty
  // (the ~1/30 `WavefrontIncrementalReRunsOnlyDirty` failure -- a
  // transient-zero pending, NOT a data race, which is why TSan was silent). The
  // guard keeps pending>=1 until seeding is done; releasing it below is what
  // actually arms completion.
  st->pending.store(1, std::memory_order_relaxed);
  st->refcount.store(1, std::memory_order_relaxed); // this waiter fiber's ref
  st->had_err.store(false, std::memory_order_relaxed);
  st->waker =
      sched::current_waker(); // whom the last node wakes (set BEFORE any spawn)

  // Seed: spawn the captured 0-dirty-dep nodes (they fan out onto workers).
  // Uses the list captured single-threaded above -- NOT a remaining[] re-read
  // -- so no seed/completion double-spawn race.
  for (std::uint32_t s = 0; s < seeds.size(); ++s)
    wf_spawn(alloc, st, seeds[s]);

  // Release the seeding guard. If real tasks are still outstanding (fetch_sub
  // returns >1) we suspend and the TRUE last task wakes us. If every task
  // already finished during seeding (fetch_sub returns 1 == the guard was
  // last), DON'T suspend -- nobody remains to wake us. acq_rel keeps the waiter
  // in pending's release sequence, so all tasks' ran_ok stores are visible to
  // the deferred-clear below in BOTH paths. A wake landing between here and
  // suspend() is caught by the wake-before-suspend rendezvous (waker set
  // above).
  if (st->pending.fetch_sub(1, std::memory_order_acq_rel) != 1)
    sched::suspend();

  // Resumed: every task has finished (the waker fired on pending==0). ALL
  // workers are done, so we (the single waiter fiber) now apply the DEFERRED
  // dirty-clears single-threaded -- no more concurrent graph writes. A node
  // whose ran_ok==1 succeeded -> clear it; nodes gated by a failed ancestor
  // never ran (ran_ok==0) -> stay dirty (deferred, correct).
  for (std::uint32_t i = 0; i < n; ++i)
    if (st->ran_ok[i].load(std::memory_order_acquire) == 1)
      g.clear_dirty_(i);

  // Lift the result, then drop the waiter's ref (frees st + rem + ran_ok +
  // outcome; succ/succ_off die with this frame).
  Status result;
  if (st->had_err.load(std::memory_order_acquire)) {
    st->err_mu.lock();
    result = st->first_err;
    st->err_mu.unlock();
  }
  st->drop_ref();
  return result;
}

} // namespace hyle::graph
