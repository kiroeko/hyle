#include "hyle/graph/parallel.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <memory>

#include "hyle/graph/taskgraph.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

namespace hyle::graph {
namespace {

using mem::system_allocator;
using sched::Fiber;
using sched::kDefaultStackSize;
using sched::Scheduler;
using sched::Task;
using mem::Allocator;

// A plain malloc-backed Allocator for the wavefront concurrency tests (same rationale as
// sched/join_test.cc:29): system_allocator() is mimalloc, whose OWN mmap arena bypasses
// TSan's malloc interceptors -- so under the wavefront's rapid spawn/free/REUSE churn TSan
// cannot see the free->reuse happens-before edge and FALSE-POSITIVES a data race on the
// reused coroutine-frame address (the exact [anon:mimalloc] reuse the trace showed). This
// malloc-backed allocator is TSan-VISIBLE, keeping the race gate meaningful. Non-wavefront
// tests keep system_allocator() (the shipping path).
#if defined(_MSC_VER)
void* tv_a(void*, std::size_t n, std::size_t al) noexcept {
  if (al < sizeof(void*)) al = sizeof(void*);
  return _aligned_malloc(n, al);
}
void tv_f(void*, void* p, std::size_t, std::size_t) noexcept { _aligned_free(p); }
void* tv_r(void*, void* p, std::size_t, std::size_t nn, std::size_t al) noexcept {
  if (al < sizeof(void*)) al = sizeof(void*);
  return _aligned_realloc(p, nn, al);
}
#else
void* tv_a(void*, std::size_t n, std::size_t al) noexcept {
  if (al < sizeof(void*)) al = sizeof(void*);
  return std::aligned_alloc(al, (n + al - 1) & ~(al - 1));
}
void tv_f(void*, void* p, std::size_t, std::size_t) noexcept { std::free(p); }
void* tv_r(void*, void* p, std::size_t, std::size_t nn, std::size_t) noexcept {
  return std::realloc(p, nn);
}
#endif
Allocator tsan_visible_allocator() noexcept { return Allocator(&tv_a, &tv_f, &tv_r, nullptr); }

// Node ctx: writes value + counts runs, like the serial test. `peak` shared across a
// level's nodes lets us observe concurrency (see WideLevelRunsConcurrently).
struct Cell {
  int value = 0;
  std::atomic<int> runs{0};
};
struct Source {
  Cell cell;
  int seed = 0;
};
Status source_fn(void* p) noexcept {
  auto* s = static_cast<Source*>(p);
  s->cell.value = s->seed;
  s->cell.runs.fetch_add(1, std::memory_order_relaxed);
  return Status();
}
struct Add {
  Cell cell;
  const Cell* a = nullptr;
  const Cell* b = nullptr;
};
Status add_fn(void* p) noexcept {
  auto* n = static_cast<Add*>(p);
  n->cell.value = n->a->value + n->b->value;
  n->cell.runs.fetch_add(1, std::memory_order_relaxed);
  return Status();
}

// Fiber-driven harness: run `body(alloc)` (a coroutine returning int) on a Scheduler and
// return its result. body calls evaluate_parallel, which must run from a fiber.
struct Drive {
  Task<int> task;
  std::atomic<bool> finished{false};
};
void drive_entry(void* p) {
  auto* d = static_cast<Drive*>(p);
  d->task.resume();
  d->finished.store(true, std::memory_order_release);
}

// ---- correctness: parallel result == serial result (the oracle) --------------
Task<int> diamond_coro(mem::Allocator alloc, Scheduler* sched, Source* a, Add* b, Add* c,
                       Add* e, int* out_val) {
  TaskGraph g(alloc);
  NodeId ia = g.add_source(source_fn, a);
  b->a = &a->cell; b->b = &a->cell;
  c->a = &a->cell; c->b = &a->cell;
  NodeId da[] = {ia};
  NodeId ib = g.add_node(add_fn, b, da, 1);
  NodeId ic = g.add_node(add_fn, c, da, 1);
  e->a = &b->cell; e->b = &c->cell;
  NodeId dbc[] = {ib, ic};
  (void)g.add_node(add_fn, e, dbc, 2);

  Status s = evaluate_parallel(g, *sched, alloc);
  if (!s.ok()) co_return -1;
  *out_val = e->cell.value;
  co_return 0;
}

TEST(GraphParallel, DiamondMatchesSerial) {
  Source a{{}, 5};
  Add b, c, e;
  int val = -1;
  Scheduler sched(4, system_allocator(), 256);
  auto d = std::make_unique<Drive>();
  d->task = diamond_coro(system_allocator(), &sched, &a, &b, &c, &e, &val);
  Fiber f(kDefaultStackSize, &drive_entry, d.get());
  sched.spawn(&f);
  sched.drain();
  sched.shutdown();
  EXPECT_TRUE(d->finished.load());
  EXPECT_EQ(d->task.done() ? d->task.result() : -999, 0);
  EXPECT_EQ(val, 20);  // (2*5)+(2*5), same as serial DiamondFloodsAllPathsOnce
  EXPECT_EQ(b.cell.runs.load(), 1);
  EXPECT_EQ(c.cell.runs.load(), 1);
  EXPECT_EQ(e.cell.runs.load(), 1);
}

// ---- incremental: after a full eval, dirty one input -> only its subgraph re-runs ----
Task<int> incr_coro(mem::Allocator alloc, Scheduler* sched, Source* a, Source* b, Add* c,
                    int* out_val) {
  TaskGraph g(alloc);
  NodeId ia = g.add_source(source_fn, a);
  NodeId ib = g.add_source(source_fn, b);
  c->a = &a->cell; c->b = &b->cell;
  NodeId deps[] = {ia, ib};
  (void)g.add_node(add_fn, c, deps, 2);
  if (!evaluate_parallel(g, *sched, alloc).ok()) co_return -1;

  a->seed = 100;
  g.mark_dirty(ia);
  if (!evaluate_parallel(g, *sched, alloc).ok()) co_return -2;
  *out_val = c->cell.value;
  co_return 0;
}

TEST(GraphParallel, IncrementalReRunsOnlyDirty) {
  Source a{{}, 3}, b{{}, 4};
  Add c;
  int val = -1;
  Scheduler sched(4, system_allocator(), 256);
  auto d = std::make_unique<Drive>();
  d->task = incr_coro(system_allocator(), &sched, &a, &b, &c, &val);
  Fiber f(kDefaultStackSize, &drive_entry, d.get());
  sched.spawn(&f);
  sched.drain();
  sched.shutdown();
  EXPECT_EQ(d->task.done() ? d->task.result() : -999, 0);
  EXPECT_EQ(val, 104);              // 100 + 4
  EXPECT_EQ(a.cell.runs.load(), 2);  // a re-ran
  EXPECT_EQ(b.cell.runs.load(), 1);  // b skipped (clean, cache reused)
  EXPECT_EQ(c.cell.runs.load(), 2);  // c re-ran (depends on a)
}

// ---- concurrency: a wide level's nodes actually overlap across workers --------
// Each node bumps a live-counter on entry, spins until it sees >1 live (or a bound),
// then decrements. If the level truly runs in parallel, the peak live count exceeds 1.
struct SpinCell {
  std::atomic<int>* live;
  std::atomic<int>* peak;
  int value = 0;
};
Status spin_fn(void* p) noexcept {
  auto* s = static_cast<SpinCell*>(p);
  int now = s->live->fetch_add(1, std::memory_order_acq_rel) + 1;
  int cur_peak = s->peak->load(std::memory_order_relaxed);
  while (now > cur_peak &&
         !s->peak->compare_exchange_weak(cur_peak, now, std::memory_order_relaxed)) {
  }
  // Spin briefly so siblings have a chance to be live simultaneously (bounded, no hang).
  volatile int sink = 0;
  for (int i = 0; i < 200000; ++i) sink = sink + 1;
  (void)sink;
  s->live->fetch_sub(1, std::memory_order_acq_rel);
  s->value = 1;
  return Status();
}
Task<int> wide_coro(mem::Allocator alloc, Scheduler* sched, SpinCell* cells, int nwide,
                    std::atomic<int>* peak) {
  TaskGraph g(alloc);
  for (int i = 0; i < nwide; ++i) (void)g.add_source(spin_fn, &cells[i]);
  if (!evaluate_parallel(g, *sched, alloc).ok()) co_return -1;
  co_return peak->load(std::memory_order_relaxed);
}

TEST(GraphParallel, WideLevelRunsConcurrently) {
  constexpr int kWide = 8;
  std::atomic<int> live{0}, peak{0};
  SpinCell cells[kWide];
  for (auto& c : cells) { c.live = &live; c.peak = &peak; }
  Scheduler sched(4, system_allocator(), 256);
  auto d = std::make_unique<Drive>();
  d->task = wide_coro(system_allocator(), &sched, cells, kWide, &peak);
  Fiber f(kDefaultStackSize, &drive_entry, d.get());
  sched.spawn(&f);
  sched.drain();
  sched.shutdown();
  int observed_peak = d->task.done() ? d->task.result() : -999;
  // With 4 workers and 8 independent same-level nodes, peak live must exceed 1 (true
  // concurrency). Exact value is racy; >1 is the meaningful assertion.
  EXPECT_GT(observed_peak, 1) << "same-level nodes did not overlap -> not parallel";
  for (auto& c : cells) EXPECT_EQ(c.value, 1);
}

// ---- failure: a failing node surfaces its Status and stays dirty --------------
struct FailCell {
  bool fail = true;
  int runs = 0;
};
Status fail_fn(void* p) noexcept {
  auto* f = static_cast<FailCell*>(p);
  ++f->runs;
  return f->fail ? Status::Internal("boom") : Status();
}
Task<int> fail_coro(mem::Allocator alloc, Scheduler* sched, FailCell* f, bool* was_err,
                    bool* still_dirty) {
  TaskGraph g(alloc);
  NodeId nf = g.add_source(fail_fn, f);
  Status s = evaluate_parallel(g, *sched, alloc);
  *was_err = !s.ok();
  *still_dirty = g.is_dirty(nf);
  co_return 0;
}

TEST(GraphParallel, FailingNodeSurfacesAndStaysDirty) {
  FailCell f;
  bool was_err = false, still_dirty = false;
  Scheduler sched(4, system_allocator(), 256);
  auto d = std::make_unique<Drive>();
  d->task = fail_coro(system_allocator(), &sched, &f, &was_err, &still_dirty);
  Fiber ff(kDefaultStackSize, &drive_entry, d.get());
  sched.spawn(&ff);
  sched.drain();
  sched.shutdown();
  EXPECT_TRUE(was_err);
  EXPECT_TRUE(still_dirty);
}

// A node failing in a MIDDLE level must STOP later levels from running (verifies the
// break). Chain a(src) -> mid(fails, level 1) -> tail(level 2): tail must NOT run.
Task<int> mid_fail_coro(mem::Allocator alloc, Scheduler* sched, Source* a, FailCell* mid,
                        FailCell* tail, bool* mid_dirty, int* tail_runs) {
  TaskGraph g(alloc);
  NodeId ia = g.add_source(source_fn, a);
  NodeId da[] = {ia};
  NodeId im = g.add_node(fail_fn, mid, da, 1);   // level 1, fails
  NodeId dm[] = {im};
  (void)g.add_node(fail_fn, tail, dm, 1);        // level 2, must not run
  Status s = evaluate_parallel(g, *sched, alloc);
  *mid_dirty = g.is_dirty(im);
  *tail_runs = tail->runs;
  co_return s.ok() ? 0 : 1;
}

TEST(GraphParallel, MidLevelFailureStopsLaterLevels) {
  Source a{{}, 1};
  FailCell mid{true, 0}, tail{true, 0};
  bool mid_dirty = false;
  int tail_runs = -1, rc = -1;
  Scheduler sched(4, system_allocator(), 256);
  auto d = std::make_unique<Drive>();
  d->task = mid_fail_coro(system_allocator(), &sched, &a, &mid, &tail, &mid_dirty, &tail_runs);
  Fiber f(kDefaultStackSize, &drive_entry, d.get());
  sched.spawn(&f);
  sched.drain();
  sched.shutdown();
  rc = d->task.done() ? d->task.result() : -999;
  EXPECT_EQ(rc, 1);              // evaluate returned the error
  EXPECT_EQ(mid.runs, 1);        // mid ran (and failed)
  EXPECT_EQ(tail_runs, 0);       // tail NEVER ran -- later level stopped
  EXPECT_TRUE(mid_dirty);        // failed node stays dirty
}

// A wide level with a MIX of dirty + clean nodes: only the dirty ones re-run in parallel;
// clean siblings at the same level are skipped. Also a 3-level chain under parallel.
Task<int> mixed_coro(mem::Allocator alloc, Scheduler* sched, Source* s0, Source* s1,
                     Add* mid, Add* tail, int* out) {
  TaskGraph g(alloc);
  NodeId i0 = g.add_source(source_fn, s0);   // level 0
  NodeId i1 = g.add_source(source_fn, s1);   // level 0
  mid->a = &s0->cell; mid->b = &s1->cell;
  NodeId dmid[] = {i0, i1};
  NodeId im = g.add_node(add_fn, mid, dmid, 2);  // level 1
  tail->a = &mid->cell; tail->b = &mid->cell;
  NodeId dt[] = {im};
  (void)g.add_node(add_fn, tail, dt, 1);         // level 2
  if (!evaluate_parallel(g, *sched, alloc).ok()) co_return -1;

  // Edit only s1; s0 stays clean. Level 0 now has one dirty (s1) + one clean (s0).
  s1->seed = 50;
  g.mark_dirty(i1);
  if (!evaluate_parallel(g, *sched, alloc).ok()) co_return -2;
  *out = tail->cell.value;
  co_return 0;
}

TEST(GraphParallel, MixedDirtyCleanAndMultiLevel) {
  Source s0{{}, 2}, s1{{}, 3};
  Add mid, tail;
  int out = -1;
  Scheduler sched(4, system_allocator(), 256);
  auto d = std::make_unique<Drive>();
  d->task = mixed_coro(system_allocator(), &sched, &s0, &s1, &mid, &tail, &out);
  Fiber f(kDefaultStackSize, &drive_entry, d.get());
  sched.spawn(&f);
  sched.drain();
  sched.shutdown();
  EXPECT_EQ(d->task.done() ? d->task.result() : -999, 0);
  EXPECT_EQ(out, 104);             // mid = 50+2 = 52; tail = 52+52 = 104
  EXPECT_EQ(s0.cell.runs.load(), 1);  // s0 clean -> skipped on 2nd eval
  EXPECT_EQ(s1.cell.runs.load(), 2);  // s1 dirty -> re-ran
  EXPECT_EQ(mid.cell.runs.load(), 2); // downstream of s1 -> re-ran
  EXPECT_EQ(tail.cell.runs.load(), 2);
}

// Empty graph + all-clean re-evaluate are both no-ops (no spawn, no crash).
Task<int> empty_and_clean_coro(mem::Allocator alloc, Scheduler* sched, Source* a) {
  TaskGraph empty(alloc);
  if (!evaluate_parallel(empty, *sched, alloc).ok()) co_return -1;  // n==0 early return

  TaskGraph g(alloc);
  (void)g.add_source(source_fn, a);
  if (!evaluate_parallel(g, *sched, alloc).ok()) co_return -2;
  if (!evaluate_parallel(g, *sched, alloc).ok()) co_return -3;  // all clean -> no-op
  co_return a->cell.runs.load();  // must be 1 (ran once, second eval skipped)
}

TEST(GraphParallel, EmptyAndAllCleanAreNoOps) {
  Source a{{}, 7};
  Scheduler sched(4, system_allocator(), 256);
  auto d = std::make_unique<Drive>();
  d->task = empty_and_clean_coro(system_allocator(), &sched, &a);
  Fiber f(kDefaultStackSize, &drive_entry, d.get());
  sched.spawn(&f);
  sched.drain();
  sched.shutdown();
  EXPECT_EQ(d->task.done() ? d->task.result() : -999, 1);  // ran exactly once
}

Task<int> admission_failure_coro(mem::Allocator alloc, Scheduler* sched,
                                 Source* sources, int count, bool wavefront,
                                 int* ran, int* dirty) {
  TaskGraph g(alloc);
  NodeId ids[8]{};
  for (int i = 0; i < count; ++i) ids[i] = g.add_source(source_fn, &sources[i]);

  Status s = wavefront ? evaluate_parallel_wavefront(g, *sched, alloc)
                       : evaluate_parallel(g, *sched, alloc);
  *ran = 0;
  *dirty = 0;
  for (int i = 0; i < count; ++i) {
    *ran += sources[i].cell.runs.load(std::memory_order_relaxed);
    *dirty += g.is_dirty(ids[i]) ? 1 : 0;
  }
  co_return s.ok() ? 0 : (s.IsResourceExhausted() ? 1 : 2);
}

int run_admission_failure_case(bool wavefront, Source* sources, int count,
                               int* ran, int* dirty) {
  // Four runnable slots total (one local + one global queue, capacity two each)
  // cannot hold eight seeds while this sole worker is still inside the caller
  // fiber.  This deterministically exercises worker-side admission failure.
  Scheduler sched(1, system_allocator(), 2);
  auto d = std::make_unique<Drive>();
  d->task = admission_failure_coro(tsan_visible_allocator(), &sched, sources,
                                   count, wavefront, ran, dirty);
  Fiber f(kDefaultStackSize, &drive_entry, d.get());
  EXPECT_TRUE(sched.spawn(&f));
  sched.drain();
  sched.shutdown();
  EXPECT_TRUE(d->finished.load(std::memory_order_acquire));
  return d->task.done() ? d->task.result() : -999;
}

Task<int> successor_admission_failure_coro(
    mem::Allocator alloc, Scheduler* sched, Source* root, Source* successors,
    int count, int* ran, int* dirty) {
  TaskGraph g(alloc);
  const NodeId root_id = g.add_source(source_fn, root);
  NodeId ids[8]{};
  for (int i = 0; i < count; ++i) {
    ids[i] = g.add_node(source_fn, &successors[i], &root_id, 1);
  }

  Status s = evaluate_parallel_wavefront(g, *sched, alloc);
  *ran = root->cell.runs.load(std::memory_order_relaxed);
  *dirty = g.is_dirty(root_id) ? 1 : 0;
  for (int i = 0; i < count; ++i) {
    *ran += successors[i].cell.runs.load(std::memory_order_relaxed);
    *dirty += g.is_dirty(ids[i]) ? 1 : 0;
  }
  co_return s.ok() ? 0 : (s.IsResourceExhausted() ? 1 : 2);
}

int run_successor_admission_failure_case(Source* root, Source* successors,
                                         int count, int* ran, int* dirty) {
  Scheduler sched(1, system_allocator(), 2);
  auto d = std::make_unique<Drive>();
  d->task = successor_admission_failure_coro(
      tsan_visible_allocator(), &sched, root, successors, count, ran, dirty);
  Fiber f(kDefaultStackSize, &drive_entry, d.get());
  EXPECT_TRUE(sched.spawn(&f));
  sched.drain();
  sched.shutdown();
  EXPECT_TRUE(d->finished.load(std::memory_order_acquire));
  return d->task.done() ? d->task.result() : -999;
}

TEST(GraphParallel, BoundedQueueAdmissionFailureStaysStructured) {
  constexpr int kNodes = 8;
  Source level_sources[kNodes];
  Source wavefront_sources[kNodes];
  for (int i = 0; i < kNodes; ++i) {
    level_sources[i].seed = i;
    wavefront_sources[i].seed = i;
  }

  int level_ran = -1, level_dirty = -1;
  EXPECT_EQ(run_admission_failure_case(false, level_sources, kNodes,
                                       &level_ran, &level_dirty),
            1);
  EXPECT_GT(level_ran, 0);
  EXPECT_LT(level_ran, kNodes);
  EXPECT_EQ(level_dirty, kNodes - level_ran);

  int wavefront_ran = -1, wavefront_dirty = -1;
  EXPECT_EQ(run_admission_failure_case(true, wavefront_sources, kNodes,
                                       &wavefront_ran, &wavefront_dirty),
            1);
  EXPECT_GT(wavefront_ran, 0);
  EXPECT_LT(wavefront_ran, kNodes);
  EXPECT_EQ(wavefront_dirty, kNodes - wavefront_ran);

  // Exercise the other wf_spawn call site: one admitted seed releases a fan-out
  // from inside wf_node_task.  The parent task keeps completion accounting live
  // while the fifth successor is rejected by the same bounded queues.
  Source root{{}, 100};
  Source successors[kNodes];
  for (int i = 0; i < kNodes; ++i) successors[i].seed = i;
  int successor_ran = -1, successor_dirty = -1;
  EXPECT_EQ(run_successor_admission_failure_case(
                &root, successors, kNodes, &successor_ran, &successor_dirty),
            1);
  EXPECT_GT(successor_ran, 1);           // root + at least one successor ran
  EXPECT_LT(successor_ran, kNodes + 1);  // at least one successor was rejected
  EXPECT_EQ(successor_dirty, kNodes + 1 - successor_ran);
}

// ==== step-3 wavefront: evaluate_parallel_wavefront ==========================

// Correctness oracle: wavefront gives the SAME result as the level-barrier path on a
// diamond (a -> b,c -> e). Each node runs exactly once.
Task<int> wf_diamond_coro(mem::Allocator alloc, Scheduler* sched, Source* a, Add* b, Add* c,
                          Add* e, int* out_val) {
  TaskGraph g(alloc);
  NodeId ia = g.add_source(source_fn, a);
  b->a = &a->cell; b->b = &a->cell;
  c->a = &a->cell; c->b = &a->cell;
  NodeId da[] = {ia};
  NodeId ib = g.add_node(add_fn, b, da, 1);
  NodeId ic = g.add_node(add_fn, c, da, 1);
  e->a = &b->cell; e->b = &c->cell;
  NodeId dbc[] = {ib, ic};
  (void)g.add_node(add_fn, e, dbc, 2);
  if (!evaluate_parallel_wavefront(g, *sched, alloc).ok()) co_return -1;
  *out_val = e->cell.value;
  co_return 0;
}

TEST(GraphParallel, WavefrontDiamondMatchesSerial) {
  Source a{{}, 5};
  Add b, c, e;
  int val = -1;
  Scheduler sched(4, system_allocator(), 256);
  auto d = std::make_unique<Drive>();
  d->task = wf_diamond_coro(tsan_visible_allocator(), &sched, &a, &b, &c, &e, &val);
  Fiber f(kDefaultStackSize, &drive_entry, d.get());
  sched.spawn(&f);
  sched.drain();
  sched.shutdown();
  EXPECT_EQ(d->task.done() ? d->task.result() : -999, 0);
  EXPECT_EQ(val, 20);  // (2*5)+(2*5) -- same as fork-join DiamondMatchesSerial
  EXPECT_EQ(b.cell.runs.load(), 1);
  EXPECT_EQ(c.cell.runs.load(), 1);
  EXPECT_EQ(e.cell.runs.load(), 1);
}

// Incremental under wavefront: after a full eval, dirty one input -> only its cone re-runs.
Task<int> wf_incr_coro(mem::Allocator alloc, Scheduler* sched, Source* a, Source* b, Add* c,
                       int* out_val) {
  TaskGraph g(alloc);
  NodeId ia = g.add_source(source_fn, a);
  NodeId ib = g.add_source(source_fn, b);
  c->a = &a->cell; c->b = &b->cell;
  NodeId deps[] = {ia, ib};
  (void)g.add_node(add_fn, c, deps, 2);
  if (!evaluate_parallel_wavefront(g, *sched, alloc).ok()) co_return -1;
  a->seed = 100;
  g.mark_dirty(ia);
  if (!evaluate_parallel_wavefront(g, *sched, alloc).ok()) co_return -2;
  *out_val = c->cell.value;
  co_return 0;
}

TEST(GraphParallel, WavefrontIncrementalReRunsOnlyDirty) {
  Source a{{}, 3}, b{{}, 4};
  Add c;
  int val = -1;
  Scheduler sched(4, system_allocator(), 256);
  auto d = std::make_unique<Drive>();
  d->task = wf_incr_coro(tsan_visible_allocator(), &sched, &a, &b, &c, &val);
  Fiber f(kDefaultStackSize, &drive_entry, d.get());
  sched.spawn(&f);
  sched.drain();
  sched.shutdown();
  EXPECT_EQ(d->task.done() ? d->task.result() : -999, 0);
  EXPECT_EQ(val, 104);
  EXPECT_EQ(a.cell.runs.load(), 2);
  EXPECT_EQ(b.cell.runs.load(), 1);  // clean -> skipped
  EXPECT_EQ(c.cell.runs.load(), 2);
}

// THE point of wavefront: the short leg's downstream finishes WITHOUT waiting for a slow
// same-"level" sibling. Graph: slow(level 0, sleeps) beside a chain fast0->fast1->fast2
// (levels 0,1,2). Under a LEVEL barrier, fast1 (level 1) could not start until slow
// (level 0) finished. Under wavefront, fast1/fast2 run while slow is still sleeping. We
// prove it by recording whether the fast chain's tail completed BEFORE slow did.
struct Timed {
  std::atomic<long>* clock;   // shared logical clock (fetch_add on each completion)
  long finish_tick = -1;
  int ms = 0;                 // busy-spin duration knob
  int value = 0;
};
Status timed_fn(void* p) noexcept {
  auto* t = static_cast<Timed*>(p);
  volatile long sink = 0;
  const long spins = static_cast<long>(t->ms) * 300000L;
  for (long i = 0; i < spins; ++i) sink = sink + 1;
  (void)sink;
  t->finish_tick = t->clock->fetch_add(1, std::memory_order_acq_rel);
  t->value = 1;
  return Status();
}
Task<int> wf_legs_coro(mem::Allocator alloc, Scheduler* sched, Timed* slow, Timed* f0,
                       Timed* f1, Timed* f2) {
  TaskGraph g(alloc);
  (void)g.add_source(timed_fn, slow);          // level 0, SLOW
  NodeId if0 = g.add_source(timed_fn, f0);     // level 0, fast
  NodeId if1 = g.add_node(timed_fn, f1, &if0, 1);  // level 1
  (void)g.add_node(timed_fn, f2, &if1, 1);     // level 2
  if (!evaluate_parallel_wavefront(g, *sched, alloc).ok()) co_return -1;
  co_return 0;
}

TEST(GraphParallel, WavefrontShortLegNotBlockedBySlowSibling) {
  std::atomic<long> clock{0};
  Timed slow{&clock, -1, 60, 0};   // ~slow
  Timed f0{&clock, -1, 0, 0}, f1{&clock, -1, 0, 0}, f2{&clock, -1, 0, 0};  // fast chain
  Scheduler sched(4, system_allocator(), 256);
  auto d = std::make_unique<Drive>();
  d->task = wf_legs_coro(tsan_visible_allocator(), &sched, &slow, &f0, &f1, &f2);
  Fiber f(kDefaultStackSize, &drive_entry, d.get());
  sched.spawn(&f);
  sched.drain();
  sched.shutdown();
  EXPECT_EQ(d->task.done() ? d->task.result() : -999, 0);
  // The fast chain's TAIL (f2, a level-2 node) must finish BEFORE the slow level-0 node.
  // A level barrier would force f1/f2 to wait for slow (same/earlier level) -> impossible.
  // Wavefront lets the fast chain race ahead. finish_tick is the completion order.
  EXPECT_GE(slow.finish_tick, 0);
  EXPECT_GE(f2.finish_tick, 0);
  EXPECT_LT(f2.finish_tick, slow.finish_tick)
      << "fast chain tail did not overtake the slow sibling -> not wavefront";
}

// Failure under wavefront: a failing node surfaces its Status, stays dirty, and its
// successors are never released (never run).
Task<int> wf_fail_coro(mem::Allocator alloc, Scheduler* sched, FailCell* f, Add* down,
                       Cell* dummy, bool* was_err, bool* f_dirty, int* down_runs) {
  TaskGraph g(alloc);
  NodeId nf = g.add_source(fail_fn, f);
  // down depends on the failing node; it must NEVER run, so its input cells are unused
  // placeholders (fail_fn writes no Cell). It stays gated because nf fails.
  down->a = dummy; down->b = dummy;
  NodeId df[] = {nf};
  (void)g.add_node(add_fn, down, df, 1);
  Status s = evaluate_parallel_wavefront(g, *sched, alloc);
  *was_err = !s.ok();
  *f_dirty = g.is_dirty(nf);
  *down_runs = down->cell.runs.load();
  co_return 0;
}

TEST(GraphParallel, WavefrontFailureStopsSuccessors) {
  FailCell f;
  Add down;
  Cell dummy;
  bool was_err = false, f_dirty = false;
  int down_runs = -1;
  Scheduler sched(4, system_allocator(), 256);
  auto d = std::make_unique<Drive>();
  d->task = wf_fail_coro(tsan_visible_allocator(), &sched, &f, &down, &dummy, &was_err, &f_dirty, &down_runs);
  Fiber ff(kDefaultStackSize, &drive_entry, d.get());
  sched.spawn(&ff);
  sched.drain();
  sched.shutdown();
  EXPECT_TRUE(was_err);
  EXPECT_TRUE(f_dirty);
  EXPECT_EQ(down_runs, 0);  // successor never released after the failure
}

// REGRESSION for the red-team-found stack-Channel UAF: MANY independent TERMINAL dirty
// nodes (no successors) finishing near-simultaneously. This is the exact shape that races
// the driver's drain-to-zero against a sender's post-unlock notify_one() -- if the driver
// returned (destroying the stack Channel) before every task's send() fully returned, the
// trailing notify_one() would hit freed memory. The fix (join ALL handles before return)
// must hold. Run under TSan/ASan to actually catch a regression; here we at least exercise
// the shape heavily and assert correctness. Uses `Cell` + a trivial source per node.
struct Leaf { Cell cell; std::atomic<int>* live; std::atomic<int>* peak; };
Status leaf_fn(void* p) noexcept {
  auto* l = static_cast<Leaf*>(p);
  int now = l->live->fetch_add(1, std::memory_order_acq_rel) + 1;
  int cp = l->peak->load(std::memory_order_relaxed);
  while (now > cp && !l->peak->compare_exchange_weak(cp, now, std::memory_order_relaxed)) {}
  volatile int s = 0; for (int i = 0; i < 50000; ++i) s = s + 1; (void)s;
  l->live->fetch_sub(1, std::memory_order_acq_rel);
  l->cell.value = 1;
  return Status();
}
Task<int> wf_many_terminals_coro(mem::Allocator alloc, Scheduler* sched, Leaf* leaves,
                                 int nleaves, std::atomic<int>* peak) {
  TaskGraph g(alloc);
  for (int i = 0; i < nleaves; ++i) (void)g.add_source(leaf_fn, &leaves[i]);  // all terminal
  if (!evaluate_parallel_wavefront(g, *sched, alloc).ok()) co_return -1;
  co_return peak->load(std::memory_order_relaxed);
}

TEST(GraphParallel, WavefrontManyIndependentTerminalsNoUAF) {
  constexpr int kLeaves = 16;
  std::atomic<int> live{0}, peak{0};
  Leaf leaves[kLeaves];
  for (auto& l : leaves) { l.live = &live; l.peak = &peak; }
  Scheduler sched(4, system_allocator(), 256);
  auto d = std::make_unique<Drive>();
  d->task = wf_many_terminals_coro(tsan_visible_allocator(), &sched, leaves, kLeaves, &peak);
  Fiber f(kDefaultStackSize, &drive_entry, d.get());
  sched.spawn(&f);
  sched.drain();
  sched.shutdown();
  int observed = d->task.done() ? d->task.result() : -999;
  EXPECT_GE(observed, 1);                     // completed cleanly (UAF would crash under ASan)
  for (auto& l : leaves) EXPECT_EQ(l.cell.value, 1);
}

// A node with MIXED dirty + clean deps: only the dirty dep gates it (the clean dep is
// already fresh). Full eval, then dirty ONE of two inputs; the sink recomputes with the
// stale-but-clean input reused + the freshly recomputed one.
Task<int> wf_mixed_deps_coro(mem::Allocator alloc, Scheduler* sched, Source* a, Source* b,
                             Add* sink, int* out) {
  TaskGraph g(alloc);
  NodeId ia = g.add_source(source_fn, a);
  NodeId ib = g.add_source(source_fn, b);
  sink->a = &a->cell; sink->b = &b->cell;
  NodeId deps[] = {ia, ib};
  (void)g.add_node(add_fn, sink, deps, 2);
  if (!evaluate_parallel_wavefront(g, *sched, alloc).ok()) co_return -1;
  // Dirty only a. On the next wavefront, sink has ONE dirty dep (a) + ONE clean (b).
  a->seed = 30;
  g.mark_dirty(ia);
  if (!evaluate_parallel_wavefront(g, *sched, alloc).ok()) co_return -2;
  *out = sink->cell.value;
  co_return 0;
}

TEST(GraphParallel, WavefrontMixedDirtyCleanDeps) {
  Source a{{}, 1}, b{{}, 2};
  Add sink;
  int out = -1;
  Scheduler sched(4, system_allocator(), 256);
  auto d = std::make_unique<Drive>();
  d->task = wf_mixed_deps_coro(tsan_visible_allocator(), &sched, &a, &b, &sink, &out);
  Fiber f(kDefaultStackSize, &drive_entry, d.get());
  sched.spawn(&f);
  sched.drain();
  sched.shutdown();
  EXPECT_EQ(d->task.done() ? d->task.result() : -999, 0);
  EXPECT_EQ(out, 32);                 // 30 (re-run a) + 2 (clean b reused)
  EXPECT_EQ(a.cell.runs.load(), 2);   // dirty dep re-ran
  EXPECT_EQ(b.cell.runs.load(), 1);   // clean dep did NOT gate/run
  EXPECT_EQ(sink.cell.runs.load(), 2);
}

// Empty + all-clean wavefront are no-ops (dirty_total==0 early return).
Task<int> wf_empty_clean_coro(mem::Allocator alloc, Scheduler* sched, Source* a) {
  TaskGraph empty(alloc);
  if (!evaluate_parallel_wavefront(empty, *sched, alloc).ok()) co_return -1;  // n==0
  TaskGraph g(alloc);
  (void)g.add_source(source_fn, a);
  if (!evaluate_parallel_wavefront(g, *sched, alloc).ok()) co_return -2;
  if (!evaluate_parallel_wavefront(g, *sched, alloc).ok()) co_return -3;  // all clean -> no-op
  co_return a->cell.runs.load();
}

TEST(GraphParallel, WavefrontEmptyAndAllCleanNoOp) {
  Source a{{}, 7};
  Scheduler sched(4, system_allocator(), 256);
  auto d = std::make_unique<Drive>();
  d->task = wf_empty_clean_coro(tsan_visible_allocator(), &sched, &a);
  Fiber f(kDefaultStackSize, &drive_entry, d.get());
  sched.spawn(&f);
  sched.drain();
  sched.shutdown();
  EXPECT_EQ(d->task.done() ? d->task.result() : -999, 1);  // ran exactly once
}

}  // namespace
}  // namespace hyle::graph
