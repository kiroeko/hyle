#include "hyle/graph/parallel.h"

#include <gtest/gtest.h>

#include <atomic>
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
  Fiber f(system_allocator(), kDefaultStackSize, &drive_entry, d.get());
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
  Fiber f(system_allocator(), kDefaultStackSize, &drive_entry, d.get());
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
  Fiber f(system_allocator(), kDefaultStackSize, &drive_entry, d.get());
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
  Fiber ff(system_allocator(), kDefaultStackSize, &drive_entry, d.get());
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
  Fiber f(system_allocator(), kDefaultStackSize, &drive_entry, d.get());
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
  Fiber f(system_allocator(), kDefaultStackSize, &drive_entry, d.get());
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
  Fiber f(system_allocator(), kDefaultStackSize, &drive_entry, d.get());
  sched.spawn(&f);
  sched.drain();
  sched.shutdown();
  EXPECT_EQ(d->task.done() ? d->task.result() : -999, 1);  // ran exactly once
}

}  // namespace
}  // namespace hyle::graph
