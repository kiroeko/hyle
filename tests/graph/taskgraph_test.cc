#include "hyle/graph/taskgraph.h"

#include <gtest/gtest.h>

#include "hyle/mem/allocator.h"

// Threadsafe death style on Linux only (re-exec vs fork). Windows death tests are always
// spawn-based, so the flag is unneeded -- and testing::FLAGS_gtest_death_test_style
// doesn't link against the gtest DLL there (see compute/ggml_bridge_test.cc:37).
#if defined(_WIN32)
#  define HYLE_THREADSAFE_DEATH() ((void)0)
#else
#  define HYLE_THREADSAFE_DEATH() GTEST_FLAG_SET(death_test_style, "threadsafe")
#endif

namespace hyle::graph {
namespace {

using mem::system_allocator;

// A node's ctx: it reads its inputs (pointers to upstream outputs) and writes its own
// output, and BUMPS a run-counter so a test can prove exactly which nodes recomputed.
struct Cell {
  int value = 0;
  int runs = 0;  // how many times this node's ComputeFn actually executed
};

// source node: output = its `seed` (a user-editable input). Counts runs.
struct Source {
  Cell cell;
  int seed = 0;
};
Status source_fn(void* p) noexcept {
  auto* s = static_cast<Source*>(p);
  s->cell.value = s->seed;
  ++s->cell.runs;
  return Status();
}

// binary node: output = a->value + b->value. Counts runs.
struct Add {
  Cell cell;
  const Cell* a = nullptr;
  const Cell* b = nullptr;
};
Status add_fn(void* p) noexcept {
  auto* n = static_cast<Add*>(p);
  n->cell.value = n->a->value + n->b->value;
  ++n->cell.runs;
  return Status();
}

TEST(TaskGraph, FirstEvaluateRunsEveryNode) {
  TaskGraph g(system_allocator());
  Source a{{}, 3}, b{{}, 4};
  Add c;
  NodeId ia = g.add_source(source_fn, &a);
  NodeId ib = g.add_source(source_fn, &b);
  c.a = &a.cell;
  c.b = &b.cell;
  NodeId deps[] = {ia, ib};
  NodeId ic = g.add_node(add_fn, &c, deps, 2);

  EXPECT_EQ(g.num_nodes(), 3u);
  EXPECT_TRUE(g.is_dirty(ia));
  EXPECT_TRUE(g.is_dirty(ic));

  ASSERT_TRUE(g.evaluate().ok());
  EXPECT_EQ(c.cell.value, 7);              // 3 + 4
  EXPECT_EQ(a.cell.runs, 1);
  EXPECT_EQ(b.cell.runs, 1);
  EXPECT_EQ(c.cell.runs, 1);
  EXPECT_FALSE(g.is_dirty(ia));
  EXPECT_FALSE(g.is_dirty(ic));
}

TEST(TaskGraph, SecondEvaluateAllCleanIsNoOp) {
  TaskGraph g(system_allocator());
  Source a{{}, 3}, b{{}, 4};
  Add c;
  NodeId ia = g.add_source(source_fn, &a);
  NodeId ib = g.add_source(source_fn, &b);
  c.a = &a.cell;
  c.b = &b.cell;
  NodeId deps[] = {ia, ib};
  (void)g.add_node(add_fn, &c, deps, 2);

  ASSERT_TRUE(g.evaluate().ok());
  ASSERT_TRUE(g.evaluate().ok());  // nothing dirty -> nothing runs
  EXPECT_EQ(a.cell.runs, 1);
  EXPECT_EQ(b.cell.runs, 1);
  EXPECT_EQ(c.cell.runs, 1);
}

// THE incremental-recompute proof: change ONE input, re-evaluate, and ONLY the changed
// node + its downstream recompute; the untouched sibling is skipped (reuses its cache).
TEST(TaskGraph, IncrementalRecomputeOnlyDirtySubgraph) {
  TaskGraph g(system_allocator());
  Source a{{}, 3}, b{{}, 4};
  Add c;
  NodeId ia = g.add_source(source_fn, &a);
  NodeId ib = g.add_source(source_fn, &b);
  c.a = &a.cell;
  c.b = &b.cell;
  NodeId deps[] = {ia, ib};
  NodeId ic = g.add_node(add_fn, &c, deps, 2);

  ASSERT_TRUE(g.evaluate().ok());
  EXPECT_EQ(c.cell.value, 7);

  // Edit only source a; mark it dirty. b is untouched.
  a.seed = 10;
  g.mark_dirty(ia);
  EXPECT_TRUE(g.is_dirty(ia));
  EXPECT_TRUE(g.is_dirty(ic));   // downstream flooded dirty
  EXPECT_FALSE(g.is_dirty(ib));  // sibling stays clean

  ASSERT_TRUE(g.evaluate().ok());
  EXPECT_EQ(c.cell.value, 14);   // 10 + 4
  EXPECT_EQ(a.cell.runs, 2);     // a recomputed
  EXPECT_EQ(b.cell.runs, 1);     // b SKIPPED -- the incremental win (cache reused)
  EXPECT_EQ(c.cell.runs, 2);     // c recomputed (depends on a)
}

// Diamond: d = (a) -> b, c -> e. Dirtying `a` must flood BOTH b and c and the sink e,
// and each downstream node runs exactly once (not once per path).
TEST(TaskGraph, DiamondFloodsAllPathsOnce) {
  TaskGraph g(system_allocator());
  Source a{{}, 1};
  Add b, c, e;
  NodeId ia = g.add_source(source_fn, &a);
  // b and c both depend on a (reuse add_fn with a as both inputs -> value = 2*a).
  b.a = &a.cell; b.b = &a.cell;
  c.a = &a.cell; c.b = &a.cell;
  NodeId da[] = {ia};
  NodeId ib = g.add_node(add_fn, &b, da, 1);
  NodeId ic = g.add_node(add_fn, &c, da, 1);
  // e depends on b and c.
  e.a = &b.cell; e.b = &c.cell;
  NodeId dbc[] = {ib, ic};
  NodeId ie = g.add_node(add_fn, &e, dbc, 2);

  ASSERT_TRUE(g.evaluate().ok());
  EXPECT_EQ(e.cell.value, 4);  // (2*1) + (2*1)

  a.seed = 5;
  g.mark_dirty(ia);
  EXPECT_TRUE(g.is_dirty(ib));
  EXPECT_TRUE(g.is_dirty(ic));
  EXPECT_TRUE(g.is_dirty(ie));
  ASSERT_TRUE(g.evaluate().ok());
  EXPECT_EQ(e.cell.value, 20);  // (2*5) + (2*5)
  EXPECT_EQ(b.cell.runs, 2);
  EXPECT_EQ(c.cell.runs, 2);
  EXPECT_EQ(e.cell.runs, 2);    // sink ran once per evaluate, not once per incoming edge
}

// A node whose input didn't change is NOT re-run even if a DIFFERENT branch did.
TEST(TaskGraph, UnaffectedBranchNotRecomputed) {
  TaskGraph g(system_allocator());
  Source a{{}, 1}, b{{}, 2}, x{{}, 100};
  Add ab, unrelated;
  NodeId ia = g.add_source(source_fn, &a);
  NodeId ib = g.add_source(source_fn, &b);
  NodeId ix = g.add_source(source_fn, &x);
  ab.a = &a.cell; ab.b = &b.cell;
  NodeId dab[] = {ia, ib};
  (void)g.add_node(add_fn, &ab, dab, 2);
  // `unrelated` depends only on x.
  unrelated.a = &x.cell; unrelated.b = &x.cell;
  NodeId dx[] = {ix};
  (void)g.add_node(add_fn, &unrelated, dx, 1);

  ASSERT_TRUE(g.evaluate().ok());
  EXPECT_EQ(unrelated.cell.runs, 1);

  b.seed = 20;
  g.mark_dirty(ib);
  ASSERT_TRUE(g.evaluate().ok());
  EXPECT_EQ(ab.cell.value, 21);           // 1 + 20
  EXPECT_EQ(unrelated.cell.runs, 1);      // x-branch untouched -> not re-run
  EXPECT_EQ(x.cell.runs, 1);
}

// A failing ComputeFn aborts the sweep, surfaces its Status, and leaves the node dirty
// so a later evaluate retries it.
struct Failing {
  Cell cell;
  bool should_fail = true;
};
Status failing_fn(void* p) noexcept {
  auto* f = static_cast<Failing*>(p);
  if (f->should_fail) return Status::Internal("boom");
  f->cell.value = 42;
  ++f->cell.runs;
  return Status();
}

TEST(TaskGraph, FailingNodeAbortsAndStaysDirty) {
  TaskGraph g(system_allocator());
  Failing f;
  NodeId nf = g.add_source(failing_fn, &f);

  Status s = g.evaluate();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(s.IsInternal());
  EXPECT_TRUE(g.is_dirty(nf));  // stays dirty -> retried

  f.should_fail = false;
  ASSERT_TRUE(g.evaluate().ok());  // retry succeeds
  EXPECT_EQ(f.cell.value, 42);
  EXPECT_FALSE(g.is_dirty(nf));
}

// Downstream of a failed node does not run (sweep stopped at the failure).
TEST(TaskGraph, DownstreamOfFailureDoesNotRun) {
  TaskGraph g(system_allocator());
  Failing f;
  Add downstream;
  NodeId iff = g.add_source(failing_fn, &f);
  downstream.a = &f.cell; downstream.b = &f.cell;
  NodeId df[] = {iff};
  (void)g.add_node(add_fn, &downstream, df, 1);

  EXPECT_FALSE(g.evaluate().ok());
  EXPECT_EQ(downstream.cell.runs, 0);  // never reached
}

TEST(TaskGraph, MoveTransfersState) {
  TaskGraph g(system_allocator());
  Source a{{}, 9};
  NodeId ia = g.add_source(source_fn, &a);
  ASSERT_TRUE(g.evaluate().ok());
  EXPECT_EQ(a.cell.value, 9);

  TaskGraph moved = std::move(g);
  EXPECT_EQ(moved.num_nodes(), 1u);
  EXPECT_FALSE(moved.is_dirty(ia));  // clean state carried across the move
  a.seed = 11;
  moved.mark_dirty(ia);
  ASSERT_TRUE(moved.evaluate().ok());
  EXPECT_EQ(a.cell.value, 11);
  EXPECT_EQ(a.cell.runs, 2);
}

// A moved-FROM graph must be a VALID EMPTY graph (reusable), not a broken one -- the
// CSR seed (dep_off_[0]=0) must survive the move so a fresh add_node doesn't read OOB.
TEST(TaskGraph, MovedFromIsReusable) {
  TaskGraph g(system_allocator());
  Source a{{}, 1};
  (void)g.add_source(source_fn, &a);
  TaskGraph moved = std::move(g);

  // Reuse the moved-from `g` as if fresh: build a 2-node dependency and evaluate.
  Source b{{}, 2};
  Add c;
  NodeId ib = g.add_source(source_fn, &b);
  c.a = &b.cell; c.b = &b.cell;
  NodeId dbc[] = {ib};
  (void)g.add_node(add_fn, &c, dbc, 1);
  EXPECT_EQ(g.num_nodes(), 2u);
  ASSERT_TRUE(g.evaluate().ok());
  EXPECT_EQ(c.cell.value, 4);  // 2 + 2 -- proves dep_off_ CSR is intact after move
}

// Transitive propagation to arbitrary DEPTH: a 4-node linear chain a->b->c->d. Dirtying
// the head must reach the tail (a bug capping the flood at depth 2-3 would pass Diamond).
TEST(TaskGraph, DeepChainPropagatesToTail) {
  TaskGraph g(system_allocator());
  Source a{{}, 1};
  Add b, c, d;
  NodeId ia = g.add_source(source_fn, &a);
  b.a = &a.cell; b.b = &a.cell;           // b = 2a
  NodeId da[] = {ia};
  NodeId ib = g.add_node(add_fn, &b, da, 1);
  c.a = &b.cell; c.b = &b.cell;           // c = 2b = 4a
  NodeId db[] = {ib};
  NodeId ic = g.add_node(add_fn, &c, db, 1);
  d.a = &c.cell; d.b = &c.cell;           // d = 2c = 8a
  NodeId dc[] = {ic};
  NodeId id = g.add_node(add_fn, &d, dc, 1);

  ASSERT_TRUE(g.evaluate().ok());
  EXPECT_EQ(d.cell.value, 8);  // 8*1

  a.seed = 3;
  g.mark_dirty(ia);
  EXPECT_TRUE(g.is_dirty(ib));
  EXPECT_TRUE(g.is_dirty(ic));
  EXPECT_TRUE(g.is_dirty(id));  // reached the tail 3 hops down
  ASSERT_TRUE(g.evaluate().ok());
  EXPECT_EQ(d.cell.value, 24);  // 8*3
  EXPECT_EQ(d.cell.runs, 2);
}

// Marking an INTERIOR (non-source) node dirty re-runs it + downstream, but NOT its
// upstream (which didn't change). Also exercises add_node on an already-evaluated graph.
TEST(TaskGraph, InteriorDirtyAndLateAdd) {
  TaskGraph g(system_allocator());
  Source a{{}, 5};
  Add b;
  NodeId ia = g.add_source(source_fn, &a);
  b.a = &a.cell; b.b = &a.cell;
  NodeId da[] = {ia};
  NodeId ib = g.add_node(add_fn, &b, da, 1);
  ASSERT_TRUE(g.evaluate().ok());
  EXPECT_EQ(a.cell.runs, 1);
  EXPECT_EQ(b.cell.runs, 1);

  // Add a downstream node AFTER a full evaluate (only it is dirty).
  Add c;
  c.a = &b.cell; c.b = &b.cell;
  NodeId db[] = {ib};
  NodeId ic = g.add_node(add_fn, &c, db, 1);
  EXPECT_TRUE(g.is_dirty(ic));
  EXPECT_FALSE(g.is_dirty(ia));
  EXPECT_FALSE(g.is_dirty(ib));
  ASSERT_TRUE(g.evaluate().ok());
  EXPECT_EQ(c.cell.value, 20);   // (2*5) + (2*5)
  EXPECT_EQ(a.cell.runs, 1);     // untouched
  EXPECT_EQ(b.cell.runs, 1);     // untouched -- only the new node ran
  EXPECT_EQ(c.cell.runs, 1);

  // Now mark the INTERIOR node b dirty: b + c re-run, a stays put.
  g.mark_dirty(ib);
  EXPECT_FALSE(g.is_dirty(ia));
  EXPECT_TRUE(g.is_dirty(ib));
  EXPECT_TRUE(g.is_dirty(ic));
  ASSERT_TRUE(g.evaluate().ok());
  EXPECT_EQ(a.cell.runs, 1);     // upstream NOT re-run
  EXPECT_EQ(b.cell.runs, 2);
  EXPECT_EQ(c.cell.runs, 2);
}

// ---- contract death tests (caller bug -> fatal assert) ----------------------
void add_null_fn() {
  TaskGraph g(system_allocator());
  (void)g.add_source(nullptr, nullptr);  // fn null -> assert
}
void dep_not_smaller() {
  TaskGraph g(system_allocator());
  Source a{{}, 1};
  NodeId ia = g.add_source(source_fn, &a);
  // depend on a not-yet-created id (id 5) -> violates decision ② (dep < new id).
  NodeId bad[] = {static_cast<NodeId>(5)};
  Add c;
  (void)g.add_node(add_fn, &c, bad, 1);
  (void)ia;
}

TEST(TaskGraphDeathTest, AddNullFn) {
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(add_null_fn(), "HYLE fatal");
}
TEST(TaskGraphDeathTest, DepNotSmaller) {
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(dep_not_smaller(), "HYLE fatal");
}

}  // namespace
}  // namespace hyle::graph
