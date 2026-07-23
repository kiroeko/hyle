#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "hyle/sched/fiber.h"

using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;

namespace {

// Shared probe: a sequence log (to assert interleave order) and a value carried
// in a FIBER-LOCAL variable across yields (to prove the stack is saved/restored).
struct Probe {
  std::vector<int> seq;
  int carried = 0;
};

// Yields from a NESTED frame -- the thing stackful fibers can do and stackless
// coroutines cannot. If the asm switch didn't preserve the full call stack, the
// resume after this yield would land in the wrong place / corrupt `local`.
void deep_call(Probe* p) {
  p->seq.push_back(3);
  Fiber::yield();
  p->seq.push_back(5);
}

void probe_entry(void* arg) {
  auto* p = static_cast<Probe*>(arg);
  int local = 100;          // lives on the fiber's own stack
  p->seq.push_back(1);
  Fiber::yield();           // (A) yield from the entry frame
  local += 10;              // mutate AFTER resume -> stack must have survived
  p->seq.push_back(2);
  deep_call(p);             // (B) yield from a nested frame, resume back here
  local += 1;
  p->seq.push_back(6);
  p->carried = local;       // 111 iff `local` survived both yields
}

void no_yield_entry(void* arg);

}  // namespace

TEST(Fiber, RecoverableCreateValidatesAndRuns) {
  int value = 0;
  auto invalid = Fiber::create(kDefaultStackSize, nullptr, &value);
  ASSERT_FALSE(invalid);
  EXPECT_TRUE(invalid.error().IsInvalidArgument());

  auto created = Fiber::create(kDefaultStackSize, &no_yield_entry, &value);
  ASSERT_TRUE(created);
  Fiber fiber = *std::move(created);
  fiber.resume();
  EXPECT_TRUE(fiber.done());
  EXPECT_EQ(value, 7);
}

TEST(Fiber, StackPreservedAcrossNestedYields) {
  Probe p;
  Fiber f(kDefaultStackSize, &probe_entry, &p);

  p.seq.push_back(0);
  EXPECT_FALSE(f.done());

  f.resume();                  // 1, then yields at (A)
  EXPECT_FALSE(f.done());
  p.seq.push_back(10);

  f.resume();                  // 2, deep_call pushes 3, yields at (B)
  EXPECT_FALSE(f.done());
  p.seq.push_back(20);

  f.resume();                  // back into deep_call: 5, then 6, returns
  EXPECT_TRUE(f.done());

  EXPECT_EQ(p.carried, 111);   // 100 +10 +1 -- local survived TWO yields
  std::vector<int> expected{0, 1, 10, 2, 3, 20, 5, 6};
  EXPECT_EQ(p.seq, expected);  // exact interleave of main <-> fiber
}

namespace {
void no_yield_entry(void* arg) { *static_cast<int*>(arg) = 7; }
}  // namespace

TEST(Fiber, RunsToCompletionWithoutYield) {
  int v = 0;
  Fiber f(kDefaultStackSize, &no_yield_entry, &v);
  EXPECT_FALSE(f.done());
  f.resume();
  EXPECT_TRUE(f.done());
  EXPECT_EQ(v, 7);
}

namespace {
// Each fiber increments its OWN local between two yields; values must not bleed
// across fibers -> proves each has an independent stack.
struct Indep {
  int start;
  int out;
};
void indep_entry(void* arg) {
  auto* x = static_cast<Indep*>(arg);
  int local = x->start;
  Fiber::yield();
  ++local;
  Fiber::yield();
  x->out = local;
}
}  // namespace

TEST(Fiber, TwoFibersHaveIndependentStacks) {
  Indep a{10, -1}, b{20, -1};
  Fiber fa(kDefaultStackSize, &indep_entry, &a);
  Fiber fb(kDefaultStackSize, &indep_entry, &b);
  // Interleave the two fibers' resumes; their locals must stay separate.
  fa.resume(); fb.resume();   // both at first yield
  fa.resume(); fb.resume();   // both incremented, at second yield
  fa.resume(); fb.resume();   // both finish
  EXPECT_TRUE(fa.done());
  EXPECT_TRUE(fb.done());
  EXPECT_EQ(a.out, 11);
  EXPECT_EQ(b.out, 21);
}

// yield() with no running fiber is a caller bug -> fail-fast abort.
TEST(FiberDeath, YieldOutsideFiberAborts) {
  EXPECT_DEATH({ Fiber::yield(); }, "");
}
