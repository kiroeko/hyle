#include <gtest/gtest.h>

#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/run_queue.h"
#include "hyle/sched/worker.h"

using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::RunQueue;
using hyle::sched::Worker;

// ---- RunQueue: pure pointer routing (never dereferences the Fiber*) ---------
// Distinct non-null fake handles; RunQueue only stores/returns them.
namespace {
int g_slots[16];
Fiber* fake(int i) { return reinterpret_cast<Fiber*>(&g_slots[i]); }
}  // namespace

TEST(RunQueue, FifoOrder) {
  RunQueue rq(system_allocator(), 8);
  rq.schedule_fifo(fake(0));
  rq.schedule_fifo(fake(1));
  rq.schedule_fifo(fake(2));
  EXPECT_EQ(rq.next(), fake(0));
  EXPECT_EQ(rq.next(), fake(1));
  EXPECT_EQ(rq.next(), fake(2));
  EXPECT_EQ(rq.next(), nullptr);
}

TEST(RunQueue, LifoSlotServedBeforeFifo) {
  RunQueue rq(system_allocator(), 8);
  rq.schedule_fifo(fake(0));
  rq.schedule_lifo(fake(9));     // run-next
  EXPECT_EQ(rq.next(), fake(9));  // LIFO slot first
  EXPECT_EQ(rq.next(), fake(0));  // then FIFO
}

TEST(RunQueue, LifoEvictBumpsPrevToFifo) {
  RunQueue rq(system_allocator(), 8);
  rq.schedule_lifo(fake(1));     // slot = 1
  rq.schedule_lifo(fake(2));     // slot = 2, 1 bumped to FIFO tail
  EXPECT_EQ(rq.next(), fake(2));  // newest in slot
  EXPECT_EQ(rq.next(), fake(1));  // evicted one from FIFO
}

TEST(RunQueue, LifoBudgetDemotesToFifo) {
  RunQueue rq(system_allocator(), 8);
  Fiber* x = fake(7);
  rq.schedule_fifo(x);  // a FIFO resident that must eventually get served
  // Re-fill the LIFO slot each round; after kMaxLifoPolls consecutive slot
  // services the budget forces a FIFO turn so x can't be starved.
  int served_from_fifo_at = -1;
  for (int i = 0; i < RunQueue::kMaxLifoPolls + 1; ++i) {
    rq.schedule_lifo(fake(i));
    Fiber* got = rq.next();
    if (got == x) {
      served_from_fifo_at = i;
      break;
    }
  }
  // The FIFO resident must be served exactly when the budget runs out.
  EXPECT_EQ(served_from_fifo_at, RunQueue::kMaxLifoPolls);
}

TEST(RunQueue, EmptyNextIsNull) {
  RunQueue rq(system_allocator(), 8);
  EXPECT_EQ(rq.next(), nullptr);
}

// Eviction must never drop a fiber: if the slot is occupied and the FIFO is full
// the new schedule_lifo is rejected and prior state stays intact.
TEST(RunQueue, LifoEvictRejectedWhenFifoFull) {
  RunQueue rq(system_allocator(), 2);  // tiny ring (power of two)
  EXPECT_TRUE(rq.schedule_fifo(fake(0)));
  EXPECT_TRUE(rq.schedule_fifo(fake(1)));  // FIFO now full
  EXPECT_TRUE(rq.schedule_lifo(fake(8)));  // slot = 8 (FIFO untouched)
  EXPECT_FALSE(rq.schedule_lifo(fake(9))); // can't evict 8 into full FIFO
  // Nothing lost: 8 still in slot, 0/1 still in FIFO; 9 stayed with the caller.
  EXPECT_EQ(rq.next(), fake(8));
  EXPECT_EQ(rq.next(), fake(0));
  EXPECT_EQ(rq.next(), fake(1));
  EXPECT_EQ(rq.next(), nullptr);
}

// ---- Worker: real fibers, prove round-robin drive ---------------------------
namespace {
struct FArg {
  int id;
  int yields;
  std::vector<int>* log;
};
void worker_fiber(void* p) {
  auto* a = static_cast<FArg*>(p);
  for (int i = 0; i < a->yields; ++i) {
    a->log->push_back(a->id);
    Fiber::yield();
  }
  a->log->push_back(a->id);  // final segment
}
}  // namespace

TEST(Worker, RoundRobinDrivesAllToCompletion) {
  std::vector<int> log;
  FArg a0{0, 2, &log}, a1{1, 2, &log}, a2{2, 2, &log};
  Fiber f0(system_allocator(), kDefaultStackSize, &worker_fiber, &a0);
  Fiber f1(system_allocator(), kDefaultStackSize, &worker_fiber, &a1);
  Fiber f2(system_allocator(), kDefaultStackSize, &worker_fiber, &a2);

  Worker w(system_allocator(), 64);
  w.schedule(&f0);
  w.schedule(&f1);
  w.schedule(&f2);
  std::size_t resumes = w.run();

  EXPECT_TRUE(f0.done());
  EXPECT_TRUE(f1.done());
  EXPECT_TRUE(f2.done());
  EXPECT_EQ(resumes, 9u);  // 3 fibers x (2 yields + 1 final) resumes
  // FIFO round-robin: 0,1,2 repeated -- NOT 0,0,0,1,1,1 (which would mean one
  // fiber ran to completion before the others got a turn).
  std::vector<int> expected{0, 1, 2, 0, 1, 2, 0, 1, 2};
  EXPECT_EQ(log, expected);
}

TEST(Worker, NoYieldFiberRunsOnce) {
  std::vector<int> log;
  FArg a{5, 0, &log};  // zero yields: runs straight to done
  Fiber f(system_allocator(), kDefaultStackSize, &worker_fiber, &a);
  Worker w(system_allocator(), 8);
  w.schedule(&f);
  EXPECT_EQ(w.run(), 1u);
  EXPECT_TRUE(f.done());
  EXPECT_EQ(log, (std::vector<int>{5}));
}
