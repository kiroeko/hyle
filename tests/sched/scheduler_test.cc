#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;

namespace {
// A fiber body that yields `yields` times, bumping a shared atomic each segment.
struct Body {
  int yields;
  std::atomic<int>* ran;       // total resume segments observed (sanity)
  std::atomic<int>* completed;  // fibers that reached the end
};
void body_entry(void* p) {
  auto* b = static_cast<Body*>(p);
  for (int i = 0; i < b->yields; ++i) {
    b->ran->fetch_add(1, std::memory_order_relaxed);
    Fiber::yield();
  }
  b->completed->fetch_add(1, std::memory_order_acq_rel);
}
}  // namespace

TEST(Scheduler, ManyFibersAllComplete) {
  constexpr int kFibers = 1000;
  constexpr int kYields = 3;
  std::atomic<int> ran{0};
  std::atomic<int> completed{0};

  std::vector<Body> bodies(kFibers, Body{kYields, &ran, &completed});
  std::vector<std::unique_ptr<Fiber>> fibers;
  fibers.reserve(kFibers);

  {
    Scheduler sched(8, system_allocator(), 1024);
    for (int i = 0; i < kFibers; ++i) {
      fibers.push_back(std::make_unique<Fiber>(system_allocator(),
                                               kDefaultStackSize, &body_entry,
                                               &bodies[i]));
      sched.spawn(fibers[i].get());
    }
    sched.shutdown();  // drains (waits all done) then joins workers
  }

  EXPECT_EQ(completed.load(), kFibers);
  EXPECT_EQ(ran.load(), kFibers * kYields);  // every segment ran exactly once
}

// Force stealing: a single "seed" fiber, running on one worker, spawns many
// children onto THAT worker's local queue. The other workers have nothing of
// their own and must steal to make progress -> they will, and everything still
// completes exactly once.
namespace {
struct SeedArgs {
  Scheduler* sched;
  int children;
  std::vector<std::unique_ptr<Fiber>>* pool;
  std::vector<Body>* bodies;
  std::atomic<int>* ran;
  std::atomic<int>* completed;
};
void seed_entry(void* p) {
  auto* a = static_cast<SeedArgs*>(p);
  for (int i = 0; i < a->children; ++i) {
    (*a->bodies)[i] = Body{2, a->ran, a->completed};
    // child fibers are pre-created in the pool; spawn from inside the seed
    // fiber so they land on this worker's LOCAL queue (forcing others to steal).
    a->sched->spawn((*a->pool)[i].get());
  }
  a->completed->fetch_add(1, std::memory_order_acq_rel);  // seed itself
}
}  // namespace

TEST(Scheduler, StealingDistributesLocalWork) {
  constexpr int kChildren = 500;
  std::atomic<int> ran{0};
  std::atomic<int> completed{0};

  std::vector<Body> bodies(kChildren);
  std::vector<std::unique_ptr<Fiber>> pool;
  pool.reserve(kChildren);

  {
    Scheduler sched(4, system_allocator(), 1024);
    for (int i = 0; i < kChildren; ++i) {
      pool.push_back(std::make_unique<Fiber>(
          system_allocator(), kDefaultStackSize, &body_entry, &bodies[i]));
    }
    SeedArgs sa{&sched, kChildren, &pool, &bodies, &ran, &completed};
    Fiber seed(system_allocator(), kDefaultStackSize, &seed_entry, &sa);
    sched.spawn(&seed);
    sched.shutdown();
  }

  EXPECT_EQ(completed.load(), kChildren + 1);  // children + seed
  EXPECT_EQ(ran.load(), kChildren * 2);
}

TEST(Scheduler, SingleWorkerStillWorks) {
  std::atomic<int> ran{0};
  std::atomic<int> completed{0};
  Body b{5, &ran, &completed};
  auto f = std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                   &body_entry, &b);
  {
    Scheduler sched(1, system_allocator(), 64);
    sched.spawn(f.get());
    sched.shutdown();
  }
  EXPECT_EQ(completed.load(), 1);
  EXPECT_EQ(ran.load(), 5);
}
