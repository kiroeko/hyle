#include <gtest/gtest.h>

#include "hyle/mem/fiber_stack.h"
#include "hyle/sched/fiber.h"

using hyle::mem::FiberStack;
using hyle::mem::FiberStackPool;
using hyle::mem::kFiberStackDefault;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::kLargeStackSize;
using hyle::sched::kSmallStackSize;

namespace {

void mark_entry(void* arg) noexcept {
  *static_cast<int*>(arg) = 99;
  Fiber::yield();
  *static_cast<int*>(arg) = 100;
}

void deep_use_stack(void* arg) noexcept {
  // Touch a moderate amount of stack frame space without overflowing.
  volatile char buf[2048];
  for (int i = 0; i < 2048; ++i) buf[i] = static_cast<char>(i);
  *static_cast<int*>(arg) = static_cast<int>(buf[100]);
}

}  // namespace

TEST(FiberGuard, DefaultCtorHasGuard) {
  int v = 0;
  Fiber f(kDefaultStackSize, &mark_entry, &v);
  EXPECT_TRUE(f.has_stack_guard());
  EXPECT_GE(f.stack_size(), kDefaultStackSize);
  f.resume();
  EXPECT_EQ(v, 99);
  EXPECT_FALSE(f.done());
  f.resume();
  EXPECT_EQ(v, 100);
  EXPECT_TRUE(f.done());
}

TEST(FiberGuard, TierSizes) {
  int a = 0, b = 0, c = 0;
  Fiber fs(kSmallStackSize, &deep_use_stack, &a);
  Fiber fd(kDefaultStackSize, &deep_use_stack, &b);
  Fiber fl(kLargeStackSize, &deep_use_stack, &c);
  EXPECT_TRUE(fs.has_stack_guard());
  EXPECT_TRUE(fd.has_stack_guard());
  EXPECT_TRUE(fl.has_stack_guard());
  EXPECT_LT(fs.stack_size(), fd.stack_size());
  EXPECT_LT(fd.stack_size(), fl.stack_size());
  fs.resume();
  fd.resume();
  fl.resume();
  EXPECT_TRUE(fs.done());
  EXPECT_TRUE(fd.done());
  EXPECT_TRUE(fl.done());
}

TEST(FiberGuard, PoolRecycleAcrossFibers) {
  FiberStackPool pool(kFiberStackDefault, /*cache_cap=*/4);
  void* first_base = nullptr;
  {
    FiberStack s = pool.acquire().value();
    first_base = s.base();
    int v = 0;
    Fiber f(std::move(s), &mark_entry, &v, &pool);
    EXPECT_TRUE(f.has_stack_guard());
    f.resume();
    f.resume();
    EXPECT_TRUE(f.done());
  }
  EXPECT_EQ(pool.cached(), 1u);
  {
    FiberStack s = pool.acquire().value();
    EXPECT_EQ(s.base(), first_base);
    int v = 0;
    Fiber f(std::move(s), &mark_entry, &v, &pool);
    f.resume();
    f.resume();
  }
  EXPECT_EQ(pool.cached(), 1u);
}

TEST(FiberGuard, MoveFiberKeepsStack) {
  int v = 0;
  Fiber f1(kDefaultStackSize, &mark_entry, &v);
  Fiber f2 = std::move(f1);
  EXPECT_TRUE(f2.has_stack_guard());
  f2.resume();
  f2.resume();
  EXPECT_TRUE(f2.done());
  EXPECT_EQ(v, 100);
}
