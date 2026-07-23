#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>  // std::abort in the guard-page death test
#include <cstring>
#include <thread>
#include <vector>

#include "hyle/mem/fiber_stack.h"
#include "hyle/platform/virtual_memory.h"

using hyle::mem::FiberStack;
using hyle::mem::FiberStackPool;
using hyle::mem::kFiberStackDefault;
using hyle::mem::kFiberStackLarge;
using hyle::mem::kFiberStackSmall;
using hyle::platform::page_align_up;
using hyle::platform::page_size;

TEST(FiberStack, AllocateDefaultHasGuardAndUsable) {
  FiberStack s = FiberStack::allocate(kFiberStackDefault).value();
  ASSERT_TRUE(s.valid());
  EXPECT_TRUE(s.has_guard());
  EXPECT_NE(s.base(), nullptr);
  EXPECT_NE(s.guard(), nullptr);
  EXPECT_EQ(s.size(), page_align_up(kFiberStackDefault));
  EXPECT_EQ(s.total_reserved(), page_size() + s.size());
  // usable starts one page above guard
  EXPECT_EQ(static_cast<char*>(s.base()) - static_cast<char*>(s.guard()),
            static_cast<std::ptrdiff_t>(page_size()));
  EXPECT_EQ(static_cast<char*>(s.top()) - static_cast<char*>(s.base()),
            static_cast<std::ptrdiff_t>(s.size()));
}

TEST(FiberStack, UsableIsWritable) {
  FiberStack s = FiberStack::allocate(kFiberStackSmall).value();
  ASSERT_TRUE(s.valid());
  auto* p = static_cast<unsigned char*>(s.base());
  std::memset(p, 0xA5, s.size());
  EXPECT_EQ(p[0], 0xA5);
  EXPECT_EQ(p[s.size() - 1], 0xA5);
  // top-1 is still inside usable
  auto* top_m1 = static_cast<unsigned char*>(s.top()) - 1;
  *top_m1 = 0x5A;
  EXPECT_EQ(*top_m1, 0x5A);
}

TEST(FiberStack, TierSizes) {
  EXPECT_LT(kFiberStackSmall, kFiberStackDefault);
  EXPECT_LT(kFiberStackDefault, kFiberStackLarge);

  FiberStack small = FiberStack::allocate(kFiberStackSmall).value();
  FiberStack def = FiberStack::allocate(kFiberStackDefault).value();
  FiberStack large = FiberStack::allocate(kFiberStackLarge).value();
  ASSERT_TRUE(small.valid());
  ASSERT_TRUE(def.valid());
  ASSERT_TRUE(large.valid());
  EXPECT_EQ(small.size(), page_align_up(kFiberStackSmall));
  EXPECT_EQ(def.size(), page_align_up(kFiberStackDefault));
  EXPECT_EQ(large.size(), page_align_up(kFiberStackLarge));
  EXPECT_LT(small.size(), def.size());
  EXPECT_LT(def.size(), large.size());
}

TEST(FiberStack, MoveTransfersOwnership) {
  FiberStack a = FiberStack::allocate(kFiberStackDefault).value();
  ASSERT_TRUE(a.valid());
  void* base = a.base();
  FiberStack b = std::move(a);
  EXPECT_FALSE(a.valid());
  EXPECT_TRUE(b.valid());
  EXPECT_EQ(b.base(), base);
  EXPECT_TRUE(b.reset().ok());
  EXPECT_FALSE(b.valid());
}

TEST(FiberStackPool, AcquireReleaseRecycles) {
  FiberStackPool pool(kFiberStackDefault, /*cache_cap=*/8);
  FiberStack s1 = pool.acquire().value();
  ASSERT_TRUE(s1.valid());
  void* base1 = s1.base();
  pool.release(std::move(s1));
  EXPECT_EQ(pool.cached(), 1u);

  FiberStack s2 = pool.acquire().value();
  ASSERT_TRUE(s2.valid());
  EXPECT_EQ(s2.base(), base1);  // recycled
  EXPECT_EQ(pool.cached(), 0u);
  pool.release(std::move(s2));
}

TEST(FiberStackPool, CacheCapDropsExtra) {
  FiberStackPool pool(kFiberStackSmall, /*cache_cap=*/2);
  FiberStack a = pool.acquire().value();
  FiberStack b = pool.acquire().value();
  FiberStack c = pool.acquire().value();
  ASSERT_TRUE(a.valid());
  ASSERT_TRUE(b.valid());
  ASSERT_TRUE(c.valid());
  pool.release(std::move(a));
  pool.release(std::move(b));
  pool.release(std::move(c));  // third dropped
  EXPECT_EQ(pool.cached(), 2u);
}

TEST(FiberStackPool, WrongSizeNotCached) {
  FiberStackPool pool(kFiberStackDefault, /*cache_cap=*/8);
  FiberStack wrong = FiberStack::allocate(kFiberStackSmall).value();
  ASSERT_TRUE(wrong.valid());
  pool.release(std::move(wrong));
  EXPECT_EQ(pool.cached(), 0u);
}

TEST(FiberStackPool, ConcurrentAcquireRelease) {
  FiberStackPool pool(kFiberStackSmall, /*cache_cap=*/32);
  constexpr int kThreads = 4;
  constexpr int kIters = 200;
  std::atomic<int> ok{0};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&] {
      for (int i = 0; i < kIters; ++i) {
        auto acquired = pool.acquire();
        if (!acquired) return;
        FiberStack s = *std::move(acquired);
        if (!s.valid() || !s.has_guard()) return;
        // touch usable so the page is real
        *static_cast<volatile unsigned char*>(s.base()) = 1;
        pool.release(std::move(s));
      }
      ok.fetch_add(1, std::memory_order_relaxed);
    });
  }
  for (auto& th : threads) th.join();
  EXPECT_EQ(ok.load(), kThreads);
  EXPECT_LE(pool.cached(), 32u);
}

// Prove the guard page actually TRAPS on access (not just "has_guard()==true").
// A fiber stack grows DOWN toward base()/guard; an overflow past the stack hits
// the permanent no-access guard page. Death test (forked child captures the fault).
TEST(FiberStackDeathTest, GuardPageWriteFaults) {
  EXPECT_DEATH(
      {
        auto allocated = FiberStack::allocate(kFiberStackSmall);
        if (!allocated) std::abort();
        FiberStack s = *std::move(allocated);
        if (!s.has_guard()) std::abort();
        // guard() is the low no-access page one page below base(); writing it is
        // exactly the stack-overflow-past-the-bottom the guard exists to catch.
        volatile auto* g = static_cast<volatile unsigned char*>(s.guard());
        g[0] = 0x1;  // must trap
      },
      ".*");
}
