#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/mem/object_pool.h"
#include "hyle/mem/stack_allocator.h"
#include "hyle/platform/thread.h"

using hyle::mem::ObjectPool;
using hyle::mem::StackAllocator;
using hyle::mem::system_allocator;

// ---- ObjectPool: ctor/dtor balance (the bug #3 proof) -----------------------

namespace {
std::atomic<int> g_live{0};  // live instances; must return to 0
struct Counted {
  int v;
  explicit Counted(int x) : v(x) { g_live.fetch_add(1, std::memory_order_relaxed); }
  ~Counted() { g_live.fetch_sub(1, std::memory_order_relaxed); }
};
}  // namespace

TEST(ObjectPool, CreateRunsCtorDestroyRunsDtor) {
  g_live.store(0);
  ObjectPool<Counted> pool(system_allocator(), 8);
  Counted* a = pool.create(42);
  ASSERT_NE(a, nullptr);
  EXPECT_EQ(a->v, 42);
  EXPECT_EQ(g_live.load(), 1);
  pool.destroy(a);
  EXPECT_EQ(g_live.load(), 0);  // dtor ran (and slot freed via the arena)
}

TEST(ObjectPool, CtorDtorBalanceAcrossManyAndReuse) {
  g_live.store(0);
  ObjectPool<Counted> pool(system_allocator(), 4);
  for (int round = 0; round < 100; ++round) {
    std::vector<Counted*> v;
    for (int i = 0; i < 4; ++i) {
      Counted* c = pool.create(i);
      ASSERT_NE(c, nullptr);
      v.push_back(c);
    }
    EXPECT_EQ(g_live.load(), 4);
    EXPECT_EQ(pool.create(99), nullptr);  // exhausted at capacity
    for (Counted* c : v) pool.destroy(c);
    EXPECT_EQ(g_live.load(), 0);  // all dtors ran each round -> no leak, balanced
  }
}

TEST(ObjectPool, SlotReused) {
  ObjectPool<int> pool(system_allocator(), 2);
  int* a = pool.create(1);
  pool.destroy(a);
  int* b = pool.create(2);
  EXPECT_EQ(a, b);  // freed slot comes back
  pool.destroy(b);
}

TEST(ObjectPool, ConcurrentCreateDestroyBalances) {
  g_live.store(0);
  constexpr int kThreads = 8;
  constexpr int kIters = 50000;
  ObjectPool<Counted> pool(system_allocator(), 256);
  struct Args {
    ObjectPool<Counted>* pool;
  } a{&pool};
  std::vector<hyle::platform::Thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    auto r = hyle::platform::Thread::spawn(
        [](void* p) {
          auto* x = static_cast<Args*>(p);
          for (int i = 0; i < kIters; ++i) {
            Counted* c = x->pool->create(i);
            if (c == nullptr) {
              hyle::platform::this_thread::yield();
              continue;
            }
            x->pool->destroy(c);
          }
        },
        &a);
    ASSERT_TRUE(r.has_value());
    threads.push_back(std::move(*r));
  }
  for (auto& th : threads) th.join();
  EXPECT_EQ(g_live.load(), 0);  // every create matched by a destroy
}

// ---- StackAllocator ---------------------------------------------------------

TEST(StackAllocator, BumpOrderAndAlignment) {
  StackAllocator sa(system_allocator(), 4096);
  void* a = sa.allocate(16, 16);
  void* b = sa.allocate(16, 16);
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  EXPECT_LT(a, b);  // bumps upward
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(a) % 16, 0u);
  void* c = sa.allocate(1, 64);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(c) % 64, 0u);  // honors alignment
}

TEST(StackAllocator, MarkRewindReleasesLifo) {
  StackAllocator sa(system_allocator(), 1024);
  sa.allocate(100);
  auto m = sa.mark();
  void* x = sa.allocate(200);
  ASSERT_NE(x, nullptr);
  std::size_t used_before = sa.used();
  sa.rewind(m);
  EXPECT_LT(sa.used(), used_before);
  void* y = sa.allocate(200);  // same region reused after rewind
  EXPECT_EQ(x, y);
}

TEST(StackAllocator, ResetClears) {
  StackAllocator sa(system_allocator(), 256);
  sa.allocate(100);
  EXPECT_GT(sa.used(), 0u);
  sa.reset();
  EXPECT_EQ(sa.used(), 0u);
  EXPECT_EQ(sa.capacity(), 256u);
}

TEST(StackAllocator, OverflowReturnsNull) {
  StackAllocator sa(system_allocator(), 128);
  // alignment 1 isolates pure size overflow (no alignment padding eating space).
  EXPECT_NE(sa.allocate(100, 1), nullptr);
  EXPECT_EQ(sa.allocate(100, 1), nullptr);  // won't fit -> null, no heap fallback
  EXPECT_NE(sa.allocate(28, 1), nullptr);   // exact remainder still usable
}
