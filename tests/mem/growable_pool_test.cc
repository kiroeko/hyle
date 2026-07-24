#include "hyle/mem/growable_pool.h"

#include <gtest/gtest.h>

#include <atomic>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/mem/stats.h"

using hyle::mem::GrowablePool;
using hyle::mem::StatsAllocator;
using hyle::mem::system_allocator;

TEST(GrowablePool, CreateDestroyRoundTrip) {
  GrowablePool<int> pool(system_allocator(), 8);
  int* a = pool.create(42);
  ASSERT_NE(a, nullptr);
  EXPECT_EQ(*a, 42);
  EXPECT_EQ(pool.live(), 1u);
  pool.destroy(a);
  EXPECT_EQ(pool.live(), 0u);
}

// Growth across many blocks (tiny initial -> geometric doubling). All pointers
// stay valid (slots never move once handed out).
TEST(GrowablePool, GrowsAndKeepsPointersValid) {
  GrowablePool<int> pool(system_allocator(), 2);
  std::vector<int*> ptrs;
  for (int i = 0; i < 1000; ++i) ptrs.push_back(pool.create(i));
  EXPECT_EQ(pool.live(), 1000u);
  for (int i = 0; i < 1000; ++i) EXPECT_EQ(*ptrs[i], i);  // none moved/corrupted
  for (int* p : ptrs) pool.destroy(p);
  EXPECT_EQ(pool.live(), 0u);
}

// A freed slot is reused by the next create (free-list works).
TEST(GrowablePool, ReusesFreedSlots) {
  GrowablePool<long> pool(system_allocator(), 4);
  long* a = pool.create(1);
  pool.destroy(a);
  long* b = pool.create(2);
  EXPECT_EQ(a, b);  // same slot recycled
  pool.destroy(b);
}

// Non-trivial destructor runs once per destroy().
TEST(GrowablePool, DestructorRunsOnDestroy) {
  static std::atomic<int> live{0};
  struct Counted {
    Counted() { live.fetch_add(1); }
    ~Counted() { live.fetch_sub(1); }
  };
  GrowablePool<Counted> pool(system_allocator(), 4);
  Counted* objs[20];
  for (auto& o : objs) o = pool.create();
  EXPECT_EQ(live.load(), 20);
  for (auto* o : objs) pool.destroy(o);
  EXPECT_EQ(live.load(), 0);
}

TEST(GrowablePool, ConstructorThrowReturnsSlotToPool) {
  struct ThrowsOnce {
    explicit ThrowsOnce(bool fail) {
      if (fail) throw std::runtime_error("ctor");
    }
  };
  GrowablePool<ThrowsOnce> pool(system_allocator(), 1);
  EXPECT_THROW((void)pool.create(true), std::runtime_error);
  EXPECT_EQ(pool.live(), 0u);
  auto* value = pool.create(false);
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(pool.live(), 1u);
  pool.destroy(value);
}

// Move transfers ownership (and the allocator -- clay's move-assign bug). The
// moved-from pool is empty; the moved-to frees correctly via the right allocator.
TEST(GrowablePool, MoveTransfersOwnership) {
  GrowablePool<int> a(system_allocator(), 8);
  int* p = a.create(7);
  EXPECT_EQ(a.live(), 1u);

  GrowablePool<int> b(std::move(a));
  EXPECT_EQ(b.live(), 1u);
  EXPECT_EQ(*p, 7);  // object survived the move (slot didn't move)
  b.destroy(p);

  // Move-assignment: the destination's old blocks are released, then it adopts c.
  GrowablePool<int> c(system_allocator(), 8);
  int* q = c.create(9);
  b = std::move(c);
  EXPECT_EQ(b.live(), 1u);
  EXPECT_EQ(*q, 9);
  b.destroy(q);
}

// Interleaved create/destroy keeps live() and reuse consistent.
TEST(GrowablePool, InterleavedChurn) {
  GrowablePool<std::uint64_t> pool(system_allocator(), 16);
  std::unordered_set<std::uint64_t*> live;
  for (int round = 0; round < 5; ++round) {
    for (int i = 0; i < 500; ++i) live.insert(pool.create(i));
    EXPECT_EQ(pool.live(), live.size());
    // Free half.
    int k = 0;
    for (auto it = live.begin(); it != live.end() && k < 250;) {
      pool.destroy(*it);
      it = live.erase(it);
      ++k;
    }
  }
  for (auto* p : live) pool.destroy(p);
  EXPECT_EQ(pool.live(), 0u);
}

TEST(GrowablePool, ClearReclaimsAndReusable) {
  StatsAllocator sa = StatsAllocator::create(system_allocator()).value();
  auto backing = sa.get();
  // create N (forces geometric growth), destroy all, clear -> blocks handed back.
  GrowablePool<std::uint64_t> pool(backing, 16);
  std::vector<std::uint64_t*> ps;
  for (int i = 0; i < 500; ++i) ps.push_back(pool.create(i));
  const auto after_grow = sa.snapshot().live_bytes;
  EXPECT_GT(after_grow, 0u);              // blocks are held
  for (auto* p : ps) pool.destroy(p);
  EXPECT_EQ(pool.live(), 0u);
  pool.clear();                            // hand every block back to the allocator
  EXPECT_EQ(sa.snapshot().live_bytes, 0u); // pool holds nothing now
  // Still usable after clear (grows again from the initial size).
  auto* q = pool.create(42);
  ASSERT_NE(q, nullptr);
  EXPECT_EQ(*q, 42u);
  EXPECT_EQ(pool.live(), 1u);
  pool.destroy(q);
}

#if defined(HYLE_TESTING)
TEST(GrowablePoolDiagnosticsDeathTest, DoubleDestroyTraps) {
  EXPECT_DEATH(
      {
        GrowablePool<int> pool(system_allocator(), 1);
        int* p = pool.create(7);
        pool.destroy(p);
        pool.destroy(p);
      },
      ".*");
}

TEST(GrowablePoolDiagnosticsDeathTest, ForeignPointerTraps) {
  EXPECT_DEATH(
      {
        GrowablePool<int> pool(system_allocator(), 1);
        int foreign = 0;
        pool.destroy(&foreign);
      },
      ".*");
}

TEST(GrowablePoolDiagnosticsDeathTest, OutstandingObjectAtDestructionTraps) {
  EXPECT_DEATH(
      {
        GrowablePool<int> pool(system_allocator(), 1);
        (void)pool.create(7);
      },
      ".*");
}
#endif
