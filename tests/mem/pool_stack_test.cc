#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/mem/fixed_pool.h"
#include "hyle/mem/growable_pool.h"
#include "hyle/mem/object_pool.h"
#include "hyle/mem/region.h"
#include "hyle/mem/shared_pool.h"
#include "hyle/mem/stack_allocator.h"
#include "hyle/platform/thread.h"

using hyle::mem::Allocator;
using hyle::mem::FixedPool;
using hyle::mem::FixedRegion;
using hyle::mem::GrowablePool;
using hyle::mem::ObjectPool;
using hyle::mem::ObjectPoolCache;
using hyle::mem::PreservingObjectPool;
using hyle::mem::Region;
using hyle::mem::RegionOptions;
using hyle::mem::StackAllocator;
using hyle::mem::system_allocator;

// ---- ObjectPool: ctor/dtor balance (the bug #3 proof) -----------------------

namespace {
std::atomic<int> g_live{0}; // live instances; must return to 0
struct Counted {
  int v;
  explicit Counted(int x) : v(x) {
    g_live.fetch_add(1, std::memory_order_relaxed);
  }
  ~Counted() { g_live.fetch_sub(1, std::memory_order_relaxed); }
};
bool g_throw_next = false;
struct ThrowsOnce {
  ThrowsOnce() {
    if (g_throw_next) {
      g_throw_next = false;
      throw 7;
    }
  }
};
struct PreserveThrowsOnThird {
  inline static int attempts = 0;
  inline static int live = 0;
  PreserveThrowsOnThird() {
    if (attempts++ == 2)
      throw 9;
    ++live;
  }
  ~PreserveThrowsOnThird() { --live; }
};
void *oom_alloc(void *, std::size_t, std::size_t) noexcept { return nullptr; }
void oom_free(void *, void *, std::size_t, std::size_t) noexcept {}
void *oom_realloc(void *, void *, std::size_t, std::size_t,
                  std::size_t) noexcept {
  return nullptr;
}

struct FailAllocatorState {
  std::size_t calls = 0;
  std::size_t fail_at = std::numeric_limits<std::size_t>::max();
  std::size_t outstanding = 0;
  std::size_t last_size = 0;
};

void *fail_alloc(void *opaque, std::size_t size,
                 std::size_t alignment) noexcept {
  auto &state = *static_cast<FailAllocatorState *>(opaque);
  const std::size_t call = state.calls++;
  state.last_size = size;
  if (call == state.fail_at)
    return nullptr;
  void *p = system_allocator().allocate(size, alignment);
  if (p != nullptr)
    ++state.outstanding;
  return p;
}

void fail_free(void *opaque, void *ptr, std::size_t size,
               std::size_t alignment) noexcept {
  if (ptr == nullptr)
    return;
  auto &state = *static_cast<FailAllocatorState *>(opaque);
  EXPECT_GT(state.outstanding, 0u);
  --state.outstanding;
  system_allocator().deallocate(ptr, size, alignment);
}

void *fail_realloc(void *, void *, std::size_t, std::size_t,
                   std::size_t) noexcept {
  return nullptr; // Region never reallocates backing blocks.
}
} // namespace

TEST(ObjectPool, ConstructionOomIsObservable) {
  hyle::mem::Allocator oom(&oom_alloc, &oom_free, &oom_realloc, nullptr);
  ObjectPool<int> hard(oom, 8);
  EXPECT_FALSE(hard.valid());
  EXPECT_EQ(hard.create(1), nullptr);
  EXPECT_FALSE(hard.try_loan(1));
}

TEST(SharedPool, EveryConstructorAllocationFailureIsLeakFree) {
  for (std::size_t fail_at = 0; fail_at != 3; ++fail_at) {
    FailAllocatorState state;
    state.fail_at = fail_at;
    Allocator failing(&fail_alloc, &fail_free, &fail_realloc, &state);
    {
      hyle::mem::SharedPool pool(failing, 64, 16, 64);
      EXPECT_FALSE(pool.valid()) << "fail_at=" << fail_at;
      EXPECT_EQ(pool.allocate(), nullptr);
    }
    EXPECT_EQ(state.outstanding, 0u) << "fail_at=" << fail_at;
  }
}

TEST(StackAllocator, ConstructorAllocationFailureIsLeakFree) {
  FailAllocatorState state;
  state.fail_at = 0;
  Allocator failing(&fail_alloc, &fail_free, &fail_realloc, &state);
  {
    StackAllocator stack(failing, 4096);
    EXPECT_FALSE(stack.valid());
    EXPECT_EQ(stack.allocate(32), nullptr);
  }
  EXPECT_EQ(state.outstanding, 0u);
}

TEST(GrowablePool, EveryGrowthAllocationFailureIsRecoverableAndLeakFree) {
  for (std::size_t fail_at = 0; fail_at != 5; ++fail_at) {
    FailAllocatorState state;
    state.fail_at = fail_at;
    Allocator failing(&fail_alloc, &fail_free, &fail_realloc, &state);
    {
      GrowablePool<int> pool(failing, 1);
      std::vector<int *> live;
      bool observed_failure = false;
      for (int value = 0; value != 64; ++value) {
        int *p = pool.create(value);
        if (p == nullptr) {
          EXPECT_FALSE(observed_failure) << "fail_at=" << fail_at;
          observed_failure = true;
          continue;
        }
        live.push_back(p);
      }
      EXPECT_TRUE(observed_failure) << "fail_at=" << fail_at;
      const std::size_t failed_value = (std::size_t{1} << fail_at) - 1;
      for (std::size_t i = 0; i < live.size(); ++i)
        EXPECT_EQ(*live[i],
                  static_cast<int>(i + (i >= failed_value ? 1 : 0)));
      for (int *p : live)
        pool.destroy(p);
    }
    EXPECT_EQ(state.outstanding, 0u) << "fail_at=" << fail_at;
  }
}

TEST(PreservingObjectPool, EveryBackingAllocationFailureIsLeakFree) {
  for (std::size_t fail_at = 0; fail_at != 4; ++fail_at) {
    FailAllocatorState state;
    state.fail_at = fail_at;
    Allocator failing(&fail_alloc, &fail_free, &fail_realloc, &state);
    {
      PreservingObjectPool<int> pool(failing, 4, 7);
      EXPECT_FALSE(pool.valid()) << "fail_at=" << fail_at;
      EXPECT_EQ(pool.acquire(), nullptr);
    }
    EXPECT_EQ(state.outstanding, 0u) << "fail_at=" << fail_at;
  }
}

TEST(ObjectPool, CreateRunsCtorDestroyRunsDtor) {
  g_live.store(0);
  ObjectPool<Counted> pool(system_allocator(), 8);
  Counted *a = pool.create(42);
  ASSERT_NE(a, nullptr);
  EXPECT_EQ(a->v, 42);
  EXPECT_EQ(g_live.load(), 1);
  pool.destroy(a);
  EXPECT_EQ(g_live.load(), 0); // dtor ran and the slot returned to the pool
}

TEST(ObjectPool, CtorDtorBalanceAcrossManyAndReuse) {
  g_live.store(0);
  ObjectPool<Counted> pool(system_allocator(), 4);
  for (int round = 0; round < 100; ++round) {
    std::vector<Counted *> v;
    for (int i = 0; i < 4; ++i) {
      Counted *c = pool.create(i);
      ASSERT_NE(c, nullptr);
      v.push_back(c);
    }
    EXPECT_EQ(g_live.load(), 4);
    EXPECT_EQ(pool.create(99), nullptr); // exhausted at capacity
    for (Counted *c : v)
      pool.destroy(c);
    EXPECT_EQ(g_live.load(),
              0); // all dtors ran each round -> no leak, balanced
  }
}

TEST(ObjectPool, SlotReused) {
  ObjectPool<int> pool(system_allocator(), 2);
  int *a = pool.create(1);
  pool.destroy(a);
  int *b = pool.create(2);
  EXPECT_EQ(a, b); // freed slot comes back
  pool.destroy(b);
}

TEST(ObjectPool, ThrowingConstructorReturnsSlot) {
  ObjectPool<ThrowsOnce> pool(system_allocator(), 2);
  g_throw_next = true;
  EXPECT_THROW((void)pool.create(), int);
  ThrowsOnce *a = pool.create();
  ThrowsOnce *b = pool.create();
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(pool.create(), nullptr); // failed construction did not leak a slot
  pool.destroy(a);
  pool.destroy(b);
}

TEST(ObjectPoolCache, ExplicitTypedCachePublishes) {
  ObjectPool<int> pool(system_allocator(), 4);
  std::array<std::uint32_t, 4> storage{};
  ObjectPoolCache<int> cache(pool, storage, 2);
  int *p = cache.create(7);
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(*p, 7);
  cache.destroy(p);
  EXPECT_GT(cache.cached(), 0u);
  EXPECT_LE(cache.cached(), 2u);
  cache.publish();
  EXPECT_EQ(cache.cached(), 0u);
}

TEST(ObjectPool, ConcurrentCreateDestroyBalances) {
  g_live.store(0);
  constexpr int kThreads = 8;
  constexpr int kIters = 50000;
  ObjectPool<Counted> pool(system_allocator(), 256);
  struct Args {
    ObjectPool<Counted> *pool;
  } a{&pool};
  std::vector<hyle::platform::Thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    auto r = hyle::platform::Thread::spawn(
        [](void *p) {
          auto *x = static_cast<Args *>(p);
          for (int i = 0; i < kIters; ++i) {
            Counted *c = x->pool->create(i);
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
  for (auto &th : threads)
    th.join();
  EXPECT_EQ(g_live.load(), 0); // every create matched by a destroy
}

TEST(ObjectPool, ScopedLoanReturnsOnEarlyExit) {
  ObjectPool<Counted> pool(system_allocator(), 1);
  g_live.store(0);
  {
    auto loan = pool.try_loan(9);
    ASSERT_TRUE(loan);
    EXPECT_EQ(loan->v, 9);
    EXPECT_EQ(g_live.load(), 1);
  }
  EXPECT_EQ(g_live.load(), 0);
  auto again = pool.try_loan(10);
  EXPECT_TRUE(again);
}

#if defined(HYLE_TESTING)
TEST(ObjectPoolDiagnosticsDeathTest, DoubleDestroyTrapsBeforeSecondDestructor) {
  EXPECT_DEATH(
      {
        ObjectPool<Counted> pool(system_allocator(), 1);
        Counted* p = pool.create(1);
        pool.destroy(p);
        pool.destroy(p);
      },
      ".*");
}

TEST(ObjectPoolDiagnosticsDeathTest, ForeignPointerTrapsBeforeDestructor) {
  EXPECT_DEATH(
      {
        ObjectPool<Counted> pool(system_allocator(), 1);
        Counted foreign(1);
        pool.destroy(&foreign);
      },
      ".*");
}

TEST(ObjectPoolDiagnosticsDeathTest, OutstandingLoanAtPoolDestructionTraps) {
  EXPECT_DEATH(
      {
        auto* pool = new ObjectPool<int>(system_allocator(), 1);
        auto loan = pool->try_loan(7);
        ASSERT_TRUE(loan);
        delete pool;
      },
      ".*");
}
#endif

TEST(PreservingObjectPool, ConstructsOnceAndPreservesStateAcrossLoans) {
  g_live.store(0);
  {
    PreservingObjectPool<Counted> pool(system_allocator(), 2, 5);
    ASSERT_TRUE(pool.valid());
    EXPECT_EQ(g_live.load(), 2);
    {
      auto loan = pool.try_loan();
      ASSERT_TRUE(loan);
      EXPECT_EQ(loan->v, 5);
      loan->v = 77;
      EXPECT_EQ(pool.outstanding(), 1u);
    }
    EXPECT_EQ(pool.outstanding(), 0u);
    auto a = pool.try_loan();
    auto b = pool.try_loan();
    ASSERT_TRUE(a);
    ASSERT_TRUE(b);
    EXPECT_TRUE(a->v == 77 || b->v == 77);
    EXPECT_FALSE(pool.try_loan());
  }
  EXPECT_EQ(g_live.load(), 0);
}

TEST(PreservingObjectPool, ConstructorFailureDestroysAndReturnsEveryPriorSlot) {
  PreserveThrowsOnThird::attempts = 0;
  PreserveThrowsOnThird::live = 0;
  EXPECT_THROW(
      (PreservingObjectPool<PreserveThrowsOnThird>(system_allocator(), 4)),
      int);
  EXPECT_EQ(PreserveThrowsOnThird::attempts, 3);
  EXPECT_EQ(PreserveThrowsOnThird::live, 0);
}

#if defined(HYLE_TESTING)
TEST(PreservingObjectPoolDiagnosticsDeathTest, DuplicateReturnTraps) {
  EXPECT_DEATH(
      {
        PreservingObjectPool<int> pool(system_allocator(), 2, 7);
        int* p = pool.acquire();
        pool.return_object(p);
        pool.return_object(p);
      },
      ".*");
}
#endif

// ---- StackAllocator ---------------------------------------------------------

TEST(StackAllocator, BumpOrderAndAlignment) {
  StackAllocator sa(system_allocator(), 4096);
  void *a = sa.allocate(16, 16);
  void *b = sa.allocate(16, 16);
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  EXPECT_LT(a, b); // bumps upward
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(a) % 16, 0u);
  void *c = sa.allocate(1, 64);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(c) % 64, 0u); // honors alignment
}

TEST(StackAllocator, MarkRewindReleasesLifo) {
  StackAllocator sa(system_allocator(), 1024);
  sa.allocate(100);
  auto m = sa.mark();
  void *x = sa.allocate(200);
  ASSERT_NE(x, nullptr);
  std::size_t used_before = sa.used();
  sa.rewind(m);
  EXPECT_LT(sa.used(), used_before);
  void *y = sa.allocate(200); // same region reused after rewind
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
  // alignment 1 isolates pure size overflow (no alignment padding eating
  // space).
  EXPECT_NE(sa.allocate(100, 1), nullptr);
  EXPECT_EQ(sa.allocate(100, 1),
            nullptr);                     // won't fit -> null, no heap fallback
  EXPECT_NE(sa.allocate(28, 1), nullptr); // exact remainder still usable
}

TEST(StackAllocator, UnrepresentableAlignmentReturnsNull) {
  StackAllocator sa(system_allocator(), 128);
  constexpr std::size_t kHugeAlignment = std::size_t{1}
                                         << (sizeof(std::size_t) * 8 - 1);
  EXPECT_EQ(sa.allocate(1, kHugeAlignment), nullptr);
  EXPECT_EQ(sa.used(), 0u);
}

TEST(StackAllocator, RemainingTracksCursor) {
  StackAllocator sa(system_allocator(), 256);
  EXPECT_EQ(sa.remaining(), 256u);
  (void)sa.allocate(64, 1);
  EXPECT_EQ(sa.remaining(), 192u);
  sa.reset();
  EXPECT_EQ(sa.remaining(), 256u);
}

TEST(StackAllocator, AsAllocatorInjectsAndReallocCopies) {
  StackAllocator sa(system_allocator(), 4096);
  hyle::mem::Allocator a = sa.as_allocator();
  ASSERT_TRUE(a.valid());
  auto *p =
      static_cast<unsigned char *>(a.allocate(32, 8)); // bump from the slab
  ASSERT_NE(p, nullptr);
  for (int i = 0; i < 32; ++i)
    p[i] = static_cast<unsigned char>(i);
  // realloc bumps a fresh slice + copies old bytes (a stack can't grow in
  // place)
  auto *q = static_cast<unsigned char *>(a.reallocate(p, 32, 64, 8));
  ASSERT_NE(q, nullptr);
  for (int i = 0; i < 32; ++i)
    EXPECT_EQ(q[i], static_cast<unsigned char>(i));
  a.deallocate(q, 64, 8); // no-op (bulk free via reset); must not crash
}

// alignment 0 slipped the old `(a & (a-1)) == 0` guard (0 & SIZE_MAX == 0
// passed) and yielded a wild pointer; #1 switched to is_pow2(), which fail-fast
// traps it. Locks in #1 for StackAllocator::allocate.
TEST(StackAllocatorDeathTest, ZeroAlignmentTraps) {
  EXPECT_DEATH(
      {
        StackAllocator sa(system_allocator(), 4096);
        (void)sa.allocate(16, 0);
      },
      ".*");
}

// ---- Region family ---------------------------------------------------------

TEST(FixedRegion, ExternalStorageHasNoFallbackAndResets) {
  alignas(64) std::byte storage[256];
  FixedRegion region(storage, sizeof(storage));
  ASSERT_TRUE(region.valid());
  void *a = region.allocate(33, 32);
  ASSERT_NE(a, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(a) % 32, 0u);
  EXPECT_EQ(region.allocate(1024), nullptr);
  const std::size_t used = region.used();
  EXPECT_GT(used, 0u);
  region.clear();
  EXPECT_EQ(region.used(), 0u);
  EXPECT_EQ(region.allocate(33, 32), a);
}

TEST(FixedRegion, RejectsUnrepresentableExternalAddressRange) {
  std::byte storage{};
  FixedRegion region(&storage, std::numeric_limits<std::size_t>::max());
  EXPECT_FALSE(region.valid());
}

TEST(Region, GrowsHonorsLimitAndBulkReuses) {
  RegionOptions options;
  options.initial_block_size = 128;
  options.max_block_size = 256;
  options.large_allocation_threshold = 192;
  options.size_limit = 2048;
  Region region(system_allocator(), options);
  void *a = region.allocate(80, 64);
  void *b = region.allocate(100, 32);
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(a) % 64, 0u);
  EXPECT_EQ(region.bytes_used(), 180u);
  const std::size_t retained = region.reserved();
  EXPECT_GT(retained, 0u);
  region.clear();
  EXPECT_EQ(region.bytes_used(), 0u);
  EXPECT_LE(region.reserved(), retained);
  EXPECT_NE(region.allocate(80, 64), nullptr);
  region.release();
  EXPECT_EQ(region.reserved(), 0u);
}

TEST(Region, OrdinaryBlockSizeTargetsTotalBackingBytes) {
  FailAllocatorState state;
  {
    Allocator counting(&fail_alloc, &fail_free, &fail_realloc, &state);
    RegionOptions options;
    options.initial_block_size = 4096;
    options.max_block_size = 4096;
    options.large_allocation_threshold = 8192;
    Region region(counting, options);

    void *p = region.allocate(64, 512);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % 512, 0u);
    EXPECT_EQ(state.last_size, 4096u);
    EXPECT_EQ(region.reserved(), 4096u);
  }
  EXPECT_EQ(state.outstanding, 0u);
}

TEST(Region, ClearReusesMultipleRetainedBlocksWithoutBackingAllocation) {
  FailAllocatorState state;
  {
    hyle::mem::Allocator counting(&fail_alloc, &fail_free, &fail_realloc,
                                  &state);
    RegionOptions options;
    options.initial_block_size = 128;
    options.max_block_size = 256;
    options.large_allocation_threshold = 4096;
    Region region(counting, options);

    constexpr int kAllocations = 12;
    for (int i = 0; i < kAllocations; ++i)
      ASSERT_NE(region.allocate(80, 16), nullptr);
    const std::size_t warmed_calls = state.calls;
    ASSERT_GT(warmed_calls, 1u);

    region.clear();
    for (int i = 0; i < kAllocations; ++i)
      ASSERT_NE(region.allocate(80, 16), nullptr);
    EXPECT_EQ(state.calls, warmed_calls);
  }
  EXPECT_EQ(state.outstanding, 0u);
}

TEST(Region, MoveTransfersCachedCursorWithoutOverlapOrBackingGrowth) {
  FailAllocatorState state;
  {
    hyle::mem::Allocator counting(&fail_alloc, &fail_free, &fail_realloc,
                                  &state);
    RegionOptions options;
    options.initial_block_size = 512;
    options.max_block_size = 512;
    options.large_allocation_threshold = 4096;
    Region source(counting, options);

    auto *first = static_cast<std::byte *>(source.allocate(96, 16));
    ASSERT_NE(first, nullptr);
    std::fill_n(first, 96, std::byte{0x5a});
    ASSERT_EQ(state.calls, 1u);

    Region moved(std::move(source));
    auto *second = static_cast<std::byte *>(moved.allocate(96, 16));
    ASSERT_NE(second, nullptr);
    EXPECT_NE(second, first);
    EXPECT_EQ(state.calls, 1u);
    for (std::size_t i = 0; i < 96; ++i)
      EXPECT_EQ(first[i], std::byte{0x5a});
  }
  EXPECT_EQ(state.outstanding, 0u);
}

TEST(Region, HardReservedByteLimitReturnsNull) {
  RegionOptions options;
  options.initial_block_size = 128;
  options.max_block_size = 128;
  options.large_allocation_threshold = 64;
  options.size_limit = 64; // smaller than payload + block header
  Region region(system_allocator(), options);
  EXPECT_EQ(region.allocate(64), nullptr);
  EXPECT_EQ(region.reserved(), 0u);
}

TEST(Region, EveryBackingAllocationFailureIsRecoverableAndLeakFree) {
  for (std::size_t fail_at = 0; fail_at != 5; ++fail_at) {
    FailAllocatorState state;
    state.fail_at = fail_at;
    hyle::mem::Allocator failing(&fail_alloc, &fail_free, &fail_realloc,
                                 &state);
    {
      RegionOptions options;
      options.initial_block_size = 128;
      options.max_block_size = 128;
      options.large_allocation_threshold = 96;
      Region region(failing, options);
      for (int i = 0; i < 5; ++i) {
        void *p = region.allocate(96, 64); // one dedicated block per request
        if (static_cast<std::size_t>(i) == fail_at)
          EXPECT_EQ(p, nullptr);
        else
          EXPECT_NE(p, nullptr);
      }
    }
    EXPECT_EQ(state.outstanding, 0u) << "fail_at=" << fail_at;
  }
}

TEST(PooledRegion, ExhaustionLeavesSecondRegionInvalid) {
  hyle::mem::SharedPool pool(system_allocator(), 256, 1, 64);
  hyle::mem::PooledRegion region(&pool);
  ASSERT_TRUE(region.valid());
  EXPECT_NE(region.allocate(64, 64), nullptr);
  hyle::mem::PooledRegion exhausted(&pool);
  EXPECT_FALSE(exhausted.valid());
}

TEST(PooledRegion, ClearUsesRegionFamilySpellingAndReusesBlock) {
  hyle::mem::SharedPool pool(system_allocator(), 256, 1, 64);
  hyle::mem::PooledRegion region(&pool);
  void* first = region.allocate(64, 16);
  ASSERT_NE(first, nullptr);
  region.clear();
  EXPECT_EQ(region.used(), 0u);
  EXPECT_EQ(region.allocate(64, 16), first);
}

// ---- external variable-size pool ------------------------------------------

TEST(FixedPool, AllocatesAlignedReusesAndCoalescesExternalStorage) {
  alignas(64) std::byte storage[2048];
  FixedPool pool(storage, sizeof(storage));
  ASSERT_TRUE(pool.valid());
  void *a = pool.allocate(100, 64);
  void *b = pool.allocate(200, 32);
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(a) % 64, 0u);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(b) % 32, 0u);
  EXPECT_EQ(pool.used(), 300u);
  pool.deallocate(a);
  pool.deallocate(b);
  EXPECT_EQ(pool.used(), 0u);
  EXPECT_NE(pool.allocate(1200, 64), nullptr); // adjacent free blocks coalesced
}

TEST(FixedPool, ReallocatePreservesBytesAndAllocatorAdapterWorks) {
  alignas(64) std::byte storage[2048];
  FixedPool pool(storage, sizeof(storage));
  auto alloc = pool.as_allocator();
  auto *p = static_cast<unsigned char *>(alloc.allocate(64, 16));
  ASSERT_NE(p, nullptr);
  for (int i = 0; i < 64; ++i)
    p[i] = static_cast<unsigned char>(i);
  auto *q = static_cast<unsigned char *>(alloc.reallocate(p, 64, 256, 32));
  ASSERT_NE(q, nullptr);
  for (int i = 0; i < 64; ++i)
    EXPECT_EQ(q[i], static_cast<unsigned char>(i));
  alloc.deallocate(q, 256, 32);
  EXPECT_EQ(pool.used(), 0u);
}

TEST(FixedPool, ReallocateGrowsIntoAdjacentFreeBlockInPlace) {
  alignas(64) std::byte storage[4096];
  FixedPool pool(storage, sizeof(storage));
  auto *first = static_cast<unsigned char *>(pool.allocate(256, 16));
  void *adjacent = pool.allocate(512, 16);
  void *barrier = pool.allocate(2400, 16);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(adjacent, nullptr);
  ASSERT_NE(barrier, nullptr);
  std::memset(first, 0xA5, 256);

  pool.deallocate(adjacent);
  auto *grown = static_cast<unsigned char *>(
      pool.reallocate(first, 256, 600, 16));
  ASSERT_EQ(grown, first);
  for (std::size_t i = 0; i < 256; ++i)
    EXPECT_EQ(grown[i], 0xA5);
  EXPECT_EQ(pool.used(), 3000u);

  // The unused part of the merged neighbor remains independently usable.
  void *remainder = pool.allocate(64, 16);
  ASSERT_NE(remainder, nullptr);
  pool.deallocate(remainder);
  pool.deallocate(grown);
  pool.deallocate(barrier);
  EXPECT_EQ(pool.used(), 0u);
}

TEST(FixedPool, ReallocateShrinkRetainsSpanForConstantTimeRegrow) {
  alignas(64) std::byte storage[2048];
  FixedPool pool(storage, sizeof(storage));
  void *p = pool.allocate(1400, 16);
  ASSERT_NE(p, nullptr);
  ASSERT_EQ(pool.allocate(800, 16), nullptr);

  ASSERT_EQ(pool.reallocate(p, 1400, 128, 16), p);
  EXPECT_EQ(pool.allocate(800, 16), nullptr);
  ASSERT_EQ(pool.reallocate(p, 128, 1400, 16), p);
  pool.deallocate(p);
  EXPECT_EQ(pool.used(), 0u);
}

TEST(FixedPool, RecentFreeMissPublishesBlockAndResetReclaimsEverything) {
  alignas(64) std::byte storage[4096];
  FixedPool pool(storage, sizeof(storage));
  void *small = pool.allocate(64, 16);
  void *barrier = pool.allocate(64, 16);
  ASSERT_NE(small, nullptr);
  ASSERT_NE(barrier, nullptr);

  pool.deallocate(small); // retained as the one-entry free cursor
  void *large = pool.allocate(1024, 64); // miss recent, use the large tail
  ASSERT_NE(large, nullptr);
  EXPECT_EQ(pool.used(), 1088u);

  pool.deallocate(barrier);
  pool.deallocate(large);
  pool.reset();
  EXPECT_EQ(pool.used(), 0u);
  EXPECT_NE(pool.allocate(3000, 64), nullptr);
}

TEST(FixedPool, SplitRemainderCursorCoalescesAcrossAdjacentReturns) {
  alignas(64) std::byte storage[4096];
  FixedPool pool(storage, sizeof(storage));
  void *first = pool.allocate(256, 16);
  void *second = pool.allocate(64, 16);
  void *third = pool.allocate(64, 16);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_NE(third, nullptr);

  pool.deallocate(first);
  first = pool.allocate(64, 16); // leaves a cursor immediately before second
  ASSERT_NE(first, nullptr);
  pool.deallocate(second);       // merges the preceding cursor directly
  pool.deallocate(first);        // merges the following cursor directly
  pool.deallocate(third);

  EXPECT_EQ(pool.used(), 0u);
  EXPECT_NE(pool.allocate(3000, 64), nullptr);
}

TEST(FixedPool, RejectsUnrepresentableExternalAddressRange) {
  std::byte storage{};
  FixedPool pool(&storage, std::numeric_limits<std::size_t>::max());
  EXPECT_FALSE(pool.valid());
}

TEST(FixedPool, LargeAlignmentAndOverflowKeepIndexInvariant) {
  alignas(4096) std::byte storage[64 * 1024];
  FixedPool pool(storage, sizeof(storage));
  ASSERT_TRUE(pool.valid());

  void *page_aligned = pool.allocate(257, 4096);
  ASSERT_NE(page_aligned, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(page_aligned) % 4096, 0u);
  pool.deallocate(page_aligned);

  EXPECT_EQ(pool.allocate((std::numeric_limits<std::size_t>::max)() - 32,
                          64),
            nullptr);
  constexpr std::size_t highest_alignment =
      std::size_t{1} << (std::numeric_limits<std::size_t>::digits - 1);
  EXPECT_EQ(pool.allocate(64, highest_alignment), nullptr);

  // Failed conservative mappings leave all free-list bitmaps untouched.
  void *large = pool.allocate(48 * 1024, 64);
  ASSERT_NE(large, nullptr);
  pool.deallocate(large);
  EXPECT_EQ(pool.used(), 0u);
}

TEST(FixedPool, RandomizedMixedSizeModelPreservesLiveAllocations) {
  alignas(64) std::byte storage[64 * 1024];
  FixedPool pool(storage, sizeof(storage));
  struct Live {
    unsigned char *ptr;
    std::size_t size;
    unsigned char tag;
  };
  std::vector<Live> live;
  std::uint32_t random = 0x9e3779b9u;
  auto next_random = [&] {
    random = random * 1664525u + 1013904223u;
    return random;
  };
  auto verify = [](const Live &allocation) {
    for (std::size_t i = 0; i < allocation.size; ++i)
      ASSERT_EQ(allocation.ptr[i], allocation.tag);
  };

  for (int step = 0; step < 10000; ++step) {
    const bool should_allocate =
        live.empty() || (live.size() < 160 && (next_random() & 3u) != 0);
    if (should_allocate) {
      const std::size_t bytes = 1 + next_random() % 512;
      const std::size_t alignment = std::size_t{1} << (next_random() % 7);
      auto *p = static_cast<unsigned char *>(pool.allocate(bytes, alignment));
      if (p != nullptr) {
        const auto tag = static_cast<unsigned char>(1 + step % 251);
        std::memset(p, tag, bytes);
        live.push_back({p, bytes, tag});
      }
    } else {
      const std::size_t index = next_random() % live.size();
      verify(live[index]);
      pool.deallocate(live[index].ptr);
      live[index] = live.back();
      live.pop_back();
    }
  }

  std::size_t expected_used = 0;
  for (const Live &allocation : live) {
    verify(allocation);
    expected_used += allocation.size;
  }
  EXPECT_EQ(pool.used(), expected_used);
  for (const Live &allocation : live)
    pool.deallocate(allocation.ptr);
  EXPECT_EQ(pool.used(), 0u);
  EXPECT_NE(pool.allocate(48 * 1024, 64), nullptr)
      << "all adjacent blocks should coalesce after randomized churn";
}

TEST(FixedPool, RandomizedReallocateModelPreservesDataAndAccounting) {
  alignas(64) std::byte storage[64 * 1024];
  FixedPool pool(storage, sizeof(storage));
  struct Live {
    unsigned char *ptr;
    std::size_t size;
    unsigned char tag;
  };
  std::vector<Live> live;
  std::uint32_t random = 0x243f6a88u;
  auto next_random = [&] {
    random = random * 1664525u + 1013904223u;
    return random;
  };
  auto verify = [](const Live &allocation, std::size_t bytes) {
    for (std::size_t i = 0; i < bytes; ++i)
      ASSERT_EQ(allocation.ptr[i], allocation.tag);
  };

  for (int step = 0; step < 12000; ++step) {
    const unsigned op = live.empty() ? 0 : next_random() % 3;
    if (op == 0 && live.size() < 128) {
      const std::size_t bytes = 1 + next_random() % 768;
      auto *p = static_cast<unsigned char *>(pool.allocate(bytes, 16));
      if (p != nullptr) {
        const auto tag = static_cast<unsigned char>(1 + step % 251);
        std::memset(p, tag, bytes);
        live.push_back({p, bytes, tag});
      }
    } else if (op == 1) {
      const std::size_t index = next_random() % live.size();
      Live &allocation = live[index];
      verify(allocation, allocation.size);
      const std::size_t new_size = 1 + next_random() % 1024;
      auto *resized = static_cast<unsigned char *>(pool.reallocate(
          allocation.ptr, allocation.size, new_size, 16));
      if (resized != nullptr) {
        const std::size_t preserved =
            allocation.size < new_size ? allocation.size : new_size;
        allocation.ptr = resized;
        verify(allocation, preserved);
        allocation.size = new_size;
        std::memset(allocation.ptr, allocation.tag, allocation.size);
      } else {
        verify(allocation, allocation.size);
      }
    } else {
      const std::size_t index = next_random() % live.size();
      verify(live[index], live[index].size);
      pool.deallocate(live[index].ptr);
      live[index] = live.back();
      live.pop_back();
    }
  }

  std::size_t expected_used = 0;
  for (const Live &allocation : live) {
    verify(allocation, allocation.size);
    expected_used += allocation.size;
  }
  EXPECT_EQ(pool.used(), expected_used);
  for (const Live &allocation : live)
    pool.deallocate(allocation.ptr);
  EXPECT_EQ(pool.used(), 0u);
}

TEST(FixedPool, FragmentedLookupExaminesAtMostTwoCandidates) {
  alignas(64) std::byte storage[64 * 1024];
  FixedPool pool(storage, sizeof(storage));
  std::vector<void *> holes;
  std::vector<void *> barriers;
  holes.reserve(96);
  barriers.reserve(96);

  // Model a long-lived packet/cache arena: differently sized values are
  // separated by live guards, then returned out of size order. The final small
  // return occupies the one-entry cursor, while many unsuitable and suitable
  // holes remain indexed.
  for (std::size_t i = 0; i < 96; ++i) {
    void *hole = pool.allocate(192 + (i % 12) * 32, 16);
    void *barrier = pool.allocate(48, 16);
    ASSERT_NE(hole, nullptr);
    ASSERT_NE(barrier, nullptr);
    holes.push_back(hole);
    barriers.push_back(barrier);
  }
  for (std::size_t i = 1; i < holes.size(); ++i)
    pool.deallocate(holes[i]);
  pool.deallocate(holes[0]); // too small recent cursor

  void *selected = pool.allocate(400, 64);
  ASSERT_NE(selected, nullptr);
  EXPECT_LE(pool.debug_last_fit_candidates(), 2u);

  pool.deallocate(selected);
  for (void *barrier : barriers)
    pool.deallocate(barrier);
  EXPECT_EQ(pool.used(), 0u);
}

TEST(FixedPoolDeathTest, DoubleReturnTraps) {
  EXPECT_DEATH(
      {
        alignas(64) std::byte storage[2048];
        FixedPool pool(storage, sizeof(storage));
        void *p = pool.allocate(64);
        pool.deallocate(p);
        pool.deallocate(p);
      },
      ".*");
}

TEST(FixedPoolDeathTest, ForeignPointerTrapsBeforeDereference) {
  EXPECT_DEATH(
      {
        alignas(64) std::byte storage[2048];
        FixedPool pool(storage, sizeof(storage));
        int foreign = 0;
        pool.deallocate(&foreign);
      },
      ".*");
}
