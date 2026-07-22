#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/mem/fixed_pool.h"
#include "hyle/mem/object_pool.h"
#include "hyle/mem/region.h"
#include "hyle/mem/shared_pool.h"
#include "hyle/mem/stack_allocator.h"
#include "hyle/platform/thread.h"

using hyle::mem::FixedPool;
using hyle::mem::FixedRegion;
using hyle::mem::ObjectPool;
using hyle::mem::ObjectPoolCached;
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
};

void *fail_alloc(void *opaque, std::size_t size,
                 std::size_t alignment) noexcept {
  auto &state = *static_cast<FailAllocatorState *>(opaque);
  const std::size_t call = state.calls++;
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
  ObjectPoolCached<int> cached(oom, 8);
  EXPECT_FALSE(hard.valid());
  EXPECT_FALSE(cached.valid());
  EXPECT_EQ(hard.create(1), nullptr);
  EXPECT_EQ(cached.create(1), nullptr);
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

TEST(ObjectPoolCached, TypedCachedShellPublishes) {
  ObjectPoolCached<int> pool(system_allocator(), 4);
  int *p = pool.create(7);
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(*p, 7);
  pool.destroy(p);
  pool.publish_thread_cache();
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

TEST(PooledRegion, ExplicitNameAndArenaAliasShareSingleBlockContract) {
  hyle::mem::SharedPool pool(system_allocator(), 256, 1, 64);
  hyle::mem::PooledRegion region(&pool);
  ASSERT_TRUE(region.valid());
  EXPECT_NE(region.allocate(64, 64), nullptr);
  hyle::mem::Arena exhausted_alias(&pool);
  EXPECT_FALSE(exhausted_alias.valid());
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

TEST(FixedPool, RejectsUnrepresentableExternalAddressRange) {
  std::byte storage{};
  FixedPool pool(&storage, std::numeric_limits<std::size_t>::max());
  EXPECT_FALSE(pool.valid());
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
