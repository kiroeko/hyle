#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory_resource>
#include <stdexcept>
#include <thread>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/mem/pmr.h"
#include "hyle/mem/quota_allocator.h"
#include "hyle/mem/shared_pool.h"
#include "hyle/mem/stats.h"
#include "hyle/mem/system_inline.h"

using hyle::mem::Allocator;
using hyle::mem::kDefaultAlignment;
using hyle::mem::QuotaAllocator;
using hyle::mem::StatsAllocator;
using hyle::mem::system_allocator;
using hyle::mem::system_good_size;
using hyle::mem::system_usable_size;
using hyle::mem::SystemAllocator;

namespace {
bool is_aligned(const void *p, std::size_t a) {
  return (reinterpret_cast<std::uintptr_t>(p) & (a - 1)) == 0;
}
void *quota_oom_alloc(void *, std::size_t, std::size_t) noexcept {
  return nullptr;
}
void quota_oom_free(void *, void *, std::size_t, std::size_t) noexcept {}
void *quota_oom_realloc(void *, void *, std::size_t, std::size_t,
                        std::size_t) noexcept {
  return nullptr;
}
} // namespace

TEST(Allocator, SystemAllocatorIsValid) {
  Allocator a = system_allocator();
  EXPECT_TRUE(a.valid());
  EXPECT_NE(a.alloc_fn(), nullptr);
}

TEST(Allocator, DefaultAllocatorIsInvalid) {
  Allocator a;
  EXPECT_FALSE(a.valid());
}

TEST(Allocator, AllocateWriteFree) {
  Allocator a = system_allocator();
  void *p = a.allocate(128);
  ASSERT_NE(p, nullptr);
  EXPECT_TRUE(is_aligned(p, kDefaultAlignment));
  std::memset(p, 0xAB, 128); // must be fully writable
  a.deallocate(p, 128);
}

TEST(Allocator, AllocateZeroReturnsNull) {
  Allocator a = system_allocator();
  EXPECT_EQ(a.allocate(0), nullptr); // new_size==0 collapses to the free path
}

// The headline of mem-01: alignment is honored end-to-end (clay dropped it).
TEST(Allocator, RespectsExplicitAlignment) {
  Allocator a = system_allocator();
  for (std::size_t align :
       {std::size_t{16}, std::size_t{32}, std::size_t{64}, std::size_t{128},
        std::size_t{256}, std::size_t{4096}}) {
    void *p = a.allocate(align * 4, align);
    ASSERT_NE(p, nullptr) << "align=" << align;
    EXPECT_TRUE(is_aligned(p, align)) << "align=" << align;
    a.deallocate(p, align * 4, align);
  }
}

TEST(Allocator, SmallAlignmentBumpedToDefault) {
  Allocator a = system_allocator();
  void *p = a.allocate(8, 1); // sub-default request still safe-aligned
  ASSERT_NE(p, nullptr);
  EXPECT_TRUE(is_aligned(p, kDefaultAlignment));
  a.deallocate(p, 8, 1);
}

TEST(Allocator, ReallocatePreservesData) {
  Allocator a = system_allocator();
  constexpr int n = 64;
  auto *p = static_cast<int *>(a.allocate(sizeof(int) * n, 64));
  ASSERT_NE(p, nullptr);
  for (int i = 0; i < n; ++i)
    p[i] = i * 7;

  auto *q = static_cast<int *>(
      a.reallocate(p, sizeof(int) * n, sizeof(int) * n * 2, 64));
  ASSERT_NE(q, nullptr);
  EXPECT_TRUE(is_aligned(q, 64));
  for (int i = 0; i < n; ++i)
    EXPECT_EQ(q[i], i * 7); // old data intact
  a.deallocate(q, sizeof(int) * n * 2, 64);
}

TEST(Allocator, ReallocateFromNullIsAllocate) {
  Allocator a = system_allocator();
  void *p = a.reallocate(nullptr, 0, 256, 64);
  ASSERT_NE(p, nullptr);
  EXPECT_TRUE(is_aligned(p, 64));
  a.deallocate(p, 256, 64);
}

TEST(Allocator, ManyDistinctNonOverlapping) {
  Allocator a = system_allocator();
  constexpr int kN = 256;
  std::vector<void *> ps;
  ps.reserve(kN);
  for (int i = 0; i < kN; ++i) {
    void *p = a.allocate(64, 64);
    ASSERT_NE(p, nullptr);
    EXPECT_TRUE(is_aligned(p, 64));
    std::memset(p, i & 0xFF, 64); // write each fully
    ps.push_back(p);
  }
  for (void *p : ps)
    a.deallocate(p, 64, 64);
}

// ---- typed convenience layer (mem-01b, clay allocator.h parity) -------------
namespace {
std::atomic<int> g_typed_live{0};
struct TCounted {
  int v;
  explicit TCounted(int x) : v(x) { g_typed_live.fetch_add(1); }
  ~TCounted() { g_typed_live.fetch_sub(1); }
};
} // namespace

TEST(AllocatorTyped, AllocateNAlignedAndWritable) {
  Allocator a = system_allocator();
  int *p = a.allocate_n<int>(5);
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % alignof(int), 0u);
  for (int i = 0; i < 5; ++i)
    p[i] = i; // writable
  EXPECT_EQ(p[4], 4);
  a.deallocate_n(p, 5);
}

TEST(AllocatorTyped, AllocateZeroedZeroes) {
  Allocator a = system_allocator();
  int *p = a.allocate_zeroed<int>(8);
  ASSERT_NE(p, nullptr);
  for (int i = 0; i < 8; ++i)
    EXPECT_EQ(p[i], 0);
  a.deallocate_n(p, 8);
}

TEST(AllocatorTyped, CountOverflowReturnsNull) {
  Allocator a = system_allocator();
  constexpr std::size_t too_many = SIZE_MAX / sizeof(std::uint64_t) + 2;
  EXPECT_EQ(a.allocate_n<std::uint64_t>(too_many), nullptr);
  EXPECT_EQ(a.allocate_zeroed<std::uint64_t>(too_many), nullptr);
}

TEST(SystemAllocatorTyped, CountOverflowReturnsNull) {
  constexpr std::size_t too_many = SIZE_MAX / sizeof(std::uint64_t) + 2;
  EXPECT_EQ(SystemAllocator::allocate_n<std::uint64_t>(too_many), nullptr);
  EXPECT_EQ(SystemAllocator::allocate_zeroed<std::uint64_t>(too_many), nullptr);
}

TEST(SystemAllocatorTypedDeathTest, DeallocateCountOverflowTraps) {
  EXPECT_DEATH(
      {
        SystemAllocator::deallocate_n<std::uint64_t>(
            nullptr, SIZE_MAX / sizeof(std::uint64_t) + 2);
      },
      ".*");
}

namespace {
struct ThrowsInCtor {
  ThrowsInCtor() { throw std::runtime_error("ctor failure"); }
};
} // namespace

TEST(AllocatorTyped, ConstructorFailureReleasesStorage) {
  StatsAllocator tracked(system_allocator());
  Allocator a = tracked.get();
  EXPECT_THROW((void)a.construct<ThrowsInCtor>(), std::runtime_error);
  EXPECT_TRUE(tracked.balanced());
  EXPECT_EQ(tracked.snapshot().total_allocs, 1u);
  EXPECT_EQ(tracked.snapshot().total_frees, 1u);
}

TEST(AllocatorTyped, ConstructRunsCtorDestroyRunsDtor) {
  g_typed_live.store(0);
  Allocator a = system_allocator();
  TCounted *c = a.construct<TCounted>(7);
  ASSERT_NE(c, nullptr);
  EXPECT_EQ(c->v, 7);
  EXPECT_EQ(g_typed_live.load(), 1);
  a.destroy(c); // ~T + free through THIS allocator (not global delete)
  EXPECT_EQ(g_typed_live.load(), 0);
}

TEST(AllocatorTyped, MakeUniqueAutoDestroys) {
  g_typed_live.store(0);
  Allocator a = system_allocator();
  {
    auto u = a.make_unique<TCounted>(99);
    ASSERT_NE(u.get(), nullptr);
    EXPECT_EQ(u->v, 99);
    EXPECT_EQ(g_typed_live.load(), 1);
  } // unique_ptr deleter destroys through the allocator
  EXPECT_EQ(g_typed_live.load(), 0);
}

// ---- usable_size / good_size (SystemAllocator only) -------------------------

TEST(SystemSize, UsableSizeAtLeastRequested) {
  auto a = system_allocator();
  void *p = a.allocate(100);
  ASSERT_NE(p, nullptr);
  std::size_t u = system_usable_size(p);
  EXPECT_GE(u, 100u); // usable is never less than requested
  // The whole usable range is writable (mimalloc rounded up to a size class).
  std::memset(p, 0xAB, u);
  EXPECT_EQ(static_cast<unsigned char *>(p)[u - 1], 0xAB);
  a.deallocate(p, 100);
  EXPECT_EQ(system_usable_size(nullptr), 0u); // null -> 0
}

TEST(SystemSize, GoodSizeIsSafeRecommendedRequestNotUsablePrediction) {
  // Some mimalloc platform/instrumentation modes expose exactly the requested
  // payload from usable_size even though good_size recommends a larger class.
  std::size_t g = system_good_size(100);
  EXPECT_GE(g, 100u);
  auto a = system_allocator();
  void *p = a.allocate(g);
  ASSERT_NE(p, nullptr);
  EXPECT_GE(system_usable_size(p), g);
  a.deallocate(p, g);
  EXPECT_EQ(system_good_size(0), 0u); // 0 -> 0
}

TEST(QuotaAllocator, EnforcesStrictRequestedByteLimit) {
  QuotaAllocator quota(system_allocator(), 128);
  Allocator a = quota.get();
  void *p = a.allocate(80, 16);
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(quota.used(), 80u);
  EXPECT_EQ(quota.remaining(), 48u);
  EXPECT_EQ(a.allocate(49, 16), nullptr);
  void *q = a.allocate(48, 16);
  ASSERT_NE(q, nullptr);
  EXPECT_EQ(quota.used(), 128u);
  EXPECT_EQ(quota.peak(), 128u);
  a.deallocate(q, 48, 16);
  a.deallocate(p, 80, 16);
  EXPECT_EQ(quota.used(), 0u);
  EXPECT_EQ(quota.peak(), 128u);
}

TEST(QuotaAllocator, ReallocateReservesAndRollsBackExactly) {
  QuotaAllocator quota(system_allocator(), 128);
  Allocator a = quota.get();
  auto *p = static_cast<unsigned char *>(a.allocate(64, 16));
  ASSERT_NE(p, nullptr);
  p[0] = 7;
  EXPECT_EQ(a.reallocate(p, 64, 129, 16), nullptr);
  EXPECT_EQ(quota.used(), 64u);
  auto *grown = static_cast<unsigned char *>(a.reallocate(p, 64, 96, 16));
  ASSERT_NE(grown, nullptr);
  EXPECT_EQ(grown[0], 7);
  EXPECT_EQ(quota.used(), 96u);
  auto *shrunk = static_cast<unsigned char *>(a.reallocate(grown, 96, 32, 16));
  ASSERT_NE(shrunk, nullptr);
  EXPECT_EQ(quota.used(), 32u);
  a.deallocate(shrunk, 32, 16);
  EXPECT_EQ(quota.used(), 0u);
}

TEST(QuotaAllocator, BackingOomRollsReservationBack) {
  Allocator oom(&quota_oom_alloc, &quota_oom_free, &quota_oom_realloc, nullptr);
  QuotaAllocator quota(oom, 1024);
  EXPECT_EQ(quota.get().allocate(100), nullptr);
  EXPECT_EQ(quota.used(), 0u);
  EXPECT_EQ(quota.peak(), 0u); // failed backing allocation never becomes live
}

TEST(QuotaAllocator, ConcurrentAdmissionNeverExceedsHardLimit) {
  constexpr std::size_t kLimit = 512;
  constexpr std::size_t kRequest = 32;
  QuotaAllocator quota(system_allocator(), kLimit);
  Allocator alloc = quota.get();
  std::atomic<std::size_t> observed_max{0};
  std::vector<std::thread> threads;
  for (int thread = 0; thread < 8; ++thread) {
    threads.emplace_back([&] {
      for (int i = 0; i < 2000; ++i) {
        void *p = alloc.allocate(kRequest, 16);
        if (p == nullptr) {
          std::this_thread::yield();
          continue;
        }
        const std::size_t used = quota.used();
        std::size_t prior = observed_max.load(std::memory_order_relaxed);
        while (used > prior && !observed_max.compare_exchange_weak(
                                   prior, used, std::memory_order_relaxed,
                                   std::memory_order_relaxed)) {
        }
        EXPECT_LE(used, kLimit);
        alloc.deallocate(p, kRequest, 16);
      }
    });
  }
  for (auto &thread : threads)
    thread.join();
  EXPECT_EQ(quota.used(), 0u);
  EXPECT_LE(quota.peak(), kLimit);
  EXPECT_GT(observed_max.load(std::memory_order_relaxed), 0u);
}

TEST(AllocatorComposition, StatsOverQuotaOverSharedPoolKeepsOneTruth) {
  hyle::mem::SharedPool pool(system_allocator(), 128, 1, 64);
  QuotaAllocator quota(pool.as_allocator(), 512);
  StatsAllocator stats(quota.get(), /*track_leaks=*/true);
  Allocator alloc = stats.get();

  auto *p = static_cast<unsigned char *>(alloc.allocate(32, 16));
  ASSERT_NE(p, nullptr);
  p[0] = 0x3C;
  EXPECT_EQ(alloc.reallocate(p, 32, 96, 16), p);
  EXPECT_EQ(p[0], 0x3C);
  EXPECT_EQ(quota.used(), 96u);
  EXPECT_EQ(stats.snapshot().live_bytes, 96u);
  ASSERT_NE(stats.leaks(), nullptr);
  EXPECT_EQ(stats.leaks()->outstanding(), 1u);

  EXPECT_EQ(alloc.reallocate(p, 96, 256, 16), nullptr);
  EXPECT_EQ(p[0], 0x3C);
  EXPECT_EQ(quota.used(), 96u);
  EXPECT_EQ(stats.snapshot().live_bytes, 96u);
  EXPECT_EQ(stats.leaks()->outstanding(), 1u);

  EXPECT_EQ(alloc.reallocate(p, 96, 0, 16), nullptr);
  EXPECT_EQ(quota.used(), 0u);
  EXPECT_EQ(stats.snapshot().live_count, 0u);
  EXPECT_EQ(stats.snapshot().live_bytes, 0u);
  EXPECT_EQ(stats.leaks()->outstanding(), 0u);

  void *reused = alloc.allocate(64, 16);
  ASSERT_NE(reused, nullptr) << "balanced decorators must not hide a leaked slot";
  alloc.deallocate(reused, 64, 16);
  EXPECT_TRUE(stats.balanced());
  EXPECT_EQ(quota.used(), 0u);
}

TEST(AllocatorComposition, QuotaOverStatsOverSharedPoolKeepsOneTruth) {
  hyle::mem::SharedPool pool(system_allocator(), 128, 1, 64);
  StatsAllocator stats(pool.as_allocator(), /*track_leaks=*/true);
  QuotaAllocator quota(stats.get(), 512);
  Allocator alloc = quota.get();

  void *p = alloc.reallocate(nullptr, 0, 64, 16);
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(quota.used(), 64u);
  EXPECT_EQ(stats.snapshot().live_bytes, 64u);
  EXPECT_EQ(alloc.reallocate(p, 64, 0, 16), nullptr);
  EXPECT_EQ(quota.used(), 0u);
  EXPECT_TRUE(stats.balanced());
  EXPECT_EQ(stats.leaks()->outstanding(), 0u);
  void *reused = pool.allocate();
  ASSERT_NE(reused, nullptr);
  pool.deallocate(reused);
}

TEST(AllocatorComposition, PmrOverDecoratedSharedPoolReturnsCapacity) {
  hyle::mem::SharedPool pool(system_allocator(), 128, 1, 64);
  QuotaAllocator quota(pool.as_allocator(), 128);
  StatsAllocator stats(quota.get());
  hyle::mem::PmrResource resource(stats.get());
  std::pmr::polymorphic_allocator<std::byte> alloc(&resource);

  std::byte *p = alloc.allocate(64);
  ASSERT_NE(p, nullptr);
  alloc.deallocate(p, 64);
  EXPECT_TRUE(stats.balanced());
  EXPECT_EQ(quota.used(), 0u);
  void *reused = pool.allocate();
  ASSERT_NE(reused, nullptr);
  pool.deallocate(reused);
}
