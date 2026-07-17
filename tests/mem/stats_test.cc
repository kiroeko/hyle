#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/mem/stats.h"

using hyle::mem::AllocStats;
using hyle::mem::AllocStatsSnapshot;
using hyle::mem::LeakEntry;
using hyle::mem::LeakTracker;
using hyle::mem::StatsAllocator;
using hyle::mem::system_allocator;

// clay bug reproduction: unsigned (0 - old_size) must NOT wrap counters.
TEST(AllocStats, FreeDoesNotUnderflow) {
  AllocStats st;
  st.record_alloc(1u << 20);  // 1 MiB
  auto mid = st.snapshot();
  EXPECT_EQ(mid.live_count, 1u);
  EXPECT_EQ(mid.live_bytes, 1u << 20);

  st.record_free(1u << 20);
  auto end = st.snapshot();
  EXPECT_EQ(end.live_count, 0u);
  EXPECT_EQ(end.live_bytes, 0u);
  EXPECT_EQ(end.total_frees, 1u);
  // If clay-style unsigned (0 - size) were used, live_bytes would be ~2^64-1.
  EXPECT_LT(end.live_bytes, 1u << 30);
}

TEST(AllocStats, ReallocFreePathNoUnderflow) {
  AllocStats st;
  st.record_realloc(/*old=*/0, /*new=*/4096);
  st.record_realloc(/*old=*/4096, /*new=*/0);
  auto s = st.snapshot();
  EXPECT_EQ(s.live_count, 0u);
  EXPECT_EQ(s.live_bytes, 0u);
  EXPECT_EQ(s.total_frees, 1u);
  EXPECT_EQ(s.reallocs, 2u);
}

TEST(AllocStats, ReallocGrowAndShrink) {
  AllocStats st;
  st.record_alloc(100);
  st.record_realloc(100, 400);
  auto mid = st.snapshot();
  EXPECT_EQ(mid.live_count, 1u);
  EXPECT_EQ(mid.live_bytes, 400u);
  EXPECT_EQ(mid.peak_live_bytes, 400u);

  st.record_realloc(400, 50);
  auto end = st.snapshot();
  EXPECT_EQ(end.live_count, 1u);
  EXPECT_EQ(end.live_bytes, 50u);
  EXPECT_EQ(end.peak_live_bytes, 400u);  // peak sticky
  st.record_free(50);
  EXPECT_TRUE(st.balanced());
}

TEST(AllocStats, PeakTracksHighWater) {
  AllocStats st;
  st.record_alloc(10);
  st.record_alloc(90);
  EXPECT_EQ(st.snapshot().peak_live_bytes, 100u);
  st.record_free(10);
  EXPECT_EQ(st.snapshot().live_bytes, 90u);
  EXPECT_EQ(st.snapshot().peak_live_bytes, 100u);
  st.record_free(90);
}

TEST(StatsAllocator, BalancedRoundTrip) {
  StatsAllocator sa(system_allocator());
  auto a = sa.get();
  void* p = a.allocate(128);
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(sa.snapshot().live_count, 1u);
  EXPECT_EQ(sa.snapshot().live_bytes, 128u);
  a.deallocate(p, 128);
  EXPECT_TRUE(sa.balanced());
  EXPECT_EQ(sa.snapshot().total_allocs, 1u);
  EXPECT_EQ(sa.snapshot().total_frees, 1u);
}

TEST(StatsAllocator, ReallocPreservesAccounting) {
  StatsAllocator sa(system_allocator());
  auto a = sa.get();
  void* p = a.allocate(64);
  ASSERT_NE(p, nullptr);
  void* q = a.reallocate(p, 64, 256);
  ASSERT_NE(q, nullptr);
  auto s = sa.snapshot();
  EXPECT_EQ(s.live_count, 1u);
  EXPECT_EQ(s.live_bytes, 256u);
  a.deallocate(q, 256);
  EXPECT_TRUE(sa.balanced());
}

TEST(StatsAllocator, ReallocToZeroIsFree) {
  StatsAllocator sa(system_allocator());
  auto a = sa.get();
  void* p = a.allocate(32);
  ASSERT_NE(p, nullptr);
  void* q = a.reallocate(p, 32, 0);
  EXPECT_EQ(q, nullptr);
  EXPECT_TRUE(sa.balanced());
}

TEST(StatsAllocator, ManyAllocsPeak) {
  StatsAllocator sa(system_allocator());
  auto a = sa.get();
  void* ps[16];
  for (int i = 0; i < 16; ++i) {
    ps[i] = a.allocate(1024);
    ASSERT_NE(ps[i], nullptr);
  }
  EXPECT_EQ(sa.snapshot().live_count, 16u);
  EXPECT_EQ(sa.snapshot().live_bytes, 16u * 1024u);
  EXPECT_EQ(sa.snapshot().peak_live_bytes, 16u * 1024u);
  for (int i = 0; i < 16; ++i) a.deallocate(ps[i], 1024);
  EXPECT_TRUE(sa.balanced());
  EXPECT_EQ(sa.snapshot().peak_live_bytes, 16u * 1024u);
}

TEST(StatsAllocator, ConcurrentNoLostCounts) {
  StatsAllocator sa(system_allocator());
  auto a = sa.get();
  constexpr int kThreads = 4;
  constexpr int kIters = 500;
  std::vector<std::thread> th;
  th.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    th.emplace_back([a] {
      for (int i = 0; i < kIters; ++i) {
        void* p = a.allocate(64);
        if (p) a.deallocate(p, 64);
      }
    });
  }
  for (auto& x : th) x.join();
  auto s = sa.snapshot();
  EXPECT_TRUE(sa.balanced());
  EXPECT_EQ(s.total_allocs, static_cast<std::uint64_t>(kThreads * kIters));
  EXPECT_EQ(s.total_frees, static_cast<std::uint64_t>(kThreads * kIters));
}

TEST(LeakTracker, DetectsOutstanding) {
  StatsAllocator sa(system_allocator(), /*track_leaks=*/true);
  ASSERT_NE(sa.leaks(), nullptr);
  auto a = sa.get();
  void* p = a.allocate(77);
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(sa.leaks()->outstanding(), 1u);
  LeakEntry e[4]{};
  std::size_t n = sa.leaks()->copy_leaks(e, 4);
  EXPECT_EQ(n, 1u);
  EXPECT_EQ(e[0].ptr, p);
  EXPECT_EQ(e[0].size, 77u);
  a.deallocate(p, 77);
  EXPECT_EQ(sa.leaks()->outstanding(), 0u);
  EXPECT_TRUE(sa.balanced());
}

TEST(LeakTracker, LeakRemainsUntilFreed) {
  StatsAllocator sa(system_allocator(), /*track_leaks=*/true);
  auto a = sa.get();
  void* leak = a.allocate(16);
  void* ok = a.allocate(16);
  a.deallocate(ok, 16);
  EXPECT_EQ(sa.leaks()->outstanding(), 1u);
  EXPECT_EQ(sa.snapshot().live_count, 1u);
  // Intentionally free the "leak" so the process stays clean.
  a.deallocate(leak, 16);
  EXPECT_EQ(sa.leaks()->outstanding(), 0u);
}

TEST(LeakTracker, DoubleFreeNotTracked) {
  LeakTracker lt;
  void* fake = reinterpret_cast<void*>(static_cast<std::uintptr_t>(0x1000));
  lt.track_alloc(fake, 8);
  EXPECT_TRUE(lt.track_free(fake));
  EXPECT_FALSE(lt.track_free(fake));  // second free
  EXPECT_EQ(lt.outstanding(), 0u);
}

TEST(StatsAllocator, SystemAllocatorUnaffected) {
  // system path has no hidden stats — allocate without StatsAllocator is fine.
  auto a = system_allocator();
  void* p = a.allocate(8);
  ASSERT_NE(p, nullptr);
  a.deallocate(p, 8);
}
