#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <limits>
#include <thread>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/mem/stats.h"

using hyle::mem::AllocStats;
using hyle::mem::AllocStatsSnapshot;
using hyle::mem::AllocatorRef;
using hyle::mem::LeakEntry;
using hyle::mem::LeakTracker;
using hyle::mem::StatsAllocator;
using hyle::mem::system_allocator;

namespace {

struct MetadataBackingState {
  std::size_t calls = 0;
  std::size_t fail_at = (std::numeric_limits<std::size_t>::max)();
  std::size_t outstanding = 0;
};

void* metadata_alloc(void* opaque, std::size_t size,
                     std::size_t alignment) noexcept {
  auto& state = *static_cast<MetadataBackingState*>(opaque);
  const std::size_t call = state.calls++;
  if (call == state.fail_at) return nullptr;
  void* pointer = system_allocator().allocate(size, alignment);
  if (pointer != nullptr) ++state.outstanding;
  return pointer;
}

void metadata_free(void* opaque, void* pointer, std::size_t size,
                   std::size_t alignment) noexcept {
  if (pointer == nullptr) return;
  auto& state = *static_cast<MetadataBackingState*>(opaque);
  EXPECT_GT(state.outstanding, 0u);
  --state.outstanding;
  system_allocator().deallocate(pointer, size, alignment);
}

void* metadata_realloc(void* opaque, void* pointer, std::size_t old_size,
                       std::size_t new_size,
                       std::size_t alignment) noexcept {
  auto& state = *static_cast<MetadataBackingState*>(opaque);
  if (new_size == 0) {
    metadata_free(opaque, pointer, old_size, alignment);
    return nullptr;
  }
  const std::size_t call = state.calls++;
  if (call == state.fail_at) return nullptr;
  void* out = system_allocator().reallocate(pointer, old_size, new_size,
                                             alignment);
  if (pointer == nullptr && out != nullptr) ++state.outstanding;
  return out;
}

hyle::mem::AllocatorRef metadata_backing(MetadataBackingState* state) {
  return hyle::mem::AllocatorRef(&metadata_alloc, &metadata_free,
                              &metadata_realloc, state);
}

}  // namespace

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
  StatsAllocator sa = StatsAllocator::create(system_allocator()).value();
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

TEST(StatsAllocator, ConstructionOomReturnsStatusAndLeaksNothing) {
  for (std::size_t fail_at = 0; fail_at != 2; ++fail_at) {
    MetadataBackingState state;
    state.fail_at = fail_at;
    auto created = StatsAllocator::create(system_allocator(),
                                          /*track_leaks=*/true,
                                          metadata_backing(&state));
    ASSERT_FALSE(created) << "fail_at=" << fail_at;
    EXPECT_TRUE(created.error().IsResourceExhausted());
    EXPECT_EQ(state.outstanding, 0u) << "fail_at=" << fail_at;
  }
}

TEST(StatsAllocator, StateAndLeakMapUseInjectedBacking) {
  MetadataBackingState state;
  {
    auto created = StatsAllocator::create(system_allocator(),
                                          /*track_leaks=*/true,
                                          metadata_backing(&state));
    ASSERT_TRUE(created);
    StatsAllocator stats = *std::move(created);
    EXPECT_GE(state.outstanding, 2u);  // LeakTracker::Impl + StatsState.

    const std::size_t calls_before_data = state.calls;
    hyle::mem::AllocatorRef measured = stats.get();
    void* pointer = measured.allocate(64, 16);
    ASSERT_NE(pointer, nullptr);
    // The measured allocation uses the separate system backing. At least one
    // PMR allocation here proves leak metadata did not fall back to global new.
    EXPECT_GE(state.calls, calls_before_data + 1);
    measured.deallocate(pointer, 64, 16);
  }
  EXPECT_EQ(state.outstanding, 0u);
}

TEST(LeakTracker, ConstructionOomReturnsStatus) {
  MetadataBackingState state;
  state.fail_at = 0;
  auto created = LeakTracker::create(metadata_backing(&state));
  ASSERT_FALSE(created);
  EXPECT_TRUE(created.error().IsResourceExhausted());
  EXPECT_EQ(state.outstanding, 0u);
}

TEST(LeakTracker, MetadataOomMakesInventoryObservablyIncomplete) {
  MetadataBackingState state;
  {
    auto created = StatsAllocator::create(system_allocator(),
                                          /*track_leaks=*/true,
                                          metadata_backing(&state));
    ASSERT_TRUE(created);
    StatsAllocator stats = *std::move(created);
    ASSERT_NE(stats.leaks(), nullptr);
    EXPECT_TRUE(stats.leaks()->complete());

    // Fail the next metadata allocation: the measured allocation itself still
    // succeeds, counters remain exact, and the optional map reports its loss.
    state.fail_at = state.calls;
    AllocatorRef measured = stats.get();
    void* pointer = measured.allocate(64, 16);
    ASSERT_NE(pointer, nullptr);
    EXPECT_EQ(stats.snapshot().live_count, 1u);
    EXPECT_EQ(stats.leaks()->outstanding(), 0u);
    EXPECT_EQ(stats.leaks()->dropped_records(), 1u);
    EXPECT_FALSE(stats.leaks()->complete());

    measured.deallocate(pointer, 64, 16);
    EXPECT_TRUE(stats.balanced());
    stats.leaks()->clear();
    EXPECT_EQ(stats.leaks()->dropped_records(), 0u);
    EXPECT_TRUE(stats.leaks()->complete());
  }
  EXPECT_EQ(state.outstanding, 0u);
}

TEST(StatsAllocator, ReallocPreservesAccounting) {
  StatsAllocator sa = StatsAllocator::create(system_allocator()).value();
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

TEST(StatsAllocator, ReallocUsesRecordedOldSizeWhenTrackingLeaks) {
  StatsAllocator sa =
      StatsAllocator::create(system_allocator(), /*track_leaks=*/true).value();
  auto a = sa.get();
  void* p = a.allocate(64);
  ASSERT_NE(p, nullptr);
  // The backing allocator owns the real size metadata. The diagnostic wrapper
  // must not corrupt its counters merely because a caller supplied a bad hint.
  void* q = a.reallocate(p, 999, 128);
  ASSERT_NE(q, nullptr);
  EXPECT_EQ(sa.snapshot().live_count, 1u);
  EXPECT_EQ(sa.snapshot().live_bytes, 128u);
  a.deallocate(q, 128);
  EXPECT_TRUE(sa.balanced());
}

TEST(StatsAllocator, ReallocToZeroIsFree) {
  StatsAllocator sa = StatsAllocator::create(system_allocator()).value();
  auto a = sa.get();
  void* p = a.allocate(32);
  ASSERT_NE(p, nullptr);
  void* q = a.reallocate(p, 32, 0);
  EXPECT_EQ(q, nullptr);
  EXPECT_TRUE(sa.balanced());
}

TEST(StatsAllocator, ManyAllocsPeak) {
  StatsAllocator sa = StatsAllocator::create(system_allocator()).value();
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
  StatsAllocator sa = StatsAllocator::create(system_allocator()).value();
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

TEST(StatsAllocator, ConcurrentPeakCoversQuiescentHighWater) {
  StatsAllocator sa = StatsAllocator::create(system_allocator()).value();
  const AllocatorRef allocator = sa.get();
  constexpr int kThreads = 8;
  constexpr std::size_t kBytes = 64;
  std::atomic<int> ready{0};
  std::atomic<bool> allocation_failed{false};
  std::atomic<bool> release{false};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&] {
      void* pointer = allocator.allocate(kBytes);
      if (pointer == nullptr)
        allocation_failed.store(true, std::memory_order_relaxed);
      ready.fetch_add(1, std::memory_order_release);
      while (!release.load(std::memory_order_acquire))
        std::this_thread::yield();
      if (pointer != nullptr)
        allocator.deallocate(pointer, kBytes);
    });
  }
  while (ready.load(std::memory_order_acquire) != kThreads)
    std::this_thread::yield();
  const auto high = sa.snapshot();
  EXPECT_FALSE(allocation_failed.load(std::memory_order_relaxed));
  EXPECT_EQ(high.live_bytes, kThreads * kBytes);
  EXPECT_GE(high.peak_live_bytes, high.live_bytes);
  release.store(true, std::memory_order_release);
  for (auto& thread : threads) thread.join();
  EXPECT_TRUE(sa.balanced());
  EXPECT_GE(sa.snapshot().peak_live_bytes, kThreads * kBytes);
}

TEST(AllocStatsDeathTest, ResetWithLiveAllocationTraps) {
  EXPECT_DEATH(
      {
        AllocStats stats;
        stats.record_alloc(8);
        stats.reset();
      },
      ".*");
}

TEST(StatsAllocatorDeathTest, DestroyWithLiveAllocationTraps) {
  EXPECT_DEATH(
      {
        StatsAllocator stats = StatsAllocator::create(system_allocator()).value();
        AllocatorRef allocator = stats.get();
        (void)allocator.allocate(8);
      },
      ".*");
}

TEST(StatsAllocatorDeathTest, MoveOverwriteWithLiveAllocationTraps) {
  EXPECT_DEATH(
      {
        StatsAllocator target = StatsAllocator::create(system_allocator()).value();
        AllocatorRef allocator = target.get();
        (void)allocator.allocate(8);
        StatsAllocator source = StatsAllocator::create(system_allocator()).value();
        target = std::move(source);
      },
      ".*");
}

TEST(LeakTracker, DetectsOutstanding) {
  StatsAllocator sa =
      StatsAllocator::create(system_allocator(), /*track_leaks=*/true).value();
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
  StatsAllocator sa =
      StatsAllocator::create(system_allocator(), /*track_leaks=*/true).value();
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
  LeakTracker lt = LeakTracker::create(system_allocator()).value();
  void* fake = reinterpret_cast<void*>(static_cast<std::uintptr_t>(0x1000));
  lt.track_alloc(fake, 8);
  EXPECT_TRUE(lt.track_free(fake));
  EXPECT_FALSE(lt.track_free(fake));  // second free
  EXPECT_EQ(lt.outstanding(), 0u);
}

TEST(LeakTracker, TrackFreeReturnsRecordedSize) {
  LeakTracker lt = LeakTracker::create(system_allocator()).value();
  void* fake = reinterpret_cast<void*>(static_cast<std::uintptr_t>(0x2000));
  lt.track_alloc(fake, 4096);
  std::size_t got = 0;
  EXPECT_TRUE(lt.track_free(fake, &got));
  EXPECT_EQ(got, 4096u);
  // Untracked ptr leaves out_size untouched.
  std::size_t sentinel = 123u;
  EXPECT_FALSE(lt.track_free(fake, &sentinel));
  EXPECT_EQ(sentinel, 123u);
}

TEST(StatsAllocator, WrongFreeSizeAccountedByRecorded) {
  // With leak tracking, a caller that frees with the WRONG size must not
  // corrupt live_bytes — accounting uses the recorded alloc size.
  StatsAllocator sa =
      StatsAllocator::create(system_allocator(), /*track_leaks=*/true).value();
  auto a = sa.get();
  void* p = a.allocate(200);
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(sa.snapshot().live_bytes, 200u);
  a.deallocate(p, 999);  // deliberately wrong size
  EXPECT_TRUE(sa.balanced());
  EXPECT_EQ(sa.snapshot().live_bytes, 0u);
}

TEST(StatsAllocator, SystemAllocatorUnaffected) {
  // system path has no hidden stats — allocate without StatsAllocator is fine.
  auto a = system_allocator();
  void* p = a.allocate(8);
  ASSERT_NE(p, nullptr);
  a.deallocate(p, 8);
}
