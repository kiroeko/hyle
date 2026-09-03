#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory_resource>
#include <stdexcept>
#include <thread>
#include <vector>

#include "hyle/mem/allocator.h"
#include "system_internal.h"
#include "hyle/mem/pmr.h"
#include "hyle/mem/quota_allocator.h"
#include "hyle/mem/shared_pool.h"
#include "hyle/mem/stats.h"
#include "hyle/mem/system_inline.h"

#if defined(_ADDRESS_SANITIZER) || defined(__SANITIZE_ADDRESS__)
#include <sanitizer/asan_interface.h>
#define HYLE_MEM_TEST_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#include <sanitizer/asan_interface.h>
#define HYLE_MEM_TEST_ASAN 1
#endif
#endif

using hyle::mem::AllocatorRef;
using hyle::mem::kDefaultAlignment;
using hyle::mem::QuotaAllocator;
using hyle::mem::StatsAllocator;
using hyle::mem::system_allocator;
using hyle::mem::system_collect;
using hyle::mem::system_good_size;
using hyle::mem::system_memory_snapshot;
using hyle::mem::system_usable_size;
using hyle::mem::SystemAllocator;
#if defined(HYLE_TESTING)
using hyle::mem::system_backend_version_for_testing;
#endif

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
struct QuotaReallocFailState {
  bool fail_realloc = false;
  std::size_t outstanding = 0;
};
void *quota_fail_alloc(void *opaque, std::size_t size,
                       std::size_t alignment) noexcept {
  auto &state = *static_cast<QuotaReallocFailState *>(opaque);
  void *p = system_allocator().allocate(size, alignment);
  if (p != nullptr)
    ++state.outstanding;
  return p;
}
void quota_fail_free(void *opaque, void *ptr, std::size_t size,
                     std::size_t alignment) noexcept {
  if (ptr == nullptr)
    return;
  auto &state = *static_cast<QuotaReallocFailState *>(opaque);
  EXPECT_GT(state.outstanding, 0u);
  --state.outstanding;
  system_allocator().deallocate(ptr, size, alignment);
}
void *quota_fail_realloc(void *opaque, void *ptr, std::size_t old_size,
                         std::size_t new_size,
                         std::size_t alignment) noexcept {
  auto &state = *static_cast<QuotaReallocFailState *>(opaque);
  if (state.fail_realloc)
    return nullptr;
  return system_allocator().reallocate(ptr, old_size, new_size, alignment);
}
} // namespace

TEST(AllocatorRef, SystemAllocatorIsValid) {
  AllocatorRef a = system_allocator();
  EXPECT_TRUE(a.valid());
  EXPECT_NE(a.alloc_fn(), nullptr);
}

#if defined(HYLE_TESTING)
TEST(AllocatorRef, ShippingBackendIsReviewedMimallocVersion) {
  // mi_version uses major*10000 + minor*100 + patch: v3.5.1 == 30501.
  EXPECT_EQ(system_backend_version_for_testing(), 30501);
}
#endif

TEST(AllocatorRef, SystemCollectPreservesLiveAllocations) {
  AllocatorRef allocator = system_allocator();
  auto *live = allocator.allocate_n<std::uint64_t>(128);
  ASSERT_NE(live, nullptr);
  live[0] = 0x123456789abcdef0ULL;
  system_collect();
  EXPECT_EQ(live[0], 0x123456789abcdef0ULL);
  allocator.deallocate_n(live, 128);
  system_collect(true);
}

TEST(AllocatorRef, SystemMemorySnapshotIsStableAndSelfConsistent) {
  const auto before = system_memory_snapshot();
  ASSERT_TRUE(before);
  EXPECT_GE(before->allocator_peak_reserved_bytes,
            before->allocator_reserved_bytes);
  EXPECT_GE(before->allocator_peak_committed_bytes,
            before->allocator_committed_bytes);
  EXPECT_GE(before->process_peak_resident_bytes,
            before->process_resident_bytes);
  EXPECT_GE(before->process_peak_committed_bytes,
            before->process_committed_bytes);

  constexpr std::size_t kProbeBytes = 8u * 1024u * 1024u;
  AllocatorRef allocator = system_allocator();
  void *probe = allocator.allocate(kProbeBytes, 64);
  ASSERT_NE(probe, nullptr);
  std::memset(probe, 0xA7, kProbeBytes);

  const auto during = system_memory_snapshot();
  ASSERT_TRUE(during);
  EXPECT_GE(during->allocator_peak_reserved_bytes,
            during->allocator_reserved_bytes);
  EXPECT_GE(during->allocator_peak_committed_bytes,
            during->allocator_committed_bytes);
  EXPECT_GE(during->process_peak_resident_bytes,
            during->process_resident_bytes);
  EXPECT_GE(during->process_peak_committed_bytes,
            during->process_committed_bytes);
  EXPECT_GE(during->allocator_peak_reserved_bytes,
            before->allocator_peak_reserved_bytes);
  EXPECT_GE(during->allocator_peak_committed_bytes,
            before->allocator_peak_committed_bytes);
  EXPECT_GE(during->process_peak_resident_bytes,
            before->process_peak_resident_bytes);
  EXPECT_GE(during->process_peak_committed_bytes,
            before->process_peak_committed_bytes);

  allocator.deallocate(probe, kProbeBytes, 64);
}

TEST(AllocatorRef, CrossThreadSystemFreeThenOwnerCollectIsSafe) {
  struct RemoteBlock {
    void* pointer;
    std::size_t size;
    std::size_t alignment;
    unsigned char marker;
  };

  constexpr std::size_t kBlocks = 4096;
  AllocatorRef owner = system_allocator();
  std::vector<RemoteBlock> blocks;
  blocks.reserve(kBlocks);
  for (std::size_t i = 0; i < kBlocks; ++i) {
    const std::size_t alignment = (i & 1u) == 0 ? 16u : 64u;
    const std::size_t size = 24u + i % 128u;
    void* pointer = owner.allocate(size, alignment);
    ASSERT_NE(pointer, nullptr);
    const auto marker = static_cast<unsigned char>(i * 37u + 11u);
    std::memset(pointer, marker, size);
    blocks.push_back(RemoteBlock{pointer, size, alignment, marker});
  }

  std::atomic<bool> corrupted{false};
  std::thread remote_free([&] {
    AllocatorRef consumer = system_allocator();
    for (const RemoteBlock& block : blocks) {
      const auto* bytes = static_cast<const unsigned char*>(block.pointer);
      if (bytes[0] != block.marker || bytes[block.size - 1] != block.marker) {
        corrupted.store(true, std::memory_order_relaxed);
      }
      consumer.deallocate(block.pointer, block.size, block.alignment);
    }
  });
  remote_free.join();
  EXPECT_FALSE(corrupted.load(std::memory_order_relaxed));

  // v3 keeps remote frees in the allocating thread heap until that owner runs
  // maintenance. This is the contract path long-lived workers rely on.
  system_collect(true);
  void* probe = owner.allocate(4096, 64);
  ASSERT_NE(probe, nullptr);
  EXPECT_TRUE(is_aligned(probe, 64));
  owner.deallocate(probe, 4096, 64);
  system_collect(true);
}

TEST(AllocatorRef, DefaultAllocatorIsInvalid) {
  AllocatorRef a;
  EXPECT_FALSE(a.valid());
}

TEST(AllocatorRef, AllocateWriteFree) {
  AllocatorRef a = system_allocator();
  void *p = a.allocate(128);
  ASSERT_NE(p, nullptr);
  EXPECT_TRUE(is_aligned(p, kDefaultAlignment));
  std::memset(p, 0xAB, 128); // must be fully writable
  a.deallocate(p, 128);
}

TEST(SystemAllocatorAlignmentDeathTest, StaticBindingRejectsZeroAndNonPowerOfTwo) {
  EXPECT_DEATH((void)SystemAllocator::allocate(8, 0), ".*");
  EXPECT_DEATH((void)SystemAllocator::allocate(8, 3), ".*");
  EXPECT_DEATH((void)SystemAllocator::reallocate(nullptr, 0, 8, 6), ".*");
#if !defined(NDEBUG)
  EXPECT_DEATH(SystemAllocator::deallocate(nullptr, 0, 3), ".*");
#endif
}

TEST(SystemAllocatorAlignmentDeathTest, RuntimeHandleRejectsZeroAndNonPowerOfTwo) {
  const AllocatorRef allocator = system_allocator();
  EXPECT_DEATH((void)allocator.allocate(8, 0), ".*");
  EXPECT_DEATH((void)allocator.allocate(8, 3), ".*");
  EXPECT_DEATH((void)allocator.reallocate(nullptr, 0, 8, 6), ".*");
#if !defined(NDEBUG)
  EXPECT_DEATH(allocator.deallocate(nullptr, 0, 3), ".*");
#endif
}

#if defined(HYLE_MEM_TEST_ASAN)
TEST(AllocatorAsan, SystemFreePoisonsUsableStorage) {
  AllocatorRef a = system_allocator();
  void *p = a.allocate(64);
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(__asan_address_is_poisoned(p), 0);
  a.deallocate(p, 64);
  EXPECT_NE(__asan_address_is_poisoned(p), 0);
}
#endif

TEST(AllocatorRef, AllocateZeroReturnsNull) {
  AllocatorRef a = system_allocator();
  EXPECT_EQ(a.allocate(0), nullptr); // new_size==0 collapses to the free path
}

// The headline of mem-01: alignment is honored end-to-end (clay dropped it).
TEST(AllocatorRef, RespectsExplicitAlignment) {
  AllocatorRef a = system_allocator();
  for (std::size_t align :
       {std::size_t{16}, std::size_t{32}, std::size_t{64}, std::size_t{128},
        std::size_t{256}, std::size_t{4096}}) {
    void *p = a.allocate(align * 4, align);
    ASSERT_NE(p, nullptr) << "align=" << align;
    EXPECT_TRUE(is_aligned(p, align)) << "align=" << align;
    a.deallocate(p, align * 4, align);
  }
}

TEST(AllocatorRef, SmallAlignmentBumpedToDefault) {
  AllocatorRef a = system_allocator();
  for (std::size_t size : {std::size_t{1}, std::size_t{8}, std::size_t{37}}) {
    for (std::size_t alignment :
         {std::size_t{1}, std::size_t{2}, std::size_t{4}, std::size_t{8},
          kDefaultAlignment}) {
      void *p = a.allocate(size, alignment);
      ASSERT_NE(p, nullptr) << "size=" << size << " align=" << alignment;
      EXPECT_TRUE(is_aligned(p, kDefaultAlignment))
          << "size=" << size << " align=" << alignment;
      void *q = a.reallocate(p, size, size + 13, alignment);
      ASSERT_NE(q, nullptr) << "size=" << size << " align=" << alignment;
      EXPECT_TRUE(is_aligned(q, kDefaultAlignment))
          << "size=" << size << " align=" << alignment;
      a.deallocate(q, size + 13, alignment);
    }
  }
}

TEST(AllocatorRef, CompileTimeSmallAlignmentBumpedToDefault) {
  for (std::size_t size : {std::size_t{1}, std::size_t{8}, std::size_t{37}}) {
    void *p = SystemAllocator::allocate(size, 1);
    ASSERT_NE(p, nullptr) << "size=" << size;
    EXPECT_TRUE(is_aligned(p, kDefaultAlignment)) << "size=" << size;
    void *q = SystemAllocator::reallocate(p, size, size + 13, 1);
    ASSERT_NE(q, nullptr) << "size=" << size;
    EXPECT_TRUE(is_aligned(q, kDefaultAlignment)) << "size=" << size;
    SystemAllocator::deallocate(q, size + 13, 1);
  }
}

TEST(AllocatorRef, ReallocatePreservesData) {
  AllocatorRef a = system_allocator();
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

TEST(AllocatorRef, ReallocateFromNullIsAllocate) {
  AllocatorRef a = system_allocator();
  void *p = a.reallocate(nullptr, 0, 256, 64);
  ASSERT_NE(p, nullptr);
  EXPECT_TRUE(is_aligned(p, 64));
  a.deallocate(p, 256, 64);
}

TEST(AllocatorRef, CompileTimeReallocateHandlesNullAndPreservesAlignment) {
  void *p = SystemAllocator::reallocate(nullptr, 0, 128);
  ASSERT_NE(p, nullptr);
  std::memset(p, 0x5A, 128);
  p = SystemAllocator::reallocate(p, 128, 256);
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(static_cast<unsigned char *>(p)[0], 0x5A);
  SystemAllocator::deallocate(p, 256);

  void *aligned = SystemAllocator::reallocate(nullptr, 0, 128, 64);
  ASSERT_NE(aligned, nullptr);
  EXPECT_TRUE(is_aligned(aligned, 64));
  aligned = SystemAllocator::reallocate(aligned, 128, 256, 64);
  ASSERT_NE(aligned, nullptr);
  EXPECT_TRUE(is_aligned(aligned, 64));
  SystemAllocator::deallocate(aligned, 256, 64);

  void* released = SystemAllocator::allocate(8, 1);
  ASSERT_NE(released, nullptr);
  EXPECT_EQ(SystemAllocator::reallocate(released, 8, 0, 1), nullptr);
}

TEST(AllocatorRef, ManyDistinctNonOverlapping) {
  AllocatorRef a = system_allocator();
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
  AllocatorRef a = system_allocator();
  int *p = a.allocate_n<int>(5);
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % alignof(int), 0u);
  for (int i = 0; i < 5; ++i)
    p[i] = i; // writable
  EXPECT_EQ(p[4], 4);
  a.deallocate_n(p, 5);
}

TEST(AllocatorTyped, AllocateZeroedZeroes) {
  AllocatorRef a = system_allocator();
  int *p = a.allocate_zeroed<int>(8);
  ASSERT_NE(p, nullptr);
  for (int i = 0; i < 8; ++i)
    EXPECT_EQ(p[i], 0);
  a.deallocate_n(p, 8);
}

TEST(AllocatorTyped, ArrayResizePreservesExplicitAlignmentAndContents) {
  AllocatorRef a = system_allocator();
  int *p = a.allocate_zeroed<int>(8, 64);
  ASSERT_NE(p, nullptr);
  EXPECT_TRUE(is_aligned(p, 64));
  p[0] = 17;
  p[7] = 29;
  p = a.reallocate_n(p, 8, 32, 64);
  ASSERT_NE(p, nullptr);
  EXPECT_TRUE(is_aligned(p, 64));
  EXPECT_EQ(p[0], 17);
  EXPECT_EQ(p[7], 29);
  a.deallocate_n(p, 32, 64);
}

TEST(AllocatorTyped, CountOverflowReturnsNull) {
  AllocatorRef a = system_allocator();
  constexpr std::size_t too_many = SIZE_MAX / sizeof(std::uint64_t) + 2;
  EXPECT_EQ(a.allocate_n<std::uint64_t>(too_many), nullptr);
  EXPECT_EQ(a.allocate_zeroed<std::uint64_t>(too_many), nullptr);
  EXPECT_EQ(a.reallocate_n<std::uint64_t>(nullptr, 0, too_many), nullptr);
}

TEST(SystemAllocatorTyped, CountOverflowReturnsNull) {
  constexpr std::size_t too_many = SIZE_MAX / sizeof(std::uint64_t) + 2;
  EXPECT_EQ(SystemAllocator::allocate_n<std::uint64_t>(too_many), nullptr);
  EXPECT_EQ(SystemAllocator::allocate_zeroed<std::uint64_t>(too_many), nullptr);
  EXPECT_EQ(SystemAllocator::reallocate_n<std::uint64_t>(nullptr, 0, too_many),
            nullptr);
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
  StatsAllocator tracked =
      StatsAllocator::create(system_allocator()).value();
  AllocatorRef a = tracked.get();
  EXPECT_THROW((void)a.construct<ThrowsInCtor>(), std::runtime_error);
  EXPECT_TRUE(tracked.balanced());
  EXPECT_EQ(tracked.snapshot().total_allocs, 1u);
  EXPECT_EQ(tracked.snapshot().total_frees, 1u);
}

TEST(AllocatorTyped, ConstructRunsCtorDestroyRunsDtor) {
  g_typed_live.store(0);
  AllocatorRef a = system_allocator();
  TCounted *c = a.construct<TCounted>(7);
  ASSERT_NE(c, nullptr);
  EXPECT_EQ(c->v, 7);
  EXPECT_EQ(g_typed_live.load(), 1);
  a.destroy(c); // ~T + free through THIS allocator (not global delete)
  EXPECT_EQ(g_typed_live.load(), 0);
}

TEST(AllocatorTyped, MakeUniqueAutoDestroys) {
  g_typed_live.store(0);
  AllocatorRef a = system_allocator();
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
  AllocatorRef a = quota.get();
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
  AllocatorRef a = quota.get();
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
  AllocatorRef oom(&quota_oom_alloc, &quota_oom_free, &quota_oom_realloc, nullptr);
  QuotaAllocator quota(oom, 1024);
  EXPECT_EQ(quota.get().allocate(100), nullptr);
  EXPECT_EQ(quota.used(), 0u);
  EXPECT_EQ(quota.peak(), 0u); // failed backing allocation never becomes live
}

TEST(QuotaAllocator, BackingReallocateFailurePreservesPointerAndAccounting) {
  QuotaReallocFailState state;
  AllocatorRef failing(&quota_fail_alloc, &quota_fail_free, &quota_fail_realloc,
                    &state);
  QuotaAllocator quota(failing, 1024);
  AllocatorRef alloc = quota.get();
  auto *p = static_cast<unsigned char *>(alloc.allocate(64, 16));
  ASSERT_NE(p, nullptr);
  p[0] = 0x5a;

  state.fail_realloc = true;
  EXPECT_EQ(alloc.reallocate(p, 64, 96, 16), nullptr);
  EXPECT_EQ(p[0], 0x5a);
  EXPECT_EQ(quota.used(), 64u);
  EXPECT_EQ(quota.peak(), 64u);
  EXPECT_EQ(alloc.reallocate(p, 64, 32, 16), nullptr);
  EXPECT_EQ(p[0], 0x5a);
  EXPECT_EQ(quota.used(), 64u);

  alloc.deallocate(p, 64, 16);
  EXPECT_EQ(quota.used(), 0u);
  EXPECT_EQ(state.outstanding, 0u);
}

TEST(QuotaAllocator, ConcurrentAdmissionNeverExceedsHardLimit) {
  constexpr std::size_t kLimit = 512;
  constexpr std::size_t kRequest = 32;
  QuotaAllocator quota(system_allocator(), kLimit);
  AllocatorRef alloc = quota.get();
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

TEST(AllocatorComposition,
     StatsOverQuotaOverConcurrentBlockPoolKeepsOneTruth) {
  hyle::mem::ConcurrentBlockPool pool(system_allocator(), 128, 1, 64);
  QuotaAllocator quota(pool.as_allocator(), 512);
  StatsAllocator stats =
      StatsAllocator::create(quota.get(), /*track_leaks=*/true).value();
  AllocatorRef alloc = stats.get();

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

TEST(AllocatorComposition,
     QuotaOverStatsOverConcurrentBlockPoolKeepsOneTruth) {
  hyle::mem::ConcurrentBlockPool pool(system_allocator(), 128, 1, 64);
  StatsAllocator stats = StatsAllocator::create(
                             pool.as_allocator(), /*track_leaks=*/true)
                             .value();
  QuotaAllocator quota(stats.get(), 512);
  AllocatorRef alloc = quota.get();

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

TEST(AllocatorComposition,
     PmrOverDecoratedConcurrentBlockPoolReturnsCapacity) {
  hyle::mem::ConcurrentBlockPool pool(system_allocator(), 128, 1, 64);
  QuotaAllocator quota(pool.as_allocator(), 128);
  StatsAllocator stats = StatsAllocator::create(quota.get()).value();
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
