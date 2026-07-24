#include <gtest/gtest.h>

#include <climits>
#include <cstring>

#include "hyle/mem/hugepage.h"
#include "hyle/platform/numa.h"
#include "hyle/platform/virtual_memory.h"

using hyle::mem::advise_hugepage_range;
using hyle::mem::AllocatorHugepagePolicy;
using hyle::mem::enable_allocator_hugepages;
using hyle::mem::HugeFlags;
using hyle::mem::hugepage_region_threshold;
using hyle::mem::HugeRegion;
using hyle::platform::large_page_align_up;
using hyle::platform::large_page_size;
using hyle::platform::page_size;

TEST(HugeRegion, AllocateDefaultIsWritableAndAligned) {
  const std::size_t want = hugepage_region_threshold(); // ~2 MiB
  HugeRegion r = HugeRegion::allocate(want, HugeFlags::kThpHint).value();
  ASSERT_TRUE(r.valid());
  EXPECT_GE(r.size(), want);
  EXPECT_EQ(r.size(), large_page_align_up(want));
  // Fully writable.
  std::memset(r.data(), 0xA5, r.size());
  EXPECT_EQ(static_cast<unsigned char *>(r.data())[0], 0xA5);
  EXPECT_EQ(static_cast<unsigned char *>(r.data())[r.size() - 1], 0xA5);
}

TEST(HugeRegion, ReportsRequestedAndAcceptedPolicyWithoutOverclaiming) {
  const auto requested = HugeFlags::kThpHint | HugeFlags::kPrefault;
  HugeRegion r =
      HugeRegion::allocate(2u * 1024u * 1024u, requested).value();
  ASSERT_TRUE(r.valid());
  EXPECT_EQ(r.requested_flags(), requested);
  EXPECT_TRUE(any(r.accepted_flags() & HugeFlags::kPrefault));
  // THP advice is best-effort and platform-dependent; accepted_flags reports
  // the call result.
  EXPECT_EQ(r.requested_numa_node(), hyle::platform::kNumaAny);
}

TEST(HugeRegion, PrefaultTouchesPages) {
  // Smaller than 2 MiB still works (aligned up). Prefault is optional path.
  HugeRegion r = HugeRegion::allocate(
                     page_size() * 8,
                     HugeFlags::kThpHint | HugeFlags::kPrefault)
                     .value();
  ASSERT_TRUE(r.valid());
  auto *p = static_cast<unsigned char *>(r.data());
  p[0] = 1;
  p[r.size() - 1] = 2;
  EXPECT_EQ(p[0], 1);
  EXPECT_EQ(p[r.size() - 1], 2);
}

TEST(HugeRegion, MoveAndReset) {
  HugeRegion a = HugeRegion::allocate(page_size() * 4).value();
  ASSERT_TRUE(a.valid());
  void *base = a.data();
  HugeRegion b = std::move(a);
  EXPECT_FALSE(a.valid());
  EXPECT_TRUE(b.valid());
  EXPECT_EQ(b.data(), base);
  EXPECT_TRUE(b.reset().ok());
  EXPECT_FALSE(b.valid());
}

TEST(HugeRegion, ZeroBytesInvalid) {
  auto r = HugeRegion::allocate(0);
  ASSERT_FALSE(r);
  EXPECT_TRUE(r.error().IsInvalidArgument());
}

TEST(HugeRegion, RejectsUnknownAndContradictoryPlacementFlags) {
  auto unknown = HugeRegion::allocate(
      page_size(), static_cast<HugeFlags>(std::uint8_t{1} << 7));
  ASSERT_FALSE(unknown);
  EXPECT_TRUE(unknown.error().IsInvalidArgument());

  auto conflict = HugeRegion::allocate(
      page_size(), HugeFlags::kNumaBind | HugeFlags::kNumaInterleave);
  ASSERT_FALSE(conflict);
  EXPECT_TRUE(conflict.error().IsInvalidArgument());

  auto named_node_interleave = HugeRegion::allocate(
      page_size(), HugeFlags::kNumaInterleave, 0);
  ASSERT_FALSE(named_node_interleave);
  EXPECT_TRUE(named_node_interleave.error().IsInvalidArgument());

  auto local_interleave =
      HugeRegion::allocate_local(page_size(), HugeFlags::kNumaInterleave);
  ASSERT_FALSE(local_interleave);
  EXPECT_TRUE(local_interleave.error().IsInvalidArgument());
}

TEST(HugeRegion, HardBindRejectsInvalidNode) {
  const unsigned count = hyle::platform::numa_node_count();
  ASSERT_LE(count, static_cast<unsigned>(INT_MAX - 100));
  const int invalid = static_cast<int>(count) + 100;
  ASSERT_NE(invalid, hyle::platform::kNumaAny);
  ASSERT_GE(static_cast<unsigned>(invalid), count);
  auto r = HugeRegion::allocate(page_size(), HugeFlags::kNumaBind, invalid);
  ASSERT_FALSE(r);
  EXPECT_TRUE(r.error().IsOutOfRange());
}

TEST(HugeRegion, AdviseHugepageRangeHelper) {
  HugeRegion r =
      HugeRegion::allocate(page_size() * 4, HugeFlags::kNone).value();
  ASSERT_TRUE(r.valid());
  // Soft — may be false off Linux.
  (void)advise_hugepage_range(r.data(), r.size());
  (void)r.advise_hugepage();
  std::memset(r.data(), 0, 64);
}

TEST(HugepagePolicy, EnableDoesNotCrash) {
  AllocatorHugepagePolicy pol;
  pol.allow_thp = true;
  pol.allow_large_os_pages = false; // no privilege required
  enable_allocator_hugepages(pol);
  // Second call with large pages requested (may no-op without privileges).
  pol.allow_large_os_pages = true;
  enable_allocator_hugepages(pol);
  pol.allow_thp = false;
  pol.allow_large_os_pages = false;
  enable_allocator_hugepages(pol);
}

TEST(HugepagePolicy, ThresholdMatchesLargePageWhenKnown) {
  std::size_t t = hugepage_region_threshold();
  EXPECT_GE(t, page_size());
  if (large_page_size() != 0) {
    EXPECT_EQ(t, large_page_size());
  }
}
