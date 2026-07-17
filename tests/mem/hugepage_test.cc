#include <gtest/gtest.h>

#include <cstring>

#include "hyle/mem/hugepage.h"
#include "hyle/platform/virtual_memory.h"

using hyle::mem::advise_hugepage_range;
using hyle::mem::AllocatorHugepagePolicy;
using hyle::mem::enable_allocator_hugepages;
using hyle::mem::HugeFlags;
using hyle::mem::HugeRegion;
using hyle::mem::hugepage_region_threshold;
using hyle::platform::large_page_align_up;
using hyle::platform::large_page_size;
using hyle::platform::page_size;

TEST(HugeRegion, AllocateDefaultIsWritableAndAligned) {
  const std::size_t want = hugepage_region_threshold();  // ~2 MiB
  HugeRegion r = HugeRegion::allocate(want, HugeFlags::kThpHint);
  ASSERT_TRUE(r.valid());
  EXPECT_GE(r.size(), want);
  EXPECT_EQ(r.size(), large_page_align_up(want));
  // Fully writable.
  std::memset(r.data(), 0xA5, r.size());
  EXPECT_EQ(static_cast<unsigned char*>(r.data())[0], 0xA5);
  EXPECT_EQ(static_cast<unsigned char*>(r.data())[r.size() - 1], 0xA5);
}

TEST(HugeRegion, PrefaultTouchesPages) {
  // Smaller than 2 MiB still works (aligned up). Prefault is optional path.
  HugeRegion r =
      HugeRegion::allocate(page_size() * 8, HugeFlags::kThpHint | HugeFlags::kPrefault);
  ASSERT_TRUE(r.valid());
  auto* p = static_cast<unsigned char*>(r.data());
  p[0] = 1;
  p[r.size() - 1] = 2;
  EXPECT_EQ(p[0], 1);
  EXPECT_EQ(p[r.size() - 1], 2);
}

TEST(HugeRegion, MoveAndReset) {
  HugeRegion a = HugeRegion::allocate(page_size() * 4);
  ASSERT_TRUE(a.valid());
  void* base = a.data();
  HugeRegion b = std::move(a);
  EXPECT_FALSE(a.valid());
  EXPECT_TRUE(b.valid());
  EXPECT_EQ(b.data(), base);
  b.reset();
  EXPECT_FALSE(b.valid());
}

TEST(HugeRegion, ZeroBytesInvalid) {
  HugeRegion r = HugeRegion::allocate(0);
  EXPECT_FALSE(r.valid());
}

TEST(HugeRegion, AdviseHugepageRangeHelper) {
  HugeRegion r = HugeRegion::allocate(page_size() * 4, HugeFlags::kNone);
  ASSERT_TRUE(r.valid());
  // Soft — may be false off Linux.
  (void)advise_hugepage_range(r.data(), r.size());
  (void)r.advise_hugepage();
  std::memset(r.data(), 0, 64);
}

TEST(HugepagePolicy, EnableDoesNotCrash) {
  AllocatorHugepagePolicy pol;
  pol.allow_thp = true;
  pol.allow_large_os_pages = false;  // no privilege required
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
