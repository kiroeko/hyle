#include <gtest/gtest.h>

#include <cstring>
#include <limits>
#include <string_view>
#include <utility>

#include "hyle/mem/hugepage.h"
#include "hyle/platform.h"
#include "hyle/platform/numa.h"
#include "hyle/platform/virtual_memory.h"

using hyle::mem::enable_allocator_numa;
using hyle::mem::HugeFlags;
using hyle::mem::HugeRegion;
using hyle::mem::reserve_huge_1g_pages;
using hyle::Status;
using hyle::platform::kNumaAny;
using hyle::platform::numa_current_node;
using hyle::platform::numa_node_available;
using hyle::platform::numa_node_count;
using hyle::platform::numa_nodes;
using hyle::platform::page_size;

TEST(MemNuma, HugeRegionOnAvailableNode) {
  HugeRegion r = HugeRegion::allocate(page_size() * 8, HugeFlags::kThpHint,
                                      static_cast<int>(numa_nodes().front()))
                     .value();
  ASSERT_TRUE(r.valid());
  std::memset(r.data(), 0xAB, 128);
  EXPECT_EQ(static_cast<unsigned char*>(r.data())[0], 0xAB);
}

TEST(MemNuma, HugeRegionLocal) {
  HugeRegion r = HugeRegion::allocate_local(page_size() * 8,
                                             HugeFlags::kThpHint)
                     .value();
  ASSERT_TRUE(r.valid());
  EXPECT_GE(r.size(), page_size() * 8);
  std::memset(r.data(), 0xCD, 64);
  EXPECT_EQ(static_cast<unsigned char*>(r.data())[0], 0xCD);
}

TEST(MemNuma, HugeRegionAnyNode) {
  HugeRegion r = HugeRegion::allocate(page_size() * 4, HugeFlags::kNone,
                                      kNumaAny)
                     .value();
  ASSERT_TRUE(r.valid());
}

TEST(MemNuma, EnableAllocatorNuma) {
  enable_allocator_numa(0);  // all nodes
  enable_allocator_numa(1);  // clamp to 1
  enable_allocator_numa(0);
  // Topology still sane after policy tweaks.
  EXPECT_GE(numa_node_count(), 1u);
  EXPECT_TRUE(numa_node_available(numa_current_node()));
}

TEST(MemNuma, HugeRegionInterleaved) {
  HugeRegion r =
      HugeRegion::allocate_interleaved(page_size() * 16).value();
  ASSERT_TRUE(r.valid());
  std::memset(r.data(), 0x3C, 64);
  EXPECT_EQ(static_cast<unsigned char*>(r.data())[0], 0x3C);
}

TEST(MemNuma, HugeRegionHardBindFlag) {
  auto result =
      HugeRegion::allocate(page_size() * 8,
                           HugeFlags::kThpHint | HugeFlags::kNumaBind,
                           static_cast<int>(numa_nodes().front()));
#if defined(HYLE_OS_WINDOWS)
  if (numa_nodes().size() > 1u) {
    EXPECT_FALSE(result);
    EXPECT_TRUE(result.error().IsUnimplemented());
    return;
  }
#endif
  HugeRegion r = *std::move(result);
  ASSERT_TRUE(r.valid());
}

TEST(MemNuma, ReserveHuge1gSoft) {
  // Usually ResourceExhausted without admin/hugetlb; must not crash or hang.
  EXPECT_TRUE(reserve_huge_1g_pages(0).ok());
  reserve_huge_1g_pages(1, kNumaAny, /*timeout_ms=*/50).IgnoreError();
  reserve_huge_1g_pages(1, static_cast<int>(numa_nodes().front()), 50)
      .IgnoreError();

  Status invalid_node = reserve_huge_1g_pages(1, -2, 1);
  EXPECT_TRUE(invalid_node.IsOutOfRange());
  EXPECT_NE(invalid_node.message().find("NUMA node"), std::string_view::npos);

  Status overflow = reserve_huge_1g_pages(
      (std::numeric_limits<std::size_t>::max)() / (std::size_t{1} << 30) + 1,
      kNumaAny, 1);
  EXPECT_TRUE(overflow.IsOutOfRange());
}
