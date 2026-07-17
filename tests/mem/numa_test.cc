#include <gtest/gtest.h>

#include <cstring>

#include "hyle/mem/allocator.h"
#include "hyle/mem/hugepage.h"
#include "hyle/mem/memory_arena.h"
#include "hyle/platform/numa.h"
#include "hyle/platform/virtual_memory.h"

using hyle::mem::enable_allocator_numa;
using hyle::mem::HugeFlags;
using hyle::mem::HugeRegion;
using hyle::mem::MemoryArena;
using hyle::mem::reserve_huge_1g_pages;
using hyle::mem::system_allocator;
using hyle::platform::kNumaAny;
using hyle::platform::numa_current_node;
using hyle::platform::numa_node_count;
using hyle::platform::page_size;

TEST(MemNuma, HugeRegionOnNodeZero) {
  HugeRegion r = HugeRegion::allocate(page_size() * 8, HugeFlags::kThpHint, 0);
  ASSERT_TRUE(r.valid());
  std::memset(r.data(), 0xAB, 128);
  EXPECT_EQ(static_cast<unsigned char*>(r.data())[0], 0xAB);
}

TEST(MemNuma, HugeRegionLocal) {
  HugeRegion r =
      HugeRegion::allocate_local(page_size() * 8, HugeFlags::kThpHint);
  ASSERT_TRUE(r.valid());
  EXPECT_GE(r.size(), page_size() * 8);
  std::memset(r.data(), 0xCD, 64);
  EXPECT_EQ(static_cast<unsigned char*>(r.data())[0], 0xCD);
}

TEST(MemNuma, HugeRegionAnyNode) {
  HugeRegion r =
      HugeRegion::allocate(page_size() * 4, HugeFlags::kNone, kNumaAny);
  ASSERT_TRUE(r.valid());
}

TEST(MemNuma, EnableAllocatorNuma) {
  enable_allocator_numa(0);  // all nodes
  enable_allocator_numa(1);  // clamp to 1
  enable_allocator_numa(0);
  // Topology still sane after policy tweaks.
  EXPECT_GE(numa_node_count(), 1u);
  EXPECT_LT(numa_current_node(), numa_node_count());
}

TEST(MemNuma, HugeRegionInterleaved) {
  HugeRegion r = HugeRegion::allocate_interleaved(page_size() * 16);
  ASSERT_TRUE(r.valid());
  std::memset(r.data(), 0x3C, 64);
  EXPECT_EQ(static_cast<unsigned char*>(r.data())[0], 0x3C);
}

TEST(MemNuma, HugeRegionHardBindFlag) {
  HugeRegion r = HugeRegion::allocate(page_size() * 8,
                                      HugeFlags::kThpHint | HugeFlags::kNumaBind, 0);
  ASSERT_TRUE(r.valid());
}

TEST(MemNuma, ReserveHuge1gSoft) {
  // Almost always false without admin/hugetlb — must not crash or hang long.
  (void)reserve_huge_1g_pages(0);  // no-op true
  EXPECT_TRUE(reserve_huge_1g_pages(0));
  (void)reserve_huge_1g_pages(1, kNumaAny, /*timeout_ms=*/50);
  (void)reserve_huge_1g_pages(1, 0, 50);
}

TEST(MemNuma, LargeMemoryArenaUsesHugePath) {
  // Slab >= ~2MiB → HugeRegion path inside MemoryArena.
  constexpr std::size_t kBlock = 64 * 1024;
  constexpr std::uint32_t kCount = 64;  // 4 MiB slab
  MemoryArena arena(system_allocator(), kBlock, kCount);
  void* a = arena.allocate();
  void* b = arena.allocate();
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  EXPECT_NE(a, b);
  std::memset(a, 0x11, kBlock);
  arena.deallocate(a);
  arena.deallocate(b);
  // dtor frees huge or allocator slab cleanly
}
