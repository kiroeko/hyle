#include <gtest/gtest.h>

#include <array>
#include <climits>
#include <cstring>
#include <utility>

#include "hyle/mem/hugepage.h"
#include "hyle/platform/numa.h"
#include "hyle/platform/virtual_memory.h"

using hyle::mem::advise_hugepage_range;
using hyle::mem::HugeFlags;
using hyle::mem::hugepage_region_threshold;
using hyle::mem::HugeMapping;
using hyle::platform::large_page_align_up;
using hyle::platform::large_page_size;
using hyle::platform::page_size;

TEST(HugeMapping, AllocateDefaultIsWritableAndAligned) {
  const std::size_t want = hugepage_region_threshold(); // ~2 MiB
  HugeMapping r = HugeMapping::allocate(want, HugeFlags::kThpHint).value();
  ASSERT_TRUE(r.valid());
  EXPECT_GE(r.size(), want);
  EXPECT_EQ(r.size(), large_page_align_up(want));
  // Fully writable.
  std::memset(r.data(), 0xA5, r.size());
  EXPECT_EQ(static_cast<unsigned char *>(r.data())[0], 0xA5);
  EXPECT_EQ(static_cast<unsigned char *>(r.data())[r.size() - 1], 0xA5);
}

TEST(HugeMapping, ReportsRequestedAndAcceptedPolicyWithoutOverclaiming) {
  const auto requested = HugeFlags::kThpHint | HugeFlags::kPrefault;
  HugeMapping r =
      HugeMapping::allocate(2u * 1024u * 1024u, requested).value();
  ASSERT_TRUE(r.valid());
  EXPECT_EQ(r.requested_flags(), requested);
  EXPECT_TRUE(any(r.accepted_flags() & HugeFlags::kPrefault));
  // THP advice is best-effort and platform-dependent; accepted_flags reports
  // the call result.
  EXPECT_EQ(r.requested_numa_node(), hyle::platform::kNumaAny);
}

TEST(HugeMapping, PrefaultTouchesPages) {
  // Smaller than 2 MiB still works (aligned up). Prefault is optional path.
  HugeMapping r = HugeMapping::allocate(
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

TEST(HugeMapping, MoveAndReset) {
  HugeMapping a = HugeMapping::allocate(page_size() * 4).value();
  ASSERT_TRUE(a.valid());
  void *base = a.data();
  HugeMapping b = std::move(a);
  EXPECT_FALSE(a.valid());
  EXPECT_TRUE(b.valid());
  EXPECT_EQ(b.data(), base);
  EXPECT_TRUE(b.reset().ok());
  EXPECT_FALSE(b.valid());
}

TEST(HugeMapping, ZeroBytesInvalid) {
  auto r = HugeMapping::allocate(0);
  ASSERT_FALSE(r);
  EXPECT_TRUE(r.error().IsInvalidArgument());
}

TEST(HugeMapping, RejectsUnknownAndContradictoryPlacementFlags) {
  auto unknown = HugeMapping::allocate(
      page_size(), static_cast<HugeFlags>(std::uint8_t{1} << 7));
  ASSERT_FALSE(unknown);
  EXPECT_TRUE(unknown.error().IsInvalidArgument());

  auto conflict = HugeMapping::allocate(
      page_size(), HugeFlags::kNumaBind | HugeFlags::kNumaInterleave);
  ASSERT_FALSE(conflict);
  EXPECT_TRUE(conflict.error().IsInvalidArgument());

  auto bind_without_node =
      HugeMapping::allocate(page_size(), HugeFlags::kNumaBind);
  ASSERT_FALSE(bind_without_node);
  EXPECT_TRUE(bind_without_node.error().IsInvalidArgument());

  auto named_node_interleave = HugeMapping::allocate(
      page_size(), HugeFlags::kNumaInterleave,
      static_cast<int>(hyle::platform::numa_nodes().front()));
  ASSERT_FALSE(named_node_interleave);
  EXPECT_TRUE(named_node_interleave.error().IsInvalidArgument());

  auto local_interleave =
      HugeMapping::allocate_local(page_size(), HugeFlags::kNumaInterleave);
  ASSERT_FALSE(local_interleave);
  EXPECT_TRUE(local_interleave.error().IsInvalidArgument());
}

TEST(HugeMapping, HardBindRejectsInvalidNode) {
  const unsigned highest = hyle::platform::numa_nodes().back();
  ASSERT_LE(highest, static_cast<unsigned>(INT_MAX - 100));
  const int invalid = static_cast<int>(highest) + 100;
  ASSERT_NE(invalid, hyle::platform::kNumaAny);
  auto r = HugeMapping::allocate(page_size(), HugeFlags::kNumaBind, invalid);
  ASSERT_FALSE(r);
  EXPECT_TRUE(r.error().IsOutOfRange());
}

TEST(HugeMapping, SelectedNodesAndChunkAreWritableAndObservable) {
  const std::size_t p = page_size();
  const std::array<unsigned, 1> selected{
      hyle::platform::numa_nodes().front()};
  auto result = HugeMapping::allocate_interleaved(
      p * 8, selected, p * 2, HugeFlags::kPrefault);
  if (hyle::platform::numa_nodes().size() > 1u &&
      !hyle::platform::has_numa_capability(
          hyle::platform::numa_capabilities(),
          hyle::platform::NumaCapability::kSelectedInterleave)) {
    ASSERT_FALSE(result);
    EXPECT_TRUE(result.error().IsUnimplemented());
    return;
  }
  ASSERT_TRUE(result);
  HugeMapping region = *std::move(result);
  EXPECT_EQ(region.interleave_chunk_size(), p * 2);
  EXPECT_TRUE(any(region.requested_flags() & HugeFlags::kNumaInterleave));
  EXPECT_TRUE(any(region.accepted_flags() & HugeFlags::kNumaInterleave));
  EXPECT_TRUE(any(region.accepted_flags() & HugeFlags::kPrefault));
  std::memset(region.data(), 0x4A, region.size());
  EXPECT_EQ(static_cast<unsigned char*>(region.data())[region.size() - 1],
            0x4A);
}

TEST(HugeMapping, SelectedNodesAcceptWeightedPlacement) {
  const unsigned node = hyle::platform::numa_nodes().front();
  const std::array<unsigned, 2> weighted{node, node};
  auto result = HugeMapping::allocate_interleaved(
      page_size(), weighted, page_size(), HugeFlags::kNone);
  ASSERT_TRUE(result);
  EXPECT_TRUE(result->valid());
  EXPECT_EQ(result->interleave_chunk_size(), page_size());
}

TEST(HugeMapping, AdviseHugepageRangeHelper) {
  HugeMapping r =
      HugeMapping::allocate(page_size() * 4, HugeFlags::kNone).value();
  ASSERT_TRUE(r.valid());
  // Soft — may be false off Linux.
  (void)advise_hugepage_range(r.data(), r.size());
  (void)r.advise_hugepage();
  std::memset(r.data(), 0, 64);
}

TEST(HugepagePolicy, ThresholdMatchesLargePageWhenKnown) {
  std::size_t t = hugepage_region_threshold();
  EXPECT_GE(t, page_size());
  if (large_page_size() != 0) {
    EXPECT_EQ(t, large_page_size());
  }
}
