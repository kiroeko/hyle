#include <gtest/gtest.h>

#include <array>
#include <climits>
#include <cstring>
#include <utility>

#include "hyle/mem/large_mapping.h"
#include "hyle/platform/numa.h"
#include "hyle/platform/virtual_memory.h"

using hyle::mem::LargeMappingFlags;
using hyle::mem::LargeMapping;
using hyle::platform::large_page_align_up;
using hyle::platform::large_page_size;
using hyle::platform::page_size;

TEST(LargeMapping, AllocateDefaultIsWritableAndAligned) {
  const std::size_t want = page_size() * 4;
  LargeMapping r = LargeMapping::allocate(want, LargeMappingFlags::kThpHint).value();
  ASSERT_TRUE(r.valid());
  EXPECT_GE(r.size(), want);
  EXPECT_EQ(r.size(), large_page_align_up(want));
  // Fully writable.
  std::memset(r.data(), 0xA5, r.size());
  EXPECT_EQ(static_cast<unsigned char *>(r.data())[0], 0xA5);
  EXPECT_EQ(static_cast<unsigned char *>(r.data())[r.size() - 1], 0xA5);
}

TEST(LargeMapping, ReportsRequestedAndAcceptedPolicyWithoutOverclaiming) {
  const auto requested = LargeMappingFlags::kThpHint | LargeMappingFlags::kPrefault;
  LargeMapping r =
      LargeMapping::allocate(2u * 1024u * 1024u, requested).value();
  ASSERT_TRUE(r.valid());
  EXPECT_EQ(r.requested_flags(), requested);
  EXPECT_TRUE(any(r.accepted_flags() & LargeMappingFlags::kPrefault));
  // THP advice is best-effort and platform-dependent; accepted_flags reports
  // the call result.
  EXPECT_EQ(r.requested_numa_node(), hyle::platform::kNumaAny);
}

TEST(LargeMapping, PrefaultTouchesPages) {
  // Smaller than 2 MiB still works (aligned up). Prefault is optional path.
  LargeMapping r = LargeMapping::allocate(
                     page_size() * 8,
                     LargeMappingFlags::kThpHint | LargeMappingFlags::kPrefault)
                     .value();
  ASSERT_TRUE(r.valid());
  auto *p = static_cast<unsigned char *>(r.data());
  p[0] = 1;
  p[r.size() - 1] = 2;
  EXPECT_EQ(p[0], 1);
  EXPECT_EQ(p[r.size() - 1], 2);
}

TEST(LargeMapping, MoveAndReset) {
  LargeMapping a = LargeMapping::allocate(page_size() * 4).value();
  ASSERT_TRUE(a.valid());
  void *base = a.data();
  LargeMapping b = std::move(a);
  EXPECT_FALSE(a.valid());
  EXPECT_TRUE(b.valid());
  EXPECT_EQ(b.data(), base);
  EXPECT_TRUE(b.reset().ok());
  EXPECT_FALSE(b.valid());
  EXPECT_EQ(b.requested_flags(), LargeMappingFlags::kNone);
  EXPECT_EQ(b.accepted_flags(), LargeMappingFlags::kNone);
  EXPECT_EQ(b.requested_numa_node(), hyle::platform::kNumaAny);
}

TEST(LargeMapping, ZeroBytesInvalid) {
  auto r = LargeMapping::allocate(0);
  ASSERT_FALSE(r);
  EXPECT_TRUE(r.error().IsInvalidArgument());
}

TEST(LargeMapping, RejectsUnknownAndContradictoryPlacementFlags) {
  auto unknown = LargeMapping::allocate(
      page_size(), static_cast<LargeMappingFlags>(std::uint8_t{1} << 7));
  ASSERT_FALSE(unknown);
  EXPECT_TRUE(unknown.error().IsInvalidArgument());

  auto conflict = LargeMapping::allocate(
      page_size(), LargeMappingFlags::kNumaBind | LargeMappingFlags::kNumaInterleave);
  ASSERT_FALSE(conflict);
  EXPECT_TRUE(conflict.error().IsInvalidArgument());

  auto bind_without_node =
      LargeMapping::allocate(page_size(), LargeMappingFlags::kNumaBind);
  ASSERT_FALSE(bind_without_node);
  EXPECT_TRUE(bind_without_node.error().IsInvalidArgument());

  auto named_node_interleave = LargeMapping::allocate(
      page_size(), LargeMappingFlags::kNumaInterleave,
      static_cast<int>(hyle::platform::numa_nodes().front()));
  ASSERT_FALSE(named_node_interleave);
  EXPECT_TRUE(named_node_interleave.error().IsInvalidArgument());

  auto local_interleave =
      LargeMapping::allocate_local(page_size(), LargeMappingFlags::kNumaInterleave);
  ASSERT_FALSE(local_interleave);
  EXPECT_TRUE(local_interleave.error().IsInvalidArgument());
}

TEST(LargeMapping, HardBindRejectsInvalidNode) {
  const unsigned highest = hyle::platform::numa_nodes().back();
  ASSERT_LE(highest, static_cast<unsigned>(INT_MAX - 100));
  const int invalid = static_cast<int>(highest) + 100;
  ASSERT_NE(invalid, hyle::platform::kNumaAny);
  auto r = LargeMapping::allocate(page_size(), LargeMappingFlags::kNumaBind, invalid);
  ASSERT_FALSE(r);
  EXPECT_TRUE(r.error().IsOutOfRange());
}

TEST(LargeMapping, SelectedNodesAndChunkAreWritableAndObservable) {
  const std::size_t p = page_size();
  const std::array<unsigned, 1> selected{
      hyle::platform::numa_nodes().front()};
  auto result = LargeMapping::allocate_interleaved(
      p * 8, selected, p * 2, LargeMappingFlags::kPrefault);
  if (hyle::platform::numa_nodes().size() > 1u &&
      !hyle::platform::has_numa_capability(
          hyle::platform::numa_capabilities(),
          hyle::platform::NumaCapability::kSelectedInterleave)) {
    ASSERT_FALSE(result);
    EXPECT_TRUE(result.error().IsUnimplemented());
    return;
  }
  ASSERT_TRUE(result);
  LargeMapping region = *std::move(result);
  EXPECT_EQ(region.interleave_chunk_size(), p * 2);
  EXPECT_TRUE(any(region.requested_flags() & LargeMappingFlags::kNumaInterleave));
  EXPECT_TRUE(any(region.accepted_flags() & LargeMappingFlags::kNumaInterleave));
  EXPECT_TRUE(any(region.accepted_flags() & LargeMappingFlags::kPrefault));
  std::memset(region.data(), 0x4A, region.size());
  EXPECT_EQ(static_cast<unsigned char*>(region.data())[region.size() - 1],
            0x4A);
}

TEST(LargeMapping, SelectedNodesAcceptWeightedPlacement) {
  const unsigned node = hyle::platform::numa_nodes().front();
  const std::array<unsigned, 2> weighted{node, node};
  auto result = LargeMapping::allocate_interleaved(
      page_size(), weighted, page_size(), LargeMappingFlags::kNone);
  ASSERT_TRUE(result);
  EXPECT_TRUE(result->valid());
  EXPECT_EQ(result->interleave_chunk_size(), page_size());
}

TEST(LargeMapping, AdviseHugepageRecordsOutcome) {
  LargeMapping r =
      LargeMapping::allocate(page_size() * 4, LargeMappingFlags::kNone).value();
  ASSERT_TRUE(r.valid());
  const hyle::Status status = r.advise_hugepage();
  EXPECT_TRUE(any(r.requested_flags() & LargeMappingFlags::kThpHint));
  EXPECT_EQ(any(r.accepted_flags() & LargeMappingFlags::kThpHint), status.ok());
  std::memset(r.data(), 0, 64);
}
