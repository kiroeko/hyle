#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "hyle/mem/lifetime_plan.h"

namespace {

using namespace hyle::mem;

bool intervals_overlap(const LifetimeRequest& left,
                       const LifetimeRequest& right) {
  return left.begin < right.end && right.begin < left.end;
}

bool ranges_overlap(MemoryPlacement left, MemoryPlacement right) {
  return left.offset < right.offset + right.size &&
         right.offset < left.offset + left.size;
}

TEST(LifetimePlan, ReusesHalfOpenIntervalsWithDeterministicBestFit) {
  const std::array requests{
      LifetimeRequest{.size = 64, .begin = 0, .end = 1},
      LifetimeRequest{.size = 128, .begin = 0, .end = 1},
      LifetimeRequest{.size = 60, .begin = 1, .end = 2},
  };
  LifetimePlan plan = LifetimePlanner::build(requests).value();
  ASSERT_EQ(plan.request_count(), 3u);
  ASSERT_EQ(plan.resource_count(), 1u);
  const MemoryPlacement first = plan.placement(0).value();
  const MemoryPlacement second = plan.placement(1).value();
  const MemoryPlacement reused = plan.placement(2).value();
  EXPECT_EQ(reused.offset, first.offset);
  EXPECT_NE(reused.offset, second.offset);
  EXPECT_EQ(plan.resource_plan(0)->peak_bytes, 192u);
}

TEST(LifetimePlan, KeepsOverlappingRequestsDisjointAcrossDenseResources) {
  const std::array requests{
      LifetimeRequest{.resource_slot = 0,
                      .size = 64,
                      .alignment = 64,
                      .begin = 0,
                      .end = 3},
      LifetimeRequest{.resource_slot = 0,
                      .size = 16,
                      .alignment = 16,
                      .begin = 1,
                      .end = 2},
      LifetimeRequest{.resource_slot = 1,
                      .size = 96,
                      .alignment = 32,
                      .begin = 0,
                      .end = 2},
  };
  LifetimePlan plan = LifetimePlanner::build(requests).value();
  const MemoryPlacement first = plan.placement(0).value();
  const MemoryPlacement second = plan.placement(1).value();
  EXPECT_FALSE(ranges_overlap(first, second));
  EXPECT_EQ(first.offset % first.alignment, 0u);
  EXPECT_EQ(second.offset % second.alignment, 0u);
  EXPECT_EQ(plan.resource_count(), 2u);
  EXPECT_EQ(plan.resource_plan(0)->peak_bytes, 80u);
  EXPECT_EQ(plan.resource_plan(1)->peak_bytes, 96u);
}

TEST(LifetimePlan, ResolvesForwardAndChainedAliasesWithoutTensorTypes) {
  const std::array requests{
      LifetimeRequest{.size = 16,
                      .alignment = 16,
                      .begin = 2,
                      .end = 3,
                      .alias_of = 2,
                      .alias_offset = 16},
      LifetimeRequest{.size = 16,
                      .alignment = 16,
                      .begin = 3,
                      .end = 4,
                      .alias_of = 0,
                      .alias_offset = 16},
      LifetimeRequest{.size = 64,
                      .alignment = 64,
                      .begin = 0,
                      .end = 1},
  };
  LifetimePlan plan = LifetimePlanner::build(requests).value();
  const MemoryPlacement root = plan.placement(2).value();
  EXPECT_EQ(plan.placement(0)->offset, root.offset + 16);
  EXPECT_EQ(plan.placement(1)->offset, root.offset + 32);
  EXPECT_EQ(plan.resource_plan(0)->peak_bytes, 64u);
}

TEST(LifetimePlan, PersistentAndOutputFactsPreventLaterReuse) {
  const std::array requests{
      LifetimeRequest{.size = 32,
                      .begin = 0,
                      .end = 1,
                      .flags = LifetimeFlag::kPersistent},
      LifetimeRequest{.size = 32, .begin = 1, .end = 2},
      LifetimeRequest{.size = 32,
                      .begin = 2,
                      .end = 3,
                      .flags = LifetimeFlag::kOutput},
  };
  LifetimePlan plan = LifetimePlanner::build(requests).value();
  EXPECT_NE(plan.placement(0)->offset, plan.placement(1)->offset);
  EXPECT_EQ(plan.placement(2)->offset, plan.placement(1)->offset);
  EXPECT_EQ(plan.resource_plan(0)->peak_bytes, 64u);
}

TEST(LifetimePlan, InstantiatesBackingAndSurvivesPlanDestruction) {
  const std::array requests{
      LifetimeRequest{.size = 64,
                      .alignment = 64,
                      .begin = 0,
                      .end = 2},
      LifetimeRequest{.size = 16,
                      .alignment = 16,
                      .begin = 0,
                      .end = 2,
                      .alias_of = 0,
                      .alias_offset = 16},
      LifetimeRequest{.size = 64,
                      .alignment = 64,
                      .begin = 2,
                      .end = 3},
  };
  LifetimePlan plan = LifetimePlanner::build(requests).value();
  const std::array resources{system_resource()};
  PlannedBuffers buffers = plan.instantiate(resources).value();
  plan = {};

  BufferView first = buffers.view(0).value();
  BufferView alias = buffers.view(1).value();
  BufferView reused = buffers.view(2).value();
  EXPECT_EQ(first.native_handle(), reused.native_handle());
  EXPECT_EQ(first.offset(), reused.offset());
  EXPECT_EQ(alias.offset(), first.offset() + 16);
  EXPECT_TRUE(alias.fill(std::byte{0x5a}).ok());
  std::array<std::byte, 16> observed{};
  EXPECT_TRUE(first.subview(16, 16)->read(observed).ok());
  for (std::byte value : observed) EXPECT_EQ(value, std::byte{0x5a});
  EXPECT_EQ(buffers.backing(0)->size(), 64u);
}

TEST(LifetimePlan, EmptyPlanIsAValidNoAllocationValue) {
  LifetimePlan plan = LifetimePlanner::build({}).value();
  EXPECT_TRUE(plan.valid());
  EXPECT_EQ(plan.request_count(), 0u);
  EXPECT_EQ(plan.resource_count(), 0u);
  PlannedBuffers buffers =
      plan.instantiate(std::span<const MemoryResourceRef>{}).value();
  EXPECT_TRUE(buffers.valid());
  EXPECT_EQ(buffers.request_count(), 0u);
}

TEST(LifetimePlan, RejectsMalformedIntervalsResourcesAliasesAndOverflow) {
  EXPECT_TRUE(LifetimePlanner::build(std::array{
                  LifetimeRequest{.size = 1, .begin = 1, .end = 1}})
                  .error()
                  .IsInvalidArgument());
  EXPECT_TRUE(LifetimePlanner::build(std::array{
                  LifetimeRequest{.size = 1,
                                  .alignment = 3,
                                  .begin = 0,
                                  .end = 1}})
                  .error()
                  .IsInvalidArgument());
  EXPECT_TRUE(LifetimePlanner::build(std::array{
                  LifetimeRequest{.resource_slot = 1,
                                  .size = 1,
                                  .begin = 0,
                                  .end = 1}})
                  .error()
                  .IsInvalidArgument());

  const std::array cycle{
      LifetimeRequest{.size = 1, .alias_of = 1},
      LifetimeRequest{.size = 1, .alias_of = 0},
  };
  EXPECT_TRUE(LifetimePlanner::build(cycle).error().IsInvalidArgument());

  const std::array cross_resource{
      LifetimeRequest{.resource_slot = 0, .size = 8},
      LifetimeRequest{.resource_slot = 1, .size = 8, .alias_of = 0},
  };
  EXPECT_TRUE(
      LifetimePlanner::build(cross_resource).error().IsInvalidArgument());

  const std::array bad_alias_alignment{
      LifetimeRequest{.size = 64, .alignment = 64},
      LifetimeRequest{.size = 8,
                      .alignment = 16,
                      .alias_of = 0,
                      .alias_offset = 1},
  };
  EXPECT_TRUE(LifetimePlanner::build(bad_alias_alignment)
                  .error()
                  .IsInvalidArgument());

  const std::array overflowing_alias{
      LifetimeRequest{.size = 1},
      LifetimeRequest{
          .size = (std::numeric_limits<std::size_t>::max)(),
          .alignment = 1,
          .alias_of = 0,
          .alias_offset = 1},
  };
  EXPECT_TRUE(
      LifetimePlanner::build(overflowing_alias).error().IsOutOfRange());
}

TEST(LifetimePlan, PropertyReplayNeverOverlapsSimultaneouslyLiveRequests) {
  constexpr std::size_t kCount = 512;
  std::vector<LifetimeRequest> requests;
  requests.reserve(kCount);
  std::uint64_t state = 0x9e3779b97f4a7c15ull;
  auto next = [&state] {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return state * 0x2545f4914f6cdd1dull;
  };
  for (std::size_t index = 0; index < kCount; ++index) {
    const std::uint64_t begin = next() % 128;
    requests.push_back(LifetimeRequest{
        .resource_slot = static_cast<std::uint32_t>(index % 2),
        .size = static_cast<std::size_t>(next() % 256 + 1),
        .alignment = std::size_t{1} << (4 + next() % 3),
        .begin = begin,
        .end = begin + next() % 16 + 1,
    });
  }

  LifetimePlan plan = LifetimePlanner::build(requests).value();
  for (std::size_t left = 0; left < requests.size(); ++left) {
    const MemoryPlacement a = plan.placement(left).value();
    EXPECT_EQ(a.offset % a.alignment, 0u);
    EXPECT_LE(a.offset + a.size,
              plan.resource_plan(a.resource_slot)->peak_bytes);
    for (std::size_t right = left + 1; right < requests.size(); ++right) {
      if (requests[left].resource_slot == requests[right].resource_slot &&
          intervals_overlap(requests[left], requests[right])) {
        EXPECT_FALSE(ranges_overlap(a, plan.placement(right).value()));
      }
    }
  }
}

TEST(LifetimePlan, ScalesPastPeerFixedBlockCapsWithoutChangingPeak) {
  constexpr std::size_t kCount = 10000;
  std::vector<LifetimeRequest> requests;
  requests.reserve(kCount);
  for (std::size_t index = 0; index < kCount; ++index) {
    const std::uint64_t begin = kCount - index;
    requests.push_back(LifetimeRequest{
        .size = 64,
        .alignment = 64,
        .begin = begin,
        .end = begin + 1,
    });
  }
  LifetimePlan first = LifetimePlanner::build(requests).value();
  LifetimePlan second = LifetimePlanner::build(requests).value();
  ASSERT_EQ(first.resource_plan(0)->peak_bytes, 64u);
  for (std::size_t index = 0; index < kCount; ++index) {
    EXPECT_EQ(first.placement(index)->offset,
              second.placement(index)->offset);
  }
}

}  // namespace
