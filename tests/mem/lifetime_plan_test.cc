#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "hyle/mem/lifetime_plan.h"

namespace {

using hyle::Result;
using hyle::Status;
using namespace hyle::mem;

bool intervals_overlap(const LifetimeRequest& left,
                       const LifetimeRequest& right) {
  return left.begin < right.end && right.begin < left.end;
}

bool ranges_overlap(MemoryPlacement left, MemoryPlacement right) {
  return left.offset < right.offset + right.size &&
         right.offset < left.offset + left.size;
}

std::span<const MemoryResourceRef> one_system_resource() {
  static const std::array resources{system_resource()};
  return resources;
}

std::span<const MemoryResourceRef> two_system_resources() {
  static const std::array resources{system_resource(), system_resource()};
  return resources;
}

class BoundedHostResource final : public MemoryResource {
 public:
  explicit BoundedHostResource(
      std::size_t maximum_allocation,
      std::size_t minimum_alignment = alignof(std::max_align_t),
      std::size_t fail_after_successes = SIZE_MAX) noexcept
      : maximum_allocation_(maximum_allocation),
        minimum_alignment_(minimum_alignment),
        fail_after_successes_(fail_after_successes) {}

  std::string_view name() const noexcept override {
    return "test.bounded-lifetime";
  }
  MemoryProperties properties() const noexcept override {
    return MemoryProperties{
        .flags = MemoryProperty::kHostAccessible |
                 MemoryProperty::kHostDeviceCoherent,
        .minimum_alignment = minimum_alignment_,
        .maximum_allocation = maximum_allocation_,
        .domain = MemoryDomainId{this, 0},
    };
  }
  Result<ResourceAllocation> allocate(std::size_t size,
                                      std::size_t alignment) noexcept override {
    ++attempts_;
    if (size > maximum_allocation_ ||
        successes_ == fail_after_successes_) {
      return std::unexpected(
          Status::ResourceExhausted("test resource allocation failed"));
    }
    alignment = (std::max)(alignment, minimum_alignment_);
    auto allocation = system_resource().resource().allocate(size, alignment);
    if (allocation) {
      ++successes_;
      live_bytes_ += size;
    }
    return allocation;
  }
  void deallocate(ResourceAllocation allocation) noexcept override {
    system_resource().resource().deallocate(allocation);
    ++deallocations_;
    live_bytes_ -= allocation.size;
  }
  Result<ResourceHostMapping> map_host(
      void* native_handle, std::size_t offset, std::size_t size,
      HostAccess) noexcept override {
    void* data = size == 0
                     ? nullptr
                     : static_cast<void*>(
                           static_cast<std::byte*>(native_handle) + offset);
    return ResourceHostMapping{data, nullptr};
  }

  std::size_t attempts() const noexcept { return attempts_; }
  std::size_t successes() const noexcept { return successes_; }
  std::size_t deallocations() const noexcept { return deallocations_; }
  std::size_t live_bytes() const noexcept { return live_bytes_; }

 private:
  std::size_t maximum_allocation_;
  std::size_t minimum_alignment_;
  std::size_t fail_after_successes_;
  std::size_t attempts_ = 0;
  std::size_t successes_ = 0;
  std::size_t deallocations_ = 0;
  std::size_t live_bytes_ = 0;
};

TEST(LifetimePlan, ReusesHalfOpenIntervalsWithDeterministicBestFit) {
  const std::array requests{
      LifetimeRequest{.size = 64, .begin = 0, .end = 1},
      LifetimeRequest{.size = 128, .begin = 0, .end = 1},
      LifetimeRequest{.size = 60, .begin = 1, .end = 2},
  };
  LifetimePlan plan =
      LifetimePlanner::build(requests, one_system_resource()).value();
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
  LifetimePlan plan =
      LifetimePlanner::build(requests, two_system_resources()).value();
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
  LifetimePlan plan =
      LifetimePlanner::build(requests, one_system_resource()).value();
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
  LifetimePlan plan =
      LifetimePlanner::build(requests, one_system_resource()).value();
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
  LifetimePlan plan =
      LifetimePlanner::build(requests, one_system_resource()).value();
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
  EXPECT_EQ(buffers.backing(0, 0)->size(), 64u);
}

TEST(LifetimePlan, SplitsPhysicalBackingsAtProviderMaximum) {
  BoundedHostResource resource(128);
  const std::array providers{MemoryResourceRef(resource)};
  const std::array requests{
      LifetimeRequest{.size = 80, .alignment = 16, .begin = 0, .end = 3},
      LifetimeRequest{.size = 80, .alignment = 16, .begin = 0, .end = 3},
  };

  LifetimePlan plan = LifetimePlanner::build(requests, providers).value();
  ASSERT_EQ(plan.resource_plan(0)->peak_bytes, 160u);
  ASSERT_EQ(plan.resource_plan(0)->backing_count, 2u);
  EXPECT_EQ(plan.backing_plan(0, 0)->size, 80u);
  EXPECT_EQ(plan.backing_plan(0, 1)->size, 80u);
  EXPECT_EQ(plan.placement(0)->backing_index, 0u);
  EXPECT_EQ(plan.placement(1)->backing_index, 1u);
  EXPECT_EQ(plan.placement(0)->offset, 0u);
  EXPECT_EQ(plan.placement(1)->offset, 0u);

  {
    PlannedBuffers buffers = plan.instantiate(providers).value();
    EXPECT_EQ(buffers.backing_count(0).value(), 2u);
    EXPECT_NE(buffers.view(0)->native_handle(),
              buffers.view(1)->native_handle());
    EXPECT_EQ(buffers.backing(0, 0)->size(), 80u);
    EXPECT_EQ(buffers.backing(0, 1)->size(), 80u);
  }
  EXPECT_EQ(resource.successes(), 2u);
  EXPECT_EQ(resource.deallocations(), 2u);
  EXPECT_EQ(resource.live_bytes(), 0u);
}

TEST(LifetimePlan, PacksExactBoundaryAndSplitsOnAlignmentGap) {
  BoundedHostResource resource(128);
  const std::array providers{MemoryResourceRef(resource)};
  const std::array exact_requests{
      LifetimeRequest{.size = 64, .alignment = 64, .begin = 0, .end = 2},
      LifetimeRequest{.size = 64, .alignment = 64, .begin = 0, .end = 2},
  };
  LifetimePlan exact =
      LifetimePlanner::build(exact_requests, providers).value();
  ASSERT_EQ(exact.resource_plan(0)->backing_count, 1u);
  EXPECT_EQ(exact.backing_plan(0, 0)->size, 128u);
  EXPECT_EQ(exact.placement(1)->offset, 64u);

  const std::array gap_requests{
      LifetimeRequest{.size = 65, .alignment = 1, .begin = 0, .end = 2},
      LifetimeRequest{.size = 64, .alignment = 64, .begin = 0, .end = 2},
  };
  LifetimePlan gap = LifetimePlanner::build(gap_requests, providers).value();
  ASSERT_EQ(gap.resource_plan(0)->backing_count, 2u);
  EXPECT_EQ(gap.resource_plan(0)->peak_bytes, 129u);
  EXPECT_EQ(gap.backing_plan(0, 0)->size, 65u);
  EXPECT_EQ(gap.backing_plan(0, 1)->size, 64u);
  EXPECT_EQ(gap.placement(1)->backing_index, 1u);
  EXPECT_EQ(gap.placement(1)->offset, 0u);
}

TEST(LifetimePlan, ReusedLifetimeSlotKeepsItsPhysicalBacking) {
  BoundedHostResource resource(128);
  const std::array providers{MemoryResourceRef(resource)};
  const std::array requests{
      LifetimeRequest{.size = 80, .begin = 0, .end = 1},
      LifetimeRequest{.size = 80, .begin = 0, .end = 3},
      LifetimeRequest{.size = 60, .begin = 1, .end = 2},
  };

  LifetimePlan plan = LifetimePlanner::build(requests, providers).value();
  ASSERT_EQ(plan.resource_plan(0)->backing_count, 2u);
  EXPECT_EQ(plan.placement(2)->backing_index,
            plan.placement(0)->backing_index);
  EXPECT_EQ(plan.placement(2)->offset, plan.placement(0)->offset);
  EXPECT_NE(plan.placement(2)->backing_index,
            plan.placement(1)->backing_index);
}

TEST(LifetimePlan, KeepsResourceLocalBackingIndicesAndFlatOwnersIndependent) {
  BoundedHostResource first_resource(64);
  BoundedHostResource second_resource(96, 32);
  const std::array providers{MemoryResourceRef(first_resource),
                             MemoryResourceRef(second_resource)};
  const std::array requests{
      LifetimeRequest{.resource_slot = 0, .size = 64, .begin = 0, .end = 2},
      LifetimeRequest{.resource_slot = 0, .size = 64, .begin = 0, .end = 2},
      LifetimeRequest{.resource_slot = 1,
                      .size = 96,
                      .alignment = 32,
                      .begin = 0,
                      .end = 2},
  };

  LifetimePlan plan = LifetimePlanner::build(requests, providers).value();
  EXPECT_EQ(plan.resource_plan(0)->backing_count, 2u);
  EXPECT_EQ(plan.resource_plan(1)->backing_count, 1u);
  EXPECT_EQ(plan.placement(0)->backing_index, 0u);
  EXPECT_EQ(plan.placement(1)->backing_index, 1u);
  EXPECT_EQ(plan.placement(2)->backing_index, 0u);
  PlannedBuffers buffers = plan.instantiate(providers).value();
  EXPECT_EQ(buffers.backing_count(0).value(), 2u);
  EXPECT_EQ(buffers.backing_count(1).value(), 1u);
  EXPECT_EQ(buffers.backing(1, 0)->size(), 96u);
}

TEST(LifetimePlan, RejectsOversizedAliasGroupAndIncompatibleReplay) {
  BoundedHostResource bounded(64);
  const std::array bounded_provider{MemoryResourceRef(bounded)};
  const std::array oversized_alias{
      LifetimeRequest{.size = 32},
      LifetimeRequest{.size = 40, .alias_of = 0, .alias_offset = 32},
  };
  EXPECT_TRUE(LifetimePlanner::build(oversized_alias, bounded_provider)
                  .error()
                  .IsOutOfRange());

  BoundedHostResource build_resource(128);
  const std::array build_provider{MemoryResourceRef(build_resource)};
  const std::array requests{
      LifetimeRequest{.size = 96, .alignment = 32, .begin = 0, .end = 2},
  };
  LifetimePlan plan = LifetimePlanner::build(requests, build_provider).value();

  BoundedHostResource too_small(64);
  const std::array incompatible{MemoryResourceRef(too_small)};
  EXPECT_TRUE(plan.instantiate(incompatible).error().IsOutOfRange());
  EXPECT_EQ(too_small.attempts(), 0u);

  BoundedHostResource stronger(256, 64);
  const std::array compatible{MemoryResourceRef(stronger)};
  {
    PlannedBuffers buffers = plan.instantiate(compatible).value();
    EXPECT_GE(buffers.backing(0, 0)->resource().properties().minimum_alignment,
              64u);
  }
  EXPECT_EQ(stronger.successes(), 1u);
  EXPECT_EQ(stronger.deallocations(), 1u);
}

TEST(LifetimePlan, RollsBackEveryCreatedBackingOnPartialOom) {
  BoundedHostResource planner_resource(64);
  const std::array planner_provider{MemoryResourceRef(planner_resource)};
  const std::array requests{
      LifetimeRequest{.size = 64, .begin = 0, .end = 2},
      LifetimeRequest{.size = 64, .begin = 0, .end = 2},
      LifetimeRequest{.size = 64, .begin = 0, .end = 2},
  };
  LifetimePlan plan =
      LifetimePlanner::build(requests, planner_provider).value();
  ASSERT_EQ(plan.resource_plan(0)->backing_count, 3u);

  BoundedHostResource failing(64, alignof(std::max_align_t), 2);
  const std::array failing_provider{MemoryResourceRef(failing)};
  EXPECT_TRUE(
      plan.instantiate(failing_provider).error().IsResourceExhausted());
  EXPECT_EQ(failing.attempts(), 3u);
  EXPECT_EQ(failing.successes(), 2u);
  EXPECT_EQ(failing.deallocations(), 2u);
  EXPECT_EQ(failing.live_bytes(), 0u);
}

TEST(LifetimePlan, ZeroOnlyResourceRetainsProviderIdentity) {
  BoundedHostResource resource(0, 64);
  const std::array providers{MemoryResourceRef(resource)};
  const std::array requests{LifetimeRequest{.size = 0}};
  LifetimePlan plan = LifetimePlanner::build(requests, providers).value();
  ASSERT_EQ(plan.resource_plan(0)->peak_bytes, 0u);
  ASSERT_EQ(plan.resource_plan(0)->backing_count, 1u);
  EXPECT_EQ(plan.backing_plan(0, 0)->size, 0u);
  EXPECT_EQ(plan.backing_plan(0, 0)->alignment, 64u);
  {
    PlannedBuffers buffers = plan.instantiate(providers).value();
    BufferView view = buffers.view(0).value();
    EXPECT_TRUE(view.valid());
    EXPECT_TRUE(view.empty());
    EXPECT_EQ(view.resource(), MemoryResourceRef(resource));
  }
  EXPECT_EQ(resource.successes(), 1u);
  EXPECT_EQ(resource.deallocations(), 1u);
}

TEST(LifetimePlan, EmptyPlanIsAValidNoAllocationValue) {
  LifetimePlan plan = LifetimePlanner::build({}, {}).value();
  EXPECT_TRUE(plan.valid());
  EXPECT_EQ(plan.request_count(), 0u);
  EXPECT_EQ(plan.resource_count(), 0u);
  PlannedBuffers buffers =
      plan.instantiate(std::span<const MemoryResourceRef>{}).value();
  EXPECT_TRUE(buffers.valid());
  EXPECT_EQ(buffers.request_count(), 0u);
}

TEST(LifetimePlan, RejectsResourceContractMismatchesBeforePlanning) {
  const std::array requests{LifetimeRequest{.size = 1}};
  EXPECT_TRUE(LifetimePlanner::build(requests, {}).error().IsInvalidArgument());
  EXPECT_TRUE(LifetimePlanner::build({}, one_system_resource())
                  .error()
                  .IsInvalidArgument());

  const std::array invalid_provider{MemoryResourceRef{}};
  EXPECT_TRUE(LifetimePlanner::build(requests, invalid_provider)
                  .error()
                  .IsInvalidArgument());

  BoundedHostResource invalid_alignment(64, 3);
  const std::array malformed{MemoryResourceRef(invalid_alignment)};
  EXPECT_TRUE(LifetimePlanner::build(requests, malformed)
                  .error()
                  .IsInvalidArgument());
}

TEST(LifetimePlan, ReportsInvalidAndOutOfRangeQueriesWithoutAllocation) {
  LifetimePlan invalid;
  EXPECT_TRUE(invalid.placement(0).error().IsFailedPrecondition());
  EXPECT_TRUE(invalid.resource_plan(0).error().IsFailedPrecondition());
  EXPECT_TRUE(invalid.backing_plan(0, 0).error().IsFailedPrecondition());
  EXPECT_TRUE(invalid.instantiate({}).error().IsFailedPrecondition());

  const std::array requests{LifetimeRequest{.size = 1}};
  LifetimePlan plan =
      LifetimePlanner::build(requests, one_system_resource()).value();
  EXPECT_TRUE(plan.placement(1).error().IsOutOfRange());
  EXPECT_TRUE(plan.resource_plan(1).error().IsOutOfRange());
  EXPECT_TRUE(plan.backing_plan(0, 1).error().IsOutOfRange());
  EXPECT_TRUE(plan.backing_plan(1, 0).error().IsOutOfRange());

  PlannedBuffers buffers =
      plan.instantiate(one_system_resource()).value();
  EXPECT_TRUE(buffers.backing_count(1).error().IsOutOfRange());
  EXPECT_TRUE(buffers.backing(0, 1).error().IsOutOfRange());
  EXPECT_TRUE(buffers.backing(1, 0).error().IsOutOfRange());

  const void* native = buffers.view(0)->native_handle();
  PlannedBuffers moved = std::move(buffers);
  EXPECT_FALSE(buffers.valid());
  EXPECT_EQ(moved.view(0)->native_handle(), native);
  moved.reset();
  EXPECT_FALSE(moved.valid());
}

TEST(LifetimePlan, RejectsMalformedIntervalsResourcesAliasesAndOverflow) {
  EXPECT_TRUE(LifetimePlanner::build(
                  std::array{
                      LifetimeRequest{.size = 1, .begin = 1, .end = 1}},
                  one_system_resource())
                  .error()
                  .IsInvalidArgument());
  EXPECT_TRUE(LifetimePlanner::build(
                  std::array{LifetimeRequest{.size = 1,
                                             .alignment = 3,
                                             .begin = 0,
                                             .end = 1}},
                  one_system_resource())
                  .error()
                  .IsInvalidArgument());
  EXPECT_TRUE(LifetimePlanner::build(
                  std::array{LifetimeRequest{.resource_slot = 1,
                                             .size = 1,
                                             .begin = 0,
                                             .end = 1}},
                  two_system_resources())
                  .error()
                  .IsInvalidArgument());

  const std::array cycle{
      LifetimeRequest{.size = 1, .alias_of = 1},
      LifetimeRequest{.size = 1, .alias_of = 0},
  };
  EXPECT_TRUE(LifetimePlanner::build(cycle, one_system_resource())
                  .error()
                  .IsInvalidArgument());

  const std::array cross_resource{
      LifetimeRequest{.resource_slot = 0, .size = 8},
      LifetimeRequest{.resource_slot = 1, .size = 8, .alias_of = 0},
  };
  EXPECT_TRUE(
      LifetimePlanner::build(cross_resource, two_system_resources())
          .error()
          .IsInvalidArgument());

  const std::array bad_alias_alignment{
      LifetimeRequest{.size = 64, .alignment = 64},
      LifetimeRequest{.size = 8,
                      .alignment = 16,
                      .alias_of = 0,
                      .alias_offset = 1},
  };
  EXPECT_TRUE(LifetimePlanner::build(bad_alias_alignment,
                                     one_system_resource())
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
      LifetimePlanner::build(overflowing_alias, one_system_resource())
          .error()
          .IsOutOfRange());

  const std::array overflowing_peak{
      LifetimeRequest{.size = (std::numeric_limits<std::size_t>::max)(),
                      .alignment = 1,
                      .begin = 0,
                      .end = 2},
      LifetimeRequest{.size = 1,
                      .alignment = 1,
                      .begin = 0,
                      .end = 2},
  };
  EXPECT_TRUE(LifetimePlanner::build(overflowing_peak, one_system_resource())
                  .error()
                  .IsOutOfRange());
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

  LifetimePlan plan =
      LifetimePlanner::build(requests, two_system_resources()).value();
  for (std::size_t left = 0; left < requests.size(); ++left) {
    const MemoryPlacement a = plan.placement(left).value();
    EXPECT_EQ(a.offset % a.alignment, 0u);
    EXPECT_LE(a.offset + a.size,
              plan.backing_plan(a.resource_slot, a.backing_index)->size);
    for (std::size_t right = left + 1; right < requests.size(); ++right) {
      if (a.backing_index == plan.placement(right)->backing_index &&
          requests[left].resource_slot == requests[right].resource_slot &&
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
  LifetimePlan first =
      LifetimePlanner::build(requests, one_system_resource()).value();
  LifetimePlan second =
      LifetimePlanner::build(requests, one_system_resource()).value();
  ASSERT_EQ(first.resource_plan(0)->peak_bytes, 64u);
  for (std::size_t index = 0; index < kCount; ++index) {
    EXPECT_EQ(first.placement(index)->offset,
              second.placement(index)->offset);
  }
}

}  // namespace
