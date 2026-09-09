#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "hyle/mem/range_allocator.h"

namespace {

using hyle::mem::AllocatorRef;
using hyle::mem::RangeAllocation;
using hyle::mem::RangeAllocator;
using hyle::mem::RangeAllocatorOptions;
using hyle::mem::system_allocator;

struct MetadataAllocatorState {
  std::size_t calls = 0;
  std::size_t fail_at = (std::numeric_limits<std::size_t>::max)();
  std::size_t outstanding = 0;
};

void* metadata_allocate(void* opaque, std::size_t size,
                        std::size_t alignment) noexcept {
  auto& state = *static_cast<MetadataAllocatorState*>(opaque);
  const std::size_t call = state.calls++;
  if (call == state.fail_at) return nullptr;
  void* allocation = system_allocator().allocate(size, alignment);
  if (allocation != nullptr) ++state.outstanding;
  return allocation;
}

void metadata_deallocate(void* opaque, void* allocation, std::size_t size,
                         std::size_t alignment) noexcept {
  if (allocation == nullptr) return;
  auto& state = *static_cast<MetadataAllocatorState*>(opaque);
  EXPECT_NE(state.outstanding, 0u);
  --state.outstanding;
  system_allocator().deallocate(allocation, size, alignment);
}

void* metadata_reallocate(void*, void*, std::size_t, std::size_t,
                          std::size_t) noexcept {
  return nullptr;
}

AllocatorRef metadata_allocator(MetadataAllocatorState& state) noexcept {
  return AllocatorRef(&metadata_allocate, &metadata_deallocate,
                      &metadata_reallocate, &state);
}

RangeAllocator make_ranges(std::uint64_t capacity,
                           std::uint32_t max_allocations) {
  return RangeAllocator::create(
             capacity,
             RangeAllocatorOptions{.max_allocations = max_allocations})
      .value();
}

TEST(RangeAllocator, RequiresExplicitValidCapacityAndMetadata) {
  RangeAllocatorOptions options{.max_allocations = 4};
  EXPECT_TRUE(RangeAllocator::create(0, options).error().IsInvalidArgument());

  options.max_allocations = 0;
  EXPECT_TRUE(RangeAllocator::create(1024, options)
                  .error()
                  .IsInvalidArgument());

  options.max_allocations = 4;
  options.metadata_allocator = {};
  EXPECT_TRUE(RangeAllocator::create(1024, options)
                  .error()
                  .IsInvalidArgument());
}

TEST(RangeAllocator, EveryMetadataAllocationFailureRollsBack) {
  for (std::size_t fail_at = 0; fail_at != 3; ++fail_at) {
    MetadataAllocatorState state{.fail_at = fail_at};
    auto created = RangeAllocator::create(
        4096, {.max_allocations = 8,
               .metadata_allocator = metadata_allocator(state)});
    EXPECT_TRUE(created.error().IsResourceExhausted()) << fail_at;
    EXPECT_EQ(state.outstanding, 0u) << fail_at;
  }
}

TEST(RangeAllocator, MetadataControlLivesThroughAnOrphanedLease) {
  MetadataAllocatorState state;
  RangeAllocation lease;
  {
    RangeAllocator ranges = RangeAllocator::create(
                                4096,
                                {.max_allocations = 8,
                                 .metadata_allocator =
                                     metadata_allocator(state)})
                                .value();
    EXPECT_EQ(state.outstanding, 3u);
    lease = ranges.allocate(64).value();
  }
  EXPECT_EQ(state.outstanding, 3u);
  EXPECT_FALSE(lease.valid());
  lease.reset();
  EXPECT_EQ(state.outstanding, 0u);
}

TEST(RangeAllocator, AllocatesAlignedRangesAndReportsExactSummary) {
  RangeAllocator ranges = make_ranges(4096, 8);
  RangeAllocation first = ranges.allocate(100, 64).value();
  RangeAllocation second = ranges.allocate(513, 256).value();

  EXPECT_EQ(first.offset() % 64, 0u);
  EXPECT_EQ(second.offset() % 256, 0u);
  EXPECT_LE(first.offset() + first.size(), second.offset());

  const auto stats = ranges.stats();
  EXPECT_EQ(stats.capacity_bytes, 4096u);
  EXPECT_EQ(stats.allocated_bytes, 613u);
  EXPECT_EQ(stats.free_bytes, 3483u);
  EXPECT_EQ(stats.peak_allocated_bytes, 613u);
  EXPECT_EQ(stats.live_allocations, 2u);
  EXPECT_EQ(stats.max_allocations, 8u);
  EXPECT_EQ(stats.outstanding_handles, 2u);
  EXPECT_GT(stats.metadata_bytes, 0u);

  const auto detailed = ranges.detailed_stats();
  EXPECT_EQ(detailed.summary.allocated_bytes, 613u);
  EXPECT_GE(detailed.free_range_count, 1u);
  EXPECT_GT(detailed.largest_free_range, 0u);
}

TEST(RangeAllocator, IndividualReleaseCoalescesAllAdjacentRanges) {
  RangeAllocator ranges = make_ranges(1024, 4);
  RangeAllocation a = ranges.allocate(128).value();
  RangeAllocation b = ranges.allocate(256).value();
  RangeAllocation c = ranges.allocate(64).value();
  b.reset();
  a.reset();
  c.reset();

  const auto detailed = ranges.detailed_stats();
  EXPECT_EQ(detailed.summary.allocated_bytes, 0u);
  EXPECT_EQ(detailed.summary.live_allocations, 0u);
  EXPECT_EQ(detailed.free_range_count, 1u);
  EXPECT_EQ(detailed.largest_free_range, 1024u);
  EXPECT_EQ(detailed.summary.peak_allocated_bytes, 448u);
}

TEST(RangeAllocator, BulkResetCannotReleaseAReusedNodeThroughStaleLease) {
  RangeAllocator ranges = make_ranges(1024, 2);
  RangeAllocation stale = ranges.allocate(512).value();
  ranges.reset();
  EXPECT_FALSE(stale.valid());
  EXPECT_EQ(ranges.stats().outstanding_handles, 1u);

  RangeAllocation current = ranges.allocate(1024).value();
  EXPECT_EQ(current.offset(), 0u);
  stale.reset();
  EXPECT_TRUE(current.valid());
  EXPECT_EQ(ranges.stats().allocated_bytes, 1024u);
  EXPECT_EQ(ranges.stats().outstanding_handles, 1u);
}

TEST(RangeAllocator, OwnerMayMoveOrDieBeforeAnOutstandingLease) {
  RangeAllocation lease;
  {
    RangeAllocator first = make_ranges(1024, 4);
    lease = first.allocate(128).value();
    RangeAllocator second = std::move(first);
    EXPECT_FALSE(first.valid());
    EXPECT_TRUE(second.valid());
    EXPECT_TRUE(lease.valid());
  }
  EXPECT_FALSE(lease.valid());
  lease.reset();
  EXPECT_FALSE(lease.valid());
}

TEST(RangeAllocator, OwnerMoveAssignmentInvalidatesOnlyReplacedControl) {
  RangeAllocator source = make_ranges(1024, 4);
  RangeAllocator destination = make_ranges(2048, 4);
  RangeAllocation source_lease = source.allocate(128).value();
  RangeAllocation replaced_lease = destination.allocate(256).value();

  destination = std::move(source);
  EXPECT_FALSE(source.valid());
  EXPECT_TRUE(destination.valid());
  EXPECT_TRUE(source_lease.valid());
  EXPECT_FALSE(replaced_lease.valid());
  EXPECT_EQ(destination.stats().allocated_bytes, 128u);

  replaced_lease.reset();
  EXPECT_TRUE(source_lease.valid());
  source_lease.reset();
  EXPECT_EQ(destination.stats().allocated_bytes, 0u);
}

TEST(RangeAllocator, BulkResetRetainsEveryOutstandingLeaseReference) {
  RangeAllocator ranges = make_ranges(1024, 2);
  std::vector<RangeAllocation> stale;
  stale.reserve(6);
  for (int generation = 0; generation < 3; ++generation) {
    stale.push_back(ranges.allocate(64).value());
    stale.push_back(ranges.allocate(128).value());
    ranges.reset();
  }
  EXPECT_EQ(ranges.stats().live_allocations, 0u);
  EXPECT_EQ(ranges.stats().outstanding_handles, 6u);
  for (const RangeAllocation& lease : stale) EXPECT_FALSE(lease.valid());
  stale.clear();
  EXPECT_EQ(ranges.stats().outstanding_handles, 0u);
}

TEST(RangeAllocator, EnforcesLiveMetadataCapacityWithoutHotGrowth) {
  RangeAllocator ranges = make_ranges(1024, 1);
  RangeAllocation only = ranges.allocate(1).value();
  EXPECT_TRUE(ranges.allocate(1).error().IsResourceExhausted());
  only.reset();
  EXPECT_TRUE(ranges.allocate(1).has_value());
}

TEST(RangeAllocator, SupportsOffsetsBeyondFourGiB) {
  constexpr std::uint64_t kGiB = std::uint64_t{1} << 30;
  RangeAllocator ranges = make_ranges(8 * kGiB, 4);
  RangeAllocation prefix = ranges.allocate(5 * kGiB, 4096).value();
  RangeAllocation high = ranges.allocate(1 * kGiB, 2 * 1024 * 1024).value();
  EXPECT_GT(high.offset(), std::uint64_t{4} * kGiB);
  EXPECT_EQ(high.offset() % (2 * 1024 * 1024), 0u);
  EXPECT_EQ(prefix.size(), 5 * kGiB);
}

TEST(RangeAllocator, AddressSensitiveFallbackDoesNotRejectAnExactFit) {
  RangeAllocator ranges = make_ranges(128, 2);
  // size + alignment - 1 exceeds capacity, but offset zero is already aligned.
  RangeAllocation exact = ranges.allocate(128, 128).value();
  EXPECT_EQ(exact.offset(), 0u);
  EXPECT_EQ(exact.size(), 128u);
}

TEST(RangeAllocator, RejectsInvalidRequestsAndPreservesStateOnFailure) {
  RangeAllocator ranges = make_ranges(256, 4);
  EXPECT_TRUE(ranges.allocate(0).error().IsInvalidArgument());
  EXPECT_TRUE(ranges.allocate(1, 0).error().IsInvalidArgument());
  EXPECT_TRUE(ranges.allocate(1, 3).error().IsInvalidArgument());
  EXPECT_TRUE(ranges.allocate(257).error().IsResourceExhausted());
  EXPECT_EQ(ranges.stats().allocated_bytes, 0u);
  EXPECT_EQ(ranges.detailed_stats().largest_free_range, 256u);
}

TEST(RangeAllocator, RandomizedChurnMatchesNonOverlapModel) {
  constexpr std::uint64_t kCapacity = 1u << 20;
  RangeAllocator ranges = make_ranges(kCapacity, 256);
  std::vector<RangeAllocation> live;
  live.reserve(256);
  std::uint64_t state = 0x9e3779b97f4a7c15ull;
  auto random = [&state] {
    state ^= state << 7;
    state ^= state >> 9;
    return state;
  };

  for (int iteration = 0; iteration < 10000; ++iteration) {
    const bool release = !live.empty() &&
                         (live.size() == 256 || (random() & 3u) == 0);
    if (release) {
      const std::size_t index =
          static_cast<std::size_t>(random() % live.size());
      live.erase(live.begin() + static_cast<std::ptrdiff_t>(index));
      continue;
    }

    const std::uint64_t size = 1 + random() % 16384;
    const std::uint64_t alignment = std::uint64_t{1} << (random() % 13);
    auto allocated = ranges.allocate(size, alignment);
    if (!allocated) {
      EXPECT_TRUE(allocated.error().IsResourceExhausted());
      continue;
    }
    EXPECT_EQ(allocated->offset() % alignment, 0u);
    EXPECT_LE(allocated->size(), kCapacity - allocated->offset());
    for (const RangeAllocation& other : live) {
      EXPECT_TRUE(allocated->offset() + allocated->size() <= other.offset() ||
                  other.offset() + other.size() <= allocated->offset());
    }
    live.push_back(std::move(*allocated));
  }

  live.clear();
  const auto final = ranges.detailed_stats();
  EXPECT_EQ(final.summary.allocated_bytes, 0u);
  EXPECT_EQ(final.free_range_count, 1u);
  EXPECT_EQ(final.largest_free_range, kCapacity);
}

static_assert(!std::is_copy_constructible_v<RangeAllocation>);
static_assert(std::is_nothrow_move_constructible_v<RangeAllocation>);
static_assert(!std::is_copy_constructible_v<RangeAllocator>);
static_assert(std::is_nothrow_move_constructible_v<RangeAllocator>);

}  // namespace
