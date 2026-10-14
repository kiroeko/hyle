#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include "hyle/compute/detail/copy_ranges.h"

namespace {
using namespace hyle::compute;
using namespace hyle::compute::detail;

struct Scratch final {
  std::array<CopyInterval, 128> intervals{};
  std::array<ResolvedCopyRange, 64> copies{};
  std::array<std::uint64_t, 64> closure{};
  CopyRangeScratch view() noexcept { return {intervals, copies, closure}; }
};
CopyRange range(std::uint32_t source, std::uint32_t destination,
                std::size_t source_offset, std::size_t destination_offset,
                std::size_t bytes, std::uint32_t phase = 0) {
  return {{source}, {destination}, {phase}, source_offset, destination_offset, bytes};
}

TEST(CopyRanges, EmptyExactSelfCopyAndZeroRangesNeedNoNativeWork) {
  Scratch scratch;
  auto empty = validate_copy_ranges({}, {}, {}, {}, scratch.view());
  ASSERT_TRUE(empty.has_value());
  EXPECT_EQ(empty->active_copies, 0u);
  const std::array resources{PhysicalCopyResource{1, 7, 0, 32, 4}, PhysicalCopyResource{1, 7, 0, 32, 4}};
  const std::array phases{CopyPhase{}};
  const std::array copies{range(0, 1, 1, 1, 3), range(0, 1, 32, 32, 0)};
  auto result = validate_copy_ranges(resources, phases, copies, {}, scratch.view());
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->no_op_copies, 2u);
  EXPECT_EQ(result->logical_bytes, 3u);
  EXPECT_EQ(result->physical_bytes, 0u);
  EXPECT_EQ(result->phase_completions, 0u);
}

TEST(CopyRanges, RepeatedReadsAndDistinctDestinationWindowsAreIndependent) {
  Scratch scratch;
  const std::array resources{PhysicalCopyResource{1, 1, 0, 64}, PhysicalCopyResource{1, 2, 0, 64}};
  const std::array phases{CopyPhase{}};
  const std::array copies{range(0, 1, 0, 0, 8), range(0, 1, 0, 8, 8)};
  auto result = validate_copy_ranges(resources, phases, copies, {}, scratch.view());
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->active_copies, 2u);
  EXPECT_EQ(result->physical_bytes, 16u);
  EXPECT_EQ(scratch.copies[0].input_index, 0u);
  EXPECT_EQ(scratch.copies[1].input_index, 1u);
  auto overlap = copies;
  overlap[1].destination_offset = 7;
  EXPECT_FALSE(validate_copy_ranges(resources, phases, overlap, {}, scratch.view()).has_value());
}

TEST(CopyRanges, PhysicalAliasesAcrossDifferentBindingsDoNotEvadeChecks) {
  Scratch scratch;
  const std::array resources{PhysicalCopyResource{3, 7, 100, 64}, PhysicalCopyResource{3, 7, 102, 64}};
  const std::array phases{CopyPhase{}};
  const std::array copies{range(0, 1, 0, 0, 4)};
  EXPECT_FALSE(validate_copy_ranges(resources, phases, copies, {}, scratch.view()).has_value());
  const std::array exact{range(0, 1, 2, 0, 4)};
  auto result = validate_copy_ranges(resources, phases, exact, {}, scratch.view());
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->no_op_copies, 1u);
}

TEST(CopyRanges, LateReadonlySourceCannotInventPhysicalProof) {
  Scratch scratch;
  std::array resources{PhysicalCopyResource{0, 0, 0, 32, 4},
                       PhysicalCopyResource{3, 7, 100, 32, 4}};
  const std::array phases{CopyPhase{}};
  const std::array copies{range(0, 1, 0, 0, 8)};
  const std::array unresolved{CopyResourceId{0}};
  auto admitted = validate_copy_ranges(resources, phases, copies, {}, scratch.view(), unresolved);
  ASSERT_TRUE(admitted.has_value());
  EXPECT_EQ(admitted->active_copies, 1u);
  EXPECT_EQ(admitted->no_op_copies, 0u);
  auto oversized = copies;
  oversized[0].source_offset = 28;
  EXPECT_FALSE(validate_copy_ranges(resources, phases, oversized, {}, scratch.view(), unresolved));
  const std::array reversed{range(1, 0, 0, 0, 8)};
  EXPECT_FALSE(validate_copy_ranges(resources, phases, reversed, {}, scratch.view(), unresolved));
  resources[0].backing = 7;
  EXPECT_FALSE(validate_copy_ranges(resources, phases, copies, {}, scratch.view(), unresolved));
  // Resolution proves a real partial overlap; a logical admission never
  // licenses executing these bytes before this full physical rejection.
  resources[0] = {3, 7, 102, 32, 4};
  EXPECT_FALSE(validate_copy_ranges(resources, phases, copies, {}, scratch.view()));
}

TEST(CopyRanges, LateSourceStillRejectsUnorderedDestinationHazards) {
  Scratch scratch;
  const std::array resources{PhysicalCopyResource{0, 0, 0, 32, 4},
      PhysicalCopyResource{3, 7, 100, 32, 4}, PhysicalCopyResource{3, 7, 104, 32, 4}};
  const std::array unresolved{CopyResourceId{0}};
  std::array phases{CopyPhase{}, CopyPhase{}};
  const std::array copies{range(0, 1, 0, 0, 8), range(0, 2, 0, 0, 8, 1)};
  EXPECT_FALSE(validate_copy_ranges(resources, phases, copies, {}, scratch.view(), unresolved));
  phases[1].dependency_mask = 1;
  auto ordered = validate_copy_ranges(resources, phases, copies, {}, scratch.view(), unresolved);
  ASSERT_TRUE(ordered.has_value());
  EXPECT_EQ(ordered->physical_bytes, 16u);
}

TEST(CopyRanges, ProducerConsumerChainRequiresExplicitTransitivePhaseEdges) {
  Scratch scratch;
  const std::array resources{PhysicalCopyResource{1, 1, 0, 64}, PhysicalCopyResource{1, 2, 0, 64},
                            PhysicalCopyResource{1, 3, 0, 64}};
  std::array phases{CopyPhase{}, CopyPhase{}, CopyPhase{}};
  const std::array copies{range(0, 1, 0, 0, 8, 0), range(1, 2, 0, 0, 8, 1), range(2, 0, 0, 0, 8, 2)};
  EXPECT_FALSE(validate_copy_ranges(resources, phases, copies, {}, scratch.view()).has_value());
  phases[1].dependency_mask = 1;
  phases[2].dependency_mask = 2;
  auto result = validate_copy_ranges(resources, phases, copies, {}, scratch.view());
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(scratch.closure[2], 3u);
  EXPECT_EQ(result->active_copies, 3u);
  auto one_phase = copies;
  for (auto &copy : one_phase) copy.phase = {0};
  EXPECT_FALSE(validate_copy_ranges(resources, phases, one_phase, {}, scratch.view()).has_value());
  phases[1].dependency_mask = 2;
  EXPECT_FALSE(validate_copy_ranges(resources, phases, copies, {}, scratch.view()).has_value());
}

TEST(CopyRanges, ExplicitPaddingIsChargedAndStaysInsideOwnedNativeRanges) {
  Scratch scratch;
  std::array resources{PhysicalCopyResource{1, 1, 0, 4, 4}, PhysicalCopyResource{1, 2, 0, 4, 4}};
  const std::array phases{CopyPhase{}};
  std::array copies{range(0, 1, 0, 0, 3)};
  EXPECT_FALSE(validate_copy_ranges(resources, phases, copies, {}, scratch.view()).has_value());
  copies[0].physical_bytes = 4;
  auto result = validate_copy_ranges(resources, phases, copies, {}, scratch.view());
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->logical_bytes, 3u);
  EXPECT_EQ(result->physical_bytes, 4u);
  resources[1].capacity = 3;
  EXPECT_FALSE(validate_copy_ranges(resources, phases, copies, {}, scratch.view()).has_value());
  resources[1].capacity = 4;
  copies[0].physical_bytes = 2;
  EXPECT_FALSE(validate_copy_ranges(resources, phases, copies, {}, scratch.view()).has_value());
  copies[0].physical_bytes = 4; copies[0].bytes = 0;
  EXPECT_FALSE(validate_copy_ranges(resources, phases, copies, {}, scratch.view()).has_value());
}

TEST(CopyRanges, IDsByteCapsArithmeticAndScratchRefuseBeforeAdmission) {
  Scratch scratch;
  const auto maximum = (std::numeric_limits<std::size_t>::max)();
  std::array resources{PhysicalCopyResource{1, 1, 0, 32}, PhysicalCopyResource{1, 2, 0, 32}};
  const std::array phases{CopyPhase{}};
  std::array copies{range(0, 1, 0, 0, 8)};
  auto small = CopyBatchOptions{}; small.max_bytes = 7;
  auto refused = validate_copy_ranges(resources, phases, copies, small, scratch.view());
  ASSERT_FALSE(refused.has_value()); EXPECT_TRUE(refused.error().IsResourceExhausted());
  auto metadata = scratch.view(); metadata.intervals = metadata.intervals.first(1);
  EXPECT_FALSE(validate_copy_ranges(resources, phases, copies, {}, metadata).has_value());
  copies[0].source.value = 2;
  EXPECT_FALSE(validate_copy_ranges(resources, phases, copies, {}, scratch.view()).has_value());
  copies[0].source.value = 0; copies[0].source_offset = maximum;
  EXPECT_FALSE(validate_copy_ranges(resources, phases, copies, {}, scratch.view()).has_value());
  copies[0].source_offset = 0; resources[0].origin = maximum - 3;
  EXPECT_FALSE(validate_copy_ranges(resources, phases, copies, {}, scratch.view()).has_value());
  resources[0].origin = 0; resources[0].copy_alignment = 3;
  EXPECT_FALSE(validate_copy_ranges(resources, phases, copies, {}, scratch.view()).has_value());
}

// Independent quadratic geometric oracle for small diagnostic inputs. It is
// not a native math/route qualification and does not copy the product sweep.
bool oracle(const std::array<PhysicalCopyResource, 4> &resources,
            const std::array<CopyPhase, 4> &phases, const std::vector<CopyRange> &copies) {
  bool reaches[4][4]{};
  for (std::size_t later = 0; later < 4; ++later)
    for (std::size_t earlier = 0; earlier < later; ++earlier)
      reaches[later][earlier] = (phases[later].dependency_mask & (std::uint64_t{1} << earlier)) != 0;
  for (std::size_t middle = 0; middle < 4; ++middle)
    for (std::size_t from = 0; from < 4; ++from)
      for (std::size_t to = 0; to < 4; ++to)
        reaches[from][to] = reaches[from][to] || (reaches[from][middle] && reaches[middle][to]);
  struct Access { std::uint64_t backing; std::size_t begin, end, phase; bool write; };
  std::vector<Access> accesses;
  for (const auto &copy : copies) {
    const auto &source = resources[copy.source.value], &destination = resources[copy.destination.value];
    const auto begin_source = source.origin + copy.source_offset, begin_destination = destination.origin + copy.destination_offset;
    if (copy.bytes == 0 || (source.backing == destination.backing && begin_source == begin_destination)) continue;
    accesses.push_back({source.backing, begin_source, begin_source + copy.bytes, copy.phase.value, false});
    accesses.push_back({destination.backing, begin_destination, begin_destination + copy.bytes, copy.phase.value, true});
  }
  for (std::size_t left = 0; left < accesses.size(); ++left) {
    for (std::size_t right = 0; right < left; ++right) {
      const auto &a = accesses[left], &b = accesses[right];
      if (a.backing != b.backing || (!a.write && !b.write) || a.end <= b.begin || b.end <= a.begin) continue;
      if (a.phase == b.phase || !reaches[(std::max)(a.phase, b.phase)][(std::min)(a.phase, b.phase)]) return false;
    }
  }
  return true;
}

TEST(CopyRanges, SweepMatchesIndependentSmallGeometricAndReachabilityModel) {
  Scratch scratch;
  const std::array resources{PhysicalCopyResource{1, 1, 0, 64}, PhysicalCopyResource{1, 1, 8, 64},
                            PhysicalCopyResource{1, 2, 0, 64}, PhysicalCopyResource{1, 2, 8, 64}};
  std::uint32_t random = 0x53a78b1u;
  auto next = [&]() { random ^= random << 13; random ^= random >> 17; random ^= random << 5; return random; };
  for (std::size_t trial = 0; trial < 500; ++trial) {
    std::array<CopyPhase, 4> phases{};
    for (std::size_t phase = 0; phase < phases.size(); ++phase)
      phases[phase].dependency_mask = next() & ((std::uint64_t{1} << phase) - 1);
    std::vector<CopyRange> copies;
    const auto count = next() % 12;
    for (std::size_t index = 0; index < count; ++index)
      copies.push_back(range(next() % 4, next() % 4, next() % 24, next() % 24, next() % 9, next() % 4));
    const bool expected = oracle(resources, phases, copies);
    const auto actual = validate_copy_ranges(resources, phases, copies, {}, scratch.view());
    EXPECT_EQ(actual.has_value(), expected) << "trial " << trial;
  }
}

} // namespace
