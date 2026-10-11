#include "hyle/compute/detail/copy_ranges.h"

#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <tuple>

namespace hyle::compute::detail {
namespace {

bool same_backing(const CopyInterval &left, const CopyInterval &right) noexcept {
  return left.address_space == right.address_space && left.backing == right.backing;
}

bool ordered(std::uint32_t left, std::uint32_t right,
             std::span<const std::uint64_t> closure) noexcept {
  if (left == right) return false;
  const auto first = (std::min)(left, right), last = (std::max)(left, right);
  return (closure[last] & (std::uint64_t{1} << first)) != 0;
}

Result<std::size_t> origin(const PhysicalCopyResource &resource,
                           std::size_t offset, std::size_t bytes) noexcept {
  if (offset > resource.capacity || bytes > resource.capacity - offset)
    return std::unexpected(Status::OutOfRange("copy range exceeds owned capacity"));
  const auto maximum = (std::numeric_limits<std::size_t>::max)();
  if (offset > maximum - resource.origin || bytes > maximum - resource.origin - offset)
    return std::unexpected(Status::OutOfRange("copy physical coordinate overflows"));
  return resource.origin + offset;
}

} // namespace

Result<CopyBatchReport> validate_copy_ranges(
    std::span<const PhysicalCopyResource> resources,
    std::span<const CopyPhase> phases, std::span<const CopyRange> ranges,
    const CopyBatchOptions &limits, CopyRangeScratch scratch) noexcept {
  if (limits.max_resources == 0 || limits.max_copies == 0 || limits.max_phases == 0 ||
      limits.max_phases > 64 || limits.max_bytes == 0)
    return std::unexpected(Status::InvalidArgument("copy batch limits are invalid"));
  if (resources.size() > limits.max_resources || phases.size() > limits.max_phases ||
      ranges.size() > limits.max_copies)
    return std::unexpected(Status::ResourceExhausted("copy batch capacity exhausted"));
  if (ranges.size() > scratch.intervals.size() / 2 || ranges.size() > scratch.copies.size() ||
      phases.size() > scratch.dependency_closure.size())
    return std::unexpected(Status::ResourceExhausted("copy validation metadata capacity exhausted"));
  for (const auto &resource : resources) {
    const auto alignment = resource.copy_alignment;
    if (alignment == 0 || (alignment & (alignment - 1)) != 0)
      return std::unexpected(Status::InvalidArgument("copy alignment is invalid"));
  }
  auto closure = scratch.dependency_closure.first(phases.size());
  for (std::size_t phase = 0; phase < phases.size(); ++phase) {
    const auto mask = phases[phase].dependency_mask;
    if ((mask >> phase) != 0)
      return std::unexpected(Status::InvalidArgument("copy dependency is not a preceding phase"));
    auto dependencies = mask;
    closure[phase] = mask;
    while (dependencies != 0) {
      const auto preceding = std::countr_zero(dependencies);
      closure[phase] |= closure[static_cast<std::size_t>(preceding)];
      dependencies &= dependencies - 1;
    }
  }
  CopyBatchReport report{.resources = static_cast<std::uint32_t>(resources.size()),
                        .copies = static_cast<std::uint32_t>(ranges.size()),
                        .phases = static_cast<std::uint32_t>(phases.size())};
  std::size_t interval_count = 0;
  for (std::size_t index = 0; index < ranges.size(); ++index) {
    const auto &copy = ranges[index];
    if (copy.source.value >= resources.size() || copy.destination.value >= resources.size() ||
        copy.phase.value >= phases.size())
      return std::unexpected(Status::InvalidArgument("copy resource or phase ID is invalid"));
    const auto bytes = copy.physical_bytes == 0 ? copy.bytes : copy.physical_bytes;
    if (bytes < copy.bytes || (copy.bytes == 0 && bytes != 0))
      return std::unexpected(Status::InvalidArgument("copy logical and physical sizes are inconsistent"));
    if (copy.bytes > limits.max_bytes - report.logical_bytes)
      return std::unexpected(Status::ResourceExhausted("copy byte capacity exhausted"));
    const auto &source = resources[copy.source.value], &destination = resources[copy.destination.value];
    HYLE_ASSIGN_OR_RETURN(auto source_begin, origin(source, copy.source_offset, bytes));
    HYLE_ASSIGN_OR_RETURN(auto destination_begin, origin(destination, copy.destination_offset, bytes));
    report.logical_bytes += copy.bytes;
    if (bytes == 0 || (source.address_space == destination.address_space && source.backing == destination.backing &&
                      source_begin == destination_begin)) {
      ++report.no_op_copies;
      continue;
    }
    if (bytes > limits.max_bytes - report.physical_bytes)
      return std::unexpected(Status::ResourceExhausted("copy physical byte capacity exhausted"));
    if (source_begin % source.copy_alignment != 0 || bytes % source.copy_alignment != 0 ||
        destination_begin % destination.copy_alignment != 0 || bytes % destination.copy_alignment != 0)
      return std::unexpected(Status::InvalidArgument("copy origin or size violates native alignment"));
    report.physical_bytes += bytes;
    scratch.copies[report.active_copies++] = {
        copy.source.value, copy.destination.value, copy.phase.value, static_cast<std::uint32_t>(index),
        copy.source_offset, copy.destination_offset, copy.bytes, bytes};
    scratch.intervals[interval_count++] = {source.address_space, source.backing, source_begin,
        source_begin + bytes, copy.phase.value, false};
    scratch.intervals[interval_count++] = {destination.address_space, destination.backing, destination_begin,
        destination_begin + bytes, copy.phase.value, true};
  }
  auto intervals = scratch.intervals.first(interval_count);
  std::sort(intervals.begin(), intervals.end(), [](const CopyInterval &left, const CopyInterval &right) noexcept {
    return std::tie(left.address_space, left.backing, left.begin, left.end, left.phase, left.write) <
           std::tie(right.address_space, right.backing, right.begin, right.end, right.phase, right.write);
  });
  // One maximum end per phase/permission is enough to discover any preceding
  // overlap. The phase ceiling is one machine word, so this is at most 64
  // checks per interval after O(R log R) sorting, not R squared pairs.
  std::array<std::size_t, 64> read_ends{}, write_ends{};
  const CopyInterval *previous = nullptr;
  for (const auto &interval : intervals) {
    if (previous == nullptr || !same_backing(*previous, interval)) {
      read_ends.fill(0); write_ends.fill(0);
    }
    for (std::size_t phase = 0; phase < phases.size(); ++phase) {
      const bool overlap = write_ends[phase] > interval.begin ||
                           (interval.write && read_ends[phase] > interval.begin);
      if (!overlap) continue;
      if (phase == interval.phase)
        return std::unexpected(Status::InvalidArgument("dependent or overlapping writes within one copy phase"));
      if (!ordered(static_cast<std::uint32_t>(phase), interval.phase, closure))
        return std::unexpected(Status::InvalidArgument("copy hazard has no explicit phase dependency"));
    }
    auto &end = interval.write ? write_ends[interval.phase] : read_ends[interval.phase];
    end = (std::max)(end, interval.end);
    previous = &interval;
  }
  auto active = scratch.copies.first(report.active_copies);
  std::sort(active.begin(), active.end(), [](const ResolvedCopyRange &left,
                                          const ResolvedCopyRange &right) noexcept {
    return std::tie(left.phase, left.input_index) < std::tie(right.phase, right.input_index);
  });
  return report;
}

} // namespace hyle::compute::detail
