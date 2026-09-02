/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary */

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "hyle/core/compiler.h"
#include "hyle/time/clock.h"

namespace {

using Duration = hyle::time::Duration;
using TimePoint = hyle::time::MonotonicTimePoint;

constexpr std::size_t kBatch = 64;
constexpr std::uint64_t kTimeoutNs = 30'000'000'000u;

std::uint64_t peer_to_ns(TimePoint point) noexcept {
  const auto value = point.time_since_epoch().count();
  return value > 0 ? static_cast<std::uint64_t>(value) : 0;
}

std::uint64_t peer_saturating_add(std::uint64_t base,
                                  std::uint64_t delta) noexcept {
  return delta > UINT64_MAX - base ? UINT64_MAX : base + delta;
}

Duration peer_nonnegative_difference(TimePoint later,
                                     TimePoint earlier) noexcept {
  using Rep = Duration::rep;
  const Rep later_count = later.time_since_epoch().count();
  const Rep earlier_count = earlier.time_since_epoch().count();
  if (later_count <= earlier_count) return Duration::zero();
  constexpr Rep kMax = (std::numeric_limits<Rep>::max)();
  if (earlier_count < 0 && later_count > kMax + earlier_count) {
    return (Duration::max)();
  }
  return Duration(later_count - earlier_count);
}

}  // namespace

// This test-only object is never linked into a product. The unresolved sink
// keeps both begin samples observable without importing a benchmark framework.
extern "C" void hyle_time_codegen_consume(std::uint64_t) noexcept;

extern "C" {

HYLE_NOINLINE std::uint64_t hyle_time_probe_instant_phase() noexcept {
  std::uint64_t checksum = 0;
  for (std::size_t index = 0; index < kBatch; ++index) {
    const hyle::time::Instant begin = hyle::time::Instant::now();
    hyle_time_codegen_consume(
        hyle::time::to_monotonic_ns(begin.time_point()));
    const std::uint64_t deadline = hyle::time::saturating_add_ns(
        hyle::time::to_monotonic_ns(begin.time_point()), kTimeoutNs + index);
    const hyle::time::Instant end = hyle::time::Instant::now();
    const std::uint64_t elapsed = begin.until_ns(end);
    const std::uint64_t end_ns =
        hyle::time::to_monotonic_ns(end.time_point());
    const std::uint64_t remaining = deadline > end_ns ? deadline - end_ns : 0;
    checksum += elapsed ^ remaining ^ deadline;
  }
  return checksum;
}

HYLE_NOINLINE std::uint64_t hyle_time_probe_chrono_phase() noexcept {
  std::uint64_t checksum = 0;
  for (std::size_t index = 0; index < kBatch; ++index) {
    const TimePoint begin = std::chrono::time_point_cast<Duration>(
        std::chrono::steady_clock::now());
    hyle_time_codegen_consume(peer_to_ns(begin));
    const std::uint64_t deadline =
        peer_saturating_add(peer_to_ns(begin), kTimeoutNs + index);
    const TimePoint end = std::chrono::time_point_cast<Duration>(
        std::chrono::steady_clock::now());
    const std::uint64_t elapsed = static_cast<std::uint64_t>(
        peer_nonnegative_difference(end, begin).count());
    const std::uint64_t end_ns = peer_to_ns(end);
    const std::uint64_t remaining = deadline > end_ns ? deadline - end_ns : 0;
    checksum += elapsed ^ remaining ^ deadline;
  }
  return checksum;
}

}  // extern "C"
