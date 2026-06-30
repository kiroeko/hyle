#include <gtest/gtest.h>

#include <cstdint>

#include "hyle/time/clock.h"

using hyle::time::fast_monotonic_available;
using hyle::time::fast_monotonic_ns;
using hyle::time::Instant;
using hyle::time::monotonic_ns;
using hyle::time::realtime_ns;
using hyle::time::sleep_for_ms;
using hyle::time::sleep_for_ns;

TEST(Clock, MonotonicNonDecreasing) {
  std::uint64_t prev = monotonic_ns();
  for (int i = 0; i < 100000; ++i) {
    std::uint64_t now = monotonic_ns();
    ASSERT_GE(now, prev) << "monotonic clock went backwards at i=" << i;
    prev = now;
  }
}

TEST(Clock, MonotonicAdvancesOverSleep) {
  std::uint64_t a = monotonic_ns();
  sleep_for_ms(5);
  std::uint64_t b = monotonic_ns();
  EXPECT_GT(b, a);
  // Should have advanced by at least ~4ms (allow scheduler slack below request).
  EXPECT_GE(b - a, 3'000'000ull);
}

TEST(Clock, RealtimeIsPlausiblyNow) {
  // realtime_ns must be after 2020-01-01 and before 2100-01-01 -- a sanity check
  // that the epoch conversion is right (catches 1601/1970 offset bugs).
  std::uint64_t now = realtime_ns();
  constexpr std::uint64_t k2020 = 1'577'836'800'000'000'000ull;  // 2020-01-01
  constexpr std::uint64_t k2100 = 4'102'444'800'000'000'000ull;  // 2100-01-01
  EXPECT_GT(now, k2020);
  EXPECT_LT(now, k2100);
}

TEST(Clock, RealtimeAdvances) {
  std::uint64_t a = realtime_ns();
  sleep_for_ms(5);
  std::uint64_t b = realtime_ns();
  EXPECT_GT(b, a);
}

TEST(Clock, SleepForNsSleepsAtLeastRequested) {
  // sleep_for must not under-sleep (measured by the monotonic clock).
  constexpr std::uint64_t req = 10'000'000ull;  // 10 ms
  std::uint64_t a = monotonic_ns();
  sleep_for_ns(req);
  std::uint64_t elapsed = monotonic_ns() - a;
  // Allow generous OS slack above, but it must be at least ~req (minus a small
  // tolerance for clock granularity differences between sleep and measurement).
  EXPECT_GE(elapsed, req - 1'000'000ull);
}

TEST(Clock, InstantElapsed) {
  Instant start = Instant::now();
  sleep_for_ms(5);
  std::uint64_t e = start.elapsed_ns();
  EXPECT_GE(e, 3'000'000ull);
  EXPECT_LT(e, 1'000'000'000ull);  // sanity: under a second
}

TEST(Clock, InstantUntil) {
  Instant a = Instant::now();
  sleep_for_ms(2);
  Instant b = Instant::now();
  EXPECT_GT(a.until_ns(b), 0u);
  EXPECT_EQ(b.until_ns(a), 0u);  // backwards -> 0, no underflow
}

TEST(Clock, ZeroSleepReturnsPromptly) {
  // sleep_for_ns(0) yields; must return ~immediately (well under 100ms).
  std::uint64_t a = monotonic_ns();
  sleep_for_ns(0);
  EXPECT_LT(monotonic_ns() - a, 100'000'000ull);
}

TEST(Clock, FastMonotonicNonDecreasing) {
  // Whether it uses rdtsc or falls back to monotonic_ns, it must never go back.
  std::uint64_t prev = fast_monotonic_ns();
  for (int i = 0; i < 100000; ++i) {
    std::uint64_t now = fast_monotonic_ns();
    ASSERT_GE(now, prev) << "fast monotonic clock went backwards at i=" << i;
    prev = now;
  }
}

TEST(Clock, FastMonotonicTracksMonotonicOverSleep) {
  // The fast clock's elapsed must agree with the OS monotonic clock's elapsed
  // over the same interval -- this is what proves the TSC calibration is right
  // (a wrong tsc_hz would make the fast delta drift from the real one).
  std::uint64_t f0 = fast_monotonic_ns();
  std::uint64_t m0 = monotonic_ns();
  sleep_for_ms(50);
  std::uint64_t f1 = fast_monotonic_ns();
  std::uint64_t m1 = monotonic_ns();
  double fast_delta = static_cast<double>(f1 - f0);
  double mono_delta = static_cast<double>(m1 - m0);
  ASSERT_GT(mono_delta, 0.0);
  double rel_err = (fast_delta - mono_delta) / mono_delta;
  if (rel_err < 0) rel_err = -rel_err;
  // 2% tolerance: calibration target is 1%, plus scheduling jitter over 50ms.
  EXPECT_LT(rel_err, 0.02) << "fast=" << fast_delta << " mono=" << mono_delta;
}

TEST(Clock, FastMonotonicFallbackEqualsMonotonic) {
  // When the hardware fast path is unavailable, fast_monotonic_ns must BE the
  // safe clock (a plain forward), so the two read essentially the same value.
  if (fast_monotonic_available()) {
    GTEST_SKIP() << "invariant TSC present; fast path active, not fallback";
  }
  std::uint64_t f = fast_monotonic_ns();
  std::uint64_t m = monotonic_ns();
  // Consecutive reads of the same underlying clock: within a few ms.
  std::uint64_t diff = m > f ? m - f : f - m;
  EXPECT_LT(diff, 5'000'000ull);
}
