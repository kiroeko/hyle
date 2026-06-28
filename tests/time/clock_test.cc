#include <gtest/gtest.h>

#include <cstdint>

#include "hyle/time/clock.h"

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
