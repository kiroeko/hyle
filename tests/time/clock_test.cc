#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <ratio>
#include <time.h>
#include <type_traits>

#include "hyle/core/environment.h"
#if HYLE_OS_WINDOWS
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

#include "hyle/time/clock.h"
#include "clock_internal.h"

using hyle::time::coarse_realtime_available;
using hyle::time::coarse_realtime_ns;
using hyle::time::coarse_monotonic_available;
using hyle::time::coarse_monotonic_ns;
using hyle::time::current_thread_cpu_ns;
using hyle::time::Instant;
using hyle::time::MonotonicTimePoint;
using hyle::time::monotonic_ns;
using hyle::time::process_cpu_ns;
using hyle::time::ProcessCpuClock;
using hyle::time::profiling_clock_available;
using hyle::time::profiling_clock_ns;
using hyle::time::profiling_measurement_start_ns;
using hyle::time::profiling_measurement_stop_ns;
using hyle::time::realtime_ns;
using hyle::time::remaining_ns;
using hyle::time::saturating_add_ns;
using hyle::time::sleep_for_ms;
using hyle::time::sleep_for_ns;
using hyle::time::sleep_until_ns;
using hyle::time::ThreadCpuClock;

namespace {

template <class Clock>
consteval bool is_steady_chrono_clock_shape() {
  return Clock::is_steady &&
         std::is_same_v<typename Clock::duration,
                        std::chrono::duration<typename Clock::rep,
                                              typename Clock::period>> &&
         std::is_same_v<typename Clock::time_point,
                        std::chrono::time_point<Clock,
                                                typename Clock::duration>> &&
         std::is_same_v<decltype(Clock::now()), typename Clock::time_point> &&
         noexcept(Clock::now());
}

bool force_tsc_timing_calibration_requested() noexcept {
#if HYLE_OS_WINDOWS
  char value[2] = {};
  const DWORD length = ::GetEnvironmentVariableA(
      "HYLE_TIME_FORCE_TSC_TIMING_CALIBRATION", value,
      static_cast<DWORD>(sizeof(value)));
  return length == 1 && value[0] == '1';
#else
  const char* value =
      std::getenv("HYLE_TIME_FORCE_TSC_TIMING_CALIBRATION");
  return value != nullptr && value[0] == '1' && value[1] == '\0';
#endif
}

}  // namespace

TEST(Clock, MonotonicNonDecreasing) {
  std::uint64_t prev = monotonic_ns();
  for (int i = 0; i < 100000; ++i) {
    std::uint64_t now = monotonic_ns();
    ASSERT_GE(now, prev) << "monotonic clock went backwards at i=" << i;
    prev = now;
  }
}

#if HYLE_ARCH_ARM64 && \
    (HYLE_COMPILER_CLANG || HYLE_COMPILER_GCC)
TEST(Clock, ArmProfilingCounterIsAvailableAndNonDecreasing) {
  ASSERT_TRUE(profiling_clock_available());
  std::uint64_t previous = profiling_clock_ns();
  for (int i = 0; i < 100000; ++i) {
    const std::uint64_t current = profiling_clock_ns();
    ASSERT_GE(current, previous);
    previous = current;
  }
}
#endif

TEST(Clock, MonotonicAdvancesOverSleep) {
  std::uint64_t a = monotonic_ns();
  sleep_for_ms(5);
  std::uint64_t b = monotonic_ns();
  EXPECT_GT(b, a);
  // Should have advanced by at least ~4ms (allow scheduler slack below request).
  EXPECT_GE(b - a, 3'000'000ull);
}

TEST(Clock, OrderedProfilingMeasurementBoundariesAreReadable) {
  const std::uint64_t start = profiling_measurement_start_ns();
  const std::uint64_t stop = profiling_measurement_stop_ns();
  EXPECT_NE(start, 0u);
  EXPECT_NE(stop, 0u);
}

TEST(Clock, CpuidTscTupleDerivationFailsClosed) {
  using hyle::time::detail::derive_tsc_hz;
  using hyle::time::detail::tsc_scale_multiplier;

  EXPECT_EQ(derive_tsc_hz({2, 156, 38'400'000, 0}), 2'995'200'000.0);
  EXPECT_EQ(derive_tsc_hz({2, 156, 0, 2'995}), 2'994'999'930.0);
  EXPECT_EQ(derive_tsc_hz({0, 156, 38'400'000, 0}), 0.0);
  EXPECT_EQ(derive_tsc_hz({2, 0, 38'400'000, 0}), 0.0);
  EXPECT_EQ(derive_tsc_hz({2, 156, 0, 0}), 0.0);

  // These injected hypervisor tuples overflow the old three-factor u64
  // product, exceed the 32-bit crystal field, or truncate the crystal to zero.
  EXPECT_EQ(derive_tsc_hz(
                {(std::numeric_limits<std::uint32_t>::max)(), 1, 0, 65'535}),
            0.0);
  EXPECT_EQ(derive_tsc_hz({2, 1, 0, 3'000}), 0.0);
  EXPECT_EQ(derive_tsc_hz(
                {1, (std::numeric_limits<std::uint32_t>::max)(), 0, 1}),
            0.0);

  EXPECT_GT(tsc_scale_multiplier(2'995'200'000.0), 0u);
  EXPECT_EQ(tsc_scale_multiplier(derive_tsc_hz(
                {1, (std::numeric_limits<std::uint32_t>::max)(),
                 (std::numeric_limits<std::uint32_t>::max)(), 0})),
            0u);
  EXPECT_EQ(tsc_scale_multiplier(0.0), 0u);
  EXPECT_EQ(tsc_scale_multiplier(-1.0), 0u);
  EXPECT_EQ(tsc_scale_multiplier(1.0), 0u);
  EXPECT_EQ(tsc_scale_multiplier(
                (std::numeric_limits<double>::infinity)()),
            0u);
  EXPECT_EQ(tsc_scale_multiplier(
                (std::numeric_limits<double>::quiet_NaN)()),
            0u);
  EXPECT_EQ(tsc_scale_multiplier(0.1), 0u);
  EXPECT_EQ(tsc_scale_multiplier(1e20), 0u);
  EXPECT_EQ(tsc_scale_multiplier(1e9 * 0x1p32), 1u);
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

TEST(Clock, UnixTimestampIsCanonicalAndRoundTripsChronoRange) {
  using hyle::time::UnixTimePoint;
  using hyle::time::unix_time_point;
  using hyle::time::unix_timestamp;
  using hyle::time::unix_timestamp_from_time_point;

  const auto before_epoch = unix_timestamp_from_time_point(
      UnixTimePoint(std::chrono::nanoseconds(-1)));
  EXPECT_EQ(before_epoch.seconds, -1);
  EXPECT_EQ(before_epoch.nanoseconds, 999'999'999u);
  ASSERT_TRUE(unix_time_point(before_epoch).has_value());
  EXPECT_EQ(unix_time_point(before_epoch)->time_since_epoch().count(), -1);

  for (const std::int64_t count : {
           (std::numeric_limits<std::int64_t>::min)(), std::int64_t{-1},
           std::int64_t{0}, std::int64_t{1},
           (std::numeric_limits<std::int64_t>::max)()}) {
    const UnixTimePoint point{std::chrono::nanoseconds(count)};
    ASSERT_TRUE(unix_time_point(
                    unix_timestamp_from_time_point(point)).has_value());
    EXPECT_EQ(*unix_time_point(unix_timestamp_from_time_point(point)), point);
  }

  EXPECT_FALSE(unix_time_point(unix_timestamp(0, 1'000'000'000u)));
  EXPECT_FALSE(unix_time_point(unix_timestamp(
      (std::numeric_limits<std::int64_t>::max)(), 0)));
}

TEST(Clock, UnixTimestampHasNaturalValueSemantics) {
  using hyle::time::UnixTimestamp;
  using hyle::time::unix_timestamp;
  using hyle::time::unix_timestamp_valid;

  constexpr UnixTimestamp epoch;
  static_assert(epoch == unix_timestamp(0));
  static_assert(unix_timestamp(-1, 999'999'999u) < epoch);
  static_assert(epoch < unix_timestamp(0, 1u));
  static_assert(unix_timestamp(1) > unix_timestamp(0, 999'999'999u));
  EXPECT_TRUE(unix_timestamp_valid(epoch));
  EXPECT_FALSE(unix_timestamp_valid(unix_timestamp(0, 1'000'000'000u)));
}

TEST(Clock, UnixTimestampConvertsOsRepresentationsWithoutEpochClamp) {
  using hyle::time::unix_timestamp_from_timespec;
  using hyle::time::unix_timestamp_from_windows_filetime_ticks;
  using hyle::time::unix_timestamp_to_timespec;

  constexpr std::uint64_t kUnixEpochTicks = 116'444'736'000'000'000ull;
  auto at_epoch =
      unix_timestamp_from_windows_filetime_ticks(kUnixEpochTicks);
  EXPECT_EQ(at_epoch.seconds, 0);
  EXPECT_EQ(at_epoch.nanoseconds, 0u);
  auto before_epoch =
      unix_timestamp_from_windows_filetime_ticks(kUnixEpochTicks - 1);
  EXPECT_EQ(before_epoch.seconds, -1);
  EXPECT_EQ(before_epoch.nanoseconds, 999'999'900u);
  auto after_epoch =
      unix_timestamp_from_windows_filetime_ticks(kUnixEpochTicks + 1);
  EXPECT_EQ(after_epoch.seconds, 0);
  EXPECT_EQ(after_epoch.nanoseconds, 100u);

  ::timespec native{};
  native.tv_sec = -1;
  native.tv_nsec = 250'000'000;
  auto timestamp = unix_timestamp_from_timespec(native);
  ASSERT_TRUE(timestamp.has_value());
  EXPECT_EQ(timestamp->seconds, -1);
  EXPECT_EQ(timestamp->nanoseconds, 250'000'000u);
  ASSERT_TRUE(unix_timestamp_to_timespec(*timestamp).has_value());
  EXPECT_EQ(unix_timestamp_to_timespec(*timestamp)->tv_sec, -1);
  EXPECT_EQ(unix_timestamp_to_timespec(*timestamp)->tv_nsec, 250'000'000);

  native.tv_nsec = -1;
  EXPECT_FALSE(unix_timestamp_from_timespec(native));
}

TEST(Clock, UnixTimestampConvertsToWindowsFiletimeCheckedAndFloored) {
  using hyle::time::unix_timestamp;
  using hyle::time::unix_timestamp_from_windows_filetime_ticks;
  using hyle::time::unix_timestamp_to_windows_filetime_ticks;

  constexpr std::uint64_t kUnixEpochTicks =
      116'444'736'000'000'000ull;
  for (const std::uint64_t ticks : {
           std::uint64_t{0}, kUnixEpochTicks - 1, kUnixEpochTicks,
           kUnixEpochTicks + 1, (std::numeric_limits<std::uint64_t>::max)()}) {
    const auto timestamp =
        unix_timestamp_from_windows_filetime_ticks(ticks);
    ASSERT_TRUE(
        unix_timestamp_to_windows_filetime_ticks(timestamp).has_value());
    EXPECT_EQ(*unix_timestamp_to_windows_filetime_ticks(timestamp), ticks);
  }

  // Explicit floor policy, including the easy-to-get-wrong pre-epoch side.
  EXPECT_EQ(*unix_timestamp_to_windows_filetime_ticks(unix_timestamp(0, 99u)),
            kUnixEpochTicks);
  EXPECT_EQ(
      *unix_timestamp_to_windows_filetime_ticks(unix_timestamp(0, 199u)),
      kUnixEpochTicks + 1);
  EXPECT_EQ(*unix_timestamp_to_windows_filetime_ticks(
                unix_timestamp(-1, 999'999'999u)),
            kUnixEpochTicks - 1);

  const auto minimum = unix_timestamp_from_windows_filetime_ticks(0);
  EXPECT_FALSE(unix_timestamp_to_windows_filetime_ticks(
      unix_timestamp(minimum.seconds - 1, 999'999'999u)));
  EXPECT_FALSE(unix_timestamp_to_windows_filetime_ticks(
      unix_timestamp(0, 1'000'000'000u)));

  const auto maximum = unix_timestamp_from_windows_filetime_ticks(
      (std::numeric_limits<std::uint64_t>::max)());
  const auto beyond_maximum =
      maximum.nanoseconds <= 999'999'899u
          ? unix_timestamp(maximum.seconds, maximum.nanoseconds + 100u)
          : unix_timestamp(maximum.seconds + 1, 0);
  EXPECT_FALSE(unix_timestamp_to_windows_filetime_ticks(beyond_maximum));
}

TEST(Clock, RealtimeAdvances) {
  std::uint64_t a = realtime_ns();
  sleep_for_ms(5);
  std::uint64_t b = realtime_ns();
  EXPECT_GT(b, a);
}

TEST(Clock, CoarseRealtimeIsPlausiblyNow) {
  // Same epoch sanity as realtime_ns -- catches a wrong 1601/1970 offset or a
  // unit error in the coarse path. Must land in [2020, 2100).
  std::uint64_t now = coarse_realtime_ns();
  constexpr std::uint64_t k2020 = 1'577'836'800'000'000'000ull;
  constexpr std::uint64_t k2100 = 4'102'444'800'000'000'000ull;
  EXPECT_GT(now, k2020);
  EXPECT_LT(now, k2100);
}

TEST(Clock, CoarseRealtimeTracksRealtime) {
  // The coarse clock must agree with realtime_ns to within its resolution (~1-15
  // ms tick) + slack. A wrong scale/epoch would drift far more than that. Compare
  // the two taken back-to-back: |fast - precise| should be under one coarse tick
  // plus a generous margin (50 ms covers any tick granularity + scheduling).
  std::uint64_t p = realtime_ns();
  std::uint64_t f = coarse_realtime_ns();
  std::uint64_t diff = p > f ? p - f : f - p;
  EXPECT_LT(diff, 50'000'000ull) << "precise=" << p << " coarse=" << f;
}

TEST(Clock, CoarseRealtimeAvailabilityMatchesFallbackContract) {
#if defined(_WIN32) || defined(CLOCK_REALTIME_COARSE)
  EXPECT_TRUE(coarse_realtime_available());
#else
  EXPECT_FALSE(coarse_realtime_available());
#endif
}

TEST(Clock, CoarseMonotonicSharesPreciseClockDomain) {
  const std::uint64_t precise = monotonic_ns();
  const std::uint64_t coarse = coarse_monotonic_ns();
  const std::uint64_t diff = precise > coarse ? precise - coarse
                                               : coarse - precise;
  EXPECT_LT(diff, 100'000'000ull)
      << "precise=" << precise << " coarse=" << coarse;

  const auto typed = hyle::time::coarse_monotonic_now();
  const std::uint64_t typed_ns = hyle::time::to_monotonic_ns(typed);
  const std::uint64_t typed_diff =
      typed_ns > coarse ? typed_ns - coarse : coarse - typed_ns;
  EXPECT_LT(typed_diff, 100'000'000ull);

#if HYLE_OS_WINDOWS || defined(CLOCK_MONOTONIC_COARSE)
  EXPECT_TRUE(coarse_monotonic_available());
#else
  EXPECT_FALSE(coarse_monotonic_available());
#endif
}

TEST(Clock, CoarseRealtimeHasTypedAndTimestampViews) {
  const auto typed = hyle::time::coarse_realtime_now();
  const auto timestamp = hyle::time::coarse_realtime_timestamp();
  EXPECT_TRUE(hyle::time::unix_timestamp_valid(timestamp));

  const std::uint64_t raw = coarse_realtime_ns();
  const std::uint64_t typed_ns = hyle::time::to_realtime_ns(typed);
  const std::uint64_t diff = raw > typed_ns ? raw - typed_ns
                                            : typed_ns - raw;
  EXPECT_LT(diff, 100'000'000ull);
}

TEST(Clock, CpuAccountingClocksAreTypedMonotonicAndExcludeSleep) {
  static_assert(is_steady_chrono_clock_shape<ThreadCpuClock>());
  static_assert(is_steady_chrono_clock_shape<ProcessCpuClock>());
  static_assert(!std::is_same_v<ThreadCpuClock, hyle::time::MonotonicClock>);
  static_assert(!std::is_same_v<ProcessCpuClock, hyle::time::RealtimeClock>);

  const std::uint64_t thread_before_sleep = current_thread_cpu_ns();
  const std::uint64_t process_before_sleep = process_cpu_ns();
  sleep_for_ms(30);
  const std::uint64_t thread_after_sleep = current_thread_cpu_ns();
  const std::uint64_t process_after_sleep = process_cpu_ns();
  ASSERT_GE(thread_after_sleep, thread_before_sleep);
  ASSERT_GE(process_after_sleep, process_before_sleep);

  const std::uint64_t thread_before_busy = current_thread_cpu_ns();
  const std::uint64_t process_before_busy = process_cpu_ns();
  const std::uint64_t wall_start = monotonic_ns();
  while (monotonic_ns() - wall_start < 80'000'000ull) {
    // The native monotonic reads are observable calls and keep this a real
    // executing interval without adding unrelated allocation or synchronization.
  }
  const std::uint64_t thread_after_busy = current_thread_cpu_ns();
  const std::uint64_t process_after_busy = process_cpu_ns();

  const std::uint64_t sleep_thread =
      thread_after_sleep - thread_before_sleep;
  const std::uint64_t busy_thread = thread_after_busy - thread_before_busy;
  const std::uint64_t busy_process = process_after_busy - process_before_busy;
  EXPECT_GT(busy_thread, sleep_thread);
  EXPECT_GT(busy_thread, 5'000'000ull);
  EXPECT_GT(busy_process, 5'000'000ull);
  EXPECT_GE(ThreadCpuClock::now().time_since_epoch().count(), 0);
  EXPECT_GE(ProcessCpuClock::now().time_since_epoch().count(), 0);
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
  // A zero duration is a no-op, not an implicit scheduler yield.
  std::uint64_t a = monotonic_ns();
  sleep_for_ns(0);
  EXPECT_LT(monotonic_ns() - a, 100'000'000ull);
}

TEST(Clock, SleepUntilPastDeadlineReturnsPromptly) {
  const std::uint64_t now = monotonic_ns();
  sleep_until_ns(now > 0 ? now - 1 : 0);
  EXPECT_LT(monotonic_ns() - now, 100'000'000ull);
}

TEST(Clock, SleepUntilFutureDeadlineDoesNotReturnEarly) {
  const std::uint64_t start = monotonic_ns();
  const std::uint64_t deadline = saturating_add_ns(start, 5'000'000ull);
  sleep_until_ns(deadline);
  EXPECT_GE(monotonic_ns(), deadline);
}

TEST(Clock, SaturatingDeadlineArithmetic) {
  EXPECT_EQ(saturating_add_ns(10, 20), 30u);
  EXPECT_EQ(saturating_add_ns(UINT64_MAX - 5, 6), UINT64_MAX);
  EXPECT_EQ(saturating_add_ns(UINT64_MAX, 0), UINT64_MAX);

  const std::uint64_t future =
      hyle::time::deadline_after_ns(1'000'000'000ull);
  EXPECT_GT(remaining_ns(future), 0u);
  EXPECT_EQ(remaining_ns(0), 0u);
}

TEST(Clock, InstantElapsedClampsFutureInstant) {
  Instant future{MonotonicTimePoint::max()};
  EXPECT_EQ(future.elapsed_ns(), 0u);
}

TEST(Clock, InstantConvenienceDifferencesSaturateAtRepresentationEdges) {
  const Instant minimum{MonotonicTimePoint::min()};
  const Instant maximum{MonotonicTimePoint::max()};
  const Instant minus_one{
      MonotonicTimePoint(hyle::time::Duration(-1))};

  EXPECT_EQ(minimum.elapsed(), (hyle::time::Duration::max)());
  EXPECT_EQ(minimum.until(maximum), (hyle::time::Duration::max)());
  EXPECT_EQ(minimum.until(minus_one), (hyle::time::Duration::max)());
  EXPECT_EQ(maximum.until(minimum), hyle::time::Duration::zero());
  EXPECT_EQ(minimum.until(minimum), hyle::time::Duration::zero());
}

TEST(Clock, ChronoFacadePreservesUnitsAndRoundsUp) {
  using namespace std::chrono_literals;
  EXPECT_EQ(hyle::time::duration_ns_saturated(1500us), 1'500'000u);
  EXPECT_EQ(hyle::time::duration_ns_saturated(-1ns), 0u);
  EXPECT_EQ(hyle::time::duration_ns_saturated(
                std::chrono::duration<double, std::nano>(1.25)),
            2u);

  const auto start = hyle::time::monotonic_now();
  const auto deadline = start + 2ms;
  EXPECT_EQ(hyle::time::to_monotonic_ns(deadline),
            hyle::time::to_monotonic_ns(start) + 2'000'000u);
  EXPECT_EQ(hyle::time::monotonic_time_point(
                hyle::time::to_monotonic_ns(deadline)),
            deadline);
  const auto wall = hyle::time::realtime_now();
  EXPECT_EQ(hyle::time::realtime_time_point(hyle::time::to_realtime_ns(wall)),
            wall);
  EXPECT_GE(hyle::time::deadline_after(1ms), start);
}

TEST(Clock, IntegralDurationConversionStaysExactAcrossDoubleBoundary) {
  using SignedNs = std::chrono::duration<std::int64_t, std::nano>;
  using UnsignedNs = std::chrono::duration<std::uint64_t, std::nano>;
  constexpr std::uint64_t kDoubleIntegerBoundary = std::uint64_t{1} << 53;

  EXPECT_EQ(hyle::time::duration_ns_saturated(
                UnsignedNs(kDoubleIntegerBoundary - 1)),
            kDoubleIntegerBoundary - 1);
  EXPECT_EQ(hyle::time::duration_ns_saturated(
                UnsignedNs(kDoubleIntegerBoundary)),
            kDoubleIntegerBoundary);
  EXPECT_EQ(hyle::time::duration_ns_saturated(
                UnsignedNs(kDoubleIntegerBoundary + 1)),
            kDoubleIntegerBoundary + 1);
  EXPECT_EQ(hyle::time::duration_ns_saturated(
                SignedNs(static_cast<std::int64_t>(kDoubleIntegerBoundary + 1))),
            kDoubleIntegerBoundary + 1);
}

TEST(Clock, IntegralDurationConversionHandlesSignedAndUnsignedLimits) {
  using SignedNs = std::chrono::duration<std::int64_t, std::nano>;
  using UnsignedNs = std::chrono::duration<std::uint64_t, std::nano>;
  using UnsignedMs = std::chrono::duration<std::uint64_t, std::milli>;
  using UnsignedFemto =
      std::chrono::duration<std::uint64_t, std::femto>;
  using OverflowingCeil =
      std::chrono::duration<std::uint64_t, std::ratio<7, 5'000'000'000>>;

  EXPECT_EQ(hyle::time::duration_ns_saturated(
                SignedNs(std::numeric_limits<std::int64_t>::min())),
            0u);
  EXPECT_EQ(hyle::time::duration_ns_saturated(
                SignedNs(std::numeric_limits<std::int64_t>::max())),
            static_cast<std::uint64_t>(
                std::numeric_limits<std::int64_t>::max()));
  EXPECT_EQ(hyle::time::duration_ns_saturated(
                UnsignedNs(std::numeric_limits<std::uint64_t>::max())),
            std::numeric_limits<std::uint64_t>::max());
  EXPECT_EQ(hyle::time::duration_ns_saturated(
                UnsignedMs(std::numeric_limits<std::uint64_t>::max())),
            std::numeric_limits<std::uint64_t>::max());

  constexpr std::uint64_t kScale = 1'000'000;
  constexpr std::uint64_t kMax =
      std::numeric_limits<std::uint64_t>::max();
  constexpr std::uint64_t kCeilMaxDivScale =
      kMax / kScale + (kMax % kScale != 0 ? 1u : 0u);
  EXPECT_EQ(hyle::time::duration_ns_saturated(UnsignedFemto(kMax)),
            kCeilMaxDivScale);
  // count * 7 / 5 has quotient UINT64_MAX and remainder 2. The upward
  // rounding step itself must saturate instead of wrapping back to zero.
  EXPECT_EQ(hyle::time::duration_ns_saturated(
                OverflowingCeil(13'176'245'766'935'394'011ull)),
            kMax);
}

TEST(Clock, IntegralDurationConversionRoundsFractionalNanosecondsUpExactly) {
  using ThirdSeconds =
      std::chrono::duration<std::uint64_t, std::ratio<1, 3>>;
  using Picoseconds =
      std::chrono::duration<std::int64_t, std::pico>;
  using AlmostOneSecond = std::chrono::duration<
      std::uint64_t,
      std::ratio<std::numeric_limits<std::intmax_t>::max(),
                 std::numeric_limits<std::intmax_t>::max() - 1>>;

  EXPECT_EQ(hyle::time::duration_ns_saturated(ThirdSeconds(1)),
            333'333'334u);
  EXPECT_EQ(hyle::time::duration_ns_saturated(ThirdSeconds(2)),
            666'666'667u);
  EXPECT_EQ(hyle::time::duration_ns_saturated(ThirdSeconds(3)),
            1'000'000'000u);
  EXPECT_EQ(hyle::time::duration_ns_saturated(Picoseconds(1)), 1u);
  EXPECT_EQ(hyle::time::duration_ns_saturated(Picoseconds(1000)), 1u);
  EXPECT_EQ(hyle::time::duration_ns_saturated(Picoseconds(1001)), 2u);
  EXPECT_EQ(hyle::time::duration_ns_saturated(AlmostOneSecond(1)),
            1'000'000'001u);
  // Exercises both portable wide paths: the first multiplication overflows
  // u64, while the second has a fractional remainder whose *1e9 overflows u64.
  EXPECT_EQ(hyle::time::duration_ns_saturated(AlmostOneSecond(3)),
            3'000'000'001u);
  constexpr std::intmax_t kMax =
      std::numeric_limits<std::intmax_t>::max();
  using AlmostHalfSecond = std::chrono::duration<
      std::uint64_t, std::ratio<(kMax - 1) / 2, kMax>>;
  EXPECT_EQ(hyle::time::duration_ns_saturated(AlmostHalfSecond(3)),
            1'500'000'000u);
}

TEST(Clock, IntegralDurationConversionAcceptsBooleanRep) {
  using BooleanThirdSecond = std::chrono::duration<bool, std::ratio<1, 3>>;
  EXPECT_EQ(hyle::time::duration_ns_saturated(BooleanThirdSecond(false)), 0u);
  EXPECT_EQ(hyle::time::duration_ns_saturated(BooleanThirdSecond(true)),
            333'333'334u);
}

TEST(Clock, InstantDifferenceIsTypedAndSigned) {
  using namespace std::chrono_literals;
  const Instant earlier{MonotonicTimePoint(10ns)};
  const Instant later{MonotonicTimePoint(25ns)};
  EXPECT_EQ(later - earlier, 15ns);
  EXPECT_EQ(earlier - later, -15ns);
  EXPECT_LT(earlier, later);
}

TEST(Clock, ChronoSleepUntilDoesNotReturnEarly) {
  using namespace std::chrono_literals;
  const auto deadline = hyle::time::monotonic_now() + 2ms;
  hyle::time::sleep_until(deadline);
  EXPECT_GE(hyle::time::monotonic_now(), deadline);
}

TEST(Clock, ProfilingClockTracksMonotonicOverSleep) {
  // The profiling clock's elapsed must agree with the OS clock's elapsed
  // over the same interval -- this is what proves the TSC calibration is right
  // (a wrong tsc_hz would make the fast delta drift from the real one).
  std::uint64_t f0 = profiling_clock_ns();
  std::uint64_t m0 = monotonic_ns();
  sleep_for_ms(50);
  std::uint64_t f1 = profiling_clock_ns();
  std::uint64_t m1 = monotonic_ns();
  double fast_delta = static_cast<double>(f1 - f0);
  double mono_delta = static_cast<double>(m1 - m0);
  ASSERT_GT(mono_delta, 0.0);
  double rel_err = (fast_delta - mono_delta) / mono_delta;
  if (rel_err < 0) rel_err = -rel_err;
  // 2% tolerance: calibration target is 1%, plus scheduling jitter over 50ms.
  EXPECT_LT(rel_err, 0.02) << "fast=" << fast_delta << " mono=" << mono_delta;
}

TEST(Clock, ProfilingClockFallbackEqualsMonotonic) {
  // When the hardware path is unavailable, profiling_clock_ns must BE the
  // safe clock (a plain forward), so the two read essentially the same value.
  if (profiling_clock_available()) {
    GTEST_SKIP() << "invariant TSC present; fast path active, not fallback";
  }
  std::uint64_t f = profiling_clock_ns();
  std::uint64_t m = monotonic_ns();
  // Consecutive reads of the same underlying clock: within a few ms.
  std::uint64_t diff = m > f ? m - f : f - m;
  EXPECT_LT(diff, 5'000'000ull);
}

TEST(Clock, ProfilingClockTimingCalibrationIsLowCpu) {
#if !HYLE_ARCH_X64
  GTEST_SKIP() << "TSC timing calibration exists only on x86-64";
#endif
  if (!force_tsc_timing_calibration_requested()) {
    GTEST_SKIP() << "fresh-process timing-fallback test only";
  }

  const std::uint64_t cpu_start = current_thread_cpu_ns();
  const auto wall_start = std::chrono::steady_clock::now();
  const bool available = profiling_clock_available();
  const auto wall_stop = std::chrono::steady_clock::now();
  const std::uint64_t cpu_stop = current_thread_cpu_ns();
  if (!available) {
    GTEST_SKIP() << "host has no invariant TSC";
  }

  const double wall_ms =
      std::chrono::duration<double, std::milli>(wall_stop - wall_start).count();
  const double cpu_ms =
      static_cast<double>(cpu_stop - cpu_start) / 1'000'000.0;
  RecordProperty("calibration_wall_ms", wall_ms);
  RecordProperty("calibration_cpu_ms", cpu_ms);
  ASSERT_GE(wall_ms, 50.0)
      << "forced fallback did not run two accounting-sized sample windows";
  EXPECT_LT(wall_ms, 1000.0);
  EXPECT_LT(cpu_ms, wall_ms * 0.25 + 1.0)
      << "timing fallback must wait rather than occupy a core";
}
