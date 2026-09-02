#include "hyle/core/environment.h"

#if HYLE_OS_WINDOWS
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <winsock2.h>
#  include <windows.h>
#endif

#include "hyle/time/native_interop.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <limits>

namespace {

using hyle::time::Duration;
using hyle::time::duration_from_timespec;
using hyle::time::duration_from_timeval;
using hyle::time::duration_to_timespec;
using hyle::time::duration_to_timeval_trunc;
using hyle::time::timeout_to_timeval;
using hyle::time::unix_timestamp;
using hyle::time::unix_timestamp_from_timeval;
using hyle::time::unix_timestamp_to_timeval;

TEST(TimeInterop, TimespecPreservesSignedNanosecondDuration) {
  for (const std::int64_t value : {
           (std::numeric_limits<std::int64_t>::min)(), std::int64_t{-1},
           std::int64_t{0}, std::int64_t{1},
           (std::numeric_limits<std::int64_t>::max)()}) {
    const Duration duration(value);
    const auto native = duration_to_timespec(duration);
    ASSERT_TRUE(native.has_value());
    EXPECT_EQ(duration_from_timespec(*native), duration);
  }

  ::timespec invalid{};
  invalid.tv_nsec = -1;
  EXPECT_FALSE(duration_from_timespec(invalid));
  invalid.tv_nsec = 1'000'000'000L;
  EXPECT_FALSE(duration_from_timespec(invalid));
}

TEST(TimeInterop, AbsoluteTimevalIsStrictCheckedAndFloored) {
  ::timeval native{};
  native.tv_sec = -1;
  native.tv_usec = 250'000;
  EXPECT_EQ(unix_timestamp_from_timeval(native),
            unix_timestamp(-1, 250'000'000u));

  native.tv_usec = -1;
  EXPECT_FALSE(unix_timestamp_from_timeval(native));
  native.tv_usec = 1'000'000;
  EXPECT_FALSE(unix_timestamp_from_timeval(native));

  ::timeval output{};
  output.tv_sec = 7;
  output.tv_usec = 8;
  ASSERT_TRUE(
      unix_timestamp_to_timeval(unix_timestamp(-1, 999'999'999u), output));
  EXPECT_EQ(output.tv_sec, -1);
  EXPECT_EQ(output.tv_usec, 999'999);

  output.tv_sec = 7;
  output.tv_usec = 8;
  EXPECT_FALSE(unix_timestamp_to_timeval(
      unix_timestamp(0, 1'000'000'000u), output));
  EXPECT_EQ(output.tv_sec, 7);
  EXPECT_EQ(output.tv_usec, 8);

  using NativeSeconds = decltype(output.tv_sec);
  if constexpr (sizeof(NativeSeconds) < sizeof(std::int64_t)) {
    output.tv_sec = 7;
    output.tv_usec = 8;
    const auto beyond_native = static_cast<std::int64_t>(
                                   (std::numeric_limits<NativeSeconds>::max)()) +
                               1;
    EXPECT_FALSE(
        unix_timestamp_to_timeval(unix_timestamp(beyond_native), output));
    EXPECT_EQ(output.tv_sec, 7);
    EXPECT_EQ(output.tv_usec, 8);
  }
}

TEST(TimeInterop, TimevalDurationTruncatesBeforeCanonicalSplit) {
  ::timeval output{};
  ASSERT_TRUE(duration_to_timeval_trunc(Duration(-1), output));
  EXPECT_EQ(output.tv_sec, 0);
  EXPECT_EQ(output.tv_usec, 0);

  ASSERT_TRUE(duration_to_timeval_trunc(Duration(-1'001), output));
  EXPECT_EQ(output.tv_sec, -1);
  EXPECT_EQ(output.tv_usec, 999'999);
  EXPECT_EQ(duration_from_timeval(output), Duration(-1'000));

  ASSERT_TRUE(duration_to_timeval_trunc(Duration(1'001), output));
  EXPECT_EQ(output.tv_sec, 0);
  EXPECT_EQ(output.tv_usec, 1);
  EXPECT_EQ(duration_from_timeval(output), Duration(1'000));

  using NativeSeconds = decltype(output.tv_sec);
  output.tv_sec = 7;
  output.tv_usec = 8;
  const bool minimum_represented =
      duration_to_timeval_trunc(Duration::min(), output);
  if constexpr (sizeof(NativeSeconds) >= sizeof(std::int64_t)) {
    ASSERT_TRUE(minimum_represented);
    ASSERT_TRUE(duration_from_timeval(output));
    EXPECT_EQ(duration_from_timeval(output)->count(),
              Duration::min().count() / 1'000 * 1'000);
  } else {
    EXPECT_FALSE(minimum_represented);
    EXPECT_EQ(output.tv_sec, 7);
    EXPECT_EQ(output.tv_usec, 8);
  }

  if constexpr (sizeof(NativeSeconds) >= sizeof(std::int64_t)) {
    ::timeval beyond_duration{};
    beyond_duration.tv_sec = static_cast<NativeSeconds>(9'223'372'037ll);
    beyond_duration.tv_usec = 0;
    EXPECT_FALSE(duration_from_timeval(beyond_duration));
  }
}

TEST(TimeInterop, TimeoutRoundsPositiveValuesUpAndPollsOtherwise) {
  ::timeval output{};
  ASSERT_TRUE(timeout_to_timeval(Duration(-1), output));
  EXPECT_EQ(output.tv_sec, 0);
  EXPECT_EQ(output.tv_usec, 0);
  ASSERT_TRUE(timeout_to_timeval(Duration::zero(), output));
  EXPECT_EQ(output.tv_sec, 0);
  EXPECT_EQ(output.tv_usec, 0);

  ASSERT_TRUE(timeout_to_timeval(Duration(1), output));
  EXPECT_EQ(output.tv_sec, 0);
  EXPECT_EQ(output.tv_usec, 1);
  ASSERT_TRUE(timeout_to_timeval(Duration(1'001), output));
  EXPECT_EQ(output.tv_sec, 0);
  EXPECT_EQ(output.tv_usec, 2);

  output.tv_sec = 7;
  output.tv_usec = 8;
  const bool represented = timeout_to_timeval(Duration::max(), output);
  using NativeSeconds = decltype(output.tv_sec);
  if constexpr (sizeof(NativeSeconds) >= sizeof(std::int64_t)) {
    EXPECT_TRUE(represented);
  } else {
    EXPECT_FALSE(represented);
    EXPECT_EQ(output.tv_sec, 7);
    EXPECT_EQ(output.tv_usec, 8);
  }
}

#if HYLE_OS_WINDOWS
TEST(TimeInterop, FiletimeStructAdapterPreservesFieldOrderAndRange) {
  using hyle::time::unix_timestamp_from_windows_filetime;
  using hyle::time::unix_timestamp_to_windows_filetime;

  constexpr std::uint64_t kUnixEpochTicks = 116'444'736'000'000'000ull;
  for (const std::uint64_t ticks : {
           std::uint64_t{0}, kUnixEpochTicks,
           (std::numeric_limits<std::uint64_t>::max)()}) {
    const FILETIME native{static_cast<DWORD>(ticks),
                          static_cast<DWORD>(ticks >> 32)};
    const auto timestamp = unix_timestamp_from_windows_filetime(native);
    FILETIME round_trip{};
    ASSERT_TRUE(unix_timestamp_to_windows_filetime(timestamp, round_trip));
    const std::uint64_t packed =
        static_cast<std::uint64_t>(round_trip.dwLowDateTime) |
        (static_cast<std::uint64_t>(round_trip.dwHighDateTime) << 32);
    EXPECT_EQ(packed, ticks);
  }

  FILETIME output{7, 8};
  ASSERT_TRUE(unix_timestamp_to_windows_filetime(
      unix_timestamp(-1, 999'999'999u), output));
  const std::uint64_t before_epoch_packed =
      static_cast<std::uint64_t>(output.dwLowDateTime) |
      (static_cast<std::uint64_t>(output.dwHighDateTime) << 32);
  EXPECT_EQ(before_epoch_packed, kUnixEpochTicks - 1);

  output = FILETIME{7, 8};
  EXPECT_FALSE(unix_timestamp_to_windows_filetime(
      unix_timestamp(0, 1'000'000'000u), output));
  EXPECT_EQ(output.dwLowDateTime, 7u);
  EXPECT_EQ(output.dwHighDateTime, 8u);
}
#endif

}  // namespace
