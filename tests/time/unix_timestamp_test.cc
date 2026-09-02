#include "hyle/time/unix_timestamp.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>

namespace {

using hyle::time::UnixTimestamp;
using hyle::time::unix_timestamp;
using hyle::time::unix_timestamp_from_unix_microseconds;
using hyle::time::unix_timestamp_from_unix_milliseconds;
using hyle::time::unix_timestamp_from_unix_nanoseconds;
using hyle::time::unix_timestamp_from_unix_seconds;
using hyle::time::unix_timestamp_to_unix_microseconds;
using hyle::time::unix_timestamp_to_unix_milliseconds;
using hyle::time::unix_timestamp_to_unix_nanoseconds;
using hyle::time::unix_timestamp_to_unix_seconds;

TEST(UnixTimestamp, SignedUnitFactoriesCanonicalizeAcrossEpoch) {
  static_assert(unix_timestamp_from_unix_seconds(-1) == unix_timestamp(-1));
  static_assert(unix_timestamp_from_unix_milliseconds(-1) ==
                unix_timestamp(-1, 999'000'000u));
  static_assert(unix_timestamp_from_unix_microseconds(-1) ==
                unix_timestamp(-1, 999'999'000u));
  static_assert(unix_timestamp_from_unix_nanoseconds(-1) ==
                unix_timestamp(-1, 999'999'999u));

  EXPECT_EQ(unix_timestamp_from_unix_milliseconds(1),
            unix_timestamp(0, 1'000'000u));
  EXPECT_EQ(unix_timestamp_from_unix_microseconds(1),
            unix_timestamp(0, 1'000u));
  EXPECT_EQ(unix_timestamp_from_unix_nanoseconds(1), unix_timestamp(0, 1u));
}

TEST(UnixTimestamp, SignedUnitsRoundTripTheirEntireInputRange) {
  constexpr std::array values{
      (std::numeric_limits<std::int64_t>::min)(), std::int64_t{-1},
      std::int64_t{0}, std::int64_t{1},
      (std::numeric_limits<std::int64_t>::max)()};

  for (const std::int64_t value : values) {
    EXPECT_EQ(unix_timestamp_to_unix_seconds(
                  unix_timestamp_from_unix_seconds(value)),
              value);
    EXPECT_EQ(unix_timestamp_to_unix_milliseconds(
                  unix_timestamp_from_unix_milliseconds(value)),
              value);
    EXPECT_EQ(unix_timestamp_to_unix_microseconds(
                  unix_timestamp_from_unix_microseconds(value)),
              value);
    EXPECT_EQ(unix_timestamp_to_unix_nanoseconds(
                  unix_timestamp_from_unix_nanoseconds(value)),
              value);
  }
}

TEST(UnixTimestamp, LowerPrecisionExtractionFloorsBeforeEpoch) {
  const UnixTimestamp minus_one_nanosecond = unix_timestamp(-1, 999'999'999u);
  EXPECT_EQ(unix_timestamp_to_unix_seconds(minus_one_nanosecond), -1);
  EXPECT_EQ(unix_timestamp_to_unix_milliseconds(minus_one_nanosecond), -1);
  EXPECT_EQ(unix_timestamp_to_unix_microseconds(minus_one_nanosecond), -1);
  EXPECT_EQ(unix_timestamp_to_unix_nanoseconds(minus_one_nanosecond), -1);

  EXPECT_EQ(unix_timestamp_to_unix_milliseconds(
                unix_timestamp(-1, 998'999'999u)),
            -2);
  EXPECT_EQ(unix_timestamp_to_unix_microseconds(
                unix_timestamp(-1, 999'998'999u)),
            -2);
}

TEST(UnixTimestamp, SignedUnitExtractionRejectsInvalidAndOverflow) {
  const UnixTimestamp invalid = unix_timestamp(0, 1'000'000'000u);
  EXPECT_FALSE(unix_timestamp_to_unix_seconds(invalid));
  EXPECT_FALSE(unix_timestamp_to_unix_milliseconds(invalid));
  EXPECT_FALSE(unix_timestamp_to_unix_microseconds(invalid));
  EXPECT_FALSE(unix_timestamp_to_unix_nanoseconds(invalid));

  EXPECT_FALSE(unix_timestamp_to_unix_milliseconds(
      unix_timestamp((std::numeric_limits<std::int64_t>::max)(), 0)));
  EXPECT_FALSE(unix_timestamp_to_unix_nanoseconds(
      unix_timestamp((std::numeric_limits<std::int64_t>::min)(),
                     999'999'999u)));
}

}  // namespace
