#include "hyle/math/scalar.h"

#include <gtest/gtest.h>
#include <cstdint>
#include <limits>

using hyle::math::ceil_div;
using hyle::math::round_up_to_multiple;
using hyle::math::try_ceil_div;
using hyle::math::try_round_up_to_multiple;

TEST(CeilDiv, NormalAndSignedCounts) {
  EXPECT_EQ(ceil_div<unsigned>(0, 32), 0u);
  EXPECT_EQ(ceil_div<unsigned>(1, 32), 1u);
  EXPECT_EQ(ceil_div<unsigned>(32, 32), 1u);
  EXPECT_EQ(ceil_div<unsigned>(33, 32), 2u);
  EXPECT_EQ(ceil_div<std::int64_t>(65, 64), 2);
}

TEST(CeilDiv, CheckedBoundaryAndNoOverflowNearMax) {
  constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();
  EXPECT_EQ(ceil_div<std::uint64_t>(kMax, 1), kMax);
  EXPECT_EQ(ceil_div<std::uint64_t>(kMax, 2), kMax / 2 + 1);
  EXPECT_EQ(ceil_div<std::uint64_t>(kMax, kMax), 1u);
  EXPECT_FALSE(try_ceil_div<unsigned>(1, 0).has_value());
  EXPECT_FALSE(try_ceil_div<int>(-1, 2).has_value());
  EXPECT_FALSE(try_ceil_div<int>(1, -2).has_value());
}

TEST(RoundUpToMultiple, CheckedNonPowerOfTwo) {
  EXPECT_EQ(round_up_to_multiple<unsigned>(0, 6), 0u);
  EXPECT_EQ(round_up_to_multiple<unsigned>(1, 6), 6u);
  EXPECT_EQ(round_up_to_multiple<unsigned>(6, 6), 6u);
  EXPECT_EQ(round_up_to_multiple<unsigned>(7, 6), 12u);
  EXPECT_EQ(round_up_to_multiple<unsigned>(100, 10), 100u);
  EXPECT_EQ(round_up_to_multiple<int>(7, 6), 12);
  EXPECT_FALSE(try_round_up_to_multiple<unsigned>(1, 0).has_value());
  EXPECT_FALSE(try_round_up_to_multiple<int>(-1, 2).has_value());
  EXPECT_FALSE(try_round_up_to_multiple<int>(1, -2).has_value());
}

TEST(RoundUpToMultiple, RejectsUnrepresentableResult) {
  constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();
  EXPECT_EQ(round_up_to_multiple<std::uint64_t>(kMax, 1), kMax);
  EXPECT_EQ(round_up_to_multiple<std::uint64_t>(kMax - 1, 2), kMax - 1);
  EXPECT_FALSE(try_round_up_to_multiple<std::uint64_t>(kMax, 2).has_value());
  EXPECT_FALSE(try_round_up_to_multiple<std::int64_t>(
                   std::numeric_limits<std::int64_t>::max(), 2)
                   .has_value());
}

TEST(ScalarIntegerMath, PropertyAgainstSmallExactOracle) {
  for (std::uint32_t value = 0; value != 256; ++value) {
    for (std::uint32_t divisor = 1; divisor != 32; ++divisor) {
      const std::uint32_t expected_ceil =
          (value + divisor - 1) / divisor;
      EXPECT_EQ(ceil_div(value, divisor), expected_ceil);
      EXPECT_EQ(round_up_to_multiple(value, divisor),
                expected_ceil * divisor);
    }
  }
}

TEST(Scalar, Constexpr) {
  static_assert(ceil_div<unsigned>(65, 64) == 2u);
  static_assert(ceil_div<std::int64_t>(65, 64) == 2);
  static_assert(try_ceil_div<unsigned>(1, 0) == std::nullopt);
  static_assert(round_up_to_multiple<unsigned>(7, 6) == 12u);
  static_assert(try_round_up_to_multiple<unsigned>(7, 6) == 12u);
  static_assert(!try_round_up_to_multiple<std::uint64_t>(
                     std::numeric_limits<std::uint64_t>::max(), 2)
                     .has_value());
  SUCCEED();
}
