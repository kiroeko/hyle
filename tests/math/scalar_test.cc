#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "hyle/math/scalar.h"

using hyle::math::approx_eq;
using hyle::math::div_ceil;
using hyle::math::round_up_to_multiple;
using hyle::math::sgn;

TEST(DivCeil, Basic) {
  EXPECT_EQ(div_ceil<unsigned>(0, 32), 0u);
  EXPECT_EQ(div_ceil<unsigned>(1, 32), 1u);   // rounds up
  EXPECT_EQ(div_ceil<unsigned>(32, 32), 1u);  // exact, no extra
  EXPECT_EQ(div_ceil<unsigned>(33, 32), 2u);
  EXPECT_EQ(div_ceil<unsigned>(65, 64), 2u);
}

// The reason we don't use (a+b-1)/b: it overflows near the max. a/b+(a%b!=0)
// does not -- max/1 must be max exactly, not wrap to 0.
TEST(DivCeil, NoOverflowNearMax) {
  constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();
  EXPECT_EQ(div_ceil<std::uint64_t>(kMax, 1), kMax);
  EXPECT_EQ(div_ceil<std::uint64_t>(kMax, kMax), 1u);
}

TEST(RoundUpToMultiple, NonPowerOfTwo) {
  EXPECT_EQ(round_up_to_multiple<unsigned>(0, 6), 0u);
  EXPECT_EQ(round_up_to_multiple<unsigned>(1, 6), 6u);
  EXPECT_EQ(round_up_to_multiple<unsigned>(6, 6), 6u);
  EXPECT_EQ(round_up_to_multiple<unsigned>(7, 6), 12u);
  EXPECT_EQ(round_up_to_multiple<unsigned>(100, 10), 100u);
}

TEST(Sgn, DeadZone) {
  EXPECT_EQ(sgn(1.0), 1);
  EXPECT_EQ(sgn(-1.0), -1);
  EXPECT_EQ(sgn(0.0), 0);
  EXPECT_EQ(sgn(1e-9), 0);   // inside default eps band -> zero
  EXPECT_EQ(sgn(-1e-9), 0);
  EXPECT_EQ(sgn(1e-3), 1);   // outside band
}

TEST(ApproxEq, ScalesWithMagnitude) {
  EXPECT_TRUE(approx_eq(1.0, 1.0));
  EXPECT_FALSE(approx_eq(1.0, 2.0));
  // The point of relative epsilon: the SAME absolute gap (1.0) is "equal" at
  // large magnitude but "not equal" at small, under the SAME relative tolerance.
  EXPECT_TRUE(approx_eq(1e8, 1e8 + 1.0, 1e-6));  // threshold 1e8*1e-6=100 >> 1
  EXPECT_FALSE(approx_eq(1.0, 2.0, 1e-6));        // threshold 1e-6 << diff 1
  // Tunable tolerance.
  EXPECT_TRUE(approx_eq(1.0, 1.05, 0.1));
  EXPECT_FALSE(approx_eq(1.0, 1.2, 0.1));
  // Default eps is ~1 ulp (the tightest): a multi-ulp gap is NOT equal.
  EXPECT_FALSE(approx_eq(1e15, 1e15 + 1.0));
}

// Everything here is usable in constant expressions.
TEST(Scalar, Constexpr) {
  static_assert(div_ceil<unsigned>(65, 64) == 2u);
  static_assert(round_up_to_multiple<unsigned>(7, 6) == 12u);
  static_assert(sgn(2.0) == 1);
  static_assert(approx_eq(1.0, 1.0));
  SUCCEED();
}
