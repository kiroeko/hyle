#include "hyle/math/scalar.h"

#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <type_traits>

using hyle::math::ceil_div;
using hyle::math::round_up_to_multiple;
using hyle::math::saturating_add;
using hyle::math::saturating_cast;
using hyle::math::saturating_divide;
using hyle::math::saturating_multiply;
using hyle::math::saturating_subtract;
using hyle::math::try_add;
using hyle::math::try_bit_ceil;
using hyle::math::try_ceil_div;
using hyle::math::try_divide;
using hyle::math::try_integer_cast;
using hyle::math::try_lcm;
using hyle::math::try_multiply;
using hyle::math::try_multiply_add;
using hyle::math::try_product;
using hyle::math::try_remainder;
using hyle::math::try_round_up_to_multiple;
using hyle::math::try_subtract;
using hyle::math::try_sum;

namespace {

template <class Lhs, class Rhs>
concept CanTryAdd = requires(Lhs lhs, Rhs rhs) { try_add(lhs, rhs); };

static_assert(CanTryAdd<std::size_t, std::size_t>);
static_assert(!CanTryAdd<bool, bool>);
static_assert(!CanTryAdd<int, unsigned>);

template <class Lhs, class Rhs>
concept CanTryLcm = requires(Lhs lhs, Rhs rhs) { try_lcm(lhs, rhs); };

template <class T>
concept CanTryBitCeil = requires(T value) { try_bit_ceil(value); };

static_assert(CanTryLcm<std::size_t, std::size_t>);
static_assert(!CanTryLcm<bool, bool>);
static_assert(!CanTryLcm<int, unsigned>);
static_assert(CanTryBitCeil<std::size_t>);
static_assert(!CanTryBitCeil<bool>);

template <class To, class From>
concept CanTryIntegerCast = requires(From value) {
  try_integer_cast<To>(value);
};

static_assert(CanTryIntegerCast<std::size_t, std::int64_t>);
static_assert(!CanTryIntegerCast<bool, int>);
static_assert(!CanTryIntegerCast<int, bool>);
enum class EnumValue : unsigned { kZero };
static_assert(!CanTryIntegerCast<unsigned, EnumValue>);

template <class T, class Wide>
void expect_small_exact(T lhs, T rhs) {
  const Wide wide_lhs = static_cast<Wide>(lhs);
  const Wide wide_rhs = static_cast<Wide>(rhs);

  const auto check = [](const std::optional<T>& actual, Wide expected) {
    constexpr Wide minimum =
        static_cast<Wide>((std::numeric_limits<T>::min)());
    constexpr Wide maximum =
        static_cast<Wide>((std::numeric_limits<T>::max)());
    const bool representable = expected >= minimum && expected <= maximum;
    ASSERT_EQ(actual.has_value(), representable);
    if (representable) {
      EXPECT_EQ(*actual, static_cast<T>(expected));
    }
  };

  check(try_add(lhs, rhs), wide_lhs + wide_rhs);
  check(try_subtract(lhs, rhs), wide_lhs - wide_rhs);
  check(try_multiply(lhs, rhs), wide_lhs * wide_rhs);
}

template <class T>
void expect_integral_boundaries() {
  constexpr T kMin = (std::numeric_limits<T>::min)();
  constexpr T kMax = (std::numeric_limits<T>::max)();
  EXPECT_EQ(try_add(T{0}, T{0}), T{0});
  EXPECT_EQ(try_add(kMax, T{0}), kMax);
  EXPECT_FALSE(try_add(kMax, T{1}).has_value());
  EXPECT_EQ(try_subtract(kMax, T{0}), kMax);
  EXPECT_EQ(try_multiply(kMax, T{0}), T{0});
  EXPECT_EQ(try_multiply(kMax, T{1}), kMax);
  EXPECT_FALSE(try_multiply(kMax, T{2}).has_value());

  if constexpr (std::signed_integral<T>) {
    EXPECT_EQ(try_add(kMin, T{0}), kMin);
    EXPECT_FALSE(try_add(kMin, T{-1}).has_value());
    EXPECT_EQ(try_add(kMin, T{1}), static_cast<T>(kMin + T{1}));
    EXPECT_FALSE(try_subtract(kMin, T{1}).has_value());
    EXPECT_FALSE(try_subtract(kMax, T{-1}).has_value());
    EXPECT_EQ(try_subtract(kMin, T{-1}), static_cast<T>(kMin + T{1}));
    EXPECT_EQ(try_multiply(kMin, T{1}), kMin);
    EXPECT_FALSE(try_multiply(kMin, T{-1}).has_value());
    EXPECT_EQ(try_multiply(T{-7}, T{-6}), T{42});
    EXPECT_EQ(try_multiply(T{-7}, T{6}), T{-42});
  } else {
    EXPECT_FALSE(try_subtract(T{0}, T{1}).has_value());
    EXPECT_EQ(try_subtract(kMax, T{1}), static_cast<T>(kMax - T{1}));
  }
}

}  // namespace

TEST(CheckedArithmetic, ExhaustiveEightBitOracle) {
  for (int lhs = (std::numeric_limits<std::int8_t>::min)();
       lhs <= (std::numeric_limits<std::int8_t>::max)(); ++lhs) {
    for (int rhs = (std::numeric_limits<std::int8_t>::min)();
         rhs <= (std::numeric_limits<std::int8_t>::max)(); ++rhs) {
      expect_small_exact<std::int8_t, int>(static_cast<std::int8_t>(lhs),
                                           static_cast<std::int8_t>(rhs));
    }
  }
  for (unsigned lhs = 0; lhs <= (std::numeric_limits<std::uint8_t>::max)();
       ++lhs) {
    for (unsigned rhs = 0;
         rhs <= (std::numeric_limits<std::uint8_t>::max)(); ++rhs) {
      expect_small_exact<std::uint8_t, int>(static_cast<std::uint8_t>(lhs),
                                            static_cast<std::uint8_t>(rhs));
    }
  }
}

TEST(CheckedArithmetic, BoundariesAcrossStandardWidths) {
  expect_integral_boundaries<std::int8_t>();
  expect_integral_boundaries<std::uint8_t>();
  expect_integral_boundaries<std::int16_t>();
  expect_integral_boundaries<std::uint16_t>();
  expect_integral_boundaries<std::int32_t>();
  expect_integral_boundaries<std::uint32_t>();
  expect_integral_boundaries<std::int64_t>();
  expect_integral_boundaries<std::uint64_t>();
}

TEST(IntegerCast, HandlesSignednessAndWidthWithoutNarrowingFirst) {
  EXPECT_EQ(try_integer_cast<std::uint32_t>(std::int64_t{42}), 42u);
  EXPECT_FALSE(try_integer_cast<std::uint32_t>(std::int64_t{-1}));
  EXPECT_FALSE(try_integer_cast<std::int32_t>(
      (std::numeric_limits<std::uint64_t>::max)()));
  EXPECT_EQ(try_integer_cast<std::int8_t>(std::int16_t{127}), 127);
  EXPECT_FALSE(try_integer_cast<std::int8_t>(std::int16_t{128}));
  EXPECT_FALSE(try_integer_cast<std::int8_t>(std::int16_t{-129}));

  EXPECT_EQ(try_integer_cast<int>(char{42}), 42);
  EXPECT_EQ(try_integer_cast<char16_t>(std::uint8_t{250}), char16_t{250});
  EXPECT_FALSE(try_integer_cast<char>(
      static_cast<int>((std::numeric_limits<char>::max)()) + 1));
  EXPECT_FALSE(
      try_integer_cast<unsigned char>(static_cast<wchar_t>(-1)));
}

TEST(DivideAndRemainder, RejectInvalidDivisionWithoutEvaluatingUb) {
  constexpr auto kMin = (std::numeric_limits<std::int64_t>::min)();
  EXPECT_EQ(try_divide<std::int64_t>(21, -3), -7);
  EXPECT_EQ(try_remainder<std::int64_t>(22, -3), 1);
  EXPECT_FALSE(try_divide<std::int64_t>(1, 0));
  EXPECT_FALSE(try_remainder<std::int64_t>(1, 0));
  EXPECT_FALSE(try_divide<std::int64_t>(kMin, -1));
  EXPECT_EQ(try_remainder<std::int64_t>(kMin, -1), 0);
}

TEST(DiscreteSizing, LeastCommonMultipleHandlesSignsAndBoundaries) {
  constexpr auto kSignedMin = (std::numeric_limits<std::int64_t>::min)();
  constexpr auto kSignedMax = (std::numeric_limits<std::int64_t>::max)();
  constexpr auto kUnsignedMax = (std::numeric_limits<std::uint64_t>::max)();

  EXPECT_EQ(try_lcm<std::int64_t>(0, kSignedMin), 0);
  EXPECT_EQ(try_lcm<std::int64_t>(kSignedMin, 0), 0);
  EXPECT_EQ(try_lcm<std::int64_t>(-6, 8), 24);
  EXPECT_EQ(try_lcm<std::int64_t>(-21, -6), 42);
  EXPECT_EQ(try_lcm<std::int64_t>(kSignedMax, 1), kSignedMax);
  EXPECT_FALSE(try_lcm<std::int64_t>(kSignedMin, 1));
  EXPECT_FALSE(try_lcm<std::int64_t>(kSignedMin, kSignedMin));
  EXPECT_FALSE(try_lcm<std::int64_t>(kSignedMax, 2));
  EXPECT_EQ(try_lcm<std::uint64_t>(kUnsignedMax, 1), kUnsignedMax);
  EXPECT_FALSE(try_lcm<std::uint64_t>(kUnsignedMax, 2));
  EXPECT_EQ(try_lcm(char16_t{6}, char16_t{8}), char16_t{24});
}

TEST(DiscreteSizing, BitCeilHandlesRepresentabilityBeforeCallingStandard) {
  constexpr std::int64_t kSignedPower = std::int64_t{1} << 62;
  constexpr std::uint64_t kUnsignedPower = std::uint64_t{1} << 63;

  EXPECT_FALSE(try_bit_ceil<std::int64_t>(-1));
  EXPECT_EQ(try_bit_ceil<std::int64_t>(0), 1);
  EXPECT_EQ(try_bit_ceil<std::int64_t>(1), 1);
  EXPECT_EQ(try_bit_ceil<std::int64_t>(2), 2);
  EXPECT_EQ(try_bit_ceil<std::int64_t>(3), 4);
  EXPECT_EQ(try_bit_ceil(kSignedPower), kSignedPower);
  EXPECT_FALSE(try_bit_ceil(kSignedPower + 1));
  EXPECT_EQ(try_bit_ceil(kUnsignedPower), kUnsignedPower);
  EXPECT_FALSE(try_bit_ceil(kUnsignedPower + 1));
  EXPECT_FALSE(try_bit_ceil((std::numeric_limits<std::uint64_t>::max)()));
  EXPECT_EQ(try_bit_ceil(char16_t{65}), char16_t{128});
}

TEST(DiscreteSizing, ExhaustiveEightBitOracle) {
  for (int lhs = (std::numeric_limits<std::int8_t>::min)();
       lhs <= (std::numeric_limits<std::int8_t>::max)(); ++lhs) {
    for (int rhs = (std::numeric_limits<std::int8_t>::min)();
         rhs <= (std::numeric_limits<std::int8_t>::max)(); ++rhs) {
      const int lhs_magnitude = lhs < 0 ? -lhs : lhs;
      const int rhs_magnitude = rhs < 0 ? -rhs : rhs;
      const int expected = lhs == 0 || rhs == 0
                               ? 0
                               : lhs_magnitude /
                                     std::gcd(lhs_magnitude, rhs_magnitude) *
                                     rhs_magnitude;
      const auto actual =
          try_lcm(static_cast<std::int8_t>(lhs),
                  static_cast<std::int8_t>(rhs));
      const bool representable =
          expected <= (std::numeric_limits<std::int8_t>::max)();
      ASSERT_EQ(actual.has_value(), representable);
      if (representable) {
        EXPECT_EQ(*actual, expected);
      }
    }
  }

  for (unsigned lhs = 0;
       lhs <= (std::numeric_limits<std::uint8_t>::max)(); ++lhs) {
    for (unsigned rhs = 0;
         rhs <= (std::numeric_limits<std::uint8_t>::max)(); ++rhs) {
      const unsigned expected =
          lhs == 0 || rhs == 0 ? 0 : lhs / std::gcd(lhs, rhs) * rhs;
      const auto actual =
          try_lcm(static_cast<std::uint8_t>(lhs),
                  static_cast<std::uint8_t>(rhs));
      const bool representable =
          expected <= (std::numeric_limits<std::uint8_t>::max)();
      ASSERT_EQ(actual.has_value(), representable);
      if (representable) {
        EXPECT_EQ(*actual, expected);
      }
    }
  }

  for (unsigned value = 0;
       value <= (std::numeric_limits<std::uint8_t>::max)(); ++value) {
    unsigned expected = 1;
    while (expected < value) expected *= 2;
    const auto actual = try_bit_ceil(static_cast<std::uint8_t>(value));
    const bool representable =
        expected <= (std::numeric_limits<std::uint8_t>::max)();
    ASSERT_EQ(actual.has_value(), representable);
    if (representable) {
      EXPECT_EQ(*actual, expected);
    }

    if (value <= static_cast<unsigned>(
                     (std::numeric_limits<std::int8_t>::max)())) {
      const auto signed_actual =
          try_bit_ceil(static_cast<std::int8_t>(value));
      const bool signed_representable =
          expected <= static_cast<unsigned>(
                          (std::numeric_limits<std::int8_t>::max)());
      ASSERT_EQ(signed_actual.has_value(), signed_representable);
      if (signed_representable) {
        EXPECT_EQ(*signed_actual, expected);
      }
    }
  }
  EXPECT_FALSE(try_bit_ceil(std::int8_t{-1}));
}

TEST(CompoundArithmetic, UsesExactStepsAndSinglePassIdentities) {
  EXPECT_EQ(try_multiply_add<std::uint64_t>(7, 6, 1), 43u);
  EXPECT_FALSE(try_multiply_add<std::int8_t>(100, 2, -100));
  EXPECT_FALSE(try_multiply_add<std::uint64_t>(
      (std::numeric_limits<std::uint64_t>::max)(), 1, 1));

  std::array<std::uint32_t, 4> values{2, 3, 4, 5};
  EXPECT_EQ(try_sum(std::span{values}), 14u);
  EXPECT_EQ(try_product(std::span{values}), 120u);
  EXPECT_EQ(try_sum(std::span<const std::uint32_t>{}), 0u);
  EXPECT_EQ(try_product(std::span<const std::uint32_t>{}), 1u);

  std::array<std::uint8_t, 2> overflow_sum{255, 1};
  std::array<std::uint8_t, 2> overflow_product{128, 2};
  EXPECT_FALSE(try_sum(std::span{overflow_sum}));
  EXPECT_FALSE(try_product(std::span{overflow_product}));
}

TEST(SaturatingArithmetic, ClampsEachDirection) {
  constexpr auto kMin = (std::numeric_limits<std::int32_t>::min)();
  constexpr auto kMax = (std::numeric_limits<std::int32_t>::max)();
  EXPECT_EQ(saturating_add(kMax, 1), kMax);
  EXPECT_EQ(saturating_add(kMin, -1), kMin);
  EXPECT_EQ(saturating_subtract(kMin, 1), kMin);
  EXPECT_EQ(saturating_subtract(kMax, -1), kMax);
  EXPECT_EQ(saturating_multiply(kMax, 2), kMax);
  EXPECT_EQ(saturating_multiply(kMin, 2), kMin);
  EXPECT_EQ(saturating_multiply(kMin, -1), kMax);
  EXPECT_EQ(saturating_divide(kMin, -1), kMax);
  EXPECT_EQ(saturating_divide(-21, 3), -7);

  EXPECT_EQ(saturating_add<std::uint8_t>(255, 1), 255);
  EXPECT_EQ(saturating_subtract<std::uint8_t>(0, 1), 0);
  EXPECT_EQ(saturating_multiply<std::uint8_t>(200, 2), 255);
  EXPECT_EQ(saturating_cast<std::uint8_t>(-1), 0);
  EXPECT_EQ(saturating_cast<std::int8_t>(1000), 127);
  EXPECT_EQ(saturating_cast<std::int8_t>(-1000), -128);
  EXPECT_EQ(saturating_cast<std::int16_t>(std::uint8_t{250}), 250);
  EXPECT_EQ(saturating_cast<char16_t>(-1), char16_t{0});
  EXPECT_EQ(saturating_cast<char>(
                static_cast<int>((std::numeric_limits<char>::max)()) + 1),
            (std::numeric_limits<char>::max)());
}

TEST(SaturatingArithmeticDeathTest, DivisionByZeroIsAPrecondition) {
  EXPECT_DEATH((void)saturating_divide(1, 0), "");
}

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
  static_assert(try_add<std::int64_t>(40, 2) == 42);
  static_assert(!try_add<std::int64_t>(
                     (std::numeric_limits<std::int64_t>::max)(), 1)
                     .has_value());
  static_assert(try_subtract<std::uint64_t>(42, 2) == 40);
  static_assert(!try_subtract<std::uint64_t>(0, 1).has_value());
  static_assert(try_multiply<std::int64_t>(-7, 6) == -42);
  static_assert(!try_multiply<std::int64_t>(
                     (std::numeric_limits<std::int64_t>::min)(), -1)
                     .has_value());
  static_assert(try_integer_cast<std::uint32_t>(std::int64_t{42}) == 42u);
  static_assert(!try_integer_cast<std::uint32_t>(std::int64_t{-1}));
  static_assert(try_divide<std::int32_t>(21, -3) == -7);
  static_assert(try_remainder<std::int32_t>(22, -3) == 1);
  static_assert(try_lcm<std::int32_t>(-6, 8) == 24);
  static_assert(
      try_lcm<std::int32_t>(
          0, (std::numeric_limits<std::int32_t>::min)()) == 0);
  static_assert(!try_lcm<std::int8_t>(64, 3));
  static_assert(try_bit_ceil<std::uint32_t>(0) == 1u);
  static_assert(try_bit_ceil<std::uint32_t>(65) == 128u);
  static_assert(!try_bit_ceil<std::int32_t>(-1));
  static_assert(!try_bit_ceil<std::uint8_t>(129));
  static_assert(try_multiply_add<std::int32_t>(6, 7, 1) == 43);
  constexpr std::array values{2, 3, 4};
  static_assert(try_sum(std::span{values}) == 9);
  static_assert(try_product(std::span{values}) == 24);
  static_assert(saturating_add<std::int8_t>(127, 1) == 127);
  static_assert(saturating_subtract<std::int8_t>(-128, 1) == -128);
  static_assert(saturating_multiply<std::int8_t>(64, 2) == 127);
  static_assert(saturating_divide<std::int8_t>(-128, -1) == 127);
  static_assert(saturating_cast<std::uint8_t>(-1) == 0);
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
