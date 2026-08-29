#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include "hyle/simd/native.h"

namespace hs = hyle::simd;

TEST(SimdNative, WholeRangeAlgorithmsOwnTailAndAliasRules) {
  constexpr std::size_t kCount = 37;
  std::array<std::uint32_t, kCount> generated{};
  std::array<std::uint32_t, kCount> unary{};
  std::array<std::uint32_t, kCount> binary{};
  std::array<std::uint32_t, kCount> ternary{};
  std::array<std::uint32_t, kCount> compacted{};

  hs::native::generate(generated.data(), generated.size(),
                       [](auto d, auto index) {
                         return hs::Add(index, hs::Set(d, 5u));
                       });
  for (std::size_t i = 0; i < kCount; ++i)
    EXPECT_EQ(generated[i], i + 5);

  hs::native::transform(generated.data(), unary.data(), kCount,
                        [](auto d, auto v) {
                          return hs::Add(v, hs::Set(d, 3u));
                        });
  hs::native::transform(unary.data(), kCount, [](auto d, auto v) {
    return hs::Xor(v, hs::Set(d, 7u));
  });
  hs::native::transform(generated.data(), unary.data(), binary.data(), kCount,
                        [](auto, auto a, auto b) { return hs::Add(a, b); });
  hs::native::transform(generated.data(), unary.data(), binary.data(),
                        ternary.data(), kCount,
                        [](auto, auto a, auto b, auto c) {
                          return hs::Add(hs::Add(a, b), c);
                        });
  for (std::size_t i = 0; i < kCount; ++i) {
    const std::uint32_t original = static_cast<std::uint32_t>(i + 5);
    const std::uint32_t changed = (original + 3u) ^ 7u;
    EXPECT_EQ(unary[i], changed);
    EXPECT_EQ(binary[i], original + changed);
    EXPECT_EQ(ternary[i], original + changed + binary[i]);
  }

  hs::native::replace(unary.data(), unary.size(), unary[3], 999u);
  EXPECT_EQ(unary[3], 999u);
  hs::native::replace_if(unary.data(), unary.size(), 111u,
                         [](auto d, auto v) {
                           return hs::Gt(v, hs::Set(d, 32u));
                         });
  for (std::uint32_t value : unary) EXPECT_LE(value, 111u);

  const auto divisible_by_four = [](auto d, auto v) {
    return hs::Eq(hs::And(v, hs::Set(d, 3u)), hs::Zero(d));
  };
  std::size_t expected_count = 0;
  std::size_t expected_first = kCount;
  for (std::size_t i = 0; i < kCount; ++i) {
    if ((generated[i] & 3u) == 0) {
      if (expected_first == kCount) expected_first = i;
      compacted[expected_count++] = generated[i];
    }
  }
  EXPECT_EQ(hs::native::count_if(generated.data(), kCount,
                                 divisible_by_four),
            expected_count);
  EXPECT_EQ(hs::native::find_if(generated.data(), kCount,
                                divisible_by_four),
            expected_first);

  std::array<std::uint32_t, kCount> output{};
  const std::size_t output_count = hs::native::copy_if(
      generated.data(), output.data(), kCount, divisible_by_four);
  EXPECT_EQ(output_count, expected_count);
  EXPECT_TRUE(std::equal(output.begin(), output.begin() + output_count,
                         compacted.begin()));

  // LoadN zero-fills inactive lanes. The tail mask must keep those synthetic
  // zeros out of predicate results.
  const auto is_zero = [](auto d, auto v) { return hs::Eq(v, hs::Zero(d)); };
  EXPECT_EQ(hs::native::count_if(generated.data(), kCount, is_zero), 0u);
  EXPECT_EQ(hs::native::find_if(generated.data(), kCount, is_zero), kCount);

  auto in_place = generated;
  const std::size_t in_place_count = hs::native::copy_if(
      in_place.data(), in_place.data(), kCount, divisible_by_four);
  EXPECT_EQ(in_place_count, expected_count);
  EXPECT_TRUE(std::equal(in_place.begin(), in_place.begin() + in_place_count,
                         compacted.begin()));
}
