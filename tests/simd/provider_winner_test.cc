/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary */

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "hyle/simd/provider/hyle_winners-inl.h"

namespace {

#if defined(HYLE_SIMD_TARGET_AVX2)

namespace hw = hyle::simd::provider::hyle_winner::avx2;
namespace hp = hyle::simd::provider;

template <class V, class T, std::size_t N>
std::array<T, N> lanes(V value) {
  std::array<T, N> out{};
  if constexpr (std::is_same_v<T, float>) {
    _mm256_storeu_ps(out.data(), value.raw);
  } else {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out.data()), value.raw);
  }
  return out;
}

template <class V, class T, std::size_t N>
V vector_from(const std::array<T, N>& values) {
  if constexpr (std::is_same_v<T, float>) {
    return V{_mm256_loadu_ps(values.data())};
  } else {
    return V{_mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(values.data()))};
  }
}

std::uint32_t reverse_bits_scalar(std::uint32_t value) {
  std::uint32_t reversed = 0;
  for (unsigned bit = 0; bit < 32; ++bit) {
    reversed |= ((value >> bit) & 1u) << (31u - bit);
  }
  return reversed;
}

static_assert(
    hp::kQualifiedProviderFor<
        hp::Operation::kConvertTo,
        hp::ConversionVariant<float, std::int32_t>> == hp::Provider::kHyle);
static_assert(
    hp::kQualifiedProviderFor<
        hp::Operation::kDemoteTo,
        hp::ConversionVariant<std::int32_t, std::int16_t,
                              hp::ConversionContract::kTwoInputSaturating>> ==
    hp::Provider::kHyle);
static_assert(
    hp::kQualifiedProviderFor<hp::Operation::kDiv,
                              hp::InvariantDivisorVariant<std::int32_t>> ==
    hp::Provider::kHyle);

TEST(SimdFiniteHyleWinner, Avx2ConversionsAndSignSelect) {
  const std::array<float, 8> input = {-3.75f, -2.1f, -1.0f, -0.0f,
                                      0.0f,   1.2f,  2.5f,  9.9f};
  const hw::F32 value = vector_from<hw::F32>(input);

  EXPECT_EQ((lanes<hw::I32, std::int32_t, 8>(hw::convert_to_i32(value))),
            (std::array<std::int32_t, 8>{-3, -2, -1, 0, 0, 1, 2, 9}));
  EXPECT_EQ((lanes<hw::I32, std::int32_t, 8>(hw::ceil_to_i32(value))),
            (std::array<std::int32_t, 8>{-3, -2, -1, 0, 0, 2, 3, 10}));
  EXPECT_EQ((lanes<hw::I32, std::int32_t, 8>(hw::floor_to_i32(value))),
            (std::array<std::int32_t, 8>{-4, -3, -1, 0, 0, 1, 2, 9}));

  const std::array<float, 8> round_input = {-3.4f, -2.6f, -1.4f, -0.4f,
                                            0.4f,  1.4f,  2.6f,  3.4f};
  EXPECT_EQ((lanes<hw::I32, std::int32_t, 8>(
                hw::round_to_i32(vector_from<hw::F32>(round_input)))),
            (std::array<std::int32_t, 8>{-3, -3, -1, 0, 0, 1, 3, 3}));

  const hw::MF32 mask{_mm256_castsi256_ps(
      _mm256_setr_epi32(-1, 0, -1, 0, -1, 0, -1, 0))};
  EXPECT_EQ((lanes<hw::I32, std::int32_t, 8>(
                hw::masked_convert_to_i32(mask, value))),
            (std::array<std::int32_t, 8>{-3, 0, -1, 0, 0, 0, 2, 0}));

  const std::array<float, 8> payload = {1, 2, 3, 4, 5, 6, 7, 8};
  EXPECT_EQ((lanes<hw::F32, float, 8>(hw::if_negative_then_else_zero(
                value, vector_from<hw::F32>(payload)))),
            (std::array<float, 8>{1, 2, 3, 0, 0, 0, 0, 0}));
  EXPECT_EQ((lanes<hw::F32, float, 8>(hw::if_negative_then_zero_else(
                value, vector_from<hw::F32>(payload)))),
            (std::array<float, 8>{0, 0, 0, 4, 5, 6, 7, 8}));
}

TEST(SimdFiniteHyleWinner, Avx2DemotePartialAndExpand) {
  const auto a = vector_from<hw::I32>(std::array<std::int32_t, 8>{
      -40000, -32768, -7, 0, 1, 9, 32767, 40000});
  const auto b = vector_from<hw::I32>(std::array<std::int32_t, 8>{
      10, 11, 12, 13, 14, 15, 16, 17});
  EXPECT_EQ((lanes<hw::I16, std::int16_t, 16>(
                hw::demote_i32_to_i16(a, b))),
            (std::array<std::int16_t, 16>{
                -32768, -32768, -7, 0, 1, 9, 32767, 32767,
                10, 11, 12, 13, 14, 15, 16, 17}));

  alignas(32) std::array<std::int32_t, 8> source{1, 2, 3, 4, 5, 6, 7, 8};
  EXPECT_EQ((lanes<hw::I32, std::int32_t, 8>(hw::load_n_i32(source.data(), 4))),
            (std::array<std::int32_t, 8>{1, 2, 3, 4, 0, 0, 0, 0}));
  std::array<std::int32_t, 10> destination{};
  destination.fill(99);
  hw::store_n_i32(vector_from<hw::I32>(source), destination.data() + 1, 4);
  EXPECT_EQ(destination,
            (std::array<std::int32_t, 10>{99, 1, 2, 3, 4, 99, 99, 99, 99, 99}));

  const hw::MI32 mask{
      _mm256_setr_epi32(-1, 0, -1, 0, 0, -1, -1, 0)};
  const auto packed = vector_from<hw::I32>(
      std::array<std::int32_t, 8>{10, 20, 30, 40, 50, 60, 70, 80});
  const std::array<std::int32_t, 8> expected{10, 0, 20, 0, 0, 30, 40, 0};
  EXPECT_EQ((lanes<hw::I32, std::int32_t, 8>(hw::expand_i32(packed, mask))),
            expected);
  EXPECT_EQ((lanes<hw::I32, std::int32_t, 8>(
                hw::load_expand_i32(mask, source.data()))),
            (std::array<std::int32_t, 8>{1, 0, 2, 0, 0, 3, 4, 0}));
}

TEST(SimdFiniteHyleWinner, Avx2IntegerAlgorithms) {
  const auto values = vector_from<hw::U32>(std::array<std::uint32_t, 8>{
      0u, 1u, 3u, 0xFFFFFFFFu, 0x80000000u, 0x01010101u, 0xFu,
      0x12345678u});
  EXPECT_EQ((lanes<hw::U32, std::uint32_t, 8>(
                hw::population_count_u32(values))),
            (std::array<std::uint32_t, 8>{0, 1, 2, 32, 1, 4, 4, 13}));

  const auto signed_values = vector_from<hw::I32>(
      std::array<std::int32_t, 8>{0, 1, 2, 3, 0x12345678,
                                  static_cast<std::int32_t>(0x80000000u),
                                  -1, 0x01020304});
  const auto reversed = lanes<hw::I32, std::int32_t, 8>(
      hw::reverse_bits_i32(signed_values));
  const auto original = lanes<hw::I32, std::int32_t, 8>(signed_values);
  for (std::size_t i = 0; i < original.size(); ++i) {
    EXPECT_EQ(static_cast<std::uint32_t>(reversed[i]),
              reverse_bits_scalar(
                  std::bit_cast<std::uint32_t>(original[i])));
  }

  const auto amounts = vector_from<hw::I32>(
      std::array<std::int32_t, 8>{0, 1, 2, 3, 4, 5, 6, 7});
  const auto positive = vector_from<hw::I32>(
      std::array<std::int32_t, 8>{1, 3, 7, 15, 31, 63, 127, 255});
  EXPECT_EQ((lanes<hw::I32, std::int32_t, 8>(
                hw::rounding_shr_i32(positive, amounts))),
            (std::array<std::int32_t, 8>{1, 2, 2, 2, 2, 2, 2, 2}));

  const auto a = vector_from<hw::I32>(
      std::array<std::int32_t, 8>{9, 2, 8, 3, 7, 4, 6, 5});
  const auto b = vector_from<hw::I32>(
      std::array<std::int32_t, 8>{20, 1, 30, 2, 40, 3, 50, 4});
  EXPECT_EQ((lanes<hw::I32, std::int32_t, 8>(hw::pairwise_sub128_i32(a, b))),
            (std::array<std::int32_t, 8>{7, 5, 19, 28, 3, 1, 37, 46}));
  EXPECT_EQ((lanes<hw::I32, std::int32_t, 8>(hw::pairwise_sub_i32(a, b))),
            (std::array<std::int32_t, 8>{7, 5, 3, 1, 19, 28, 37, 46}));
}

TEST(SimdFiniteHyleWinner, Avx2InvariantDivision) {
  const auto input = vector_from<hw::I32>(std::array<std::int32_t, 8>{
      std::numeric_limits<std::int32_t>::min(), -1001, -37, -1,
      0, 1, 37, 1001});
  for (const std::int32_t divisor_value : {-17, -8, -1, 1, 8, 17}) {
    const hw::InvariantDivisorI32 divisor(divisor_value);
    const auto quotient =
        lanes<hw::I32, std::int32_t, 8>(hw::divide(input, divisor));
    const auto remainder =
        lanes<hw::I32, std::int32_t, 8>(hw::modulo(input, divisor));
    const auto source = lanes<hw::I32, std::int32_t, 8>(input);
    for (std::size_t lane = 0; lane < source.size(); ++lane) {
      const std::int64_t q = static_cast<std::int64_t>(source[lane]) /
                             static_cast<std::int64_t>(divisor_value);
      const std::int64_t r = static_cast<std::int64_t>(source[lane]) -
                             q * static_cast<std::int64_t>(divisor_value);
      EXPECT_EQ(quotient[lane], static_cast<std::int32_t>(q));
      EXPECT_EQ(remainder[lane], static_cast<std::int32_t>(r));
    }
    const hw::MI32 mask{
        _mm256_setr_epi32(-1, 0, -1, 0, -1, 0, -1, 0)};
    const auto no = vector_from<hw::I32>(
        std::array<std::int32_t, 8>{9, 9, 9, 9, 9, 9, 9, 9});
    const auto masked = lanes<hw::I32, std::int32_t, 8>(
        hw::masked_modulo_or(mask, no, input, divisor));
    for (std::size_t lane = 0; lane < masked.size(); ++lane) {
      EXPECT_EQ(masked[lane], lane % 2 == 0 ? remainder[lane] : 9);
    }
  }
}

TEST(SimdFiniteHyleWinner, Avx2ComplexAndLayout) {
  const auto a = vector_from<hw::F32>(
      std::array<float, 8>{1, 2, 3, 4, -1, 0.5f, 2, -3});
  const auto b = vector_from<hw::F32>(
      std::array<float, 8>{5, 6, -2, 1, 4, -2, 0.25f, 2});
  const auto product = lanes<hw::F32, float, 8>(hw::mul_complex(a, b));
  const auto product_conj =
      lanes<hw::F32, float, 8>(hw::mul_complex_conj(a, b));
  const auto aa = lanes<hw::F32, float, 8>(a);
  const auto bb = lanes<hw::F32, float, 8>(b);
  for (std::size_t i = 0; i < 8; i += 2) {
    EXPECT_FLOAT_EQ(product[i], aa[i] * bb[i] - aa[i + 1] * bb[i + 1]);
    EXPECT_FLOAT_EQ(product[i + 1],
                    aa[i] * bb[i + 1] + aa[i + 1] * bb[i]);
    EXPECT_FLOAT_EQ(product_conj[i],
                    aa[i] * bb[i] + aa[i + 1] * bb[i + 1]);
    EXPECT_FLOAT_EQ(product_conj[i + 1],
                    aa[i + 1] * bb[i] - aa[i] * bb[i + 1]);
  }
  const hw::MF32 mask{_mm256_castsi256_ps(
      _mm256_setr_epi32(-1, -1, 0, 0, -1, -1, 0, 0))};
  const auto masked =
      lanes<hw::F32, float, 8>(hw::masked_mul_complex(mask, a, b));
  const auto masked_conj =
      lanes<hw::F32, float, 8>(hw::masked_mul_complex_conj(mask, a, b));
  for (std::size_t i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(masked[i], (i < 2 || (4 <= i && i < 6)) ? product[i] : 0);
    EXPECT_FLOAT_EQ(masked_conj[i],
                    (i < 2 || (4 <= i && i < 6)) ? product_conj[i] : 0);
  }

  const auto v0 = vector_from<hw::F32>(
      std::array<float, 8>{0, 1, 2, 3, 4, 5, 6, 7});
  const auto v1 = vector_from<hw::F32>(
      std::array<float, 8>{10, 11, 12, 13, 14, 15, 16, 17});
  const auto v2 = vector_from<hw::F32>(
      std::array<float, 8>{20, 21, 22, 23, 24, 25, 26, 27});
  const auto v3 = vector_from<hw::F32>(
      std::array<float, 8>{30, 31, 32, 33, 34, 35, 36, 37});
  std::array<float, 32> interleaved{};
  hw::store_interleaved4_f32(v0, v1, v2, v3, interleaved.data());
  for (std::size_t lane = 0; lane < 8; ++lane) {
    EXPECT_FLOAT_EQ(interleaved[4 * lane + 0], static_cast<float>(lane));
    EXPECT_FLOAT_EQ(interleaved[4 * lane + 1], static_cast<float>(10 + lane));
    EXPECT_FLOAT_EQ(interleaved[4 * lane + 2], static_cast<float>(20 + lane));
    EXPECT_FLOAT_EQ(interleaved[4 * lane + 3], static_cast<float>(30 + lane));
  }
}

TEST(SimdFiniteHyleWinner, Avx2BitShuffleAndSad) {
  const std::array<std::uint64_t, 4> source = {
      0x0123456789ABCDEFULL, 0xFEDCBA9876543210ULL,
      0xAAAAAAAA55555555ULL, 0x8000000000000001ULL};
  std::array<std::uint8_t, 32> indices{};
  for (std::size_t i = 0; i < indices.size(); ++i) {
    indices[i] = static_cast<std::uint8_t>((i * 7 + 3) & 63);
  }
  const auto shuffled = lanes<hw::U64, std::uint64_t, 4>(hw::bit_shuffle_u64(
      vector_from<hw::U64>(source), vector_from<hw::U8>(indices)));
  for (std::size_t lane = 0; lane < 4; ++lane) {
    std::uint64_t expected = 0;
    for (std::size_t bit = 0; bit < 8; ++bit) {
      expected |= ((source[lane] >> indices[lane * 8 + bit]) & 1ULL) << bit;
    }
    EXPECT_EQ(shuffled[lane], expected);
  }

  std::array<std::uint8_t, 32> left{};
  std::array<std::uint8_t, 32> right{};
  for (std::size_t i = 0; i < 32; ++i) {
    left[i] = static_cast<std::uint8_t>(i * 3);
    right[i] = static_cast<std::uint8_t>(90 - i);
  }
  const auto actual = lanes<hw::U16, std::uint16_t, 16>(
      hw::sums_of_shuffled_quad_abs_diff<3, 2, 1, 0>(
          vector_from<hw::U8>(left), vector_from<hw::U8>(right)));
  constexpr std::array<std::size_t, 4> kSourceBase{0, 2, 8, 10};
  for (std::size_t block = 0; block < 2; ++block) {
    for (std::size_t lane = 0; lane < 8; ++lane) {
      std::uint16_t expected = 0;
      const std::size_t pair = lane / 2;
      const std::size_t odd = lane % 2;
      const std::size_t a_offset =
          block * 16 + kSourceBase[pair] + odd;
      const std::size_t b_offset = block * 16 + pair * 4;
      for (std::size_t j = 0; j < 4; ++j) {
        const int delta = static_cast<int>(left[a_offset + j]) -
                          static_cast<int>(right[b_offset + j]);
        expected = static_cast<std::uint16_t>(expected + std::abs(delta));
      }
      EXPECT_EQ(actual[block * 8 + lane], expected);
    }
  }
}

#elif defined(HYLE_SIMD_TARGET_NEON)

namespace hw = hyle::simd::provider::hyle_winner::neon;
namespace hp = hyle::simd::provider;

static_assert(
    hp::kQualifiedProviderFor<
        hp::Operation::kDemoteTo,
        hp::ConversionVariant<std::int32_t, std::int16_t,
                              hp::ConversionContract::kTwoInputSaturating>> ==
    hp::Provider::kHyle);

TEST(SimdFiniteHyleWinner, NeonCompactExpandAndDemote) {
  const std::array<std::int32_t, 4> input{10, 20, 30, 40};
  const std::array<std::uint32_t, 4> mask_lanes{~0u, 0u, ~0u, 0u};
  const hw::I32 value{vld1q_s32(input.data())};
  const hw::MI32 mask{vld1q_u32(mask_lanes.data())};
  std::array<std::int32_t, 4> compact{};
  EXPECT_EQ(hw::compress_blended_store_i32(value, mask, compact.data()), 2u);
  EXPECT_EQ(compact[0], 10);
  EXPECT_EQ(compact[1], 30);

  std::array<std::int32_t, 4> expanded{};
  vst1q_s32(expanded.data(), hw::expand_i32(value, mask).raw);
  EXPECT_EQ(expanded,
            (std::array<std::int32_t, 4>{10, 0, 20, 0}));
  vst1q_s32(expanded.data(), hw::load_expand_i32(mask, compact.data()).raw);
  EXPECT_EQ(expanded,
            (std::array<std::int32_t, 4>{10, 0, 30, 0}));

  const std::array<std::int32_t, 4> wide0_lanes{-40000, -7, 32767,
                                                 40000};
  const std::array<std::int32_t, 4> wide1_lanes{1, 2, 3, 4};
  const hw::I32 wide0{vld1q_s32(wide0_lanes.data())};
  const hw::I32 wide1{vld1q_s32(wide1_lanes.data())};
  std::array<std::int16_t, 8> narrow{};
  vst1q_s16(narrow.data(), hw::demote_i32_to_i16(wide0, wide1).raw);
  EXPECT_EQ(narrow, (std::array<std::int16_t, 8>{
                        -32768, -7, 32767, 32767, 1, 2, 3, 4}));
}

TEST(SimdFiniteHyleWinner, NeonCompressNotAndStreamLifecycle) {
  const std::array<std::uint64_t, 2> input{11, 22};
  const std::array<std::uint64_t, 2> mask_lanes{~0ULL, 0ULL};
  const hw::U64 value{vld1q_u64(input.data())};
  const hw::MU64 mask{vld1q_u64(mask_lanes.data())};
  std::array<std::uint64_t, 2> output{};
  vst1q_u64(output.data(), hw::compress_not_u64(value, mask).raw);
  EXPECT_EQ(output[0], 22u);

  const ::hwy::HWY_NAMESPACE::ScalableTag<std::uint64_t> d;
  alignas(16) std::array<std::uint64_t, 2> streamed{};
  hw::stream(value, d, streamed.data());
  hw::flush_stream();
  EXPECT_EQ(streamed, (std::array<std::uint64_t, 2>{11, 22}));
}

#endif

}  // namespace
