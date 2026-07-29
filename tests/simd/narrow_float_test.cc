#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "hyle/simd/narrow_algorithm.h"

namespace hs = hyle::simd;

namespace {

constexpr std::size_t kMaxLanes = 64;

template <typename T> struct alignas(64) Buffer {
  T lanes[kMaxLanes]{};
};

template <class DN> void CheckStorageRoundTrip(DN dn) {
  const std::size_t n = hs::Lanes(dn);
  Buffer<typename DN::T> input;
  Buffer<typename DN::T> output;
  Buffer<typename DN::T> loaded;
  for (std::size_t i = 0; i < n; ++i)
    input.lanes[i] =
        DN::T::from_bits(static_cast<std::uint16_t>(0x9E37u * i + 0x1234u));
  hs::StoreNarrowU(hs::LoadNarrowU(dn, input.lanes), dn, output.lanes);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(output.lanes[i].bits, input.lanes[i].bits) << i;

  for (std::size_t count = 0; count <= n; ++count) {
    const auto partial = hs::LoadNarrowN(dn, input.lanes, count);
    hs::StoreNarrowU(partial, dn, loaded.lanes);
    for (std::size_t i = 0; i < count; ++i)
      EXPECT_EQ(loaded.lanes[i].bits, input.lanes[i].bits)
          << "count=" << count << " i=" << i;
    for (std::size_t i = count; i < n; ++i)
      EXPECT_EQ(loaded.lanes[i].bits, 0u) << "count=" << count << " i=" << i;

    for (std::size_t i = 0; i < kMaxLanes; ++i)
      output.lanes[i] = DN::T::from_bits(0xCAFEu);
    hs::StoreNarrowN(partial, dn, output.lanes, count);
    for (std::size_t i = 0; i < count; ++i)
      EXPECT_EQ(output.lanes[i].bits, input.lanes[i].bits)
          << "count=" << count << " i=" << i;
    for (std::size_t i = count; i < n; ++i)
      EXPECT_EQ(output.lanes[i].bits, 0xCAFEu)
          << "count=" << count << " i=" << i;
  }
}

template <class DN, class ToFloat, class FromFloat>
void CheckConversionSpace(DN dn, ToFloat to_float, FromFloat from_float) {
  using T = typename DN::T;
  using DF = typename DN::FloatTag;
  const DF df;
  const std::size_t n = hs::Lanes(df);
  Buffer<T> narrow;
  Buffer<T> encoded;
  Buffer<float> promoted;
  Buffer<float> input;

  // Every representable f16/bf16 bit pattern, including signed zero,
  // subnormals, infinities and all NaN payloads.
  for (std::uint32_t base = 0; base < 65536u;
       base += static_cast<std::uint32_t>(n)) {
    for (std::size_t lane = 0; lane < n; ++lane)
      narrow.lanes[lane] =
          T::from_bits(static_cast<std::uint16_t>(base + lane));
    hs::StoreU(hs::PromoteNarrowTo(df, dn, hs::LoadNarrowU(dn, narrow.lanes)),
               df, promoted.lanes);
    for (std::size_t lane = 0; lane < n; ++lane) {
      const float expected = to_float(narrow.lanes[lane]);
      if (std::isnan(expected)) {
        EXPECT_TRUE(std::isnan(promoted.lanes[lane])) << (base + lane);
      } else {
        EXPECT_EQ(std::bit_cast<std::uint32_t>(promoted.lanes[lane]),
                  std::bit_cast<std::uint32_t>(expected))
            << (base + lane);
      }
    }
  }

  // Deterministically spread 65,536 inputs across the entire f32 encoding
  // space. This complements the hand-picked rounding boundaries in
  // CheckConversions.
  constexpr std::uint32_t kStride = 0x9E3779B9u;
  for (std::uint32_t base = 0; base < 65536u;
       base += static_cast<std::uint32_t>(n)) {
    for (std::size_t lane = 0; lane < n; ++lane)
      input.lanes[lane] = std::bit_cast<float>(
          (base + static_cast<std::uint32_t>(lane)) * kStride);
    hs::StoreNarrowU(hs::DemoteNarrowTo(dn, hs::LoadU(df, input.lanes)), dn,
                     encoded.lanes);
    for (std::size_t lane = 0; lane < n; ++lane) {
      const T expected = from_float(input.lanes[lane]);
      if (std::isnan(input.lanes[lane])) {
        EXPECT_TRUE(std::isnan(to_float(encoded.lanes[lane]))) << (base + lane);
      } else {
        EXPECT_EQ(encoded.lanes[lane].bits, expected.bits) << (base + lane);
      }
    }
  }
}

template <class DN, class ToFloat, class FromFloat>
void CheckConversions(DN dn, ToFloat to_float, FromFloat from_float) {
  using T = typename DN::T;
  using DF = typename DN::FloatTag;
  const DF df;
  const std::size_t n = hs::Lanes(df);
  Buffer<T> narrow;
  Buffer<T> encoded;
  Buffer<float> promoted;
  Buffer<float> input;

  constexpr std::array<std::uint32_t, 16> kF32Bits = {
      0x00000000u, 0x80000000u, 0x3F800000u, 0xBF800000u,
      0x33800000u, 0x33000000u, 0x387FC000u, 0x38800000u,
      0x477FE000u, 0x47800000u, 0x7F7FFFFFu, 0x7F800000u,
      0xFF800000u, 0x7FC12345u, 0x00000001u, 0x00800000u};
  constexpr std::array<std::uint16_t, 16> kNarrowBits = {
      0x0000u, 0x8000u, 0x3C00u, 0xBC00u, 0x0001u, 0x03FFu, 0x0400u, 0x3C01u,
      0x7BFFu, 0x7C00u, 0xFC00u, 0x7E00u, 0x3555u, 0xB555u, 0x7FFFu, 0xFFFFu};

  for (std::size_t i = 0; i < n; ++i) {
    narrow.lanes[i] = T::from_bits(kNarrowBits[i % kNarrowBits.size()]);
    input.lanes[i] = std::bit_cast<float>(kF32Bits[i % kF32Bits.size()]);
  }

  hs::StoreU(hs::PromoteNarrowTo(df, dn, hs::LoadNarrowU(dn, narrow.lanes)), df,
             promoted.lanes);
  hs::StoreNarrowU(hs::DemoteNarrowTo(dn, hs::LoadU(df, input.lanes)), dn,
                   encoded.lanes);

  for (std::size_t i = 0; i < n; ++i) {
    const float expected = to_float(narrow.lanes[i]);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(promoted.lanes[i])) << i;
    } else {
      EXPECT_EQ(std::bit_cast<std::uint32_t>(promoted.lanes[i]),
                std::bit_cast<std::uint32_t>(expected))
          << i;
    }
    const auto expected_encoded = from_float(input.lanes[i]);
    if (std::isnan(input.lanes[i])) {
      EXPECT_TRUE(std::isnan(to_float(encoded.lanes[i]))) << i;
    } else {
      EXPECT_EQ(encoded.lanes[i].bits, expected_encoded.bits) << i;
    }
  }
}

template <class DN, class ToFloat, class FromFloat>
void CheckArithmetic(DN dn, ToFloat to_float, FromFloat from_float) {
  const std::size_t n = hs::Lanes(dn);
  Buffer<typename DN::T> a;
  Buffer<typename DN::T> b;
  Buffer<typename DN::T> c;
  Buffer<typename DN::T> out;
  for (std::size_t i = 0; i < n; ++i) {
    a.lanes[i] = from_float(static_cast<float>(i + 1) * 0.5f);
    b.lanes[i] = from_float(static_cast<float>((i % 3) + 1));
    c.lanes[i] = from_float(0.25f);
  }
  const auto va = hs::LoadNarrowU(dn, a.lanes);
  const auto vb = hs::LoadNarrowU(dn, b.lanes);
  const auto vc = hs::LoadNarrowU(dn, c.lanes);

  hs::StoreNarrowU(hs::AddNarrow(dn, va, vb), dn, out.lanes);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(out.lanes[i].bits,
              from_float(to_float(a.lanes[i]) + to_float(b.lanes[i])).bits)
        << i;

  hs::StoreNarrowU(hs::MulNarrow(dn, va, vb), dn, out.lanes);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(out.lanes[i].bits,
              from_float(to_float(a.lanes[i]) * to_float(b.lanes[i])).bits)
        << i;

  hs::StoreNarrowU(hs::MulAddNarrow(dn, va, vb, vc), dn, out.lanes);
  for (std::size_t i = 0; i < n; ++i) {
    const float expected =
        to_float(a.lanes[i]) * to_float(b.lanes[i]) + to_float(c.lanes[i]);
    EXPECT_EQ(to_float(out.lanes[i]), to_float(from_float(expected))) << i;
  }
}

template <typename T>
void CheckScalarTotalOrder(std::array<std::uint16_t, 8> landmarks) {
  for (std::uint32_t bits = 0; bits < 65536u; ++bits) {
    const T value = T::from_bits(static_cast<std::uint16_t>(bits));
    const std::uint16_t key = hs::narrow_total_order_key(value);
    EXPECT_EQ(hs::narrow_from_total_order_key<T>(key).bits, bits);
  }
  for (std::size_t i = 1; i < landmarks.size(); ++i) {
    EXPECT_LT(hs::narrow_total_order_key(T::from_bits(landmarks[i - 1])),
              hs::narrow_total_order_key(T::from_bits(landmarks[i])));
  }
}

template <class DN, class FromFloat>
void CheckCompareAndReduce(DN dn, FromFloat from_float,
                           std::uint16_t negative_qnan,
                           std::uint16_t positive_qnan,
                           std::uint16_t canonical_qnan) {
  using T = typename DN::T;
  using DF = typename DN::FloatTag;
  const DF df;
  const typename DN::StorageTag db;
  const std::size_t n = hs::Lanes(dn);
  Buffer<T> a;
  Buffer<T> b;
  Buffer<T> ordered;
  Buffer<T> restored;
  Buffer<std::uint16_t> keys;

  for (std::size_t i = 0; i < n; ++i) {
    a.lanes[i] = T::from_bits(0x0000u);
    b.lanes[i] = T::from_bits(0x8000u);
    ordered.lanes[i] =
        from_float(static_cast<float>(static_cast<int>(i % 7) - 3));
  }
  const auto va = hs::LoadNarrowU(dn, a.lanes);
  const auto vb = hs::LoadNarrowU(dn, b.lanes);
  EXPECT_EQ(hs::CountTrue(df, hs::EqNarrow(dn, va, vb)), n);
  EXPECT_EQ(hs::CountTrue(db, hs::EqBitsNarrow(dn, va, vb)), 0u);
  EXPECT_EQ(hs::CountTrue(db, hs::LtTotalOrderNarrow(dn, vb, va)), n);

  const auto vo = hs::LoadNarrowU(dn, ordered.lanes);
  const auto vk = hs::TotalOrderKeyNarrow(dn, vo);
  hs::StoreU(vk, db, keys.lanes);
  hs::StoreNarrowU(hs::NarrowFromTotalOrderKey(dn, vk), dn, restored.lanes);
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(keys.lanes[i], hs::narrow_total_order_key(ordered.lanes[i])) << i;
    EXPECT_EQ(restored.lanes[i].bits, ordered.lanes[i].bits) << i;
  }

  const T min_expected = from_float(-3.0f);
  float max_value = -3.0f;
  float sum_expected = 0.0f;
  for (std::size_t i = 0; i < n; ++i) {
    const float value = static_cast<float>(static_cast<int>(i % 7) - 3);
    sum_expected += value;
    if (value > max_value)
      max_value = value;
  }
  const T max_expected = from_float(max_value);
  EXPECT_EQ(hs::ReduceSumNarrow(dn, vo), sum_expected);
  EXPECT_EQ(hs::ReduceMinNarrow(dn, vo).bits, min_expected.bits);
  EXPECT_EQ(hs::ReduceMaxNarrow(dn, vo).bits, max_expected.bits);
  EXPECT_EQ(hs::ReduceMinNumberNarrow(dn, vo).bits, min_expected.bits);
  EXPECT_EQ(hs::ReduceMaxNumberNarrow(dn, vo).bits, max_expected.bits);

  for (std::size_t i = 0; i < n; ++i)
    ordered.lanes[i] = T::from_bits((i & 1u) == 0 ? 0x0000u : 0x8000u);
  const auto signed_zero = hs::LoadNarrowU(dn, ordered.lanes);
  EXPECT_EQ(hs::ReduceMinNarrow(dn, signed_zero).bits, 0x8000u);
  EXPECT_EQ(hs::ReduceMaxNarrow(dn, signed_zero).bits, 0x0000u);
  EXPECT_EQ(hs::ReduceMinNumberNarrow(dn, signed_zero).bits, 0x8000u);
  EXPECT_EQ(hs::ReduceMaxNumberNarrow(dn, signed_zero).bits, 0x0000u);

  for (std::size_t i = 0; i < n; ++i)
    ordered.lanes[i] = from_float(static_cast<float>(i + 1));
  ordered.lanes[0] = T::from_bits(negative_qnan);
  if (n > 1)
    ordered.lanes[1] = T::from_bits(positive_qnan);
  const auto with_nan = hs::LoadNarrowU(dn, ordered.lanes);
  EXPECT_EQ(hs::ReduceMinNarrow(dn, with_nan).bits, canonical_qnan);
  EXPECT_EQ(hs::ReduceMaxNarrow(dn, with_nan).bits, canonical_qnan);
  EXPECT_EQ(hs::ReduceMinTotalOrderNarrow(dn, with_nan).bits, negative_qnan);
  if (n > 1) {
    EXPECT_EQ(hs::ReduceMaxTotalOrderNarrow(dn, with_nan).bits, positive_qnan);
  }
  EXPECT_EQ(hs::ReduceMinNumberNarrow(dn, with_nan).bits,
            from_float(n > 1 ? 3.0f : 2.0f).bits);
  EXPECT_EQ(hs::ReduceMaxNumberNarrow(dn, with_nan).bits,
            from_float(static_cast<float>(n)).bits);

  for (std::size_t i = 0; i < n; ++i)
    ordered.lanes[i] = T::from_bits(positive_qnan);
  const auto all_nan = hs::LoadNarrowU(dn, ordered.lanes);
  EXPECT_EQ(hs::ReduceMinNumberNarrow(dn, all_nan).bits, canonical_qnan);
  EXPECT_EQ(hs::ReduceMaxNumberNarrow(dn, all_nan).bits, canonical_qnan);
}

template <class DN, class FromFloat>
void CheckRangeAlgorithms(DN dn, FromFloat from_float,
                          std::uint16_t negative_qnan,
                          std::uint16_t positive_qnan,
                          std::uint16_t positive_inf) {
  using T = typename DN::T;
  const std::size_t width = hs::Lanes(dn);
  const std::size_t size = 4 * width + 3;
  std::vector<T> values(size);
  float expected_sum = 0.0f;
  for (std::size_t i = 0; i < size; ++i) {
    const float value = static_cast<float>(static_cast<int>(i % 7) - 3);
    values[i] = from_float(value);
    expected_sum += value;
  }

  EXPECT_EQ(hs::SumNarrow(dn, values.data(), size), expected_sum);
  EXPECT_EQ(hs::CountEqualNarrow(dn, values.data(), size, from_float(2.0f)),
            (size + 1) / 7);
  const std::size_t first_two =
      hs::FindEqualNarrow(dn, values.data(), size, from_float(2.0f));
  ASSERT_LT(first_two, size);
  EXPECT_EQ(values[first_two].bits, from_float(2.0f).bits);

  std::array<T, 8> special = {T::from_bits(negative_qnan),
                              from_float(-2.0f),
                              T::from_bits(0x8000u),
                              T::from_bits(0x0000u),
                              from_float(3.0f),
                              T::from_bits(positive_inf),
                              T::from_bits(positive_qnan),
                              T::from_bits(positive_qnan)};
  EXPECT_EQ(hs::CountEqualNarrow(dn, special.data(), special.size(),
                                 T::from_bits(0x0000u)),
            2u);
  EXPECT_EQ(hs::FindEqualNarrow(dn, special.data(), special.size(),
                                T::from_bits(0x0000u)),
            2u);
  EXPECT_EQ(hs::CountBitsNarrow(dn, special.data(), special.size(),
                                T::from_bits(0x0000u)),
            1u);
  EXPECT_EQ(hs::FindBitsNarrow(dn, special.data(), special.size(),
                               T::from_bits(0x0000u)),
            3u);
  EXPECT_EQ(hs::CountEqualNarrow(dn, special.data(), special.size(),
                                 T::from_bits(positive_qnan)),
            0u);
  EXPECT_EQ(hs::FindEqualNarrow(dn, special.data(), special.size(),
                                T::from_bits(positive_qnan)),
            special.size());
  EXPECT_EQ(hs::CountBitsNarrow(dn, special.data(), special.size(),
                                T::from_bits(positive_qnan)),
            2u);
  EXPECT_EQ(hs::FindBitsNarrow(dn, special.data(), special.size(),
                               T::from_bits(positive_qnan)),
            6u);

  std::array<T, 8> total_sorted = special;
  std::sort(total_sorted.begin(), total_sorted.end(), [](T a, T b) {
    return hs::narrow_total_order_key(a) < hs::narrow_total_order_key(b);
  });
  EXPECT_TRUE(hs::IsSortedTotalOrderNarrow(dn, total_sorted.data(),
                                           total_sorted.size()));
  EXPECT_FALSE(hs::IsSortedTotalOrderNarrow(dn, total_sorted.data(),
                                            total_sorted.size(), false));
  std::reverse(total_sorted.begin(), total_sorted.end());
  EXPECT_TRUE(hs::IsSortedTotalOrderNarrow(dn, total_sorted.data(),
                                           total_sorted.size(), false));
}

} // namespace

TEST(NarrowFloat, TagsAndStorage) {
  using DF = hs::ScalableTag<float>;
  const hs::F16Tag<DF> dh;
  const hs::BF16Tag<DF> db;
  EXPECT_EQ(hs::Lanes(dh), hs::Lanes(DF{}));
  EXPECT_EQ(hs::Lanes(db), hs::Lanes(DF{}));
  EXPECT_EQ(hs::MaxLanes(dh), hs::MaxLanes(DF{}));
  EXPECT_EQ(hs::MaxLanes(db), hs::MaxLanes(DF{}));
  CheckStorageRoundTrip(dh);
  CheckStorageRoundTrip(db);
}

TEST(NarrowFloat, TotalOrderScalarExhaustive) {
  CheckScalarTotalOrder<hs::float16_t>(
      {0xFE00u, 0xFC01u, 0xFC00u, 0x8000u, 0x0000u, 0x7C00u, 0x7C01u, 0x7E00u});
  CheckScalarTotalOrder<hs::bfloat16_t>(
      {0xFFC0u, 0xFF81u, 0xFF80u, 0x8000u, 0x0000u, 0x7F80u, 0x7F81u, 0x7FC0u});
}

TEST(NarrowFloat, F16ConvertAndCompute) {
  using DF = hs::ScalableTag<float>;
  const hs::F16Tag<DF> d;
  CheckConversions(d, hs::f16_to_f32, hs::f32_to_f16);
  CheckConversionSpace(d, hs::f16_to_f32, hs::f32_to_f16);
  CheckArithmetic(d, hs::f16_to_f32, hs::f32_to_f16);
}

TEST(NarrowFloat, BF16ConvertAndCompute) {
  using DF = hs::ScalableTag<float>;
  const hs::BF16Tag<DF> d;
  CheckConversions(d, hs::bf16_to_f32, hs::f32_to_bf16);
  CheckConversionSpace(d, hs::bf16_to_f32, hs::f32_to_bf16);
  CheckArithmetic(d, hs::bf16_to_f32, hs::f32_to_bf16);
}

TEST(NarrowFloat, CompareAndReduce) {
  using DF = hs::ScalableTag<float>;
  CheckCompareAndReduce(hs::F16Tag<DF>{}, hs::f32_to_f16, 0xFE00u, 0x7E00u,
                        0x7E00u);
  CheckCompareAndReduce(hs::BF16Tag<DF>{}, hs::f32_to_bf16, 0xFFC0u, 0x7FC0u,
                        0x7FC0u);
}

TEST(NarrowFloat, RangeAlgorithms) {
  using DF = hs::ScalableTag<float>;
  CheckRangeAlgorithms(hs::F16Tag<DF>{}, hs::f32_to_f16, 0xFE00u, 0x7E00u,
                       0x7C00u);
  CheckRangeAlgorithms(hs::BF16Tag<DF>{}, hs::f32_to_bf16, 0xFFC0u, 0x7FC0u,
                       0x7F80u);
}
