#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "hyle/simd/narrow_float.h"

namespace hs = hyle::simd;

namespace {

constexpr std::size_t kMaxLanes = 64;

template <typename T>
struct alignas(64) Buffer {
  T lanes[kMaxLanes]{};
};

template <class DN>
void CheckStorageRoundTrip(DN dn) {
  const std::size_t n = hs::Lanes(dn);
  Buffer<typename DN::T> input;
  Buffer<typename DN::T> output;
  Buffer<typename DN::T> loaded;
  for (std::size_t i = 0; i < n; ++i)
    input.lanes[i] = DN::T::from_bits(static_cast<std::uint16_t>(0x9E37u * i + 0x1234u));
  hs::StoreNarrowU(hs::LoadNarrowU(dn, input.lanes), dn, output.lanes);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(output.lanes[i].bits, input.lanes[i].bits) << i;

  for (std::size_t count = 0; count <= n; ++count) {
    const auto partial = hs::LoadNarrowN(dn, input.lanes, count);
    hs::StoreNarrowU(partial, dn, loaded.lanes);
    for (std::size_t i = 0; i < count; ++i)
      EXPECT_EQ(loaded.lanes[i].bits, input.lanes[i].bits) << "count=" << count << " i=" << i;
    for (std::size_t i = count; i < n; ++i)
      EXPECT_EQ(loaded.lanes[i].bits, 0u) << "count=" << count << " i=" << i;

    for (std::size_t i = 0; i < kMaxLanes; ++i)
      output.lanes[i] = DN::T::from_bits(0xCAFEu);
    hs::StoreNarrowN(partial, dn, output.lanes, count);
    for (std::size_t i = 0; i < count; ++i)
      EXPECT_EQ(output.lanes[i].bits, input.lanes[i].bits) << "count=" << count << " i=" << i;
    for (std::size_t i = count; i < n; ++i)
      EXPECT_EQ(output.lanes[i].bits, 0xCAFEu) << "count=" << count << " i=" << i;
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

  // Every representable f16/bf16 bit pattern, including signed zero, subnormals,
  // infinities and all NaN payloads.
  for (std::uint32_t base = 0; base < 65536u; base += static_cast<std::uint32_t>(n)) {
    for (std::size_t lane = 0; lane < n; ++lane)
      narrow.lanes[lane] = T::from_bits(static_cast<std::uint16_t>(base + lane));
    hs::StoreU(hs::PromoteNarrowTo(df, dn, hs::LoadNarrowU(dn, narrow.lanes)), df,
               promoted.lanes);
    for (std::size_t lane = 0; lane < n; ++lane) {
      const float expected = to_float(narrow.lanes[lane]);
      if (std::isnan(expected))
        EXPECT_TRUE(std::isnan(promoted.lanes[lane])) << (base + lane);
      else
        EXPECT_EQ(std::bit_cast<std::uint32_t>(promoted.lanes[lane]),
                  std::bit_cast<std::uint32_t>(expected)) << (base + lane);
    }
  }

  // Deterministically spread 65,536 inputs across the entire f32 encoding space.
  // This complements the hand-picked rounding boundaries in CheckConversions.
  constexpr std::uint32_t kStride = 0x9E3779B9u;
  for (std::uint32_t base = 0; base < 65536u; base += static_cast<std::uint32_t>(n)) {
    for (std::size_t lane = 0; lane < n; ++lane)
      input.lanes[lane] = std::bit_cast<float>((base + static_cast<std::uint32_t>(lane)) * kStride);
    hs::StoreNarrowU(hs::DemoteNarrowTo(dn, hs::LoadU(df, input.lanes)), dn,
                     encoded.lanes);
    for (std::size_t lane = 0; lane < n; ++lane) {
      const T expected = from_float(input.lanes[lane]);
      if (std::isnan(input.lanes[lane]))
        EXPECT_TRUE(std::isnan(to_float(encoded.lanes[lane]))) << (base + lane);
      else
        EXPECT_EQ(encoded.lanes[lane].bits, expected.bits) << (base + lane);
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
      0x0000u, 0x8000u, 0x3C00u, 0xBC00u,
      0x0001u, 0x03FFu, 0x0400u, 0x3C01u,
      0x7BFFu, 0x7C00u, 0xFC00u, 0x7E00u,
      0x3555u, 0xB555u, 0x7FFFu, 0xFFFFu};

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
    if (std::isnan(expected)) EXPECT_TRUE(std::isnan(promoted.lanes[i])) << i;
    else EXPECT_EQ(std::bit_cast<std::uint32_t>(promoted.lanes[i]),
                   std::bit_cast<std::uint32_t>(expected)) << i;
    const auto expected_encoded = from_float(input.lanes[i]);
    if (std::isnan(input.lanes[i]))
      EXPECT_TRUE(std::isnan(to_float(encoded.lanes[i]))) << i;
    else
      EXPECT_EQ(encoded.lanes[i].bits, expected_encoded.bits) << i;
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
    EXPECT_EQ(out.lanes[i].bits, from_float(to_float(a.lanes[i]) + to_float(b.lanes[i])).bits) << i;

  hs::StoreNarrowU(hs::MulNarrow(dn, va, vb), dn, out.lanes);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(out.lanes[i].bits, from_float(to_float(a.lanes[i]) * to_float(b.lanes[i])).bits) << i;

  hs::StoreNarrowU(hs::MulAddNarrow(dn, va, vb, vc), dn, out.lanes);
  for (std::size_t i = 0; i < n; ++i) {
    const float expected = to_float(a.lanes[i]) * to_float(b.lanes[i]) + to_float(c.lanes[i]);
    EXPECT_EQ(to_float(out.lanes[i]), to_float(from_float(expected))) << i;
  }
}

}  // namespace

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
