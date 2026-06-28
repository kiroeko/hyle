#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "hyle/simd/vec.h"

namespace hs = hyle::simd;

// ---- D-tag contract -----------------------------------------------------------------

TEST(Vec, ScalableTagLaneCountMatchesNativeWidth) {
  // v1 floor emulates a 128-bit vector: 4 float lanes, 2 double, 16 uint8.
  EXPECT_EQ(hs::Lanes(hs::ScalableTag<float>()), hs::kNativeVectorBytes / 4);
  EXPECT_EQ(hs::Lanes(hs::ScalableTag<double>()), hs::kNativeVectorBytes / 8);
  EXPECT_EQ(hs::Lanes(hs::ScalableTag<std::uint8_t>()), hs::kNativeVectorBytes);
}

TEST(Vec, FixedTagIsExact) {
  EXPECT_EQ(hs::Lanes(hs::FixedTag<float, 4>()), 4u);
  EXPECT_EQ(hs::MaxLanes(hs::FixedTag<std::int32_t, 8>()), 8u);
}

TEST(Vec, CappedTagClampsToNativeAndPow2) {
  // Cap above native -> native; non-pow2 cap -> floor pow2.
  EXPECT_EQ(hs::Lanes(hs::CappedTag<float, 1024>()), hs::kNativeVectorBytes / 4);
  EXPECT_EQ(hs::Lanes(hs::CappedTag<float, 3>()), 2u);  // floor_pow2(3) = 2
}

TEST(Vec, HalfTwiceTags) {
  using D = hs::FixedTag<float, 4>;
  EXPECT_EQ(hs::MaxLanes(hs::Half<D>()), 2u);
  EXPECT_EQ(hs::MaxLanes(hs::Twice<D>()), 8u);
}

TEST(Vec, TFromDIsLaneType) {
  static_assert(std::is_same_v<hs::TFromD<hs::FixedTag<double, 2>>, double>);
  SUCCEED();
}

// Lanes() must be usable in a constexpr context for the fixed-width floor (the SVE hook
// flips this off; for v1 it must be precomputable so loop bounds fold).
TEST(Vec, LanesIsConstexprForFixedWidth) {
  constexpr std::size_t n = hs::Lanes(hs::FixedTag<float, 4>());
  static_assert(n == 4);
  EXPECT_EQ(n, 4u);
}

// ---- scalar backend ops, paired against naive per-lane reference --------------------

TEST(Vec, SetBroadcastsAndGetLane) {
  const hs::FixedTag<std::int32_t, 4> d;
  auto v = hs::Set(d, 7);
  EXPECT_EQ(hs::GetLane(v), 7);
  std::int32_t out[4];
  hs::Store(v, d, out);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(out[i], 7);
}

TEST(Vec, LoadStoreRoundTrip) {
  const hs::FixedTag<float, 4> d;
  float in[4] = {1.5f, -2.0f, 3.25f, 0.0f};
  auto v = hs::Load(d, in);
  float out[4];
  hs::Store(v, d, out);
  for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(out[i], in[i]);
}

TEST(Vec, ArithmeticMatchesScalar) {
  const hs::FixedTag<std::int32_t, 4> d;
  std::int32_t a[4] = {10, 20, 30, 40};
  std::int32_t b[4] = {1, 2, 3, 4};
  auto va = hs::Load(d, a);
  auto vb = hs::Load(d, b);
  std::int32_t out[4];

  hs::Store(hs::Add(va, vb), d, out);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(out[i], a[i] + b[i]);
  hs::Store(hs::Sub(va, vb), d, out);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(out[i], a[i] - b[i]);
  hs::Store(hs::Mul(va, vb), d, out);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(out[i], a[i] * b[i]);
  hs::Store(hs::Min(va, vb), d, out);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(out[i], b[i]);
  hs::Store(hs::Max(va, vb), d, out);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(out[i], a[i]);
}

TEST(Vec, NegAbs) {
  const hs::FixedTag<std::int32_t, 4> d;
  std::int32_t a[4] = {-3, 5, -7, 0};
  auto v = hs::Load(d, a);
  std::int32_t out[4];
  hs::Store(hs::Neg(v), d, out);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(out[i], -a[i]);
  hs::Store(hs::Abs(v), d, out);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(out[i], a[i] < 0 ? -a[i] : a[i]);
}

TEST(Vec, MulAddFamily) {
  const hs::FixedTag<float, 4> d;
  float a[4] = {2, 3, 4, 5}, b[4] = {10, 10, 10, 10}, c[4] = {1, 1, 1, 1};
  auto va = hs::Load(d, a), vb = hs::Load(d, b), vc = hs::Load(d, c);
  float out[4];
  hs::Store(hs::MulAdd(va, vb, vc), d, out);
  for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(out[i], a[i] * b[i] + c[i]);
  hs::Store(hs::NegMulAdd(va, vb, vc), d, out);
  for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(out[i], -(a[i] * b[i]) + c[i]);
}

TEST(Vec, BitwiseOnIntegers) {
  const hs::FixedTag<std::uint32_t, 4> d;
  std::uint32_t a[4] = {0xF0F0F0F0u, 0xFFFFu, 0x1u, 0xAAAAu};
  std::uint32_t b[4] = {0x0F0F0F0Fu, 0xFF00u, 0x3u, 0x5555u};
  auto va = hs::Load(d, a), vb = hs::Load(d, b);
  std::uint32_t out[4];
  hs::Store(hs::And(va, vb), d, out);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(out[i], a[i] & b[i]);
  hs::Store(hs::Or(va, vb), d, out);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(out[i], a[i] | b[i]);
  hs::Store(hs::Xor(va, vb), d, out);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(out[i], a[i] ^ b[i]);
  hs::Store(hs::AndNot(va, vb), d, out);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(out[i], ~a[i] & b[i]);  // Highway order
}

TEST(Vec, CompareToMaskAndBlend) {
  const hs::FixedTag<std::int32_t, 4> d;
  std::int32_t a[4] = {1, 5, 3, 9};
  std::int32_t b[4] = {4, 5, 2, 1};
  auto va = hs::Load(d, a), vb = hs::Load(d, b);
  auto m = hs::Gt(va, vb);  // a>b: lanes 2,3
  EXPECT_EQ(hs::CountTrue(d, m), 2u);
  EXPECT_EQ(hs::FindFirstTrue(d, m), 2);
  EXPECT_FALSE(hs::AllTrue(d, m));
  EXPECT_FALSE(hs::AllFalse(d, m));

  // blend: where a>b take a else b -> elementwise max via mask.
  std::int32_t out[4];
  hs::Store(hs::IfThenElse(m, va, vb), d, out);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(out[i], a[i] > b[i] ? a[i] : b[i]);

  auto meq = hs::Eq(va, vb);  // lane 1 only
  EXPECT_EQ(hs::CountTrue(d, meq), 1u);
  EXPECT_TRUE(hs::AllTrue(d, hs::Ge(va, hs::Set(d, -100))));
}

TEST(Vec, Reductions) {
  const hs::FixedTag<std::int32_t, 4> d;
  std::int32_t a[4] = {3, 1, 4, 1};
  auto v = hs::Load(d, a);
  EXPECT_EQ(hs::ReduceSum(d, v), 3 + 1 + 4 + 1);
  EXPECT_EQ(hs::ReduceMin(d, v), 1);
  EXPECT_EQ(hs::ReduceMax(d, v), 4);
}

// A loop written against ScalableTag (the idiomatic form) computes a correct dot
// product regardless of how many lanes the compiled backend has.
TEST(Vec, ScalableDotProduct) {
  const hs::ScalableTag<float> d;
  const std::size_t n = hs::Lanes(d);
  constexpr std::size_t kCount = 64;
  ASSERT_EQ(kCount % 16, 0u);  // divisible by any v1 native width (<=16)
  float xs[kCount], ys[kCount];
  double ref = 0.0;
  for (std::size_t i = 0; i < kCount; ++i) {
    xs[i] = static_cast<float>(i);
    ys[i] = static_cast<float>(2 * i + 1);
    ref += static_cast<double>(xs[i]) * ys[i];
  }
  auto acc = hs::Zero(d);
  for (std::size_t i = 0; i + n <= kCount; i += n) {
    acc = hs::MulAdd(hs::Load(d, xs + i), hs::Load(d, ys + i), acc);
  }
  EXPECT_FLOAT_EQ(hs::ReduceSum(d, acc), static_cast<float>(ref));
}
