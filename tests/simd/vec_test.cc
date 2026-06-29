#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <type_traits>

#include "hyle/simd/vec.h"

// vec_test -- WIDTH-AGNOSTIC: this ONE source is compiled three times (scalar / sse4 /
// avx2 targets, see tests/CMakeLists.txt) and validates whichever backend is compiled
// in. So the op tests must NOT assume a lane count: they use ScalableTag<T> + the
// runtime Lanes(d) + a naive per-lane reference. The D-tag CONTRACT tests below are
// pure compile-time tag math (backend-independent) and stay as-is.

namespace hs = hyle::simd;

// ---- D-tag contract (tag math only -- no vectors created, valid on every backend) ---

TEST(Vec, ScalableTagLaneCountMatchesNativeWidth) {
  // Lanes scale with the compiled backend's native width (16B scalar/sse4, 32B avx2).
  EXPECT_EQ(hs::Lanes(hs::ScalableTag<float>()), hs::kNativeVectorBytes / 4);
  EXPECT_EQ(hs::Lanes(hs::ScalableTag<double>()), hs::kNativeVectorBytes / 8);
  EXPECT_EQ(hs::Lanes(hs::ScalableTag<std::int32_t>()), hs::kNativeVectorBytes / 4);
}

TEST(Vec, FixedTagIsExact) {
  EXPECT_EQ(hs::Lanes(hs::FixedTag<float, 4>()), 4u);
  EXPECT_EQ(hs::MaxLanes(hs::FixedTag<std::int32_t, 8>()), 8u);
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

TEST(Vec, LanesIsConstexprForFixedWidth) {
  constexpr std::size_t n = hs::Lanes(hs::FixedTag<float, 4>());
  static_assert(n == 4);
  EXPECT_EQ(n, 4u);
}

// ---- backend ops, paired against a naive per-lane reference -------------------------
// Buffers are 64B-aligned so the aligned Load/Store path is exercised on every backend
// (scalar memcpy / SSE 16B / AVX2 32B all satisfied). Sized for the widest v1 backend.
namespace {
constexpr std::size_t kMaxLanes = 16;  // >= any v1 native width (avx2 f32 = 8)

template <typename T>
struct alignas(64) Buf {
  T data[kMaxLanes] = {};
};
}  // namespace

TEST(Vec, SetBroadcastAndStore) {
  const hs::ScalableTag<std::int32_t> d;
  const std::size_t n = hs::Lanes(d);
  auto v = hs::Set(d, 7);
  EXPECT_EQ(hs::GetLane(v), 7);
  Buf<std::int32_t> out;
  hs::Store(v, d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], 7);
}

TEST(Vec, LoadStoreRoundTrip) {
  const hs::ScalableTag<float> d;
  const std::size_t n = hs::Lanes(d);
  Buf<float> in;
  for (std::size_t i = 0; i < n; ++i) in.data[i] = static_cast<float>(i) - 2.5f;
  auto v = hs::Load(d, in.data);
  Buf<float> out;
  hs::Store(v, d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_FLOAT_EQ(out.data[i], in.data[i]);
  // unaligned path agrees too
  auto vu = hs::LoadU(d, in.data);
  hs::StoreU(vu, d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_FLOAT_EQ(out.data[i], in.data[i]);
}

// Generic arithmetic check over a lane type T with integer-exact inputs.
template <typename T>
void CheckArith() {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  Buf<T> a, b, out;
  for (std::size_t i = 0; i < n; ++i) {
    a.data[i] = static_cast<T>(i + 3);
    b.data[i] = static_cast<T>(i + 1);
  }
  auto va = hs::Load(d, a.data);
  auto vb = hs::Load(d, b.data);

  hs::Store(hs::Add(va, vb), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], static_cast<T>(a.data[i] + b.data[i]));
  hs::Store(hs::Sub(va, vb), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], static_cast<T>(a.data[i] - b.data[i]));
  hs::Store(hs::Mul(va, vb), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], static_cast<T>(a.data[i] * b.data[i]));
  hs::Store(hs::Min(va, vb), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], b.data[i]);
  hs::Store(hs::Max(va, vb), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], a.data[i]);
}

TEST(Vec, ArithmeticInt32) { CheckArith<std::int32_t>(); }
TEST(Vec, ArithmeticUint32) { CheckArith<std::uint32_t>(); }
TEST(Vec, ArithmeticFloat) { CheckArith<float>(); }
TEST(Vec, ArithmeticDouble) { CheckArith<double>(); }

TEST(Vec, NegAbsInt32) {
  const hs::ScalableTag<std::int32_t> d;
  const std::size_t n = hs::Lanes(d);
  Buf<std::int32_t> a, out;
  for (std::size_t i = 0; i < n; ++i) a.data[i] = (i % 2 ? -1 : 1) * static_cast<int>(i + 1);
  auto v = hs::Load(d, a.data);
  hs::Store(hs::Neg(v), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], -a.data[i]);
  hs::Store(hs::Abs(v), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], a.data[i] < 0 ? -a.data[i] : a.data[i]);
}

TEST(Vec, FloatDivSqrtAbs) {
  const hs::ScalableTag<float> d;
  const std::size_t n = hs::Lanes(d);
  Buf<float> a, b, out;
  for (std::size_t i = 0; i < n; ++i) {
    a.data[i] = static_cast<float>((i + 1) * (i + 1));  // perfect squares
    b.data[i] = 2.0f;
  }
  auto va = hs::Load(d, a.data), vb = hs::Load(d, b.data);
  hs::Store(hs::Div(va, vb), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_FLOAT_EQ(out.data[i], a.data[i] / 2.0f);
  hs::Store(hs::Sqrt(va), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_FLOAT_EQ(out.data[i], static_cast<float>(i + 1));
  hs::Store(hs::Abs(hs::Neg(va)), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_FLOAT_EQ(out.data[i], a.data[i]);
}

TEST(Vec, MulAddFamily) {
  const hs::ScalableTag<float> d;
  const std::size_t n = hs::Lanes(d);
  Buf<float> a, b, c, out;
  for (std::size_t i = 0; i < n; ++i) {
    a.data[i] = static_cast<float>(i + 2);
    b.data[i] = 10.0f;
    c.data[i] = 1.0f;
  }
  auto va = hs::Load(d, a.data), vb = hs::Load(d, b.data), vc = hs::Load(d, c.data);
  // FMA (avx2) single-rounds vs naive two-rounding -> tolerate a small epsilon.
  hs::Store(hs::MulAdd(va, vb, vc), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_NEAR(out.data[i], a.data[i] * b.data[i] + c.data[i], 1e-3f);
  hs::Store(hs::NegMulAdd(va, vb, vc), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_NEAR(out.data[i], -(a.data[i] * b.data[i]) + c.data[i], 1e-3f);
}

TEST(Vec, BitwiseUint32) {
  const hs::ScalableTag<std::uint32_t> d;
  const std::size_t n = hs::Lanes(d);
  Buf<std::uint32_t> a, b, out;
  for (std::size_t i = 0; i < n; ++i) {
    a.data[i] = 0xF0F0F0F0u ^ static_cast<std::uint32_t>(i * 0x01010101u);
    b.data[i] = 0x0F0F0F0Fu ^ static_cast<std::uint32_t>(i * 0x10101010u);
  }
  auto va = hs::Load(d, a.data), vb = hs::Load(d, b.data);
  hs::Store(hs::And(va, vb), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], a.data[i] & b.data[i]);
  hs::Store(hs::Or(va, vb), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], a.data[i] | b.data[i]);
  hs::Store(hs::Xor(va, vb), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], a.data[i] ^ b.data[i]);
  hs::Store(hs::AndNot(va, vb), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], ~a.data[i] & b.data[i]);  // Highway order
}

// Comparison -> mask -> blend, over a lane type T. Builds inputs where exactly the even
// lanes satisfy a > b, so the expected mask population is deterministic and width-agnostic.
template <typename T>
void CheckCompareBlend() {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  Buf<T> a, b, out;
  std::size_t expect_gt = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (i % 2 == 0) { a.data[i] = static_cast<T>(9); b.data[i] = static_cast<T>(2); ++expect_gt; }
    else { a.data[i] = static_cast<T>(1); b.data[i] = static_cast<T>(5); }
  }
  auto va = hs::Load(d, a.data), vb = hs::Load(d, b.data);
  auto m = hs::Gt(va, vb);
  EXPECT_EQ(hs::CountTrue(d, m), expect_gt);
  EXPECT_EQ(hs::FindFirstTrue(d, m), 0);  // lane 0 is even -> a>b
  EXPECT_FALSE(hs::AllTrue(d, m));
  EXPECT_FALSE(hs::AllFalse(d, m));
  EXPECT_TRUE(hs::AllTrue(d, hs::Ge(va, hs::Min(va, vb))));  // a >= min(a,b) always

  // blend: where a>b take a else b == elementwise max
  hs::Store(hs::IfThenElse(m, va, vb), d, out.data);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(out.data[i], a.data[i] > b.data[i] ? a.data[i] : b.data[i]);

  // Eq / Lt / Le / Ne consistency
  EXPECT_TRUE(hs::AllTrue(d, hs::Eq(va, va)));
  EXPECT_TRUE(hs::AllFalse(d, hs::Ne(va, va)));
  EXPECT_TRUE(hs::AllTrue(d, hs::Le(hs::Min(va, vb), hs::Max(va, vb))));
  EXPECT_TRUE(hs::AllTrue(d, hs::Lt(hs::Sub(va, hs::Set(d, static_cast<T>(0))), hs::Add(va, hs::Set(d, static_cast<T>(1))))));
}

TEST(Vec, CompareBlendInt32) { CheckCompareBlend<std::int32_t>(); }
TEST(Vec, CompareBlendUint32) { CheckCompareBlend<std::uint32_t>(); }
TEST(Vec, CompareBlendFloat) { CheckCompareBlend<float>(); }
TEST(Vec, CompareBlendDouble) { CheckCompareBlend<double>(); }

// NaN / signed-zero edge cases: must be identical across every backend. These caught a
// real divergence (red team 2026-06): AVX2/AVX512 Ne used the ordered predicate
// _CMP_NEQ_OQ (false on NaN) while scalar/SSE4/NEON return true; and AVX512 Neg used 0-a
// (Neg(+0.0)=+0.0) vs xor-signbit (-0.0) elsewhere. CheckCompareBlend feeds only finite
// values so it could not see either.
template <typename T>
void CheckFloatEdgeCases() {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  const T nan = std::numeric_limits<T>::quiet_NaN();
  Buf<T> a;
  for (std::size_t i = 0; i < n; ++i) a.data[i] = nan;
  auto vnan = hs::Load(d, a.data);
  // NaN != NaN is true on every lane; NaN == NaN is false on every lane.
  EXPECT_TRUE(hs::AllTrue(d, hs::Ne(vnan, vnan)));
  EXPECT_TRUE(hs::AllFalse(d, hs::Eq(vnan, vnan)));
  // Ne and Eq are logical complements even on NaN (the portable IfThenElse(Ne(x,x),..) idiom).
  EXPECT_EQ(hs::CountTrue(d, hs::Ne(vnan, vnan)), n);
  EXPECT_EQ(hs::CountTrue(d, hs::Eq(vnan, vnan)), 0u);

  // Neg(+0.0) == -0.0: bit pattern must have the sign bit set (1/-0.0 == -inf).
  Buf<T> zeros, out;
  for (std::size_t i = 0; i < n; ++i) zeros.data[i] = static_cast<T>(0);
  hs::Store(hs::Neg(hs::Load(d, zeros.data)), d, out.data);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_TRUE(std::signbit(out.data[i])) << "Neg(+0.0) lost the sign bit at lane " << i;
}

TEST(Vec, FloatEdgeCasesFloat) { CheckFloatEdgeCases<float>(); }
TEST(Vec, FloatEdgeCasesDouble) { CheckFloatEdgeCases<double>(); }

template <typename T>
void CheckReductions() {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  Buf<T> a;
  T sum = T{0}, mn = T{0}, mx = T{0};
  for (std::size_t i = 0; i < n; ++i) {
    // values 1..n permuted a bit so min/max aren't at the ends
    T val = static_cast<T>(((i * 7 + 3) % n) + 1);
    a.data[i] = val;
    sum = static_cast<T>(sum + val);
    if (i == 0) { mn = mx = val; }
    else { mn = val < mn ? val : mn; mx = val > mx ? val : mx; }
  }
  auto v = hs::Load(d, a.data);
  EXPECT_EQ(hs::ReduceSum(d, v), sum);
  EXPECT_EQ(hs::ReduceMin(d, v), mn);
  EXPECT_EQ(hs::ReduceMax(d, v), mx);
}

TEST(Vec, ReductionsInt32) { CheckReductions<std::int32_t>(); }
TEST(Vec, ReductionsUint32) { CheckReductions<std::uint32_t>(); }
TEST(Vec, ReductionsFloat) { CheckReductions<float>(); }
TEST(Vec, ReductionsDouble) { CheckReductions<double>(); }

// The idiomatic ScalableTag dot-product: correct regardless of the backend's lane count.
TEST(Vec, ScalableDotProduct) {
  const hs::ScalableTag<float> d;
  const std::size_t n = hs::Lanes(d);
  constexpr std::size_t kCount = 256;
  ASSERT_EQ(kCount % 8, 0u);  // divisible by any v1 native width (<=8 float lanes)
  alignas(64) float xs[kCount], ys[kCount];
  double ref = 0.0;
  for (std::size_t i = 0; i < kCount; ++i) {
    xs[i] = static_cast<float>(i % 13);
    ys[i] = static_cast<float>((2 * i + 1) % 7);
    ref += static_cast<double>(xs[i]) * ys[i];
  }
  auto acc = hs::Zero(d);
  for (std::size_t i = 0; i + n <= kCount; i += n)
    acc = hs::MulAdd(hs::Load(d, xs + i), hs::Load(d, ys + i), acc);
  EXPECT_NEAR(hs::ReduceSum(d, acc), static_cast<float>(ref), 1.0f);
}
