#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <bit>
#include <limits>
#include <type_traits>
#include <vector>
#include <algorithm>

#include "hyle/simd/vec.h"
#include "hyle/simd/dtype.h"
#include "hyle/simd/random.h"

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
constexpr std::size_t kMaxLanes = 64;  // >= widest lane count (AVX-512 i8 = 64; avx2 i8 = 32)

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
TEST(Vec, ArithmeticInt8) { CheckArith<std::int8_t>(); }
TEST(Vec, ArithmeticUint8) { CheckArith<std::uint8_t>(); }
TEST(Vec, ArithmeticInt16) { CheckArith<std::int16_t>(); }
TEST(Vec, ArithmeticUint16) { CheckArith<std::uint16_t>(); }
TEST(Vec, ArithmeticInt64) { CheckArith<std::int64_t>(); }
TEST(Vec, ArithmeticUint64) { CheckArith<std::uint64_t>(); }

// Neg/Abs over each signed integer width (exercises the per-width intrinsics + i64 emul).
template <typename T>
void CheckNegAbs() {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  Buf<T> a, out;
  for (std::size_t i = 0; i < n; ++i)
    a.data[i] = static_cast<T>((i % 2 ? -1 : 1) * static_cast<int>(i % 40 + 1));
  auto v = hs::Load(d, a.data);
  hs::Store(hs::Neg(v), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], static_cast<T>(-a.data[i]));
  hs::Store(hs::Abs(v), d, out.data);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(out.data[i], a.data[i] < 0 ? static_cast<T>(-a.data[i]) : a.data[i]);
}
TEST(Vec, NegAbsInt8) { CheckNegAbs<std::int8_t>(); }
TEST(Vec, NegAbsInt16) { CheckNegAbs<std::int16_t>(); }
TEST(Vec, NegAbsInt64) { CheckNegAbs<std::int64_t>(); }

// Shifts: compile-time immediate count, per integer width. ShiftRight is arithmetic for
// signed (sign-extending) and logical for unsigned -- checked against the C++ operator on
// the lane type (the scalar backend IS that operator, so this validates the wide paths).
template <typename T, int kBits>
void CheckShift() {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  Buf<T> a, out;
  for (std::size_t i = 0; i < n; ++i)
    a.data[i] = static_cast<T>((i % 2 ? -1 : 1) * static_cast<long long>(i * 7 + 1));
  auto v = hs::Load(d, a.data);
  hs::Store(hs::template ShiftLeft<kBits>(v), d, out.data);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(out.data[i],
              static_cast<T>(static_cast<std::make_unsigned_t<T>>(a.data[i]) << kBits))
        << "shl i=" << i;
  hs::Store(hs::template ShiftRight<kBits>(v), d, out.data);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(out.data[i], static_cast<T>(a.data[i] >> kBits)) << "shr i=" << i;
}
TEST(Vec, ShiftInt8) { CheckShift<std::int8_t, 3>(); }
TEST(Vec, ShiftUint8) { CheckShift<std::uint8_t, 3>(); }
TEST(Vec, ShiftInt16) { CheckShift<std::int16_t, 5>(); }
TEST(Vec, ShiftUint16) { CheckShift<std::uint16_t, 5>(); }
TEST(Vec, ShiftInt32) { CheckShift<std::int32_t, 10>(); }
TEST(Vec, ShiftUint32) { CheckShift<std::uint32_t, 10>(); }
TEST(Vec, ShiftInt64) { CheckShift<std::int64_t, 20>(); }
TEST(Vec, ShiftUint64) { CheckShift<std::uint64_t, 20>(); }

// Saturated add/sub (8/16-bit): clamp to [min,max] instead of wrapping. Feed values near
// the type limits so saturation actually triggers, vs a wide-int clamped reference.
template <typename T>
void CheckSaturated() {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  constexpr long long lo = static_cast<long long>(std::numeric_limits<T>::min());
  constexpr long long hi = static_cast<long long>(std::numeric_limits<T>::max());
  Buf<T> a, b, out;
  for (std::size_t i = 0; i < n; ++i) {
    a.data[i] = static_cast<T>(hi - static_cast<long long>(i % 5));       // near max
    b.data[i] = static_cast<T>((i % 3) + 2);                             // small positive
  }
  auto va = hs::Load(d, a.data), vb = hs::Load(d, b.data);
  auto clamp = [](long long x) { return static_cast<T>(x < lo ? lo : (x > hi ? hi : x)); };
  hs::Store(hs::SaturatedAdd(va, vb), d, out.data);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(out.data[i], clamp(static_cast<long long>(a.data[i]) + b.data[i])) << "satadd i=" << i;
  hs::Store(hs::SaturatedSub(vb, va), d, out.data);  // small - big -> saturates low
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(out.data[i], clamp(static_cast<long long>(b.data[i]) - a.data[i])) << "satsub i=" << i;
}
TEST(Vec, SaturatedInt8) { CheckSaturated<std::int8_t>(); }
TEST(Vec, SaturatedUint8) { CheckSaturated<std::uint8_t>(); }
TEST(Vec, SaturatedInt16) { CheckSaturated<std::int16_t>(); }
TEST(Vec, SaturatedUint16) { CheckSaturated<std::uint16_t>(); }

// Rounding average (u8/u16): (a+b+1)>>1.
template <typename T>
void CheckAverage() {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  Buf<T> a, b, out;
  for (std::size_t i = 0; i < n; ++i) {
    a.data[i] = static_cast<T>(i * 3 + 1);
    b.data[i] = static_cast<T>(i * 2 + 4);
  }
  auto va = hs::Load(d, a.data), vb = hs::Load(d, b.data);
  hs::Store(hs::AverageRound(va, vb), d, out.data);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(out.data[i], static_cast<T>((static_cast<unsigned>(a.data[i]) + b.data[i] + 1u) >> 1))
        << "avg i=" << i;
}
TEST(Vec, AverageUint8) { CheckAverage<std::uint8_t>(); }
TEST(Vec, AverageUint16) { CheckAverage<std::uint16_t>(); }

// Type conversion f32<->i32 (same lane count). ConvertTo truncates toward zero;
// RoundToInt rounds to nearest-even. The quantization (de)quant core.
TEST(Vec, ConvertF32I32) {
  const hs::ScalableTag<float> df;
  const hs::ScalableTag<std::int32_t> di;
  const std::size_t n = hs::Lanes(df);
  ASSERT_EQ(n, hs::Lanes(di));  // f32 and i32 have the same lane count on every backend
  Buf<float> f;
  for (std::size_t i = 0; i < n; ++i)
    f.data[i] = (i % 2 ? -1.0f : 1.0f) * (static_cast<float>(i) + 0.4f);  // .4 -> trunc/round differ near .5 avoided
  auto vf = hs::Load(df, f.data);
  Buf<std::int32_t> oi;
  hs::Store(hs::ConvertTo(di, vf), di, oi.data);  // truncate toward zero
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(oi.data[i], static_cast<std::int32_t>(f.data[i])) << "cvtt i=" << i;
  hs::Store(hs::RoundToInt(di, vf), di, oi.data);  // round to nearest
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(oi.data[i], static_cast<std::int32_t>(std::nearbyint(f.data[i]))) << "round i=" << i;
  // i32 -> f32 round trip of exact integers
  Buf<std::int32_t> ii;
  for (std::size_t i = 0; i < n; ++i) ii.data[i] = static_cast<std::int32_t>(i) * 3 - 5;
  auto vi = hs::Load(di, ii.data);
  Buf<float> of;
  hs::Store(hs::ConvertTo(df, vi), df, of.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_FLOAT_EQ(of.data[i], static_cast<float>(ii.data[i]));
}

// BitCast: reinterpret bits, no value conversion. Same-width case (f32<->u32, same lane
// count on every backend): each lane's bits must equal std::bit_cast. Distinct from
// ConvertTo (which changes the numeric value). Then a width-CHANGING case (i32->i8, lane
// count 4x) checks the total byte stream is preserved (the low-level use gather/mask rely on).
TEST(Vec, BitCastSameWidth) {
  const hs::ScalableTag<float> df;
  const hs::ScalableTag<std::uint32_t> du;
  const std::size_t n = hs::Lanes(df);
  ASSERT_EQ(n, hs::Lanes(du));
  Buf<float> f;
  for (std::size_t i = 0; i < n; ++i) f.data[i] = (i % 2 ? -1.0f : 1.0f) * (static_cast<float>(i) * 1.5f + 0.25f);
  auto vf = hs::Load(df, f.data);
  Buf<std::uint32_t> ou;
  hs::Store(hs::BitCast(du, vf), du, ou.data);  // bits, not value
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(ou.data[i], std::bit_cast<std::uint32_t>(f.data[i])) << "bitcast f32->u32 i=" << i;
  // round trip back to float reproduces the exact bit pattern
  auto vu = hs::Load(du, ou.data);
  Buf<float> of;
  hs::Store(hs::BitCast(df, vu), df, of.data);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(std::bit_cast<std::uint32_t>(of.data[i]), std::bit_cast<std::uint32_t>(f.data[i])) << i;
}

TEST(Vec, BitCastWidthChanging) {
  // i32 vector -> i8 vector: same total bytes, 4x the lane count. The byte stream must be
  // identical (little-endian lane layout), which is what index/mask bit-tricks depend on.
  const hs::ScalableTag<std::int32_t> di;
  const std::size_t n32 = hs::Lanes(di);
  Buf<std::int32_t> in;
  for (std::size_t i = 0; i < n32; ++i) in.data[i] = static_cast<std::int32_t>(0x01020300u + i);
  auto vi = hs::Load(di, in.data);
  // dst tag: i8 with 4x lanes (same total bytes). FixedTag pins the exact width.
  const hs::FixedTag<std::int8_t, hs::kNativeVectorBytes> d8;  // kNativeVectorBytes lanes of i8 = full reg
  Buf<std::int8_t> o8;
  hs::Store(hs::BitCast(d8, vi), d8, o8.data);
  // Reference: reinterpret the i32 buffer's bytes.
  const auto* ref = reinterpret_cast<const std::int8_t*>(in.data);
  for (std::size_t b = 0; b < n32 * sizeof(std::int32_t); ++b)
    EXPECT_EQ(o8.data[b], ref[b]) << "byte " << b;
}

// ---- A-3 FP classification ----------------------------------------------------------
// Feed a mix of NaN / +Inf / -Inf / finite lanes and check IsNaN/IsInf/IsFinite/IsEitherNaN
// against the std references. Float lanes only.
template <typename T>
void CheckFpClassify() {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  const T qnan = std::numeric_limits<T>::quiet_NaN();
  const T inf = std::numeric_limits<T>::infinity();
  Buf<T> v, out;
  for (std::size_t i = 0; i < n; ++i) {
    switch (i % 4) {
      case 0: v.data[i] = qnan; break;
      case 1: v.data[i] = inf; break;
      case 2: v.data[i] = -inf; break;
      default: v.data[i] = static_cast<T>(i) * T(1.5) - T(2); break;  // finite
    }
  }
  auto vv = hs::Load(d, v.data);
  hs::Store(hs::VecFromMask(d, hs::IsNaN(vv)), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i] != T{0}, std::isnan(v.data[i]) != 0) << "IsNaN i=" << i;
  hs::Store(hs::VecFromMask(d, hs::IsInf(vv)), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i] != T{0}, std::isinf(v.data[i]) != 0) << "IsInf i=" << i;
  hs::Store(hs::VecFromMask(d, hs::IsFinite(vv)), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i] != T{0}, std::isfinite(v.data[i]) != 0) << "IsFinite i=" << i;
  // IsEitherNaN(v, shifted): true where either operand is NaN.
  Buf<T> w;
  for (std::size_t i = 0; i < n; ++i) w.data[i] = (i % 5 == 0) ? qnan : T(1);
  auto vw = hs::Load(d, w.data);
  hs::Store(hs::VecFromMask(d, hs::IsEitherNaN(vv, vw)), d, out.data);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(out.data[i] != T{0}, (std::isnan(v.data[i]) || std::isnan(w.data[i]))) << "IsEitherNaN i=" << i;
}
TEST(Vec, FpClassifyFloat) { CheckFpClassify<float>(); }
TEST(Vec, FpClassifyDouble) { CheckFpClassify<double>(); }

// ---- A-4 TableLookupBytes (PSHUFB byte shuffle, per-128-bit block) -------------------
// u8 lanes: table[i]=some pattern, idx[i]=lookup indices; result[i]=table[block*16+(idx&15)].
// Or0 zeroes where idx bit7 set. Reference mirrors the scalar block-local semantics.
TEST(Vec, TableLookupBytes) {
  const hs::ScalableTag<std::uint8_t> d;
  const std::size_t n = hs::Lanes(d);  // 16 (sse4/scalar) or 32 (avx2)
  Buf<std::uint8_t> tbl, idx, out;
  for (std::size_t i = 0; i < n; ++i) {
    tbl.data[i] = static_cast<std::uint8_t>(0xA0 + (i & 15));   // distinguishable per block-slot
    idx.data[i] = static_cast<std::uint8_t>((i * 7) & 15);      // in-range 0..15
  }
  auto vt = hs::Load(d, tbl.data), vi = hs::Load(d, idx.data);
  hs::Store(hs::TableLookupBytes(vt, vi), d, out.data);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t block = (i / 16) * 16;
    EXPECT_EQ(out.data[i], tbl.data[block + (idx.data[i] & 15)]) << "TLB i=" << i;
  }
  // Or0: set bit7 on odd lanes -> those zero, others look up (idx & 15).
  Buf<std::uint8_t> idx0;
  for (std::size_t i = 0; i < n; ++i)
    idx0.data[i] = static_cast<std::uint8_t>((i % 2) ? (0x80 | (i & 15)) : (i & 15));
  auto vi0 = hs::Load(d, idx0.data);
  hs::Store(hs::TableLookupBytesOr0(vt, vi0), d, out.data);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t block = (i / 16) * 16;
    std::uint8_t expect = (idx0.data[i] & 0x80) ? 0 : tbl.data[block + (idx0.data[i] & 15)];
    EXPECT_EQ(out.data[i], expect) << "TLBOr0 i=" << i;
  }
}

// ---- A-5 widen-dot (int8/16 GEMM quantization core) ---------------------------------
// SatWidenMulPairwiseAdd (u8*i8 -> saturated i16 pairwise), the ReorderWidenMulAccumulate +
// RearrangeToOddPlusEven pair (i16 dot accumulate), and SumOfMulQuadAccumulate (i8 4-way).
// All checked against the exact scalar reference computed in wide integer arithmetic.
TEST(Vec, SatWidenMulPairwiseAdd) {
  const hs::ScalableTag<std::uint8_t> du;   // u8 input lanes
  const hs::ScalableTag<std::int16_t> di;   // i16 output lanes (half the count)
  const std::size_t nu = hs::Lanes(du);
  const std::size_t ni = hs::Lanes(di);
  ASSERT_EQ(nu, 2 * ni);
  Buf<std::uint8_t> a; Buf<std::int8_t> b; Buf<std::int16_t> out;
  for (std::size_t i = 0; i < nu; ++i) {
    a.data[i] = static_cast<std::uint8_t>(200 + (i % 40));            // large unsigned
    b.data[i] = static_cast<std::int8_t>((i % 2) ? 120 : -110);      // large signed
  }
  auto va = hs::Load(du, a.data);
  // load i8 b through a same-width signed tag
  const hs::ScalableTag<std::int8_t> ds;
  auto vb = hs::Load(ds, b.data);
  hs::Store(hs::SatWidenMulPairwiseAdd(di, va, vb), di, out.data);
  for (std::size_t i = 0; i < ni; ++i) {
    int p = static_cast<int>(a.data[2*i]) * b.data[2*i] + static_cast<int>(a.data[2*i+1]) * b.data[2*i+1];
    int sat = p < -32768 ? -32768 : (p > 32767 ? 32767 : p);
    EXPECT_EQ(out.data[i], static_cast<std::int16_t>(sat)) << "sat i=" << i;
  }
}
TEST(Vec, ReorderWidenMulAccumulate) {
  const hs::ScalableTag<std::int16_t> d16;
  const hs::ScalableTag<std::int32_t> d32;
  const std::size_t n16 = hs::Lanes(d16), n32 = hs::Lanes(d32);
  ASSERT_EQ(n16, 2 * n32);
  Buf<std::int16_t> a, b;
  for (std::size_t i = 0; i < n16; ++i) { a.data[i] = static_cast<std::int16_t>(i * 3 - 100); b.data[i] = static_cast<std::int16_t>(i - 50); }
  auto va = hs::Load(d16, a.data), vb = hs::Load(d16, b.data);
  auto sum0 = hs::Set(d32, 0), sum1 = hs::Set(d32, 0);
  sum0 = hs::ReorderWidenMulAccumulate(d32, va, vb, sum0, sum1);
  sum0 = hs::ReorderWidenMulAccumulate(d32, va, vb, sum0, sum1);  // accumulate twice
  Buf<std::int32_t> out;
  hs::Store(hs::RearrangeToOddPlusEven(sum0, sum1), d32, out.data);
  for (std::size_t i = 0; i < n32; ++i) {
    long long pair = static_cast<long long>(a.data[2*i]) * b.data[2*i] + static_cast<long long>(a.data[2*i+1]) * b.data[2*i+1];
    EXPECT_EQ(out.data[i], static_cast<std::int32_t>(2 * pair)) << "reorder i=" << i;  // 2x from two accumulates
  }
}
TEST(Vec, SumOfMulQuadAccumulate) {
  const hs::ScalableTag<std::int8_t> d8;
  const hs::ScalableTag<std::int32_t> d32;
  const std::size_t n8 = hs::Lanes(d8), n32 = hs::Lanes(d32);
  ASSERT_EQ(n8, 4 * n32);
  Buf<std::int8_t> a, b; Buf<std::int32_t> out;
  for (std::size_t i = 0; i < n8; ++i) { a.data[i] = static_cast<std::int8_t>(i - 60); b.data[i] = static_cast<std::int8_t>(40 - i); }
  auto va = hs::Load(d8, a.data), vb = hs::Load(d8, b.data);
  auto sum = hs::Set(d32, 7);  // nonzero initial accumulator
  hs::Store(hs::SumOfMulQuadAccumulate(d32, va, vb, sum), d32, out.data);
  for (std::size_t i = 0; i < n32; ++i) {
    int acc = 7;
    for (int k = 0; k < 4; ++k) acc += static_cast<int>(a.data[4*i+k]) * b.data[4*i+k];
    EXPECT_EQ(out.data[i], acc) << "quad i=" << i;
  }
}

// ---- A-6 interleaved load/store (AoS <-> SoA) ---------------------------------------
// StoreInterleavedK writes K vectors as interleaved AoS; LoadInterleavedK reads it back.
// Check both the interleaved memory layout AND the deinterleave round-trip.
TEST(Vec, InterleavedLoadStore) {
  const hs::ScalableTag<std::int32_t> d;
  const std::size_t n = hs::Lanes(d);
  Buf<std::int32_t> s0, s1, s2, s3;
  for (std::size_t i = 0; i < n; ++i) {
    s0.data[i] = static_cast<std::int32_t>(i);          s1.data[i] = static_cast<std::int32_t>(100 + i);
    s2.data[i] = static_cast<std::int32_t>(200 + i);    s3.data[i] = static_cast<std::int32_t>(300 + i);
  }
  auto v0 = hs::Load(d, s0.data), v1 = hs::Load(d, s1.data), v2 = hs::Load(d, s2.data), v3 = hs::Load(d, s3.data);
  // --- 2-way ---
  std::int32_t buf2[kMaxLanes * 2];
  hs::StoreInterleaved2(v0, v1, d, buf2);
  for (std::size_t i = 0; i < n; ++i) { EXPECT_EQ(buf2[2*i], s0.data[i]); EXPECT_EQ(buf2[2*i+1], s1.data[i]); }
  Buf<std::int32_t> r0, r1, r2, r3;
  hs::VFromD<decltype(d)> lv0, lv1, lv2, lv3;
  hs::LoadInterleaved2(d, buf2, lv0, lv1);
  hs::Store(lv0, d, r0.data); hs::Store(lv1, d, r1.data);
  for (std::size_t i = 0; i < n; ++i) { EXPECT_EQ(r0.data[i], s0.data[i]); EXPECT_EQ(r1.data[i], s1.data[i]); }
  // --- 3-way ---
  std::int32_t buf3[kMaxLanes * 3];
  hs::StoreInterleaved3(v0, v1, v2, d, buf3);
  for (std::size_t i = 0; i < n; ++i) { EXPECT_EQ(buf3[3*i], s0.data[i]); EXPECT_EQ(buf3[3*i+1], s1.data[i]); EXPECT_EQ(buf3[3*i+2], s2.data[i]); }
  hs::LoadInterleaved3(d, buf3, lv0, lv1, lv2);
  hs::Store(lv0,d,r0.data); hs::Store(lv1,d,r1.data); hs::Store(lv2,d,r2.data);
  for (std::size_t i = 0; i < n; ++i) { EXPECT_EQ(r0.data[i], s0.data[i]); EXPECT_EQ(r1.data[i], s1.data[i]); EXPECT_EQ(r2.data[i], s2.data[i]); }
  // --- 4-way ---
  std::int32_t buf4[kMaxLanes * 4];
  hs::StoreInterleaved4(v0, v1, v2, v3, d, buf4);
  for (std::size_t i = 0; i < n; ++i) { EXPECT_EQ(buf4[4*i], s0.data[i]); EXPECT_EQ(buf4[4*i+3], s3.data[i]); }
  hs::LoadInterleaved4(d, buf4, lv0, lv1, lv2, lv3);
  hs::Store(lv0,d,r0.data); hs::Store(lv1,d,r1.data); hs::Store(lv2,d,r2.data); hs::Store(lv3,d,r3.data);
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(r0.data[i], s0.data[i]); EXPECT_EQ(r1.data[i], s1.data[i]);
    EXPECT_EQ(r2.data[i], s2.data[i]); EXPECT_EQ(r3.data[i], s3.data[i]);
  }
}

// Type-generic interleave round-trip: covers the register-transpose paths (32/64-bit) AND the
// store+reload fallback (8/16-bit) across every lane type, on top of the i32 test above. Verifies
// StoreInterleavedK writes correct AoS and LoadInterleavedK reads it back to the originals.
namespace {
template <typename T>
void InterleaveRoundTrip() {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  Buf<T> s0, s1, s2, s3, r0, r1, r2, r3;
  for (std::size_t i = 0; i < n; ++i) {
    s0.data[i] = static_cast<T>(1 + i);   s1.data[i] = static_cast<T>(40 + i);
    s2.data[i] = static_cast<T>(80 + i);  s3.data[i] = static_cast<T>(120 + i);
  }
  const auto v0 = hs::Load(d, s0.data), v1 = hs::Load(d, s1.data),
             v2 = hs::Load(d, s2.data), v3 = hs::Load(d, s3.data);
  hs::VFromD<decltype(d)> l0, l1, l2, l3;
  T buf[kMaxLanes * 4];
  // 2-way
  hs::StoreInterleaved2(v0, v1, d, buf);
  for (std::size_t i = 0; i < n; ++i) { EXPECT_EQ(buf[2*i], s0.data[i]); EXPECT_EQ(buf[2*i+1], s1.data[i]); }
  hs::LoadInterleaved2(d, buf, l0, l1);
  hs::Store(l0, d, r0.data); hs::Store(l1, d, r1.data);
  for (std::size_t i = 0; i < n; ++i) { EXPECT_EQ(r0.data[i], s0.data[i]); EXPECT_EQ(r1.data[i], s1.data[i]); }
  // 3-way
  hs::StoreInterleaved3(v0, v1, v2, d, buf);
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(buf[3*i], s0.data[i]); EXPECT_EQ(buf[3*i+1], s1.data[i]); EXPECT_EQ(buf[3*i+2], s2.data[i]);
  }
  hs::LoadInterleaved3(d, buf, l0, l1, l2);
  hs::Store(l0, d, r0.data); hs::Store(l1, d, r1.data); hs::Store(l2, d, r2.data);
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(r0.data[i], s0.data[i]); EXPECT_EQ(r1.data[i], s1.data[i]); EXPECT_EQ(r2.data[i], s2.data[i]);
  }
  // 4-way
  hs::StoreInterleaved4(v0, v1, v2, v3, d, buf);
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(buf[4*i], s0.data[i]);   EXPECT_EQ(buf[4*i+1], s1.data[i]);
    EXPECT_EQ(buf[4*i+2], s2.data[i]); EXPECT_EQ(buf[4*i+3], s3.data[i]);
  }
  hs::LoadInterleaved4(d, buf, l0, l1, l2, l3);
  hs::Store(l0, d, r0.data); hs::Store(l1, d, r1.data); hs::Store(l2, d, r2.data); hs::Store(l3, d, r3.data);
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(r0.data[i], s0.data[i]); EXPECT_EQ(r1.data[i], s1.data[i]);
    EXPECT_EQ(r2.data[i], s2.data[i]); EXPECT_EQ(r3.data[i], s3.data[i]);
  }
}
}  // namespace
TEST(Vec, InterleaveRoundTripAllWidths) {
  InterleaveRoundTrip<std::uint32_t>();  // register transpose (shuffle_ps tree)
  InterleaveRoundTrip<float>();
  InterleaveRoundTrip<std::int64_t>();   // register transpose (64-bit unpack)
  InterleaveRoundTrip<double>();
  InterleaveRoundTrip<std::uint16_t>();  // store+reload fallback
  InterleaveRoundTrip<std::uint8_t>();
}

// ---- A-7 math / arith extensions ----------------------------------------------------
TEST(Vec, MathExtFloat) {
  const hs::ScalableTag<float> d;
  const std::size_t n = hs::Lanes(d);
  Buf<float> a, b, out;
  for (std::size_t i = 0; i < n; ++i) { a.data[i] = static_cast<float>(i) + 1.5f; b.data[i] = static_cast<float>(i % 3) + 0.5f; }
  auto va = hs::Load(d, a.data), vb = hs::Load(d, b.data);
  // ApproximateReciprocal: HARDWARE approximation. x86 rcp_ps is ~12-bit (2^-12); ARM vrecpeq
  // is coarser -- Highway documents "error on Arm is closer to 1%". So the tolerance must admit
  // the WORST backend (ARM ~1%), not just x86 -- a 1e-3 bound passed on x86 but a real ARM run
  // (mac M4) measured ~0.06% error exceeding it on some lanes. Use 1% (the op's contract is
  // "approximate"; exact reciprocal is Div). Same for rsqrt.
  hs::Store(hs::ApproximateReciprocal(va), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_NEAR(out.data[i], 1.0f / a.data[i], std::abs(1.0f/a.data[i]) * 1e-2f) << "rcp i=" << i;
  hs::Store(hs::ApproximateReciprocalSqrt(va), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_NEAR(out.data[i], 1.0f / std::sqrt(a.data[i]), (1.0f/std::sqrt(a.data[i])) * 1e-2f) << "rsqrt i=" << i;
  // Newton-refined: one step lifts the ~12-bit estimate to near-full f32 (~1e-4 rel, >> the raw
  // 1e-2). HYLE superiority (xsimd/Highway ship only the raw estimate). Tolerance 5e-4 admits
  // the worst backend after one step while still ~20x tighter than the raw estimate's 1e-2.
  hs::Store(hs::ReciprocalNewton(va), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_NEAR(out.data[i], 1.0f / a.data[i], std::abs(1.0f/a.data[i]) * 5e-4f) << "rcpN i=" << i;
  hs::Store(hs::ReciprocalSqrtNewton(va), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_NEAR(out.data[i], 1.0f / std::sqrt(a.data[i]), (1.0f/std::sqrt(a.data[i])) * 5e-4f) << "rsqrtN i=" << i;
  // GetExponent = floor(log2).
  hs::Store(hs::GetExponent(va), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_FLOAT_EQ(out.data[i], std::logb(a.data[i])) << "getexp i=" << i;
  // MulRound = round(a*b).
  hs::Store(hs::MulRound(va, vb), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_FLOAT_EQ(out.data[i], std::nearbyint(a.data[i] * b.data[i])) << "mulround i=" << i;
  // AddSub: a-b even, a+b odd.
  hs::Store(hs::AddSub(va, vb), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_FLOAT_EQ(out.data[i], (i&1) ? a.data[i]+b.data[i] : a.data[i]-b.data[i]) << "addsub i=" << i;
  // MinNumber/MaxNumber with a NaN operand: return the non-NaN.
  Buf<float> na; for (std::size_t i = 0; i < n; ++i) na.data[i] = (i%2) ? std::numeric_limits<float>::quiet_NaN() : a.data[i];
  auto vna = hs::Load(d, na.data);
  hs::Store(hs::MinNumber(vna, vb), d, out.data);
  for (std::size_t i = 0; i < n; ++i) { float e = std::isnan(na.data[i]) ? b.data[i] : std::min(na.data[i], b.data[i]); EXPECT_FLOAT_EQ(out.data[i], e) << "minnum i=" << i; }
  // MulByPow2(a, e) = a * 2^e ; e is an i32-lane vector of matching width (ldexp).
  const hs::ScalableTag<std::int32_t> di;
  Buf<std::int32_t> ei; for (std::size_t i = 0; i < n; ++i) ei.data[i] = static_cast<std::int32_t>(i % 5) - 2;  // -2..2
  hs::Store(hs::MulByPow2(va, hs::Load(di, ei.data)), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_FLOAT_EQ(out.data[i], std::ldexp(a.data[i], static_cast<int>(ei.data[i]))) << "mulbypow2 i=" << i;
  // Red-team: the exp=-2..2 sweep above only exercises normal values -- the vectorized 3-part
  // exponent build must also be correct through OVERFLOW (->inf), DENORMAL, and total UNDERFLOW
  // (->0). These are exactly the ranges the single-2^e naive form would get wrong.
  Buf<float> big; Buf<std::int32_t> ebig;
  const int wide[] = {0, 1, -1, 130, -130, 200, -200, 127, -149, -150, 50, -50, 255, -255, 10, -10};
  const float vals[] = {1.0f, 1.5f, 0.5f, 3.0f, 1e20f, 1e-20f, -2.0f, 7.0f,
                        1.0f, 1.0f, -1.5f, 2.0f, 1.0f, 1.0f, 0.0f, -0.0f};
  for (std::size_t i = 0; i < n; ++i) { big.data[i] = vals[i % 16]; ebig.data[i] = wide[i % 16]; }
  hs::Store(hs::MulByPow2(hs::Load(d, big.data), hs::Load(di, ebig.data)), d, out.data);
  for (std::size_t i = 0; i < n; ++i) {
    const float ref = std::ldexp(big.data[i], static_cast<int>(ebig.data[i]));
    if (std::isnan(ref)) EXPECT_TRUE(std::isnan(out.data[i])) << "mulbypow2 wide nan i=" << i;
    else EXPECT_EQ(out.data[i], ref) << "mulbypow2 wide i=" << i << " v=" << big.data[i] << " e=" << ebig.data[i];
  }
}
TEST(Vec, MathExtSaturatedInt) {
  const hs::ScalableTag<std::int16_t> d;
  const std::size_t n = hs::Lanes(d);
  Buf<std::int16_t> a, out;
  const std::int16_t mn = std::numeric_limits<std::int16_t>::min();
  for (std::size_t i = 0; i < n; ++i) a.data[i] = (i % 4 == 0) ? mn : static_cast<std::int16_t>(static_cast<int>(i) * 7 - 300);
  auto va = hs::Load(d, a.data);
  hs::Store(hs::SaturatedNeg(va), d, out.data);
  for (std::size_t i = 0; i < n; ++i) { int e = (a.data[i]==mn) ? 32767 : -a.data[i]; EXPECT_EQ(out.data[i], static_cast<std::int16_t>(e)) << "satneg i=" << i; }
  hs::Store(hs::SaturatedAbs(va), d, out.data);
  for (std::size_t i = 0; i < n; ++i) { int e = (a.data[i]==mn) ? 32767 : (a.data[i]<0?-a.data[i]:a.data[i]); EXPECT_EQ(out.data[i], static_cast<std::int16_t>(e)) << "satabs i=" << i; }
  // MulFixedPoint15: Q1.15 mul = (2ab rounded) >> 16.
  Buf<std::int16_t> b, out2;
  for (std::size_t i = 0; i < n; ++i) { a.data[i] = static_cast<std::int16_t>(i * 1000 - 15000); b.data[i] = static_cast<std::int16_t>(20000 - i * 800); }
  a.data[0] = mn; b.data[0] = mn;
  auto va2 = hs::Load(d, a.data), vb2 = hs::Load(d, b.data);
  hs::Store(hs::MulFixedPoint15(va2, vb2), d, out2.data);
  for (std::size_t i = 0; i < n; ++i) { int e = (static_cast<int>(a.data[i]) * b.data[i] + 0x4000) >> 15; EXPECT_EQ(out2.data[i], static_cast<std::int16_t>(e)) << "q15 i=" << i; }
}

// ---- C complex arithmetic -----------------------------------------------------------
TEST(Vec, ComplexArith) {
  const hs::ScalableTag<float> d;
  const std::size_t n = hs::Lanes(d);
  if (n < 2) return;  // need pairs
  Buf<float> a, b, c, out;
  for (std::size_t i = 0; i < n; ++i) { a.data[i] = static_cast<float>(i) - 3.0f; b.data[i] = static_cast<float>(i % 5) + 1.0f; c.data[i] = static_cast<float>(i) * 0.5f; }
  auto va = hs::Load(d, a.data), vb = hs::Load(d, b.data), vc = hs::Load(d, c.data);
  // ComplexConj: negate odd (imag) lanes.
  hs::Store(hs::ComplexConj(va), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_FLOAT_EQ(out.data[i], (i%2) ? -a.data[i] : a.data[i]) << "conj i=" << i;
  // MulComplex: per complex pair (re,im).
  hs::Store(hs::MulComplex(va, vb), d, out.data);
  for (std::size_t j = 0; j + 1 < n; j += 2) {
    float re = a.data[j]*b.data[j] - a.data[j+1]*b.data[j+1];
    float im = a.data[j]*b.data[j+1] + a.data[j+1]*b.data[j];
    EXPECT_NEAR(out.data[j], re, 1e-4f) << "mulc re j=" << j;
    EXPECT_NEAR(out.data[j+1], im, 1e-4f) << "mulc im j=" << j;
  }
  // MulComplexAdd = a*b + c.
  hs::Store(hs::MulComplexAdd(va, vb, vc), d, out.data);
  for (std::size_t j = 0; j + 1 < n; j += 2) {
    float re = a.data[j]*b.data[j] - a.data[j+1]*b.data[j+1] + c.data[j];
    float im = a.data[j]*b.data[j+1] + a.data[j+1]*b.data[j] + c.data[j+1];
    EXPECT_NEAR(out.data[j], re, 1e-4f) << "mulcadd re j=" << j;
    EXPECT_NEAR(out.data[j+1], im, 1e-4f) << "mulcadd im j=" << j;
  }
  // MulComplexConj: a * conj(b).
  hs::Store(hs::MulComplexConj(va, vb), d, out.data);
  for (std::size_t j = 0; j + 1 < n; j += 2) {
    float re = a.data[j]*b.data[j] + a.data[j+1]*b.data[j+1];
    float im = a.data[j+1]*b.data[j] - a.data[j]*b.data[j+1];
    EXPECT_NEAR(out.data[j], re, 1e-4f) << "mulcconj re j=" << j;
    EXPECT_NEAR(out.data[j+1], im, 1e-4f) << "mulcconj im j=" << j;
  }
}

// ---- C crypto: AES round primitives + carryless multiply ----------------------------
// The op contract is a 128-bit AES block. ScalableTag<uint8_t> yields exactly the backend's
// native width -- 16B (scalar/sse4/neon), 32B (avx2 = 2 blocks), 64B (avx512 = 4 blocks) --
// which matches each backend's static_assert. We verify against a LOCAL software oracle
// (independent FIPS-197 code path, NOT the library's) applied per 128-bit block, plus
// deterministic FIPS-197 anchors that hold across the hardware and software paths.
namespace crypto_oracle {
// FIPS-197 S-box / inverse (independent copy -- the whole point is a second implementation).
constexpr std::uint8_t kS[256] = {
  0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
  0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
  0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
  0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
  0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
  0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
  0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
  0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
  0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
  0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
  0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
  0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
  0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
  0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
  0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
  0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16};
inline std::uint8_t xtime(std::uint8_t a) { std::uint8_t h = a & 0x80; a <<= 1; return h ? (a ^ 0x1b) : a; }
inline std::uint8_t gmul(std::uint8_t a, std::uint8_t b) {
  std::uint8_t p = 0;
  for (int i = 0; i < 8; ++i) { if (b & 1) p ^= a; a = xtime(a); b >>= 1; }
  return p;
}
// AESRound = MixColumns(SubBytes(ShiftRows(s))) ^ k  (AES-NI aesenc semantics).
inline void round(std::uint8_t s[16], const std::uint8_t k[16], bool mix) {
  std::uint8_t t[16];
  for (int c = 0; c < 4; ++c) for (int r = 0; r < 4; ++r) t[c*4+r] = kS[s[((c+r)%4)*4+r]];  // ShiftRows+SubBytes
  if (mix) for (int c = 0; c < 4; ++c) {
    std::uint8_t* col = t + c*4; std::uint8_t a0=col[0],a1=col[1],a2=col[2],a3=col[3];
    col[0]=gmul(a0,2)^gmul(a1,3)^a2^a3; col[1]=a0^gmul(a1,2)^gmul(a2,3)^a3;
    col[2]=a0^a1^gmul(a2,2)^gmul(a3,3); col[3]=gmul(a0,3)^a1^a2^gmul(a3,2);
  }
  for (int i = 0; i < 16; ++i) s[i] = t[i] ^ k[i];
}
inline void clmul(std::uint64_t a, std::uint64_t b, std::uint64_t out[2]) {
  std::uint64_t lo = 0, hi = 0;
  for (int i = 0; i < 64; ++i) if ((b >> i) & 1) { lo ^= a << i; if (i) hi ^= a >> (64 - i); }
  out[0] = lo; out[1] = hi;
}
// AESKeyGenAssist(v, rcon), matching _mm_aeskeygenassist_si128 exactly.
inline void keygen(std::uint8_t s[16], std::uint8_t rcon) {
  std::uint8_t x[16];
  for (int i = 0; i < 16; ++i) x[i] = kS[s[i]];
  std::uint8_t w1[4] = {x[4], x[5], x[6], x[7]}, w3[4] = {x[12], x[13], x[14], x[15]};
  s[0]=w1[0]; s[1]=w1[1]; s[2]=w1[2]; s[3]=w1[3];
  s[4]=static_cast<std::uint8_t>(w1[1]^rcon); s[5]=w1[2]; s[6]=w1[3]; s[7]=w1[0];
  s[8]=w3[0]; s[9]=w3[1]; s[10]=w3[2]; s[11]=w3[3];
  s[12]=static_cast<std::uint8_t>(w3[1]^rcon); s[13]=w3[2]; s[14]=w3[3]; s[15]=w3[0];
}
}  // namespace crypto_oracle

TEST(Vec, CryptoAES) {
  const hs::ScalableTag<std::uint8_t> d;
  const std::size_t n = hs::Lanes(d);
  ASSERT_EQ(n % 16u, 0u);  // native width is a whole number of 128-bit AES blocks
  Buf<std::uint8_t> st, key, out;
  for (std::size_t i = 0; i < n; ++i) { st.data[i] = static_cast<std::uint8_t>(i * 7 + 1); key.data[i] = static_cast<std::uint8_t>(i * 3 + 5); }
  auto vs = hs::Load(d, st.data), vk = hs::Load(d, key.data);

  // AESRound vs oracle (per 128-bit block).
  hs::Store(hs::AESRound(vs, vk), d, out.data);
  for (std::size_t blk = 0; blk < n; blk += 16) {
    std::uint8_t s[16], k[16];
    for (int i = 0; i < 16; ++i) { s[i] = st.data[blk+i]; k[i] = key.data[blk+i]; }
    crypto_oracle::round(s, k, /*mix=*/true);
    for (int i = 0; i < 16; ++i) EXPECT_EQ(out.data[blk+i], s[i]) << "AESRound blk=" << blk << " i=" << i;
  }
  // AESLastRound vs oracle (no MixColumns).
  hs::Store(hs::AESLastRound(vs, vk), d, out.data);
  for (std::size_t blk = 0; blk < n; blk += 16) {
    std::uint8_t s[16], k[16];
    for (int i = 0; i < 16; ++i) { s[i] = st.data[blk+i]; k[i] = key.data[blk+i]; }
    crypto_oracle::round(s, k, /*mix=*/false);
    for (int i = 0; i < 16; ++i) EXPECT_EQ(out.data[blk+i], s[i]) << "AESLastRound blk=" << blk << " i=" << i;
  }
  // Deterministic FIPS-197 anchor: SubBytes(ShiftRows(0)) ^ 0 = SBox[0] = 0x63 everywhere.
  hs::Store(hs::AESLastRound(hs::Zero(d), hs::Zero(d)), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], 0x63) << "anchor i=" << i;

  // Round-trip: InvSubBytes(InvShiftRows(.)) inverts SubBytes(ShiftRows(.)) since SubBytes and
  // ShiftRows commute; with key=0 the last-round enc/dec pair is an exact identity.
  hs::Store(hs::AESLastRoundInv(hs::AESLastRound(vs, hs::Zero(d)), hs::Zero(d)), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], st.data[i]) << "roundtrip i=" << i;

  // AESInvMixColumns undoes AESRound's MixColumns: AESInvMixColumns applied to a full round
  // (key=0) leaves SubBytes(ShiftRows(state)) = AESLastRound(state,0).
  hs::Store(hs::AESInvMixColumns(hs::AESRound(vs, hs::Zero(d))), d, out.data);
  Buf<std::uint8_t> ref;
  hs::Store(hs::AESLastRound(vs, hs::Zero(d)), d, ref.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], ref.data[i]) << "invmix i=" << i;

  // AESKeyGenAssist<0x01> vs oracle (key-schedule helper, rcon is a compile-time immediate).
  hs::Store(hs::AESKeyGenAssist<0x01>(vs), d, out.data);
  for (std::size_t blk = 0; blk < n; blk += 16) {
    std::uint8_t s[16];
    for (int i = 0; i < 16; ++i) s[i] = st.data[blk+i];
    crypto_oracle::keygen(s, 0x01);
    for (int i = 0; i < 16; ++i) EXPECT_EQ(out.data[blk+i], s[i]) << "keygen blk=" << blk << " i=" << i;
  }
}

TEST(Vec, CryptoCLMul) {
  const hs::ScalableTag<std::uint8_t> d;
  const std::size_t n = hs::Lanes(d);
  ASSERT_EQ(n % 16u, 0u);
  // Fill each 128-bit block with two distinct 64-bit values in the low/high halves.
  Buf<std::uint8_t> ab, bb, out;
  for (std::size_t blk = 0; blk < n; blk += 16) {
    std::uint64_t a0 = 0x0123456789abcdefULL ^ (blk * 0x1000ULL), a1 = 0xfedcba9876543210ULL ^ blk;
    std::uint64_t b0 = 0x00000000000000FFULL ^ blk, b1 = 0x8000000000000001ULL ^ (blk << 8);
    std::memcpy(ab.data + blk, &a0, 8); std::memcpy(ab.data + blk + 8, &a1, 8);
    std::memcpy(bb.data + blk, &b0, 8); std::memcpy(bb.data + blk + 8, &b1, 8);
  }
  auto va = hs::Load(d, ab.data), vb = hs::Load(d, bb.data);

  hs::Store(hs::CLMulLower(va, vb), d, out.data);
  for (std::size_t blk = 0; blk < n; blk += 16) {
    std::uint64_t a0, b0, o[2];
    std::memcpy(&a0, ab.data + blk, 8); std::memcpy(&b0, bb.data + blk, 8);
    crypto_oracle::clmul(a0, b0, o);
    std::uint64_t lo, hi; std::memcpy(&lo, out.data + blk, 8); std::memcpy(&hi, out.data + blk + 8, 8);
    EXPECT_EQ(lo, o[0]) << "CLMulLower lo blk=" << blk;
    EXPECT_EQ(hi, o[1]) << "CLMulLower hi blk=" << blk;
  }
  hs::Store(hs::CLMulUpper(va, vb), d, out.data);
  for (std::size_t blk = 0; blk < n; blk += 16) {
    std::uint64_t a1, b1, o[2];
    std::memcpy(&a1, ab.data + blk + 8, 8); std::memcpy(&b1, bb.data + blk + 8, 8);
    crypto_oracle::clmul(a1, b1, o);
    std::uint64_t lo, hi; std::memcpy(&lo, out.data + blk, 8); std::memcpy(&hi, out.data + blk + 8, 8);
    EXPECT_EQ(lo, o[0]) << "CLMulUpper lo blk=" << blk;
    EXPECT_EQ(hi, o[1]) << "CLMulUpper hi blk=" << blk;
  }
  // Deterministic anchor: 1*x = x (carryless), so low=x, high=0.
  Buf<std::uint8_t> one{}, xb, oo;
  for (std::size_t blk = 0; blk < n; blk += 16) {
    std::uint64_t lo1 = 1; std::memcpy(one.data + blk, &lo1, 8);
    std::uint64_t xv = 0xDEADBEEFCAFEBABEULL; std::memcpy(xb.data + blk, &xv, 8);
  }
  hs::Store(hs::CLMulLower(hs::Load(d, one.data), hs::Load(d, xb.data)), d, oo.data);
  for (std::size_t blk = 0; blk < n; blk += 16) {
    std::uint64_t lo, hi; std::memcpy(&lo, oo.data + blk, 8); std::memcpy(&hi, oo.data + blk + 8, 8);
    EXPECT_EQ(lo, 0xDEADBEEFCAFEBABEULL) << "clmul1 lo blk=" << blk;
    EXPECT_EQ(hi, 0u) << "clmul1 hi blk=" << blk;
  }
}

// ---- B-5 compress / gather / masked-memory completion -------------------------------
TEST(Vec, CompressAndMaskedMemory) {
  const hs::ScalableTag<std::int32_t> d;
  const std::size_t n = hs::Lanes(d);
  Buf<std::int32_t> a, no, out;
  for (std::size_t i = 0; i < n; ++i) { a.data[i] = static_cast<std::int32_t>(i + 1); no.data[i] = -9; }
  auto va = hs::Load(d, a.data), vno = hs::Load(d, no.data);
  Buf<std::int32_t> par; for (std::size_t i = 0; i < n; ++i) par.data[i] = static_cast<std::int32_t>(i % 2);
  auto m = hs::Eq(hs::Load(d, par.data), hs::Set(d, 0));  // even lanes true
  // CompressNot(v, m) = Compress(v, ~m): keeps ODD lanes packed low.
  hs::Store(hs::CompressNot(va, m), d, out.data);
  { std::size_t k = 0; for (std::size_t i = 0; i < n; ++i) if (i % 2) EXPECT_EQ(out.data[k++], a.data[i]) << "cnot"; }
  // MaskedGatherIndexOr: gather a[idx] on active lanes, else `no`.
  Buf<std::int32_t> idx; for (std::size_t i = 0; i < n; ++i) idx.data[i] = static_cast<std::int32_t>((n - 1) - i);
  auto vidx = hs::Load(d, idx.data);
  hs::Store(hs::MaskedGatherIndexOr(vno, m, d, a.data, vidx), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], (i%2==0) ? a.data[(n-1)-i] : -9) << "mgather i=" << i;
  // MaskedGatherIndex (zero default).
  hs::Store(hs::MaskedGatherIndex(m, d, a.data, vidx), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], (i%2==0) ? a.data[(n-1)-i] : 0) << "mgather0 i=" << i;
  // GatherOffset: byte offsets (idx*4 bytes for i32).
  Buf<std::int32_t> boff; for (std::size_t i = 0; i < n; ++i) boff.data[i] = static_cast<std::int32_t>(i * sizeof(std::int32_t));
  hs::Store(hs::GatherOffset(d, a.data, hs::Load(d, boff.data)), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], a.data[i]) << "goff i=" << i;
  // LoadNOr: load 2 lanes, rest = no.
  hs::Store(hs::LoadNOr(vno, d, a.data, 2), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], (i < 2) ? a.data[i] : -9) << "loadnor i=" << i;
  // SafeFillN / SafeCopyN.
  Buf<std::int32_t> fill; for (std::size_t i = 0; i < n; ++i) fill.data[i] = 0;
  hs::SafeFillN(3, 77, d, fill.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(fill.data[i], (i < 3) ? 77 : 0) << "fillN i=" << i;
  Buf<std::int32_t> cp; for (std::size_t i = 0; i < n; ++i) cp.data[i] = 0;
  hs::SafeCopyN(2, d, a.data, cp.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(cp.data[i], (i < 2) ? a.data[i] : 0) << "copyN i=" << i;
  // ScatterOffset: inverse of GatherOffset (byte offsets). Scatter a[i] to dst[i] via i*4 bytes.
  Buf<std::int32_t> sdst; for (std::size_t i = 0; i < n; ++i) sdst.data[i] = 0;
  hs::ScatterOffset(va, d, sdst.data, hs::Load(d, boff.data));
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(sdst.data[i], a.data[i]) << "soff i=" << i;
  // CompressBits: compress by a mask packed in a bitfield (even lanes -> bit i set for even i).
  std::uint8_t packed[8] = {};
  for (std::size_t i = 0; i < n; ++i) if (i % 2 == 0) packed[i / 8] |= static_cast<std::uint8_t>(1u << (i % 8));
  hs::Store(hs::CompressBits(va, packed), d, out.data);
  { std::size_t k = 0; for (std::size_t i = 0; i < n; ++i) if (i % 2 == 0) EXPECT_EQ(out.data[k++], a.data[i]) << "cbits i=" << i; }
  // LoadExpand: scatter first CountTrue(m) contiguous elems of p out to the true (even) lanes.
  hs::Store(hs::LoadExpand(m, d, a.data), d, out.data);
  { std::size_t k = 0; for (std::size_t i = 0; i < n; ++i) if (i % 2 == 0) EXPECT_EQ(out.data[i], a.data[k++]) << "lexp i=" << i; }
}

// ---- B-4 ternary bit fusion + sign/reduce helpers -----------------------------------
TEST(Vec, TernaryBitAndSign) {
  const hs::ScalableTag<std::uint32_t> d;
  const std::size_t n = hs::Lanes(d);
  Buf<std::uint32_t> a, b, c, out;
  for (std::size_t i = 0; i < n; ++i) { a.data[i] = 0xF0F0F0F0u + i; b.data[i] = 0x0FF00FF0u; c.data[i] = 0x33333333u; }
  auto va = hs::Load(d, a.data), vb = hs::Load(d, b.data), vc = hs::Load(d, c.data);
  hs::Store(hs::Xor3(va, vb, vc), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], a.data[i]^b.data[i]^c.data[i]) << "xor3 i=" << i;
  hs::Store(hs::Or3(va, vb, vc), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], a.data[i]|b.data[i]|c.data[i]) << "or3 i=" << i;
  hs::Store(hs::OrAnd(va, vb, vc), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], a.data[i]|(b.data[i]&c.data[i])) << "orand i=" << i;
  hs::Store(hs::BitwiseIfThenElse(vb, va, vc), d, out.data);  // (b&a)|(~b&c)
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], (b.data[i]&a.data[i])|(~b.data[i]&c.data[i])) << "bitsel i=" << i;
  // AllBits0 / AllBits1.
  EXPECT_TRUE(hs::AllBits0(hs::Zero(d)));
  EXPECT_FALSE(hs::AllBits0(va));
  EXPECT_TRUE(hs::AllBits1(hs::Set(d, 0xFFFFFFFFu)));
}
TEST(Vec, SignAndVectorReduce) {
  const hs::ScalableTag<std::int32_t> d;
  const std::size_t n = hs::Lanes(d);
  Buf<std::int32_t> a, out;
  int total = 0, mn = 1<<30, mx = -(1<<30);
  for (std::size_t i = 0; i < n; ++i) { a.data[i] = static_cast<std::int32_t>(i) * (i%2?-3:2) - 1; total += a.data[i]; mn = std::min(mn, a.data[i]); mx = std::max(mx, a.data[i]); }
  auto va = hs::Load(d, a.data);
  // IsNegative -> mask.
  hs::Store(hs::VecFromMask(d, hs::IsNegative(va)), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i] != 0, a.data[i] < 0) << "isneg i=" << i;
  // IfNegativeThenElseZero: a where a<0 else 0.
  hs::Store(hs::IfNegativeThenElseZero(va, va), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], a.data[i] < 0 ? a.data[i] : 0) << "ifneg i=" << i;
  // Vector-returning reductions: all lanes = the reduction.
  hs::Store(hs::SumOfLanes(d, va), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], total) << "sumlanes i=" << i;
  hs::Store(hs::MinOfLanes(d, va), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], mn) << "minlanes i=" << i;
  hs::Store(hs::MaxOfLanes(d, va), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], mx) << "maxlanes i=" << i;
}

// ---- B-4 SumsOf2/4 + RoundingShiftRight + Rol/Ror/TestBit ----------------------------
TEST(Vec, SumsOfNAndRoundShift) {
  const hs::ScalableTag<std::uint8_t> d8;
  const std::size_t n8 = hs::Lanes(d8);
  Buf<std::uint8_t> a;
  for (std::size_t i = 0; i < n8; ++i) a.data[i] = static_cast<std::uint8_t>(i * 3 + 1);
  auto va = hs::Load(d8, a.data);
  // SumsOf2 -> u16 (N/2 lanes).
  if (n8 % 2 == 0) {
    const hs::FixedTag<std::uint16_t, hs::ScalableTag<std::uint8_t>::kMaxLanes / 2> d16;
    Buf<std::uint16_t> o16; hs::Store(hs::SumsOf2(va), d16, o16.data);
    for (std::size_t g = 0; g < n8/2; ++g) EXPECT_EQ(o16.data[g], static_cast<std::uint16_t>(a.data[2*g]+a.data[2*g+1])) << "sums2 g=" << g;
  }
  // SumsOf4 -> u32 (N/4 lanes).
  if (n8 % 4 == 0) {
    const hs::FixedTag<std::uint32_t, hs::ScalableTag<std::uint8_t>::kMaxLanes / 4> d32;
    Buf<std::uint32_t> o32; hs::Store(hs::SumsOf4(va), d32, o32.data);
    for (std::size_t g = 0; g < n8/4; ++g) EXPECT_EQ(o32.data[g], static_cast<std::uint32_t>(a.data[4*g]+a.data[4*g+1]+a.data[4*g+2]+a.data[4*g+3])) << "sums4 g=" << g;
  }
  // RoundingShiftRight<3> on u32: (v + 4) >> 3.
  const hs::ScalableTag<std::uint32_t> du;
  Buf<std::uint32_t> u, ou;
  for (std::size_t i = 0; i < hs::Lanes(du); ++i) u.data[i] = static_cast<std::uint32_t>(i * 100 + 3);
  hs::Store(hs::RoundingShiftRight<3>(hs::Load(du, u.data)), du, ou.data);
  for (std::size_t i = 0; i < hs::Lanes(du); ++i) EXPECT_EQ(ou.data[i], (u.data[i] + 4u) >> 3) << "rsr i=" << i;
  // Rol/Ror<4> on u32 (= RotateLeft/Right).
  hs::Store(hs::Rol<4>(hs::Load(du, u.data)), du, ou.data);
  for (std::size_t i = 0; i < hs::Lanes(du); ++i) { std::uint32_t x = u.data[i]; EXPECT_EQ(ou.data[i], (x<<4)|(x>>28)) << "rol i=" << i; }
  // TestBit: bit 2 set?
  hs::Store(hs::VecFromMask(du, hs::TestBit(hs::Load(du, u.data), hs::Set(du, 4u))), du, ou.data);
  for (std::size_t i = 0; i < hs::Lanes(du); ++i) EXPECT_EQ(ou.data[i] != 0, (u.data[i] & 4u) != 0) << "testbit i=" << i;
}

// ---- B-3 lane access + init (generic layer) -----------------------------------------
TEST(Vec, LaneAccessAndInit) {
  const hs::ScalableTag<std::int32_t> d;
  const std::size_t n = hs::Lanes(d);
  // Iota: ramp from a base.
  auto vi = hs::Iota(d, 10);
  Buf<std::int32_t> out; hs::Store(vi, d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], static_cast<std::int32_t>(10 + i)) << "iota i=" << i;
  // ExtractLane / InsertLane.
  EXPECT_EQ(hs::ExtractLane(vi, 0), 10);
  if (n >= 2) EXPECT_EQ(hs::ExtractLane(vi, 1), 11);
  auto vj = hs::InsertLane(vi, 0, 999);
  hs::Store(vj, d, out.data);
  EXPECT_EQ(out.data[0], 999); for (std::size_t i = 1; i < n; ++i) EXPECT_EQ(out.data[i], static_cast<std::int32_t>(10+i));
  // Undefined = Zero (we alias it).
  hs::Store(hs::Undefined(d), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], 0);
  // MaskedSet: value on even lanes, 0 else.
  Buf<std::int32_t> par; for (std::size_t i = 0; i < n; ++i) par.data[i] = static_cast<std::int32_t>(i % 2);
  auto m = hs::Eq(hs::Load(d, par.data), hs::Set(d, 0));
  hs::Store(hs::MaskedSet(d, m, 42), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], (i%2==0) ? 42 : 0) << "maskset i=" << i;
}
TEST(Vec, FloatToIntRounding) {
  const hs::ScalableTag<float> d;
  const hs::ScalableTag<std::int32_t> di;
  const std::size_t n = hs::Lanes(d);
  Buf<float> f; Buf<std::int32_t> out;
  for (std::size_t i = 0; i < n; ++i) f.data[i] = (i % 2 ? -1.0f : 1.0f) * (static_cast<float>(i) + 0.6f);
  auto vf = hs::Load(d, f.data);
  hs::Store(hs::NearestInt(vf), di, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], static_cast<std::int32_t>(std::nearbyint(f.data[i]))) << "nearest i=" << i;
  hs::Store(hs::CeilInt(vf), di, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], static_cast<std::int32_t>(std::ceil(f.data[i]))) << "ceil i=" << i;
  hs::Store(hs::FloorInt(vf), di, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], static_cast<std::int32_t>(std::floor(f.data[i]))) << "floor i=" << i;
  // SignBit: MSB-only pattern; reinterpret to int and check it's negative-zero bit.
  Buf<float> sb; hs::Store(hs::SignBit(d), d, sb.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(std::bit_cast<std::uint32_t>(sb.data[i]), 0x80000000u) << "signbit i=" << i;
}

// ---- B-3 narrowing / strided widen --------------------------------------------------
TEST(Vec, TruncateAndStridedPromote) {
  // TruncateTo u32 -> u8 (low byte, modular).
  const hs::ScalableTag<std::uint32_t> d32;
  const std::size_t n32 = hs::Lanes(d32);
  const hs::FixedTag<std::uint8_t, hs::ScalableTag<std::uint32_t>::kMaxLanes> d8;
  Buf<std::uint32_t> a; Buf<std::uint8_t> o8;
  for (std::size_t i = 0; i < n32; ++i) a.data[i] = 0x11223300u + static_cast<std::uint32_t>(i * 17);
  auto va = hs::Load(d32, a.data);
  hs::Store(hs::TruncateTo(d8, va), d8, o8.data);
  for (std::size_t i = 0; i < n32; ++i) EXPECT_EQ(o8.data[i], static_cast<std::uint8_t>(a.data[i])) << "trunc i=" << i;
  // U8FromU32 (clamped values).
  Buf<std::uint32_t> c; for (std::size_t i = 0; i < n32; ++i) c.data[i] = static_cast<std::uint32_t>(i % 256);
  auto vc = hs::Load(d32, c.data);
  hs::Store(hs::U8FromU32(vc), d8, o8.data);
  for (std::size_t i = 0; i < n32; ++i) EXPECT_EQ(o8.data[i], static_cast<std::uint8_t>(c.data[i])) << "u8fromu32 i=" << i;
  // PromoteEvenTo / PromoteOddTo: i16 (2N lanes) -> i32 (N lanes), even/odd.
  const hs::ScalableTag<std::int16_t> d16;
  const hs::ScalableTag<std::int32_t> di32;
  const std::size_t n16 = hs::Lanes(d16), ni32 = hs::Lanes(di32);
  if (n16 == 2 * ni32) {
    Buf<std::int16_t> s; for (std::size_t i = 0; i < n16; ++i) s.data[i] = static_cast<std::int16_t>(i * 5 - 200);
    auto vs = hs::Load(d16, s.data);
    Buf<std::int32_t> oe;
    hs::Store(hs::PromoteEvenTo(di32, vs), di32, oe.data);
    for (std::size_t i = 0; i < ni32; ++i) EXPECT_EQ(oe.data[i], static_cast<std::int32_t>(s.data[2*i])) << "promEven i=" << i;
    hs::Store(hs::PromoteOddTo(di32, vs), di32, oe.data);
    for (std::size_t i = 0; i < ni32; ++i) EXPECT_EQ(oe.data[i], static_cast<std::int32_t>(s.data[2*i+1])) << "promOdd i=" << i;
  }
}

// ---- B-2 swizzle completion ---------------------------------------------------------
TEST(Vec, SwizzleCompletion) {
  const hs::ScalableTag<std::int32_t> d;
  const std::size_t n = hs::Lanes(d);
  Buf<std::int32_t> a, b, out;
  for (std::size_t i = 0; i < n; ++i) { a.data[i] = static_cast<std::int32_t>(i); b.data[i] = static_cast<std::int32_t>(100 + i); }
  auto va = hs::Load(d, a.data), vb = hs::Load(d, b.data);
  // Reverse4: within each group of 4, out[i]=in[i^3]. (n>=4 on sse4+; scalar n=1 skips.)
  if (n >= 4) {
    hs::Store(hs::Reverse4(d, va), d, out.data);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], a.data[(i & ~std::size_t{3}) | ((i & 3) ^ 3)]) << "rev4 i=" << i;
  }
  // InterleaveEven/Odd.
  if (n >= 2) {
    hs::Store(hs::InterleaveEven(va, vb), d, out.data);
    for (std::size_t i = 0; i < n/2; ++i) { EXPECT_EQ(out.data[2*i], a.data[2*i]); EXPECT_EQ(out.data[2*i+1], b.data[2*i]); }
    hs::Store(hs::InterleaveOdd(va, vb), d, out.data);
    for (std::size_t i = 0; i < n/2; ++i) { EXPECT_EQ(out.data[2*i], a.data[2*i+1]); EXPECT_EQ(out.data[2*i+1], b.data[2*i+1]); }
  }
  // Slide1Up/Down (zero-fill).
  hs::Store(hs::Slide1Up(d, va), d, out.data);
  EXPECT_EQ(out.data[0], 0); for (std::size_t i = 1; i < n; ++i) EXPECT_EQ(out.data[i], a.data[i-1]) << "s1up i=" << i;
  hs::Store(hs::Slide1Down(d, va), d, out.data);
  if (n) EXPECT_EQ(out.data[n-1], 0); for (std::size_t i = 0; i+1 < n; ++i) EXPECT_EQ(out.data[i], a.data[i+1]) << "s1dn i=" << i;
  // SlideUpLanes(amt=2).
  if (n >= 2) {
    hs::Store(hs::SlideUpLanes(d, va, 2), d, out.data);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], (i >= 2) ? a.data[i-2] : 0) << "slup i=" << i;
  }
  // BroadcastLane<1> (n>=2).
  if (n >= 2) {
    hs::Store(hs::BroadcastLane<1>(va), d, out.data);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], a.data[1]) << "bcast i=" << i;
  }
  // ZipLower == InterleaveLower.
  hs::Store(hs::ZipLower(va, vb), d, out.data);
  Buf<std::int32_t> il; hs::Store(hs::InterleaveLower(va, vb), d, il.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], il.data[i]) << "zip i=" << i;
  // ReverseBits / ReverseLaneBytes on u32.
  const hs::ScalableTag<std::uint32_t> du;
  Buf<std::uint32_t> uu, uo;
  for (std::size_t i = 0; i < hs::Lanes(du); ++i) uu.data[i] = 0x12345678u + static_cast<std::uint32_t>(i);
  auto vu = hs::Load(du, uu.data);
  hs::Store(hs::ReverseLaneBytes(vu), du, uo.data);
  for (std::size_t i = 0; i < hs::Lanes(du); ++i) {
    std::uint32_t x = uu.data[i];
    std::uint32_t bswap = ((x & 0xFFu) << 24) | ((x & 0xFF00u) << 8) | ((x >> 8) & 0xFF00u) | (x >> 24);
    EXPECT_EQ(uo.data[i], bswap) << "revbytes i=" << i;
  }
  hs::Store(hs::ReverseBits(vu), du, uo.data);
  for (std::size_t i = 0; i < hs::Lanes(du); ++i) {
    std::uint32_t x = uu.data[i], y = 0; for (int b2 = 0; b2 < 32; ++b2) { y = (y<<1)|(x&1); x >>= 1; }
    EXPECT_EQ(uo.data[i], y) << "revbits i=" << i;
  }
}

// ---- B-1 Masked* family -------------------------------------------------------------
// Masked<Op> zeroes off-lanes; Masked<Op>Or fills them from `no`. Check against a per-lane
// reference driven by the same mask (lane i active iff i is even).
TEST(Vec, MaskedFamily) {
  const hs::ScalableTag<std::int32_t> d;
  const std::size_t n = hs::Lanes(d);
  Buf<std::int32_t> a, b, no, out;
  for (std::size_t i = 0; i < n; ++i) { a.data[i] = static_cast<std::int32_t>(i + 5); b.data[i] = static_cast<std::int32_t>(i + 2); no.data[i] = -7; }
  auto va = hs::Load(d, a.data), vb = hs::Load(d, b.data), vno = hs::Load(d, no.data);
  // mask: even lanes active (lane i & 1 == 0). Build via Eq(iota&1, 0)-style: use a vector.
  Buf<std::int32_t> par; for (std::size_t i = 0; i < n; ++i) par.data[i] = static_cast<std::int32_t>(i % 2);
  auto m = hs::Eq(hs::Load(d, par.data), hs::Set(d, 0));  // true on even lanes
  auto active = [&](std::size_t i){ return (i % 2) == 0; };
  hs::Store(hs::MaskedAdd(m, va, vb), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], active(i) ? a.data[i]+b.data[i] : 0) << "MaskedAdd i=" << i;
  hs::Store(hs::MaskedAddOr(m, vno, va, vb), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], active(i) ? a.data[i]+b.data[i] : -7) << "MaskedAddOr i=" << i;
  hs::Store(hs::MaskedMin(m, va, vb), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], active(i) ? std::min(a.data[i],b.data[i]) : 0) << "MaskedMin i=" << i;
  hs::Store(hs::MaskedAbs(m, hs::Sub(vb, va)), d, out.data);  // |b-a| on active lanes
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], active(i) ? std::abs(b.data[i]-a.data[i]) : 0) << "MaskedAbs i=" << i;
  // MaskedEq returns a mask ANDed with m: true only on active AND equal lanes.
  auto meq = hs::MaskedEq(m, va, va);  // va==va always true, so = m
  hs::Store(hs::VecFromMask(d, meq), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i] != 0, active(i)) << "MaskedEq i=" << i;
  // MaskedMulAdd(m,a,b,c)=a*b+c on active, 0 else.
  hs::Store(hs::MaskedMulAdd(m, va, vb, vno), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], active(i) ? a.data[i]*b.data[i]+no.data[i] : 0) << "MaskedMulAdd i=" << i;
}

TEST(Vec, MaskedShiftAndReduceCompletion) {
  const hs::ScalableTag<std::uint32_t> d;
  const std::size_t n = hs::Lanes(d);
  Buf<std::uint32_t> a, bits, flags, no, out;
  for (std::size_t i = 0; i < n; ++i) {
    a.data[i] = static_cast<std::uint32_t>(0x10203u + i * 17u);
    bits.data[i] = static_cast<std::uint32_t>(i % 4);
    flags.data[i] = static_cast<std::uint32_t>(i & 1u);
    no.data[i] = 0xDEADBEEFu;
  }
  const auto va = hs::Load(d, a.data);
  const auto vb = hs::Load(d, bits.data);
  const auto vno = hs::Load(d, no.data);
  const auto m = hs::Eq(hs::Load(d, flags.data), hs::Zero(d));
  const auto active = [](std::size_t i) { return (i & 1u) == 0; };
  hs::Store(hs::MaskedShl(m, va, vb), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], active(i) ? a.data[i] << bits.data[i] : 0u);
  hs::Store(hs::MaskedShlOr(m, vno, va, vb), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], active(i) ? a.data[i] << bits.data[i] : no.data[i]);
  hs::Store(hs::MaskedShr(m, va, vb), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], active(i) ? a.data[i] >> bits.data[i] : 0u);
  hs::Store(hs::MaskedShrOr(m, vno, va, vb), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], active(i) ? a.data[i] >> bits.data[i] : no.data[i]);
  hs::Store(hs::MaskedShiftLeft<3>(m, va), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], active(i) ? a.data[i] << 3 : 0u);
  hs::Store(hs::MaskedShiftLeftOr<3>(m, vno, va), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], active(i) ? a.data[i] << 3 : no.data[i]);
  hs::Store(hs::MaskedShiftRight<3>(m, va), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], active(i) ? a.data[i] >> 3 : 0u);
  hs::Store(hs::MaskedShiftRightOr<3>(m, vno, va), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], active(i) ? a.data[i] >> 3 : no.data[i]);

  std::uint32_t sum = 0, minv = std::numeric_limits<std::uint32_t>::max(), maxv = 0;
  for (std::size_t i = 0; i < n; ++i) if (active(i)) { sum += a.data[i]; minv = std::min(minv, a.data[i]); maxv = std::max(maxv, a.data[i]); }
  EXPECT_EQ(hs::MaskedReduceSum(d, m, va), sum);
  EXPECT_EQ(hs::MaskedReduceMin(d, m, va), minv);
  EXPECT_EQ(hs::MaskedReduceMax(d, m, va), maxv);
  const auto none = hs::Eq(hs::Load(d, flags.data), hs::Set(d, 7u));
  EXPECT_EQ(hs::MaskedReduceSum(d, none, va), 0u);
  EXPECT_EQ(hs::MaskedReduceMin(d, none, va), std::numeric_limits<std::uint32_t>::max());
  EXPECT_EQ(hs::MaskedReduceMax(d, none, va), 0u);

  const hs::ScalableTag<std::int32_t> di;
  Buf<std::int32_t> si, siflags;
  for (std::size_t i = 0; i < hs::Lanes(di); ++i) { si.data[i] = -static_cast<std::int32_t>(i + 1); siflags.data[i] = 1; }
  const auto svi = hs::Load(di, si.data);
  const auto sinone = hs::Eq(hs::Load(di, siflags.data), hs::Zero(di));
  EXPECT_EQ(hs::MaskedReduceMin(di, sinone, svi), std::numeric_limits<std::int32_t>::max());
  EXPECT_EQ(hs::MaskedReduceMax(di, sinone, svi), std::numeric_limits<std::int32_t>::lowest());

  const hs::ScalableTag<float> df;
  Buf<float> sf, sfflags;
  for (std::size_t i = 0; i < hs::Lanes(df); ++i) { sf.data[i] = -static_cast<float>(i + 1); sfflags.data[i] = 1.0f; }
  const auto svf = hs::Load(df, sf.data);
  const auto sfnone = hs::Eq(hs::Load(df, sfflags.data), hs::Zero(df));
  EXPECT_EQ(hs::MaskedReduceMin(df, sfnone, svf), std::numeric_limits<float>::infinity());
  EXPECT_EQ(hs::MaskedReduceMax(df, sfnone, svf), -std::numeric_limits<float>::infinity());
}

TEST(Vec, MaskedApproxConvertAndCountCompletion) {
  const hs::ScalableTag<float> df;
  const hs::ScalableTag<std::int32_t> di;
  const hs::ScalableTag<std::uint32_t> du;
  const std::size_t n = hs::Lanes(df);
  Buf<float> f, fflags, fout, fref;
  Buf<std::int32_t> flags, iout;
  Buf<std::uint32_t> u, uflags, uout;
  for (std::size_t i = 0; i < n; ++i) {
    f.data[i] = static_cast<float>(i + 1);
    fflags.data[i] = static_cast<float>(i & 1u);
    flags.data[i] = static_cast<std::int32_t>(i & 1u);
    uflags.data[i] = static_cast<std::uint32_t>(i & 1u);
    u.data[i] = i == 0 ? 0u : static_cast<std::uint32_t>(1u << (i % 31));
  }
  const auto vf = hs::Load(df, f.data);
  const auto mf = hs::Eq(hs::Load(df, fflags.data), hs::Zero(df));
  const auto mi = hs::Eq(hs::Load(di, flags.data), hs::Zero(di));
  hs::Store(hs::ApproximateReciprocal(vf), df, fref.data);
  hs::Store(hs::MaskedApproximateReciprocal(mf, vf), df, fout.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(fout.data[i], (i & 1u) == 0 ? fref.data[i] : 0.0f);
  hs::Store(hs::ApproximateReciprocalSqrt(vf), df, fref.data);
  hs::Store(hs::MaskedApproximateReciprocalSqrt(mf, vf), df, fout.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(fout.data[i], (i & 1u) == 0 ? fref.data[i] : 0.0f);
  hs::Store(hs::MaskedConvertTo(mi, di, vf), di, iout.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(iout.data[i], (i & 1u) == 0 ? static_cast<std::int32_t>(f.data[i]) : 0);
  const auto mu = hs::Eq(hs::Load(du, uflags.data), hs::Zero(du));
  hs::Store(hs::MaskedLeadingZeroCount(mu, hs::Load(du, u.data)), du, uout.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(uout.data[i], (i & 1u) == 0 ? std::countl_zero(u.data[i]) : 0u);
}

TEST(Vec, MaskedFixedPointAndWidenCompletion) {
  const hs::ScalableTag<std::int16_t> d16;
  const hs::ScalableTag<std::int32_t> d32;
  const std::size_t n16 = hs::Lanes(d16), n32 = hs::Lanes(d32);
  Buf<std::int16_t> a, b, no, qout, qref;
  Buf<std::int32_t> flags, wout, wref;
  for (std::size_t i = 0; i < n16; ++i) {
    a.data[i] = static_cast<std::int16_t>(i == 0 ? INT16_MIN : 1000 + i * 17);
    b.data[i] = static_cast<std::int16_t>(i == 0 ? INT16_MIN : 2000 - i * 13);
    no.data[i] = -123;
  }
  for (std::size_t i = 0; i < n32; ++i) flags.data[i] = static_cast<std::int32_t>(i & 1u);
  const auto va = hs::Load(d16, a.data), vb = hs::Load(d16, b.data), vno = hs::Load(d16, no.data);
  Buf<std::int16_t> flags16;
  for (std::size_t i = 0; i < n16; ++i) flags16.data[i] = static_cast<std::int16_t>(i & 1u);
  const auto m16 = hs::Eq(hs::Load(d16, flags16.data), hs::Zero(d16));
  hs::Store(hs::MulFixedPoint15(va, vb), d16, qref.data);
  hs::Store(hs::MaskedMulFixedPoint15(m16, va, vb), d16, qout.data);
  for (std::size_t i = 0; i < n16; ++i) EXPECT_EQ(qout.data[i], (i & 1u) == 0 ? qref.data[i] : 0);
  hs::Store(hs::MaskedSatAddOr(m16, vno, va, vb), d16, qout.data);
  hs::Store(hs::MaskedSaturatedAddOr(m16, vno, va, vb), d16, qref.data);
  for (std::size_t i = 0; i < n16; ++i) EXPECT_EQ(qout.data[i], qref.data[i]);
  hs::Store(hs::MaskedSatSubOr(m16, vno, va, vb), d16, qout.data);
  hs::Store(hs::MaskedSaturatedSubOr(m16, vno, va, vb), d16, qref.data);
  for (std::size_t i = 0; i < n16; ++i) EXPECT_EQ(qout.data[i], qref.data[i]);
  const auto m32 = hs::Eq(hs::Load(d32, flags.data), hs::Zero(d32));
  hs::Store(hs::WidenMulPairwiseAdd(d32, va, vb), d32, wref.data);
  hs::Store(hs::MaskedWidenMulPairwiseAdd(d32, m32, va, vb), d32, wout.data);
  for (std::size_t i = 0; i < n32; ++i) EXPECT_EQ(wout.data[i], (i & 1u) == 0 ? wref.data[i] : 0);
}

TEST(Vec, MaskedComplexCompletion) {
  const hs::ScalableTag<float> d;
  const std::size_t n = hs::Lanes(d);
  Buf<float> a, b, c, no, flags, out, ref;
  for (std::size_t i = 0; i < n; ++i) {
    a.data[i] = static_cast<float>(i + 1);
    b.data[i] = static_cast<float>(i + 3) * 0.25f;
    c.data[i] = static_cast<float>(i + 5) * -0.5f;
    no.data[i] = -77.0f;
    flags.data[i] = static_cast<float>((i / 2) & 1u);
  }
  const auto va = hs::Load(d, a.data), vb = hs::Load(d, b.data), vc = hs::Load(d, c.data);
  const auto vno = hs::Load(d, no.data);
  const auto m = hs::Eq(hs::Load(d, flags.data), hs::Zero(d));
  const auto check = [&](auto got, auto want, float inactive) {
    hs::Store(got, d, out.data); hs::Store(want, d, ref.data);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_EQ(out.data[i], ((i / 2) & 1u) == 0 ? ref.data[i] : inactive) << "i=" << i;
  };
  check(hs::MaskedMulComplex(m, va, vb), hs::MulComplex(va, vb), 0.0f);
  check(hs::MaskedMulComplexOr(m, vno, va, vb), hs::MulComplex(va, vb), -77.0f);
  check(hs::MaskedMulComplexConj(m, va, vb), hs::MulComplexConj(va, vb), 0.0f);
  check(hs::MaskedMulComplexAdd(m, va, vb, vc), hs::MulComplexAdd(va, vb, vc), 0.0f);
  check(hs::MaskedMulComplexConjAdd(m, va, vb, vc), hs::MulComplexConjAdd(va, vb, vc), 0.0f);
}

// ---- A-2 mask-domain logic ----------------------------------------------------------
// Build two masks from comparisons, then check the mask And/Or/Xor/AndNot/Not/ExclusiveNeither
// against per-lane bool references, plus IfThenZeroElse and MaskFromVec. Uses i32 lanes.
TEST(Vec, MaskDomainLogic) {
  const hs::ScalableTag<std::int32_t> d;
  const std::size_t n = hs::Lanes(d);
  Buf<std::int32_t> a, b, out;
  for (std::size_t i = 0; i < n; ++i) { a.data[i] = static_cast<std::int32_t>(i); b.data[i] = 3; }
  auto va = hs::Load(d, a.data), vb = hs::Load(d, b.data);
  auto ma = hs::Lt(va, vb);   // lane i true iff i < 3
  auto mb = hs::Gt(va, hs::Set(d, 1));  // lane i true iff i > 1
  auto ref_a = [&](std::size_t i) { return a.data[i] < 3; };
  auto ref_b = [&](std::size_t i) { return a.data[i] > 1; };
  // Materialize a mask to a vector (0xFFFFFFFF / 0) and read lanes to check truth.
  auto check = [&](auto mask, auto ref, const char* tag) {
    hs::Store(hs::VecFromMask(d, mask), d, out.data);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_EQ(out.data[i] != 0, ref(i)) << tag << " i=" << i;
  };
  check(hs::And(ma, mb), [&](std::size_t i){ return ref_a(i) && ref_b(i); }, "And");
  check(hs::Or(ma, mb), [&](std::size_t i){ return ref_a(i) || ref_b(i); }, "Or");
  check(hs::Xor(ma, mb), [&](std::size_t i){ return ref_a(i) != ref_b(i); }, "Xor");
  check(hs::AndNot(ma, mb), [&](std::size_t i){ return !ref_a(i) && ref_b(i); }, "AndNot");
  check(hs::Not(ma), [&](std::size_t i){ return !ref_a(i); }, "Not");
  check(hs::ExclusiveNeither(ma, mb), [&](std::size_t i){ return !ref_a(i) && !ref_b(i); }, "Nor");
  // IfThenZeroElse: zero where ma true, else vb.
  hs::Store(hs::IfThenZeroElse(ma, vb), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], ref_a(i) ? 0 : 3) << "ifzeroelse i=" << i;
  // MaskFromVec: build a vector whose lanes have the sign bit set on odd lanes.
  Buf<std::int32_t> s;
  for (std::size_t i = 0; i < n; ++i) s.data[i] = (i % 2) ? -1 : 0;  // MSB set on odd
  auto mfv = hs::MaskFromVec(hs::Load(d, s.data));
  hs::Store(hs::VecFromMask(d, mfv), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i] != 0, (i % 2) != 0) << "maskfromvec i=" << i;
}

// BitsFromMask/LoadMaskBits round-trip + prefix-scan (SetBeforeFirst/OnlyFirst/etc).
TEST(Vec, MaskBitsAndPrefixScan) {
  const hs::ScalableTag<std::int32_t> d;
  const std::size_t n = hs::Lanes(d);
  Buf<std::int32_t> a;
  for (std::size_t i = 0; i < n; ++i) a.data[i] = (i == 2 || i == 5 % n) ? 1 : 0;  // some true lanes
  auto m = hs::Eq(hs::Load(d, a.data), hs::Set(d, 1));
  // BitsFromMask -> LoadMaskBits round-trips the mask.
  std::uint64_t bits = hs::BitsFromMask(d, m);
  auto m2 = hs::LoadMaskBits(d, bits);
  Buf<std::int32_t> o1, o2;
  hs::Store(hs::VecFromMask(d, m), d, o1.data);
  hs::Store(hs::VecFromMask(d, m2), d, o2.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o1.data[i], o2.data[i]) << "roundtrip i=" << i;
  // First true lane index (find via bits).
  std::ptrdiff_t first = -1;
  for (std::size_t i = 0; i < n; ++i) if (a.data[i]) { first = static_cast<std::ptrdiff_t>(i); break; }
  ASSERT_GE(first, 0);
  // SetBeforeFirst: lanes strictly before `first`.
  hs::Store(hs::VecFromMask(d, hs::SetBeforeFirst(m)), d, o1.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o1.data[i] != 0, static_cast<std::ptrdiff_t>(i) < first) << "before i=" << i;
  // SetOnlyFirst: only `first`.
  hs::Store(hs::VecFromMask(d, hs::SetOnlyFirst(m)), d, o1.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o1.data[i] != 0, static_cast<std::ptrdiff_t>(i) == first) << "only i=" << i;
  // SetAtOrAfterFirst: lanes >= first.
  hs::Store(hs::VecFromMask(d, hs::SetAtOrAfterFirst(m)), d, o1.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o1.data[i] != 0, static_cast<std::ptrdiff_t>(i) >= first) << "atorafter i=" << i;
  // SetAtOrBeforeFirst: lanes <= first.
  hs::Store(hs::VecFromMask(d, hs::SetAtOrBeforeFirst(m)), d, o1.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o1.data[i] != 0, static_cast<std::ptrdiff_t>(i) <= first) << "atorbefore i=" << i;
  // FindKnownFirstTrue matches.
  EXPECT_EQ(static_cast<std::ptrdiff_t>(hs::FindKnownFirstTrue(d, m)), first);
}

// Highway public-op-face parity wrappers (audit 2026-07-12): SetTableIndices / CompressBlocksNot /
// ZeroExtendResizeBitCast / Dup128MaskFromMaskBits -- each vs its underlying primitive.
TEST(Vec, HighwayParityWrappers) {
  const hs::ScalableTag<std::int32_t> d;
  const std::size_t n = hs::Lanes(d);
  Buf<std::int32_t> a, idx, out, ref;
  for (std::size_t i = 0; i < n; ++i) { a.data[i] = static_cast<std::int32_t>(i * 10 + 1); idx.data[i] = static_cast<std::int32_t>(n - 1 - i); }
  const auto va = hs::Load(d, a.data);
  // SetTableIndices(d, ptr) == IndicesFromVec(d, LoadU(ptr)): reverse permute.
  hs::Store(hs::TableLookupLanes(va, hs::SetTableIndices(d, idx.data)), d, out.data);
  hs::Store(hs::TableLookupLanes(va, hs::IndicesFromVec(d, hs::Load(d, idx.data))), d, ref.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], ref.data[i]) << "SetTableIndices i=" << i;

  // ZeroExtendResizeBitCast == ResizeBitCast (same-width identity here: i32 tag -> i32 tag).
  hs::Store(hs::ZeroExtendResizeBitCast(d, d, va), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], a.data[i]) << "ZeroExtendResizeBitCast i=" << i;

  // CompressBlocksNot(v, m) == CompressNot(v, m) on u64.
  {
    const hs::ScalableTag<std::uint64_t> d64;
    const std::size_t n64 = hs::Lanes(d64);
    Buf<std::uint64_t> b, c1, c2;
    for (std::size_t i = 0; i < n64; ++i) b.data[i] = i;
    const auto vb = hs::Load(d64, b.data);
    const auto mb = hs::Eq(hs::And(vb, hs::Set(d64, 1)), hs::Set(d64, 1));  // odd lanes true
    hs::Store(hs::CompressBlocksNot(vb, mb), d64, c1.data);
    hs::Store(hs::CompressNot(vb, mb), d64, c2.data);
    for (std::size_t i = 0; i < n64; ++i) EXPECT_EQ(c1.data[i], c2.data[i]) << "CompressBlocksNot i=" << i;
  }

  // Dup128MaskFromMaskBits: low-128-block bits replicated to every block. For i32, a 128-bit block
  // is 4 lanes; bits=0b0101 (lanes 0,2 true) must hold in each block.
  {
    const unsigned block_bits = 0b0101u;  // lanes 0 and 2 of each 128-bit block
    const auto m = hs::Dup128MaskFromMaskBits(d, block_bits);
    hs::Store(hs::VecFromMask(d, m), d, out.data);
    for (std::size_t i = 0; i < n; ++i) {
      const bool want = ((block_bits >> (i % 4)) & 1u) != 0;  // per-4-lane block pattern
      EXPECT_EQ(out.data[i] != 0, want) << "Dup128MaskFromMaskBits i=" << i;
    }
  }
}

// Prefix-scan on i8 lanes with the ONLY true lane at the LAST position (lane N-1). On avx2
// N=32, so SetAtOrBeforeFirst computed `1u << (countr_zero+1)` = `1u << 32` = shift UB before
// the fix (x86 masked it to 1u<<0 -> upto=0 -> wrong all-clear instead of all-set). This test
// pins that corner across every backend/width.
TEST(Vec, PrefixScanLastLaneI8) {
  const hs::ScalableTag<std::int8_t> d;
  const std::size_t n = hs::Lanes(d);
  Buf<std::int8_t> a{};
  a.data[n - 1] = 1;  // sole true lane at the very end
  auto m = hs::Eq(hs::Load(d, a.data), hs::Set(d, std::int8_t{1}));
  Buf<std::int8_t> o;
  // SetAtOrBeforeFirst: every lane (<= the last-and-only true lane) must be set.
  hs::Store(hs::VecFromMask(d, hs::SetAtOrBeforeFirst(m)), d, o.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_NE(o.data[i], 0) << "atorbefore-last i=" << i;
  // SetBeforeFirst: all lanes strictly before the last are set; the last itself is clear.
  hs::Store(hs::VecFromMask(d, hs::SetBeforeFirst(m)), d, o.data);
  for (std::size_t i = 0; i + 1 < n; ++i) EXPECT_NE(o.data[i], 0) << "before-last i=" << i;
  EXPECT_EQ(o.data[n - 1], 0) << "before-last tail";
}

// Float rounding to an integral value (result stays float): Round(nearest-even), Floor
// (toward -inf), Ceil (toward +inf), Trunc (toward zero). Feed .5 ties and both signs so
// nearest-even vs directed rounding actually diverge, vs the libm reference (== scalar).
template <typename T>
void CheckRounding() {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  Buf<T> a, out;
  // values spanning negatives/positives with .25/.5/.75 fractions incl exact ties
  const T src[] = {static_cast<T>(-2.5), static_cast<T>(-1.5), static_cast<T>(-0.5),
                   static_cast<T>(0.5),  static_cast<T>(1.5),  static_cast<T>(2.5),
                   static_cast<T>(2.25), static_cast<T>(-2.75)};
  for (std::size_t i = 0; i < n; ++i) a.data[i] = src[i % 8];
  auto v = hs::Load(d, a.data);
  hs::Store(hs::Round(v), d, out.data);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(out.data[i], static_cast<T>(std::nearbyint(a.data[i]))) << "round i=" << i;
  hs::Store(hs::Floor(v), d, out.data);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(out.data[i], static_cast<T>(std::floor(a.data[i]))) << "floor i=" << i;
  hs::Store(hs::Ceil(v), d, out.data);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(out.data[i], static_cast<T>(std::ceil(a.data[i]))) << "ceil i=" << i;
  hs::Store(hs::Trunc(v), d, out.data);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(out.data[i], static_cast<T>(std::trunc(a.data[i]))) << "trunc i=" << i;
}
TEST(Vec, RoundingFloat) { CheckRounding<float>(); }
TEST(Vec, RoundingDouble) { CheckRounding<double>(); }

// Masked / partial load-store (tail handling): FirstN builds a first-n-lanes mask;
// MaskedLoad zeroes off lanes; MaskedLoadOr keeps 'no' off lanes; BlendedStore writes
// only on-lanes; LoadN/StoreN take a runtime count. Sweep k=0..n so partial + full +
// empty tails are all exercised, width-agnostic (naive per-lane reference).
template <typename T>
void CheckMaskedTail() {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  Buf<T> in, out;
  for (std::size_t i = 0; i < n; ++i) in.data[i] = static_cast<T>(i + 1);

  for (std::size_t k = 0; k <= n; ++k) {
    auto m = hs::FirstN(d, k);
    EXPECT_EQ(hs::CountTrue(d, m), k) << "FirstN count k=" << k;

    hs::Store(hs::MaskedLoad(m, d, in.data), d, out.data);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_EQ(out.data[i], i < k ? in.data[i] : T{0}) << "MaskedLoad k=" << k << " i=" << i;

    Buf<T> no;
    for (std::size_t i = 0; i < n; ++i) no.data[i] = static_cast<T>(90 + i);
    hs::Store(hs::MaskedLoadOr(hs::Load(d, no.data), m, d, in.data), d, out.data);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_EQ(out.data[i], i < k ? in.data[i] : no.data[i]) << "MaskedLoadOr k=" << k;

    for (std::size_t i = 0; i < n; ++i) out.data[i] = static_cast<T>(7);
    hs::BlendedStore(hs::Load(d, in.data), m, d, out.data);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_EQ(out.data[i], i < k ? in.data[i] : T{7}) << "BlendedStore k=" << k;

    hs::Store(hs::LoadN(d, in.data, k), d, out.data);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_EQ(out.data[i], i < k ? in.data[i] : T{0}) << "LoadN k=" << k;

    for (std::size_t i = 0; i < n; ++i) out.data[i] = static_cast<T>(5);
    hs::StoreN(hs::Load(d, in.data), d, out.data, k);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_EQ(out.data[i], i < k ? in.data[i] : T{5}) << "StoreN k=" << k;
  }
}
TEST(Vec, MaskedTailFloat) { CheckMaskedTail<float>(); }
TEST(Vec, MaskedTailDouble) { CheckMaskedTail<double>(); }
TEST(Vec, MaskedTailInt32) { CheckMaskedTail<std::int32_t>(); }
TEST(Vec, MaskedTailInt8) { CheckMaskedTail<std::int8_t>(); }
TEST(Vec, MaskedTailInt16) { CheckMaskedTail<std::int16_t>(); }
TEST(Vec, MaskedTailInt64) { CheckMaskedTail<std::int64_t>(); }
TEST(Vec, MaskedTailUint8) { CheckMaskedTail<std::uint8_t>(); }

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
  hs::Store(hs::MulSub(va, vb, vc), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_NEAR(out.data[i], a.data[i] * b.data[i] - c.data[i], 1e-3f);
  hs::Store(hs::NegMulSub(va, vb, vc), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_NEAR(out.data[i], -(a.data[i] * b.data[i]) - c.data[i], 1e-3f);
  hs::Store(hs::MulAddSub(va, vb, vc), d, out.data);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_NEAR(out.data[i], a.data[i] * b.data[i] + (i % 2 ? c.data[i] : -c.data[i]), 1e-3f);
  hs::Store(hs::MulSubAdd(va, vb, vc), d, out.data);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_NEAR(out.data[i], a.data[i] * b.data[i] + (i % 2 ? -c.data[i] : c.data[i]), 1e-3f);
}

TEST(Vec, MulSignVariantsUint32) {
  const hs::ScalableTag<std::uint32_t> d;
  const std::size_t n = hs::Lanes(d);
  Buf<std::uint32_t> a, b, c, out;
  for (std::size_t i = 0; i < n; ++i) {
    a.data[i] = static_cast<std::uint32_t>(i + 2);
    b.data[i] = static_cast<std::uint32_t>(i * 3 + 5);
    c.data[i] = static_cast<std::uint32_t>(i * 7 + 1);
  }
  const auto va = hs::Load(d, a.data), vb = hs::Load(d, b.data), vc = hs::Load(d, c.data);
  hs::Store(hs::MulSub(va, vb, vc), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], a.data[i] * b.data[i] - c.data[i]);
  hs::Store(hs::NegMulSub(va, vb, vc), d, out.data);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(out.data[i], std::uint32_t{0} - (a.data[i] * b.data[i] + c.data[i]));
  hs::Store(hs::MulAddSub(va, vb, vc), d, out.data);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(out.data[i], a.data[i] * b.data[i] +
                              (i % 2 ? c.data[i] : std::uint32_t{0} - c.data[i]));
  hs::Store(hs::MulSubAdd(va, vb, vc), d, out.data);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(out.data[i], a.data[i] * b.data[i] +
                              (i % 2 ? std::uint32_t{0} - c.data[i] : c.data[i]));
}

TEST(Vec, MagnitudeMinMax) {
  {
    const hs::ScalableTag<std::int32_t> d;
    const std::size_t n = hs::Lanes(d);
    Buf<std::int32_t> a, b, out;
    const std::int32_t av[] = {-5, 5, std::numeric_limits<std::int32_t>::min(), -7, 8};
    const std::int32_t bv[] = {3, -5, 4, 7, -9};
    const std::int32_t min_ref[] = {3, -5, 4, -7, 8};
    const std::int32_t max_ref[] = {-5, 5, std::numeric_limits<std::int32_t>::min(), 7, -9};
    for (std::size_t i = 0; i < n; ++i) { a.data[i] = av[i % 5]; b.data[i] = bv[i % 5]; }
    const auto va = hs::Load(d, a.data), vb = hs::Load(d, b.data);
    hs::Store(hs::MinMagnitude(va, vb), d, out.data);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], min_ref[i % 5]) << "min lane " << i;
    hs::Store(hs::MaxMagnitude(va, vb), d, out.data);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], max_ref[i % 5]) << "max lane " << i;
  }
  {
    const hs::ScalableTag<std::uint32_t> d;
    const std::size_t n = hs::Lanes(d);
    Buf<std::uint32_t> a, b, out;
    for (std::size_t i = 0; i < n; ++i) { a.data[i] = static_cast<std::uint32_t>(i + 9); b.data[i] = static_cast<std::uint32_t>(i + 3); }
    const auto va = hs::Load(d, a.data), vb = hs::Load(d, b.data);
    hs::Store(hs::MinMagnitude(va, vb), d, out.data);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], b.data[i]);
    hs::Store(hs::MaxMagnitude(va, vb), d, out.data);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], a.data[i]);
  }
  {
    const hs::ScalableTag<float> d;
    const std::size_t n = hs::Lanes(d);
    ASSERT_GE(n, 4u);
    Buf<float> a, b, out;
    const float av[] = {0.0f, -0.0f, 4.0f, -4.0f};
    const float bv[] = {-0.0f, 0.0f, -4.0f, 4.0f};
#if defined(HYLE_SIMD_TARGET_NEON)
    // Highway's AArch64 Min/Max are IEEE minimumNumber/maximumNumber, which
    // canonicalize an opposite-signed zero tie to -0 for Min and +0 for Max.
    const float min_ref[] = {-0.0f, -0.0f, -4.0f, -4.0f};
    const float max_ref[] = {0.0f, 0.0f, 4.0f, 4.0f};
#else
    const float min_ref[] = {-0.0f, 0.0f, -4.0f, -4.0f};
    const float max_ref[] = {0.0f, -0.0f, 4.0f, 4.0f};
#endif
    for (std::size_t i = 0; i < n; ++i) { a.data[i] = av[i % 4]; b.data[i] = bv[i % 4]; }
    const auto va = hs::Load(d, a.data), vb = hs::Load(d, b.data);
    hs::Store(hs::MinMagnitude(va, vb), d, out.data);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_EQ(std::bit_cast<std::uint32_t>(out.data[i]), std::bit_cast<std::uint32_t>(min_ref[i % 4]))
          << "min lane " << i;
    hs::Store(hs::MaxMagnitude(va, vb), d, out.data);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_EQ(std::bit_cast<std::uint32_t>(out.data[i]), std::bit_cast<std::uint32_t>(max_ref[i % 4]))
          << "max lane " << i;
  }
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
TEST(Vec, CompareBlendInt8) { CheckCompareBlend<std::int8_t>(); }
TEST(Vec, CompareBlendUint8) { CheckCompareBlend<std::uint8_t>(); }
TEST(Vec, CompareBlendInt16) { CheckCompareBlend<std::int16_t>(); }
TEST(Vec, CompareBlendUint16) { CheckCompareBlend<std::uint16_t>(); }
TEST(Vec, CompareBlendInt64) { CheckCompareBlend<std::int64_t>(); }
TEST(Vec, CompareBlendUint64) { CheckCompareBlend<std::uint64_t>(); }

// FindLastTrue (highest set lane) + VecFromMask (mask->0xFF vector) + SumsOf8 (u8->u64 SAD).
// These three are the string count/rfind primitives. Byte-lane focused (that's the consumer).
TEST(Vec, FindLastTrueByte) {
  const hs::ScalableTag<std::uint8_t> d;
  const std::size_t n = hs::Lanes(d);
  Buf<std::uint8_t> a;
  for (std::size_t i = 0; i < n; ++i) a.data[i] = 'a';
  a.data[1] = 'z'; a.data[n - 2] = 'z';  // matches at 1 and n-2
  auto m = hs::Eq(hs::Load(d, a.data), hs::Set(d, static_cast<std::uint8_t>('z')));
  EXPECT_EQ(hs::FindLastTrue(d, m), static_cast<std::ptrdiff_t>(n - 2));
  EXPECT_EQ(hs::FindFirstTrue(d, m), 1);
  auto m0 = hs::Eq(hs::Load(d, a.data), hs::Set(d, static_cast<std::uint8_t>('Q')));
  EXPECT_EQ(hs::FindLastTrue(d, m0), -1);  // none set
}
TEST(Vec, VecFromMaskAndSumsOf8Count) {
  // The count core: SumsOf8(VecFromMask(Eq(v,needle))) sums 255*matches per 8-lane group.
  const hs::ScalableTag<std::uint8_t> d;
  const std::size_t n = hs::Lanes(d);
  ASSERT_EQ(n % 8, 0u);
  Buf<std::uint8_t> a;
  std::size_t want = 0;
  for (std::size_t i = 0; i < n; ++i) { a.data[i] = (i % 4 == 0) ? 'x' : 'o'; if (a.data[i] == 'x') ++want; }
  auto m = hs::Eq(hs::Load(d, a.data), hs::Set(d, static_cast<std::uint8_t>('x')));
  auto mv = hs::VecFromMask(d, m);
  // VecFromMask lane is all-ones (0xFF) where matched, 0 else.
  Buf<std::uint8_t> mvb;
  hs::Store(mv, d, mvb.data);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(mvb.data[i], a.data[i] == 'x' ? 0xFFu : 0x00u) << "VecFromMask " << i;
  const hs::ScalableTag<std::uint64_t> d64;
  std::uint64_t total = hs::ReduceSum(d64, hs::SumsOf8(mv));
  EXPECT_EQ(total / 255u, want);           // SAD-style count matches
  EXPECT_EQ(hs::CountTrue(d, m), want);    // cross-check the popcount path
}

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
TEST(Vec, ReductionsInt16) { CheckReductions<std::int16_t>(); }
TEST(Vec, ReductionsUint16) { CheckReductions<std::uint16_t>(); }
TEST(Vec, ReductionsInt64) { CheckReductions<std::int64_t>(); }
TEST(Vec, ReductionsUint64) { CheckReductions<std::uint64_t>(); }

// ReduceMul / ReduceAnd / ReduceOr (family completion) over integer lanes. Small factors
// keep the product in range for every width; And/Or checked against a naive fold.
template <typename T>
void CheckReduceMulBitwise() {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  Buf<T> a;
  for (std::size_t i = 0; i < n; ++i) a.data[i] = static_cast<T>((i % 3) + 1);  // 1..3, product fits
  auto v = hs::Load(d, a.data);
  T mul = a.data[0], andv = a.data[0], orv = a.data[0];
  for (std::size_t i = 1; i < n; ++i) {
    mul = static_cast<T>(mul * a.data[i]);
    andv = static_cast<T>(andv & a.data[i]);
    orv = static_cast<T>(orv | a.data[i]);
  }
  EXPECT_EQ(hs::ReduceMul(d, v), mul);
  EXPECT_EQ(hs::ReduceAnd(d, v), andv);
  EXPECT_EQ(hs::ReduceOr(d, v), orv);
}
TEST(Vec, ReduceMulBitwiseInt32) { CheckReduceMulBitwise<std::int32_t>(); }
TEST(Vec, ReduceMulBitwiseUint32) { CheckReduceMulBitwise<std::uint32_t>(); }
TEST(Vec, ReduceMulBitwiseInt8) { CheckReduceMulBitwise<std::int8_t>(); }
TEST(Vec, ReduceMulBitwiseInt16) { CheckReduceMulBitwise<std::int16_t>(); }
TEST(Vec, ReduceMulBitwiseInt64) { CheckReduceMulBitwise<std::int64_t>(); }

// ReduceMul over float (exact for small integer-valued factors).
TEST(Vec, ReduceMulFloat) {
  const hs::ScalableTag<float> d;
  const std::size_t n = hs::Lanes(d);
  Buf<float> a;
  float mul = 1.0f;
  for (std::size_t i = 0; i < n; ++i) { a.data[i] = static_cast<float>((i % 4) + 1); mul *= a.data[i]; }
  EXPECT_FLOAT_EQ(hs::ReduceMul(d, hs::Load(d, a.data)), mul);
}

// Broadcast<i>: every lane becomes lane i. Checked for a couple of fixed low indices
// (valid on every v1 width -- the narrowest full vector still has >= 2 lanes).
template <typename T>
void CheckBroadcast() {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  Buf<T> a, out;
  for (std::size_t i = 0; i < n; ++i) a.data[i] = static_cast<T>(i * 2 + 1);
  auto v = hs::Load(d, a.data);
  hs::Store(hs::Broadcast<0>(v), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], a.data[0]);
  hs::Store(hs::Broadcast<1>(v), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], a.data[1]);
}
TEST(Vec, BroadcastFloat) { CheckBroadcast<float>(); }
TEST(Vec, BroadcastDouble) { CheckBroadcast<double>(); }
TEST(Vec, BroadcastInt32) { CheckBroadcast<std::int32_t>(); }
TEST(Vec, BroadcastInt8) { CheckBroadcast<std::int8_t>(); }
TEST(Vec, BroadcastInt64) { CheckBroadcast<std::int64_t>(); }

// Width-changing promote/demote (half-at-a-time forms). On a fixed-width backend the narrow
// ScalableTag has 2x the lanes of the wide one, so PromoteLower/UpperTo map narrow(2n)->wide(n)
// and DemoteTo maps two wide(n)->narrow(2n). Checked against per-lane static_cast (promote)
// and a saturating reference (demote), width-agnostic via Lanes().
template <typename Narrow, typename Wide>
void CheckPromote() {
  const hs::ScalableTag<Narrow> dn;
  const hs::ScalableTag<Wide> dw;
  const std::size_t nn = hs::Lanes(dn), nw = hs::Lanes(dw);
  ASSERT_EQ(nn, 2 * nw);
  Buf<Narrow> in;
  for (std::size_t i = 0; i < nn; ++i)
    in.data[i] = static_cast<Narrow>((i % 2 ? -1 : 1) * static_cast<long long>(i % 50 + 1));
  auto v = hs::Load(dn, in.data);
  Buf<Wide> out;
  hs::Store(hs::PromoteLowerTo(dw, v), dw, out.data);
  for (std::size_t i = 0; i < nw; ++i) EXPECT_EQ(out.data[i], static_cast<Wide>(in.data[i])) << "lower i=" << i;
  hs::Store(hs::PromoteUpperTo(dw, v), dw, out.data);
  for (std::size_t i = 0; i < nw; ++i) EXPECT_EQ(out.data[i], static_cast<Wide>(in.data[nw + i])) << "upper i=" << i;
}
TEST(Vec, PromoteI8I16) { CheckPromote<std::int8_t, std::int16_t>(); }
TEST(Vec, PromoteU8U16) { CheckPromote<std::uint8_t, std::uint16_t>(); }
TEST(Vec, PromoteI16I32) { CheckPromote<std::int16_t, std::int32_t>(); }
TEST(Vec, PromoteU16U32) { CheckPromote<std::uint16_t, std::uint32_t>(); }
TEST(Vec, PromoteI32I64) { CheckPromote<std::int32_t, std::int64_t>(); }
TEST(Vec, PromoteU32U64) { CheckPromote<std::uint32_t, std::uint64_t>(); }

template <typename Wide, typename Narrow>
void CheckDemote() {
  const hs::ScalableTag<Wide> dw;
  const hs::ScalableTag<Narrow> dn;
  const std::size_t nw = hs::Lanes(dw);
  ASSERT_EQ(hs::Lanes(dn), 2 * nw);
  Buf<Wide> a, b;
  for (std::size_t i = 0; i < nw; ++i) {
    a.data[i] = static_cast<Wide>((i % 2 ? -1 : 1) * static_cast<long long>(i * 37 + 3));  // some exceed narrow range
    b.data[i] = static_cast<Wide>((i % 3 ? 1 : -1) * static_cast<long long>(i * 53 + 7));
  }
  constexpr long long lo = static_cast<long long>(std::numeric_limits<Narrow>::min());
  constexpr long long hi = static_cast<long long>(std::numeric_limits<Narrow>::max());
  auto clampn = [&](Wide x) { long long v = static_cast<long long>(x); return static_cast<Narrow>(v < lo ? lo : (v > hi ? hi : v)); };
  Buf<Narrow> out;
  hs::Store(hs::DemoteTo(dn, hs::Load(dw, a.data), hs::Load(dw, b.data)), dn, out.data);
  for (std::size_t i = 0; i < nw; ++i) EXPECT_EQ(out.data[i], clampn(a.data[i])) << "demote a i=" << i;
  for (std::size_t i = 0; i < nw; ++i) EXPECT_EQ(out.data[nw + i], clampn(b.data[i])) << "demote b i=" << i;
}
TEST(Vec, DemoteI16I8) { CheckDemote<std::int16_t, std::int8_t>(); }
TEST(Vec, DemoteI16U8) { CheckDemote<std::int16_t, std::uint8_t>(); }
TEST(Vec, DemoteI32I16) { CheckDemote<std::int32_t, std::int16_t>(); }
TEST(Vec, DemoteI32U16) { CheckDemote<std::int32_t, std::uint16_t>(); }

// ReorderDemote2To: same saturating narrowing as DemoteTo but lane ORDER IS UNSPECIFIED --
// scalar/NEON return the ordered [a...,b...] while the wide x86 backends return the cheaper
// block-interleaved permutation. So the CONTRACT is a multiset match, NOT positional. This
// test pins exactly that (order-independent) so callers can't accidentally rely on positions.
TEST(Vec, ReorderDemote2ToIsMultiset) {
  const hs::ScalableTag<std::int32_t> dw;
  const hs::ScalableTag<std::int16_t> dn;
  const std::size_t nw = hs::Lanes(dw);
  ASSERT_EQ(hs::Lanes(dn), 2 * nw);
  Buf<std::int32_t> a, b;
  for (std::size_t i = 0; i < nw; ++i) {
    a.data[i] = static_cast<std::int32_t>((i % 2 ? -1 : 1) * static_cast<long long>(i * 37 + 3));
    b.data[i] = static_cast<std::int32_t>((i % 3 ? 1 : -1) * static_cast<long long>(i * 53 + 7));
  }
  constexpr long long lo = std::numeric_limits<std::int16_t>::min();
  constexpr long long hi = std::numeric_limits<std::int16_t>::max();
  auto clampn = [&](std::int32_t x) { long long v = x; return static_cast<std::int16_t>(v < lo ? lo : (v > hi ? hi : v)); };
  Buf<std::int16_t> out;
  hs::Store(hs::ReorderDemote2To(dn, hs::Load(dw, a.data), hs::Load(dw, b.data)), dn, out.data);
  std::vector<std::int16_t> got(out.data, out.data + 2 * nw), want;
  for (std::size_t i = 0; i < nw; ++i) want.push_back(clampn(a.data[i]));
  for (std::size_t i = 0; i < nw; ++i) want.push_back(clampn(b.data[i]));
  std::sort(got.begin(), got.end()); std::sort(want.begin(), want.end());
  EXPECT_EQ(got, want) << "ReorderDemote2To multiset mismatch";
}

// f32<->f64 promote/demote (ordinary narrowing/widening cast, no saturation).
TEST(Vec, PromoteDemoteF32F64) {
  const hs::ScalableTag<float> df;
  const hs::ScalableTag<double> dd;
  const std::size_t nf = hs::Lanes(df), nd = hs::Lanes(dd);
  ASSERT_EQ(nf, 2 * nd);
  Buf<float> f;
  for (std::size_t i = 0; i < nf; ++i) f.data[i] = static_cast<float>(i) * 0.5f - 1.0f;
  auto vf = hs::Load(df, f.data);
  Buf<double> od;
  hs::Store(hs::PromoteLowerTo(dd, vf), dd, od.data);
  for (std::size_t i = 0; i < nd; ++i) EXPECT_DOUBLE_EQ(od.data[i], static_cast<double>(f.data[i]));
  hs::Store(hs::PromoteUpperTo(dd, vf), dd, od.data);
  for (std::size_t i = 0; i < nd; ++i) EXPECT_DOUBLE_EQ(od.data[i], static_cast<double>(f.data[nd + i]));
  Buf<double> a, b;
  for (std::size_t i = 0; i < nd; ++i) { a.data[i] = static_cast<double>(i) * 1.5 - 3.0; b.data[i] = static_cast<double>(i) * -2.25 + 1.0; }
  Buf<float> of;
  hs::Store(hs::DemoteTo(df, hs::Load(dd, a.data), hs::Load(dd, b.data)), df, of.data);
  for (std::size_t i = 0; i < nd; ++i) EXPECT_FLOAT_EQ(of.data[i], static_cast<float>(a.data[i]));
  for (std::size_t i = 0; i < nd; ++i) EXPECT_FLOAT_EQ(of.data[nd + i], static_cast<float>(b.data[i]));
}

// Per-lane variable shifts Shl/Shr: each lane shifts by its own count (from a 2nd vector).
// Signed Shr is arithmetic, unsigned logical (== scalar >>). 8-bit is not provided (no
// native x86 variable byte shift), so test 16/32/64 widths where every backend supports it.
template <typename T>
void CheckVarShift() {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  const int bits = sizeof(T) * 8;
  Buf<T> a, cnt, out;
  for (std::size_t i = 0; i < n; ++i) {
    a.data[i] = static_cast<T>((i % 2 ? -1 : 1) * static_cast<long long>(i * 7 + 1));
    cnt.data[i] = static_cast<T>(i % (bits - 1));  // 0..bits-2 (valid count)
  }
  auto va = hs::Load(d, a.data), vc = hs::Load(d, cnt.data);
  hs::Store(hs::Shl(va, vc), d, out.data);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(out.data[i], static_cast<T>(static_cast<std::make_unsigned_t<T>>(a.data[i]) << cnt.data[i])) << "shl i=" << i;
  hs::Store(hs::Shr(va, vc), d, out.data);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(out.data[i], static_cast<T>(a.data[i] >> cnt.data[i])) << "shr i=" << i;
}
TEST(Vec, VarShiftInt16) { CheckVarShift<std::int16_t>(); }
TEST(Vec, VarShiftUint16) { CheckVarShift<std::uint16_t>(); }
TEST(Vec, VarShiftInt32) { CheckVarShift<std::int32_t>(); }
TEST(Vec, VarShiftUint32) { CheckVarShift<std::uint32_t>(); }
TEST(Vec, VarShiftInt64) { CheckVarShift<std::int64_t>(); }
TEST(Vec, VarShiftUint64) { CheckVarShift<std::uint64_t>(); }

// MulHigh: top 16 bits of the i16/u16 product.
template <typename T>
void CheckMulHigh() {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  using W = std::conditional_t<std::is_signed_v<T>, std::int32_t, std::uint32_t>;
  Buf<T> a, b, out;
  for (std::size_t i = 0; i < n; ++i) { a.data[i] = static_cast<T>(i * 137 + 3); b.data[i] = static_cast<T>(i * 251 + 7); }
  hs::Store(hs::MulHigh(hs::Load(d, a.data), hs::Load(d, b.data)), d, out.data);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(out.data[i], static_cast<T>((static_cast<W>(a.data[i]) * static_cast<W>(b.data[i])) >> 16)) << "i=" << i;
}
TEST(Vec, MulHighInt16) { CheckMulHigh<std::int16_t>(); }
TEST(Vec, MulHighUint16) { CheckMulHigh<std::uint16_t>(); }

// WidenMulPairwiseAdd: multiply i16 pairs, sum adjacent into i32 (fixed-point dot core).
TEST(Vec, WidenMulPairwiseAdd) {
  const hs::ScalableTag<std::int16_t> d16;
  const hs::ScalableTag<std::int32_t> d32;
  const std::size_t n16 = hs::Lanes(d16), n32 = hs::Lanes(d32);
  ASSERT_EQ(n16, 2 * n32);
  Buf<std::int16_t> a, b;
  for (std::size_t i = 0; i < n16; ++i) {
    a.data[i] = static_cast<std::int16_t>((i % 2 ? -1 : 1) * static_cast<int>(i * 11 + 1));
    b.data[i] = static_cast<std::int16_t>(i * 13 + 2);
  }
  Buf<std::int32_t> out;
  hs::Store(hs::WidenMulPairwiseAdd(d32, hs::Load(d16, a.data), hs::Load(d16, b.data)), d32, out.data);
  for (std::size_t i = 0; i < n32; ++i)
    EXPECT_EQ(out.data[i], static_cast<std::int32_t>(a.data[2 * i]) * b.data[2 * i] +
                               static_cast<std::int32_t>(a.data[2 * i + 1]) * b.data[2 * i + 1]) << "i=" << i;
}

// MulEven/MulOdd: select alternating source lanes, widen once, then multiply.
template <typename T>
void CheckMulEvenOdd() {
  using W = hs::MakeWide<T>;
  const hs::ScalableTag<T> ds;
  const hs::RepartitionToWide<decltype(ds)> dw;
  const std::size_t ns = hs::Lanes(ds), nw = hs::Lanes(dw);
  ASSERT_EQ(ns, 2 * nw);
  Buf<T> a, b;
  Buf<W> out;
  for (std::size_t i = 0; i < ns; ++i) {
    const int sign = i % 3 == 0 ? -1 : 1;
    a.data[i] = static_cast<T>(sign * static_cast<int>(i + 2));
    b.data[i] = static_cast<T>(static_cast<int>(i * 5 + 3));
  }
  const auto va = hs::Load(ds, a.data), vb = hs::Load(ds, b.data);
  hs::Store(hs::MulEven(va, vb), dw, out.data);
  for (std::size_t i = 0; i < nw; ++i)
    EXPECT_EQ(out.data[i], static_cast<W>(a.data[2 * i]) * static_cast<W>(b.data[2 * i]))
        << "even lane " << i;
  hs::Store(hs::MulOdd(va, vb), dw, out.data);
  for (std::size_t i = 0; i < nw; ++i)
    EXPECT_EQ(out.data[i], static_cast<W>(a.data[2 * i + 1]) * static_cast<W>(b.data[2 * i + 1]))
        << "odd lane " << i;
}
TEST(Vec, MulEvenOddInt8) { CheckMulEvenOdd<std::int8_t>(); }
TEST(Vec, MulEvenOddUint8) { CheckMulEvenOdd<std::uint8_t>(); }
TEST(Vec, MulEvenOddInt16) { CheckMulEvenOdd<std::int16_t>(); }
TEST(Vec, MulEvenOddUint16) { CheckMulEvenOdd<std::uint16_t>(); }
TEST(Vec, MulEvenOddInt32) { CheckMulEvenOdd<std::int32_t>(); }
TEST(Vec, MulEvenOddUint32) { CheckMulEvenOdd<std::uint32_t>(); }

TEST(Vec, Bf16MulEvenOddAdd) {
  const hs::ScalableTag<std::uint16_t> dbf;
  const hs::ScalableTag<float> df;
  const std::size_t nbf = hs::Lanes(dbf), nf = hs::Lanes(df);
  ASSERT_EQ(nbf, 2 * nf);
  Buf<std::uint16_t> a, b;
  Buf<float> c, out;
  for (std::size_t i = 0; i < nbf; ++i) {
    a.data[i] = hs::f32_to_bf16(static_cast<float>(static_cast<int>(i % 7) - 3)).bits;
    b.data[i] = hs::f32_to_bf16(static_cast<float>(i % 5 + 1)).bits;
  }
  for (std::size_t i = 0; i < nf; ++i) c.data[i] = static_cast<float>(i) * 0.25f;
  const auto va = hs::Load(dbf, a.data), vb = hs::Load(dbf, b.data);
  const auto vc = hs::Load(df, c.data);
  hs::Store(hs::MulEvenAdd(df, va, vb, vc), df, out.data);
  for (std::size_t i = 0; i < nf; ++i) {
    const float expected = hs::bf16_to_f32(hs::bfloat16_t::from_bits(a.data[2 * i])) *
                               hs::bf16_to_f32(hs::bfloat16_t::from_bits(b.data[2 * i])) + c.data[i];
    EXPECT_FLOAT_EQ(out.data[i], expected) << "even lane " << i;
  }
  hs::Store(hs::MulOddAdd(df, va, vb, vc), df, out.data);
  for (std::size_t i = 0; i < nf; ++i) {
    const float expected = hs::bf16_to_f32(hs::bfloat16_t::from_bits(a.data[2 * i + 1])) *
                               hs::bf16_to_f32(hs::bfloat16_t::from_bits(b.data[2 * i + 1])) + c.data[i];
    EXPECT_FLOAT_EQ(out.data[i], expected) << "odd lane " << i;
  }
}

// AbsDiff: |a-b| per lane.
template <typename T>
void CheckAbsDiff() {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  Buf<T> a, b, out;
  for (std::size_t i = 0; i < n; ++i) { a.data[i] = static_cast<T>(i * 3 % 50); b.data[i] = static_cast<T>(i * 5 % 50); }
  hs::Store(hs::AbsDiff(hs::Load(d, a.data), hs::Load(d, b.data)), d, out.data);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(out.data[i], a.data[i] > b.data[i] ? static_cast<T>(a.data[i] - b.data[i]) : static_cast<T>(b.data[i] - a.data[i])) << "i=" << i;
}
TEST(Vec, AbsDiffInt32) { CheckAbsDiff<std::int32_t>(); }
TEST(Vec, AbsDiffUint8) { CheckAbsDiff<std::uint8_t>(); }
TEST(Vec, AbsDiffInt16) { CheckAbsDiff<std::int16_t>(); }
TEST(Vec, AbsDiffInt64) { CheckAbsDiff<std::int64_t>(); }
TEST(Vec, AbsDiffFloat) {
  const hs::ScalableTag<float> d;
  const std::size_t n = hs::Lanes(d);
  Buf<float> a, b, out;
  for (std::size_t i = 0; i < n; ++i) { a.data[i] = static_cast<float>(i) - 3.0f; b.data[i] = 2.0f * static_cast<float>(i) - 1.0f; }
  hs::Store(hs::AbsDiff(hs::Load(d, a.data), hs::Load(d, b.data)), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_FLOAT_EQ(out.data[i], std::abs(a.data[i] - b.data[i]));
}

// Swizzle (lane rearrangement). Semantics are Highway's: most ops are whole-vector, but
// InterleaveLower/Upper are PER-128-BIT-BLOCK (matters on 256b avx2 = 2 blocks; on 128b
// sse4/neon there's one block so it reduces to whole-vector). blk = lanes per 128b block.
template <typename T>
void CheckSwizzle() {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  const std::size_t blk = 16 / sizeof(T);
  Buf<T> a, b, out;
  for (std::size_t i = 0; i < n; ++i) { a.data[i] = static_cast<T>(i + 1); b.data[i] = static_cast<T>(100 + i); }
  auto va = hs::Load(d, a.data), vb = hs::Load(d, b.data);

  hs::Store(hs::OddEven(va, vb), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], (i & 1) ? a.data[i] : b.data[i]) << "OddEven " << i;
  hs::Store(hs::Reverse(d, va), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], a.data[n - 1 - i]) << "Reverse " << i;
  hs::Store(hs::Reverse2(d, va), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], a.data[i ^ 1]) << "Reverse2 " << i;
  hs::Store(hs::InterleaveLower(va, vb), d, out.data);
  for (std::size_t i = 0; i < n; ++i) { std::size_t base = (i / blk) * blk, o = i % blk;
    EXPECT_EQ(out.data[i], (o & 1) ? b.data[base + o / 2] : a.data[base + o / 2]) << "IntLo " << i; }
  hs::Store(hs::InterleaveUpper(va, vb), d, out.data);
  for (std::size_t i = 0; i < n; ++i) { std::size_t base = (i / blk) * blk, o = i % blk;
    EXPECT_EQ(out.data[i], (o & 1) ? b.data[base + blk / 2 + o / 2] : a.data[base + blk / 2 + o / 2]) << "IntUp " << i; }
  hs::Store(hs::ConcatLowerLower(va, vb), d, out.data);  // (hi=va, lo=vb) -> [vb.lo, va.lo]
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], i < n / 2 ? b.data[i] : a.data[i - n / 2]) << "CLL " << i;
  hs::Store(hs::ConcatUpperUpper(va, vb), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], i < n / 2 ? b.data[n / 2 + i] : a.data[i]) << "CUU " << i;
  hs::Store(hs::ConcatLowerUpper(va, vb), d, out.data);  // [vb.upper, va.lower]
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], i < n / 2 ? b.data[n / 2 + i] : a.data[i - n / 2]) << "CLU " << i;
  hs::Store(hs::ConcatUpperLower(va, vb), d, out.data);  // [vb.lower, va.upper]
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], i < n / 2 ? b.data[i] : a.data[i]) << "CUL " << i;
  hs::Store(hs::ConcatEven(va, vb), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], i < n / 2 ? b.data[2 * i] : a.data[2 * (i - n / 2)]) << "CEven " << i;
  hs::Store(hs::ConcatOdd(va, vb), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], i < n / 2 ? b.data[2 * i + 1] : a.data[2 * (i - n / 2) + 1]) << "COdd " << i;
  hs::Store(hs::DupEven(va), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], a.data[i & ~std::size_t{1}]) << "DupEven " << i;
  hs::Store(hs::DupOdd(va), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], a.data[(i & ~std::size_t{1}) | 1]) << "DupOdd " << i;

  // TableLookupLanes: reverse permute via an index vector.
  Buf<T> idx;
  for (std::size_t i = 0; i < n; ++i) idx.data[i] = static_cast<T>(n - 1 - i);
  auto ind = hs::IndicesFromVec(d, hs::Load(d, idx.data));
  hs::Store(hs::TableLookupLanes(va, ind), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], a.data[n - 1 - i]) << "TLL " << i;
}
TEST(Vec, SwizzleInt32) { CheckSwizzle<std::int32_t>(); }
TEST(Vec, SwizzleFloat) { CheckSwizzle<float>(); }
TEST(Vec, SwizzleInt64) { CheckSwizzle<std::int64_t>(); }
TEST(Vec, SwizzleDouble) { CheckSwizzle<double>(); }
TEST(Vec, SwizzleInt16) { CheckSwizzle<std::int16_t>(); }
TEST(Vec, SwizzleInt8) { CheckSwizzle<std::int8_t>(); }
TEST(Vec, SwizzleUint8) { CheckSwizzle<std::uint8_t>(); }

// Gather/Scatter (indexed memory) + Compress/CompressStore/Expand (mask pack/unpack).
// Gather index vector is a same-lane-count vector of the matching-width integer type.
template <typename T, typename IdxT>
void CheckGatherScatter() {
  const hs::ScalableTag<T> d;
  const hs::ScalableTag<IdxT> di;
  const std::size_t n = hs::Lanes(d);
  ASSERT_EQ(hs::Lanes(di), n);
  alignas(64) T table[64];
  for (std::size_t i = 0; i < 2 * n && i < 64; ++i) table[i] = static_cast<T>(i * 2 + 1);
  Buf<IdxT> idx;
  for (std::size_t i = 0; i < n; ++i) idx.data[i] = static_cast<IdxT>((n - 1 - i));
  Buf<T> out;
  hs::Store(hs::GatherIndex(d, table, hs::Load(di, idx.data)), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], table[idx.data[i]]) << "Gather " << i;

  alignas(64) T dst[64] = {};
  Buf<T> v; Buf<IdxT> sidx;
  for (std::size_t i = 0; i < n; ++i) { v.data[i] = static_cast<T>(i * 7 + 3); sidx.data[i] = static_cast<IdxT>(i); }
  hs::ScatterIndex(hs::Load(d, v.data), d, dst, hs::Load(di, sidx.data));
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(dst[i], v.data[i]) << "Scatter " << i;
}
TEST(Vec, GatherScatterI32) { CheckGatherScatter<std::int32_t, std::int32_t>(); }
TEST(Vec, GatherScatterFloat) { CheckGatherScatter<float, std::int32_t>(); }
TEST(Vec, GatherScatterI64) { CheckGatherScatter<std::int64_t, std::int64_t>(); }
TEST(Vec, GatherScatterDouble) { CheckGatherScatter<double, std::int64_t>(); }

template <typename T>
void CheckCompress() {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  Buf<T> v, lo, hi, out;
  for (std::size_t i = 0; i < n; ++i) { v.data[i] = static_cast<T>(i + 1); lo.data[i] = static_cast<T>(i % 2); hi.data[i] = 0; }
  auto m = hs::Gt(hs::Load(d, lo.data), hs::Load(d, hi.data));  // odd lanes true
  T ref[64]{}; std::size_t k = 0;  // AVX-512 i8 = 64 lanes
  for (std::size_t i = 0; i < n; ++i) if (i % 2 == 1) ref[k++] = v.data[i];
  // Compress CONTRACT: the first CountTrue(m) lanes hold the active elements in order; the tail is
  // UNSPECIFIED (matches Highway/xsimd -- avx2 has no zeroing-compress, so a zero-tail guarantee
  // would cost ~2x). Only assert the defined prefix; do NOT assert the tail.
  hs::Store(hs::Compress(hs::Load(d, v.data), m), d, out.data);
  for (std::size_t i = 0; i < k; ++i) EXPECT_EQ(out.data[i], ref[i]) << "Compress " << i;
  Buf<T> sp;
  for (std::size_t i = 0; i < n; ++i) sp.data[i] = static_cast<T>(0);
  std::size_t cnt = hs::CompressStore(hs::Load(d, v.data), m, d, sp.data);
  EXPECT_EQ(cnt, k);
  for (std::size_t i = 0; i < k; ++i) EXPECT_EQ(sp.data[i], ref[i]) << "CompressStore " << i;
  // Expand(compress-packed, m) restores the true lanes.
  hs::Store(hs::Expand(hs::LoadU(d, ref), m), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], (i % 2 == 1) ? v.data[i] : T{0}) << "Expand " << i;

  const auto check_mask = [&](std::uint64_t bits) {
    for (std::size_t i = 0; i < n; ++i) lo.data[i] = static_cast<T>((bits >> i) & 1u);
    m = hs::Gt(hs::Load(d, lo.data), hs::Load(d, hi.data));
    k = 0;
    for (std::size_t i = 0; i < n; ++i) if ((bits >> i) & 1u) ref[k++] = v.data[i];
    hs::Store(hs::Compress(hs::Load(d, v.data), m), d, out.data);
    for (std::size_t i = 0; i < k; ++i) EXPECT_EQ(out.data[i], ref[i]) << "Compress bits=" << bits << " i=" << i;
    hs::Store(hs::Expand(hs::LoadU(d, ref), m), d, out.data);
    std::size_t packed = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const T expected = ((bits >> i) & 1u) ? ref[packed++] : T{0};
      EXPECT_EQ(out.data[i], expected) << "Expand bits=" << bits << " i=" << i;
    }
  };
  if (n <= 8) {
    for (std::uint64_t bits = 0; bits < (std::uint64_t{1} << n); ++bits) check_mask(bits);
  } else {
    const std::uint64_t all = n == 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << n) - 1;
    const std::uint64_t masks[] = {
        0, all, 1, std::uint64_t{1} << (n - 1), 1 | (std::uint64_t{1} << (n - 1)),
        0x0180 & all, 0xA53C & all, 0xA5A5A5A5A5A5A5A5ull & all,
        all >> (n / 2), all ^ (all >> (n / 2))};
    for (const std::uint64_t bits : masks) check_mask(bits);
  }
}
TEST(Vec, CompressI32) { CheckCompress<std::int32_t>(); }
TEST(Vec, CompressFloat) { CheckCompress<float>(); }
TEST(Vec, CompressI64) { CheckCompress<std::int64_t>(); }
TEST(Vec, CompressI16) { CheckCompress<std::int16_t>(); }
TEST(Vec, CompressI8) { CheckCompress<std::int8_t>(); }

// Bit counts (popcount/lzcnt/tzcnt), Not, Clamp, Rotate, and sign helpers.
template <typename T>
void CheckBitCount() {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  using U = std::make_unsigned_t<T>;
  Buf<T> a, out;
  for (std::size_t i = 0; i < n; ++i) a.data[i] = static_cast<T>(i * 0x9E37 + 1);
  auto v = hs::Load(d, a.data);
  hs::Store(hs::PopulationCount(v), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], static_cast<T>(std::popcount(static_cast<U>(a.data[i])))) << "popcnt " << i;
  hs::Store(hs::LeadingZeroCount(v), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], static_cast<T>(std::countl_zero(static_cast<U>(a.data[i])))) << "lzcnt " << i;
  hs::Store(hs::TrailingZeroCount(v), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], static_cast<T>(std::countr_zero(static_cast<U>(a.data[i])))) << "tzcnt " << i;
}
TEST(Vec, BitCountUint32) { CheckBitCount<std::uint32_t>(); }
TEST(Vec, BitCountInt32) { CheckBitCount<std::int32_t>(); }
TEST(Vec, BitCountUint64) { CheckBitCount<std::uint64_t>(); }
TEST(Vec, BitCountUint16) { CheckBitCount<std::uint16_t>(); }
TEST(Vec, BitCountUint8) { CheckBitCount<std::uint8_t>(); }

// Not / Clamp / RotateLeft / RotateRight over integer lanes.
template <typename T>
void CheckIntMisc() {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  using U = std::make_unsigned_t<T>;
  const int kb = sizeof(T) * 8;
  Buf<T> a, out;
  for (std::size_t i = 0; i < n; ++i) a.data[i] = static_cast<T>((i % 2 ? -1 : 1) * static_cast<long long>(i * 13 + 1));
  auto v = hs::Load(d, a.data);
  hs::Store(hs::Not(v), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], static_cast<T>(~static_cast<U>(a.data[i]))) << "Not " << i;
  const T clo = std::is_signed_v<T> ? static_cast<T>(-5) : static_cast<T>(3), chi = static_cast<T>(20);
  hs::Store(hs::Clamp(v, hs::Set(d, clo), hs::Set(d, chi)), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], a.data[i] < clo ? clo : (a.data[i] > chi ? chi : a.data[i])) << "Clamp " << i;
  hs::Store(hs::template RotateRight<3>(v), d, out.data);
  for (std::size_t i = 0; i < n; ++i) { U u = static_cast<U>(a.data[i]); EXPECT_EQ(out.data[i], static_cast<T>((u >> 3) | (u << (kb - 3)))) << "RotR " << i; }
  hs::Store(hs::template RotateLeft<3>(v), d, out.data);
  for (std::size_t i = 0; i < n; ++i) { U u = static_cast<U>(a.data[i]); EXPECT_EQ(out.data[i], static_cast<T>((u << 3) | (u >> (kb - 3)))) << "RotL " << i; }
}
TEST(Vec, IntMiscInt32) { CheckIntMisc<std::int32_t>(); }
TEST(Vec, IntMiscUint32) { CheckIntMisc<std::uint32_t>(); }
TEST(Vec, IntMiscInt64) { CheckIntMisc<std::int64_t>(); }
TEST(Vec, IntMiscInt16) { CheckIntMisc<std::int16_t>(); }
TEST(Vec, IntMiscInt8) { CheckIntMisc<std::int8_t>(); }

// Sign helpers over signed integer lanes: BroadcastSignBit / IfNegativeThenElse / ZeroIfNegative.
template <typename T>
void CheckSignHelpers() {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  Buf<T> a, out;
  for (std::size_t i = 0; i < n; ++i) a.data[i] = static_cast<T>((i % 2 ? -1 : 1) * static_cast<long long>(i + 1));
  auto v = hs::Load(d, a.data);
  hs::Store(hs::BroadcastSignBit(v), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], a.data[i] < 0 ? static_cast<T>(~T{0}) : T{0}) << "BSB " << i;
  hs::Store(hs::IfNegativeThenElse(v, hs::Set(d, static_cast<T>(77)), hs::Set(d, static_cast<T>(9))), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], a.data[i] < 0 ? static_cast<T>(77) : static_cast<T>(9)) << "INTE " << i;
  hs::Store(hs::ZeroIfNegative(v), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], a.data[i] < 0 ? T{0} : a.data[i]) << "ZIN " << i;
}
TEST(Vec, SignHelpersInt32) { CheckSignHelpers<std::int32_t>(); }
TEST(Vec, SignHelpersInt16) { CheckSignHelpers<std::int16_t>(); }
TEST(Vec, SignHelpersInt8) { CheckSignHelpers<std::int8_t>(); }

// CopySign over float lanes.
template <typename T>
void CheckCopySign() {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  Buf<T> m, s, out;
  for (std::size_t i = 0; i < n; ++i) { m.data[i] = static_cast<T>(i + 1); s.data[i] = (i % 2 ? -1 : 1) * static_cast<T>(i); }
  hs::Store(hs::CopySign(hs::Load(d, m.data), hs::Load(d, s.data)), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], std::copysign(m.data[i], s.data[i])) << "CopySign " << i;
}
TEST(Vec, CopySignFloat) { CheckCopySign<float>(); }
TEST(Vec, CopySignDouble) { CheckCopySign<double>(); }


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

// ---- vector integer division by a run-time scalar (simd-09 线②#1) --------------------
// Oracle: the naive scalar `a / b`. Div() must be bit-identical on every backend. We sweep
// each divisor across a battery of dividends (edge values + a ramp) and compare per lane.
namespace {
template <typename T>
void CheckDivU32(std::uint32_t divisor) {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  const hs::Divisor<T> dv(static_cast<T>(divisor));
  Buf<T> in, out;
  // A spread of dividends: 0, 1, divisor-1, divisor, 2^32-1, and a scattered ramp.
  const T seeds[] = {0u, 1u, static_cast<T>(divisor - 1), divisor,
                     static_cast<T>(divisor + 1), 0xFFFFFFFFu, 0x80000000u, 123456789u};
  for (std::size_t base = 0; base < sizeof(seeds) / sizeof(seeds[0]); ++base) {
    for (std::size_t i = 0; i < n; ++i)
      in.data[i] = static_cast<T>(seeds[base] + static_cast<T>(i) * 0x01234567u);
    hs::Store(hs::Div(hs::Load(d, in.data), dv), d, out.data);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_EQ(out.data[i], static_cast<T>(in.data[i] / static_cast<T>(divisor)))
          << "u32 div=" << divisor << " a=" << in.data[i];
  }
}
template <typename T>
void CheckDivI32(std::int32_t divisor) {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  const hs::Divisor<T> dv(static_cast<T>(divisor));
  Buf<T> in, out;
  const T seeds[] = {0, 1, -1, divisor, static_cast<T>(-divisor),
                     std::numeric_limits<T>::max(), std::numeric_limits<T>::min(),
                     -123456789, 987654321};
  const T kMin = std::numeric_limits<T>::min();
  for (std::size_t base = 0; base < sizeof(seeds) / sizeof(seeds[0]); ++base) {
    for (std::size_t i = 0; i < n; ++i)  // build the ramp in unsigned to avoid signed-overflow UB
      in.data[i] = static_cast<T>(static_cast<std::uint32_t>(seeds[base]) +
                                  static_cast<std::uint32_t>(i) * (i % 2 ? 0xFF373737u : 0x74CBB1u));
    hs::Store(hs::Div(hs::Load(d, in.data), dv), d, out.data);
    for (std::size_t i = 0; i < n; ++i) {
      // INT_MIN / -1 is UB in C++ (traps on x86 idiv); the vector path yields -a wrapping to
      // INT_MIN. Define that as the contract and skip the trapping scalar oracle for that lane.
      const T expected = (divisor == -1 && in.data[i] == kMin)
                             ? kMin
                             : static_cast<T>(in.data[i] / static_cast<T>(divisor));
      EXPECT_EQ(out.data[i], expected) << "i32 div=" << divisor << " a=" << in.data[i];
    }
  }
}
}  // namespace

TEST(Vec, IntDivU32) {
  const std::uint32_t divisors[] = {1u,  2u,   3u,     4u,       7u,     8u,
                                    100u, 128u, 1000u,  65535u,   65536u, 999983u,
                                    2147483647u /*prime near 2^31*/, 4294967291u /*prime near 2^32*/};
  for (std::uint32_t dvz : divisors) CheckDivU32<std::uint32_t>(dvz);
}

TEST(Vec, IntDivI32) {
  const std::int32_t divisors[] = {1,     -1,   2,      -2,     3,     -3,   7,      -7,
                                   100,   -100, 128,    -128,   1000,  1024, -1024,  65535,
                                   -65536, 999983, -999983, 2147483647 /*INT_MAX prime*/};
  for (std::int32_t dvz : divisors) CheckDivI32<std::int32_t>(dvz);
}

namespace {
template <typename T>
void CheckMod(T divisor) {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  const hs::Divisor<T> dv(divisor);
  Buf<T> in, out;
  const auto modular_neg = [](T value) noexcept {
    if constexpr (std::is_signed_v<T>) return static_cast<T>(-value);
    else return T{0} - value;
  };
  const T seeds[] = {T{0}, T{1}, modular_neg(T{1}), divisor, modular_neg(divisor),
                     std::numeric_limits<T>::max(), std::numeric_limits<T>::min(),
                     modular_neg(static_cast<T>(1234567))};
  const auto ref = [divisor](T a) {
    if constexpr (std::is_signed_v<T>) {
      if (divisor == -1 && a == std::numeric_limits<T>::min()) return T{0};
    }
    return static_cast<T>(a % divisor);
  };
  for (std::size_t base = 0; base < sizeof(seeds) / sizeof(seeds[0]); ++base) {
    for (std::size_t i = 0; i < n; ++i) {
      in.data[i] = static_cast<T>(static_cast<std::make_unsigned_t<T>>(seeds[base]) +
                                  static_cast<std::make_unsigned_t<T>>(i) * 0x71u);
    }
    const auto v = hs::Load(d, in.data);
    hs::Store(hs::Mod(v, dv), d, out.data);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_EQ(out.data[i], ref(in.data[i])) << "mod d=" << divisor << " lane=" << i;

    const auto active = hs::Eq(hs::And(v, hs::Set(d, static_cast<T>(1))), hs::Zero(d));
    const T no = static_cast<T>(-77);
    hs::Store(hs::MaskedModOr(active, hs::Set(d, no), v, dv), d, out.data);
    for (std::size_t i = 0; i < n; ++i) {
      const bool is_active = (static_cast<std::make_unsigned_t<T>>(in.data[i]) & 1u) == 0;
      EXPECT_EQ(out.data[i], is_active ? ref(in.data[i]) : no) << "masked mod lane=" << i;
    }
    hs::Store(hs::MaskedModOr(hs::FirstN(d, 0), hs::Set(d, no), v, dv), d, out.data);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], no) << "all-off lane=" << i;
  }
}
}  // namespace

TEST(Vec, IntModU32) {
  const std::uint32_t divisors[] = {1u, 2u, 3u, 7u, 65535u, 999983u, 4294967291u};
  for (const auto divisor : divisors) CheckMod(divisor);
}

TEST(Vec, IntModI32) {
  const std::int32_t divisors[] = {1, -1, 2, -2, 3, -3, 7, -7, 65535, -65536, 999983, -999983};
  for (const auto divisor : divisors) CheckMod(divisor);
}

// 128-bit integer compare: each ADJACENT PAIR of u64 lanes = one 128-bit value (even=low,
// odd=high). Reference computes the same __uint128-style comparison per pair and checks the
// mask broadcasts the verdict to BOTH lanes. ScalableTag<u64> gives >=2 lanes on every backend
// (scalar/sse4/neon=2, avx2=4, avx512=8), so there is always at least one pair.
TEST(Vec, Compare128) {
  const hs::ScalableTag<std::uint64_t> d;
  const std::size_t n = hs::Lanes(d);
  ASSERT_EQ(n % 2u, 0u);
  Buf<std::uint64_t> ab, bb, out;
  // Craft pairs exercising: hi decides; hi equal + lo decides; fully equal.
  std::uint64_t seed = 0x9E3779B97F4A7C15ull;
  auto nextv = [&]() { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed; };
  for (std::size_t p = 0; p < n; p += 2) {
    switch ((p / 2) % 4) {
      case 0: ab.data[p]=5; ab.data[p+1]=10; bb.data[p]=999; bb.data[p+1]=11; break;  // a.hi<b.hi -> a<b
      case 1: ab.data[p]=7; ab.data[p+1]=20; bb.data[p]=3;   bb.data[p+1]=20; break;  // hi eq, a.lo>b.lo -> a>b
      case 2: ab.data[p]=42; ab.data[p+1]=99; bb.data[p]=42; bb.data[p+1]=99; break;  // equal
      default: ab.data[p]=nextv(); ab.data[p+1]=nextv(); bb.data[p]=nextv(); bb.data[p+1]=nextv(); break;
    }
  }
  auto va = hs::Load(d, ab.data), vb = hs::Load(d, bb.data);
  auto ref_lt = [&](std::size_t p) {
    return (ab.data[p+1] < bb.data[p+1]) || (ab.data[p+1] == bb.data[p+1] && ab.data[p] < bb.data[p]);
  };
  // Lt128
  hs::Store(hs::VecFromMask(d, hs::Lt128(d, va, vb)), d, out.data);
  for (std::size_t p = 0; p < n; p += 2) {
    const bool lt = ref_lt(p);
    EXPECT_EQ(out.data[p] != 0, lt) << "lt128 lo p=" << p;
    EXPECT_EQ(out.data[p+1] != 0, lt) << "lt128 hi p=" << p;  // verdict broadcast to both lanes
  }
  // Eq128
  hs::Store(hs::VecFromMask(d, hs::Eq128(d, va, vb)), d, out.data);
  for (std::size_t p = 0; p < n; p += 2) {
    const bool eq = ab.data[p]==bb.data[p] && ab.data[p+1]==bb.data[p+1];
    EXPECT_EQ(out.data[p] != 0, eq) << "eq128 p=" << p;
  }
  // Ne128 = !Eq128
  hs::Store(hs::VecFromMask(d, hs::Ne128(d, va, vb)), d, out.data);
  for (std::size_t p = 0; p < n; p += 2) {
    const bool ne = ab.data[p]!=bb.data[p] || ab.data[p+1]!=bb.data[p+1];
    EXPECT_EQ(out.data[p] != 0, ne) << "ne128 p=" << p;
  }
  // Lt128Upper: only high 64 decides
  hs::Store(hs::VecFromMask(d, hs::Lt128Upper(d, va, vb)), d, out.data);
  for (std::size_t p = 0; p < n; p += 2)
    EXPECT_EQ(out.data[p+1] != 0, ab.data[p+1] < bb.data[p+1]) << "lt128u p=" << p;
  // Min128 / Max128: whole 128-bit pair selected
  hs::Store(hs::Min128(d, va, vb), d, out.data);
  for (std::size_t p = 0; p < n; p += 2) {
    const bool a_lt = ref_lt(p);
    EXPECT_EQ(out.data[p],   a_lt ? ab.data[p]   : bb.data[p])   << "min128 lo p=" << p;
    EXPECT_EQ(out.data[p+1], a_lt ? ab.data[p+1] : bb.data[p+1]) << "min128 hi p=" << p;
  }
  hs::Store(hs::Max128(d, va, vb), d, out.data);
  for (std::size_t p = 0; p < n; p += 2) {
    const bool b_lt = (bb.data[p+1] < ab.data[p+1]) || (bb.data[p+1]==ab.data[p+1] && bb.data[p] < ab.data[p]);
    EXPECT_EQ(out.data[p],   b_lt ? ab.data[p]   : bb.data[p])   << "max128 lo p=" << p;
    EXPECT_EQ(out.data[p+1], b_lt ? ab.data[p+1] : bb.data[p+1]) << "max128 hi p=" << p;
  }
}

// ---- simd-09-rng: N parallel xoshiro256** streams --------------------------------------
// Lane 0's state is the initial scalar-seeded xoshiro (before any Jump), so lane 0's Next()
// sequence MUST match a scalar xoshiro256** seeded identically -- the correctness oracle.
namespace {
struct ScalarXoshiro {  // reference xoshiro256** (Blackman-Vigna), independent of the lib impl
  std::uint64_t s[4];
  static std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  explicit ScalarXoshiro(std::uint64_t seed) {
    std::uint64_t z = seed;  // SplitMix64 bootstrap
    for (auto& e : s) {
      z += 0x9e3779b97f4a7c15ull;
      std::uint64_t t = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
      t = (t ^ (t >> 27)) * 0x94d049bb133111ebull;
      e = t ^ (t >> 31);
    }
  }
  std::uint64_t next() {
    const std::uint64_t r = rotl(s[0] + s[3], 23) + s[0];
    const std::uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = rotl(s[3], 45);
    return r;
  }
};
}  // namespace
TEST(Vec, RandomXoshiro) {
  const hs::ScalableTag<std::uint64_t> d;
  const std::size_t n = hs::Lanes(d);
  // Lane 0 must match the scalar oracle for many draws.
  hs::VectorXoshiro rng{0xC0FFEEull};
  ScalarXoshiro oracle{0xC0FFEEull};
  Buf<std::uint64_t> out;
  for (int iter = 0; iter < 32; ++iter) {
    hs::Store(rng.Next(), d, out.data);
    EXPECT_EQ(out.data[0], oracle.next()) << "xoshiro lane0 mismatch iter=" << iter;
  }
  // Streams are decorrelated (Jump per lane): no two lanes produce the same value in a draw.
  if (n >= 2) {
    hs::VectorXoshiro rng2{0x1234ull};
    hs::Store(rng2.Next(), d, out.data);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = i + 1; j < n; ++j)
        EXPECT_NE(out.data[i], out.data[j]) << "lanes " << i << "," << j << " correlated";
  }
  // Reproducible: same seed -> identical sequence.
  hs::VectorXoshiro a{42}, b{42};
  Buf<std::uint64_t> oa, ob;
  for (int iter = 0; iter < 8; ++iter) {
    hs::Store(a.Next(), d, oa.data);
    hs::Store(b.Next(), d, ob.data);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(oa.data[i], ob.data[i]) << "repro i=" << i;
  }
  // Uniform() in [0,1).
  const hs::ScalableTag<double> df;
  const std::size_t nf = hs::Lanes(df);
  alignas(64) double u[hs::ScalableTag<double>::kMaxLanes];
  hs::VectorXoshiro urng{7};
  double sum = 0.0;
  int count = 0;
  for (int iter = 0; iter < 200; ++iter) {
    hs::Store(urng.Uniform(), df, u);
    for (std::size_t i = 0; i < nf; ++i) {
      EXPECT_GE(u[i], 0.0) << "uniform<0";
      EXPECT_LT(u[i], 1.0) << "uniform>=1";
      sum += u[i]; ++count;
    }
  }
  EXPECT_NEAR(sum / count, 0.5, 0.05) << "uniform mean off (n=" << count << ")";  // ~0.5 expected
}

// ---- simd-17 batch1: fixed-pattern Shuffle + byte/lane shifts ----------------------
// Oracle is the scalar backend's per-128-bit-block contract. Lane values = global index
// so multi-block backends (avx2=2, avx512=4) must keep blocks independent.
TEST(Vec, ShuffleFixedPatterns32) {
  const hs::ScalableTag<std::int32_t> d;
  const std::size_t n = hs::Lanes(d);
  ASSERT_GE(n, 4u);
  Buf<std::int32_t> in, out;
  for (std::size_t i = 0; i < n; ++i) in.data[i] = static_cast<std::int32_t>(i + 10);
  const auto v = hs::Load(d, in.data);

  auto check_map = [&](auto op, auto map4, const char* name) {
    hs::Store(op(v), d, out.data);
    for (std::size_t base = 0; base + 3 < n; base += 4) {
      for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(out.data[base + i], in.data[base + map4(i)])
            << name << " base=" << base << " i=" << i;
      }
    }
  };
  check_map([](auto x) { return hs::Shuffle2301(x); },
            [](std::size_t i) { return i ^ 1; }, "Shuffle2301");
  check_map([](auto x) { return hs::Shuffle1032(x); },
            [](std::size_t i) { return i ^ 2; }, "Shuffle1032");
  check_map([](auto x) { return hs::Shuffle0321(x); },
            [](std::size_t i) { return (i + 1) & 3; }, "Shuffle0321");
  check_map([](auto x) { return hs::Shuffle2103(x); },
            [](std::size_t i) { return (i + 3) & 3; }, "Shuffle2103");
  check_map([](auto x) { return hs::Shuffle0123(x); },
            [](std::size_t i) { return i ^ 3; }, "Shuffle0123");
}

TEST(Vec, Shuffle01_64) {
  const hs::ScalableTag<std::int64_t> d;
  const std::size_t n = hs::Lanes(d);
  ASSERT_GE(n, 2u);
  Buf<std::int64_t> in, out;
  for (std::size_t i = 0; i < n; ++i) in.data[i] = static_cast<std::int64_t>(100 + i);
  hs::Store(hs::Shuffle01(hs::Load(d, in.data)), d, out.data);
  for (std::size_t base = 0; base + 1 < n; base += 2) {
    EXPECT_EQ(out.data[base + 0], in.data[base + 1]) << "Shuffle01 base=" << base;
    EXPECT_EQ(out.data[base + 1], in.data[base + 0]) << "Shuffle01 base=" << base;
  }
}

TEST(Vec, ShiftBytesAndLanes) {
  // Byte shifts on i8 so each byte is independently visible.
  {
    const hs::ScalableTag<std::uint8_t> d;
    const std::size_t n = hs::Lanes(d);
    Buf<std::uint8_t> in, out;
    for (std::size_t i = 0; i < n; ++i) in.data[i] = static_cast<std::uint8_t>(i + 1);
    const auto v = hs::Load(d, in.data);

    hs::Store(hs::ShiftLeftBytes<3>(v), d, out.data);
    for (std::size_t base = 0; base < n; base += 16) {
      for (std::size_t i = 0; i < 16 && base + i < n; ++i) {
        const std::uint8_t exp = (i >= 3) ? in.data[base + i - 3] : std::uint8_t{0};
        EXPECT_EQ(out.data[base + i], exp) << "ShiftLeftBytes<3> base=" << base << " i=" << i;
      }
    }
    hs::Store(hs::ShiftRightBytes<5>(v), d, out.data);
    for (std::size_t base = 0; base < n; base += 16) {
      for (std::size_t i = 0; i < 16 && base + i < n; ++i) {
        const std::uint8_t exp = (i + 5 < 16) ? in.data[base + i + 5] : std::uint8_t{0};
        EXPECT_EQ(out.data[base + i], exp) << "ShiftRightBytes<5> base=" << base << " i=" << i;
      }
    }
    // k=0 identity
    hs::Store(hs::ShiftLeftBytes<0>(v), d, out.data);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], in.data[i]);
  }
  // Lane shifts on i32
  {
    const hs::ScalableTag<std::int32_t> d;
    const std::size_t n = hs::Lanes(d);
    Buf<std::int32_t> in, out;
    for (std::size_t i = 0; i < n; ++i) in.data[i] = static_cast<std::int32_t>(i + 7);
    hs::Store(hs::ShiftLeftLanes<1>(hs::Load(d, in.data)), d, out.data);
    for (std::size_t base = 0; base + 3 < n; base += 4) {
      EXPECT_EQ(out.data[base + 0], 0);
      EXPECT_EQ(out.data[base + 1], in.data[base + 0]);
      EXPECT_EQ(out.data[base + 2], in.data[base + 1]);
      EXPECT_EQ(out.data[base + 3], in.data[base + 2]);
    }
    hs::Store(hs::ShiftRightLanes<1>(hs::Load(d, in.data)), d, out.data);
    for (std::size_t base = 0; base + 3 < n; base += 4) {
      EXPECT_EQ(out.data[base + 0], in.data[base + 1]);
      EXPECT_EQ(out.data[base + 1], in.data[base + 2]);
      EXPECT_EQ(out.data[base + 2], in.data[base + 3]);
      EXPECT_EQ(out.data[base + 3], 0);
    }
  }
}

TEST(Vec, CombineShiftRightBytes) {
  const hs::ScalableTag<std::uint8_t> d;
  const std::size_t n = hs::Lanes(d);
  Buf<std::uint8_t> lo, hi, out;
  for (std::size_t i = 0; i < n; ++i) {
    lo.data[i] = static_cast<std::uint8_t>(0x10 + i);
    hi.data[i] = static_cast<std::uint8_t>(0x80 + i);
  }
  hs::Store(hs::CombineShiftRightBytes<4>(hs::Load(d, hi.data), hs::Load(d, lo.data)), d, out.data);
  for (std::size_t base = 0; base < n; base += 16) {
    for (std::size_t i = 0; i < 16 && base + i < n; ++i) {
      const std::size_t src = 4 + i;
      const std::uint8_t exp = (src < 16) ? lo.data[base + src] : hi.data[base + src - 16];
      EXPECT_EQ(out.data[base + i], exp) << "CombineShiftRightBytes base=" << base << " i=" << i;
    }
  }
  // Lane form: CombineShiftRightLanes<1> on i32 == CombineShiftRightBytes<4>
  {
    const hs::ScalableTag<std::int32_t> di;
    const std::size_t ni = hs::Lanes(di);
    Buf<std::int32_t> a, b, oa, ob;
    for (std::size_t i = 0; i < ni; ++i) {
      a.data[i] = static_cast<std::int32_t>(i + 1);
      b.data[i] = static_cast<std::int32_t>(100 + i);
    }
    const auto va = hs::Load(di, a.data);
    const auto vb = hs::Load(di, b.data);
    hs::Store(hs::CombineShiftRightLanes<1>(vb, va), di, oa.data);
    hs::Store(hs::CombineShiftRightBytes<4>(vb, va), di, ob.data);
    for (std::size_t i = 0; i < ni; ++i) EXPECT_EQ(oa.data[i], ob.data[i]);
  }
}

// ---- simd-17 batch2: *Same shifts, Pairwise, Stream, LoadDup128, exp domain ----------
TEST(Vec, ShiftSameAndRotateSame) {
  const hs::ScalableTag<std::int32_t> d;
  const std::size_t n = hs::Lanes(d);
  Buf<std::int32_t> in, out;
  for (std::size_t i = 0; i < n; ++i) in.data[i] = static_cast<std::int32_t>((i + 1) * 3);
  const auto v = hs::Load(d, in.data);
  hs::Store(hs::ShiftLeftSame(v, 2), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], in.data[i] << 2);
  hs::Store(hs::ShiftRightSame(v, 1), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], in.data[i] >> 1);
  hs::Store(hs::RotateLeftSame(v, 4), d, out.data);
  for (std::size_t i = 0; i < n; ++i) {
    const auto u = static_cast<std::uint32_t>(in.data[i]);
    EXPECT_EQ(static_cast<std::uint32_t>(out.data[i]), (u << 4) | (u >> 28));
  }
  hs::Store(hs::RoundingShiftRightSame(v, 2), d, out.data);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(out.data[i], (in.data[i] + 2) >> 2);
}

TEST(Vec, PairwiseAddSub) {
  const hs::ScalableTag<std::int32_t> d;
  const std::size_t n = hs::Lanes(d);
  ASSERT_GE(n, 4u);
  Buf<std::int32_t> a, b, out;
  for (std::size_t i = 0; i < n; ++i) {
    a.data[i] = static_cast<std::int32_t>(i + 1);
    b.data[i] = static_cast<std::int32_t>(10 + i);
  }
  hs::Store(hs::PairwiseAdd(hs::Load(d, a.data), hs::Load(d, b.data)), d, out.data);
  for (std::size_t i = 0; i < n / 2; ++i) {
    EXPECT_EQ(out.data[i], a.data[2 * i] + a.data[2 * i + 1]) << "PAdd a i=" << i;
    EXPECT_EQ(out.data[n / 2 + i], b.data[2 * i] + b.data[2 * i + 1]) << "PAdd b i=" << i;
  }
  hs::Store(hs::PairwiseAdd128(hs::Load(d, a.data), hs::Load(d, b.data)), d, out.data);
  for (std::size_t base = 0; base + 3 < n; base += 4) {
    EXPECT_EQ(out.data[base + 0], a.data[base + 0] + a.data[base + 1]);
    EXPECT_EQ(out.data[base + 1], a.data[base + 2] + a.data[base + 3]);
    EXPECT_EQ(out.data[base + 2], b.data[base + 0] + b.data[base + 1]);
    EXPECT_EQ(out.data[base + 3], b.data[base + 2] + b.data[base + 3]);
  }
}

TEST(Vec, StreamAndLoadDup128) {
  const hs::ScalableTag<std::int32_t> d;
  const std::size_t n = hs::Lanes(d);
  Buf<std::int32_t> in, out;
  for (std::size_t i = 0; i < n; ++i) in.data[i] = static_cast<std::int32_t>(i + 5);
  hs::Stream(hs::Load(d, in.data), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], in.data[i]);

  Buf<std::int32_t> src128{};
  for (std::size_t i = 0; i < 4 && i < n; ++i) src128.data[i] = static_cast<std::int32_t>(100 + i);
  hs::Store(hs::LoadDup128(d, src128.data), d, out.data);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(out.data[i], src128.data[i % 4]) << "LoadDup128 i=" << i;
}

TEST(Vec, GetBiasedExponentAndMulByFloorPow2) {
  const hs::ScalableTag<float> d;
  const std::size_t n = hs::Lanes(d);
  Buf<float> in, exp, outf;
  Buf<std::uint32_t> outu;
  for (std::size_t i = 0; i < n; ++i) {
    in.data[i] = (i % 3 == 0) ? 1.0f : (i % 3 == 1) ? 2.0f : 0.5f;
    exp.data[i] = 2.5f;
  }
  hs::Store(hs::GetBiasedExponent(hs::Load(d, in.data)),
            hs::RebindToUnsigned<hs::ScalableTag<float>>{}, outu.data);
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint32_t expect = (in.data[i] == 1.0f) ? 127u : (in.data[i] == 2.0f) ? 128u : 126u;
    EXPECT_EQ(outu.data[i], expect) << "biased i=" << i;
  }
  hs::Store(hs::MulByFloorPow2(hs::Load(d, in.data), hs::Load(d, exp.data)), d, outf.data);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_FLOAT_EQ(outf.data[i], in.data[i] * 4.0f) << "MulByFloorPow2 i=" << i;

  // Negative operand: GetBiasedExponent uses |v|, so -2.0 still yields biased 128 (no sign leak).
  Buf<float> neg;
  for (std::size_t i = 0; i < n; ++i) neg.data[i] = -2.0f;
  hs::Store(hs::GetBiasedExponent(hs::Load(d, neg.data)),
            hs::RebindToUnsigned<hs::ScalableTag<float>>{}, outu.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(outu.data[i], 128u) << "biased neg i=" << i;

  // Highway special cases (the red-team gap the earlier exp=2.5-only test missed):
  //   MulByFloorPow2(v, NaN)=NaN, (0,+inf)=NaN, (inf,-inf)=NaN. Non-special inf paths pass through.
  const float kNaN = std::numeric_limits<float>::quiet_NaN();
  const float kInf = std::numeric_limits<float>::infinity();
  Buf<float> sv, sexp, so;
  for (std::size_t i = 0; i < n; ++i) {
    const int c = static_cast<int>(i % 5);
    if (c == 0) { sv.data[i] = 3.0f; sexp.data[i] = kNaN; }        // NaN exp -> NaN
    else if (c == 1) { sv.data[i] = 0.0f; sexp.data[i] = kInf; }   // (0,+inf) -> NaN
    else if (c == 2) { sv.data[i] = kInf; sexp.data[i] = -kInf; }  // (inf,-inf) -> NaN
    else if (c == 3) { sv.data[i] = 4.0f; sexp.data[i] = kInf; }   // (finite,+inf) -> +inf
    else { sv.data[i] = 3.0f; sexp.data[i] = 2.0f; }               // normal -> 12
  }
  hs::Store(hs::MulByFloorPow2(hs::Load(d, sv.data), hs::Load(d, sexp.data)), d, so.data);
  for (std::size_t i = 0; i < n; ++i) {
    const int c = static_cast<int>(i % 5);
    if (c <= 2) EXPECT_TRUE(std::isnan(so.data[i])) << "special NaN i=" << i;
    else if (c == 3) EXPECT_TRUE(std::isinf(so.data[i]) && so.data[i] > 0) << "special +inf i=" << i;
    else EXPECT_FLOAT_EQ(so.data[i], 12.0f) << "special normal i=" << i;
  }
  // Red-team wide finite exponents (overflow->inf, denormal, underflow->0) with fractional floor.
  Buf<float> wv, we, wo;
  const float wvals[] = {1.5f, 3.0f, 1e20f, 1e-20f, 2.0f, -4.0f, 0.5f, 1.0f,
                         7.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  const float wexps[] = {130.9f, 200.1f, 50.5f, -200.7f, -150.2f, 10.5f, -149.9f, 127.0f,
                         -10.3f, 3.5f, 0.0f, -1.0f, 2.0f, -2.0f, 5.5f, -5.5f};
  for (std::size_t i = 0; i < n; ++i) { wv.data[i] = wvals[i % 16]; we.data[i] = wexps[i % 16]; }
  hs::Store(hs::MulByFloorPow2(hs::Load(d, wv.data), hs::Load(d, we.data)), d, wo.data);
  for (std::size_t i = 0; i < n; ++i) {
    const float ref = std::ldexp(wv.data[i], static_cast<int>(std::floor(we.data[i])));
    EXPECT_EQ(wo.data[i], ref) << "mulbyfloorpow2 wide i=" << i << " v=" << wv.data[i] << " e=" << we.data[i];
  }
}

// ---- simd-17 batch4: half-vector family ---------------------------------------------
// Round-trip iron test: Combine(UpperHalf, LowerHalf) == original.
// Also covers ZeroExtendVector (high half zero), InsertIntoUpper (only high changes),
// ResizeBitCast (widen half->full zeros high; narrow full->half keeps low).
// Note: Store/Load signatures take Simd<T,N,0> (kPow2==0); Half tags have kPow2=-1, so
// store half results via DFromV (which rebinds to kPow2=0).
TEST(Vec, HalfVectorFamily) {
  const hs::ScalableTag<std::int32_t> d;
  using DH = typename hs::ScalableTag<std::int32_t>::Half;
  const DH dh{};
  const std::size_t n = hs::Lanes(d);
  const std::size_t nh = n / 2;
  ASSERT_GE(n, 2u);

  Buf<std::int32_t> in, out, lo_out, hi_out, zext_out, ins_out;
  for (std::size_t i = 0; i < n; ++i) in.data[i] = static_cast<std::int32_t>(100 + i);
  const auto v = hs::Load(d, in.data);

  const auto lo = hs::LowerHalf(dh, v);
  const auto hi = hs::UpperHalf(dh, v);
  using DH0 = hs::DFromV<decltype(lo)>;
  const DH0 dh0{};

  // LowerHalf / UpperHalf lane contents.
  hs::Store(lo, dh0, lo_out.data);
  hs::Store(hi, dh0, hi_out.data);
  for (std::size_t i = 0; i < nh; ++i) {
    EXPECT_EQ(lo_out.data[i], in.data[i]) << "LowerHalf i=" << i;
    EXPECT_EQ(hi_out.data[i], in.data[nh + i]) << "UpperHalf i=" << i;
  }

  // Convenience overloads (no tag) match tagged forms.
  hs::Store(hs::LowerHalf(v), dh0, lo_out.data);
  hs::Store(hs::UpperHalf(v), dh0, hi_out.data);
  for (std::size_t i = 0; i < nh; ++i) {
    EXPECT_EQ(lo_out.data[i], in.data[i]);
    EXPECT_EQ(hi_out.data[i], in.data[nh + i]);
  }

  // Round-trip: Combine(d, UpperHalf, LowerHalf) == v  (Highway order: hi, lo).
  hs::Store(hs::Combine(d, hi, lo), d, out.data);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(out.data[i], in.data[i]) << "Combine round-trip i=" << i;

  // ZeroExtendVector: low = lo, high = 0.
  hs::Store(hs::ZeroExtendVector(d, lo), d, zext_out.data);
  for (std::size_t i = 0; i < nh; ++i)
    EXPECT_EQ(zext_out.data[i], in.data[i]) << "ZeroExtend low i=" << i;
  for (std::size_t i = nh; i < n; ++i)
    EXPECT_EQ(zext_out.data[i], 0) << "ZeroExtend high i=" << i;

  // InsertIntoUpper: low stays, high replaced by hi.
  Buf<std::int32_t> base;
  for (std::size_t i = 0; i < n; ++i) base.data[i] = static_cast<std::int32_t>(i);  // 0..n-1
  const auto bv = hs::Load(d, base.data);
  // hi = LowerHalf of original (100, 101, ...)
  hs::Store(hs::InsertIntoUpper(d, lo, bv), d, ins_out.data);
  for (std::size_t i = 0; i < nh; ++i)
    EXPECT_EQ(ins_out.data[i], base.data[i]) << "InsertIntoUpper low i=" << i;
  for (std::size_t i = 0; i < nh; ++i)
    EXPECT_EQ(ins_out.data[nh + i], in.data[i]) << "InsertIntoUpper high i=" << i;

  // ResizeBitCast: half -> full (widen, high zeroed); full -> half (narrow, keep low).
  hs::Store(hs::ResizeBitCast(d, lo), d, out.data);
  for (std::size_t i = 0; i < nh; ++i)
    EXPECT_EQ(out.data[i], in.data[i]) << "ResizeBitCast widen low i=" << i;
  for (std::size_t i = nh; i < n; ++i)
    EXPECT_EQ(out.data[i], 0) << "ResizeBitCast widen high i=" << i;

  // Narrow full -> half: use dh0 (kPow2=0) as destination tag so Load/Store match.
  hs::Store(hs::ResizeBitCast(dh0, v), dh0, lo_out.data);
  for (std::size_t i = 0; i < nh; ++i)
    EXPECT_EQ(lo_out.data[i], in.data[i]) << "ResizeBitCast narrow i=" << i;
}

// Float lane variant of half-vector family (catches f32 cast/extract paths on x86).
TEST(Vec, HalfVectorFamilyF32) {
  const hs::ScalableTag<float> d;
  using DH = typename hs::ScalableTag<float>::Half;
  const DH dh{};
  const std::size_t n = hs::Lanes(d);
  const std::size_t nh = n / 2;
  ASSERT_GE(n, 2u);

  Buf<float> in, out;
  for (std::size_t i = 0; i < n; ++i) in.data[i] = static_cast<float>(i) + 0.5f;
  const auto v = hs::Load(d, in.data);
  hs::Store(hs::Combine(d, hs::UpperHalf(dh, v), hs::LowerHalf(dh, v)), d, out.data);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_FLOAT_EQ(out.data[i], in.data[i]) << "f32 Combine round-trip i=" << i;

  // Edge: all-zero vector round-trip + ZeroExtend of zero stays zero.
  const auto z = hs::Zero(d);
  hs::Store(hs::Combine(d, hs::UpperHalf(dh, z), hs::LowerHalf(dh, z)), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_FLOAT_EQ(out.data[i], 0.0f);
  hs::Store(hs::ZeroExtendVector(d, hs::LowerHalf(dh, z)), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_FLOAT_EQ(out.data[i], 0.0f);
  (void)nh;
}

// ---- simd-17 batch5: SumsOf8AbsDiff (SAD) + 128-bit block ops (avx2/512) ------------
TEST(Vec, SumsOf8AbsDiff) {
  const hs::ScalableTag<std::uint8_t> d;
  const std::size_t n = hs::Lanes(d);
  ASSERT_GE(n, 8u);
  ASSERT_EQ(n % 8, 0u);
  Buf<std::uint8_t> a, b;
  for (std::size_t i = 0; i < n; ++i) {
    a.data[i] = static_cast<std::uint8_t>(i * 3 + 1);
    b.data[i] = static_cast<std::uint8_t>(i * 2);
  }
  const auto sa = hs::SumsOf8AbsDiff(hs::Load(d, a.data), hs::Load(d, b.data));
  using D64 = hs::DFromV<decltype(sa)>;
  const D64 d64{};
  const std::size_t ng = hs::Lanes(d64);
  Buf<std::uint64_t> out;
  hs::Store(sa, d64, out.data);
  for (std::size_t g = 0; g < ng; ++g) {
    std::uint64_t expect = 0;
    for (std::size_t j = 0; j < 8; ++j) {
      const int aa = a.data[g * 8 + j];
      const int bb = b.data[g * 8 + j];
      expect += static_cast<std::uint64_t>(aa > bb ? aa - bb : bb - aa);
    }
    EXPECT_EQ(out.data[g], expect) << "SAD group " << g;
  }
  // Unary vs zero equals SumsOf8.
  hs::Store(hs::SumsOf8AbsDiff(hs::Load(d, a.data)), d64, out.data);
  Buf<std::uint64_t> out2;
  hs::Store(hs::SumsOf8(hs::Load(d, a.data)), d64, out2.data);
  for (std::size_t g = 0; g < ng; ++g) EXPECT_EQ(out.data[g], out2.data[g]);
}

#if defined(HYLE_SIMD_TARGET_AVX2) || defined(HYLE_SIMD_TARGET_AVX512)
TEST(Vec, BroadcastExtractBlock) {
  // 128-bit blocks inside 256 (2 blocks) or 512 (4 blocks).
  const hs::ScalableTag<std::int32_t> d;
  const std::size_t n = hs::Lanes(d);
  ASSERT_GE(n, 8u);  // at least 2 x 128-bit blocks of i32
  Buf<std::int32_t> in, out, blk;
  for (std::size_t i = 0; i < n; ++i) in.data[i] = static_cast<std::int32_t>(1000 + i);
  const auto v = hs::Load(d, in.data);

  // ExtractBlock<0> = low 128 bits = first 4 i32 lanes.
  {
    const auto b0 = hs::ExtractBlock<0>(v);
    using DB = hs::DFromV<decltype(b0)>;
    hs::Store(b0, DB{}, blk.data);
    for (std::size_t i = 0; i < 4; ++i) EXPECT_EQ(blk.data[i], in.data[i]) << "Extract0 i=" << i;
  }
  // ExtractBlock<1> = second 128 bits.
  {
    const auto b1 = hs::ExtractBlock<1>(v);
    using DB = hs::DFromV<decltype(b1)>;
    hs::Store(b1, DB{}, blk.data);
    for (std::size_t i = 0; i < 4; ++i) EXPECT_EQ(blk.data[i], in.data[4 + i]) << "Extract1 i=" << i;
  }

  // BroadcastBlock<0>: every 128-bit block equals the low 128 of input.
  hs::Store(hs::BroadcastBlock<0>(v), d, out.data);
  for (std::size_t base = 0; base < n; base += 4) {
    for (std::size_t i = 0; i < 4; ++i)
      EXPECT_EQ(out.data[base + i], in.data[i]) << "Bcast0 base=" << base << " i=" << i;
  }
  // BroadcastBlock<1>: every block equals the second 128 of input.
  hs::Store(hs::BroadcastBlock<1>(v), d, out.data);
  for (std::size_t base = 0; base < n; base += 4) {
    for (std::size_t i = 0; i < 4; ++i)
      EXPECT_EQ(out.data[base + i], in.data[4 + i]) << "Bcast1 base=" << base << " i=" << i;
  }
}
#endif  // AVX2 || AVX512


// ---- simd-17 batch5: BitShuffle + MultiRotateRight (u64 data, u8 index, 8 per lane) --------
// Portable on all five fixed backends; AVX3_DL takes the native VPSHUFBITQMB / VPMULTISHIFTQB
// path (SDE), the AVX512 baseline the store+reload fallback -- both must match this oracle.
TEST(Vec, BitShuffleAndMultiRotate) {
  const hs::ScalableTag<std::uint64_t> d;
  const std::size_t nq = hs::Lanes(d);          // u64 data lanes
  using DI = hs::ScalableTag<std::uint8_t>;
  const DI di;
  const std::size_t ni = hs::Lanes(di);         // u8 index lanes
  ASSERT_EQ(ni, nq * 8);                          // exactly 8 index bytes per data lane

  Buf<std::uint64_t> vin, out;
  Buf<std::uint8_t> iin;
  for (std::size_t q = 0; q < nq; ++q)
    vin.data[q] = 0x0123456789ABCDEFull ^ (0x1111111111111111ull * q);

  // --- BitShuffle: idx[q*8+j] selects which bit of v[q] lands in output bit j. ---
  for (std::size_t q = 0; q < nq; ++q)
    for (std::size_t j = 0; j < 8; ++j)
      iin.data[q * 8 + j] = static_cast<std::uint8_t>((j * 7 + q * 3) & 63);  // spread bit picks
  hs::Store(hs::BitShuffle(hs::Load(d, vin.data), hs::Load(di, iin.data)), d, out.data);
  for (std::size_t q = 0; q < nq; ++q) {
    std::uint64_t expect = 0;
    for (std::size_t j = 0; j < 8; ++j) {
      const unsigned bit = iin.data[q * 8 + j] & 63u;
      expect |= ((vin.data[q] >> bit) & 1ull) << j;
    }
    EXPECT_EQ(out.data[q], expect) << "BitShuffle lane " << q;
  }
  // Edge: idx all >=64 wrap to bit (idx&63); idx==0 for all -> output byte = 8 copies of bit0.
  for (std::size_t i = 0; i < ni; ++i) iin.data[i] = 0;
  hs::Store(hs::BitShuffle(hs::Load(d, vin.data), hs::Load(di, iin.data)), d, out.data);
  for (std::size_t q = 0; q < nq; ++q) {
    const std::uint64_t b0 = vin.data[q] & 1ull;
    EXPECT_EQ(out.data[q], b0 ? 0xFFull : 0x00ull) << "BitShuffle bit0 lane " << q;
  }

  // --- MultiRotateRight: output byte j = low 8 bits of rotr64(v[q], idx[q*8+j]&63). ---
  for (std::size_t q = 0; q < nq; ++q)
    for (std::size_t j = 0; j < 8; ++j)
      iin.data[q * 8 + j] = static_cast<std::uint8_t>((j * 8) & 63);  // 0,8,16,...,56 -> each byte
  hs::Store(hs::MultiRotateRight(hs::Load(d, vin.data), hs::Load(di, iin.data)), d, out.data);
  for (std::size_t q = 0; q < nq; ++q) {
    std::uint64_t expect = 0;
    for (std::size_t j = 0; j < 8; ++j) {
      const unsigned rot = (iin.data[q * 8 + j]) & 63u;
      const std::uint64_t rr = rot == 0 ? vin.data[q] : ((vin.data[q] >> rot) | (vin.data[q] << (64 - rot)));
      expect |= (rr & 0xFFull) << (j * 8);
    }
    EXPECT_EQ(out.data[q], expect) << "MultiRotate lane " << q;
  }
  // With rot j*8, output byte j = byte j of v[q] -> identity (nice cross-check).
  for (std::size_t q = 0; q < nq; ++q)
    EXPECT_EQ(out.data[q], vin.data[q]) << "MultiRotate identity lane " << q;
  // Edge: all rot 0 -> every output byte = low byte of v[q] (splat).
  for (std::size_t i = 0; i < ni; ++i) iin.data[i] = 0;
  hs::Store(hs::MultiRotateRight(hs::Load(d, vin.data), hs::Load(di, iin.data)), d, out.data);
  for (std::size_t q = 0; q < nq; ++q) {
    const std::uint64_t lo = vin.data[q] & 0xFFull;
    EXPECT_EQ(out.data[q], lo * 0x0101010101010101ull) << "MultiRotate splat lane " << q;
  }
}

// ---- simd-18 批A: generic mask family (RebindMask/MaskFalse/SetMask/Combine/Half/Promote/
// Demote/SlideMask*) -- masks round-trip through vector domain. Oracle: build masks via compare,
// apply the vector analogue, compare mask->vector patterns lane by lane. ------------------------
TEST(Vec, GenericMaskFamily) {
  const hs::ScalableTag<std::int32_t> d;
  const std::size_t n = hs::Lanes(d);
  ASSERT_GE(n, 2u);
  Buf<std::int32_t> a, b, mv_out, ref_out;
  for (std::size_t i = 0; i < n; ++i) { a.data[i] = int(i); b.data[i] = int(i) % 2; }  // mask = even lanes? b: lane%2
  const auto va = hs::Load(d, a.data);
  // mask m = (a > threshold) -> mix of true/false
  const auto m = hs::Gt(va, hs::Set(d, int(n / 2)));

  // MaskFalse: all lanes false -> VecFromMask all zero.
  hs::Store(hs::VecFromMask(d, hs::MaskFalse(d)), d, mv_out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(mv_out.data[i], 0) << "MaskFalse i=" << i;

  // SetMask(true/false): all-ones / all-zero.
  hs::Store(hs::VecFromMask(d, hs::SetMask(d, true)), d, mv_out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(mv_out.data[i], -1) << "SetMask true i=" << i;
  hs::Store(hs::VecFromMask(d, hs::SetMask(d, false)), d, mv_out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(mv_out.data[i], 0) << "SetMask false i=" << i;

  // RebindMask i32<->f32 (same width): pattern preserved.
  const hs::ScalableTag<float> df;
  const auto mf = hs::RebindMask(df, m);
  Buf<float> mf_out;
  hs::Store(hs::VecFromMask(df, mf), df, mf_out.data);
  hs::Store(hs::VecFromMask(d, m), d, ref_out.data);
  for (std::size_t i = 0; i < n; ++i) {
    std::int32_t bits; std::memcpy(&bits, &mf_out.data[i], 4);
    EXPECT_EQ(bits, ref_out.data[i]) << "RebindMask i=" << i;
  }

  // SlideMask1Up/Down: mask analogue of vector slide.
  hs::Store(hs::VecFromMask(d, hs::SlideMask1Up(d, m)), d, mv_out.data);
  hs::Store(hs::Slide1Up(d, hs::VecFromMask(d, m)), d, ref_out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(mv_out.data[i], ref_out.data[i]) << "SlideMask1Up i=" << i;
  hs::Store(hs::VecFromMask(d, hs::SlideMask1Down(d, m)), d, mv_out.data);
  hs::Store(hs::Slide1Down(d, hs::VecFromMask(d, m)), d, ref_out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(mv_out.data[i], ref_out.data[i]) << "SlideMask1Down i=" << i;

  // SlideMaskUp/DownLanes by 1.
  hs::Store(hs::VecFromMask(d, hs::SlideMaskUpLanes(d, m, 1)), d, mv_out.data);
  hs::Store(hs::SlideUpLanes(d, hs::VecFromMask(d, m), 1), d, ref_out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(mv_out.data[i], ref_out.data[i]) << "SlideMaskUpLanes i=" << i;
  hs::Store(hs::VecFromMask(d, hs::SlideMaskDownLanes(d, m, 1)), d, mv_out.data);
  hs::Store(hs::SlideDownLanes(d, hs::VecFromMask(d, m), 1), d, ref_out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(mv_out.data[i], ref_out.data[i]) << "SlideMaskDownLanes i=" << i;
}

// Half-of-mask + Combine (needs >=2 lanes; skip on scalar 1-lane wide types via ASSERT).
TEST(Vec, MaskHalfAndCombine) {
  const hs::ScalableTag<std::int32_t> d;
  using DH = typename hs::ScalableTag<std::int32_t>::Half;
  const DH dh{};
  const std::size_t n = hs::Lanes(d);
  const std::size_t nh = n / 2;
  ASSERT_GE(n, 2u);
  Buf<std::int32_t> a;
  for (std::size_t i = 0; i < n; ++i) a.data[i] = int(i);
  const auto va = hs::Load(d, a.data);
  const auto m = hs::Gt(va, hs::Set(d, int(nh)));  // some true in both halves potentially

  using DH0 = hs::DFromV<decltype(hs::LowerHalf(hs::VecFromMask(d, m)))>;
  const DH0 dh0{};
  Buf<std::int32_t> lo_out, hi_out, ref;
  // LowerHalfOfMask / UpperHalfOfMask
  hs::Store(hs::VecFromMask(dh0, hs::LowerHalfOfMask(dh0, m)), dh0, lo_out.data);
  hs::Store(hs::VecFromMask(dh0, hs::UpperHalfOfMask(dh0, m)), dh0, hi_out.data);
  hs::Store(hs::VecFromMask(d, m), d, ref.data);
  for (std::size_t i = 0; i < nh; ++i) {
    EXPECT_EQ(lo_out.data[i], ref.data[i]) << "LowerHalfOfMask i=" << i;
    EXPECT_EQ(hi_out.data[i], ref.data[nh + i]) << "UpperHalfOfMask i=" << i;
  }
  // CombineMasks(UpperHalfOfMask, LowerHalfOfMask) round-trips to m.
  const auto lo = hs::LowerHalfOfMask(dh0, m);
  const auto hi = hs::UpperHalfOfMask(dh0, m);
  Buf<std::int32_t> rt;
  hs::Store(hs::VecFromMask(d, hs::CombineMasks(d, hi, lo)), d, rt.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(rt.data[i], ref.data[i]) << "CombineMasks round-trip i=" << i;
  (void)dh;
}

// ---- simd-18 批B: Inf/NaN/IfVecThenElse/IfNegativeThenNegOrUndefIfZero/HighestSetBitIndex +
// GatherIndexN/Or + ScatterIndexN ------------------------------------------------------------
TEST(Vec, VectorMiscAndGatherN) {
  const hs::ScalableTag<float> df;
  const std::size_t nf = hs::Lanes(df);
  Buf<float> f_out;
  // Inf / NaN splats.
  hs::Store(hs::Inf(df), df, f_out.data);
  for (std::size_t i = 0; i < nf; ++i) EXPECT_TRUE(std::isinf(f_out.data[i]) && f_out.data[i] > 0) << "Inf i=" << i;
  hs::Store(hs::NaN(df), df, f_out.data);
  for (std::size_t i = 0; i < nf; ++i) EXPECT_TRUE(std::isnan(f_out.data[i])) << "NaN i=" << i;

  // IfNegativeThenNegOrUndefIfZero (float): mask<0 -> -v ; mask>0 -> v.
  Buf<float> mask, v, r;
  for (std::size_t i = 0; i < nf; ++i) { mask.data[i] = (i % 2) ? -1.0f : 2.0f; v.data[i] = float(i) + 1.0f; }
  hs::Store(hs::IfNegativeThenNegOrUndefIfZero(hs::Load(df, mask.data), hs::Load(df, v.data)), df, r.data);
  for (std::size_t i = 0; i < nf; ++i) {
    const float want = (mask.data[i] < 0) ? -v.data[i] : v.data[i];
    EXPECT_FLOAT_EQ(r.data[i], want) << "IfNegNeg float i=" << i;
  }

  // Integer variant + IfVecThenElse + HighestSetBitIndex.
  const hs::ScalableTag<std::int32_t> d;
  const std::size_t n = hs::Lanes(d);
  Buf<std::int32_t> im, iv, ir;
  for (std::size_t i = 0; i < n; ++i) { im.data[i] = (i % 2) ? -5 : 7; iv.data[i] = int(i) - 2; }
  hs::Store(hs::IfNegativeThenNegOrUndefIfZero(hs::Load(d, im.data), hs::Load(d, iv.data)), d, ir.data);
  for (std::size_t i = 0; i < n; ++i) {
    const int want = (im.data[i] < 0) ? -iv.data[i] : iv.data[i];
    EXPECT_EQ(ir.data[i], want) << "IfNegNeg int i=" << i;
  }
  // IfVecThenElse: selector vector MSB set -> yes.
  Buf<std::int32_t> sel, yes, no, out;
  for (std::size_t i = 0; i < n; ++i) { sel.data[i] = (i % 2) ? -1 : 0; yes.data[i] = 100 + int(i); no.data[i] = 200 + int(i); }
  hs::Store(hs::IfVecThenElse(hs::Load(d, sel.data), hs::Load(d, yes.data), hs::Load(d, no.data)), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], (sel.data[i] < 0) ? yes.data[i] : no.data[i]) << "IfVec i=" << i;
  // HighestSetBitIndex (unsigned lanes for clean floor-log2).
  const hs::ScalableTag<std::uint32_t> du;
  const std::size_t nu = hs::Lanes(du);
  Buf<std::uint32_t> uv, uo;
  for (std::size_t i = 0; i < nu; ++i) uv.data[i] = std::uint32_t(1) << (i % 20);  // 2^k
  hs::Store(hs::HighestSetBitIndex(hs::Load(du, uv.data)), du, uo.data);
  for (std::size_t i = 0; i < nu; ++i) EXPECT_EQ(uo.data[i], std::uint32_t(i % 20)) << "HighestSetBit i=" << i;

  // GatherIndexN / Or / ScatterIndexN.
  constexpr std::size_t kTab = 64;
  std::vector<std::int32_t> table(kTab);
  for (std::size_t i = 0; i < kTab; ++i) table[i] = int(1000 + i);
  Buf<std::int32_t> gidx, g_out;
  for (std::size_t i = 0; i < n; ++i) gidx.data[i] = int((i * 3) % kTab);
  const std::size_t take = n / 2;
  hs::Store(hs::GatherIndexN(d, table.data(), hs::Load(d, gidx.data), take), d, g_out.data);
  for (std::size_t i = 0; i < n; ++i) {
    const int want = (i < take) ? table[(i * 3) % kTab] : 0;
    EXPECT_EQ(g_out.data[i], want) << "GatherIndexN i=" << i;
  }
  Buf<std::int32_t> no_v;
  for (std::size_t i = 0; i < n; ++i) no_v.data[i] = -9;
  hs::Store(hs::GatherIndexNOr(hs::Load(d, no_v.data), d, table.data(), hs::Load(d, gidx.data), take), d, g_out.data);
  for (std::size_t i = 0; i < n; ++i) {
    const int want = (i < take) ? table[(i * 3) % kTab] : -9;
    EXPECT_EQ(g_out.data[i], want) << "GatherIndexNOr i=" << i;
  }
  // ScatterIndexN: only first `take` written.
  std::vector<std::int32_t> dst(kTab, -1);
  Buf<std::int32_t> sidx, sval;
  for (std::size_t i = 0; i < n; ++i) { sidx.data[i] = int(i); sval.data[i] = int(500 + i); }
  hs::ScatterIndexN(hs::Load(d, sval.data), d, dst.data(), hs::Load(d, sidx.data), take);
  for (std::size_t i = 0; i < n && i < kTab; ++i)
    EXPECT_EQ(dst[i], (i < take) ? int(500 + i) : -1) << "ScatterIndexN i=" << i;
}

// ---- simd-18 批C(部分): Blocks / OrderedTruncate2To / TruncateStore --------------------------
TEST(Vec, DemoteTruncateHelpers) {
  // Blocks: constexpr block count = bytes/16 (>=1).
  const hs::ScalableTag<std::int32_t> d;
  const std::size_t nbytes = hs::Lanes(d) * sizeof(std::int32_t);
  EXPECT_EQ(hs::Blocks(d), nbytes <= 16 ? std::size_t(1) : nbytes / 16);

  // OrderedTruncate2To: two u32 vectors -> u16, ordered [a..., b...] keeping low 16 bits.
  const hs::ScalableTag<std::uint32_t> du32;
  const std::size_t nw = hs::Lanes(du32);
  using DN = hs::Repartition<std::uint16_t, hs::ScalableTag<std::uint32_t>>;  // 2x lanes u16
  const DN dn{};
  Buf<std::uint32_t> a, b;
  for (std::size_t i = 0; i < nw; ++i) { a.data[i] = 0x11110000u + i; b.data[i] = 0x22220000u + i; }
  Buf<std::uint16_t> out;
  hs::Store(hs::OrderedTruncate2To(dn, hs::Load(du32, a.data), hs::Load(du32, b.data)), dn, out.data);
  for (std::size_t i = 0; i < nw; ++i) {
    EXPECT_EQ(out.data[i], std::uint16_t(a.data[i] & 0xFFFF)) << "OrdTrunc a i=" << i;
    EXPECT_EQ(out.data[nw + i], std::uint16_t(b.data[i] & 0xFFFF)) << "OrdTrunc b i=" << i;
  }

  // TruncateStore: u32 vector -> u8 buffer, low byte each (same lane count). Narrow tag must be
  // kPow2=0 (TruncateTo takes Simd<DstT,N,0>); build it directly from the u32 lane count.
  const hs::FixedTag<std::uint8_t, hs::kNativeVectorBytes / sizeof(std::uint32_t)> dn8{};
  Buf<std::uint32_t> src;
  for (std::size_t i = 0; i < nw; ++i) src.data[i] = 0xABCD0000u + i * 7;
  Buf<std::uint8_t> tout;
  hs::TruncateStore(hs::Load(du32, src.data), dn8, tout.data);
  for (std::size_t i = 0; i < hs::Lanes(dn8); ++i)
    EXPECT_EQ(tout.data[i], std::uint8_t(src.data[i] & 0xFF)) << "TruncStore i=" << i;

  // TruncateTo native PSHUFB path: cover every integer width combo with NON-trivial bit patterns
  // (the store+reload path was replaced by a byte-gather; a wrong index table would silently emit
  // wrong low bytes). Modular truncation = keep the low sizeof(DstT) bytes of each source lane.
  {  // u32 -> u16
    const hs::FixedTag<std::uint16_t, hs::kNativeVectorBytes / sizeof(std::uint32_t)> d16{};
    Buf<std::uint32_t> s; Buf<std::uint16_t> o;
    for (std::size_t i = 0; i < nw; ++i) s.data[i] = 0xDEAD0000u + static_cast<std::uint32_t>(i * 37 + 1);
    hs::Store(hs::TruncateTo(d16, hs::Load(du32, s.data)), d16, o.data);
    for (std::size_t i = 0; i < nw; ++i) EXPECT_EQ(o.data[i], std::uint16_t(s.data[i] & 0xFFFFu)) << "t u32->u16 i=" << i;
  }
  {  // u64 -> u32 / u16 / u8
    const hs::ScalableTag<std::uint64_t> d64;
    const std::size_t n64 = hs::Lanes(d64);
    Buf<std::uint64_t> s;
    for (std::size_t i = 0; i < n64; ++i) s.data[i] = 0x0123456789ABCDEFull + static_cast<std::uint64_t>(i) * 0x1111ull;
    auto v = hs::Load(d64, s.data);
    const hs::FixedTag<std::uint32_t, hs::ScalableTag<std::uint64_t>::kMaxLanes> d32n{};
    const hs::FixedTag<std::uint16_t, hs::ScalableTag<std::uint64_t>::kMaxLanes> d16n{};
    const hs::FixedTag<std::uint8_t,  hs::ScalableTag<std::uint64_t>::kMaxLanes> d8n{};
    Buf<std::uint32_t> o32; Buf<std::uint16_t> o16; Buf<std::uint8_t> o8;
    hs::Store(hs::TruncateTo(d32n, v), d32n, o32.data);
    hs::Store(hs::TruncateTo(d16n, v), d16n, o16.data);
    hs::Store(hs::TruncateTo(d8n,  v), d8n,  o8.data);
    for (std::size_t i = 0; i < n64; ++i) {
      EXPECT_EQ(o32.data[i], std::uint32_t(s.data[i] & 0xFFFFFFFFull)) << "t u64->u32 i=" << i;
      EXPECT_EQ(o16.data[i], std::uint16_t(s.data[i] & 0xFFFFull))     << "t u64->u16 i=" << i;
      EXPECT_EQ(o8.data[i],  std::uint8_t (s.data[i] & 0xFFull))       << "t u64->u8 i="  << i;
    }
  }
}

// ---- simd-18 C remaining + D: OrderedDemote / InRange / shift-demote2 / WidenMulAccum ----
TEST(Vec, OrderedDemote2ToFamily) {
  // OrderedDemote2To: i32->i16 and i16->i8, lanes ordered [a..., b...].
  {
    const hs::ScalableTag<std::int32_t> dw;
    const std::size_t nw = hs::Lanes(dw);
    using DN = hs::Repartition<std::int16_t, hs::ScalableTag<std::int32_t>>;
    const DN dn{};
    Buf<std::int32_t> a, b;
    for (std::size_t i = 0; i < nw; ++i) {
      a.data[i] = static_cast<std::int32_t>(i * 3 - 10);
      b.data[i] = static_cast<std::int32_t>(i * 5 + 20);
    }
    Buf<std::int16_t> out;
    hs::Store(hs::OrderedDemote2To(dn, hs::Load(dw, a.data), hs::Load(dw, b.data)), dn, out.data);
    for (std::size_t i = 0; i < nw; ++i) {
      const auto sa = static_cast<std::int16_t>(
          a.data[i] < -32768 ? -32768 : (a.data[i] > 32767 ? 32767 : a.data[i]));
      const auto sb = static_cast<std::int16_t>(
          b.data[i] < -32768 ? -32768 : (b.data[i] > 32767 ? 32767 : b.data[i]));
      EXPECT_EQ(out.data[i], sa) << "OrdDemote i32->i16 a i=" << i;
      EXPECT_EQ(out.data[nw + i], sb) << "OrdDemote i32->i16 b i=" << i;
    }
  }
  {
    const hs::ScalableTag<std::int16_t> dw;
    const std::size_t nw = hs::Lanes(dw);
    using DN = hs::Repartition<std::int8_t, hs::ScalableTag<std::int16_t>>;
    const DN dn{};
    Buf<std::int16_t> a, b;
    for (std::size_t i = 0; i < nw; ++i) {
      a.data[i] = static_cast<std::int16_t>(i * 7 - 40);
      b.data[i] = static_cast<std::int16_t>(i * 2 + 5);
    }
    Buf<std::int8_t> out;
    hs::Store(hs::OrderedDemote2To(dn, hs::Load(dw, a.data), hs::Load(dw, b.data)), dn, out.data);
    for (std::size_t i = 0; i < nw; ++i) {
      const auto sa = static_cast<std::int8_t>(
          a.data[i] < -128 ? -128 : (a.data[i] > 127 ? 127 : a.data[i]));
      const auto sb = static_cast<std::int8_t>(
          b.data[i] < -128 ? -128 : (b.data[i] > 127 ? 127 : b.data[i]));
      EXPECT_EQ(out.data[i], sa) << "OrdDemote i16->i8 a i=" << i;
      EXPECT_EQ(out.data[nw + i], sb) << "OrdDemote i16->i8 b i=" << i;
    }
  }
}

TEST(Vec, InRangeConvertAndDemote) {
  // ConvertInRangeTo f32->i32 equals ConvertTo on in-domain inputs.
  {
    const hs::ScalableTag<float> df;
    const hs::ScalableTag<std::int32_t> di;
    const std::size_t n = hs::Lanes(df);
    Buf<float> src;
    for (std::size_t i = 0; i < n; ++i) src.data[i] = static_cast<float>(i) * 1.5f - 3.0f;
    Buf<std::int32_t> a, b;
    hs::Store(hs::ConvertInRangeTo(di, hs::Load(df, src.data)), di, a.data);
    hs::Store(hs::ConvertTo(di, hs::Load(df, src.data)), di, b.data);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(a.data[i], b.data[i]) << "InRange cvt i=" << i;
  }
  // DemoteInRangeTo f64->i32: in-domain truncate, same lane count.
  {
    const hs::ScalableTag<double> dd;
    const std::size_t n = hs::Lanes(dd);
    // dest tag: same N, i32 (half the register width on wide backends)
    const hs::FixedTag<std::int32_t, hs::kNativeVectorBytes / sizeof(double)> di{};
    ASSERT_EQ(hs::Lanes(di), n);
    Buf<double> src;
    for (std::size_t i = 0; i < n; ++i) src.data[i] = static_cast<double>(i) * 2.7 - 5.5;
    Buf<std::int32_t> out;
    hs::Store(hs::DemoteInRangeTo(di, hs::Load(dd, src.data)), di, out.data);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_EQ(out.data[i], static_cast<std::int32_t>(src.data[i])) << "DemoteInRange i=" << i;
  }
}

// @opus: PromoteMaskTo / DemoteMaskTo / OrderedDemote2MasksTo -- lane-width mask conversions via
// bitfield round-trip. Ground truth = the per-lane bool predicate is width-independent, so promote
// and demote must reproduce the SAME lane predicates in the target width. Uses i16<->i32 (Rebind =
// same lane count, different width). VecFromMask stores all-ones/zero per lane; compare bit i true.
TEST(Vec, MaskPromoteDemote) {
  // Anchor on the WIDE type (full native lane count). The narrow tag is Rebind = SAME lane count,
  // half register occupancy (a partial tag) -- matching Highway's contract that d_from/d_to for a
  // width change share lane COUNT. (Rebinding the narrow full tag to wider would 2x the register.)
  const hs::ScalableTag<std::int32_t> d32;          // wide anchor, N lanes, full register
  const std::size_t n = hs::Lanes(d32);
  using D16 = hs::Rebind<std::int16_t, hs::ScalableTag<std::int32_t>>;  // same N, half occupancy
  const D16 d16{};
  ASSERT_EQ(hs::Lanes(d16), n);
  // Verify via BitsFromMask (independent op: reports which lanes the RESULT mask actually has set,
  // regardless of how it was built) against the bool oracle. Storing the partial narrow-tag vector
  // isn't offered on fixed-width backends, so BitsFromMask is the clean per-lane readout.
  // Build a mask on the narrow tag: lane i true iff i is odd.
  Buf<std::int16_t> s16;
  for (std::size_t i = 0; i < n; ++i) s16.data[i] = static_cast<std::int16_t>((i & 1) ? -1 : 3);
  const auto m16 = hs::Lt(hs::Load(d16, s16.data), hs::Zero(d16));  // odd lanes true
  // PromoteMaskTo: i16 mask -> i32 mask, same predicates.
  const std::uint64_t up = hs::BitsFromMask(d32, hs::PromoteMaskTo(d32, d16, m16));
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(((up >> i) & 1) != 0, (i & 1) != 0) << "PromoteMaskTo i=" << i;
  // DemoteMaskTo: build a mask on the wide tag, narrow it back; predicates preserved.
  Buf<std::int32_t> s32;
  for (std::size_t i = 0; i < n; ++i) s32.data[i] = static_cast<std::int32_t>((i % 3 == 0) ? -7 : 2);
  const auto m32 = hs::Lt(hs::Load(d32, s32.data), hs::Zero(d32));  // every 3rd lane true
  const std::uint64_t dn = hs::BitsFromMask(d16, hs::DemoteMaskTo(d16, d32, m32));
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(((dn >> i) & 1) != 0, (i % 3 == 0)) << "DemoteMaskTo i=" << i;
  // Round-trip: PromoteMaskTo then DemoteMaskTo == identity.
  const std::uint64_t rt =
      hs::BitsFromMask(d16, hs::DemoteMaskTo(d16, d32, hs::PromoteMaskTo(d32, d16, m16)));
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(((rt >> i) & 1) != 0, (i & 1) != 0) << "Promote/Demote mask round-trip i=" << i;
}

// @opus: OrderedDemote2MasksTo -- two N-lane wide masks -> one 2N-lane narrow mask, ordered
// [a-lanes..., b-lanes...]. Bitfields concatenated: a low, b high. i32(x2) -> i16.
TEST(Vec, OrderedDemote2MasksToFamily) {
  const hs::ScalableTag<std::int32_t> dw;           // wide source, Nw lanes
  const std::size_t nw = hs::Lanes(dw);
  using DN = hs::Repartition<std::int16_t, hs::ScalableTag<std::int32_t>>;  // 2Nw narrow lanes
  const DN dn{};
  ASSERT_EQ(hs::Lanes(dn), 2 * nw);
  Buf<std::int32_t> sa, sb;
  for (std::size_t i = 0; i < nw; ++i) {
    sa.data[i] = static_cast<std::int32_t>((i & 1) ? -1 : 5);       // a: odd lanes true
    sb.data[i] = static_cast<std::int32_t>((i % 2 == 0) ? -9 : 4);  // b: even lanes true
  }
  const auto ma = hs::Lt(hs::Load(dw, sa.data), hs::Zero(dw));
  const auto mb = hs::Lt(hs::Load(dw, sb.data), hs::Zero(dw));
  Buf<std::int16_t> out;
  hs::Store(hs::VecFromMask(dn, hs::OrderedDemote2MasksTo(dn, dw, ma, mb)), dn, out.data);
  for (std::size_t i = 0; i < nw; ++i) {
    EXPECT_EQ(out.data[i] != 0, (i & 1) != 0) << "OrderedDemote2MasksTo a-half i=" << i;
    EXPECT_EQ(out.data[nw + i] != 0, (i % 2 == 0)) << "OrderedDemote2MasksTo b-half i=" << i;
  }
}

// @opus: DemoteToNearestInt f64->i32 round-to-nearest-EVEN + sign-saturate on overflow/NaN.
// Oracle = Highway emu128 semantics reimplemented per-lane (bias+round-even; |v|>INT32_MAX or NaN
// -> signbit ? INT32_MIN : INT32_MAX). Covers halfway ties (round-even), negatives, and saturation.
TEST(Vec, DemoteToNearestIntFamily) {
  const hs::ScalableTag<double> dd;
  const std::size_t n = hs::Lanes(dd);
  const hs::FixedTag<std::int32_t, hs::kNativeVectorBytes / sizeof(double)> di{};
  ASSERT_EQ(hs::Lanes(di), n);
  auto oracle = [](double x) -> std::int32_t {
    const double kMaxD = 2147483647.0;
    if (!(std::fabs(x) <= kMaxD))  // NaN or too large
      return std::signbit(x) ? INT32_MIN : INT32_MAX;
    const double bias = (x < 0.0) ? -0.5 : 0.5;
    std::int32_t rounded = static_cast<std::int32_t>(x + bias);
    if (rounded == 0) return 0;
    const double rf = static_cast<double>(rounded);
    if ((rounded & 1) && std::fabs(rf - x) == 0.5)  // round half to even
      rounded -= std::signbit(x) ? -1 : 1;
    return rounded;
  };
  // A spread of exact halves (tie->even), ordinary values, negatives, and out-of-range/NaN.
  const double kInf = std::numeric_limits<double>::infinity();
  const double kNaN = std::numeric_limits<double>::quiet_NaN();
  const double kPool[] = {0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 2.3, -2.3, 7.0,
                          123456.5, -123456.5, 3e9, -3e9, kInf, -kInf, kNaN, 0.0};
  const std::size_t kPoolN = sizeof(kPool) / sizeof(kPool[0]);
  Buf<double> src;
  for (std::size_t base = 0; base < kPoolN; base += n) {
    for (std::size_t i = 0; i < n; ++i) src.data[i] = kPool[(base + i) % kPoolN];
    Buf<std::int32_t> out;
    hs::Store(hs::DemoteToNearestInt(di, hs::Load(dd, src.data)), di, out.data);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_EQ(out.data[i], oracle(src.data[i]))
          << "DemoteToNearestInt v=" << src.data[i] << " i=" << i;
  }
  // Fast in-range variant: on inputs the caller guarantees are finite & in [MIN,MAX], it must equal
  // the safe form (which reduces to plain RNE there). Pool restricted to representable values.
  const double kInPool[] = {0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 2.3, -2.3, 7.0, 123456.5, -123456.5};
  const std::size_t kInPoolN = sizeof(kInPool) / sizeof(kInPool[0]);
  for (std::size_t base = 0; base < kInPoolN; base += n) {
    for (std::size_t i = 0; i < n; ++i) src.data[i] = kInPool[(base + i) % kInPoolN];
    Buf<std::int32_t> fast, safe;
    hs::Store(hs::DemoteToNearestIntInRange(di, hs::Load(dd, src.data)), di, fast.data);
    hs::Store(hs::DemoteToNearestInt(di, hs::Load(dd, src.data)), di, safe.data);
    for (std::size_t i = 0; i < n; ++i) {
      EXPECT_EQ(fast.data[i], oracle(src.data[i])) << "DemoteToNearestIntInRange v=" << src.data[i];
      EXPECT_EQ(fast.data[i], safe.data[i]) << "InRange==safe on in-range v=" << src.data[i];
    }
  }
}

TEST(Vec, ShiftRightAndDemote2Family) {
  // OrderedShiftRightAndDemote2To / Reorder* vs hand composition; k=1 and k=3.
  const hs::ScalableTag<std::int32_t> dw;
  const std::size_t nw = hs::Lanes(dw);
  using DN = hs::Repartition<std::int16_t, hs::ScalableTag<std::int32_t>>;
  const DN dn{};
  Buf<std::int32_t> a, b;
  for (std::size_t i = 0; i < nw; ++i) {
    a.data[i] = static_cast<std::int32_t>((i + 1) * 17);
    b.data[i] = static_cast<std::int32_t>((i + 3) * 11);
  }
  auto va = hs::Load(dw, a.data);
  auto vb = hs::Load(dw, b.data);
  for (int k : {1, 3}) {
    Buf<std::int16_t> got, ref;
    if (k == 1) {
      hs::Store(hs::OrderedShiftRightAndDemote2To<1>(dn, va, vb), dn, got.data);
      hs::Store(hs::OrderedDemote2To(dn, hs::ShiftRight<1>(va), hs::ShiftRight<1>(vb)), dn, ref.data);
      Buf<std::int16_t> rgot, rref;
      hs::Store(hs::ReorderShiftRightAndDemote2To<1>(dn, va, vb), dn, rgot.data);
      hs::Store(hs::ReorderDemote2To(dn, hs::ShiftRight<1>(va), hs::ShiftRight<1>(vb)), dn, rref.data);
      for (std::size_t i = 0; i < 2 * nw; ++i) {
        EXPECT_EQ(got.data[i], ref.data[i]) << "OrdShiftDemote k=1 i=" << i;
        EXPECT_EQ(rgot.data[i], rref.data[i]) << "ReordShiftDemote k=1 i=" << i;
      }
      hs::Store(hs::OrderedRoundingShiftRightAndDemote2To<1>(dn, va, vb), dn, got.data);
      hs::Store(hs::OrderedDemote2To(dn, hs::RoundingShiftRight<1>(va), hs::RoundingShiftRight<1>(vb)),
                dn, ref.data);
      for (std::size_t i = 0; i < 2 * nw; ++i)
        EXPECT_EQ(got.data[i], ref.data[i]) << "OrdRoundShiftDemote k=1 i=" << i;
    } else {
      hs::Store(hs::OrderedShiftRightAndDemote2To<3>(dn, va, vb), dn, got.data);
      hs::Store(hs::OrderedDemote2To(dn, hs::ShiftRight<3>(va), hs::ShiftRight<3>(vb)), dn, ref.data);
      for (std::size_t i = 0; i < 2 * nw; ++i)
        EXPECT_EQ(got.data[i], ref.data[i]) << "OrdShiftDemote k=3 i=" << i;
      hs::Store(hs::ReorderRoundingShiftRightAndDemote2To<3>(dn, va, vb), dn, got.data);
      hs::Store(hs::ReorderDemote2To(dn, hs::RoundingShiftRight<3>(va), hs::RoundingShiftRight<3>(vb)),
                dn, ref.data);
      for (std::size_t i = 0; i < 2 * nw; ++i)
        EXPECT_EQ(got.data[i], ref.data[i]) << "ReordRoundShiftDemote k=3 i=" << i;
    }
  }
}

// Single-vector Promote/Demote convenience forms (Highway's Rebind-tag shape, SAME lane count).
// DemoteTo(dn, v): N wide -> N narrow (half register); PromoteTo(dw, v): N narrow -> N wide.
TEST(Vec, SingleVectorPromoteDemote) {
  // The single-vector result is a PARTIAL vector (Rebind tag, kPow2=-1); HYLE's Store only takes a
  // concrete Simd<T,N,0> tag, so read the result back through DFromV<decltype(result)> (the same
  // pattern HalfVectorFamily uses for LowerHalf results).
  // DemoteTo i32 -> i16, saturating, N lanes -> N lanes.
  {
    const hs::ScalableTag<std::int32_t> dw;
    const std::size_t n = hs::Lanes(dw);
    const hs::Rebind<std::int16_t, hs::ScalableTag<std::int32_t>> dn;
    Buf<std::int32_t> in;
    Buf<std::int16_t> out;
    for (std::size_t i = 0; i < n; ++i)
      in.data[i] = static_cast<std::int32_t>(i) * 9000 - 20000;  // spans i16 saturation range
    const auto r = hs::DemoteTo(dn, hs::Load(dw, in.data));
    hs::Store(r, hs::DFromV<decltype(r)>{}, out.data);
    for (std::size_t i = 0; i < n; ++i) {
      const std::int32_t c = std::clamp(in.data[i], -32768, 32767);
      EXPECT_EQ(out.data[i], static_cast<std::int16_t>(c)) << "DemoteTo i32->i16 i=" << i;
    }
  }
  // DemoteTo f64 -> f32, N lanes -> N lanes.
  {
    const hs::ScalableTag<double> dw;
    const std::size_t n = hs::Lanes(dw);
    const hs::Rebind<float, hs::ScalableTag<double>> dn;
    Buf<double> in;
    Buf<float> out;
    for (std::size_t i = 0; i < n; ++i) in.data[i] = 1.5 * static_cast<double>(i) - 3.25;
    const auto r = hs::DemoteTo(dn, hs::Load(dw, in.data));
    hs::Store(r, hs::DFromV<decltype(r)>{}, out.data);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_EQ(out.data[i], static_cast<float>(in.data[i])) << "DemoteTo f64->f32 i=" << i;
  }
  // PromoteTo i16 -> i32 (sign extend) and u8 -> u16 (zero extend), N lanes -> N lanes. The narrow
  // INPUT is loaded via a concrete Simd<T,N,0> tag (Load likewise needs kPow2=0); the wrapper takes
  // the wide dest tag dw and infers the narrow Rebind internally.
  {
    const hs::ScalableTag<std::int32_t> dw;
    const std::size_t n = hs::Lanes(dw);
    const hs::Simd<std::int16_t, hs::ScalableTag<std::int32_t>::kMaxLanes, 0> dn_load;
    Buf<std::int16_t> in;
    Buf<std::int32_t> out;
    for (std::size_t i = 0; i < n; ++i)
      in.data[i] = static_cast<std::int16_t>(-9 + static_cast<int>(i) * 3);
    hs::Store(hs::PromoteTo(dw, hs::Load(dn_load, in.data)), dw, out.data);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_EQ(out.data[i], static_cast<std::int32_t>(in.data[i])) << "PromoteTo i16->i32 i=" << i;
  }
  {
    const hs::ScalableTag<std::uint16_t> dw;
    const std::size_t n = hs::Lanes(dw);
    const hs::Simd<std::uint8_t, hs::ScalableTag<std::uint16_t>::kMaxLanes, 0> dn_load;
    Buf<std::uint8_t> in;
    Buf<std::uint16_t> out;
    for (std::size_t i = 0; i < n; ++i) in.data[i] = static_cast<std::uint8_t>(i * 7 + 1);
    hs::Store(hs::PromoteTo(dw, hs::Load(dn_load, in.data)), dw, out.data);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_EQ(out.data[i], static_cast<std::uint16_t>(in.data[i])) << "PromoteTo u8->u16 i=" << i;
  }
  // ShiftRightAndDemoteTo<k> / RoundingShiftRightAndDemoteTo<k>: i32 -> i16 single-vector.
  {
    const hs::ScalableTag<std::int32_t> dw;
    const std::size_t n = hs::Lanes(dw);
    const hs::Rebind<std::int16_t, hs::ScalableTag<std::int32_t>> dn;
    Buf<std::int32_t> in;
    Buf<std::int16_t> got, ref;
    for (std::size_t i = 0; i < n; ++i) in.data[i] = static_cast<std::int32_t>((i + 1) * 137);
    const auto v = hs::Load(dw, in.data);
    using DR = hs::DFromV<decltype(hs::DemoteTo(dn, v))>;
    hs::Store(hs::ShiftRightAndDemoteTo<3>(dn, v), DR{}, got.data);
    hs::Store(hs::DemoteTo(dn, hs::ShiftRight<3>(v)), DR{}, ref.data);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_EQ(got.data[i], ref.data[i]) << "ShiftRightAndDemoteTo i=" << i;
    hs::Store(hs::RoundingShiftRightAndDemoteTo<3>(dn, v), DR{}, got.data);
    hs::Store(hs::DemoteTo(dn, hs::RoundingShiftRight<3>(v)), DR{}, ref.data);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_EQ(got.data[i], ref.data[i]) << "RoundingShiftRightAndDemoteTo i=" << i;
  }
}

TEST(Vec, WidenMulAccumulateFamily) {
  // D1: WidenMulAccumulate i16*i16 -> dual i32 acc.
  {
    const hs::ScalableTag<std::int16_t> dn;
    const std::size_t nn = hs::Lanes(dn);
    using DW = hs::RepartitionToWide<hs::ScalableTag<std::int16_t>>;
    const DW dw{};
    const std::size_t nw = hs::Lanes(dw);
    ASSERT_EQ(nn, 2 * nw);
    Buf<std::int16_t> mul, x;
    Buf<std::int32_t> low, high;
    for (std::size_t i = 0; i < nn; ++i) {
      mul.data[i] = static_cast<std::int16_t>(i + 1);
      x.data[i] = static_cast<std::int16_t>(i + 2);
    }
    for (std::size_t i = 0; i < nw; ++i) {
      low.data[i] = static_cast<std::int32_t>(100 + i);
      high.data[i] = static_cast<std::int32_t>(200 + i);
    }
    auto vmul = hs::Load(dn, mul.data);
    auto vx = hs::Load(dn, x.data);
    auto vlow = hs::Load(dw, low.data);
    auto vhigh = hs::Load(dw, high.data);
    auto new_low = hs::WidenMulAccumulate(dw, vmul, vx, vlow, vhigh);
    Buf<std::int32_t> olow, ohigh;
    hs::Store(new_low, dw, olow.data);
    hs::Store(vhigh, dw, ohigh.data);
    for (std::size_t i = 0; i < nw; ++i) {
      const std::int32_t exp_lo =
          low.data[i] + static_cast<std::int32_t>(mul.data[i]) * static_cast<std::int32_t>(x.data[i]);
      const std::int32_t exp_hi =
          high.data[i] + static_cast<std::int32_t>(mul.data[nw + i]) *
                             static_cast<std::int32_t>(x.data[nw + i]);
      EXPECT_EQ(olow.data[i], exp_lo) << "WMA lo i=" << i;
      EXPECT_EQ(ohigh.data[i], exp_hi) << "WMA hi i=" << i;
    }
  }
  // D2: SatWidenMulPairwiseAccumulate i16 -> i32 sat. Oracle = int64-exact then saturate
  // (Highway generic:5854; the INT32_MIN pmaddwd corner is compensated to equal exact math).
  // Oracle mirrors the HARDWARE contract, not idealized int64: pmaddwd itself saturates the
  // pairwise sum to i32 (the 4x(-32768) corner => +2^31 saturates to INT32_MAX on real x86,
  // NOT the wrap-to-INT32_MIN that Highway's compensation assumes -- verified via SDE: the
  // AVX512 madd result is 0x7FFFFFFF there). So the reference is TWO-level saturation:
  // product = sat_i32(a0*b0 + a1*b1); result = sat_i32(sum + product). All five backends match.
  auto sat32 = [](std::int64_t v) -> std::int32_t {
    if (v > 2147483647LL) return 2147483647;
    if (v < -2147483648LL) return -2147483648;
    return static_cast<std::int32_t>(v);
  };
  auto d2_oracle = [&](std::int32_t s, std::int16_t a0, std::int16_t a1, std::int16_t b0,
                       std::int16_t b1) -> std::int32_t {
    const std::int32_t product =
        sat32(static_cast<std::int64_t>(a0) * b0 + static_cast<std::int64_t>(a1) * b1);
    return sat32(static_cast<std::int64_t>(s) + static_cast<std::int64_t>(product));
  };
  {
    const hs::ScalableTag<std::int16_t> d16;
    const std::size_t n16 = hs::Lanes(d16);
    using D32 = hs::RepartitionToWide<hs::ScalableTag<std::int16_t>>;
    const D32 d32{};
    const std::size_t n32 = hs::Lanes(d32);
    ASSERT_EQ(n16, 2 * n32);
    // Ordinary values.
    Buf<std::int16_t> a, b;
    Buf<std::int32_t> sum, out;
    for (std::size_t i = 0; i < n16; ++i) {
      a.data[i] = static_cast<std::int16_t>((i % 7) - 3);
      b.data[i] = static_cast<std::int16_t>((i % 5) + 1);
    }
    for (std::size_t i = 0; i < n32; ++i) sum.data[i] = static_cast<std::int32_t>(i * 10 - 15);
    hs::Store(hs::SatWidenMulPairwiseAccumulate(d32, hs::Load(d16, a.data), hs::Load(d16, b.data),
                                                hs::Load(d32, sum.data)), d32, out.data);
    for (std::size_t i = 0; i < n32; ++i)
      EXPECT_EQ(out.data[i], d2_oracle(sum.data[i], a.data[2*i], a.data[2*i+1], b.data[2*i], b.data[2*i+1]))
          << "SatWMA ordinary i=" << i;

    // 4x(-32768) corner: pairwise sum = +2^31, unrepresentable in i32. HARDWARE DIVERGES here:
    // SSE2/AVX2 vpmaddwd WRAPS to INT32_MIN (Highway's compensation then recovers the exact 2^31),
    // but AVX512 vpmaddwd SATURATES to INT32_MAX (2^31-1) -- verified via SDE. Both are the
    // legal per-ISA contract; the two results differ by 1 and CANNOT be unified (it is a genuine
    // cross-generation hardware behavior of the madd instruction, not a bug). So the corner lane
    // is checked against the SET of both legal values; ordinary lanes (above) stay exact.
    for (int sign_case = 0; sign_case < 2; ++sign_case) {
      for (std::size_t i = 0; i < n16; ++i) { a.data[i] = -32768; b.data[i] = -32768; }
      for (std::size_t i = 0; i < n32; ++i)
        sum.data[i] = sign_case == 0 ? static_cast<std::int32_t>(i) : -static_cast<std::int32_t>(i + 1);
      hs::Store(hs::SatWidenMulPairwiseAccumulate(d32, hs::Load(d16, a.data), hs::Load(d16, b.data),
                                                  hs::Load(d32, sum.data)), d32, out.data);
      for (std::size_t i = 0; i < n32; ++i) {
        const std::int64_t exact = static_cast<std::int64_t>(sum.data[i]) + 2147483648LL;   // wrap+compensate path
        const std::int64_t satur = static_cast<std::int64_t>(sum.data[i]) + 2147483647LL;   // saturate path
        auto clamp = [](std::int64_t v) {
          return static_cast<std::int32_t>(v > 2147483647LL ? 2147483647LL
                    : (v < -2147483648LL ? -2147483648LL : v));
        };
        const std::int32_t v_exact = clamp(exact), v_sat = clamp(satur);
        EXPECT_TRUE(out.data[i] == v_exact || out.data[i] == v_sat)
            << "SatWMA corner sign_case=" << sign_case << " i=" << i << " got=" << out.data[i]
            << " legal={" << v_exact << "," << v_sat << "}";
      }
    }
  }
}

// ---- simd-18 批E: 128-bit block ops + TwoTablesLookupLanes + D3 ----------------------
TEST(Vec, BlockOpsFamily) {
  const hs::ScalableTag<std::int32_t> d;
  const std::size_t n = hs::Lanes(d);
  const std::size_t blocks = hs::Blocks(d);
  Buf<std::int32_t> a, b, out;
  for (std::size_t i = 0; i < n; ++i) {
    a.data[i] = static_cast<std::int32_t>(i + 1);
    b.data[i] = static_cast<std::int32_t>(100 + i);
  }
  auto va = hs::Load(d, a.data);
  auto vb = hs::Load(d, b.data);

  // SwapAdjacentBlocks / ReverseBlocks: identity when blocks==1; swap halves when blocks==2.
  hs::Store(hs::SwapAdjacentBlocks(va), d, out.data);
  if (blocks == 1) {
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], a.data[i]);
  } else if (blocks == 2) {
    const std::size_t h = n / 2;
    for (std::size_t i = 0; i < h; ++i) {
      EXPECT_EQ(out.data[i], a.data[h + i]) << "swap lo i=" << i;
      EXPECT_EQ(out.data[h + i], a.data[i]) << "swap hi i=" << i;
    }
  }
  // ReverseBlocks equals Swap when blocks<=2
  hs::Store(hs::ReverseBlocks(d, va), d, out.data);
  if (blocks <= 2) {
    auto sw = hs::SwapAdjacentBlocks(va);
    Buf<std::int32_t> ref;
    hs::Store(sw, d, ref.data);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], ref.data[i]);
  }

  // OddEvenBlocks: even block from even, odd from odd.
  hs::Store(hs::OddEvenBlocks(vb, va), d, out.data);
  if (blocks == 1) {
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], a.data[i]);  // returns even
  } else if (blocks == 2) {
    const std::size_t h = n / 2;
    for (std::size_t i = 0; i < h; ++i) {
      EXPECT_EQ(out.data[i], a.data[i]);       // even block from va
      EXPECT_EQ(out.data[h + i], b.data[h + i]);  // odd block from vb
    }
  }

  // SlideUpBlocks<0> / SlideDownBlocks<0> identity
  hs::Store(hs::SlideUpBlocks<0>(d, va), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], a.data[i]);
  hs::Store(hs::SlideDownBlocks<0>(d, va), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], a.data[i]);

  // SlideUp/DownBlocks<1> only valid when Blocks>=2 (compile-time on fixed backends).
  // One 128-bit block = 16/sizeof(T) lanes (not n/2 — avx512 has 4 blocks).
  if constexpr (hs::kNativeVectorBytes >= 32) {
    constexpr std::size_t kBlk = 16 / sizeof(std::int32_t);
    hs::Store(hs::SlideUpBlocks<1>(d, va), d, out.data);
    for (std::size_t i = 0; i < kBlk; ++i) EXPECT_EQ(out.data[i], 0) << "up1 zero i=" << i;
    for (std::size_t i = 0; i + kBlk < n; ++i)
      EXPECT_EQ(out.data[kBlk + i], a.data[i]) << "up1 data i=" << i;

    hs::Store(hs::SlideDownBlocks<1>(d, va), d, out.data);
    for (std::size_t i = 0; i + kBlk < n; ++i)
      EXPECT_EQ(out.data[i], a.data[i + kBlk]) << "dn1 i=" << i;
    for (std::size_t i = n - kBlk; i < n; ++i) EXPECT_EQ(out.data[i], 0) << "dn1 zero i=" << i;
  }

  // InterleaveWholeLower: continuous a0,b0,a1,b1,... for first N/2 pairs from low halves
  hs::Store(hs::InterleaveWholeLower(d, va, vb), d, out.data);
  for (std::size_t i = 0; i < n / 2; ++i) {
    EXPECT_EQ(out.data[2 * i], a.data[i]) << "iwl a i=" << i;
    EXPECT_EQ(out.data[2 * i + 1], b.data[i]) << "iwl b i=" << i;
  }

  // Per4LaneBlockShuffle 2301 on i32
  if (n >= 4 && sizeof(std::int32_t) == 4) {
    hs::Store(hs::Per4LaneBlockShuffle<2, 3, 0, 1>(va), d, out.data);
    constexpr std::size_t blk = 4;
    for (std::size_t base = 0; base + 3 < n; base += blk) {
      EXPECT_EQ(out.data[base + 0], a.data[base + 1]);
      EXPECT_EQ(out.data[base + 1], a.data[base + 0]);
      EXPECT_EQ(out.data[base + 2], a.data[base + 3]);
      EXPECT_EQ(out.data[base + 3], a.data[base + 2]);
    }
  }
}

// directive2: Per4LaneBlockShuffle 8/16-bit lane 补(此前 static_assert 只允 32/64-bit)。
// 语义:每 128-bit block 内,每 4 lane 一组按 <i0,i1,i2,i3> 重排;逐组重复。i16(8 lane/block=
// 2 组)+ i8(16 lane/block=4 组)。oracle 逐 block 逐组算。
template <typename T>
static void CheckPer4Lane8_16() {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  constexpr std::size_t kLanesPer128 = 16 / sizeof(T);
  if (n < kLanesPer128) return;  // need a full 128-bit block
  Buf<T> a, out;
  for (std::size_t i = 0; i < n; ++i) a.data[i] = static_cast<T>(i * 3 + 1);
  auto va = hs::Load(d, a.data);
  // shuffle indices <i3,i2,i1,i0> = <2,3,0,1>: within each group of 4, out[0]=src1,out[1]=src0,
  // out[2]=src3,out[3]=src2.
  hs::Store(hs::Per4LaneBlockShuffle<2, 3, 0, 1>(va), d, out.data);
  const int idx[4] = {1, 0, 3, 2};  // {i0,i1,i2,i3}
  for (std::size_t blk = 0; blk + kLanesPer128 <= n; blk += kLanesPer128) {
    for (std::size_t lane = 0; lane < kLanesPer128; ++lane) {
      const std::size_t group = lane / 4;
      const std::size_t src = blk + group * 4 + static_cast<std::size_t>(idx[lane % 4]);
      EXPECT_EQ(out.data[blk + lane], a.data[src])
          << "Per4Lane" << (sizeof(T) * 8) << " blk=" << blk << " lane=" << lane;
    }
  }
}
TEST(Vec, Per4LaneBlockShuffle8And16) {
  CheckPer4Lane8_16<std::int16_t>();
  CheckPer4Lane8_16<std::uint8_t>();
}

// simd-18 批E 补: wide 类型(f32/f64/i64)+ Upper/Blocks/Insert 覆盖(grok 只测了 i32)。
// backend-aware: 单 block 后端(Blocks==1)下 block ops 退化为 identity/alias。
template <typename T>
static void CheckBlockOpsWide() {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  const std::size_t nb = hs::Blocks(d);
  const std::size_t blk = 16 / sizeof(T);  // lanes per 128-bit block
  Buf<T> a, b, out;
  for (std::size_t i = 0; i < n; ++i) {
    a.data[i] = static_cast<T>(i + 1);
    b.data[i] = static_cast<T>(100 + i);
  }
  auto va = hs::Load(d, a.data), vb = hs::Load(d, b.data);

  // SwapAdjacentBlocks: block-reverse of adjacent pairs [1,0,3,2].
  hs::Store(hs::SwapAdjacentBlocks(va), d, out.data);
  for (std::size_t bi = 0; bi < nb; ++bi) {
    const std::size_t src = (bi ^ 1) < nb ? (bi ^ 1) : bi;  // swap within pair
    for (std::size_t j = 0; j < blk; ++j)
      EXPECT_EQ(out.data[bi * blk + j], a.data[src * blk + j]) << "swap bi=" << bi << " j=" << j;
  }
  // ReverseBlocks: [nb-1..0].
  hs::Store(hs::ReverseBlocks(d, va), d, out.data);
  for (std::size_t bi = 0; bi < nb; ++bi)
    for (std::size_t j = 0; j < blk; ++j)
      EXPECT_EQ(out.data[bi * blk + j], a.data[(nb - 1 - bi) * blk + j]) << "rev bi=" << bi;
  // OddEvenBlocks(odd=vb, even=va): even blocks from va, odd from vb.
  hs::Store(hs::OddEvenBlocks(vb, va), d, out.data);
  for (std::size_t bi = 0; bi < nb; ++bi) {
    const T* src = (bi % 2 == 0) ? a.data : b.data;
    for (std::size_t j = 0; j < blk; ++j)
      EXPECT_EQ(out.data[bi * blk + j], src[bi * blk + j]) << "oddeven bi=" << bi;
  }
  // InterleaveWholeUpper: continuous a[N/2+i],b[N/2+i] pairs.
  hs::Store(hs::InterleaveWholeUpper(d, va, vb), d, out.data);
  for (std::size_t i = 0; i < n / 2; ++i) {
    EXPECT_EQ(out.data[2 * i], a.data[n / 2 + i]) << "iwu a i=" << i;
    EXPECT_EQ(out.data[2 * i + 1], b.data[n / 2 + i]) << "iwu b i=" << i;
  }
  // Interleave{Lower,Upper}Blocks: Lower=[a.b0,b.b0,a.b1,b.b1], Upper=[a.b2,b.b2,a.b3,b.b3].
  if (nb >= 2) {
    hs::Store(hs::InterleaveLowerBlocks(d, va, vb), d, out.data);
    // blocks: 0<-a.b0, 1<-b.b0, 2<-a.b1, 3<-b.b1
    for (std::size_t ob = 0; ob < nb; ++ob) {
      const bool from_a = (ob % 2 == 0);
      const std::size_t srcb = ob / 2;  // a.b0,b.b0,a.b1,b.b1
      const T* src = from_a ? a.data : b.data;
      for (std::size_t j = 0; j < blk; ++j)
        EXPECT_EQ(out.data[ob * blk + j], src[srcb * blk + j]) << "ilb ob=" << ob << " j=" << j;
    }
    hs::Store(hs::InterleaveUpperBlocks(d, va, vb), d, out.data);
    for (std::size_t ob = 0; ob < nb; ++ob) {
      const bool from_a = (ob % 2 == 0);
      const std::size_t srcb = nb / 2 + ob / 2;  // a.b2,b.b2,a.b3,b.b3
      const T* src = from_a ? a.data : b.data;
      for (std::size_t j = 0; j < blk; ++j)
        EXPECT_EQ(out.data[ob * blk + j], src[srcb * blk + j]) << "iub ob=" << ob;
    }
  }
  // Per4LaneBlockShuffle for 32/64-bit: reverse the 4 (or per-block) dwords <0,1,2,3>=identity.
  hs::Store(hs::Per4LaneBlockShuffle<3, 2, 1, 0>(va), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], a.data[i]) << "p4identity i=" << i;
}
TEST(Vec, BlockOpsFamilyWideF32) { CheckBlockOpsWide<float>(); }
TEST(Vec, BlockOpsFamilyWideF64) { CheckBlockOpsWide<double>(); }
TEST(Vec, BlockOpsFamilyWideI64) { CheckBlockOpsWide<std::int64_t>(); }

TEST(Vec, TwoTablesLookupLanesFamily) {
  const hs::ScalableTag<std::int32_t> d;
  const std::size_t n = hs::Lanes(d);
  Buf<std::int32_t> a, b, idx, out;
  for (std::size_t i = 0; i < n; ++i) {
    a.data[i] = static_cast<std::int32_t>(i + 1);
    b.data[i] = static_cast<std::int32_t>(1000 + i);
    // cover both tables: even from a, odd from b
    idx.data[i] = static_cast<std::int32_t>((i % 2 == 0) ? (i / 2) % n : n + (i / 2) % n);
  }
  auto ind = hs::IndicesFromVec(d, hs::Load(d, idx.data));
  hs::Store(hs::TwoTablesLookupLanes(d, hs::Load(d, a.data), hs::Load(d, b.data), ind), d, out.data);
  for (std::size_t i = 0; i < n; ++i) {
    const auto k = static_cast<std::size_t>(idx.data[i]);
    const auto exp = (k < n) ? a.data[k] : b.data[k - n];
    EXPECT_EQ(out.data[i], exp) << "2tbl i=" << i;
  }
}

// @opus: Dup128VecFromValues -- k immediates fill one 128-bit block, broadcast to all blocks.
// Oracle: out lane i == the (i % k)-th supplied value, for every block. Covers 32/16/8/64-bit widths.
TEST(Vec, Dup128VecFromValues) {
  {  // 32-bit: 4 values per block
    const hs::ScalableTag<std::int32_t> d;
    const std::size_t n = hs::Lanes(d);
    Buf<std::int32_t> out;
    hs::Store(hs::Dup128VecFromValues(d, 10, 20, 30, 40), d, out.data);
    const std::int32_t vals[4] = {10, 20, 30, 40};
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], vals[i % 4]) << "dup32 i=" << i;
  }
  {  // 16-bit: 8 values per block
    const hs::ScalableTag<std::int16_t> d;
    const std::size_t n = hs::Lanes(d);
    Buf<std::int16_t> out;
    hs::Store(hs::Dup128VecFromValues(d, std::int16_t(1), std::int16_t(2), std::int16_t(3),
                                      std::int16_t(4), std::int16_t(5), std::int16_t(6),
                                      std::int16_t(7), std::int16_t(8)),
              d, out.data);
    const std::int16_t vals[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], vals[i % 8]) << "dup16 i=" << i;
  }
  {  // 8-bit: 16 values per block
    const hs::ScalableTag<std::uint8_t> d;
    const std::size_t n = hs::Lanes(d);
    Buf<std::uint8_t> out;
    hs::Store(hs::Dup128VecFromValues(d, std::uint8_t(0), std::uint8_t(11), std::uint8_t(22),
                                      std::uint8_t(33), std::uint8_t(44), std::uint8_t(55),
                                      std::uint8_t(66), std::uint8_t(77), std::uint8_t(88),
                                      std::uint8_t(99), std::uint8_t(110), std::uint8_t(121),
                                      std::uint8_t(132), std::uint8_t(143), std::uint8_t(154),
                                      std::uint8_t(165)),
              d, out.data);
    const std::uint8_t vals[16] = {0,  11, 22, 33,  44,  55,  66,  77,
                                   88, 99, 110, 121, 132, 143, 154, 165};
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], vals[i % 16]) << "dup8 i=" << i;
  }
  {  // 64-bit: 2 values per block
    const hs::ScalableTag<std::int64_t> d;
    const std::size_t n = hs::Lanes(d);
    Buf<std::int64_t> out;
    hs::Store(hs::Dup128VecFromValues(d, std::int64_t(-5), std::int64_t(123456789012LL)), d, out.data);
    const std::int64_t vals[2] = {-5, 123456789012LL};
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], vals[i % 2]) << "dup64 i=" << i;
  }
}

TEST(Vec, SatWidenMulAccumFixedPoint) {
  // D3 smoke: just ensure it compiles and runs; Q15-ish values.
  const hs::ScalableTag<std::int16_t> d16;
  const std::size_t n16 = hs::Lanes(d16);
  using D32 = hs::RepartitionToWide<hs::ScalableTag<std::int16_t>>;
  const D32 d32{};
  const std::size_t n32 = hs::Lanes(d32);
  Buf<std::int16_t> a, b;
  Buf<std::int32_t> sum, out;
  for (std::size_t i = 0; i < n16; ++i) {
    a.data[i] = static_cast<std::int16_t>((i % 5) + 1);
    b.data[i] = static_cast<std::int16_t>((i % 3) + 2);
  }
  for (std::size_t i = 0; i < n32; ++i) sum.data[i] = 10;
  hs::Store(hs::SatWidenMulAccumFixedPoint(d32, hs::Load(d16, a.data), hs::Load(d16, b.data),
                                           hs::Load(d32, sum.data)),
            d32, out.data);
  // result is finite i32 (no crash); spot-check non-zero for positive inputs
  bool any = false;
  for (std::size_t i = 0; i < n32; ++i) if (out.data[i] != 10) any = true;
  EXPECT_TRUE(any || n32 == 0);
}

// ---- simd-18 批F: SAD video quad ----------------------------------------------------
TEST(Vec, SumsOfAdjQuadAbsDiffFamily) {
  const hs::ScalableTag<std::uint8_t> d8;
  const std::size_t n = hs::Lanes(d8);
  if (n < 16) GTEST_SKIP() << "need >=16 u8 lanes";
  using D16 = hs::Repartition<std::uint16_t, hs::ScalableTag<std::uint8_t>>;
  // Result has n/2 u16 lanes in same register width.
  const hs::FixedTag<std::uint16_t, /*filled below*/ 1> _ignore{};
  (void)_ignore;
  Buf<std::uint8_t> a, b;
  for (std::size_t i = 0; i < n; ++i) {
    a.data[i] = static_cast<std::uint8_t>(i * 3 + 1);
    b.data[i] = static_cast<std::uint8_t>(i * 2 + 5);
  }
  // Use kA=0, kB=0
  auto va = hs::Load(d8, a.data);
  auto vb = hs::Load(d8, b.data);
  auto r = hs::SumsOfAdjQuadAbsDiff<0, 0>(va, vb);
  const std::size_t n16 = n / 2;
  Buf<std::uint16_t> out;
  // Store through a matching tag: same byte width as d8.
  const hs::FixedTag<std::uint16_t, hs::kNativeVectorBytes / 2> d16{};
  // For full native: kNativeVectorBytes/2 == n/2 when sizeof(u8)*n == kNative.
  ASSERT_EQ(hs::Lanes(d16), n16);
  hs::Store(r, d16, out.data);
  for (std::size_t blk = 0; blk < n; blk += 16) {
    for (int i = 0; i < 8; ++i) {
      unsigned sum = 0;
      for (int j = 0; j < 4; ++j) {
        const int da = a.data[blk + static_cast<std::size_t>(i + j)];
        const int db = b.data[blk + static_cast<std::size_t>(j)];
        sum += static_cast<unsigned>(da >= db ? da - db : db - da);
      }
      EXPECT_EQ(out.data[blk / 2 + static_cast<std::size_t>(i)], static_cast<std::uint16_t>(sum))
          << "SAD blk=" << blk << " i=" << i;
    }
  }
  // F1 non-trivial offsets: kAOffset=1, kBOffset=2.
  {
    auto r12 = hs::SumsOfAdjQuadAbsDiff<1, 2>(va, vb);
    hs::Store(r12, d16, out.data);
    for (std::size_t blk = 0; blk < n; blk += 16)
      for (int i = 0; i < 8; ++i) {
        unsigned sum = 0;
        for (int j = 0; j < 4; ++j) {
          const int da = a.data[blk + static_cast<std::size_t>(1 * 4 + i + j)];
          const int db = b.data[blk + static_cast<std::size_t>(2 * 4 + j)];
          sum += static_cast<unsigned>(da >= db ? da - db : db - da);
        }
        EXPECT_EQ(out.data[blk / 2 + static_cast<std::size_t>(i)], static_cast<std::uint16_t>(sum))
            << "SAD<1,2> blk=" << blk << " i=" << i;
      }
  }
  // F2 SumsOfShuffledQuadAbsDiff: EXACT Highway lane layout (per 128-bit block).
  //   S = per-block u32 shuffle of a by template (i0,i1,i2,i3); Sbase={0,2,8,10} per pair.
  //   out[pair*2+odd] = sum_j | b[pair*4+j] - S[Sbase[pair]+odd+j] |
  // Must match on EVERY backend including avx512 native dbsad (multi-platform gate).
  auto check_shuffled = [&](auto rs, int s0, int s1, int s2, int s3, const char* tag) {
    hs::Store(rs, d16, out.data);
    const int sel[4] = {s0, s1, s2, s3};
    constexpr int kSBase[4] = {0, 2, 8, 10};
    for (std::size_t blk = 0; blk < n; blk += 16) {
      std::uint8_t sh[16];
      for (int q = 0; q < 4; ++q)
        for (int bi = 0; bi < 4; ++bi)
          sh[q * 4 + bi] = a.data[blk + static_cast<std::size_t>(sel[q] * 4 + bi)];
      for (int pair = 0; pair < 4; ++pair)
        for (int odd = 0; odd < 2; ++odd) {
          unsigned sum = 0;
          for (int j = 0; j < 4; ++j) {
            const int da = sh[kSBase[pair] + odd + j];
            const int db = b.data[blk + static_cast<std::size_t>(pair * 4 + j)];
            sum += static_cast<unsigned>(da >= db ? da - db : db - da);
          }
          EXPECT_EQ(out.data[blk / 2 + static_cast<std::size_t>(pair * 2 + odd)],
                    static_cast<std::uint16_t>(sum))
              << "Shuffled" << tag << " blk=" << blk << " pair=" << pair << " odd=" << odd;
        }
    }
  };
  check_shuffled(hs::SumsOfShuffledQuadAbsDiff<3, 2, 1, 0>(va, vb), 0, 1, 2, 3, "<3210>");
  check_shuffled(hs::SumsOfShuffledQuadAbsDiff<0, 1, 2, 3>(va, vb), 3, 2, 1, 0, "<0123>");
  check_shuffled(hs::SumsOfShuffledQuadAbsDiff<2, 0, 3, 1>(va, vb), 1, 3, 0, 2, "<2031>");
}
