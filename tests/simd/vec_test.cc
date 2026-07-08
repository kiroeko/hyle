#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <bit>
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
constexpr std::size_t kMaxLanes = 32;  // >= widest v1 lane count (avx2 i8 = 32)

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
  // FindKnownFirstTrue matches.
  EXPECT_EQ(static_cast<std::ptrdiff_t>(hs::FindKnownFirstTrue(d, m)), first);
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
  T ref[32]; std::size_t k = 0;
  for (std::size_t i = 0; i < n; ++i) if (i % 2 == 1) ref[k++] = v.data[i];
  for (std::size_t j = k; j < n; ++j) ref[j] = T{0};
  hs::Store(hs::Compress(hs::Load(d, v.data), m), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], ref[i]) << "Compress " << i;
  Buf<T> sp;
  for (std::size_t i = 0; i < n; ++i) sp.data[i] = static_cast<T>(0);
  std::size_t cnt = hs::CompressStore(hs::Load(d, v.data), m, d, sp.data);
  EXPECT_EQ(cnt, k);
  for (std::size_t i = 0; i < k; ++i) EXPECT_EQ(sp.data[i], ref[i]) << "CompressStore " << i;
  // Expand(compress-packed, m) restores the true lanes.
  hs::Store(hs::Expand(hs::Load(d, ref), m), d, out.data);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out.data[i], (i % 2 == 1) ? v.data[i] : T{0}) << "Expand " << i;
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
