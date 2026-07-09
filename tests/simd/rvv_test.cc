// RVV backend test -- the RISC-V Vector backend's op face cross-checked against a plain scalar
// oracle. Like sve_test.cc, this is VL-AGNOSTIC: it queries the runtime Lanes(d) (vsetvlmax) and
// checks every implemented op for every lane type, so the code is correct at whatever VLEN the
// hardware/emulator provides. Cross-compiled (riscv64-linux-gnu-g++ -march=rv64gcv) and run under
// qemu-riscv64-static. See scripts/rvv-qemu-test.sh and docs/architecture/simd-design.md.
//
// Op surface mirrors backend/rvv.h (LMUL=1 first version). Ops NOT yet in rvv.h -- native
// gather/compress/scatter, width-changing convert, block swizzle, Zvbb popcount/clz -- are
// intentionally absent (documented boundary, same as the SVE landing).
#include <gtest/gtest.h>

#define HYLE_SIMD_TARGET_RVV 1
#include "hyle/simd/vec.h"

#include <cmath>
#include <cstdint>

namespace hs = hyle::simd;

namespace {
constexpr std::size_t kBuf = 512;  // >= any reasonable VLEN's element count at LMUL=1
}  // namespace

template <typename T>
void CheckLoadStore() {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  ASSERT_GE(n, std::size_t{1});
  alignas(64) T buf[kBuf];
  hs::Store(hs::Set(d, static_cast<T>(7)), d, buf);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(buf[i], static_cast<T>(7));
  hs::Store(hs::Zero(d), d, buf);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(buf[i], static_cast<T>(0));
  alignas(64) T src[kBuf], dst[kBuf];
  for (std::size_t i = 0; i < n; ++i) src[i] = static_cast<T>(i * 3 + 1);
  hs::Store(hs::Load(d, src), d, dst);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(dst[i], src[i]);
}

TEST(Rvv, LoadStoreAllTypes) {
  CheckLoadStore<float>();
  CheckLoadStore<double>();
  CheckLoadStore<std::int8_t>();
  CheckLoadStore<std::uint8_t>();
  CheckLoadStore<std::int16_t>();
  CheckLoadStore<std::uint16_t>();
  CheckLoadStore<std::int32_t>();
  CheckLoadStore<std::uint32_t>();
  CheckLoadStore<std::int64_t>();
  CheckLoadStore<std::uint64_t>();
}

template <typename T>
void CheckIntArith() {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  alignas(64) T a[kBuf], b[kBuf], o[kBuf];
  for (std::size_t i = 0; i < n; ++i) { a[i] = static_cast<T>(i + 1); b[i] = static_cast<T>((i % 5) + 1); }
  auto va = hs::Load(d, a), vb = hs::Load(d, b);

  hs::Store(hs::Add(va, vb), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], static_cast<T>(a[i] + b[i]));
  hs::Store(hs::Sub(va, vb), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], static_cast<T>(a[i] - b[i]));
  hs::Store(hs::Mul(va, vb), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], static_cast<T>(a[i] * b[i]));
  hs::Store(hs::Min(va, vb), d, o);  // exercises the unsigned-mnemonic split (vminu)
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], static_cast<T>(a[i] < b[i] ? a[i] : b[i]));
  hs::Store(hs::Max(va, vb), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], static_cast<T>(a[i] > b[i] ? a[i] : b[i]));
  hs::Store(hs::And(va, vb), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], static_cast<T>(a[i] & b[i]));
  hs::Store(hs::Or(va, vb), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], static_cast<T>(a[i] | b[i]));
  hs::Store(hs::Xor(va, vb), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], static_cast<T>(a[i] ^ b[i]));

  auto m = hs::Lt(va, vb);  // unsigned compare uses vmsltu
  hs::Store(hs::IfThenElse(m, va, vb), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], static_cast<T>(a[i] < b[i] ? a[i] : b[i]));
  std::size_t ref = 0;
  for (std::size_t i = 0; i < n; ++i) if (a[i] < b[i]) ++ref;
  EXPECT_EQ(hs::CountTrue(d, m), ref);
  EXPECT_TRUE(hs::AllTrue(d, hs::Eq(va, va)));
  EXPECT_TRUE(hs::AllFalse(d, hs::Lt(va, va)));
  EXPECT_EQ(hs::GetLane(va), a[0]);

  T rs = 0; for (std::size_t i = 0; i < n; ++i) rs += a[i];
  EXPECT_EQ(hs::ReduceSum(d, va), rs);
  EXPECT_EQ(hs::ReduceMin(d, va), a[0]);
  EXPECT_EQ(hs::ReduceMax(d, va), a[n - 1]);

  hs::Store(hs::ShiftLeft<2>(va), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], static_cast<T>(a[i] << 2));
  hs::Store(hs::ShiftRight<1>(va), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], static_cast<T>(a[i] >> 1));
}

TEST(Rvv, IntArithAllTypes) {
  CheckIntArith<std::int8_t>();
  CheckIntArith<std::uint8_t>();
  CheckIntArith<std::int16_t>();
  CheckIntArith<std::uint16_t>();
  CheckIntArith<std::int32_t>();
  CheckIntArith<std::uint32_t>();
  CheckIntArith<std::int64_t>();
  CheckIntArith<std::uint64_t>();
}

TEST(Rvv, FloatArith) {
  const hs::ScalableTag<float> d;
  const std::size_t n = hs::Lanes(d);
  alignas(64) float a[kBuf], b[kBuf], c[kBuf], o[kBuf];
  for (std::size_t i = 0; i < n; ++i) { a[i] = static_cast<float>(i + 1); b[i] = 2.0f; c[i] = 0.5f; }
  auto va = hs::Load(d, a), vb = hs::Load(d, b), vc = hs::Load(d, c);
  hs::Store(hs::Div(va, vb), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_NEAR(o[i], a[i] / b[i], 1e-5f);
  hs::Store(hs::Sqrt(va), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_NEAR(o[i], std::sqrt(a[i]), 1e-4f);
  hs::Store(hs::MulAdd(va, vb, vc), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_NEAR(o[i], a[i] * b[i] + c[i], 1e-4f);
  hs::Store(hs::NegMulAdd(va, vb, vc), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_NEAR(o[i], c[i] - a[i] * b[i], 1e-4f);
  float fs = 0; for (std::size_t i = 0; i < n; ++i) fs += a[i];
  EXPECT_NEAR(hs::ReduceSum(d, va), fs, 1e-2f);
}

// ---- tail ops mirrored from SVE: bitcast/convert/reverse/gather/scatter/compress/tail ----
TEST(Rvv, TailOps) {
  const hs::ScalableTag<std::int32_t> di;
  const hs::ScalableTag<float> df;
  const hs::ScalableTag<std::uint32_t> du;
  const std::size_t n = hs::Lanes(di);
  alignas(64) std::int32_t a[kBuf], o[kBuf];
  for (std::size_t i = 0; i < n; ++i) a[i] = static_cast<std::int32_t>(i * 7 + 3);
  auto va = hs::Load(di, a);

  hs::Store(hs::Abs(hs::Sub(hs::Zero(di), va)), di, o);  // Abs(-a) == a
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], a[i]);
  hs::Store(hs::Clamp(va, hs::Set(di, 5), hs::Set(di, 50)), di, o);
  for (std::size_t i = 0; i < n; ++i) { std::int32_t e = a[i] < 5 ? 5 : (a[i] > 50 ? 50 : a[i]); EXPECT_EQ(o[i], e); }
  hs::Store(hs::BitCast(di, hs::BitCast(df, va)), di, o);  // i32->f32->i32 round-trip
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], a[i]);

  alignas(64) float f[kBuf]; for (std::size_t i = 0; i < n; ++i) f[i] = static_cast<float>(i) * 1.4f + 0.6f;
  auto vf = hs::Load(df, f);
  hs::Store(hs::ConvertTo(di, vf), di, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], static_cast<std::int32_t>(f[i]));  // truncate
  hs::Store(hs::RoundToInt(di, vf), di, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], static_cast<std::int32_t>(std::nearbyint(f[i])));

  hs::Store(hs::Reverse(di, va), di, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], a[n - 1 - i]);

  const std::size_t k = n > 3 ? 3 : n;
  alignas(64) std::int32_t o2[kBuf] = {};
  hs::StoreN(hs::LoadN(di, a, k), di, o2, k);
  for (std::size_t i = 0; i < k; ++i) EXPECT_EQ(o2[i], a[i]);

  // classification
  alignas(64) float ff[kBuf]; for (std::size_t i = 0; i < hs::Lanes(df); ++i) ff[i] = static_cast<float>(i);
  ff[0] = std::numeric_limits<float>::quiet_NaN();
  ff[1] = std::numeric_limits<float>::infinity();
  auto vff = hs::Load(df, ff);
  EXPECT_EQ(hs::CountTrue(df, hs::IsNaN(vff)), std::size_t{1});
  EXPECT_EQ(hs::CountTrue(df, hs::IsInf(vff)), std::size_t{1});

  // gather / scatter (byte-offset scaling under the hood)
  alignas(64) float tab[1024]; for (int i = 0; i < 1024; ++i) tab[i] = static_cast<float>(i * 10);
  alignas(64) std::uint32_t ix[kBuf]; for (std::size_t i = 0; i < hs::Lanes(df); ++i) ix[i] = static_cast<std::uint32_t>((i * 7) % 1024);
  alignas(64) float of[kBuf];
  hs::Store(hs::GatherIndex(df, tab, hs::Load(du, ix)), df, of);
  for (std::size_t i = 0; i < hs::Lanes(df); ++i) EXPECT_EQ(of[i], tab[ix[i]]);
  alignas(64) std::int32_t stab[1024] = {};
  alignas(64) std::uint32_t ix2[kBuf]; for (std::size_t i = 0; i < n; ++i) ix2[i] = static_cast<std::uint32_t>(i * 3);
  hs::ScatterIndex(va, di, stab, hs::Load(du, ix2));
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(stab[i * 3], a[i]);

  // compress + compressstore (even lanes)
  alignas(64) std::int32_t mb[kBuf]; for (std::size_t i = 0; i < n; ++i) mb[i] = (i % 2 == 0) ? 1 : 0;
  auto mm = hs::Eq(hs::Load(di, mb), hs::Set(di, 1));
  hs::Store(hs::Compress(va, mm), di, o);
  std::size_t j = 0;
  for (std::size_t i = 0; i < n; ++i) if (i % 2 == 0) EXPECT_EQ(o[j++], a[i]);
  alignas(64) std::int32_t cs[kBuf];
  const std::size_t wrote = hs::CompressStore(va, mm, di, cs);
  EXPECT_EQ(wrote, hs::CountTrue(di, mm));

  // saturating u8
  const hs::ScalableTag<std::uint8_t> du8;
  const std::size_t n8 = hs::Lanes(du8);
  alignas(64) std::uint8_t p[kBuf], q[kBuf], o8[kBuf];
  for (std::size_t i = 0; i < n8; ++i) { p[i] = static_cast<std::uint8_t>(200 + i % 60); q[i] = static_cast<std::uint8_t>(100 + i % 50); }
  hs::Store(hs::SaturatedAdd(hs::Load(du8, p), hs::Load(du8, q)), du8, o8);
  for (std::size_t i = 0; i < n8; ++i) { int s = p[i] + q[i]; if (s > 255) s = 255; EXPECT_EQ(o8[i], static_cast<std::uint8_t>(s)); }
}

// ---- ops added to reach SVE parity: variable shift / reduce-and-or / swizzle / broadcast ----
template <typename T>
void CheckParityOps() {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  alignas(64) T a[kBuf], b[kBuf], o[kBuf];
  for (std::size_t i = 0; i < n; ++i) { a[i] = static_cast<T>(i + 1); b[i] = static_cast<T>(i % 3); }
  auto va = hs::Load(d, a), vb = hs::Load(d, b);

  // variable per-lane shift
  hs::Store(hs::Shl(va, vb), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], static_cast<T>(a[i] << b[i])) << "Shl " << i;
  hs::Store(hs::Shr(va, vb), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], static_cast<T>(a[i] >> b[i])) << "Shr " << i;

  // ReduceAnd / ReduceOr
  T ra = static_cast<T>(~T{0}), ro = 0;
  for (std::size_t i = 0; i < n; ++i) { ra &= a[i]; ro |= a[i]; }
  EXPECT_EQ(hs::ReduceAnd(d, va), ra) << "ReduceAnd";
  EXPECT_EQ(hs::ReduceOr(d, va), ro) << "ReduceOr";

  // InterleaveLower/Upper (whole-vector): lower = [a0,b0,a1,b1,...] for first n/2 of each
  hs::Store(hs::InterleaveLower(va, vb), d, o);
  for (std::size_t k = 0; k < n / 2; ++k) { EXPECT_EQ(o[2 * k], a[k]) << "IntLo a " << k; EXPECT_EQ(o[2 * k + 1], b[k]) << "IntLo b " << k; }
  hs::Store(hs::InterleaveUpper(va, vb), d, o);
  for (std::size_t k = 0; k < n / 2; ++k) { EXPECT_EQ(o[2 * k], a[n / 2 + k]) << "IntHi a " << k; EXPECT_EQ(o[2 * k + 1], b[n / 2 + k]) << "IntHi b " << k; }

  // ConcatEven/Odd(hi=b, lo=a): [a even lanes, b even lanes] / [a odd, b odd]
  hs::Store(hs::ConcatEven(d, vb, va), d, o);
  for (std::size_t k = 0; k < n / 2; ++k) { EXPECT_EQ(o[k], a[2 * k]) << "CatEven lo " << k; EXPECT_EQ(o[n / 2 + k], b[2 * k]) << "CatEven hi " << k; }
  hs::Store(hs::ConcatOdd(d, vb, va), d, o);
  for (std::size_t k = 0; k < n / 2; ++k) { EXPECT_EQ(o[k], a[2 * k + 1]) << "CatOdd lo " << k; EXPECT_EQ(o[n / 2 + k], b[2 * k + 1]) << "CatOdd hi " << k; }

  // OddEven(odd=b, even=a): even positions from a, odd positions from b
  hs::Store(hs::OddEven(vb, va), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], (i % 2 == 0) ? a[i] : b[i]) << "OddEven " << i;

  // Broadcast<2>: splat lane 2
  if (n > 2) {
    hs::Store(hs::Broadcast<2>(va), d, o);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], a[2]) << "Broadcast " << i;
  }
}

TEST(Rvv, ParityOpsAllTypes) {
  CheckParityOps<std::int8_t>();
  CheckParityOps<std::uint8_t>();
  CheckParityOps<std::int16_t>();
  CheckParityOps<std::uint16_t>();
  CheckParityOps<std::int32_t>();
  CheckParityOps<std::uint32_t>();
  CheckParityOps<std::int64_t>();
  CheckParityOps<std::uint64_t>();
}

// MaskedLoadOr: active lanes from p, inactive take `no`.
TEST(Rvv, MaskedLoadOr) {
  const hs::ScalableTag<std::int32_t> d;
  const std::size_t n = hs::Lanes(d);
  alignas(64) std::int32_t src[kBuf], mb[kBuf], o[kBuf];
  for (std::size_t i = 0; i < n; ++i) { src[i] = static_cast<std::int32_t>(i * 2 + 1); mb[i] = (i % 2 == 0) ? 1 : 0; }
  auto m = hs::Eq(hs::Load(d, mb), hs::Set(d, 1));  // even lanes active
  auto no = hs::Set(d, -99);
  hs::Store(hs::MaskedLoadOr(no, m, d, src), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], (i % 2 == 0) ? src[i] : -99) << "MaskedLoadOr " << i;
}

// Round/Floor/Ceil/Trunc (float->float) vs libm; FindFirstTrue/FindLastTrue vs scalar scan.
TEST(Rvv, RoundingAndFindTrue) {
  const hs::ScalableTag<float> d;
  const std::size_t n = hs::Lanes(d);
  alignas(64) float a[kBuf], o[kBuf];
  for (std::size_t i = 0; i < n; ++i) a[i] = (static_cast<float>(i) - static_cast<float>(n) / 2) * 1.375f;  // mix +/- fractional
  auto va = hs::Load(d, a);
  hs::Store(hs::Round(va), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_NEAR(o[i], std::nearbyint(a[i]), 1e-5f) << "Round " << i;
  hs::Store(hs::Floor(va), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_NEAR(o[i], std::floor(a[i]), 1e-5f) << "Floor " << i;
  hs::Store(hs::Ceil(va), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_NEAR(o[i], std::ceil(a[i]), 1e-5f) << "Ceil " << i;
  hs::Store(hs::Trunc(va), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_NEAR(o[i], std::trunc(a[i]), 1e-5f) << "Trunc " << i;

  const hs::ScalableTag<std::int32_t> di;
  const std::size_t ni = hs::Lanes(di);
  alignas(64) std::int32_t iv[kBuf];
  for (std::size_t i = 0; i < ni; ++i) iv[i] = static_cast<std::int32_t>(i);
  // first-true/last-true against a mask "lane in [2, ni-2)"
  auto vi = hs::Load(di, iv);
  auto m = hs::And(hs::Ge(vi, hs::Set(di, 2)), hs::Lt(vi, hs::Set(di, static_cast<std::int32_t>(ni - 2 > 2 ? ni - 2 : ni))));
  std::ptrdiff_t ff = -1, fl = -1;
  for (std::size_t i = 0; i < ni; ++i) { bool act = (std::int32_t)i >= 2 && i < (ni - 2 > 2 ? ni - 2 : ni); if (act) { if (ff < 0) ff = (std::ptrdiff_t)i; fl = (std::ptrdiff_t)i; } }
  EXPECT_EQ(hs::FindFirstTrue(di, m), ff);
  EXPECT_EQ(hs::FindLastTrue(di, m), fl);
  // all-false -> -1
  auto mf = hs::Lt(vi, hs::Set(di, -1));
  EXPECT_EQ(hs::FindFirstTrue(di, mf), std::ptrdiff_t{-1});
  EXPECT_EQ(hs::FindLastTrue(di, mf), std::ptrdiff_t{-1});
}
