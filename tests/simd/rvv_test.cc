// RVV backend test -- the RISC-V Vector backend's op face cross-checked against a plain scalar
// oracle. Like sve_test.cc, this is VL-AGNOSTIC: it queries the runtime Lanes(d) (vsetvlmax) and
// checks every implemented op for every lane type, so the code is correct at whatever VLEN the
// hardware/emulator provides. Cross-compiled (riscv64-linux-gnu-g++ -march=rv64gcv) and run under
// qemu-riscv64-static. See scripts/rvv-qemu-test.sh and docs/architecture/simd-design.md.
//
// Op surface mirrors backend/rvv.h with LMUL=1 native values. Zvbb/Zvbc reclaim ops are gated
// under #if defined(__riscv_zvbb)/__riscv_zvbc (see simd-16). scalable P0-P5 covered here.
#include <gtest/gtest.h>

#include "hyle/simd/vec.h"

#include <bit>
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

TEST(Rvv, Block128Ops) {
  const hs::ScalableTag<std::uint32_t> d;
  const std::size_t n = hs::Lanes(d);
  constexpr std::size_t kLanesPerBlock = 4;
  ASSERT_GE(n, kLanesPerBlock);
  alignas(64) std::uint32_t a[kBuf], b[kBuf], out[kBuf];
  for (std::size_t i = 0; i < n; ++i) {
    a[i] = static_cast<std::uint32_t>(100 + i);
    b[i] = static_cast<std::uint32_t>(1000 + i);
  }
  const auto va = hs::Load(d, a);
  const auto vb = hs::Load(d, b);

  hs::Store(hs::LoadDup128(d, a), d, out);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out[i], a[i % kLanesPerBlock]);

  hs::Store(hs::OddEvenBlocks(va, vb), d, out);
  for (std::size_t i = 0; i < n; ++i) {
    const bool even_block = ((i / kLanesPerBlock) & 1u) == 0;
    EXPECT_EQ(out[i], even_block ? b[i] : a[i]);
  }

  hs::Store(hs::SwapAdjacentBlocks(va), d, out);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t expected = n == kLanesPerBlock ? i : (i ^ kLanesPerBlock);
    EXPECT_EQ(out[i], a[expected]);
  }

  hs::Store(hs::ReverseBlocks(d, va), d, out);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t block = i / kLanesPerBlock;
    const std::size_t within = i % kLanesPerBlock;
    const std::size_t expected = (n / kLanesPerBlock - 1 - block) * kLanesPerBlock + within;
    EXPECT_EQ(out[i], a[expected]);
  }

  hs::Store(hs::Shuffle1032(va), d, out);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out[i], a[i ^ 2u]);
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

  T rs = 0, rmin = a[0], rmax = a[0];
  for (std::size_t i = 0; i < n; ++i) { rs += a[i]; rmin = a[i] < rmin ? a[i] : rmin; rmax = a[i] > rmax ? a[i] : rmax; }
  EXPECT_EQ(hs::ReduceSum(d, va), rs);
  EXPECT_EQ(hs::ReduceMin(d, va), rmin);
  EXPECT_EQ(hs::ReduceMax(d, va), rmax);

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

template <typename T>
void CheckFloatVecFromMaskBits() {
  using D = hs::ScalableTag<T>;
  using DI = hs::RebindToSigned<D>;
  using I = hs::TFromD<DI>;
  const D d;
  const DI di;
  const std::size_t n = hs::Lanes(d);
  alignas(64) T lanes[kBuf];
  alignas(64) I bits[kBuf];
  for (std::size_t i = 0; i < n; ++i) lanes[i] = (i & 1u) == 0 ? T{1} : T{-1};
  const auto mask = hs::Gt(hs::Load(d, lanes), hs::Zero(d));
  hs::Store(hs::BitCast(di, hs::VecFromMask(d, mask)), di, bits);
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(bits[i], (i & 1u) == 0 ? I{-1} : I{0}) << "lane " << i;
  }
}

TEST(Rvv, FloatingVecFromMaskIsBitwiseCanonical) {
  CheckFloatVecFromMaskBits<float>();
  CheckFloatVecFromMaskBits<double>();
}

TEST(Rvv, P4MulModMagnitude) {
  const hs::ScalableTag<float> df;
  const std::size_t nf = hs::Lanes(df);
  alignas(64) float fa[kBuf], fb[kBuf], fc[kBuf], fo[kBuf];
  for (std::size_t i = 0; i < nf; ++i) {
    fa[i] = static_cast<float>(i + 2); fb[i] = 2.0f; fc[i] = 0.5f;
  }
  const auto vfa = hs::Load(df, fa), vfb = hs::Load(df, fb), vfc = hs::Load(df, fc);
  hs::Store(hs::MulSub(vfa, vfb, vfc), df, fo);
  for (std::size_t i = 0; i < nf; ++i) EXPECT_NEAR(fo[i], fa[i] * fb[i] - fc[i], 1e-5f);
  hs::Store(hs::NegMulSub(vfa, vfb, vfc), df, fo);
  for (std::size_t i = 0; i < nf; ++i) EXPECT_NEAR(fo[i], -(fa[i] * fb[i]) - fc[i], 1e-5f);
  hs::Store(hs::MulAddSub(vfa, vfb, vfc), df, fo);
  for (std::size_t i = 0; i < nf; ++i)
    EXPECT_NEAR(fo[i], fa[i] * fb[i] + (i % 2 ? fc[i] : -fc[i]), 1e-5f);
  hs::Store(hs::MulSubAdd(vfa, vfb, vfc), df, fo);
  for (std::size_t i = 0; i < nf; ++i)
    EXPECT_NEAR(fo[i], fa[i] * fb[i] + (i % 2 ? -fc[i] : fc[i]), 1e-5f);

  const hs::ScalableTag<std::int32_t> di;
  const std::size_t ni = hs::Lanes(di);
  alignas(64) std::int32_t ia[kBuf], ib[kBuf], io[kBuf];
  for (std::size_t i = 0; i < ni; ++i) {
    ia[i] = i % 2 ? -5 : std::numeric_limits<std::int32_t>::min();
    ib[i] = i % 2 ? 3 : 4;
  }
  const auto via = hs::Load(di, ia), vib = hs::Load(di, ib);
  hs::Store(hs::MinMagnitude(via, vib), di, io);
  for (std::size_t i = 0; i < ni; ++i) EXPECT_EQ(io[i], ib[i]);
  hs::Store(hs::MaxMagnitude(via, vib), di, io);
  for (std::size_t i = 0; i < ni; ++i) EXPECT_EQ(io[i], ia[i]);

  const hs::ScalableTag<std::uint16_t> du16;
  const hs::ScalableTag<std::uint32_t> du32;
  const std::size_t n16 = hs::Lanes(du16), n32 = hs::Lanes(du32);
  ASSERT_EQ(n16, 2 * n32);
  alignas(64) std::uint16_t ma[kBuf], mb[kBuf]; alignas(64) std::uint32_t mo[kBuf];
  for (std::size_t i = 0; i < n16; ++i) {
    ma[i] = static_cast<std::uint16_t>(i + 2); mb[i] = static_cast<std::uint16_t>(i * 3 + 1);
  }
  const auto vma = hs::Load(du16, ma), vmb = hs::Load(du16, mb);
  hs::Store(hs::MulEven(vma, vmb), du32, mo);
  for (std::size_t i = 0; i < n32; ++i) EXPECT_EQ(mo[i], std::uint32_t{ma[2 * i]} * mb[2 * i]);
  hs::Store(hs::MulOdd(vma, vmb), du32, mo);
  for (std::size_t i = 0; i < n32; ++i) EXPECT_EQ(mo[i], std::uint32_t{ma[2 * i + 1]} * mb[2 * i + 1]);

  alignas(64) std::uint16_t ba[kBuf], bb[kBuf]; alignas(64) float acc[kBuf];
  for (std::size_t i = 0; i < n16; ++i) { ba[i] = i % 2 ? 0xC000u : 0x3F80u; bb[i] = 0x4000u; }
  for (std::size_t i = 0; i < n32; ++i) acc[i] = 0.5f;
  const auto vba = hs::Load(du16, ba), vbb = hs::Load(du16, bb);
  const auto vacc = hs::Load(df, acc);
  hs::Store(hs::MulEvenAdd(df, vba, vbb, vacc), df, fo);
  for (std::size_t i = 0; i < n32; ++i) EXPECT_FLOAT_EQ(fo[i], 2.5f);
  hs::Store(hs::MulOddAdd(df, vba, vbb, vacc), df, fo);
  for (std::size_t i = 0; i < n32; ++i) EXPECT_FLOAT_EQ(fo[i], -3.5f);

  const hs::Divisor<std::int32_t> d3(3);
  hs::Store(hs::Mod(via, d3), di, io);
  for (std::size_t i = 0; i < ni; ++i) EXPECT_EQ(io[i], ia[i] % 3);
  const auto even = hs::Eq(hs::And(via, hs::Set(di, 1)), hs::Zero(di));
  hs::Store(hs::MaskedModOr(even, hs::Set(di, -77), via, d3), di, io);
  for (std::size_t i = 0; i < ni; ++i) EXPECT_EQ(io[i], (ia[i] & 1) == 0 ? ia[i] % 3 : -77);

  alignas(64) std::uint32_t ua[kBuf];
  for (std::size_t i = 0; i < n32; ++i) ua[i] = static_cast<std::uint32_t>(i * 23 + 5);
  const hs::Divisor<std::uint32_t> d7(7);
  hs::Store(hs::Mod(hs::Load(du32, ua), d7), du32, mo);
  for (std::size_t i = 0; i < n32; ++i) EXPECT_EQ(mo[i], ua[i] % 7u);
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
  const std::int32_t hi = static_cast<std::int32_t>(ni - 2 > 2 ? ni - 2 : ni);
  // build the "in [2,hi)" mask from a per-lane flag array (masks are vbool_t, not vectors -> can't
  // combine two compares with a vector And).
  alignas(64) std::int32_t flag[kBuf];
  for (std::size_t i = 0; i < ni; ++i) flag[i] = ((std::int32_t)i >= 2 && (std::int32_t)i < hi) ? 1 : 0;
  auto m = hs::Gt(hs::Load(di, flag), hs::Set(di, 0));
  std::ptrdiff_t ff = -1, fl = -1;
  for (std::size_t i = 0; i < ni; ++i) { if (flag[i]) { if (ff < 0) ff = (std::ptrdiff_t)i; fl = (std::ptrdiff_t)i; } }
  EXPECT_EQ(hs::FindFirstTrue(di, m), ff);
  EXPECT_EQ(hs::FindLastTrue(di, m), fl);
  // all-false -> -1
  alignas(64) std::int32_t zero[kBuf] = {};
  auto mf = hs::Gt(hs::Load(di, zero), hs::Set(di, 0));
  EXPECT_EQ(hs::FindFirstTrue(di, mf), std::ptrdiff_t{-1});
  EXPECT_EQ(hs::FindLastTrue(di, mf), std::ptrdiff_t{-1});
}

// Width-changing Promote/Demote (2x integer). PromoteLowerTo/UpperTo widen the lower/upper half of
// the source; DemoteTo saturates two wide vectors into one narrow (dst 2x lanes).
TEST(Rvv, WidenPromoteDemote) {
  const hs::ScalableTag<std::int16_t> d16;
  const hs::ScalableTag<std::int32_t> d32;
  const std::size_t n16 = hs::Lanes(d16);  // source lanes for promote
  const std::size_t n32 = hs::Lanes(d32);  // dst lanes for promote = n16/2
  alignas(64) std::int16_t s16[kBuf]; alignas(64) std::int32_t o32[kBuf];
  for (std::size_t i = 0; i < n16; ++i) s16[i] = static_cast<std::int16_t>((i * 37) - 300);  // mix +/-
  auto v16 = hs::Load(d16, s16);
  hs::Store(hs::PromoteLowerTo(d32, v16), d32, o32);
  for (std::size_t i = 0; i < n32; ++i) EXPECT_EQ(o32[i], static_cast<std::int32_t>(s16[i])) << "PromLo " << i;
  hs::Store(hs::PromoteUpperTo(d32, v16), d32, o32);
  for (std::size_t i = 0; i < n32; ++i) EXPECT_EQ(o32[i], static_cast<std::int32_t>(s16[n16 / 2 + i])) << "PromHi " << i;

  // DemoteTo i32->i16 (saturating): dst[0:n32]=sat(a), dst[n32:2*n32]=sat(b)
  alignas(64) std::int32_t a32[kBuf], b32[kBuf]; alignas(64) std::int16_t od16[kBuf];
  for (std::size_t i = 0; i < n32; ++i) { a32[i] = static_cast<std::int32_t>(i * 1000 - 40000); b32[i] = static_cast<std::int32_t>(50000 - i * 900); }
  auto va = hs::Load(d32, a32), vb = hs::Load(d32, b32);
  hs::Store(hs::DemoteTo(d16, va, vb), d16, od16);
  auto sat16 = [](std::int32_t x) -> std::int16_t { return static_cast<std::int16_t>(x < -32768 ? -32768 : (x > 32767 ? 32767 : x)); };
  for (std::size_t i = 0; i < n32; ++i) { EXPECT_EQ(od16[i], sat16(a32[i])) << "DemLo " << i; EXPECT_EQ(od16[n32 + i], sat16(b32[i])) << "DemHi " << i; }

  // unsigned promote u8->u16
  const hs::ScalableTag<std::uint8_t> du8; const hs::ScalableTag<std::uint16_t> du16;
  const std::size_t nu8 = hs::Lanes(du8); const std::size_t nu16 = hs::Lanes(du16);
  alignas(64) std::uint8_t su8[kBuf]; alignas(64) std::uint16_t ou16[kBuf];
  for (std::size_t i = 0; i < nu8; ++i) su8[i] = static_cast<std::uint8_t>(i * 5 + 1);
  hs::Store(hs::PromoteLowerTo(du16, hs::Load(du8, su8)), du16, ou16);
  for (std::size_t i = 0; i < nu16; ++i) EXPECT_EQ(ou16[i], static_cast<std::uint16_t>(su8[i])) << "PromLoU " << i;

  // WidenMulPairwiseAdd i16->i32: r[k] = a[2k]*b[2k] + a[2k+1]*b[2k+1]
  alignas(64) std::int16_t wa[kBuf], wb[kBuf]; alignas(64) std::int32_t wr[kBuf];
  for (std::size_t i = 0; i < n16; ++i) { wa[i] = static_cast<std::int16_t>((i * 13) % 500 - 250); wb[i] = static_cast<std::int16_t>((i * 7) % 400 - 200); }
  hs::Store(hs::WidenMulPairwiseAdd(d32, hs::Load(d16, wa), hs::Load(d16, wb)), d32, wr);
  for (std::size_t k = 0; k < n32; ++k) {
    std::int32_t e = static_cast<std::int32_t>(wa[2 * k]) * wb[2 * k] + static_cast<std::int32_t>(wa[2 * k + 1]) * wb[2 * k + 1];
    EXPECT_EQ(wr[k], e) << "WidenMulPairwiseAdd " << k;
  }
}

// P0: instantiate the generic complex/Newton/rotate/128-bit compare layer against RVV.
TEST(Rvv, ScalableGenericP0) {
  const hs::ScalableTag<float> df;
  const std::size_t nf = hs::Lanes(df);
  alignas(64) float a[kBuf], b[kBuf], o[kBuf];
  for (std::size_t i = 0; i < nf; i += 2) {
    a[i] = static_cast<float>(i + 1); a[i + 1] = static_cast<float>(i + 2);
    b[i] = 0.5f; b[i + 1] = -0.25f;
  }
  const auto va = hs::Load(df, a);
  const auto vb = hs::Load(df, b);
  hs::Store(hs::AddSub(va, vb), df, o);
  for (std::size_t i = 0; i < nf; ++i)
    EXPECT_FLOAT_EQ(o[i], (i & 1) ? a[i] + b[i] : a[i] - b[i]);
  hs::Store(hs::DupEven(va), df, o);
  for (std::size_t i = 0; i < nf; ++i) EXPECT_EQ(o[i], a[i & ~std::size_t{1}]);
  hs::Store(hs::DupOdd(va), df, o);
  for (std::size_t i = 0; i < nf; ++i) EXPECT_EQ(o[i], a[(i & ~std::size_t{1}) | 1]);
  hs::Store(hs::Reverse2(df, va), df, o);
  for (std::size_t i = 0; i < nf; ++i) EXPECT_EQ(o[i], a[i ^ 1]);

  hs::Store(hs::MulComplex(va, vb), df, o);
  for (std::size_t i = 0; i < nf; i += 2) {
    EXPECT_NEAR(o[i], a[i] * b[i] - a[i + 1] * b[i + 1], 1e-5f);
    EXPECT_NEAR(o[i + 1], a[i] * b[i + 1] + a[i + 1] * b[i], 1e-5f);
  }

  for (std::size_t i = 0; i < nf; ++i) a[i] = 1.25f + static_cast<float>(i) * 0.125f;
  const auto vx = hs::Load(df, a);
  hs::Store(hs::ReciprocalNewton(vx), df, o);
  for (std::size_t i = 0; i < nf; ++i) EXPECT_NEAR(o[i], 1.0f / a[i], 5e-4f);
  hs::Store(hs::ReciprocalSqrtNewton(vx), df, o);
  for (std::size_t i = 0; i < nf; ++i) EXPECT_NEAR(o[i], 1.0f / std::sqrt(a[i]), 5e-4f);

  const hs::ScalableTag<std::uint32_t> du32;
  const std::size_t n32 = hs::Lanes(du32);
  alignas(64) std::uint32_t u32[kBuf], r32[kBuf];
  for (std::size_t i = 0; i < n32; ++i) u32[i] = 0x81234567u + static_cast<std::uint32_t>(i);
  const auto vu32 = hs::Load(du32, u32);
  hs::Store(hs::Rol<5>(vu32), du32, r32);
  for (std::size_t i = 0; i < n32; ++i) EXPECT_EQ(r32[i], (u32[i] << 5) | (u32[i] >> 27));
  hs::Store(hs::Ror<7>(vu32), du32, r32);
  for (std::size_t i = 0; i < n32; ++i) EXPECT_EQ(r32[i], (u32[i] >> 7) | (u32[i] << 25));
  for (std::size_t i = 0; i < n32; ++i) u32[i] = (i & 1) ? 1u : 0x80000000u;
  hs::Store(hs::VecFromMask(du32, hs::MaskFromVec(hs::Load(du32, u32))), du32, r32);
  for (std::size_t i = 0; i < n32; ++i) EXPECT_EQ(r32[i], (i & 1) ? 0u : ~0u);

  const hs::ScalableTag<std::uint64_t> du64;
  const std::size_t n64 = hs::Lanes(du64);
  alignas(64) std::uint64_t x[kBuf], y[kBuf], m[kBuf];
  for (std::size_t i = 0; i < n64; i += 2) {
    const std::size_t p = i / 2;
    x[i] = 5; x[i + 1] = p % 3 == 1 ? 2 : 1;
    y[i] = p % 3 == 0 ? 6 : 5; y[i + 1] = p % 3 == 1 ? 1 : 1;
  }
  const auto vx128 = hs::Load(du64, x);
  const auto vy128 = hs::Load(du64, y);
  hs::Store(hs::VecFromMask(du64, hs::Lt128(du64, vx128, vy128)), du64, m);
  for (std::size_t i = 0; i < n64; i += 2) {
    const bool expected = (i / 2) % 3 == 0;
    EXPECT_EQ(m[i], expected ? ~std::uint64_t{0} : 0); EXPECT_EQ(m[i + 1], m[i]);
  }
  hs::Store(hs::VecFromMask(du64, hs::Eq128(du64, vx128, vy128)), du64, m);
  for (std::size_t i = 0; i < n64; i += 2) {
    const bool expected = (i / 2) % 3 == 2;
    EXPECT_EQ(m[i], expected ? ~std::uint64_t{0} : 0); EXPECT_EQ(m[i + 1], m[i]);
  }
}

template <typename T>
void CheckScalableP1() {
  using D = hs::ScalableTag<T>;
  using DU = hs::RebindToUnsigned<D>;
  using U = hs::TFromD<DU>;
  const D d; const DU du;
  const std::size_t n = hs::Lanes(d);
  alignas(64) T a[kBuf], b[kBuf], o[kBuf];
  alignas(64) U idx[kBuf];
  for (std::size_t i = 0; i < n; ++i) {
    a[i] = static_cast<T>((i * 3 + 1) % 97);
    b[i] = static_cast<T>((i * 5 + 2) % 89);
    idx[i] = static_cast<U>(n - 1 - i);
  }
  const auto va = hs::Load(d, a), vb = hs::Load(d, b);
  hs::Store(hs::TableLookupLanes(va, hs::IndicesFromVec(d, hs::Load(du, idx))), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], a[n - 1 - i]) << "Table " << i;

  const std::size_t amt = n > 2 ? 2 : 1;
  hs::Store(hs::SlideUpLanes(d, va, amt), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], i < amt ? T{0} : a[i - amt]) << "Up " << i;
  hs::Store(hs::SlideDownLanes(d, va, amt), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], i + amt < n ? a[i + amt] : T{0}) << "Down " << i;
  hs::Store(hs::Slide1Up(d, va), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], i == 0 ? T{0} : a[i - 1]);
  hs::Store(hs::Slide1Down(d, va), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], i + 1 < n ? a[i + 1] : T{0});

  hs::Store(hs::InterleaveEven(va, vb), d, o);
  for (std::size_t i = 0; i < n / 2; ++i) {
    EXPECT_EQ(o[2 * i], a[2 * i]); EXPECT_EQ(o[2 * i + 1], b[2 * i]);
  }
  hs::Store(hs::InterleaveOdd(va, vb), d, o);
  for (std::size_t i = 0; i < n / 2; ++i) {
    EXPECT_EQ(o[2 * i], a[2 * i + 1]); EXPECT_EQ(o[2 * i + 1], b[2 * i + 1]);
  }
  if (n >= 4) {
    hs::Store(hs::Reverse4(d, va), d, o);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], a[i ^ 3]) << "Reverse4 " << i;
  }
  if (n >= 8) {
    hs::Store(hs::Reverse8(d, va), d, o);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], a[i ^ 7]) << "Reverse8 " << i;
  }
  if constexpr (std::is_integral_v<T>) {
    hs::Store(hs::ReverseLaneBytes(va), d, o);
    using UT = std::make_unsigned_t<T>;
    for (std::size_t i = 0; i < n; ++i) {
      UT x = static_cast<UT>(a[i]), expected = 0;
      for (std::size_t byte = 0; byte < sizeof(T); ++byte) {
        expected = static_cast<UT>((expected << 8) | (x & UT{0xFF})); x >>= 8;
      }
      EXPECT_EQ(o[i], static_cast<T>(expected)) << "ReverseLaneBytes " << i;
    }
  }

  alignas(64) T aos[4 * kBuf], round[4 * kBuf], s0[kBuf], s1[kBuf], s2[kBuf], s3[kBuf];
  for (std::size_t i = 0; i < 4 * n; ++i) aos[i] = static_cast<T>((i * 7 + 3) % 101);
  auto v0 = hs::Zero(d), v1 = hs::Zero(d), v2 = hs::Zero(d), v3 = hs::Zero(d);
  hs::LoadInterleaved2(d, aos, v0, v1);
  hs::Store(v0, d, s0); hs::Store(v1, d, s1);
  for (std::size_t i = 0; i < n; ++i) { EXPECT_EQ(s0[i], aos[2 * i]); EXPECT_EQ(s1[i], aos[2 * i + 1]); }
  hs::StoreInterleaved2(v0, v1, d, round);
  for (std::size_t i = 0; i < 2 * n; ++i) EXPECT_EQ(round[i], aos[i]);
  hs::LoadInterleaved3(d, aos, v0, v1, v2);
  hs::Store(v0, d, s0); hs::Store(v1, d, s1); hs::Store(v2, d, s2);
  for (std::size_t i = 0; i < n; ++i) { EXPECT_EQ(s0[i], aos[3 * i]); EXPECT_EQ(s1[i], aos[3 * i + 1]); EXPECT_EQ(s2[i], aos[3 * i + 2]); }
  hs::StoreInterleaved3(v0, v1, v2, d, round);
  for (std::size_t i = 0; i < 3 * n; ++i) EXPECT_EQ(round[i], aos[i]);
  hs::LoadInterleaved4(d, aos, v0, v1, v2, v3);
  hs::Store(v0, d, s0); hs::Store(v1, d, s1); hs::Store(v2, d, s2); hs::Store(v3, d, s3);
  for (std::size_t i = 0; i < n; ++i) { EXPECT_EQ(s0[i], aos[4 * i]); EXPECT_EQ(s1[i], aos[4 * i + 1]); EXPECT_EQ(s2[i], aos[4 * i + 2]); EXPECT_EQ(s3[i], aos[4 * i + 3]); }
  hs::StoreInterleaved4(v0, v1, v2, v3, d, round);
  for (std::size_t i = 0; i < 4 * n; ++i) EXPECT_EQ(round[i], aos[i]);
}

TEST(Rvv, ScalableP1Swizzle) {
  CheckScalableP1<float>(); CheckScalableP1<double>();
  CheckScalableP1<std::int8_t>(); CheckScalableP1<std::uint8_t>();
  CheckScalableP1<std::int16_t>(); CheckScalableP1<std::uint16_t>();
  CheckScalableP1<std::int32_t>(); CheckScalableP1<std::uint32_t>();
  CheckScalableP1<std::int64_t>(); CheckScalableP1<std::uint64_t>();

  const hs::ScalableTag<std::uint8_t> d;
  const std::size_t n = hs::Lanes(d);
  alignas(64) std::uint8_t table[kBuf], idx[kBuf], out[kBuf];
  for (std::size_t i = 0; i < n; ++i) { table[i] = static_cast<std::uint8_t>(i); idx[i] = static_cast<std::uint8_t>(15 - (i & 15)); }
  const auto vt = hs::Load(d, table);
  hs::Store(hs::TableLookupBytes(vt, hs::Load(d, idx)), d, out);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out[i], static_cast<std::uint8_t>((i & ~std::size_t{15}) + idx[i]));
  for (std::size_t i = 0; i < n; ++i) if (i % 3 == 0) idx[i] |= 0x80;
  hs::Store(hs::TableLookupBytesOr0(vt, hs::Load(d, idx)), d, out);
  for (std::size_t i = 0; i < n; ++i) {
    const auto expected = (idx[i] & 0x80) ? std::uint8_t{0} : static_cast<std::uint8_t>((i & ~std::size_t{15}) + (idx[i] & 15));
    EXPECT_EQ(out[i], expected);
  }
}

TEST(Rvv, ScalableP2WidenQuantize) {
  const hs::ScalableTag<std::uint8_t> du8;
  const hs::ScalableTag<std::int8_t> di8;
  const hs::ScalableTag<std::int16_t> di16;
  const hs::ScalableTag<std::uint32_t> du32;
  const hs::ScalableTag<std::int32_t> di32;
  const std::size_t n8 = hs::Lanes(du8), n16 = hs::Lanes(di16), n32 = hs::Lanes(di32);
  ASSERT_EQ(n8, 2 * n16); ASSERT_EQ(n16, 2 * n32);

  alignas(64) std::int16_t s16[kBuf]; alignas(64) std::int32_t o32[kBuf];
  for (std::size_t i = 0; i < n16; ++i) s16[i] = static_cast<std::int16_t>(i * 17 - 300);
  const auto v16 = hs::Load(di16, s16);
  hs::Store(hs::PromoteEvenTo(di32, v16), di32, o32);
  for (std::size_t i = 0; i < n32; ++i) EXPECT_EQ(o32[i], s16[2 * i]);
  hs::Store(hs::PromoteOddTo(di32, v16), di32, o32);
  for (std::size_t i = 0; i < n32; ++i) EXPECT_EQ(o32[i], s16[2 * i + 1]);

  const hs::ScalableTag<float> df32; const hs::ScalableTag<double> df64;
  const std::size_t nf32 = hs::Lanes(df32), nf64 = hs::Lanes(df64);
  ASSERT_EQ(nf32, 2 * nf64);
  alignas(64) float f32[kBuf]; alignas(64) double f64[kBuf];
  for (std::size_t i = 0; i < nf32; ++i) f32[i] = static_cast<float>(i) * 1.25f - 7.0f;
  const auto vf32 = hs::Load(df32, f32);
  hs::Store(hs::PromoteEvenTo(df64, vf32), df64, f64);
  for (std::size_t i = 0; i < nf64; ++i) EXPECT_EQ(f64[i], static_cast<double>(f32[2 * i]));
  hs::Store(hs::PromoteOddTo(df64, vf32), df64, f64);
  for (std::size_t i = 0; i < nf64; ++i) EXPECT_EQ(f64[i], static_cast<double>(f32[2 * i + 1]));

  using DTrunc8 = hs::Rebind<std::uint8_t, hs::ScalableTag<std::uint32_t>>;
  const DTrunc8 dt8;
  ASSERT_EQ(hs::Lanes(dt8), n32);
  alignas(64) std::uint32_t u32[kBuf]; alignas(64) std::uint8_t o8[kBuf];
  for (std::size_t i = 0; i < n32; ++i) u32[i] = 0xA1B2C300u + static_cast<std::uint32_t>(i * 19 + 7);
  const auto vu32 = hs::Load(du32, u32);
  hs::Store(hs::TruncateTo(dt8, vu32), dt8, o8);
  for (std::size_t i = 0; i < n32; ++i) EXPECT_EQ(o8[i], static_cast<std::uint8_t>(u32[i]));
  for (std::size_t i = 0; i < n32; ++i) u32[i] = static_cast<std::uint32_t>((i * 13) & 255);
  hs::Store(hs::U8FromU32(hs::Load(du32, u32)), dt8, o8);
  for (std::size_t i = 0; i < n32; ++i) EXPECT_EQ(o8[i], static_cast<std::uint8_t>(u32[i]));

  const hs::ScalableTag<std::uint64_t> du64; const std::size_t n64 = hs::Lanes(du64);
  using DTrunc16 = hs::Rebind<std::uint16_t, hs::ScalableTag<std::uint64_t>>;
  using DTrunc8From64 = hs::Rebind<std::uint8_t, hs::ScalableTag<std::uint64_t>>;
  const DTrunc16 dt16; const DTrunc8From64 dt8x;
  alignas(64) std::uint64_t u64[kBuf]; alignas(64) std::uint16_t o16u[kBuf];
  for (std::size_t i = 0; i < n64; ++i) u64[i] = 0xA1B2C3D4E5F60700ull + i * 23;
  const auto vu64 = hs::Load(du64, u64);
  hs::Store(hs::TruncateTo(dt16, vu64), dt16, o16u); hs::Store(hs::TruncateTo(dt8x, vu64), dt8x, o8);
  for (std::size_t i = 0; i < n64; ++i) { EXPECT_EQ(o16u[i], static_cast<std::uint16_t>(u64[i])); EXPECT_EQ(o8[i], static_cast<std::uint8_t>(u64[i])); }

  alignas(64) std::uint8_t a8[kBuf]; alignas(64) std::int8_t b8[kBuf];
  alignas(64) std::int16_t o16[kBuf];
  for (std::size_t i = 0; i < n8; ++i) { a8[i] = 255; b8[i] = static_cast<std::int8_t>(((i / 2) & 1) ? -128 : 127); }
  hs::Store(hs::SatWidenMulPairwiseAdd(di16, hs::Load(du8, a8), hs::Load(di8, b8)), di16, o16);
  for (std::size_t i = 0; i < n16; ++i) {
    int x = static_cast<int>(a8[2 * i]) * b8[2 * i] + static_cast<int>(a8[2 * i + 1]) * b8[2 * i + 1];
    x = x < -32768 ? -32768 : (x > 32767 ? 32767 : x);
    EXPECT_EQ(o16[i], static_cast<std::int16_t>(x));
  }

  alignas(64) std::int16_t ma[kBuf], mb[kBuf];
  for (std::size_t i = 0; i < n16; ++i) { ma[i] = static_cast<std::int16_t>(i * 3 - 100); mb[i] = static_cast<std::int16_t>(i - 50); }
  auto sum0 = hs::Zero(di32), sum1 = hs::Zero(di32);
  sum0 = hs::ReorderWidenMulAccumulate(di32, hs::Load(di16, ma), hs::Load(di16, mb), sum0, sum1);
  hs::Store(hs::RearrangeToOddPlusEven(sum0, sum1), di32, o32);
  for (std::size_t i = 0; i < n32; ++i)
    EXPECT_EQ(o32[i], static_cast<std::int32_t>(ma[2 * i]) * mb[2 * i] + static_cast<std::int32_t>(ma[2 * i + 1]) * mb[2 * i + 1]);

  auto quad = hs::Set(di32, 7);
  hs::Store(hs::SumOfMulQuadAccumulate(di32, hs::Load(di8, b8), hs::Load(di8, b8), quad), di32, o32);
  for (std::size_t i = 0; i < n32; ++i) {
    int expected = 7; for (std::size_t k = 0; k < 4; ++k) expected += static_cast<int>(b8[4 * i + k]) * b8[4 * i + k];
    EXPECT_EQ(o32[i], expected);
  }

  using DS2 = hs::FixedTag<std::uint16_t, hs::ScalableTag<std::uint8_t>::kMaxLanes / 2>;
  using DS4 = hs::FixedTag<std::uint32_t, hs::ScalableTag<std::uint8_t>::kMaxLanes / 4>;
  using DS8 = hs::FixedTag<std::uint64_t, hs::ScalableTag<std::uint8_t>::kMaxLanes / 8>;
  const DS2 ds2; const DS4 ds4; const DS8 ds8;
  alignas(64) std::uint16_t s2[kBuf]; alignas(64) std::uint32_t s4[kBuf]; alignas(64) std::uint64_t s8[kBuf];
  for (std::size_t i = 0; i < n8; ++i) a8[i] = static_cast<std::uint8_t>(i % 29);
  const auto bytes = hs::Load(du8, a8);
  hs::Store(hs::SumsOf2(bytes), ds2, s2); hs::Store(hs::SumsOf4(bytes), ds4, s4); hs::Store(hs::SumsOf8(bytes), ds8, s8);
  for (std::size_t i = 0; i < n8 / 2; ++i) EXPECT_EQ(s2[i], static_cast<std::uint16_t>(a8[2 * i] + a8[2 * i + 1]));
  for (std::size_t i = 0; i < n8 / 4; ++i) EXPECT_EQ(s4[i], static_cast<std::uint32_t>(a8[4 * i] + a8[4 * i + 1] + a8[4 * i + 2] + a8[4 * i + 3]));
  for (std::size_t i = 0; i < n8 / 8; ++i) { std::uint64_t expected = 0; for (std::size_t k = 0; k < 8; ++k) expected += a8[8 * i + k]; EXPECT_EQ(s8[i], expected); }

  for (std::size_t i = 0; i < n32; ++i) o32[i] = i < 3 ? static_cast<std::int32_t>(i + 2) : 1;
  std::int32_t product = 1; for (std::size_t i = 0; i < n32; ++i) product *= o32[i];
  EXPECT_EQ(hs::ReduceMul(di32, hs::Load(di32, o32)), product);

  alignas(64) std::int16_t qa[kBuf], qb[kBuf];
  const auto min16 = std::numeric_limits<std::int16_t>::min();
  for (std::size_t i = 0; i < n16; ++i) { qa[i] = i % 5 == 0 ? min16 : static_cast<std::int16_t>(i * 911 - 12000); qb[i] = static_cast<std::int16_t>(18000 - i * 431); }
  qa[0] = min16; qb[0] = min16;
  const auto vqa = hs::Load(di16, qa);
  hs::Store(hs::SaturatedNeg(vqa), di16, o16);
  for (std::size_t i = 0; i < n16; ++i) EXPECT_EQ(o16[i], qa[i] == min16 ? std::numeric_limits<std::int16_t>::max() : static_cast<std::int16_t>(-qa[i]));
  hs::Store(hs::SaturatedAbs(vqa), di16, o16);
  for (std::size_t i = 0; i < n16; ++i) EXPECT_EQ(o16[i], qa[i] == min16 ? std::numeric_limits<std::int16_t>::max() : static_cast<std::int16_t>(qa[i] < 0 ? -qa[i] : qa[i]));
  hs::Store(hs::MulFixedPoint15(vqa, hs::Load(di16, qb)), di16, o16);
  for (std::size_t i = 0; i < n16; ++i) EXPECT_EQ(o16[i], static_cast<std::int16_t>((static_cast<int>(qa[i]) * qb[i] + 0x4000) >> 15));

  alignas(64) std::int32_t da[kBuf], db[kBuf]; alignas(64) std::int16_t demoted[kBuf];
  for (std::size_t i = 0; i < n32; ++i) { da[i] = static_cast<std::int32_t>(i * 1000 - 40000); db[i] = static_cast<std::int32_t>(50000 - i * 900); }
  hs::Store(hs::ReorderDemote2To(di16, hs::Load(di32, da), hs::Load(di32, db)), di16, demoted);
  auto sat16 = [](std::int32_t x) { return static_cast<std::int16_t>(x < -32768 ? -32768 : (x > 32767 ? 32767 : x)); };
  for (std::size_t i = 0; i < n32; ++i) { EXPECT_EQ(demoted[i], sat16(da[i])); EXPECT_EQ(demoted[n32 + i], sat16(db[i])); }

  alignas(64) std::uint32_t ua[kBuf], ub[kBuf];
  for (std::size_t i = 0; i < n32; ++i) { ua[i] = static_cast<std::uint32_t>(i * 20000); ub[i] = static_cast<std::uint32_t>(70000 - i * 1000); }
  hs::Store(hs::ReorderDemote2To(di16, hs::Load(du32, ua), hs::Load(du32, ub)), di16, demoted);
  for (std::size_t i = 0; i < n32; ++i) { EXPECT_EQ(demoted[i], static_cast<std::int16_t>(ua[i] > 32767 ? 32767 : ua[i])); EXPECT_EQ(demoted[n32 + i], static_cast<std::int16_t>(ub[i] > 32767 ? 32767 : ub[i])); }
  const hs::ScalableTag<std::uint16_t> du16;
  alignas(64) std::uint16_t demoted_u[kBuf];
  hs::Store(hs::ReorderDemote2To(du16, hs::Load(di32, da), hs::Load(di32, db)), du16, demoted_u);
  auto satu16 = [](std::int32_t x) { return static_cast<std::uint16_t>(x < 0 ? 0 : (x > 65535 ? 65535 : x)); };
  for (std::size_t i = 0; i < n32; ++i) { EXPECT_EQ(demoted_u[i], satu16(da[i])); EXPECT_EQ(demoted_u[n32 + i], satu16(db[i])); }
}

TEST(Rvv, PartialTagSemantics) {
  const hs::FixedTag<std::int32_t, 2> d;
  ASSERT_EQ(hs::Lanes(d), std::size_t{2});
  alignas(64) std::int32_t src[kBuf] = {3, 5};
  alignas(64) std::int32_t out[kBuf];
  for (auto& x : out) x = -99;
  const auto v = hs::Load(d, src);
  hs::Store(v, d, out);
  EXPECT_EQ(out[0], 3); EXPECT_EQ(out[1], 5); EXPECT_EQ(out[2], -99);
  EXPECT_EQ(hs::ReduceSum(d, v), 8);
  EXPECT_EQ(hs::CountTrue(d, hs::Eq(v, v)), std::size_t{2});
  EXPECT_EQ(hs::FindLastTrue(d, hs::Eq(v, v)), std::ptrdiff_t{1});
  hs::Store(hs::Reverse(d, v), d, out);
  EXPECT_EQ(out[0], 5); EXPECT_EQ(out[1], 3);
  hs::Store(hs::Slide1Down(d, v), d, out);
  EXPECT_EQ(out[0], 5); EXPECT_EQ(out[1], 0);
  alignas(64) std::int32_t src2[kBuf] = {7, 11};
  const auto v2 = hs::Load(d, src2);
  hs::Store(hs::InterleaveUpper(d, v, v2), d, out);
  EXPECT_EQ(out[0], 5); EXPECT_EQ(out[1], 11);
  hs::Store(hs::ConcatEven(d, v2, v), d, out);
  EXPECT_EQ(out[0], 3); EXPECT_EQ(out[1], 7);
  hs::Store(hs::ConcatOdd(d, v2, v), d, out);
  EXPECT_EQ(out[0], 5); EXPECT_EQ(out[1], 11);

  const hs::FixedTag<std::int8_t, 4> di8;
  const hs::FixedTag<std::int16_t, 2> di16;
  alignas(64) std::int8_t narrow[kBuf] = {1, 2, 3, 4}, narrow_out[kBuf];
  alignas(64) std::int16_t wide_a[kBuf] = {-200, 300}, wide_b[kBuf] = {400, -500}, wide_out[kBuf];
  const auto vn = hs::Load(di8, narrow);
  hs::Store(hs::PromoteUpperTo(di16, vn), di16, wide_out);
  EXPECT_EQ(wide_out[0], 3); EXPECT_EQ(wide_out[1], 4);
  hs::Store(hs::PromoteEvenTo(di16, vn), di16, wide_out);
  EXPECT_EQ(wide_out[0], 1); EXPECT_EQ(wide_out[1], 3);
  hs::Store(hs::PromoteOddTo(di16, vn), di16, wide_out);
  EXPECT_EQ(wide_out[0], 2); EXPECT_EQ(wide_out[1], 4);
  hs::Store(hs::ReorderDemote2To(di8, hs::Load(di16, wide_a), hs::Load(di16, wide_b)), di8, narrow_out);
  EXPECT_EQ(narrow_out[0], -128); EXPECT_EQ(narrow_out[1], 127);
  EXPECT_EQ(narrow_out[2], 127); EXPECT_EQ(narrow_out[3], -128);

  for (auto& x : out) x = -99;
  hs::StoreN(v, d, out, 4);
  EXPECT_EQ(out[0], 3); EXPECT_EQ(out[1], 5); EXPECT_EQ(out[2], -99);
  using DR = hs::Rebind<std::uint32_t, hs::FixedTag<std::uint8_t, 2>>;
  const DR dr;
  EXPECT_EQ(hs::Lanes(dr), std::size_t{2});

  const hs::ScalableTag<std::int32_t> df;
  const std::size_t n = hs::Lanes(df), k = n > 3 ? 3 : n;
  for (std::size_t i = 0; i < n; ++i) { src[i] = static_cast<std::int32_t>(i + 1); out[i] = -1; }
  hs::Store(hs::LoadN(df, src, k), df, out);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out[i], i < k ? src[i] : 0);
  alignas(64) std::int32_t flags[kBuf];
  for (std::size_t i = 0; i < n; ++i) flags[i] = (i & 1) ? 0 : 1;
  const auto mask = hs::Eq(hs::Load(df, flags), hs::Set(df, 1));
  hs::Store(hs::MaskedLoad(mask, df, src), df, out);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out[i], (i & 1) ? 0 : src[i]);

  const hs::ScalableTag<std::uint8_t> d8;
  const std::size_t n8 = hs::Lanes(d8);
  alignas(64) std::uint8_t flags8[kBuf] = {};
  flags8[n8 - 1] = 1;
  const auto last = hs::Eq(hs::Load(d8, flags8), hs::Set(d8, 1));
  EXPECT_EQ(hs::FindLastTrue(d8, last), static_cast<std::ptrdiff_t>(n8 - 1));
  EXPECT_EQ(hs::CountTrue(d8, hs::FirstN(d8, n8)), n8);
}
TEST(Rvv, P5TrivialAndMask) {
  // A1 BroadcastSignBit
  {
    const hs::ScalableTag<std::int32_t> d;
    const std::size_t n = hs::Lanes(d);
    alignas(64) std::int32_t a[kBuf], o[kBuf];
    for (std::size_t i = 0; i < n; ++i) a[i] = (i % 3 == 0) ? -5 : (i % 3 == 1) ? 0 : 7;
    hs::Store(hs::BroadcastSignBit(hs::Load(d, a)), d, o);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], a[i] < 0 ? -1 : 0) << "i32 " << i;
  }
  {
    const hs::ScalableTag<std::int8_t> d;
    const std::size_t n = hs::Lanes(d);
    alignas(64) std::int8_t a[kBuf], o[kBuf];
    for (std::size_t i = 0; i < n; ++i) a[i] = static_cast<std::int8_t>((i & 1) ? -1 : 42);
    hs::Store(hs::BroadcastSignBit(hs::Load(d, a)), d, o);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], a[i] < 0 ? -1 : 0) << "i8 " << i;
  }
  {
    const hs::ScalableTag<std::int64_t> d;
    const std::size_t n = hs::Lanes(d);
    alignas(64) std::int64_t a[kBuf], o[kBuf];
    for (std::size_t i = 0; i < n; ++i) a[i] = (i == 0) ? std::numeric_limits<std::int64_t>::min()
                                                          : static_cast<std::int64_t>(i);
    hs::Store(hs::BroadcastSignBit(hs::Load(d, a)), d, o);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], a[i] < 0 ? -1 : 0) << "i64 " << i;
  }

  // A2 ExclusiveNeither truth table via two flag arrays
  {
    const hs::ScalableTag<std::int32_t> d;
    const std::size_t n = hs::Lanes(d);
    alignas(64) std::int32_t fa[kBuf], fb[kBuf], out[kBuf];
    for (std::size_t i = 0; i < n; ++i) {
      fa[i] = (i % 4 == 0 || i % 4 == 1) ? 1 : 0;  // T T F F
      fb[i] = (i % 4 == 0 || i % 4 == 2) ? 1 : 0;  // T F T F
    }
    auto ma = hs::Gt(hs::Load(d, fa), hs::Zero(d));
    auto mb = hs::Gt(hs::Load(d, fb), hs::Zero(d));
    hs::Store(hs::VecFromMask(d, hs::ExclusiveNeither(d, ma, mb)), d, out);
    for (std::size_t i = 0; i < n; ++i) {
      const bool expected = !fa[i] && !fb[i];  // only FF -> T
      EXPECT_EQ(out[i] != 0, expected) << "ExclusiveNeither " << i;
    }
  }

  // A3 IsEitherNaN
  {
    const hs::ScalableTag<float> d;
    const std::size_t n = hs::Lanes(d);
    alignas(64) float a[kBuf], b[kBuf];
    alignas(64) std::int32_t out[kBuf];
    const hs::ScalableTag<std::int32_t> di;
    for (std::size_t i = 0; i < n; ++i) {
      a[i] = static_cast<float>(i);
      b[i] = static_cast<float>(i + 1);
    }
    if (n > 0) a[0] = std::numeric_limits<float>::quiet_NaN();
    if (n > 1) b[1] = std::numeric_limits<float>::quiet_NaN();
    if (n > 2) a[2] = std::numeric_limits<float>::infinity();
    hs::Store(hs::VecFromMask(di, hs::IsEitherNaN(hs::Load(d, a), hs::Load(d, b))), di, out);
    for (std::size_t i = 0; i < n; ++i) {
      const bool expected = std::isnan(a[i]) || std::isnan(b[i]);
      EXPECT_EQ(out[i] != 0, expected) << "IsEitherNaN " << i;
    }
  }

  // A4 MulRound
  {
    const hs::ScalableTag<float> d;
    const std::size_t n = hs::Lanes(d);
    alignas(64) float a[kBuf], b[kBuf], o[kBuf];
    for (std::size_t i = 0; i < n; ++i) {
      a[i] = (i % 3 == 0) ? 2.4f : (i % 3 == 1) ? 2.5f : 3.5f;
      b[i] = 1.0f;
    }
    hs::Store(hs::MulRound(hs::Load(d, a), hs::Load(d, b)), d, o);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_NEAR(o[i], std::nearbyint(a[i] * b[i]), 1e-5f) << "MulRound " << i;
  }

  // A5 GetExponent
  {
    const hs::ScalableTag<float> d;
    const std::size_t n = hs::Lanes(d);
    alignas(64) float a[kBuf], o[kBuf];
    for (std::size_t i = 0; i < n; ++i) {
      if (i % 3 == 0) a[i] = 1.0f;
      else if (i % 3 == 1) a[i] = 8.0f;
      else a[i] = 0.5f;
    }
    hs::Store(hs::GetExponent(hs::Load(d, a)), d, o);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_NEAR(o[i], std::logb(a[i]), 1e-5f) << "GetExponent f32 " << i;
  }
  {
    const hs::ScalableTag<double> d;
    const std::size_t n = hs::Lanes(d);
    alignas(64) double a[kBuf], o[kBuf];
    for (std::size_t i = 0; i < n; ++i) a[i] = (i & 1) ? 16.0 : 0.25;
    hs::Store(hs::GetExponent(hs::Load(d, a)), d, o);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_NEAR(o[i], std::logb(a[i]), 1e-12) << "GetExponent f64 " << i;
  }

  // A6 BroadcastLane
  {
    const hs::ScalableTag<std::int32_t> d;
    const std::size_t n = hs::Lanes(d);
    ASSERT_GE(n, std::size_t{2});
    alignas(64) std::int32_t a[kBuf], o[kBuf];
    for (std::size_t i = 0; i < n; ++i) a[i] = static_cast<std::int32_t>(i * 10 + 3);
    hs::Store(hs::BroadcastLane<0>(hs::Load(d, a)), d, o);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], a[0]);
    hs::Store(hs::BroadcastLane<1>(hs::Load(d, a)), d, o);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], a[1]);
  }

  // B1-B4 Set*First family: first true at lane 0 / mid / last / none / all
  {
    const hs::ScalableTag<std::int32_t> d;
    const std::size_t n = hs::Lanes(d);
    alignas(64) std::int32_t flag[kBuf], out[kBuf];

    auto check_set = [&](std::size_t first_true, bool has_true) {
      for (std::size_t i = 0; i < n; ++i) flag[i] = 0;
      if (has_true) {
        flag[first_true] = 1;
        if (first_true + 2 < n) flag[first_true + 2] = 1;  // extra true after first
      }
      auto m = hs::Gt(hs::Load(d, flag), hs::Zero(d));

      hs::Store(hs::VecFromMask(d, hs::SetBeforeFirst(d, m)), d, out);
      for (std::size_t i = 0; i < n; ++i) {
        const bool expected = has_true ? (i < first_true) : true;
        EXPECT_EQ(out[i] != 0, expected) << "Before f=" << first_true << " i=" << i;
      }
      hs::Store(hs::VecFromMask(d, hs::SetAtOrBeforeFirst(d, m)), d, out);
      for (std::size_t i = 0; i < n; ++i) {
        const bool expected = has_true ? (i <= first_true) : true;
        EXPECT_EQ(out[i] != 0, expected) << "AtOrBefore f=" << first_true << " i=" << i;
      }
      hs::Store(hs::VecFromMask(d, hs::SetOnlyFirst(d, m)), d, out);
      for (std::size_t i = 0; i < n; ++i) {
        const bool expected = has_true && (i == first_true);
        EXPECT_EQ(out[i] != 0, expected) << "Only f=" << first_true << " i=" << i;
      }
      hs::Store(hs::VecFromMask(d, hs::SetAtOrAfterFirst(d, m)), d, out);
      for (std::size_t i = 0; i < n; ++i) {
        const bool expected = has_true ? (i >= first_true) : false;
        EXPECT_EQ(out[i] != 0, expected) << "AtOrAfter f=" << first_true << " i=" << i;
      }
    };
    check_set(0, true);
    if (n > 2) check_set(n / 2, true);
    check_set(n - 1, true);
    check_set(0, false);  // no true
    // all true
    for (std::size_t i = 0; i < n; ++i) flag[i] = 1;
    auto mall = hs::Gt(hs::Load(d, flag), hs::Zero(d));
    hs::Store(hs::VecFromMask(d, hs::SetOnlyFirst(d, mall)), d, out);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out[i] != 0, i == 0) << "Only all-true " << i;
  }

  // B5-B6 FindKnownFirst/LastTrue
  {
    const hs::ScalableTag<std::int32_t> d;
    const std::size_t n = hs::Lanes(d);
    alignas(64) std::int32_t flag[kBuf];
    for (std::size_t i = 0; i < n; ++i) flag[i] = 0;
    flag[0] = 1;
    if (n > 1) flag[n - 1] = 1;
    if (n > 3) flag[2] = 1;
    auto m = hs::Gt(hs::Load(d, flag), hs::Zero(d));
    EXPECT_EQ(hs::FindKnownFirstTrue(d, m), std::size_t{0});
    EXPECT_EQ(hs::FindKnownLastTrue(d, m), n - 1);
    for (std::size_t i = 0; i < n; ++i) flag[i] = 0;
    flag[n / 2] = 1;
    m = hs::Gt(hs::Load(d, flag), hs::Zero(d));
    EXPECT_EQ(hs::FindKnownFirstTrue(d, m), n / 2);
    EXPECT_EQ(hs::FindKnownLastTrue(d, m), n / 2);
  }
}

// P5 batch C: MulByPow2 (store+reload ldexp) / Expand (viota + vrgather).
// TrailingZeroCount is Zvbb-gated (like Pop/LeadingZeroCount) -> not in base rv64gcv.
TEST(Rvv, P5BatchC) {
  {
    const hs::ScalableTag<float> d;
    const hs::ScalableTag<std::int32_t> di;
    const std::size_t n = hs::Lanes(d);
    alignas(64) float a[kBuf], o[kBuf]; alignas(64) std::int32_t b[kBuf];
    for (std::size_t i = 0; i < n; ++i) { a[i] = 1.5f + static_cast<float>(i) * 0.25f; b[i] = static_cast<std::int32_t>((i % 7)) - 3; }
    hs::Store(hs::MulByPow2(hs::Load(d, a), hs::Load(di, b)), d, o);
    for (std::size_t i = 0; i < n; ++i) EXPECT_NEAR(o[i], std::ldexp(a[i], b[i]), 1e-5f) << "MulByPow2 f32 " << i;
    // Red-team wide exponents: the vectorized 3-part build must survive overflow/denormal/underflow.
    const int we[] = {0, 130, -130, 200, -200, 127, -149, -150, 50, -50, 255, -255, 10, 1, -1, 90};
    const float wv[] = {1.0f, 3.0f, 1e20f, 1e-20f, 2.0f, 7.0f, 1.0f, 1.0f, 1.5f, 0.5f, 1.0f, 1.0f, -2.0f, 1.0f, 1.0f, 1.0f};
    for (std::size_t i = 0; i < n; ++i) { a[i] = wv[i % 16]; b[i] = we[i % 16]; }
    hs::Store(hs::MulByPow2(hs::Load(d, a), hs::Load(di, b)), d, o);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], std::ldexp(a[i], b[i])) << "MulByPow2 f32 wide " << i;
  }
  {
    const hs::ScalableTag<double> d;
    const hs::ScalableTag<std::int64_t> di;
    const std::size_t n = hs::Lanes(d);
    alignas(64) double a[kBuf], o[kBuf]; alignas(64) std::int64_t b[kBuf];
    for (std::size_t i = 0; i < n; ++i) { a[i] = 0.75 + static_cast<double>(i); b[i] = static_cast<std::int64_t>((i % 5)) - 2; }
    hs::Store(hs::MulByPow2(hs::Load(d, a), hs::Load(di, b)), d, o);
    for (std::size_t i = 0; i < n; ++i) EXPECT_NEAR(o[i], std::ldexp(a[i], static_cast<int>(b[i])), 1e-12) << "MulByPow2 f64 " << i;
    const long long we[] = {0, 1023, -1023, 1100, -1100, -1074, 5, -5};
    const double wv[] = {1.0, 2.0, 0.5, 1e100, 1e-100, 1.0, 3.0, 7.0};
    for (std::size_t i = 0; i < n; ++i) { a[i] = wv[i % 8]; b[i] = we[i % 8]; }
    hs::Store(hs::MulByPow2(hs::Load(d, a), hs::Load(di, b)), d, o);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], std::ldexp(a[i], static_cast<int>(b[i]))) << "MulByPow2 f64 wide " << i;
  }

  // Expand: dst[i] = m[i] ? src[k++] : 0
  {
    const hs::ScalableTag<std::int32_t> d;
    const std::size_t n = hs::Lanes(d);
    alignas(64) std::int32_t src[kBuf], flag[kBuf], o[kBuf];
    for (std::size_t i = 0; i < n; ++i) { src[i] = static_cast<std::int32_t>(i * 10 + 1); flag[i] = (i % 3 == 0) ? 1 : 0; }
    auto m = hs::Gt(hs::Load(d, flag), hs::Zero(d));
    hs::Store(hs::Expand(hs::Load(d, src), m), d, o);
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i) { const std::int32_t e = flag[i] ? src[k++] : 0; EXPECT_EQ(o[i], e) << "Expand " << i; }
  }
  {
    const hs::ScalableTag<std::int32_t> d;
    const std::size_t n = hs::Lanes(d);
    alignas(64) std::int32_t src[kBuf], flag[kBuf], o[kBuf];
    for (std::size_t i = 0; i < n; ++i) { src[i] = static_cast<std::int32_t>(i + 100); flag[i] = 0; }
    auto m0 = hs::Gt(hs::Load(d, flag), hs::Zero(d));
    hs::Store(hs::Expand(hs::Load(d, src), m0), d, o);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], 0) << "Expand all-false " << i;
    for (std::size_t i = 0; i < n; ++i) flag[i] = 1;
    auto m1 = hs::Gt(hs::Load(d, flag), hs::Zero(d));
    hs::Store(hs::Expand(hs::Load(d, src), m1), d, o);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], src[i]) << "Expand all-true " << i;
  }
}

// ---- simd-16: Zvbb bit counts / reverse / rotate (gated; base tier skips) -------------
#if defined(__riscv_zvbb)
namespace {
template <typename T>
T revbits_scalar(T v) {
  using U = std::make_unsigned_t<T>;
  U x = static_cast<U>(v), y = 0;
  for (std::size_t b = 0; b < sizeof(T) * 8; ++b) {
    y = static_cast<U>((y << 1) | (x & 1));
    x >>= 1;
  }
  return static_cast<T>(y);
}

template <typename T>
void CheckZvbbBitOps() {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  alignas(64) T a[kBuf], o[kBuf];
  using U = std::make_unsigned_t<T>;
  constexpr int kBits = static_cast<int>(sizeof(T) * 8);
  for (std::size_t i = 0; i < n; ++i) {
    // cover 0, 1, high-bit, mixed, all-ones
    if (i == 0) a[i] = T{0};
    else if (i == 1) a[i] = T{1};
    else if (i == 2) a[i] = static_cast<T>(U{1} << (kBits - 1));
    else if (i == 3) a[i] = static_cast<T>(~U{0});
    else a[i] = static_cast<T>((U{0xA5} << ((i % (sizeof(T)))) * 8) | static_cast<U>(i * 17 + 3));
  }
  const auto va = hs::Load(d, a);

  hs::Store(hs::LeadingZeroCount(va), d, o);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(o[i], static_cast<T>(std::countl_zero(static_cast<U>(a[i])))) << "clz " << i;

  hs::Store(hs::TrailingZeroCount(va), d, o);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(o[i], static_cast<T>(std::countr_zero(static_cast<U>(a[i])))) << "ctz " << i;

  hs::Store(hs::ReverseBits(va), d, o);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(o[i], revbits_scalar(a[i])) << "brev " << i;

  hs::Store(hs::PopulationCount(va), d, o);  // Zvbb vcpop.v (inline-asm; GCC-14 lacks C wrapper)
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(o[i], static_cast<T>(std::popcount(static_cast<U>(a[i])))) << "vcpop " << i;

  // RotateRight/Left native Zvbb path (same oracle as shift-or)
  constexpr int kR = kBits > 8 ? 5 : 3;
  hs::Store(hs::RotateRight<kR>(va), d, o);
  for (std::size_t i = 0; i < n; ++i) {
    const U u = static_cast<U>(a[i]);
    const U e = static_cast<U>((u >> kR) | (u << (kBits - kR)));
    EXPECT_EQ(o[i], static_cast<T>(e)) << "ror " << i;
  }
  hs::Store(hs::RotateLeft<kR>(va), d, o);
  for (std::size_t i = 0; i < n; ++i) {
    const U u = static_cast<U>(a[i]);
    const U e = static_cast<U>((u << kR) | (u >> (kBits - kR)));
    EXPECT_EQ(o[i], static_cast<T>(e)) << "rol " << i;
  }
}
}  // namespace

TEST(Rvv, Simd16Zvbb) {
  CheckZvbbBitOps<std::uint8_t>();
  CheckZvbbBitOps<std::uint16_t>();
  CheckZvbbBitOps<std::uint32_t>();
  CheckZvbbBitOps<std::uint64_t>();
  CheckZvbbBitOps<std::int8_t>();
  CheckZvbbBitOps<std::int16_t>();
  CheckZvbbBitOps<std::int32_t>();
  CheckZvbbBitOps<std::int64_t>();
}
#endif  // __riscv_zvbb

// ---- simd-16: Zvbc CLMul (128-bit-block semantics) -----------------------------------
#if defined(__riscv_zvbc)
namespace {
// Software clmul64: 128-bit product of two 64-bit operands without carries (scalar oracle).
void clmul64_soft(std::uint64_t a, std::uint64_t b, std::uint64_t o[2]) {
  o[0] = 0; o[1] = 0;
  for (int i = 0; i < 64; ++i) {
    if ((b >> i) & 1ull) {
      o[0] ^= a << i;
      if (i) o[1] ^= a >> (64 - i);
    }
  }
}
}  // namespace

TEST(Rvv, Simd16ZvbcClmul) {
  const hs::ScalableTag<std::uint64_t> d;
  const std::size_t n = hs::Lanes(d);
  ASSERT_GE(n, std::size_t{2});
  ASSERT_EQ(n % 2, std::size_t{0});
  alignas(64) std::uint64_t a[kBuf], b[kBuf], o[kBuf];
  for (std::size_t i = 0; i < n; ++i) {
    a[i] = 0x0123456789ABCDEFull * (i + 1) + 0x1111ull * i;
    b[i] = 0xFEDCBA9876543210ull ^ (0x55ull << (i % 56));
  }
  // Force known simple cases into block 0.
  a[0] = 1; b[0] = 1;          // lower: 1*1 = 1
  a[1] = 0x2; b[1] = 0x3;      // upper of block 0
  if (n >= 4) { a[2] = 0; b[2] = ~0ull; a[3] = ~0ull; b[3] = 1; }

  hs::Store(hs::CLMulLower(hs::Load(d, a), hs::Load(d, b)), d, o);
  for (std::size_t blk = 0; blk < n; blk += 2) {
    std::uint64_t exp[2];
    clmul64_soft(a[blk], b[blk], exp);
    EXPECT_EQ(o[blk], exp[0]) << "CLMulLower lo blk=" << blk;
    EXPECT_EQ(o[blk + 1], exp[1]) << "CLMulLower hi blk=" << blk;
  }

  hs::Store(hs::CLMulUpper(hs::Load(d, a), hs::Load(d, b)), d, o);
  for (std::size_t blk = 0; blk < n; blk += 2) {
    std::uint64_t exp[2];
    clmul64_soft(a[blk + 1], b[blk + 1], exp);
    EXPECT_EQ(o[blk], exp[0]) << "CLMulUpper lo blk=" << blk;
    EXPECT_EQ(o[blk + 1], exp[1]) << "CLMulUpper hi blk=" << blk;
  }
}
#endif  // __riscv_zvbc

// ---- simd-17 batch3: whole-vector half Concat* (scalable) ----------------------------
TEST(Rvv, ConcatHalves) {
  const hs::ScalableTag<std::int32_t> d;
  const std::size_t n = hs::Lanes(d);
  ASSERT_GE(n, std::size_t{2});
  ASSERT_EQ(n % 2, std::size_t{0});
  const std::size_t h = n / 2;
  alignas(64) std::int32_t lo[kBuf], hi[kBuf], out[kBuf];
  for (std::size_t i = 0; i < n; ++i) {
    lo[i] = static_cast<std::int32_t>(i);
    hi[i] = static_cast<std::int32_t>(100 + i);
  }
  const auto vlo = hs::Load(d, lo);
  const auto vhi = hs::Load(d, hi);

  hs::Store(hs::ConcatLowerLower(d, vhi, vlo), d, out);
  for (std::size_t i = 0; i < h; ++i) {
    EXPECT_EQ(out[i], lo[i]) << "LL lo i=" << i;
    EXPECT_EQ(out[h + i], hi[i]) << "LL hi i=" << i;
  }
  hs::Store(hs::ConcatUpperUpper(d, vhi, vlo), d, out);
  for (std::size_t i = 0; i < h; ++i) {
    EXPECT_EQ(out[i], lo[h + i]) << "UU lo i=" << i;
    EXPECT_EQ(out[h + i], hi[h + i]) << "UU hi i=" << i;
  }
  hs::Store(hs::ConcatLowerUpper(d, vhi, vlo), d, out);
  for (std::size_t i = 0; i < h; ++i) {
    EXPECT_EQ(out[i], lo[h + i]) << "LU lo i=" << i;
    EXPECT_EQ(out[h + i], hi[i]) << "LU hi i=" << i;
  }
  hs::Store(hs::ConcatUpperLower(d, vhi, vlo), d, out);
  for (std::size_t i = 0; i < h; ++i) {
    EXPECT_EQ(out[i], lo[i]) << "UL lo i=" << i;
    EXPECT_EQ(out[h + i], hi[h + i]) << "UL hi i=" << i;
  }
}

TEST(Rvv, CryptoCapabilityFloor) {
  const hs::ScalableTag<std::uint8_t> d;
  const std::size_t n = hs::Lanes(d);
  ASSERT_EQ(n % 16u, 0u);
  alignas(64) std::uint8_t st[kBuf], key[kBuf], out[kBuf], ref[kBuf];
  for (std::size_t i = 0; i < n; ++i) {
    st[i] = static_cast<std::uint8_t>(i * 7 + 1);
    key[i] = static_cast<std::uint8_t>(i * 3 + 5);
  }
  const auto vs = hs::Load(d, st), vk = hs::Load(d, key), z = hs::Zero(d);
  hs::Store(hs::AESLastRound(z, z), d, out);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out[i], 0x63) << "anchor i=" << i;
  hs::Store(hs::AESLastRoundInv(hs::AESLastRound(vs, z), z), d, out);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out[i], st[i]) << "roundtrip i=" << i;
  hs::Store(hs::AESInvMixColumns(hs::AESRound(vs, z)), d, out);
  hs::Store(hs::AESLastRound(vs, z), d, ref);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out[i], ref[i]) << "invmix i=" << i;
  hs::Store(hs::AESRound(vs, vk), d, out);
  hs::Store(hs::AESRound(vs, z), d, ref);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out[i], ref[i] ^ key[i]) << "roundkey i=" << i;
  hs::Store(hs::AESRoundInv(z, z), d, out);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out[i], 0x52) << "inv anchor i=" << i;
  hs::Store(hs::AESKeyGenAssist<0x01>(z), d, out);
  for (std::size_t b = 0; b < n; b += 16)
    for (std::size_t i = 0; i < 16; ++i)
      EXPECT_EQ(out[b + i], (i == 4 || i == 12) ? 0x62 : 0x63) << "keygen i=" << b + i;
}

TEST(Rvv, BitShuffleMultiRotateAndStream) {
  const hs::ScalableTag<std::uint64_t> d64;
  const hs::ScalableTag<std::uint8_t> d8;
  const std::size_t n64 = hs::Lanes(d64);
  const std::size_t n8 = hs::Lanes(d8);
  ASSERT_EQ(n8, n64 * 8);
  alignas(64) std::uint64_t data[kBuf / 8], out[kBuf / 8];
  alignas(64) std::uint8_t idx[kBuf];
  for (std::size_t q = 0; q < n64; ++q) {
    data[q] = 0x0123456789ABCDEFULL ^ (0x1111111111111111ULL * q);
    for (std::size_t j = 0; j < 8; ++j)
      idx[q * 8 + j] = static_cast<std::uint8_t>((q * 29 + j * 37 + 192) & 255);
  }
  const auto values = hs::Load(d64, data);
  const auto indices = hs::Load(d8, idx);
  hs::Store(hs::BitShuffle(values, indices), d64, out);
  for (std::size_t q = 0; q < n64; ++q) {
    std::uint64_t expected = 0;
    for (std::size_t j = 0; j < 8; ++j)
      expected |= ((data[q] >> (idx[q * 8 + j] & 63)) & 1u) << j;
    EXPECT_EQ(out[q], expected) << "bitshuffle q=" << q;
  }
  hs::Store(hs::MultiRotateRight(values, indices), d64, out);
  for (std::size_t q = 0; q < n64; ++q) {
    std::uint64_t expected = 0;
    for (std::size_t j = 0; j < 8; ++j) {
      const unsigned s = idx[q * 8 + j] & 63;
      const std::uint64_t rotated = s == 0 ? data[q] : (data[q] >> s) | (data[q] << (64 - s));
      expected |= (rotated & 0xFFu) << (8 * j);
    }
    EXPECT_EQ(out[q], expected) << "multirotate q=" << q;
  }
  for (std::size_t q = 0; q < n64; ++q) out[q] = 0;
  hs::Stream(values, d64, out);
  for (std::size_t q = 0; q < n64; ++q) EXPECT_EQ(out[q], data[q]) << "stream q=" << q;
}

TEST(Rvv, ScalableSadFamilies) {
  const hs::ScalableTag<std::uint8_t> d8;
  const hs::ScalableTag<std::uint16_t> d16;
  const hs::ScalableTag<std::uint64_t> d64;
  const std::size_t n = hs::Lanes(d8);
  alignas(64) std::uint8_t a[kBuf], b[kBuf];
  alignas(64) std::uint16_t o16[kBuf];
  alignas(64) std::uint64_t o64[kBuf];
  for (std::size_t i = 0; i < n; ++i) { a[i] = static_cast<std::uint8_t>(i * 7u + 3u); b[i] = static_cast<std::uint8_t>(251u - i * 5u); }
  hs::Store(hs::SumsOf8AbsDiff(hs::Load(d8, a), hs::Load(d8, b)), d64, o64);
  for (std::size_t g = 0; g < n / 8; ++g) { std::uint64_t expected = 0; for (std::size_t j = 0; j < 8; ++j) expected += static_cast<std::uint64_t>(std::abs(static_cast<int>(a[8 * g + j]) - static_cast<int>(b[8 * g + j]))); EXPECT_EQ(o64[g], expected); }
  hs::Store(hs::SumsOfAdjQuadAbsDiff<1, 2>(hs::Load(d8, a), hs::Load(d8, b)), d16, o16);
  for (std::size_t block = 0; block < n; block += 16) for (std::size_t i = 0; i < 8; ++i) { unsigned expected = 0; for (std::size_t j = 0; j < 4; ++j) expected += static_cast<unsigned>(std::abs(static_cast<int>(a[block + 4 + i + j]) - static_cast<int>(b[block + 8 + j]))); EXPECT_EQ(o16[block / 2 + i], expected); }
  hs::Store(hs::SumsOfShuffledQuadAbsDiff<0, 3, 1, 2>(hs::Load(d8, a), hs::Load(d8, b)), d16, o16);
  constexpr int sel[4] = {2, 1, 3, 0}; constexpr int start[4] = {0, 2, 8, 10};
  for (std::size_t block = 0; block < n; block += 16) for (int pair = 0; pair < 4; ++pair) for (int odd = 0; odd < 2; ++odd) { std::uint8_t shuffled[16]; for (int q = 0; q < 4; ++q) for (int j = 0; j < 4; ++j) shuffled[q * 4 + j] = a[block + sel[q] * 4 + j]; unsigned expected = 0; for (int j = 0; j < 4; ++j) expected += static_cast<unsigned>(std::abs(static_cast<int>(shuffled[start[pair] + odd + j]) - static_cast<int>(b[block + pair * 4 + j]))); EXPECT_EQ(o16[block / 2 + pair * 2 + odd], expected); }
}

TEST(Rvv, WholeVectorPairwiseLookupDemoteAndScale) {
  const hs::ScalableTag<std::int32_t> d;
  const std::size_t n = hs::Lanes(d);
  alignas(64) std::int32_t a[kBuf], b[kBuf], idx[kBuf], out[kBuf];
  for (std::size_t i = 0; i < n; ++i) { a[i] = static_cast<std::int32_t>(i + 1); b[i] = static_cast<std::int32_t>(100 + i); idx[i] = static_cast<std::int32_t>((i & 1u) == 0 ? i / 2 : n + i / 2); }
  const auto va = hs::Load(d, a), vb = hs::Load(d, b);
  hs::Store(hs::PairwiseAdd(va, vb), d, out);
  for (std::size_t i = 0; i < n / 2; ++i) { EXPECT_EQ(out[i], a[2 * i] + a[2 * i + 1]); EXPECT_EQ(out[n / 2 + i], b[2 * i] + b[2 * i + 1]); }
  hs::Store(hs::PairwiseSub(va, vb), d, out);
  for (std::size_t i = 0; i < n / 2; ++i) { EXPECT_EQ(out[i], a[2 * i] - a[2 * i + 1]); EXPECT_EQ(out[n / 2 + i], b[2 * i] - b[2 * i + 1]); }
  hs::Store(hs::InterleaveWholeLower(d, va, vb), d, out);
  for (std::size_t i = 0; i < n / 2; ++i) { EXPECT_EQ(out[2 * i], a[i]); EXPECT_EQ(out[2 * i + 1], b[i]); }
  hs::Store(hs::InterleaveWholeUpper(d, va, vb), d, out);
  for (std::size_t i = 0; i < n / 2; ++i) { EXPECT_EQ(out[2 * i], a[n / 2 + i]); EXPECT_EQ(out[2 * i + 1], b[n / 2 + i]); }
  const auto indices = hs::IndicesFromVec(d, hs::Load(d, idx));
  hs::Store(hs::TwoTablesLookupLanes(d, va, vb, indices), d, out);
  for (std::size_t i = 0; i < n; ++i) { const std::size_t k = static_cast<std::size_t>(idx[i]); EXPECT_EQ(out[i], k < n ? a[k] : b[k - n]); }

  const hs::ScalableTag<std::int16_t> d16;
  alignas(64) std::int16_t narrow[kBuf];
  hs::Store(hs::OrderedDemote2To(d16, va, vb), d16, narrow);
  for (std::size_t i = 0; i < n; ++i) { EXPECT_EQ(narrow[i], a[i]); EXPECT_EQ(narrow[n + i], b[i]); }

  const hs::ScalableTag<float> df;
  const std::size_t nf = hs::Lanes(df);
  alignas(64) float fv[kBuf], fe[kBuf], fo[kBuf];
  for (std::size_t i = 0; i < nf; ++i) { fv[i] = static_cast<float>(i + 1); fe[i] = static_cast<float>(static_cast<int>(i % 5) - 2) + 0.75f; }
  hs::Store(hs::MulByFloorPow2(hs::Load(df, fv), hs::Load(df, fe)), df, fo);
  for (std::size_t i = 0; i < nf; ++i) EXPECT_EQ(fo[i], std::ldexp(fv[i], static_cast<int>(std::floor(fe[i]))));

  const hs::ScalableTag<std::uint32_t> du;
  const std::size_t nu = hs::Lanes(du);
  alignas(64) std::uint32_t ua[kBuf], ub[kBuf], uc[kBuf], uo[kBuf];
  for (std::size_t i = 0; i < nu; ++i) { ua[i] = 0x80000000u + static_cast<std::uint32_t>(i * 97u); ub[i] = 0x10203040u + static_cast<std::uint32_t>(i); uc[i] = 0x55AA00FFu; }
  const auto uva = hs::Load(du, ua), uvb = hs::Load(du, ub), uvc = hs::Load(du, uc);
  hs::Store(hs::Xor3(uva, uvb, uvc), du, uo);
  for (std::size_t i = 0; i < nu; ++i) EXPECT_EQ(uo[i], ua[i] ^ ub[i] ^ uc[i]);
  hs::Store(hs::ShiftRightUniform(uva, 3), du, uo);
  for (std::size_t i = 0; i < nu; ++i) EXPECT_EQ(uo[i], ua[i] >> 3);
  hs::Store(hs::MulHigh(uva, uvb), du, uo);
  for (std::size_t i = 0; i < nu; ++i) EXPECT_EQ(uo[i], static_cast<std::uint32_t>((static_cast<std::uint64_t>(ua[i]) * ub[i]) >> 32));
  auto uvhi = hs::Zero(du);
  hs::Store(hs::MulHighLow(uva, uvb, uvhi), du, uo);
  for (std::size_t i = 0; i < nu; ++i)
    EXPECT_EQ(uo[i], static_cast<std::uint32_t>(ua[i] * ub[i]));
  hs::Store(uvhi, du, uo);
  for (std::size_t i = 0; i < nu; ++i)
    EXPECT_EQ(uo[i], static_cast<std::uint32_t>(
                         (static_cast<std::uint64_t>(ua[i]) * ub[i]) >> 32));
  hs::Store(hs::LoadNOr(hs::Set(du, 0xDEADBEEFu), du, ua, nu / 2), du, uo);
  for (std::size_t i = 0; i < nu; ++i) EXPECT_EQ(uo[i], i < nu / 2 ? ua[i] : 0xDEADBEEFu);
}

TEST(Rvv, RuntimeShiftsRoundedAverageAndBlockShuffles) {
  const hs::ScalableTag<std::int32_t> d;
  const std::size_t n = hs::Lanes(d);
  alignas(64) std::int32_t in[kBuf], amounts[kBuf], out[kBuf];
  for (std::size_t i = 0; i < n; ++i) { in[i] = static_cast<std::int32_t>(i * 137) - 300; amounts[i] = static_cast<std::int32_t>(i & 3u); }
  const auto v = hs::Load(d, in);
  hs::Store(hs::ShiftLeftSame(v, 2), d, out);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(static_cast<std::uint32_t>(out[i]), static_cast<std::uint32_t>(in[i]) << 2);
  hs::Store(hs::ShiftRightSame(v, 2), d, out);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out[i], in[i] >> 2);
  hs::Store(hs::RotateLeftSame(v, 4), d, out);
  for (std::size_t i = 0; i < n; ++i) { const auto u = static_cast<std::uint32_t>(in[i]); EXPECT_EQ(static_cast<std::uint32_t>(out[i]), (u << 4) | (u >> 28)); }
  hs::Store(hs::RoundingShr(v, hs::Load(d, amounts)), d, out);
  for (std::size_t i = 0; i < n; ++i) { const int s = amounts[i]; const auto expected = s == 0 ? in[i] : (in[i] + (std::int32_t{1} << (s - 1))) >> s; EXPECT_EQ(out[i], expected); }
  hs::Store(hs::RoundingShiftRightSame(v, 3), d, out);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out[i], (in[i] + 4) >> 3);

  auto check32 = [&](auto op, auto map, const char* name) {
    hs::Store(op(v), d, out);
    for (std::size_t base = 0; base < n; base += 4)
      for (std::size_t i = 0; i < 4; ++i) EXPECT_EQ(out[base + i], in[base + map(i)]) << name << " lane=" << base + i;
  };
  check32([](auto x) { return hs::Shuffle2301(x); }, [](std::size_t i) { return i ^ 1u; }, "2301");
  check32([](auto x) { return hs::Shuffle0321(x); }, [](std::size_t i) { return (i + 1u) & 3u; }, "0321");
  check32([](auto x) { return hs::Shuffle2103(x); }, [](std::size_t i) { return (i + 3u) & 3u; }, "2103");
  check32([](auto x) { return hs::Shuffle0123(x); }, [](std::size_t i) { return i ^ 3u; }, "0123");

  hs::Store(hs::ShiftLeftLanes<1>(v), d, out);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out[i], (i & 3u) == 0 ? 0 : in[i - 1]);
  hs::Store(hs::ShiftRightLanes<1>(v), d, out);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out[i], (i & 3u) == 3 ? 0 : in[i + 1]);
  const auto vb32 = hs::Load(d, amounts);
  hs::Store(hs::CombineShiftRightLanes<1>(vb32, v), d, out);
  for (std::size_t i = 0; i < n; ++i) { const std::size_t local = i & 3u; EXPECT_EQ(out[i], local < 3 ? in[i + 1] : amounts[i - local]); }
  hs::Store(hs::PairwiseAdd128(v, vb32), d, out);
  for (std::size_t base = 0; base < n; base += 4) { EXPECT_EQ(out[base], in[base] + in[base + 1]); EXPECT_EQ(out[base + 1], in[base + 2] + in[base + 3]); EXPECT_EQ(out[base + 2], amounts[base] + amounts[base + 1]); EXPECT_EQ(out[base + 3], amounts[base + 2] + amounts[base + 3]); }
  hs::Store(hs::PairwiseSub128(v, vb32), d, out);
  for (std::size_t base = 0; base < n; base += 4) { EXPECT_EQ(out[base], in[base] - in[base + 1]); EXPECT_EQ(out[base + 1], in[base + 2] - in[base + 3]); EXPECT_EQ(out[base + 2], amounts[base] - amounts[base + 1]); EXPECT_EQ(out[base + 3], amounts[base + 2] - amounts[base + 3]); }
  hs::Store(hs::Per4LaneBlockShuffle<2, 3, 0, 1>(v), d, out);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out[i], in[i ^ 1u]);

  const hs::Half<hs::ScalableTag<std::int32_t>> dh;
  hs::Store(hs::InsertIntoUpper(d, hs::Load(dh, amounts), v), d, out);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out[i], i < n / 2 ? in[i] : amounts[i - n / 2]);
  if (n >= 8) {
    constexpr std::size_t kBlockLanes = 4;
    const std::size_t blocks = n / kBlockLanes;
    hs::Store(hs::SlideUpBlocks<1>(d, v), d, out);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out[i], i < 4 ? 0 : in[i - 4]);
    hs::Store(hs::SlideDownBlocks<1>(d, v), d, out);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out[i], i + 4 < n ? in[i + 4] : 0);
    hs::Store(hs::InterleaveEvenBlocks(d, v, vb32), d, out);
    for (std::size_t i = 0; i < n; ++i) { const std::size_t ob = i / 4, local = i & 3u, src = (ob & ~std::size_t{1}) * 4 + local; EXPECT_EQ(out[i], (ob & 1u) == 0 ? in[src] : amounts[src]); }
    hs::Store(hs::InterleaveOddBlocks(d, v, vb32), d, out);
    for (std::size_t i = 0; i < n; ++i) { const std::size_t ob = i / 4, local = i & 3u, sb = ob | 1u; const std::int32_t expected = sb < blocks ? ((ob & 1u) == 0 ? in[sb * 4 + local] : amounts[sb * 4 + local]) : 0; EXPECT_EQ(out[i], expected); }
    auto check_lu = [&](auto result, bool upper) { hs::Store(result, d, out); for (std::size_t i = 0; i < n; ++i) { const std::size_t ob = i / 4, local = i & 3u, sb = ob / 2 + (upper ? blocks / 2 : 0), src = sb * 4 + local; EXPECT_EQ(out[i], (ob & 1u) == 0 ? in[src] : amounts[src]); } };
    check_lu(hs::InterleaveLowerBlocks(d, v, vb32), false);
    check_lu(hs::InterleaveUpperBlocks(d, v, vb32), true);
    const hs::CappedTag<std::int32_t, 4> db;
    alignas(64) std::int32_t block_out[kBuf];
    hs::Store(hs::ExtractBlock<1>(v), db, block_out);
    for (std::size_t i = 0; i < 4; ++i) EXPECT_EQ(block_out[i], in[4 + i]);
    hs::Store(hs::BroadcastBlock<1>(v), d, out);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out[i], in[4 + (i & 3u)]);
    hs::Store(hs::InsertBlock<1>(d, v, hs::ExtractBlock<0>(vb32)), d, out);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out[i], i >= 4 && i < 8 ? amounts[i - 4] : in[i]);
  }

  const hs::ScalableTag<std::uint8_t> d8;
  const std::size_t n8 = hs::Lanes(d8);
  alignas(64) std::uint8_t a8[kBuf], b8[kBuf], o8[kBuf];
  for (std::size_t i = 0; i < n8; ++i) { a8[i] = static_cast<std::uint8_t>(i * 3u); b8[i] = static_cast<std::uint8_t>(255u - i); }
  hs::Store(hs::AverageRound(hs::Load(d8, a8), hs::Load(d8, b8)), d8, o8);
  for (std::size_t i = 0; i < n8; ++i) EXPECT_EQ(o8[i], static_cast<std::uint8_t>((static_cast<unsigned>(a8[i]) + b8[i] + 1u) >> 1));
  const auto va8 = hs::Load(d8, a8), vb8 = hs::Load(d8, b8);
  hs::Store(hs::ResizeBitCast(d, hs::ResizeBitCast(d8, v)), d, out);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(out[i], in[i]);
  const hs::CappedTag<std::uint8_t, 16> d8_block;
  hs::Store(hs::ResizeBitCast(d8, d8_block, hs::Load(d8_block, a8)), d8, o8);
  for (std::size_t i = 0; i < n8; ++i) EXPECT_EQ(o8[i], i < 16 ? a8[i] : 0);
  hs::Store(hs::ShiftLeftBytes<3>(va8), d8, o8);
  for (std::size_t i = 0; i < n8; ++i) { const std::size_t local = i & 15u; EXPECT_EQ(o8[i], local < 3 ? 0 : a8[i - 3]); }
  hs::Store(hs::ShiftRightBytes<5>(va8), d8, o8);
  for (std::size_t i = 0; i < n8; ++i) { const std::size_t local = i & 15u; EXPECT_EQ(o8[i], local + 5 < 16 ? a8[i + 5] : 0); }
  hs::Store(hs::CombineShiftRightBytes<4>(vb8, va8), d8, o8);
  for (std::size_t i = 0; i < n8; ++i) { const std::size_t local = i & 15u; EXPECT_EQ(o8[i], local + 4 < 16 ? a8[i + 4] : b8[i - 12]); }

  const hs::ScalableTag<std::uint64_t> d64;
  const std::size_t n64 = hs::Lanes(d64);
  alignas(64) std::uint64_t a64[kBuf], o64[kBuf];
  for (std::size_t i = 0; i < n64; ++i) a64[i] = 100u + i;
  hs::Store(hs::Shuffle01(hs::Load(d64, a64)), d64, o64);
  for (std::size_t i = 0; i < n64; i += 2) { EXPECT_EQ(o64[i], a64[i + 1]); EXPECT_EQ(o64[i + 1], a64[i]); }
}

// mask <-> packed bits byte-buffer round-trip (VL>64 fix): RVV VLEN can exceed a uint64 lane count,
// so StoreMaskBits/LoadMaskBits use a byte buffer (1 bit/lane). Round-trip must reproduce the mask.
template <typename T>
void CheckMaskBits() {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  alignas(64) T a[kBuf];
  for (std::size_t i = 0; i < n; ++i) a[i] = static_cast<T>((i % 3 == 0) ? 5 : -1);
  const auto m = hs::Gt(hs::Load(d, a), hs::Zero(d));  // lane true where a>0 (i%3==0)
  std::uint8_t bits[kBuf / 8 + 1] = {};
  const std::size_t nbytes = hs::StoreMaskBits(d, m, bits);
  EXPECT_EQ(nbytes, (n + 7) / 8) << "StoreMaskBits nbytes";
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(((bits[i >> 3] >> (i & 7)) & 1u) != 0, (i % 3 == 0)) << "bit i=" << i;
  const auto m2 = hs::LoadMaskBits(d, bits);
  EXPECT_EQ(hs::CountTrue(d, m2), hs::CountTrue(d, m)) << "roundtrip CountTrue";
  alignas(64) T v1[kBuf], v2[kBuf];
  hs::Store(hs::VecFromMask(d, m), d, v1);
  hs::Store(hs::VecFromMask(d, m2), d, v2);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(v1[i] != 0, v2[i] != 0) << "roundtrip lane i=" << i;
}
TEST(Rvv, MaskBitsByteBuffer) {
  CheckMaskBits<std::int8_t>();   // widest lane count -> exercises >64 lanes at large VLEN
  CheckMaskBits<std::int16_t>();
  CheckMaskBits<std::int32_t>();
  CheckMaskBits<std::int64_t>();
}
