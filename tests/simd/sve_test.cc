// SVE backend test -- the scalable-width backend's op face cross-checked against a plain scalar
// oracle. Unlike the fixed-width vec_test (which pins Lanes to a compile-time constant), this test
// is VL-AGNOSTIC: it queries the runtime Lanes(d) and checks every implemented op for every lane
// type, so the SAME binary is correct at any SVE vector length. Run it under qemu-aarch64 at
// multiple VLs (128/256/512) -- the scalable contract is "one binary, every width".
//
// This is NOT registered as an auto-run x86 ctest: SVE has no x86 host and no native dev machine,
// so it is cross-compiled (aarch64-linux-gnu-g++ -march=armv8-a+sve) and run under qemu-aarch64-
// static. See docs/architecture/simd-design.md (SVE section) for the exact recipe. The op surface
// here mirrors backend/sve.h; ops NOT yet in sve.h (block-relative swizzle, width-changing
// promote/demote, SVE2 crypto) are intentionally absent (documented boundary).
#include <gtest/gtest.h>

#define HYLE_SIMD_TARGET_SVE 1
#include "hyle/simd/vec.h"

#include <cmath>
#include <cstdint>
#include <limits>

namespace hs = hyle::simd;

namespace {
// A scratch big enough for the widest VL we test (2048-bit => 64 x u8). kNativeVectorBytes is 256.
constexpr std::size_t kBuf = 256;
}  // namespace

// ---- Phase A: Set / Zero / Load / Store + runtime Lanes -----------------------------
template <typename T>
void CheckLoadStore() {
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  ASSERT_GE(n, std::size_t{1});
  ASSERT_LE(n * sizeof(T), hs::kNativeVectorBytes);  // within the upper bound

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

TEST(Sve, LoadStoreAllTypes) {
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

// ---- Phase B: arithmetic / bitwise / compare / select / mask-reduce -----------------
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
  hs::Store(hs::Min(va, vb), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], static_cast<T>(a[i] < b[i] ? a[i] : b[i]));
  hs::Store(hs::Max(va, vb), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], static_cast<T>(a[i] > b[i] ? a[i] : b[i]));
  hs::Store(hs::And(va, vb), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], static_cast<T>(a[i] & b[i]));
  hs::Store(hs::Or(va, vb), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], static_cast<T>(a[i] | b[i]));
  hs::Store(hs::Xor(va, vb), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], static_cast<T>(a[i] ^ b[i]));

  // compare -> mask -> select
  auto m = hs::Lt(va, vb);
  hs::Store(hs::IfThenElse(m, va, vb), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], static_cast<T>(a[i] < b[i] ? a[i] : b[i]));
  std::size_t ref = 0;
  for (std::size_t i = 0; i < n; ++i) if (a[i] < b[i]) ++ref;
  EXPECT_EQ(hs::CountTrue(d, m), ref);
  EXPECT_TRUE(hs::AllTrue(d, hs::Eq(va, va)));
  EXPECT_TRUE(hs::AllFalse(d, hs::Lt(va, va)));
  EXPECT_EQ(hs::GetLane(va), a[0]);
}

TEST(Sve, IntArithAllTypes) {
  CheckIntArith<std::int8_t>();
  CheckIntArith<std::uint8_t>();
  CheckIntArith<std::int16_t>();
  CheckIntArith<std::uint16_t>();
  CheckIntArith<std::int32_t>();
  CheckIntArith<std::uint32_t>();
  CheckIntArith<std::int64_t>();
  CheckIntArith<std::uint64_t>();
}

TEST(Sve, FloatArith) {
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
}

// ---- Phase C: reduce / shift / reverse / tail / compress / popcount -----------------
TEST(Sve, ReduceShiftReverse) {
  using T = std::int32_t;
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  alignas(64) T a[kBuf], o[kBuf];
  for (std::size_t i = 0; i < n; ++i) a[i] = static_cast<T>(i + 1);
  auto va = hs::Load(d, a);

  T rs = 0; for (std::size_t i = 0; i < n; ++i) rs += a[i];
  EXPECT_EQ(hs::ReduceSum(d, va), rs);
  EXPECT_EQ(hs::ReduceMin(d, va), a[0]);
  EXPECT_EQ(hs::ReduceMax(d, va), a[n - 1]);

  hs::Store(hs::ShiftLeft<2>(va), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], static_cast<T>(a[i] << 2));
  hs::Store(hs::ShiftRight<1>(va), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], static_cast<T>(a[i] >> 1));

  alignas(64) T b[kBuf]; for (std::size_t i = 0; i < n; ++i) b[i] = static_cast<T>(i % 4);
  hs::Store(hs::Shl(va, hs::Load(d, b)), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], static_cast<T>(a[i] << b[i]));

  hs::Store(hs::Reverse(d, va), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], a[n - 1 - i]);
}

TEST(Sve, TailAndCompress) {
  using T = std::int32_t;
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  alignas(64) T a[kBuf], o[kBuf] = {};
  for (std::size_t i = 0; i < n; ++i) a[i] = static_cast<T>(i + 1);

  const std::size_t k = n > 3 ? 3 : n;
  hs::StoreN(hs::LoadN(d, a, k), d, o, k);
  for (std::size_t i = 0; i < k; ++i) EXPECT_EQ(o[i], a[i]);

  // Compress: keep even-index lanes, packed to front.
  auto va = hs::Load(d, a);
  alignas(64) T mb[kBuf]; for (std::size_t i = 0; i < n; ++i) mb[i] = (i % 2 == 0) ? 1 : 0;
  auto mm = hs::Eq(hs::Load(d, mb), hs::Set(d, static_cast<T>(1)));
  hs::Store(hs::Compress(va, mm), d, o);
  std::size_t j = 0;
  for (std::size_t i = 0; i < n; ++i) if (i % 2 == 0) EXPECT_EQ(o[j++], a[i]);
}

TEST(Sve, PopCountLeadingZero) {
  using T = std::uint32_t;
  const hs::ScalableTag<T> d;
  const std::size_t n = hs::Lanes(d);
  alignas(64) T a[kBuf], o[kBuf];
  for (std::size_t i = 0; i < n; ++i) a[i] = static_cast<T>(i * 7 + 3);
  auto va = hs::Load(d, a);
  hs::Store(hs::PopulationCount(va), d, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], static_cast<T>(__builtin_popcount(a[i])));
  hs::Store(hs::LeadingZeroCount(va), d, o);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(o[i], static_cast<T>(a[i] ? __builtin_clz(a[i]) : 32));
}

// ---- conversions + gather -----------------------------------------------------------
TEST(Sve, ConvertGather) {
  const hs::ScalableTag<float> df;
  const hs::ScalableTag<std::int32_t> di;
  const hs::ScalableTag<std::uint32_t> du;
  const std::size_t n = hs::Lanes(df);

  alignas(64) float f[kBuf]; alignas(64) std::int32_t o[kBuf];
  for (std::size_t i = 0; i < n; ++i) f[i] = static_cast<float>(i) * 1.4f + 0.6f;
  auto vf = hs::Load(df, f);
  hs::Store(hs::ConvertTo(di, vf), di, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], static_cast<std::int32_t>(f[i]));  // truncate
  hs::Store(hs::RoundToInt(di, vf), di, o);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(o[i], static_cast<std::int32_t>(std::nearbyint(f[i])));

  alignas(64) float table[512]; for (int i = 0; i < 512; ++i) table[i] = static_cast<float>(i * 10);
  alignas(64) std::uint32_t idx[kBuf]; for (std::size_t i = 0; i < n; ++i) idx[i] = static_cast<std::uint32_t>((i * 7) % 512);
  alignas(64) float of[kBuf];
  hs::Store(hs::GatherIndex(df, table, hs::Load(du, idx)), df, of);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(of[i], table[idx[i]]);
}

// Round/Floor/Ceil/Trunc (float->float, svrintX) vs libm; FindFirstTrue/FindLastTrue vs scalar scan.
TEST(Sve, RoundingAndFindTrue) {
  const hs::ScalableTag<float> d;
  const std::size_t n = hs::Lanes(d);
  alignas(64) float a[kBuf], o[kBuf];
  for (std::size_t i = 0; i < n; ++i) a[i] = (static_cast<float>(i) - static_cast<float>(n) / 2) * 1.375f;
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
  // build the "in [2,hi)" mask from a per-lane flag array (avoids combining two masks with a
  // vector And -- masks are svbool_t, not vectors).
  alignas(64) std::int32_t flag[kBuf];
  for (std::size_t i = 0; i < ni; ++i) flag[i] = ((std::int32_t)i >= 2 && (std::int32_t)i < hi) ? 1 : 0;
  auto m = hs::Gt(hs::Load(di, flag), hs::Set(di, 0));
  std::ptrdiff_t ff = -1, fl = -1;
  for (std::size_t i = 0; i < ni; ++i) { if (flag[i]) { if (ff < 0) ff = (std::ptrdiff_t)i; fl = (std::ptrdiff_t)i; } }
  EXPECT_EQ(hs::FindFirstTrue(di, m), ff);
  EXPECT_EQ(hs::FindLastTrue(di, m), fl);
  alignas(64) std::int32_t zero[kBuf] = {};
  auto mf = hs::Gt(hs::Load(di, zero), hs::Set(di, 0));  // all false
  EXPECT_EQ(hs::FindFirstTrue(di, mf), std::ptrdiff_t{-1});
  EXPECT_EQ(hs::FindLastTrue(di, mf), std::ptrdiff_t{-1});
}

// ---- clean-native batch: bitcast / saturating / classify / scatter / compressstore --
TEST(Sve, CleanNativeBatch) {
  const hs::ScalableTag<std::int32_t> di;
  const std::size_t n = hs::Lanes(di);
  alignas(64) std::int32_t a[kBuf], o[kBuf];
  for (std::size_t i = 0; i < n; ++i) a[i] = static_cast<std::int32_t>(i * 7 + 3);
  auto va = hs::Load(di, a);

  hs::Store(hs::Clamp(va, hs::Set(di, 5), hs::Set(di, 50)), di, o);
  for (std::size_t i = 0; i < n; ++i) {
    std::int32_t e = a[i] < 5 ? 5 : (a[i] > 50 ? 50 : a[i]);
    EXPECT_EQ(o[i], e);
  }
  // BitCast i32 -> f32 -> i32 round-trip
  const hs::ScalableTag<float> df;
  hs::Store(hs::BitCast(di, hs::BitCast(df, va)), di, o);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(o[i], a[i]);

  // SaturatedAdd u8
  const hs::ScalableTag<std::uint8_t> du8;
  const std::size_t n8 = hs::Lanes(du8);
  alignas(64) std::uint8_t p[kBuf], q[kBuf], o8[kBuf];
  for (std::size_t i = 0; i < n8; ++i) { p[i] = static_cast<std::uint8_t>(200 + i % 60); q[i] = static_cast<std::uint8_t>(100 + i % 50); }
  hs::Store(hs::SaturatedAdd(hs::Load(du8, p), hs::Load(du8, q)), du8, o8);
  for (std::size_t i = 0; i < n8; ++i) {
    int s = p[i] + q[i]; if (s > 255) s = 255;
    EXPECT_EQ(o8[i], static_cast<std::uint8_t>(s));
  }

  // classification
  const hs::ScalableTag<float> df2;
  const std::size_t nf = hs::Lanes(df2);
  alignas(64) float ff[kBuf]; for (std::size_t i = 0; i < nf; ++i) ff[i] = static_cast<float>(i);
  ff[0] = std::numeric_limits<float>::quiet_NaN();
  ff[1] = std::numeric_limits<float>::infinity();
  auto vff = hs::Load(df2, ff);
  EXPECT_EQ(hs::CountTrue(df2, hs::IsNaN(vff)), std::size_t{1});
  EXPECT_EQ(hs::CountTrue(df2, hs::IsInf(vff)), std::size_t{1});

  // ScatterIndex
  alignas(64) std::int32_t tab[512] = {};
  alignas(64) std::uint32_t ix[kBuf]; for (std::size_t i = 0; i < n; ++i) ix[i] = static_cast<std::uint32_t>(i * 3);
  const hs::ScalableTag<std::uint32_t> du;
  hs::ScatterIndex(va, di, tab, hs::Load(du, ix));
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(tab[i * 3], a[i]);

  // CompressStore even lanes
  alignas(64) std::int32_t cs[kBuf];
  alignas(64) std::int32_t mb[kBuf]; for (std::size_t i = 0; i < n; ++i) mb[i] = (i % 2 == 0) ? 1 : 0;
  auto mm = hs::Eq(hs::Load(di, mb), hs::Set(di, 1));
  const std::size_t wrote = hs::CompressStore(va, mm, di, cs);
  EXPECT_EQ(wrote, hs::CountTrue(di, mm));
  std::size_t j = 0;
  for (std::size_t i = 0; i < n; ++i) if (i % 2 == 0) EXPECT_EQ(cs[j++], a[i]);
}
