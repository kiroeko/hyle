#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "hyle/platform.h"
#include "hyle/simd/dtype.h"
#include "hyle/simd/dtype_ops.h"

// dtype_test -- fp16/bf16 conversion correctness. Two layers:
//  (1) element conversions (dtype.h) vs known IEEE-754 bit patterns + exhaustive
//      round-trip over ALL 65536 fp16 patterns.
//  (2) batch + dispatch (dtype_ops.h): on an F16C host the batch path runs the HARDWARE
//      kernel; we cross-check it against the scalar element core for every fp16 pattern
//      (finite: bit-equal; NaN: equal-by-class) -- the hardware-vs-software gate.

using hyle::simd::active_dtype_variant;
using hyle::simd::bf16_to_f32;
using hyle::simd::bfloat16_t;
using hyle::simd::f16_to_f32;
using hyle::simd::f16_to_f32_batch;
using hyle::simd::f32_to_bf16;
using hyle::simd::f32_to_f16;
using hyle::simd::f32_to_f16_batch;
using hyle::simd::float16_t;
using hyle::simd::Isa;

namespace {
bool is_nan_f32(float f) { return f != f; }
}  // namespace

// ---- fp16 element: known bit patterns -----------------------------------------------
TEST(Fp16, KnownValues) {
  EXPECT_EQ(f16_to_f32(float16_t::from_bits(0x0000)), 0.0f);   // +0
  EXPECT_EQ(f16_to_f32(float16_t::from_bits(0x3C00)), 1.0f);   // 1.0
  EXPECT_EQ(f16_to_f32(float16_t::from_bits(0xC000)), -2.0f);  // -2.0
  EXPECT_EQ(f16_to_f32(float16_t::from_bits(0x7BFF)), 65504.0f);  // max normal
  EXPECT_TRUE(std::isinf(f16_to_f32(float16_t::from_bits(0x7C00))));   // +inf
  EXPECT_TRUE(std::isinf(f16_to_f32(float16_t::from_bits(0xFC00))));   // -inf
  EXPECT_TRUE(is_nan_f32(f16_to_f32(float16_t::from_bits(0x7E00))));   // qNaN
  // smallest subnormal = 2^-24
  EXPECT_FLOAT_EQ(f16_to_f32(float16_t::from_bits(0x0001)), std::ldexp(1.0f, -24));
  // smallest normal = 2^-14
  EXPECT_FLOAT_EQ(f16_to_f32(float16_t::from_bits(0x0400)), std::ldexp(1.0f, -14));
}

TEST(Fp16, F32ToF16KnownAndRounding) {
  EXPECT_EQ(f32_to_f16(1.0f).bits, 0x3C00);
  EXPECT_EQ(f32_to_f16(-2.0f).bits, 0xC000);
  EXPECT_EQ(f32_to_f16(0.0f).bits, 0x0000);
  EXPECT_EQ(f32_to_f16(-0.0f).bits, 0x8000);
  // overflow -> +inf
  EXPECT_EQ(f32_to_f16(1e30f).bits, 0x7C00);
  EXPECT_EQ(f32_to_f16(-1e30f).bits, 0xFC00);
  // tiny -> signed zero
  EXPECT_EQ(f32_to_f16(1e-30f).bits, 0x0000);
  // NaN stays NaN (mantissa nonzero, exp all-ones)
  const float16_t n = f32_to_f16(std::nanf(""));
  EXPECT_EQ(n.bits & 0x7C00, 0x7C00);
  EXPECT_NE(n.bits & 0x03FF, 0);
}

// f32 -> f16 ROUND-TO-NEAREST-EVEN with hand-computed oracles. The exhaustive round-trip
// test only feeds exactly-representable f32, so it NEVER exercises rounding -- this does.
// fp16 1.0 = 0x3C00 (mantissa 0, even); 0x3C01 = 1+2^-10 (odd); 0x3C02 = 1+2^-9 (even).
TEST(Fp16, F32ToF16RoundToNearestEven) {
  const float ulp = std::ldexp(1.0f, -10);   // fp16 ulp at exponent 0
  const float half = std::ldexp(1.0f, -11);  // half-ulp
  // exact tie between 0x3C00(even) and 0x3C01(odd) -> rounds to EVEN = 0x3C00
  EXPECT_EQ(f32_to_f16(1.0f + half).bits, 0x3C00);
  // exact tie between 0x3C01(odd) and 0x3C02(even) -> rounds to EVEN = 0x3C02
  EXPECT_EQ(f32_to_f16(1.0f + ulp + half).bits, 0x3C02);
  // not a tie: 0.75 of the way up -> rounds toward nearer = 0x3C01
  EXPECT_EQ(f32_to_f16(1.0f + ulp * 0.75f).bits, 0x3C01);
  // not a tie: 0.25 of the way up -> rounds down = 0x3C00
  EXPECT_EQ(f32_to_f16(1.0f + ulp * 0.25f).bits, 0x3C00);
}

// Exhaustive: every fp16 bit pattern survives f16->f32->f16 unchanged (f32 represents
// every fp16 exactly; NaN stays NaN though its mantissa may canonicalize to 0x3FF).
TEST(Fp16, ExhaustiveRoundTrip) {
  for (std::uint32_t b = 0; b <= 0xFFFF; ++b) {
    const auto h = float16_t::from_bits(static_cast<std::uint16_t>(b));
    const float f = f16_to_f32(h);
    const float16_t r = f32_to_f16(f);
    const bool is_nan16 = ((b & 0x7C00) == 0x7C00) && ((b & 0x03FF) != 0);
    if (is_nan16) {
      EXPECT_EQ(r.bits & 0x7C00, 0x7C00) << "b=" << b;       // still inf-exp
      EXPECT_NE(r.bits & 0x03FF, 0) << "b=" << b;            // still nan (mantissa!=0)
    } else {
      EXPECT_EQ(r.bits, static_cast<std::uint16_t>(b)) << "b=" << b;
    }
  }
}

// ---- bf16 element -------------------------------------------------------------------
TEST(Bf16, KnownAndShiftSemantics) {
  // bf16 is the top 16 bits of fp32: 1.0f = 0x3F800000 -> 0x3F80.
  EXPECT_EQ(f32_to_bf16(1.0f).bits, 0x3F80);
  EXPECT_EQ(f32_to_bf16(-2.0f).bits, 0xC000);
  EXPECT_FLOAT_EQ(bf16_to_f32(bfloat16_t::from_bits(0x3F80)), 1.0f);
  // promote(demote(x)) for a value exactly representable in bf16 is exact
  EXPECT_FLOAT_EQ(bf16_to_f32(f32_to_bf16(1.5f)), 1.5f);
}

TEST(Bf16, RoundToNearestEvenAndNan) {
  // Round-half-to-even at the bf16 boundary. 0x3F808000 is exactly halfway between
  // 0x3F80 and 0x3F81; RNE -> even (0x3F80).
  const float halfway = [] { std::uint32_t u = 0x3F808000u; float f; std::memcpy(&f, &u, 4); return f; }();
  EXPECT_EQ(f32_to_bf16(halfway).bits, 0x3F80);
  // just above halfway rounds up
  const float above = [] { std::uint32_t u = 0x3F808001u; float f; std::memcpy(&f, &u, 4); return f; }();
  EXPECT_EQ(f32_to_bf16(above).bits, 0x3F81);
  // NaN stays NaN (must not collapse to inf)
  const bfloat16_t n = f32_to_bf16(std::nanf(""));
  EXPECT_EQ(n.bits & 0x7F80, 0x7F80);
  EXPECT_NE(n.bits & 0x007F, 0);
}

// ---- batch + dispatch: hardware (F16C) vs scalar core -------------------------------

// On an F16C host the batch path runs the hardware kernel; it must agree with the scalar
// element core for EVERY fp16 pattern (finite bit-equal; NaN equal-by-class). This is the
// hardware-vs-software cross-check the milestone hinges on.
TEST(DtypeBatch, F16ToF32HardwareMatchesScalar) {
#if defined(HYLE_ARCH_X64)
  if (!hyle::simd::cpu_features().f16c)
    GTEST_SKIP() << "no F16C on this host -- batch would fall back to scalar (tautology)";
#elif defined(HYLE_ARCH_ARM64)
  // aarch64: batch runs the NEON native-vcvt kernel (ARMv8.2 FP16) -- validate it here.
#else
  GTEST_SKIP() << "unknown arch -- no hardware fp16 path";
#endif
  std::vector<float16_t> src(0x10000);
  for (std::uint32_t b = 0; b <= 0xFFFF; ++b) src[b] = float16_t::from_bits(static_cast<std::uint16_t>(b));
  std::vector<float> got(0x10000);
  f16_to_f32_batch(src.data(), got.data(), src.size());  // hardware on F16C host
  for (std::uint32_t b = 0; b <= 0xFFFF; ++b) {
    const float ref = f16_to_f32(src[b]);  // scalar oracle
    if (is_nan_f32(ref)) {
      EXPECT_TRUE(is_nan_f32(got[b])) << "b=" << b;
    } else {
      EXPECT_EQ(std::memcmp(&got[b], &ref, 4), 0) << "b=" << b;  // bit-equal
    }
  }
}

// f32 -> f16 hardware vs scalar over a wide sweep of f32 inputs (finite bit-equal).
TEST(DtypeBatch, F32ToF16HardwareMatchesScalar) {
#if defined(HYLE_ARCH_X64)
  if (!hyle::simd::cpu_features().f16c)
    GTEST_SKIP() << "no F16C on this host -- batch would fall back to scalar (tautology)";
#elif defined(HYLE_ARCH_ARM64)
  // aarch64: batch runs the NEON native-vcvt narrow -- validate it (finite sweep, RNE).
#else
  GTEST_SKIP() << "unknown arch -- no hardware fp16 path";
#endif
  std::vector<float> src;
  for (int e = -20; e <= 20; ++e)
    for (int m = 0; m < 64; ++m)
      src.push_back(std::ldexp(1.0f + static_cast<float>(m) / 64.0f, e) * (m & 1 ? -1 : 1));
  src.push_back(0.0f); src.push_back(-0.0f); src.push_back(65504.0f); src.push_back(1e30f);
  std::vector<float16_t> got(src.size());
  f32_to_f16_batch(src.data(), got.data(), src.size());  // hardware on F16C host
  for (std::size_t i = 0; i < src.size(); ++i)
    EXPECT_EQ(got[i].bits, f32_to_f16(src[i]).bits) << "i=" << i << " v=" << src[i];
}

// Batch handles a tail (n not a multiple of 8).
TEST(DtypeBatch, TailHandled) {
  for (std::size_t n : {std::size_t{1}, std::size_t{7}, std::size_t{8}, std::size_t{9}, std::size_t{31}}) {
    std::vector<float> src(n);
    for (std::size_t i = 0; i < n; ++i) src[i] = 0.5f + static_cast<float>(i);
    std::vector<float16_t> h(n);
    std::vector<float> back(n);
    f32_to_f16_batch(src.data(), h.data(), n);
    f16_to_f32_batch(h.data(), back.data(), n);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_FLOAT_EQ(back[i], f16_to_f32(f32_to_f16(src[i]))) << "n=" << n << " i=" << i;
  }
}

TEST(DtypeBatch, Bf16RoundTrip) {
  std::vector<float> src{1.0f, -2.0f, 3.14159f, 1e10f, -0.001f, 0.0f};
  std::vector<bfloat16_t> b(src.size());
  std::vector<float> back(src.size());
  f32_to_bf16_batch(src.data(), b.data(), src.size());
  bf16_to_f32_batch(b.data(), back.data(), src.size());
  for (std::size_t i = 0; i < src.size(); ++i) {
    const float ref = bf16_to_f32(f32_to_bf16(src[i]));  // element-wise oracle
    EXPECT_EQ(back[i], ref) << "i=" << i;
  }
}

// bf16 AVX2 kernel vs scalar oracle over ALL 0x10000 bf16 patterns. bf16->f32 is a pure
// shift (no rounding), so the AVX2 path is BIT-IDENTICAL to scalar -- NaN payloads too
// (unlike F16C fp16). Exact equality everywhere.
TEST(DtypeBatch, Bf16ToF32Avx2MatchesScalarExact) {
#if defined(HYLE_ARCH_X64)
  if (!hyle::simd::cpu_features().avx2)
    GTEST_SKIP() << "no AVX2 on this host -- batch falls back to scalar (tautology)";
#elif defined(HYLE_ARCH_ARM64)
  // aarch64: batch runs the NEON integer bf16 kernel -- bit-identical to scalar, validate.
#else
  GTEST_SKIP() << "unknown arch -- no vectorized bf16 path";
#endif
  std::vector<bfloat16_t> src(0x10000);
  for (std::uint32_t b = 0; b <= 0xFFFF; ++b)
    src[b] = bfloat16_t::from_bits(static_cast<std::uint16_t>(b));
  std::vector<float> got(0x10000);
  bf16_to_f32_batch(src.data(), got.data(), src.size());  // AVX2 on an AVX2 host
  for (std::uint32_t b = 0; b <= 0xFFFF; ++b) {
    const float ref = bf16_to_f32(src[b]);  // scalar oracle
    EXPECT_EQ(std::memcmp(&got[b], &ref, 4), 0) << "b=" << b;  // bit-equal (NaN incl.)
  }
}

// f32 -> bf16 AVX2 vs scalar over a wide f32 sweep incl. inf/NaN(s/q)/ties/±0. The convert
// is integer RNE bit-math, so AVX2 must match scalar EXACTLY on the bf16 bits everywhere.
TEST(DtypeBatch, F32ToBf16Avx2MatchesScalarExact) {
#if defined(HYLE_ARCH_X64)
  if (!hyle::simd::cpu_features().avx2)
    GTEST_SKIP() << "no AVX2 on this host -- batch falls back to scalar (tautology)";
#elif defined(HYLE_ARCH_ARM64)
  // aarch64: batch runs the NEON integer RNE bf16 kernel -- bit-identical to scalar.
#else
  GTEST_SKIP() << "unknown arch -- no vectorized bf16 path";
#endif
  std::vector<float> src;
  for (int e = -30; e <= 30; ++e)
    for (int m = 0; m < 128; ++m)
      src.push_back(std::ldexp(1.0f + static_cast<float>(m) / 128.0f, e) * (m & 1 ? -1 : 1));
  // ties (low f32 mantissa bits exactly at the bf16 rounding point) + specials by raw bits.
  for (std::uint32_t bits : {0x00000000u, 0x80000000u, 0x7F800000u, 0xFF800000u,
                             0x7FC00000u, 0x7F800001u, 0x00008000u, 0x00018000u,
                             0x3F808000u, 0x3F818000u}) {
    float f;
    std::memcpy(&f, &bits, 4);
    src.push_back(f);
  }
  std::vector<bfloat16_t> got(src.size());
  f32_to_bf16_batch(src.data(), got.data(), src.size());  // AVX2 on an AVX2 host
  for (std::size_t i = 0; i < src.size(); ++i)
    EXPECT_EQ(got[i].bits, f32_to_bf16(src[i]).bits) << "i=" << i << " v=" << src[i];
}

// f32 -> bf16 NATIVE AVX-512-BF16 (VCVTNEPS2BF16) vs scalar oracle. On a host with the
// avx512bf16 feature the batch path runs the hardware convert; for FINITE inputs hardware
// RNE matches the scalar oracle bit-for-bit, but for NaN the hardware quiets to its own
// QNaN pattern (need not match the oracle's mantissa), so NaN is checked equal-by-CLASS
// (stays NaN), finite is checked bit-exact -- same stance as the F16C fp16 cross-check.
// SKIPS on a CPU without avx512bf16 (this dev box is AVX2-only) -- there the batch falls to
// the AVX2 path (already covered bit-exact above), so running it here would be a tautology.
TEST(DtypeBatch, F32ToBf16Avx512MatchesScalar) {
#if defined(HYLE_ARCH_X64)
  if (!hyle::simd::cpu_features().avx512bf16)
    GTEST_SKIP() << "no AVX-512-BF16 on this host -- batch uses the AVX2 path (covered above)";
#else
  GTEST_SKIP() << "non-x86 -- no AVX-512-BF16 path";
#endif
  std::vector<float> src;
  for (int e = -30; e <= 30; ++e)
    for (int m = 0; m < 128; ++m)
      src.push_back(std::ldexp(1.0f + static_cast<float>(m) / 128.0f, e) * (m & 1 ? -1 : 1));
  for (std::uint32_t bits : {0x00000000u, 0x80000000u, 0x7F800000u, 0xFF800000u,
                             0x7FC00000u, 0x7F800001u, 0x00008000u, 0x00018000u,
                             0x3F808000u, 0x3F818000u}) {
    float f;
    std::memcpy(&f, &bits, 4);
    src.push_back(f);
  }
  std::vector<bfloat16_t> got(src.size());
  f32_to_bf16_batch(src.data(), got.data(), src.size());  // VCVTNEPS2BF16 on a bf16 host
  for (std::size_t i = 0; i < src.size(); ++i) {
    const bfloat16_t ref = f32_to_bf16(src[i]);  // scalar oracle
    const bool ref_nan = (ref.bits & 0x7F80) == 0x7F80 && (ref.bits & 0x007F) != 0;
    if (ref_nan) {
      // equal-by-class: hardware may pick a different QNaN mantissa, must still be NaN
      EXPECT_EQ(got[i].bits & 0x7F80, 0x7F80) << "i=" << i << " v=" << src[i];
      EXPECT_NE(got[i].bits & 0x007F, 0) << "i=" << i << " v=" << src[i];
    } else {
      EXPECT_EQ(got[i].bits, ref.bits) << "i=" << i << " v=" << src[i];  // finite: bit-exact
    }
  }
}

// On an x86 host with F16C the dispatcher must engage the hardware tier (not the floor).
TEST(DtypeBatch, ActiveVariantEngagesHardwareWhenAvailable) {
#if defined(HYLE_ARCH_X64)
  if (hyle::simd::cpu_features().f16c)
    EXPECT_EQ(active_dtype_variant(), Isa::kAvx2);
  else
    EXPECT_EQ(active_dtype_variant(), Isa::kScalar);
#elif defined(HYLE_ARCH_ARM64)
  // aarch64: fp16 via native vcvt + bf16 via NEON integer are ALWAYS compiled (NEON is the
  // ARMv8 baseline), so the dtype dispatcher engages the NEON tier, never the scalar floor.
  EXPECT_EQ(active_dtype_variant(), Isa::kNeon);
#else
  EXPECT_EQ(active_dtype_variant(), Isa::kScalar);
#endif
}
