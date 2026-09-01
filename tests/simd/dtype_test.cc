#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#include "hyle/core/environment.h"
#include "hyle/simd/dtype.h"
#include "hyle/simd/dtype_ops.h"
#include "src/simd/runtime_internal.h"

// dtype_test -- public storage conversion correctness. Two layers:
//  (1) element conversions (dtype.h) vs known IEEE-754 bit patterns + exhaustive
//      round-trip over ALL 65536 fp16 patterns.
//  (2) whole-buffer conversion (dtype_ops.h): every compiled runtime table and the
//      selected public route are checked against the scalar element contract.

using hyle::simd::bf16_to_f32;
using hyle::simd::bfloat16_t;
using hyle::simd::f16_to_f32;
using hyle::simd::convert;
using hyle::simd::f32_to_bf16;
using hyle::simd::f32_to_f16;
using hyle::simd::float16_t;
using hyle::simd::Isa;
using hyle::simd::f8e4m3_to_f32;
using hyle::simd::f32_to_f8e4m3;
using hyle::simd::float8_e4m3_t;
using hyle::simd::f8e5m2_to_f32;
using hyle::simd::f32_to_f8e5m2;
using hyle::simd::float8_e5m2_t;

namespace {
bool is_nan_f32(float f) { return f != f; }

template <typename T>
float decode_storage(T value) {
  if constexpr (std::is_same_v<T, float>) return value;
  if constexpr (std::is_same_v<T, float16_t>) return f16_to_f32(value);
  if constexpr (std::is_same_v<T, bfloat16_t>) return bf16_to_f32(value);
  if constexpr (std::is_same_v<T, float8_e4m3_t>) return f8e4m3_to_f32(value);
  if constexpr (std::is_same_v<T, float8_e5m2_t>) return f8e5m2_to_f32(value);
}

template <typename T>
T encode_storage(float value) {
  if constexpr (std::is_same_v<T, float>) return value;
  if constexpr (std::is_same_v<T, float16_t>) return f32_to_f16(value);
  if constexpr (std::is_same_v<T, bfloat16_t>) return f32_to_bf16(value);
  if constexpr (std::is_same_v<T, float8_e4m3_t>) return f32_to_f8e4m3(value);
  if constexpr (std::is_same_v<T, float8_e5m2_t>) return f32_to_f8e5m2(value);
}

template <typename T>
bool same_storage_value(T actual, T expected) {
  if (is_nan_f32(decode_storage(expected)))
    return is_nan_f32(decode_storage(actual));
  if constexpr (std::is_same_v<T, float>)
    return std::bit_cast<std::uint32_t>(actual) ==
           std::bit_cast<std::uint32_t>(expected);
  else
    return actual.bits == expected.bits;
}

template <typename Source, typename Destination>
void expect_conversion(const std::vector<Source>& source) {
  std::vector<Destination> actual(source.size());
  convert(source.data(), actual.data(), source.size());
  for (std::size_t i = 0; i < source.size(); ++i) {
    const Destination expected =
        encode_storage<Destination>(decode_storage(source[i]));
    EXPECT_TRUE(same_storage_value(actual[i], expected)) << "i=" << i;
  }
}

template <typename Source, typename Destination>
void expect_zero_conversion() {
  convert(static_cast<const Source*>(nullptr),
          static_cast<Destination*>(nullptr), 0);
}

std::vector<float> f32_conversion_inputs() {
  std::vector<float> values;
  for (int exponent = -30; exponent <= 30; ++exponent) {
    for (int mantissa = 0; mantissa < 64; ++mantissa) {
      values.push_back(std::ldexp(1.0f + static_cast<float>(mantissa) / 64.0f,
                                  exponent) *
                       (mantissa & 1 ? -1.0f : 1.0f));
    }
  }
  for (std::uint32_t bits :
       {0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u,
        0x7F7FFFFFu, 0xFF7FFFFFu, 0x7F800000u, 0xFF800000u,
        0x7FC00000u, 0x7F800001u, 0x3F808000u, 0x3F818000u}) {
    values.push_back(std::bit_cast<float>(bits));
  }
  return values;
}

template <typename T>
std::vector<T> all_storage_patterns() {
  constexpr std::size_t kCount = sizeof(T) == 1 ? 256 : 65536;
  std::vector<T> values(kCount);
  for (std::size_t bits = 0; bits < kCount; ++bits) {
    if constexpr (sizeof(T) == 1)
      values[bits] = T::from_bits(static_cast<std::uint8_t>(bits));
    else
      values[bits] = T::from_bits(static_cast<std::uint16_t>(bits));
  }
  return values;
}
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

// ---- whole-buffer conversion + dispatch vs scalar core -----------------------------

// The public selected route must agree with the element core for every fp16
// pattern. This remains meaningful for fat and exact artifacts alike.
TEST(DtypeBatch, F16ToF32SelectedRouteMatchesScalar) {
  std::vector<float16_t> src(0x10000);
  for (std::uint32_t b = 0; b <= 0xFFFF; ++b) src[b] = float16_t::from_bits(static_cast<std::uint16_t>(b));
  std::vector<float> got(0x10000);
  convert(src.data(), got.data(), src.size());
  for (std::uint32_t b = 0; b <= 0xFFFF; ++b) {
    const float ref = f16_to_f32(src[b]);  // scalar oracle
    if (is_nan_f32(ref)) {
      EXPECT_TRUE(is_nan_f32(got[b])) << "b=" << b;
    } else {
      EXPECT_EQ(std::memcmp(&got[b], &ref, 4), 0) << "b=" << b;  // bit-equal
    }
  }
}

// f32 -> f16 selected route vs scalar over a wide input sweep.
TEST(DtypeBatch, F32ToF16SelectedRouteMatchesScalar) {
  std::vector<float> src;
  for (int e = -20; e <= 20; ++e)
    for (int m = 0; m < 64; ++m)
      src.push_back(std::ldexp(1.0f + static_cast<float>(m) / 64.0f, e) * (m & 1 ? -1 : 1));
  src.push_back(0.0f); src.push_back(-0.0f); src.push_back(65504.0f); src.push_back(1e30f);
  std::vector<float16_t> got(src.size());
  convert(src.data(), got.data(), src.size());
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
    convert(src.data(), h.data(), n);
    convert(h.data(), back.data(), n);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_FLOAT_EQ(back[i], f16_to_f32(f32_to_f16(src[i]))) << "n=" << n << " i=" << i;
  }
}

TEST(DtypeBatch, Bf16RoundTrip) {
  std::vector<float> src{1.0f, -2.0f, 3.14159f, 1e10f, -0.001f, 0.0f};
  std::vector<bfloat16_t> b(src.size());
  std::vector<float> back(src.size());
  convert(src.data(), b.data(), src.size());
  convert(b.data(), back.data(), src.size());
  for (std::size_t i = 0; i < src.size(); ++i) {
    const float ref = bf16_to_f32(f32_to_bf16(src[i]));  // element-wise oracle
    EXPECT_EQ(back[i], ref) << "i=" << i;
  }
}

// bf16 -> f32 is a pure expansion, so every selected target must preserve all
// 65,536 encodings bit-for-bit, including NaN payloads.
TEST(DtypeBatch, Bf16ToF32SelectedRouteMatchesScalarExact) {
  std::vector<bfloat16_t> src(0x10000);
  for (std::uint32_t b = 0; b <= 0xFFFF; ++b)
    src[b] = bfloat16_t::from_bits(static_cast<std::uint16_t>(b));
  std::vector<float> got(0x10000);
  convert(src.data(), got.data(), src.size());
  for (std::uint32_t b = 0; b <= 0xFFFF; ++b) {
    const float ref = bf16_to_f32(src[b]);  // scalar oracle
    EXPECT_EQ(std::memcmp(&got[b], &ref, 4), 0) << "b=" << b;  // bit-equal (NaN incl.)
  }
}

// f32 -> bf16 selected route vs scalar over finite values, infinities, NaNs,
// ties and signed zero. The public RNE/NaN contract is bit-exact.
TEST(DtypeBatch, F32ToBf16SelectedRouteMatchesScalarExact) {
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
  convert(src.data(), got.data(), src.size());  // AVX2 on an AVX2 host
  for (std::size_t i = 0; i < src.size(); ++i)
    EXPECT_EQ(got[i].bits, f32_to_bf16(src[i]).bits) << "i=" << i << " v=" << src[i];
}

// f32 -> bf16 AVX-512-BF16 specialization vs the scalar oracle. The product
// patches hardware NaN payloads back to HYLE's public conversion contract, so
// every result is bit-exact. Skip unless the selected artifact actually routes
// through the specialization (CPU support alone is insufficient for exact builds).
TEST(DtypeBatch, F32ToBf16Avx512MatchesScalar) {
#if HYLE_ARCH_X64
  if ((hyle::simd::runtime().dtype.features &
       hyle::simd::Runtime::kFeatureAvx512Bf16) == 0)
    GTEST_SKIP() << "selected artifact does not use AVX-512-BF16";
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
  convert(src.data(), got.data(), src.size());  // VCVTNEPS2BF16 on a bf16 host
  for (std::size_t i = 0; i < src.size(); ++i)
    EXPECT_EQ(got[i].bits, f32_to_bf16(src[i]).bits)
        << "i=" << i << " v=" << src[i];
}

// Fat artifacts may add precise-feature specializations from the host probe;
// exact artifacts expose only features owned by their sole compiled product tier.
TEST(DtypeBatch, ActiveVariantEngagesHardwareWhenAvailable) {
#if HYLE_ARCH_X64
#if defined(HYLE_SIMD_EXACT)
  const bool tier_owns_f16c = hyle::simd::runtime().isa == Isa::kAvx2 ||
                              hyle::simd::runtime().isa == Isa::kAvx512;
  EXPECT_EQ((hyle::simd::runtime().dtype.features &
             hyle::simd::Runtime::kFeatureF16c) != 0,
            tier_owns_f16c);
#else
  if (hyle::platform::cpu_features().f16c) {
    EXPECT_NE(hyle::simd::runtime().dtype.features &
                  hyle::simd::Runtime::kFeatureF16c,
              0u);
  } else {
    EXPECT_EQ(hyle::simd::runtime().dtype.features &
                  hyle::simd::Runtime::kFeatureF16c,
              0u);
  }
#endif
#elif HYLE_ARCH_ARM64
  // AArch64 always has the NEON floor. A fat SVE artifact promotes the common
  // conversion graph to its selected scalable tier instead of stopping at NEON.
  EXPECT_EQ(hyle::simd::runtime().dtype.isa, hyle::simd::runtime().isa);
#elif HYLE_ARCH_RISCV64
  // A compiled RVV product owns the common conversion graph directly.
  EXPECT_EQ(hyle::simd::runtime().dtype.isa, hyle::simd::runtime().isa);
#else
  EXPECT_EQ(hyle::simd::runtime().dtype.isa, Isa::kScalar);
#endif
}

TEST(DtypeGraph, CoversEveryDirectedStorageEdge) {
  const auto f32 = f32_conversion_inputs();
  expect_conversion<float, float16_t>(f32);
  expect_conversion<float, bfloat16_t>(f32);
  expect_conversion<float, float8_e4m3_t>(f32);
  expect_conversion<float, float8_e5m2_t>(f32);

  const auto f16 = all_storage_patterns<float16_t>();
  expect_conversion<float16_t, float>(f16);
  expect_conversion<float16_t, bfloat16_t>(f16);
  expect_conversion<float16_t, float8_e4m3_t>(f16);
  expect_conversion<float16_t, float8_e5m2_t>(f16);

  const auto bf16 = all_storage_patterns<bfloat16_t>();
  expect_conversion<bfloat16_t, float>(bf16);
  expect_conversion<bfloat16_t, float16_t>(bf16);
  expect_conversion<bfloat16_t, float8_e4m3_t>(bf16);
  expect_conversion<bfloat16_t, float8_e5m2_t>(bf16);

  const auto e4m3 = all_storage_patterns<float8_e4m3_t>();
  expect_conversion<float8_e4m3_t, float>(e4m3);
  expect_conversion<float8_e4m3_t, float16_t>(e4m3);
  expect_conversion<float8_e4m3_t, bfloat16_t>(e4m3);
  expect_conversion<float8_e4m3_t, float8_e5m2_t>(e4m3);

  const auto e5m2 = all_storage_patterns<float8_e5m2_t>();
  expect_conversion<float8_e5m2_t, float>(e5m2);
  expect_conversion<float8_e5m2_t, float16_t>(e5m2);
  expect_conversion<float8_e5m2_t, bfloat16_t>(e5m2);
  expect_conversion<float8_e5m2_t, float8_e4m3_t>(e5m2);
}

TEST(DtypeGraph, CommonConversionsMatchScalarOnEveryRuntimeTier) {
  const auto f16 = all_storage_patterns<float16_t>();
  const auto bf16 = all_storage_patterns<bfloat16_t>();
  std::vector<float> f16_wide(f16.size());
  std::vector<float> bf16_wide(bf16.size());
  for (std::size_t i = 0; i < f16.size(); ++i) {
    f16_wide[i] = f16_to_f32(f16[i]);
    bf16_wide[i] = bf16_to_f32(bf16[i]);
  }
  std::vector<float> f16_f32_out(f16.size());
  std::vector<float> bf16_f32_out(bf16.size());
  std::vector<float16_t> f32_f16_out(f16.size());
  std::vector<bfloat16_t> f32_bf16_out(bf16.size());
  std::vector<bfloat16_t> bf16_out(f16.size());
  std::vector<float16_t> f16_out(bf16.size());
  for (const Isa isa : {Isa::kScalar, Isa::kSse42, Isa::kAvx2,
                         Isa::kAvx512, Isa::kNeon, Isa::kSve, Isa::kRvv}) {
    const hyle::simd::Runtime* runtime = hyle::simd::runtime_for(isa);
    if (runtime == nullptr) continue;
    SCOPED_TRACE(static_cast<int>(isa));
    runtime->dtype_convert.f16_to_f32(f16.data(), f16_f32_out.data(),
                                      f16.size());
    runtime->dtype_convert.f32_to_f16(f16_wide.data(), f32_f16_out.data(),
                                      f16_wide.size());
    runtime->dtype_convert.bf16_to_f32(bf16.data(), bf16_f32_out.data(),
                                       bf16.size());
    runtime->dtype_convert.f32_to_bf16(bf16_wide.data(), f32_bf16_out.data(),
                                       bf16_wide.size());
    runtime->dtype_convert.f16_to_bf16(f16.data(), bf16_out.data(),
                                       f16.size());
    runtime->dtype_convert.bf16_to_f16(bf16.data(), f16_out.data(),
                                       bf16.size());
    for (std::size_t i = 0; i < f16.size(); ++i) {
      EXPECT_EQ(std::bit_cast<std::uint32_t>(f16_f32_out[i]),
                std::bit_cast<std::uint32_t>(f16_wide[i]))
          << "f16 widen bits=" << i;
      EXPECT_EQ(f32_f16_out[i].bits, f32_to_f16(f16_wide[i]).bits)
          << "f16 narrow bits=" << i;
      EXPECT_EQ(std::bit_cast<std::uint32_t>(bf16_f32_out[i]),
                std::bit_cast<std::uint32_t>(bf16_wide[i]))
          << "bf16 widen bits=" << i;
      EXPECT_EQ(f32_bf16_out[i].bits, f32_to_bf16(bf16_wide[i]).bits)
          << "bf16 narrow bits=" << i;
      EXPECT_EQ(bf16_out[i].bits,
                f32_to_bf16(f16_to_f32(f16[i])).bits)
          << "f16->bf16 bits=" << i;
      EXPECT_EQ(f16_out[i].bits, f32_to_f16(bf16_to_f32(bf16[i])).bits)
          << "bf16->f16 bits=" << i;
    }
  }
}

TEST(DtypeGraph, ZeroAcceptsNullForEveryDirectedEdge) {
  expect_zero_conversion<float, float16_t>();
  expect_zero_conversion<float, bfloat16_t>();
  expect_zero_conversion<float, float8_e4m3_t>();
  expect_zero_conversion<float, float8_e5m2_t>();
  expect_zero_conversion<float16_t, float>();
  expect_zero_conversion<float16_t, bfloat16_t>();
  expect_zero_conversion<float16_t, float8_e4m3_t>();
  expect_zero_conversion<float16_t, float8_e5m2_t>();
  expect_zero_conversion<bfloat16_t, float>();
  expect_zero_conversion<bfloat16_t, float16_t>();
  expect_zero_conversion<bfloat16_t, float8_e4m3_t>();
  expect_zero_conversion<bfloat16_t, float8_e5m2_t>();
  expect_zero_conversion<float8_e4m3_t, float>();
  expect_zero_conversion<float8_e4m3_t, float16_t>();
  expect_zero_conversion<float8_e4m3_t, bfloat16_t>();
  expect_zero_conversion<float8_e4m3_t, float8_e5m2_t>();
  expect_zero_conversion<float8_e5m2_t, float>();
  expect_zero_conversion<float8_e5m2_t, float16_t>();
  expect_zero_conversion<float8_e5m2_t, bfloat16_t>();
  expect_zero_conversion<float8_e5m2_t, float8_e4m3_t>();
}

TEST(DtypeGraph, EqualWidthExactAliasIsSupported) {
  alignas(std::uint16_t) std::array<std::byte, 31 * sizeof(std::uint16_t)>
      wide_storage{};
  std::array<float16_t, 31> f16{};
  for (std::size_t i = 0; i < f16.size(); ++i)
    f16[i] = float16_t::from_bits(static_cast<std::uint16_t>(i * 2003));
  std::memcpy(wide_storage.data(), f16.data(), sizeof(f16));
  convert(reinterpret_cast<const float16_t*>(wide_storage.data()),
          reinterpret_cast<bfloat16_t*>(wide_storage.data()), f16.size());
  std::array<bfloat16_t, 31> bf16{};
  std::memcpy(bf16.data(), wide_storage.data(), sizeof(bf16));
  for (std::size_t i = 0; i < f16.size(); ++i) {
    const auto expected = f32_to_bf16(f16_to_f32(f16[i]));
    EXPECT_TRUE(same_storage_value(bf16[i], expected)) << "i=" << i;
  }
  convert(reinterpret_cast<const bfloat16_t*>(wide_storage.data()),
          reinterpret_cast<float16_t*>(wide_storage.data()), bf16.size());
  std::array<float16_t, 31> f16_back{};
  std::memcpy(f16_back.data(), wide_storage.data(), sizeof(f16_back));
  for (std::size_t i = 0; i < bf16.size(); ++i) {
    const auto expected = f32_to_f16(bf16_to_f32(bf16[i]));
    EXPECT_TRUE(same_storage_value(f16_back[i], expected)) << "i=" << i;
  }

  alignas(std::uint8_t) std::array<std::byte, 31> byte_storage{};
  std::array<float8_e4m3_t, 31> e4m3{};
  for (std::size_t i = 0; i < e4m3.size(); ++i)
    e4m3[i] = float8_e4m3_t::from_bits(static_cast<std::uint8_t>(i * 7));
  std::memcpy(byte_storage.data(), e4m3.data(), e4m3.size());
  convert(reinterpret_cast<const float8_e4m3_t*>(byte_storage.data()),
          reinterpret_cast<float8_e5m2_t*>(byte_storage.data()), e4m3.size());
  std::array<float8_e5m2_t, 31> e5m2{};
  std::memcpy(e5m2.data(), byte_storage.data(), e5m2.size());
  for (std::size_t i = 0; i < e4m3.size(); ++i) {
    const auto expected = f32_to_f8e5m2(f8e4m3_to_f32(e4m3[i]));
    EXPECT_TRUE(same_storage_value(e5m2[i], expected)) << "i=" << i;
  }
  convert(reinterpret_cast<const float8_e5m2_t*>(byte_storage.data()),
          reinterpret_cast<float8_e4m3_t*>(byte_storage.data()), e5m2.size());
  std::array<float8_e4m3_t, 31> e4m3_back{};
  std::memcpy(e4m3_back.data(), byte_storage.data(), e4m3_back.size());
  for (std::size_t i = 0; i < e5m2.size(); ++i) {
    const auto expected = f32_to_f8e4m3(f8e5m2_to_f32(e5m2[i]));
    EXPECT_TRUE(same_storage_value(e4m3_back[i], expected)) << "i=" << i;
  }
}

// ---- fp8 (OCP E4M3FN / E5M2) --------------------------------------------------------
// Encoding transcribed from CUDA cuda_fp8.hpp. Known values pin the format-specific rules
// (E4M3 max normal 448, no inf, sole NaN 0x7F; E5M2 max normal 57344, IEEE inf/NaN).

TEST(Fp8E4M3, KnownValues) {
  // decode
  EXPECT_EQ(f8e4m3_to_f32(float8_e4m3_t::from_bits(0x00)), 0.0f);   // +0
  EXPECT_EQ(f8e4m3_to_f32(float8_e4m3_t::from_bits(0x80)), 0.0f);   // -0 (compares equal to 0)
  EXPECT_EQ(f8e4m3_to_f32(float8_e4m3_t::from_bits(0x38)), 1.0f);   // 1.0 (exp=7,man=0)
  EXPECT_EQ(f8e4m3_to_f32(float8_e4m3_t::from_bits(0xB8)), -1.0f);  // -1.0
  EXPECT_EQ(f8e4m3_to_f32(float8_e4m3_t::from_bits(0x7E)), 448.0f);      // max normal
  EXPECT_EQ(f8e4m3_to_f32(float8_e4m3_t::from_bits(0xFE)), -448.0f);
  EXPECT_TRUE(is_nan_f32(f8e4m3_to_f32(float8_e4m3_t::from_bits(0x7F))));  // sole NaN
  EXPECT_TRUE(is_nan_f32(f8e4m3_to_f32(float8_e4m3_t::from_bits(0xFF))));
  // smallest subnormal = 2^-9 (exp field 0, mant 1: 2^(1-7) * 1/8 = 2^-9)
  EXPECT_FLOAT_EQ(f8e4m3_to_f32(float8_e4m3_t::from_bits(0x01)), std::ldexp(1.0f, -9));
  // encode
  EXPECT_EQ(f32_to_f8e4m3(1.0f).bits, 0x38);
  EXPECT_EQ(f32_to_f8e4m3(-1.0f).bits, 0xB8);
  EXPECT_EQ(f32_to_f8e4m3(448.0f).bits, 0x7E);
  EXPECT_EQ(f32_to_f8e4m3(0.0f).bits, 0x00);
  // overflow saturates to max normal (OCP SATFINITE), NOT inf (E4M3 has none)
  EXPECT_EQ(f32_to_f8e4m3(1e30f).bits, 0x7E);
  EXPECT_EQ(f32_to_f8e4m3(-1e30f).bits, 0xFE);
  // NaN -> canonical 0x7F
  EXPECT_EQ(f32_to_f8e4m3(std::nanf("")).bits & 0x7F, 0x7F);
}

TEST(Fp8E5M2, KnownValues) {
  EXPECT_EQ(f8e5m2_to_f32(float8_e5m2_t::from_bits(0x00)), 0.0f);
  EXPECT_EQ(f8e5m2_to_f32(float8_e5m2_t::from_bits(0x3C)), 1.0f);   // exp=15,man=0
  EXPECT_EQ(f8e5m2_to_f32(float8_e5m2_t::from_bits(0xBC)), -1.0f);
  EXPECT_EQ(f8e5m2_to_f32(float8_e5m2_t::from_bits(0x7B)), 57344.0f);  // max normal
  EXPECT_TRUE(std::isinf(f8e5m2_to_f32(float8_e5m2_t::from_bits(0x7C))));  // +inf (exp all-1, man 0)
  EXPECT_TRUE(std::isinf(f8e5m2_to_f32(float8_e5m2_t::from_bits(0xFC))));  // -inf
  EXPECT_TRUE(is_nan_f32(f8e5m2_to_f32(float8_e5m2_t::from_bits(0x7E))));  // NaN (man != 0)
  // smallest subnormal = 2^-16 (exp 0, mant 1: 2^(1-15) * 1/4 = 2^-16)
  EXPECT_FLOAT_EQ(f8e5m2_to_f32(float8_e5m2_t::from_bits(0x01)), std::ldexp(1.0f, -16));
  EXPECT_EQ(f32_to_f8e5m2(1.0f).bits, 0x3C);
  EXPECT_EQ(f32_to_f8e5m2(57344.0f).bits, 0x7B);
  EXPECT_EQ(f32_to_f8e5m2(0.0f).bits, 0x00);
  EXPECT_EQ(f32_to_f8e5m2(1e30f).bits, 0x7B);  // saturate to max normal
  EXPECT_EQ(f32_to_f8e5m2(std::nanf("")).bits & 0x7F, 0x7E);  // canonical NaN
}

// Exhaustive: fp8 has only 256 patterns, so round-trip fp8 -> f32 -> fp8 over ALL of them is
// COMPLETE (unlike fp16's 65536 sample). Every finite/inf value must round-trip bit-exact
// (it's a widening then a re-narrowing of an exactly-representable value); NaN stays NaN.
TEST(Fp8E4M3, ExhaustiveRoundTrip) {
  for (int b = 0; b <= 0xFF; ++b) {
    const auto v = float8_e4m3_t::from_bits(static_cast<std::uint8_t>(b));
    const float f = f8e4m3_to_f32(v);
    const auto r = f32_to_f8e4m3(f);
    if (is_nan_f32(f)) {
      EXPECT_EQ(r.bits & 0x7F, 0x7F) << "b=" << b;  // still the sole NaN
    } else {
      EXPECT_EQ(r.bits, static_cast<std::uint8_t>(b)) << "b=" << b << " f=" << f;
    }
  }
}
TEST(Fp8E5M2, ExhaustiveRoundTrip) {
  for (int b = 0; b <= 0xFF; ++b) {
    const auto v = float8_e5m2_t::from_bits(static_cast<std::uint8_t>(b));
    const float f = f8e5m2_to_f32(v);
    const auto r = f32_to_f8e5m2(f);
    if (is_nan_f32(f)) {
      EXPECT_EQ(r.bits & 0x7F, 0x7E) << "b=" << b;  // canonical NaN class (exp all-1, man!=0)
      EXPECT_NE(r.bits & 0x03, 0) << "b=" << b;
    } else {
      EXPECT_EQ(r.bits, static_cast<std::uint8_t>(b)) << "b=" << b << " f=" << f;
    }
  }
}

// batch conversion agrees with the element core for every pattern (fp8 has no hardware tier;
// this validates the batch loop wiring == the scalar oracle).
TEST(DtypeBatch, Fp8E4M3BatchMatchesElement) {
  std::vector<float8_e4m3_t> src(256);
  for (int b = 0; b <= 0xFF; ++b) src[b] = float8_e4m3_t::from_bits(static_cast<std::uint8_t>(b));
  std::vector<float> got(256);
  convert(src.data(), got.data(), src.size());
  for (int b = 0; b <= 0xFF; ++b) {
    const float ref = f8e4m3_to_f32(src[b]);
    if (is_nan_f32(ref)) {
      EXPECT_TRUE(is_nan_f32(got[b])) << "b=" << b;
    }
    else EXPECT_EQ(std::memcmp(&got[b], &ref, 4), 0) << "b=" << b;
  }
  // and the reverse batch direction round-trips
  std::vector<float8_e4m3_t> back(256);
  convert(got.data(), back.data(), got.size());
  for (int b = 0; b <= 0xFF; ++b) {
    if (is_nan_f32(got[b])) {
      EXPECT_EQ(back[b].bits & 0x7F, 0x7F) << "b=" << b;
    }
    else EXPECT_EQ(back[b].bits, src[b].bits) << "b=" << b;
  }
}

// f16 -> fp8 batch, EXHAUSTIVE over all 65536 f16. The batch must equal the scalar element oracle
// f32_to_f8(f16_to_f32(h)) for EVERY f16 -- on the scalar backend trivially, and on the AVX10.2
// hardware kernel this is the bit-identity gate (f16->f32 exact => single rounding; the E5M2 kernel
// restores OCP inf/NaN that Intel's vcvtph2bf8 saturates). Same test passes on both paths, so a
// build/run with the AVX10.2 kernel (clang-20 + SDE -dmr) proves hw == scalar bit-for-bit.
TEST(Fp8FromF16, ExhaustiveMatchesScalar) {
  std::vector<float16_t> src(65536);
  for (int b = 0; b < 65536; ++b) src[b] = float16_t::from_bits(static_cast<std::uint16_t>(b));
  std::vector<float8_e4m3_t> e4(65536);
  std::vector<float8_e5m2_t> e5(65536);
  convert(src.data(), e4.data(), src.size());
  convert(src.data(), e5.data(), src.size());
  for (int b = 0; b < 65536; ++b) {
    const float f = f16_to_f32(src[b]);
    EXPECT_EQ(e4[b].bits, f32_to_f8e4m3(f).bits) << "e4m3 f16=0x" << std::hex << b;
    EXPECT_EQ(e5[b].bits, f32_to_f8e5m2(f).bits) << "e5m2 f16=0x" << std::hex << b;
  }
}

// f32 -> fp8 batch (dispatch). On an AVX10.2 host (or SDE) the batch path runs the round-to-odd
// hardware kernel; otherwise the scalar oracle. Either way the RESULT must be bit-identical to the
// direct single-rounding scalar f32_to_f8* -- that's the whole point of round-to-odd (it removes
// the f32->f16->fp8 double-rounding error). Sweep a large deterministic set: all 65536 f16-grid
// values promoted to f32, PLUS every mantissa at a stride across all fp8-relevant exponents and
// both signs, PLUS the inf/NaN/overflow boundaries. (The portable-scalar version of this sweep was
// validated exhaustively -- every mantissa, 486.6M values, 0 mismatch -- during bring-up.)
TEST(Fp8FromF32, RoundToOddMatchesDirectScalar) {
  std::vector<float> src;
  src.reserve(1 << 20);
  auto push = [&](std::uint32_t u) { float f; std::memcpy(&f, &u, 4); src.push_back(f); };
  // (a) all f16-grid values as f32 (dense in the fp8 range).
  for (int b = 0; b < 65536; ++b) push(hyle::simd::detail::f32_to_bits(f16_to_f32(
      float16_t::from_bits(static_cast<std::uint16_t>(b)))));
  // (b) strided mantissa across exponents that matter for fp8 (subnormal .. overflow) + both signs.
  for (int e = -12; e <= 18; ++e)
    for (std::uint32_t s = 0; s < 2; ++s)
      for (std::uint32_t m = 0; m < 0x800000u; m += 0x400u)  // stride 1024 -> 8192 steps/exp
        push((s << 31) | (static_cast<std::uint32_t>(e + 127) << 23) | m);
  // (c) explicit specials: +/-0, +/-inf, a signaling & quiet NaN, values just over f16 max (65504).
  for (std::uint32_t u : {0x00000000u, 0x80000000u, 0x7F800000u, 0xFF800000u, 0x7FC00000u,
                          0x7F800001u, 0x47800000u /*65536*/, 0x47802000u})
    push(u);

  const std::size_t n = src.size();
  std::vector<float8_e4m3_t> e4(n);
  std::vector<float8_e5m2_t> e5(n);
  convert(src.data(), e4.data(), n);
  convert(src.data(), e5.data(), n);
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(e4[i].bits, f32_to_f8e4m3(src[i]).bits)
        << "e4m3 i=" << i << " f=" << src[i];
    EXPECT_EQ(e5[i].bits, f32_to_f8e5m2(src[i]).bits)
        << "e5m2 i=" << i << " f=" << src[i];
  }
}
