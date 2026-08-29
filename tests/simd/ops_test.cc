#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "hyle/platform.h"
#include "hyle/simd/dispatch.h"
#include "hyle/simd/ops.h"
#include "src/simd/runtime_internal.h"

// ops_test -- validates the RUNTIME-DISPATCHED ops (simd-04). Unlike vec_test (compiled
// per-ISA), this links the ONE dispatcher and checks: (1) results match a naive scalar
// reference regardless of which variant the host selected; (2) runtime().isa really
// picked the host's best compiled tier -- proving best_isa() is now LIVE (consumed to
// select a kernel), not the dead detector the simd-01 audit flagged.

using hyle::simd::add;
using hyle::platform::best_isa;
using hyle::simd::dot;
using hyle::simd::Isa;
using hyle::simd::runtime;
using hyle::simd::saxpy;

namespace {
std::vector<float> gen(std::size_t n, float base) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i)
    v[i] = base + static_cast<float>((i * 7 + 1) % 17) * 0.25f;
  return v;
}
double ref_dot(const std::vector<float>& x, const std::vector<float>& y) {
  double s = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) s += double(x[i]) * y[i];
  return s;
}
std::vector<double> gend(std::size_t n, double base) {
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; ++i)
    v[i] = base + static_cast<double>((i * 7 + 1) % 17) * 0.25;
  return v;
}
long double ref_dot_d(const std::vector<double>& x, const std::vector<double>& y) {
  long double s = 0.0L;
  for (std::size_t i = 0; i < x.size(); ++i) s += static_cast<long double>(x[i]) * y[i];
  return s;
}

std::vector<const hyle::simd::Runtime*> available_runtimes() {
  std::vector<const hyle::simd::Runtime*> out;
  for (hyle::simd::Isa isa : {hyle::simd::Isa::kScalar, hyle::simd::Isa::kSse42,
                              hyle::simd::Isa::kAvx2, hyle::simd::Isa::kAvx512,
                              hyle::simd::Isa::kNeon, hyle::simd::Isa::kSve,
                              hyle::simd::Isa::kRvv}) {
    if (const auto* table = hyle::simd::runtime_for(isa)) out.push_back(table);
  }
  return out;
}

float ref_dot_bf16(const std::vector<hyle::simd::bfloat16_t>& x,
                   const std::vector<hyle::simd::bfloat16_t>& y) {
  float sum = 0.0f;
  for (std::size_t i = 0; i < x.size(); ++i)
    sum += hyle::simd::bf16_to_f32(x[i]) * hyle::simd::bf16_to_f32(y[i]);
  return sum;
}

float ref_dot_f16(const std::vector<hyle::simd::float16_t>& x,
                  const std::vector<hyle::simd::float16_t>& y) {
  float sum = 0.0f;
  for (std::size_t i = 0; i < x.size(); ++i)
    sum += hyle::simd::f16_to_f32(x[i]) * hyle::simd::f16_to_f32(y[i]);
  return sum;
}

std::int64_t ref_dot_i16(const std::vector<std::int16_t>& x,
                         const std::vector<std::int16_t>& y) {
  std::uint64_t sum = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const std::int64_t product = static_cast<std::int64_t>(x[i]) * y[i];
    sum += static_cast<std::uint64_t>(product);
  }
  return std::bit_cast<std::int64_t>(sum);
}

std::int64_t ref_dot_i8(const std::vector<std::int8_t>& x,
                        const std::vector<std::int8_t>& y) {
  std::uint64_t sum = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const std::int64_t product = static_cast<std::int64_t>(x[i]) * y[i];
    sum += static_cast<std::uint64_t>(product);
  }
  return std::bit_cast<std::int64_t>(sum);
}
}  // namespace

// dot result matches a DOUBLE-precision reference across sizes that exercise full vectors
// + a tail (n not a multiple of any backend width: 1,7,8,31,64,1000). Tolerance is
// RELATIVE to the magnitude: fp reorder (tree-reduce vs sequential) drifts ~1e-5 relative,
// but a real logic bug (wrong lane, off-by-one tail) shifts the result by ~one element's
// worth = a large relative error, so this still catches bugs. Exact per-op math is pinned
// separately in vec_test (integer oracles); here we validate the DISPATCH produces a sane
// result from whichever variant the host picked.
TEST(Ops, DotMatchesReference) {
  for (std::size_t n : {std::size_t{1}, std::size_t{7}, std::size_t{8}, std::size_t{31},
                        std::size_t{64}, std::size_t{1000}}) {
    auto x = gen(n, 0.5f), y = gen(n, -0.3f);
    const double ref = ref_dot(x, y);
    float got = dot(x.data(), y.data(), n);
    const float tol = static_cast<float>(std::abs(ref)) * 1e-4f + 1e-4f;
    EXPECT_NEAR(got, static_cast<float>(ref), tol) << "n=" << n;
  }
}

TEST(Ops, DotEmptyIsZero) {
  EXPECT_EQ(dot(static_cast<const float*>(nullptr), static_cast<const float*>(nullptr), 0), 0.0f);
}

// f64 dot vs a long-double reference. f64 tolerance is far tighter than f32 (the whole point of
// the double path): ~1e-12 relative covers the tree-reduce reorder while still catching logic bugs.
TEST(Ops, DotF64MatchesReference) {
  for (std::size_t n : {std::size_t{1}, std::size_t{7}, std::size_t{8}, std::size_t{31},
                        std::size_t{64}, std::size_t{1000}}) {
    auto x = gend(n, 0.5), y = gend(n, -0.3);
    const long double ref = ref_dot_d(x, y);
    double got = dot(x.data(), y.data(), n);
    const double tol = static_cast<double>(std::abs(ref)) * 1e-12 + 1e-12;
    EXPECT_NEAR(got, static_cast<double>(ref), tol) << "n=" << n;
  }
}

TEST(Ops, DotF64EmptyIsZero) {
  EXPECT_EQ(dot(static_cast<const double*>(nullptr), static_cast<const double*>(nullptr), 0), 0.0);
}

TEST(Ops, MixedPrecisionDotsMatchReferenceOnEveryRuntimeTier) {
  for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{7},
                        std::size_t{8}, std::size_t{31}, std::size_t{64},
                        std::size_t{257}, std::size_t{1000}}) {
    std::vector<float> f32(n);
    std::vector<hyle::simd::bfloat16_t> bf0(n), bf1(n);
    std::vector<hyle::simd::float16_t> f16a(n), f16b(n);
    for (std::size_t i = 0; i < n; ++i) {
      const float a = (static_cast<int>(i % 23) - 11) * 0.125f;
      const float b = (static_cast<int>((i * 7) % 29) - 14) * 0.0625f;
      f32[i] = a + 0.0001f;
      bf0[i] = hyle::simd::f32_to_bf16(a);
      bf1[i] = hyle::simd::f32_to_bf16(b);
      f16a[i] = hyle::simd::f32_to_f16(a);
      f16b[i] = hyle::simd::f32_to_f16(b);
    }

    const float bf_ref = ref_dot_bf16(bf0, bf1);
    const float f16_ref = ref_dot_f16(f16a, f16b);
    float mixed_ref = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
      mixed_ref += f32[i] * hyle::simd::bf16_to_f32(bf1[i]);
    const float bf_tol = std::abs(bf_ref) * 2e-5f + 2e-5f;
    const float f16_tol = std::abs(f16_ref) * 2e-5f + 2e-5f;
    const float mixed_tol = std::abs(mixed_ref) * 2e-5f + 2e-5f;

    EXPECT_NEAR(dot(bf0.data(), bf1.data(), n), bf_ref, bf_tol) << "public bf16 n=" << n;
    EXPECT_NEAR(dot(f32.data(), bf1.data(), n), mixed_ref, mixed_tol)
        << "public f32xbf16 n=" << n;
    EXPECT_NEAR(dot(bf1.data(), f32.data(), n), mixed_ref, mixed_tol)
        << "public bf16xf32 n=" << n;
    EXPECT_NEAR(dot(f16a.data(), f16b.data(), n), f16_ref, f16_tol)
        << "public f16 n=" << n;

    for (const auto* table : available_runtimes()) {
      EXPECT_NEAR(table->ops.dot_bf16_bf16(bf0.data(), bf1.data(), n),
                  bf_ref, bf_tol) << "bf16 tier=" << static_cast<int>(table->isa) << " n=" << n;
      EXPECT_NEAR(table->ops.dot_f32_bf16(f32.data(), bf1.data(), n),
                  mixed_ref, mixed_tol)
          << "mixed tier=" << static_cast<int>(table->isa) << " n=" << n;
      EXPECT_NEAR(table->ops.dot_f16_f16(f16a.data(), f16b.data(), n),
                  f16_ref, f16_tol) << "f16 tier=" << static_cast<int>(table->isa) << " n=" << n;
    }
  }
}

TEST(Ops, MixedPrecisionDotsPreserveSpecialAndInputSemantics) {
  std::vector<hyle::simd::bfloat16_t> bf_nan(
      64, hyle::simd::f32_to_bf16(1.0f));
  const std::vector<hyle::simd::bfloat16_t> bf_one(
      64, hyle::simd::f32_to_bf16(1.0f));
  bf_nan[17] = hyle::simd::f32_to_bf16(NAN);  // inside a full vector block
  EXPECT_TRUE(std::isnan(dot(bf_nan.data(), bf_one.data(), bf_nan.size())));

  std::vector<hyle::simd::float16_t> f16_inf_zero(
      64, hyle::simd::f32_to_f16(1.0f));
  std::vector<hyle::simd::float16_t> f16_zero_one(
      64, hyle::simd::f32_to_f16(1.0f));
  f16_inf_zero[9] = hyle::simd::f32_to_f16(INFINITY);
  f16_zero_one[9] = hyle::simd::f32_to_f16(0.0f);
  EXPECT_TRUE(std::isnan(dot(f16_inf_zero.data(), f16_zero_one.data(), 64)));

  const float full_precision = 1.0001f;
  const auto one = hyle::simd::f32_to_bf16(1.0f);
  EXPECT_FLOAT_EQ(dot(&full_precision, &one, 1), full_precision);
  EXPECT_FLOAT_EQ(dot(&one, &full_precision, 1), full_precision);
}

TEST(Ops, I16DotIsExactAndModuloDefinedOnEveryRuntimeTier) {
  constexpr std::size_t kCount = 1025;
  std::vector<std::int16_t> x(kCount), y(kCount);
  for (std::size_t i = 0; i < kCount; ++i) {
    x[i] = (i % 5 == 0) ? std::int16_t{-32768}
                        : static_cast<std::int16_t>(static_cast<int>(i % 65536) - 32768);
    y[i] = (i % 7 == 0) ? std::int16_t{-32768}
                        : static_cast<std::int16_t>(32767 - static_cast<int>(i % 65536));
  }
  const std::int64_t expected = ref_dot_i16(x, y);
  EXPECT_EQ(dot(x.data(), y.data(), x.size()), expected);
  for (const auto* table : available_runtimes())
    EXPECT_EQ(table->ops.dot_i16_i16(x.data(), y.data(), x.size()), expected)
        << "tier=" << static_cast<int>(table->isa);

  EXPECT_EQ(dot(static_cast<const std::int16_t*>(nullptr),
                static_cast<const std::int16_t*>(nullptr), 0), 0);
}

TEST(Ops, I8DotIsExactAndModuloDefinedOnEveryRuntimeTier) {
  constexpr std::size_t kCount = 262147;  // crosses the x86 i32 accumulator flush boundary
  std::vector<std::int8_t> x(kCount), y(kCount);
  for (std::size_t i = 0; i < kCount; ++i) {
    x[i] = (i % 5 == 0) ? std::int8_t{-128}
                        : static_cast<std::int8_t>(static_cast<int>(i % 255) - 127);
    y[i] = (i % 7 == 0) ? std::int8_t{-128}
                        : static_cast<std::int8_t>(127 - static_cast<int>((i * 13) % 255));
  }
  const std::int64_t expected = ref_dot_i8(x, y);
  EXPECT_EQ(dot(x.data(), y.data(), x.size()), expected);
  for (const auto* table : available_runtimes())
    EXPECT_EQ(table->ops.dot_i8_i8(x.data(), y.data(), x.size()), expected)
        << "tier=" << static_cast<int>(table->isa);

  EXPECT_EQ(dot(static_cast<const std::int8_t*>(nullptr),
                static_cast<const std::int8_t*>(nullptr), 0), 0);
}

// saxpy result matches naive reference; tail handled.
TEST(Ops, SaxpyMatchesReference) {
  for (std::size_t n : {std::size_t{1}, std::size_t{8}, std::size_t{31}, std::size_t{257}}) {
    auto x = gen(n, 1.0f);
    auto y = gen(n, 2.0f);
    std::vector<float> expect(n);
    const float a = 1.5f;
    for (std::size_t i = 0; i < n; ++i) expect[i] = a * x[i] + y[i];
    saxpy(a, x.data(), y.data(), n);
    for (std::size_t i = 0; i < n; ++i) EXPECT_FLOAT_EQ(y[i], expect[i]) << "n=" << n << " i=" << i;
  }
}

// f64 saxpy: y[i] = a*x[i] + y[i] over doubles; tail handled.
TEST(Ops, SaxpyF64MatchesReference) {
  for (std::size_t n : {std::size_t{1}, std::size_t{8}, std::size_t{31}, std::size_t{257}}) {
    auto x = gend(n, 1.0);
    auto y = gend(n, 2.0);
    std::vector<double> expect(n);
    const double a = 1.5;
    for (std::size_t i = 0; i < n; ++i) expect[i] = a * x[i] + y[i];
    saxpy(a, x.data(), y.data(), n);
    for (std::size_t i = 0; i < n; ++i) EXPECT_DOUBLE_EQ(y[i], expect[i]) << "n=" << n << " i=" << i;
  }
}

// elementwise add: z = x + y (f32/f64). Tail handled at non-SIMD-width sizes.
TEST(Ops, AddF32MatchesReference) {
  for (std::size_t n : {std::size_t{1}, std::size_t{8}, std::size_t{31}, std::size_t{257}, std::size_t{1000}}) {
    auto x = gen(n, 0.5f), y = gen(n, -0.3f);
    std::vector<float> z(n, -1.0f);
    add(x.data(), y.data(), z.data(), n);
    for (std::size_t i = 0; i < n; ++i) EXPECT_FLOAT_EQ(z[i], x[i] + y[i]) << "n=" << n << " i=" << i;
  }
}
TEST(Ops, AddF64MatchesReference) {
  for (std::size_t n : {std::size_t{1}, std::size_t{8}, std::size_t{31}, std::size_t{257}, std::size_t{1000}}) {
    auto x = gend(n, 0.5), y = gend(n, -0.3);
    std::vector<double> z(n, -1.0);
    add(x.data(), y.data(), z.data(), n);
    for (std::size_t i = 0; i < n; ++i) EXPECT_DOUBLE_EQ(z[i], x[i] + y[i]) << "n=" << n << " i=" << i;
  }
}

// THE simd-04 point: the dispatcher selected the host's best COMPILED tier. We ship
// scalar/sse4/avx2/avx512, so on an x86 host runtime().isa must equal best_isa() mapped
// to {scalar,sse42,avx2,avx512} -- never below what the CPU supports, never above what we
// compiled. This proves best_isa() is actually consumed (was dead before simd-04).
TEST(Ops, ActiveVariantMatchesBestCompiledTier) {
  const Isa active = runtime().isa;
#if defined(HYLE_ARCH_X64)
  const auto best = static_cast<std::uint32_t>(best_isa());
  Isa expect;
  if (best >= static_cast<std::uint32_t>(Isa::kAvx512)) expect = Isa::kAvx512;
  else if (best >= static_cast<std::uint32_t>(Isa::kAvx2)) expect = Isa::kAvx2;
  else if (best >= static_cast<std::uint32_t>(Isa::kSse42)) expect = Isa::kSse42;
  else expect = Isa::kScalar;
  EXPECT_EQ(active, expect);
  // On any modern x86-64 dev box this should be a real SIMD tier, not the floor.
  EXPECT_GE(static_cast<std::uint32_t>(active), static_cast<std::uint32_t>(Isa::kSse42));
#elif defined(HYLE_ARCH_ARM64)
  // Linux aarch64 may additionally ship the SVE scalable variant. The active
  // Runtime must match the highest variant actually compiled into this binary.
#if defined(HYLE_SIMD_HAS_SVE_VARIANT)
  if (best_isa() == Isa::kSve) {
    EXPECT_EQ(active, Isa::kSve);
  } else if (best_isa() == Isa::kNeon) {
    EXPECT_EQ(active, Isa::kNeon);
  } else {
    EXPECT_EQ(active, Isa::kScalar);
  }
#else
  if (best_isa() == Isa::kNeon) {
    EXPECT_EQ(active, Isa::kNeon);
  } else {
    EXPECT_EQ(active, Isa::kScalar);
  }
#endif
#elif defined(HYLE_ARCH_RISCV64)
#if defined(HYLE_SIMD_HAS_RVV_VARIANT)
  EXPECT_EQ(active, best_isa() == Isa::kRvv ? Isa::kRvv : Isa::kScalar);
#else
  EXPECT_EQ(active, Isa::kScalar);
#endif
#else
  EXPECT_EQ(active, Isa::kScalar);  // unknown arch: portable floor only
#endif
}

// Dispatch is stable: repeated calls return the same cached selection (no re-resolve).
TEST(Ops, ActiveVariantStable) {
  EXPECT_EQ(&runtime(), &runtime());
}
