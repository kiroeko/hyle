#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "hyle/core/environment.h"
#include "hyle/simd/target.h"
#include "hyle/simd/ops.h"
#include "src/simd/runtime_internal.h"

// ops_test -- validates the RUNTIME-DISPATCHED ops (simd-04). Unlike vec_test (compiled
// per-target), this links the one dispatcher and checks: (1) results match a
// naive scalar reference regardless of which variant the host selected; and
// (2) the runtime selected the best compiled target whose explicit feature
// requirements are satisfied.

using hyle::simd::add;
using hyle::simd::dot;
using hyle::simd::KernelTarget;
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
  for (hyle::simd::KernelTarget isa : {hyle::simd::KernelTarget::kScalar, hyle::simd::KernelTarget::kSse4,
                              hyle::simd::KernelTarget::kAvx2, hyle::simd::KernelTarget::kAvx512,
                              hyle::simd::KernelTarget::kNeon, hyle::simd::KernelTarget::kSve,
                              hyle::simd::KernelTarget::kRvv}) {
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
                  bf_ref, bf_tol) << "bf16 tier=" << static_cast<int>(table->target) << " n=" << n;
      EXPECT_NEAR(table->ops.dot_f32_bf16(f32.data(), bf1.data(), n),
                  mixed_ref, mixed_tol)
          << "mixed tier=" << static_cast<int>(table->target) << " n=" << n;
      EXPECT_NEAR(table->ops.dot_f16_f16(f16a.data(), f16b.data(), n),
                  f16_ref, f16_tol) << "f16 tier=" << static_cast<int>(table->target) << " n=" << n;
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
        << "tier=" << static_cast<int>(table->target);

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
        << "tier=" << static_cast<int>(table->target);

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

TEST(Ops, CompleteF32RowAlgebraMatchesReferenceOnEveryRuntimeTier) {
  constexpr std::size_t kCount = 31;
  std::vector<float> x(kCount), y(kCount), out(kCount);
  for (std::size_t i = 0; i < kCount; ++i) {
    x[i] = 0.25f + static_cast<float>(i) * 0.125f;
    y[i] = 1.0f + static_cast<float>(i % 7) * 0.25f;
  }

  for (const auto* table : available_runtimes()) {
    const auto& ops = table->ops;
    ops.sub_f32(x.data(), y.data(), out.data(), kCount);
    for (std::size_t i = 0; i < kCount; ++i) EXPECT_FLOAT_EQ(out[i], x[i] - y[i]);
    ops.mul_f32(x.data(), y.data(), out.data(), kCount);
    for (std::size_t i = 0; i < kCount; ++i) EXPECT_FLOAT_EQ(out[i], x[i] * y[i]);
    ops.div_f32(x.data(), y.data(), out.data(), kCount);
    for (std::size_t i = 0; i < kCount; ++i) EXPECT_FLOAT_EQ(out[i], x[i] / y[i]);
    ops.scale_f32(x.data(), 1.5f, out.data(), kCount);
    for (std::size_t i = 0; i < kCount; ++i) EXPECT_FLOAT_EQ(out[i], x[i] * 1.5f);
    ops.add_scalar_f32(x.data(), -0.5f, out.data(), kCount);
    for (std::size_t i = 0; i < kCount; ++i) EXPECT_FLOAT_EQ(out[i], x[i] - 0.5f);
    ops.affine_f32(x.data(), 1.5f, -0.25f, out.data(), kCount);
    for (std::size_t i = 0; i < kCount; ++i)
      EXPECT_NEAR(out[i], x[i] * 1.5f - 0.25f, 1e-6f);

    float expected_sum = 0.0f;
    for (float value : x) expected_sum += value;
    EXPECT_NEAR(ops.sum_f32(x.data(), kCount), expected_sum,
                std::abs(expected_sum) * 1e-6f + 1e-6f);

    std::vector<float> signed_x = x;
    for (std::size_t i = 0; i < kCount; i += 2) signed_x[i] = -signed_x[i];
    ops.abs_f32(signed_x.data(), out.data(), kCount);
    for (std::size_t i = 0; i < kCount; ++i) EXPECT_FLOAT_EQ(out[i], std::abs(signed_x[i]));
    ops.neg_f32(x.data(), out.data(), kCount);
    for (std::size_t i = 0; i < kCount; ++i) EXPECT_FLOAT_EQ(out[i], -x[i]);
    ops.square_f32(x.data(), out.data(), kCount);
    for (std::size_t i = 0; i < kCount; ++i) EXPECT_FLOAT_EQ(out[i], x[i] * x[i]);
    ops.sqrt_f32(x.data(), out.data(), kCount);
    for (std::size_t i = 0; i < kCount; ++i) EXPECT_FLOAT_EQ(out[i], std::sqrt(x[i]));

    auto in_place = x;
    ops.mul_f32(in_place.data(), y.data(), in_place.data(), kCount);
    for (std::size_t i = 0; i < kCount; ++i) EXPECT_FLOAT_EQ(in_place[i], x[i] * y[i]);
  }
}

TEST(Ops, CompleteF64RowAlgebraAndZeroContract) {
  constexpr std::size_t kCount = 31;
  auto x = gend(kCount, 0.5), y = gend(kCount, 1.0);
  std::vector<double> out(kCount);
  hyle::simd::subtract(x.data(), y.data(), out.data(), kCount);
  hyle::simd::multiply(x.data(), y.data(), out.data(), kCount);
  hyle::simd::divide(x.data(), y.data(), out.data(), kCount);
  hyle::simd::scale(x.data(), 2.0, out.data(), kCount);
  hyle::simd::add_scalar(x.data(), -1.0, out.data(), kCount);
  hyle::simd::affine(x.data(), 2.0, -1.0, out.data(), kCount);
  hyle::simd::abs(x.data(), out.data(), kCount);
  hyle::simd::negate(x.data(), out.data(), kCount);
  hyle::simd::square(x.data(), out.data(), kCount);
  hyle::simd::sqrt(x.data(), out.data(), kCount);
  long double expected_sum = 0.0;
  for (double value : x) expected_sum += value;
  EXPECT_NEAR(hyle::simd::sum(x.data(), kCount), static_cast<double>(expected_sum),
              static_cast<double>(expected_sum) * 1e-12 + 1e-12);

  const double* null_in = nullptr;
  double* null_out = nullptr;
  hyle::simd::subtract(null_in, null_in, null_out, 0);
  hyle::simd::multiply(null_in, null_in, null_out, 0);
  hyle::simd::divide(null_in, null_in, null_out, 0);
  hyle::simd::scale(null_in, 1.0, null_out, 0);
  hyle::simd::add_scalar(null_in, 1.0, null_out, 0);
  hyle::simd::affine(null_in, 1.0, 1.0, null_out, 0);
  EXPECT_EQ(hyle::simd::sum(null_in, 0), 0.0);
  hyle::simd::abs(null_in, null_out, 0);
  hyle::simd::negate(null_in, null_out, 0);
  hyle::simd::square(null_in, null_out, 0);
  hyle::simd::sqrt(null_in, null_out, 0);
}

// Fat artifacts select the best compiled host tier. Exact artifacts deliberately
// contain one builder-selected tier and must not reintroduce host probing.
TEST(Ops, ActiveVariantMatchesBestCompiledTier) {
  const KernelTarget active = runtime().target;
#if defined(HYLE_SIMD_EXACT_TARGET_SCALAR)
  EXPECT_EQ(active, KernelTarget::kScalar);
#elif defined(HYLE_SIMD_EXACT_TARGET_SSE4)
  EXPECT_EQ(active, KernelTarget::kSse4);
#elif defined(HYLE_SIMD_EXACT_TARGET_AVX2)
  EXPECT_EQ(active, KernelTarget::kAvx2);
#elif defined(HYLE_SIMD_EXACT_TARGET_AVX512)
  EXPECT_EQ(active, KernelTarget::kAvx512);
#elif defined(HYLE_SIMD_EXACT_TARGET_NEON)
  EXPECT_EQ(active, KernelTarget::kNeon);
#elif defined(HYLE_SIMD_EXACT_TARGET_SVE)
  EXPECT_EQ(active, KernelTarget::kSve);
#elif defined(HYLE_SIMD_EXACT_TARGET_RVV)
  EXPECT_EQ(active, KernelTarget::kRvv);
#elif HYLE_ARCH_X64
  const auto& features = hyle::platform::cpu_features();
  const KernelTarget expect =
      hyle::simd::kernel_target_supported(KernelTarget::kAvx512, features)
          ? KernelTarget::kAvx512
      : hyle::simd::kernel_target_supported(KernelTarget::kAvx2, features)
          ? KernelTarget::kAvx2
      : hyle::simd::kernel_target_supported(KernelTarget::kSse4, features)
          ? KernelTarget::kSse4
          : KernelTarget::kScalar;
  EXPECT_EQ(active, expect);
#elif HYLE_ARCH_ARM64
  // Linux aarch64 may additionally ship the SVE scalable variant. The active
  // Runtime must match the highest variant actually compiled into this binary.
#if defined(HYLE_SIMD_HAS_SVE_VARIANT)
  if (hyle::simd::kernel_target_supported(KernelTarget::kSve)) {
    EXPECT_EQ(active, KernelTarget::kSve);
  } else if (hyle::simd::kernel_target_supported(KernelTarget::kNeon)) {
    EXPECT_EQ(active, KernelTarget::kNeon);
  } else {
    EXPECT_EQ(active, KernelTarget::kScalar);
  }
#else
  if (hyle::simd::kernel_target_supported(KernelTarget::kNeon)) {
    EXPECT_EQ(active, KernelTarget::kNeon);
  } else {
    EXPECT_EQ(active, KernelTarget::kScalar);
  }
#endif
#elif HYLE_ARCH_RISCV64
#if defined(HYLE_SIMD_HAS_RVV_VARIANT)
  EXPECT_EQ(active,
            hyle::simd::kernel_target_supported(KernelTarget::kRvv)
                ? KernelTarget::kRvv
                : KernelTarget::kScalar);
#else
  EXPECT_EQ(active, KernelTarget::kScalar);
#endif
#else
  EXPECT_EQ(active, KernelTarget::kScalar);  // unknown arch: portable floor only
#endif
}

// Dispatch is stable: repeated calls return the same cached selection (no re-resolve).
TEST(Ops, ActiveVariantStable) {
  EXPECT_EQ(&runtime(), &runtime());
}
