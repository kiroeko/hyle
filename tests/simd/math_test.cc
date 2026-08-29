#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

#include "hyle/simd/dispatch.h"
#include "hyle/simd/math.h"
#include "hyle/simd/runtime.h"

// math_test -- ordinary batch operations vs the std::/libm oracle. Runtime routes to a
// fixed-target locked-Highway implementation where qualified and otherwise to the scalar
// semantic floor. The public result must satisfy the same complete-domain contract either way.

using hyle::simd::exp_f32_batch;
using hyle::simd::Isa;
using hyle::simd::log_f32_batch;
using hyle::simd::tanh_f32_batch;

namespace {
// Relative-error tolerance vs a double-precision libm reference. Set from MEASURED worst
// case (red team 2026-06 flagged the old 1e-5 ~= 84 ULP as toothless -- a broken
// polynomial would pass). Qualified locked-Highway functions are in the low-single-digit
// ULP class, so the bound is 8e-7 ~= 6.7 ULP: enough composition margin while remaining
// still ~12x tighter than the old toothless 1e-5, so a kernel off by >=7 ULP fails.
// Absolute floor 1e-6 covers lanes crossing zero (log(1), tanh(0)) where rel err undefined.
constexpr double kRelTol = 8e-7;
constexpr double kAbsTol = 1e-6;
void check(void (*fn)(const float*, float*, std::size_t), double (*ref)(double),
           const std::vector<float>& in, const char* name) {
  std::vector<float> out(in.size());
  fn(in.data(), out.data(), in.size());
  for (std::size_t i = 0; i < in.size(); ++i) {
    const double want = ref(static_cast<double>(in[i]));
    const double got = static_cast<double>(out[i]);
    const double tol = std::abs(want) * kRelTol + kAbsTol;
    EXPECT_NEAR(got, want, tol) << name << " in=" << in[i] << " i=" << i;
  }
}
}  // namespace

TEST(Math, ExpMatchesLibm) {
  std::vector<float> in;
  for (int k = -300; k <= 300; ++k) in.push_back(static_cast<float>(k) * 0.1f);  // [-30,30]
  check(exp_f32_batch, std::exp, in, "exp");
}

TEST(Math, LogMatchesLibm) {
  std::vector<float> in;
  for (int k = 1; k <= 1000; ++k) in.push_back(static_cast<float>(k) * 0.05f);  // (0, 50]
  check(log_f32_batch, std::log, in, "log");
}

TEST(Math, TanhMatchesLibm) {
  std::vector<float> in;
  for (int k = -200; k <= 200; ++k) in.push_back(static_cast<float>(k) * 0.05f);  // [-10,10]
  check(tanh_f32_batch, std::tanh, in, "tanh");
}

// f64 transcendentals vs a long-double libm reference. The vendored cores are ~3.5-ULP class in
// double too, so the bound is ~1e-14 relative (~45 ULP of double -- still ~billions x tighter than
// the f32 8e-7, and catches any broken f64 polynomial / wrong-precision dispatch).
namespace {
constexpr double kRelTolF64 = 1e-14;
constexpr double kAbsTolF64 = 1e-15;
void check_f64(void (*fn)(const double*, double*, std::size_t), long double (*ref)(long double),
               const std::vector<double>& in, const char* name) {
  std::vector<double> out(in.size());
  fn(in.data(), out.data(), in.size());
  for (std::size_t i = 0; i < in.size(); ++i) {
    const long double want = ref(static_cast<long double>(in[i]));
    const double tol = static_cast<double>(std::abs(want)) * kRelTolF64 + kAbsTolF64;
    EXPECT_NEAR(out[i], static_cast<double>(want), tol) << name << " in=" << in[i] << " i=" << i;
  }
}
}  // namespace
TEST(Math, F64MatchesLibm) {
  std::vector<double> ein, lin, tin;
  for (int k = -300; k <= 300; ++k) ein.push_back(static_cast<double>(k) * 0.1);   // exp [-30,30]
  for (int k = 1; k <= 1000; ++k) lin.push_back(static_cast<double>(k) * 0.05);    // log (0,50]
  for (int k = -200; k <= 200; ++k) tin.push_back(static_cast<double>(k) * 0.05);  // tanh [-10,10]
  check_f64(hyle::simd::exp_f64_batch, std::expl, ein, "exp_f64");
  check_f64(hyle::simd::log_f64_batch, std::logl, lin, "log_f64");
  check_f64(hyle::simd::tanh_f64_batch, std::tanhl, tin, "tanh_f64");
}

namespace {

std::uint64_t ordered_f64_bits(double x) noexcept {
  const std::uint64_t bits = std::bit_cast<std::uint64_t>(x);
  return (bits & (std::uint64_t{1} << 63)) != 0 ? ~bits + 1 : bits | (std::uint64_t{1} << 63);
}

std::uint64_t f64_ulp_distance(double a, double b) noexcept {
  const std::uint64_t oa = ordered_f64_bits(a);
  const std::uint64_t ob = ordered_f64_bits(b);
  return oa >= ob ? oa - ob : ob - oa;
}

// Full-domain ULP gate for prospective P2-1 table reductions. std::exp/log is the production
// scalar floor, so this catches a regression relative to the API's existing fallback semantics.
void check_f64_ulp(void (*fn)(const double*, double*, std::size_t), double (*ref)(double),
                   const std::vector<double>& in, std::uint64_t max_ulp, const char* name) {
  std::vector<double> out(in.size());
  fn(in.data(), out.data(), in.size());
  std::uint64_t observed = 0;
  for (std::size_t i = 0; i < in.size(); ++i) {
    const double want = ref(in[i]);
    const double got = out[i];
    if (std::isnan(want)) {
      EXPECT_TRUE(std::isnan(got)) << name << " i=" << i;
      continue;
    }
    if (std::isinf(want)) {
      EXPECT_EQ(got, want) << name << " i=" << i;
      continue;
    }
    const std::uint64_t ulp = f64_ulp_distance(got, want);
    observed = std::max(observed, ulp);
    EXPECT_LE(ulp, max_ulp) << name << " in=" << std::hexfloat << in[i] << std::defaultfloat
                            << " i=" << i << " got=" << std::hexfloat << got
                            << " want=" << want << std::defaultfloat;
  }
  std::fprintf(stderr, "P2-1 %s max_ulp=%llu samples=%zu\n", name,
               static_cast<unsigned long long>(observed), in.size());
}

std::vector<double> f64_exp_ulp_inputs() {
  std::vector<double> in = {
      -std::numeric_limits<double>::infinity(), -745.0, -744.4400719213812,
      -708.3964185322641, -1.0, -0.0, 0.0, 1.0, 709.0, 709.782712893384,
      std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()};
  std::mt19937_64 rng(0xE6F64ULL);
  std::uniform_real_distribution<double> dist(-745.0, 709.0);
  for (std::size_t i = 0; i < 16384; ++i) in.push_back(dist(rng));
  return in;
}

std::vector<double> f64_log_ulp_inputs() {
  std::vector<double> in = {
      -std::numeric_limits<double>::infinity(), -1.0, -0.0, 0.0,
      std::numeric_limits<double>::denorm_min(), std::nextafter(std::numeric_limits<double>::min(), 0.0),
      std::numeric_limits<double>::min(), std::nextafter(std::numeric_limits<double>::min(), 1.0),
      0.5, 1.0, 2.0, std::numeric_limits<double>::max(),
      std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()};
  std::mt19937_64 rng(0x10AF64ULL);
  std::uniform_real_distribution<double> exponent(-1074.0, 1023.0);
  std::uniform_real_distribution<double> mantissa(1.0, 2.0);
  for (std::size_t i = 0; i < 16384; ++i) in.push_back(std::scalbn(mantissa(rng), static_cast<int>(exponent(rng))));
  return in;
}

std::vector<double> f64_exp_in_range_inputs() {
  std::vector<double> in = {-700.0, -1.0, -0.0, 0.0, 1.0, 700.0};
  std::mt19937_64 rng(0xE6F64A11ULL);
  std::uniform_real_distribution<double> dist(-700.0, 700.0);
  for (std::size_t i = 0; i < 16384; ++i) in.push_back(dist(rng));
  return in;
}

std::vector<double> f64_log_in_range_inputs() {
  std::vector<double> in = {std::numeric_limits<double>::min(), 0.5, 1.0, 2.0,
                            std::numeric_limits<double>::max()};
  std::mt19937_64 rng(0x10AF64A11ULL);
  std::uniform_real_distribution<double> exponent(-1022.0, 1022.0);
  std::uniform_real_distribution<double> mantissa(1.0, 2.0);
  for (std::size_t i = 0; i < 16384; ++i)
    in.push_back(std::scalbn(mantissa(rng), static_cast<int>(exponent(rng))));
  return in;
}

}  // namespace

TEST(Math, F64ExpLogFullDomainUlp) {
  // Existing Highway standard kernels target 1 ULP upstream. Keep a small integration margin for
  // compiler/FMA differences; any P2-1 replacement must tighten this gate rather than weaken it.
  check_f64_ulp(hyle::simd::exp_f64_batch, std::exp, f64_exp_ulp_inputs(), 4, "exp_f64");
  check_f64_ulp(hyle::simd::log_f64_batch, std::log, f64_log_ulp_inputs(), 4, "log_f64");
}

TEST(Math, F64ExpLogInRangeUlpAndAlias) {
  const auto exp_in = f64_exp_in_range_inputs();
  const auto log_in = f64_log_in_range_inputs();
  check_f64_ulp(hyle::simd::exp_in_range_f64_batch, std::exp, exp_in, 4, "exp_in_range_f64");
  check_f64_ulp(hyle::simd::log_in_range_f64_batch, std::log, log_in, 4, "log_in_range_f64");

  auto inplace = exp_in;
  hyle::simd::exp_in_range_f64_batch(inplace.data(), inplace.data(), inplace.size());
  for (std::size_t i = 0; i < inplace.size(); ++i)
    EXPECT_LE(f64_ulp_distance(inplace[i], std::exp(exp_in[i])), 4U) << "exp in-place i=" << i;
  inplace = log_in;
  hyle::simd::log_in_range_f64_batch(inplace.data(), inplace.data(), inplace.size());
  for (std::size_t i = 0; i < inplace.size(); ++i)
    EXPECT_LE(f64_ulp_distance(inplace[i], std::log(log_in[i])), 4U) << "log in-place i=" << i;
}

TEST(Math, F32ExpLogInRangeAndAlias) {
  std::vector<float> exp_in;
  std::vector<float> log_in;
  for (int k = -800; k <= 800; ++k)
    exp_in.push_back(static_cast<float>(k) * 0.1f);
  for (int k = 1; k <= 2000; ++k)
    log_in.push_back(static_cast<float>(k) * 0.05f);

  check(hyle::simd::exp_in_range_f32_batch, std::exp, exp_in,
        "exp_in_range_f32");
  check(hyle::simd::log_in_range_f32_batch, std::log, log_in,
        "log_in_range_f32");

  auto inplace = exp_in;
  hyle::simd::exp_in_range_f32_batch(inplace.data(), inplace.data(),
                                     inplace.size());
  for (std::size_t i = 0; i < inplace.size(); ++i)
    EXPECT_NEAR(inplace[i], std::exp(exp_in[i]),
                std::abs(std::exp(exp_in[i])) * kRelTol + kAbsTol)
        << "exp in-place i=" << i;
  inplace = log_in;
  hyle::simd::log_in_range_f32_batch(inplace.data(), inplace.data(),
                                     inplace.size());
  for (std::size_t i = 0; i < inplace.size(); ++i)
    EXPECT_NEAR(inplace[i], std::log(log_in[i]),
                std::abs(std::log(log_in[i])) * kRelTol + kAbsTol)
        << "log in-place i=" << i;
}

// The remaining f64 transcendentals, each on its natural domain, against long-double libm at
// the ~1e-14 f64 bound. Product routing is not encoded in the test name or oracle.
namespace {
std::vector<double> rampd(double lo, double hi, int steps) {
  std::vector<double> v;
  for (int k = 0; k <= steps; ++k) v.push_back(lo + (hi - lo) * static_cast<double>(k) / steps);
  return v;
}
}  // namespace
TEST(Math, F64RestMatchesLibm) {
  check_f64(hyle::simd::sin_f64_batch, std::sinl, rampd(-3.14, 3.14, 600), "sin_f64");
  check_f64(hyle::simd::cos_f64_batch, std::cosl, rampd(-3.14, 3.14, 600), "cos_f64");
  check_f64(hyle::simd::exp2_f64_batch, std::exp2l, rampd(-20.0, 20.0, 600), "exp2_f64");
  check_f64(hyle::simd::log2_f64_batch, std::log2l, rampd(0.01, 100.0, 600), "log2_f64");
  check_f64(hyle::simd::erf_f64_batch, std::erfl, rampd(-4.0, 4.0, 600), "erf_f64");
  check_f64(hyle::simd::atan_f64_batch, std::atanl, rampd(-20.0, 20.0, 600), "atan_f64");
  check_f64(hyle::simd::cbrt_f64_batch, std::cbrtl, rampd(-100.0, 100.0, 600), "cbrt_f64");
  check_f64(hyle::simd::expm1_f64_batch, std::expm1l, rampd(-2.0, 2.0, 600), "expm1_f64");
  check_f64(hyle::simd::log1p_f64_batch, std::log1pl, rampd(-0.9, 20.0, 600), "log1p_f64");
  check_f64(hyle::simd::asinh_f64_batch, std::asinhl, rampd(-20.0, 20.0, 600), "asinh_f64");
  check_f64(hyle::simd::acosh_f64_batch, std::acoshl, rampd(1.0, 50.0, 600), "acosh_f64");
  check_f64(hyle::simd::tgamma_f64_batch, std::tgammal, rampd(0.5, 15.0, 400), "tgamma_f64");
  check_f64(hyle::simd::lgamma_f64_batch, std::lgammal, rampd(0.5, 40.0, 400), "lgamma_f64");
}
// hypot f64 (binary), incl overflow inputs (>1e150 where a^2 overflows double -- Highway's exponent
// scaling must match std::hypot, not return inf).
TEST(Math, HypotF64MatchesLibm) {
  std::vector<double> a, b;
  for (int k = -50; k <= 50; ++k) { a.push_back(static_cast<double>(k) * 3.0); b.push_back(static_cast<double>(k) * 2.0 + 1.0); }
  a.push_back(3e300); b.push_back(4e300);   // a^2/b^2 overflow double -> tests exponent scaling
  a.push_back(1e308); b.push_back(1.0);
  std::vector<double> out(a.size());
  hyle::simd::hypot_f64_batch(a.data(), b.data(), out.data(), a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    const long double want = std::hypotl(static_cast<long double>(a[i]), static_cast<long double>(b[i]));
    const double tol = static_cast<double>(std::abs(want)) * 1e-14 + 1e-15;
    EXPECT_NEAR(out[i], static_cast<double>(want), tol) << "hypot_f64 i=" << i;
  }
}

// Breadth functions vs libm. Ranges follow each function's domain; whether the implementation is
// a qualified Highway primitive, a minimal HYLE composition, or the scalar floor is private.
namespace {
std::vector<float> ramp(float lo, float hi, int steps) {
  std::vector<float> v;
  for (int k = 0; k <= steps; ++k) v.push_back(lo + (hi - lo) * static_cast<float>(k) / static_cast<float>(steps));
  return v;
}
}  // namespace
TEST(Math, Simd09MatchesLibm) {
  check(hyle::simd::sin_f32_batch, std::sin, ramp(-3.14f, 3.14f, 600), "sin");
  check(hyle::simd::cos_f32_batch, std::cos, ramp(-3.14f, 3.14f, 600), "cos");
  check(hyle::simd::exp2_f32_batch, std::exp2, ramp(-20.0f, 20.0f, 600), "exp2");
  check(hyle::simd::log2_f32_batch, std::log2, ramp(0.01f, 100.0f, 600), "log2");
  check(hyle::simd::erf_f32_batch, std::erf, ramp(-4.0f, 4.0f, 600), "erf");
  check(hyle::simd::atan_f32_batch, std::atan, ramp(-20.0f, 20.0f, 600), "atan");
  check(hyle::simd::cbrt_f32_batch, std::cbrt, ramp(-100.0f, 100.0f, 600), "cbrt");
  check(hyle::simd::expm1_f32_batch, std::expm1, ramp(-2.0f, 2.0f, 600), "expm1");
  check(hyle::simd::log1p_f32_batch, std::log1p, ramp(-0.9f, 20.0f, 600), "log1p");
  check(hyle::simd::asinh_f32_batch, std::asinh, ramp(-20.0f, 20.0f, 600), "asinh");
  check(hyle::simd::acosh_f32_batch, std::acosh, ramp(1.0f, 50.0f, 600), "acosh");
}

TEST(Math, DirectHighwayPreservesSignedZero) {
  constexpr std::size_t n = 64;
  alignas(64) float f_input[n], f_y[n], f_x[n], f_out[n];
  alignas(64) double d_input[n], d_y[n], d_x[n], d_out[n];
  for (std::size_t i = 0; i < n; ++i) {
    const bool negative_y = (i & 1U) != 0;
    const bool negative_x = (i & 2U) == 0;
    f_input[i] = negative_y ? -0.0f : 0.0f;
    d_input[i] = negative_y ? -0.0 : 0.0;
    f_y[i] = f_input[i];
    d_y[i] = d_input[i];
    f_x[i] = negative_x ? -0.0f : 0.0f;
    d_x[i] = negative_x ? -0.0 : 0.0;
  }

  for (const auto fn : {hyle::simd::sin_f32_batch,
                        hyle::simd::expm1_f32_batch}) {
    fn(f_input, f_out, n);
    for (std::size_t i = 0; i < n; ++i) {
      EXPECT_EQ(f_out[i], 0.0f) << "f32 lane=" << i;
      EXPECT_EQ(std::signbit(f_out[i]), std::signbit(f_input[i]))
          << "f32 lane=" << i;
    }
  }
  for (const auto fn : {hyle::simd::sin_f64_batch,
                        hyle::simd::expm1_f64_batch}) {
    fn(d_input, d_out, n);
    for (std::size_t i = 0; i < n; ++i) {
      EXPECT_EQ(d_out[i], 0.0) << "f64 lane=" << i;
      EXPECT_EQ(std::signbit(d_out[i]), std::signbit(d_input[i]))
          << "f64 lane=" << i;
    }
  }

  hyle::simd::atan2_f32_batch(f_y, f_x, f_out, n);
  hyle::simd::atan2_f64_batch(d_y, d_x, d_out, n);
  for (std::size_t i = 0; i < n; ++i) {
    const float f_want = std::atan2(f_y[i], f_x[i]);
    const double d_want = std::atan2(d_y[i], d_x[i]);
    EXPECT_NEAR(f_out[i], f_want, 4.0f * std::numeric_limits<float>::epsilon())
        << "atan2 f32 lane=" << i;
    EXPECT_NEAR(d_out[i], d_want, 8.0 * std::numeric_limits<double>::epsilon())
        << "atan2 f64 lane=" << i;
    EXPECT_EQ(std::signbit(f_out[i]), std::signbit(f_want))
        << "atan2 f32 lane=" << i;
    EXPECT_EQ(std::signbit(d_out[i]), std::signbit(d_want))
        << "atan2 f64 lane=" << i;
  }
}

TEST(Math, DirectHighwayFullRangeSpecialValues) {
  const auto check_f32_specials = [](auto fn, auto ref, const char* name) {
    const float inf = std::numeric_limits<float>::infinity();
    const std::array<float, 8> pattern = {
        -inf, -1.0f, -0.0f, 0.0f, 1.0f, inf,
        std::numeric_limits<float>::quiet_NaN(), 2.0f};
    std::vector<float> in;
    for (int repeat = 0; repeat < 8; ++repeat)
      in.insert(in.end(), pattern.begin(), pattern.end());
    std::vector<float> out(in.size());
    fn(in.data(), out.data(), out.size());
    for (std::size_t i = 0; i < in.size(); ++i) {
      const float want = ref(in[i]);
      if (std::isnan(want)) {
        EXPECT_TRUE(std::isnan(out[i])) << name << " i=" << i;
      } else if (std::isinf(want) || want == 0.0f) {
        EXPECT_EQ(out[i], want) << name << " i=" << i;
        EXPECT_EQ(std::signbit(out[i]), std::signbit(want))
            << name << " i=" << i;
      } else {
        EXPECT_NEAR(out[i], want, std::abs(want) * 8e-7f + 1e-6f)
            << name << " i=" << i;
      }
    }
  };
  const auto check_f64_specials = [](auto fn, auto ref, const char* name) {
    const double inf = std::numeric_limits<double>::infinity();
    const std::array<double, 8> pattern = {
        -inf, -1.0, -0.0, 0.0, 1.0, inf,
        std::numeric_limits<double>::quiet_NaN(), 2.0};
    std::vector<double> in;
    for (int repeat = 0; repeat < 8; ++repeat)
      in.insert(in.end(), pattern.begin(), pattern.end());
    std::vector<double> out(in.size());
    fn(in.data(), out.data(), out.size());
    for (std::size_t i = 0; i < in.size(); ++i) {
      const double want = ref(in[i]);
      if (std::isnan(want)) {
        EXPECT_TRUE(std::isnan(out[i])) << name << " i=" << i;
      } else if (std::isinf(want) || want == 0.0) {
        EXPECT_EQ(out[i], want) << name << " i=" << i;
        EXPECT_EQ(std::signbit(out[i]), std::signbit(want))
            << name << " i=" << i;
      } else {
        EXPECT_NEAR(out[i], want, std::abs(want) * 1e-14 + 1e-15)
            << name << " i=" << i;
      }
    }
  };

  check_f32_specials(hyle::simd::tanh_f32_batch,
                     [](float x) { return std::tanh(x); }, "tanh_f32");
  check_f32_specials(hyle::simd::erf_f32_batch,
                     [](float x) { return std::erf(x); }, "erf_f32");
  check_f32_specials(hyle::simd::atan_f32_batch,
                     [](float x) { return std::atan(x); }, "atan_f32");
  check_f32_specials(hyle::simd::cbrt_f32_batch,
                     [](float x) { return std::cbrt(x); }, "cbrt_f32");
  check_f32_specials(hyle::simd::asinh_f32_batch,
                     [](float x) { return std::asinh(x); }, "asinh_f32");

  check_f64_specials(hyle::simd::tanh_f64_batch,
                     [](double x) { return std::tanh(x); }, "tanh_f64");
  check_f64_specials(hyle::simd::erf_f64_batch,
                     [](double x) { return std::erf(x); }, "erf_f64");
  check_f64_specials(hyle::simd::atan_f64_batch,
                     [](double x) { return std::atan(x); }, "atan_f64");
  check_f64_specials(hyle::simd::cbrt_f64_batch,
                     [](double x) { return std::cbrt(x); }, "cbrt_f64");
  check_f64_specials(hyle::simd::asinh_f64_batch,
                     [](double x) { return std::asinh(x); }, "asinh_f64");
}

// The current locked Highway release directly provides lgamma/tgamma. Ranges avoid the poles
// (0, negative integers) and f32 overflow: tgamma overflows past ~35, so cap at 10
// (tgamma(10)=362880). Gamma has a wider upstream ULP class than the common transcendentals, so
// keep a dedicated tolerance instead of weakening the general math gate.
TEST(Math, DirectHighwayGammaMatchesLibm) {
  auto check_gamma = [](void (*fn)(const float*, float*, std::size_t), double (*ref)(double),
                        const std::vector<float>& in, const char* name, double reltol) {
    std::vector<float> out(in.size());
    fn(in.data(), out.data(), in.size());
    for (std::size_t i = 0; i < in.size(); ++i) {
      const double want = ref(static_cast<double>(in[i]));
      const double got = static_cast<double>(out[i]);
      const double tol = std::abs(want) * reltol + 1e-5;
      EXPECT_NEAR(got, want, tol) << name << " in=" << in[i] << " i=" << i;
    }
  };
  // lgamma over (0, 50] (skip the pole at 0): log-gamma, modest magnitude.
  check_gamma(hyle::simd::lgamma_f32_batch, std::lgamma, ramp(0.05f, 50.0f, 600), "lgamma", 4e-6);
  // tgamma over (0, 10]: Gamma itself, capped below f32 overflow (~35).
  check_gamma(hyle::simd::tgamma_f32_batch, std::tgamma, ramp(0.05f, 10.0f, 600), "tgamma", 4e-6);
}

// Full-range hypot uses Highway exponent scaling and must match std::hypot even when a*a overflows.
TEST(Math, HypotMatchesLibm) {
  std::vector<float> a, b;
  // grid of (a,b) pairs across small, large, and overflow-prone magnitudes + signs.
  for (float av : {-1e3f, -3.5f, 0.0f, 2.0f, 100.0f, 1e18f, 3e19f})
    for (float bv : {-1e3f, -0.5f, 0.0f, 4.0f, 100.0f, 1e18f, 3e19f}) { a.push_back(av); b.push_back(bv); }
  std::vector<float> out(a.size());
  hyle::simd::hypot_f32_batch(a.data(), b.data(), out.data(), a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double want = std::hypot(static_cast<double>(a[i]), static_cast<double>(b[i]));
    const double got = static_cast<double>(out[i]);
    const double tol = std::abs(want) * 8e-7 + 1e-6;
    EXPECT_NEAR(got, want, tol) << "hypot a=" << a[i] << " b=" << b[i] << " i=" << i;
  }
}

TEST(Math, HypotInRangeWithinContract) {
  std::vector<float> a, b;
  for (float av : {-1e3f, -3.5f, 0.0f, 1e-10f, 2.0f, 100.0f, 1.2e19f})
    for (float bv : {-1e3f, -0.5f, 0.0f, 1e-10f, 4.0f, 100.0f, 1.2e19f}) {
      a.push_back(av);
      b.push_back(bv);
    }
  const auto check = [&](const std::vector<float>& out) {
    for (std::size_t i = 0; i < a.size(); ++i) {
      const float want = std::sqrt(std::fma(a[i], a[i], b[i] * b[i]));
      EXPECT_NEAR(out[i], want, std::abs(want) * 8e-7f + 1e-6f) << "i=" << i;
    }
  };
  std::vector<float> out(a.size());
  hyle::simd::hypot_in_range_f32_batch(a.data(), b.data(), out.data(), a.size());
  check(out);
  auto in_place_a = a;
  hyle::simd::hypot_in_range_f32_batch(in_place_a.data(), b.data(), in_place_a.data(), a.size());
  check(in_place_a);
  auto in_place_b = b;
  hyle::simd::hypot_in_range_f32_batch(a.data(), in_place_b.data(), in_place_b.data(), a.size());
  check(in_place_b);
}

// Tail handling across values below, at, and above representative vector widths.
TEST(Math, TailSizes) {
  for (std::size_t n : {std::size_t{1}, std::size_t{7}, std::size_t{8}, std::size_t{9}, std::size_t{17}}) {
    std::vector<float> in(n);
    for (std::size_t i = 0; i < n; ++i) in[i] = 0.1f + 0.3f * static_cast<float>(i);
    std::vector<float> oe(n), ol(n), ot(n);
    exp_f32_batch(in.data(), oe.data(), n);
    log_f32_batch(in.data(), ol.data(), n);
    tanh_f32_batch(in.data(), ot.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
      EXPECT_NEAR(oe[i], std::exp(in[i]), std::abs(std::exp(in[i])) * kRelTol + kAbsTol) << "exp n=" << n;
      EXPECT_NEAR(ol[i], std::log(in[i]), std::abs(std::log(in[i])) * kRelTol + kAbsTol) << "log n=" << n;
      EXPECT_NEAR(ot[i], std::tanh(in[i]), std::abs(std::tanh(in[i])) * kRelTol + kAbsTol) << "tanh n=" << n;
    }
  }
}

// Active-tier sanity. Correctness is proved independently above; this property ensures the
// runtime diagnostic names the fixed target actually compiled for the host.
TEST(Math, ActiveVariantReported) {
  const Isa v = hyle::simd::runtime().math.isa;
#if defined(HYLE_ARCH_X64)
  // x86: scalar floor, or locked-Highway AVX2 / AVX-512 target when available.
  EXPECT_TRUE(v == Isa::kScalar || v == Isa::kAvx2 || v == Isa::kAvx512);
#elif defined(HYLE_ARCH_ARM64)
  // aarch64: Highway NEON cores are always compiled (ARMv8 baseline) -> kNeon.
  EXPECT_EQ(v, Isa::kNeon);
#else
  EXPECT_EQ(v, Isa::kScalar);
#endif
}

// AI activation combined cores vs scalar reference. Tolerance is the ~3.5-ULP transcendental
// class the underlying exp/tanh/erf inherit, with margin (the combine arithmetic adds a little).
TEST(Math, Activations) {
  std::vector<float> in;
  for (int k = -100; k <= 100; ++k) in.push_back(static_cast<float>(k) * 0.1f);  // [-10,10]
  const std::size_t n = in.size();
  const float sq2inv = 0.70710678118654752440f, sq2pi = 0.79788456080286535588f, ca = 0.044715f;
  std::vector<float> out(n);

  hyle::simd::gelu_f32_batch(in.data(), out.data(), n);
  for (std::size_t i = 0; i < n; ++i) {
    const double ref = 0.5 * in[i] * (1.0 + std::erf(in[i] * (double)sq2inv));
    EXPECT_NEAR(out[i], ref, std::abs(ref) * 1e-4 + 1e-5) << "gelu " << in[i];
  }
  hyle::simd::gelu_tanh_f32_batch(in.data(), out.data(), n);
  for (std::size_t i = 0; i < n; ++i) {
    const double x = in[i];
    const double ref = 0.5 * x * (1.0 + std::tanh(sq2pi * x * (1.0 + ca * x * x)));
    EXPECT_NEAR(out[i], ref, std::abs(ref) * 1e-4 + 1e-5) << "gelu_tanh " << in[i];
  }
  hyle::simd::sigmoid_f32_batch(in.data(), out.data(), n);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_NEAR(out[i], 1.0 / (1.0 + std::exp(-(double)in[i])), 1e-5) << "sigmoid " << in[i];
  hyle::simd::silu_f32_batch(in.data(), out.data(), n);
  for (std::size_t i = 0; i < n; ++i)
    EXPECT_NEAR(out[i], in[i] / (1.0 + std::exp(-(double)in[i])), std::abs(in[i]) * 1e-5 + 1e-5) << "silu " << in[i];
}

// softmax / log_softmax: match a double-precision naive reference; softmax sums to 1.
TEST(Math, Softmax) {
  for (std::size_t n : {std::size_t{1}, std::size_t{7}, std::size_t{64}, std::size_t{1000}}) {
    std::vector<float> x(n), sm(n);
    for (std::size_t i = 0; i < n; ++i) x[i] = std::sin(static_cast<float>(i) * 0.7f) * 5.0f;  // spread
    // reference (stable, double)
    double mx = x[0];
    for (std::size_t i = 1; i < n; ++i) mx = x[i] > mx ? x[i] : mx;
    double sum = 0.0;
    std::vector<double> ref(n);
    for (std::size_t i = 0; i < n; ++i) { ref[i] = std::exp((double)x[i] - mx); sum += ref[i]; }
    for (std::size_t i = 0; i < n; ++i) ref[i] /= sum;

    sm = x;
    hyle::simd::softmax_f32(sm.data(), n);
    double got_sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) { EXPECT_NEAR(sm[i], ref[i], 1e-5) << "softmax n=" << n << " i=" << i; got_sum += sm[i]; }
    EXPECT_NEAR(got_sum, 1.0, 1e-4) << "softmax sum n=" << n;

    // log_softmax = log(softmax)
    std::vector<float> ls = x;
    hyle::simd::log_softmax_f32(ls.data(), n);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_NEAR(ls[i], std::log(ref[i]), 1e-4) << "log_softmax n=" << n << " i=" << i;
  }
}

// ---- COMPLEX SoA compositions over real primitives vs std::complex oracle ------------
// SoA planar (re/im separate arrays). Ranges kept moderate (complex exp/pow overflow fast).
// Tolerance is the ~3.5-ULP transcendental class scaled by magnitude.
namespace {
void check_cunary(void (*fn)(const float*, const float*, float*, float*, std::size_t),
                  std::complex<double> (*ref)(std::complex<double>), const char* name) {
  constexpr int N = 200;
  std::vector<float> sr(N), si(N), dr(N), di(N);
  // Offset both ramps so no sample lands on (0,0): clog/csqrt have a branch-point singularity
  // there (std::log(0)=(-inf,0)); that's a domain edge, not a core-accuracy question.
  for (int i = 0; i < N; ++i) { sr[i] = -3.0f + 6.0f * i / N + 0.017f; si[i] = -2.5f + 5.0f * i / N + 0.013f; }
  fn(sr.data(), si.data(), dr.data(), di.data(), N);
  for (int i = 0; i < N; ++i) {
    std::complex<double> want = ref(std::complex<double>(sr[i], si[i]));
    const double tol = std::abs(want) * 4e-6 + 1e-5;
    EXPECT_NEAR(dr[i], want.real(), tol) << name << " re i=" << i;
    EXPECT_NEAR(di[i], want.imag(), tol) << name << " im i=" << i;
  }
}
std::complex<double> r_exp(std::complex<double> z) { return std::exp(z); }
std::complex<double> r_log(std::complex<double> z) { return std::log(z); }
std::complex<double> r_sqrt(std::complex<double> z) { return std::sqrt(z); }
std::complex<double> r_sin(std::complex<double> z) { return std::sin(z); }
std::complex<double> r_cos(std::complex<double> z) { return std::cos(z); }
std::complex<double> r_tan(std::complex<double> z) { return std::tan(z); }
std::complex<double> r_sinh(std::complex<double> z) { return std::sinh(z); }
std::complex<double> r_cosh(std::complex<double> z) { return std::cosh(z); }
std::complex<double> r_tanh(std::complex<double> z) { return std::tanh(z); }

void check_cunary_f64(
    void (*fn)(const double*, const double*, double*, double*, std::size_t),
    std::complex<double> (*ref)(std::complex<double>), const char* name) {
  constexpr int N = 201;
  std::vector<double> sr(N), si(N), dr(N), di(N);
  for (int i = 0; i < N; ++i) {
    sr[i] = -3.0 + 6.0 * i / N + 0.017;
    si[i] = -2.5 + 5.0 * i / N + 0.013;
  }
  fn(sr.data(), si.data(), dr.data(), di.data(), N);
  for (int i = 0; i < N; ++i) {
    const std::complex<double> want = ref({sr[i], si[i]});
    const double tol = std::abs(want) * 3e-11 + 2e-12;
    EXPECT_NEAR(dr[i], want.real(), tol) << name << " re i=" << i;
    EXPECT_NEAR(di[i], want.imag(), tol) << name << " im i=" << i;
  }
}
}  // namespace

TEST(Math, ComplexTranscendentals) {
  check_cunary(hyle::simd::cexp_f32_batch, r_exp, "cexp");
  check_cunary(hyle::simd::clog_f32_batch, r_log, "clog");
  check_cunary(hyle::simd::csqrt_f32_batch, r_sqrt, "csqrt");
  check_cunary(hyle::simd::csin_f32_batch, r_sin, "csin");
  check_cunary(hyle::simd::ccos_f32_batch, r_cos, "ccos");
  check_cunary(hyle::simd::ctan_f32_batch, r_tan, "ctan");
  check_cunary(hyle::simd::csinh_f32_batch, r_sinh, "csinh");
  check_cunary(hyle::simd::ccosh_f32_batch, r_cosh, "ccosh");
  check_cunary(hyle::simd::ctanh_f32_batch, r_tanh, "ctanh");
}

TEST(Math, ComplexTranscendentalsF64) {
  check_cunary_f64(hyle::simd::cexp_f64_batch, r_exp, "cexp_f64");
  check_cunary_f64(hyle::simd::clog_f64_batch, r_log, "clog_f64");
  check_cunary_f64(hyle::simd::csqrt_f64_batch, r_sqrt, "csqrt_f64");
  check_cunary_f64(hyle::simd::csin_f64_batch, r_sin, "csin_f64");
  check_cunary_f64(hyle::simd::ccos_f64_batch, r_cos, "ccos_f64");
  check_cunary_f64(hyle::simd::ctan_f64_batch, r_tan, "ctan_f64");
  check_cunary_f64(hyle::simd::csinh_f64_batch, r_sinh, "csinh_f64");
  check_cunary_f64(hyle::simd::ccosh_f64_batch, r_cosh, "ccosh_f64");
  check_cunary_f64(hyle::simd::ctanh_f64_batch, r_tanh, "ctanh_f64");
}

TEST(Math, ComplexF64SpecialTailAndExactAlias) {
  using Fn = void (*)(const double*, const double*, double*, double*, std::size_t);
  struct Op {
    const char* name;
    Fn fn;
    std::complex<double> (*ref)(std::complex<double>);
  };
  const Op ops[] = {
      {"cexp", hyle::simd::cexp_f64_batch, r_exp},
      {"clog", hyle::simd::clog_f64_batch, r_log},
      {"csqrt", hyle::simd::csqrt_f64_batch, r_sqrt},
      {"csin", hyle::simd::csin_f64_batch, r_sin},
      {"ccos", hyle::simd::ccos_f64_batch, r_cos},
      {"ctan", hyle::simd::ctan_f64_batch, r_tan},
      {"csinh", hyle::simd::csinh_f64_batch, r_sinh},
      {"ccosh", hyle::simd::ccosh_f64_batch, r_cosh},
      {"ctanh", hyle::simd::ctanh_f64_batch, r_tanh},
  };
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double denorm = std::numeric_limits<double>::denorm_min();
  const std::vector<std::complex<double>> input = {
      {0.0, 0.0},       {-0.0, 0.0},  {0.0, -0.0}, {denorm, -denorm},
      {1.0e-200, 2e-200}, {1.0, -2.0},  {inf, 0.0},  {-inf, 0.0},
      {0.0, inf},       {0.0, -inf}, {nan, 1.0},  {1.0, nan},
      {2.0, -3.0},      {-0.5, 0.75}, {3.0, 2.0}, {0.25, -0.5},
      {1.5, 0.125},
  };
  for (const Op& op : ops) {
    std::vector<double> re(input.size()), im(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
      re[i] = input[i].real();
      im[i] = input[i].imag();
    }
    op.fn(re.data(), im.data(), re.data(), im.data(), input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
      const std::complex<double> want = op.ref(input[i]);
      for (const auto& [got, expected, part] :
           {std::tuple{re[i], want.real(), "re"}, std::tuple{im[i], want.imag(), "im"}}) {
        if (std::isnan(expected)) {
          EXPECT_TRUE(std::isnan(got)) << op.name << ' ' << part << " i=" << i;
        } else if (std::isinf(expected)) {
          EXPECT_TRUE(std::isinf(got)) << op.name << ' ' << part << " i=" << i;
          EXPECT_EQ(std::signbit(got), std::signbit(expected))
              << op.name << ' ' << part << " i=" << i;
        } else {
          EXPECT_NEAR(got, expected, std::abs(expected) * 4e-11 + 2e-12)
              << op.name << ' ' << part << " i=" << i;
          if (expected == 0.0) {
            EXPECT_EQ(std::signbit(got), std::signbit(expected))
                << op.name << ' ' << part << " i=" << i;
          }
        }
      }
    }
  }
}

TEST(Math, ComplexF64PowTailAndExactAlias) {
  constexpr std::size_t n = 67;
  std::vector<double> zr(n), zi(n), wr(n), wi(n);
  for (std::size_t i = 0; i < n; ++i) {
    zr[i] = 0.25 + 0.013 * double(i);
    zi[i] = -0.75 + 0.017 * double(i);
    wr[i] = 0.5 + 0.009 * double(i);
    wi[i] = 0.2 + 0.003 * double(i);
  }
  const std::vector<double> original_re = zr;
  const std::vector<double> original_im = zi;
  hyle::simd::cpow_f64_batch(zr.data(), zi.data(), wr.data(), wi.data(), zr.data(), zi.data(), n);
  for (std::size_t i = 0; i < n; ++i) {
    const std::complex<double> want =
        std::pow(std::complex<double>(original_re[i], original_im[i]),
                 std::complex<double>(wr[i], wi[i]));
    EXPECT_NEAR(zr[i], want.real(), std::abs(want) * 5e-11 + 2e-12) << "re i=" << i;
    EXPECT_NEAR(zi[i], want.imag(), std::abs(want) * 5e-11 + 2e-12) << "im i=" << i;
  }
}

void expect_complex_component(float got, float want, const char* part, std::size_t i) {
  if (std::isnan(want)) {
    EXPECT_TRUE(std::isnan(got)) << part << " i=" << i;
  } else if (std::isinf(want)) {
    EXPECT_TRUE(std::isinf(got)) << part << " i=" << i;
    EXPECT_EQ(std::signbit(got), std::signbit(want)) << part << " i=" << i;
  } else {
    EXPECT_NEAR(got, want, std::abs(want) * 5e-6f + 1e-6f) << part << " i=" << i;
    if (want == 0.0f) {
      EXPECT_EQ(std::signbit(got), std::signbit(want)) << part << " i=" << i;
    }
  }
}

TEST(Math, ComplexSinCosSpecialTailAndAlias) {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::vector<std::complex<float>> input = {
      {0.0f, 0.0f}, {-0.0f, 0.0f}, {0.0f, -0.0f}, {1.0f, -2.0f},
      {0.0f, inf}, {0.0f, -inf}, {inf, 0.0f}, {-inf, 0.0f},
      {nan, 1.0f}, {1.0f, nan}, {1e-38f, -1e-38f}, {3.0f, 100.0f},
      {-2.0f, -100.0f}, {0.25f, 0.5f}, {-0.75f, 1.25f}, {2.0f, -3.0f}, {0.1f, 0.2f}};
  std::vector<float> re(input.size()), im(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) { re[i] = input[i].real(); im[i] = input[i].imag(); }

  for (bool cosine : {false, true}) {
    std::vector<float> got_re = re;
    std::vector<float> got_im = im;
    if (cosine)
      hyle::simd::ccos_f32_batch(got_re.data(), got_im.data(), got_re.data(), got_im.data(), input.size());
    else
      hyle::simd::csin_f32_batch(got_re.data(), got_im.data(), got_re.data(), got_im.data(), input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
      const std::complex<float> want = cosine ? std::cos(input[i]) : std::sin(input[i]);
      expect_complex_component(got_re[i], want.real(), cosine ? "ccos re" : "csin re", i);
      expect_complex_component(got_im[i], want.imag(), cosine ? "ccos im" : "csin im", i);
    }
  }
}

TEST(Math, ComplexTanhSpecialTailAndAlias) {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float denorm = std::numeric_limits<float>::denorm_min();
  const std::vector<std::complex<float>> input = {
      {0.0f, 0.0f}, {-0.0f, 0.0f}, {0.0f, -0.0f}, {0.7f, 0.0f},
      {0.7f, -0.0f}, {1e-38f, 0.7f}, {-1e-38f, -0.7f}, {0.7f, denorm},
      {0.7f, -denorm}, {0.01f, 1.5707963f - 0.001f}, {-0.01f, -1.5707963f + 0.001f},
      {3.0f, -2.0f}, {-3.0f, 2.0f}, {39.0f, 0.3f}, {-39.0f, -0.3f},
      {45.0f, 0.2f}, {-45.0f, -0.2f}, {inf, 0.0f}, {-inf, 0.0f},
      {nan, 1.0f}, {1.0f, nan}};
  std::vector<float> re(input.size()), im(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    re[i] = input[i].real();
    im[i] = input[i].imag();
  }
  hyle::simd::ctanh_f32_batch(re.data(), im.data(), re.data(), im.data(), input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    const std::complex<float> want = std::tanh(input[i]);
    expect_complex_component(re[i], want.real(), "ctanh re", i);
    expect_complex_component(im[i], want.imag(), "ctanh im", i);
  }

  std::vector<float> sr(input.size() + 2, -123.0f), si(input.size() + 2, -123.0f);
  std::vector<float> dr(input.size() + 2, -321.0f), di(input.size() + 2, -321.0f);
  for (std::size_t i = 0; i < input.size(); ++i) {
    sr[i + 1] = input[i].real();
    si[i + 1] = input[i].imag();
  }
  hyle::simd::ctanh_f32_batch(sr.data() + 1, si.data() + 1, dr.data() + 1, di.data() + 1,
                               input.size());
  EXPECT_EQ(sr.front(), -123.0f);
  EXPECT_EQ(sr.back(), -123.0f);
  EXPECT_EQ(si.front(), -123.0f);
  EXPECT_EQ(si.back(), -123.0f);
  EXPECT_EQ(dr.front(), -321.0f);
  EXPECT_EQ(dr.back(), -321.0f);
  EXPECT_EQ(di.front(), -321.0f);
  EXPECT_EQ(di.back(), -321.0f);
  for (std::size_t i = 0; i < input.size(); ++i) {
    const std::complex<float> want = std::tanh(input[i]);
    expect_complex_component(dr[i + 1], want.real(), "ctanh unaligned re", i);
    expect_complex_component(di[i + 1], want.imag(), "ctanh unaligned im", i);
  }
}

TEST(Math, ComplexTanhExtremeImaginaryFallsBack) {
  const float max = std::numeric_limits<float>::max();
  const std::vector<std::complex<float>> input = {
      {25.0f, max}, {-25.0f, -max}, {25.0f, 0.75f * max}, {-25.0f, -0.75f * max},
      {25.0f, 0.25f}, {-25.0f, -0.25f}, {30.0f, 0.5f}, {-30.0f, -0.5f}};
  std::vector<float> re(input.size()), im(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    re[i] = input[i].real();
    im[i] = input[i].imag();
  }
  hyle::simd::ctanh_f32_batch(re.data(), im.data(), re.data(), im.data(), input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    const std::complex<float> want = std::tanh(input[i]);
    expect_complex_component(re[i], want.real(), "ctanh extreme re", i);
    expect_complex_component(im[i], want.imag(), "ctanh extreme im", i);
  }
}

TEST(Math, ComplexSinCosTinyImaginary) {
  constexpr std::size_t n = 16;
  const float tiny[] = {1e-38f, -1e-38f, 1e-8f, -1e-8f, 1e-7f, -1e-7f,
                        1e-6f, -1e-6f, 1e-5f, -1e-5f, 1e-4f, -1e-4f,
                        1e-3f, -1e-3f, 1e-2f, -1e-2f};
  std::vector<float> re(n), im(n), dr(n), di(n);
  for (std::size_t i = 0; i < n; ++i) { re[i] = 0.25f + 0.03f * static_cast<float>(i); im[i] = tiny[i]; }
  for (bool cosine : {false, true}) {
    if (cosine) hyle::simd::ccos_f32_batch(re.data(), im.data(), dr.data(), di.data(), n);
    else hyle::simd::csin_f32_batch(re.data(), im.data(), dr.data(), di.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
      const std::complex<float> want = cosine
          ? std::cos(std::complex<float>(re[i], im[i]))
          : std::sin(std::complex<float>(re[i], im[i]));
      EXPECT_NEAR(dr[i], want.real(), std::abs(want.real()) * 5e-6f + 1e-7f);
      ASSERT_NE(want.imag(), 0.0f);
      EXPECT_NEAR(di[i] / want.imag(), 1.0f, 2e-5f) << "i=" << i;
    }
  }
}

TEST(Math, ComplexPow) {
  constexpr int N = 128;
  std::vector<float> zr(N), zi(N), wr(N), wi(N), dr(N), di(N);
  for (int i = 0; i < N; ++i) {
    zr[i] = 0.3f + 2.0f * i / N; zi[i] = -1.0f + 2.0f * i / N;  // base away from 0 (log branch)
    wr[i] = 0.5f + 1.0f * i / N; wi[i] = 0.2f;
  }
  hyle::simd::cpow_f32_batch(zr.data(), zi.data(), wr.data(), wi.data(), dr.data(), di.data(), N);
  for (int i = 0; i < N; ++i) {
    std::complex<double> want = std::pow(std::complex<double>(zr[i], zi[i]),
                                         std::complex<double>(wr[i], wi[i]));
    const double tol = std::abs(want) * 1e-5 + 1e-5;
    EXPECT_NEAR(dr[i], want.real(), tol) << "cpow re i=" << i;
    EXPECT_NEAR(di[i], want.imag(), tol) << "cpow im i=" << i;
  }
}

TEST(Math, ComplexPowSpecialTailAndAlias) {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float denorm = std::numeric_limits<float>::denorm_min();
  const std::vector<std::complex<float>> base = {
      {0.0f, 0.0f}, {-0.0f, 0.0f}, {0.0f, -0.0f}, {-2.0f, 0.0f},
      {-2.0f, -0.0f}, {1e-38f, 0.25f}, {0.25f, 1e-38f}, {1e18f, 0.5f},
      {0.5f, -0.75f}, {-1.5f, 2.0f}, {inf, 1.0f}, {1.0f, inf},
      {nan, 1.0f}, {1.0f, nan}, {2.0f, -3.0f}, {-0.25f, 0.75f}, {3.0f, 2.0f},
      {0.5f, -0.75f}, {0.5f, -0.75f}, {0.5f, -0.75f}, {0.5f, -0.75f},
      {0.5f, -0.75f}, {0.5f, -0.75f}};
  const std::vector<std::complex<float>> exponent = {
      {0.0f, 0.0f}, {1.0f, 0.0f}, {-1.0f, 0.0f}, {0.5f, 0.25f},
      {0.5f, -0.25f}, {1.5f, 0.0f}, {-0.5f, 1.0f}, {0.1f, -0.2f},
      {0.0f, 1.0f}, {2.0f, -0.5f}, {1.0f, 0.0f}, {0.5f, 0.25f},
      {1.0f, 0.0f}, {0.5f, -0.5f}, {3.0f, 2.0f}, {-2.0f, 1.5f}, {0.75f, -0.4f},
      {0.0f, 0.3f}, {-0.0f, 0.3f}, {0.3f, 0.0f}, {0.3f, -0.0f},
      {denorm, 0.3f}, {0.3f, -denorm}};
  ASSERT_EQ(base.size(), exponent.size());
  std::vector<float> zr(base.size()), zi(base.size()), wr(base.size()), wi(base.size());
  for (std::size_t i = 0; i < base.size(); ++i) {
    zr[i] = base[i].real();
    zi[i] = base[i].imag();
    wr[i] = exponent[i].real();
    wi[i] = exponent[i].imag();
  }
  hyle::simd::cpow_f32_batch(zr.data(), zi.data(), wr.data(), wi.data(), zr.data(), zi.data(),
                              base.size());
  for (std::size_t i = 0; i < base.size(); ++i) {
    const std::complex<float> want = std::pow(base[i], exponent[i]);
    expect_complex_component(zr[i], want.real(), "cpow special re", i);
    expect_complex_component(zi[i], want.imag(), "cpow special im", i);
  }

  std::vector<float> ur(base.size() + 2, -123.0f), ui(base.size() + 2, -123.0f);
  std::vector<float> uwr(base.size() + 2, -123.0f), uwi(base.size() + 2, -123.0f);
  std::vector<float> dr(base.size() + 2, -321.0f), di(base.size() + 2, -321.0f);
  for (std::size_t i = 0; i < base.size(); ++i) {
    ur[i + 1] = base[i].real();
    ui[i + 1] = base[i].imag();
    uwr[i + 1] = exponent[i].real();
    uwi[i + 1] = exponent[i].imag();
  }
  hyle::simd::cpow_f32_batch(ur.data() + 1, ui.data() + 1, uwr.data() + 1, uwi.data() + 1,
                              dr.data() + 1, di.data() + 1, base.size());
  for (const auto* guarded : {ur.data(), ui.data(), uwr.data(), uwi.data(), dr.data(), di.data()}) {
    EXPECT_EQ(guarded[0], guarded == dr.data() || guarded == di.data() ? -321.0f : -123.0f);
    EXPECT_EQ(guarded[base.size() + 1], guarded == dr.data() || guarded == di.data() ? -321.0f : -123.0f);
  }
  for (std::size_t i = 0; i < base.size(); ++i) {
    const std::complex<float> want = std::pow(base[i], exponent[i]);
    expect_complex_component(dr[i + 1], want.real(), "cpow unaligned re", i);
    expect_complex_component(di[i + 1], want.imag(), "cpow unaligned im", i);
  }
}

// simd-18 批G smoke: new APIs compile, dispatch, and match libm loosely.
TEST(Math, Simd18BreadthG) {
  constexpr std::size_t n = 64;
  alignas(64) float x[n], y[n], z[n], s[n], c[n];
  for (std::size_t i = 0; i < n; ++i) {
    x[i] = 0.01f * static_cast<float>(i) - 0.3f;  // in-domain for most
    y[i] = 0.02f * static_cast<float>(i) + 0.5f;
  }
  hyle::simd::acos_f32_batch(x, z, n);
  for (std::size_t i = 0; i < n; ++i) {
    if (x[i] >= -1.f && x[i] <= 1.f) {
      EXPECT_NEAR(z[i], std::acos(x[i]), 1e-4f) << "acos i=" << i;
    }
  }
  hyle::simd::asin_f32_batch(x, z, n);
  hyle::simd::sinh_f32_batch(x, z, n);
  hyle::simd::cosh_f32_batch(x, z, n);
  hyle::simd::tan_f32_batch(x, z, n);
  hyle::simd::atanh_f32_batch(x, z, n);  // |x|<1 for domain
  hyle::simd::log10_f32_batch(y, z, n);  // y>0
  hyle::simd::atan2_f32_batch(x, y, z, n);
  hyle::simd::pow_f32_batch(y, x, z, n);
  hyle::simd::sincos_f32_batch(x, s, c, n);
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(s[i], std::sin(x[i]), 1e-4f);
    EXPECT_NEAR(c[i], std::cos(x[i]), 1e-4f);
  }
  // Vendor-gap operations: assert the HYLE composition/scalar floor against libm.
  hyle::simd::erfc_f32_batch(x, z, n);
  for (std::size_t i = 0; i < n; ++i) EXPECT_NEAR(z[i], std::erfc(x[i]), 1e-4f) << "erfc i=" << i;
  hyle::simd::exp10_f32_batch(x, z, n);
  for (std::size_t i = 0; i < n; ++i) EXPECT_NEAR(z[i], std::pow(10.f, x[i]), 1e-3f) << "exp10 i=" << i;
  hyle::simd::fmod_f32_batch(y, x, z, n);
  for (std::size_t i = 0; i < n; ++i) {
    if (x[i] != 0.f) {
      EXPECT_NEAR(z[i], std::fmod(y[i], x[i]), 1e-4f) << "fmod i=" << i;
    }
  }
  hyle::simd::remainder_f32_batch(y, x, z, n);
  for (std::size_t i = 0; i < n; ++i) {
    if (x[i] != 0.f) {
      EXPECT_NEAR(z[i], std::remainder(y[i], x[i]), 1e-4f) << "rem i=" << i;
    }
  }
  hyle::simd::nextafter_f32_batch(x, y, z, n);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(z[i], std::nextafter(x[i], y[i])) << "nextafter i=" << i;
  hyle::simd::fdim_f32_batch(y, x, z, n);
  for (std::size_t i = 0; i < n; ++i) EXPECT_NEAR(z[i], std::fdim(y[i], x[i]), 1e-4f) << "fdim i=" << i;
  // f64 smoke
  alignas(64) double dx[n], dy[n], dz[n], ds[n], dc[n];
  for (std::size_t i = 0; i < n; ++i) {
    dx[i] = 0.01 * static_cast<double>(i) - 0.3;
    dy[i] = 0.02 * static_cast<double>(i) + 0.5;
  }
  hyle::simd::acos_f64_batch(dx, dz, n);
  hyle::simd::atan2_f64_batch(dx, dy, dz, n);
  hyle::simd::pow_f64_batch(dy, dx, dz, n);
  hyle::simd::sincos_f64_batch(dx, ds, dc, n);
  hyle::simd::erfc_f64_batch(dx, dz, n);
  for (std::size_t i = 0; i < n; ++i) EXPECT_NEAR(dz[i], std::erfc(dx[i]), 1e-9) << "erfc64 i=" << i;
  hyle::simd::exp10_f64_batch(dx, dz, n);
  for (std::size_t i = 0; i < n; ++i) EXPECT_NEAR(dz[i], std::pow(10.0, dx[i]), 1e-6) << "exp10_64 i=" << i;
  hyle::simd::fmod_f64_batch(dy, dx, dz, n);
  for (std::size_t i = 0; i < n; ++i) {
    if (dx[i] != 0.0) {
      EXPECT_NEAR(dz[i], std::fmod(dy[i], dx[i]), 1e-9) << "fmod64 i=" << i;
    }
  }
  hyle::simd::remainder_f64_batch(dy, dx, dz, n);
  for (std::size_t i = 0; i < n; ++i) {
    if (dx[i] != 0.0) {
      EXPECT_NEAR(dz[i], std::remainder(dy[i], dx[i]), 1e-9) << "rem64 i=" << i;
    }
  }
  hyle::simd::nextafter_f64_batch(dx, dy, dz, n);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(dz[i], std::nextafter(dx[i], dy[i])) << "nextafter64 i=" << i;
  hyle::simd::fdim_f64_batch(dy, dx, dz, n);
  for (std::size_t i = 0; i < n; ++i) EXPECT_NEAR(dz[i], std::fdim(dy[i], dx[i]), 1e-9) << "fdim64 i=" << i;
  SUCCEED();
}
