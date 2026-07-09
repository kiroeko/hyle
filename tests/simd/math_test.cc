#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

#include "hyle/simd/dispatch.h"
#include "hyle/simd/math.h"

// math_test -- exp/log/tanh batch ops vs libm reference. The batch path runs whichever
// tier is live (SLEEF AVX2 if -DHYLE_SIMD_SLEEF + AVX2 host, else scalar libm). Either
// way the result must match std:: within tolerance: scalar is exact-libm; SLEEF exp is
// _u10 (1-ULP), log/tanh are _u35 (3.5-ULP -- the AI-appropriate, ~2-3x faster tier; see
// transcendental_sleef.cc). The tolerance below is set to the 3.5-ULP class with margin.

using hyle::simd::active_math_variant;
using hyle::simd::exp_f32_batch;
using hyle::simd::Isa;
using hyle::simd::log_f32_batch;
using hyle::simd::tanh_f32_batch;

namespace {
// Relative-error tolerance vs a double-precision libm reference. Set from MEASURED worst
// case (red team 2026-06 flagged the old 1e-5 ~= 84 ULP as toothless -- a broken
// polynomial would pass): scalar tier worst 0.7 ULP (8.4e-8), SLEEF exp _u10 worst 0.5
// ULP. log/tanh moved to _u35 (3.5-ULP worst, the tier xsimd/Highway also use and the
// AI hot path needs) -- so the bound is 8e-7 ~= 6.7 ULP: ~2x margin over _u35's 3.5 ULP,
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

// simd-09 best-of functions vs libm. Same tolerance (both vendor cores are ~3.5-ULP class).
// Range chosen per function's domain. The batch op routes to the bench-decided winner
// (Highway for most, xsimd for exp2/erf) or the scalar floor -- either must match libm.
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

// xsimd-ONLY borrows (lgamma/tgamma) vs libm. Highway has no gamma, so these are a single-source
// borrow (no best-of). Ranges avoid the poles (0, negative integers) and f32 overflow: tgamma
// overflows past ~35, so cap at 10 (tgamma(10)=362880). lgamma grows slowly, safe to ~50.
// Uses a dedicated tolerance: gamma is NOT the 3.5-ULP transcendental class -- xsimd's gamma
// polynomial is looser, and tgamma's large dynamic range magnifies the trailing-arg f32 error.
TEST(Math, XsimdGammaMatchesLibm) {
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

// hypot(a,b): BINARY best-of op (xsimd won the bench). vs std::hypot over a mix of magnitudes
// including the overflow-prone case (large a,b whose squares overflow f32 but the hypotenuse
// doesn't -- std::hypot and xsimd::hypot are both overflow-safe). ~3.5-ULP class.
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

// tail handling: n not a multiple of 8 (the SLEEF kernel's vector width). All three ops
// exercised (red team: log/tanh tails were previously untested -- only exp).
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

// active tier sanity. NOTE: HYLE_SIMD_SLEEF is a library-PRIVATE define (set on hyle_simd,
// not on this test TU), so the test cannot know whether SLEEF was wired in -- it only
// observes the runtime result. Either tier is valid: scalar libm floor, or kAvx2 when the
// lib was built with SLEEF AND the host has AVX2. Correctness is proven by the
// Exp/Log/Tanh-vs-libm tests above regardless of which tier ran.
TEST(Math, ActiveVariantReported) {
  const Isa v = active_math_variant();
#if defined(HYLE_ARCH_X64)
  // x86: scalar floor, or AVX2 / AVX-512 borrow cores when the CPU has them.
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

// ---- COMPLEX transcendentals (xsimd-only borrow) vs std::complex oracle -------------
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
std::complex<double> r_tanh(std::complex<double> z) { return std::tanh(z); }
}  // namespace

TEST(Math, ComplexTranscendentals) {
  check_cunary(hyle::simd::cexp_f32_batch, r_exp, "cexp");
  check_cunary(hyle::simd::clog_f32_batch, r_log, "clog");
  check_cunary(hyle::simd::csqrt_f32_batch, r_sqrt, "csqrt");
  check_cunary(hyle::simd::csin_f32_batch, r_sin, "csin");
  check_cunary(hyle::simd::ccos_f32_batch, r_cos, "ccos");
  check_cunary(hyle::simd::ctanh_f32_batch, r_tanh, "ctanh");
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
