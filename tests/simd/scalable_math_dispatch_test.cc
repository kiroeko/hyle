#include <gtest/gtest.h>

#include "hyle/simd/math.h"
#include "hyle/simd/runtime.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>

namespace hs = hyle::simd;

namespace {

#if defined(HYLE_ARCH_ARM64)
constexpr hs::Isa kExpectedIsa = hs::Isa::kSve;
#elif defined(HYLE_ARCH_RISCV64)
constexpr hs::Isa kExpectedIsa = hs::Isa::kRvv;
#endif

template <class T>
void ExpectNearRelative(T actual, T expected, T tolerance) {
  const T scale = std::max(T{1}, std::abs(expected));
  EXPECT_NEAR(actual, expected, tolerance * scale);
}

}  // namespace

TEST(ScalableMathDispatch, RuntimeSelectsScalableProductTier) {
  EXPECT_EQ(hs::runtime().isa, kExpectedIsa);
  EXPECT_EQ(hs::runtime().math.isa, kExpectedIsa);

  const hs::Runtime* exact = hs::runtime_for(kExpectedIsa);
  ASSERT_NE(exact, nullptr);
  EXPECT_EQ(exact->isa, kExpectedIsa);
  EXPECT_EQ(exact->math.isa, kExpectedIsa);
}

TEST(ScalableMathDispatch, PublicBatchApiUsesScalableCoreAndTail) {
  constexpr std::size_t n = 37;
  alignas(64) float src[n], exp_out[n], log_out[n];
  alignas(64) double a[n], b[n], hypot_out[n];
  for (std::size_t i = 0; i < n; ++i) {
    src[i] = -2.0f + 4.0f * float(i + 1) / float(n + 1);
    a[i] = -3.0 + 6.0 * double(i + 1) / double(n + 1);
    b[i] = 0.25 + 0.125 * double(i % 7);
  }

  hs::exp_f32_batch(src, exp_out, n);
  hs::log_f32_batch(exp_out, log_out, n);
  hs::hypot_f64_batch(a, b, hypot_out, n);
  for (std::size_t i = 0; i < n; ++i) {
    ExpectNearRelative(exp_out[i], std::exp(src[i]), 4e-6f);
    ExpectNearRelative(log_out[i], src[i], 8e-6f);
    ExpectNearRelative(hypot_out[i], std::hypot(a[i], b[i]), 3e-13);
  }
}

TEST(ScalableMathDispatch, PublicBatchApiIncludesHyleDerivedTier) {
  constexpr std::size_t n = 37;
  alignas(64) float a[n], b[n], out[n], out_im[n];
  alignas(64) double x[n], gamma[n];
  for (std::size_t i = 0; i < n; ++i) {
    a[i] = 0.1f + 0.075f * float(i % 31);
    b[i] = 0.3f + 0.125f * float(i % 11);
    x[i] = 0.5 + 0.125 * double(i % 29);
  }

  hs::fmod_f32_batch(a, b, out, n);
  hs::tgamma_f64_batch(x, gamma, n);
  for (std::size_t i = 0; i < n; ++i) {
    ExpectNearRelative(out[i], std::fmod(a[i], b[i]), 2e-6f);
    ExpectNearRelative(gamma[i], std::tgamma(x[i]), 2e-12);
  }

  hs::clog_f32_batch(a, b, out, out_im, n);
  for (std::size_t i = 0; i < n; ++i) {
    const std::complex<float> expected = std::log(std::complex<float>(a[i], b[i]));
    ExpectNearRelative(out[i], expected.real(), 2e-5f);
    ExpectNearRelative(out_im[i], expected.imag(), 2e-5f);
  }
  hs::ctanh_f32_batch(a, b, out, out_im, n);
  for (std::size_t i = 0; i < n; ++i) {
    const std::complex<float> expected = std::tanh(std::complex<float>(a[i], b[i]));
    ExpectNearRelative(out[i], expected.real(), 2e-5f);
    ExpectNearRelative(out_im[i], expected.imag(), 2e-5f);
  }
  hs::cpow_f32_batch(a, b, b, a, out, out_im, n);
  for (std::size_t i = 0; i < n; ++i) {
    const std::complex<float> expected =
        std::pow(std::complex<float>(a[i], b[i]), std::complex<float>(b[i], a[i]));
    ExpectNearRelative(out[i], expected.real(), 3e-5f);
    ExpectNearRelative(out_im[i], expected.imag(), 3e-5f);
  }
}
