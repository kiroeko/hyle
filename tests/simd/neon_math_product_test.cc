#include <gtest/gtest.h>

#include "hyle/platform.h"
#include "src/simd/math/kernels.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>

#if defined(HYLE_ARCH_ARM64)

namespace {

template <class T>
void ExpectNearRelative(T actual, T expected, T tolerance) {
  EXPECT_NEAR(actual, expected, tolerance * std::max(T{1}, std::abs(expected)));
}

}  // namespace

TEST(NeonMathProduct, TypedComplexKernelsAndTail) {
  constexpr std::size_t n = 37;
  alignas(64) float re32[n], im32[n], out32[n], out32_im[n];
  alignas(64) double re64[n], im64[n], wr64[n], wi64[n], out64[n], out64_im[n];
  for (std::size_t i = 0; i < n; ++i) {
    re32[i] = 0.1f + 0.035f * float(i);
    im32[i] = -0.6f + 0.025f * float(i);
    re64[i] = 0.25 + 0.02 * double(i);
    im64[i] = -0.8 + 0.03 * double(i);
    wr64[i] = 0.5 + 0.01 * double(i);
    wi64[i] = 0.2 + 0.005 * double(i);
  }

  hyle::simd::detail::ctan_f32_hyle_neon(re32, im32, out32, out32_im, n);
  for (std::size_t i = 0; i < n; ++i) {
    const std::complex<float> expected = std::tan(std::complex<float>(re32[i], im32[i]));
    ExpectNearRelative(out32[i], expected.real(), 3e-5f);
    ExpectNearRelative(out32_im[i], expected.imag(), 3e-5f);
  }

  hyle::simd::detail::ccosh_f64_hyle_neon(re64, im64, out64, out64_im, n);
  for (std::size_t i = 0; i < n; ++i) {
    const std::complex<double> expected = std::cosh(std::complex<double>(re64[i], im64[i]));
    ExpectNearRelative(out64[i], expected.real(), 4e-11);
    ExpectNearRelative(out64_im[i], expected.imag(), 4e-11);
  }

  hyle::simd::detail::cpow_f64_hyle_neon(re64, im64, wr64, wi64, out64, out64_im, n);
  for (std::size_t i = 0; i < n; ++i) {
    const std::complex<double> expected =
        std::pow(std::complex<double>(re64[i], im64[i]),
                 std::complex<double>(wr64[i], wi64[i]));
    ExpectNearRelative(out64[i], expected.real(), 6e-11);
    ExpectNearRelative(out64_im[i], expected.imag(), 6e-11);
  }
}

#endif
