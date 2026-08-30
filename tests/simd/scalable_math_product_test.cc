#include <gtest/gtest.h>

#include "src/simd/math/kernels.h"
#include "hwy/highway.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>

namespace hs = hyle::simd;
namespace hn = hwy::HWY_NAMESPACE;

#if defined(HYLE_SIMD_TARGET_SVE)
#define HYLE_SCALABLE_PRODUCT(name) hs::detail::name##_sve
#elif defined(HYLE_SIMD_TARGET_RVV)
#define HYLE_SCALABLE_PRODUCT(name) hs::detail::name##_rvv
#endif

namespace {
constexpr std::size_t kMaxLanes = 256;

template <class T>
void ExpectNearRelative(T actual, T expected, T tolerance) {
  const T scale = std::max(T{1}, std::abs(expected));
  EXPECT_NEAR(actual, expected, tolerance * scale);
}
}  // namespace

TEST(ScalableMathProduct, LengthAgnosticF32CoreAndTail) {
  const std::size_t w = hn::Lanes(hn::ScalableTag<float>{});
  const std::size_t n = 2 * w + 3;
  ASSERT_LE(n, kMaxLanes);
  alignas(64) float src[kMaxLanes], exp_out[kMaxLanes], log_out[kMaxLanes];
  for (std::size_t i = 0; i < n; ++i) src[i] = -2.0f + 4.0f * float(i + 1) / float(n + 1);

  HYLE_SCALABLE_PRODUCT(exp_f32_highway)(src, exp_out, n);
  HYLE_SCALABLE_PRODUCT(log_f32_highway)(exp_out, log_out, n);
  for (std::size_t i = 0; i < n; ++i) {
    ExpectNearRelative(exp_out[i], std::exp(src[i]), 4e-6f);
    ExpectNearRelative(log_out[i], src[i], 8e-6f);
  }
}

TEST(ScalableMathProduct, F64TableBinaryAndSinCos) {
  const std::size_t w = hn::Lanes(hn::ScalableTag<double>{});
  const std::size_t n = 2 * w + 1;
  ASSERT_LE(n, kMaxLanes);
  alignas(64) double a[kMaxLanes], b[kMaxLanes], out[kMaxLanes];
  alignas(64) double s[kMaxLanes], c[kMaxLanes];
  for (std::size_t i = 0; i < n; ++i) {
    a[i] = -3.0 + 6.0 * double(i + 1) / double(n + 1);
    b[i] = 0.25 + 0.125 * double(i % 7);
  }

  HYLE_SCALABLE_PRODUCT(exp_in_range_f64_highway)(a, out, n);
  for (std::size_t i = 0; i < n; ++i)
    ExpectNearRelative(out[i], std::exp(a[i]), 2e-13);

  HYLE_SCALABLE_PRODUCT(hypot_f64_highway)(a, b, out, n);
  HYLE_SCALABLE_PRODUCT(sincos_f64_highway)(a, s, c, n);
  for (std::size_t i = 0; i < n; ++i) {
    ExpectNearRelative(out[i], std::hypot(a[i], b[i]), 3e-13);
    ExpectNearRelative(s[i], std::sin(a[i]), 3e-13);
    ExpectNearRelative(c[i], std::cos(a[i]), 3e-13);
  }
}

TEST(ScalableMathProduct, HyleDerivedRealKernelsAndTail) {
  const std::size_t w32 = hn::Lanes(hn::ScalableTag<float>{});
  const std::size_t n32 = 2 * w32 + 3;
  ASSERT_LE(n32, kMaxLanes);
  alignas(64) float x32[kMaxLanes], y32[kMaxLanes], out32[kMaxLanes];
  for (std::size_t i = 0; i < n32; ++i) {
    x32[i] = 0.1f + 0.075f * float(i % 53);
    y32[i] = 0.3f + 0.125f * float(i % 11);
  }

  HYLE_SCALABLE_PRODUCT(erfc_f32_hyle)(x32, out32, n32);
  for (std::size_t i = 0; i < n32; ++i)
    ExpectNearRelative(out32[i], std::erfc(x32[i]), 5e-6f);
  HYLE_SCALABLE_PRODUCT(exp10_f32_hyle)(x32, out32, n32);
  for (std::size_t i = 0; i < n32; ++i)
    ExpectNearRelative(out32[i], std::pow(10.0f, x32[i]), 6e-6f);
  for (std::size_t i = 0; i < n32; ++i)
    x32[i] = 0.05f + 9.95f * float(i + 1) / float(n32 + 1);
  HYLE_SCALABLE_PRODUCT(tgamma_f32_highway)(x32, out32, n32);
  for (std::size_t i = 0; i < n32; ++i)
    ExpectNearRelative(out32[i], std::tgamma(x32[i]), 5e-6f);
  HYLE_SCALABLE_PRODUCT(fdim_f32_hyle)(x32, y32, out32, n32);
  for (std::size_t i = 0; i < n32; ++i)
    ExpectNearRelative(out32[i], std::fdim(x32[i], y32[i]), 2e-6f);
  HYLE_SCALABLE_PRODUCT(hypot_in_range_f32_hyle)(x32, y32, out32, n32);
  for (std::size_t i = 0; i < n32; ++i)
    ExpectNearRelative(out32[i], std::sqrt(std::fma(x32[i], x32[i], y32[i] * y32[i])),
                       2e-6f);

  const std::size_t w64 = hn::Lanes(hn::ScalableTag<double>{});
  const std::size_t n64 = 2 * w64 + 3;
  ASSERT_LE(n64, kMaxLanes);
  alignas(64) double x64[kMaxLanes], y64[kMaxLanes], out64[kMaxLanes];
  for (std::size_t i = 0; i < n64; ++i) {
    x64[i] = 0.1 + 0.075 * double(i % 53);
    y64[i] = 0.3 + 0.125 * double(i % 11);
  }
  HYLE_SCALABLE_PRODUCT(erfc_f64_hyle)(x64, out64, n64);
  for (std::size_t i = 0; i < n64; ++i)
    ExpectNearRelative(out64[i], std::erfc(x64[i]), 2e-12);
  HYLE_SCALABLE_PRODUCT(exp10_f64_hyle)(x64, out64, n64);
  for (std::size_t i = 0; i < n64; ++i)
    ExpectNearRelative(out64[i], std::pow(10.0, x64[i]), 2e-12);
  for (std::size_t i = 0; i < n64; ++i)
    x64[i] = 0.5 + 14.5 * double(i + 1) / double(n64 + 1);
  HYLE_SCALABLE_PRODUCT(tgamma_f64_highway)(x64, out64, n64);
  for (std::size_t i = 0; i < n64; ++i)
    ExpectNearRelative(out64[i], std::tgamma(x64[i]), 2e-12);
  HYLE_SCALABLE_PRODUCT(fdim_f64_hyle)(x64, y64, out64, n64);
  for (std::size_t i = 0; i < n64; ++i)
    ExpectNearRelative(out64[i], std::fdim(x64[i], y64[i]), 2e-12);
}

TEST(ScalableMathProduct, ActivationsAndRowReductionsCoverWideTail) {
  const std::size_t width = hn::Lanes(hn::ScalableTag<float>{});
  const std::size_t n = 2 * width + 3;
  ASSERT_LE(n, kMaxLanes);
  alignas(64) float input[kMaxLanes], output[kMaxLanes], alias[kMaxLanes];
  for (std::size_t i = 0; i < n; ++i) {
    input[i] = -8.0f + 16.0f * float(i + 1) / float(n + 1);
  }

  const auto check_unary = [&](auto function, auto reference,
                               float tolerance) {
    std::copy_n(input, n, alias);
    function(input, output, n);
    function(alias, alias, n);
    for (std::size_t i = 0; i < n; ++i) {
      ExpectNearRelative(output[i], reference(input[i]), tolerance);
      EXPECT_EQ(alias[i], output[i]);
    }
  };
  check_unary(HYLE_SCALABLE_PRODUCT(gelu_f32_hyle),
              [](float x) {
                return 0.5f * x *
                       (1.0f + std::erf(x * 0.70710678118654752440f));
              },
              8e-6f);
  check_unary(HYLE_SCALABLE_PRODUCT(gelu_tanh_f32_route),
              [](float x) {
                return 0.5f * x *
                       (1.0f +
                        std::tanh(0.79788456080286535588f * x *
                                  (1.0f + 0.044715f * x * x)));
              },
              8e-6f);
  check_unary(HYLE_SCALABLE_PRODUCT(sigmoid_f32_hyle),
              [](float x) {
                const float e = std::exp(-std::abs(x));
                return x < 0.0f ? e / (1.0f + e) : 1.0f / (1.0f + e);
              },
              8e-6f);
  check_unary(HYLE_SCALABLE_PRODUCT(silu_f32_hyle),
              [](float x) {
                const float e = std::exp(-std::abs(x));
                const float s =
                    x < 0.0f ? e / (1.0f + e) : 1.0f / (1.0f + e);
                return x * s;
              },
              8e-6f);
  check_unary(HYLE_SCALABLE_PRODUCT(relu_f32_hyle),
              [](float x) { return x > 0.0f ? x : 0.0f; }, 0.0f);
  check_unary(HYLE_SCALABLE_PRODUCT(hard_sigmoid_f32_hyle),
              [](float x) {
                return std::min(1.0f,
                                std::max(0.0f, (x + 3.0f) / 6.0f));
              },
              2e-6f);
  check_unary(HYLE_SCALABLE_PRODUCT(hard_swish_f32_hyle),
              [](float x) {
                if (x <= -3.0f) return 0.0f;
                if (x >= 3.0f) return x;
                return x * ((x + 3.0f) / 6.0f);
              },
              2e-6f);

  HYLE_SCALABLE_PRODUCT(leaky_relu_f32_hyle)(input, 0.125f, output, n);
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(output[i], input[i] >= 0.0f ? input[i] : 0.125f * input[i]);
  }
  HYLE_SCALABLE_PRODUCT(elu_f32_hyle)(input, 1.25f, output, n);
  for (std::size_t i = 0; i < n; ++i) {
    const float expected =
        input[i] >= 0.0f ? input[i] : 1.25f * std::expm1(input[i]);
    ExpectNearRelative(output[i], expected, 8e-6f);
  }

  HYLE_SCALABLE_PRODUCT(softmax_f32_hyle)(input, output, n);
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) sum += output[i];
  EXPECT_NEAR(sum, 1.0, 2e-5);
  HYLE_SCALABLE_PRODUCT(log_softmax_f32_hyle)(input, alias, n);
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(alias[i], std::log(output[i]), 2e-5f);
  }
}

TEST(ScalableMathProduct, HyleDerivedComplexKernelsCoverWideExceptionalLanes) {
  const std::size_t w = hn::Lanes(hn::ScalableTag<float>{});
  const std::size_t n = 2 * w + 3;
  ASSERT_LE(n, kMaxLanes);
  alignas(64) float zr[kMaxLanes], zi[kMaxLanes], wr[kMaxLanes], wi[kMaxLanes];
  alignas(64) float dr[kMaxLanes], di[kMaxLanes];
  for (std::size_t i = 0; i < n; ++i) {
    zr[i] = (i % 7 == 0) ? 0.0f : 0.25f + 0.02f * float(i % 31);
    zi[i] = -0.8f + 0.04f * float(i % 37);
    wr[i] = (i % 13 == 0) ? 0.0f : 0.5f + 0.01f * float(i % 29);
    wi[i] = 0.2f + 0.005f * float(i % 17);
  }

  const auto check_unary = [&](auto fn, auto oracle) {
    fn(zr, zi, dr, di, n);
    for (std::size_t i = 0; i < n; ++i) {
      const std::complex<float> expected = oracle(std::complex<float>(zr[i], zi[i]));
      ExpectNearRelative(dr[i], expected.real(), 2e-5f);
      ExpectNearRelative(di[i], expected.imag(), 2e-5f);
    }
  };
  check_unary(HYLE_SCALABLE_PRODUCT(cexp_f32_hyle),
              [](std::complex<float> z) { return std::exp(z); });
  check_unary(HYLE_SCALABLE_PRODUCT(clog_f32_hyle),
              [](std::complex<float> z) { return std::log(z); });
  check_unary(HYLE_SCALABLE_PRODUCT(csqrt_f32_hyle),
              [](std::complex<float> z) { return std::sqrt(z); });
  check_unary(HYLE_SCALABLE_PRODUCT(csin_f32_hyle),
              [](std::complex<float> z) { return std::sin(z); });
  check_unary(HYLE_SCALABLE_PRODUCT(ccos_f32_hyle),
              [](std::complex<float> z) { return std::cos(z); });
  check_unary(HYLE_SCALABLE_PRODUCT(ctan_f32_hyle),
              [](std::complex<float> z) { return std::tan(z); });
  check_unary(HYLE_SCALABLE_PRODUCT(csinh_f32_hyle),
              [](std::complex<float> z) { return std::sinh(z); });
  check_unary(HYLE_SCALABLE_PRODUCT(ccosh_f32_hyle),
              [](std::complex<float> z) { return std::cosh(z); });
  check_unary(HYLE_SCALABLE_PRODUCT(ctanh_f32_hyle),
              [](std::complex<float> z) { return std::tanh(z); });

  HYLE_SCALABLE_PRODUCT(cpow_f32_hyle)(zr, zi, wr, wi, dr, di, n);
  for (std::size_t i = 0; i < n; ++i) {
    const std::complex<float> expected =
        std::pow(std::complex<float>(zr[i], zi[i]), std::complex<float>(wr[i], wi[i]));
    ExpectNearRelative(dr[i], expected.real(), 3e-5f);
    ExpectNearRelative(di[i], expected.imag(), 3e-5f);
  }

  const std::size_t w64 = hn::Lanes(hn::ScalableTag<double>{});
  const std::size_t n64 = 2 * w64 + 3;
  ASSERT_LE(n64, kMaxLanes);
  alignas(64) double zr64[kMaxLanes], zi64[kMaxLanes], wr64[kMaxLanes], wi64[kMaxLanes];
  alignas(64) double dr64[kMaxLanes], di64[kMaxLanes];
  for (std::size_t i = 0; i < n64; ++i) {
    zr64[i] = 0.25 + 0.02 * double(i % 31);
    zi64[i] = -0.8 + 0.04 * double(i % 37);
    wr64[i] = 0.5 + 0.01 * double(i % 29);
    wi64[i] = 0.2 + 0.005 * double(i % 17);
  }
  const auto check_unary64 = [&](auto fn, auto oracle) {
    fn(zr64, zi64, dr64, di64, n64);
    for (std::size_t i = 0; i < n64; ++i) {
      const std::complex<double> expected = oracle(std::complex<double>(zr64[i], zi64[i]));
      ExpectNearRelative(dr64[i], expected.real(), 4e-11);
      ExpectNearRelative(di64[i], expected.imag(), 4e-11);
    }
  };
  check_unary64(HYLE_SCALABLE_PRODUCT(cexp_f64_hyle),
                [](std::complex<double> z) { return std::exp(z); });
  check_unary64(HYLE_SCALABLE_PRODUCT(clog_f64_hyle),
                [](std::complex<double> z) { return std::log(z); });
  check_unary64(HYLE_SCALABLE_PRODUCT(csqrt_f64_hyle),
                [](std::complex<double> z) { return std::sqrt(z); });
  check_unary64(HYLE_SCALABLE_PRODUCT(csin_f64_hyle),
                [](std::complex<double> z) { return std::sin(z); });
  check_unary64(HYLE_SCALABLE_PRODUCT(ccos_f64_hyle),
                [](std::complex<double> z) { return std::cos(z); });
  check_unary64(HYLE_SCALABLE_PRODUCT(ctan_f64_hyle),
                [](std::complex<double> z) { return std::tan(z); });
  check_unary64(HYLE_SCALABLE_PRODUCT(csinh_f64_hyle),
                [](std::complex<double> z) { return std::sinh(z); });
  check_unary64(HYLE_SCALABLE_PRODUCT(ccosh_f64_hyle),
                [](std::complex<double> z) { return std::cosh(z); });
  check_unary64(HYLE_SCALABLE_PRODUCT(ctanh_f64_hyle),
                [](std::complex<double> z) { return std::tanh(z); });

  HYLE_SCALABLE_PRODUCT(cpow_f64_hyle)(zr64, zi64, wr64, wi64, dr64, di64, n64);
  for (std::size_t i = 0; i < n64; ++i) {
    const std::complex<double> expected =
        std::pow(std::complex<double>(zr64[i], zi64[i]),
                 std::complex<double>(wr64[i], wi64[i]));
    ExpectNearRelative(dr64[i], expected.real(), 6e-11);
    ExpectNearRelative(di64[i], expected.imag(), 6e-11);
  }
}

#undef HYLE_SCALABLE_PRODUCT
