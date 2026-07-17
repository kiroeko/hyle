#include <gtest/gtest.h>

#include "hyle/simd/vec.h"
#include "src/simd/math/kernels.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace hs = hyle::simd;

namespace {
constexpr std::size_t kMaxLanes = 256;

template <class T>
void ExpectNearRelative(T actual, T expected, T tolerance) {
  const T scale = std::max(T{1}, std::abs(expected));
  EXPECT_NEAR(actual, expected, tolerance * scale);
}
}  // namespace

TEST(SveMathProduct, LengthAgnosticF32CoreAndTail) {
  const std::size_t w = hs::Lanes(hs::ScalableTag<float>{});
  const std::size_t n = 2 * w + 3;
  ASSERT_LE(n, kMaxLanes);
  alignas(64) float src[kMaxLanes], exp_out[kMaxLanes], log_out[kMaxLanes];
  for (std::size_t i = 0; i < n; ++i) src[i] = -2.0f + 4.0f * float(i + 1) / float(n + 1);

  hs::detail::exp_f32_highway_sve(src, exp_out, n);
  hs::detail::log_f32_highway_sve(exp_out, log_out, n);
  for (std::size_t i = 0; i < n; ++i) {
    ExpectNearRelative(exp_out[i], std::exp(src[i]), 4e-6f);
    ExpectNearRelative(log_out[i], src[i], 8e-6f);
  }
}

TEST(SveMathProduct, F64TableBinaryAndSinCos) {
  const std::size_t w = hs::Lanes(hs::ScalableTag<double>{});
  const std::size_t n = 2 * w + 1;
  ASSERT_LE(n, kMaxLanes);
  alignas(64) double a[kMaxLanes], b[kMaxLanes], out[kMaxLanes];
  alignas(64) double s[kMaxLanes], c[kMaxLanes];
  for (std::size_t i = 0; i < n; ++i) {
    a[i] = -3.0 + 6.0 * double(i + 1) / double(n + 1);
    b[i] = 0.25 + 0.125 * double(i % 7);
  }

  hs::detail::exp_in_range_f64_highway_sve(a, out, n);
  for (std::size_t i = 0; i < n; ++i)
    ExpectNearRelative(out[i], std::exp(a[i]), 2e-13);

  hs::detail::hypot_f64_highway_sve(a, b, out, n);
  hs::detail::sincos_f64_highway_sve(a, s, c, n);
  for (std::size_t i = 0; i < n; ++i) {
    ExpectNearRelative(out[i], std::hypot(a[i], b[i]), 3e-13);
    ExpectNearRelative(s[i], std::sin(a[i]), 3e-13);
    ExpectNearRelative(c[i], std::cos(a[i]), 3e-13);
  }
}
