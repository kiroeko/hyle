#include <gtest/gtest.h>

#include "hyle/simd/math.h"
#include "hyle/simd/runtime.h"

#include <algorithm>
#include <cmath>
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
