#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "hyle/simd/dispatch.h"
#include "hyle/simd/math.h"

// math_test -- exp/log/tanh batch ops vs libm reference. The batch path runs whichever
// tier is live (SLEEF AVX2 if -DHYLE_SIMD_SLEEF + AVX2 host, else scalar libm). Either
// way the result must match std:: within tolerance: scalar is exact-libm; SLEEF _u10 is
// 1-ULP, so a small relative tolerance covers both. This is runnable on EVERY platform
// (scalar tier always present), and additionally exercises the SLEEF cores when wired.

using hyle::simd::active_math_variant;
using hyle::simd::exp_f32_batch;
using hyle::simd::Isa;
using hyle::simd::log_f32_batch;
using hyle::simd::tanh_f32_batch;

namespace {
// relative error vs a double-precision reference; tolerant of 1-ULP SLEEF + f32 rounding.
void check(void (*fn)(const float*, float*, std::size_t), double (*ref)(double),
           const std::vector<float>& in, const char* name) {
  std::vector<float> out(in.size());
  fn(in.data(), out.data(), in.size());
  for (std::size_t i = 0; i < in.size(); ++i) {
    const double want = ref(static_cast<double>(in[i]));
    const double got = static_cast<double>(out[i]);
    const double tol = std::abs(want) * 1e-5 + 1e-6;
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

// tail handling: n not a multiple of 8 (the SLEEF kernel's vector width).
TEST(Math, TailSizes) {
  for (std::size_t n : {std::size_t{1}, std::size_t{7}, std::size_t{8}, std::size_t{9}, std::size_t{17}}) {
    std::vector<float> in(n);
    for (std::size_t i = 0; i < n; ++i) in[i] = 0.1f + 0.3f * static_cast<float>(i);
    std::vector<float> out(n);
    exp_f32_batch(in.data(), out.data(), n);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_NEAR(out[i], std::exp(in[i]), std::exp(in[i]) * 1e-5f + 1e-6f) << "n=" << n;
  }
}

// active tier sanity. NOTE: HYLE_SIMD_SLEEF is a library-PRIVATE define (set on hyle_simd,
// not on this test TU), so the test cannot know whether SLEEF was wired in -- it only
// observes the runtime result. Either tier is valid: scalar libm floor, or kAvx2 when the
// lib was built with SLEEF AND the host has AVX2. Correctness is proven by the
// Exp/Log/Tanh-vs-libm tests above regardless of which tier ran.
TEST(Math, ActiveVariantReported) {
  const Isa v = active_math_variant();
  EXPECT_TRUE(v == Isa::kScalar || v == Isa::kAvx2);
}
