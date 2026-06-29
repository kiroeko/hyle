#include <gtest/gtest.h>

#include <cmath>
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
  EXPECT_TRUE(v == Isa::kScalar || v == Isa::kAvx2);
}
