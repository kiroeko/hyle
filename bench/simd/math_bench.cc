// math_bench -- vectorized transcendentals (exp/log/tanh f32) vs scalar libm. The AI hot
// path (softmax/gelu/attention) is dominated by these; performance.md claims SIMD math is
// "4-16x vs scalar / transcendental 是大头". This bench is the real number: HYLE's
// exp/log/tanh_f32_batch (SLEEF AVX2 8-wide when built with -DHYLE_SIMD_SLEEF=ON, else
// scalar libm) vs a plain std::exp/log/tanh loop. SLEEF IS the transcendental SOTA (HYLE
// borrows its verified cores), so the comparison measures the vectorization win itself.
// active_math_variant() prints which tier is live -- a "kScalar" line means SLEEF is NOT
// wired in and the HYLE row will TIE the baseline (honest: no vectorization without SLEEF).
#include <nanobench.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "hyle/simd/dispatch.h"
#include "hyle/simd/math.h"

namespace {
constexpr std::size_t kN = 4096;  // fits L1; isolates compute throughput
namespace nb = ankerl::nanobench;

// Per-op safe input range: exp avoids overflow, log needs positive, tanh anything.
std::vector<float> gen(std::size_t n, float lo, float hi, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto& e : v) e = dist(rng);
  return v;
}

const char* variant_name(hyle::simd::Isa i) {
  return i == hyle::simd::Isa::kAvx2 ? "AVX2 (SLEEF)" : "scalar libm (no SLEEF wired)";
}
}  // namespace

int main() {
  std::printf("HYLE math tier live: %s\n\n", variant_name(hyle::simd::active_math_variant()));

  std::vector<float> dst(kN);

  struct Case {
    const char* title;
    std::vector<float> src;
    void (*hyle_batch)(const float*, float*, std::size_t) noexcept;
    float (*scalar_fn)(float);
  };
  Case cases[] = {
      {"exp f32 (4096)", gen(kN, -10.0f, 10.0f, 1), &hyle::simd::exp_f32_batch,
       [](float x) { return std::exp(x); }},
      {"log f32 (4096)", gen(kN, 0.01f, 100.0f, 2), &hyle::simd::log_f32_batch,
       [](float x) { return std::log(x); }},
      {"tanh f32 (4096)", gen(kN, -5.0f, 5.0f, 3), &hyle::simd::tanh_f32_batch,
       [](float x) { return std::tanh(x); }},
  };

  for (auto& c : cases) {
    nb::Bench b;
    b.title(c.title).relative(true).batch(kN).minEpochIterations(500);
    // Baseline: scalar libm element loop (what the default HYLE build also does).
    b.run("scalar libm loop", [&] {
      for (std::size_t i = 0; i < kN; ++i) dst[i] = c.scalar_fn(c.src[i]);
      nb::doNotOptimizeAway(dst.data());
    });
    // HYLE batch op: SLEEF AVX2 when built with it, else its own scalar tier.
    b.run("HYLE *_f32_batch", [&] {
      c.hyle_batch(c.src.data(), dst.data(), kN);
      nb::doNotOptimizeAway(dst.data());
    });
  }
  return 0;
}
