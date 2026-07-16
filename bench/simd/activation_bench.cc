// activation_bench -- AI activation + softmax + RNG throughput (directive3 P1 blind-spot fill).
// These are inference hot paths (GELU/SiLU in transformer FFN, softmax in attention) that had ZERO
// bench coverage. Answers: (1) does the vectorized activation beat a plain scalar libm loop, and
// (2) does softmax's max-stabilize + exp + normalize scale. RNG row = VectorXoshiro fill throughput.
//
// Not a vs-Highway/xsimd parity bench: these are HYLE composite kernels (they call the best-of math
// batch ops internally), so the meaningful baseline is the scalar reference a naive user would write.
#include <nanobench.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "hyle/simd/math.h"
#include "hyle/simd/native/random.h"
#include "hyle/simd/random.h"

namespace {
namespace nb = ankerl::nanobench;
constexpr std::size_t kN = 4096;  // L1-resident

std::vector<float> gen(std::size_t n, float lo, float hi, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto& e : v) e = dist(rng);
  return v;
}

// Scalar references (what a user writes without HYLE). Exact-erf GELU + tanh-approx GELU + SiLU.
float gelu_scalar(float x) { return 0.5f * x * (1.0f + std::erf(x * 0.70710678f)); }
float gelu_tanh_scalar(float x) {
  const float k = 0.7978845608f;  // sqrt(2/pi)
  return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
}
float sigmoid_scalar(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float silu_scalar(float x) { return x / (1.0f + std::exp(-x)); }
}  // namespace

int main() {
  std::vector<float> src = gen(kN, -6.0f, 6.0f, 1), dst(kN);

  struct Case {
    const char* title;
    void (*hyle)(const float*, float*, std::size_t) noexcept;
    float (*scalar)(float);
  };
  const Case cases[] = {
      {"gelu f32 (4096)", &hyle::simd::gelu_f32_batch, &gelu_scalar},
      {"gelu_tanh f32 (4096)", &hyle::simd::gelu_tanh_f32_batch, &gelu_tanh_scalar},
      {"sigmoid f32 (4096)", &hyle::simd::sigmoid_f32_batch, &sigmoid_scalar},
      {"silu f32 (4096)", &hyle::simd::silu_f32_batch, &silu_scalar},
  };
  for (const auto& c : cases) {
    nb::Bench b;
    b.title(c.title).relative(true).batch(kN).minEpochIterations(1000);
    b.run("scalar libm loop", [&] {
      for (std::size_t i = 0; i < kN; ++i) dst[i] = c.scalar(src[i]);
      nb::doNotOptimizeAway(dst.data());
    });
    b.run("HYLE *_f32_batch", [&] {
      c.hyle(src.data(), dst.data(), kN);
      nb::doNotOptimizeAway(dst.data());
    });
  }

  // softmax / log_softmax: in-place, reduction-heavy (max-stabilize + exp + sum + normalize). The
  // scalar reference does the same 3 passes by hand; HYLE vectorizes each. Attention hot path.
  {
    nb::Bench b;
    b.title("softmax f32 (4096, in-place)").relative(true).batch(kN).minEpochIterations(1000);
    b.run("scalar 3-pass", [&] {
      for (std::size_t i = 0; i < kN; ++i) dst[i] = src[i];
      float m = dst[0];
      for (std::size_t i = 1; i < kN; ++i) m = dst[i] > m ? dst[i] : m;
      float s = 0.0f;
      for (std::size_t i = 0; i < kN; ++i) { dst[i] = std::exp(dst[i] - m); s += dst[i]; }
      const float inv = 1.0f / s;
      for (std::size_t i = 0; i < kN; ++i) dst[i] *= inv;
      nb::doNotOptimizeAway(dst.data());
    });
    b.run("HYLE softmax_f32", [&] {
      for (std::size_t i = 0; i < kN; ++i) dst[i] = src[i];
      hyle::simd::softmax_f32(dst.data(), kN);
      nb::doNotOptimizeAway(dst.data());
    });
  }
  {
    nb::Bench b;
    b.title("log_softmax f32 (4096, in-place)").relative(true).batch(kN).minEpochIterations(1000);
    b.run("HYLE log_softmax_f32", [&] {
      for (std::size_t i = 0; i < kN; ++i) dst[i] = src[i];
      hyle::simd::log_softmax_f32(dst.data(), kN);
      nb::doNotOptimizeAway(dst.data());
    });
  }

  // RNG: VectorXoshiro fill throughput vs std::mt19937_64. SIMD xoshiro256 is the fast path for
  // dropout masks / sampling; the value is raw bytes/sec, so bench a bulk fill. Next() returns a
  // WHOLE VECTOR of N independent-stream u64, stored per SIMD width.
  {
    alignas(64) std::vector<std::uint64_t> out(kN);
    nb::Bench b;
    b.title("RNG u64 fill (4096)").relative(true).batch(kN).minEpochIterations(1000);
    b.run("std::mt19937_64", [&] {
      std::mt19937_64 rng(12345);
      for (std::size_t i = 0; i < kN; ++i) out[i] = rng();
      nb::doNotOptimizeAway(out.data());
    });
    b.run("HYLE runtime random_fill", [&] {
      hyle::simd::RandomState state(12345);
      hyle::simd::random_fill(state, out.data(), out.size());
      nb::doNotOptimizeAway(out.data());
    });
    b.run("HYLE native random_fill", [&] {
      hyle::simd::RandomState state(12345);
      hyle::simd::native::random_fill(state, out.data(), out.size());
      nb::doNotOptimizeAway(out.data());
    });
  }
  return 0;
}
