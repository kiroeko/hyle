// math_bench -- vectorized transcendentals (exp/log/tanh f32). performance.md claims SIMD
// math is "4-16x vs scalar". TWO questions this answers:
//   (1) vectorization win: HYLE's *_f32_batch (best-of-breed: borrows xsimd's exp/tanh +
//       Highway's log core, vendored unconditionally) vs a plain std::exp/log/tanh loop.
//   (2) SAME-ISA SOTA (gap-fill #36): is our borrow-core wiring competitive with the OTHER
//       vectorized-math libs at the SAME -mavx2? Gated rivals, compiled at AVX2:
//         * xsimd (-DHYLE_BENCH_XSIMD): xsimd::exp/log/tanh (its OWN polynomial cores)
//         * Highway (-DHYLE_BENCH_HIGHWAY): hn::Exp/Log/Tanh (contrib/math)
//       HYLE borrows xsimd/Highway's verified cores, so a tie with the BORROWED lib = "our
//       wiring is SOTA"; a loss vs the borrowed lib = our dispatch/build has overhead (this
//       bench caught exactly that: the product Highway log core shipped 128-bit on Linux
//       until the baseline-ISA fix -- see performance.md math section).
// active_math_variant() prints the live tier; "kScalar" => no vectorized math wired (HYLE row
// ties the scalar baseline -- honest: no vectorization on a non-AVX2 host).
#include <nanobench.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "hyle/simd/dispatch.h"
#include "hyle/simd/math.h"

#if defined(HYLE_BENCH_XSIMD)
#  include "xsimd/xsimd.hpp"
#endif
#if defined(HYLE_BENCH_HIGHWAY)
#  include "hwy/highway.h"
#  include "hwy/contrib/math/math-inl.h"
namespace hn = hwy::HWY_NAMESPACE;
#endif

namespace {
namespace nb = ankerl::nanobench;
constexpr std::size_t kN = 4096;  // fits L1; isolates compute throughput

// Per-op safe input range: exp avoids overflow, log needs positive, tanh anything.
std::vector<float> gen(std::size_t n, float lo, float hi, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto& e : v) e = dist(rng);
  return v;
}

const char* variant_name(hyle::simd::Isa i) {
  return i == hyle::simd::Isa::kAvx2 ? "AVX2 (vectorized: Highway/SLEEF per build)"
                                     : "scalar libm (no vectorized math wired)";
}

enum class Op { kExp, kLog, kTanh };

#if defined(HYLE_BENCH_XSIMD)
void xsimd_run(Op op, const float* src, float* dst, std::size_t n) {
  using batch = xsimd::batch<float>;
  constexpr std::size_t w = batch::size;
  std::size_t i = 0;
  for (; i + w <= n; i += w) {
    batch v = batch::load_unaligned(src + i);
    batch r = op == Op::kExp ? xsimd::exp(v) : op == Op::kLog ? xsimd::log(v) : xsimd::tanh(v);
    r.store_unaligned(dst + i);
  }
  for (; i < n; ++i) dst[i] = op == Op::kExp ? std::exp(src[i])
                              : op == Op::kLog ? std::log(src[i]) : std::tanh(src[i]);
}
#endif
#if defined(HYLE_BENCH_HIGHWAY)
void highway_run(Op op, const float* src, float* dst, std::size_t n) {
  const hn::ScalableTag<float> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w) {
    auto v = hn::LoadU(d, src + i);
    auto r = op == Op::kExp ? hn::Exp(d, v) : op == Op::kLog ? hn::Log(d, v) : hn::Tanh(d, v);
    hn::StoreU(r, d, dst + i);
  }
  for (; i < n; ++i) dst[i] = op == Op::kExp ? std::exp(src[i])
                              : op == Op::kLog ? std::log(src[i]) : std::tanh(src[i]);
}
#endif
}  // namespace

int main() {
  std::printf("HYLE math tier live: %s\n\n", variant_name(hyle::simd::active_math_variant()));

  std::vector<float> dst(kN);

  struct Case {
    const char* title;
    Op op;
    std::vector<float> src;
    void (*hyle_batch)(const float*, float*, std::size_t) noexcept;
    float (*scalar_fn)(float);
  };
  Case cases[] = {
      {"exp f32 (4096)", Op::kExp, gen(kN, -10.0f, 10.0f, 1), &hyle::simd::exp_f32_batch,
       [](float x) { return std::exp(x); }},
      {"log f32 (4096)", Op::kLog, gen(kN, 0.01f, 100.0f, 2), &hyle::simd::log_f32_batch,
       [](float x) { return std::log(x); }},
      {"tanh f32 (4096)", Op::kTanh, gen(kN, -5.0f, 5.0f, 3), &hyle::simd::tanh_f32_batch,
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
    // HYLE batch op: the wired vectorized tier (Highway-math if -DHYLE_SIMD_HIGHWAY_MATH,
    // else SLEEF if -DHYLE_SIMD_SLEEF, else scalar). Dispatcher prefers Highway > SLEEF.
    b.run("HYLE *_f32_batch (vectorized)", [&] {
      c.hyle_batch(c.src.data(), dst.data(), kN);
      nb::doNotOptimizeAway(dst.data());
    });
#if defined(HYLE_BENCH_XSIMD)
    b.run("xsimd (same ISA SOTA)", [&] {
      xsimd_run(c.op, c.src.data(), dst.data(), kN);
      nb::doNotOptimizeAway(dst.data());
    });
#endif
#if defined(HYLE_BENCH_HIGHWAY)
    b.run("Highway (same ISA SOTA)", [&] {
      highway_run(c.op, c.src.data(), dst.data(), kN);
      nb::doNotOptimizeAway(dst.data());
    });
#endif
  }
  return 0;
}
