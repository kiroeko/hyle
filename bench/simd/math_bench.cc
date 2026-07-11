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

// simd-09: candidate functions for the best-of-breed expansion. exp/log/tanh already wired in
// HYLE (simd-06); the rest are UNWIRED candidates -- this bench decides who (xsimd vs Highway)
// wins each at the same ISA, so math_dispatch can borrow the fastest core per op.
enum class Op {
  kExp, kLog, kTanh,
  kSin, kCos, kExp2, kLog2, kErf, kAtan, kCbrt, kExpm1, kLog1p, kAsinh, kAcosh
};

float scalar_of(Op op, float x) {
  switch (op) {
    case Op::kExp: return std::exp(x);   case Op::kLog: return std::log(x);
    case Op::kTanh: return std::tanh(x); case Op::kSin: return std::sin(x);
    case Op::kCos: return std::cos(x);   case Op::kExp2: return std::exp2(x);
    case Op::kLog2: return std::log2(x); case Op::kErf: return std::erf(x);
    case Op::kAtan: return std::atan(x); case Op::kCbrt: return std::cbrt(x);
    case Op::kExpm1: return std::expm1(x); case Op::kLog1p: return std::log1p(x);
    case Op::kAsinh: return std::asinh(x); case Op::kAcosh: return std::acosh(x);
  }
  return x;
}
// f64 doubles of the same ops -- the f64 best-of winners diverge from f32 (memory: exp/tanh/exp2
// flip xsimd->Highway), so the f64 dispatch table must be re-benched, not extrapolated from f32.
double scalar_of64(Op op, double x) {
  switch (op) {
    case Op::kExp: return std::exp(x);   case Op::kLog: return std::log(x);
    case Op::kTanh: return std::tanh(x); case Op::kSin: return std::sin(x);
    case Op::kCos: return std::cos(x);   case Op::kExp2: return std::exp2(x);
    case Op::kLog2: return std::log2(x); case Op::kErf: return std::erf(x);
    case Op::kAtan: return std::atan(x); case Op::kCbrt: return std::cbrt(x);
    case Op::kExpm1: return std::expm1(x); case Op::kLog1p: return std::log1p(x);
    case Op::kAsinh: return std::asinh(x); case Op::kAcosh: return std::acosh(x);
  }
  return x;
}
std::vector<double> gen64(std::size_t n, double lo, double hi, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(lo, hi);
  std::vector<double> v(n);
  for (auto& e : v) e = dist(rng);
  return v;
}

#if defined(HYLE_BENCH_XSIMD)
void xsimd_run(Op op, const float* src, float* dst, std::size_t n) {
  using batch = xsimd::batch<float>;
  constexpr std::size_t w = batch::size;
  std::size_t i = 0;
  for (; i + w <= n; i += w) {
    batch v = batch::load_unaligned(src + i);
    batch r;
    switch (op) {
      case Op::kExp: r = xsimd::exp(v); break;   case Op::kLog: r = xsimd::log(v); break;
      case Op::kTanh: r = xsimd::tanh(v); break; case Op::kSin: r = xsimd::sin(v); break;
      case Op::kCos: r = xsimd::cos(v); break;   case Op::kExp2: r = xsimd::exp2(v); break;
      case Op::kLog2: r = xsimd::log2(v); break; case Op::kErf: r = xsimd::erf(v); break;
      case Op::kAtan: r = xsimd::atan(v); break; case Op::kCbrt: r = xsimd::cbrt(v); break;
      case Op::kExpm1: r = xsimd::expm1(v); break; case Op::kLog1p: r = xsimd::log1p(v); break;
      case Op::kAsinh: r = xsimd::asinh(v); break; case Op::kAcosh: r = xsimd::acosh(v); break;
    }
    r.store_unaligned(dst + i);
  }
  for (; i < n; ++i) dst[i] = scalar_of(op, src[i]);
}
#endif
#if defined(HYLE_BENCH_HIGHWAY)
void highway_run(Op op, const float* src, float* dst, std::size_t n) {
  const hn::ScalableTag<float> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w) {
    auto v = hn::LoadU(d, src + i);
    decltype(v) r;
    switch (op) {
      case Op::kExp: r = hn::Exp(d, v); break;   case Op::kLog: r = hn::Log(d, v); break;
      case Op::kTanh: r = hn::Tanh(d, v); break; case Op::kSin: r = hn::Sin(d, v); break;
      case Op::kCos: r = hn::Cos(d, v); break;   case Op::kExp2: r = hn::Exp2(d, v); break;
      case Op::kLog2: r = hn::Log2(d, v); break; case Op::kErf: r = hn::Erf(d, v); break;
      case Op::kAtan: r = hn::Atan(d, v); break; case Op::kCbrt: r = hn::Cbrt(d, v); break;
      case Op::kExpm1: r = hn::Expm1(d, v); break; case Op::kLog1p: r = hn::Log1p(d, v); break;
      case Op::kAsinh: r = hn::Asinh(d, v); break; case Op::kAcosh: r = hn::Acosh(d, v); break;
    }
    hn::StoreU(r, d, dst + i);
  }
  for (; i < n; ++i) dst[i] = scalar_of(op, src[i]);
}
#endif
#if defined(HYLE_BENCH_XSIMD)
void xsimd_run64(Op op, const double* src, double* dst, std::size_t n) {
  using batch = xsimd::batch<double>;
  constexpr std::size_t w = batch::size;
  std::size_t i = 0;
  for (; i + w <= n; i += w) {
    batch v = batch::load_unaligned(src + i);
    batch r;
    switch (op) {
      case Op::kExp: r = xsimd::exp(v); break;   case Op::kLog: r = xsimd::log(v); break;
      case Op::kTanh: r = xsimd::tanh(v); break; case Op::kSin: r = xsimd::sin(v); break;
      case Op::kCos: r = xsimd::cos(v); break;   case Op::kExp2: r = xsimd::exp2(v); break;
      case Op::kLog2: r = xsimd::log2(v); break; case Op::kErf: r = xsimd::erf(v); break;
      case Op::kAtan: r = xsimd::atan(v); break; case Op::kCbrt: r = xsimd::cbrt(v); break;
      case Op::kExpm1: r = xsimd::expm1(v); break; case Op::kLog1p: r = xsimd::log1p(v); break;
      case Op::kAsinh: r = xsimd::asinh(v); break; case Op::kAcosh: r = xsimd::acosh(v); break;
    }
    r.store_unaligned(dst + i);
  }
  for (; i < n; ++i) dst[i] = scalar_of64(op, src[i]);
}
#endif
#if defined(HYLE_BENCH_HIGHWAY)
void highway_run64(Op op, const double* src, double* dst, std::size_t n) {
  const hn::ScalableTag<double> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w) {
    auto v = hn::LoadU(d, src + i);
    decltype(v) r;
    switch (op) {
      case Op::kExp: r = hn::Exp(d, v); break;   case Op::kLog: r = hn::Log(d, v); break;
      case Op::kTanh: r = hn::Tanh(d, v); break; case Op::kSin: r = hn::Sin(d, v); break;
      case Op::kCos: r = hn::Cos(d, v); break;   case Op::kExp2: r = hn::Exp2(d, v); break;
      case Op::kLog2: r = hn::Log2(d, v); break; case Op::kErf: r = hn::Erf(d, v); break;
      case Op::kAtan: r = hn::Atan(d, v); break; case Op::kCbrt: r = hn::Cbrt(d, v); break;
      case Op::kExpm1: r = hn::Expm1(d, v); break; case Op::kLog1p: r = hn::Log1p(d, v); break;
      case Op::kAsinh: r = hn::Asinh(d, v); break; case Op::kAcosh: r = hn::Acosh(d, v); break;
    }
    hn::StoreU(r, d, dst + i);
  }
  for (; i < n; ++i) dst[i] = scalar_of64(op, src[i]);
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
    void (*hyle_batch)(const float*, float*, std::size_t) noexcept;  // nullptr = unwired candidate
  };
  Case cases[] = {
      // Already WIRED in HYLE (simd-06): hyle_batch present -> HYLE row runs.
      {"exp f32 (4096)", Op::kExp, gen(kN, -10.0f, 10.0f, 1), &hyle::simd::exp_f32_batch},
      {"log f32 (4096)", Op::kLog, gen(kN, 0.01f, 100.0f, 2), &hyle::simd::log_f32_batch},
      {"tanh f32 (4096)", Op::kTanh, gen(kN, -5.0f, 5.0f, 3), &hyle::simd::tanh_f32_batch},
      // simd-09 CANDIDATES (unwired: hyle_batch=nullptr -> only scalar/xsimd/Highway rows). This
      // bench decides the per-op winner so math_dispatch borrows the fastest core. Input ranges
      // chosen safe/representative per function.
      // simd-09 candidates are now WIRED (dispatch chose a winner) -> HYLE row runs, regressing the
      // choice against the same-ISA rivals (was nullptr "unwired candidate" before simd-09 landed).
      {"sin f32 (4096)", Op::kSin, gen(kN, -3.14f, 3.14f, 4), &hyle::simd::sin_f32_batch},
      {"cos f32 (4096)", Op::kCos, gen(kN, -3.14f, 3.14f, 5), &hyle::simd::cos_f32_batch},
      {"exp2 f32 (4096)", Op::kExp2, gen(kN, -10.0f, 10.0f, 6), &hyle::simd::exp2_f32_batch},
      {"log2 f32 (4096)", Op::kLog2, gen(kN, 0.01f, 100.0f, 7), &hyle::simd::log2_f32_batch},
      {"erf f32 (4096)", Op::kErf, gen(kN, -3.0f, 3.0f, 8), &hyle::simd::erf_f32_batch},
      {"atan f32 (4096)", Op::kAtan, gen(kN, -10.0f, 10.0f, 9), &hyle::simd::atan_f32_batch},
      {"cbrt f32 (4096)", Op::kCbrt, gen(kN, -100.0f, 100.0f, 10), &hyle::simd::cbrt_f32_batch},
      {"expm1 f32 (4096)", Op::kExpm1, gen(kN, -1.0f, 1.0f, 11), &hyle::simd::expm1_f32_batch},
      {"log1p f32 (4096)", Op::kLog1p, gen(kN, 0.0f, 10.0f, 12), &hyle::simd::log1p_f32_batch},
      {"asinh f32 (4096)", Op::kAsinh, gen(kN, -10.0f, 10.0f, 13), &hyle::simd::asinh_f32_batch},
      {"acosh f32 (4096)", Op::kAcosh, gen(kN, 1.0f, 50.0f, 14), &hyle::simd::acosh_f32_batch},
  };

  for (auto& c : cases) {
    nb::Bench b;
    b.title(c.title).relative(true).batch(kN).minEpochIterations(500);
    // Baseline: scalar libm element loop (what the default HYLE build also does).
    b.run("scalar libm loop", [&] {
      for (std::size_t i = 0; i < kN; ++i) dst[i] = scalar_of(c.op, c.src[i]);
      nb::doNotOptimizeAway(dst.data());
    });
    // HYLE batch op (only for already-wired ops): the live vectorized tier.
    if (c.hyle_batch) {
      b.run("HYLE *_f32_batch (vectorized)", [&] {
        c.hyle_batch(c.src.data(), dst.data(), kN);
        nb::doNotOptimizeAway(dst.data());
      });
    }
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

  // ---- f64 best-of regression: all 14 unary transcendentals are WIRED in HYLE (f64 dispatch chose
  // a winner per op; memory notes exp/tanh/exp2 winners FLIP vs f32). This section re-benches so the
  // f64 dispatch choice is proven, not extrapolated. HYLE row = the live f64 batch; xsimd/Highway =
  // same-ISA rivals. If HYLE loses to the OTHER lib here, the f64 dispatch picked the wrong core. ----
  std::vector<double> dst64(kN);
  struct Case64 {
    const char* title;
    Op op;
    std::vector<double> src;
    void (*hyle_batch)(const double*, double*, std::size_t) noexcept;
  };
  Case64 cases64[] = {
      {"exp f64 (4096)", Op::kExp, gen64(kN, -10.0, 10.0, 31), &hyle::simd::exp_f64_batch},
      {"log f64 (4096)", Op::kLog, gen64(kN, 0.01, 100.0, 32), &hyle::simd::log_f64_batch},
      {"tanh f64 (4096)", Op::kTanh, gen64(kN, -5.0, 5.0, 33), &hyle::simd::tanh_f64_batch},
      {"sin f64 (4096)", Op::kSin, gen64(kN, -3.14, 3.14, 34), &hyle::simd::sin_f64_batch},
      {"cos f64 (4096)", Op::kCos, gen64(kN, -3.14, 3.14, 35), &hyle::simd::cos_f64_batch},
      {"exp2 f64 (4096)", Op::kExp2, gen64(kN, -10.0, 10.0, 36), &hyle::simd::exp2_f64_batch},
      {"log2 f64 (4096)", Op::kLog2, gen64(kN, 0.01, 100.0, 37), &hyle::simd::log2_f64_batch},
      {"erf f64 (4096)", Op::kErf, gen64(kN, -3.0, 3.0, 38), &hyle::simd::erf_f64_batch},
      {"atan f64 (4096)", Op::kAtan, gen64(kN, -10.0, 10.0, 39), &hyle::simd::atan_f64_batch},
      {"cbrt f64 (4096)", Op::kCbrt, gen64(kN, -100.0, 100.0, 40), &hyle::simd::cbrt_f64_batch},
      {"expm1 f64 (4096)", Op::kExpm1, gen64(kN, -1.0, 1.0, 41), &hyle::simd::expm1_f64_batch},
      {"log1p f64 (4096)", Op::kLog1p, gen64(kN, 0.0, 10.0, 42), &hyle::simd::log1p_f64_batch},
      {"asinh f64 (4096)", Op::kAsinh, gen64(kN, -10.0, 10.0, 43), &hyle::simd::asinh_f64_batch},
      {"acosh f64 (4096)", Op::kAcosh, gen64(kN, 1.0, 50.0, 44), &hyle::simd::acosh_f64_batch},
  };
  for (auto& c : cases64) {
    nb::Bench b;
    b.title(c.title).relative(true).batch(kN).minEpochIterations(500);
    b.run("scalar libm loop", [&] {
      for (std::size_t i = 0; i < kN; ++i) dst64[i] = scalar_of64(c.op, c.src[i]);
      nb::doNotOptimizeAway(dst64.data());
    });
    b.run("HYLE *_f64_batch (vectorized)", [&] {
      c.hyle_batch(c.src.data(), dst64.data(), kN);
      nb::doNotOptimizeAway(dst64.data());
    });
#if defined(HYLE_BENCH_XSIMD)
    b.run("xsimd (same ISA SOTA)", [&] {
      xsimd_run64(c.op, c.src.data(), dst64.data(), kN);
      nb::doNotOptimizeAway(dst64.data());
    });
#endif
#if defined(HYLE_BENCH_HIGHWAY)
    b.run("Highway (same ISA SOTA)", [&] {
      highway_run64(c.op, c.src.data(), dst64.data(), kN);
      nb::doNotOptimizeAway(dst64.data());
    });
#endif
  }

  // hypot: a BINARY best-of candidate (BOTH xsimd and Highway have it -- unlike lgamma/tgamma
  // which are xsimd-only). Benched separately because it takes two inputs (a,b -> sqrt(a*a+b*b)
  // overflow-safe), which the unary Op switch above can't express. This decides the per-op winner
  // so hypot_f32_batch (if wired) borrows the fastest core -- same measure-don't-assume rule.
  {
    auto a = gen(kN, -1e3f, 1e3f, 20);
    auto bb = gen(kN, -1e3f, 1e3f, 21);
    nb::Bench b;
    b.title("hypot f32 (4096, binary)").relative(true).batch(kN).minEpochIterations(500);
    b.run("scalar libm loop", [&] {
      for (std::size_t i = 0; i < kN; ++i) dst[i] = std::hypot(a[i], bb[i]);
      nb::doNotOptimizeAway(dst.data());
    });
    b.run("HYLE safe", [&] {
      hyle::simd::hypot_f32_batch(a.data(), bb.data(), dst.data(), kN);
      nb::doNotOptimizeAway(dst.data());
    });
    b.run("HYLE fast", [&] {
      hyle::simd::hypot_in_range_f32_batch(a.data(), bb.data(), dst.data(), kN);
      nb::doNotOptimizeAway(dst.data());
    });
#if defined(HYLE_BENCH_XSIMD)
    b.run("xsimd (same ISA SOTA)", [&] {
      using batch = xsimd::batch<float>;
      constexpr std::size_t w = batch::size;
      std::size_t i = 0;
      for (; i + w <= kN; i += w)
        xsimd::hypot(batch::load_unaligned(a.data() + i), batch::load_unaligned(bb.data() + i))
            .store_unaligned(dst.data() + i);
      for (; i < kN; ++i) dst[i] = std::hypot(a[i], bb[i]);
      nb::doNotOptimizeAway(dst.data());
    });
#endif
#if defined(HYLE_BENCH_HIGHWAY)
    b.run("Highway (same ISA SOTA)", [&] {
      const hn::ScalableTag<float> d;
      const std::size_t w = hn::Lanes(d);
      std::size_t i = 0;
      for (; i + w <= kN; i += w)
        hn::StoreU(hn::Hypot(d, hn::LoadU(d, a.data() + i), hn::LoadU(d, bb.data() + i)), d,
                   dst.data() + i);
      for (; i < kN; ++i) dst[i] = std::hypot(a[i], bb[i]);
      nb::doNotOptimizeAway(dst.data());
    });
#endif
  }
  return 0;
}
