// simd_bench -- races the dot/saxpy kernels: scalar-autovec baseline vs HYLE sse4 vs HYLE
// avx2 (vs Highway avx2 if -DHYLE_BENCH_HIGHWAY). nanobench .relative(true) prints each
// relative to the first (scalar). Answers "is HYLE SIMD at SOTA?" with real numbers, and
// "how much over auto-vec?" -- both printed, win or lose.
//
// ALSO races the PUBLIC dispatched entry points (hyle::simd::dot_f32 / saxpy_f32 from
// ops.h) against the same-ISA local kernel. Those exported symbols go through one cached
// fn-pointer (the "one binary, any CPU" dispatch model ops.h documents); this is the only
// bench that measures that indirection cost -- the local hyle_bench::dot_avx2 above is the
// zero-indirection ceiling, so the gap between them IS the dispatch tax, printed honestly.
#include <nanobench.h>

#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "bench/simd/simd_kernels.h"
#include "hyle/platform.h"
#include "hyle/simd/ops.h"  // public dispatched dot_f32 / saxpy_f32 / active_variant

namespace {
constexpr std::size_t kN = 4096;  // fits L1; isolates compute throughput, not memory BW

std::vector<float> gen(std::size_t n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(n);
  for (auto& e : v) e = dist(rng);
  return v;
}

using DirectDotFn = float (*)(const float*, const float*, std::size_t);

// hyle_simd's variant entry symbols are private to the DLL, as they must be. The benchmark has
// same-ISA local x64 anchors for scalar/SSE4/AVX2; use one only when it matches the public tier.
// AVX-512 and NEON intentionally omit this comparison rather than crossing an ABI boundary or
// comparing different widths and mislabeling the result as dispatch overhead.
DirectDotFn direct_active_dot() noexcept {
#if defined(HYLE_ARCH_X64)
  switch (hyle::simd::active_variant()) {
    case hyle::simd::Isa::kAvx2: return &hyle_bench::dot_avx2;
    case hyle::simd::Isa::kSse42: return &hyle_bench::dot_sse4;
    case hyle::simd::Isa::kScalar: return &hyle_bench::dot_scalar;
    default: return nullptr;
  }
#else
  return nullptr;
#endif
}
}  // namespace

int main() {
  const std::vector<float> x = gen(kN, 1);
  const std::vector<float> y = gen(kN, 2);

  {
    ankerl::nanobench::Bench b;
    b.title("dot product (4096 f32)").relative(true).batch(kN).minEpochIterations(2000);
    b.run("scalar (auto-vec)", [&] {
      ankerl::nanobench::doNotOptimizeAway(hyle_bench::dot_scalar(x.data(), y.data(), kN));
    });
    b.run("HYLE sse4 (128b)", [&] {
      ankerl::nanobench::doNotOptimizeAway(hyle_bench::dot_sse4(x.data(), y.data(), kN));
    });
    b.run("HYLE avx2 (256b+FMA)", [&] {
      ankerl::nanobench::doNotOptimizeAway(hyle_bench::dot_avx2(x.data(), y.data(), kN));
    });
#if defined(HYLE_BENCH_HIGHWAY)
    b.run("Highway avx2 (naive 1-acc)", [&] {
      ankerl::nanobench::doNotOptimizeAway(hyle_bench::dot_highway(x.data(), y.data(), kN));
    });
    b.run("Highway contrib/dot (SOTA 4-acc)", [&] {
      ankerl::nanobench::doNotOptimizeAway(hyle_bench::dot_highway_contrib(x.data(), y.data(), kN));
    });
#endif
    // PUBLIC dispatched entry (fn-ptr indirection): the shipped library symbol.
    // Gap vs "HYLE avx2" above = the dispatch tax (one cached indirect call).
    b.run("HYLE dot_f32 (public dispatch)", [&] {
      ankerl::nanobench::doNotOptimizeAway(hyle::simd::dot_f32(x.data(), y.data(), kN));
    });
  }

  // The 4096 row above shows the steady-state ceiling. Sweep batch sizes with the SAME active
  // implementation on both sides so this reports only the cached dispatcher indirect-call tax.
  if (const DirectDotFn direct_dot = direct_active_dot()) {
    for (const std::size_t n : std::array<std::size_t, 6>{1, 16, 64, 256, 1024, kN}) {
      ankerl::nanobench::Bench b;
      const std::string title = "dispatch overhead dot_f32 (n=" + std::to_string(n) + ")";
      b.title(title.c_str()).relative(true).batch(n).minEpochIterations(n <= 16 ? 500000 : 10000);
      b.run("direct active kernel", [&] {
        ankerl::nanobench::doNotOptimizeAway(direct_dot(x.data(), y.data(), n));
      });
      b.run("HYLE dot_f32 (public dispatch)", [&] {
        ankerl::nanobench::doNotOptimizeAway(hyle::simd::dot_f32(x.data(), y.data(), n));
      });
    }
  } else {
    std::printf("dispatch scaling omitted: no same-ISA local direct anchor for tier=%s\n",
                hyle::simd::isa_name(hyle::simd::active_variant()));
  }

  {
    std::vector<float> ys = y;  // saxpy mutates y; fresh copy per variant via reset
    ankerl::nanobench::Bench b;
    b.title("saxpy y=a*x+y (4096 f32)").relative(true).batch(kN).minEpochIterations(2000);
    b.run("scalar (auto-vec)", [&] {
      hyle_bench::saxpy_scalar(1.001f, x.data(), ys.data(), kN);
      ankerl::nanobench::doNotOptimizeAway(ys[0]);
    });
    b.run("HYLE sse4 (128b)", [&] {
      hyle_bench::saxpy_sse4(1.001f, x.data(), ys.data(), kN);
      ankerl::nanobench::doNotOptimizeAway(ys[0]);
    });
    b.run("HYLE avx2 (256b+FMA)", [&] {
      hyle_bench::saxpy_avx2(1.001f, x.data(), ys.data(), kN);
      ankerl::nanobench::doNotOptimizeAway(ys[0]);
    });
#if defined(HYLE_BENCH_HIGHWAY)
    b.run("Highway avx2 (SOTA)", [&] {
      hyle_bench::saxpy_highway(1.001f, x.data(), ys.data(), kN);
      ankerl::nanobench::doNotOptimizeAway(ys[0]);
    });
#endif
    b.run("HYLE saxpy_f32 (public dispatch)", [&] {
      hyle::simd::saxpy_f32(1.001f, x.data(), ys.data(), kN);
      ankerl::nanobench::doNotOptimizeAway(ys[0]);
    });
  }

  {
    ankerl::nanobench::Bench b;
    b.title("reduce_max (4096 f32)").relative(true).batch(kN).minEpochIterations(2000);
    b.run("scalar (auto-vec)", [&] {
      ankerl::nanobench::doNotOptimizeAway(hyle_bench::reduce_max_scalar(x.data(), kN));
    });
    b.run("HYLE sse4 (128b)", [&] {
      ankerl::nanobench::doNotOptimizeAway(hyle_bench::reduce_max_sse4(x.data(), kN));
    });
    b.run("HYLE avx2 (256b)", [&] {
      ankerl::nanobench::doNotOptimizeAway(hyle_bench::reduce_max_avx2(x.data(), kN));
    });
#if defined(HYLE_BENCH_HIGHWAY)
    b.run("Highway avx2 (SOTA)", [&] {
      ankerl::nanobench::doNotOptimizeAway(hyle_bench::reduce_max_highway(x.data(), kN));
    });
#endif
  }

  {
    std::vector<float> ys(kN);
    ankerl::nanobench::Bench b;
    b.title("scale y=a*x (4096 f32)").relative(true).batch(kN).minEpochIterations(2000);
    b.run("scalar (auto-vec)", [&] {
      hyle_bench::scale_scalar(2.0f, x.data(), ys.data(), kN);
      ankerl::nanobench::doNotOptimizeAway(ys[0]);
    });
    b.run("HYLE sse4 (128b)", [&] {
      hyle_bench::scale_sse4(2.0f, x.data(), ys.data(), kN);
      ankerl::nanobench::doNotOptimizeAway(ys[0]);
    });
    b.run("HYLE avx2 (256b)", [&] {
      hyle_bench::scale_avx2(2.0f, x.data(), ys.data(), kN);
      ankerl::nanobench::doNotOptimizeAway(ys[0]);
    });
#if defined(HYLE_BENCH_HIGHWAY)
    b.run("Highway avx2 (SOTA)", [&] {
      hyle_bench::scale_highway(2.0f, x.data(), ys.data(), kN);
      ankerl::nanobench::doNotOptimizeAway(ys[0]);
    });
#endif
  }

  // f64 dot/saxpy (directive3): CAE/geometry solvers run in double; simd_bench only covered f32.
  // Bench the PUBLIC dispatched dot_f64/saxpy_f64 (shipped symbols) vs a scalar reference. f64 is
  // half the lane count of f32 (4 vs 8 on AVX2), so the vectorization win is ~half but still real,
  // and the wider accumulator is exactly why solvers want f64.
  {
    std::vector<double> xd(kN), yd(kN);
    for (std::size_t i = 0; i < kN; ++i) { xd[i] = static_cast<double>(x[i]); yd[i] = static_cast<double>(y[i]); }
    {
      ankerl::nanobench::Bench b;
      b.title("dot product (4096 f64)").relative(true).batch(kN).minEpochIterations(2000);
      b.run("scalar (auto-vec)", [&] {
        double s = 0.0;
        for (std::size_t i = 0; i < kN; ++i) s += xd[i] * yd[i];
        ankerl::nanobench::doNotOptimizeAway(s);
      });
      b.run("HYLE dot_f64 (public dispatch)", [&] {
        ankerl::nanobench::doNotOptimizeAway(hyle::simd::dot_f64(xd.data(), yd.data(), kN));
      });
    }
    {
      std::vector<double> yds = yd;
      ankerl::nanobench::Bench b;
      b.title("saxpy y=a*x+y (4096 f64)").relative(true).batch(kN).minEpochIterations(2000);
      b.run("scalar (auto-vec)", [&] {
        for (std::size_t i = 0; i < kN; ++i) yds[i] = 1.001 * xd[i] + yds[i];
        ankerl::nanobench::doNotOptimizeAway(yds[0]);
      });
      b.run("HYLE saxpy_f64 (public dispatch)", [&] {
        hyle::simd::saxpy_f64(1.001, xd.data(), yds.data(), kN);
        ankerl::nanobench::doNotOptimizeAway(yds[0]);
      });
    }
  }
  return 0;
}
