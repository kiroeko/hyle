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

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "bench/simd/simd_kernels.h"
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
  return 0;
}
