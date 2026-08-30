// Ordinary pointer/count throughput: shipping HYLE API versus scalar
// auto-vectorization and, when enabled, the locked Highway peer. Provider
// qualification is intentionally not repeated here.
#include <nanobench.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "bench/simd/simd_kernels.h"
#include "hyle/simd/ops.h"  // public runtime-dispatched dot / saxpy

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
#if defined(HYLE_BENCH_HIGHWAY)
    b.run("Highway avx2 (naive 1-acc)", [&] {
      ankerl::nanobench::doNotOptimizeAway(hyle_bench::dot_highway(x.data(), y.data(), kN));
    });
    b.run("Highway contrib/dot (SOTA 4-acc)", [&] {
      ankerl::nanobench::doNotOptimizeAway(hyle_bench::dot_highway_contrib(x.data(), y.data(), kN));
    });
#endif
    b.run("HYLE dot_f32 (public dispatch)", [&] {
      ankerl::nanobench::doNotOptimizeAway(hyle::simd::dot(x.data(), y.data(), kN));
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
#if defined(HYLE_BENCH_HIGHWAY)
    b.run("Highway avx2 (SOTA)", [&] {
      hyle_bench::saxpy_highway(1.001f, x.data(), ys.data(), kN);
      ankerl::nanobench::doNotOptimizeAway(ys[0]);
    });
#endif
    b.run("HYLE saxpy_f32 (public dispatch)", [&] {
      hyle::simd::saxpy(1.001f, x.data(), ys.data(), kN);
      ankerl::nanobench::doNotOptimizeAway(ys[0]);
    });
  }

  {
    std::vector<float> ys(kN);
    ankerl::nanobench::Bench b;
    b.title("scale y=a*x (4096 f32)").relative(true).batch(kN).minEpochIterations(2000);
    b.run("scalar (auto-vec)", [&] {
      hyle_bench::scale_scalar(2.0f, x.data(), ys.data(), kN);
      ankerl::nanobench::doNotOptimizeAway(ys[0]);
    });
#if defined(HYLE_BENCH_HIGHWAY)
    b.run("Highway avx2 (SOTA)", [&] {
      hyle_bench::scale_highway(2.0f, x.data(), ys.data(), kN);
      ankerl::nanobench::doNotOptimizeAway(ys[0]);
    });
#endif
    b.run("HYLE scale_f32 (public dispatch)", [&] {
      hyle::simd::scale(x.data(), 2.0f, ys.data(), kN);
      ankerl::nanobench::doNotOptimizeAway(ys[0]);
    });
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
        ankerl::nanobench::doNotOptimizeAway(hyle::simd::dot(xd.data(), yd.data(), kN));
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
        hyle::simd::saxpy(1.001, xd.data(), yds.data(), kN);
        ankerl::nanobench::doNotOptimizeAway(yds[0]);
      });
    }
  }
  return 0;
}
