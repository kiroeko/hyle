// radix_sort_bench -- crossover: std::sort / HYLE sort (auto) / explicit radix_sort vs n.
#include <nanobench.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "hyle/simd/sort.h"

namespace {
std::vector<std::uint32_t> gen(std::size_t n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<std::uint32_t> d;
  std::vector<std::uint32_t> v(n);
  for (auto& e : v) e = d(rng);
  return v;
}
}  // namespace

int main() {
  for (std::size_t n : {std::size_t{1024}, std::size_t{4096}, std::size_t{16384},
                        std::size_t{65536}, std::size_t{262144}, std::size_t{1048576}}) {
    const auto base = gen(n, 1);
    ankerl::nanobench::Bench b;
    b.title("sort u32 n=" + std::to_string(n)).relative(true).unit("sort").minEpochIterations(5);
    b.run("std::sort", [&] {
      auto v = base;
      std::sort(v.begin(), v.end());
      ankerl::nanobench::doNotOptimizeAway(v.data());
    });
    b.run("HYLE sort (auto)", [&] {
      auto v = base;
      hyle::simd::sort(v.data(), n);
      ankerl::nanobench::doNotOptimizeAway(v.data());
    });
    b.run("HYLE radix_sort", [&] {
      auto v = base;
      hyle::simd::radix_sort(v.data(), n);
      ankerl::nanobench::doNotOptimizeAway(v.data());
    });
  }
  return 0;
}
