#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#include "hwy/highway.h"
#include "hwy/contrib/algo/count-inl.h"
#include "hwy/contrib/algo/find-inl.h"
#include "hwy/contrib/algo/is_sorted-inl.h"
#include "hwy/contrib/algo/minmax-inl.h"
#include "nanobench.h"

#include "hyle/platform.h"
#include "hyle/simd/runtime.h"
#include "hyle/simd/search.h"

namespace nb = ankerl::nanobench;
namespace hs = hyle::simd;
namespace hn = hwy::HWY_NAMESPACE;

namespace {

const hs::Runtime& same_isa_runtime() {
#if defined(HYLE_ARCH_X64)
  if (const auto* rt = hs::runtime_for(hs::Isa::kAvx2)) return *rt;
#elif defined(HYLE_ARCH_ARM64)
  if (const auto* rt = hs::runtime_for(hs::Isa::kNeon)) return *rt;
#endif
  return *hs::runtime_for(hs::Isa::kScalar);
}

template <typename T>
void run_type(const char* name) {
  constexpr std::size_t n = 1u << 20;
  std::vector<T> data(n);
  for (std::size_t i = 0; i < n; ++i)
    data[i] = static_cast<T>(static_cast<int>(i % 251) - 125);
  const T needle = static_cast<T>(17);
  const T missing = static_cast<T>(126);
  const hn::ScalableTag<T> d;
  const auto& fixed = same_isa_runtime();

  nb::Bench bench;
  bench.title(std::string("search ") + name + " n=1M")
      .unit("element")
      .batch(n)
      .minEpochIterations(128);

  bench.run("std count", [&] {
    auto result = std::count(data.begin(), data.end(), needle);
    nb::doNotOptimizeAway(result);
  });
  bench.run("Highway Count", [&] {
    auto result = hn::Count(d, needle, data.data(), n);
    nb::doNotOptimizeAway(result);
  });
  bench.run("HYLE fixed Count", [&] {
    std::size_t result;
    if constexpr (std::is_same_v<T, std::uint32_t>)
      result = fixed.search.count_equal_u32(data.data(), n, needle);
    else
      result = fixed.search.count_equal_f32(data.data(), n, needle);
    nb::doNotOptimizeAway(result);
  });
  bench.run("HYLE public Count", [&] {
    auto result = hs::count_equal(data.data(), n, needle);
    nb::doNotOptimizeAway(result);
  });

  bench.run("std find absent", [&] {
    auto result = std::find(data.begin(), data.end(), missing);
    nb::doNotOptimizeAway(result);
  });
  bench.run("Highway Find absent", [&] {
    auto result = hn::Find(d, missing, data.data(), n);
    nb::doNotOptimizeAway(result);
  });
  bench.run("HYLE fixed Find absent", [&] {
    std::size_t result;
    if constexpr (std::is_same_v<T, std::uint32_t>)
      result = fixed.search.find_equal_u32(data.data(), n, missing);
    else
      result = fixed.search.find_equal_f32(data.data(), n, missing);
    nb::doNotOptimizeAway(result);
  });
  bench.run("HYLE public Find absent", [&] {
    auto result = hs::find_equal(data.data(), n, missing);
    nb::doNotOptimizeAway(result);
  });

  bench.run("std minmax", [&] {
    auto result = std::minmax_element(data.begin(), data.end());
    nb::doNotOptimizeAway(result);
  });
  bench.run("Highway Min+Max", [&] {
    auto lo = hn::MinValue(d, data.data(), n);
    auto hi = hn::MaxValue(d, data.data(), n);
    nb::doNotOptimizeAway(lo);
    nb::doNotOptimizeAway(hi);
  });
  bench.run("HYLE fixed MinMax", [&] {
    T lo{}, hi{};
    if constexpr (std::is_same_v<T, std::uint32_t>)
      fixed.search.minmax_value_u32(data.data(), n, &lo, &hi);
    else
      fixed.search.minmax_value_f32(data.data(), n, &lo, &hi);
    nb::doNotOptimizeAway(lo);
    nb::doNotOptimizeAway(hi);
  });
  bench.run("HYLE public MinMax", [&] {
    T lo{}, hi{};
    hs::minmax_value(data.data(), n, &lo, &hi);
    nb::doNotOptimizeAway(lo);
    nb::doNotOptimizeAway(hi);
  });

  std::sort(data.begin(), data.end());
  bench.run("std is_sorted", [&] {
    auto result = std::is_sorted(data.begin(), data.end());
    nb::doNotOptimizeAway(result);
  });
  bench.run("Highway IsSorted", [&] {
    auto result = hn::IsSorted(d, data.data(), n);
    nb::doNotOptimizeAway(result);
  });
  bench.run("HYLE fixed IsSorted", [&] {
    bool result;
    if constexpr (std::is_same_v<T, std::uint32_t>)
      result = fixed.search.is_sorted_asc_u32(data.data(), n);
    else
      result = fixed.search.is_sorted_asc_f32(data.data(), n);
    nb::doNotOptimizeAway(result);
  });
  bench.run("HYLE public IsSorted", [&] {
    auto result = hs::is_sorted(data.data(), n);
    nb::doNotOptimizeAway(result);
  });
}

}  // namespace

int main() {
  std::printf("HYLE public ISA=%u, fixed comparison ISA=%u\n",
              static_cast<unsigned>(hs::runtime().isa),
              static_cast<unsigned>(same_isa_runtime().isa));
  run_type<std::uint32_t>("u32");
  run_type<float>("f32");
}
