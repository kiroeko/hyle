#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "hwy/contrib/algo/copy-inl.h"
#include "hwy/contrib/algo/count-inl.h"
#include "hwy/contrib/algo/find-inl.h"
#include "hwy/contrib/algo/transform-inl.h"
#include "hwy/highway.h"
#include "nanobench.h"

#include "hyle/simd/native/algorithm.h"

namespace nb = ankerl::nanobench;
namespace hs = hyle::simd;
namespace hn = hwy::HWY_NAMESPACE;

namespace {

constexpr std::size_t kCount = 1u << 20;

void run_transform() {
  std::vector<std::uint32_t> hyle(kCount);
  std::vector<std::uint32_t> highway(kCount);
  for (std::size_t i = 0; i < kCount; ++i) {
    hyle[i] = static_cast<std::uint32_t>(i * 2654435761u);
    highway[i] = hyle[i];
  }
  const hn::ScalableTag<std::uint32_t> hd;

  nb::Bench bench;
  bench.title("native unary transform u32 n=1M")
      .unit("element")
      .batch(kCount)
      .relative(true)
      .minEpochIterations(32);
  bench.run("HYLE native transform", [&] {
    hs::native::transform(hyle.data(), hyle.size(), [](auto d, auto v) {
      return hs::Xor(v, hs::Set(d, UINT32_C(0x9E3779B9)));
    });
    nb::doNotOptimizeAway(hyle.data());
  });
  bench.run("Highway Transform", [&] {
    hn::Transform(hd, highway.data(), highway.size(), [](auto d, auto v) {
      return hn::Xor(v, hn::Set(d, UINT32_C(0x9E3779B9)));
    });
    nb::doNotOptimizeAway(highway.data());
  });
}

void run_predicates() {
  std::vector<std::uint32_t> input(kCount);
  std::vector<std::uint32_t> hyle_output(kCount);
  std::vector<std::uint32_t> highway_output(kCount);
  for (std::size_t i = 0; i < kCount; ++i)
    input[i] = static_cast<std::uint32_t>(i * 17u + 1u);
  const hn::ScalableTag<std::uint32_t> hd;

  const auto hyle_keep = [](auto d, auto v) {
    return hs::Eq(hs::And(v, hs::Set(d, 7u)), hs::Zero(d));
  };
  const auto highway_keep = [](auto d, auto v) {
    return hn::Eq(hn::And(v, hn::Set(d, 7u)), hn::Zero(d));
  };
  const auto hyle_absent = [](auto d, auto v) {
    return hs::Eq(v, hs::Set(d, UINT32_MAX));
  };
  const auto highway_absent = [](auto d, auto v) {
    return hn::Eq(v, hn::Set(d, UINT32_MAX));
  };

  nb::Bench bench;
  bench.title("native predicate algorithms u32 n=1M")
      .unit("element")
      .batch(kCount)
      .relative(true)
      .minEpochIterations(32);
  bench.run("HYLE count_if 1/8", [&] {
    const auto result =
        hs::native::count_if(input.data(), input.size(), hyle_keep);
    nb::doNotOptimizeAway(result);
  });
  bench.run("Highway CountIf 1/8", [&] {
    const auto result = hn::CountIf(hd, input.data(), input.size(), highway_keep);
    nb::doNotOptimizeAway(result);
  });
  bench.run("HYLE find_if absent", [&] {
    const auto result =
        hs::native::find_if(input.data(), input.size(), hyle_absent);
    nb::doNotOptimizeAway(result);
  });
  bench.run("Highway FindIf absent", [&] {
    const auto result = hn::FindIf(hd, input.data(), input.size(), highway_absent);
    nb::doNotOptimizeAway(result);
  });
  bench.run("HYLE copy_if 1/8", [&] {
    const auto result = hs::native::copy_if(
        input.data(), hyle_output.data(), input.size(), hyle_keep);
    nb::doNotOptimizeAway(result);
    nb::doNotOptimizeAway(hyle_output.data());
  });
  bench.run("Highway CopyIf 1/8", [&] {
    const auto* end = hn::CopyIf(hd, input.data(), input.size(),
                                 highway_output.data(), highway_keep);
    nb::doNotOptimizeAway(end);
    nb::doNotOptimizeAway(highway_output.data());
  });
}

}  // namespace

int main() {
  std::printf("native algorithm lanes: HYLE=%zu Highway=%zu\n",
              hs::Lanes(hs::ScalableTag<std::uint32_t>{}),
              hn::Lanes(hn::ScalableTag<std::uint32_t>{}));
  run_transform();
  run_predicates();
}
