// Stream is a non-temporal STORE contract, so an L1-resident op-parity loop is the wrong
// instrument. This workload writes a buffer larger than typical LLCs, fences x86 streaming
// stores before stopping the clock, alternates HYLE/Highway order, and reports useful GiB/s.
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <vector>

#include "hyle/simd/vec.h"
#define HWY_HEADER_ONLY
#include "hwy/highway.h"

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace hs = hyle::simd;
namespace hn = hwy::HWY_NAMESPACE;

namespace {

constexpr std::size_t kBytes = 256u * 1024u * 1024u;
constexpr int kRuns = 6;

void stream_fence() noexcept {
#if defined(_M_X64) || defined(__x86_64__)
  _mm_sfence();
#endif
}

template <class F>
double time_seconds(std::uint8_t* dst, F&& write) {
  std::memset(dst, 0, kBytes);  // fault pages in; not timed
  const auto begin = std::chrono::steady_clock::now();
  write();
  stream_fence();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(end - begin).count();
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return (values[values.size() / 2 - 1] + values[values.size() / 2]) * 0.5;
}

double gib_per_second(double seconds) {
  return (static_cast<double>(kBytes) / (1024.0 * 1024.0 * 1024.0)) / seconds;
}

}  // namespace

int main() {
  auto* dst = static_cast<std::uint8_t*>(
      ::operator new[](kBytes, std::align_val_t{64}));

  const hs::ScalableTag<std::uint64_t> hd;
  const hn::ScalableTag<std::uint64_t> gd;
  const std::size_t hw = hs::Lanes(hd);
  const std::size_t gw = hn::Lanes(gd);
  const auto hv = hs::Set(hd, UINT64_C(0x9e3779b97f4a7c15));
  const auto gv = hn::Set(gd, UINT64_C(0x9e3779b97f4a7c15));

  std::vector<double> hyle_times, highway_times, cached_times;
  hyle_times.reserve(kRuns);
  highway_times.reserve(kRuns);
  cached_times.reserve(kRuns);

  const auto run_hyle = [&] {
    auto* out = reinterpret_cast<std::uint64_t*>(dst);
    const std::size_t n = kBytes / sizeof(std::uint64_t);
    for (std::size_t i = 0; i + hw <= n; i += hw) hs::Stream(hv, hd, out + i);
  };
  const auto run_highway = [&] {
    auto* out = reinterpret_cast<std::uint64_t*>(dst);
    const std::size_t n = kBytes / sizeof(std::uint64_t);
    for (std::size_t i = 0; i + gw <= n; i += gw) hn::Stream(gv, gd, out + i);
  };
  const auto run_cached = [&] {
    auto* out = reinterpret_cast<std::uint64_t*>(dst);
    const std::size_t n = kBytes / sizeof(std::uint64_t);
    for (std::size_t i = 0; i + hw <= n; i += hw) hs::Store(hv, hd, out + i);
  };

  for (int run = 0; run < kRuns; ++run) {
    if ((run & 1) == 0) {
      hyle_times.push_back(time_seconds(dst, run_hyle));
      highway_times.push_back(time_seconds(dst, run_highway));
    } else {
      highway_times.push_back(time_seconds(dst, run_highway));
      hyle_times.push_back(time_seconds(dst, run_hyle));
    }
    cached_times.push_back(time_seconds(dst, run_cached));
  }

  const double hyle = median(hyle_times);
  const double highway = median(highway_times);
  const double cached = median(cached_times);
  std::printf("Stream working_set_mib=256 runs=%d hyle_lanes=%zu highway_lanes=%zu\n",
              kRuns, hw, gw);
  std::printf("implementation\tseconds\tGiB/s\n");
  std::printf("HYLE Stream\t%.9f\t%.3f\n", hyle, gib_per_second(hyle));
  std::printf("Highway Stream\t%.9f\t%.3f\n", highway, gib_per_second(highway));
  std::printf("HYLE cached Store\t%.9f\t%.3f\n", cached, gib_per_second(cached));
  std::printf("HYLE/Highway throughput ratio\t%.4f\n", highway / hyle);

  ::operator delete[](dst, std::align_val_t{64});
  return 0;
}
