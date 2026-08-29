// SIMD-09 bounded search/scan integration instrument.
//
// The candidate is the exact shipping target body instantiated in this TU.
// The peer is direct Highway (search) or an independently spelled composition
// of the same Highway primitives (scan), with the same AVX2/NEON target,
// pointer/count, input domain, output traffic and alias contract. The shared
// parity engine supplies placement replicas, reverse order/layout, A/A
// calibration and CPU pinning. Public/body rows are informational and isolate
// the one allowed whole-buffer dispatch from algorithm parity.

#include <nanobench.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "hyle/simd/scan.h"
#include "hyle/simd/search.h"
#include "op_parity_engine.h"

#define HWY_COMPILE_ONLY_STATIC 1
#define HYLE_SIMD_VARIANT_SUFFIX parity
#include "src/simd/search/kernel_impl.h"
#include "src/simd/scan/kernel_impl.h"
#undef HYLE_SIMD_VARIANT_SUFFIX
#undef HWY_COMPILE_ONLY_STATIC

namespace nb = ankerl::nanobench;
namespace hs = hyle::simd;
namespace hn = hwy::HWY_NAMESPACE;

namespace {

constexpr std::size_t kResidentCount = 4096;
constexpr std::size_t kStreamingCount = 1u << 20;

HWY_INLINE void peer_add(const float* x, const float* y, float* dst,
                         std::size_t n) noexcept {
  const hn::ScalableTag<float> d;
  const std::size_t lanes = hn::Lanes(d);
  std::size_t i = 0;
  for (; lanes <= n - i; i += lanes) {
    hn::StoreU(hn::Add(hn::LoadU(d, x + i), hn::LoadU(d, y + i)), d,
               dst + i);
  }
  for (; i < n; ++i) dst[i] = x[i] + y[i];
}

HWY_INLINE std::size_t peer_count(const float* data, std::size_t n,
                                  float value) noexcept {
  return hn::Count(hn::ScalableTag<float>(), value, data, n);
}

HWY_INLINE std::size_t peer_argmax(const float* data,
                                   std::size_t n) noexcept {
  return hn::IndexOfMax(hn::ScalableTag<float>(), data, n);
}

template <std::size_t kLanes, class D, class V>
HWY_INLINE V peer_scan_step(D d, V value) noexcept {
  if constexpr (kLanes >= HWY_MAX_LANES_D(D)) {
    return value;
  } else {
    if (kLanes >= hn::Lanes(d)) return value;
    return hn::Add(value, hn::SlideUpLanes(d, value, kLanes));
  }
}

template <class D, class V>
HWY_INLINE V peer_scan_vector(D d, V value) noexcept {
  value = peer_scan_step<1>(d, value);
  value = peer_scan_step<2>(d, value);
  value = peer_scan_step<4>(d, value);
  value = peer_scan_step<8>(d, value);
  value = peer_scan_step<16>(d, value);
  value = peer_scan_step<32>(d, value);
  return value;
}

HWY_INLINE void peer_inclusive_scan(const std::uint32_t* src,
                                    std::uint32_t* dst,
                                    std::size_t n) noexcept {
  if (n == 0) return;
  const hn::ScalableTag<std::uint32_t> d;
  const std::size_t lanes = hn::Lanes(d);
  std::uint32_t carry = 0;
  std::size_t i = 0;
  for (; lanes <= n - i; i += lanes) {
    const auto inclusive = peer_scan_vector(d, hn::LoadU(d, src + i));
    const auto with_carry = hn::Add(inclusive, hn::Set(d, carry));
    hn::StoreU(with_carry, d, dst + i);
    carry = hn::ExtractLane(with_carry, lanes - 1);
  }
  for (; i < n; ++i) {
    carry += src[i];
    dst[i] = carry;
  }
}

std::size_t epoch_iterations(std::size_t n) {
  // Each epoch remains above the timer/noise floor without turning the two
  // 1M streaming rows into a multi-minute benchmark.
  return n == kResidentCount ? 8192 : 512;
}

template <class Candidate, class Peer>
void run_pair(const std::string& title, std::size_t n, Candidate candidate,
              Peer peer) {
  nb::Bench bench;
  bench.title(title).relative(true).batch(n).minEpochIterations(
      epoch_iterations(n));
  HYLE_RUN_PAIR(
      bench,
      [&](auto rep) {
        hyle_bench::stagger(rep);
        candidate();
        hyle_bench::mark(rep);
      },
      [&](auto rep) {
        hyle_bench::stagger(rep);
        peer();
        hyle_bench::mark(rep);
      });
}

template <class Public, class Body>
void run_public_info(const std::string& title, std::size_t n,
                     Public public_call, Body body_call) {
  nb::Bench bench;
  bench.title(title).relative(true).batch(n).minEpochIterations(
      epoch_iterations(n));
  HYLE_RUN_INFO_PAIR(
      bench,
      [&](auto rep) {
        hyle_bench::stagger(rep);
        public_call();
        hyle_bench::mark(rep);
      },
      [&](auto rep) {
        hyle_bench::stagger(rep);
        body_call();
        hyle_bench::mark(rep);
      });
}

void run_size(std::size_t n) {
  std::vector<float> input(n), other(n), output(n);
  std::vector<std::uint32_t> scan_input(n), scan_output(n);
  for (std::size_t i = 0; i < n; ++i) {
    input[i] = static_cast<float>(static_cast<int>((i * 17) % 257) - 128) /
               64.0f;
    other[i] = static_cast<float>(static_cast<int>((i * 29) % 251) - 125) /
               64.0f;
    scan_input[i] = static_cast<std::uint32_t>((i * 13 + 7) & 31u);
  }
  // Keep the timed argmax domain finite-normal and give it one unambiguous
  // maximum. NaN/first-tie semantics belong to the differential gate.
  input[n / 2] = 128.0f;
  const float needle = 0.25f;
  const std::string suffix = " (" + std::to_string(n) + ")";

  if (n == kResidentCount) {
    run_pair("Add f32 (4096)", n,
             [&] { peer_add(input.data(), other.data(), output.data(), n); },
             [&] { peer_add(input.data(), other.data(), output.data(), n); });
  }

  run_pair("search count f32" + suffix, n,
           [&] {
             nb::doNotOptimizeAway(hs::detail::count_equal_f32_parity(
                 input.data(), n, needle));
           },
           [&] {
             nb::doNotOptimizeAway(peer_count(input.data(), n, needle));
           });
  run_pair("search argmax f32" + suffix, n,
           [&] {
             nb::doNotOptimizeAway(
                 hs::detail::index_of_max_f32_parity(input.data(), n));
           },
           [&] { nb::doNotOptimizeAway(peer_argmax(input.data(), n)); });
  run_pair("inclusive scan u32" + suffix, n,
           [&] {
             hs::detail::inclusive_scan_u32_parity(
                 scan_input.data(), scan_output.data(), n);
             nb::doNotOptimizeAway(scan_output);
           },
           [&] {
             peer_inclusive_scan(scan_input.data(), scan_output.data(), n);
             nb::doNotOptimizeAway(scan_output);
           });

  run_public_info("public/body search count" + suffix, n,
                  [&] {
                    nb::doNotOptimizeAway(
                        hs::count_equal(input.data(), n, needle));
                  },
                  [&] {
                    nb::doNotOptimizeAway(
                        hs::detail::count_equal_f32_parity(
                            input.data(), n, needle));
                  });
  run_public_info("public/body search argmax" + suffix, n,
                  [&] {
                    nb::doNotOptimizeAway(
                        hs::index_of_max(input.data(), n));
                  },
                  [&] {
                    nb::doNotOptimizeAway(
                        hs::detail::index_of_max_f32_parity(
                            input.data(), n));
                  });
  run_public_info("public/body inclusive scan" + suffix, n,
                  [&] {
                    hs::inclusive_scan(scan_input.data(), scan_output.data(),
                                       n);
                    nb::doNotOptimizeAway(scan_output);
                  },
                  [&] {
                    hs::detail::inclusive_scan_u32_parity(
                        scan_input.data(), scan_output.data(), n);
                    nb::doNotOptimizeAway(scan_output);
                  });
}

}  // namespace

int main() {
  hyle_bench::pin_and_boost();
  std::printf(
      "SIMD-09 search/scan parity: shipping target body vs direct Highway\n");
  run_size(kResidentCount);
  run_size(kStreamingCount);
}
