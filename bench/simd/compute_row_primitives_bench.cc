// SIMD v7 Compute-row qualification. The timed pairs keep the same f32
// storage, full-precision/NaN/signed-zero semantics, fixed ISA and allocation
// contract. Scalar code is used only as a correctness oracle; performance is
// compared with an independently spelled direct-Highway lowering.
#include <nanobench.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

#include "hyle/core/compiler.h"
#include "hyle/simd/ops.h"
#include "hyle/simd/search.h"

#include "../common/perf_process.h"

#define HWY_COMPILE_ONLY_STATIC 1
#include "hwy/highway.h"
#undef HWY_COMPILE_ONLY_STATIC

namespace {
namespace nb = ankerl::nanobench;
namespace hn = hwy::HWY_NAMESPACE;
namespace hs = hyle::simd;

constexpr std::size_t kHidden = 4096;
constexpr std::size_t kFfn = 11008;

bool environment_present(const char *name) {
#if defined(_WIN32)
  char *value = nullptr;
  std::size_t size = 0;
  const bool present = _dupenv_s(&value, &size, name) == 0 && value != nullptr;
  std::free(value);
  return present;
#else
  return std::getenv(name) != nullptr;
#endif
}

template <class Candidate, class Peer>
void run_pair(nb::Bench &bench, bool peer_first,
              std::string_view candidate_name, Candidate &&candidate,
              std::string_view peer_name, Peer &&peer) {
  if (peer_first) {
    bench.run(peer_name, std::forward<Peer>(peer));
    bench.run(candidate_name, std::forward<Candidate>(candidate));
  } else {
    bench.run(candidate_name, std::forward<Candidate>(candidate));
    bench.run(peer_name, std::forward<Peer>(peer));
  }
}

template <bool kMinimum, class V>
HYLE_FORCE_INLINE V number_extreme(V a, V b) noexcept {
  const hn::DFromV<V> d;
  const auto zero = hn::Zero(d);
  const auto both_zero = hn::And(hn::Eq(a, zero), hn::Eq(b, zero));
  const auto numeric = kMinimum ? hn::MinNumber(a, b) : hn::MaxNumber(a, b);
  const auto signed_zero = kMinimum ? hn::Or(a, b) : hn::And(a, b);
  return hn::IfThenElse(both_zero, signed_zero, numeric);
}

HYLE_NOINLINE void highway_reciprocal_sqrt_peer(const float *input,
                                                float *output,
                                                std::size_t count) noexcept {
  const hn::ScalableTag<float> d;
  const std::size_t lanes = hn::Lanes(d);
  const auto one = hn::Set(d, 1.0F);
  std::size_t i = 0;
  for (; lanes <= count - i; i += lanes) {
    hn::StoreU(hn::Div(one, hn::Sqrt(hn::LoadU(d, input + i))), d, output + i);
  }
  for (; i < count; ++i)
    output[i] = 1.0F / std::sqrt(input[i]);
}

HYLE_NOINLINE void highway_fmin_peer(const float *left, const float *right,
                                     float *output,
                                     std::size_t count) noexcept {
  const hn::ScalableTag<float> d;
  const std::size_t lanes = hn::Lanes(d);
  std::size_t i = 0;
  for (; lanes <= count - i; i += lanes) {
    hn::StoreU(
        number_extreme<true>(hn::LoadU(d, left + i), hn::LoadU(d, right + i)),
        d, output + i);
  }
  for (; i < count; ++i)
    output[i] = std::fmin(left[i], right[i]);
}

HYLE_NOINLINE void highway_select_peer(const std::uint8_t *condition,
                                       const float *when_true,
                                       const float *when_false, float *output,
                                       std::size_t count) noexcept {
  const hn::ScalableTag<float> df;
  const hn::Rebind<std::uint8_t, decltype(df)> dc;
  const hn::RebindToUnsigned<decltype(df)> du;
  const std::size_t lanes = hn::Lanes(df);
  std::size_t i = 0;
  for (; lanes <= count - i; i += lanes) {
    const auto condition_u32 = hn::PromoteTo(du, hn::LoadU(dc, condition + i));
    const auto mask = hn::RebindMask(df, hn::Ne(condition_u32, hn::Zero(du)));
    hn::StoreU(hn::IfThenElse(mask, hn::LoadU(df, when_true + i),
                              hn::LoadU(df, when_false + i)),
               df, output + i);
  }
  for (; i < count; ++i)
    output[i] = condition[i] != 0 ? when_true[i] : when_false[i];
}

HYLE_NOINLINE float highway_min_value_peer(const float *input,
                                           std::size_t count) noexcept {
  if (count == 0)
    return std::numeric_limits<float>::infinity();
  const hn::ScalableTag<float> d;
  const std::size_t lanes = hn::Lanes(d);
  const auto identity = hn::Set(d, std::numeric_limits<float>::infinity());
  auto acc0 = identity;
  auto acc1 = identity;
  auto acc2 = identity;
  auto acc3 = identity;
  auto nan = hn::FirstN(d, 0);
  std::size_t i = 0;
  for (; 4 * lanes <= count - i; i += 4 * lanes) {
    const auto v0 = hn::LoadU(d, input + i);
    const auto v1 = hn::LoadU(d, input + i + lanes);
    const auto v2 = hn::LoadU(d, input + i + 2 * lanes);
    const auto v3 = hn::LoadU(d, input + i + 3 * lanes);
    nan = hn::Or(nan, hn::Or(hn::Or(hn::IsNaN(v0), hn::IsNaN(v1)),
                             hn::Or(hn::IsNaN(v2), hn::IsNaN(v3))));
    acc0 = hn::Min(acc0, v0);
    acc1 = hn::Min(acc1, v1);
    acc2 = hn::Min(acc2, v2);
    acc3 = hn::Min(acc3, v3);
  }
  auto acc = hn::Min(hn::Min(acc0, acc1), hn::Min(acc2, acc3));
  for (; lanes <= count - i; i += lanes) {
    const auto value = hn::LoadU(d, input + i);
    nan = hn::Or(nan, hn::IsNaN(value));
    acc = hn::Min(acc, value);
  }
  float result = hn::ReduceMin(d, acc);
  for (; i < count; ++i) {
    if (std::isnan(input[i]))
      return std::numeric_limits<float>::quiet_NaN();
    if (input[i] < result || (input[i] == 0.0F && result == 0.0F &&
                              std::signbit(input[i]) && !std::signbit(result)))
      result = input[i];
  }
  if (!hn::AllFalse(d, nan))
    return std::numeric_limits<float>::quiet_NaN();
  if (result != 0.0F)
    return result;
  for (std::size_t index = 0; index < count; ++index) {
    if (input[index] == 0.0F && std::signbit(input[index]))
      return -0.0F;
  }
  return 0.0F;
}

bool same_value(float actual, float expected) {
  return (std::isnan(actual) && std::isnan(expected)) ||
         std::bit_cast<std::uint32_t>(actual) ==
             std::bit_cast<std::uint32_t>(expected);
}

} // namespace

int main() {
  const auto qos =
      hyle::bench::perf::configure_thread_qos("HYLE_BENCH_APPLE_QOS");
  const auto affinity =
      hyle::bench::perf::configure_thread_affinity("HYLE_PERF_CALLER_CPU");
  const bool peer_first = environment_present("HYLE_BENCH_RIVAL_FIRST");
  std::fprintf(stderr,
               "# HYLE_SIMD_COMPUTE_ROWS_RUN execution_order=%s "
               "apple_qos=%s qos_ok=%s affinity_topology=%s caller_cpu=%s "
               "affinity_ok=%s run_nonce=%s\n",
               peer_first ? "peer-first" : "candidate-first", qos.name, qos.ok,
               affinity.topology, affinity.cpu, affinity.ok,
               hyle::bench::perf::run_nonce());

  std::vector<float> left(kFfn), right(kFfn), output(kFfn), peer_output(kFfn);
  std::vector<std::uint8_t> condition(kFfn);
  for (std::size_t i = 0; i < kFfn; ++i) {
    left[i] =
        0.125F + std::abs(std::sin(static_cast<float>(i) * 0.0137F)) * 31.0F;
    right[i] = 0.25F + std::cos(static_cast<float>(i) * 0.0191F) * 17.0F;
    condition[i] = static_cast<std::uint8_t>((i % 5 == 0) ? 0 : (i % 251 + 1));
  }

  hs::reciprocal_sqrt(left.data(), output.data(), kHidden);
  highway_reciprocal_sqrt_peer(left.data(), peer_output.data(), kHidden);
  for (std::size_t i = 0; i < kHidden; ++i)
    if (!same_value(output[i], peer_output[i]))
      return 2;

  hs::fmin(left.data(), right.data(), output.data(), kFfn);
  highway_fmin_peer(left.data(), right.data(), peer_output.data(), kFfn);
  for (std::size_t i = 0; i < kFfn; ++i)
    if (!same_value(output[i], peer_output[i]))
      return 3;

  hs::select_values(condition.data(), left.data(), right.data(), output.data(),
                    kHidden);
  highway_select_peer(condition.data(), left.data(), right.data(),
                      peer_output.data(), kHidden);
  for (std::size_t i = 0; i < kHidden; ++i)
    if (!same_value(output[i], peer_output[i]))
      return 4;

  const float hyle_min = hs::min_value(right.data(), kHidden);
  const float peer_min = highway_min_value_peer(right.data(), kHidden);
  if (!same_value(hyle_min, peer_min))
    return 5;

  nb::Bench rsqrt;
  rsqrt.title("compute dense reciprocal-sqrt f32 N=4096")
      .relative(true)
      .batch(1)
      .minEpochTime(std::chrono::milliseconds(8));
  run_pair(
      rsqrt, peer_first, "HYLE reciprocal-sqrt f32 N=4096",
      [&] {
        hs::reciprocal_sqrt(left.data(), output.data(), kHidden);
        nb::doNotOptimizeAway(output.data());
      },
      "Highway reciprocal-sqrt f32 N=4096",
      [&] {
        highway_reciprocal_sqrt_peer(left.data(), output.data(), kHidden);
        nb::doNotOptimizeAway(output.data());
      });

  nb::Bench minimum;
  minimum.title("compute dense number-min f32 N=11008")
      .relative(true)
      .batch(1)
      .minEpochTime(std::chrono::milliseconds(8));
  run_pair(
      minimum, peer_first, "HYLE number-min f32 N=11008",
      [&] {
        hs::fmin(left.data(), right.data(), output.data(), kFfn);
        nb::doNotOptimizeAway(output.data());
      },
      "Highway number-min f32 N=11008",
      [&] {
        highway_fmin_peer(left.data(), right.data(), output.data(), kFfn);
        nb::doNotOptimizeAway(output.data());
      });

  nb::Bench select;
  select.title("compute dense byte-bool select f32 N=4096")
      .relative(true)
      .batch(1)
      .minEpochTime(std::chrono::milliseconds(8));
  run_pair(
      select, peer_first, "HYLE byte-bool select f32 N=4096",
      [&] {
        hs::select_values(condition.data(), left.data(), right.data(),
                          output.data(), kHidden);
        nb::doNotOptimizeAway(output.data());
      },
      "Highway byte-bool select f32 N=4096",
      [&] {
        highway_select_peer(condition.data(), left.data(), right.data(),
                            output.data(), kHidden);
        nb::doNotOptimizeAway(output.data());
      });

  nb::Bench reduction;
  reduction.title("compute dense propagating-min f32 N=4096")
      .relative(true)
      .batch(1)
      .minEpochTime(std::chrono::milliseconds(8));
  run_pair(
      reduction, peer_first, "HYLE propagating-min f32 N=4096",
      [&] { nb::doNotOptimizeAway(hs::min_value(right.data(), kHidden)); },
      "Highway propagating-min f32 N=4096",
      [&] {
        nb::doNotOptimizeAway(highway_min_value_peer(right.data(), kHidden));
      });
}
