// Qualified SIMD inference-row replay. Each pair has the same storage,
// arithmetic, output, ISA and allocation contract; only the public HYLE route
// versus an independently spelled current-provider/consumer algorithm differs.
#include <nanobench.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <utility>
#include <vector>

#include "hyle/core/compiler.h"
#include "hyle/simd/math.h"
#include "hyle/simd/sort.h"

#include "../common/perf_process.h"

#define HWY_COMPILE_ONLY_STATIC 1
#include "hwy/contrib/math/math-inl.h"
#include "hwy/highway.h"
#undef HWY_COMPILE_ONLY_STATIC

namespace {
namespace nb = ankerl::nanobench;
namespace hn = hwy::HWY_NAMESPACE;
namespace hs = hyle::simd;

constexpr std::size_t kHidden = 4096;
constexpr std::size_t kFfn = 11008;
constexpr std::size_t kExperts = 256;
constexpr std::size_t kSelectedExperts = 6;
constexpr float kEpsilon = 1e-5f;

bool environment_present(const char* name) {
#if defined(_WIN32)
  char* value = nullptr;
  std::size_t size = 0;
  const bool present = _dupenv_s(&value, &size, name) == 0 && value != nullptr;
  std::free(value);
  return present;
#else
  return std::getenv(name) != nullptr;
#endif
}

template <class Candidate, class Peer>
void run_pair(nb::Bench& bench, bool peer_first,
              std::string_view candidate_name, Candidate&& candidate,
              std::string_view peer_name, Peer&& peer) {
  if (peer_first) {
    bench.run(peer_name, std::forward<Peer>(peer));
    bench.run(candidate_name, std::forward<Candidate>(candidate));
  } else {
    bench.run(candidate_name, std::forward<Candidate>(candidate));
    bench.run(peer_name, std::forward<Peer>(peer));
  }
}

HYLE_NOINLINE void highway_swiglu_peer(const float* gate, const float* value,
                                       float* dst, std::size_t n) noexcept {
  const hn::ScalableTag<float> d;
  const std::size_t width = hn::Lanes(d);
  std::size_t i = 0;
  for (; width <= n - i; i += width) {
    const auto x = hn::LoadU(d, gate + i);
    const auto linear = hn::LoadU(d, value + i);
    const auto one = hn::Set(d, 1.0f);
    const auto e = hn::Exp(d, hn::Neg(hn::Abs(x)));
    const auto denominator = hn::Add(one, e);
    const auto sigmoid =
        hn::IfThenElse(hn::Lt(x, hn::Zero(d)), hn::Div(e, denominator),
                       hn::Div(one, denominator));
    auto activated = hn::Mul(x, sigmoid);
    activated = hn::IfThenElse(hn::And(hn::IsInf(x), hn::Lt(x, hn::Zero(d))),
                               hn::Zero(d), activated);
    hn::StoreU(hn::Mul(activated, linear), d, dst + i);
  }
  for (; i < n; ++i) {
    const float x = gate[i];
    const float e = std::exp(-std::abs(x));
    const float sigmoid = x < 0.0f ? e / (1.0f + e) : 1.0f / (1.0f + e);
    dst[i] = x * sigmoid * value[i];
  }
}

HYLE_NOINLINE void highway_rms_norm_peer(const float* src, const float* weight,
                                         float* dst, std::size_t n,
                                         float epsilon) noexcept {
  const hn::ScalableTag<float> d;
  const std::size_t width = hn::Lanes(d);
  auto sum_squares_v = hn::Zero(d);
  std::size_t i = 0;
  for (; width <= n - i; i += width) {
    const auto value = hn::LoadU(d, src + i);
    sum_squares_v = hn::MulAdd(value, value, sum_squares_v);
  }
  float sum_squares = hn::ReduceSum(d, sum_squares_v);
  for (; i < n; ++i) sum_squares = std::fma(src[i], src[i], sum_squares);
  const float inverse =
      1.0f / std::sqrt(sum_squares / static_cast<float>(n) + epsilon);
  const auto inverse_v = hn::Set(d, inverse);
  i = 0;
  for (; width <= n - i; i += width) {
    hn::StoreU(hn::Mul(hn::Mul(hn::LoadU(d, src + i), inverse_v),
                       hn::LoadU(d, weight + i)),
               d, dst + i);
  }
  for (; i < n; ++i) dst[i] = src[i] * inverse * weight[i];
}

struct BetterScoredIndex {
  bool operator()(const hs::ScoredIndexF32& lhs,
                  const hs::ScoredIndexF32& rhs) const noexcept {
    const bool lhs_nan = std::isnan(lhs.score);
    const bool rhs_nan = std::isnan(rhs.score);
    if (lhs_nan != rhs_nan) return !lhs_nan;
    if (!lhs_nan) {
      if (lhs.score > rhs.score) return true;
      if (lhs.score < rhs.score) return false;
    }
    return lhs.index < rhs.index;
  }
};

// Source-equivalent to ds4's bounded expert-selection insertion algorithm,
// adapted only to HYLE's stronger score+index and deterministic tie contract.
HYLE_NOINLINE void ds4_small_top_k_peer(const float* scores, std::size_t n,
                                        hs::ScoredIndexF32* result,
                                        std::size_t k) noexcept {
  const BetterScoredIndex better;
  std::size_t retained = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const hs::ScoredIndexF32 candidate{scores[i],
                                       static_cast<std::uint32_t>(i)};
    if (retained == k && !better(candidate, result[k - 1])) continue;
    std::size_t position = retained < k ? retained++ : k - 1;
    while (position > 0 && better(candidate, result[position - 1])) {
      result[position] = result[position - 1];
      --position;
    }
    result[position] = candidate;
  }
}

bool same_result(const float* lhs, const float* rhs, std::size_t n,
                 float tolerance) {
  for (std::size_t i = 0; i < n; ++i) {
    if (std::abs(lhs[i] - rhs[i]) > tolerance + tolerance * std::abs(rhs[i]))
      return false;
  }
  return true;
}

}  // namespace

int main() {
  const auto qos =
      hyle::bench::perf::configure_thread_qos("HYLE_BENCH_APPLE_QOS");
  const auto affinity =
      hyle::bench::perf::configure_thread_affinity("HYLE_PERF_CALLER_CPU");
  const bool peer_first = environment_present("HYLE_BENCH_RIVAL_FIRST");
  std::fprintf(stderr,
               "# HYLE_SIMD_INFERENCE_RUN execution_order=%s apple_qos=%s "
               "qos_ok=%s affinity_topology=%s caller_cpu=%s affinity_ok=%s "
               "run_nonce=%s\n",
               peer_first ? "peer-first" : "candidate-first", qos.name, qos.ok,
               affinity.topology, affinity.cpu, affinity.ok,
               hyle::bench::perf::run_nonce());

  std::vector<float> gate(kFfn), value(kFfn), weight(kHidden);
  std::vector<float> output(kFfn), peer_output(kFfn);
  for (std::size_t i = 0; i < gate.size(); ++i) {
    gate[i] = 7.0f * std::sin(static_cast<float>(i) * 0.0137f);
    value[i] = 2.0f * std::cos(static_cast<float>(i) * 0.0191f);
  }
  for (std::size_t i = 0; i < weight.size(); ++i)
    weight[i] = 0.75f + static_cast<float>(i % 127) / 254.0f;

  hs::swiglu_f32_batch(gate.data(), value.data(), output.data(), kFfn);
  highway_swiglu_peer(gate.data(), value.data(), peer_output.data(), kFfn);
  if (!same_result(output.data(), peer_output.data(), kFfn, 2e-6f)) return 2;

  hs::rms_norm_f32(gate.data(), weight.data(), output.data(), kHidden,
                   kEpsilon);
  highway_rms_norm_peer(gate.data(), weight.data(), peer_output.data(), kHidden,
                        kEpsilon);
  if (!same_result(output.data(), peer_output.data(), kHidden, 3e-5f)) return 3;

  std::array<float, kExperts> scores{};
  for (std::size_t i = 0; i < scores.size(); ++i) {
    scores[i] = std::sin(static_cast<float>(i) * 0.131f) +
                0.01f * static_cast<float>(i % 17);
  }
  std::array<hs::ScoredIndexF32, kSelectedExperts> selected{};
  std::array<hs::ScoredIndexF32, kSelectedExperts> peer_selected{};
  hs::top_k_f32(scores.data(), scores.size(), selected.data(), selected.size());
  ds4_small_top_k_peer(scores.data(), scores.size(), peer_selected.data(),
                       peer_selected.size());
  for (std::size_t i = 0; i < selected.size(); ++i) {
    if (selected[i].score != peer_selected[i].score ||
        selected[i].index != peer_selected[i].index)
      return 4;
  }

  nb::Bench swiglu;
  swiglu.title("inference split SwiGLU f32 N=11008")
      .relative(true)
      .batch(1)
      .minEpochTime(std::chrono::milliseconds(8));
  run_pair(
      swiglu, peer_first, "HYLE split SwiGLU f32 N=11008",
      [&] {
        hs::swiglu_f32_batch(gate.data(), value.data(), output.data(), kFfn);
        nb::doNotOptimizeAway(output.data());
      },
      "Highway split SwiGLU f32 N=11008",
      [&] {
        highway_swiglu_peer(gate.data(), value.data(), output.data(), kFfn);
        nb::doNotOptimizeAway(output.data());
      });

  nb::Bench rms_norm;
  rms_norm.title("inference weighted RMSNorm f32 N=4096")
      .relative(true)
      .batch(1)
      .minEpochTime(std::chrono::milliseconds(8));
  run_pair(
      rms_norm, peer_first, "HYLE weighted RMSNorm f32 N=4096",
      [&] {
        hs::rms_norm_f32(gate.data(), weight.data(), output.data(), kHidden,
                         kEpsilon);
        nb::doNotOptimizeAway(output.data());
      },
      "Highway weighted RMSNorm f32 N=4096",
      [&] {
        highway_rms_norm_peer(gate.data(), weight.data(), output.data(),
                              kHidden, kEpsilon);
        nb::doNotOptimizeAway(output.data());
      });

  nb::Bench top_k;
  top_k.title("inference expert top-k f32 N=256 K=6")
      .relative(true)
      .batch(1)
      .minEpochTime(std::chrono::milliseconds(8));
  run_pair(
      top_k, peer_first, "HYLE expert top-k f32 N=256 K=6",
      [&] {
        hs::top_k_f32(scores.data(), scores.size(), selected.data(),
                      selected.size());
        nb::doNotOptimizeAway(selected.data());
      },
      "ds4-shape expert top-k f32 N=256 K=6",
      [&] {
        ds4_small_top_k_peer(scores.data(), scores.size(), selected.data(),
                             selected.size());
        nb::doNotOptimizeAway(selected.data());
      });
}
