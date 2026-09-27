#pragma once
// Complete, synchronous consumer tasks over the ordinary Direct and Graph APIs.
// Mathematical recipes/oracles come from the already qualified D5 fixtures.
#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif
#include <gtest/gtest.h>
#include <nanobench.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../tests/compute_models/d5_moe_execution_fragments.h"
#include "../../tests/compute_models/d5_qwen35_fixture.h"
#include "../common/perf_process.h"

namespace hyle_d5_perf {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace mem = hyle::mem;
namespace sem = d5_qwen38_24t_semantic_fragments;
namespace nb = ankerl::nanobench;
using Clock = std::chrono::steady_clock;
using Id = sem::ValueId;
using BF16 = hyle::simd::bfloat16_t;

void require(hyle::Status status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
template <class T> T take(hyle::Result<T> result) {
  if (!result)
    throw std::runtime_error(std::string(result.error().message()));
  return std::move(*result);
}
std::string environment(const char *name) {
#if defined(_WIN32)
  char *value = nullptr;
  std::size_t size = 0;
  if (_dupenv_s(&value, &size, name) != 0)
    throw std::runtime_error("environment read failed");
  std::string result = value == nullptr ? "" : value;
  std::free(value);
  return result;
#else
  const auto *value = std::getenv(name);
  return value == nullptr ? "" : value;
#endif
}

sem::Fragment qwen_ffn() {
  const auto root = d5_qwen35_fixture::root();
  if (!root)
    throw std::runtime_error("HYLE_D5_QWEN35_FIXTURE is required; no skip");
  auto source = d5_qwen35_fixture::load_ffn(*root);
  sem::Fragment f{.profile = &workload_test::kModel, .region = 2};
  constexpr auto h = d5_qwen35_fixture::kHidden;
  constexpr auto n = d5_qwen35_fixture::kFfnHidden;
  const auto type = c::DataType::kBFloat16;
  auto x = f.input(sem::dense({1, h}, type));
  auto scale = f.constant(sem::dense({h}, type), std::move(source.norm_scale));
  auto wg = f.constant(take(sem::dense({n, h}, type).transposed()),
                       std::move(source.weights[0]));
  auto wu = f.constant(take(sem::dense({n, h}, type).transposed()),
                       std::move(source.weights[1]));
  auto wd = f.constant(take(sem::dense({h, n}, type).transposed()),
                       std::move(source.weights[2]));
  auto normalized = f.one(c::Operation(c::RmsNorm{}), {x, scale});
  auto gate = f.one(c::Operation(c::MatMul{}), {normalized, wg});
  auto up = f.one(c::Operation(c::MatMul{}), {normalized, wu});
  auto activated = f.one(c::Operation(c::GatedActivation{
      .inputs = c::GatedInputKind::kSeparate}), {gate, up});
  f.outputs = {f.one(c::Operation(c::MatMul{}), {activated, wd})};
  for (std::size_t i = 0; i < source.activations.size(); ++i)
    f.replays.push_back({{std::move(source.activations[i])},
                         {std::move(source.direct_expected[i])}});
  return f;
}

sem::Fragment deepseek_moe() {
  auto f = d5_moe_execution_fragments::hash_moe(
      d5_moe_execution_cases::kCases[5]);
  // The consumer publishes its full result, not the diagnostic intermediate
  // taps from the correctness suite. Both arms get exactly this same DAG.
  f.outputs = {f.outputs[5]};
  f.exact_outputs.clear();
  for (auto &replay : f.replays) {
    auto final = std::move(replay.outputs[5]);
    replay.outputs = {std::move(final)};
  }
  std::vector<bool> needed(f.descriptions.size());
  needed[f.outputs[0].ordinal()] = true;
  for (auto node = f.nodes.rbegin(); node != f.nodes.rend(); ++node)
    if (std::ranges::any_of(node->outputs,
                           [&](Id id) { return needed[id.ordinal()]; }))
      for (auto id : node->inputs)
        needed[id.ordinal()] = true;
  std::erase_if(f.nodes, [&](const auto &node) {
    return std::ranges::none_of(node.outputs,
                               [&](Id id) { return needed[id.ordinal()]; });
  });
  return f;
}

inline void verify_output(const sem::Fragment &fixture, std::size_t replay,
                          std::span<const std::byte> bytes, float qwen_absolute = 1e-8F) {
  std::vector<float> actual;
  if (fixture.description(fixture.outputs[0]).data_type() == c::DataType::kBFloat16) {
    actual.resize(bytes.size() / sizeof(BF16));
    for (std::size_t i = 0; i < actual.size(); ++i) {
      BF16 value;
      std::memcpy(&value, bytes.data() + i * sizeof(value), sizeof(value));
      actual[i] = static_cast<float>(half_test::decode(value));
    }
  } else {
    actual.resize(bytes.size() / sizeof(float));
    std::memcpy(actual.data(), bytes.data(), bytes.size());
  }
  const auto &expected = fixture.replays[replay].outputs[0];
  if (actual.size() != expected.size()) throw std::runtime_error("oracle size mismatch");
  for (float value : actual)
    if (!std::isfinite(value)) throw std::runtime_error("nonfinite result");
  if (fixture.profile == &workload_test::kModel) {
    for (std::size_t i = 0; i < actual.size(); ++i)
      if (std::abs(actual[i] - expected[i]) > std::abs(expected[i]) * .008F + qwen_absolute) {
        std::fprintf(stderr, "# QWEN_ORACLE replay=%zu index=%zu actual=%.10g expected=%.10g limit=%.10g\n",
                     replay, i, double(actual[i]), double(expected[i]),
                     double(std::abs(expected[i]) * .008F + qwen_absolute));
        throw std::runtime_error("Qwen BF16 FFN oracle mismatch");
      }
  } else {
    sem::expect_output(fixture, actual, expected);
    if (::testing::Test::HasFailure()) throw std::runtime_error("DeepSeek MoE oracle mismatch");
  }
}
template <class Route>
void measure(sem::Fragment fixture, const char *name, bool verify_only) {
  const auto aa = environment("HYLE_BENCH_AA_ROUTE");
  if (!aa.empty() && aa != "selected" && aa != "peer")
    throw std::runtime_error("invalid A/A route");
  const bool allocation_first = !environment("HYLE_BENCH_PEER_ALLOCATION_FIRST").empty();
  std::unique_ptr<Route> candidate, peer;
  if (allocation_first) peer = std::make_unique<Route>(fixture, aa == "selected");
  candidate = std::make_unique<Route>(fixture, aa != "peer");
  if (!allocation_first) peer = std::make_unique<Route>(fixture, aa == "selected");
  candidate->verify();
  peer->verify();
  std::fprintf(stderr, "# verify PASS profile=%s replays=%zu aa=%s\n", name,
               fixture.replays.size(), aa.empty() ? "off" : aa.c_str());
  if (verify_only) return;
  nb::Bench bench;
  bench.title(name).relative(true).minEpochTime(std::chrono::milliseconds(8));
  const auto selected = [&] { require(candidate->run()); };
  const auto reference = [&] { require(peer->run()); };
  if (!environment("HYLE_BENCH_RIVAL_FIRST").empty())
    (void)bench.compare("Direct complete fragment", reference, "Graph complete fragment", selected);
  else
    (void)bench.compare("Graph complete fragment", selected, "Direct complete fragment", reference);
  candidate->verify();
  peer->verify();
}
template <class Route> int run_main(int argc, char **argv) {
  try {
    ::testing::InitGoogleTest(&argc, argv);
    const bool verify = !environment("HYLE_BENCH_VERIFY_ONLY").empty();
    const bool managed = !environment("HYLE_PERF_RUN_NONCE").empty();
    const auto aa = environment("HYLE_BENCH_AA_ROUTE");
    if (!verify && !managed && environment("HYLE_D5_DIAGNOSTIC") != "1")
      throw std::runtime_error("sampling requires the instrument or explicit diagnostic mode");
    if (!verify && managed && aa.empty()) {
      std::string ready;
      std::ifstream(std::filesystem::absolute(argv[0]).parent_path() /
                    "d5-fragment-aa.ready") >> ready;
      if (ready != "PASS")
        throw std::runtime_error("this artifact has no passing workload A/A receipt");
    }
    const auto affinity = hyle::bench::perf::configure_thread_affinity("HYLE_PERF_CALLER_CPU");
#if defined(_WIN32)
    DWORD_PTR mask = 0, system_mask = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &mask, &system_mask))
      throw std::runtime_error("cannot read process affinity");
    const auto priority = GetPriorityClass(GetCurrentProcess());
    if (managed && GetCurrentProcessorNumber() != std::stoul(affinity.cpu))
      throw std::runtime_error("caller did not land on the declared CPU");
    std::fprintf(stderr,
        "# HYLE_D5_FRAGMENT_RUN execution_order=%s allocation_order=%s aa_route=%s "
        "paired_mode=nanobench-compare affinity_topology=%s caller_cpu=%s affinity_ok=%s "
        "parent_mask=0x%llx priority=%s run_nonce=%s verify_only=%d\n",
        environment("HYLE_BENCH_RIVAL_FIRST").empty() ? "candidate-first" : "peer-first",
        environment("HYLE_BENCH_PEER_ALLOCATION_FIRST").empty() ? "candidate-first" : "peer-first",
        aa.empty() ? "off" : aa.c_str(), affinity.topology, affinity.cpu, affinity.ok,
        static_cast<unsigned long long>(mask),
        priority == ABOVE_NORMAL_PRIORITY_CLASS ? "above-normal" : "other",
        hyle::bench::perf::run_nonce(), verify);
#else
    nb::doNotOptimizeAway(affinity);
#endif
    measure<Route>(qwen_ffn(), "Qwen3.5-2B official BF16 FFN M1 H2048 N6144", verify);
    measure<Route>(deepseek_moe(), "DeepSeek V4 Flash UD-Q4_K_XL hash-MoE H4096 N2048 top6", verify);
    return ::testing::Test::HasFailure() ? 1 : 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "D5 fragment benchmark failed: %s\n", error.what());
    return 1;
  }
}

} // namespace hyle_d5_perf
