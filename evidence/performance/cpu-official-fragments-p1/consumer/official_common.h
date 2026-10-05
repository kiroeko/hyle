#pragma once
// Reuse the existing D5 nanobench measurement and machine policy unchanged.
#include "../../../../bench/graph_route_cpu/d5_common.h"
#include "../../../../tests/compute_models/d5_grouped_fp8_fragments.h"
#include "../../../../tests/compute_models/d5_deepseek_official_fragments.h"
namespace hyle_d5_perf {
inline sem::Fragment publish_result(sem::Fragment f, std::size_t port) {
  const bool bfloat = f.bfloat_chain_output(port);
  f.outputs = {f.outputs.at(port)};
  f.exact_outputs.clear();
  f.bfloat_chain_outputs = bfloat ? std::vector<std::size_t>{0}
                                 : std::vector<std::size_t>{};
  for (auto &replay : f.replays) {
    auto output = std::move(replay.outputs.at(port));
    replay.outputs = {std::move(output)};
  }
  // Publish the complete consumer result; diagnostic taps do not change the DAG.
  std::vector<bool> needed(f.descriptions.size());
  needed[f.outputs[0].ordinal()] = true;
  for (auto node = f.nodes.rbegin(); node != f.nodes.rend(); ++node)
    if (std::ranges::any_of(node->outputs,
                           [&](Id id) { return needed[id.ordinal()]; }))
      for (auto id : node->inputs) needed[id.ordinal()] = true;
  std::erase_if(f.nodes, [&](const auto &node) {
    return std::ranges::none_of(node.outputs,
                               [&](Id id) { return needed[id.ordinal()]; });
  });
  return f;
}
inline sem::Fragment glm_official() {
  return publish_result(d5_grouped_fp8_fragments::glm_moe(true), 7);
}
inline sem::Fragment deepseek_official() {
  return publish_result(d5_deepseek_official_fragments::moe_chain(
      d5_moe_execution_cases::kCases[4], false, nullptr, false), 5);
}
template <class Route> int run_official_main(int argc, char **argv) {
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
#elif defined(__APPLE__)
    const auto qos = hyle::bench::perf::configure_thread_qos();
    std::fprintf(stderr,
        "# HYLE_D5_FRAGMENT_RUN execution_order=%s allocation_order=%s aa_route=%s "
        "paired_mode=nanobench-compare apple_qos=%s qos_ok=%s run_nonce=%s verify_only=%d\n",
        environment("HYLE_BENCH_RIVAL_FIRST").empty() ? "candidate-first" : "peer-first",
        environment("HYLE_BENCH_PEER_ALLOCATION_FIRST").empty() ? "candidate-first" : "peer-first",
        aa.empty() ? "off" : aa.c_str(), qos.name, qos.ok,
        hyle::bench::perf::run_nonce(), verify);
    nb::doNotOptimizeAway(affinity);
#else
    nb::doNotOptimizeAway(affinity);
#endif
    measure<Route>(glm_official(), "GLM 5.3 Flash official FP8 MoE H4096 N2048 top8 cycle2", verify);
    measure<Route>(deepseek_official(), "DeepSeek V4 Flash official FP4 FP8 MoE H4096 N2048 top6 cycle3", verify);
    return ::testing::Test::HasFailure() ? 1 : 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "D5 fragment benchmark failed: %s\n", error.what());
    return 1;
  }
}


} // namespace hyle_d5_perf
