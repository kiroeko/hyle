#pragma once

#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#include "../common/perf_process.h"

namespace hyle::bench::cuda {

[[noreturn]] inline void reject_host_policy(const char *message) {
  std::fprintf(stderr, "CUDA benchmark host policy: %s\n", message);
  std::exit(86);
}

struct HostPolicy {
  std::string mask = "not-applicable";
  const char *priority = "not-applicable";

  HostPolicy() { check(); }

  // Cold checks only. Native helpers retain SDK ownership and are not assigned
  // fabricated worker IDs or pinned by this benchmark.
  void check() {
#if defined(_WIN32)
    DWORD_PTR process_mask = 0, system_mask = 0;
    if (!::GetProcessAffinityMask(::GetCurrentProcess(), &process_mask, &system_mask))
      reject_host_policy("cannot read process affinity");
    char text[2 + sizeof(DWORD_PTR) * 2 + 1]{};
    std::snprintf(text, sizeof(text), "0x%llx",
                  static_cast<unsigned long long>(process_mask));
    mask = text;
    const auto priority_class = ::GetPriorityClass(::GetCurrentProcess());
    priority = priority_class == ABOVE_NORMAL_PRIORITY_CLASS ? "above-normal"
               : priority_class == NORMAL_PRIORITY_CLASS ? "normal" : "other";
    const char *topology = std::getenv("HYLE_PERF_SAMPLING_TOPOLOGY");
    if (topology == nullptr) return;
    const char *expected_mask = std::getenv("HYLE_PERF_PARENT_MASK");
    const char *expected_priority = std::getenv("HYLE_PERF_PRIORITY");
    const char *caller = std::getenv("HYLE_PERF_CALLER_CPU");
    if (std::strcmp(topology, "native-runtime-pair") != 0 || !expected_mask ||
        !expected_priority || !caller || mask != expected_mask ||
        std::strcmp(priority, expected_priority) != 0 ||
        std::popcount(static_cast<unsigned long long>(process_mask)) != 2)
      reject_host_policy("native runtime CPU budget/priority does not match");
    char *end = nullptr;
    const auto cpu = std::strtoul(caller, &end, 10);
    if (end == caller || *end || cpu >= sizeof(DWORD_PTR) * 8 ||
        !(process_mask & (DWORD_PTR{1} << cpu)) ||
        ::GetCurrentProcessorNumber() != cpu)
      reject_host_policy("caller is not on its declared CPU");
#endif
  }
};

inline void require_aa_ready(bool aa_run, bool graph) {
#if defined(_WIN32)
  if (aa_run || std::getenv("HYLE_PERF_RUN_NONCE") == nullptr) return;
  wchar_t executable[32768]{};
  const auto size = ::GetModuleFileNameW(nullptr, executable, 32768);
  if (size == 0 || size >= 32768)
    reject_host_policy("cannot locate qualification receipt");
  const auto path = std::filesystem::path(executable).parent_path() /
                    (graph ? "hyle_cuda_graph_aa.ready" : "hyle_cuda_compute_aa.ready");
  std::ifstream input(path);
  std::string status;
  input >> status;
  if (status != "PASS")
    reject_host_policy("A/A qualification has not passed for this build");
#else
  (void)aa_run;
  (void)graph;
#endif
}

} // namespace hyle::bench::cuda
