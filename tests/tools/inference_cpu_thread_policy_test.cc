#include "../../engine/inference/cpu_thread_policy.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>

int main() {
  using namespace hyle;
  using platform::CpuCacheClassInfo;
  using platform::CpuKind;
  using platform::ProcessCpuCapacity;
  const auto check = [](bool value) { if (!value) std::abort(); };
  const std::array mixed{
      CpuCacheClassInfo{.kind = CpuKind::kPerformance, .logical_cpus = 16, .physical_cores = 8},
      CpuCacheClassInfo{.kind = CpuKind::kEfficiency, .logical_cpus = 16, .physical_cores = 16}};
  check(inference::select_cpu_native_threads(mixed, {32}, 16) == 8);
  check(inference::select_cpu_native_threads(mixed, {3}, 16) == 3);
  check(inference::select_cpu_native_threads(mixed, {32, platform::CpuQuota{150000, 100000}}, 16) == 2);
  check(inference::select_cpu_native_threads(mixed, {32}, 4) == 4);
  const std::array uniform{CpuCacheClassInfo{.logical_cpus = 16, .physical_cores = 8}};
  check(inference::select_cpu_native_threads(uniform, {16}, 16) == 8);
  const std::array unknown{CpuCacheClassInfo{.logical_cpus = 6}};
  check(inference::select_cpu_native_threads(unknown, {6}, 16) == 6);
  check(inference::select_cpu_native_threads({}, {32}, 16) == 1);
  check(inference::select_cpu_native_threads(mixed, {}, 16) == 1);
  check(inference::select_cpu_native_threads(mixed, {32}, 1) == 1);
  const std::array huge{CpuCacheClassInfo{.logical_cpus = UINT32_MAX},
                        CpuCacheClassInfo{.logical_cpus = UINT32_MAX}};
  check(inference::select_cpu_native_threads(huge, {(std::numeric_limits<std::size_t>::max)()}, 512) == 512);
  std::cout << "PASS CPU native thread policy: physical cores, P/E, quota, unknown topology and overflow\n";
}
