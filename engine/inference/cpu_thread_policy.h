#pragma once

#include <algorithm>
#include <cstdint>
#include <span>

#include "hyle/platform/cpu.h"

namespace hyle::inference {

// Cold engine default for ggml's existing Context-owned native team. Explicit
// caller options keep their original budget. Available classes are disjoint;
// a known physical count avoids charging SMT siblings as separate cores.
inline std::uint32_t select_cpu_native_threads(
    std::span<const platform::CpuCacheClassInfo> classes,
    const platform::ProcessCpuCapacity &capacity,
    std::uint32_t native_limit) noexcept {
  const auto ceiling = (std::max)(std::size_t{1}, (std::min)(
      capacity.parallelism_ceiling(), std::size_t{native_limit}));
  std::size_t all = 0, performance = 0;
  for (const auto &group : classes) {
    const auto cores = group.physical_cores ? group.physical_cores : group.logical_cpus;
    all += (std::min)(std::size_t{cores}, ceiling - all);
    if (group.kind == platform::CpuKind::kPerformance)
      performance += (std::min)(std::size_t{cores}, ceiling - performance);
  }
  return static_cast<std::uint32_t>(performance ? performance : all ? all : 1);
}

} // namespace hyle::inference
