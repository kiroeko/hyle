#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <string_view>

#include "hyle/core/environment.h"
#include "hyle/platform/cpu.h"

#if HYLE_OS_LINUX && HYLE_ARCH_X64
#include <sys/utsname.h>
#endif

namespace hyle::inference {

inline bool uses_wsl2_guest_cpu_topology() noexcept {
#if HYLE_OS_LINUX && HYLE_ARCH_X64
  struct utsname kernel {};
  return ::uname(&kernel) == 0 &&
         std::string_view(kernel.release).ends_with("-microsoft-standard-WSL2");
#else
  return false;
#endif
}

// Cold engine default for ggml's existing Context-owned native team. Explicit
// caller options keep their original budget. Available classes are disjoint;
// a known physical count avoids charging SMT siblings as separate cores.
inline std::uint32_t select_cpu_native_threads(
    std::span<const platform::CpuCacheClassInfo> classes,
    const platform::ProcessCpuCapacity &capacity,
    std::uint32_t native_limit,
    bool prefer_compact_uniform_guest_team = false) noexcept {
  const auto ceiling = (std::max)(std::size_t{1}, (std::min)(
      capacity.parallelism_ceiling(), std::size_t{native_limit}));
  std::size_t all = 0, performance = 0;
  bool has_efficiency_class = false;
  for (const auto &group : classes) {
    const auto cores = group.physical_cores ? group.physical_cores : group.logical_cpus;
    all += (std::min)(std::size_t{cores}, ceiling - all);
    has_efficiency_class |= group.kind == platform::CpuKind::kEfficiency;
    if (group.kind == platform::CpuKind::kPerformance)
      performance += (std::min)(std::size_t{cores}, ceiling - performance);
  }
  auto selected = performance ? performance : all;
  // WSL2 can report the same CPU capacity for every virtual core, including
  // on a heterogeneous host. Use a compact team for CPU-only inference with
  // uniform guest capacity, within the original process budget.
  if (prefer_compact_uniform_guest_team && !has_efficiency_class)
    selected = (std::min)(selected, std::size_t{8});
  return static_cast<std::uint32_t>((std::max)(selected, std::size_t{1}));
}

} // namespace hyle::inference
