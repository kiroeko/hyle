#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>

#include "hyle/error/result.h"
#include "hyle/platform/visibility.h"

namespace hyle::platform {

// One live host-memory snapshot for allocation and placement planning. Physical
// values are always present on success. Optional values are absent when the OS
// cannot expose a trustworthy process/container view; zero remains a valid
// observed amount rather than doubling as an "unknown" sentinel.
struct HostMemoryInfo {
  std::uint64_t physical_total_bytes = 0;
  std::uint64_t physical_available_bytes = 0;
  std::optional<std::uint64_t> commit_limit_bytes;
  std::optional<std::uint64_t> commit_available_bytes;
  std::optional<std::uint64_t> swap_total_bytes;
  std::optional<std::uint64_t> swap_available_bytes;
  std::optional<std::uint64_t> process_limit_bytes;
  std::optional<std::uint64_t> process_available_bytes;

  // Conservative bytes a host allocation planner may still request now. This
  // is host capacity only: a Compute backend must combine it with its dynamic
  // device/UMA budget instead of treating the two as independent pools.
  std::uint64_t effective_available_bytes() const noexcept {
    std::uint64_t value = physical_available_bytes;
    if (commit_available_bytes)
      value = (std::min)(value, *commit_available_bytes);
    if (process_limit_bytes) value = (std::min)(value, *process_limit_bytes);
    if (process_available_bytes)
      value = (std::min)(value, *process_available_bytes);
    return value;
  }

  bool constrained() const noexcept {
    return effective_available_bytes() < physical_available_bytes;
  }
};

// One uncached whole-process memory snapshot. Resident values include every
// allocator and mapping domain in the process. Private committed values are
// present only when the OS exposes that exact concept; virtual address size,
// host-wide commit and an individual allocator's committed bytes are not
// substitutes for an absent field.
struct ProcessMemoryInfo {
  std::uint64_t resident_bytes = 0;
  std::uint64_t peak_resident_bytes = 0;
  std::optional<std::uint64_t> private_committed_bytes;
  std::optional<std::uint64_t> peak_private_committed_bytes;
};

// Dynamic, fallible and intentionally uncached: physical availability and
// cgroup/process limits may change between calls. Windows leaves process_*
// empty because documented APIs cannot fold an unknown nested Job chain. The
// snapshot is internally coherent enough for planning but is not a reservation;
// allocation may still race another process and fail.
HYLE_PLATFORM_API Result<HostMemoryInfo> host_memory_info() noexcept;

// Dynamic, fallible and intentionally uncached. This is a cold diagnostic and
// planning query; it adds no state or work to allocation and mapping hot paths.
HYLE_PLATFORM_API Result<ProcessMemoryInfo> process_memory_info() noexcept;

}  // namespace hyle::platform
