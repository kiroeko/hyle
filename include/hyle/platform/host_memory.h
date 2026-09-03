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
    if (commit_available_bytes) value = (std::min)(value, *commit_available_bytes);
    if (process_available_bytes) value = (std::min)(value, *process_available_bytes);
    return value;
  }

  bool constrained() const noexcept {
    return effective_available_bytes() < physical_available_bytes;
  }
};

// Dynamic, fallible and intentionally uncached: physical availability and
// cgroup/job/process limits may change between calls. The snapshot is internally
// coherent enough for planning but is not a reservation; allocation may still
// race another process and fail.
HYLE_PLATFORM_API Result<HostMemoryInfo> host_memory_info() noexcept;

}  // namespace hyle::platform
