#pragma once

#include "p1_common.h"

#include <cstddef>
#include <cstdint>

namespace hyle::bench::sched_pinned_task_p1 {

inline constexpr std::size_t kWarmupEpisodes = 256;
inline constexpr std::size_t kMeasuredEpisodes = 4'096;

constexpr std::uint64_t episode_value(unsigned worker,
                                      std::size_t episode) noexcept {
  return (static_cast<std::uint64_t>(worker) + 1) *
         (static_cast<std::uint64_t>(episode) + 0x9e3779b9ULL);
}

constexpr std::uint64_t expected_checksum() noexcept {
  std::uint64_t result = 0;
  for (unsigned worker = 0;
       worker < hyle::bench::sched_p1::kWorkerCount; ++worker) {
    for (std::size_t episode = 0; episode < kMeasuredEpisodes; ++episode) {
      result += episode_value(worker, episode);
    }
  }
  return result;
}

}  // namespace hyle::bench::sched_pinned_task_p1
