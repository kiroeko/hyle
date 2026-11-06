#include "../../engine/inference/mtp_policy.h"

#include <array>
#include <cstdio>
#include <stdexcept>

int main() {
  namespace m = hyle::inference::mtp;
  std::size_t checks = 0;
  constexpr std::array capacities{std::size_t{3}, std::size_t{4}, std::size_t{32768}, std::size_t{32769}};
  constexpr std::array counts{std::size_t{0}, std::size_t{1}, std::size_t{2}};
  constexpr std::array drafts{std::size_t{0}, std::size_t{1}, std::size_t{3}, std::size_t{4}};
  constexpr std::array modes{m::RequestedMode::kOff, m::RequestedMode::kAuto, m::RequestedMode::kOn};
  for (unsigned bits = 0; bits < (1u << 11); ++bits)
    for (const auto capacity : capacities)
      for (const auto sessions : counts)
        for (const auto active : counts)
          for (const auto depth : drafts)
            for (const auto mode : modes) {
    const auto flag = [bits](unsigned bit) { return (bits & (1u << bit)) != 0; };
    const m::DefaultRequest request{mode, flag(0), flag(1), flag(2), flag(3), flag(4),
        sessions, active, capacity, flag(5), flag(6), flag(7), flag(8),
        depth, flag(9), flag(10)};
    const bool inspect = mode != m::RequestedMode::kOff && depth > 0 && request.default_backend &&
        request.greedy && request.automatic_head && request.automatic_vocabulary;
    if (m::inspect_prepared_vocabulary(request) != inspect)
      throw std::runtime_error("prepared asset inspection changed");
    for (const bool exists : {false, true}) {
      const bool prepared = inspect && exists;
      const bool route = request.natural_generation && !request.multiple_schedules &&
          !request.prefix_enabled && !request.scheduler_progress && !request.warmup_planned_shapes &&
          sessions <= 1 && active <= 1 && capacity >= 4 && capacity <= 32768 &&
          request.default_backend && !request.cpu_ffn && request.greedy && request.default_replay;
      const bool automatic = prepared && route && depth <= 3;
      const bool enabled = mode == m::RequestedMode::kOn || automatic;
      const auto selected = m::select_default(request, exists);
      if (selected.automatic != automatic || selected.enabled != enabled ||
          selected.prepared_assets != (enabled && prepared))
        throw std::runtime_error("cold MTP policy changed an existing default or fallback");
      ++checks;
    }
  }
  std::printf("MTP cold policy: %zu configuration checks passed\n", checks);
}
