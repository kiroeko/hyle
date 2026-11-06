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
  constexpr std::array limits{std::size_t{0}, std::size_t{1}, std::size_t{3}, std::size_t{4}};
  constexpr std::array modes{m::RequestedMode::kOff, m::RequestedMode::kAuto, m::RequestedMode::kOn};
  for (unsigned bits = 0; bits < (1u << 10); ++bits)
    for (const auto capacity : capacities)
      for (const auto sessions : counts)
        for (const auto active : counts)
          for (const auto depth : drafts)
            for (const auto limit : limits)
            for (const auto mode : modes) {
    const auto flag = [bits](unsigned bit) { return (bits & (1u << bit)) != 0; };
    const m::DefaultRequest request{mode, flag(0), flag(1), flag(2), flag(3),
        sessions, active, capacity, flag(4), flag(5), flag(6), flag(7),
        depth, limit, flag(8), flag(9)};
    const bool inspect = mode != m::RequestedMode::kOff && depth > 0 && request.default_backend &&
        request.greedy && request.automatic_head && request.automatic_vocabulary;
    if (m::inspect_prepared_vocabulary(request) != inspect)
      throw std::runtime_error("prepared asset inspection changed");
    for (const bool exists : {false, true}) for (const bool original : {false, true}) {
      const bool prepared = inspect && exists;
      const bool route = request.natural_generation && !request.multiple_schedules &&
          !request.prefix_enabled && !request.scheduler_progress &&
          sessions <= 1 && active <= 1 && capacity >= 4 &&
          request.default_backend && !request.cpu_ffn && request.greedy && request.replay_supports_chain;
      const bool automatic = (prepared || (inspect && original)) && route && depth <= limit;
      const bool enabled = mode == m::RequestedMode::kOn || automatic;
      const auto selected = m::select_default(request, exists, original);
      if (selected.automatic != automatic || selected.enabled != enabled ||
          selected.prepared_assets != (enabled && prepared))
        throw std::runtime_error("cold MTP policy changed an existing default or fallback");
      ++checks;
    }
  }
  m::DefaultRequest typical{.mode=m::RequestedMode::kAuto, .natural_generation=true,
      .capacity=32768, .default_backend=true, .greedy=true, .replay_supports_chain=true,
      .max_drafts=1, .maximum_automatic_drafts=3, .automatic_head=true, .automatic_vocabulary=true};
  if (m::selection_reason(typical, m::select_default(typical,false)) != "no_eligible_source" ||
      m::selection_reason(typical, m::select_default(typical,false,true)) != "original_projection" ||
      m::selection_reason(typical, m::select_default(typical,true)) != "prepared_vocabulary")
    throw std::runtime_error("automatic source reason differs");
  typical.greedy=false;
  if (m::selection_reason(typical, m::select_default(typical,true)) != "sampling_or_draft_threshold")
    throw std::runtime_error("sampling fallback reason differs");
  typical.greedy=true; typical.maximum_automatic_drafts=0;
  if (m::selection_reason(typical, m::select_default(typical,true)) != "unsupported_draft_depth")
    throw std::runtime_error("model chain capability reason differs");
  typical.maximum_automatic_drafts=3; typical.replay_supports_chain=false;
  if (m::selection_reason(typical, m::select_default(typical,true)) != "replay_commands")
    throw std::runtime_error("replay fallback reason differs");
  typical.mode=m::RequestedMode::kOff;
  if (m::selection_reason(typical, m::select_default(typical,true)) != "explicit_off")
    throw std::runtime_error("explicit off reason differs");
  std::printf("MTP cold policy: %zu configuration checks passed\n", checks);
}
