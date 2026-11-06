#pragma once

#include <cstddef>
#include <string_view>

namespace hyle::inference::mtp {

enum class RequestedMode { kOff, kAuto, kOn };

// Cold configuration for the current prepared default. This policy does not
// assert model or backend support; the selected source is validated separately.
struct DefaultRequest final {
  RequestedMode mode = RequestedMode::kAuto;
  bool natural_generation = false;
  bool multiple_schedules = false;
  bool prefix_enabled = false;
  bool scheduler_progress = false;
  std::size_t max_sessions = 0;
  std::size_t active_sequences = 0;
  std::size_t capacity = 0;
  bool default_backend = false;
  bool cpu_ffn = false;
  bool greedy = false;
  bool replay_supports_chain = false;
  std::size_t max_drafts = 0;
  std::size_t maximum_automatic_drafts = 0;
  bool automatic_head = false;
  bool automatic_vocabulary = false;
};

// Avoid inspecting an unused artifact, including errors from its filesystem.
constexpr bool inspect_prepared_vocabulary(const DefaultRequest &request) noexcept {
  return request.mode != RequestedMode::kOff && request.max_drafts > 0 &&
      request.default_backend && request.greedy && request.automatic_head &&
      request.automatic_vocabulary;
}

struct DefaultSelection final {
  bool automatic = false;
  bool enabled = false;
  bool prepared_assets = false;
  std::string_view reason;
};

// One cold policy decision supplies both admission and its explanation.
// Asset inspection is separate because explicit MTP can use prepared assets
// even when the automatic route is not selected.
constexpr std::string_view automatic_constraint(const DefaultRequest &request) noexcept {
  if (!request.max_drafts) return "drafting_disabled";
  if (!request.natural_generation) return "non_natural_request";
  if (!request.default_backend) return "non_cuda_backend";
  if (!request.greedy) return "sampling_or_draft_threshold";
  if (!request.automatic_head || !request.automatic_vocabulary) return "manual_draft_asset";
  if (request.multiple_schedules || request.max_sessions > 1 || request.active_sequences > 1)
    return "multiple_sessions_or_sequences";
  if (request.prefix_enabled) return "prefix_cache";
  if (request.scheduler_progress) return "scheduler";
  if (request.cpu_ffn) return "cpu_ffn";
  if (!request.replay_supports_chain) return "replay_commands";
  // This is an automatic cost preference, not a backend rejection.
  if (request.capacity < 4) return "insufficient_capacity";
  if (request.max_drafts > request.maximum_automatic_drafts) return "unsupported_draft_depth";
  return {};
}

constexpr DefaultSelection select_default(const DefaultRequest &request,
    bool prepared_vocabulary_exists, bool original_projection_eligible = false) noexcept {
  const bool prepared = inspect_prepared_vocabulary(request) && prepared_vocabulary_exists;
  const bool available = prepared || (inspect_prepared_vocabulary(request) && original_projection_eligible);
  const auto constraint = automatic_constraint(request);
  const bool automatic = available && constraint.empty();
  const bool enabled = request.mode == RequestedMode::kOn || automatic;
  const auto reason = request.mode == RequestedMode::kOff ? "explicit_off" :
      request.mode == RequestedMode::kOn ? "explicit_on" :
      !constraint.empty() ? constraint : !automatic ? "no_eligible_source" :
      prepared ? "prepared_vocabulary" : "original_projection";
  return {automatic, enabled, enabled && prepared, reason};
}

} // namespace hyle::inference::mtp
