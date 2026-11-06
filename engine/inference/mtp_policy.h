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
};

constexpr DefaultSelection select_default(const DefaultRequest &request,
    bool prepared_vocabulary_exists, bool original_projection_eligible = false) noexcept {
  const bool prepared = inspect_prepared_vocabulary(request) && prepared_vocabulary_exists;
  const bool available = prepared || (inspect_prepared_vocabulary(request) && original_projection_eligible);
  // Very short contexts are a default cost policy, not a model capability:
  // explicit MTP remains available and the execution path clips its depth.
  const bool automatic = available && request.natural_generation && !request.multiple_schedules &&
      !request.prefix_enabled && !request.scheduler_progress &&
      request.max_sessions <= 1 && request.active_sequences <= 1 && request.capacity >= 4 &&
      !request.cpu_ffn && request.replay_supports_chain &&
      request.max_drafts <= request.maximum_automatic_drafts;
  const bool enabled = request.mode == RequestedMode::kOn || automatic;
  return {automatic, enabled, enabled && prepared};
}

// Report the first cold configuration constraint, not a per-token decision.
constexpr std::string_view selection_reason(const DefaultRequest &request,
    const DefaultSelection &selection) noexcept {
  if (request.mode == RequestedMode::kOff) return "explicit_off";
  if (request.mode == RequestedMode::kOn) return "explicit_on";
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
  // This reason describes the automatic policy, not a backend rejection.
  if (request.capacity < 4) return "insufficient_capacity";
  if (!request.max_drafts || request.max_drafts > request.maximum_automatic_drafts)
    return "unsupported_draft_depth";
  if (!selection.automatic) return "no_eligible_source";
  return selection.prepared_assets ? "prepared_vocabulary" : "original_projection";
}

} // namespace hyle::inference::mtp
