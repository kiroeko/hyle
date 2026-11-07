#pragma once

#include "cold_workload.h"

#include <cstddef>
#include <string_view>

namespace hyle::inference::mtp {

enum class RequestedMode { kOff, kAuto, kOn };

// Cold configuration for the current prepared default. This policy does not
// assert model or backend support; the selected source is validated separately.
struct DefaultRequest final {
  RequestedMode mode = RequestedMode::kAuto;
  ColdWorkloadFacts workload;
  bool automatic_backend_eligible = false;
  std::size_t max_drafts = 0;
  bool automatic_vocabulary = false;

  constexpr bool unmodified_greedy_drafting() const noexcept {
    return workload.greedy && workload.zero_draft_threshold;
  }
};

// Avoid inspecting an unused artifact, including errors from its filesystem.
constexpr bool inspect_prepared_vocabulary(const DefaultRequest &request) noexcept {
  return request.mode != RequestedMode::kOff && request.max_drafts > 0 &&
      request.automatic_backend_eligible && request.unmodified_greedy_drafting() &&
      request.automatic_vocabulary;
}

struct DefaultSelection final {
  bool automatic = false;
  bool enabled = false;
  bool prepared_assets = false;
  std::string_view reason;
};

struct DraftDepthSelection final {
  std::size_t max_drafts = 0;
  std::string_view source;
};

// A validated artifact may recommend a depth; an explicit request wins.
// The model and execution plan validate or limit the chosen depth later.
constexpr DraftDepthSelection select_draft_depth(const DefaultSelection &selection,
    bool explicit_depth, std::size_t requested_depth, std::size_t recommended_depth) noexcept {
  if (!selection.enabled) return {requested_depth, "disabled"};
  if (explicit_depth) return {requested_depth, "explicit"};
  if (recommended_depth) return {recommended_depth, "artifact_recommendation"};
  return {requested_depth, "configured_default"};
}

// One cold policy decision supplies both admission and its explanation.
// Asset inspection is separate because explicit MTP can use prepared assets
// even when the automatic route is not selected.
constexpr std::string_view automatic_constraint(const DefaultRequest &request) noexcept {
  const auto &workload = request.workload;
  if (!request.max_drafts) return "drafting_disabled";
  if (!workload.natural_generation) return "non_natural_request";
  if (!request.automatic_backend_eligible) return "backend_policy";
  if (!workload.greedy || !workload.zero_draft_threshold) return "sampling_or_draft_threshold";
  if (workload.multiple_schedules || workload.max_sessions > 1 || workload.active_sequences > 1)
    return "multiple_sessions_or_sequences";
  if (workload.prefix_enabled) return "prefix_cache";
  if (workload.scheduler_progress) return "scheduler";
  if (workload.cpu_assist) return "cpu_ffn";
  // A draft needs at least one input row and one remaining position.
  if (workload.capacity < 2) return "insufficient_capacity";
  return {};
}

constexpr DefaultSelection select_default(const DefaultRequest &request,
    bool prepared_vocabulary_exists, bool original_projection_eligible = false,
    bool explicit_asset_selection = false) noexcept {
  const bool prepared = inspect_prepared_vocabulary(request) && prepared_vocabulary_exists;
  const bool available = prepared || original_projection_eligible || explicit_asset_selection;
  const auto constraint = automatic_constraint(request);
  const bool automatic = request.mode != RequestedMode::kOff && available && constraint.empty();
  const bool enabled = request.mode == RequestedMode::kOn || automatic;
  const auto reason = request.mode == RequestedMode::kOff ? "explicit_off" :
      request.mode == RequestedMode::kOn ? "explicit_on" :
      !constraint.empty() ? constraint : !automatic ? "no_eligible_source" :
      explicit_asset_selection ? "explicit_draft_asset" :
      prepared ? "prepared_vocabulary" : "original_projection";
  return {automatic, enabled, enabled && prepared, reason};
}

} // namespace hyle::inference::mtp
