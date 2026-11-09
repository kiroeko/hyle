#pragma once

#include "cold_workload.h"
#include "route_reason.h"

#include <cstddef>

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
  // A provider's measured default preference, not an execution capability.
  // Explicit requests remain subject to the provider's real validation.
  RouteReason automatic_preference = RouteReason::kNone;

  constexpr bool unmodified_greedy_drafting() const noexcept {
    return workload.greedy;
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
  RouteReason reason = RouteReason::kNone;
};

struct DraftDepthSelection final {
  std::size_t max_drafts = 0;
  RouteReason source = RouteReason::kNone;
};

// A validated artifact may recommend a depth; an explicit request wins.
// The model and execution plan validate or limit the chosen depth later.
constexpr DraftDepthSelection select_draft_depth(const DefaultSelection &selection,
    bool explicit_depth, std::size_t requested_depth, std::size_t recommended_depth) noexcept {
  if (!selection.enabled) return {requested_depth, RouteReason::kDisabled};
  if (explicit_depth) return {requested_depth, RouteReason::kExplicit};
  if (recommended_depth) return {recommended_depth, RouteReason::kArtifactRecommendation};
  return {requested_depth, RouteReason::kConfiguredDefault};
}

// One cold policy decision supplies both admission and its explanation.
// Asset inspection is separate because explicit MTP can use prepared assets
// even when the automatic route is not selected.
constexpr RouteReason automatic_constraint(const DefaultRequest &request) noexcept {
  const auto &workload = request.workload;
  if (!request.max_drafts) return RouteReason::kDraftingDisabled;
  if (!workload.natural_generation) return RouteReason::kNonNaturalRequest;
  if (!request.automatic_backend_eligible) return RouteReason::kBackendPolicy;
  if (!workload.greedy) return RouteReason::kSampling;
  if (request.automatic_preference != RouteReason::kNone) return request.automatic_preference;
  if (workload.prefix_enabled) return RouteReason::kPrefixCache;
  if (workload.scheduler_progress) return RouteReason::kScheduler;
  if (workload.cpu_assist) return RouteReason::kCpuAssist;
  // A draft needs at least one input row and one remaining position.
  if (workload.capacity < 2) return RouteReason::kInsufficientCapacity;
  return RouteReason::kNone;
}

constexpr DefaultSelection select_default(const DefaultRequest &request,
    bool prepared_vocabulary_exists, bool original_projection_eligible = false,
    bool explicit_asset_selection = false) noexcept {
  const bool prepared = inspect_prepared_vocabulary(request) && prepared_vocabulary_exists;
  const bool available = prepared || original_projection_eligible || explicit_asset_selection;
  const auto constraint = automatic_constraint(request);
  const bool automatic = request.mode != RequestedMode::kOff && available && constraint == RouteReason::kNone;
  const bool enabled = request.mode == RequestedMode::kOn || automatic;
  const auto reason = request.mode == RequestedMode::kOff ? RouteReason::kExplicitOff :
      request.mode == RequestedMode::kOn ? RouteReason::kExplicitOn :
      constraint != RouteReason::kNone ? constraint : !automatic ? RouteReason::kNoEligibleSource :
      explicit_asset_selection ? RouteReason::kExplicitDraftAsset :
      prepared ? RouteReason::kPreparedVocabulary : RouteReason::kOriginalProjection;
  return {automatic, enabled, enabled && prepared, reason};
}

} // namespace hyle::inference::mtp
