#pragma once

#include "route_reason.h"

#include <cstddef>

namespace hyle::inference::mtp {

enum class RequestedMode { kOff, kAuto, kOn };

// The model/provider decides which automatic route is appropriate and which
// sources are available. Explicit requests still undergo actual validation.
struct DefaultRequest final {
  RequestedMode mode = RequestedMode::kAuto;
  RouteReason automatic_refusal = RouteReason::kNone;
};

struct DefaultSourceReasons final {
  RouteReason prepared = RouteReason::kSelected;
  RouteReason fallback = RouteReason::kSelected;
  RouteReason explicit_source = RouteReason::kSelected;
};

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

// Selection is cold. Source inspection and automatic preference belong to the
// provider; this function does not assume a sampler, cache or backend.
constexpr DefaultSelection select_default(const DefaultRequest &request,
    bool prepared, bool fallback = false, bool explicit_source = false,
    DefaultSourceReasons reasons = {}) noexcept {
  const bool available = prepared || fallback || explicit_source;
  const bool automatic = request.mode != RequestedMode::kOff && available &&
      request.automatic_refusal == RouteReason::kNone;
  const bool enabled = request.mode == RequestedMode::kOn || automatic;
  const auto reason = request.mode == RequestedMode::kOff ? RouteReason::kExplicitOff :
      request.mode == RequestedMode::kOn ? RouteReason::kExplicitOn :
      request.automatic_refusal != RouteReason::kNone ? request.automatic_refusal :
      !automatic ? RouteReason::kNoEligibleSource : explicit_source ? reasons.explicit_source :
      prepared ? reasons.prepared : reasons.fallback;
  return {automatic, enabled, enabled && prepared, reason};
}

} // namespace hyle::inference::mtp
