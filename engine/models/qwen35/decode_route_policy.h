#pragma once

#include "../../inference/cold_workload.h"
#include "../../inference/route_choice.h"

#include <cstddef>

namespace qwen35 {

using hyle::inference::ColdWorkloadFacts;
using hyle::inference::RouteReason;
using DecodeRouteChoice = hyle::inference::RouteChoice;

// Cold observations consumed by Qwen diagnostics, never a state owner.
struct RouteDecisionFacts final {
  RouteReason mtp_selection_reason = RouteReason::kNotConfigured;
  RouteReason mtp_depth_source = RouteReason::kNotConfigured;
  RouteReason draft_vocabulary_selection_reason = RouteReason::kNotConfigured;
  RouteReason bounded_generation_plan_reason = RouteReason::kNotConfigured;
  RouteReason decode_overlap_plan_reason = RouteReason::kNotConfigured;
  RouteReason device_verification_plan_reason = RouteReason::kNotConfigured;
  RouteReason overlap_depth_plan_reason = RouteReason::kNotConfigured;
};

// Qwen's current prepared routes, not requirements imposed on other models.
// Workload form, replay/trace and source layout constrain these implementations;
// the resident single-sequence preference and budget fallback order are cold
// policy. Native permissions and actual retirement remain with Graph/Compute.

// Cold, configuration-derived facts. The backend and the model still own
// their actual resource and execution capability checks.
struct DecodeRouteFacts final {
  ColdWorkloadFacts workload;
  bool overlap_requested = false;
  bool mtp_enabled = false;
  std::size_t max_drafts = 0;
  std::size_t max_device_drafts = 0;
  bool stored_draft_head = false;
  bool relay_eligible = false;
  RouteReason relay_refusal = RouteReason::kNone;
  bool commands_replay = false;
  bool has_decode = false;
  std::size_t resolved_active_sequences = 0;
};

// Bounds the workload's shape independently of how the backend progresses it.
// Scheduler progress can reject device relay without invalidating this bound.
struct BoundedGenerationFacts final {
  ColdWorkloadFacts workload;
  bool supported_backend = false;
  bool mtp_enabled = false;
  std::size_t max_drafts = 0;
};

constexpr DecodeRouteChoice select_bounded_generation(const BoundedGenerationFacts &facts) noexcept {
  const auto &workload = facts.workload;
  if (!workload.natural_generation) return {false, RouteReason::kNonNaturalRequest};
  if (workload.multiple_schedules || workload.max_sessions > 1 || workload.active_sequences > 1)
    return {false, RouteReason::kMultipleSessionsOrSequences};
  if (workload.prefix_enabled) return {false, RouteReason::kPrefixCache};
  if (workload.planned_shapes) return {false, RouteReason::kPlannedShapes};
  if (!facts.supported_backend) return {false, RouteReason::kBackendCapability};
  if (workload.cpu_assist) return {false, RouteReason::kCpuAssist};
  if (!workload.greedy) return {false, RouteReason::kSampling};
  if (facts.mtp_enabled && workload.capacity <= 1) return {false, RouteReason::kInsufficientCapacity};
  if (facts.mtp_enabled && !facts.max_drafts) return {false, RouteReason::kDraftingDisabled};
  return {true, RouteReason::kSelected};
}

struct DecodeBudgetRoutes final {
  DecodeRouteChoice overlap;
  DecodeRouteChoice verification;
  std::size_t overlap_depth;
  RouteReason overlap_depth_reason = RouteReason::kNone;
};

// The model owns the capacity calculation. This policy only chooses which
// optional route to relinquish, in order, when the minimum workload cannot fit.
template<class Fits>
DecodeBudgetRoutes select_decode_budget_routes(DecodeBudgetRoutes routes, Fits fits) {
  if (routes.verification.enabled && !fits(routes))
    routes.verification = {false, RouteReason::kCacheBudget};
  if (routes.overlap.enabled && routes.overlap_depth == 3 && !fits(routes)) {
    routes.overlap_depth = 2;
    routes.overlap_depth_reason = RouteReason::kCacheBudget;
  }
  if (routes.overlap.enabled && !fits(routes)) {
    routes.overlap = {false, RouteReason::kCacheBudget};
    routes.overlap_depth_reason = RouteReason::kCacheBudget;
  }
  return routes;
}

constexpr DecodeRouteChoice select_decode_overlap(const DecodeRouteFacts &facts) noexcept {
  if (!facts.overlap_requested) return {false, RouteReason::kUserDisabled};
  if (!facts.workload.natural_generation) return {false, RouteReason::kNonNaturalRequest};
  if (facts.mtp_enabled) {
    if (!facts.max_drafts) return {false, RouteReason::kDraftingDisabled};
    if (facts.max_drafts > facts.max_device_drafts) return {false, RouteReason::kDeviceChainCapacity};
    if (facts.max_drafts > 1 && !facts.stored_draft_head) return {false, RouteReason::kDraftProjection};
  }
  if (!facts.relay_eligible)
    return {false, facts.relay_refusal == RouteReason::kNone ? RouteReason::kRelayCapability : facts.relay_refusal};
  if (facts.commands_replay) return {false, RouteReason::kCommandsReplay};
  if (facts.workload.state_trace) return {false, RouteReason::kStateTrace};
  if (!facts.has_decode) return {false, RouteReason::kNoDecode};
  return {true, RouteReason::kSelected};
}

// A resident executor prepares this route before requests arrive. Its single
// configured slot is independent of whether token selection is random;
// sampled MTP and other unsupported cases are rejected by relay capability.
constexpr DecodeRouteChoice select_resident_decode_overlap(const DecodeRouteFacts &facts) noexcept {
  const auto route = select_decode_overlap(facts);
  if (!route.enabled) return route;
  if (facts.workload.max_sessions != 1 || facts.workload.active_sequences != 1)
    return {false, RouteReason::kMultipleSessionsOrSequences};
  return route;
}

// Verification capacity also supports standalone draft chains, independently
// of overlap and a compound projection. Tracing disables packet handoff but
// can still use this capacity for an independent chain.
constexpr DecodeRouteChoice select_device_verification(const DecodeRouteFacts &facts) noexcept {
  if (!facts.mtp_enabled) return {false, RouteReason::kMtpDisabled};
  if (facts.max_drafts <= 1) return {false, RouteReason::kSingleOrNoDraft};
  if (facts.max_drafts > facts.max_device_drafts) return {false, RouteReason::kDeviceChainCapacity};
  if (!facts.relay_eligible)
    return {false, facts.relay_refusal == RouteReason::kNone ? RouteReason::kRelayCapability : facts.relay_refusal};
  if (facts.resolved_active_sequences != 1) return {false, RouteReason::kActiveSequences};
  if (facts.commands_replay) return {false, RouteReason::kCommandsReplay};
  if (!facts.has_decode) return {false, RouteReason::kNoDecode};
  return {true, RouteReason::kSelected};
}

struct DecodeRoutePlans final {
  DecodeRouteChoice overlap;
  DecodeRouteChoice verification;
};

constexpr DecodeRoutePlans plan_decode_routes(const DecodeRouteFacts &facts) noexcept {
  return {select_decode_overlap(facts), select_device_verification(facts)};
}

constexpr DecodeRouteChoice select_continuous_mtp(const DecodeRouteFacts &facts, bool overlap_selected,
    std::size_t cached_sessions, bool has_attention, bool exact_token_ids) noexcept {
  if (!overlap_selected) return {false, RouteReason::kOverlapDisabled};
  if (!facts.mtp_enabled) return {false, RouteReason::kMtpDisabled};
  if (facts.workload.prefix_enabled) return {false, RouteReason::kPrefixCache};
  if (cached_sessions != 1 || facts.resolved_active_sequences != 1) return {false, RouteReason::kMultipleSessionsOrSequences};
  if (!has_attention) return {false, RouteReason::kNoAttentionLayers};
  if (!exact_token_ids) return {false, RouteReason::kTokenIdPrecision};
  return {true, RouteReason::kSelected};
}

} // namespace qwen35
