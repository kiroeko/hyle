#pragma once

#include <cstddef>
#include <string_view>

namespace hyle::inference {

// Cold, configuration-derived facts. The backend and the model still own
// their actual resource and execution capability checks.
struct DecodeRouteFacts final {
  bool overlap_requested = false;
  bool natural_generation = false;
  bool mtp_enabled = false;
  std::size_t max_drafts = 0;
  std::size_t max_device_drafts = 0;
  bool zero_draft_threshold = false;
  bool stored_draft_head = false;
  bool relay_eligible = false;
  std::string_view relay_refusal;
  bool commands_replay = false;
  bool state_trace = false;
  bool has_decode = false;
  std::size_t active_sequences = 0;
};

struct DecodeRouteChoice final {
  bool enabled;
  std::string_view reason;
};

// Bounds the workload's shape independently of how the backend progresses it.
// Scheduler progress can reject device relay without invalidating this bound.
struct BoundedGenerationFacts final {
  bool natural_generation = false, multiple_schedules = false, prefix_cache = false;
  bool planned_shapes = false, supported_backend = false, cpu_assist = false, greedy = false;
  bool mtp_enabled = false, zero_draft_threshold = false;
  std::size_t max_sessions = 0, active_sequences = 0, capacity = 0, max_drafts = 0;
};

constexpr DecodeRouteChoice select_bounded_generation(const BoundedGenerationFacts &facts) noexcept {
  if (!facts.natural_generation) return {false, "non_natural_request"};
  if (facts.multiple_schedules || facts.max_sessions > 1 || facts.active_sequences > 1)
    return {false, "multiple_sessions_or_sequences"};
  if (facts.prefix_cache) return {false, "prefix_cache"};
  if (facts.planned_shapes) return {false, "planned_shapes"};
  if (!facts.supported_backend) return {false, "backend_capability"};
  if (facts.cpu_assist) return {false, "cpu_assist"};
  if (!facts.greedy) return {false, "sampling"};
  if (facts.mtp_enabled && facts.capacity <= 1) return {false, "insufficient_capacity"};
  if (facts.mtp_enabled && !facts.max_drafts) return {false, "drafting_disabled"};
  if (facts.mtp_enabled && !facts.zero_draft_threshold) return {false, "draft_threshold"};
  return {true, "selected"};
}

struct DecodeBudgetRoutes final {
  DecodeRouteChoice overlap;
  DecodeRouteChoice verification;
  std::size_t overlap_depth;
  std::string_view overlap_depth_reason;
};

// The model owns the capacity calculation. This policy only chooses which
// optional route to relinquish, in order, when the minimum workload cannot fit.
template<class Fits>
DecodeBudgetRoutes select_decode_budget_routes(DecodeBudgetRoutes routes, Fits fits) {
  if (routes.verification.enabled && !fits(routes))
    routes.verification = {false, "cache_budget"};
  if (routes.overlap.enabled && routes.overlap_depth == 3 && !fits(routes)) {
    routes.overlap_depth = 2;
    routes.overlap_depth_reason = "cache_budget";
  }
  if (routes.overlap.enabled && !fits(routes)) {
    routes.overlap = {false, "cache_budget"};
    routes.overlap_depth_reason = "cache_budget";
  }
  return routes;
}

constexpr DecodeRouteChoice select_decode_overlap(const DecodeRouteFacts &facts) noexcept {
  if (!facts.overlap_requested) return {false, "user_disabled"};
  if (!facts.natural_generation) return {false, "non_natural_request"};
  if (facts.mtp_enabled) {
    if (!facts.max_drafts) return {false, "drafting_disabled"};
    if (facts.max_drafts > facts.max_device_drafts) return {false, "device_chain_capacity"};
    if (!facts.zero_draft_threshold) return {false, "draft_threshold"};
    if (facts.max_drafts > 1 && !facts.stored_draft_head) return {false, "draft_projection"};
  }
  if (!facts.relay_eligible) return {false, facts.relay_refusal.empty() ? "relay_capability" : facts.relay_refusal};
  if (facts.commands_replay) return {false, "commands_replay"};
  if (facts.state_trace) return {false, "state_trace"};
  if (!facts.has_decode) return {false, "no_decode"};
  return {true, "selected"};
}

constexpr DecodeRouteChoice select_device_verification(const DecodeRouteFacts &facts) noexcept {
  if (!facts.mtp_enabled) return {false, "mtp_disabled"};
  if (facts.max_drafts <= 1) return {false, "single_or_no_draft"};
  if (facts.max_drafts > facts.max_device_drafts) return {false, "device_chain_capacity"};
  if (!facts.zero_draft_threshold) return {false, "draft_threshold"};
  if (!facts.relay_eligible) return {false, facts.relay_refusal.empty() ? "relay_capability" : facts.relay_refusal};
  if (facts.active_sequences != 1) return {false, "active_sequences"};
  if (facts.commands_replay) return {false, "commands_replay"};
  if (!facts.has_decode) return {false, "no_decode"};
  return {true, "selected"};
}

constexpr DecodeRouteChoice select_continuous_mtp(const DecodeRouteFacts &facts, bool overlap_selected,
    bool prefix_cache, std::size_t cached_sessions, bool has_attention, bool exact_token_ids) noexcept {
  if (!overlap_selected) return {false, "overlap_disabled"};
  if (!facts.mtp_enabled) return {false, "mtp_disabled"};
  if (prefix_cache) return {false, "prefix_cache"};
  if (cached_sessions != 1 || facts.active_sequences != 1) return {false, "multiple_sessions_or_sequences"};
  if (!has_attention) return {false, "no_attention_layers"};
  if (!exact_token_ids) return {false, "token_id_precision"};
  return {true, "selected"};
}

} // namespace hyle::inference
