#pragma once

#include <array>
#include <cstddef>
#include <string_view>

namespace hyle::inference {

// Each cold decision has a typed reason and a single observable spelling.
#define HYLE_ROUTE_REASONS(X) \
  X(kActiveSequences, "active_sequences") \
  X(kArtifactRecommendation, "artifact_recommendation") \
  X(kAttentionCapability, "attention_capability") \
  X(kBackend, "backend") \
  X(kBackendCapability, "backend_capability") \
  X(kBackendPolicy, "backend_policy") \
  X(kCacheBudget, "cache_budget") \
  X(kCommandsReplay, "commands_replay") \
  X(kConfiguredDefault, "configured_default") \
  X(kCpuAssist, "cpu_assist") \
  X(kDeviceChainCapacity, "device_chain_capacity") \
  X(kDisabled, "disabled") \
  X(kDraftProjection, "draft_projection") \
  X(kDraftingDisabled, "drafting_disabled") \
  X(kExplicit, "explicit") \
  X(kExplicitDraftAsset, "explicit_draft_asset") \
  X(kExplicitFile, "explicit_file") \
  X(kExplicitFull, "explicit_full") \
  X(kExplicitOff, "explicit_off") \
  X(kExplicitOn, "explicit_on") \
  X(kInsufficientCapacity, "insufficient_capacity") \
  X(kLogitsModifiers, "logits_modifiers") \
  X(kMtpDisabled, "mtp_disabled") \
  X(kMultipleSessionsOrSequences, "multiple_sessions_or_sequences") \
  X(kNoAttentionLayers, "no_attention_layers") \
  X(kNoDecode, "no_decode") \
  X(kNoEligibleSource, "no_eligible_source") \
  X(kNonNaturalRequest, "non_natural_request") \
  X(kNotConfigured, "not_configured") \
  X(kNotInspected, "not_inspected") \
  X(kNotPresent, "not_present") \
  X(kNotSelected, "not_selected") \
  X(kOriginalProjection, "original_projection") \
  X(kOverlapDisabled, "overlap_disabled") \
  X(kPlannedShapes, "planned_shapes") \
  X(kPrefixCache, "prefix_cache") \
  X(kPreparedVocabulary, "prepared_vocabulary") \
  X(kRelayCapability, "relay_capability") \
  X(kSampling, "sampling") \
  X(kScheduler, "scheduler") \
  X(kSelected, "selected") \
  X(kSingleOrNoDraft, "single_or_no_draft") \
  X(kStateTrace, "state_trace") \
  X(kTokenIdPrecision, "token_id_precision") \
  X(kUnsupportedSourceOrBackend, "unsupported_source_or_backend") \
  X(kUserDisabled, "user_disabled")

enum class RouteReason {
  kNone,
#define HYLE_DECLARE_ROUTE_REASON(name, spelling) name,
  HYLE_ROUTE_REASONS(HYLE_DECLARE_ROUTE_REASON)
#undef HYLE_DECLARE_ROUTE_REASON
  kCount,
};

inline constexpr std::array<std::string_view, static_cast<std::size_t>(RouteReason::kCount)> kRouteReasonNames{
    "",
#define HYLE_ROUTE_REASON_NAME(name, spelling) spelling,
    HYLE_ROUTE_REASONS(HYLE_ROUTE_REASON_NAME)
#undef HYLE_ROUTE_REASON_NAME
};
#undef HYLE_ROUTE_REASONS

constexpr std::string_view route_reason_name(RouteReason reason) noexcept {
  const auto index = static_cast<std::size_t>(reason);
  return index < kRouteReasonNames.size() ? kRouteReasonNames[index] : std::string_view{};
}

constexpr bool route_reason_names_unique() noexcept {
  for (std::size_t i = 1; i < kRouteReasonNames.size(); ++i) {
    if (kRouteReasonNames[i].empty()) return false;
    for (std::size_t j = 1; j < i; ++j)
      if (kRouteReasonNames[i] == kRouteReasonNames[j]) return false;
  }
  return true;
}
static_assert(route_reason_names_unique());

} // namespace hyle::inference
