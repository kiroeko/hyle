#pragma once

#include "../../inference/cold_workload.h"
#include "../../inference/mtp_policy.h"

#include <cstddef>

namespace qwen35 {

// Qwen's measured multi-session default is conservative: the prepared subset
// can help mixed Chinese/English prompts but regress on the measured multilingual
// set. This preference is not a capability check; explicit MTP is validated by
// the model's preparation and execution paths.
constexpr hyle::inference::RouteReason automatic_mtp_preference(
    const hyle::inference::ColdWorkloadFacts &workload) noexcept {
  return workload.multiple_schedules || workload.max_sessions > 1 || workload.active_sequences > 1
      ? hyle::inference::RouteReason::kMultipleSessionsOrSequences
      : hyle::inference::RouteReason::kNone;
}

// Qwen's cold source and default policy. These choices describe its current
// routes; a different model may support sampling, prefix reuse or other modes.
struct MtpDefaultRequest final {
  hyle::inference::mtp::RequestedMode mode = hyle::inference::mtp::RequestedMode::kAuto;
  hyle::inference::ColdWorkloadFacts workload;
  bool automatic_backend_eligible = false;
  std::size_t max_drafts = 0;
  bool automatic_vocabulary = false;

  constexpr bool unmodified_greedy_drafting() const noexcept { return workload.greedy; }
};

constexpr bool inspect_prepared_vocabulary(const MtpDefaultRequest &request) noexcept {
  return request.mode != hyle::inference::mtp::RequestedMode::kOff && request.max_drafts > 0 &&
      request.automatic_backend_eligible && request.unmodified_greedy_drafting() &&
      request.automatic_vocabulary;
}

constexpr hyle::inference::RouteReason automatic_mtp_constraint(const MtpDefaultRequest &request) noexcept {
  using hyle::inference::RouteReason;
  const auto &workload = request.workload;
  if (!request.max_drafts) return RouteReason::kDraftingDisabled;
  if (!workload.natural_generation) return RouteReason::kNonNaturalRequest;
  if (!request.automatic_backend_eligible) return RouteReason::kBackendPolicy;
  if (!workload.greedy) return RouteReason::kSampling;
  if (const auto preference = automatic_mtp_preference(workload); preference != RouteReason::kNone) return preference;
  if (workload.prefix_enabled) return RouteReason::kPrefixCache;
  if (workload.scheduler_progress) return RouteReason::kScheduler;
  if (workload.cpu_assist) return RouteReason::kCpuAssist;
  if (workload.capacity < 2) return RouteReason::kInsufficientCapacity;
  return RouteReason::kNone;
}

constexpr hyle::inference::mtp::DefaultSelection select_mtp_default(const MtpDefaultRequest &request,
    bool prepared_vocabulary_exists, bool original_projection_eligible = false,
    bool explicit_asset_selection = false) noexcept {
  namespace mtp = hyle::inference::mtp;
  using hyle::inference::RouteReason;
  const bool prepared = inspect_prepared_vocabulary(request) && prepared_vocabulary_exists;
  return mtp::select_default({request.mode, automatic_mtp_constraint(request)}, prepared,
      original_projection_eligible, explicit_asset_selection,
      {RouteReason::kPreparedVocabulary, RouteReason::kOriginalProjection, RouteReason::kExplicitDraftAsset});
}

} // namespace qwen35
