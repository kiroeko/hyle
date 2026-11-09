#pragma once

#include "../../inference/cold_workload.h"
#include "../../inference/route_reason.h"

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

} // namespace qwen35
