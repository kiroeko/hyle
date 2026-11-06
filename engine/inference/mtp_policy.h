#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string_view>

namespace hyle::inference::mtp {

enum class RequestedMode { kOff, kAuto, kOn };
enum class TargetProjection { kDense, kRowBlockQuantized };

// Reports from model/backend preparation, rather than model fingerprints in
// the request driver. Availability is distinct from a default preference.
struct PreparedCapabilities final {
  bool model = false;
  bool bounded_device_execution = false;
  bool prepared_draft_projection = false;
  TargetProjection target = TargetProjection::kDense;
  std::size_t maximum_drafts = 0;
  TargetProjection draft = TargetProjection::kDense;
};

struct Request final {
  RequestedMode mode = RequestedMode::kAuto;
  bool automatic_contract = false;
  std::optional<std::size_t> maximum_drafts;
};

struct DefaultPreference final {
  TargetProjection target;
  TargetProjection draft;
  std::size_t drafts;
  bool automatic;
};

// Fixed defaults only. These entries neither deny explicit configurations nor
// establish mathematical/backend support. A failed preference tries the next
// supported entry; K2 remains an explicit choice.
inline constexpr std::array default_preferences{
    DefaultPreference{TargetProjection::kDense, TargetProjection::kRowBlockQuantized, 3, true},
    DefaultPreference{TargetProjection::kDense, TargetProjection::kRowBlockQuantized, 1, true},
    DefaultPreference{TargetProjection::kDense, TargetProjection::kDense, 1, true},
    DefaultPreference{TargetProjection::kRowBlockQuantized, TargetProjection::kRowBlockQuantized, 1, false},
    DefaultPreference{TargetProjection::kRowBlockQuantized, TargetProjection::kDense, 1, false}};

constexpr bool automatic_default_available(TargetProjection target, TargetProjection draft) noexcept {
  for (const auto preference : default_preferences)
    if (preference.target == target && preference.draft == draft && preference.automatic) return true;
  return false;
}

constexpr std::optional<std::size_t> preferred_drafts(TargetProjection target,
    TargetProjection draft, std::size_t maximum) noexcept {
  for (const auto preference : default_preferences)
    if (preference.target == target && preference.draft == draft && preference.drafts <= maximum) return preference.drafts;
  return {};
}

enum class SelectionReason {
  kRequestedOff,
  kRequestedOn,
  kPreparedDefault,
  kAutomaticContractUnavailable,
  kAutomaticDefaultDisabled,
  kModelUnavailable,
  kDeviceExecutionUnavailable,
  kPreparedProjectionUnavailable,
  kDraftDepthUnavailable,
};

constexpr std::string_view reason_name(SelectionReason reason) noexcept {
  switch (reason) {
    case SelectionReason::kRequestedOff: return "requested_off";
    case SelectionReason::kRequestedOn: return "requested_on";
    case SelectionReason::kPreparedDefault: return "prepared_default";
    case SelectionReason::kAutomaticContractUnavailable: return "automatic_contract_unavailable";
    case SelectionReason::kAutomaticDefaultDisabled: return "automatic_default_disabled";
    case SelectionReason::kModelUnavailable: return "model_unavailable";
    case SelectionReason::kDeviceExecutionUnavailable: return "device_execution_unavailable";
    case SelectionReason::kPreparedProjectionUnavailable: return "prepared_projection_unavailable";
    case SelectionReason::kDraftDepthUnavailable: return "draft_depth_unavailable";
  }
  return "unknown";
}

struct Selection final {
  bool enabled = false;
  std::size_t maximum_drafts = 1;
  SelectionReason reason = SelectionReason::kRequestedOff;
};

constexpr Selection select(Request request, PreparedCapabilities capabilities) noexcept {
  const auto explicit_selection = Selection{request.mode == RequestedMode::kOn,
      request.maximum_drafts.value_or(1), SelectionReason::kRequestedOn};
  if (request.mode == RequestedMode::kOff)
    return {false, request.maximum_drafts.value_or(1), SelectionReason::kRequestedOff};
  if (request.mode == RequestedMode::kOn && request.maximum_drafts)
    return explicit_selection;
  const auto unavailable = [&](SelectionReason reason) {
    return request.mode == RequestedMode::kOn ? explicit_selection
        : Selection{false, request.maximum_drafts.value_or(1), reason};
  };
  if (!request.automatic_contract) return unavailable(SelectionReason::kAutomaticContractUnavailable);
  if (request.mode == RequestedMode::kAuto && !automatic_default_available(capabilities.target, capabilities.draft))
    return unavailable(SelectionReason::kAutomaticDefaultDisabled);
  if (!capabilities.model) return unavailable(SelectionReason::kModelUnavailable);
  if (!capabilities.bounded_device_execution) return unavailable(SelectionReason::kDeviceExecutionUnavailable);
  if (!capabilities.prepared_draft_projection) return unavailable(SelectionReason::kPreparedProjectionUnavailable);
  if (request.maximum_drafts) {
    if (*request.maximum_drafts && *request.maximum_drafts <= capabilities.maximum_drafts)
      return {true, *request.maximum_drafts, SelectionReason::kPreparedDefault};
    return unavailable(SelectionReason::kDraftDepthUnavailable);
  }
  if (const auto drafts = preferred_drafts(capabilities.target, capabilities.draft, capabilities.maximum_drafts))
    return {true, *drafts, SelectionReason::kPreparedDefault};
  return unavailable(SelectionReason::kDraftDepthUnavailable);
}

} // namespace hyle::inference::mtp
