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
  bool reduced_draft_projection = false;
  TargetProjection target = TargetProjection::kDense;
  std::size_t maximum_drafts = 0;
};

struct Request final {
  RequestedMode mode = RequestedMode::kAuto;
  bool automatic_contract = false;
  std::optional<std::size_t> maximum_drafts;
};

struct DefaultPreference final {
  TargetProjection target;
  std::size_t drafts;
};

// Fixed defaults only. These entries neither deny explicit configurations nor
// establish mathematical/backend support. A failed preference tries the next
// supported entry; K2 remains an explicit choice.
inline constexpr std::array default_preferences{
    DefaultPreference{TargetProjection::kDense, 3},
    DefaultPreference{TargetProjection::kDense, 1},
    DefaultPreference{TargetProjection::kRowBlockQuantized, 1}};

constexpr std::optional<std::size_t> preferred_drafts(TargetProjection target,
    std::size_t maximum) noexcept {
  for (const auto preference : default_preferences)
    if (preference.target == target && preference.drafts <= maximum) return preference.drafts;
  return {};
}

enum class SelectionReason {
  kRequestedOff,
  kRequestedOn,
  kPreparedDefault,
  kAutomaticContractUnavailable,
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
  if (!capabilities.model) return unavailable(SelectionReason::kModelUnavailable);
  if (!capabilities.bounded_device_execution) return unavailable(SelectionReason::kDeviceExecutionUnavailable);
  if (!capabilities.reduced_draft_projection) return unavailable(SelectionReason::kPreparedProjectionUnavailable);
  if (request.maximum_drafts) {
    if (*request.maximum_drafts && *request.maximum_drafts <= capabilities.maximum_drafts)
      return {true, *request.maximum_drafts, SelectionReason::kPreparedDefault};
    return unavailable(SelectionReason::kDraftDepthUnavailable);
  }
  if (const auto drafts = preferred_drafts(capabilities.target, capabilities.maximum_drafts))
    return {true, *drafts, SelectionReason::kPreparedDefault};
  return unavailable(SelectionReason::kDraftDepthUnavailable);
}

} // namespace hyle::inference::mtp
