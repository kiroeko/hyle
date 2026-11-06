#include "../../engine/inference/mtp_policy.h"

#include <cstdio>
#include <stdexcept>

namespace {
std::size_t checks = 0;
void require(bool value) {
  if (!value) throw std::runtime_error("MTP default policy contract failed");
  ++checks;
}
}

int main() {
  namespace m = hyle::inference::mtp;
  using Reason = m::SelectionReason;
  const m::PreparedCapabilities dense{true, true, true, m::TargetProjection::kDense, 3,
      m::TargetProjection::kRowBlockQuantized};
  auto packed = dense; packed.target = m::TargetProjection::kRowBlockQuantized;
  const m::Request automatic{m::RequestedMode::kAuto, true, {}};
  const auto check = [&](m::Request request, m::PreparedCapabilities available,
                         bool enabled, std::size_t drafts, Reason reason) {
    const auto selection = m::select(request, available);
    require(selection.enabled == enabled);
    require(selection.maximum_drafts == drafts);
    require(selection.reason == reason);
  };
  check(automatic, dense, true, 3, Reason::kPreparedDefault);
  check(automatic, packed, false, 1, Reason::kAutomaticDefaultDisabled);
  auto packed_explicit_depth = automatic; packed_explicit_depth.maximum_drafts = 3;
  check(packed_explicit_depth, packed, false, 3, Reason::kAutomaticDefaultDisabled);
  auto original_dense = dense; original_dense.draft = m::TargetProjection::kDense;
  check(automatic, original_dense, true, 1, Reason::kPreparedDefault);
  auto explicit_dense = automatic; explicit_dense.mode = m::RequestedMode::kOn; explicit_dense.maximum_drafts = 3;
  check(explicit_dense, original_dense, true, 3, Reason::kRequestedOn);
  explicit_dense.mode = m::RequestedMode::kAuto; explicit_dense.maximum_drafts = 2;
  check(explicit_dense, original_dense, true, 2, Reason::kPreparedDefault);
  auto limited = dense; limited.maximum_drafts = 1;
  check(automatic, limited, true, 1, Reason::kPreparedDefault);
  limited.maximum_drafts = 0;
  check(automatic, limited, false, 1, Reason::kDraftDepthUnavailable);
  auto unavailable = dense; unavailable.model = false;
  check(automatic, unavailable, false, 1, Reason::kModelUnavailable);
  unavailable = dense; unavailable.bounded_device_execution = false;
  check(automatic, unavailable, false, 1, Reason::kDeviceExecutionUnavailable);
  unavailable = dense; unavailable.prepared_draft_projection = false;
  check(automatic, unavailable, false, 1, Reason::kPreparedProjectionUnavailable);
  auto request = automatic; request.automatic_contract = false;
  check(request, dense, false, 1, Reason::kAutomaticContractUnavailable);
  request = automatic; request.maximum_drafts = 2;
  check(request, dense, true, 2, Reason::kPreparedDefault);
  request.maximum_drafts = 4;
  check(request, dense, false, 4, Reason::kDraftDepthUnavailable);
  request.maximum_drafts = 0;
  check(request, dense, false, 0, Reason::kDraftDepthUnavailable);
  request = automatic; request.mode = m::RequestedMode::kOff;
  check(request, dense, false, 1, Reason::kRequestedOff);
  request.maximum_drafts = 3;
  check(request, packed, false, 3, Reason::kRequestedOff);
  request = automatic; request.mode = m::RequestedMode::kOn;
  check(request, dense, true, 3, Reason::kPreparedDefault);
  check(request, packed, true, 1, Reason::kPreparedDefault);
  check(request, {}, true, 1, Reason::kRequestedOn);
  request.maximum_drafts = 4;
  check(request, dense, true, 4, Reason::kRequestedOn);
  request.maximum_drafts = 0;
  check(request, {}, true, 0, Reason::kRequestedOn);
  std::printf("MTP default policy: %zu checks passed\n", checks);
}
