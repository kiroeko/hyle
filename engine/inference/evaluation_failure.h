#pragma once

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "retirement_observation.h"

namespace hyle::inference {

// Ephemeral error facts only: no Session, native work, lease or cleanup owner.
// A participant index is qualified only by successful exact output decoding:
// one ordinary Text/EmitNext member is 0 and every other member is 1.
// No index means the owner must not classify this as a request-local failure.
class EvaluationFailure final : public std::runtime_error {
public:
  explicit EvaluationFailure(Status status, std::optional<std::size_t> participant = {})
      : std::runtime_error(std::string(status.message())), status_(std::move(status)),
        participant_(participant) {}
  const Status &status() const noexcept { return status_; }
  std::optional<std::size_t> participant() const noexcept { return participant_; }
  bool retirement_observed() const noexcept { return retirement_observed_; }
  const RetirementObservation &retirement() const noexcept { return retirement_; }
  // An outer original cleanup may drive pending work again. Preserve its real
  // first error while updating the actual proof, never infer proof from status.
  void observe_retirement(const RetirementObservation &observed) noexcept {
    record_retirement_status(retirement_.status, observed.status);
    retirement_.retired = observed.retired;
    retirement_observed_ = true;
  }
private:
  Status status_;
  std::optional<std::size_t> participant_;
  RetirementObservation retirement_{};
  bool retirement_observed_ = false;
};

} // namespace hyle::inference
