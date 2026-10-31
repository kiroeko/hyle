#pragma once

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "retirement_observation.h"

namespace hyle::inference {

// Ephemeral error facts only: no Session, native work, lease or cleanup owner.
// A participant names an original submitted member proven by exact validity
// decoding (one 0, all others 1) or that member's explicit prefix validator.
// Private auxiliary work has no invented target participant. An absent index
// provides no batch attribution; a separate original-request boundary can
// supply its qualified key without granting retirement or retry permission.
class EvaluationFailure final : public std::runtime_error {
public:
  explicit EvaluationFailure(Status status, std::optional<std::size_t> participant = {})
      : std::runtime_error(std::string(status.message())), status_(std::move(status)),
        participant_(participant) {}
  const Status &status() const noexcept { return status_; }
  std::optional<std::size_t> participant() const noexcept { return participant_; }
  // Ephemeral preparation origin only. It proves neither retirement nor any
  // input/output permission, and is never inferred from an error code/message.
  bool unsubmitted_reuse_refusal() const noexcept { return unsubmitted_reuse_refusal_; }
  void observe_unsubmitted_reuse_refusal() noexcept { unsubmitted_reuse_refusal_ = true; }
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
  bool unsubmitted_reuse_refusal_ = false;
};

} // namespace hyle::inference
