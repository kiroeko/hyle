#pragma once

#include <concepts>
#include <cstddef>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

#include "evaluation_failure.h"
#include "request_window.h"

namespace hyle::inference {

// Error facts only. A key names an original qualified output participant or
// explicit argument preflight/retired private validator. A key never grants
// retirement, retry or release permission.
// Preserve the evaluation error and its cleanup observation independently.
class RequestExecutionFailure final : public std::exception {
public:
  // Only an explicit argument check at a known request boundary may use this
  // path. The caller supplies that registered request's complete original key;
  // this factory does not infer a participant, native acceptance or retirement.
  // Lease/device/runtime errors must keep their original evaluation path.
  static RequestExecutionFailure from_preflight(Status status, RequestKey original) {
    if (!status.IsInvalidArgument() || status.has_native_error() || !original.generation ||
        original.slot == (std::numeric_limits<std::size_t>::max)())
      throw std::logic_error("request preflight failure requires an argument error and an original request key");
    return RequestExecutionFailure(EvaluationFailure(std::move(status)), original);
  }

  // An explicit one-request private numeric validator (for example proposal
  // q before target submission) has no submitted target participant. Only its
  // actual observed retirement can qualify this original caller key. This
  // adds error attribution, never cleanup, retry or publication permission.
  static RequestExecutionFailure from_request_evaluation(const EvaluationFailure &failure, RequestKey original) {
    if (!failure.status().IsInvalidArgument() || failure.status().has_native_error() || failure.participant() ||
        !failure.retirement_observed() || !failure.retirement().retired || failure.retirement().status.has_native_error() ||
        !original.generation || original.slot == (std::numeric_limits<std::size_t>::max)())
      throw std::logic_error("request numeric failure requires its original key and actual retired private validator");
    return RequestExecutionFailure(failure, original);
  }

  // Call while the actual submitted bucket is still available. key_of returns
  // that bucket member's original complete key, never active/member/slot index
  // arithmetic. Unknown, invalid or native-fault attribution stays batch-wide.
  // This runs only on failure; ordinary execution builds no key table.
  template<class KeyOf>
    requires requires(KeyOf &key_of, std::size_t index) {
      { key_of(index) } -> std::same_as<RequestKey>;
    }
  static RequestExecutionFailure from_evaluation(const EvaluationFailure &failure,
      std::size_t participants, KeyOf &&key_of) {
    std::optional<RequestKey> key;
    const auto participant = failure.participant();
    const bool native_fault = failure.status().has_native_error() ||
        (failure.retirement_observed() && failure.retirement().status.has_native_error());
    if (!native_fault && participant && *participant < participants)
      key = key_of(*participant);
    return RequestExecutionFailure(failure, key);
  }

  const char *what() const noexcept override { return evaluation_.what(); }
  const EvaluationFailure &evaluation() const noexcept { return evaluation_; }
  std::optional<RequestKey> request_key() const noexcept { return request_key_; }

private:
  RequestExecutionFailure(const EvaluationFailure &failure, std::optional<RequestKey> key)
      : evaluation_(failure), request_key_(key) {}

  EvaluationFailure evaluation_;
  std::optional<RequestKey> request_key_;
};

} // namespace hyle::inference
