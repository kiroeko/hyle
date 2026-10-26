#pragma once

#include <concepts>
#include <cstddef>
#include <exception>
#include <optional>

#include "evaluation_failure.h"
#include "request_window.h"

namespace hyle::inference {

// Error facts only. A key attributes the original qualified participant; it
// does not grant retirement, retry, cancellation or request-release permission.
// Preserve the evaluation error and its cleanup observation independently.
class RequestExecutionFailure final : public std::exception {
public:
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
