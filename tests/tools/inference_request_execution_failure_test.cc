// Logical error attribution only. No execution, native fault or test framework.
#include "../../engine/inference/request_execution_failure.h"

#include <array>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace {
namespace i = hyle::inference;

void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}

void require_original(const i::RequestExecutionFailure &mapped, const i::EvaluationFailure &original) {
  const auto &actual = mapped.evaluation();
  require(actual.status() == original.status() && actual.participant() == original.participant() &&
      actual.retirement_observed() == original.retirement_observed() &&
      actual.retirement().retired == original.retirement().retired &&
      actual.retirement().status == original.retirement().status &&
      std::string_view(mapped.what()) == original.what(),
      "request attribution changed the original status, member, message or retirement facts");
}

void maps_the_actual_bucket_key_without_changing_identity() {
  i::RequestWindow owner; owner.prepare(4, 2);
  const auto old = owner.enqueue(3);
  require(owner.cancel_waiting(old), "could not retire the old fixture key");
  owner.release(old);
  const auto replacement = owner.enqueue(3);
  const auto peer = owner.enqueue(1);
  i::RequestWindow foreign; foreign.prepare(4, 2);
  const auto foreign_old = foreign.enqueue(0);
  require(foreign.cancel_waiting(foreign_old), "could not retire the foreign fixture key");
  foreign.release(foreign_old);
  const auto other = foreign.enqueue(3);
  require(other.slot == replacement.slot && other.generation == replacement.generation && other != replacement,
      "foreign fixture did not isolate the key owner identity");
  const std::array bucket{replacement, peer};
  i::EvaluationFailure original(hyle::Status::InvalidArgument("qualified original member"), 0);
  original.observe_retirement({true, {}});
  std::size_t lookups = 0;
  const auto mapped = i::RequestExecutionFailure::from_evaluation(original, bucket.size(), [&](std::size_t index) {
    ++lookups; return bucket[index];
  });
  require_original(mapped, original);
  require(lookups == 1 && mapped.request_key() == replacement && mapped.request_key()->slot == 3 &&
      *mapped.request_key() != old && *mapped.request_key() != other &&
      owner.matches(*mapped.request_key()) && !foreign.matches(*mapped.request_key()) &&
      owner.matches(peer) && !owner.matches(old),
      "member zero was confused with slot, generation or a foreign window");
  const std::array reordered{peer, replacement};
  const auto reordered_error = i::RequestExecutionFailure::from_evaluation(original, reordered.size(),
      [&](std::size_t index) { return reordered[index]; });
  require(reordered_error.request_key() == peer, "error mapping ignored the actual reordered bucket");
}

void unknown_and_out_of_range_members_remain_batch_wide() {
  std::size_t lookups = 0;
  const auto key_of = [&](std::size_t) { ++lookups; return i::RequestKey{}; };
  for (const auto participant : {std::optional<std::size_t>{}, std::optional<std::size_t>{2}}) {
    i::EvaluationFailure original(hyle::Status::InvalidArgument("unqualified original member"), participant);
    original.observe_retirement({true, {}});
    const auto mapped = i::RequestExecutionFailure::from_evaluation(original, 2, key_of);
    require_original(mapped, original);
    require(!mapped.request_key(), "unknown/out-of-range member acquired a request identity");
  }
  const i::EvaluationFailure empty(hyle::Status::InvalidArgument("empty original bucket"), 0);
  const auto mapped = i::RequestExecutionFailure::from_evaluation(empty, 0, key_of);
  require_original(mapped, empty);
  require(!mapped.request_key() && !lookups, "batch-wide failure called the member lookup");
}

void native_faults_never_acquire_a_local_key() {
  const auto native = hyle::Status::FromNative(hyle::StatusCode::DataLoss,
      {hyle::NativeErrorDomain::CudaDriver, 700}, "test native error fact");
  std::size_t lookups = 0;
  const auto key_of = [&](std::size_t) { ++lookups; return i::RequestKey{}; };
  i::EvaluationFailure original(native, 0);
  original.observe_retirement({false, {}});
  auto mapped = i::RequestExecutionFailure::from_evaluation(original, 1, key_of);
  require_original(mapped, original);
  require(!mapped.request_key(), "native computation fault was made request-local");
  i::EvaluationFailure cleanup(hyle::Status::InvalidArgument("qualified output, native cleanup failure"), 0);
  cleanup.observe_retirement({false, native});
  mapped = i::RequestExecutionFailure::from_evaluation(cleanup, 1, key_of);
  require_original(mapped, cleanup);
  require(!mapped.request_key() && !lookups, "native retirement fault acquired a member identity");
}

void computation_and_retirement_stay_independent() {
  i::RequestWindow owner; owner.prepare(1, 1); const auto key = owner.enqueue(0);
  const auto key_of = [&](std::size_t) { return key; };
  const i::EvaluationFailure unobserved(hyle::Status::InvalidArgument("qualified member without retirement observation"), 0);
  auto mapped = i::RequestExecutionFailure::from_evaluation(unobserved, 1, key_of);
  require_original(mapped, unobserved);
  require(mapped.request_key() == key && !mapped.evaluation().retirement_observed() &&
      !mapped.evaluation().retirement().retired && owner.matches(key),
      "identity attribution manufactured retirement or released a key");
  i::EvaluationFailure observed(hyle::Status::InvalidArgument("original invalid output"), 0);
  observed.observe_retirement({true, hyle::Status::FailedPrecondition("independent cleanup observation")});
  mapped = i::RequestExecutionFailure::from_evaluation(observed, 1, key_of);
  require_original(mapped, observed);
  require(mapped.evaluation().status().IsInvalidArgument() &&
      mapped.evaluation().retirement().status.IsFailedPrecondition() &&
      mapped.evaluation().retirement().retired && owner.matches(key),
      "cleanup status replaced the computation error or implicitly consumed the request");
}
} // namespace

int main() {
  try {
    maps_the_actual_bucket_key_without_changing_identity();
    unknown_and_out_of_range_members_remain_batch_wide();
    native_faults_never_acquire_a_local_key();
    computation_and_retirement_stay_independent();
    std::puts("request execution failure: 4 host contract groups passed; no model/device loaded");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "request execution failure: %s\n", error.what());
    return 1;
  }
}
