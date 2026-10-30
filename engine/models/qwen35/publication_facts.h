#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>

#include "execution_mode.h"
#include "state_boundary.h"
#include "../../inference/commit_observation.h"
#include "../../inference/retirement_observation.h"
#include "../../inference/request_stop_policy.h"

namespace qwen35 {

struct EvaluationAttemptFacts final {
  GraphRole role;
  TextCompletion completion;
  GenerationBoundary before;
  const hyle::inference::RequestStopPolicy &stop_policy;
  std::size_t input_tokens;
  std::int32_t first_input;
  bool complete_state, withdrawn;
};

struct EvaluationResultFacts final {
  GraphRole role;
  GenerationBoundary after;
  std::size_t requests, member_index, accepted, draft_rows, verified_rows;
  std::span<const std::int32_t> accepted_inputs, verification_inputs;
  std::int32_t next_id;
  bool state_only, stopped;
};

// Raw mathematical consumers need no request ledger. These synchronous static
// projections introduce no callback table, persistent borrow or new owner.
struct NoEvaluationPublication final {
  hyle::inference::PreparedRecordCommit *prepare(std::size_t, const EvaluationAttemptFacts &) noexcept { return nullptr; }
  template<class Result> void selected(std::size_t, const Result &, std::size_t) noexcept {}
  void complete(std::size_t, const EvaluationResultFacts &) noexcept {}
  void finished(std::chrono::steady_clock::time_point, double) noexcept {}
  void retired(const hyle::inference::RetirementObservation &) noexcept {}
};

} // namespace qwen35
