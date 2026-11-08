#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>

#include "request_progress.h"

namespace hyle::inference {

// Cold checks for the bounded schedule policy used by this caller. This is
// not a universal context rule: sliding-window or other model policies may
// select different bounds. Future output quotas are not combined with prompt
// lengths: a previous turn can stop early, so its actual remaining context is
// checked at that boundary.
enum class RequestAdmissionError {
  kInvalidLimits, kNoTurns, kInvalidLiveTurns, kTurnLimit, kCapacity, kToken
};

inline std::optional<RequestAdmissionError> check_request_schedule(
    const GenerationSessionSchedule &schedule, std::size_t context, std::size_t vocabulary) noexcept {
  if (!context || !vocabulary) return RequestAdmissionError::kInvalidLimits;
  if (schedule.turns.empty()) return RequestAdmissionError::kNoTurns;
  if (schedule.accepts_live_input() && schedule.turns.size() != 1)
    return RequestAdmissionError::kInvalidLiveTurns;
  if (schedule.turns.size() > context) return RequestAdmissionError::kTurnLimit;
  std::size_t prompt_total = 0, output_total = 0;
  for (std::size_t index = 0; index < schedule.turns.size(); ++index) {
    const auto &turn = schedule.turns[index];
    if (!turn.max_outputs || (!index && turn.prompt.empty()) || turn.prompt.size() > context ||
        (index && turn.prompt.size() >= context) ||
        (!index && turn.max_outputs - 1 > context - turn.prompt.size()) ||
        (index && turn.max_outputs > context) || turn.prompt.size() > context - prompt_total ||
        turn.max_outputs > context - output_total)
      return RequestAdmissionError::kCapacity;
    prompt_total += turn.prompt.size();
    output_total += turn.max_outputs;
    if (std::ranges::any_of(turn.prompt, [&](auto id) {
          return id < 0 || static_cast<std::size_t>(id) >= vocabulary;
        })) return RequestAdmissionError::kToken;
  }
  return std::nullopt;
}

inline const char *request_admission_message(RequestAdmissionError error) noexcept {
  switch (error) {
    case RequestAdmissionError::kInvalidLimits: return "reusable request requires valid model limits";
    case RequestAdmissionError::kNoTurns: return "reusable request requires an initial turn";
    case RequestAdmissionError::kInvalidLiveTurns: return "live input requires one initial turn and turn boundaries";
    case RequestAdmissionError::kTurnLimit: return "reusable request exceeds its configured turn bound";
    case RequestAdmissionError::kCapacity: return "reusable request exceeds its configured input or output bounds";
    case RequestAdmissionError::kToken: return "reusable request contains a token outside the model vocabulary";
  }
  return "invalid reusable request";
}

} // namespace hyle::inference
