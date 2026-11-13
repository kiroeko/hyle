#pragma once

#include "generation_state.h"
#include "../../inference/model_executor.h"
#include "../../inference/request_admission.h"

namespace qwen35 {

// Admission-only inputs. The caller chooses the visible turn span and explicit
// capacities; installation never discovers future turns or prepares programs.
// Null prepared_model preserves the original single-profile execution path.
struct RequestInstallation final {
  const inference::GenerationSessionSchedule &schedule;
  std::span<const inference::GenerationTurn> visible_turns;
  Model &model;
  Model *prepared_model = nullptr;
  cache::PrefixCache *prefix = nullptr;
  std::size_t output_capacity, turn_capacity, record_capacity;
  bool open_turns = false;
  std::size_t sampling_index = 0;
  inference::RequestStopPolicy stop;
  std::filesystem::path trace;
  Clock::time_point arrival{};
  inference::LogitsInputProvider logits_provider;
  cache::TokenPlanRef token_plan;
};

inline GenerationRequest install_generation_request(RequestInstallation inputs) {
  const auto &schedule = inputs.schedule;
  const auto turns = inputs.visible_turns;
  if (turns.empty() || turns.front().prompt.empty() || !inputs.output_capacity ||
      inputs.turn_capacity < turns.size() || !inputs.record_capacity)
    throw std::invalid_argument("request installation requires visible input and explicit nonzero capacities");
  const auto ceiling = inputs.model.cache_plan.draft_cache ? inputs.model.max_drafts() : 0;
  if (schedule.max_draft_tokens && *schedule.max_draft_tokens > ceiling)
    throw std::invalid_argument("requested draft ceiling exceeds the cold-prepared execution plan");
  GenerationRequest request(schedule.id, turns.front().prompt, std::move(inputs.trace),
      inputs.output_capacity, inputs.sampling_index, std::move(inputs.stop));
  request.prepared_model = inputs.prepared_model;
  request.prepared_prefix = inputs.prefix;
  request.execution_profile = schedule.execution_profile;
  request.max_draft_tokens = schedule.max_draft_tokens.value_or(ceiling);
  request.cache_sharing_domain = schedule.cache_sharing_domain;
  if (schedule.accepts_live_input()) {
    request.live_turns.reserve(inputs.turn_capacity);
    request.live_turns.push_back(turns.front());
    request.progress.bind_open_turns(request.live_turns, inputs.turn_capacity, inputs.turn_capacity);
  } else if (inputs.open_turns)
    request.progress.bind_open_turns(turns, inputs.turn_capacity, inputs.turn_capacity);
  else request.progress.bind_turns(turns);
  request.logits_provider = std::move(inputs.logits_provider);
  request.progress.initial_sampling = inference::initial_sampling_cursor(schedule, inputs.model.sampling_seed);
  request.turn_observations.reserve(inputs.turn_capacity);
  request.turn_observations.resize(turns.size());
  if (inputs.open_turns) request.turn_observations[0].arrival = inputs.arrival;
  request.warm_inputs.reserve(inputs.turn_capacity);
  for (std::size_t i = 0; i < turns.size(); ++i) {
    const auto &ids = turns[i].prompt;
    auto owner = std::make_shared<std::vector<std::int32_t>>(cache::add(ids.size(), i ? 1 : 0));
    std::copy(ids.begin(), ids.end(), owner->begin() + (i ? 1 : 0));
    request.warm_inputs.push_back(std::move(owner));
  }
  request.prompt = *request.warm_inputs.front();
  request.ledger = std::make_unique<GenerationLedger>(inputs.record_capacity, inputs.model.cache_plan.candidate_boundaries);
  request.round_emitted.resize(inputs.output_capacity - 1);
  for (auto &emitted : request.round_emitted) emitted.reserve(inputs.model.cache_plan.candidate_boundaries);
  if (inputs.prefix)
    request.input = inputs.token_plan ? std::move(inputs.token_plan) : cache::token_plan(request.prompt, inputs.model.capacity);
  return request;
}

} // namespace qwen35
