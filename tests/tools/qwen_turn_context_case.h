#pragma once

// Included after the real model implementation by the lifecycle consumer.
// The cached untrained V64 fixture makes the context boundary independent of EOS.
namespace qwen_turn_context_case {
namespace q = qwen35;
namespace i = hyle::inference;

inline void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}

inline q::GenerationRequest request(std::string name,
    const std::shared_ptr<std::vector<std::int32_t>> &input, std::size_t outputs) {
  q::GenerationRequest value(std::move(name), *input, {}, outputs);
  value.progress.initial_sampling = {123, 0, 0};
  value.warm_inputs.push_back(input);
  value.ledger = std::make_unique<q::GenerationLedger>(outputs, 1);
  value.round_emitted.resize(outputs - 1);
  for (auto &row : value.round_emitted) row.reserve(1);
  return value;
}

inline void execute(q::Model &model, q::GenerationRequest &value) {
  std::array calls{q::prepare_generation_call(model, value, value.progress.output_limit(), nullptr, {})};
  q::execute_generation_calls(model, calls, false);
}

inline void retire(q::GenerationRequest &value) {
  q::check(value.session->stop());
  q::check(value.session->drain());
  require(q::take(value.session->try_retire()), "context-boundary payload did not actually retire");
  value.session.reset();
}

inline int run(const std::filesystem::path &input_directory) {
  const auto manifest = input_directory / "weights.tsv";
  const auto geometry = q::read_weight_geometry(manifest);
  require(geometry.hidden == 64 && geometry.intermediate == 128 && geometry.vocabulary == 64 && geometry.layers == "A",
      "turn-context case requires the cached untrained A/H64/FFN128/V64 fixture");
  const auto plan = q::cache::plan(geometry, 16, 1, 15,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 1, 1, false, true);
  q::Model model(manifest, geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands, 16u << 20,
      {.enabled = true, .policy = {1.F, 8, .9F}, .seed = 123}, 0, {.primary = i::BackendKind::kCpu});
  const std::array shapes{std::size_t{15}, std::size_t{1}};
  model.prepare_family(shapes);
  for (auto rows : shapes) model.plan_replay_uses(rows, 1, 2);
  model.prepare_execution(shapes);
  require(model.execution.host_only() && !model.cache_plan.draft_cache && !model.physical_mtp_roles,
      "context-boundary case must remain ordinary pure CPU");
  const auto prompt = std::make_shared<std::vector<std::int32_t>>(15, 1);

  // No-turn-span callers have the same maximum-journey admission as schedules.
  auto oversized = request("oversized", prompt, 3);
  const std::array oversized_turns{i::GenerationTurn{*prompt, 3}, i::GenerationTurn{}};
  oversized.progress.bind_open_turns(std::span(oversized_turns).first(1), 2, 15);
  oversized.prompt = oversized.progress.begin_turn(*oversized.warm_inputs[0]);
  std::string refusal;
  try { (void)q::prepare_generation_call(model, oversized, 3, nullptr, {}); }
  catch (const std::runtime_error &error) { refusal = error.what(); }
  require(refusal == "complete turn input and output quota exceed the remaining context",
      "oversized turn was not refused by its model-context admission");
  require(!oversized.session && !oversized.ledger->committed_count && !oversized.ledger->sampling &&
      oversized.progress.generated.empty() && !oversized.progress.observed_commits() &&
      !oversized.progress.prefilled && !oversized.progress.turn_done &&
      oversized.input_turn == SIZE_MAX && !oversized.turn_input_consumed &&
      !model.active_step && !model.carrier_borrowed && !model.snapshot_versions.live,
      "context refusal published Session, input admission, ledger, output or model state");

  // The last returned output is not another KV input: P15/quota2 fits C16.
  auto exact = request("exact", prompt, 2);
  execute(model, exact);
  require(exact.session->length() == 15 && exact.progress.generated.size() == 1 &&
      !exact.progress.turn_done && exact.session->head->sampling.draw == 1,
      "exact-fit first output has an invalid state boundary");
  const auto pending = exact.session->head->next_id;
  execute(model, exact);
  require(exact.session->length() == 16 && exact.progress.generated.size() == 2 &&
      exact.progress.request_done() && exact.ledger->committed_count == 2 &&
      exact.progress.observed_commits() == 2 && exact.session->head->sampling.draw == 2 &&
      exact.ledger->records[1].pending_input == pending && exact.ledger->records[1].accepted == 1 &&
      exact.ledger->records[1].before.length == 15 && exact.ledger->records[1].after.length == 16 &&
      exact.session->head->next_id == exact.progress.generated.back(),
      "exact-fit final output was consumed, dropped or not completed at context capacity");
  const auto exact_ids = exact.progress.generated;
  retire(exact);

  // A later empty suffix consumes exactly one historical pending in the last row.
  const std::array turns{i::GenerationTurn{*prompt, 1}, i::GenerationTurn{{}, 1}};
  auto continuation = request("pending-only", prompt, 2);
  continuation.progress.bind_turns(turns);
  continuation.warm_inputs.push_back(std::make_shared<std::vector<std::int32_t>>(1, -1));
  continuation.prompt = continuation.progress.begin_turn(*continuation.warm_inputs[0]);
  execute(model, continuation);
  require(continuation.progress.turn_done && continuation.session->length() == 15,
      "pending-only first turn did not reach its quota");
  const auto historical_pending = continuation.session->head->next_id;
  continuation.progress.advance_turn();
  continuation.prompt = continuation.progress.begin_turn(*continuation.warm_inputs[1], historical_pending);
  execute(model, continuation);
  require(continuation.session->length() == 16 && continuation.progress.request_done() &&
      continuation.progress.generated.size() == 2 && continuation.ledger->committed_count == 2 &&
      continuation.session->head->sampling.draw == 2 && continuation.ledger->records[1].ids().size() == 1 &&
      continuation.ledger->records[1].ids()[0] == historical_pending &&
      continuation.ledger->records[1].before.length == 15 && continuation.ledger->records[1].after.length == 16,
      "last-row pending-only continuation changed its input, quota or random cursor");
  const auto continued_ids = continuation.progress.generated;
  retire(continuation);
  require(!model.snapshot_versions.live && !model.active_step && !model.carrier_borrowed,
      "context-boundary case retained native admission or a state version");
  std::cout << "{\"status\":\"PASS\",\"ordinary\":true,\"backend\":\"cpu\","
      "\"fixture\":\"cached_untrained_A_H64_FFN128_V64\",\"capacity\":16,"
      "\"oversized_quota\":3,\"early_refused_outputs\":0,\"early_refused_commits\":0,"
      "\"exact_quota\":2,\"exact_length\":16,\"exact_draw\":2,\"exact_ids\":["
      << exact_ids[0] << ',' << exact_ids[1] << "],\"pending_only_length\":16,"
      "\"pending_only_draw\":2,\"pending_only_ids\":[" << continued_ids[0] << ',' << continued_ids[1]
      << "],\"final_versions\":0,\"GPU\":false,\"HTTP\":false}\n";
  return 0;
}
} // namespace qwen_turn_context_case
