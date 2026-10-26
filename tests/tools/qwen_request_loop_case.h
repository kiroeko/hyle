#pragma once

// Actual tiny Model/Graph consumer of the same adapter as the natural CLI.
// The caller supplies the existing untrained A/H64/FFN128/V64 fixture.
namespace qwen_request_loop_case {
namespace q = qwen35;
namespace i = hyle::inference;

inline void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}

inline int run(const std::filesystem::path &input) {
  const auto geometry = q::read_weight_geometry(input / "weights.tsv");
  require(geometry.hidden == 64 && geometry.intermediate == 128 &&
      geometry.vocabulary == 64 && geometry.layers == "A",
      "request-loop case requires the cached untrained A/H64/FFN128/V64 fixture");
  const q::cache::KvRowBytes rows{
      q::cache::multiply(q::cache::multiply(static_cast<std::size_t>(geometry.kv_heads), static_cast<std::size_t>(geometry.key_depth)), 2),
      q::cache::multiply(q::cache::multiply(static_cast<std::size_t>(geometry.kv_heads), static_cast<std::size_t>(geometry.value_depth)), 2)};
  const auto plan = q::cache::plan(geometry, rows, 16, 1, 3,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 1, 1, false, true);
  q::Model model(input / "weights.tsv", geometry, 16, plan, {}, false,
      q::ReplayPolicy::kCommands, 16u << 20,
      {.enabled = true, .policy = {1.F, 8, .9F}, .seed = 123}, 0,
      {.primary = i::BackendKind::kCpu});
  const std::array shapes{std::size_t{1}, std::size_t{2}, std::size_t{3}};
  model.prepare_family(shapes);
  for (auto count : shapes) model.plan_replay_uses(count, 1, 16);
  model.prepare_execution(shapes);

  const std::array a_turns{i::GenerationTurn{{1, 2}, 1}, i::GenerationTurn{{3, 4}, 2}};
  const std::array peer_turns{i::GenerationTurn{{8}, 1}};
  std::vector<q::GenerationRequest> requests;
  requests.reserve(3);
  requests.emplace_back("A", a_turns[0].prompt, std::filesystem::path{}, 3);
  requests.emplace_back("B", peer_turns[0].prompt, std::filesystem::path{}, 1);
  requests.emplace_back("C", peer_turns[0].prompt, std::filesystem::path{}, 1);
  for (std::size_t index = 0; index < requests.size(); ++index) {
    auto &request = requests[index];
    const auto turns = index ? std::span<const i::GenerationTurn>(peer_turns)
                             : std::span<const i::GenerationTurn>(a_turns);
    if (!index) request.progress.bind_open_turns(turns.first(1), 2, 4);
    else request.progress.bind_turns(turns);
    request.progress.initial_sampling = {123, 20 + index, 0};
    request.ledger = std::make_unique<q::GenerationLedger>(q::generation_record_capacity(turns), 1);
    request.round_emitted.resize(request.progress.maximum_outputs() - 1);
    for (auto &emitted : request.round_emitted) emitted.reserve(1);
    request.turn_observations.resize(turns.size());
    for (std::size_t turn = 0; turn < turns.size(); ++turn) {
      auto owner = std::make_shared<std::vector<std::int32_t>>(turns[turn].prompt.size() + (turn ? 1 : 0));
      std::copy(turns[turn].prompt.begin(), turns[turn].prompt.end(), owner->begin() + (turn ? 1 : 0));
      request.warm_inputs.push_back(std::move(owner));
    }
  }
  auto &a = requests[0];
  i::RequestLoop loop;
  loop.prepare(3, 1);
  std::vector<std::size_t> evictions;
  evictions.reserve(2);
  std::optional<q::GenerationBoundary> expected_resume;
  std::size_t checked_resumes = 0;
  const auto execute = [&](std::span<const i::RequestKey> active) {
    require(active.size() == 1, "N1 owner admitted multiple simultaneous requests");
    auto &request = requests[active.front().slot];
    if (expected_resume && active.front().slot == 0) {
      require(q::generation_boundary(*request.session->head) == *expected_resume &&
          request.progress.generated.size() == 2 && request.progress.observed_commits() == 2 &&
          request.progress.turn_started() && request.progress.prefilled,
          "mid-turn admission changed committed pending/RNG, input or observed output");
      expected_resume.reset(); ++checked_resumes;
    }
    std::array calls{q::prepare_generation_call(model, request, request.progress.output_limit(), nullptr, requests)};
    q::execute_generation_calls(model, calls, false, nullptr, requests);
    q::publish_generation_prompt(model, calls[0], nullptr);
  };
  q::GenerationRequestAdapter adapter{model, static_cast<q::cache::PrefixCache *>(nullptr),
      std::span(requests), false, &evictions, execute};
  a.request_key = loop.enqueue(0);
  const auto key = a.request_key;
  const auto first = loop.pump_once(adapter);
  require(first.admitted == 1 && first.executed && !first.completed &&
      a.progress.waiting_for_input() && loop.active().empty() && loop.matches(key),
      "open input was released or retained an active execution slot");

  requests[1].request_key = loop.enqueue(1);
  const auto second = loop.pump_once(adapter);
  require(second.completed == 1 && !second.waiting_retirement && !a.session &&
      loop.matches(key) && a.idle_evictions == 1 && !model.snapshot_versions.live,
      "new request did not evict only the paused resident and actually retire independently");

  a.progress.extend_turns(a_turns);
  a.progress.close_input();
  loop.wake(key);
  const auto third = loop.pump_once(adapter);
  require(third.admitted == 1 && !third.completed && a.restore_count == 1 &&
      a.progress.turn_index == 1 && a.progress.generated.size() == 2 &&
      a.progress.observed_commits() == 2 && a.session->head->sampling.draw == 2,
      "open-input wake restarted the old turn or lost its complete recovery");
  expected_resume = q::generation_boundary(*a.session->head);
  (void)loop.pause(key);

  requests[2].request_key = loop.enqueue(2);
  const auto fourth = loop.pump_once(adapter);
  require(fourth.completed == 1 && !a.session && loop.matches(key) &&
      a.idle_evictions == 2 && !model.snapshot_versions.live,
      "mid-turn pause lost the request identity or retained its old native owner");
  loop.wake(key);
  const auto last = loop.pump_once(adapter);
  require(last.completed == 1 && !last.waiting_retirement && checked_resumes == 1 &&
      a.restore_count == 2 && a.progress.request_done() && a.progress.generated.size() == 3 &&
      a.progress.observed_commits() == 3 && a.ledger->records[2].after.sampling.draw == 3 &&
      !loop.matches(key) && loop.active().empty() && !loop.waiting_front() &&
      !model.active_step && !model.carrier_borrowed && !model.snapshot_versions.live &&
      evictions == std::vector<std::size_t>{0, 0},
      "request-loop completion duplicated work or released identity before actual retirement");
  std::cout << "{\"status\":\"PASS\",\"fixture\":\"untrained_A_H64\","
      "\"shared_cli_adapter\":true,\"open_input_pause\":true,\"idle_evictions\":2,"
      "\"restores\":2,\"midturn_boundary_equal\":true,\"sampling_draw\":3,\"final_versions\":0}\n";
  return 0;
}
} // namespace qwen_request_loop_case
