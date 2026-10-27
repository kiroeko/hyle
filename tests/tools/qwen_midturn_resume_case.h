#pragma once

#include "../../engine/inference/output_delivery.h"

// Real ordinary Model/Graph lifecycle on the existing small CPU fixture.
// No native fault injection, new executor, model download or timing claim.
namespace qwen_midturn_resume_case {
namespace q = qwen35;
namespace i = hyle::inference;

inline void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}

inline void retire(q::GenerationRequest &request) {
  const auto observed = q::retire_generation_request(request);
  if (observed.retired) request.session.reset();
  q::check(observed.status);
  require(observed.retired, "mid-turn owner did not actually retire its Session");
}

inline int run(const std::filesystem::path &input) {
  const auto geometry = q::read_weight_geometry(input / "weights.tsv");
  require(geometry.hidden == 64 && geometry.intermediate == 128 &&
      geometry.vocabulary == 64 && geometry.layers == "A",
      "mid-turn resume requires the cached untrained A/H64/FFN128/V64 fixture");
  // This case uses the original plain 16-bit K/V cache carriers.
  const q::cache::KvRowBytes cache_rows{
      q::cache::multiply(q::cache::multiply(static_cast<std::size_t>(geometry.kv_heads), static_cast<std::size_t>(geometry.key_depth)), 2),
      q::cache::multiply(q::cache::multiply(static_cast<std::size_t>(geometry.kv_heads), static_cast<std::size_t>(geometry.value_depth)), 2)};
  const auto plan = q::cache::plan(geometry, {.target = cache_rows}, 16, 2, 2,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 1, 1, false, true);
  q::Model model(input / "weights.tsv", geometry, 16, plan, {}, false,
      q::ReplayPolicy::kCommands, 16u << 20,
      {.enabled = true, .policy = {1.F, 8, .9F}, .seed = 123}, 0,
      {.primary = i::BackendKind::kCpu});
  const std::array state_rows{std::size_t{2}};
  const std::array emit_rows{std::size_t{1}, std::size_t{2}};
  model.prepare_family(state_rows, false, 1, q::GraphRole::kText, false, q::TextCompletion::kStateOnly);
  model.prepare_family(emit_rows);
  model.plan_replay_uses(2, 1, 24, q::GraphRole::kText, q::TextCompletion::kStateOnly);
  for (const auto rows : emit_rows) model.plan_replay_uses(rows, 1, 24);
  model.prepare_execution(state_rows, 1, q::GraphRole::kText, q::TextCompletion::kStateOnly);
  model.prepare_execution(emit_rows);

  const std::array a_turns{i::GenerationTurn{{1, 2}, 1}, i::GenerationTurn{{3, 4, 5, 6, 7}, 2}};
  const std::array other_turns{i::GenerationTurn{{8}, 1}};
  std::vector<q::GenerationRequest> requests;
  requests.reserve(3);
  requests.emplace_back("A", a_turns[0].prompt, std::filesystem::path{}, 3);
  requests.emplace_back("C", other_turns[0].prompt, std::filesystem::path{}, 1);
  requests.emplace_back("D", other_turns[0].prompt, std::filesystem::path{}, 1);
  for (std::size_t index = 0; index < requests.size(); ++index) {
    auto &request = requests[index];
    const auto turns = index ? std::span<const i::GenerationTurn>(other_turns)
                             : std::span<const i::GenerationTurn>(a_turns);
    request.progress.bind_turns(turns);
    request.progress.initial_sampling = {123, 10 + index, 0};
    request.ledger = std::make_unique<q::GenerationLedger>(q::generation_record_capacity(turns, 2), 1);
    request.round_emitted.resize(request.progress.maximum_outputs() - 1);
    for (auto &row : request.round_emitted) row.reserve(1);
    request.turn_observations.resize(turns.size());
    for (std::size_t turn = 0; turn < turns.size(); ++turn) {
      auto owner = std::make_shared<std::vector<std::int32_t>>(turns[turn].prompt.size() + (turn ? 1 : 0));
      std::copy(turns[turn].prompt.begin(), turns[turn].prompt.end(), owner->begin() + (turn ? 1 : 0));
      request.warm_inputs.push_back(std::move(owner));
    }
  }
  auto &a = requests[0];
  i::RequestWindow window;
  window.prepare(3, 1);
  i::OutputDelivery delivery(1);
  q::cache::PrefixCache prefix(1);
  const auto key_a = window.enqueue(0);
  a.request_key = key_a;
  const auto begin_if_needed = [&](q::GenerationRequest &request) {
    if (!request.progress.turn_started()) {
      request.prompt = request.progress.begin_turn(*request.warm_inputs[request.progress.turn_index],
          request.progress.turn_index ? std::optional<std::int32_t>(request.session->head->next_id) : std::nullopt);
      request.turn_observations[request.progress.turn_index].output_begin = request.progress.turn_output_begin;
    }
  };
  const auto execute = [&](std::size_t index) {
    auto &request = requests[index];
    const auto credit = index ? std::size_t{1} : delivery.available(request.progress.generated.size());
    std::array calls{q::prepare_generation_call(model, request, request.progress.output_limit(),
        nullptr, requests, credit, 2)};
    q::execute_generation_calls(model, calls, false, nullptr, requests);
  };
  const auto run_waiter = [&](std::size_t index) {
    auto &request = requests[index];
    const auto key = window.enqueue(index);
    request.request_key = key;
    require(window.waiting_front() == key, "paused A blocked another FIFO request");
    const auto victim = window.idle_victim([&](auto member) { return bool(requests[member].session); },
        [&](auto member) { return requests[member].last_idle; });
    require(victim && *victim == 0 && model.snapshot_versions.live == 2,
        "new arrival did not encounter the actual paused resident/version pressure");
    retire(a); ++a.idle_evictions;
    require(window.matches(key_a) && !a.session && prefix.size() == 1 && model.snapshot_versions.live == 1,
        "paused eviction released a request identity or its immutable complete prefix");
    begin_if_needed(request); window.admitted(key); execute(index);
    require(request.progress.request_done(), "new arrival did not complete while A was paused");
    window.release_turn(key, false); retire(request); window.release(key);
    require(window.matches(key_a) && !a.session && model.snapshot_versions.live == 1,
        "new arrival did not retire independently of paused A");
  };
  const auto resume_a = [&] {
    window.wake(key_a);
    require(window.waiting_front() == key_a, "A did not resume through the same FIFO/key");
    q::restore_generation_session(model, a, &prefix, false, requests);
    begin_if_needed(a); window.admitted(key_a);
  };

  begin_if_needed(a); window.admitted(key_a); execute(0);
  require(a.progress.turn_done && a.progress.generated.size() == 1 && a.session->length() == 2,
      "first turn did not produce its complete boundary");
  q::publish_generation_completed(model, a, &prefix);
  const auto first_boundary = q::generation_boundary(*a.session->head);
  // The same consumed key can be published by another request's RNG wrapper.
  // Direct recovery must retain this request's historical cursor, without a draw.
  auto peer = model.sampling_fork(a.session->checkpoint(), {123, 99, 7}, first_boundary.next_id);
  prefix.publish(a.completed_checkpoint.input, peer, a.completed_checkpoint.replay);
  peer.reset();
  window.release_turn(key_a, true); retire(a);
  q::restore_generation_session(model, a, &prefix, false, requests);
  require(a.checkpoint_resume_hits == 1 && !a.restore_count &&
      q::generation_boundary(*a.session->head) == first_boundary &&
      a.progress.generated.size() == 1 && a.progress.observed_commits() == 1,
      "exact complete checkpoint did not preserve the original request's boundary");
  delivery.acknowledge_prefix(a.progress.generated.size(), 1);
  a.progress.advance_turn(); begin_if_needed(a); window.admitted(key_a); execute(0);
  require(a.turn_input_consumed == 2 && !a.progress.prefilled && a.session->head->next_id == -1 &&
      a.session->head->sampling.draw == 1 && a.progress.generated.size() == 1,
      "intermediate chunk emitted a token or advanced the RNG");
  const auto partial = q::generation_boundary(*a.session->head);
  const auto *const turn_data = a.prompt.data();
  const auto partial_commits = a.ledger->committed_count;
  a.last_idle = window.pause(key_a);
  run_waiter(1);
  resume_a();
  require(a.request_key == key_a && q::generation_boundary(*a.session->head) == partial &&
      a.turn_input_consumed == 2 && a.input_turn == 1 && a.prompt.data() == turn_data &&
      a.progress.turn_started() && !a.progress.prefilled && a.progress.turn_output_begin == 1 &&
      a.ledger->committed_count == partial_commits && a.progress.observed_commits() == partial_commits &&
      a.progress.generated.size() == 1 && delivery.acknowledged() == 1 &&
      a.restore_count == 1 && a.checkpoint_resume_misses == 1,
      "state-only wake reused an obsolete checkpoint or restarted/reobserved its turn");
  execute(0);
  require(a.turn_input_consumed == 4 && a.session->head->sampling.draw == 1 && a.progress.generated.size() == 1,
      "resumed intermediate chunk duplicated input/output or RNG");
  window.requeue(key_a); window.admitted(key_a); execute(0);
  require(a.turn_input_consumed == 6 && a.progress.prefilled && a.progress.generated.size() == 2 &&
      a.session->head->sampling.draw == 2 && !a.progress.turn_done && !delivery.available(2),
      "final chunk did not emit exactly one credit-bounded output");
  const auto emitted_boundary = q::generation_boundary(*a.session->head);
  const auto emitted = a.progress.generated;
  const auto emitted_commits = a.ledger->committed_count;
  a.last_idle = window.pause(key_a);
  run_waiter(2);
  delivery.acknowledge_prefix(a.progress.generated.size(), 2);
  resume_a();
  require(q::generation_boundary(*a.session->head) == emitted_boundary &&
      a.progress.generated == emitted && a.progress.observed_commits() == emitted_commits &&
      a.turn_input_consumed == 6 && a.progress.prefilled && delivery.acknowledged() == 2 &&
      a.restore_count == 2 && a.checkpoint_resume_misses == 2,
      "credit-paused wake duplicated delivered output, pending input or random draws");
  execute(0);
  require(a.progress.request_done() && a.progress.generated.size() == 3 &&
      a.session->length() == 9 && a.session->head->sampling.draw == 3 &&
      a.ledger->records[a.ledger->committed_count - 1].pending_input == emitted_boundary.next_id,
      "restored decode did not consume its historical pending exactly once");
  window.release_turn(key_a, false);
  require(window.matches(key_a) && delivery.pending(a.progress.generated).size() == 1,
      "terminal key was released before final output delivery");
  delivery.acknowledge_prefix(a.progress.generated.size(), 3);
  const auto ids = a.progress.generated;
  retire(a); window.release(key_a); prefix.evict_one();
  require(!window.matches(key_a) && !model.snapshot_versions.live && !model.active_step && !model.carrier_borrowed,
      "mid-turn recovery retained a key, version or original execution carrier");
  std::cout << "{\"status\":\"PASS\",\"backend\":\"cpu\","
      "\"fixture\":\"cached_untrained_A_H64_FFN128_V64\",\"same_key\":true,"
      "\"checkpoint_hits\":1,\"obsolete_checkpoint_misses\":2,\"actual_evictions\":2,"
      "\"ledger_rebuilds\":2,\"final_sampling_draw\":3,\"final_versions\":0,\"ids\":["
      << ids[0] << ',' << ids[1] << ',' << ids[2] << "],\"GPU\":false,\"HTTP\":false}\n";
  return 0;
}
} // namespace qwen_midturn_resume_case
