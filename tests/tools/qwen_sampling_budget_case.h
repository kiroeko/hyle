#pragma once

// Included after the real Model implementation. No synthetic native failures:
// this checks ordinary random address admission on the retained tiny CPU model.
namespace qwen_sampling_budget_case {
namespace q = qwen35;
namespace i = hyle::inference;
inline void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
template<class F> inline void exhausted(F &&action) {
  bool rejected = false;
  try { action(); } catch (const std::runtime_error &error) {
    rejected = std::string_view(error.what()) == "sampling draw cursor is exhausted";
  }
  require(rejected, "expected exact sampling address exhaustion before admission");
}
inline q::GenerationRequest request(std::string name, std::span<const std::int32_t> prompt,
    std::size_t outputs, std::uint64_t draw, std::size_t records, std::size_t sampling = 0) {
  q::GenerationRequest value(std::move(name), prompt, {}, outputs, sampling);
  value.progress.initial_sampling = {123, 10, draw};
  value.warm_inputs.push_back(std::make_shared<std::vector<std::int32_t>>(prompt.begin(), prompt.end()));
  value.prompt = *value.warm_inputs.front();
  value.ledger = std::make_unique<q::GenerationLedger>(records, 1);
  value.round_emitted.resize(outputs - 1);
  for (auto &emitted : value.round_emitted) emitted.reserve(1);
  return value;
}
inline void retire(q::GenerationRequest &value) {
  const auto observed = q::retire_generation_request(value);
  if (observed.retired) value.session.reset();
  q::check(observed.status);
  require(observed.retired, "sampling-budget Session did not actually retire");
}
inline void full_hit_peer_preflight_retry(q::Model &model, q::cache::PrefixCache &prefix,
    const q::cache::TokenPlanRef &input, std::span<const std::int32_t> prompt,
    std::uint64_t maximum) {
  std::vector<q::GenerationRequest> owners; owners.reserve(2);
  owners.push_back(request("full-hit-unobserved", prompt, 1, maximum - 1, 1));
  owners.push_back(request("peer-preflight", prompt, 1, 0, 1, 1));
  for (auto &owner : owners) { owner.input = input; owner.turn_observations.resize(1); }
  auto &a = owners[0]; auto &b = owners[1];
  std::size_t provider_calls = 0;
  // This is a real request argument error: this greedy configuration has no
  // external bias/mask ports. The original preflight must reject B, not call
  // its provider or manufacture a target participant/retirement observation.
  b.logits_provider = [&](std::span<const std::int32_t>, std::span<const std::int32_t>)
      -> hyle::Result<q::LogitsInputs> { ++provider_calls; return q::LogitsInputs{}; };
  i::OutputDelivery delivery(1);
  i::RequestLoop loop; loop.prepare(2, 2);
  struct Saved final {
    const void *head = nullptr;
    std::uint64_t epoch = 0;
    q::GenerationBoundary boundary;
    q::GenerationCommitRecord record; // Host facts only; no native version pin.
  };
  std::optional<Saved> saved;
  std::size_t prepared_waves = 0, executed_waves = 0, target_submissions = 0;
  const auto frontier_calls = model.frontier_sampling_graph_calls;
  const auto execute_wave = [&](std::span<const i::RequestKey> active) {
    ++prepared_waves;
    require((!saved && active.size() == 2 && active[0] == a.request_key && active[1] == b.request_key) ||
        (saved && active.size() == 1 && active[0] == a.request_key),
        "full-hit recovery changed its original wave/key order");
    std::vector<q::PreparedGenerationCall> calls; calls.reserve(active.size());
    for (const auto key : active)
      calls.push_back(q::prepare_generation_call(model, owners[key.slot], 1, &prefix, owners,
          loop.credit(key, owners[key.slot].progress.generated.size())));
    q::execute_generation_calls(model, calls, false, &prefix, owners, [&] { ++target_submissions; });
    ++executed_waves;
    require(saved && calls.size() == 1 && calls[0].full_hit &&
        calls[0].disposition == q::EvaluationDisposition::kPublished &&
        a.session->head.get() == saved->head && a.session->head.epoch() == saved->epoch &&
        q::generation_boundary(*a.session->head) == saved->boundary &&
        a.ledger->committed_count == 1 && a.ledger->records[0].same_result(saved->record) &&
        a.ledger->records[0].committed && a.ledger->records[0].imported &&
        a.ledger->records[0].user_output && a.ledger->records[0].ids().data() == saved->record.ids().data() &&
        a.ledger->records[0].ids().size() == saved->record.ids().size() &&
        a.progress.generated == std::vector<std::int32_t>{saved->boundary.next_id} &&
        a.progress.observed_commits() == 1 && a.progress.prefilled && a.progress.request_done(),
        "full-hit recovery reran sampling, changed its committed boundary/ledger or duplicated its output");
    for (auto &call : calls) q::publish_generation_prompt(model, call, &prefix);
  };
  q::SynchronousGenerationWave wave{model, execute_wave};
  q::GenerationRequestAdapter adapter{model, &prefix, std::span(owners), false,
      static_cast<std::vector<std::size_t> *>(nullptr), wave};
  a.request_key = loop.enqueue(0, &delivery); b.request_key = loop.enqueue(1);
  const auto key_a = a.request_key, key_b = b.request_key;
  bool rejected_b = false;
  try { (void)loop.pump_once(adapter); }
  catch (const i::RequestExecutionFailure &failure) {
    const auto &evaluation = failure.evaluation();
    rejected_b = failure.request_key() == key_b && evaluation.status().IsInvalidArgument() &&
        !evaluation.status().has_native_error() && !evaluation.participant() && !evaluation.retirement_observed();
  }
  require(rejected_b && prepared_waves == 1 && !executed_waves && !target_submissions && !provider_calls &&
      loop.matches(key_a) && loop.matches(key_b) && a.request_key == key_a && b.request_key == key_b &&
      a.session && a.session->head && a.session->head->sampling.draw == maximum &&
      a.prefix_full_hits == 1 && a.prefix_sampling_calls == 1 &&
      model.frontier_sampling_graph_calls == frontier_calls + 1 &&
      a.ledger->committed_count == 1 && a.ledger->records[0].committed && a.ledger->records[0].imported &&
      a.progress.observed_commits() == 1 && a.progress.generated.empty() && !a.progress.prefilled &&
      !a.progress.turn_done && a.progress.input_consumed() == prompt.size() && !delivery.acknowledged() &&
      !b.session && !b.ledger->committed_count && b.progress.generated.empty() &&
      !model.active_step && !model.successor_step && !model.carrier_borrowed,
      "peer preflight did not preserve A's real unobserved full-hit result and B's original error/key");
  saved.emplace(Saved{a.session->head.get(), a.session->head.epoch(),
      q::generation_boundary(*a.session->head), a.ledger->records[0]});
  q::check(loop.cancel(key_b, adapter));
  // No new random address is needed: A's only output is already sampled at
  // MAX. The next real pump must observe it without preparing another draw.
  const auto recovered = loop.pump_once(adapter);
  require(recovered.executed && recovered.completed == 1 && recovered.cancelled == 1 &&
      recovered.waiting_output && !recovered.waiting_retirement &&
      prepared_waves == 2 && executed_waves == 1 && !target_submissions && !provider_calls &&
      model.frontier_sampling_graph_calls == frontier_calls + 1 &&
      loop.matches(key_a) && !loop.matches(key_b) && a.request_key == key_a && b.request_key == key_b &&
      !a.session && !b.session && !delivery.acknowledged() &&
      delivery.pending(a.progress.generated).size() == 1 &&
      delivery.pending(a.progress.generated).front() == saved->boundary.next_id &&
      a.ledger->committed_count == 1 && a.ledger->records[0].same_result(saved->record),
      "full-hit recovery submitted again, lost its key or retired before preserving one pending delivery");
  require(loop.acknowledge(key_a, 1, adapter) && loop.acknowledge(key_a, 1, adapter),
      "full-hit result rejected its original or duplicate delivery acknowledgement");
  const auto released = loop.pump_once(adapter);
  require(released.completed == 1 && !released.executed && !released.cancelled &&
      !released.waiting_output && !released.waiting_retirement && !loop.matches(key_a) &&
      loop.active().empty() && !loop.waiting_front() && delivery.acknowledged() == 1 &&
      delivery.pending(a.progress.generated).empty() && a.progress.generated.size() == 1 &&
      model.frontier_sampling_graph_calls == frontier_calls + 1 &&
      !model.active_step && !model.successor_step && !model.carrier_borrowed,
      "full-hit delivery acknowledgement duplicated output or retained its original request authority");
}
inline int run(const std::filesystem::path &input) {
  const auto geometry = q::read_weight_geometry(input / "weights.tsv");
  require(geometry.hidden == 64 && geometry.intermediate == 128 &&
      geometry.vocabulary == 64 && geometry.layers == "A", "sampling-budget case requires cached A/H64/FFN128/V64");
  // This case uses the original plain 16-bit K/V cache carriers.
  const q::cache::KvRowBytes cache_rows{
      q::cache::multiply(q::cache::multiply(static_cast<std::size_t>(geometry.kv_heads), static_cast<std::size_t>(geometry.key_depth)), 2),
      q::cache::multiply(q::cache::multiply(static_cast<std::size_t>(geometry.kv_heads), static_cast<std::size_t>(geometry.value_depth)), 2)};
  const auto plan = q::cache::plan(geometry, {.target = cache_rows}, 16, 3, 2,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 1, 1, false, true);
  const std::array additional{q::SamplingConfiguration{false, {}}};
  q::Model model(input / "weights.tsv", geometry, 16, plan, {}, false,
      q::ReplayPolicy::kCommands, 16u << 20,
      {.enabled = true, .policy = {1.F, 8, .9F}, .seed = 123}, 0,
      {.primary = i::BackendKind::kCpu}, additional);
  const std::array emit_rows{std::size_t{1}, std::size_t{2}};
  const std::array state_rows{std::size_t{1}};
  for (const auto sampling : {std::size_t{0}, std::size_t{1}}) {
    model.prepare_family(emit_rows, false, 1, q::GraphRole::kText, false, q::TextCompletion::kEmitNext, sampling);
    for (auto rows : emit_rows) model.plan_replay_uses(rows, 1, 32, q::GraphRole::kText, q::TextCompletion::kEmitNext, sampling);
  }
  model.prepare_family(state_rows, false, 1, q::GraphRole::kText, false, q::TextCompletion::kStateOnly);
  model.plan_replay_uses(1, 1, 16, q::GraphRole::kText, q::TextCompletion::kStateOnly);
  for (const auto sampling : {std::size_t{0}, std::size_t{1}})
    model.prepare_execution(emit_rows, 1, q::GraphRole::kText, q::TextCompletion::kEmitNext, sampling);
  model.prepare_execution(state_rows, 1, q::GraphRole::kText, q::TextCompletion::kStateOnly);
  const auto maximum = (std::numeric_limits<std::uint64_t>::max)();
  const std::array<std::int32_t, 2> prompt{1, 2};
  const auto execute = [&](q::GenerationRequest &value, q::cache::PrefixCache *prefix = nullptr,
                           std::size_t quantum = SIZE_MAX) {
    std::array calls{q::prepare_generation_call(model, value, value.progress.output_limit(), prefix, {}, 1, quantum)};
    q::execute_generation_calls(model, calls, false, prefix);
  };

  // First admission must fail before auto-begin, policy binding or state-only
  // input publication. The same not-yet-admitted request can use a legal cursor.
  auto initial = request("initial-budget", prompt, 2, maximum - 1, 3);
  exhausted([&] { (void)q::prepare_generation_call(model, initial, 2, nullptr, {}, 1, 1); });
  require(!initial.progress.turn_started() && initial.progress.generated.empty() &&
      !initial.progress.observed_commits() && !initial.progress.prefilled && !initial.progress.turn_done &&
      initial.progress.input_turn() == SIZE_MAX && !initial.progress.input_consumed() && !initial.session &&
      !initial.ledger->committed_count && !initial.ledger->sampling && !model.active_step &&
      !model.carrier_borrowed && !model.snapshot_versions.live,
      "exhausted whole turn changed progress, ledger, input, Session or carrier");
  initial.progress.initial_sampling.draw = maximum - 2;
  execute(initial, nullptr, 1);
  require(initial.session->length() == 1 && initial.session->head->sampling.draw == maximum - 2 &&
      initial.session->head->next_id == -1 && initial.progress.generated.empty() &&
      initial.ledger->committed_count == 1 && initial.progress.input_consumed() == 1,
      "state-only chunk spent a random address or produced a pending output");
  retire(initial);
  q::restore_generation_session(model, initial, nullptr, false);
  require(initial.restore_count == 1 && initial.session->length() == 1 &&
      initial.session->head->sampling.draw == maximum - 2 && initial.progress.generated.empty(),
      "state-only ledger rebuild spent a random address");
  execute(initial, nullptr, 1);
  require(initial.progress.generated.size() == 1 && initial.session->head->sampling.draw == maximum - 1,
      "prefill first output did not use exactly one admitted address");
  const auto initial_pending = initial.session->head->next_id;
  execute(initial);
  require(initial.progress.request_done() && initial.progress.generated.size() == 2 &&
      initial.session->head->sampling.draw == maximum && initial.session->length() == 3 &&
      initial.ledger->records[2].pending_input == initial_pending,
      "last output incorrectly needed another pending-token draw");
  retire(initial);
  auto exact = request("one-last-address", prompt, 1, maximum - 1, 1);
  execute(exact);
  require(exact.progress.request_done() && exact.session->head->sampling.draw == maximum,
      "MAX-1/quota1 was incorrectly rejected or advanced past MAX");
  retire(exact);

  // New full-hit requests must not borrow their publisher's low draw address.
  auto publisher = request("publisher", prompt, 1, 0, 1);
  execute(publisher);
  q::cache::PrefixCache prefix(1);
  q::publish_generation_completed(model, publisher, &prefix);
  const auto published_cursor = publisher.session->head->sampling;
  auto hit = request("new-full-hit", prompt, 2, maximum - 1, 2);
  hit.input = publisher.completed_checkpoint.input;
  retire(publisher);
  const auto retained = model.snapshot_versions.live;
  exhausted([&] { (void)q::prepare_generation_call(model, hit, 2, &prefix, {}); });
  require(!hit.session && !hit.progress.turn_started() && !hit.ledger->committed_count &&
      !hit.ledger->sampling && !hit.prefix_hits && !hit.prefix_misses && !hit.prefix_full_hits &&
      hit.progress.input_turn() == SIZE_MAX && prefix.size() == 1 && model.snapshot_versions.live == retained,
      "full-hit address rejection changed lookup/import or request admission");
  hit.progress.initial_sampling.draw = maximum - 2;
  execute(hit, &prefix);
  require(hit.prefix_full_hits == 1 && hit.prefix_sampling_calls == 1 &&
      hit.session->head->sampling.draw == maximum - 1 && hit.progress.generated.size() == 1 &&
      published_cursor.draw == 1 && hit.session->head->sampling != published_cursor,
      "full hit used the publisher cursor or failed to count its first output");
  execute(hit, &prefix);
  require(hit.progress.request_done() && hit.session->head->sampling.draw == maximum &&
      hit.progress.generated.size() == 2, "full-hit final pending consumed an extra draw");
  retire(hit);
  full_hit_peer_preflight_retry(model, prefix, publisher.completed_checkpoint.input, prompt, maximum);
  prefix.evict_one();

  // A later unstarted turn uses its immutable new quota, not the previous
  // output_begin. Its live head and then original evicted ledger are decisive.
  const std::array turns{i::GenerationTurn{{1, 2}, 1}, i::GenerationTurn{{}, 2}};
  auto later = request("later-turn-budget", turns[0].prompt, 3, maximum - 2, 3);
  later.progress.bind_turns(turns);
  later.warm_inputs.push_back(std::make_shared<std::vector<std::int32_t>>(1, -1));
  q::preflight_generation_sampling(model, later);
  later.prompt = later.progress.begin_turn(*later.warm_inputs[0]);
  execute(later);
  const auto boundary = q::generation_boundary(*later.session->head);
  const auto *head = later.session->head.get();
  later.progress.advance_turn();
  (void)later.progress.initial_sampling.advanced(2); // Initial alone could fit.
  exhausted([&] { q::preflight_generation_sampling(model, later); });
  require(later.session->head.get() == head && q::generation_boundary(*later.session->head) == boundary &&
      !later.progress.turn_started() && later.progress.turn_index == 1 && later.progress.generated.size() == 1 &&
      later.ledger->committed_count == 1 && later.progress.input_turn() == 0 && !model.active_step,
      "later-turn address rejection changed its published boundary or host turn");
  retire(later);
  exhausted([&] { q::restore_generation_session(model, later, nullptr, false); });
  require(!later.session && !later.restore_count && !later.checkpoint_resume_hits &&
      !later.checkpoint_resume_misses && later.ledger->committed_count == 1 &&
      !later.progress.turn_started() && !model.snapshot_versions.live,
      "evicted request fell back to initial cursor or admitted recovery before rejection");

  const std::array legal_turns{i::GenerationTurn{{1, 2}, 1}, i::GenerationTurn{{}, 1}};
  auto neighbor = request("legal-later-turn", legal_turns[0].prompt, 2, maximum - 2, 2);
  neighbor.progress.bind_turns(legal_turns);
  neighbor.warm_inputs.push_back(std::make_shared<std::vector<std::int32_t>>(1, -1));
  q::preflight_generation_sampling(model, neighbor);
  neighbor.prompt = neighbor.progress.begin_turn(*neighbor.warm_inputs[0]); execute(neighbor);
  const auto pending = neighbor.session->head->next_id;
  neighbor.progress.advance_turn(); q::preflight_generation_sampling(model, neighbor);
  neighbor.prompt = neighbor.progress.begin_turn(*neighbor.warm_inputs[1], pending); execute(neighbor);
  require(neighbor.progress.request_done() && neighbor.progress.generated.size() == 2 &&
      neighbor.session->head->sampling.draw == maximum && neighbor.ledger->records[1].ids().size() == 1 &&
      neighbor.ledger->records[1].ids()[0] == pending && neighbor.session->length() == 3,
      "legal pending-only neighbor consumed pending/output twice");
  retire(neighbor);

  // Raw per-episode API cannot know a whole-turn quota, but must reject its
  // one EmitNext draw before installing active_step or touching shared inputs.
  const std::array rows{std::size_t{1}};
  const std::array<std::int32_t, 1> token{1};
  auto raw = model.create_session(rows, {}, q::GraphRole::kText, q::cache::SamplingCursor{123, 10, maximum});
  const auto *raw_head = raw->head.get(); const auto raw_boundary = q::generation_boundary(*raw->head);
  const auto admissions = model.shape_plan.admissions();
  exhausted([&] { model.prepare_evaluation(*raw, token); });
  require(!model.active_step && !model.carrier_borrowed && !raw->pending && raw->head.get() == raw_head &&
      q::generation_boundary(*raw->head) == raw_boundary && raw->retirement_status.ok() &&
      model.shape_plan.admissions() == admissions, "raw exhausted draw mutated an actual episode or its head");
  model.prepare_evaluation(*raw, token, nullptr, q::GraphRole::kText, q::TextCompletion::kStateOnly);
  model.submit_evaluation(*raw);
  (void)q::finish_evaluation(model, *raw, "max-state-only", {}, false);
  require(raw->length() == 1 && raw->head->sampling.draw == maximum && raw->head->next_id == -1,
      "raw state-only at exhausted output cursor spent a draw");
  raw->stop().IgnoreError(); const auto raw_retired = raw->try_retire(); q::check(raw_retired.status);
  require(raw_retired.retired, "raw address case did not retire");
  auto stale = request("retired-session-owner", prompt, 1, 0, 1);
  stale.session = raw;
  bool stale_rejected = false;
  try { q::preflight_generation_sampling(model, stale); }
  catch (const std::runtime_error &error) {
    stale_rejected = std::string_view(error.what()) == "ordinary sampling admission requires a live idle Session head";
  }
  require(stale_rejected && !raw->head && !stale.progress.turn_started(),
      "retired Session wrapper was dereferenced or readmitted");
  stale.session.reset(); raw.reset();

  auto greedy = request("greedy-independent", prompt, 2, maximum, 2, 1);
  execute(greedy); execute(greedy);
  require(greedy.progress.request_done() && greedy.progress.generated.size() == 2 &&
      greedy.session->head->sampling == q::cache::SamplingCursor{}, "greedy inherited random address exhaustion");
  retire(greedy);
  require(!model.snapshot_versions.live && !model.active_step && !model.carrier_borrowed &&
      model.draft_programs.empty() && model.prefix_sampling_programs.empty(), "sampling-budget EOF retained state or MTP resources");
  std::cout << "{\"status\":\"PASS\",\"backend\":\"cpu\",\"ordinary\":true,"
      "\"initial_rejection_atomic\":true,\"same_request_cursor_retry\":true,\"MAX_minus_1_quota_1\":true,"
      "\"full_hit_request_cursor\":true,\"later_head_budget\":true,\"evicted_ledger_budget\":true,"
      "\"full_hit_peer_preflight_recovery\":true,\"full_hit_recovery_sampling_calls\":1,\"full_hit_delivery_once\":true,"
      "\"state_only_no_draw\":true,\"raw_draw_preflight\":true,\"greedy_unchanged\":true,"
      "\"pending_only_single_consumption\":true,\"final_versions\":0,\"GPU\":false,\"HTTP\":0}\n";
  return 0;
}
} // namespace qwen_sampling_budget_case
