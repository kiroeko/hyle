#pragma once

// Real CPU Model/Graph controls on the existing untrained A/H64 fixture.
// Submission is proved; this synchronous CPU case does not claim GPU overlap.
namespace qwen_request_control_case {
namespace q = qwen35;
namespace i = hyle::inference;

inline void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}

inline int run(const std::filesystem::path &input) {
  const auto geometry = q::read_weight_geometry(input / "weights.tsv");
  require(geometry.hidden == 64 && geometry.intermediate == 128 &&
      geometry.vocabulary == 64 && geometry.layers == "A",
      "request-control case requires the cached untrained A/H64/FFN128/V64 fixture");
  const q::cache::KvRowBytes bytes{
      q::cache::multiply(q::cache::multiply(static_cast<std::size_t>(geometry.kv_heads), static_cast<std::size_t>(geometry.key_depth)), 2),
      q::cache::multiply(q::cache::multiply(static_cast<std::size_t>(geometry.kv_heads), static_cast<std::size_t>(geometry.value_depth)), 2)};
  const auto plan = q::cache::plan(geometry, bytes, 16, 3, 2,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 2, 1, false, true);
  q::Model model(input / "weights.tsv", geometry, 16, plan, {}, false,
      q::ReplayPolicy::kCommands, 16u << 20,
      {.enabled = true, .policy = {1.F, 8, .9F}, .seed = 123}, 0,
      {.primary = i::BackendKind::kCpu});
  const std::array shapes{std::size_t{1}, std::size_t{2}};
  for (const auto width : {std::size_t{1}, std::size_t{2}}) {
    model.prepare_family(shapes, false, width);
    for (auto count : shapes) model.plan_replay_uses(count, width, 12);
  }
  for (const auto width : {std::size_t{1}, std::size_t{2}}) model.prepare_execution(shapes, width);
  const std::array long_turns{i::GenerationTurn{{1, 2}, 3}};
  const std::array short_turns{i::GenerationTurn{{8}, 1}};
  std::vector<q::GenerationRequest> requests;
  requests.reserve(4);
  for (std::size_t index = 0; index < 4; ++index) {
    const auto turns = index < 2 ? std::span<const i::GenerationTurn>(long_turns)
                                 : std::span<const i::GenerationTurn>(short_turns);
    requests.emplace_back(std::string(1, static_cast<char>('A' + index)), turns[0].prompt,
        std::filesystem::path{}, turns[0].max_outputs);
    auto &request = requests.back();
    request.progress.bind_turns(turns);
    request.progress.initial_sampling = {123, 100 + index, 0};
    request.input = q::cache::token_plan(std::span<const std::int32_t>(turns[0].prompt), 16);
    request.ledger = std::make_unique<q::GenerationLedger>(q::generation_record_capacity(turns), 1);
    request.round_emitted.resize(request.progress.maximum_outputs() - 1);
    for (auto &emitted : request.round_emitted) emitted.reserve(1);
    request.turn_observations.resize(1);
    request.warm_inputs.push_back(std::make_shared<std::vector<std::int32_t>>(turns[0].prompt));
  }
  auto &a = requests[0]; auto &b = requests[1]; auto &c = requests[2]; auto &d = requests[3];
  std::array delivery{i::OutputDelivery(1), i::OutputDelivery(1), i::OutputDelivery(1), i::OutputDelivery(1)};
  i::RequestLoop loop;
  loop.prepare(requests.size(), 2);
  q::cache::PrefixCache prefix(1);
  bool cancel_after_submit = true, original_survivor = false, cancelled_unpublished = false;
  std::size_t physical_waves = 0;
  // Cancellation only needs the model's original stop operation. The full
  // adapter below owns the normal request stages and uses this same Session.
  struct StopView final {
    std::span<q::GenerationRequest> requests;
    hyle::Status stop(i::RequestKey key) const noexcept {
      return requests[key.slot].session ? requests[key.slot].session->stop() : hyle::Status::Ok();
    }
  } stop_view{requests};
  const auto execute = [&](std::span<const i::RequestKey> active) {
    std::vector<q::PreparedGenerationCall> calls;
    calls.reserve(active.size());
    for (auto key : active) {
      auto &request = requests[key.slot];
      calls.push_back(q::prepare_generation_call(model, request, request.progress.output_limit(),
          &prefix, requests, loop.credit(key, request.progress.generated.size())));
    }
    const q::cache::State *survivor = nullptr;
    const auto submitted = [&] {
      if (!cancel_after_submit) return;
      require(active.size() == 2 && active[0] == a.request_key && active[1] == b.request_key &&
          model.active_step && model.active_step->submitted && model.active_step->completion.valid(),
          "control did not observe the original submitted R2 episode");
      const auto old = q::generation_boundary(*a.session->head);
      const auto epoch = a.session->head.epoch();
      survivor = model.active_step->members[1].next.get();
      q::check(loop.cancel(a.request_key, stop_view));
      require(loop.active().size() == 2 && model.active_step->discard_stopped_members &&
          !model.active_step->cancelled && a.session->head.epoch() == epoch &&
          q::generation_boundary(*a.session->head) == old && !a.ledger->committed_count,
          "cancel changed the borrowed wave or published its stopped member");
    };
    q::execute_generation_calls(model, calls, false, &prefix, requests, submitted);
    if (cancel_after_submit) {
      original_survivor = b.session->head.get() == survivor && b.ledger->committed_count == 1;
      cancelled_unpublished = calls[0].disposition == q::EvaluationDisposition::kCancelled &&
          calls[1].disposition == q::EvaluationDisposition::kPublished &&
          a.progress.generated.empty() && !a.ledger->committed_count && !a.history->length;
      require(original_survivor && cancelled_unpublished,
          "cancelled call reached high-level output or replaced its survivor candidate");
      cancel_after_submit = false;
    }
    for (auto &call : calls) q::publish_generation_prompt(model, call, &prefix);
    ++physical_waves;
  };
  q::GenerationRequestAdapter adapter{model, &prefix, std::span(requests), false,
      static_cast<std::vector<std::size_t> *>(nullptr), execute};
  a.request_key = loop.enqueue(0, &delivery[0]);
  b.request_key = loop.enqueue(1, &delivery[1]);
  d.request_key = loop.enqueue(3, &delivery[3]);
  const auto key_a = a.request_key, key_b = b.request_key, key_d = d.request_key;
  q::check(loop.cancel(key_d, adapter));
  const auto first = loop.pump_once(adapter);
  require(first.completed == 2 && first.cancelled == 2 && !loop.matches(key_a) && !loop.matches(key_d) &&
      !a.session && !d.session && loop.matches(key_b) && b.session &&
      b.progress.generated.size() == 1 && b.session->head->sampling.draw == 1 &&
      delivery[1].available(1) == 0 && loop.active().empty() && first.waiting_output,
      "queued/submitted cancellation or credit pause returned the wrong ownership");
  require(!loop.acknowledge(key_a, 0, adapter) && loop.cancel(key_a, adapter).IsNotFound(),
      "retired request accepted a stale control");
  const auto idle = loop.pump_once(adapter);
  require(!idle.executed && idle.waiting_output && !idle.waiting_retirement && physical_waves == 1,
      "credit exhaustion resubmitted work or masqueraded as native retirement");

  c.request_key = loop.enqueue(2, &delivery[2]);
  const auto key_c = c.request_key;
  const auto peer = loop.pump_once(adapter);
  require(peer.executed && !peer.completed && peer.waiting_output && !peer.waiting_retirement &&
      !c.session && loop.matches(key_c) && c.progress.generated.size() == 1,
      "native completion released final undelivered output or kept its device Session");
  require(loop.acknowledge(key_b, 1, adapter), "valid credit ACK was refused");
  const auto resumed = loop.pump_once(adapter);
  require(resumed.executed && !resumed.completed && resumed.waiting_output &&
      b.progress.generated.size() == 2 && b.session->head->sampling.draw == 2 &&
      b.ledger->committed_count == 2 && loop.matches(key_c) && !c.session && physical_waves == 3,
      "a peer's final pending output blocked runnable decode or replayed its own result");

  q::check(loop.cancel(key_b, adapter));
  const auto stopped = loop.pump_once(adapter);
  require(!stopped.executed && !stopped.completed && stopped.waiting_output && !stopped.waiting_retirement &&
      !b.session && loop.matches(key_b) && b.progress.generated.size() == 2 &&
      delivery[1].acknowledged() == 1 && b.ledger->records[1].after.sampling.draw == 2,
      "paused cancellation dropped final pending output or executed another token");
  bool invalid_ack = false;
  try { (void)loop.acknowledge(key_b, 3, adapter); }
  catch (const std::out_of_range &) { invalid_ack = true; }
  require(invalid_ack && delivery[1].acknowledged() == 1 && loop.matches(key_b),
      "invalid ACK mutated delivery or request ownership");
  require(loop.acknowledge(key_c, 1, adapter) && loop.acknowledge(key_b, 2, adapter),
      "final committed output ACK was refused");
  const auto last = loop.pump_once(adapter);
  require(last.completed == 2 && last.cancelled == 1 && !last.executed && !last.waiting_output &&
      !last.waiting_retirement && !loop.matches(key_b) && !loop.matches(key_c) &&
      loop.active().empty() && !loop.waiting_front() && physical_waves == 3 &&
      !model.active_step && !model.carrier_borrowed,
      "final ACK failed to retire each logical identity exactly once");
  while (prefix.evict_one()) {}
  require(!model.snapshot_versions.live, "prefix cleanup retained a model version");
  std::cout << "{\"status\":\"PASS\",\"fixture\":\"untrained_A_H64\","
      "\"queued_cancel\":true,\"submitted_partial_cancel\":true,\"original_survivor\":true,"
      "\"cancelled_unpublished\":true,\"credit_pause_and_ack_wake\":true,"
      "\"final_output_retained\":true,\"unacked_peer_does_not_block\":true,"
      "\"physical_waves\":3,\"released_requests\":4,\"cancelled_requests\":3,\"final_versions\":0}\n";
  return 0;
}
} // namespace qwen_request_control_case
