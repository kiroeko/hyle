#pragma once

// Real ordinary RequestLoop/Model consumer on the cached untrained GA fixture.
// Only descriptor metadata is invalid; the original ready Bool lease stays valid.
namespace qwen_logits_preflight_case {
namespace q = qwen35;
namespace i = hyle::inference;
namespace h = qwen_host_image_case;
namespace p = qwen_packed_prefill_case;
using h::require;

inline q::LogitsInput ready_mask(q::Model &model, std::int32_t selected) {
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(model.vocabulary));
  bytes.at(static_cast<std::size_t>(selected)) = 1;
  auto buffer = q::take(q::mem::Buffer::create(model.execution.device(), bytes.size(), 256));
  q::check(model.execution.write(buffer.view(), std::as_bytes(std::span(bytes))));
  return {q::dense(q::Type::kBool, {1, model.vocabulary}),
      q::take(model.execution.adopt_buffer(std::move(buffer)))};
}

inline bool same_record(const q::GenerationCommitRecord &a, const q::GenerationCommitRecord &b) {
  return a.same_result(b) && a.committed == b.committed && a.user_output == b.user_output &&
      a.imported == b.imported && a.turn == b.turn && a.input_owner == b.input_owner &&
      a.pending_input == b.pending_input && std::ranges::equal(a.prompt_ids, b.prompt_ids) &&
      a.step_seconds == b.step_seconds && a.endpoint == b.endpoint;
}

struct SavedRequest final {
  q::cache::SnapshotRef checkpoint;
  const q::cache::State *head = nullptr;
  std::uint64_t epoch = 0;
  h::ImageBytes image;
  std::vector<q::GenerationCommitRecord> records;
  std::vector<std::int32_t> generated;
  std::size_t commits = 0, observed = 0, consumed = 0, input_turn = 0, history_length = 0;
  const q::cache::ReplayHistory *history = nullptr;
  const q::PreparedSampling *sampling = nullptr;
};

inline SavedRequest save(q::Model &model, q::GenerationRequest &request,
    const std::shared_ptr<i::HostStateStorage> &storage) {
  require(bool(request.session) && bool(request.session->head),
      "logits preflight must preserve a real admitted, committed request");
  SavedRequest result;
  result.checkpoint = request.session->checkpoint(); result.head = request.session->head.get();
  result.epoch = request.session->head.epoch(); result.image = p::capture(model, *request.session, storage);
  result.records = request.ledger->records; result.generated = request.progress.generated;
  result.commits = request.ledger->committed_count; result.observed = request.progress.observed_commits();
  result.consumed = request.turn_input_consumed; result.input_turn = request.input_turn;
  result.history = request.history.get(); result.history_length = request.history ? request.history->length : 0;
  result.sampling = request.ledger->sampling;
  return result;
}

inline void unchanged(q::Model &model, q::GenerationRequest &request, const SavedRequest &before,
    const std::shared_ptr<i::HostStateStorage> &storage) {
  require(request.session && request.session->head.get() == before.head &&
      request.session->head.epoch() == before.epoch &&
      request.progress.generated == before.generated && request.progress.prefilled && !request.progress.turn_done &&
      request.progress.observed_commits() == before.observed && request.ledger->committed_count == before.commits &&
      request.turn_input_consumed == before.consumed && request.input_turn == before.input_turn &&
      request.history.get() == before.history && (!request.history || request.history->length == before.history_length) &&
      request.ledger->sampling == before.sampling && !request.ledger->successor_ordinal() &&
      request.ledger->records.size() == before.records.size(),
      "bad logits metadata changed the original head, epoch, progress, RNG or ledger identity");
  for (std::size_t at = 0; at < before.records.size(); ++at)
    require(same_record(request.ledger->records[at], before.records[at]),
        "bad logits metadata changed a committed or reserved ledger record");
  h::same_image(before.image, p::capture(model, *request.session, storage));
  p::check_old(model, before.checkpoint, before.image, storage);
}

inline int run(const std::filesystem::path &input, q::BackendKind kind) {
  auto source = q::prepare_weight_source(input / "weights.tsv", false);
  const auto geometry = source.roles.geometry;
  require(geometry.hidden == 64 && geometry.vocabulary == 64 && geometry.context_limit >= 16 &&
      geometry.recurrent_layers() && geometry.attention_layers(),
      "logits preflight needs the cached untrained C16 GA/H64/V64 fixture");
  constexpr std::size_t capacity = 16;
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, capacity, 4, 2,
      q::recurrent_bytes(q::state_schema_for(geometry, capacity), geometry), 2, 1, false, true);
  const q::Model::Sampling sampler{.enabled = true, .policy = {1.F, 0, 1.F}, .seed = 123,
      .modifiers = {.allowed = true}};
  q::Model model(std::move(source), geometry, capacity, plan, {}, false, q::ReplayPolicy::kCommands,
      SIZE_MAX, sampler, 0, {.primary = kind});
  const std::array shapes{std::size_t{1}, std::size_t{2}};
  for (const auto width : {std::size_t{1}, std::size_t{2}}) {
    model.prepare_family(shapes, false, width);
    for (auto rows : shapes) model.plan_replay_uses(rows, width, 6);
  }
  for (const auto width : {std::size_t{1}, std::size_t{2}}) model.prepare_execution(shapes, width);
  const auto storage = q::take(i::HostStateStorage::create(model.execution, 1, 4u << 20));
  const std::array<std::array<i::GenerationTurn, 1>, 2> turns{
      std::array{i::GenerationTurn{{1, 2}, 2}}, std::array{i::GenerationTurn{{5, 6}, 2}}};
  std::vector<q::GenerationRequest> requests; requests.reserve(2);
  for (std::size_t member = 0; member < 2; ++member) {
    requests.emplace_back(member ? "bad-logits-B" : "valid-logits-A", turns[member][0].prompt,
        std::filesystem::path{}, 2);
    auto &request = requests.back(); request.progress.bind_turns(turns[member]);
    request.progress.initial_sampling = {123, 41 + member, 0};
    request.input = q::cache::token_plan(turns[member][0].prompt, capacity);
    request.ledger = std::make_unique<q::GenerationLedger>(q::generation_record_capacity(turns[member]), 1);
    request.round_emitted.resize(1); request.round_emitted[0].reserve(1);
    request.turn_observations.resize(1);
    request.warm_inputs.push_back(std::make_shared<std::vector<std::int32_t>>(turns[member][0].prompt));
    request.logits.allowed = ready_mask(model, static_cast<std::int32_t>(3 + member));
  }
  auto &a = requests[0]; auto &b = requests[1];
  std::array delivery{i::OutputDelivery(1), i::OutputDelivery(1)};
  i::RequestLoop loop; loop.prepare(2, 2);
  std::size_t submissions = 0, executed_waves = 0, rejected_key_position = SIZE_MAX;
  const auto execute = [&](std::span<const i::RequestKey> active) {
    std::vector<q::PreparedGenerationCall> calls; calls.reserve(active.size());
    for (std::size_t member = 0; member < active.size(); ++member) {
      const auto key = active[member]; auto &request = requests[key.slot];
      if (request.logits.allowed.description.rank() == 1) rejected_key_position = member;
      calls.push_back(q::prepare_generation_call(model, request, request.progress.output_limit(), nullptr,
          requests, loop.credit(key, request.progress.generated.size())));
    }
    const auto submitted = [&] {
      require(model.active_step && model.active_step->submitted && model.active_step->execution.completion.valid(),
          "logits case did not observe the original native submission");
      require((submissions == 0 && model.active_step->members.size() == 2 &&
          model.active_step->members[0].session == a.session && model.active_step->members[1].session == b.session) ||
          (submissions == 1 && model.active_step->members.size() == 1 &&
          model.active_step->members[0].session == a.session),
          "logits case did not execute its actual R2 prefill followed by the same original R1 peer");
      ++submissions;
    };
    q::execute_generation_calls(model, calls, false, nullptr, requests, submitted);
    for (auto &call : calls) q::publish_generation_prompt(model, call, nullptr);
    ++executed_waves;
  };
  q::SynchronousGenerationWave wave{model, execute};
  q::GenerationRequestAdapter adapter{model, nullptr, std::span(requests), false,
      static_cast<std::vector<std::size_t> *>(nullptr), wave};
  a.request_key = loop.enqueue(0, &delivery[0]); b.request_key = loop.enqueue(1, &delivery[1]);
  const auto key_a = a.request_key, key_b = b.request_key;
  const auto first = loop.pump_once(adapter);
  require(first.executed && first.waiting_output && !first.waiting_retirement && !first.completed &&
      submissions == 1 && executed_waves == 1 && loop.active().empty() && loop.matches(key_a) && loop.matches(key_b) &&
      a.progress.generated == std::vector<std::int32_t>{3} && b.progress.generated == std::vector<std::int32_t>{4} &&
      a.session && b.session && a.session->head->sampling.draw == 1 && b.session->head->sampling.draw == 1 &&
      a.ledger->committed_count == 1 && b.ledger->committed_count == 1,
      "valid R2 prefill did not produce the original two credit-paused requests");

  auto saved_a = save(model, a, storage), saved_b = save(model, b, storage);
  // Wake B first: its original logical slot is 1 but its next batch position is
  // 0. A failure must retain the complete B key, not manufacture an index key.
  require(loop.acknowledge(key_b, 1, adapter) && loop.acknowledge(key_a, 1, adapter),
      "valid prefill output acknowledgements were rejected");
  const auto original_description = b.logits.allowed.description;
  b.logits.allowed.description = q::dense(q::Type::kBool, {model.vocabulary});
  require(b.logits.allowed.lease.valid(), "descriptor-only corruption lost its original ready mask");
  bool typed_failure = false;
  try { (void)loop.pump_once(adapter); }
  catch (const i::RequestExecutionFailure &failure) {
    const auto &evaluation = failure.evaluation();
    typed_failure = failure.request_key() == key_b && evaluation.status().IsInvalidArgument() &&
        !evaluation.status().has_native_error() && !evaluation.participant() && !evaluation.retirement_observed();
    std::cerr << "logits_preflight key_slot=" << (failure.request_key() ? failure.request_key()->slot : SIZE_MAX)
        << " batch_position=" << rejected_key_position << " status=" << evaluation.status().message()
        << " participant=" << (evaluation.participant() ? static_cast<long long>(*evaluation.participant()) : -1)
        << " retirement_observed=" << evaluation.retirement_observed() << '\n';
  }
  require(typed_failure && rejected_key_position == 0 && key_b.slot == 1 && submissions == 1 && executed_waves == 1 &&
      loop.matches(key_a) && loop.matches(key_b) && a.request_key == key_a && b.request_key == key_b &&
      delivery[0].acknowledged() == 1 && delivery[1].acknowledged() == 1 &&
      !model.active_step && !model.successor_step && !model.carrier_borrowed,
      "bad descriptor was not a key-qualified preflight failure before any new native work");
  unchanged(model, a, saved_a, storage); unchanged(model, b, saved_b, storage);
  b.logits.allowed.description = original_description;
  q::check(loop.cancel(key_b, adapter));
  const auto continued = loop.pump_once(adapter);
  require(continued.executed && continued.cancelled == 1 && !continued.waiting_retirement && continued.waiting_output &&
      submissions == 2 && executed_waves == 2 && !loop.matches(key_b) && !b.session && loop.matches(key_a) &&
      a.request_key == key_a && a.progress.generated == std::vector<std::int32_t>({3, 3}) &&
      a.ledger->committed_count == 2 && a.progress.observed_commits() == 2 &&
      a.ledger->records[1].after.length == 3 && a.ledger->records[1].after.sampling.draw == 2 &&
      b.progress.generated == saved_b.generated && b.ledger->committed_count == saved_b.commits,
      "cancelling the bad request did not let the same-key peer continue exactly once");
  p::check_old(model, saved_a.checkpoint, saved_a.image, storage);
  p::check_old(model, saved_b.checkpoint, saved_b.image, storage);
  require(loop.acknowledge(key_a, 2, adapter), "final peer output acknowledgement was rejected");
  const auto last = loop.pump_once(adapter);
  require(last.completed == 1 && !last.executed && !last.waiting_output && !last.waiting_retirement &&
      !loop.matches(key_a) && !loop.matches(key_b) && loop.active().empty() && !loop.waiting_front() &&
      !a.session && !b.session && submissions == 2,
      "final ACK did not release the original two logical request identities");
  saved_a.checkpoint.reset(); saved_b.checkpoint.reset();
  require(!model.snapshot_versions.live && !model.active_step && !model.successor_step && !model.carrier_borrowed &&
      !storage->live_images() && model.page_inventory.free_ids.size() == plan.max_pages,
      "logits preflight consumer retained a state version, page, native episode or host image");
  std::cout << "{\"status\":\"PASS\",\"backend\":\"" << h::backend_name(kind)
      << "\",\"case\":\"logits-preflight\",\"synthetic_untrained\":true,\"requests\":2,"
      "\"valid_prefill_submissions\":1,\"preflight_submissions\":0,\"peer_decode_submissions\":1,"
      "\"typed_invalid_argument\":true,\"original_key\":true,\"batch_position\":0,\"key_slot\":1,"
      "\"complete_state_unchanged\":true,\"all_ledger_records_unchanged\":true,\"old_checkpoints_unchanged\":true,"
      "\"same_key_peer_continued\":true,\"allfalse_numeric_retested\":false,\"final_versions\":0,"
      "\"quality_claim\":false,\"performance_claim\":false}\n";
  return 0;
}
} // namespace qwen_logits_preflight_case
