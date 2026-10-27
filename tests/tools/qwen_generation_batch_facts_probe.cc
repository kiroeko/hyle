#include "../../engine/models/qwen35/main.cc"

// Finite local Model consumer of actual generation-batch observations. Both
// widths use real original proposals; taking their prefix is fixture control,
// not an assertion that natural model EOS shortened either proposal.
namespace {
namespace q = qwen35;
namespace i = hyle::inference;

constexpr std::size_t kContext = 16, kOutputLimit = 6, kCredit = 3;

void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}

struct BatchCopy final {
  q::GraphRole role;
  q::TextCompletion completion;
  bool generation = false;
  std::size_t rows = 0, members = 0;
  std::array<i::RequestKey, 2> keys{};
  std::array<double, 2> member_seconds{};
  double seconds = 0;
};

struct CaseResult final {
  std::size_t submits = 0, observations = 0, verified_rows = 0, outputs = 0;
  bool mapped_error = false;
};

CaseResult run_case(q::Model &model, std::span<const std::int32_t> prompt, bool split,
                    bool controlled_error = false) {
  require(!controlled_error || split, "controlled child failure requires the real mixed-width split");
  require(!model.active_step && !model.successor_step && !model.carrier_borrowed &&
      !model.snapshot_versions.live, "preceding batch-facts case retained an original episode or state");
  const std::array turns{i::GenerationTurn{{prompt.begin(), prompt.end()}, kOutputLimit}};
  std::array deliveries{i::OutputDelivery(kCredit), i::OutputDelivery(kCredit)};
  i::RequestLoop owner;
  owner.prepare(2, model.cache_plan.resident_sessions);
  std::vector<q::GenerationRequest> requests;
  requests.reserve(2);
  for (std::size_t member = 0; member < 2; ++member) {
    requests.emplace_back(member ? "B" : "A", turns[0].prompt, std::filesystem::path{}, kOutputLimit);
    auto &request = requests.back();
    request.progress.bind_turns(turns);
    request.ledger = std::make_unique<q::GenerationLedger>(q::generation_record_capacity(turns), 3);
    request.round_emitted.resize(kOutputLimit - 1);
    for (auto &row : request.round_emitted) row.reserve(3);
    request.turn_observations.resize(1);
    request.warm_inputs.push_back(std::make_shared<std::vector<std::int32_t>>(prompt.begin(), prompt.end()));
    request.prompt = request.progress.begin_turn(*request.warm_inputs[0]);
    request.request_key = owner.enqueue(member, &deliveries[member]);
  }
  const std::array keys{requests[0].request_key, requests[1].request_key};
  require(keys[0] != keys[1] && owner.matches(keys[0]) && owner.matches(keys[1]),
      "batch-facts fixture did not obtain distinct original request identities");
  hyle::util::ScopeGuard cleanup{[&]() noexcept {
    (void)model.abort_episode();
    for (auto &request : requests) if (request.session) {
      request.session->stop().IgnoreError(); (void)request.session->drain();
      (void)request.session->try_retire(); request.session.reset();
    }
  }};

  std::vector<q::PreparedGenerationCall> prefill;
  prefill.reserve(2);
  for (auto &request : requests)
    prefill.push_back(q::prepare_generation_call(model, request, request.progress.output_limit(),
        nullptr, requests, owner.credit(request.request_key, request.progress.generated.size())));
  q::execute_generation_calls(model, prefill, false, nullptr, requests);
  for (std::size_t member = 0; member < 2; ++member) {
    auto &request = requests[member];
    require(request.progress.prefilled && request.progress.generated.size() == 1 &&
        request.ledger->committed_count == 1 && request.progress.observed_commits() == 1 &&
        !request.progress.turn_done && request.session && !request.session->pending,
        "real R2 prefill did not publish and observe one nonterminal output per original owner");
    deliveries[member].acknowledge_prefix(request.progress.generated.size(), 1);
    require(owner.credit(keys[member], request.progress.generated.size()) == kCredit,
        "prefill acknowledgement did not restore the original bounded delivery credit");
  }
  const std::array before{q::generation_boundary(*requests[0].session->head),
      q::generation_boundary(*requests[1].session->head)};
  const std::array epochs{requests[0].session->head.epoch(), requests[1].session->head.epoch()};
  const std::array heads{requests[0].session->head.get(), requests[1].session->head.get()};
  const std::array prefill_records{requests[0].ledger->records[0], requests[1].ledger->records[0]};
  std::vector<q::PreparedGenerationCall> calls;
  calls.reserve(2);
  for (std::size_t member = 0; member < 2; ++member) {
    auto &request = requests[member];
    calls.push_back(q::prepare_generation_call(model, request, request.progress.output_limit(),
        nullptr, requests, owner.credit(keys[member], request.progress.generated.size())));
    auto &call = calls.back();
    require(call.role == q::GraphRole::kVerify && call.generation && call.maximum == 2 && call.quota == kCredit,
        "both prepared calls must enter the same real K3 verifier bucket and bounded output quota");
    const auto begin = q::Clock::now();
    auto proposals = model.propose_recursive(*request.session, call.maximum, q::model_eos_tokens);
    call.proposal_seconds = std::chrono::duration<double>(q::Clock::now() - begin).count();
    require(proposals.size() == 3 && proposals.front() == before[member].next_id,
        "local fixture must supply three actual proposal inputs without natural EOS shortening");
    // Original B becomes the first physical child, exercising identity/order
    // mapping rather than assuming the outer bucket's first request ran first.
    if (split && member == 1) proposals.resize(2);
    call.verification_ids = std::move(proposals);
  }
  const auto expected_rows = [&](std::size_t member) { return split && member == 1 ? 2u : 3u; };
  const auto expected_graphs = split ? std::size_t{2} : std::size_t{1};
  std::vector<BatchCopy> observed;
  observed.reserve(expected_graphs);
  std::size_t submits = 0;
  bool injected = false;
  const auto controlled_status = hyle::Status::InvalidArgument("controlled child");
  const auto after_submit = [&] {
    require(submits < expected_graphs && model.active_step && model.active_step->submitted &&
        !model.active_step->cancelled && model.active_step->phase &&
        model.active_step->phase->family->role == q::GraphRole::kVerify,
        "after-submit did not observe the actual original target-verifier episode");
    const auto &step = *model.active_step;
    const auto rows = split && submits == 0 ? 2u : 3u;
    require(step.input_tokens == rows && step.members.size() == (split ? 1u : 2u),
        "original Model submission did not use the real child width and member count");
    for (std::size_t member = 0; member < step.members.size(); ++member) {
      const auto original = split ? (submits == 0 ? 1u : 0u) : member;
      require(step.members[member].session == requests[original].session &&
          step.members[member].first_input == before[original].next_id,
          "actual child submission lost its original Session or pending input");
    }
    ++submits;
    if (controlled_error) {
      require(submits == 1 && step.members[0].session == requests[1].session,
          "controlled error did not follow the real first B-child submission");
      injected = true;
      // A host control failure after a real accepted submit. This is not an
      // injected native/device fault and grants no retirement by itself.
      throw i::EvaluationFailure(controlled_status, 0);
    }
  };
  const auto batch_observer = [&](const q::GenerationBatchFacts &facts) {
    require(!controlled_error, "failed unpublished child produced a successful batch observation");
    require(observed.size() < expected_graphs && submits == observed.size() + 1 &&
        facts.role == q::GraphRole::kVerify && facts.completion == q::TextCompletion::kEmitNext &&
        facts.generation && facts.members.size() == (split ? 1u : 2u) &&
        facts.input_rows == (split && observed.empty() ? 2u : 3u) &&
        std::isfinite(facts.step_seconds) && facts.step_seconds > 0,
        "batch observation did not describe one actual completed original child");
    BatchCopy copy{facts.role, facts.completion, facts.generation, facts.input_rows, facts.members.size()};
    copy.seconds = facts.step_seconds;
    for (std::size_t member = 0; member < facts.members.size(); ++member) {
      const auto original = split ? (observed.empty() ? 1u : 0u) : member;
      const auto &call = facts.members[member];
      require(call.request == &requests[original] && call.request->request_key == keys[original] &&
          owner.matches(call.request->request_key) && call.ids().size() == facts.input_rows &&
          call.disposition == q::EvaluationDisposition::kPublished &&
          std::isfinite(call.step_seconds) && call.step_seconds > 0,
          "borrowed child facts lost original key, input width or actual completion");
      copy.keys[member] = call.request->request_key;
      copy.member_seconds[member] = call.step_seconds;
    }
    // Only values leave this callback. Neither facts.members nor any borrowed
    // call/request pointer is retained past the synchronous observation.
    observed.push_back(copy);
  };
  bool mapped_error = false;
  try {
    q::execute_generation_calls(model, calls, false, nullptr, requests, after_submit, nullptr, batch_observer);
  } catch (const i::EvaluationFailure &failure) {
    if (!controlled_error) throw;
    require(injected && failure.status() == controlled_status && !failure.status().has_native_error() &&
        failure.participant() == 1 && failure.retirement_observed() &&
        failure.retirement().retired && failure.retirement().status.ok(),
        "split child failure lost its original status, parent member index or actual retirement proof");
    require(calls[1].request == &requests[1], "failed child lost its parent bucket's original B identity");
    const auto attributed = i::RequestExecutionFailure::from_evaluation(failure, calls.size(),
        [&](std::size_t member) { return calls[member].request->request_key; });
    require(attributed.request_key() == keys[1] && attributed.evaluation().status() == controlled_status &&
        attributed.evaluation().participant() == 1 && attributed.evaluation().retirement_observed() &&
        attributed.evaluation().retirement().retired && attributed.evaluation().retirement().status.ok(),
        "parent-bucket attribution did not preserve original B identity and cleanup facts");
    mapped_error = true;
  }
  CaseResult result{submits, observed.size(), 0, 0};
  if (controlled_error) {
    require(mapped_error && submits == 1 && observed.empty() && !model.active_step &&
        !model.successor_step && !model.carrier_borrowed,
        "controlled child was not rejected and actually retired before the second group ran");
    for (std::size_t member = 0; member < 2; ++member) {
      const auto &request = requests[member];
      model.validate_mtp_snapshot(*request.session->head);
      require(request.request_key == keys[member] && owner.matches(keys[member]) &&
          request.session->head.get() == heads[member] && request.session->head.epoch() == epochs[member] &&
          q::generation_boundary(*request.session->head) == before[member] && !request.session->pending &&
          request.ledger->committed_count == 1 && request.progress.observed_commits() == 1 &&
          request.ledger->records[0].committed && request.ledger->records[0].same_result(prefill_records[member]) &&
          !request.ledger->records[1].committed && request.rounds.empty() &&
          request.progress.generated.size() == 1 && request.progress.generated.back() == before[member].next_id &&
          deliveries[member].acknowledged() == 1 &&
          owner.credit(keys[member], request.progress.generated.size()) == kCredit &&
          deliveries[member].pending(request.progress.generated).empty(),
          "controlled child failure changed a published head, authoritative ledger, output or delivery credit");
    }
    result.mapped_error = true;
  } else {
    require(!mapped_error && submits == expected_graphs && observed.size() == expected_graphs &&
        !model.active_step && !model.successor_step && !model.carrier_borrowed,
        "split/combined verifier did not report and retire exactly its actual Graph submissions");
    for (std::size_t member = 0; member < 2; ++member) {
      auto &request = requests[member];
      model.validate_mtp_snapshot(*request.session->head);
      const auto &record = request.ledger->records[1];
      require(request.request_key == keys[member] && owner.matches(keys[member]) &&
          request.ledger->committed_count == 2 && request.progress.observed_commits() == 2 &&
          record.committed && record.role == q::GraphRole::kVerify && record.generation &&
          record.requests == (split ? 1u : 2u) && record.member_index == (split ? 0u : member) &&
          record.verified_rows == expected_rows(member) && record.accepted > 0 && record.accepted <= record.verified_rows &&
          record.before == before[member] && record.after == q::generation_boundary(*request.session->head) &&
          record.quota == kCredit && record.emitted_count > 0 && record.emitted_count <= kCredit &&
          record.emitted_count <= record.emitted.size() && request.rounds.size() == 1 &&
          request.rounds[0].verified_rows == record.verified_rows &&
          request.session->head.epoch() == epochs[member] + 1 &&
          request.session->head->sampling == before[member].sampling &&
          request.session->length() == before[member].length + record.accepted && !request.session->pending &&
          std::ranges::equal(record.verification_inputs, calls[member].verification_ids),
          "actual child publication changed original ledger geometry, complete boundary or greedy RNG");
      require(request.progress.generated.size() == 1 + record.emitted_count &&
          std::equal(record.emitted.begin(), record.emitted.begin() + record.emitted_count,
              request.progress.generated.begin() + 1) &&
          request.progress.generated.back() == request.session->head->next_id &&
          deliveries[member].acknowledged() == 1 &&
          owner.credit(keys[member], request.progress.generated.size()) == kCredit - record.emitted_count &&
          deliveries[member].pending(request.progress.generated).size() == record.emitted_count,
          "batch reporting advanced delivery, repeated output or exceeded the original request credit");
      result.verified_rows += record.verified_rows; result.outputs += record.emitted_count;
    }
  }
  // Destroy the original call containers before inspecting our copied facts.
  // The observer's stored representation cannot retain its temporary span.
  calls.clear(); prefill.clear();
  if (!controlled_error)
    require(observed[0].keys[0] == keys[split ? 1u : 0u] &&
        (split ? observed[1].keys[0] == keys[0] : observed[0].keys[1] == keys[1]),
        "value-copied original RequestKeys did not survive borrowed call-container destruction");
  for (auto &request : requests) {
    q::check(request.session->stop());
    const auto retired = request.session->drain();
    q::check(retired.status); require(retired.retired, "batch-facts Session did not actually retire");
    request.session.reset();
  }
  require(!model.snapshot_versions.live, "batch observations kept original model versions alive");
  return result;
}

int run(const std::filesystem::path &input, q::BackendKind backend) {
  auto source = q::prepare_weight_source(input / "weights.tsv", true);
  const auto geometry = source.roles.geometry;
  require(geometry.context_limit >= kContext && geometry.vocabulary >= 16 && geometry.mtp_layers == 1,
      "batch-facts probe requires its cached C16/MTP fixture");
  auto prompt = q::read_ids(input / "prompt.ids", geometry.vocabulary, kContext);
  require(prompt.size() >= 3, "batch-facts fixture requires at least three real prompt IDs");
  prompt.resize(3);
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, kContext, 6, 3,
      q::recurrent_bytes(q::state_schema_for(geometry, kContext), geometry), 2, 3, true);
  q::Model model(std::move(source), geometry, kContext, plan, {}, false, q::ReplayPolicy::kCommands,
      SIZE_MAX, {}, 0, {.primary = backend});
  model.configure_draft_policy({2, 0});
  const std::array prompt_rows{prompt.size()};
  const std::array verify_rows{std::size_t{2}, std::size_t{3}};
  for (const auto width : {std::size_t{1}, std::size_t{2}}) {
    model.prepare_family(prompt_rows, false, width, q::GraphRole::kMtpPrefill);
    model.plan_replay_uses(prompt.size(), width, 3, q::GraphRole::kMtpPrefill);
    model.prepare_family(verify_rows, false, width, q::GraphRole::kVerify);
    for (const auto rows : verify_rows) model.plan_replay_uses(rows, width, 3, q::GraphRole::kVerify);
  }
  model.prepare_recursive_round_programs();
  model.freeze_shape_plan();
  for (const auto width : {std::size_t{1}, std::size_t{2}}) {
    model.prepare_execution(prompt_rows, width, q::GraphRole::kMtpPrefill);
    model.prepare_execution(verify_rows, width, q::GraphRole::kVerify);
  }
  const auto split = run_case(model, prompt, true);
  const auto same = run_case(model, prompt, false);
  const auto failed = run_case(model, prompt, true, true);
  require(split.submits == 2 && split.verified_rows == 5 && same.submits == 1 && same.verified_rows == 6,
      "actual child facts confused Graph count with member participation or packed target rows");
  require(failed.mapped_error && failed.submits == 1 && failed.observations == 0 &&
      failed.verified_rows == 0 && failed.outputs == 0,
      "controlled submitted child was reported as published work or attributed to original A");
  std::cout << "{\"status\":\"PASS\",\"backend\":\""
      << (backend == q::BackendKind::kCuda ? "cuda" : "cpu")
      << "\",\"cases\":3,\"mixed_width_graphs\":" << split.submits
      << ",\"mixed_width_rows\":" << split.verified_rows << ",\"same_width_graphs\":" << same.submits
      << ",\"same_width_rows\":" << same.verified_rows
      << ",\"controlled_error_submits\":" << failed.submits
      << ",\"original_request_keys\":true,\"borrowed_spans_retained\":false,\"bounded_output_credit\":true,"
         "\"controlled_child_maps_to_B\":true,\"controlled_error_actually_retired\":true,"
         "\"native_error_injection\":false,\"ledger_and_outputs_checked\":true,\"final_live_versions\":0,\"natural_eos_shortening\":false,"
         "\"performance_claim\":false}\n";
  return 0;
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 1) { std::cerr << "SKIP: pass --cpu|--cuda PREPARED_LOCAL_INPUTS\n"; return 77; }
    if (argc != 3 || (std::string_view(argv[1]) != "--cpu" && std::string_view(argv[1]) != "--cuda"))
      throw std::runtime_error("usage: qwen_generation_batch_facts_probe --cpu|--cuda PREPARED_LOCAL_INPUTS");
    const auto backend = std::string_view(argv[1]) == "--cuda" ? q::BackendKind::kCuda : q::BackendKind::kCpu;
    const auto compiled = q::Backend::compiled_backends();
    if (std::find(compiled.begin(), compiled.end(), backend) == compiled.end()) {
      std::cerr << "SKIP: selected backend is not built\n"; return 77;
    }
    return run(argv[2], backend);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
