#pragma once

// Controlled accepted prefixes use the real target Graph, original KV/Store
// transactions and publication. The Commands arm keeps the original separate
// repair; the CUDA arm computes repair inside its original verifier Graph.
namespace qwen_recursive_repair_case {
namespace q = qwen35;
namespace h = qwen_host_image_case;
namespace s = qwen_mtp_batch_state_case;
using h::require;

inline void prefill(q::Model &model, const std::shared_ptr<q::Session> &session) {
  const std::array<std::int32_t, 3> ids{1, 2, 3};
  model.prepare_evaluation(*session, ids, nullptr, q::GraphRole::kMtpPrefill);
  model.submit_evaluations();
  const std::array outputs{q::EvaluationOutput{.session = session}};
  std::array<std::int32_t, 1> next{};
  model.finish_evaluations(outputs, next, [](auto, const auto &) {});
}

template<class Complete = std::nullptr_t>
void selected(q::Model &model, const std::shared_ptr<q::Session> &session,
    std::size_t rows, std::size_t accepted, Complete complete = nullptr) {
  const std::array ids{session->head->next_id, std::int32_t{4}, std::int32_t{5}, std::int32_t{6}};
  model.prepare_evaluation(*session, std::span(ids).first(rows), nullptr, q::GraphRole::kVerify);
  require(model.active_step->phase->recursive_repair == (model.replay_policy != q::ReplayPolicy::kCommands),
      "repair case did not select its original in-graph or serial control");
  model.submit_evaluations();
  const std::array outputs{q::EvaluationOutput{.session = session}};
  std::array<std::int32_t, 1> next{};
  model.finish_selected_evaluations(outputs, next,
      [&](std::span<const std::int32_t> candidates, q::StepState &step, std::span<std::int32_t> result) {
        step.members[0].accepted_tokens = accepted;
        result[0] = candidates[accepted - 1];
      }, [&](q::StepState &step) {
        model.repair_recursive_draft(step);
        if constexpr (!std::is_null_pointer_v<Complete>) complete(step);
      }, [](auto, const auto &) {});
}

inline std::vector<std::int32_t> natural(q::Model &model, const std::shared_ptr<q::Session> &session) {
  const auto pending = session->head->next_id;
  const auto ids = model.propose_recursive(*session, 3);
  model.prepare_evaluation(*session, ids, nullptr, q::GraphRole::kVerify);
  model.submit_evaluations();
  const std::array selections{q::MtpGenerationSelection{session, pending, 4, {}, 3}};
  const std::array outputs{q::EvaluationOutput{.session = session}};
  std::array<q::MtpRoundResult, 1> results;
  model.finish_recursive_mtp(selections, {}, outputs, results, [](auto, const auto &) {});
  return results[0].emitted;
}

// The actual request adapter grants output credit, submits the private Source
// successor and retires it on stop/failure. A low-credit RequestLoop must wait
// for acknowledgement without submitting more native work.
inline void request_boundaries(q::Model &model) {
  namespace i = hyle::inference;
  constexpr std::size_t outputs = 12;
  const std::array<std::int32_t, 3> prompt{1, 2, 3};
  for (const auto mode : {0, 1, 2}) {
    std::vector<q::GenerationRequest> requests;
    requests.emplace_back("recursive-boundary", prompt, std::filesystem::path{}, outputs);
    auto &request = requests.front();
    request.warm_inputs.push_back(std::make_shared<std::vector<std::int32_t>>(prompt.begin(), prompt.end()));
    request.prompt = *request.warm_inputs.front();
    request.input = q::cache::token_plan(prompt, model.capacity);
    request.ledger = std::make_unique<q::GenerationLedger>(outputs, 4);
    request.round_emitted.resize(outputs - 1);
    for (auto &row : request.round_emitted) row.reserve(4);
    request.turn_observations.resize(1);
    q::GenerationContinuation continuation; continuation.enabled = true; continuation.prepare(1);
    const auto unused = [](i::RequestWave, std::span<std::size_t>) -> i::RequestWaveResult {
      throw std::logic_error("manual recursive boundary cannot execute an admission wave");
    };
    q::GenerationRequestAdapter admission{model, nullptr, std::span(requests), false, nullptr, unused, {}, &continuation};
    i::RequestWindow window; window.prepare(1, 1); request.request_key = window.enqueue(0);
    admission.admission_preflight(request.request_key);
    bool interrupted = false, submitted = false;
    std::size_t commits = 0;
    q::cache::SnapshotRef checkpoint;
    std::optional<s::StateImage> before;
    std::vector<std::int32_t> ids;
    hyle::util::ScopeGuard cleanup{[&]() noexcept {
      continuation.reset(); (void)model.abort_episode();
      if (request.session) {
        request.session->stop().IgnoreError(); (void)request.session->drain();
        (void)request.session->try_retire(); request.session.reset();
      }
    }};
    while (!request.progress.turn_done && !interrupted) {
      std::array calls{q::prepare_generation_call(model, request, outputs, nullptr, requests)};
      if (calls.front().generation) {
        checkpoint = request.session->checkpoint(); before = s::image(model, *checkpoint);
        ids = request.progress.generated; commits = request.ledger->committed_count;
      }
      struct Refused {};
      try {
        q::execute_generation_calls(model, calls, false, nullptr, requests, [&] {
          if (!model.recursive_ahead || !model.recursive_ahead->submitted || !mode) return;
          submitted = true; interrupted = true;
          if (mode == 1) q::check(request.session->stop());
          else throw Refused{};
        }, &continuation);
      } catch (const Refused &) { require(mode == 2 && interrupted, "unrelated request failure escaped"); }
    }
    if (mode) {
      require(submitted && interrupted && before && !model.active_step && !model.recursive_ahead &&
          model.private_publications_retired() && s::image(model, *request.session->head) == *before &&
          s::image(model, *checkpoint) == *before && request.progress.generated == ids &&
          request.ledger->committed_count == commits,
          "cancelled or failed recursive Source successor published state/output or escaped real retirement");
    } else {
      require(request.progress.generated.size() == outputs && request.progress.turn_done,
          "request following Source cancellation/failure failed to reach its real quota");
      std::size_t rows = 0;
      for (std::size_t at = 0; at < request.ledger->committed_count; ++at)
        rows += request.ledger->records[at].draft_rows;
      require(request.session->submitted_draft_rows == rows,
          "recursive request lost speculative draft work from its original ledger");
    }
    checkpoint = {}; continuation.reset(); h::retire(request.session); cleanup.dismiss();
    require(!model.snapshot_versions.live && model.private_publications_retired(),
        "recursive request boundary retained a version or native source");
  }

  const std::array turns{i::GenerationTurn{std::vector<std::int32_t>(prompt.begin(), prompt.end()), outputs}};
  std::vector<q::GenerationRequest> requests;
  requests.emplace_back("recursive-pressure", prompt, std::filesystem::path{}, outputs);
  auto &request = requests.front(); request.progress.bind_turns(turns);
  request.warm_inputs.push_back(std::make_shared<std::vector<std::int32_t>>(prompt.begin(), prompt.end()));
  request.input = q::cache::token_plan(prompt, model.capacity);
  request.ledger = std::make_unique<q::GenerationLedger>(outputs, 4);
  request.round_emitted.resize(outputs - 1);
  for (auto &row : request.round_emitted) row.reserve(4);
  request.turn_observations.resize(1);
  i::OutputDelivery delivery(4); i::RequestLoop loop; loop.prepare(1, 1);
  q::GenerationContinuation continuation; continuation.enabled = true; continuation.prepare(1);
  const auto ready = [&](i::RequestWave wave) {
    std::array calls{q::prepare_generation_call(model, request, outputs, nullptr, requests,
        loop.credit(request.request_key, request.progress.generated.size()))};
    q::execute_generation_calls(model, calls, false, nullptr, requests, nullptr, &continuation,
        nullptr, true, wave.credits);
    q::publish_generation_prompt(model, calls.front(), nullptr);
  };
  const auto execute = [&](i::RequestWave wave, std::span<std::size_t> remaining) {
    return q::advance_generation_wave(model, wave, remaining, false, continuation, ready, nullptr, nullptr);
  };
  q::GenerationRequestAdapter adapter{model, nullptr, std::span(requests), false, nullptr, execute, {}, &continuation};
  request.request_key = loop.enqueue(0, &delivery);
  bool waited = false;
  const auto deadline = q::Clock::now() + std::chrono::seconds(30);
  while (loop.matches(request.request_key)) {
    require(q::Clock::now() < deadline, "recursive output-credit journey did not complete");
    const auto pumped = loop.pump_once(adapter);
    require(!pumped.cancelled, "recursive pressure journey unexpectedly cancelled");
    if (pumped.waiting_output) {
      const auto rows = model.submitted_draft_rows, graphs = model.draft_program_graph_calls;
      const auto observed = request.progress.generated.size(), committed = request.ledger->committed_count;
      const auto again = loop.pump_once(adapter);
      require(again.waiting_output && !again.executed && model.submitted_draft_rows == rows &&
          model.draft_program_graph_calls == graphs && request.progress.generated.size() == observed &&
          request.ledger->committed_count == committed,
          "backpressure submitted or published work without an actual output credit");
      waited = true;
      require(loop.acknowledge(request.request_key, observed, adapter),
          "original request owner rejected a current output acknowledgement");
    }
  }
  continuation.reset();
  require(waited && request.progress.request_done() && request.progress.generated.size() == outputs &&
      !request.session && !model.snapshot_versions.live && model.private_publications_retired(),
      "recursive pressure journey lost output, credit or native retirement");
}

inline void relay_contract(const std::filesystem::path &input, q::BackendKind kind) {
  auto left = q::prepare_weight_source(input / "weights.tsv", true);
  auto right = q::prepare_weight_source(input / "weights.tsv", true);
  const auto geometry = left.roles.geometry;
  const auto plan = q::cache::plan(geometry, left.cache_row_bytes, 32, 3, 4,
      q::recurrent_bytes(q::state_schema_for(geometry, 32), geometry), 1, 4, true, false, 2, 2, 2);
  q::Model control(std::move(left), geometry, 32, plan, {}, false,
      q::ReplayPolicy::kCommands, SIZE_MAX, {}, 0, {.primary = kind});
  q::Model candidate(std::move(right), geometry, 32, plan, {}, false,
      q::ReplayPolicy::kAuto, SIZE_MAX, {}, 0, {.primary = kind});
  const std::array prompt{std::size_t{3}};
  for (auto *model : {&control, &candidate}) {
    model->configure_draft_policy({3, 1});
    model->prepare_family(prompt, false, 1, q::GraphRole::kMtpPrefill);
    model->plan_replay_uses(3, 1, 64, q::GraphRole::kMtpPrefill);
    for (std::size_t rows = 1; rows <= 4; ++rows) {
      const std::array shape{rows};
      model->prepare_family(shape, false, 1, q::GraphRole::kVerify);
      model->plan_replay_uses(rows, 1, 64, q::GraphRole::kVerify);
    }
    model->prepare_recursive_round_programs(0, false, 64);
    const std::array single{std::size_t{1}};
    for (const auto role : {q::GraphRole::kMtpSingle, q::GraphRole::kMtpCatchupSingle}) {
      model->prepare_family(single, false, 1, role); model->plan_replay_uses(1, 1, 16, role);
    }
    model->freeze_shape_plan();
    model->prepare_execution(prompt, 1, q::GraphRole::kMtpPrefill);
    for (std::size_t rows = 1; rows <= 4; ++rows) {
      const std::array shape{rows}; model->prepare_execution(shape, 1, q::GraphRole::kVerify);
    }
    for (const auto role : {q::GraphRole::kMtpSingle, q::GraphRole::kMtpCatchupSingle}) model->prepare_execution(single, 1, role);
  }
  auto a = control.create_session(prompt, {}, q::GraphRole::kMtpPrefill);
  auto b = candidate.create_session(prompt, {}, q::GraphRole::kMtpPrefill);
  prefill(control, a); prefill(candidate, b);
  const auto prepare = [&]() {
    const auto begin_rows = b->submitted_draft_rows, begin_calls = candidate.draft_program_graph_calls;
    std::size_t prepaid = 0;
    const auto pending = b->head->next_id;
    const auto ids = candidate.propose_recursive(*b, 3, {}, nullptr, nullptr, {}, {}, nullptr,
        [&](std::span<const std::int32_t> prefix, const std::shared_ptr<q::StepSources> &source, bool ready) {
          if (ready) { candidate.finish_device_verification_preparation(); return; }
          std::array<std::int32_t, 4> shape{}; shape[0] = pending;
          std::copy(prefix.begin(), prefix.end(), shape.begin());
          const std::array requests{q::EvaluationRequest{b, std::span(shape).first(prefix.size() + 1),
              nullptr, {}, {}, {}, nullptr, {}, 0, 0, source}};
          candidate.prepare_evaluations(requests, nullptr, q::GraphRole::kVerify);
        }, &prepaid);
    require(ids.size() == 1 && candidate.active_step && !candidate.active_step->submitted,
        "unit-probability relay must keep its actual zero-proposal verifier");
    auto &member = candidate.active_step->members[0];
    member.verification_inputs = ids; member.prepaid_draft_rows = prepaid;
    member.proposal_input_rows = b->submitted_draft_rows - begin_rows;
    member.proposal_graph_calls = candidate.draft_program_graph_calls - begin_calls;
    candidate.prepare_submission_sources(); candidate.submit_evaluations();
    const auto successor = candidate.prepare_recursive_ahead(candidate.active_step);
    if (successor == q::inference::StepSuccessor::kPrepared) candidate.submit_recursive_ahead();
    return successor;
  };
  const auto finish = [&](q::Model &model, const std::shared_ptr<q::Session> &session) {
    const std::array selections{q::MtpGenerationSelection{session, session->head->next_id, 1, {}, 3}};
    const std::array outputs{q::EvaluationOutput{.session = session}};
    std::array<q::MtpRoundResult, 1> result;
    model.finish_recursive_mtp(selections, {}, outputs, result, [](auto, const auto &) {});
    return std::move(result[0]);
  };
  bool injected = false;
  for (std::size_t round = 0; round < 12; ++round) {
    auto checkpoint = b->checkpoint(); const auto old = s::image(candidate, *checkpoint);
    auto begin_rows = b->submitted_draft_rows;
    const auto successor = prepare();
    if (!injected && round >= 3 && successor == q::inference::StepSuccessor::kPrepared) {
      struct Refused {};
      bool failed = false;
      const std::array outputs{q::EvaluationOutput{.session = b}};
      std::array<std::int32_t, 1> next{};
      try {
        candidate.finish_selected_evaluations(outputs, next,
            [](auto rows, q::StepState &step, auto result) {
              step.members[0].accepted_tokens = 1; result[0] = rows[0];
            }, [](q::StepState &) { throw Refused{}; }, [](auto, const auto &) {});
      } catch (const Refused &) { failed = true; }
      require(failed && !candidate.active_step && candidate.private_publications_retired() &&
          s::image(candidate, *b->head) == old && s::image(candidate, *checkpoint) == old,
          "failed verifier published state or abandoned its real native draft successor");
      injected = true; begin_rows = b->submitted_draft_rows; (void)prepare();
    }
    const auto ids = control.propose_recursive(*a, 3);
    require(ids.size() == 1, "Commands probability control unexpectedly retained a proposal");
    control.prepare_evaluation(*a, ids, nullptr, q::GraphRole::kVerify); control.submit_evaluations();
    const auto expected = finish(control, a), actual = finish(candidate, b);
    candidate.retain_recursive_ahead(true);
    require(expected.emitted == actual.emitted && s::image(control, *a->head) == s::image(candidate, *b->head) &&
        s::image(candidate, *checkpoint) == old && actual.processed_draft == b->submitted_draft_rows - begin_rows,
        "relayed zero-proposal round changed complete state, checkpoint or actual work accounting");
  }
  const auto retired = candidate.discard_recursive_ahead();
  require(injected && retired.retired && retired.status.ok() && candidate.recursive_ahead_submissions > 0 &&
      candidate.recursive_ahead_hits > 0 && candidate.private_publications_retired(),
      "relay contract did not exercise real native overlap, reuse, failed publication and retirement");
  h::retire(a); h::retire(b);
  require(!control.snapshot_versions.live && !candidate.snapshot_versions.live,
      "relay contract retained a committed state version");
  request_boundaries(candidate);
}

inline int run(const std::filesystem::path &input, q::BackendKind kind) {
  require(kind == q::BackendKind::kCuda, "recursive repair contract requires CUDA");
  auto left_source = q::prepare_weight_source(input / "weights.tsv", true);
  auto right_source = q::prepare_weight_source(input / "weights.tsv", true);
  const auto geometry = left_source.roles.geometry;
  require(geometry.hidden == 64 && geometry.vocabulary == 64 && geometry.context_limit >= 32,
      "recursive repair contract requires the cached GA/H64/V64/C32 fixture");
  const auto plan = q::cache::plan(geometry, left_source.cache_row_bytes, 32, 3, 4,
      q::recurrent_bytes(q::state_schema_for(geometry, 32), geometry), 1, 4, true, false, 1, 2, 2);
  q::Model control(std::move(left_source), geometry, 32, plan, {}, false,
      q::ReplayPolicy::kCommands, SIZE_MAX, {}, 0, {.primary = kind});
  q::Model candidate(std::move(right_source), geometry, 32, plan, {}, false,
      q::ReplayPolicy::kAuto, SIZE_MAX, {}, 0, {.primary = kind});
  const std::array prompt{std::size_t{3}};
  for (auto *model : {&control, &candidate}) {
    model->configure_draft_policy({3, 0});
    model->prepare_family(prompt, false, 1, q::GraphRole::kMtpPrefill);
    model->plan_replay_uses(3, 1, 64, q::GraphRole::kMtpPrefill);
    for (const auto rows : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{4}}) {
      const std::array shape{rows};
      model->prepare_family(shape, false, 1, q::GraphRole::kVerify);
      model->plan_replay_uses(rows, 1, 64, q::GraphRole::kVerify);
    }
    model->prepare_recursive_round_programs(0, false, 64);
    model->prepare_execution(prompt, 1, q::GraphRole::kMtpPrefill);
    for (const auto rows : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{4}}) {
      const std::array shape{rows};
      model->prepare_execution(shape, 1, q::GraphRole::kVerify);
    }
  }
  std::size_t prefixes = 0;
  for (const bool lagged : {false, true}) for (const auto rows : {std::size_t{1}, std::size_t{4}})
      for (std::size_t accepted = 1; accepted <= rows; ++accepted) {
    auto a = control.create_session(prompt, {}, q::GraphRole::kMtpPrefill);
    auto b = candidate.create_session(prompt, {}, q::GraphRole::kMtpPrefill);
    prefill(control, a); prefill(candidate, b);
    if (lagged) { selected(control, a, 3, 2); selected(candidate, b, 3, 2); }
    auto old_a = a->checkpoint(), old_b = b->checkpoint();
    const auto before_a = s::image(control, *old_a), before_b = s::image(candidate, *old_b);
    require(before_a == before_b, "repair seed differs from its independent Commands control");
    const auto epoch_a = a->head.epoch(), epoch_b = b->head.epoch();
    selected(control, a, rows, accepted); selected(candidate, b, rows, accepted);
    require(a->head.epoch() == epoch_a + 1 && b->head.epoch() == epoch_b + 1 &&
        s::image(control, *a->head) == s::image(candidate, *b->head),
        "in-graph repair changed an accepted complete state or its publication epoch");
    require(s::image(control, *old_a) == before_a && s::image(candidate, *old_b) == before_b,
        "in-graph repair mutated an independently retained original checkpoint");
    require(natural(control, a) == natural(candidate, b) &&
        s::image(control, *a->head) == s::image(candidate, *b->head),
        "accepted repair did not continue with the same greedy outputs and complete state");
    old_a = {}; old_b = {}; h::retire(a); h::retire(b); ++prefixes;
  }
  {
    auto session = candidate.create_session(prompt, {}, q::GraphRole::kMtpPrefill);
    prefill(candidate, session);
    auto old = session->checkpoint(); const auto before = s::image(candidate, *old);
    struct Refused {};
    bool refused = false;
    try { selected(candidate, session, 4, 2, [](auto &) { throw Refused{}; }); }
    catch (const Refused &) { refused = true; }
    require(refused && !candidate.active_step && !candidate.carrier_borrowed &&
        s::image(candidate, *session->head) == before && s::image(candidate, *old) == before,
        "failed publication leaked repaired state, changed a head or retained its original episode");
    selected(candidate, session, 4, 2); // Same owner can retry after actual retirement.
    old = {}; h::retire(session);
  }
  require(!control.snapshot_versions.live && !candidate.snapshot_versions.live &&
      !control.active_step && !candidate.active_step &&
      control.private_publications_retired() && candidate.private_publications_retired(),
      "repair contract retained native work or a complete state version");
  relay_contract(input, kind);
  std::cout << "{\"status\":\"PASS\",\"case\":\"recursive-repair\",\"accepted_prefixes\":" << prefixes
      << ",\"lag0_lag1\":true,\"complete_state_exact\":true,\"checkpoints_immutable\":true,"
         "\"greedy_continuations\":true,\"failed_publication_retry\":true,\"real_retirement\":true,"
         "\"relayed_zero_draft_rounds\":12,\"relayed_failure_retry\":true,\"actual_work_conserved\":true,"
         "\"recursive_request_cancel_failure\":true,\"recursive_output_backpressure\":true}\n";
  return 0;
}
} // namespace qwen_recursive_repair_case
