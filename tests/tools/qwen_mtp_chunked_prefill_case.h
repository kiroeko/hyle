#pragma once

#include "qwen_mtp_ragged_prefill_case.h"

// Bounded journeys share the cached GA fixture and the original execution,
// request ledger, reconstruction and host-image paths. No acceptance/quality or
// performance claim follows from this untrained controlled-prefix check.
namespace qwen_mtp_chunked_prefill_case {
namespace q = qwen35;
namespace h = qwen_host_image_case;
namespace p = qwen_packed_prefill_case;
namespace s = qwen_mtp_batch_state_case;
namespace r = qwen_mtp_ragged_prefill_case;
namespace i = hyle::inference;
using h::require;

inline void state_only_ports(const q::Phase &phase) {
  require(phase.family->completion == q::TextCompletion::kStateOnly && !phase.logits_facts &&
      !phase.logits_port && !phase.id_port && !phase.sampling_identity_port && !phase.sampling_valid_port,
      "MTP state-only carrier contains a score, sampler or output-token port");
}

inline std::int32_t raw(q::Model &model, const std::shared_ptr<q::Session> &session,
    std::span<const std::int32_t> ids, q::GraphRole role, q::TextCompletion completion,
    bool packed_prefill = false) {
  const auto epoch = session->head.epoch();
  if (packed_prefill) {
    const std::array requests{q::EvaluationRequest{session, ids}};
    model.prepare_prefill_evaluations(requests, nullptr, role, completion);
    require(q::packs_prefill(model.active_step->phase->family->packing) && model.active_step->phase->query_rows == 4,
        "state-only MTP prefill did not retain its real packed R4 carrier");
  } else model.prepare_evaluation(*session, ids, nullptr, role, completion);
  if (completion == q::TextCompletion::kStateOnly) state_only_ports(*model.active_step->phase);
  if (role == q::GraphRole::kMtpCatchupSingle && completion == q::TextCompletion::kStateOnly) {
    const auto &step = *model.active_step;
    require(ids.size() == 1 && step.input_tokens == 1 && step.phase->query_rows == 1 &&
        !q::packs_prefill(step.phase->family->packing) && step.members.size() == 1 &&
        step.members[0].layout.facts().query_rows == 1 && step.members[0].draft_layout.facts().query_rows == 2 &&
        step.members[0].next->length == session->length() + 1 &&
        step.members[0].next->draft->frontier == session->head->draft->frontier + 2,
        "StateOnly CatchupSingle did not prepare its true one target and two draft rows");
  }
  try {
    model.submit_evaluation(*session);
    const std::array outputs{q::EvaluationOutput{.session = session}};
    std::array<std::int32_t, 1> next{-1};
    model.finish_evaluations(outputs, next, [](auto, const auto &) {});
    require(session->head.epoch() == epoch + 1 && !session->pending && !model.active_step && !model.carrier_borrowed,
        "MTP chunk did not publish once and retire its original episode");
    return next[0];
  } catch (...) { (void)model.abort_episode(); throw; }
}

inline void seed_lag1(q::Model &model, const std::shared_ptr<q::Session> &session) {
  const std::array<std::int32_t, 3> prompt{3, 2, 1};
  (void)raw(model, session, prompt, q::GraphRole::kMtpPrefill, q::TextCompletion::kEmitNext);
  const auto epoch = session->head.epoch();
  const auto cursor = session->head->sampling;
  const std::array ids{session->head->next_id, std::int32_t{4}, std::int32_t{5}};
  model.prepare_evaluation(*session, ids, nullptr, q::GraphRole::kVerify); model.submit_evaluation(*session);
  const std::array outputs{q::EvaluationOutput{.session = session}};
  std::array<std::int32_t, 1> next{-1};
  model.finish_selected_evaluations(outputs, next,
      [](std::span<const std::int32_t> candidates, q::StepState &step, std::span<std::int32_t> selected) {
        require(candidates.size() == 3 && step.input_tokens == 3 && step.members.size() == 1,
            "lag1 seed did not execute the original K3 target verifier");
        step.members[0].accepted_tokens = 2; selected[0] = candidates[1];
      }, [&](q::StepState &step) { model.repair_recursive_draft(step); }, [](auto, const auto &) {});
  require(session->length() == 5 && session->head->draft->frontier == 4 && session->head->draft->lag == 1 &&
      session->head->draft->catchup_token == ids[1] && session->head->prior_hidden.valid() &&
      session->head->sampling == cursor && session->head.epoch() == epoch + 1 && session->head->next_id == next[0] &&
      !model.active_step && !model.carrier_borrowed, "controlled prefix did not preserve its real lag1 boundary");
}

inline int run(const std::filesystem::path &input, q::BackendKind kind) {
  auto exact_source = q::prepare_weight_source(input / "weights.tsv", true);
  auto chunk_source = q::prepare_weight_source(input / "weights.tsv", true);
  const auto geometry = exact_source.roles.geometry;
  require(geometry.context_limit >= 16 && geometry.vocabulary >= 16 && geometry.mtp_layers == 1 &&
      geometry.recurrent_layers() && geometry.attention_layers(), "MTP chunked case requires the cached C16 GA fixture");
  // The independent exact catch-up consumes five target and six draft rows.
  const auto plan = q::cache::plan(geometry, exact_source.cache_row_bytes, 16, 6, 6,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 1, 3, true, true);
  const q::Model::Sampling sampler{.enabled = true, .policy = {1.F, 8, .9F}, .seed = 123};
  q::Model exact(std::move(exact_source), geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands,
      SIZE_MAX, sampler, 0, {.primary = kind});
  q::Model chunk(std::move(chunk_source), geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands,
      SIZE_MAX, sampler, 0, {.primary = kind});
  const std::array exact_prompts{std::size_t{3}, std::size_t{4}};
  const std::array seed{std::size_t{3}}, single{std::size_t{1}};
  const std::array catchup{std::size_t{2}, std::size_t{5}};
  const std::array carrier{std::size_t{4}};
  exact.prepare_family(exact_prompts, false, 1, q::GraphRole::kMtpPrefill);
  for (const auto rows : exact_prompts) exact.plan_replay_uses(rows, 1, 2, q::GraphRole::kMtpPrefill);
  exact.prepare_family(catchup, false, 1, q::GraphRole::kMtpCatchupPrefill);
  for (const auto rows : catchup) exact.plan_replay_uses(rows, 1, 1, q::GraphRole::kMtpCatchupPrefill);
  chunk.prepare_family(seed, false, 1, q::GraphRole::kMtpPrefill);
  chunk.plan_replay_uses(3, 1, 1, q::GraphRole::kMtpPrefill);
  for (const auto completion : {q::TextCompletion::kStateOnly, q::TextCompletion::kEmitNext}) {
    chunk.prepare_family(single, false, 1, q::GraphRole::kMtpSingle, false, completion);
    chunk.plan_replay_uses(1, 1, 10, q::GraphRole::kMtpSingle, completion);
  }
  chunk.prepare_family(single, false, 1, q::GraphRole::kMtpCatchupSingle, false, q::TextCompletion::kStateOnly);
  chunk.plan_replay_uses(1, 1, 1, q::GraphRole::kMtpCatchupSingle, q::TextCompletion::kStateOnly);
  for (const auto role : {q::GraphRole::kMtpPrefill, q::GraphRole::kMtpCatchupPrefill}) {
    require(chunk.prepare_prefill_family(carrier, q::TextCompletion::kStateOnly, 0, 1, role),
        "selected route did not prepare its real state-only MTP prefill carrier");
    chunk.plan_replay_uses(4, 1, 2, role, q::TextCompletion::kStateOnly, 0, q::RowPacking::kPrefill);
  }
  for (auto *model : {&exact, &chunk}) {
    model->prepare_family(seed, false, 1, q::GraphRole::kVerify);
    model->plan_replay_uses(3, 1, 1, q::GraphRole::kVerify); model->prepare_draft_program(1, false);
  }
  exact.prepare_execution(exact_prompts, 1, q::GraphRole::kMtpPrefill);
  exact.prepare_execution(catchup, 1, q::GraphRole::kMtpCatchupPrefill);
  chunk.prepare_execution(seed, 1, q::GraphRole::kMtpPrefill);
  for (const auto completion : {q::TextCompletion::kStateOnly, q::TextCompletion::kEmitNext})
    chunk.prepare_execution(single, 1, q::GraphRole::kMtpSingle, completion);
  chunk.prepare_execution(single, 1, q::GraphRole::kMtpCatchupSingle, q::TextCompletion::kStateOnly);
  for (const auto role : {q::GraphRole::kMtpPrefill, q::GraphRole::kMtpCatchupPrefill})
    chunk.prepare_prefill_execution(carrier, q::TextCompletion::kStateOnly, 0, 1, role);
  for (auto *model : {&exact, &chunk}) model->prepare_execution(seed, 1, q::GraphRole::kVerify);
  auto exact_storage = q::take(i::HostStateStorage::create(exact.execution, 1, 4u << 20));
  auto chunk_storage = q::take(i::HostStateStorage::create(chunk.execution, 2, 4u << 20));
  p::StepReport report_a, report_b, report_c;
  std::size_t rebuild_rows = 0, rebuild_calls = 0;
  bool cancelled_inflight = false;

  // A: natural request, four one-row chunks. Prefix visibility and output
  // delivery follow the original committed ledger, including reconstruction.
  {
    const std::array<std::int32_t, 4> prompt{1, 2, 3, 4};
    const std::array control_plan{std::size_t{4}};
    auto control = exact.create_session(control_plan, {}, q::GraphRole::kMtpPrefill, q::cache::SamplingCursor{123, 7, 0});
    report_a = p::StepReport{0, 0, 4, 4}; report_a.prefill = true;
    report_a.exact_next = raw(exact, control, prompt, q::GraphRole::kMtpPrefill, q::TextCompletion::kEmitNext);
    const std::vector<i::GenerationTurn> turns{{std::vector<std::int32_t>(prompt.begin(), prompt.end()), 1}};
    std::vector<q::GenerationRequest> requests; requests.reserve(1); requests.emplace_back("state-only", prompt, std::filesystem::path{}, 1);
    auto &request = requests[0]; request.progress.bind_turns(turns); request.progress.initial_sampling = {123, 7, 0};
    request.ledger = std::make_unique<q::GenerationLedger>(q::generation_record_capacity(turns, 1), 1);
    request.turn_observations.resize(1); request.warm_inputs.push_back(std::make_shared<std::vector<std::int32_t>>(prompt.begin(), prompt.end()));
    request.input = q::cache::token_plan(prompt, 16); request.prompt = request.progress.begin_turn(*request.warm_inputs[0]);
    i::RequestWindow window; window.prepare(1, 1); request.request_key = window.enqueue(0); window.admitted(request.request_key);
    const auto key = request.request_key;
    q::cache::PrefixCache prefix(1);
    const auto prepare = [&] { return q::prepare_generation_call(chunk, request, 1, &prefix, requests, 1, 1); };
    const auto execute = [&] {
      auto call = prepare(); const auto epoch = request.session->head.epoch();
      require(call.role == q::GraphRole::kMtpSingle && call.ids().size() == 1 && !call.generation,
          "Q1 MTP natural prefill replaced Single with physical padding or generation");
      const bool state_only = !call.prefill_complete;
      require(call.completion == (state_only ? q::TextCompletion::kStateOnly : q::TextCompletion::kEmitNext),
          "Q1 request completion differs from its real final-input fact");
      std::array calls{std::move(call)};
      q::execute_generation_calls(chunk, calls, false, &prefix, requests, [&] {
        require(chunk.active_step && chunk.active_step->submitted && chunk.active_step->input_tokens == 1 &&
            chunk.active_step->phase->query_rows == 1 && !q::packs_prefill(chunk.active_step->phase->family->packing),
            "Q1 request did not submit the original exact Single episode");
        if (state_only) state_only_ports(*chunk.active_step->phase);
      });
      q::publish_generation_prompt(chunk, calls[0], &prefix);
      require(request.session->head.epoch() == epoch + 1 && request.request_key == key && !request.session->pending,
          "MTP request commit changed owner/key or advanced its epoch more than once");
      if (state_only) require(request.progress.generated.empty() && !request.progress.prefilled &&
          request.session->head->next_id == -1 && request.session->head->sampling.draw == 0 && !prefix.size() &&
          request.ledger->records[request.ledger->committed_count - 1].emitted_count == 0 &&
          !request.ledger->records[request.ledger->committed_count - 1].user_output,
          "MTP StateOnly sampled, emitted or published an incomplete prompt");
    };
    execute(); execute();
    require(request.progress.input_consumed() == 2 && request.ledger->committed_count == 2 && request.progress.observed_commits() == 2,
        "two real StateOnly chunks did not advance their original input cursor/ledger exactly once");
    const auto old = request.session->checkpoint(); const auto before = s::image(chunk, *old);
    const auto boundary = q::generation_boundary(*request.session->head);
    {
      auto cancelled = prepare();
      q::prepare_generation_record(*request.ledger, *request.session, cancelled.role, cancelled.ids(), false, 0, 0,
          request.warm_inputs[0], 0, cancelled.completion, cancelled.input_begin, cancelled.prefill_complete);
      chunk.prepare_evaluation(*request.session, cancelled.ids(), nullptr, cancelled.role, cancelled.completion);
      state_only_ports(*chunk.active_step->phase); chunk.submit_evaluation(*request.session);
      cancelled_inflight = chunk.active_step->execution.run->in_flight();
      const auto stopped = chunk.abort_episode();
      if (!stopped.status.IsCancelled() || stopped.status.has_native_error()) q::check(stopped.status);
      require(stopped.retired && !chunk.active_step && !chunk.carrier_borrowed && !request.session->pending &&
          q::generation_boundary(*request.session->head) == boundary && s::image(chunk, *request.session->head) == before &&
          request.progress.input_consumed() == 2 && request.ledger->committed_count == 2 && !request.ledger->records[2].committed &&
          request.progress.observed_commits() == 2 && request.progress.generated.empty() && !prefix.size(),
          "cancelled StateOnly chunk published state/output or advanced original input/ledger facts");
    }
    h::retire(request.session); q::restore_generation_session(chunk, request, &prefix, false, requests);
    rebuild_rows = request.restored.target_rows; rebuild_calls = request.restored.graph_calls;
    require(rebuild_rows == 2 && rebuild_calls == 2 && request.restored.draft_rows == 2 &&
        q::generation_boundary(*request.session->head) == boundary && s::image(chunk, *request.session->head) == before &&
        request.request_key == key && request.progress.input_consumed() == 2 && request.progress.observed_commits() == 2 &&
        request.progress.generated.empty() && !prefix.size(),
        "partial MTP StateOnly ledger reconstruction replayed output/RNG or changed complete state and request progress");
    execute(); execute();
    require(request.progress.request_done() && request.progress.prefilled && request.progress.generated.size() == 1 &&
        request.progress.input_consumed() == 4 && request.ledger->committed_count == 4 && request.session->length() == 4 &&
        request.session->head->draft->frontier == 4 && request.session->head->draft->lag == 0 && request.session->head->sampling.draw == 1 &&
        s::image(chunk, *old) == before && prefix.size() == 1, "final Single failed to emit and publish the complete prompt exactly once");
    auto hit = prefix.lookup(*request.input, 4, 4, &chunk, chunk.math_identity());
    require(bool(hit) && hit.payload.replay.chunks == 4 && hit.payload.replay.history->chunks[0].state_only &&
        hit.payload.replay.history->chunks[1].state_only && hit.payload.replay.history->chunks[2].state_only &&
        !hit.payload.replay.history->chunks[3].state_only,
        "MTP complete prefix directory lost its actual StateOnly chunk history");
    hit.reset(); prefix.evict_one();
    report_a.bucket_next = request.progress.generated[0]; report_a.epoch = static_cast<std::size_t>(request.session->head.epoch());
    p::compare(exact, p::capture(exact, *control, exact_storage), p::capture(chunk, *request.session, chunk_storage), report_a);
    window.release_turn(key, false); h::retire(request.session); window.release(key); h::retire(control);
    require(window.active().empty() && !window.waiting_front(), "bounded request retained its original window admission");
  }

  // B: controlled K3 acceptance produces a real lag1 base. One exact five-row
  // catch-up is the independent control for StateOnly 2+2 and final Single 1.
  {
    auto control = exact.create_session(seed, {}, q::GraphRole::kMtpPrefill, q::cache::SamplingCursor{123, 8, 0});
    auto candidate = chunk.create_session(seed, {}, q::GraphRole::kMtpPrefill, q::cache::SamplingCursor{123, 8, 0});
    seed_lag1(exact, control); seed_lag1(chunk, candidate);
    const auto old_a = control->checkpoint(), old_b = candidate->checkpoint();
    const auto before_a = s::image(exact, *old_a), before_b = s::image(chunk, *old_b);
    require(before_a.pending == before_b.pending && before_a.sampling == before_b.sampling,
        "lag1 controls do not share the same original pending/cursor");
    const std::array suffix{before_a.pending, std::int32_t{6}, std::int32_t{7}, std::int32_t{8}, std::int32_t{9}};
    report_b = p::StepReport{1, 5, 5, 9}; report_b.prefill = true;
    report_b.exact_next = raw(exact, control, suffix, q::GraphRole::kMtpCatchupPrefill, q::TextCompletion::kEmitNext);
    const auto first = raw(chunk, candidate, std::span(suffix).first(2), q::GraphRole::kMtpCatchupPrefill,
        q::TextCompletion::kStateOnly, true);
    require(first == -1 && candidate->length() == 7 && candidate->head->draft->frontier == 7 &&
        !candidate->head->draft->lag && candidate->head->sampling.draw == 1 && candidate->head->next_id == -1,
        "StateOnly catch-up consumed the wrong draft rows, retained lag1 or sampled early");
    const auto partial = candidate->checkpoint(); const auto partial_bytes = s::image(chunk, *partial);
    auto image = chunk.export_session(*candidate, chunk_storage); const auto host_bytes = h::read_image(*image);
    require(host_bytes.boundary.next_id == -1 && host_bytes.boundary.evaluated_next_id == -1 && host_bytes.boundary.draft &&
        host_bytes.boundary.draft->frontier == 7 && !host_bytes.boundary.draft->lag,
        "partial MTP HostImage fabricated a completed pending output");
    h::retire(candidate); candidate = chunk.restore_session(image);
    require(candidate->head->state_identity() != partial->state_identity() && s::image(chunk, *candidate->head) == partial_bytes,
        "partial MTP HostImage did not import independent original initialized state");
    auto roundtrip = chunk.export_session(*candidate, chunk_storage);
    h::same_image(host_bytes, h::read_image(*roundtrip)); roundtrip.reset(); image.reset();
    require(!chunk_storage->live_images(), "partial MTP HostImage reservations did not retire");
    const auto second = raw(chunk, candidate, std::span(suffix).subspan(2, 2), q::GraphRole::kMtpPrefill,
        q::TextCompletion::kStateOnly, true);
    require(second == -1 && candidate->length() == 9 && candidate->head->draft->frontier == 9 &&
        !candidate->head->draft->lag && candidate->head->sampling.draw == 1,
        "second StateOnly chunk replayed catch-up, padding or a random draw");
    report_b.bucket_next = raw(chunk, candidate, std::span(suffix).last(1), q::GraphRole::kMtpSingle, q::TextCompletion::kEmitNext);
    require(candidate->length() == 10 && candidate->head->draft->frontier == 10 && !candidate->head->draft->lag &&
        candidate->head->sampling.draw == 2 && s::image(exact, *old_a) == before_a && s::image(chunk, *old_b) == before_b &&
        s::image(chunk, *partial) == partial_bytes, "bounded lag1 continuation changed checkpoints, final state bounds or RNG");
    r::preserved_prefix(before_a, s::image(exact, *control->head));
    r::preserved_prefix(before_b, s::image(chunk, *candidate->head));
    r::preserved_prefix(partial_bytes, s::image(chunk, *candidate->head));
    report_b.epoch = static_cast<std::size_t>(candidate->head.epoch());
    p::compare(exact, p::capture(exact, *control, exact_storage), p::capture(chunk, *candidate, chunk_storage), report_b);
    h::retire(control); h::retire(candidate);

    // C reuses this same actual lag1 checkpoint; it does not reseed, change a
    // frontier or force a padded Prefill path for the missing one-row case.
    const std::array short_exact{std::size_t{2}}, short_single{std::size_t{1}};
    auto short_control = exact.create_session(short_exact, old_a, q::GraphRole::kMtpCatchupPrefill);
    auto short_candidate = chunk.create_session(short_single, old_b, q::GraphRole::kMtpCatchupSingle,
        {}, false, q::TextCompletion::kStateOnly);
    report_c = p::StepReport{2, 5, 2, 2}; report_c.prefill = true;
    report_c.exact_next = raw(exact, short_control, std::span(suffix).first(2),
        q::GraphRole::kMtpCatchupPrefill, q::TextCompletion::kEmitNext);
    const auto first_single = raw(chunk, short_candidate, std::span(suffix).first(1),
        q::GraphRole::kMtpCatchupSingle, q::TextCompletion::kStateOnly);
    require(first_single == -1 && short_candidate->head->next_id == -1 && short_candidate->length() == 6 &&
        short_candidate->head->draft->frontier == 6 && !short_candidate->head->draft->lag &&
        short_candidate->head->sampling == before_b.sampling,
        "StateOnly CatchupSingle sampled, kept lag1 or advanced an incorrect cache frontier");
    const auto short_partial = short_candidate->checkpoint(); const auto short_bytes = s::image(chunk, *short_partial);
    report_c.bucket_next = raw(chunk, short_candidate, std::span(suffix).subspan(1, 1),
        q::GraphRole::kMtpSingle, q::TextCompletion::kEmitNext);
    require(short_candidate->length() == 7 && short_candidate->head->draft->frontier == 7 &&
        !short_candidate->head->draft->lag && short_candidate->head->sampling == before_b.sampling.advanced(1) &&
        s::image(exact, *old_a) == before_a && s::image(chunk, *old_b) == before_b &&
        s::image(chunk, *short_partial) == short_bytes,
        "StateOnly CatchupSingle continuation changed a checkpoint, final frontier or original RNG");
    r::preserved_prefix(before_a, s::image(exact, *short_control->head));
    r::preserved_prefix(before_b, s::image(chunk, *short_candidate->head));
    r::preserved_prefix(short_bytes, s::image(chunk, *short_candidate->head));
    report_c.epoch = static_cast<std::size_t>(short_candidate->head.epoch());
    p::compare(exact, p::capture(exact, *short_control, exact_storage),
        p::capture(chunk, *short_candidate, chunk_storage), report_c);
    h::retire(short_control); h::retire(short_candidate);
  }
  for (auto *model : {&exact, &chunk}) {
    require(!model->snapshot_versions.live && !model->active_step && !model->successor_step && !model->carrier_borrowed &&
        model->page_inventory.free_ids.size() == plan.max_pages && model->draft_inventory.free_ids.size() == plan.max_pages,
        "MTP chunked journeys retained a state version, target/draft page permission or native episode");
    for (const auto &[key, program] : model->draft_programs) {
      (void)key;
      require(!program->guard.execution.has_completion() && !program->guard.execution.cleanup_pending &&
          !program->hidden_transactions.bound() && !program->carrier.pending(), "MTP controlled repair did not retire");
    }
  }
  require(!exact_storage->live_images() && !chunk_storage->live_images(), "MTP chunked journeys retained a host image");
  const bool equal = report_a.numerically_equivalent && report_b.numerically_equivalent && report_c.numerically_equivalent;
  std::cout << "{\"backend\":\"" << h::backend_name(kind) << "\",\"mode\":\"mtp_chunked_prefill\",\"synthetic_untrained\":true"
      << ",\"A\":{\"Q\":1,\"P\":4,\"commits\":4,\"state_only_commits\":3,\"emitted\":1,\"final_draw\":1,\"final_length\":4"
      << ",\"cancelled_rows\":1,\"cancelled_submitted\":true,\"hardware_inflight_at_abort\":" << (cancelled_inflight ? "true" : "false")
      << ",\"ledger_rebuilt_rows\":" << rebuild_rows << ",\"ledger_rebuilt_calls\":" << rebuild_calls
      << ",\"no_early_output_or_prefix\":true,\"comparison\":"; p::write_step(report_a);
  std::cout << "},\"B\":{\"parts\":[2,2,1],\"seed_length\":5,\"seed_frontier\":4,\"seed_lag\":1"
      << ",\"state_only_commits\":2,\"host_restored_length\":7,\"host_restored_pending\":-1,\"final_length\":10,\"final_frontier\":10"
      << ",\"final_draw\":2,\"comparison\":"; p::write_step(report_b);
  std::cout << "},\"C\":{\"parts\":[1,1],\"seed_lag\":1,\"state_only_target_rows\":1,\"state_only_draft_rows\":2"
      << ",\"state_only_pending\":-1,\"state_only_draw\":1,\"final_length\":7,\"final_frontier\":7,\"final_draw\":2,\"comparison\":";
  p::write_step(report_c);
  std::cout << "},\"old_checkpoints_byte_unchanged\":true,\"partial_restore_byte_equal\":true,\"same_owner_commit_epoch_increment\":1"
      << ",\"restored_epoch_scope\":\"new_session_owner\",\"state_only_score_sampler_id_ports_absent\":true"
      << ",\"final_versions\":0,\"final_images\":0,\"final_native_episodes\":0,\"payloads_byte_equal\":"
      << (report_a.payloads_equal && report_b.payloads_equal && report_c.payloads_equal ? "true" : "false")
      << ",\"numerically_equivalent\":" << (equal ? "true" : "false") << ",\"gdn_state_absolute_budget\":" << p::gdn_state_absolute_budget
      << ",\"natural_acceptance_claim\":false,\"performance_claim\":false,\"status\":\"" << (equal ? "PASS" : "VALUE_OR_CONTRACT_DIFFERENCE") << "\"}\n";
  require(equal, "MTP chunked versus exact complete state exceeded the inherited numeric or exact payload/boundary contract");
  return 0;
}
} // namespace qwen_mtp_chunked_prefill_case
