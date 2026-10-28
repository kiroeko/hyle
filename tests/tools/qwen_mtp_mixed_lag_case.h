#pragma once

// Included after the real Model and the existing complete-state oracles.
// Controlled prefixes test state ownership; natural K1 uses the original selector.
namespace qwen_mtp_mixed_lag_case {
namespace q = qwen35;
namespace i = hyle::inference;
namespace h = qwen_host_image_case;
namespace s = qwen_mtp_batch_state_case;
namespace p = qwen_packed_prefill_case;
namespace r = qwen_mtp_ragged_prefill_case;
using h::require;

inline void cold(q::Model &model) {
  const std::array three{std::size_t{3}}, two{std::size_t{2}}, one{std::size_t{1}};
  model.prepare_family(three, false, 1, q::GraphRole::kMtpPrefill);
  model.plan_replay_uses(3, 1, 2, q::GraphRole::kMtpPrefill);
  model.prepare_family(two, false, 1, q::GraphRole::kVerify);
  model.plan_replay_uses(2, 1, 1, q::GraphRole::kVerify);
  for (const auto role : {q::GraphRole::kMtpRound, q::GraphRole::kMtpCatchupRound}) {
    model.prepare_family(two, false, 1, role); model.plan_replay_uses(2, 1, 2, role);
  }
  for (const auto role : {q::GraphRole::kMtpSingle, q::GraphRole::kMtpCatchupSingle}) {
    model.prepare_family(one, false, 1, role, false, q::TextCompletion::kStateOnly);
    model.plan_replay_uses(1, 1, 4, role, q::TextCompletion::kStateOnly);
  }
  model.prepare_family(two, false, 2, q::GraphRole::kMtpCatchupRound, false,
      q::TextCompletion::kEmitNext, 0, q::RowPacking::kDraft);
  model.plan_replay_uses(2, 2, 2, q::GraphRole::kMtpCatchupRound,
      q::TextCompletion::kEmitNext, 0, q::RowPacking::kDraft);
  model.prepare_draft_program(1, false);
  model.freeze_shape_plan();
  model.prepare_execution(three, 1, q::GraphRole::kMtpPrefill);
  model.prepare_execution(two, 1, q::GraphRole::kVerify);
  for (const auto role : {q::GraphRole::kMtpRound, q::GraphRole::kMtpCatchupRound}) model.prepare_execution(two, 1, role);
  for (const auto role : {q::GraphRole::kMtpSingle, q::GraphRole::kMtpCatchupSingle})
    model.prepare_execution(one, 1, role, q::TextCompletion::kStateOnly);
  model.prepare_execution(two, 2, q::GraphRole::kMtpCatchupRound,
      q::TextCompletion::kEmitNext, 0, q::RowPacking::kDraft);
}

inline void same_round(const q::MtpRoundResult &a, const q::MtpRoundResult &b) {
  require(a.pending == b.pending && a.draft == b.draft && a.target_first == b.target_first &&
      a.target_second == b.target_second && a.next_id == b.next_id && a.agreed == b.agreed &&
      a.accepted_draft == b.accepted_draft && a.stopped == b.stopped &&
      a.processed_target == b.processed_target && a.processed_draft == b.processed_draft &&
      a.proposal_count == b.proposal_count && a.consumed_drafts == b.consumed_drafts &&
      a.verified_rows == b.verified_rows && a.auxiliary_graph_calls == b.auxiliary_graph_calls && a.emitted == b.emitted,
      "mixed-lag natural K1 changed its original R1 proposal/acceptance/output facts");
}

inline int run(const std::filesystem::path &input, q::BackendKind kind) {
  auto source = q::prepare_weight_source(input / "weights.tsv", true);
  const auto geometry = source.roles.geometry;
  require(geometry.context_limit >= 16 && geometry.vocabulary == 64 && geometry.hidden == 64 &&
      geometry.recurrent_layers() && geometry.attention_layers() && geometry.mtp_layers == 1,
      "mixed-lag K1 requires the original cached C16 GA/H64/V64 MTP fixture");
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 16, 6, 3,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 2, 2, true, true);
  q::Model model(std::move(source), geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands, SIZE_MAX,
      {.enabled = true, .policy = {1.F, 8, .9F}, .seed = 123}, 0, {.primary = kind});
  cold(model);
  const auto storage = q::take(i::HostStateStorage::create(model.execution, 1, 4u << 20));
  std::array<std::shared_ptr<q::Session>, 2> sessions, controls;
  std::array<q::cache::SnapshotRef, 2> original;
  std::array<s::StateImage, 2> original_images;
  hyle::util::ScopeGuard cleanup{[&]() noexcept {
    (void)model.abort_episode();
    for (auto *group : {&sessions, &controls}) for (auto &session : *group) if (session) {
      session->stop().IgnoreError(); (void)session->drain(); (void)session->try_retire(); session.reset();
    }
    original = {};
  }};
  const std::array chunks{std::size_t{3}};
  const std::array<std::array<std::int32_t, 3>, 2> prompts{{{1, 2, 3}, {3, 2, 1}}};
  std::size_t target_submissions = 0, mixed_submissions = 0;
  for (std::size_t member = 0; member < 2; ++member) {
    sessions[member] = model.create_session(chunks, {}, q::GraphRole::kMtpPrefill,
        q::cache::SamplingCursor{123, 7 + member, 0});
    model.prepare_evaluation(*sessions[member], prompts[member], nullptr, q::GraphRole::kMtpPrefill);
    model.submit_evaluations(); ++target_submissions;
    (void)q::finish_evaluation(model, *sessions[member], "mixed-lag-prefill", {}, false);
  }
  // Produce lag1 through a real accepted two-row target prefix and original
  // one-row repair, independently of this untrained model's natural agreement.
  {
    auto &session = sessions[1]; const auto cursor = session->head->sampling;
    const std::array ids{session->head->next_id, std::int32_t{4}};
    model.prepare_evaluation(*session, ids, nullptr, q::GraphRole::kVerify);
    model.submit_evaluations(); ++target_submissions;
    const std::array outputs{q::EvaluationOutput{.session = session}};
    std::array<std::int32_t, 1> next{-1};
    model.finish_selected_evaluations(outputs, next,
        [&](auto candidates, q::StepState &step, auto selected) {
          require(step.members.size() == 1 && candidates.size() == 2, "lag1 seed did not execute Verify2");
          step.members[0].accepted_tokens = 2; selected[0] = candidates[1];
        }, [&](q::StepState &step) { model.repair_recursive_draft(step); }, [](auto, const auto &) {});
    require(session->head->draft->lag == 1 && session->head->draft->catchup_token == ids[1] &&
        session->head->prior_hidden.valid() && session->head->sampling == cursor,
        "lag1 seed lost its actual prior/catch-up token or advanced RNG");
  }
  for (std::size_t member = 0; member < 2; ++member) {
    original[member] = sessions[member]->checkpoint(); original_images[member] = s::image(model, *original[member]);
    require(original_images[member].lag == member, "mixed-lag seed failed to retain distinct original lags0/1");
    controls[member] = model.create_session({}, original[member], member ? q::GraphRole::kMtpCatchupRound : q::GraphRole::kMtpRound);
  }
  std::vector<p::StepReport> reports; reports.reserve(6);
  const auto old_unchanged = [&] {
    for (std::size_t member = 0; member < 2; ++member)
      require(s::image(model, *original[member]) == original_images[member], "mixed-lag work mutated an immutable complete checkpoint");
  };
  const auto compare_states = [&](std::size_t sequence) {
    for (std::size_t member = 0; member < 2; ++member) {
      p::StepReport report{sequence, original_images[member].length, 2, 2};
      report.epoch = sessions[member]->head.epoch();
      report.exact_next = controls[member]->head->next_id; report.bucket_next = sessions[member]->head->next_id;
      p::compare(model, p::capture(model, *controls[member], storage), p::capture(model, *sessions[member], storage), report);
      require(report.numerically_equivalent && report.exact_next == report.bucket_next,
          "mixed-lag complete state differs from its independent exact R1 control");
      reports.push_back(std::move(report));
    }
    old_unchanged();
  };
  const auto inspect_mixed = [&](bool submitted = false) {
    const auto &step = *model.active_step;
    require(step.prepared && step.submitted == submitted && step.members.size() == 2 && step.input_tokens == 2 &&
        step.active_input_rows == 4 && step.phase->prefix_rows == 2 &&
        step.phase->family->role == q::GraphRole::kMtpCatchupRound && step.phase->family->packing == q::RowPacking::kDraft,
        "different base lags were split or failed to select the one canonical compact draft family");
    auto &io = step.phase->io[step.state_slot];
    const std::vector<std::int32_t> hidden{0, 3, 1, 0}, tokens{2, 1, 3, 0};
    require(r::read_indices(model, io.draft_hidden_select) == hidden && r::read_indices(model, io.draft_token_select) == tokens,
        "mixed-lag hidden/catch-up selector differs from its original carry/prior/token order");
    const auto layout = step.draft_layout.host();
    const std::array<std::int32_t, 3> offsets{0, 1, 3};
    const auto dimensions = io.draft_positions.description.dimensions();
    require(std::ranges::equal(layout.query_indptr, offsets) && step.draft_layout.facts().query_rows == 3 &&
        dimensions.size() == 2 && dimensions[0] == 2 && dimensions[1] == 2 &&
        r::read_indices(model, io.draft_last_ids) == std::vector<std::int32_t>{0, 2},
        "mixed-lag draft did not retain exactly three actual rows in its four-row physical carrier");
    std::vector<std::int32_t> positions(4, 0);
    for (std::size_t member = 0; member < 2; ++member) {
      const auto &part = step.members[member]; const auto rows = 1 + member;
      require(part.session == sessions[member] && part.base->draft->lag == member &&
          part.draft_layout.facts().query_rows == rows && part.next->draft->frontier == part.base->draft->frontier + rows,
          "mixed-lag append belongs to another member or consumed padded rows");
      for (std::size_t row = 0; row < rows; ++row)
        positions[static_cast<std::size_t>(offsets[member]) + row] = static_cast<std::int32_t>(part.base->draft->frontier + row);
    }
    require(r::read_indices(model, io.draft_positions) == positions, "mixed-lag draft positions lost a frontier or initialized padding");
  };
  const std::array<std::size_t, 2> accepted{1, 2};
  const auto controlled = [&](std::span<const std::shared_ptr<q::Session>> selected, bool mixed) {
    std::vector<std::int32_t> pending; std::vector<q::EvaluationRequest> requests; std::vector<q::EvaluationOutput> outputs;
    std::vector<s::StateImage> before; std::vector<std::uint64_t> epochs;
    std::vector<std::int32_t> drafts(selected.size(), -1);
    pending.reserve(selected.size()); requests.reserve(selected.size()); outputs.reserve(selected.size());
    for (const auto &session : selected) {
      pending.push_back(session->head->next_id); before.push_back(s::image(model, *session->head)); epochs.push_back(session->head.epoch());
    }
    for (std::size_t member = 0; member < selected.size(); ++member) {
      requests.push_back({selected[member], std::span(&pending[member], 1)}); outputs.push_back({.session = selected[member]});
    }
    const auto role = selected[0]->head->draft->lag ? q::GraphRole::kMtpCatchupRound : q::GraphRole::kMtpRound;
    model.prepare_evaluations(requests, nullptr, role); if (mixed) inspect_mixed();
    model.submit_evaluations(); ++target_submissions; mixed_submissions += mixed;
    require(model.active_step->submitted && model.active_step->members.size() == selected.size(), "controlled K1 lost its actual submitted cohort");
    std::vector<std::int32_t> next(selected.size(), -1);
    model.finish_selected_evaluations(outputs, next,
        [&](auto candidates, q::StepState &step, auto result) {
          require(candidates.size() == selected.size() * 2, "controlled K1 did not execute both target rows per member");
          for (std::size_t member = 0; member < selected.size(); ++member) {
            const auto count = mixed ? accepted[member] : accepted[selected[0] == controls[0] ? 0 : 1];
            drafts[member] = step.members[member].draft_token;
            step.members[member].accepted_tokens = count; result[member] = candidates[member * 2 + count - 1];
          }
        }, [](auto, const auto &) {});
    for (std::size_t member = 0; member < selected.size(); ++member) {
      const auto after = s::image(model, *selected[member]->head);
      const auto count = mixed ? accepted[member] : accepted[selected[0] == controls[0] ? 0 : 1];
      require(selected[member]->head.epoch() == epochs[member] + 1 && after.length == before[member].length + count &&
          after.frontier == before[member].frontier + 1 + before[member].lag && after.pending == next[member] &&
          after.sampling == before[member].sampling.advanced(3) && after.catchup == (after.lag ? drafts[member] : -1),
          "controlled K1 changed actual epoch, accepted frontier, pending/catch-up or original sampling cursor");
      r::preserved_prefix(before[member], after);
    }
  };
  for (std::size_t member = 0; member < 2; ++member) controlled(std::span(controls).subspan(member, 1), false);
  controlled(sessions, true);
  for (std::size_t member = 0; member < 2; ++member)
    require(sessions[member]->length() == original_images[member].length + accepted[member] &&
        sessions[member]->head->draft->lag == member && sessions[member]->head->sampling == original_images[member].sampling.advanced(3),
        "controlled mixed-lag K1 changed accepted length, pending lag or original draw count");
  compare_states(0);
  std::array<q::MtpRoundResult, 2> expected, actual;
  const auto natural = [&](std::span<const std::shared_ptr<q::Session>> selected, std::span<q::MtpRoundResult> result, bool mixed) {
    std::vector<std::int32_t> pending; std::vector<q::EvaluationRequest> requests; std::vector<q::EvaluationOutput> outputs;
    std::vector<q::MtpGenerationSelection> selections;
    std::vector<s::StateImage> before; std::vector<std::uint64_t> epochs;
    pending.reserve(selected.size()); requests.reserve(selected.size()); outputs.reserve(selected.size()); selections.reserve(selected.size());
    for (const auto &session : selected) {
      pending.push_back(session->head->next_id); before.push_back(s::image(model, *session->head)); epochs.push_back(session->head.epoch());
    }
    for (std::size_t member = 0; member < selected.size(); ++member) {
      requests.push_back({selected[member], std::span(&pending[member], 1)}); outputs.push_back({.session = selected[member]});
      selections.push_back({selected[member], pending[member], 2});
    }
    const auto role = selected[0]->head->draft->lag ? q::GraphRole::kMtpCatchupRound : q::GraphRole::kMtpRound;
    model.prepare_evaluations(requests, nullptr, role); if (mixed) inspect_mixed();
    model.submit_evaluations(); ++target_submissions; mixed_submissions += mixed;
    require(model.active_step->submitted && model.active_step->members.size() == selected.size(), "natural K1 lost its actual submitted cohort");
    model.finish_mtp_rounds(selections, q::model_eos_tokens, outputs, result, [](auto, const auto &) {});
    for (std::size_t member = 0; member < selected.size(); ++member) {
      const auto after = s::image(model, *selected[member]->head);
      require(selected[member]->head.epoch() == epochs[member] + 1 &&
          after.length == before[member].length + result[member].processed_target &&
          after.frontier == before[member].frontier + 1 + before[member].lag &&
          result[member].processed_draft == 1 + before[member].lag && after.pending == result[member].next_id &&
          after.sampling == before[member].sampling.advanced(3),
          "natural K1 changed original accepted rows, draft work, epoch, pending or sampling cursor");
      r::preserved_prefix(before[member], after);
    }
  };
  for (std::size_t member = 0; member < 2; ++member)
    natural(std::span(controls).subspan(member, 1), std::span(expected).subspan(member, 1), false);
  // Observe the actual already committed frontier, as the full-hit output
  // boundary does, then execute the original generation/ledger entry. This is
  // not a source-history import or a complete RequestLoop admission test.
  {
    std::array<s::StateImage, 2> before;
    std::array<std::uint64_t, 2> epochs;
    std::vector<q::GenerationRequest> owners; owners.reserve(2);
    for (std::size_t member = 0; member < 2; ++member) {
      before[member] = s::image(model, *sessions[member]->head); epochs[member] = sessions[member]->head.epoch();
      owners.emplace_back("mixed-frontier" + std::to_string(member), std::span<const std::int32_t>{}, std::filesystem::path{}, 3);
      auto &request = owners.back(); request.session = sessions[member];
      request.ledger = std::make_unique<q::GenerationLedger>(2, 2);
      request.round_emitted.resize(2); for (auto &emitted : request.round_emitted) emitted.reserve(2);
      request.turn_observations.resize(1); request.progress.initial_sampling = request.session->head->sampling;
      (void)request.progress.begin_turn();
      request.progress.apply_outputs(std::span(&request.session->head->next_id, 1), true, true, false);
    }
    std::array calls{q::prepare_generation_call(model, owners[0], 3, nullptr, owners, 2),
        q::prepare_generation_call(model, owners[1], 3, nullptr, owners, 2)};
    require(calls[0].generation && calls[1].generation && calls[0].role == q::GraphRole::kMtpRound &&
        calls[1].role == q::GraphRole::kMtpCatchupRound && calls[0].maximum == 1 && calls[1].maximum == 1,
        "real generation preparation did not preserve both original K1 lag roles");
    std::size_t submitted_batches = 0;
    q::execute_generation_calls(model, calls, false, nullptr, owners, [&] {
      inspect_mixed(true); ++submitted_batches; ++target_submissions; ++mixed_submissions;
      require(submitted_batches == 1, "mixed-lag Engine entry split the actual K1 target submission");
    });
    require(submitted_batches == 1, "mixed-lag Engine entry did not submit its actual target");
    for (std::size_t member = 0; member < 2; ++member) {
      const auto &request = owners[member]; const auto &record = request.ledger->records[0];
      require(calls[member].disposition == q::EvaluationDisposition::kPublished &&
          request.session == sessions[member] && request.ledger->committed_count == 1 && record.committed &&
          record.role == calls[member].role && record.requests == 2 && record.member_index == member &&
          request.progress.observed_commits() == 1 && request.rounds.size() == 1 &&
          request.progress.generated.front() == before[member].pending &&
          request.progress.generated.size() == 1 + request.rounds.front().emitted.size() &&
          std::ranges::equal(std::span(request.progress.generated).subspan(1), request.rounds.front().emitted),
          "mixed-lag Engine publication changed an original member role, ledger, or observed output");
      actual[member] = request.rounds.front();
      const auto after = s::image(model, *sessions[member]->head);
      require(sessions[member]->head.epoch() == epochs[member] + 1 &&
          after.length == before[member].length + actual[member].processed_target &&
          after.frontier == before[member].frontier + 1 + before[member].lag &&
          actual[member].processed_draft == 1 + before[member].lag && after.pending == actual[member].next_id &&
          after.sampling == before[member].sampling.advanced(3),
          "natural mixed Engine K1 changed accepted rows, actual draft work, epoch, pending or RNG");
      r::preserved_prefix(before[member], after);
    }
  }
  for (std::size_t member = 0; member < 2; ++member) same_round(expected[member], actual[member]);
  compare_states(1);
  for (std::size_t member = 0; member < 2; ++member) for (auto *group : {&controls, &sessions}) {
    auto &session = (*group)[member]; const auto cursor = session->head->sampling;
    const auto epoch = session->head.epoch(); const auto length = session->length();
    const auto role = session->head->draft->lag ? q::GraphRole::kMtpCatchupSingle : q::GraphRole::kMtpSingle;
    const std::array ids{session->head->next_id};
    model.prepare_evaluation(*session, ids, nullptr, role, q::TextCompletion::kStateOnly);
    model.submit_evaluations(); ++target_submissions;
    const auto next = q::finish_evaluation(model, *session, "mixed-lag-state-only", {}, false);
    require(next == -1 && session->head->next_id == -1 && session->head->sampling == cursor &&
        session->head.epoch() == epoch + 1 && session->length() == length + 1 &&
        !session->head->draft->lag && session->head->draft->frontier == session->length(),
        "StateOnly continuation emitted a pending token or consumed RNG");
  }
  compare_states(2);
  const auto retired_handles = sessions;
  for (auto *group : {&controls, &sessions}) for (auto &session : *group) h::retire(session);
  original = {};
  require(!retired_handles[0]->head && !retired_handles[1]->head,
      "retired-handle regression did not retain actual retired Session identities");
  bool retired_rejected = false;
  const std::array<std::int32_t, 1> retired_input{1};
  const std::array retired_requests{q::EvaluationRequest{retired_handles[0], retired_input},
      q::EvaluationRequest{retired_handles[1], retired_input}};
  try { model.prepare_evaluations(retired_requests, nullptr, q::GraphRole::kMtpRound); }
  catch (const std::runtime_error &) { retired_rejected = true; }
  require(retired_rejected && !model.active_step && !model.successor_step && !model.carrier_borrowed,
      "retired Session reached lag scanning/preparation or retained an execution episode");
  for (const auto &[key, program] : model.draft_programs)
    require(!program->guard.execution.has_completion() && !program->guard.execution.cleanup_pending &&
        !program->hidden_transactions.bound() && !program->carrier.pending(), "mixed-lag repair retained its original NextN resources");
  require(!model.active_step && !model.successor_step && !model.carrier_borrowed && !model.snapshot_versions.live &&
      !storage->live_images() && model.page_inventory.free_ids.size() == plan.max_pages &&
      model.draft_inventory.free_ids.size() == plan.max_pages && mixed_submissions == 2 && target_submissions == 13,
      "mixed-lag journey split a B2 submission or retained a state, image, target/draft page owner");
  bool equivalent = true, exact = true;
  for (const auto &report : reports) { equivalent = equivalent && report.numerically_equivalent; exact = exact && report.payloads_equal; }
  std::cout << "{\"status\":\"PASS\",\"backend\":\"" << h::backend_name(kind)
      << "\",\"mode\":\"mtp_mixed_lag\",\"synthetic_untrained\":true,\"requests\":2,\"base_lags\":[0,1],"
      "\"controlled_accepted\":[1,2],\"mixed_target_submissions\":" << mixed_submissions << ",\"target_submissions\":" << target_submissions
      << ",\"target_rows_per_member\":2,\"actual_draft_rows\":3,\"physical_draft_rows\":4,"
      "\"hidden_indices\":[0,3,1,0],\"token_indices\":[2,1,3,0],\"natural_processed_target\":["
      << actual[0].processed_target << ',' << actual[1].processed_target << "],\"natural_processed_draft\":["
      << actual[0].processed_draft << ',' << actual[1].processed_draft << "],\"state_checks\":[";
  for (std::size_t index = 0; index < reports.size(); ++index) { if (index) std::cout << ','; p::write_step(reports[index]); }
  std::cout << "],\"payloads_byte_equal\":" << (exact ? "true" : "false")
      << ",\"numerically_equivalent\":" << (equivalent ? "true" : "false")
      << ",\"old_checkpoints_unchanged\":true,\"StateOnly_no_pending_or_draw\":true,\"natural_R1_results_equal\":true,"
      "\"mixed_engine_single_submission\":true,\"original_member_roles_committed\":true,\"retired_head_rejected_before_prepare\":true,"
      "\"full_request_history_import_coverage\":false,"
      "\"final_versions\":0,\"natural_acceptance_quality_claim\":false,\"performance_claim\":false}\n";
  cleanup.dismiss();
  return 0;
}
} // namespace qwen_mtp_mixed_lag_case
