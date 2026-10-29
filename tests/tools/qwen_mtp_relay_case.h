#pragma once

// Included after the current Model and the existing complete-state consumers.
// Selection is controlled through the original finite-prefix API. No device
// output, Completion, source permission or execution route is replaced.
namespace qwen_mtp_relay_case {
namespace q = qwen35;
namespace i = hyle::inference;
namespace h = qwen_host_image_case;
namespace p = qwen_packed_prefill_case;
namespace s = qwen_mtp_batch_state_case;
using h::require;
enum class Mode { kFull, kPartial, kQuotaStop, kCancel };

inline bool ordinary_cancel(const hyle::Status &status) {
  return status.ok() || (status.IsCancelled() && !status.has_native_error());
}
inline void retire(std::shared_ptr<q::Session> &session) {
  if (!session) return;
  require(ordinary_cancel(session->stop()), "K1 relay Session stop returned a native/error verdict");
  const auto drained = session->drain();
  require(drained.retired && ordinary_cancel(drained.status), "K1 relay Session did not really drain");
  const auto retired = session->try_retire();
  require(retired.retired && ordinary_cancel(retired.status), "K1 relay Session did not really retire");
  session.reset();
}
inline void same_records(const q::GenerationLedger &a, const q::GenerationLedger &b) {
  require(a.committed_count == b.committed_count, "K1 Source changed the committed ledger count");
  for (std::size_t row = 0; row < a.committed_count; ++row) {
    const auto &x = a.records[row]; const auto &y = b.records[row];
    require(x.same_result(y) && x.committed == y.committed && x.user_output == y.user_output &&
        x.imported == y.imported && x.turn == y.turn && std::ranges::equal(x.ids(), y.ids()),
        "K1 Source changed an original complete non-time ledger result");
  }
}
inline void unchanged_records(const q::GenerationLedger &ledger,
    const std::vector<q::GenerationCommitRecord> &before, std::size_t committed) {
  require(ledger.committed_count == committed && ledger.records.size() == before.size(),
      "K1 A/B submission changed the ledger before A host finish");
  for (std::size_t row = 0; row < before.size(); ++row) {
    const auto &x = ledger.records[row]; const auto &y = before[row];
    require(x.same_result(y) && x.committed == y.committed && x.user_output == y.user_output &&
        x.imported == y.imported && x.turn == y.turn && x.input_owner == y.input_owner &&
        std::ranges::equal(x.ids(), y.ids()) && x.step_seconds == y.step_seconds && x.endpoint == y.endpoint,
        "K1 A/B submission mutated an original prepared or committed ledger record");
  }
}
inline q::GraphRole role(const q::Session &session) {
  return session.head->draft->lag ? q::GraphRole::kMtpCatchupRound : q::GraphRole::kMtpRound;
}
inline void stage(q::GenerationLedger &ledger, const std::shared_ptr<q::Session> &session, std::size_t quota) {
  const auto pending = session->head->next_id;
  q::prepare_generation_record(ledger, *session, role(*session), std::span(&pending, 1), true, quota, 0);
}
inline std::int32_t finish(q::Model &model, const std::shared_ptr<q::Session> &session,
    q::GenerationLedger &ledger, std::size_t accepted, std::size_t quota) {
  const auto before = q::generation_boundary(*session->head); const auto epoch = session->head.epoch();
  const std::array outputs{q::EvaluationOutput{.session = session, .generation_ledger = &ledger}};
  std::array<std::int32_t, 1> next{-1};
  model.finish_selected_evaluations(outputs, next,
      [&](std::span<const std::int32_t> ids, q::StepState &step, std::span<std::int32_t> selected) {
        require(step.members.size() == 1 && step.input_tokens == 2 && ids.size() == 2 &&
            (accepted == 1 || accepted == 2) && accepted <= quota,
            "K1 relay controlled selection did not observe its real two target rows");
        auto &member = step.members[0]; auto &record = ledger.next();
        member.accepted_tokens = accepted; selected[0] = ids[accepted - 1];
        record.draft = member.draft_token; record.target_first = ids[0]; record.target_second = ids[1];
        record.agreed = accepted == 2; record.accepted_draft = accepted == 2;
        record.consumed_drafts = accepted - 1; record.stopped = false;
        record.emitted = {accepted == 2 ? member.draft_token : selected[0], accepted == 2 ? selected[0] : -1};
        record.emitted_count = accepted;
      }, [](auto, const auto &) {});
  const auto after = q::generation_boundary(*session->head);
  require(session->head.epoch() == epoch + 1 && after.length == before.length + accepted &&
      after.frontier == before.frontier + before.lag + 1 && after.lag == accepted - 1 &&
      after.next_id == next[0] && after.sampling == before.sampling,
      "K1 relay published the wrong prefix, draft frontier, pending, epoch or greedy RNG");
  return next[0];
}
inline void serial(q::Model &model, const std::shared_ptr<q::Session> &session,
    q::GenerationLedger &ledger, std::size_t accepted, std::size_t quota, std::size_t &submissions) {
  const auto pending = session->head->next_id; stage(ledger, session, quota);
  model.prepare_evaluation(*session, std::span(&pending, 1), nullptr, role(*session));
  model.submit_evaluations(); ++submissions; (void)finish(model, session, ledger, accepted, quota);
}

inline int run(const std::filesystem::path &input, q::BackendKind kind) {
  auto source = q::prepare_weight_source(input / "weights.tsv", true); const auto geometry = source.roles.geometry;
  require((kind == q::BackendKind::kCpu || kind == q::BackendKind::kCuda) && geometry.context_limit >= 16 &&
      geometry.hidden == 64 && geometry.vocabulary == 64 && geometry.recurrent_layers() &&
      geometry.attention_layers() && geometry.mtp_layers == 1, "K1 relay requires the original cached C16 GA MTP fixture");
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 16, 6, 3,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 1, 2, true, true, 2);
  // CUDA uses its actual captured route. Ready Source tuples prime the same
  // imported carrier before the finite pending-producer branches.
  q::Model model(std::move(source), geometry, 16, plan, {}, false, kind == q::BackendKind::kCuda ? q::ReplayPolicy::kPreferCapture : q::ReplayPolicy::kCommands,
      SIZE_MAX, {}, 0, {.primary = kind});
  const std::array three{std::size_t{3}}, two{std::size_t{2}};
  model.prepare_family(three, false, 1, q::GraphRole::kMtpPrefill);
  model.plan_replay_uses(3, 1, 8, q::GraphRole::kMtpPrefill);
  for (const auto selected : {q::GraphRole::kMtpRound, q::GraphRole::kMtpCatchupRound}) {
    model.prepare_family(two, false, 1, selected); model.plan_replay_uses(2, 1, 16, selected);
  }
  model.freeze_shape_plan(); model.prepare_execution(three, 1, q::GraphRole::kMtpPrefill);
  for (const auto selected : {q::GraphRole::kMtpRound, q::GraphRole::kMtpCatchupRound}) model.prepare_execution(two, 1, selected);
  model.prepare_relay_execution(1, 0, q::GraphRole::kMtpCatchupRound);
  std::vector<p::StepReport> checks; checks.reserve(6);
  std::size_t submissions = 0, source_submissions = 0, successor_submissions = 0, successor_discards = 0,
      cpu_declines = 0;
  const auto compare = [&](const s::StateImage &expected, const q::cache::State &actual) {
    const auto image = s::image(model, actual);
    p::StepReport report{checks.size(), actual.length, 2, 2}; report.prefill = false;
    report.exact_next = expected.pending; report.bucket_next = image.pending;
    report.boundary_equal = std::tie(expected.length, expected.frontier, expected.lag, expected.pending,
        expected.catchup, expected.sampling) == std::tie(image.length, image.frontier, image.lag,
        image.pending, image.catchup, image.sampling);
    require(expected.target == image.target && expected.payloads.size() == image.payloads.size(),
        "K1 Source changed the exact target KV or complete payload inventory");
    report.payloads_equal = expected.payloads == image.payloads;
    report.numerically_equivalent = report.boundary_equal;
    std::vector<std::size_t> recurrent_layers;
    for (std::size_t layer = 0; layer < geometry.layers.size(); ++layer)
      if (geometry.is_gdn(layer)) recurrent_layers.push_back(layer);
    for (std::size_t part = 0; part < expected.payloads.size(); ++part) {
      if (expected.payloads[part] == image.payloads[part]) continue;
      // The existing oracle permits rounding only in newly evaluated F32 GDN.
      // History, target/draft KV, hidden and all metadata stay byte exact.
      require(part < recurrent_layers.size() * 2 && part % 2 == 1,
          "K1 Source changed a non-GDN complete-state payload");
      q::cache::HostPayloadDesc layout{};
      layout.role = q::cache::HostPayloadRole::kGdnState;
      layout.layer = static_cast<std::uint32_t>(recurrent_layers[part / 2]);
      layout.bytes = expected.payloads[part].size();
      auto difference = p::compare_payload(model, layout, expected.payloads[part], image.payloads[part]);
      require(!difference.nonfinite_differences && difference.maximum_absolute <= p::gdn_state_absolute_budget,
          "K1 Source GDN state differs beyond the original absolute budget");
      report.differences.push_back(std::move(difference));
    }
    require(report.numerically_equivalent && report.exact_next == report.bucket_next,
        "K1 Source complete state differs from its independent serial control");
    checks.push_back(std::move(report));
  };
  static constexpr std::array<std::int32_t, 3> prompt{1, 2, 3};
  const auto prefill = [&](std::shared_ptr<q::Session> &session, q::GenerationLedger &ledger) {
    session = model.create_session(three, {}, q::GraphRole::kMtpPrefill);
    q::prepare_generation_record(ledger, *session, q::GraphRole::kMtpPrefill, prompt, false, 1, 0);
    model.prepare_evaluation(*session, prompt, nullptr, q::GraphRole::kMtpPrefill);
    model.submit_evaluations(); ++submissions;
    (void)q::finish_evaluation(model, *session, "K1-source-prefill", {}, false,
        nullptr, nullptr, nullptr, nullptr, {}, &ledger);
  };
  std::size_t priming_submissions = 0;
  const auto prime = [&](unsigned lag) {
    const auto before = submissions;
    q::GenerationLedger ledger(8, 2); std::shared_ptr<q::Session> session;
    std::shared_ptr<q::StepSources> ready;
    hyle::util::ScopeGuard cleanup{[&]() noexcept {
      ready.reset(); (void)model.abort_episode();
      if (session) { session->stop().IgnoreError(); (void)session->drain(); (void)session->try_retire(); session.reset(); }
    }};
    prefill(session, ledger);
    if (lag) serial(model, session, ledger, 2, 4, submissions);
    const auto pending = session->head->next_id; stage(ledger, session, 4);
    model.prepare_evaluation(*session, std::span(&pending, 1), nullptr, role(*session));
    ready = model.prepare_submission_sources(*model.active_step);
    model.submit_evaluations(); ++submissions;
    (void)finish(model, session, ledger, 2, 4);
    require(!model.active_step && !session->pending && ready->submission.valid() &&
        std::ranges::all_of(ready->published_states, [](const auto &read) { return read.valid(); }),
        "K1 cold priming did not retain the real complete ready Source guards");
    stage(ledger, session, 2);
    const std::array future{q::EvaluationRequest{session, {}}};
    auto &prepared = model.prepare_relay_evaluations(future, ready, 0, q::GraphRole::kMtpCatchupRound);
    require(prepared.capture_recorded + prepared.capture_reused > 0,
        "K1 imported tuple cold priming did not prepare a real native capture");
    model.submit_evaluations(prepared); ++submissions;
    (void)finish(model, session, ledger, 2, 2);
    ready.reset(); retire(session); cleanup.dismiss();
    priming_submissions += submissions - before;
    require(!model.active_step && !model.successor_step && !model.carrier_borrowed && !model.snapshot_versions.live,
        "K1 cold priming retained an original episode or state version");
  };
  if (kind == q::BackendKind::kCuda) for (const auto lag : {0u, 1u}) prime(lag);
  for (const auto mode : {Mode::kFull, Mode::kPartial, Mode::kQuotaStop, Mode::kCancel}) {
    require(!model.active_step && !model.successor_step && !model.carrier_borrowed && !model.snapshot_versions.live,
        "preceding K1 relay arm retained an original episode or state version");
    q::GenerationLedger control_ledger(8, 2), candidate_ledger(8, 2);
    std::shared_ptr<q::Session> control, candidate;
    q::cache::SnapshotRef original_control, original_candidate;
    std::shared_ptr<q::StepSources> sources;
    hyle::util::ScopeGuard cleanup{[&]() noexcept {
      sources.reset(); (void)model.abort_episode();
      for (auto *session : {&candidate, &control}) if (*session) {
        (*session)->stop().IgnoreError(); (void)(*session)->drain(); (void)(*session)->try_retire(); session->reset();
      }
      original_candidate.reset(); original_control.reset();
    }};
    prefill(control, control_ledger); prefill(candidate, candidate_ledger);
    // Both producer roles are real: the later three arms first commit a full
    // original K1 round to obtain lag1, rather than editing draft metadata.
    if (mode != Mode::kFull) {
      serial(model, control, control_ledger, 2, 4, submissions);
      serial(model, candidate, candidate_ledger, 2, 4, submissions);
    }
    original_control = control->checkpoint(); original_candidate = candidate->checkpoint();
    const auto old_control = s::image(model, *original_control), old_candidate = s::image(model, *original_candidate);
    same_records(control_ledger, candidate_ledger);
    const auto accepted = mode == Mode::kPartial ? std::size_t{1} : std::size_t{2};
    const auto quota = mode == Mode::kQuotaStop ? std::size_t{2} : std::size_t{4};
    serial(model, control, control_ledger, accepted, quota, submissions);
    const auto control_a = s::image(model, *control->head);
    const auto control_a_records = control_ledger.records; const auto control_a_count = control_ledger.committed_count;
    std::optional<s::StateImage> final_control;
    if (mode == Mode::kFull || mode == Mode::kPartial) {
      serial(model, control, control_ledger, 2, mode == Mode::kFull ? 2 : 3, submissions);
      final_control = s::image(model, *control->head);
    }
    const auto pending = candidate->head->next_id; const auto epoch = candidate->head.epoch();
    const auto *old_head = candidate->head.get(); const auto old_boundary = q::generation_boundary(*candidate->head);
    stage(candidate_ledger, candidate, quota);
    const auto before_records = candidate_ledger.records; const auto before_count = candidate_ledger.committed_count;
    const std::array requests{q::EvaluationRequest{candidate, std::span(&pending, 1)}};
    model.prepare_evaluations(requests, nullptr, role(*candidate)); auto *const a = model.active_step.get();
    const std::weak_ptr<q::StepState> a_lifetime = model.active_step;
    try { sources = model.prepare_submission_sources(*a); }
    catch (const i::EvaluationFailure &failure) {
      require(kind == q::BackendKind::kCpu && failure.status().IsUnimplemented() && !failure.status().has_native_error() &&
          !a->submitted && !a->execution.has_completion() && !a->execution.sources,
          "Source capability rejection consumed native work or was not a real unsupported CPU route");
      ++cpu_declines;
    }
    model.submit_evaluations(*a, true); ++submissions;
    q::StepState *b = nullptr; std::weak_ptr<q::StepState> b_lifetime;
    if (kind == q::BackendKind::kCuda) {
      require(a->execution.sources == sources && a->execution.has_completion() && !a->execution.completion.valid() &&
          sources->submission.valid() && sources->mtp_full_candidate && sources->token.valid() &&
          sources->extra_outputs.size() == 1 && sources->extra_outputs[0].valid() &&
          sources->states.size() == geometry.recurrent_layers() * 2 + 1 &&
          sources->successor_states.size() == geometry.recurrent_layers() * 2 && sources->pages.size() == model.draft_bank + 2,
          "K1 did not capture token/catch-up, hidden/state and target/draft KV from one original submission");
      for (const auto &state : sources->states) require(state.valid(), "K1 full Store source was not captured");
      for (const auto &state : sources->successor_states)
        require(state.valid() && state.destination_count() == 1, "K1 successor Store did not select its full accepted prefix");
      for (const auto &page : sources->pages) require(page.valid(), "K1 Source omitted a target/draft KV bank");
      ++source_submissions;
      const std::array future{q::EvaluationRequest{candidate, {}}};
      b = &model.prepare_relay_evaluations(future, sources, 0, q::GraphRole::kMtpCatchupRound);
      require(!b->capture_recorded && b->capture_reused,
          "pending K1 successor did not reuse the actual cold-ready native capture");
      b_lifetime = model.successor_step; model.submit_evaluations(*b); ++submissions; ++successor_submissions;
      require(model.active_step.get() == a && model.successor_step.get() == b && b->submitted &&
          b->execution.has_completion() && b->execution.incoming && b->execution.incoming->owner == sources &&
          b->execution.predecessor_episode && candidate->pending == a && model.carrier_borrowed == 2,
          "K1 B was not actually submitted against its original pending A owner before A host finish");
    } else {
      require(!a->execution.sources && a->execution.completion.valid() && a->execution.has_completion() &&
          !sources && !model.successor_step,
          "CPU optional Source refusal lost the actual original Completion or invented a successor");
    }
    require(candidate->head.get() == old_head && candidate->head.epoch() == epoch &&
        q::generation_boundary(*candidate->head) == old_boundary,
        "K1 A/B submission published a candidate, pending, epoch or RNG before A host finish");
    unchanged_records(candidate_ledger, before_records, before_count);
    sources.reset(); (void)finish(model, candidate, candidate_ledger, accepted, quota);
    const auto a_bytes = s::image(model, *candidate->head);
    compare(control_a, *candidate->head);
    require(candidate_ledger.committed_count == control_a_count, "K1 A omitted its one original ledger commit");
    for (std::size_t row = 0; row < control_a_count; ++row)
      require(candidate_ledger.records[row].same_result(control_a_records[row]), "K1 A ledger differs from serial before B publication");
    if (b) {
      require(model.active_step.get() == b && !model.successor_step && candidate->pending == b,
          "K1 A did not preserve/promote its original submitted B episode");
      if (mode == Mode::kPartial) require(b->cancelled, "partial K1 acceptance kept a full-boundary successor runnable");
      else require(!b->cancelled, "full K1 acceptance cancelled a valid successor boundary");
      if (mode == Mode::kFull) {
        stage(candidate_ledger, candidate, 2); (void)finish(model, candidate, candidate_ledger, 2, 2);
      } else {
        if (mode == Mode::kCancel)
          require(ordinary_cancel(candidate->stop()) && candidate->stopping && b->cancelled,
              "real K1 cancellation did not stop the already submitted successor");
        const auto committed = candidate_ledger.committed_count;
        const auto retired = model.abort_episode();
        require(retired.retired && ordinary_cancel(retired.status) && !model.active_step && !model.successor_step &&
            !model.carrier_borrowed && !candidate->pending && b_lifetime.expired() &&
            candidate_ledger.committed_count == committed && s::image(model, *candidate->head) == a_bytes,
            "K1 successor discard failed to retire real work or changed A's committed prefix");
        ++successor_discards;
      }
    } else if (mode == Mode::kFull) serial(model, candidate, candidate_ledger, 2, 2, submissions);
    else if (mode == Mode::kCancel) require(ordinary_cancel(candidate->stop()), "CPU cancel changed its serial A verdict");
    require(a_lifetime.expired(), "K1 original A stayed retained after its successor retired");
    if (mode == Mode::kPartial) serial(model, candidate, candidate_ledger, 2, 3, submissions);
    if (final_control) { compare(*final_control, *candidate->head); same_records(control_ledger, candidate_ledger); }
    if (mode == Mode::kQuotaStop)
      require(candidate_ledger.records[candidate_ledger.committed_count - 1].emitted_count == quota &&
          candidate_ledger.committed_count == control_ledger.committed_count,
          "K1 quota-terminal owner published an uncredited successor output");
    require(s::image(model, *original_control) == old_control &&
        s::image(model, *original_candidate) == old_candidate, "K1 successor changed an immutable complete checkpoint");
    original_candidate.reset(); original_control.reset(); retire(candidate); retire(control);
    require(!model.active_step && !model.successor_step && !model.carrier_borrowed && !model.snapshot_versions.live &&
        model.page_inventory.free_ids.size() == plan.max_pages &&
        model.draft_inventory.free_ids.size() == model.draft_inventory.owners.size(),
        "K1 relay arm retained candidate/Store/Append/page/host-image owners");
    cleanup.dismiss();
    if (kind == q::BackendKind::kCuda && (mode == Mode::kPartial || mode == Mode::kQuotaStop)) {
      // Abort really closes/resets the original carrier. Re-enter through
      // idle cold preparation, never initialize it behind a live producer.
      model.prepare_relay_execution(1, 0, q::GraphRole::kMtpCatchupRound);
      prime(1);
    }
  }
  require(checks.size() == 6 && (kind == q::BackendKind::kCuda
      ? source_submissions == 4 && successor_submissions == 4 && successor_discards == 3 && !cpu_declines && priming_submissions == 15
      : cpu_declines == 4 && !source_submissions && !successor_submissions && !successor_discards && !priming_submissions),
      "K1 relay finite branches or actual backend Source coverage changed");
  std::cout << "{\"status\":\"PASS\",\"case\":\"mtp-relay\",\"backend\":\"" << h::backend_name(kind)
      << "\",\"models\":1,\"arms\":4,\"native_submissions\":" << submissions
      << ",\"source_submissions\":" << source_submissions << ",\"successor_submissions\":" << successor_submissions
      << ",\"successor_discards\":" << successor_discards << ",\"cpu_optional_source_declines\":" << cpu_declines
      << ",\"priming_submissions\":" << priming_submissions
      << ",\"replay_commands_only\":" << (kind == q::BackendKind::kCpu ? "true" : "false")
      << ",\"cold_ready_tuple_priming\":" << (kind == q::BackendKind::kCuda ? "true" : "false")
      << ",\"full_accept_two\":true,\"partial_accept_one_retry\":true,\"quota_terminal_discard\":true,"
         "\"cancel_after_A_commit\":true,\"no_publication_before_A_finish\":true,\"original_ledger_preserved\":true,"
         "\"old_checkpoints_byte_unchanged\":true,\"real_successor_retirement\":true,\"greedy_rng_unchanged\":true,"
         "\"simultaneous_native_inflight_claim\":false,\"model_eos_run\":false,\"natural_RequestLoop_claim\":false,"
         "\"synthetic_untrained\":true,\"quality_claim\":false,\"performance_claim\":false,\"final_versions\":0,\"state_checks\":[";
  for (std::size_t row = 0; row < checks.size(); ++row) { if (row) std::cout << ','; p::write_step(checks[row]); }
  std::cout << "]}\n"; return 0;
}
} // namespace qwen_mtp_relay_case
