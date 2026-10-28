#pragma once

// Real caller-owned controls over one original MTP episode. Escaped C0 values
// are readonly leases, not another token history, state or execution owner.
namespace qwen_history_control_case {
namespace q = qwen35;
namespace i = hyle::inference;
namespace h = qwen_host_image_case;
namespace hc = qwen_history_logits_case;
namespace m = qwen_mtp_logits_case;
namespace p = qwen_packed_prefill_case;
namespace t = qwen_mtp_stop_case;
namespace u = qwen_mtp_sampling_case;
namespace e = qwen_logits_preflight_case;
using h::require;

inline void check_old(q::Model &model, const q::cache::SnapshotRef &checkpoint,
    const h::ImageBytes &expected, const std::shared_ptr<i::HostStateStorage> &storage) {
  auto reader = model.create_session({}, checkpoint, q::GraphRole::kMtpPrefill, {}, false,
      q::TextCompletion::kEmitNext, 0);
  h::same_image(expected, p::capture(model, *reader, storage)); h::retire(reader);
}

inline void cold(q::Model &model) {
  const std::array rows{std::size_t{3}};
  for (const auto width : {std::size_t{1}, std::size_t{2}})
    for (const auto role : {q::GraphRole::kMtpPrefill, q::GraphRole::kVerify}) {
      model.prepare_family(rows, false, width, role);
      model.plan_replay_uses(3, width, 6, role);
    }
  model.prepare_recursive_round_programs();
}

inline void cache_is_committed(q::Model &model, q::GenerationLedger &ledger) {
  const auto ids = hc::consumed(ledger); const auto expected = hc::counts(model, ids);
  const auto &cache = ledger.history_counts(static_cast<std::size_t>(model.vocabulary), model.execution_budget,
      q::take(model.execution_budget.metadata_allocator()));
  require(cache.ordinal() == ledger.committed_count && cache.consumed_length() == ids.size() &&
      std::ranges::equal(cache.values(), expected), "failed C0 admission retained conditional, pending or rejected token counts");
}

inline void unchanged_committed(q::Model &model, q::GenerationRequest &request, const e::SavedRequest &before,
    const std::shared_ptr<i::HostStateStorage> &storage) {
  require(request.session && request.session->head.get() == before.head && request.session->head.epoch() == before.epoch &&
      request.progress.generated == before.generated && request.progress.prefilled && !request.progress.turn_done &&
      request.progress.observed_commits() == before.observed && request.ledger->committed_count == before.commits &&
      request.turn_input_consumed == before.consumed && request.input_turn == before.input_turn &&
      request.history.get() == before.history && (!request.history || request.history->length == before.history_length) &&
      request.ledger->sampling == before.sampling && !request.ledger->successor_ordinal() &&
      request.ledger->records.size() == before.records.size(),
      "history cancellation changed a committed head, epoch, pending, RNG, input/progress or ledger identity");
  for (std::size_t at = 0; at != before.commits; ++at)
    require(e::same_record(request.ledger->records[at], before.records[at]),
        "history cancellation changed an authoritative committed record");
  const auto &slot = request.ledger->records.at(before.commits);
  const auto &old_slot = before.records.at(before.commits);
  require(!slot.committed && !slot.input_deferred && !slot.accepted && !slot.emitted_count && !slot.quota &&
      !slot.input_owner && slot.pending_input == -1 && slot.prompt_ids.empty() && slot.accepted_inputs.empty() &&
      slot.verification_inputs.empty() && !slot.draft_limit && !slot.verified_rows &&
      slot.emitted.capacity() >= old_slot.emitted.capacity() && slot.accepted_inputs.capacity() >= old_slot.accepted_inputs.capacity() &&
      slot.verification_inputs.capacity() >= old_slot.verification_inputs.capacity(),
      "cancelled ledger retained proposal/publication authority or shrank its reusable cold metadata capacity");
  // The pointer/epoch check above ties this live head to its immutable old
  // checkpoint. A stopped Session cannot admit a new export operation.
  qwen_history_control_case::check_old(model, before.checkpoint, before.image, storage);
}

inline q::MtpRoundResult control_round(q::Model &model, const std::shared_ptr<q::Session> &session,
    const q::LogitsInputs &inputs) {
  const auto pending = session->head->next_id;
  const auto cursor = session->head->sampling;
  const auto epoch = session->head.epoch();
  std::vector<q::c::ReadLease> probabilities;
  const auto ids = model.propose_recursive(*session, 2, q::model_eos_tokens, &probabilities, nullptr, inputs);
  require(ids.size() == 3 && probabilities.size() == 2, "history survivor control did not execute both original proposals");
  q::GenerationLedger ledger(1, 3); ledger.bind_stop_policy({}, model.vocabulary);
  q::prepare_generation_record(ledger, *session, q::GraphRole::kVerify, std::span(&pending, 1), true, 3, 0, {}, 2);
  model.prepare_evaluation(*session, ids, nullptr, q::GraphRole::kVerify, q::TextCompletion::kEmitNext, nullptr, inputs);
  model.submit_evaluation(*session);
  const std::array outputs{q::evaluation_output(model, session, {.name = "history-survivor-control", .trace_states = false},
      nullptr, nullptr, nullptr, nullptr, &ledger)};
  const std::array selections{q::MtpGenerationSelection{session, pending, 3, probabilities, 2}};
  std::array<q::MtpRoundResult, 1> result;
  model.finish_recursive_mtp(selections, q::model_eos_tokens, outputs, result, [](auto, const auto &) {});
  require(session->head.epoch() == epoch + 1 && session->head->sampling == cursor.advanced(5) && ledger.committed_count == 1,
      "history survivor control changed its original epoch or RNG reservation");
  probabilities.clear(); t::idle(model); return std::move(result[0]);
}

inline int run(const std::filesystem::path &input, q::BackendKind kind) {
  auto source = q::prepare_weight_source(input / "weights.tsv", true); const auto geometry = source.roles.geometry;
  require(geometry.hidden == 64 && geometry.vocabulary == 64 && geometry.context_limit >= 16 && geometry.mtp_layers == 1 &&
      geometry.recurrent_layers() && geometry.attention_layers(), "history controls require the cached GA/H64/V64/C16 fixture");
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 16, 8, 3,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 2, 3, true, true);
  q::Model model(std::move(source), geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands, SIZE_MAX,
      {.enabled = true, .policy = {.8F, 0, 1.F}, .seed = 123, .modifiers = {.history = {1.75F, .5F, .125F}}},
      0, {.primary = kind});
  model.configure_draft_policy({.max_drafts = 2}); cold(model);
  const auto storage = q::take(i::HostStateStorage::create(model.execution, 1, 4u << 20));
  const std::array<std::array<std::int32_t, 3>, 2> prompts{{{1, 2, 1}, {2, 1, 2}}};
  std::vector<q::GenerationRequest> owners; owners.reserve(2);
  i::RequestWindow window; window.prepare(2, 2);
  std::array<i::OutputDelivery, 2> delivery{i::OutputDelivery(3), i::OutputDelivery(3)};
  for (std::size_t member = 0; member != 2; ++member) {
    owners.emplace_back("history-control" + std::to_string(member), prompts[member], std::filesystem::path{}, 7);
    auto &owner = owners.back(); t::prepare_request(owner, 16); owner.progress.initial_sampling = {171, 401 + member, 0};
    owner.request_key = window.enqueue(member); window.admitted(owner.request_key);
  }
  const std::array keys{owners[0].request_key, owners[1].request_key};
  const auto prepare = [&](std::size_t member) {
    auto &owner = owners[member]; return q::prepare_generation_call(model, owner, 7, nullptr, owners,
        delivery[member].available(owner.progress.generated.size()));
  };
  {
    std::array prefills{prepare(0), prepare(1)};
    q::execute_generation_calls(model, prefills, false, nullptr, owners);
    for (std::size_t member = 0; member != 2; ++member) {
      require(!prefills[member].logits.counts.lease.valid() && owners[member].ledger->committed_count == 1 &&
          owners[member].progress.generated.size() == 1, "history controls did not finish and release their real prefill C0");
      delivery[member].acknowledge_prefix(owners[member].progress.generated.size(), 1);
    }
  }
  std::array before{e::save(model, owners[0], storage), e::save(model, owners[1], storage)};
  std::array expected{hc::counts(model, hc::consumed(*owners[0].ledger), std::span(&owners[0].session->head->next_id, 1)),
      hc::counts(model, hc::consumed(*owners[1].ledger), std::span(&owners[1].session->head->next_id, 1))};
  std::array<q::LogitsInput, 2> escaped;
  bool submitted = false;
  {
    std::array calls{prepare(0), prepare(1)};
    q::execute_generation_calls(model, calls, false, nullptr, owners, [&] {
        const auto &step = *model.active_step;
        require(step.submitted && step.phase->family->role == q::GraphRole::kVerify &&
            step.members.size() == 2 && step.input_tokens == 3 && !step.discard_stopped_members,
            "history cancel did not reach one original B2 K2 verifier");
        for (std::size_t member = 0; member != 2; ++member) {
          require(calls[member].verification_ids.size() == 3 && calls[member].draft_log_probabilities.size() == 2,
              "history cancel did not hold both real proposals per member");
          hc::same_counts(model, calls[member].logits.counts, expected[member]);
          escaped[member] = calls[member].logits.counts;
          for (const auto &row : calls[member].draft_log_probabilities)
            require(row.valid() && row.size() == static_cast<std::size_t>(model.vocabulary) * sizeof(float),
                "history cancel lost its complete normalized q owner");
        }
        submitted = true; (void)owners[0].session->stop();
        require(model.active_step->cancelled && !model.active_step->discard_stopped_members,
            "history MTP member cancellation changed the original whole-episode policy");
    });
    require(submitted && !model.active_step && !model.carrier_borrowed,
        "history cancelled cohort did not return normally after real native retirement");
    for (const auto &call : calls)
      require(call.disposition == q::EvaluationDisposition::kCancelled && !call.logits.counts.lease.valid() &&
          call.draft_log_probabilities.empty(), "normally cancelled call retained private q/C0 or claimed publication");
  } // Completed/aborted caller metadata drops q and its own C0 references.
  t::idle(model);
  for (std::size_t member = 0; member != 2; ++member) {
    unchanged_committed(model, owners[member], before[member], storage);
    hc::same_counts(model, escaped[member], expected[member]); cache_is_committed(model, *owners[member].ledger);
  }
  require(!owners[1].session->stopping && !owners[1].session->failed && owners[1].session->retirement_status.ok() &&
      owners[1].request_key == keys[1] && window.matches(keys[1]), "cancelled MTP peer permanently stopped or lost its original key");

  auto &pool = model.resources_->history_counts_pool;
  require(pool.stats().max_blocks == plan.active_sequences && pool.stats().live_blocks == 2,
      "escaped readonly C0 did not occupy the original fixed executing-cohort pool");
  const auto pending = owners[1].session->head->next_id;
  bool refused = false;
  try { (void)model.history_logits_inputs(*owners[1].ledger, std::span(&pending, 1)); }
  catch (const std::runtime_error &failure) { refused = std::string_view(failure.what()) == "lease block capacity"; }
  const auto denied = pool.allocate(static_cast<std::size_t>(model.vocabulary) * sizeof(std::int32_t));
  require(refused && !denied && denied.error().IsResourceExhausted() && pool.stats().live_blocks == 2,
      "escaped C0 caused no real bounded-pool refusal or expanded its capacity");
  cache_is_committed(model, *owners[1].ledger); unchanged_committed(model, owners[1], before[1], storage);
  escaped = {}; require(pool.stats().live_blocks == 0, "retired escaped C0 still retained original pool blocks");
  {
    auto retry = model.history_logits_inputs(*owners[1].ledger, std::span(&pending, 1));
    hc::same_counts(model, retry.counts, expected[1]); cache_is_committed(model, *owners[1].ledger);
    require(pool.stats().max_blocks == plan.active_sequences && pool.stats().live_blocks == 1,
        "same-key C0 retry used a larger or second snapshot pool");
  }
  require(pool.stats().live_blocks == 0, "C0 retry retained an unescaped snapshot");
  // Prove both old checkpoints before releasing their explicit device pins.
  // Otherwise a surviving checkpoint would legitimately prevent idle eviction
  // from freeing its version, even after the original Session is retired.
  for (auto &saved : before) { qwen_history_control_case::check_old(model, saved.checkpoint, saved.image, storage); saved.checkpoint.reset(); }

  const auto stopped = owners[0].session->try_retire();
  if (!stopped.status.IsCancelled() || stopped.status.has_native_error()) q::check(stopped.status);
  require(stopped.retired, "stopped history member was freed before shared work retired");
  owners[0].session.reset(); window.retire_active(keys[0]); window.release(keys[0]);
  require(!window.matches(keys[0]) && window.matches(keys[1]), "retiring the stopped member changed its peer's key");

  std::vector<p::StepReport> checks;
  auto image = model.export_session(*owners[1].session, storage); const auto parked = h::read_image(*image);
  const auto boundary = q::generation_boundary(*owners[1].session->head);
  const auto generated = owners[1].progress.generated; const auto observed = owners[1].progress.observed_commits();
  const auto committed = owners[1].ledger->committed_count;
  owners[1].last_idle = window.pause(keys[1]); h::retire(owners[1].session);
  require(!owners[1].session && image->ready() && !model.snapshot_versions.live,
      "idle history owner retained an original resident version after releasing its checkpoint pins");
  owners[1].session = model.restore_session(image); image.reset();
  require(q::generation_boundary(*owners[1].session->head) == boundary && owners[1].request_key == keys[1] &&
      window.matches(keys[1]) && owners[1].ledger->committed_count == committed &&
      owners[1].progress.observed_commits() == observed && owners[1].progress.generated == generated,
      "host restore resampled pending/cursor or changed the original history/request progress");
  u::same(model, parked, p::capture(model, *owners[1].session, storage), checks);
  window.wake(keys[1]); window.admitted(keys[1]);
  const std::array planned{std::size_t{3}};
  auto restored_checkpoint = owners[1].session->checkpoint();
  auto control = model.create_session(planned, restored_checkpoint, q::GraphRole::kVerify);
  const q::LogitsInputs explicit_c0{.counts = m::ready<std::int32_t>(model, q::Type::kInt32, expected[1])};
  const auto control_result = control_round(model, control, explicit_c0);
  const auto epoch = owners[1].session->head.epoch();
  const auto cursor = owners[1].session->head->sampling;
  std::array survivor{prepare(1)};
  q::execute_generation_calls(model, survivor, false, nullptr, owners, [&] {
    require(model.active_step->submitted && model.active_step->members.size() == 1 && model.active_step->input_tokens == 3,
        "history peer did not continue in its original R1 K2 geometry");
    hc::same_counts(model, survivor[0].logits.counts, expected[1]);
  });
  h::same_round(control_result, owners[1].rounds.back());
  u::same(model, p::capture(model, *control, storage), p::capture(model, *owners[1].session, storage), checks);
  require(owners[1].request_key == keys[1] && window.matches(keys[1]) && owners[1].ledger->committed_count == committed + 1 &&
      owners[1].progress.observed_commits() == observed + 1 && owners[1].session->head.epoch() == epoch + 1 &&
      owners[1].session->head->sampling == cursor.advanced(5) && !survivor[0].logits.counts.lease.valid() &&
      survivor[0].draft_log_probabilities.empty(), "history peer continuation duplicated a commit/draw or retained completed C0/q");
  cache_is_committed(model, *owners[1].ledger);
  {
    const auto next = owners[1].session->head->next_id;
    const auto expected_next = hc::counts(model, hc::consumed(*owners[1].ledger), std::span(&next, 1));
    auto next_c0 = model.history_logits_inputs(*owners[1].ledger, std::span(&next, 1));
    hc::same_counts(model, next_c0.counts, expected_next); cache_is_committed(model, *owners[1].ledger);
  }
  qwen_history_control_case::check_old(model, restored_checkpoint, parked, storage);
  delivery[1].acknowledge_prefix(owners[1].progress.generated.size(), owners[1].progress.generated.size());
  window.retire_active(keys[1]); h::retire(owners[1].session); window.release(keys[1]); h::retire(control);
  restored_checkpoint.reset();
  require(!model.snapshot_versions.live && !storage->live_images() && pool.stats().live_blocks == 0 &&
      pool.stats().max_blocks == plan.active_sequences && window.active().empty() && !window.waiting_front() &&
      model.page_inventory.free_ids.size() == plan.max_pages && model.draft_inventory.free_ids.size() == model.draft_inventory.owners.size(),
      "history controls retained old heads, images, request keys, C0 blocks or page writers"); t::idle(model);
  std::cout << "{\"status\":\"PASS\",\"case\":\"history-control\",\"backend\":\"" << h::backend_name(kind)
      << "\",\"models\":1,\"configurations\":1,\"history_only\":true,\"cancelled_B2_K2\":true,"
         "\"whole_episode_retired\":true,\"postsubmit_cancel_returned_normally\":true,\"peer_same_key_R1\":true,"
         "\"committed_counts_only\":true,\"escaped_readonly_C0\":true,\"bounded_pool_refusal\":true,"
         "\"release_then_retry\":true,\"idle_HostImage_restore\":true,\"idle_resident_versions_zero\":true,\"old_checkpoints_unchanged\":true,"
         "\"quality_claim\":false,\"performance_claim\":false,\"final_versions\":0,\"state_checks\":[";
  for (std::size_t at = 0; at != checks.size(); ++at) { if (at) std::cout << ','; p::write_step(checks[at]); }
  std::cout << "]}\n"; return 0;
}
} // namespace qwen_history_control_case
