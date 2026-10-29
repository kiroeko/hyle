#pragma once

// Cached untrained GA weights, original authenticated target calibration and
// the original BF16 NextN cache. This checks encoded storage and retirement,
// not trained-model quality or equality with a different cache precision.
namespace qwen_mtp_e4_case {
namespace q = qwen35;
namespace i = hyle::inference;
namespace h = qwen_host_image_case;
namespace p = qwen_packed_prefill_case;
namespace s = qwen_mtp_batch_state_case;
namespace c = qwen_mtp_chunked_prefill_case;
namespace t = qwen_mtp_stop_case;
namespace u = qwen_mtp_sampling_case;
using h::require;

inline void same(const h::ImageBytes &expected, const h::ImageBytes &actual, std::size_t &checks) {
  if (expected.layout == actual.layout && expected.payloads.size() == actual.payloads.size())
    for (std::size_t at = 0; at != expected.payloads.size(); ++at) if (expected.payloads[at] != actual.payloads[at]) {
      const auto &part = expected.layout[at];
      const auto mismatch = std::mismatch(expected.payloads[at].begin(), expected.payloads[at].end(),
          actual.payloads[at].begin(), actual.payloads[at].end());
      std::cerr << "mtp_e4_payload_difference role=" << static_cast<int>(part.role) << " layer=" << part.layer
          << " bytes=" << part.bytes << " first_byte=" << (mismatch.first - expected.payloads[at].begin()) << '\n';
    }
  h::same_image(expected, actual); ++checks;
}

inline void state_bytes(q::Model &model, const q::cache::State &head) {
  const auto image = s::image(model, head);
  require(image.target.size() == 2 && image.target[0].size() == head.length * 64 &&
      image.target[1].size() == head.length * 64 && image.payloads.size() == 6 &&
      image.payloads[2].size() == head.draft->frontier * 128 &&
      image.payloads[3].size() == head.draft->frontier * 128,
      "E4 target or plain BF16 draft changed its initialized physical row extent");
}

inline void cold(q::Model &model) {
  const std::array one{std::size_t{1}};
  const std::array two{std::size_t{2}};
  const std::array three{std::size_t{3}};
  model.prepare_family(three, false, 1, q::GraphRole::kMtpPrefill);
  model.plan_replay_uses(3, 1, 8, q::GraphRole::kMtpPrefill);
  for (const auto role : {q::GraphRole::kMtpSingle, q::GraphRole::kMtpCatchupSingle}) {
    model.prepare_family(one, false, 1, role);
    model.plan_replay_uses(1, 1, 12, role);
  }
  model.prepare_family(one, false, 1, q::GraphRole::kMtpSingle, false, q::TextCompletion::kStateOnly);
  model.plan_replay_uses(1, 1, 8, q::GraphRole::kMtpSingle, q::TextCompletion::kStateOnly);
  for (const auto role : {q::GraphRole::kMtpRound, q::GraphRole::kMtpCatchupRound}) {
    model.prepare_family(two, false, 1, role); model.plan_replay_uses(2, 1, 4, role);
  }
  model.prepare_family(three, false, 1, q::GraphRole::kVerify);
  model.plan_replay_uses(3, 1, 8, q::GraphRole::kVerify);
  for (const auto role : {q::GraphRole::kMtpPrefill, q::GraphRole::kVerify}) {
    model.prepare_family(three, false, 2, role); model.plan_replay_uses(3, 2, 3, role);
  }
  model.prepare_recursive_round_programs();
}

inline void storage_facts(q::Model &model, const q::CacheStoragePair &target) {
  model.prepare_state_storage();
  const auto &plan = model.cache_plan;
  require(plan.row_bytes.target == q::cache::KvRowBytes{64, 64} &&
      plan.row_bytes.draft == q::cache::KvRowBytes{128, 128} && model.precision.activation_type() == q::Type::kBFloat16 &&
      model.target_kv_layers == 1 && model.geometry.recurrent_layers() == 1,
      "E4 target calibration changed the draft, activation or GA geometry");
  const auto actual_target = model.target_attention_schema(0).capacity();
  const auto actual_draft = model.attention_schema.capacity();
  const auto plain = q::take(q::c::CacheStorage::plain(q::Type::kBFloat16));
  require(actual_target.key_storage == target[0] && actual_target.value_storage == target[1] &&
      actual_draft.key_storage == plain && actual_draft.value_storage == plain && target[0] != target[1],
      "MTP borrowed the target calibration for its plain draft schema");
  const auto target_bank = q::cache::add(q::cache::multiply(plan.max_pages, 16 * 64), 1);
  const auto draft_bank = q::cache::add(q::cache::multiply(plan.max_pages, 16 * 128), 1);
  require(plan.key_bank_bytes == target_bank && plan.value_bank_bytes == target_bank &&
      plan.draft_key_bank_bytes == draft_bank && plan.draft_value_bank_bytes == draft_bank &&
      model.page_inventory.keys[0].page_bytes() == 16 * 64 && model.page_inventory.values[0].page_bytes() == 16 * 64 &&
      model.draft_inventory.keys[0].page_bytes() == 16 * 128 && model.draft_inventory.values[0].page_bytes() == 16 * 128,
      "E4 target and BF16 draft arenas did not reserve their independent exact byte extents");
  auto expected_backing = q::cache::add(q::cache::multiply(2, target_bank), q::cache::multiply(2, draft_bank));
  for (std::size_t layer = 0; layer != model.state_schema.size(); ++layer) if (model.geometry.is_gdn(layer)) {
    const auto &state = model.state_schema[layer];
    expected_backing = q::cache::add(expected_backing, q::cache::state_pool_backing_bytes(q::take(state.history.required_bytes()), plan.state_slots));
    expected_backing = q::cache::add(expected_backing, q::cache::state_pool_backing_bytes(q::take(state.recurrent.required_bytes()), plan.state_slots));
  }
  expected_backing = q::cache::add(expected_backing, q::cache::state_pool_backing_bytes(model.geometry.hidden_bytes(), plan.hidden_slots));
  require(plan.backing_bytes == expected_backing && model.known_cache_backing_bytes == expected_backing,
      "E4 mixed-cache plan or actual inventory widened/halved an unrelated state pool");
  (void)model.cache_usage();
  const auto schema = model.host_image_schema();
  require(schema.attention.size() == 1 && schema.attention[0].row_bytes == q::cache::KvRowBytes{64, 64} &&
      schema.draft_row_bytes == q::cache::KvRowBytes{128, 128}, "host image schema lost distinct target and draft row bytes");
  for (const auto &[key, family] : model.families) {
    (void)key; std::vector<q::c::CacheStorage> encoders;
    for (std::size_t at = 0; at != family->graph.node_count(); ++at) {
      const auto operation = family->graph.operation(q::take(family->graph.node_id(at)));
      if (operation) if (const auto encoded = operation->custom_as<q::c::model_ops::CacheEncode>()) encoders.push_back(encoded->storage);
    }
    require(encoders.size() == 2 && encoders[0] == target[0] && encoders[1] == target[1],
        "target Graph lost its own K/V encoder or encoded its plain draft append");
  }
  for (const auto &[key, program] : model.draft_programs) {
    (void)key;
    for (std::size_t at = 0; at != program->graph.node_count(); ++at) {
      const auto operation = program->graph.operation(q::take(program->graph.node_id(at)));
      require(!operation || !operation->custom_as<q::c::model_ops::CacheEncode>(), "NextN encoded the original plain BF16 cache");
    }
  }
}

inline void rounds(q::Model &model, const std::shared_ptr<i::HostStateStorage> &storage, std::size_t &checks) {
  const std::array planned{std::size_t{3}};
  const std::array<std::int32_t, 3> prompt{1, 2, 3};
  const q::cache::SamplingCursor initial{317, 41, 0};
  auto control = model.create_session(planned, {}, q::GraphRole::kMtpPrefill, initial);
  auto candidate = model.create_session(planned, {}, q::GraphRole::kMtpPrefill, initial);
  require(t::raw(model, control, prompt, q::GraphRole::kMtpPrefill) ==
      t::raw(model, candidate, prompt, q::GraphRole::kMtpPrefill), "identical E4 prefill changed its output");
  auto old = candidate->checkpoint(); const auto old_bytes = s::image(model, *old);
  same(p::capture(model, *control, storage), p::capture(model, *candidate, storage), checks);
  for (const bool recursive : {false, true}) {
    const auto expected = u::round(model, control, recursive); const auto actual = u::round(model, candidate, recursive);
    h::same_round(expected, actual); same(p::capture(model, *control, storage), p::capture(model, *candidate, storage), checks);
    state_bytes(model, *control->head); state_bytes(model, *candidate->head);
    require(s::image(model, *old) == old_bytes, "E4 K1/K2 changed an independently retained complete checkpoint");
  }
  old.reset(); h::retire(control); h::retire(candidate); t::idle(model);
}

inline void partial_request(q::Model &model, const std::shared_ptr<i::HostStateStorage> &storage, std::size_t &checks) {
  const std::array<std::int32_t, 3> prompt{1, 2, 3}; const std::array planned{std::size_t{1}};
  std::vector<q::GenerationRequest> requests; requests.reserve(1); requests.emplace_back("e4-partial", prompt, std::filesystem::path{}, 4);
  auto &request = requests[0]; t::prepare_request(request, 16); request.progress.initial_sampling = {317, 61, 0};
  i::RequestWindow window; window.prepare(1, 1); request.request_key = window.enqueue(0); const auto key = request.request_key; window.admitted(key);
  i::OutputDelivery delivery(4); q::cache::PrefixCache prefix(1);
  auto control = model.create_session(planned, {}, q::GraphRole::kMtpSingle, request.progress.initial_sampling,
      false, q::TextCompletion::kStateOnly);
  q::cache::SnapshotRef frozen; s::StateImage frozen_bytes;
  for (std::size_t row = 0; row != prompt.size(); ++row) {
    std::array calls{q::prepare_generation_call(model, request, 4, &prefix, requests, delivery.available(request.progress.generated.size()), 1)};
    const bool final = row + 1 == prompt.size(); const auto completion = final ? q::TextCompletion::kEmitNext : q::TextCompletion::kStateOnly;
    const auto epoch_before = request.session->head.epoch();
    require(calls[0].role == q::GraphRole::kMtpSingle && calls[0].completion == completion,
        "E4 Q1 prefill changed its original role/completion");
    q::execute_generation_calls(model, calls, false, &prefix, requests, [&] {
      if (!final) c::state_only_ports(*model.active_step->phase);
    });
    q::publish_generation_prompt(model, calls[0], &prefix);
    const auto next = c::raw(model, control, std::span(prompt).subspan(row, 1), q::GraphRole::kMtpSingle, completion);
    require(request.request_key == key && window.matches(key) && request.ledger->committed_count == row + 1 &&
        request.progress.observed_commits() == row + 1 && request.progress.input_consumed() == row + 1 &&
        request.session->head->next_id == next && request.session->head.epoch() == epoch_before + 1,
        "E4 chunk changed its original request/ledger/input/epoch boundary");
    same(p::capture(model, *control, storage), p::capture(model, *request.session, storage), checks);
    if (!final) require(next == -1 && request.progress.generated.empty() && !prefix.size() &&
        request.session->head->sampling == request.progress.initial_sampling,
        "E4 StateOnly produced an output, prefix or RNG draw");
    if (!row) {
      frozen = request.session->checkpoint(); frozen_bytes = s::image(model, *frozen);
      auto parked = model.export_session(*request.session, storage); const auto before = h::read_image(*parked);
      const auto epoch = request.session->head.epoch();
      request.last_idle = window.pause(key); h::retire(request.session); request.session = model.restore_session(parked); parked.reset();
      same(before, p::capture(model, *request.session, storage), checks);
      require(request.request_key == key && window.matches(key) && request.session->head->next_id == -1 &&
          request.session->head->sampling == request.progress.initial_sampling && request.ledger->committed_count == 1,
          "E4 partial HostImage changed pending, RNG, key or original ledger");
      // A restored Session is a new publication owner. Its logical image is
      // exact; its positive owner epoch is not copied from the retired owner.
      require(epoch > 0 && request.session->head.epoch() > 0, "E4 host transfer failed to publish a real state owner");
      window.wake(key); window.admitted(key);
    }
    state_bytes(model, *request.session->head);
    if (frozen) require(s::image(model, *frozen) == frozen_bytes, "E4 prefill/restore changed an old partial checkpoint");
  }
  require(request.progress.generated.size() == 1 && request.session->head->sampling == request.progress.initial_sampling.advanced(1) &&
      prefix.size() == 1, "E4 final input did not publish exactly one pending sample and complete prefix");
  std::array calls{q::prepare_generation_call(model, request, 4, &prefix, requests, 3)};
  const auto expected = u::round(model, control, true);
  q::execute_generation_calls(model, calls, false, &prefix, requests);
  require(request.rounds.back().emitted == expected.emitted && calls[0].draft_log_probabilities.empty(),
      "E4 restored original request changed recursive outputs or retained completed q");
  same(p::capture(model, *control, storage), p::capture(model, *request.session, storage), checks);
  require(s::image(model, *frozen) == frozen_bytes, "E4 recursive continuation changed its old partial checkpoint");
  delivery.acknowledge_prefix(request.progress.generated.size(), request.progress.generated.size());
  window.retire_active(key); h::retire(request.session); window.release(key); h::retire(control); frozen.reset(); prefix.evict_one(); t::idle(model);
}

inline void batch_cancel(q::Model &model, const std::shared_ptr<i::HostStateStorage> &storage, std::size_t &checks) {
  const std::array planned{std::size_t{3}};
  std::array sessions{model.create_session(planned, {}, q::GraphRole::kMtpPrefill, q::cache::SamplingCursor{317, 81, 0}),
      model.create_session(planned, {}, q::GraphRole::kMtpPrefill, q::cache::SamplingCursor{317, 82, 0})};
  const std::array<std::array<std::int32_t, 3>, 2> prompt{{{1, 2, 3}, {3, 2, 1}}};
  const std::array seed{q::EvaluationRequest{sessions[0], prompt[0]}, q::EvaluationRequest{sessions[1], prompt[1]}};
  const std::array outputs{q::EvaluationOutput{.session = sessions[0]}, q::EvaluationOutput{.session = sessions[1]}};
  model.prepare_evaluations(seed, nullptr, q::GraphRole::kMtpPrefill); model.submit_evaluations();
  std::array<std::int32_t, 2> next{-1, -1}; model.finish_evaluations(outputs, next, [](auto, const auto &) {});
  for (const bool cancel : {false, true}) {
    std::array<q::cache::SnapshotRef, 2> old{sessions[0]->checkpoint(), sessions[1]->checkpoint()};
    const std::array before{s::image(model, *old[0]), s::image(model, *old[1])};
    const std::array epochs{sessions[0]->head.epoch(), sessions[1]->head.epoch()};
    std::array<std::vector<q::c::ReadLease>, 2> probabilities;
    const std::array ids{model.propose_recursive(*sessions[0], 2, q::model_eos_tokens, &probabilities[0]),
        model.propose_recursive(*sessions[1], 2, q::model_eos_tokens, &probabilities[1])};
    require(ids[0].size() == 3 && ids[1].size() == 3 && probabilities[0].size() == 2 && probabilities[1].size() == 2,
        "E4 B2 did not retain its four actual BF16-draft proposal distributions");
    std::array<q::MtpRoundResult, 2> expected;
    h::ImageBytes survivor_image;
    if (cancel) {
      // Explicit episode abort still withdraws the whole original cohort.
      // This is independent of stopping one member and committing its peer.
      {
        std::array aborted{model.create_session(planned, old[0], q::GraphRole::kVerify),
            model.create_session(planned, old[1], q::GraphRole::kVerify)};
        const std::array abort_epochs{aborted[0]->head.epoch(), aborted[1]->head.epoch()};
        const std::array abort_inputs{q::EvaluationRequest{aborted[0], ids[0]}, q::EvaluationRequest{aborted[1], ids[1]}};
        model.prepare_evaluations(abort_inputs, nullptr, q::GraphRole::kVerify); model.submit_evaluations();
        const auto retired = model.abort_episode();
        if (!retired.status.IsCancelled() || retired.status.has_native_error()) q::check(retired.status);
        require(retired.retired, "E4 explicit abort did not retire its original shared verifier");
        for (std::size_t member = 0; member != 2; ++member) {
          require(s::image(model, *aborted[member]->head) == before[member] &&
              aborted[member]->head.epoch() == abort_epochs[member],
              "E4 explicit abort published a head, byte, epoch or RNG state");
          (void)aborted[member]->stop(); const auto released = aborted[member]->try_retire();
          if (!released.status.IsCancelled() || released.status.has_native_error()) q::check(released.status);
          require(released.retired, "E4 explicit abort retained a Session owner"); aborted[member].reset();
        }
        t::idle(model);
      }
      // Resolve the same physical B2/K3 verifier with the same actual q rows.
      // The later survivor is compared to this original geometry, not to an
      // R1 re-execution after cancellation.
      std::array control{model.create_session(planned, old[0], q::GraphRole::kVerify),
          model.create_session(planned, old[1], q::GraphRole::kVerify)};
      const std::array control_inputs{q::EvaluationRequest{control[0], ids[0]}, q::EvaluationRequest{control[1], ids[1]}};
      const std::array control_outputs{q::EvaluationOutput{.session = control[0]}, q::EvaluationOutput{.session = control[1]}};
      const std::array control_selections{
          q::MtpGenerationSelection{control[0], old[0]->next_id, 3, probabilities[0], 2},
          q::MtpGenerationSelection{control[1], old[1]->next_id, 3, probabilities[1], 2}};
      model.prepare_evaluations(control_inputs, nullptr, q::GraphRole::kVerify); model.submit_evaluations();
      model.finish_recursive_mtp(control_selections, q::model_eos_tokens, control_outputs, expected, [](auto, const auto &) {});
      survivor_image = p::capture(model, *control[1], storage);
      h::retire(control[0]); h::retire(control[1]); t::idle(model);
    }
    const std::array inputs{q::EvaluationRequest{sessions[0], ids[0]}, q::EvaluationRequest{sessions[1], ids[1]}};
    model.prepare_evaluations(inputs, nullptr, q::GraphRole::kVerify); model.submit_evaluations();
    require(model.active_step->submitted && model.active_step->members.size() == 2 && model.active_step->input_tokens == 3,
        "E4 B2 did not submit one original shared K3 verifier");
    if (cancel) {
      auto *phase = model.active_step->phase; const auto slot = model.active_step->state_slot;
      auto *run = model.active_step->execution.run;
      const auto repairs = model.draft_program_graph_calls, decisions = model.prefix_decision_graph_calls;
      (void)sessions[0]->stop(); require(!model.active_step->cancelled && model.active_step->discard_stopped_members,
          "E4 stopped MTP member cancelled its survivor's original verifier");
      std::array dispositions{q::EvaluationDisposition::kNotPublished, q::EvaluationDisposition::kNotPublished};
      const std::array surviving_outputs{
          q::EvaluationOutput{.session = sessions[0], .disposition = &dispositions[0]},
          q::EvaluationOutput{.session = sessions[1], .disposition = &dispositions[1]}};
      const std::array selections{
          q::MtpGenerationSelection{sessions[0], old[0]->next_id, 3, probabilities[0], 2},
          q::MtpGenerationSelection{sessions[1], old[1]->next_id, 3, probabilities[1], 2}};
      std::array<q::MtpRoundResult, 2> actual; std::size_t observed = 0;
      model.finish_recursive_mtp(selections, q::model_eos_tokens, surviving_outputs, actual,
          [&](std::size_t member, const auto &) {
            require(member == 1, "E4 observed the withdrawn MTP member"); ++observed;
          });
      require(dispositions[0] == q::EvaluationDisposition::kCancelled &&
          dispositions[1] == q::EvaluationDisposition::kPublished && observed == 1 &&
          actual[0].emitted.empty() && !actual[0].processed_target && !actual[0].processed_draft &&
          model.draft_program_graph_calls == repairs + 1 && model.prefix_decision_graph_calls == decisions + 1 &&
          !sessions[1]->stopping && !sessions[1]->failed && sessions[1]->retirement_status.ok(),
          "E4 withdrawal delivered A, repeated B or retained A's sampling/repair work");
      require(sessions[0]->head.get() == old[0].get() && s::image(model, *sessions[0]->head) == before[0] &&
          sessions[0]->head.epoch() == epochs[0] && sessions[0]->head->sampling == before[0].sampling &&
          sessions[1]->head.epoch() == epochs[1] + 1 &&
          sessions[1]->head->sampling == before[1].sampling.advanced(5) &&
          sessions[1]->length() == before[1].length + actual[1].processed_target,
          "E4 withdrawal changed A's complete old boundary or B's one original commit");
      h::same_round(expected[1], actual[1]); same(survivor_image, p::capture(model, *sessions[1], storage), checks);
      for (std::size_t member = 0; member != 2; ++member)
        require(s::image(model, *old[member]) == before[member], "E4 withdrawn B2 changed an original complete checkpoint");
      require(!run->in_flight() && !phase->slots[slot].carrier.pending() && !phase->slots[slot].transactions.bound(),
          "E4 MTP withdrawal retained the original shared verifier or Store cohort");
      state_bytes(model, *sessions[1]->head); t::idle(model);
      probabilities = {};
      const auto released = sessions[0]->try_retire();
      if (!released.status.IsCancelled() || released.status.has_native_error()) q::check(released.status);
      require(released.retired, "E4 stopped member was released before actual shared retirement"); sessions[0].reset();
      h::retire(sessions[1]);
    } else {
      // The explicit prefix selector guarantees both the one-row lag0 and
      // two-row lag1 repairs. Natural K1/K2 sampling is checked separately;
      // this selected-state check does not invent an acceptance probability.
      const std::array<std::size_t, 2> accepted{1, 2};
      const auto select = [&](std::span<const std::int32_t> candidates, q::StepState &step, std::span<std::int32_t> selected) {
        require(step.members.size() == 2 && step.input_tokens == 3 && candidates.size() == 6,
            "E4 controlled prefix did not inspect the original B2 K3 result");
        for (std::size_t member = 0; member != 2; ++member) {
          step.members[member].accepted_tokens = accepted[member];
          selected[member] = candidates[member * 3 + accepted[member] - 1];
        }
      };
      const auto repair = [&](q::StepState &step) { model.repair_recursive_draft(step); };
      model.finish_selected_evaluations(outputs, next, select, repair, [](auto, const auto &) {});
      for (std::size_t member = 0; member != 2; ++member) {
        require(sessions[member]->head.epoch() == epochs[member] + 1 &&
            sessions[member]->head->sampling == before[member].sampling &&
            sessions[member]->length() == before[member].length + accepted[member] &&
            sessions[member]->head->draft->lag == member &&
            s::image(model, *old[member]) == before[member], "E4 controlled B2 changed selected length, unchanged RNG, epoch or retained bytes");
        state_bytes(model, *sessions[member]->head);
      }
      std::array control{model.create_session(planned, old[0], q::GraphRole::kVerify),
          model.create_session(planned, old[1], q::GraphRole::kVerify)};
      const std::array control_inputs{q::EvaluationRequest{control[0], ids[0]}, q::EvaluationRequest{control[1], ids[1]}};
      const std::array control_outputs{q::EvaluationOutput{.session = control[0]}, q::EvaluationOutput{.session = control[1]}};
      model.prepare_evaluations(control_inputs, nullptr, q::GraphRole::kVerify); model.submit_evaluations();
      std::array<std::int32_t, 2> control_next{-1, -1};
      model.finish_selected_evaluations(control_outputs, control_next, select, repair, [](auto, const auto &) {});
      require(next == control_next, "same-geometry E4 B2 selector changed its pending token");
      for (std::size_t member = 0; member != 2; ++member) {
        same(p::capture(model, *control[member], storage), p::capture(model, *sessions[member], storage), checks);
        require(s::image(model, *old[member]) == before[member], "E4 controlled prefix changed an original complete snapshot");
        h::retire(control[member]);
      }
    }
    t::idle(model);
  }
}

inline int run(const std::filesystem::path &input, q::BackendKind kind) {
  auto source = q::prepare_weight_source(input / "weights.tsv", true); const auto geometry = source.roles.geometry;
  require(geometry.layers == "GA" && geometry.hidden == 64 && geometry.vocabulary == 64 && geometry.context_limit >= 16 &&
      geometry.key_depth == 64 && geometry.value_depth == 64 && geometry.kv_heads == 1 && geometry.mtp_layers == 1 &&
      source.roles.cache_policy == q::prepared::CachePolicy::kFloat8OriginalCalibration && source.cache_storage.size() == 1 &&
      source.roles.cache_calibration.size() == 2 && source.cache_row_bytes.target == q::cache::KvRowBytes{64, 64} &&
      source.cache_row_bytes.draft == q::cache::KvRowBytes{128, 128}, "E4 MTP requires the authenticated cached GA target/plain-draft fixture");
  const auto target = source.cache_storage[0];
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 16, 8, 3,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 2, 3, true, true);
  q::Model model(std::move(source), geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands, SIZE_MAX,
      {.enabled = true, .policy = {1.F, 8, .9F}, .seed = 317}, 0, {.primary = kind});
  model.configure_draft_policy({.max_drafts = 2}); cold(model); storage_facts(model, target);
  const auto storage = q::take(i::HostStateStorage::create(model.execution, 1, 4u << 20));
  std::size_t checks = 0; rounds(model, storage, checks); partial_request(model, storage, checks); batch_cancel(model, storage, checks);
  require(checks == 11 && !model.snapshot_versions.live && !storage->live_images() && model.page_inventory.free_ids.size() == plan.max_pages &&
      model.draft_inventory.free_ids.size() == plan.max_pages, "E4 MTP retained a host image, version or target/draft page owner");
  for (const auto &[key, program] : model.draft_programs) {
    (void)key; require(!program->guard.execution.has_completion() && !program->guard.execution.cleanup_pending &&
        !program->hidden_transactions.bound() && !program->carrier.pending(), "E4 NextN retained original native work or state permission");
  }
  t::idle(model); (void)model.cache_usage();
  std::cout << "{\"status\":\"PASS\",\"case\":\"mtp-e4-cache\",\"backend\":\"" << h::backend_name(kind)
      << "\",\"models\":1,\"original_calibration_roles_used\":2,\"original_calibration_roles_available\":4,"
         "\"target_row_bytes\":64,\"draft_row_bytes\":128,\"target_storage\":\"E4M3FN\",\"draft_storage\":\"BF16\","
         "\"exact_cache_backing_bytes\":" << plan.backing_bytes << ",\"complete_byte_equal_checks\":" << checks
      << ",\"K1_K2\":true,\"StateOnly_no_draw\":true,\"partial_HostImage\":true,\"original_request_key\":true,"
         "\"B2_actual_verify\":true,\"controlled_B2_accepted_prefixes\":[1,2],\"controlled_B2_RNG_unchanged\":true,"
         "\"explicit_abort_whole_cohort\":true,\"B2_withdrawn_member_unchanged\":true,\"survivor_same_B2_commit\":true,"
         "\"old_checkpoints_unchanged\":true,\"final_versions\":0,\"synthetic_untrained\":true,"
         "\"quality_claim\":false,\"performance_claim\":false}\n";
  return 0;
}
} // namespace qwen_mtp_e4_case
