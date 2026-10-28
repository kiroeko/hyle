#pragma once

// Same cached untrained GA weights, real K1/recursive selection and original
// request owners. Repeatability is within a policy and geometry; this does not
// assert equality with a different sampler or target-only random trajectory.
namespace qwen_mtp_sampling_case {
namespace q = qwen35;
namespace i = hyle::inference;
namespace h = qwen_host_image_case;
namespace p = qwen_packed_prefill_case;
namespace c = qwen_mtp_chunked_prefill_case;
namespace s = qwen_mtp_batch_state_case;
namespace t = qwen_mtp_stop_case;
using h::require;

inline void same(q::Model &model, const h::ImageBytes &expected, const h::ImageBytes &actual,
    std::vector<p::StepReport> &checks) {
  require(expected.identity == actual.identity, "MTP sampling state comparison changed the selected configuration identity");
  t::equal(model, expected, actual, checks);
  require(checks.back().payloads_equal, "identical MTP policy/geometry did not reproduce complete state bytes");
}

inline q::MtpRoundResult round(q::Model &model, const std::shared_ptr<q::Session> &session, bool recursive) {
  const auto &selected = session->sampling_configuration();
  const auto cursor = session->head->sampling;
  const auto epoch = session->head.epoch();
  const auto old = s::image(model, *session->head);
  const auto pending = session->head->next_id;
  std::vector<q::c::ReadLease> probabilities;
  auto ids = recursive ? model.propose_recursive(*session, 2, q::model_eos_tokens, &probabilities)
                       : std::vector<std::int32_t>{pending};
  require(session->head.epoch() == epoch && session->head->sampling == cursor &&
      s::image(model, *session->head) == old && (!recursive || ids.size() == 3) &&
      probabilities.size() == (recursive && selected.configuration.enabled ? 2u : 0u),
      "MTP proposals published a head/cursor or lost the actual policy's q rows");
  const auto role = recursive ? q::GraphRole::kVerify
      : session->head->draft->lag ? q::GraphRole::kMtpCatchupRound : q::GraphRole::kMtpRound;
  q::GenerationLedger ledger(1, 3); ledger.bind_stop_policy({}, model.vocabulary);
  q::prepare_generation_record(ledger, *session, role, std::span(&pending, 1), true, 3, 0, {}, recursive ? 2 : 0);
  model.prepare_evaluation(*session, ids, nullptr, role);
  require(model.active_step->phase->family->sampling == &selected,
      "MTP phase used another request's prepared sampler");
  model.submit_evaluation(*session);
  const auto output = q::evaluation_output(model, session, {.name = "sampling-round", .trace_states = false},
      nullptr, nullptr, nullptr, nullptr, &ledger);
  const std::array outputs{output};
  const std::array selections{q::MtpGenerationSelection{session, pending, 3, probabilities, recursive ? 2u : 1u}};
  std::array<q::MtpRoundResult, 1> result;
  if (recursive) model.finish_recursive_mtp(selections, q::model_eos_tokens, outputs, result, [](auto, const auto &) {});
  else model.finish_mtp_rounds(selections, q::model_eos_tokens, outputs, result, [](auto, const auto &) {});
  const auto expected_cursor = selected.configuration.enabled ? cursor.advanced(recursive ? 5 : 3) : q::cache::SamplingCursor{};
  require(session->head.epoch() == epoch + 1 && session->head->sampling == expected_cursor &&
      session->length() == old.length + result[0].processed_target && ledger.committed_count == 1 &&
      ledger.sampling == &selected && ledger.records[0].after.sampling == expected_cursor,
      "MTP policy selection changed its real reserved RNG, accepted length or ledger identity");
  t::idle(model); return std::move(result[0]);
}

inline void cold(q::Model &model) {
  const std::array prefills{std::size_t{2}, std::size_t{3}};
  const std::array single{std::size_t{1}};
  const std::array pair{std::size_t{2}};
  const std::array verify{std::size_t{3}};
  for (std::size_t index = 0; index != 3; ++index) {
    model.prepare_family(prefills, false, 1, q::GraphRole::kMtpPrefill, false, q::TextCompletion::kEmitNext, index);
    for (const auto rows : prefills) model.plan_replay_uses(rows, 1, 24, q::GraphRole::kMtpPrefill, q::TextCompletion::kEmitNext, index);
    for (const auto role : {q::GraphRole::kMtpSingle, q::GraphRole::kMtpCatchupSingle}) {
      model.prepare_family(single, false, 1, role, false, q::TextCompletion::kEmitNext, index);
      model.plan_replay_uses(1, 1, 24, role, q::TextCompletion::kEmitNext, index);
      if (index == 2) {
        model.prepare_family(single, false, 1, role, false, q::TextCompletion::kStateOnly, index);
        model.plan_replay_uses(1, 1, 2, role, q::TextCompletion::kStateOnly, index);
      }
    }
    for (const auto role : {q::GraphRole::kMtpRound, q::GraphRole::kMtpCatchupRound}) {
      model.prepare_family(pair, false, 1, role, false, q::TextCompletion::kEmitNext, index);
      model.plan_replay_uses(2, 1, 4, role, q::TextCompletion::kEmitNext, index);
    }
    model.prepare_family(verify, false, 1, q::GraphRole::kVerify, false, q::TextCompletion::kEmitNext, index);
    model.plan_replay_uses(3, 1, 16, q::GraphRole::kVerify, q::TextCompletion::kEmitNext, index);
    model.prepare_recursive_round_programs(index);
  }
  const std::array cohort_rows{std::size_t{3}};
  for (const auto role : {q::GraphRole::kMtpPrefill, q::GraphRole::kVerify}) {
    model.prepare_family(cohort_rows, false, 2, role, false, q::TextCompletion::kEmitNext, 1);
    model.plan_replay_uses(3, 2, 1, role, q::TextCompletion::kEmitNext, 1);
  }
  for (std::size_t index = 0; index != 3; ++index) {
    const auto &head = *model.draft_programs.at({1, true, index});
    require(head.sampling_index == index, "same-width NextN program collided across sampling configurations");
    if (index) require(model.prefix_sampling_programs.at({2, index})->sampling_index == index,
        "same-width recursive sampler collided across configurations");
  }
  require(model.draft_programs.at({1, true, 1}).get() != model.draft_programs.at({1, true, 2}).get() &&
      model.prefix_sampling_programs.at({2, 1}).get() != model.prefix_sampling_programs.at({2, 2}).get(),
      "distinct request samplers share one mutable native program");
}

inline void batched(q::Model &model, const std::shared_ptr<i::HostStateStorage> &storage) {
  const std::array<std::int32_t, 3> prompt{1, 2, 3};
  std::vector<q::GenerationRequest> requests; requests.reserve(2);
  for (std::size_t member = 0; member != 2; ++member) {
    requests.emplace_back("sampling-cohort" + std::to_string(member), prompt, std::filesystem::path{}, 4, 1);
    t::prepare_request(requests.back(), 16); requests.back().progress.initial_sampling = {919, 201 + member, 0};
  }
  i::RequestWindow window; window.prepare(2, 2);
  std::array<i::OutputDelivery, 2> delivery{i::OutputDelivery(3), i::OutputDelivery(3)};
  for (std::size_t member = 0; member != 2; ++member) {
    requests[member].request_key = window.enqueue(member); window.admitted(requests[member].request_key);
  }
  const auto prepare = [&](std::size_t member) {
    auto &request = requests[member];
    return q::prepare_generation_call(model, request, 4, nullptr, requests, delivery[member].available(request.progress.generated.size()));
  };
  std::array prefills{prepare(0), prepare(1)};
  q::execute_generation_calls(model, prefills, false, nullptr, requests, [&] {
    require(model.active_step->submitted && model.active_step->members.size() == 2 &&
        model.active_step->input_tokens == 3 && model.active_step->phase->family->sampling == &model.sampling_configuration(1),
        "sampled B2 prefill did not use its original shared configuration/carrier");
  });
  std::array<q::cache::SnapshotRef, 2> checkpoints;
  std::array<s::StateImage, 2> before;
  std::array<std::uint64_t, 2> epochs;
  for (std::size_t member = 0; member != 2; ++member) {
    auto &request = requests[member]; checkpoints[member] = request.session->checkpoint();
    before[member] = s::image(model, *checkpoints[member]); epochs[member] = request.session->head.epoch();
    delivery[member].acknowledge_prefix(request.progress.generated.size(), 1);
  }
  std::array calls{prepare(0), prepare(1)}; bool actual_four_q = false;
  q::execute_generation_calls(model, calls, false, nullptr, requests, [&] {
    const auto &step = *model.active_step;
    require(step.submitted && step.members.size() == 2 && step.input_tokens == 3 &&
        step.phase->family->role == q::GraphRole::kVerify && step.phase->family->sampling == &model.sampling_configuration(1),
        "sampled B2 verification split the actual same-policy K2 cohort");
    for (const auto &call : calls) {
      require(call.verification_ids.size() == 3 && call.draft_log_probabilities.size() == 2,
          "sampled B2 verification lost its two actual proposal distributions per member");
      for (const auto &row : call.draft_log_probabilities)
        require(row.valid() && q::take(row.view()).size() == model.vocabulary * sizeof(float),
            "sampled B2 verification did not retain a valid complete q row");
    }
    const auto &program = *model.draft_programs.at({1, true, 1});
    if (program.probabilities_output.caller_owned()) {
      const auto stats = program.probability_pool.stats();
      require(stats.live_blocks == 4 && stats.max_blocks == model.cache_plan.candidate_boundaries * model.cache_plan.active_sequences,
          "caller-owned B2 proposal pool did not cover its simultaneous original per-member allowance");
    }
    actual_four_q = true;
  });
  require(actual_four_q, "sampled B2 consumer never submitted four simultaneous q rows");
  for (std::size_t member = 0; member != 2; ++member) {
    auto &request = requests[member]; const auto &record = request.ledger->records[1];
    require(calls[member].draft_log_probabilities.empty() && request.ledger->committed_count == 2 &&
        record.requests == 2 && record.member_index == member && record.verified_rows == 3 &&
        request.session->head.epoch() == epochs[member] + 1 &&
        request.session->head->sampling == before[member].sampling.advanced(5) &&
        request.session->length() == before[member].length + record.accepted &&
        s::image(model, *checkpoints[member]) == before[member],
        "sampled B2 completion retained q or changed the original cursor/epoch/accepted state/checkpoint");
    const auto image = p::capture(model, *request.session, storage);
    require(image.boundary.length == request.session->length() && image.boundary.sampling == record.after.sampling,
        "sampled B2 completed state did not export its actual complete boundary");
    window.retire_active(request.request_key);
    delivery[member].acknowledge_prefix(request.progress.generated.size(), request.progress.generated.size());
    h::retire(request.session); window.release(request.request_key); checkpoints[member].reset();
  }
  const auto &program = *model.draft_programs.at({1, true, 1});
  require((!program.probabilities_output.caller_owned() || !program.probability_pool.stats().live_blocks) &&
      !model.snapshot_versions.live && !storage->live_images() && window.active().empty() && !window.waiting_front(),
      "sampled B2 cohort did not retire its original q/state/image/request owners");
  t::idle(model);
}

inline std::size_t manual(const std::filesystem::path &input, q::BackendKind kind) {
  auto source = q::prepare_weight_source(input / "weights.tsv", true); const auto geometry = source.roles.geometry;
  require(geometry.hidden == 64 && geometry.vocabulary == 64 && geometry.context_limit >= 16 &&
      geometry.mtp_layers == 1 && geometry.recurrent_layers() && geometry.attention_layers(),
      "MTP sampler check requires the original cached C16 GA/H64/V64 fixture");
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 16, 13, 3,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 2, 3, true, true);
  const std::array additional{q::SamplingConfiguration{true, {.7F, 8, .9F}},
      q::SamplingConfiguration{true, {1.3F, 16, .95F}}};
  q::Model model(std::move(source), geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands, SIZE_MAX,
      {.enabled = false, .policy = {}, .seed = 123}, 0, {.primary = kind}, additional);
  model.configure_draft_policy({.max_drafts = 2}); cold(model);
  const auto storage = q::take(i::HostStateStorage::create(model.execution, 1, 4u << 20));
  const std::array<std::int32_t, 3> prompt{1, 2, 3}; const std::array chunks{std::size_t{3}};
  struct Pair final {
    std::shared_ptr<q::Session> control, candidate;
    q::cache::SnapshotRef frozen_control, frozen_candidate;
    h::ImageBytes old_control, old_candidate;
  };
  std::array<Pair, 3> pairs; std::vector<p::StepReport> checks;
  for (const auto index : {2u, 0u, 1u}) {
    auto &pair = pairs[index]; const auto cursor = index ? q::cache::SamplingCursor{808, 17 + index, 0} : q::cache::SamplingCursor{};
    pair.control = model.create_session(chunks, {}, q::GraphRole::kMtpPrefill, cursor, false, q::TextCompletion::kEmitNext, index);
    pair.candidate = model.create_session(chunks, {}, q::GraphRole::kMtpPrefill, cursor, false, q::TextCompletion::kEmitNext, index);
    const auto first = t::raw(model, pair.control, prompt, q::GraphRole::kMtpPrefill);
    require(t::raw(model, pair.candidate, prompt, q::GraphRole::kMtpPrefill) == first &&
        pair.candidate->head->sampling == (index ? cursor.advanced(1) : cursor), "prefill used a foreign policy/cursor");
    pair.frozen_control = pair.control->checkpoint(); pair.frozen_candidate = pair.candidate->checkpoint();
    pair.old_control = p::capture(model, *pair.control, storage); pair.old_candidate = p::capture(model, *pair.candidate, storage);
    same(model, pair.old_control, pair.old_candidate, checks);
  }
  for (const bool recursive : {false, true}) for (const auto index : {1u, 2u, 0u}) {
    auto &pair = pairs[index]; const auto a = round(model, pair.control, recursive), b = round(model, pair.candidate, recursive);
    h::same_round(a, b); same(model, p::capture(model, *pair.control, storage), p::capture(model, *pair.candidate, storage), checks);
  }
  {
    auto &pair = pairs[2]; const auto cursor = pair.control->head->sampling;
    const std::array<std::int32_t, 1> ids{5}, suffix{6};
    const auto role = pair.control->head->draft->lag ? q::GraphRole::kMtpCatchupSingle : q::GraphRole::kMtpSingle;
    for (auto &session : {pair.control, pair.candidate})
      require(c::raw(model, session, ids, role, q::TextCompletion::kStateOnly) == -1 &&
          session->head->next_id == -1 && session->head->sampling == cursor,
          "additional-policy StateOnly produced a sample, pending token or RNG draw");
    same(model, p::capture(model, *pair.control, storage), p::capture(model, *pair.candidate, storage), checks);
    const auto next = t::raw(model, pair.control, suffix, q::GraphRole::kMtpSingle);
    require(t::raw(model, pair.candidate, suffix, q::GraphRole::kMtpSingle) == next &&
        pair.candidate->head->sampling == cursor.advanced(1), "StateOnly continuation lost its selected sampler");
    same(model, p::capture(model, *pair.control, storage), p::capture(model, *pair.candidate, storage), checks);
  }
  for (std::size_t index = 0; index != pairs.size(); ++index) {
    auto &pair = pairs[index];
    auto control_reader = model.create_session({}, pair.frozen_control, q::GraphRole::kMtpPrefill,
        {}, false, q::TextCompletion::kEmitNext, index);
    auto candidate_reader = model.create_session({}, pair.frozen_candidate, q::GraphRole::kMtpPrefill,
        {}, false, q::TextCompletion::kEmitNext, index);
    h::same_image(pair.old_control, p::capture(model, *control_reader, storage));
    h::same_image(pair.old_candidate, p::capture(model, *candidate_reader, storage));
    h::retire(control_reader); h::retire(candidate_reader);
    pair.frozen_control.reset(); pair.frozen_candidate.reset(); h::retire(pair.control); h::retire(pair.candidate);
  }
  require(!model.snapshot_versions.live, "interleaved samplers retained a hidden state version");
  batched(model, storage);

  std::vector<q::GenerationRequest> requests; requests.reserve(2);
  requests.emplace_back("sampling-owner", prompt, std::filesystem::path{}, 7, 1);
  auto &owner = requests[0]; t::prepare_request(owner, 16); owner.progress.initial_sampling = {901, 71, 0};
  i::RequestWindow window; window.prepare(1, 1); const auto key = window.enqueue(0); owner.request_key = key; window.admitted(key);
  i::OutputDelivery delivery(3); q::cache::PrefixCache prefix(1);
  const auto prepare = [&](q::GenerationRequest &request, std::size_t credit, q::cache::PrefixCache *directory = nullptr) {
    return q::prepare_generation_call(model, request, request.progress.output_limit(), directory, requests, credit);
  };
  std::array prefill{prepare(owner, 3, &prefix)}; q::execute_generation_calls(model, prefill, false, &prefix, requests);
  q::publish_generation_prompt(model, prefill[0], &prefix);
  require(owner.progress.generated.size() == 1 && owner.session->head->sampling == owner.progress.initial_sampling.advanced(1),
      "sampled request failed to publish exactly its actual prefill draw");
  auto old = owner.session->checkpoint(); const auto old_bytes = s::image(model, *old);
  requests.emplace_back("sampling-hit", prompt, std::filesystem::path{}, 1, 2);
  auto &hit = requests[1]; t::prepare_request(hit, 16); hit.progress.initial_sampling = {991, 81, 0};
  std::array hit_call{prepare(hit, 1, &prefix)};
  require(hit_call[0].full_hit, "new sampling request did not hit the actual shared consumed prefix");
  q::execute_generation_calls(model, hit_call, false, &prefix, requests);
  {
    auto control = model.create_session({}, old, q::GraphRole::kMtpPrefill, hit.progress.initial_sampling, true,
        q::TextCompletion::kEmitNext, 2);
    require(hit.prefix_sampling_calls == 1 && hit.session->head->state_identity() == old->state_identity() &&
        hit.session->head->sampling == hit.progress.initial_sampling.advanced(1) &&
        &hit.session->sampling_configuration() == &model.sampling_configuration(2) &&
        hit.ledger->sampling == &model.sampling_configuration(2),
        "full prefix hit inherited its publisher's policy, ordinal or Philox cursor");
    same(model, p::capture(model, *control, storage), p::capture(model, *hit.session, storage), checks); h::retire(control);
  }
  h::retire(hit.session); delivery.acknowledge_prefix(owner.progress.generated.size(), 1);
  const auto original = p::capture(model, *owner.session, storage);
  {
    auto replay = q::restore_mtp_session(model, *owner.ledger, {}, false);
    require(&replay->sampling_configuration() == &model.sampling_configuration(1), "strict replay used default sampling");
    same(model, original, p::capture(model, *replay, storage), checks); h::retire(replay);
  }
  const auto observed = owner.progress.observed_commits(); const auto generated = owner.progress.generated;
  auto image = model.export_session(*owner.session, storage); const auto mirror = h::read_image(*image);
  require(image->identity == model.host_image_identity(1) &&
      image->validate(model.host_image_schema(), model.host_image_identity(2)).IsInvalidArgument(),
      "host image accepted another sampler's identity");
  owner.last_idle = window.pause(key); h::retire(owner.session); owner.session = model.restore_session(image); image.reset();
  same(model, mirror, p::capture(model, *owner.session, storage), checks);
  window.wake(key); window.admitted(key);
  owner.last_idle = window.pause(key); h::retire(owner.session);
  q::restore_generation_session(model, owner, nullptr, false, requests);
  require(owner.request_key == key && window.matches(key) && owner.progress.generated == generated &&
      owner.progress.observed_commits() == observed && owner.restore_count == 1 &&
      owner.ledger->sampling == &model.sampling_configuration(1), "same-key rebuild changed request policy/observations");
  auto rebuilt_expected = original; const auto rebuilt = p::capture(model, *owner.session, storage);
  require(!rebuilt_expected.boundary.sampling_wrapper && rebuilt.boundary.sampling_wrapper,
      "historical MTP rebuild failed to create its explicit sampling wrapper");
  rebuilt_expected.boundary.sampling_wrapper = true; same(model, rebuilt_expected, rebuilt, checks);
  window.wake(key); window.admitted(key);
  std::array recursive{prepare(owner, delivery.available(owner.progress.generated.size()))};
  require(recursive[0].role == q::GraphRole::kVerify && recursive[0].maximum == 2,
      "actual request credit failed to choose the original recursive sampler");
  const auto recursive_cursor = owner.session->head->sampling;
  q::execute_generation_calls(model, recursive, false, nullptr, requests);
  require(recursive[0].draft_log_probabilities.empty(),
      "completed recursive request retained its consumed proposal distributions");
  require(owner.session->head->sampling == recursive_cursor.advanced(5) && owner.ledger->committed_count == 2,
      "recursive request lost its policy's original five-address RNG reservation");
  {
    auto replay = q::restore_mtp_session(model, *owner.ledger, {}, false);
    same(model, p::capture(model, *owner.session, storage), p::capture(model, *replay, storage), checks); h::retire(replay);
  }
  delivery.acknowledge_prefix(owner.progress.generated.size(), owner.progress.generated.size());
  const auto cursor = owner.session->head->sampling;
  h::ImageBytes expected_tail;
  {
    const std::array one{std::size_t{1}}; const auto pending = owner.session->head->next_id;
    const auto role = owner.session->head->draft->lag ? q::GraphRole::kMtpCatchupSingle : q::GraphRole::kMtpSingle;
    auto control = model.create_session(one, owner.session->checkpoint(), role, {}, false, q::TextCompletion::kEmitNext, 1);
    (void)t::raw(model, control, std::span(&pending, 1), role);
    expected_tail = p::capture(model, *control, storage); h::retire(control);
  }
  std::array tail{prepare(owner, 1)}; q::execute_generation_calls(model, tail, false, nullptr, requests);
  require(owner.request_key == key && window.matches(key) && owner.ledger->committed_count == 3 &&
      owner.session->head->sampling == cursor.advanced(1), "same-key credit tail changed selected policy or original cursor");
  same(model, expected_tail, p::capture(model, *owner.session, storage), checks);
  require(s::image(model, *old) == old_bytes, "policy/prefix/recovery changed an old complete checkpoint");
  window.retire_active(key); delivery.acknowledge_prefix(owner.progress.generated.size(), owner.progress.generated.size());
  h::retire(owner.session); window.release(key); prefix.evict_one(); prefill = {}; hit_call = {}; recursive = {}; tail = {};
  // The original old checkpoint is deliberately the last read owner here.
  const auto live_before_old_release = model.snapshot_versions.live;
  require(live_before_old_release == 1 && !storage->live_images() && window.active().empty() && !window.waiting_front(),
      "MTP sampler request kept an extra head, image or delivery owner");
  old.reset();
  require(!model.snapshot_versions.live && model.page_inventory.free_ids.size() == plan.max_pages &&
      model.draft_inventory.free_ids.size() == model.draft_inventory.owners.size(), "MTP sampling owners did not really retire");
  t::idle(model);
  std::cout << "{\"status\":\"PASS\",\"case\":\"mtp-sampling-manual\",\"backend\":\"" << h::backend_name(kind)
      << "\",\"configurations\":3,\"same_loaded_model\":true,\"interleaved_K1\":3,\"interleaved_recursive\":3,"
      "\"state_only_policy\":2,\"prefix_new_policy\":true,\"strict_replay\":true,\"same_key_rebuild\":true,"
      "\"rebuild_sampling_wrapper\":true,\"host_image_identity\":true,\"completed_q_released\":true,\"actual_B2_simultaneous_q\":4,"
      "\"final_versions\":0,\"state_checks\":[";
  for (std::size_t at = 0; at < checks.size(); ++at) { if (at) std::cout << ','; p::write_step(checks[at]); }
  std::cout << "]}\n"; return checks.size();
}

inline int run(const std::filesystem::path &input, q::BackendKind kind) {
  const auto comparisons = manual(input, kind); // Fully destroy the first Model before the natural cold consumer.
  auto source = q::prepare_weight_source(input / "weights.tsv", true); const auto geometry = source.roles.geometry;
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 16, 6, 3,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 1, 3, true, true);
  const std::array additional{q::SamplingConfiguration{false, {}}, q::SamplingConfiguration{true, {1.3F, 16, .95F}}};
  q::Model model(std::move(source), geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands, SIZE_MAX,
      {.enabled = true, .policy = {.7F, 8, .9F}, .seed = 123}, 0, {.primary = kind}, additional);
  model.configure_draft_policy({.max_drafts = 2});
  const std::vector<std::int32_t> prompt{1, 2, 3};
  const std::array<i::GenerationSessionSchedule, 3> schedules{{{"natural-random", {{prompt, 4}}},
      {"natural-greedy", {{prompt, 4}}}, {"natural-random-extra", {{prompt, 4}}}}};
  const std::array plans{q::cache::token_plan(prompt, 16), q::cache::token_plan(prompt, 16), q::cache::token_plan(prompt, 16)};
  const std::array<std::size_t, 3> policies{0, 1, 2}; const auto now = q::Clock::now();
  require(q::run_generation_sessions<q::GenerationMode::kMtp>(model, schedules, {}, false, now, now, false, plans, policies) == 0,
      "natural per-request sampling cold plan failed");
  for (std::size_t index = 0; index != 3; ++index) {
    require(model.phases.contains({1, 3, q::GraphRole::kVerify, q::TextCompletion::kEmitNext, index}) &&
        model.draft_programs.at({1, true, index})->sampling_index == index,
        "natural cold planning omitted a used recursive configuration or its actual verifier carrier");
    if (index != 1) require(model.prefix_sampling_programs.at({2, index})->sampling_index == index &&
        model.prefix_sampling_programs.at({2, index})->carrier.initialized(),
        "natural recursive sampling did not execute its original configuration's prefix program");
  }
  require(!model.snapshot_versions.live && model.page_inventory.free_ids.size() == plan.max_pages &&
      model.draft_inventory.free_ids.size() == model.draft_inventory.owners.size(), "natural sampling consumer retained model state");
  t::idle(model);
  std::cout << "{\"status\":\"PASS\",\"case\":\"mtp-sampling-policies\",\"backend\":\"" << h::backend_name(kind)
      << "\",\"sequential_models\":2,\"manual_state_checks\":" << comparisons
      << ",\"natural_requests\":3,\"natural_outputs_per_request\":4,\"natural_configurations\":3,"
      "\"natural_recursive_cold_plan\":true,\"natural_default_random_extra_greedy\":true,"
      "\"completed_q_released\":true,\"actual_B2_simultaneous_q\":4,\"synthetic_untrained\":true,\"final_versions\":0,"
      "\"quality_claim\":false,\"performance_claim\":false}\n";
  return 0;
}
} // namespace qwen_mtp_sampling_case
