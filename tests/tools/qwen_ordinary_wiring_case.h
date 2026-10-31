#pragma once

#include <chrono>
#include <thread>

// Finite ordinary consumers on the cached GA fixture. The original Model,
// RequestLoop, state images and Graph/CopyBatch controls remain the owners.
namespace qwen_ordinary_wiring_case {
namespace q = qwen35;
namespace i = hyle::inference;
namespace h = qwen_host_image_case;
namespace p = qwen_packed_prefill_case;
using h::require;

inline void qualification_matrix() {
  struct Case final {
    q::OrdinaryIoCapabilities facts;
    bool packet, materialize, batch;
  };
  const std::array cases{
      Case{{.backend = q::BackendKind::kCpu}, true, false, false},
      Case{{.backend = q::BackendKind::kCuda}, true, true, true},
      Case{{.backend = q::BackendKind::kCuda, .packing = q::RowPacking::kPrefill}, true, true, true},
      Case{{.backend = q::BackendKind::kCuda, .cpu_assist = true}, true, false, false},
      Case{{.backend = q::BackendKind::kCuda, .imported_sources = true}, false, true, false},
      Case{{.backend = q::BackendKind::kCuda, .diagnostic = true}, false, false, false},
      Case{{.backend = q::BackendKind::kCuda, .completion = q::TextCompletion::kStateOnly}, true, false, false},
      Case{{.backend = q::BackendKind::kCuda, .role = q::GraphRole::kMtpPrefill}, false, false, false},
      Case{{.backend = q::BackendKind::kApple}, false, false, false},
      Case{{.backend = q::BackendKind::kVulkan}, false, false, false},
  };
  for (const auto &test : cases)
    require(test.facts.packet_inputs() == test.packet && test.facts.caller_small_results() == test.materialize &&
        test.facts.batched_small_results(true, false) == test.batch &&
        !test.facts.batched_small_results(false, false) && !test.facts.batched_small_results(true, true),
        "ordinary cold qualification changed a legal packet, materialization or two-result boundary");
  require(q::DeviceRelayCapabilities{.backend = q::BackendKind::kCuda}.eligible() &&
      !q::DeviceRelayCapabilities{.backend = q::BackendKind::kCpu}.eligible() &&
      !q::DeviceRelayCapabilities{.backend = q::BackendKind::kCuda, .cpu_assist = true}.eligible() &&
      !q::DeviceRelayCapabilities{.backend = q::BackendKind::kCuda, .scheduler = true}.eligible() &&
      !q::DeviceRelayCapabilities{.backend = q::BackendKind::kCuda, .sampling = true}.eligible() &&
      !q::DeviceRelayCapabilities{.backend = q::BackendKind::kCuda, .logits_modifiers = true}.eligible(),
      "ordinary device relay escaped its separate static sampler/backend boundary");
}

inline void cold_capacity_fallback(const std::filesystem::path &input, q::BackendKind kind) {
  auto normal_source = q::prepare_weight_source(input / "weights.tsv", false);
  auto fallback_source = q::prepare_weight_source(input / "weights.tsv", false);
  const auto geometry = normal_source.roles.geometry;
  const auto plan = q::cache::plan(geometry, normal_source.cache_row_bytes, 16, 1, 1,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 1, 1, false, true);
  const q::Model::Sampling sampler{.enabled = true, .policy = {1.F, 8, .9F}, .seed = 123};
  q::Model normal(std::move(normal_source), geometry, 16, plan, {}, false,
      q::ReplayPolicy::kCommands, SIZE_MAX, sampler, 0, {.primary = kind});
  q::Model fallback(std::move(fallback_source), geometry, 16, plan, {}, false,
      q::ReplayPolicy::kCommands, SIZE_MAX, sampler, 0, {.primary = kind});
  const std::array rows{std::size_t{1}};
  for (auto *model : {&normal, &fallback}) {
    require(model->prepare_family(rows), "ordinary fallback fixture lost its original finite Family");
    model->plan_replay_uses(1, 1, 2);
  }
  normal.prepare_execution(rows);
  const auto host = q::mem::system_resource();
  const auto before = q::take(fallback.execution_budget.usage(host));
  q::Phase *original = nullptr;
  {
    auto pressure = q::take(fallback.execution_budget.reserve(host, before.limit_bytes - before.live_bytes));
    original = &fallback.prepare_child(1, false);
    const auto &io = original->slots[0].data;
    require(!io.input_packet.prepared() && io.input_packet.allocation_refused() &&
        io.ordinary_io.inputs == q::OrdinaryInputMode::kSeparate &&
        io.ordinary_io.results == q::OrdinaryResultMode::kSeparate &&
        io.input_packet_preparation.IsResourceExhausted() && !io.input_packet_preparation.has_native_error() &&
        io.tokens.caller_owned() && io.positions.caller_owned() && io.sampling_identities.caller_owned() &&
        !io.small_results.valid() &&
        (kind != q::BackendKind::kCuda ||
            (io.small_results_preparation.IsResourceExhausted() && !io.small_results_preparation.has_native_error())) &&
        q::take(fallback.execution_budget.usage(host)).live_bytes == before.limit_bytes,
        "ordinary cold capacity refusal changed its typed cause or retained an unpublished candidate");
  }
  require(q::take(fallback.execution_budget.usage(host)).live_bytes == before.live_bytes,
      "ordinary cold fallback did not return the refused staging/metadata charge");
  fallback.prepare_execution(rows);
  auto &io = original->slots[0].data;
  const auto tokens = io.tokens.storage(), positions = io.positions.storage(), identities = io.sampling_identities.storage();
  const auto *run = q::take(original->slots[0].carrier.borrow());
  require(normal.prepare_child(1, false).slots[0].data.input_packet.prepared() &&
      normal.prepare_child(1, false).slots[0].data.ordinary_io.inputs == q::OrdinaryInputMode::kPacket &&
      (kind != q::BackendKind::kCuda || normal.prepare_child(1, false).slots[0].data.small_results.valid()),
      "ordinary control did not select its packet and applicable result batch");
  auto normal_storage = q::take(i::HostStateStorage::create(normal.execution, 1, 4u << 20));
  auto fallback_storage = q::take(i::HostStateStorage::create(fallback.execution, 1, 4u << 20));
  auto a = normal.create_session(rows, {}, q::GraphRole::kText, q::cache::SamplingCursor{123, 7, 0});
  auto b = fallback.create_session(rows, {}, q::GraphRole::kText, q::cache::SamplingCursor{123, 7, 0});
  std::int32_t id = 1;
  for (std::size_t step = 0; step < 2; ++step) {
    const auto expected = q::evaluate(normal, *a, std::span(&id, 1), "ordinary-packet-control", {}, false);
    const auto actual = q::evaluate(fallback, *b, std::span(&id, 1), "ordinary-cold-fallback", {}, false);
    require(expected == actual && a->head.epoch() == b->head.epoch() && b->length() == step + 1 &&
        b->head->sampling == q::cache::SamplingCursor{123, 7, step + 1},
        "ordinary fallback changed the original ID, epoch, boundary or random address");
    const auto control_state = p::capture(normal, *a, normal_storage);
    const auto fallback_state = p::capture(fallback, *b, fallback_storage);
    // A HostImage can restore only its own live Model/configuration. Those
    // process-local owner identities must differ across these two Models;
    // compare every logical boundary/layout/initialized byte exactly instead.
    const bool identity_distinct = control_state.identity != fallback_state.identity;
    const bool boundary_equal = h::same_boundary(control_state.boundary, fallback_state.boundary);
    const bool layout_equal = control_state.layout == fallback_state.layout;
    const bool bytes_equal = control_state.total_bytes == fallback_state.total_bytes;
    const bool payloads_equal = control_state.payloads == fallback_state.payloads;
    std::cerr << "ordinary_fallback_state step=" << step << " model_identity_distinct=" << identity_distinct
        << " boundary_equal=" << boundary_equal << " layout_equal=" << layout_equal
        << " byte_extent_equal=" << bytes_equal << " initialized_payloads_equal=" << payloads_equal << '\n';
    require(identity_distinct && normal.source_identity == fallback.source_identity &&
        control_state.identity == normal.host_image_identity(0) && fallback_state.identity == fallback.host_image_identity(0),
        "ordinary cross-Model state oracle lost either original owner identity or its equal source");
    require(boundary_equal && layout_equal && bytes_equal && payloads_equal,
        "ordinary fallback changed an exact logical boundary, state layout or initialized payload byte");
    require(&fallback.prepare_child(1, false) == original && !io.input_packet.prepared() && !io.small_results.valid() &&
        !io.ordinary_io.packet() && !io.ordinary_io.batched_results() &&
        io.tokens.storage().native_handle() == tokens.native_handle() &&
        io.positions.storage().native_handle() == positions.native_handle() &&
        io.sampling_identities.storage().native_handle() == identities.native_handle() &&
        q::take(original->slots[0].carrier.borrow()) == run && !fallback.active_step && !fallback.carrier_borrowed,
        "warm ordinary fallback replaced its cold fixed owners, Invocation or selected IO route");
    id = actual;
  }
  h::retire(a); h::retire(b);
  require(!normal.snapshot_versions.live && !fallback.snapshot_versions.live &&
      !normal_storage->live_images() && !fallback_storage->live_images(),
      "ordinary fallback left an original state/image owner after actual retirement");
}

inline void random_r2_cancel(const std::filesystem::path &input, q::BackendKind kind) {
  auto source = q::prepare_weight_source(input / "weights.tsv", false);
  const auto geometry = source.roles.geometry;
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 16, 2, 2,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 2, 1, false, true);
  q::Model model(std::move(source), geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands,
      SIZE_MAX, {.enabled = true, .policy = {1.F, 0, 1.F}, .seed = 123}, 0, {.primary = kind});
  const std::array rows{std::size_t{1}, std::size_t{2}};
  for (const auto width : {std::size_t{1}, std::size_t{2}}) {
    require(model.prepare_family(rows, false, width), "ordinary control lost its original R1/R2 Families");
    for (const auto count : rows) model.plan_replay_uses(count, width, 4);
  }
  for (const auto width : {std::size_t{1}, std::size_t{2}}) model.prepare_execution(rows, width);
  const std::array turns{i::GenerationTurn{{1, 2}, 3}};
  std::vector<q::GenerationRequest> requests; requests.reserve(2);
  for (std::size_t member = 0; member < 2; ++member) {
    requests.emplace_back(member ? "ordinary-survivor" : "ordinary-cancelled", turns[0].prompt,
        std::filesystem::path{}, turns[0].max_outputs);
    auto &request = requests.back(); request.progress.bind_turns(turns);
    request.progress.initial_sampling = {123, 100 + member, 0};
    request.input = q::cache::token_plan(turns[0].prompt, 16);
    request.ledger = std::make_unique<q::GenerationLedger>(q::generation_record_capacity(turns), 1);
    request.round_emitted.resize(2);
    for (auto &ids : request.round_emitted) ids.reserve(1);
    request.turn_observations.resize(1);
    request.warm_inputs.push_back(std::make_shared<std::vector<std::int32_t>>(turns[0].prompt));
  }
  std::array deliveries{i::OutputDelivery(3), i::OutputDelivery(3)};
  i::RequestLoop loop; loop.prepare(2, 2);
  std::size_t submissions = 0, observed_members = 0, copied_pairs = 0, r1_steps = 0;
  bool cancelled = false;
  struct StopView final {
    std::span<q::GenerationRequest> requests;
    hyle::Status stop(i::RequestKey key) const noexcept { return requests[key.slot].session->stop(); }
  } stop_view{requests};
  const auto execute = [&](std::span<const i::RequestKey> active) {
    std::vector<q::PreparedGenerationCall> calls; calls.reserve(active.size());
    for (const auto key : active) {
      auto &request = requests[key.slot];
      calls.push_back(q::prepare_generation_call(model, request, request.progress.output_limit(), nullptr,
          requests, loop.credit(key, request.progress.generated.size())));
    }
    q::Phase *phase = nullptr;
    const q::cache::State *survivor = nullptr;
    const auto submitted = [&] {
      ++submissions; phase = model.active_step->phase;
      const auto &io = phase->slots[model.active_step->state_slot].data;
      require(io.input_packet.prepared() && io.input_packet.sampling_identities().valid() &&
          io.ordinary_io.packet() && io.ordinary_io.packet_sampling_identities &&
          io.ordinary_io.batched_results() == (kind == q::BackendKind::kCuda) &&
          io.small_results.valid() == (kind == q::BackendKind::kCuda),
          "ordinary random control did not select its actual packet/result qualification");
      if (cancelled) { require(active.size() == 1 && active[0] == requests[1].request_key,
          "ordinary survivor did not resume through its original R1 carrier"); ++r1_steps; return; }
      require(active.size() == 2 && model.active_step->submitted && model.active_step->execution.has_completion(),
          "ordinary partial cancellation did not follow an accepted original R2 Graph");
      survivor = model.active_step->members[1].next.get();
      const auto before = q::generation_boundary(*requests[0].session->head);
      const auto epoch = requests[0].session->head.epoch();
      q::check(loop.cancel(requests[0].request_key, stop_view));
      require(model.active_step->discard_stopped_members && !model.active_step->cancelled &&
          requests[0].session->head.epoch() == epoch && q::generation_boundary(*requests[0].session->head) == before &&
          !requests[0].ledger->committed_count && !requests[1].session->stopping,
          "ordinary cancellation mutated its committed boundary or stopped the peer");
    };
    const auto observed = [&](const q::GenerationBatchFacts &facts) {
      require(facts.members.size() == active.size(), "ordinary batch observer lost an actual member");
      for (const auto &call : facts.members) {
        if (call.disposition == q::EvaluationDisposition::kCancelled) continue;
        require(call.disposition == q::EvaluationDisposition::kPublished && call.request == &requests[1],
            "ordinary cancelled member reached output publication");
        ++observed_members;
      }
    };
    q::execute_generation_calls(model, calls, false, nullptr, requests, submitted, nullptr, observed);
    if (!cancelled) {
      require(requests[1].session->head.get() == survivor && requests[1].ledger->committed_count == 1 &&
          !requests[0].ledger->committed_count && requests[0].progress.generated.empty() &&
          calls[0].disposition == q::EvaluationDisposition::kCancelled,
          "ordinary cancellation replaced the survivor or published a withdrawn candidate");
      cancelled = true;
    }
    if (kind == q::BackendKind::kCuda) {
      const auto &batch = phase->slots[0].data.small_results;
      const auto report = batch.report();
      require(!batch.in_flight() && report.copies == 2 && report.active_copies == 2 &&
          report.phases == 1 && report.phase_completions == 1 && report.host_waits <= 1,
          "ordinary random batch did not retire its original two-result copy phase");
      ++copied_pairs;
    }
    for (auto &call : calls) q::publish_generation_prompt(model, call, nullptr);
  };
  q::SynchronousGenerationWave wave{model, execute};
  q::GenerationRequestAdapter adapter{model, static_cast<q::cache::PrefixCache *>(nullptr), std::span(requests), false,
      static_cast<std::vector<std::size_t> *>(nullptr), wave};
  for (std::size_t member = 0; member < 2; ++member) requests[member].request_key = loop.enqueue(member, &deliveries[member]);
  const auto stopped_key = requests[0].request_key, survivor_key = requests[1].request_key;
  for (std::size_t pump = 0; loop.matches(stopped_key) || loop.matches(survivor_key); ++pump) {
    require(pump < 8, "finite ordinary random requests did not finish");
    (void)loop.pump_once(adapter);
    if (loop.matches(survivor_key))
      require(loop.acknowledge(survivor_key, requests[1].progress.generated.size(), adapter),
          "ordinary survivor output ACK was refused");
  }
  require(submissions == 3 && observed_members == 3 && r1_steps == 2 && cancelled &&
      copied_pairs == (kind == q::BackendKind::kCuda ? 3u : 0u) &&
      requests[1].progress.generated.size() == 3 && requests[1].ledger->committed_count == 3 &&
      requests[1].ledger->records[2].after.sampling == q::cache::SamplingCursor{123, 101, 3} &&
      !requests[0].ledger->committed_count && !requests[0].session && !requests[1].session &&
      !model.active_step && !model.successor_step && !model.carrier_borrowed && !model.snapshot_versions.live,
      "ordinary random cancellation/continuation changed IDs, draw budget, ledger or final ownership");
}

#if defined(HYLE_COMPUTE_CUDA)
inline void ordinary_decode_recordings(const std::filesystem::path &input) {
  const auto geometry = q::read_weight_geometry(input / "weights.tsv");
  const std::array rows{std::size_t{1}, std::size_t{3}};
  const auto make_model = [&](q::BackendKind backend) {
    auto source = q::prepare_weight_source(input / "weights.tsv", false);
    const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 16, 1, 3,
        q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 1, 1, false, true, 2);
    const std::array random{q::SamplingConfiguration{true, {1.F, 8, .9F}}};
    const auto replay = backend == q::BackendKind::kCuda ? q::ReplayPolicy::kPreferCapture : q::ReplayPolicy::kCommands;
    auto model = std::make_unique<q::Model>(std::move(source), geometry, 16, plan, std::filesystem::path{},
        false, replay, SIZE_MAX, q::Model::Sampling{.enabled = false},
        0, q::BackendSelection{.primary = backend}, random);
    require(model->prepare_family(rows), "decode recording fixture lost its finite ordinary Family");
    for (const auto count : rows) model->plan_replay_uses(count, 1, 16);
    model->freeze_shape_plan(); model->prepare_state_storage(); model->prepare_execution(rows);
    if (backend == q::BackendKind::kCuda) model->prepare_relay_execution();
    return model;
  };
  const auto idle = [&](const q::Model &model) {
    require(!model.active_step && !model.successor_step && !model.carrier_borrowed &&
        !model.snapshot_versions.live && model.page_inventory.free_ids.size() == model.cache_plan.max_pages,
        "ordinary decode recording retained private work, versions, pages or a carrier loan");
  };
  struct Result final {
    std::vector<std::int32_t> ids;
    std::vector<q::GenerationCommitRecord> records;
    std::vector<std::uint64_t> epochs;
    h::ImageBytes image;
  };
  const auto journey = [&](q::Model &model, std::shared_ptr<q::Session> prepared_session = {}) {
    const std::array<std::int32_t, 3> prompt{1, 2, 3};
    std::vector<q::GenerationRequest> requests;
    requests.emplace_back("ordinary-after-recording", prompt, std::filesystem::path{}, 4, 0,
        i::RequestStopPolicy({}, true));
    auto &request = requests.front();
    request.session = std::move(prepared_session);
    i::RequestWindow keys; keys.prepare(1, 1);
    request.request_key = keys.enqueue(0);
    request.warm_inputs.push_back(std::make_shared<std::vector<std::int32_t>>(prompt.begin(), prompt.end()));
    request.prompt = *request.warm_inputs.front();
    request.ledger = std::make_unique<q::GenerationLedger>(4, 1);
    request.round_emitted.resize(3); for (auto &emitted : request.round_emitted) emitted.reserve(1);
    q::GenerationContinuation continuation; continuation.prepare(1);
    hyle::util::ScopeGuard cleanup{[&]() noexcept {
      continuation.reset(); (void)model.abort_episode();
      if (request.session) {
        request.session->stop().IgnoreError(); (void)request.session->drain();
        (void)request.session->try_retire(); request.session.reset();
      }
    }};
    Result result;
    const auto observed = [&](const q::GenerationBatchFacts &facts) {
      require(facts.members.size() == 1 && facts.members[0].request == &request &&
          facts.members[0].disposition == q::EvaluationDisposition::kPublished &&
          request.session->head->sampling == q::cache::SamplingCursor{},
          "ordinary post-recording publication changed its request, disposition or greedy RNG");
      result.epochs.push_back(request.session->head.epoch());
    };
    while (!request.progress.turn_done) {
      require(request.ledger->committed_count < 4, "ordinary post-recording request exceeded its finite ledger");
      std::array calls{q::prepare_generation_call(model, request, 4, nullptr, requests)};
      if (!calls[0].generation) {
        require(!request.session->length() && request.session->head->next_id == -1,
            "new ordinary Session inherited a recording frontier");
        for (const auto &bytes : qwen_state_reuse_case::state_bytes(model, *request.session->head))
          require(std::ranges::all_of(bytes, [](auto byte) { return byte == std::byte{}; }),
              "new ordinary Session did not clear the recycled recording state");
      }
      q::execute_generation_calls(model, calls, false, nullptr, requests, nullptr, &continuation, observed);
    }
    continuation.reset();
    result.ids = request.progress.generated;
    result.records.assign(request.ledger->records.begin(), request.ledger->records.begin() + request.ledger->committed_count);
    auto storage = q::take(i::HostStateStorage::create(model.execution, 1, 4u << 20));
    result.image = p::capture(model, *request.session, storage);
    require(result.ids.size() == 4 && result.records.size() == 4 && result.epochs.size() == 4,
        "private decode recording polluted the real request's output or ledger counts");
    h::retire(request.session); idle(model); cleanup.dismiss(); return result;
  };
  auto control_model = make_model(q::BackendKind::kCuda), recorded_model = make_model(q::BackendKind::kCuda);
  const auto recording = recorded_model->prepare_ordinary_decode_recordings(0);
  require(recording.preparations == 5 && recording.captures > 0 && recording.seconds > 0 &&
      recording.capture_preparation_seconds > 0 && recording.capture_preparation_seconds <= recording.seconds,
      "ordinary cold recording did not prepare its five original Invocation tuples");
  idle(*recorded_model);
  std::shared_ptr<q::Session> prepared_session;
  const auto session_recording = recorded_model->prepare_ordinary_decode_recordings(0, &prepared_session);
  require(session_recording.preparations == 5 && session_recording.seconds > 0 &&
      session_recording.capture_preparation_seconds >= 0 &&
      session_recording.capture_preparation_seconds <= session_recording.seconds &&
      prepared_session && prepared_session->head && !prepared_session->length() &&
      prepared_session->head->next_id == -1 && prepared_session->head->sampling == q::cache::SamplingCursor{} &&
      !prepared_session->pending && !prepared_session->stopping && !prepared_session->failed &&
      !recorded_model->active_step && !recorded_model->successor_step && !recorded_model->carrier_borrowed &&
      recorded_model->snapshot_versions.live == 1 &&
      recorded_model->page_inventory.free_ids.size() == recorded_model->cache_plan.max_pages,
      "ordinary cold recording did not hand out exactly one empty Session without retaining private work");
  const auto control = journey(*control_model), candidate = journey(*recorded_model, std::move(prepared_session));
  require(!prepared_session, "ordinary recording handoff retained another Session owner");
  require(control.ids == candidate.ids && control.epochs == candidate.epochs &&
      control.image.identity != candidate.image.identity &&
      control.image.identity == control_model->host_image_identity(0) &&
      candidate.image.identity == recorded_model->host_image_identity(0) &&
      control_model->source_identity == recorded_model->source_identity &&
      h::same_boundary(control.image.boundary, candidate.image.boundary) &&
      control.image.layout == candidate.image.layout && control.image.total_bytes == candidate.image.total_bytes &&
      control.image.payloads == candidate.image.payloads,
      "ordinary cold recording changed real IDs, epochs or a complete logical state image");
  for (std::size_t at = 0; at != control.records.size(); ++at)
    require(control.records[at].same_result(candidate.records[at]) &&
        control.records[at].committed == candidate.records[at].committed &&
        control.records[at].user_output == candidate.records[at].user_output &&
        control.records[at].imported == candidate.records[at].imported &&
        control.records[at].turn == candidate.records[at].turn &&
        std::ranges::equal(control.records[at].ids(), candidate.records[at].ids()) &&
        std::ranges::equal(control.records[at].committed_ids(), candidate.records[at].committed_ids()),
        "ordinary cold recording changed a real authoritative ledger input, RNG or decision");
  const auto skipped = [&](q::Model &model, std::size_t sampling) {
    const auto actual = model.prepare_ordinary_decode_recordings(sampling);
    require(!actual.preparations && !actual.captures && actual.seconds == 0 && actual.capture_preparation_seconds == 0,
        "ineligible ordinary route performed private cold recording");
    idle(model);
  };
  skipped(*recorded_model, 1);
  auto &slot = recorded_model->prepare_child(1, false).slots[0];
  const auto replay = slot.replay_mode; slot.replay_mode = q::g::ReplayMode::kCommands;
  hyle::util::ScopeGuard restore_replay{[&]() noexcept { slot.replay_mode = replay; }};
  skipped(*recorded_model, 0);
  auto cpu_model = make_model(q::BackendKind::kCpu); skipped(*cpu_model, 0);
  std::cout << "{\"case\":\"ordinary-decode-recording\",\"recorded_preparations\":" << recording.preparations
      << ",\"captures\":" << recording.captures
      << ",\"session_preparations\":" << session_recording.preparations
      << ",\"session_captures\":" << session_recording.captures
      << ",\"real_ids_ledger_epochs_images_equal\":true,\"recycled_state_zeroed\":true,\"empty_session_handoff\":true,"
         "\"random_commands_cpu_skipped\":true,\"final_private_versions\":0}\n";
}

inline void pending_graph_pumps(const std::filesystem::path &input) {
  auto source = q::prepare_weight_source(input / "weights.tsv", false);
  const auto geometry = source.roles.geometry;
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 16, 2, 2,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 2, 1, false, true);
  q::Model model(std::move(source), geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands,
      SIZE_MAX, {.enabled = true, .policy = {1.F, 0, 1.F}, .seed = 123}, 0, {.primary = q::BackendKind::kCuda});
  const std::array rows{std::size_t{2}};
  require(model.prepare_family(rows, false, 2), "pending ordinary Graph lost its original R2 Family");
  model.plan_replay_uses(2, 2, 3); model.prepare_execution(rows, 2);
  auto storage = q::take(i::HostStateStorage::create(model.execution, 1, 4u << 20));
  const std::array<std::int32_t, 2> prompt{1, 2};
  std::array<h::ImageBytes, 2> expected;
  {
    // Prime this exact original R2 provider before inserting a Queue gate;
    // cold native allocation is not what the nonblocking contract measures.
    std::array<std::shared_ptr<q::Session>, 2> warm;
    std::array<q::EvaluationRequest, 2> inputs;
    std::array<q::EvaluationOutput, 2> outputs;
    std::array<std::int32_t, 2> next;
    for (std::size_t member = 0; member < 2; ++member) {
      warm[member] = model.create_session(rows, {}, q::GraphRole::kText, q::cache::SamplingCursor{123, 100 + member, 0});
      inputs[member] = {warm[member], prompt}; outputs[member].session = warm[member];
    }
    model.prepare_evaluations(inputs); model.submit_evaluations();
    model.finish_evaluations(outputs, next, [](std::size_t, const q::CompletedEvaluationView &) {});
    for (std::size_t member = 0; member < 2; ++member) {
      expected[member] = p::capture(model, *warm[member], storage); h::retire(warm[member]);
    }
  }
  const std::array turns{i::GenerationTurn{{1, 2}, 1}};
  std::vector<q::GenerationRequest> requests; requests.reserve(2);
  for (std::size_t member = 0; member < 2; ++member) {
    requests.emplace_back(member ? "pending-B" : "pending-A", prompt, std::filesystem::path{}, 1);
    auto &request = requests.back(); request.progress.bind_turns(turns);
    request.progress.initial_sampling = {123, 100 + member, 0};
    request.input = q::cache::token_plan(prompt, 16);
    request.ledger = std::make_unique<q::GenerationLedger>(q::generation_record_capacity(turns), 1);
    request.turn_observations.resize(1);
    request.warm_inputs.push_back(std::make_shared<std::vector<std::int32_t>>(turns[0].prompt));
  }
  i::RequestLoop loop; loop.prepare(2, 2);
  std::array deliveries{i::OutputDelivery(1), i::OutputDelivery(1)};
  q::GenerationContinuation continuation; continuation.enabled = false; continuation.prepare(2);
  inference_readback_batch_case::NativeGate gate;
  std::size_t submissions = 0, observations = 0;
  std::array<const q::cache::State *, 2> heads{};
  std::array<std::uint64_t, 2> epochs{};
  std::array<q::GenerationBoundary, 2> boundaries{};
  q::StepState *original = nullptr;
  const auto submitted = [&] { ++submissions; };
  const auto observed = [&](const q::GenerationBatchFacts &facts) {
    require(facts.members.size() == 2 && !gate.blocked(),
        "pending Graph published before the original native Queue was released");
    for (std::size_t member = 0; member < 2; ++member) {
      const auto &call = facts.members[member];
      require(call.disposition == q::EvaluationDisposition::kPublished && call.request == &requests[member] &&
          requests[member].ledger->committed_count == 1,
          "ready pending Graph duplicated or omitted an authoritative member record");
      h::same_image(expected[member], p::capture(model, *requests[member].session, storage));
      ++observations;
    }
  };
  const auto ready = [&](i::RequestWave wave) {
    require(!submissions && wave.keys.size() == 2, "pending Graph started another original episode");
    std::vector<q::PreparedGenerationCall> calls; calls.reserve(2);
    for (std::size_t member = 0; member < 2; ++member) {
      auto &request = requests[wave.keys[member].slot];
      calls.push_back(q::prepare_generation_call(model, request, 1, nullptr, requests, wave.credits[member].additional));
      heads[member] = request.session->head.get(); epochs[member] = request.session->head.epoch();
      boundaries[member] = q::generation_boundary(*request.session->head);
    }
    q::SynchronousGenerationStepAdapter step(model, std::span(calls), false, false, 0, submitted, observed, &continuation);
    step.prepare(i::StepPosition::kCurrent);
    gate.arm(model.execution); // Inputs are prepared before the original Queue is held.
    step.submit(i::StepPosition::kCurrent); original = model.active_step.get();
    continuation.hold_current(calls, true);
    continuation.drive_cursor = {true, i::StepSuccessor::kNone};
    require(i::drive_steps(step, {.resume = true, .cursor = &continuation.drive_cursor}) == i::StepDriveResult::kPending,
        "real pending original Graph did not yield through the shared driver");
  };
  const auto advance = [&](i::RequestWave wave, std::span<std::size_t> reserved) {
    return q::advance_generation_wave(model, wave, reserved, false, continuation, ready, submitted, observed);
  };
  q::GenerationRequestAdapter adapter{model, static_cast<q::cache::PrefixCache *>(nullptr), std::span(requests), false,
      static_cast<std::vector<std::size_t> *>(nullptr), advance, {}, &continuation};
  for (std::size_t member = 0; member < 2; ++member) requests[member].request_key = loop.enqueue(member, &deliveries[member]);
  for (std::size_t pump = 0; pump < 3; ++pump) {
    const auto result = loop.pump_once(adapter);
    require(result.waiting_retirement && !result.completed && gate.blocked() && model.active_step.get() == original &&
        !model.successor_step && model.carrier_borrowed == 1 && submissions == 1 && !observations &&
        continuation.drive_cursor.awaiting_current && continuation.pending_synchronous &&
        continuation.pending_calls.size() == 2,
        "repeated real pending pump resubmitted, lost its native owner or published output");
    for (std::size_t member = 0; member < 2; ++member) {
      const auto &request = requests[member];
      require(request.session->head.get() == heads[member] && request.session->head.epoch() == epochs[member] &&
          q::generation_boundary(*request.session->head) == boundaries[member] &&
          !request.ledger->committed_count && !request.progress.observed_commits() && request.progress.generated.empty() &&
          !request.progress.prefilled && !request.progress.input_consumed() && !deliveries[member].acknowledged() &&
          loop.credit(request.request_key, 0) == 0,
          "pending pump changed committed state identity, RNG/epoch, ledger, progress or its reserved output credit");
    }
  }
  gate.finish();
  const auto deadline = q::Clock::now() + std::chrono::seconds(30);
  for (std::size_t pump = 0; loop.matches(requests[0].request_key) || loop.matches(requests[1].request_key); ++pump) {
    require(q::Clock::now() < deadline, "released original Graph did not complete within its real completion deadline");
    const auto result = loop.pump_once(adapter);
    for (std::size_t member = 0; member < 2; ++member) if (loop.matches(requests[member].request_key))
      require(loop.acknowledge(requests[member].request_key, requests[member].progress.generated.size(), adapter),
          "released pending Graph output ACK was refused");
    if (result.waiting_retirement && !result.executed) std::this_thread::yield();
  }
  require(submissions == 1 && observations == 2 && continuation.pending_calls.empty() &&
      !continuation.drive_cursor.awaiting_current && !model.active_step && !model.successor_step &&
      !model.carrier_borrowed && !model.snapshot_versions.live && !storage->live_images(),
      "released pending Graph retained its host/native/state owner or published twice");
}
#endif

inline int run(const std::filesystem::path &input, q::BackendKind kind) {
  require(kind == q::BackendKind::kCpu || kind == q::BackendKind::kCuda,
      "ordinary wiring case requires the actual CPU/CUDA Backend");
  const auto geometry = q::read_weight_geometry(input / "weights.tsv");
  require(geometry.hidden == 64 && geometry.vocabulary == 64 && geometry.layers == "GA" && geometry.context_limit >= 16,
      "ordinary wiring requires the cached untrained GA/H64/V64/C16 fixture");
  qualification_matrix();
  cold_capacity_fallback(input, kind);
  random_r2_cancel(input, kind);
#if defined(HYLE_COMPUTE_CUDA)
  if (kind == q::BackendKind::kCuda) {
    ordinary_decode_recordings(input);
    pending_graph_pumps(input);
  }
#endif
  std::cout << "{\"status\":\"PASS\",\"case\":\"ordinary-wiring\",\"backend\":\"" << h::backend_name(kind)
      << "\",\"cold_capacity_fallback_graphs\":2,\"fixed_fallback_owners\":true,\"full_state_rng_epoch_equal\":true,"
         "\"random_r2_partial_cancel\":true,\"survivor_r1_graphs\":2,\"native_fault_injected\":false,\"cuda_pending_pumps\":"
      << (kind == q::BackendKind::kCuda ? 3 : 0) << "}\n";
  return 0;
}
} // namespace qwen_ordinary_wiring_case
