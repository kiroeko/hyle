#pragma once

// Natural greedy RequestLoop consumer. The original Model/Continuation owns
// every submit, acceptance decision, Source permission and retirement.
namespace qwen_mtp_natural_relay_case {
namespace q = qwen35;
namespace i = hyle::inference;
namespace h = qwen_host_image_case;
namespace p = qwen_packed_prefill_case;
namespace s = qwen_mtp_batch_state_case;
using h::require;
constexpr std::size_t outputs = 40;

struct Batch final {
  std::size_t ordinal, quota, reserved, additional, emitted, accepted, draft_rows;
  bool generation, discarded;
};
struct Result final {
  std::vector<std::int32_t> ids;
  std::vector<q::GenerationCommitRecord> records;
  std::optional<s::StateImage> final_state;
  std::vector<Batch> batches;
  std::size_t submits = 0, source_submits = 0, pending_pairs = 0, deferred_waves = 0,
      reserved_two_grants = 0, successors = 0, publications = 0, discards = 0, refusals = 0,
      accepted_drafts = 0, rejected_drafts = 0, actual_draft_rows = 0, discarded_draft_rows = 0;
  bool stopped = false;
};

inline Result journey(q::Model &model, std::span<const std::int32_t> prompt, bool overlap) {
  require(!model.active_step && !model.successor_step && !model.carrier_borrowed && !model.snapshot_versions.live,
      "natural K1 journey started with a retained original episode or state");
  const auto submitted_draft_begin = model.submitted_draft_rows;
  const std::array turns{i::GenerationTurn{std::vector<std::int32_t>(prompt.begin(), prompt.end()), outputs}};
  std::vector<q::GenerationRequest> requests;
  requests.reserve(1); requests.emplace_back(overlap ? "natural-source" : "natural-serial", prompt,
      std::filesystem::path{}, outputs);
  auto &request = requests.front();
  request.progress.bind_turns(turns); request.progress.initial_sampling = {};
  request.input = q::cache::token_plan(prompt, model.capacity);
  request.ledger = std::make_unique<q::GenerationLedger>(q::generation_record_capacity(turns), 2);
  request.round_emitted.resize(outputs - 1);
  for (auto &row : request.round_emitted) row.reserve(2);
  request.turn_observations.resize(1);
  request.warm_inputs.push_back(std::make_shared<std::vector<std::int32_t>>(prompt.begin(), prompt.end()));
  i::OutputDelivery delivery(4); // Two real output credits for A and two for B.
  i::RequestLoop loop; loop.prepare(1, 1);
  q::GenerationContinuation continuation; continuation.enabled = overlap; continuation.prepare(1);
  Result result; result.batches.reserve(outputs * 2);
  // Borrow only each accepted Append's immutable geometry, without retaining
  // an episode or Source. Match it to the existing ordered retirement callback.
  struct SubmittedDraft final { std::size_t rows; bool successor; };
  std::vector<SubmittedDraft> submitted_drafts; submitted_drafts.reserve(outputs * 2);
  std::size_t retired_submissions = 0;
  std::vector<std::weak_ptr<q::StepSources>> sources; sources.reserve(outputs * 2);
  std::size_t observed = 0, emitted = 0, reserved = 0, additional = 0;
  hyle::util::ScopeGuard cleanup{[&]() noexcept {
    continuation.reset(); (void)model.abort_episode();
    if (request.session) {
      request.session->stop().IgnoreError(); (void)request.session->drain();
      (void)request.session->try_retire(); request.session.reset();
    }
  }};
  const auto remember_source = [&](const q::StepState &step) {
    if (!step.execution.sources) return;
    const auto &views = *step.execution.sources;
    require(step.submitted && !step.execution.completion.valid() && step.execution.has_completion() &&
        views.submission.valid() && views.primary_output.valid() && views.extra_outputs.size() == 1 &&
        views.extra_outputs.front().valid() && views.mtp_full_candidate &&
        views.states.size() == step.phase->state_sources.size() &&
        views.pages.size() == model.draft_bank + 2,
        "natural K1 Source lost its sole Completion or complete original output/bank views");
    sources.push_back(step.execution.sources);
  };
  const auto submitted = [&] {
    ++result.submits;
    require(model.active_step && model.active_step->submitted && model.active_step->execution.has_completion(),
        "natural K1 submit callback did not observe the original accepted target");
    const bool successor = model.successor_step && model.successor_step->submitted;
    const auto &new_step = successor ? *model.successor_step : *model.active_step;
    const auto draft_rows = new_step.draft_layout.facts().query_rows;
    require(new_step.members.size() == 1 && draft_rows > 0 &&
        new_step.members.front().draft_layout.facts().query_rows == draft_rows,
        "natural K1 accepted submission lost its original member Append geometry");
    submitted_drafts.push_back({draft_rows, successor});
    if (new_step.execution.sources) { ++result.source_submits; remember_source(new_step); }
    if (!successor) return;
    const auto &a = *model.active_step; const auto &b = *model.successor_step;
    require(overlap && a.members.size() == 1 && b.members.size() == 1 &&
        a.members.front().session == request.session && b.members.front().session == request.session &&
        request.session->head.get() == a.members.front().base && request.session->pending == &a &&
        a.execution.sources && b.execution.sources && b.execution.has_completion() &&
        b.capture_reused > 0 && b.capture_recorded == 0 &&
        request.ledger->committed_count == observed &&
        observed < request.ledger->records.size() && !request.ledger->records[observed].committed,
        "natural K1 B was not actually submitted before its original A host publication");
    ++result.pending_pairs;
  };
  const auto observed_batch = [&](const q::GenerationBatchFacts &facts) {
    require(facts.members.size() == 1 && facts.members.front().request == &request &&
        facts.completion == q::TextCompletion::kEmitNext && facts.active_rows <= facts.physical_rows,
        "natural K1 batch lost its original request, completion or physical work");
    const auto &call = facts.members.front();
    require(retired_submissions < submitted_drafts.size(),
        "natural K1 retirement had no original accepted submission");
    const auto submitted_draft = submitted_drafts[retired_submissions++];
    if (call.disposition == q::EvaluationDisposition::kCancelled) {
      require(submitted_draft.successor && call.generation && submitted_draft.rows == 2,
          "natural K1 discarded batch was not an actual two-row successor Append");
      result.discarded_draft_rows = q::cache::add(result.discarded_draft_rows, submitted_draft.rows);
      result.batches.push_back({observed, call.quota, reserved, additional, 0, 0, 0, call.generation, true});
      return;
    }
    require(call.disposition == q::EvaluationDisposition::kPublished &&
        request.ledger->committed_count == observed + 1,
        "natural K1 observer duplicated or omitted an authoritative ledger commit");
    const auto &record = request.ledger->records[observed];
    require(record.committed && !record.input_deferred && record.requests == 1 && !record.member_index &&
        record.quota == call.quota && record.after == q::generation_boundary(*request.session->head) &&
        record.before.sampling == i::SamplingCursor{} && record.after.sampling == i::SamplingCursor{} &&
        record.emitted_count > 0 && record.emitted_count <= 2,
        "natural K1 commit changed its actual quota, member, boundary or greedy RNG");
    if (call.generation)
      require(record.quota > 0 && record.quota <= reserved + additional &&
          record.emitted_count <= record.quota && record.quota <= request.progress.remaining_outputs(),
          "natural K1 commit exceeded its original RequestLoop output grant");
    emitted += record.emitted_count;
    require(emitted <= outputs, "natural K1 journey exceeded its user output quota");
    result.batches.push_back({observed, record.quota, reserved, additional, record.emitted_count,
        record.accepted, record.draft_rows, record.generation, false});
    if (record.generation && record.draft_limit) {
      result.accepted_drafts += record.accepted_draft;
      result.rejected_drafts += !record.accepted_draft;
    }
    ++observed;
    if (record.stopped || emitted == outputs) {
      require(!result.final_state, "natural K1 journey published more than one terminal state");
      result.final_state.emplace(s::image(model, *request.session->head));
      result.stopped = record.stopped;
    }
  };
  const auto execute_ready = [&](i::RequestWave wave) {
    require(wave.keys.size() == 1 && wave.keys.front() == request.request_key,
        "natural K1 owner changed its original RequestKey");
    std::array calls{q::prepare_generation_call(model, request, outputs, nullptr, requests,
        loop.credit(request.request_key, request.progress.generated.size()))};
    q::execute_generation_calls(model, calls, false, nullptr, requests, submitted,
        &continuation, observed_batch, true, wave.credits);
    q::publish_generation_prompt(model, calls.front(), nullptr);
  };
  const auto execute_wave = [&](i::RequestWave wave, std::span<std::size_t> remaining) {
    require(wave.keys.size() == 1 && wave.credits.size() == 1,
        "natural K1 wave lost its original ordered output grant");
    reserved = wave.credits.front().reserved; additional = wave.credits.front().additional;
    if (!continuation.pending_calls.empty()) {
      require(reserved == 2 && additional <= 2 && !wave.credits.front().cancelled,
          "natural pending K1 B did not retain exactly its original two output reservations");
      ++result.reserved_two_grants;
    }
    return q::advance_generation_wave(model, wave, remaining, false, continuation,
        execute_ready, submitted, observed_batch);
  };
  q::GenerationRequestAdapter adapter{model, static_cast<q::cache::PrefixCache *>(nullptr), std::span(requests),
      false, static_cast<std::vector<std::size_t> *>(nullptr), execute_wave, {}, &continuation};
  request.request_key = loop.enqueue(0, &delivery);
  for (std::size_t pumps = 0; loop.matches(request.request_key); ++pumps) {
    require(pumps < outputs * 4, "natural K1 RequestLoop did not finish its finite request");
    const auto pumped = loop.pump_once(adapter);
    require(!pumped.cancelled && (!pumped.waiting_output ||
        delivery.acknowledged() < request.progress.generated.size()),
        "natural K1 normal delivery cancelled or stalled without a real output awaiting acknowledgement");
    if (!continuation.pending_calls.empty()) {
      require(model.active_step && model.active_step->submitted && !model.successor_step &&
          model.active_step->execution.sources && model.active_step->execution.has_completion(),
          "natural K1 deferred wave lost the original promoted B submission");
      ++result.deferred_waves;
    }
    delivery.acknowledge_prefix(request.progress.generated.size(), request.progress.generated.size());
  }
  require(request.progress.request_done() && result.final_state && !request.session &&
      observed == request.ledger->committed_count && observed == request.progress.observed_commits() &&
      emitted == request.progress.generated.size() && delivery.pending(request.progress.generated).empty() &&
      retired_submissions == submitted_drafts.size(),
      "natural K1 request finished without its complete observed outputs, ledger or terminal state");
  result.actual_draft_rows = model.submitted_draft_rows - submitted_draft_begin;
  require(request.actual_draft_rows() == result.actual_draft_rows,
      "natural K1 request retirement lost or duplicated its original submitted draft rows");
  result.ids = request.progress.generated;
  result.records.assign(request.ledger->records.begin(), request.ledger->records.begin() + observed);
  result.successors = continuation.successor_submissions; result.publications = continuation.successor_publications;
  result.discards = continuation.successor_discards; result.refusals = continuation.successor_refusals;
  continuation.reset();
  require(continuation.pending_calls.empty() && !model.active_step && !model.successor_step &&
      !model.carrier_borrowed && !model.snapshot_versions.live &&
      model.page_inventory.free_ids.size() == model.cache_plan.max_pages &&
      model.draft_inventory.free_ids.size() == model.draft_inventory.owners.size() &&
      std::ranges::all_of(sources, [](const auto &source) { return source.expired(); }),
      "natural K1 completion retained an original Source, candidate, Store or append page owner");
  cleanup.dismiss(); return result;
}

inline p::StepReport compare_state(q::Model &model, const s::StateImage &expected, const s::StateImage &actual,
    std::size_t rows) {
  p::StepReport report{0, actual.length, rows, rows}; report.prefill = false;
  report.exact_next = expected.pending; report.bucket_next = actual.pending;
  report.boundary_equal = std::tie(expected.length, expected.frontier, expected.lag, expected.pending,
      expected.catchup, expected.sampling) == std::tie(actual.length, actual.frontier, actual.lag,
      actual.pending, actual.catchup, actual.sampling);
  require(expected.target == actual.target && expected.payloads.size() == actual.payloads.size(),
      "natural K1 Source changed final target KV or the complete payload inventory");
  report.payloads_equal = expected.payloads == actual.payloads;
  std::vector<std::size_t> recurrent;
  for (std::size_t layer = 0; layer < model.geometry.layers.size(); ++layer)
    if (model.geometry.is_gdn(layer)) recurrent.push_back(layer);
  for (std::size_t part = 0; part < expected.payloads.size(); ++part) {
    if (expected.payloads[part] == actual.payloads[part]) continue;
    require(part < recurrent.size() * 2 && part % 2 == 1,
        "natural K1 Source changed a non-GDN final state payload");
    q::cache::HostPayloadDesc layout{}; layout.role = q::cache::HostPayloadRole::kGdnState;
    layout.layer = static_cast<std::uint32_t>(recurrent[part / 2]); layout.bytes = expected.payloads[part].size();
    auto difference = p::compare_payload(model, layout, expected.payloads[part], actual.payloads[part]);
    require(!difference.nonfinite_differences && difference.maximum_absolute <= p::gdn_state_absolute_budget,
        "natural K1 Source final GDN state exceeds the original absolute budget");
    report.differences.push_back(std::move(difference));
  }
  report.numerically_equivalent = report.boundary_equal;
  require(report.numerically_equivalent, "natural K1 Source changed the complete final logical boundary");
  return report;
}

inline int run(const std::filesystem::path &input, q::BackendKind kind) {
  require(kind == q::BackendKind::kCpu || kind == q::BackendKind::kCuda,
      "natural K1 Source consumer currently checks CPU/CUDA only");
  auto source = q::prepare_weight_source(input / "weights.tsv", true); const auto geometry = source.roles.geometry;
  const auto prompt = q::read_ids(input / "prompt.ids", geometry.vocabulary, geometry.context_limit);
  const auto capacity = (std::max)(std::size_t{256}, q::cache::add(prompt.size(), outputs + 8));
  require(!prompt.empty() && geometry.context_limit > 0 && capacity <= static_cast<std::size_t>(geometry.context_limit) && geometry.recurrent_layers() &&
      geometry.attention_layers() && geometry.mtp_layers == 1,
      "natural K1 Source requires the original cached GA/MTP prompt within its real context limit");
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, capacity, 1,
      (std::max)(prompt.size(), std::size_t{2}), q::recurrent_bytes(q::state_schema_for(geometry, capacity), geometry),
      1, 2, true, true, 2);
  require(plan.state_slots == 5 && plan.hidden_slots == 6 && plan.resident_sessions == 1,
      "natural K1 consumer must use the original exact N1/R1/D2 state capacity");
  q::Model model(std::move(source), geometry, capacity, plan, {}, false,
      kind == q::BackendKind::kCuda ? q::ReplayPolicy::kPreferCapture : q::ReplayPolicy::kCommands,
      SIZE_MAX, {}, 0, {.primary = kind});
  model.configure_draft_policy({.max_drafts = 1});
  const std::array prefill{prompt.size()}, one{std::size_t{1}}, two{std::size_t{2}};
  model.prepare_family(prefill, false, 1, q::GraphRole::kMtpPrefill);
  model.plan_replay_uses(prompt.size(), 1, 2, q::GraphRole::kMtpPrefill);
  for (const auto role : {q::GraphRole::kMtpSingle, q::GraphRole::kMtpCatchupSingle}) {
    model.prepare_family(one, false, 1, role); model.plan_replay_uses(1, 1, outputs * 2, role);
  }
  for (const auto role : {q::GraphRole::kMtpRound, q::GraphRole::kMtpCatchupRound}) {
    model.prepare_family(two, false, 1, role); model.plan_replay_uses(2, 1, outputs * 2, role);
  }
  model.freeze_shape_plan(); model.prepare_execution(prefill, 1, q::GraphRole::kMtpPrefill);
  for (const auto role : {q::GraphRole::kMtpSingle, q::GraphRole::kMtpCatchupSingle}) model.prepare_execution(one, 1, role);
  for (const auto role : {q::GraphRole::kMtpRound, q::GraphRole::kMtpCatchupRound}) model.prepare_execution(two, 1, role);
  if (kind == q::BackendKind::kCuda) model.prepare_relay_execution(1, 0, q::GraphRole::kMtpCatchupRound);
  const auto serial = journey(model, prompt, false);
  const auto candidate = journey(model, prompt, true);
  std::size_t serial_ledger_draft_rows = 0;
  for (const auto &record : serial.records)
    serial_ledger_draft_rows = q::cache::add(serial_ledger_draft_rows, record.draft_rows);
  require(!serial.discarded_draft_rows && serial.actual_draft_rows == serial_ledger_draft_rows,
      "natural K1 serial submitted draft rows differ from the original complete ledger");
  require(candidate.actual_draft_rows == q::cache::add(serial.actual_draft_rows, candidate.discarded_draft_rows),
      "natural K1 Source omitted discarded successor work or counted an original submission twice");
  require(serial.ids == candidate.ids && serial.stopped == candidate.stopped &&
      serial.records.size() == candidate.records.size(), "natural K1 Source changed generated IDs, stop or ledger length");
  for (std::size_t ordinal = 0; ordinal < serial.records.size(); ++ordinal) {
    const auto &a = serial.records[ordinal]; const auto &b = candidate.records[ordinal];
    const bool equal = a.same_result(b) && a.committed == b.committed && a.user_output == b.user_output &&
        a.imported == b.imported && a.turn == b.turn && std::ranges::equal(a.ids(), b.ids()) &&
        std::ranges::equal(a.committed_ids(), b.committed_ids());
    if (!equal) {
      std::cerr << "natural_relay_ledger_mismatch ordinal=" << ordinal << " serial_quota=" << a.quota
          << " auto_quota=" << b.quota << " serial_role=" << static_cast<int>(a.role)
          << " auto_role=" << static_cast<int>(b.role) << " serial_accepted=" << a.accepted
          << " auto_accepted=" << b.accepted << " serial_length=" << a.after.length
          << " auto_length=" << b.after.length << " serial_frontier=" << a.after.frontier
          << " auto_frontier=" << b.after.frontier << " serial_next=" << a.after.next_id
          << " auto_next=" << b.after.next_id << " serial_emitted=";
      for (auto id : std::span(a.emitted).first(a.emitted_count)) std::cerr << id << ',';
      std::cerr << " auto_emitted=";
      for (auto id : std::span(b.emitted).first(b.emitted_count)) std::cerr << id << ',';
      std::cerr << '\n';
    }
    require(equal, "natural K1 Source changed a complete original non-time ledger result");
  }
  const auto state = compare_state(model, *serial.final_state, *candidate.final_state,
      candidate.records.back().verified_rows);
  require(!serial.successors && !serial.pending_pairs && !serial.deferred_waves,
      "natural K1 serial control unexpectedly submitted a successor");
  if (kind == q::BackendKind::kCuda)
    require(candidate.successors > 0 && candidate.publications > 0 && candidate.pending_pairs > 0 &&
        candidate.deferred_waves > 0 && candidate.reserved_two_grants > 0,
        "natural K1 cached prompt did not actually submit and publish an original pending successor");
  std::cout << "{\"status\":\"PASS\",\"case\":\"mtp-natural-relay\",\"backend\":\"" << h::backend_name(kind)
      << "\",\"models\":1,\"journeys\":2,\"prompt_tokens\":" << prompt.size() << ",\"capacity\":" << capacity
      << ",\"resident_capacity\":" << plan.resident_sessions << ",\"state_slots\":" << plan.state_slots
      << ",\"hidden_slots\":" << plan.hidden_slots << ",\"output_limit\":" << outputs << ",\"outputs\":" << candidate.ids.size()
      << ",\"serial_submissions\":" << serial.submits << ",\"auto_submissions\":" << candidate.submits
      << ",\"source_submissions\":" << candidate.source_submits << ",\"successor_submissions\":" << candidate.successors
      << ",\"successor_publications\":" << candidate.publications << ",\"successor_discards\":" << candidate.discards
      << ",\"successor_refusals\":" << candidate.refusals << ",\"pending_pairs_before_A_publish\":" << candidate.pending_pairs
      << ",\"deferred_waves\":" << candidate.deferred_waves << ",\"reserved_two_grants\":" << candidate.reserved_two_grants
      << ",\"accepted_drafts\":" << candidate.accepted_drafts << ",\"rejected_drafts\":" << candidate.rejected_drafts
      << ",\"ledger_records\":" << candidate.records.size()
      << ",\"serial_actual_draft_rows\":" << serial.actual_draft_rows
      << ",\"auto_actual_draft_rows\":" << candidate.actual_draft_rows
      << ",\"discarded_draft_rows\":" << candidate.discarded_draft_rows
      << ",\"natural_RequestLoop\":true,\"controlled_acceptance\":false,\"original_ledger_equal\":true,"
         "\"generated_ids_equal\":true,\"greedy_rng_unchanged\":true,\"all_sources_expired\":true,"
         "\"simultaneous_native_inflight_claim\":false,\"quality_claim\":false,\"performance_claim\":false,"
         "\"final_versions\":0,\"state_checks\":[";
  p::write_step(state); std::cout << "],\"ids\":[";
  for (std::size_t row = 0; row < candidate.ids.size(); ++row) std::cout << (row ? "," : "") << candidate.ids[row];
  std::cout << "],\"batches\":[";
  for (std::size_t row = 0; row < candidate.batches.size(); ++row) {
    const auto &b = candidate.batches[row];
    std::cout << (row ? "," : "") << "{\"ordinal\":" << b.ordinal << ",\"quota\":" << b.quota
        << ",\"reserved\":" << b.reserved << ",\"additional\":" << b.additional << ",\"emitted\":" << b.emitted
        << ",\"accepted\":" << b.accepted << ",\"draft_rows\":" << b.draft_rows
        << ",\"generation\":" << (b.generation ? "true" : "false") << ",\"discarded\":" << (b.discarded ? "true" : "false") << '}';
  }
  std::cout << "]}\n"; return 0;
}
} // namespace qwen_mtp_natural_relay_case
