#include "generation_state.h"
#include "request_installation.h"
#include "generation_diagnostics.h"
#include <atomic>
#include "model_core.h"
#include "prepared_executor.h"
#include "device_mtp_planning.h"
#include "schedule.h"
#include "session_schedule.h"
#include "prefix_cache.h"
#include "batch_schedule.h"
#include "generation_schedule.h"
#include "generation_ledger.h"
#include "generation_publication.h"
#include "draft_asset_policy.h"
#include "mtp_default_policy.h"
#include "../../inference/session_cache.h"
#include "../../inference/request_progress.h"
#include "../../inference/request_admission.h"
#include "../../inference/request_loop.h"
#include "../../inference/request_owner.h"
#include "../../inference/token_delivery.h"
#include "../../inference/model_executor.h"
#include "../../inference/request_wave_execution.h"
#include "../../inference/step_execution.h"
#include "../../inference/continuation.h"
#include "decode_route_policy.h"
#include "../../inference/prefill_plan.h"
#include "../../inference/version_admission.h"
#include "../../inference/batch_dispatch.h"
#include "../../inference/request_batch_execution.h"
#include "../../inference/mtp_policy.h"
#include "../../reporting/execution_report.h"

namespace qwen35 {

Model &generation_model(Model &fallback, const GenerationRequest &request) noexcept {
  return request.prepared_model ? *request.prepared_model : fallback;
}
cache::PrefixCache *generation_prefix(cache::PrefixCache *fallback, const GenerationRequest &request) noexcept {
  return request.prepared_model ? request.prepared_prefix : fallback;
}

std::size_t generation_sampling_index(const Model &model, const GenerationLedger &ledger) {
  if (ledger.sampling && (ledger.sampling_owner.expired() || ledger.sampling_owner.owner_before(model.resources_) || model.resources_.owner_before(ledger.sampling_owner)))
    throw std::runtime_error("generation ledger belongs to another or destroyed model");
  return model.sampling_index(ledger.sampling);
}

// C0 includes only authoritative committed IDs and this first conditional's
// actual input. It is never an alternative history, Session or RNG owner.
// Raw EvaluationRequest callers can instead supply their own exact C0 lease.
LogitsInputs history_logits_inputs(Model &model, GenerationLedger &ledger,
    std::span<const std::int32_t> first_condition_ids, LogitsInputs inputs = {}, std::size_t sampling_index = 0) {
  const auto &prepared = model.sampling_configuration(sampling_index);
  ledger.validate_sampling(prepared, model.resources_);
  if (prepared.configuration.modifiers.history.empty()) return inputs;
  if (inputs.counts.lease.valid() || inputs.counts.description.valid())
    throw std::invalid_argument("derived history C0 cannot replace an externally supplied counts input");
  auto &cache = ledger.history_counts(static_cast<std::size_t>(model.vocabulary), model.execution_budget,
      take(model.execution_budget.metadata_allocator()));
  const auto conditional = cache.condition(first_condition_ids);
  const auto bytes = std::as_bytes(cache.values());
  auto writer = take(model.resources_->history_counts_pool.allocate(bytes.size()));
  check(model.execution.write(take(writer.view()), bytes));
  inputs.counts = {dense(Type::kInt32, {1, model.vocabulary}), take(writer.publish())};
  return inputs;
}

// Entry IO runs synchronously inside Model's original completed-output scope.
// Neither this view nor its borrowed storage may escape or reenter preparation.
void observe_completed_evaluation(Model &model, const EvaluationOutput &output,
    const EvaluationIO &io, const CompletedEvaluationView &completed) {
  auto &observation = completed.observation;
  if (!output.observation) { observation.name = io.name; hyle::engine::reporting::write_step_observation(std::cerr, observation); }
  if ((!io.trace || io.trace->empty()) && !output.capture_probes) return;
  const auto activation_suffix = model.precision.activation_type() == Type::kFloat32 ? ".f32" :
      model.precision.is_f16() ? ".f16" : ".bf16";
  const auto dump_bytes = [&](mem::ConstBufferView view, const std::filesystem::path &path) {
    std::vector<std::byte> bytes(view.size()); check(model.execution.read(view, bytes));
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!stream) throw std::runtime_error("cannot write Qwen trace: " + path.string());
  };
  if (io.trace && !io.trace->empty()) {
    const auto directory = *io.trace / io.name; std::filesystem::create_directories(directory);
    const auto dump_row = [&](c::ConstTensorView tensor, const std::string &file) {
      if (completed.packed_input_rows) {
        const auto &description = tensor.description();
        if (!description.is_dense() || description.rank() != 2)
          throw std::runtime_error("packed verification trace requires dense row outputs");
        const auto row_bytes = take(description.required_bytes()) / static_cast<std::size_t>(description.dimensions()[0]);
        dump_bytes(take(tensor.storage().subview(cache::multiply(completed.packed_input_begin, row_bytes),
            cache::multiply(completed.packed_input_rows, row_bytes))), directory / file);
      } else {
        const auto bytes = take(tensor.description().required_bytes()) / completed.requests;
        dump_bytes(take(tensor.storage().subview(completed.member_index * bytes, bytes)), directory / file);
      }
    };
    if (completed.logits.valid()) dump_row(completed.logits, "logits.f32");
    dump_row(completed.hidden, std::string("hidden") + activation_suffix);
    if (io.trace_states) {
      for (std::size_t i = 0; i < model.state_schema.size(); ++i) if (model.geometry.is_gdn(i)) {
        const auto prefix = "layer" + std::to_string(i);
        dump_bytes(take(completed.head.state[i].history.view()), directory / (prefix + ".conv" + activation_suffix));
        dump_bytes(take(completed.head.state[i].recurrent.view()), directory / (prefix + ".gdn.f32"));
      }
      std::ofstream layout_trace(directory / "kv-layout.json");
      layout_trace << "{\"visible_tokens\":" << completed.head.length << ",\"page_rows\":" << cache::Plan::page_rows << ",\"pages\":[";
      for (std::size_t i = 0; i < completed.head.paged.pages.size(); ++i) layout_trace << (i ? "," : "") << completed.head.paged.pages[i].page();
      layout_trace << "]}\n"; if (!layout_trace) throw std::runtime_error("cannot write KV layout trace");
    }
  }
  if (output.capture_probes) {
    const auto directory = io.probe_subdirectory.empty() ? model.layer0_probe / io.name
        : model.layer0_probe / std::filesystem::path(io.probe_subdirectory) / io.name;
    std::filesystem::create_directories(directory);
    for (std::size_t i = 0; i < completed.probes.size(); ++i) {
      const auto view = completed.probes[i]; const auto bytes = take(view.description().required_bytes());
      if ((completed.requests != 1 && !view.description().is_dense()) || bytes % completed.requests)
        throw std::runtime_error("published batch diagnostic is not request-major dense");
      dump_bytes(take(view.storage().subview(completed.member_index * (bytes / completed.requests), bytes / completed.requests)),
          directory / (std::string(layer0_probe_names[i]) + activation_suffix));
    }
  }
}

EvaluationOutput evaluation_output(Model &model, const std::shared_ptr<Session> &session,
    const EvaluationIO &io, double *complete_seconds = nullptr, Clock::time_point *host_endpoint = nullptr,
    inference::StepObservation *observation = nullptr, inference::PreparationObservation *preparation = nullptr) {
  return {.session = session, .capture_tensors = io.trace && !io.trace->empty(),
      .capture_probes = !model.layer0_probe.empty(), .complete_seconds = complete_seconds,
      .host_endpoint = host_endpoint, .observation = observation, .preparation = preparation};
}

MtpRoundResult finish_mtp_round(Model &model, Session &session, std::int32_t expected_pending,
    std::size_t output_quota, std::span<const std::int32_t> eos = {},
    const EvaluationOutput *output = nullptr, const EvaluationIO &io = {.name = "mtp"},
    GenerationPublicationInput request_publication = {}) {
  const std::array selections{MtpGenerationSelection{session.shared_from_this(), expected_pending, output_quota}};
  const std::array outputs{output ? *output : evaluation_output(model, session.shared_from_this(), io)};
  std::array<MtpRoundResult, 1> results;
  GenerationPublicationBatch publication({request_publication});
  model.finish_mtp_rounds(selections, eos, outputs, results, [&](std::size_t j, const CompletedEvaluationView &view) {
    observe_completed_evaluation(model, outputs[j], io, view);
  }, publication);
  return std::move(results[0]);
}

std::int32_t finish_evaluation(Model &model, Session &session, const std::string &name,
    const std::filesystem::path &trace = {}, bool trace_states = true,
    double *complete_seconds = nullptr, Clock::time_point *host_endpoint = nullptr,
    inference::StepObservation *observation = nullptr, inference::PreparationObservation *preparation = nullptr,
    std::string_view probe_subdirectory = {},
    GenerationLedger *generation_ledger = nullptr, const GenerationCommitRecord *expected_episode = nullptr) {
  const EvaluationIO io{name, &trace, trace_states, probe_subdirectory};
  const std::array outputs{evaluation_output(model, session.shared_from_this(), io,
      complete_seconds, host_endpoint, observation, preparation)};
  GenerationPublicationBatch publication({{generation_ledger, expected_episode}});
  std::int32_t next = -1;
  model.finish_evaluations(outputs, std::span(&next, 1), [&](std::size_t j, const CompletedEvaluationView &view) {
    observe_completed_evaluation(model, outputs[j], io, view);
  }, publication);
  return next;
}

std::int32_t evaluate(Model &model, Session &session, std::span<const std::int32_t> ids, const std::string &name,
    const std::filesystem::path &trace = {}, bool trace_states = true,
    double *complete_seconds = nullptr, Clock::time_point *host_endpoint = nullptr,
    inference::StepObservation *observation = nullptr, inference::PreparationObservation *preparation = nullptr,
    std::string_view probe_subdirectory = {}, const inference::RequestStopPolicy *stop_policy = nullptr,
    const LogitsInputs &logits = {}) {
  model.prepare_evaluation(session, ids, preparation, GraphRole::kText, TextCompletion::kEmitNext, stop_policy, logits);
  try { model.submit_evaluation(session); return finish_evaluation(model, session, name, trace, trace_states,
      complete_seconds, host_endpoint, observation, preparation, probe_subdirectory); }
  catch (inference::EvaluationFailure &failure) {
    failure.observe_retirement(model.retire_failed_publication(
        failure.retirement_observed() && !failure.retirement().retired));
    throw;
  } catch (...) { (void)model.retire_failed_publication(); throw; }
}

void observe_generation_preparation(GenerationRequest &request, const CompletedEvaluationView &view, bool generation) {
  if (generation || request.turn_observations.empty()) return;
  auto &turn = request.turn_observations.at(request.progress.turn_index);
  const auto &observed = view.observation;
  turn.prefill_shape_admission_seconds += observed.shape_admission_seconds;
  turn.prefill_phase_prepare_seconds += observed.phase_prepare_seconds;
  turn.prefill_slot_initialize_seconds += observed.slot_initialize_seconds;
  turn.prefill_phase_creations += observed.phase_creations;
  turn.prefill_slot_initializations += observed.slot_initializations;
}

// Cold episode count; imported source chunks have their own explicit bound.
std::size_t generation_record_capacity(std::span<const inference::GenerationTurn> turns,
    std::size_t max_prefill_rows = SIZE_MAX, std::size_t imported_records = 0) {
  const inference::PrefillPolicy prefill(max_prefill_rows);
  auto count = imported_records;
  for (std::size_t turn = 0; turn < turns.size(); ++turn) {
    const auto rows = cache::add(turns[turn].prompt.size(), turn ? 1u : 0u);
    if (!rows || !turns[turn].max_outputs) throw std::runtime_error("generation requires nonempty turn input and positive quota");
    const auto chunks = prefill.parts(rows);
    count = cache::add(count, cache::add(chunks, turns[turn].max_outputs - 1));
  }
  return count;
}

GenerationCommitRecord &prepare_generation_record(GenerationLedger &ledger, const Session &session,
    GraphRole role, std::span<const std::int32_t> ids, bool generation,
    std::size_t quota, std::size_t turn, std::shared_ptr<const void> owner = {}, std::size_t maximum = 0,
    TextCompletion completion = TextCompletion::kEmitNext, std::size_t input_begin = 0, bool prefill_complete = true) {
  if (!supports_commit_completion(role, completion, generation, prefill_complete))
    throw std::runtime_error("state-only ledger entry requires an incomplete teacher-forced prefill");
  if (ids.empty() || !session.head || (generation && ids.size() != 1))
    throw std::runtime_error("MTP replay record requires its actual input and complete base");
  auto &record = ledger.next();
  ledger.bind_sampling(session.sampling_configuration(), session.sampling_owner());
  auto emitted = std::move(record.emitted), consumed = std::move(record.accepted_inputs);
  auto verified = std::move(record.verification_inputs);
  record = {}; record.emitted = std::move(emitted); record.accepted_inputs = std::move(consumed);
  record.verification_inputs = std::move(verified); record.verification_inputs.clear();
  write_generation_tokens(record.emitted, {-1, -1}); write_generation_tokens(record.accepted_inputs, {-1, -1});
  record.role = role; record.generation = generation; record.quota = quota; record.turn = turn;
  record.completion = completion; record.input_begin = input_begin; record.prefill_complete = prefill_complete;
  record.user_output = completion == TextCompletion::kEmitNext;
  record.draft_limit = role == GraphRole::kVerify ? maximum : drafts_token(role) ? (maximum ? maximum : 1) : 0;
  record.before = generation_boundary(*session.head);
  if (generation) record.pending_input = ids.front();
  else { record.input_owner = std::move(owner); record.prompt_ids = ids; }
  return record;
}

void preflight_generation_prompt_ledger(const GenerationLedger &ledger, const cache::ReplayPrefix &prefix) {
  if (ledger.committed_count || !prefix.history || !prefix.chunks ||
      prefix.chunks > prefix.history->chunks.size() || prefix.chunks > ledger.records.size())
    throw std::runtime_error("MTP initial prompt import requires an empty bounded ledger");
  std::size_t covered = 0;
  for (std::size_t i = 0; i < prefix.chunks; ++i) {
    const auto &part = prefix.history->chunks[i];
    if (ledger.records[i].committed || ledger.records[i].emitted.capacity() < 2 ||
        !part.input || !part.count || part.first != covered || part.first > part.input->ids.size() ||
        part.count > part.input->ids.size() - part.first || covered > prefix.length ||
        part.count > prefix.length - covered || (part.state_only ? part.next_id != -1 : part.next_id < 0))
      throw std::runtime_error("prompt import requires complete original ranges and uncommitted reserved records");
    covered += part.count;
  }
  if (covered != prefix.length)
    throw std::runtime_error("prompt import history does not cover its complete boundary");
}

void import_generation_prompt_ledger(GenerationLedger &ledger, const cache::ReplayPrefix &prefix,
                              bool full_hit, bool draft_cache) {
  preflight_generation_prompt_ledger(ledger, prefix);
  std::int32_t previous = -1;
  for (std::size_t i = 0; i < prefix.chunks; ++i) {
    const auto &part = prefix.history->chunks[i];
    auto &record = ledger.records[i];
    record.role = !draft_cache ? GraphRole::kText
        : part.count == 1 ? GraphRole::kMtpSingle : GraphRole::kMtpPrefill;
    record.input_owner = part.input;
    record.prompt_ids = std::span<const std::int32_t>(part.input->ids).subspan(part.first, part.count);
    record.before = {part.first, draft_cache ? part.first : 0, 0, previous, -1};
    record.after = {part.first + part.count, draft_cache ? part.first + part.count : 0, 0, part.next_id, -1};
    record.accepted = part.count; record.draft_rows = draft_cache ? part.count : 0;
    record.completion = part.state_only ? TextCompletion::kStateOnly : TextCompletion::kEmitNext;
    record.prefill_complete = !part.state_only;
    record.input_begin = part.first;
    record.target_first = part.next_id; write_generation_tokens(record.emitted, {part.next_id, -1}); record.emitted_count = part.state_only ? 0 : 1;
    record.stopped = !part.state_only && inference::request_token_stopped(
        part.next_id, model_eos_tokens, &ledger.stop_policy());
    record.user_output = full_hit && i + 1 == prefix.chunks;
    record.imported = true;
    record.requests = 0; // Role-only source history does not claim original batch geometry.
    previous = part.next_id;
  }
  // All ranges/capacities were checked before population. shared_ptr/span and
  // scalar writes, including the two inline emitted IDs, allocate nothing.
  for (std::size_t i = 0; i < prefix.chunks; ++i) ledger.records[i].committed = true;
  ledger.committed_count = prefix.chunks;
}

void apply_generation_observation(GenerationRequest &request, const GenerationCommitRecord &record,
    const inference::CommitObservation &observation) noexcept {
  const auto output_end = request.progress.generated.size();
  if (record.user_output) {
    if (record.generation) {
      MtpRoundResult result;
      result.pending = record.pending_input; result.draft = record.draft;
      result.target_first = record.target_first; result.target_second = record.target_second;
      result.next_id = record.after.next_id; result.agreed = record.agreed;
      result.accepted_draft = record.accepted_draft; result.stopped = record.stopped;
      result.processed_target = record.accepted; result.processed_draft = record.draft_rows;
      result.proposal_count = record.verified_rows ? record.verified_rows - 1 : 0;
      result.verified_rows = record.verified_rows; result.consumed_drafts = record.consumed_drafts;
      result.auxiliary_graph_calls = record.auxiliary_graph_calls;
      result.emitted = std::move(request.round_emitted[request.rounds.size()]);
      result.emitted.assign(record.emitted.begin(), record.emitted.begin() + record.emitted_count);
      request.rounds.push_back(std::move(result));
      request.verified_rows += record.verified_rows;
      request.draft_rows += record.draft_rows; request.draft_rounds += record.draft_limit ? 1 : 0;
      request.agreements += record.agreed ? 1 : 0; request.accepts += record.consumed_drafts;
      request.auxiliary_graph_calls += record.auxiliary_graph_calls;
      request.decode_step_sum_seconds += record.step_seconds;
    }
  }
  if (!record.generation && !record.imported) {
    ++request.prefill_graph_calls; request.prefill_graph_tokens += record.ids().size();
    request.prefill_draft_rows += record.draft_rows;
    request.prefill_step_seconds += record.step_seconds;
  }
  request.visible_tokens = record.after.length; request.draft_frontier = record.after.frontier;
  request.draft_lag = record.after.lag; request.fresh_next_id = record.after.next_id;
  if (!record.imported) request.endpoint = record.endpoint;
  if (observation.prefill_complete) {
    request.prefill_endpoint = record.endpoint;
    if (!request.turn_observations.empty()) {
      auto &turn = request.turn_observations[request.progress.turn_index];
      turn.start = request.prefill_start; turn.prefill_endpoint = record.endpoint;
    }
  }
  if (!request.turn_observations.empty()) {
    auto &turn = request.turn_observations[request.progress.turn_index];
    turn.endpoint = request.endpoint; turn.output_end = output_end;
    turn.context_after = request.visible_tokens;
  }
}

// Consume committed host facts even when a subsequent diagnostic write threw.
// The authoritative ledger was installed by the original head-swap transaction.
void observe_generation_commits(GenerationRequest &request) {
  if (!request.ledger) throw std::runtime_error("generation request requires its cold-prepared commit ledger");
  while (request.progress.observed_commits() < request.ledger->committed_count) {
    const auto ordinal = request.progress.observed_commits();
    (void)inference::observe_record_commit(std::span<const GenerationCommitRecord>(request.ledger->records),
        request.ledger->committed_count, ordinal, request.progress,
        [](const GenerationCommitRecord &record) {
          if (record.emitted_count > record.emitted.size())
            throw std::runtime_error("committed output count exceeds its original record storage");
          return inference::CommitObservation{
              std::span(record.emitted).first(record.emitted_count), record.user_output,
              !record.generation && !record.imported && record.prefill_complete,
              record.completion == TextCompletion::kEmitNext && record.stopped,
              !record.generation && !record.imported
                  ? std::optional<inference::RequestInputCommit>{{record.turn, record.input_begin, record.ids()}}
                  : std::nullopt};
        },
        [&](const GenerationCommitRecord &record, const inference::CommitObservation &) {
          if (record.user_output && record.generation &&
        (request.rounds.size() == request.rounds.capacity() || request.rounds.size() >= request.round_emitted.size() ||
         record.emitted_count > request.round_emitted[request.rounds.size()].capacity()))
            throw std::runtime_error("generation observer requires its original cold-reserved round storage");
        },
        [&](const GenerationCommitRecord &record, const inference::CommitObservation &observation) noexcept {
          apply_generation_observation(request, record, observation);
        });
  }
}

// Only the serialized request owners and this directory retain committed
// Snapshots here. Hit/local Snapshot references are dropped before this gate.
inference::VersionHolder<Session, cache::State> version_holder(const Session *session) noexcept {
  return {session, session && session->head ? session->head->state_identity() : nullptr};
}

void make_generation_room(Model &model, cache::PrefixCache *prefix, Session *owner,
                    std::span<const GenerationRequest> requests) {
  const auto limit = model.cache_plan.resident_sessions;
  const std::array selected{version_holder(owner)};
  const auto residents = requests | std::views::transform([&](const auto &request) {
    return version_holder(request.prepared_model && request.prepared_model != &model ? nullptr : request.session.get());
  });
  const auto observe = [&] {
    const auto live = std::size_t(model.snapshot_versions.live);
    if (live > limit || model.active_step) throw std::runtime_error("generation admission requires retired committed heads within N");
    return inference::VersionRoomObservation{{live, limit}, prefix ? prefix->size() : 0,
        inference::version_admission({live, limit}, selected, residents,
            [&](const auto *head) { return prefix && prefix->retains(head); }).fits};
  };
  if (inference::make_version_room(observe, [&] { return prefix && prefix->evict_one(); }) !=
      inference::VersionRoomResult::kReady)
    throw std::runtime_error("generation checkpoints retain all committed versions; no unfinished session may be evicted");
}

std::shared_ptr<Session> restore_mtp_session(Model &model, const GenerationLedger &ledger,
    const std::filesystem::path &trace = {}, bool trace_states = true,
    GenerationRestoreStats *stats = nullptr, cache::PrefixCache *prefix = nullptr,
    std::span<const GenerationRequest> requests = {}) {
  const auto sampling_index = generation_sampling_index(model, ledger);
  const auto started = Clock::now();
  const auto submitted_begin = model.submitted_draft_rows;
  hyle::util::ScopeGuard record_submitted{[&]() noexcept {
    if (stats) stats->submitted_draft_rows += model.submitted_draft_rows - submitted_begin;
  }};
  ledger.stop_policy().validate(model.vocabulary);
  const auto &replay_modifiers = model.sampling_configuration(sampling_index).configuration.modifiers;
  if (!model.cache_plan.draft_cache || replay_modifiers.bias || replay_modifiers.allowed)
    throw std::runtime_error("strict MTP replay requires the original complete target/draft model without external logits modifiers");
  if (!ledger.committed_count || ledger.committed_count > ledger.records.size() ||
      ledger.records[0].generation || ledger.records[0].before.length || model.active_step || model.carrier_borrowed)
    throw std::runtime_error("MTP restore requires a complete zero-origin ledger and an idle carrier");
  for (std::size_t i = 0; i < ledger.committed_count; ++i)
    if (ledger.records[i].requests != 1 || ledger.records[i].member_index != 0)
      throw std::runtime_error("strict MTP replay requires recorded original R1 geometry; use committed-token rebuild for cache recovery");
  make_generation_room(model, prefix, nullptr, requests);
  if (model.snapshot_versions.live >= model.cache_plan.resident_sessions)
    throw std::runtime_error("retire the old MTP head before restoring within N committed versions");
  const auto &first = ledger.records[0];
  const std::array calls{first.ids().size()};
  auto session = model.create_session(calls, {}, first.role, first.before.sampling, false, first.completion, sampling_index);
  GenerationLedger actual(ledger.committed_count, model.cache_plan.candidate_boundaries);
  actual.bind_stop_policy(ledger.stop_policy(), model.vocabulary);
  const auto &eos = model_eos_tokens;
  try {
    for (std::size_t index = 0; index < ledger.committed_count; ++index) {
      const auto &expected = ledger.records[index];
      if (!expected.committed) throw std::runtime_error("MTP restore ledger contains an uncommitted gap");
      const auto pending = expected.pending_input;
      const auto ids = expected.generation ? std::span<const std::int32_t>(&pending, 1) : expected.ids();
      prepare_generation_record(actual, *session, expected.role, ids, expected.generation,
          expected.quota, expected.turn, expected.input_owner, expected.draft_limit,
          expected.completion, expected.input_begin, expected.prefill_complete);
      make_generation_room(model, prefix, session.get(), requests);
      LogitsInputs replay_logits;
      if (expected.completion == TextCompletion::kEmitNext)
        replay_logits = history_logits_inputs(model, actual, ids, {}, sampling_index);
      std::vector<c::ReadLease> q;
      const auto verification = expected.role == GraphRole::kVerify
          ? model.propose_recursive(*session, expected.draft_limit, eos, &q, &actual.stop_policy(), replay_logits) : std::vector<std::int32_t>{};
      if (!expected.generation && is_mtp_prefill(expected.role)) {
        const std::array input{EvaluationRequest{session, ids, &actual.stop_policy(), replay_logits}};
        model.prepare_prefill_evaluations(input, nullptr, expected.role, expected.completion);
      } else model.prepare_evaluation(*session, expected.role == GraphRole::kVerify ? std::span<const std::int32_t>(verification) : ids,
          nullptr, expected.role, expected.completion, &actual.stop_policy(), replay_logits);
      model.submit_evaluation(*session);
      if (stats) {
        ++stats->graph_calls;
        stats->target_rows = cache::add(stats->target_rows, expected.verified_rows);
        stats->draft_rows = cache::add(stats->draft_rows, expected.draft_rows);
      }
      const auto name = "episode" + std::to_string(index);
      double step_seconds = 0;
      if (expected.generation) {
        const EvaluationIO io{name, &trace, trace_states};
        const auto output = evaluation_output(model, session, io, &step_seconds);
        if (expected.role == GraphRole::kVerify) {
          const std::array selections{MtpGenerationSelection{session, pending, expected.quota, q, expected.draft_limit}};
          const std::array outputs{output}; std::array<MtpRoundResult, 1> results;
          GenerationPublicationBatch publication({{&actual, &expected}});
          model.finish_recursive_mtp(selections, eos, outputs, results,
              [&](std::size_t j, const CompletedEvaluationView &view) { observe_completed_evaluation(model, outputs[j], io, view); }, publication);
        } else (void)qwen35::finish_mtp_round(model, *session, pending, expected.quota, eos, &output, io, {&actual, &expected});
      } else {
        (void)qwen35::finish_evaluation(model, *session, name, trace, trace_states, &step_seconds,
            nullptr, nullptr, nullptr, {}, &actual, &expected);
      }
      if (stats) stats->step_seconds += step_seconds;
    }
  } catch (...) {
    session->stop().IgnoreError(); (void)session->drain();
    (void)session->try_retire();
    if (stats) stats->seconds += std::chrono::duration<double>(Clock::now() - started).count();
    throw;
  }
  if (stats) stats->seconds += std::chrono::duration<double>(Clock::now() - started).count();
  return session;
}

// Cache reconstruction consumes immutable committed token chunks. It does not
// resample past outputs or claim to replay an old batch's natural decisions.
std::shared_ptr<Session> rebuild_generation_session(Model &model, const GenerationLedger &ledger,
    const std::filesystem::path &trace = {}, bool trace_states = true,
    GenerationRestoreStats *stats = nullptr, cache::PrefixCache *prefix = nullptr,
    std::span<const GenerationRequest> requests = {}) {
  const auto sampling_index = generation_sampling_index(model, ledger);
  ledger.stop_policy().validate(model.vocabulary);
  const auto started = Clock::now();
  const auto submitted_begin = model.submitted_draft_rows;
  hyle::util::ScopeGuard record_submitted{[&]() noexcept {
    if (stats) stats->submitted_draft_rows += model.submitted_draft_rows - submitted_begin;
  }};
  if (!ledger.committed_count || ledger.committed_count > ledger.records.size() ||
      ledger.records[0].generation || ledger.records[0].before.length || model.active_step || model.carrier_borrowed)
    throw std::runtime_error("MTP rebuild requires a complete zero-origin committed history and idle carrier");
  std::size_t committed_length = 0;
  for (std::size_t i = 0; i < ledger.committed_count; ++i) {
    const auto &record = ledger.records[i];
    if (!record.committed || !record.accepted || (record.generation &&
        (record.accepted > record.accepted_inputs.size() || record.accepted_inputs[0] != record.pending_input ||
          record.consumed_drafts + 1 != record.accepted ||
          (record.accepted > 1 && (!record.accepted_draft || record.accepted_inputs[1] != record.draft)))))
      throw std::runtime_error("MTP rebuild has an invalid committed input boundary");
    const auto ids = record.committed_ids();
    if (record.before.length != committed_length || ids.empty() || ids.size() != record.accepted ||
        record.accepted > model.capacity - committed_length)
      throw std::runtime_error("MTP rebuild committed input chunks are not a continuous bounded prefix");
    committed_length = cache::add(committed_length, record.accepted);
    if (record.after.length != committed_length) throw std::runtime_error("MTP rebuild committed length is inconsistent");
    for (auto id : ids) if (id < 0 || id >= model.vocabulary) throw std::runtime_error("MTP rebuild token is outside vocabulary");
  }
  make_generation_room(model, prefix, nullptr, requests);
  if (model.snapshot_versions.live >= model.cache_plan.resident_sessions)
    throw std::runtime_error("retire the old MTP head before rebuilding within N committed versions");
  if (stats) stats->decision_changes.reserve(cache::add(stats->decision_changes.size(), ledger.committed_count));
  // Modifier values are per-output inputs, not replay history. Rebuild only
  // consumed state, then restore each recorded pending/cursor below; no old
  // mask/bias retention, historical resampling or extra RNG draw is required.
  const bool state_rebuild = !model.sampling_configuration(sampling_index).configuration.modifiers.empty();
  const auto first_ids = ledger.records[0].committed_ids();
  const std::array calls{first_ids.size()};
  const auto role_for = [&](std::size_t rows) {
    return !model.cache_plan.draft_cache ? GraphRole::kText
        : rows == 1 ? GraphRole::kMtpSingle : GraphRole::kMtpPrefill;
  };
  auto session = model.create_session(calls, {}, role_for(first_ids.size()),
      ledger.records[0].before.sampling, false, state_rebuild ? TextCompletion::kStateOnly : ledger.records[0].completion, sampling_index);
  GenerationLedger actual(ledger.committed_count);
  actual.bind_stop_policy(ledger.stop_policy(), model.vocabulary);
  std::size_t observed = 0;
  const auto observe = [&] {
    while (observed < actual.committed_count) {
      const auto &fresh = actual.records[observed]; const auto &old = ledger.records[observed];
      if (stats) {
        ++stats->graph_calls; stats->target_rows = cache::add(stats->target_rows, fresh.accepted);
        stats->draft_rows = cache::add(stats->draft_rows, fresh.draft_rows);
        stats->step_seconds += fresh.step_seconds;
        if (!state_rebuild && fresh.after.next_id != old.after.next_id)
          stats->decision_changes.push_back({observed, old.after.next_id, fresh.after.next_id});
      }
      ++observed;
    }
  };
  try {
    for (std::size_t i = 0; i < ledger.committed_count; ++i) {
      const auto &source = ledger.records[i]; const auto ids = source.committed_ids();
      const auto role = role_for(ids.size());
      const auto completion = state_rebuild ? TextCompletion::kStateOnly : source.completion;
      prepare_generation_record(actual, *session, role, ids, false, 0, source.turn, source.input_owner, 0,
          completion, source.input_begin, state_rebuild ? false : source.prefill_complete);
      make_generation_room(model, prefix, session.get(), requests);
      if ((!source.generation && role == GraphRole::kText) || is_mtp_prefill(role)) {
        const std::array input{EvaluationRequest{session, ids, &actual.stop_policy()}};
        model.prepare_prefill_evaluations(input, nullptr, role, completion);
      } else model.prepare_evaluation(*session, ids, nullptr, role, completion, &actual.stop_policy());
      model.submit_evaluation(*session);
      (void)qwen35::finish_evaluation(model, *session, "episode" + std::to_string(i), trace, trace_states,
          nullptr, nullptr, nullptr, nullptr, {}, &actual);
      observe();
      // Rebuilding committed tokens is not a fresh user sampling episode.
      // The historical pending token and cursor remain authoritative even if
      // a differently partitioned recomputation produces different logits.
      model.restore_pending_boundary(*session, source.after.sampling, source.after.next_id);
    }
    if (session->length() != ledger.records[ledger.committed_count - 1].after.length ||
        (session->head->draft && session->head->draft->lag))
      throw std::runtime_error("MTP committed-token rebuild has an inconsistent complete boundary");
    if (stats) stats->normalized_lag_rows = cache::add(stats->normalized_lag_rows, ledger.records[ledger.committed_count - 1].after.lag);
  } catch (...) {
    observe(); session->stop().IgnoreError(); (void)session->drain();
    (void)session->try_retire();
    if (stats) stats->seconds += std::chrono::duration<double>(Clock::now() - started).count();
    throw;
  }
  if (stats) stats->seconds += std::chrono::duration<double>(Clock::now() - started).count();
  return session;
}

// A completed checkpoint is only a possible recovery shortcut. A newer
// mid-turn commit leaves that old complete key behind; compare it with the
// actual immutable ledger at this cold recovery boundary, without a new key.
bool checkpoint_covers_generation_ledger(const cache::Checkpoint &checkpoint,
                                         const GenerationLedger &ledger) noexcept {
  if (!checkpoint.input || !checkpoint.replay.history || !ledger.committed_count ||
      ledger.committed_count > ledger.records.size()) return false;
  const auto &last = ledger.records[ledger.committed_count - 1];
  if (last.after.next_id < 0 || checkpoint.input->ids.size() != last.after.length ||
      checkpoint.replay.length != last.after.length || checkpoint.replay.chunks != ledger.committed_count ||
      checkpoint.replay.chunks > checkpoint.replay.history->chunks.size() ||
      checkpoint.replay.history->chunks[checkpoint.replay.chunks - 1].next_id != last.after.next_id)
    return false;
  std::size_t offset = 0;
  for (std::size_t i = 0; i < ledger.committed_count; ++i) {
    const auto &record = ledger.records[i]; const auto ids = record.committed_ids();
    if (!record.committed || ids.empty() || ids.size() != record.accepted || record.before.length != offset ||
        offset > checkpoint.input->ids.size() || ids.size() > checkpoint.input->ids.size() - offset ||
        !std::equal(ids.begin(), ids.end(), checkpoint.input->ids.begin() + static_cast<std::ptrdiff_t>(offset)))
      return false;
    offset += ids.size();
    if (record.after.length != offset) return false;
  }
  return offset == last.after.length;
}

// Prove the whole remaining ordinary turn against its authoritative cursor.
// This is admission arithmetic only: no RNG reservation or mutable copy. A new
// prefix hit must start from this request, never from its cached publisher.
void preflight_generation_sampling(const Model &model, const GenerationRequest &request) {
  if (!request.ledger) throw std::runtime_error("generation request requires its cold-prepared commit ledger");
  if (model.cache_plan.draft_cache ||
      !model.sampling_configuration(request.sampling_index).configuration.enabled) return;
  const auto &progress = request.progress;
  std::size_t outputs = 0;
  if (progress.turn_started() && progress.turn_done && progress.more_turns()) {
    // Open input may append its successor while the finished turn is paused.
    // Prove that actual next quota before the shared owner evicts any peer.
    outputs = progress.turns()[progress.turn_index + 1].max_outputs;
    if (progress.generated.size() > progress.maximum_outputs() ||
        outputs > progress.maximum_outputs() - progress.generated.size())
      throw std::runtime_error("request turn exceeds its prepared output limit");
  }
  else if (progress.turn_started()) outputs = progress.remaining_outputs();
  else {
    if (progress.generated.size() > progress.maximum_outputs())
      throw std::runtime_error("request outputs exceed its prepared output limit");
    const auto turns = progress.turns();
    if (turns.empty()) outputs = progress.maximum_outputs() - progress.generated.size();
    else {
      if (progress.turn_index >= turns.size())
        throw std::runtime_error("request turn is outside its prepared schedule");
      outputs = turns[progress.turn_index].max_outputs;
      if (outputs > progress.maximum_outputs() - progress.generated.size())
        throw std::runtime_error("request turn exceeds its prepared output limit");
    }
  }
  if (!outputs) return;
  auto cursor = progress.initial_sampling;
  if (request.session) {
    if (!request.session->head || request.session->stopping || request.session->failed || request.session->pending)
      throw std::runtime_error("ordinary sampling admission requires a live idle Session head");
    cursor = request.session->head->sampling;
  }
  else if (request.ledger->committed_count) {
    if (request.ledger->committed_count > request.ledger->records.size())
      throw std::runtime_error("request recovery exceeds its original committed ledger");
    cursor = request.ledger->records[request.ledger->committed_count - 1].after.sampling;
  }
  // A full hit may have sampled its new frontier before a later request's
  // preparation failed. Import adopted the source ordinals, but its first
  // output is still unobserved. Budget only future draws from that exact
  // published frontier, including when restoring its original ledger.
  const auto &ledger = *request.ledger;
  if (progress.turn_started() && !progress.turn_index && !progress.prefilled &&
      progress.generated.empty() && progress.input_admitted() &&
      progress.input_consumed() == progress.input().size() && ledger.committed_count &&
      ledger.committed_count <= ledger.records.size() && progress.observed_commits() == ledger.committed_count) {
    const auto &last = ledger.records[ledger.committed_count - 1];
    if (last.committed && last.imported && last.user_output && last.turn == progress.turn_index &&
        last.emitted_count == 1 && !last.emitted.empty() && last.emitted[0] == last.after.next_id &&
        last.after.next_id >= 0 && last.after.length == progress.input().size() && last.after.sampling == cursor &&
        (!request.session || generation_boundary(*request.session->head) == last.after))
      --outputs;
  }
  // Includes each not-yet-executed prefill/full-hit output. The last pending
  // token and StateOnly chunks require no additional draw.
  (void)cursor.advanced(outputs);
}

// Only restore the original Session. Request progress, turn input/offset,
// output delivery and its key remain with the serialized owner. In particular,
// this function never begins a turn or observes/emits historical output again.
void restore_generation_session(Model &model, GenerationRequest &request,
    cache::PrefixCache *prefix = nullptr, bool trace_states = true,
    std::span<const GenerationRequest> requests = {}) {
  if (!request.ledger) throw std::runtime_error("generation request requires its cold-prepared commit ledger");
  request.stop_policy.validate(model.vocabulary);
  request.ledger->validate_stop_policy(request.stop_policy);
  const auto &prepared_sampling = model.sampling_configuration(request.sampling_index);
  if (request.session) {
    if (&request.session->sampling_configuration() != &prepared_sampling)
      throw std::runtime_error("request sampling cannot change after Session admission");
    return;
  }
  if (!request.ledger->committed_count) return;
  request.ledger->validate_sampling(prepared_sampling, model.resources_);
  if (generation_sampling_index(model, *request.ledger) != request.sampling_index)
    throw std::runtime_error("request recovery sampling differs from its committed ledger");
  preflight_generation_sampling(model, request);
  if (model.active_step || model.carrier_borrowed)
    throw std::runtime_error("request recovery requires the original idle execution carrier");
  const auto &ledger = *request.ledger;
  if (ledger.committed_count > ledger.records.size())
    throw std::runtime_error("request recovery exceeds its original committed ledger");
  const auto &last = ledger.records[ledger.committed_count - 1];
  std::shared_ptr<Session> restored;
  if (prefix && request.completed_checkpoint.input) {
    if (ledger.sampling == &prepared_sampling &&
        checkpoint_covers_generation_ledger(request.completed_checkpoint, ledger)) {
      const auto length = request.completed_checkpoint.input->ids.size();
      auto hit = prefix->lookup(*request.completed_checkpoint.input, length, length, &model, model.math_identity(), request.cache_sharing_domain);
      if (hit) {
        // A peer may have published the same consumed key with another pending
        // or RNG cursor. Restore only those historical scalar facts; the fork
        // cannot repair an old physical length/frontier/lag/catchup boundary.
        auto head = model.sampling_fork(std::move(hit.payload.snapshot), last.after.sampling, last.after.next_id);
        hit.reset();
        if (generation_boundary(*head) == last.after &&
            bool(head->draft) == model.cache_plan.draft_cache) {
          restored = model.create_session({}, std::move(head),
              model.cache_plan.draft_cache ? GraphRole::kMtpSingle : GraphRole::kText,
              {}, false, TextCompletion::kEmitNext, request.sampling_index);
          ++request.checkpoint_resume_hits;
        }
      }
    }
    if (!restored) ++request.checkpoint_resume_misses;
  }
  if (!restored) {
    if(model.vision && std::ranges::any_of(request.progress.turns(),[](const auto &turn){return !turn.attachments.empty();}))
      throw inference::RequestRejected("image sessions require their complete resident or host checkpoint; token-only replay is insufficient");
    const auto trace = request.trace.empty() ? request.trace
        : request.trace / "restores" / ("restore" + std::to_string(request.restore_count));
    restored = rebuild_generation_session(model, ledger, trace, trace_states, &request.restored, prefix, requests);
    ++request.restore_count;
  }
  // All fallible recovery precedes installing the new original Session owner.
  request.session = std::move(restored);
  request.session_draft_rows_begin = request.session->submitted_draft_rows;
}

struct PreparedGenerationCall final {
  GenerationRequest *request = nullptr;
  GraphRole role = GraphRole::kText;
  std::span<const std::int32_t> prompt;
  std::int32_t pending = -1;
  std::size_t quota = 0, steps = 0;
  std::size_t maximum = 0;
  std::vector<std::int32_t> verification_ids;
  std::vector<c::ReadLease> draft_log_probabilities;
  bool generation = false, full_hit = false, initial_prefix = false, prefill_complete = true;
  bool verification_prepared = false;
  EvaluationDisposition disposition = EvaluationDisposition::kNotPublished;
  LogitsInputs logits; // Snapshot the caller's input leases for this prepared episode.
  inference::LogitsInputProvider logits_provider;
  inference::ConditionalLogitsInputs conditional_logits;
  TextCompletion completion = TextCompletion::kEmitNext;
  std::size_t input_begin = 0;
  std::string name;
  double step_seconds = 0, proposal_seconds = 0;
  std::span<const std::int32_t> ids() const noexcept {
    if (generation && role == GraphRole::kVerify && !verification_ids.empty()) return verification_ids;
    return generation ? std::span(&pending, 1) : prompt;
  }
  // Actual submitted target rows, independent of the cold proposal bound or
  // accepted prefix. Verify has no actual rows until its proposals are ready.
  std::size_t actual_target_rows() const noexcept {
    if (generation && role == GraphRole::kVerify) return verification_ids.size();
    return generation && drafts_token(role) ? maximum + 1 : ids().size();
  }
  std::size_t planned_rows() const noexcept { return role == GraphRole::kVerify || drafts_token(role) ? maximum + 1 : ids().size(); }
};

// Failure recovery may observe already authoritative records, but a second
// host-observation error must not replace the original Status/participant or
// retirement facts. Try every original peer independently. A rejected observer
// leaves its ordinal in the existing ledger for a later observation-only retry.
void observe_generation_commits_after_failure(std::span<const PreparedGenerationCall> calls) noexcept {
  for (const auto &call : calls) {
    try { observe_generation_commits(*call.request); }
    catch (...) {}
  }
}

void validate_generation_logits_inputs(const SamplingConfiguration &configuration,
    const PreparedGenerationCall &call, std::int64_t vocabulary) {
  if (call.conditional_logits.empty()) {
    validate_request_logits_inputs(configuration, call.logits, vocabulary, call.request->request_key); return;
  }
  const auto rows = call.conditional_logits.rows();
  check_request_logits_metadata(prefix_logits_inputs_metadata_status(configuration, call.logits, rows,
      call.ids().size(), vocabulary), call.request->request_key);
  validate_prefix_logits_input_leases(configuration, call.logits, rows);
}

std::size_t generation_prefill_capacity(const Model &model, std::span<const PreparedGenerationCall> calls) {
  if (calls.empty() || !calls[0].request || calls[0].generation || calls[0].full_hit ||
      (calls[0].role != GraphRole::kText && !is_mtp_prefill(calls[0].role))) return 0;
  std::size_t rows = 0;
  for (const auto &call : calls) {
    if (!call.request || call.generation || call.full_hit || call.role != calls[0].role ||
        call.completion != calls[0].completion || call.request->sampling_index != calls[0].request->sampling_index)
      return 0;
    rows = (std::max)(rows, call.ids().size());
  }
  return model.prefill_capacity(rows, calls[0].completion, calls[0].request->sampling_index, calls.size(), calls[0].role);
}

std::size_t generation_verification_capacity(const Model &model, std::span<const PreparedGenerationCall> calls) {
  if (calls.empty() || !calls[0].request || !calls[0].generation || calls[0].full_hit ||
      calls[0].role != GraphRole::kVerify || calls[0].completion != TextCompletion::kEmitNext) return 0;
  std::size_t rows = 0;
  for (const auto &call : calls) {
    if (!call.request || !call.generation || call.full_hit || call.role != GraphRole::kVerify ||
        call.completion != calls[0].completion || call.request->sampling_index != calls[0].request->sampling_index) return 0;
    rows = (std::max)(rows, call.verification_ids.empty() ? call.planned_rows() : call.ids().size());
  }
  return model.verification_capacity(rows, calls.size(), calls[0].request->sampling_index);
}

// One actually retired target batch, never a planned request bucket. Members
// are borrowed synchronously; a report owner copies identities before return.
// Episode durations can overlap and must not be added as a wall-time union.
struct GenerationBatchFacts final {
  GraphRole role; // First member's role; mixed K1 members retain their own roles below.
  TextCompletion completion;
  bool generation;
  std::size_t input_rows; // Maximum actual member width, never total work.
  std::size_t active_rows, physical_rows;
  std::span<const PreparedGenerationCall> members;
  double step_seconds;
};

template<class Observer>
void observe_generation_batch(Observer &observer, std::span<const PreparedGenerationCall> members,
    std::size_t rows, double seconds, std::size_t active_rows = 0, std::size_t physical_rows = 0) {
  if constexpr (!std::is_null_pointer_v<std::remove_cvref_t<Observer>>) {
    if (!active_rows)
      for (const auto &member : members) active_rows = cache::add(active_rows, member.actual_target_rows());
    if (!physical_rows) physical_rows = cache::multiply(rows, members.size());
    observer(GenerationBatchFacts{members[0].role, members[0].completion, members[0].generation,
        rows, active_rows, physical_rows, members, seconds});
  }
}

// Serialized host metadata only. Original Step/Source owns all native work;
// a deferred generation call may cross pumps without exporting another head.
// Exact Model/Graph checks, not warm-direction hints, authorize execution.
struct GenerationContinuation final {
  bool enabled = true, unavailable = false;
  inference::ContinuationCohort cohort;
  std::size_t next_slot = 0;
  std::optional<std::size_t> ready_slot;
  // Actual typed coverage-refusal recovery only. Ordinary ready execution
  // must re-prime this slot from its real predecessor before speculating again.
  // This owns no Source, captured tuple, native work or admission permission.
  std::optional<std::size_t> reprime_slot;
  std::array<bool, execution_slot_count> direction_warm{};
  std::size_t ready_steps = 0;
  std::shared_ptr<StepSources> ready;
  hyle::container::SmallVector<PreparedGenerationCall, 2> pending_calls;
  // These are original call descriptors and a host control cursor, never
  // another Completion or permission owner. A may still own an accepted B.
  inference::StepDriveCursor drive_cursor;
  bool pending_synchronous = false, pending_was_successor = false;
  hyle::container::SmallVector<PreparedGenerationCall, 2> pending_successor_calls;
  std::optional<inference::EvaluationFailure> pending_refusal;
  std::size_t completed_steps = 0, successor_submissions = 0, successor_publications = 0,
      successor_discards = 0, successor_refusals = 0;

  void prepare(std::size_t maximum_members) {
    cohort.prepare(maximum_members);
    if (!pending_calls.try_reserve(maximum_members)) throw std::bad_alloc{};
    if (!pending_successor_calls.try_reserve(maximum_members)) throw std::bad_alloc{};
  }
  void release_ready() noexcept {
    ready.reset(); ready_slot.reset(); reprime_slot.reset(); next_slot = 0;
  }
  void reset_reuse() noexcept {
    release_ready(); direction_warm.fill(false); ready_steps = 0;
  }
  void reset() noexcept { reset_reuse(); cohort.reset_ready(); }
  static inference::ContinuationMember identity(const PreparedGenerationCall &call) noexcept {
    return {call.request->request_key, call.request->progress.turn_index};
  }
  bool select(std::span<const PreparedGenerationCall> calls) {
    const auto selected = cohort.select(calls, identity, !pending_calls.empty());
    if (selected == inference::ContinuationCohort::Selection::kPendingConflict)
      throw std::logic_error("pending continuation cannot change its original cohort");
    if (selected == inference::ContinuationCohort::Selection::kRejected) { reset(); return false; }
    if (selected == inference::ContinuationCohort::Selection::kChanged) reset_reuse();
    return true;
  }
  // Admission may run for one member before the next compatible batch exists.
  // Keep a matching cohort, but release it before another owner needs slots.
  void admission(inference::RequestKey selected, std::size_t selected_turn) noexcept {
    if (cohort.admission({selected, selected_turn}, !pending_calls.empty()) == inference::ContinuationCohort::Release::kReady)
      reset();
  }
  void release(inference::RequestKey selected) noexcept {
    if (cohort.release(selected, !pending_calls.empty()) == inference::ContinuationCohort::Release::kReady) reset();
  }
  bool matches(std::span<const PreparedGenerationCall> calls) const noexcept {
    const auto members = cohort.members();
    if (!ready || calls.size() != members.size() || ready->boundaries.size() != calls.size()) return false;
    for (std::size_t j = 0; j < calls.size(); ++j) {
      const auto &request = *calls[j].request;
      if (members[j] != identity(calls[j]) || !request.session || !request.session->head)
        return false;
      const auto &boundary = ready->boundaries[j]; const auto owner = boundary.session.lock();
      if (owner != request.session || owner.owner_before(request.session) || request.session.owner_before(owner) ||
          boundary.candidate != request.session->head.get() || boundary.epoch != request.session->head.epoch()) return false;
      if (ready->mtp_full_candidate && (calls.size() != 1 || !drafts_token(calls[j].role) ||
          request.session->head->length != boundary.length || !request.session->head->draft ||
          request.session->head->draft->frontier != boundary.draft_frontier || request.session->head->draft->lag != 1))
        return false;
    }
    return true;
  }
  void clear_pending() noexcept {
    pending_calls.clear(); pending_successor_calls.clear(); pending_refusal.reset();
    drive_cursor = {}; pending_synchronous = pending_was_successor = false;
    cohort.clear_drain();
  }
  void hold_current(std::span<PreparedGenerationCall> calls, bool synchronous, bool was_successor = false) {
    if (calls.data() != pending_calls.data()) {
      pending_calls.clear();
      for (auto &call : calls) pending_calls.push_back(std::move(call));
    }
    pending_synchronous = synchronous; pending_was_successor = was_successor;
  }
  bool matches_pending_successor(const StepState *step) const noexcept {
    if (!drive_cursor.awaiting_current) return !step;
    if (drive_cursor.successor == inference::StepSuccessor::kPrepared)
      return step && step->submitted && pending_successor_calls.size() == pending_calls.size();
    // Typed preparation refusal can leave B's unpublished transactions
    // pinned until the original in-flight A really retires. It has no output
    // reservation and must never be mistaken for a submitted successor.
    if (drive_cursor.successor == inference::StepSuccessor::kRefused)
      return pending_refusal && (!step || (!step->submitted && step->cancelled && step->execution.cleanup_pending));
    return !step;
  }
};

std::span<const std::int32_t> provider_committed_outputs(const GenerationRequest &request) {
  if (!request.progress.turn_started()) return {};
  return std::span<const std::int32_t>(request.progress.generated).subspan(request.progress.turn_output_begin);
}

PreparedGenerationCall prepare_generation_call(Model &model, GenerationRequest &request, std::size_t steps,
    cache::PrefixCache *prefix, std::span<const GenerationRequest> requests,
    std::size_t output_credit = SIZE_MAX, std::size_t max_prefill_rows = SIZE_MAX) {
  if (!request.ledger) throw std::runtime_error("generation request requires its cold-prepared commit ledger");
  request.stop_policy.validate(model.vocabulary);
  request.ledger->validate_stop_policy(request.stop_policy);
  const auto &prepared_sampling = model.sampling_configuration(request.sampling_index);
  if (request.session && &request.session->sampling_configuration() != &prepared_sampling)
    throw std::runtime_error("request sampling cannot change after Session admission");
  request.ledger->validate_sampling(prepared_sampling, model.resources_);
  const auto &sampler = prepared_sampling.configuration;
  if (!max_prefill_rows) throw std::runtime_error("prefill rows require a positive quantum");
  const inference::PrefillPolicy prefill(max_prefill_rows);
  // Credit belongs to this episode, not the turn's immutable output quota.
  // Reject before prefix lookup, session creation or any other admission work.
  if (!output_credit) throw std::runtime_error("generation must wait for output credit before preparation");
  if (steps != request.progress.output_limit())
    throw std::runtime_error("generation call quota differs from its prepared request turn");
  if (request.progress.turn_started() && request.progress.turn_done)
    throw std::runtime_error("generation request has already completed its current turn");
  if (request.logits_provider && request.session && (!request.session->head || request.session->stopping ||
      request.session->failed || request.session->pending))
    throw std::runtime_error("dynamic logits preparation requires a live idle Session");
  preflight_generation_sampling(model, request);
  LogitsInputs episode_logits = request.logits;
  inference::LogitsInputProvider episode_provider;
  if (request.logits_provider && !sampler.modifiers.bias && !sampler.modifiers.allowed)
    check_request_logits_metadata(hyle::Status::InvalidArgument("dynamic logits require configured bias or mask ports"), request.request_key);
  if (!sampler.modifiers.empty()) {
    bool emits = request.progress.prefilled;
    if (!emits) {
      auto consumed = request.progress.input_consumed();
      if (consumed > request.prompt.size()) throw std::runtime_error("prefill offset exceeds its turn input");
      if (prefix && request.progress.turn_index == 0 && !request.session) {
        if (!request.input) throw std::runtime_error("prefix admission requires its complete token plan");
        // No eviction/publication or directory mutation occurs between this
        // pure length preview and the original pinned lookup below.
        consumed = prefix->match_length(*request.input, request.prompt.size(), 1, &model, model.math_identity(), request.cache_sharing_domain);
      }
      emits = prefill.next(request.prompt.size(), consumed).complete;
    }
    if (emits) {
      auto external = sampler;
      external.modifiers.history = {}; // The ledger supplies natural C0 below.
      if (request.logits_provider) {
        auto base = external; base.modifiers.bias = false; base.modifiers.allowed = false;
        validate_request_logits_inputs(base, request.logits, model.vocabulary, request.request_key);
        episode_provider = request.logits_provider;
        auto supplied = episode_provider(provider_committed_outputs(request), {});
        if (!supplied) {
          auto status = hyle::Status(supplied.error());
          if (status.IsInvalidArgument()) check_request_logits_metadata(std::move(status), request.request_key);
          throw inference::EvaluationFailure(std::move(status));
        }
        episode_logits = std::move(*supplied);
      }
      validate_request_logits_inputs(external, episode_logits, model.vocabulary, request.request_key);
    }
  }
  if (!request.progress.turn_started()) (void)request.progress.begin_turn();
  if (request.progress.turn_done) throw std::runtime_error("MTP request has already completed");
  if (request.round_emitted.size() < steps - 1)
    throw std::runtime_error("MTP observer slots must be reserved for every turn before execution");
  request.ledger->bind_stop_policy(request.stop_policy, model.vocabulary);
  PreparedGenerationCall call; call.request = &request; call.steps = steps; call.logits = std::move(episode_logits);
  call.logits_provider = std::move(episode_provider);
  call.generation = request.progress.prefilled;
  if (!call.generation) {
    if (!request.progress.input_admitted()) {
      const auto before = request.session ? request.session->length() : 0;
      if (before > model.capacity || request.prompt.empty() || request.prompt.size() > model.capacity - before)
        throw std::runtime_error("complete turn input exceeds the remaining context");
      // The turn's final output remains pending. Admit its full maximum journey
      // before any input chunk, prefix import or Session is published.
      const auto outputs = request.progress.remaining_outputs();
      if (!outputs || outputs - 1 > model.capacity - before - request.prompt.size())
        throw std::runtime_error("complete turn input and output quota exceed the remaining context");
      for (auto id : request.prompt) if (id < 0 || id >= model.vocabulary)
        throw std::runtime_error("token ID outside vocabulary");
      request.progress.bind_input(request.prompt);
    }
    request.ledger->validate_sampling(prepared_sampling, model.resources_);
    std::cerr << "memory_phase=" << request.memory_name << "_prefill_begin\n";
    if (!request.progress.input_consumed()) request.prefill_start = request.admission_start == Clock::time_point{} ? Clock::now()
        : std::exchange(request.admission_start, Clock::time_point{});
    call.initial_prefix = prefix && request.progress.turn_index == 0;
    if (call.initial_prefix && !request.session) {
      auto hit = prefix->lookup(*request.input, request.prompt.size(), 1, &model, model.math_identity(), request.cache_sharing_domain);
      if (hit) {
        if (hit.payload.snapshot->draft.has_value() != model.cache_plan.draft_cache)
          throw std::runtime_error("prefix directory contains a different target/draft state contract");
        const auto cached_lag = hit.payload.snapshot->draft ? hit.payload.snapshot->draft->lag : 0;
        const auto reused_tokens = hit.payload.snapshot->length;
        const bool full_hit = reused_tokens == request.prompt.size();
        auto history = cache::copy_history(hit.payload.replay, model.capacity);
        preflight_generation_prompt_ledger(*request.ledger, hit.payload.replay);
        request.progress.preflight_adopt_committed_history(reused_tokens);
        const auto suffix_part = prefill.next(request.prompt.size(), reused_tokens);
        const auto suffix = request.prompt.size() - suffix_part.begin;
        const auto suffix_rows = suffix_part.rows;
        const auto suffix_completion = suffix_part.complete ? TextCompletion::kEmitNext : TextCompletion::kStateOnly;
        std::string hit_name = "prefill";
        if (!suffix_part.complete) hit_name += "_chunk" + std::to_string(reused_tokens);
        const std::array chunks{suffix_rows};
        const auto prior_sampling_seconds = model.frontier_sampling_seconds;
        const auto suffix_role = !model.cache_plan.draft_cache ? GraphRole::kText : suffix_rows == 1
            ? (cached_lag ? GraphRole::kMtpCatchupSingle : GraphRole::kMtpSingle)
            : (cached_lag ? GraphRole::kMtpCatchupPrefill : GraphRole::kMtpPrefill);
        // Keep every new request fact local until the actual Session/frontier
        // succeeds. Failed native/resource admission must permit the same retry.
        const auto admission_role = !model.cache_plan.draft_cache ? GraphRole::kText
            : suffix ? suffix_role : request.prompt.size() == 1 ? GraphRole::kMtpSingle : GraphRole::kMtpPrefill;
        if (full_hit && !sampler.modifiers.history.empty())
          call.logits = history_logits_inputs(model, *request.ledger,
              std::span<const std::int32_t>(request.input->ids).first(reused_tokens), call.logits, request.sampling_index);
        std::shared_ptr<Session> adopted;
        try {
          adopted = model.create_session(suffix ? std::span<const std::size_t>(chunks) : std::span<const std::size_t>{},
              hit.payload.snapshot, admission_role,
              request.progress.initial_sampling, suffix == 0, suffix_completion, request.sampling_index, call.logits);
        } catch (const inference::EvaluationFailure &failure) {
          const auto key = request.request_key;
          if (failure.status().IsInvalidArgument() && !failure.status().has_native_error() && !failure.participant() &&
              failure.retirement_observed() && failure.retirement().retired && !failure.retirement().status.has_native_error() &&
              key.generation && key.slot != (std::numeric_limits<std::size_t>::max)())
            throw inference::RequestExecutionFailure::from_request_evaluation(failure, key);
          throw;
        }
        if (!suffix) history->chunks.back().next_id = adopted->head->next_id;
        // Only original reserved metadata remains. Preconditions are unchanged
        // since preflight; no allocation or native work follows this commit.
        import_generation_prompt_ledger(*request.ledger, hit.payload.replay, full_hit, model.cache_plan.draft_cache);
        if (sampler.enabled || !sampler.modifiers.empty() || model.sampling_configuration_count() > 1) {
          for (std::size_t i = 0; i < request.ledger->committed_count; ++i) {
            auto &record = request.ledger->records[i];
            record.before.sampling = sampler.enabled ? request.progress.initial_sampling : cache::SamplingCursor{};
            record.after.sampling = record.before.sampling;
            record.emitted_count = 0; write_generation_tokens(record.emitted, {-1, -1});
          }
          request.ledger->records[request.ledger->committed_count - 1].after.sampling = adopted->head->sampling;
        }
        if (!suffix) {
          auto &last = request.ledger->records[request.ledger->committed_count - 1];
          last.after.sampling = adopted->head->sampling;
          last.after.next_id = adopted->head->next_id;
          last.target_first = last.after.next_id; write_generation_tokens(last.emitted, {last.after.next_id, -1});
          last.emitted_count = 1; last.stopped = inference::request_token_stopped(
              last.after.next_id, model_eos_tokens, &request.stop_policy);
        }
        request.ledger->bind_sampling(prepared_sampling, model.resources_);
        request.progress.adopt_committed_history(request.ledger->committed_count, reused_tokens);
        request.history = std::move(history);
        request.session = std::move(adopted);
        request.reused_tokens = reused_tokens; request.prefix_hits = 1;
        request.prefix_full_hits = full_hit ? 1 : 0;
        call.name = std::move(hit_name);
        if (sampler.enabled || !sampler.modifiers.empty() || model.sampling_configuration_count() > 1) {
          request.prefix_sampling_calls = suffix == 0 ? 1 : 0;
          request.prefix_sampling_seconds = model.frontier_sampling_seconds - prior_sampling_seconds;
        }
        hit.reset();
      } else { request.prefix_misses = 1; request.history = std::make_shared<cache::ReplayHistory>(model.capacity); }
    }
    const auto part = request.progress.next_input(prefill);
    if(request.image_turn!=request.progress.turn_index){
      const auto &turn=request.progress.turns()[request.progress.turn_index];
      request.image_input.reset();
      if(!turn.attachments.empty()){
        if(!model.vision || model.cache_plan.draft_cache)
          throw inference::RequestRejected("image input requires a prepared ordinary vision execution profile");
        const auto before=request.session?request.session->length():0;
        const auto offset=request.session?request.session->head->rotary_offset:0;
        try {
          request.image_input=std::make_unique<ConditionedPrompt>(prepare_image_prompt(*model.vision,request.prompt,
              turn.attachments,request.prompt.size()-turn.prompt.size(),static_cast<std::int64_t>(before)+offset,model.precision.activation_type()));
        } catch(const std::invalid_argument &error){ throw inference::RequestRejected(error.what()); }
      }
      request.image_turn=request.progress.turn_index;
    }
    call.input_begin = part.begin;
    call.prefill_complete = part.complete;
    call.completion = call.prefill_complete ? TextCompletion::kEmitNext : TextCompletion::kStateOnly;
    call.prompt = part.ids;
    const auto lag = request.session && request.session->head->draft ? request.session->head->draft->lag : 0;
    call.role = !model.cache_plan.draft_cache ? GraphRole::kText
        : call.prompt.size() == 1 ? (lag ? GraphRole::kMtpCatchupSingle : GraphRole::kMtpSingle)
        : (lag ? GraphRole::kMtpCatchupPrefill : GraphRole::kMtpPrefill);
    if (!request.session) {
      make_generation_room(model, prefix, nullptr, requests);
      const std::array chunks{call.prompt.size()}; request.session = model.create_session(chunks, {}, call.role, request.progress.initial_sampling, false, call.completion, request.sampling_index);
    }
    if (!request.ledger->sampling) request.ledger->bind_sampling(prepared_sampling, model.resources_);
    if (!request.turn_observations.empty() &&
        call.input_begin == (request.progress.turn_index ? 0 : request.reused_tokens))
      request.turn_observations[request.progress.turn_index].context_before = request.session->length();
    call.full_hit = call.prompt.empty();
    if (call.name.empty()) {
      call.name = request.progress.turn_index ? "round" + std::to_string(request.progress.turn_index + 1) + "_prefill" : "prefill";
      if (!call.prefill_complete || call.input_begin > request.reused_tokens)
        call.name += "_chunk" + std::to_string(call.input_begin);
    }
  } else {
    call.quota = (std::min)(request.progress.remaining_outputs(), output_credit);
    call.pending = request.session->head->next_id;
    if (request.progress.generated.empty() || call.pending != request.progress.generated.back())
      throw std::runtime_error("generation pending differs from its last observed output");
    const auto remaining = model.capacity - request.session->length();
    if (!remaining) throw std::runtime_error("MTP has exhausted the configured context capacity");
    request.ledger->bind_sampling(prepared_sampling, model.resources_);
    call.maximum = model.cache_plan.draft_cache
        ? (std::min)({model.max_drafts(), request.max_draft_tokens, call.quota - 1, remaining - 1}) : 0;
    const bool one = call.maximum == 0;
    const auto lag = request.session->head->draft ? request.session->head->draft->lag : 0;
    call.role = !model.cache_plan.draft_cache ? GraphRole::kText
        : one ? (lag ? GraphRole::kMtpCatchupSingle : GraphRole::kMtpSingle)
        : !call.logits_provider && (!model.recursive_mtp() ||
            (call.maximum == model.max_drafts() && model.cache_plan.in_flight_batches >= 3 &&
             model.greedy_draft_chain_enabled(sampler)))
            ? (lag ? GraphRole::kMtpCatchupRound : GraphRole::kMtpRound) : GraphRole::kVerify;
    call.name = (request.progress.turn_index ? "round" + std::to_string(request.progress.turn_index + 1) + "_" : "") +
        (model.cache_plan.draft_cache ? "mtp" : "decode") + std::to_string(request.rounds.size() + 1);
  }
  if (call.completion == TextCompletion::kStateOnly) call.logits = {};
  else {
    // A wave can exceed the actual physical cohort. Allocate its bounded C0
    // only when this call enters an executing bucket below. A full-hit's
    // frontier has already retired here, so it needs no later counts pin.
    if (call.full_hit) call.logits.counts = {};
    auto external = sampler; external.modifiers.history = {};
    validate_request_logits_inputs(external, call.logits, model.vocabulary, request.request_key);
  }
  return call;
}

bool make_generation_batch_room(Model &model, cache::PrefixCache *prefix,
    std::span<const PreparedGenerationCall> calls, std::span<const GenerationRequest> requests) {
  const auto limit = model.cache_plan.resident_sessions;
  const auto selected = calls | std::views::transform([](const auto &call) {
    return version_holder(call.request->session.get());
  });
  const auto residents = requests | std::views::transform([&](const auto &request) {
    return version_holder(request.prepared_model && request.prepared_model != &model ? nullptr : request.session.get());
  });
  const auto observe = [&] {
    const auto live = std::size_t(model.snapshot_versions.live);
    if (live > limit || model.active_step) throw std::runtime_error("generation batch admission requires retired heads within N");
    return inference::VersionRoomObservation{{live, limit}, prefix ? prefix->size() : 0,
        inference::version_admission({live, limit}, selected, residents,
            [&](const auto *head) { return prefix && prefix->retains(head); }).fits};
  };
  return inference::make_version_room(observe, [&] { return prefix && prefix->evict_one(); }) ==
      inference::VersionRoomResult::kReady;
}

void finish_generation_call(Model &model, PreparedGenerationCall &call) {
  if (call.disposition == EvaluationDisposition::kCancelled) { call.logits.counts = {}; return; }
  auto &request = *call.request;
  if (!request.ledger) throw std::runtime_error("generation request requires its cold-prepared commit ledger");
  if (call.full_hit && request.progress.prefilled) { call.logits.counts = {}; return; } // A committed cache hit is observed once.
  if (call.full_hit) {
    const auto next_id = request.session->head->next_id;
    const auto emitted = std::span<const std::int32_t>(&next_id, 1);
    const bool stopped = inference::request_token_stopped(next_id, model_eos_tokens, &request.stop_policy);
    request.progress.preflight_outputs(emitted, true, true, stopped);
    // The imported request ledger already adopted source history. The new
    // frontier output is observed directly, without inventing a model ordinal.
    request.progress.apply_outputs(emitted, true, true, stopped);
    request.prefill_endpoint = Clock::now(); request.endpoint = request.prefill_endpoint;
  } else observe_generation_commits(request);
  if (!call.generation && call.prefill_complete) {
    request.endpoint = request.prefill_endpoint;
  }
  request.visible_tokens = request.session->length();
  request.draft_frontier = request.session->head->draft ? request.session->head->draft->frontier : 0;
  request.draft_lag = request.session->head->draft ? request.session->head->draft->lag : 0;
  request.fresh_next_id = request.session->head->next_id;
  if (!request.turn_observations.empty()) {
    auto &turn = request.turn_observations[request.progress.turn_index];
    if (!call.generation && call.prefill_complete) { turn.start = request.prefill_start; turn.prefill_endpoint = request.prefill_endpoint; }
    turn.endpoint = request.endpoint;
    turn.output_end = request.progress.generated.size();
    turn.context_after = request.visible_tokens;
  }
  if (call.full_hit)
    std::cerr << "prefix_hit name=prefill reused_tokens=" << request.reused_tokens
              << " prompt_graph_calls=0 frontier_sampling_graph_calls=" << request.prefix_sampling_calls
              << " visible=" << request.session->length() << '\n';
  if (!call.generation && call.prefill_complete)
    std::cerr << "memory_phase=" << request.memory_name << "_prefill_end\n"
              << "memory_phase=" << request.memory_name << "_decode_begin\n";
  if (request.progress.turn_done) std::cerr << "memory_phase=" << request.memory_name << "_decode_end\n";
  if (call.full_hit) {
    if (!request.trace.empty()) {
      const auto path = request.trace / "prefill"; std::filesystem::create_directories(path);
      std::ofstream metadata(path / "cache-hit.json");
      if (!metadata) throw std::runtime_error("cannot write MTP prefix-hit metadata");
      metadata << "{\"prefix_hit\":true,\"prompt_graph_calls\":0,\"frontier_sampling_graph_calls\":" << request.prefix_sampling_calls
               << ",\"graph_calls\":" << request.prefix_sampling_calls << ",\"tensor_payload_available\":false,\"reused_tokens\":"
               << request.reused_tokens << ",\"next_id\":" << request.progress.generated.back() << "}\n";
    }
  }
  call.logits.counts = {}; // Native consumers retired; the completed call no longer needs C0.
  (void)model;
}

// A borrowed model adapter for the common serialized step driver. It holds
// only this call's original scratch and Source pins; Model/Graph own execution.
template<class AfterSubmit, class BatchObserver>
struct GenerationStepAdapter final {
  static constexpr bool kSupportsSuccessor = true;
  using Position = inference::StepPosition;
  Model &model;
  std::span<PreparedGenerationCall> calls;
  bool trace_states;
  GenerationContinuation &continuation;
  AfterSubmit &after_submit;
  BatchObserver &batch_observer;
  std::span<const inference::RequestCreditGrant> credits;
  bool drain_requested, resumed, mtp, device_mtp, continuous_mtp, imported;
  std::size_t maximum_outputs, slot, next_slot;
  GraphRole successor_role;
  std::optional<std::size_t> prior_slot;
  hyle::container::SmallVector<PreparedGenerationCall, 2> &second;
  hyle::container::SmallVector<EvaluationRequest, 2> direct, relay;
  hyle::container::SmallVector<EvaluationOutput, 2> outputs;
  hyle::container::SmallVector<MtpGenerationSelection, 2> selections;
  hyle::container::SmallVector<MtpRoundResult, 2> results;
  hyle::container::SmallVector<std::int32_t, 2> next;
  std::shared_ptr<StepSources> a_source, b_source;
  StepState *a = nullptr, *b = nullptr;
  Clock::time_point first_start{}, second_start{};
  std::optional<inference::EvaluationFailure> &refused_failure;
  bool draining = false;

  GenerationStepAdapter(Model &selected, std::span<PreparedGenerationCall> original, bool trace,
      GenerationContinuation &pending, AfterSubmit &submitted, BatchObserver &observer,
      std::span<const inference::RequestCreditGrant> granted, bool drain, bool resume)
      : model(selected), calls(original), trace_states(trace), continuation(pending), after_submit(submitted),
        batch_observer(observer), credits(granted), drain_requested(drain),
        resumed(pending.drive_cursor.awaiting_current ? pending.pending_was_successor : resume),
        mtp(model.cache_plan.draft_cache), device_mtp(model.device_mtp_successor_enabled()),
        continuous_mtp(model.device_mtp_continuous_enabled()),
        imported(resume ? model.active_step->phase->imported_sources : !device_mtp && continuation.matches(calls)),
        maximum_outputs(mtp ? model.max_drafts()+1 : 1), slot(resume ? model.active_step->state_slot : device_mtp ? 0 : continuation.next_slot),
        next_slot(device_mtp ? (continuous_mtp ? (imported ? (slot + 1) % execution_slot_count
            : original[0].role == GraphRole::kMtpCatchupRound ? 1u : 0u) : 0u)
            : (slot + 1) % execution_slot_count),
        successor_role(device_mtp ? GraphRole::kMtpContinuation : mtp ? GraphRole::kMtpCatchupRound : GraphRole::kText),
        prior_slot(continuation.ready_slot),
        second(pending.pending_successor_calls), refused_failure(pending.pending_refusal) {
    if (pending.drive_cursor.awaiting_current) {
      a = model.active_step.get(); b = model.successor_step.get();
      first_start = a->step_start; a_source = a->execution.sources;
      if (b) { second_start = b->step_start; b_source = b->execution.sources; }
      draining = continuation.cohort.draining(drain_requested, credits);
    } else { second.clear(); refused_failure.reset(); }
    if (!pending.drive_cursor.awaiting_current) prepare_scratch();
  }
  void prepare_scratch() {
    if (!second.try_reserve(calls.size()) || !direct.try_reserve(calls.size()) ||
        !relay.try_reserve(calls.size())) throw std::bad_alloc{};
    finish_scratch();
  }
  void finish_scratch() {
    if (!outputs.try_reserve(calls.size()) ||
        !next.try_resize(calls.size(), -1) || (mtp &&
            (!selections.try_reserve(calls.size()) || !results.try_resize(calls.size())))) throw std::bad_alloc{};
  }

  void notify() {
    if constexpr (!std::is_null_pointer_v<std::remove_cvref_t<AfterSubmit>>) after_submit();
  }
  void prepare(Position) {
    for (auto &call : calls) {
      auto &request = *call.request;
      if (!resumed)
        prepare_generation_record(*request.ledger, *request.session, call.role, call.ids(), call.generation,
            call.quota, request.progress.turn_index, {}, call.maximum, call.completion,
            call.input_begin, call.prefill_complete);
      direct.push_back({request.session, call.ids(), &request.ledger->stop_policy(), call.logits});
      relay.push_back({request.session, {}, &request.ledger->stop_policy(), call.logits});
    }
    a = resumed ? model.active_step.get()
        : imported ? &model.prepare_relay_evaluations(std::span(relay.data(), relay.size()), continuation.ready, slot, successor_role)
        : (model.prepare_evaluations(std::span(direct.data(), direct.size()), nullptr, calls[0].role), model.active_step.get());
    first_start = a->step_start;
    if (resumed) a_source = a->execution.sources;
  }
  void submit(Position position) {
    if (position == Position::kCurrent) {
      a_source = model.prepare_submission_sources(*a);
      model.submit_evaluations(*a, true);
      if (!a->execution.sources) a_source.reset();
    } else {
      second_start = b->step_start;
      if (!device_mtp || continuous_mtp) b_source = model.prepare_submission_sources(*b);
      model.submit_evaluations(*b);
      ++continuation.successor_submissions;
    }
    notify();
  }
  bool cancelled() const {
    // One stopped member can leave A's other members publishable. Only actual
    // whole-episode cancellation/failure takes the common abort branch.
    return a->cancelled || std::ranges::any_of(calls, [](const auto &call) { return call.request->session->failed; });
  }
  bool current_ready() { return take(a->execution.poll()) != c::CompletionPoll::kPending; }
  void defer_current(inference::StepSuccessor) {
    continuation.hold_current(calls, false, resumed);
  }
  inference::StepSuccessor prepare_successor() {
    // A device-selected B is the terminal member of this pair. It owns no
    // outgoing Source and must not enter either legacy relay or C preparation.
    if (device_mtp && !continuous_mtp && a && a->device_input_choice) return inference::StepSuccessor::kNone;
    if (!a_source) return inference::StepSuccessor::kUnavailable;
    draining = continuation.cohort.draining(drain_requested, credits);
    if (resumed && !draining) {
      if (credits.size() != calls.size()) throw std::logic_error("pending batch lost its original request grants");
      for (std::size_t j = 0; j < calls.size(); ++j) {
        auto &call = calls[j]; auto &request = *call.request;
        if (credits[j].cancelled || request.session->stopping || request.progress.turn_done) continue;
        call.quota = request.ledger->renew_pending_quota(generation_boundary(*request.session->head),
            request.progress.observed_commits(), request.progress.turn_index, request.progress.remaining_outputs(),
            credits[j].reserved, credits[j].additional).quota;
      }
    }
    // This call's actual incoming Source is only a necessary capture-key
    // candidate. Original B preparation/submission still proves the complete
    // current tuple, permissions and whole predecessor outcome.
    if (draining || (!device_mtp && (!imported || continuation.reprime_slot)) ||
        !std::ranges::all_of(calls, [&](const auto &call) {
          const auto &request = *call.request; const auto &ledger = *request.ledger;
          return !request.session->stopping && !request.session->failed && call.quota >= 2 * maximum_outputs &&
              request.progress.remaining_outputs() >= 2 * maximum_outputs &&
              model.capacity - request.session->length() >= 2 * maximum_outputs &&
              request.round_emitted.size() - request.rounds.size() >= 2 &&
              request.rounds.capacity() - request.rounds.size() >= 2 &&
              ledger.records.size() - ledger.committed_count >= 2;
        })) return inference::StepSuccessor::kNone;
    // Candidate lookup stays with the model that owns the original Phase and
    // carrier; preparation below still proves the actual submission rights.
    if (!model.has_relay_successor_capture(a, *a_source, calls.size(), maximum_outputs,
            successor_role, calls.front().request->sampling_index, next_slot, device_mtp))
      return inference::StepSuccessor::kNone;
    for (std::size_t j = 0; j < calls.size(); ++j) {
      const auto &call = calls[j]; auto &request = *call.request;
      second.push_back(call); auto &following = second.back();
      following.pending = -1; following.quota = call.quota - maximum_outputs;
      following.role = successor_role;
      following.step_seconds = following.proposal_seconds = 0;
      following.verification_prepared = false;
      following.disposition = EvaluationDisposition::kNotPublished;
      following.name = (request.progress.turn_index ? "round" + std::to_string(request.progress.turn_index + 1) + "_" : "") +
          (mtp ? "mtp" : "decode") + std::to_string(request.rounds.size() + 2);
      request.ledger->reserve_successor(request.progress.observed_commits(), following.quota,
          request.progress.turn_index, calls.size(), j);
    }
    try { b = &model.prepare_relay_evaluations(std::span(relay.data(), relay.size()), a_source, next_slot, successor_role); }
    catch (const inference::EvaluationFailure &failure) {
      if (!(failure.status().IsUnavailable() || (device_mtp && failure.status().IsUnimplemented())) ||
          failure.status().has_native_error() ||
          (failure.retirement_observed() && !failure.retirement().status.ok()) ||
          (model.successor_step && model.successor_step->submitted)) throw;
      refused_failure = failure; ++continuation.successor_refusals;
      std::cerr << "decode_overlap_refused code=" << hyle::status_code_name(failure.status().code())
          << " native=" << failure.status().has_native_error() << " predecessor_slot=" << slot
          << " successor_slot=" << next_slot << " members=" << calls.size()
          << " committed_length=" << calls.front().request->session->length()
          << " candidate_length=" << a->members.front().next->length
          << " reason=" << failure.status().message() << '\n';
      return inference::StepSuccessor::kRefused;
    }
    return inference::StepSuccessor::kPrepared;
  }
  void finish(Position position) {
    finish_scratch();
    const bool device_successor = device_mtp && model.active_step->device_input_choice;
    const auto current = position == Position::kCurrent ? calls : std::span(second.data(), second.size());
    const auto rows = model.active_step->input_tokens;
    const auto started = model.active_step->step_start;
    outputs.clear(); selections.clear();
    for (auto &call : current) {
      auto &request = *call.request;
      outputs.push_back(evaluation_output(model, request.session, {call.name, &request.trace, trace_states},
          &call.step_seconds, &request.endpoint, nullptr, nullptr));
      outputs.back().disposition = &call.disposition;
      if (mtp) selections.push_back({request.session, call.pending, call.quota, {}, call.maximum});
    }
    GenerationPublicationBatch publication(current.size(), [&](std::size_t index) {
      return GenerationPublicationInput{current[index].request->ledger.get(), nullptr};
    });
    const auto observe = [&](std::size_t j, const CompletedEvaluationView &view) {
      auto &call = current[j]; auto &request = *call.request;
      observe_generation_preparation(request, view, call.generation);
      observe_completed_evaluation(model, outputs[j], {call.name, &request.trace, trace_states}, view);
    };
    if (mtp) model.finish_mtp_rounds(std::span(selections.data(), selections.size()), model_eos_tokens,
        std::span(outputs.data(), outputs.size()), std::span(results.data(), results.size()), observe, publication);
    else model.finish_evaluations(std::span(outputs.data(), outputs.size()), std::span(next.data(), next.size()), observe, publication);
    double seconds = 0;
    for (const auto &call : current) seconds = (std::max)(seconds, call.step_seconds);
    if (!seconds) seconds = std::chrono::duration<double>(Clock::now() - started).count();
    observe_generation_batch(batch_observer, current, rows, seconds);
    for (auto &call : current) finish_generation_call(model, call);
    ++continuation.completed_steps;
    if (position == Position::kSuccessor || (resumed && (a_source || device_successor)))
      ++continuation.successor_publications;
    if (position == Position::kCurrent && a_source) continuation.ready.reset();
    if (device_mtp && position == Position::kCurrent) a = model.active_step.get();
  }
  void observe_cancelled(std::span<PreparedGenerationCall> current, Clock::time_point started,
      Clock::time_point retired_at) {
    const auto seconds = std::chrono::duration<double>(retired_at - started).count();
    for (auto &call : current) { call.disposition = EvaluationDisposition::kCancelled; call.step_seconds = seconds; }
    observe_generation_batch(batch_observer, current, maximum_outputs, seconds);
  }
  static void require_retired(inference::RetirementObservation observation) {
    if (!observation.retired || (!observation.status.ok() &&
        (!observation.status.IsCancelled() || observation.status.has_native_error()))) {
      inference::EvaluationFailure failure(observation.status.ok()
          ? hyle::Status::Unavailable("generation continuation still retains its original episode") : observation.status);
      failure.observe_retirement(observation); throw failure;
    }
  }
  inference::RetirementObservation retire_failure(inference::EvaluationFailure *failure) noexcept {
    b_source.reset(); a_source.reset(); continuation.reset();
    return model.retire_failed_publication(
        failure && failure->retirement_observed() && !failure->retirement().retired);
  }
  void discard_records() {
    for (const auto &call : calls) {
      auto &ledger = *call.request->ledger;
      if (ledger.committed_count < ledger.records.size()) ledger.discard_uncommitted();
    }
  }
  void recover_records(bool retired) noexcept {
    observe_generation_commits_after_failure(calls);
    if (retired && !model.active_step && !model.successor_step) {
      try { discard_records(); } catch (...) {}
    }
  }
  void cancel(bool submitted_successor) {
    if (submitted_successor) ++continuation.successor_discards;
    if (resumed) ++continuation.successor_discards;
    require_retired(retire_failure(nullptr)); discard_records();
    const auto retired_at = Clock::now();
    observe_cancelled(calls, first_start, retired_at);
    if (submitted_successor) observe_cancelled(std::span(second.data(), second.size()), second_start, retired_at);
  }
  void finish_refusal() {
    inference::RetirementObservation retired{!model.active_step && !model.successor_step, {}};
    for (const auto &call : calls) {
      retired.retired = retired.retired && !call.request->session->pending;
      inference::record_retirement_status(retired.status, call.request->session->retirement_status);
    }
    refused_failure->observe_retirement(retired);
    if (!retired.retired || !refused_failure->retirement().status.ok()) throw *refused_failure;
    // A retired; B was not submitted. Its single-use Source cannot bind twice.
    // A typed coverage miss needs ordinary ready execution of the real
    // predecessor, then the refused slot. Preserve only this ordered cohort's
    // temporary route; Source/permission/capture authority is never carried.
    discard_records(); a_source.reset();
    if (!device_mtp && refused_failure->unsubmitted_reuse_refusal()) {
      continuation.reset_reuse();
      continuation.next_slot = slot;
      continuation.reprime_slot = next_slot;
    } else continuation.reset();
  }
  bool keep_successor() const {
    return inference::continuation_retention(calls, [](const auto &call) {
      const auto &request = *call.request;
      return inference::ContinuationProgress{call.disposition == EvaluationDisposition::kPublished,
          request.progress.turn_done, request.session->stopping, request.session->failed,
          request.progress.remaining_outputs()};
    }, true, model.active_step && model.active_step->cancelled, continuation.unavailable, draining) !=
        inference::ContinuationRetention::kRetirePending;
  }
  void discard_successor() {
    ++continuation.successor_discards;
    b_source.reset(); a_source.reset(); continuation.release_ready();
    auto discarded = model.discard_successful_episode();
    if (!discarded.retired) {
      continuation.reset();
      const auto retired = model.retire_failed_publication();
      inference::record_retirement_status(discarded.status, retired.status);
      discarded.retired = retired.retired;
    }
    require_retired(discarded); discard_records();
    const auto retired_at = Clock::now();
    observe_cancelled(std::span(second.data(), second.size()), second_start, retired_at);
    for (const auto &call : calls) {
      auto &request = *call.request; request.endpoint = retired_at;
      if (!request.turn_observations.empty()) request.turn_observations[request.progress.turn_index].endpoint = retired_at;
    }
  }
  void bind_successor() {
    if (device_mtp) {
      a = model.active_step.get();
      if (!a || !a->device_input_choice || calls.size() != 1 || !calls.front().request->session->head)
        throw std::logic_error("device successor lost its producer's actual published head");
      check(model.resolve_device_mtp_successor(*a, *calls.front().request->session->head));
    }
    for (std::size_t j = 0; j < calls.size(); ++j) {
      auto &request = *calls[j].request;
      auto &record = request.ledger->bind_successor(generation_boundary(*request.session->head),
          request.progress.observed_commits(), request.progress.turn_index, request.progress.generated.back(), device_mtp);
      second[j].pending = record.pending_input;
      if (device_mtp) second[j].role = record.role;
    }
  }
  void carry(inference::StepCarry disposition) {
    if (disposition == inference::StepCarry::kUnavailable) {
      discard_records(); continuation.reset(); continuation.unavailable = true;
    } else if (disposition == inference::StepCarry::kDeferred) {
      // Only host metadata crosses pumps; the promoted active_step remains
      // the original sole native owner of the submitted successor.
      continuation.pending_calls.clear();
      for (auto &call : second) continuation.pending_calls.push_back(std::move(call));
      second.clear(); continuation.pending_refusal.reset();
      continuation.pending_synchronous = false; continuation.pending_was_successor = true;
      continuation.cohort.clear_drain();
      continuation.next_slot = device_mtp && !continuous_mtp ? 0 : (next_slot + 1) % execution_slot_count;
      if (device_mtp) { a_source.reset(); continuation.release_ready(); }
    } else if (disposition == inference::StepCarry::kSuccessor) {
      if (device_mtp) {
        a_source.reset(); b_source.reset(); continuation.reset_reuse(); return;
      }
      a_source.reset(); continuation.ready = std::move(b_source); continuation.ready_slot = next_slot;
      continuation.next_slot = (next_slot + 1) % execution_slot_count;
    } else {
      if (device_mtp) {
        a_source.reset(); continuation.reset_reuse(); discard_records(); return;
      }
      continuation.ready = std::move(a_source);
      if (imported) {
        ++continuation.ready_steps;
        if (prior_slot && (*prior_slot + 1) % execution_slot_count == slot) continuation.direction_warm[slot] = true;
        continuation.ready_slot = slot; continuation.next_slot = next_slot;
        // This ordinary target has actually completed and published using the
        // correct predecessor Source. Only now may speculative attempts resume.
        if (continuation.reprime_slot == slot && prior_slot &&
            (*prior_slot + 1) % execution_slot_count == slot)
          continuation.reprime_slot.reset();
      } else { continuation.ready_slot.reset(); continuation.next_slot = slot; }
      // Partial K1 acceptance cannot seed another full-candidate Source.
      if (mtp && !continuation.matches(calls)) continuation.release_ready();
      discard_records();
    }
  }
  void finish_reuse() {
    const auto retention = inference::continuation_retention(calls, [](const auto &call) {
      const auto &request = *call.request;
      return inference::ContinuationProgress{call.disposition == EvaluationDisposition::kPublished,
          request.progress.turn_done, request.session->stopping, request.session->failed,
          request.progress.remaining_outputs()};
    }, false, false, continuation.unavailable, draining);
    if (retention == inference::ContinuationRetention::kResetReady) continuation.reset();
    else if (retention == inference::ContinuationRetention::kReleaseReady ||
        std::ranges::any_of(calls, [](const auto &call) { return call.quota <= 1; })) continuation.release_ready();
  }
};

template<class AfterSubmit, class BatchObserver>
bool execute_generation_continuation(Model &model, std::span<PreparedGenerationCall> calls, bool trace_states,
    GenerationContinuation &continuation, AfterSubmit &after_submit, BatchObserver &batch_observer,
    bool defer_successor = false, std::span<const inference::RequestCreditGrant> credits = {},
    bool drain_requested = false) {
  const bool resumed = !continuation.pending_calls.empty();
  const bool mtp = model.cache_plan.draft_cache;
  const auto successor_role = mtp ? (model.device_mtp_successor_enabled()
      ? GraphRole::kMtpContinuation : GraphRole::kMtpCatchupRound) : GraphRole::kText;
  struct RetiredCalls final {
    Model &model; GenerationContinuation &continuation;
    int exceptions = std::uncaught_exceptions();
    ~RetiredCalls() {
      if (!model.active_step && !model.successor_step &&
          (std::uncaught_exceptions() == exceptions || model.private_publications_retired())) {
        continuation.clear_pending();
      }
    }
  } retired_calls{model, continuation};
  if (!resumed && (!continuation.enabled || continuation.unavailable || model.cache_plan.in_flight_batches < 2 ||
      (mtp && (calls.size() != 1 || !model.bounded_device_mtp_policy() ||
          (model.max_drafts()>1 && !model.device_mtp_successor_enabled()))) ||
      std::ranges::any_of(calls, [&](const auto &call) {
        const auto &request = *call.request;
        return !model.supports_device_relay(request.session->sampling_configuration().index) ||
            (mtp ? !drafts_token(call.role) || call.maximum != model.max_drafts() : call.role != GraphRole::kText) ||
            call.completion != TextCompletion::kEmitNext || call.full_hit || !call.generation ||
            (trace_states && !request.trace.empty());
      }))) {
    continuation.reset(); return false;
  }
  if (resumed && (calls.data() != continuation.pending_calls.data() ||
      calls.size() != continuation.pending_calls.size() || continuation.pending_synchronous || !model.active_step ||
      !continuation.matches_pending_successor(model.successor_step.get()) ||
      !model.active_step->submitted || model.active_step->members.size() != calls.size()))
    throw std::logic_error("generation wave lost its original pending batch");
  if (!resumed && !continuation.select(calls)) return false;
  if (!resumed && continuation.ready && !continuation.matches(calls)) {
    continuation.reset(); if (!continuation.select(calls)) return false;
  }
  // A discarded successor really closed its original Invocation. Reinstall
  // missing carriers only while idle, before admitting this new predecessor.
  if (!resumed) model.prepare_relay_execution(calls.size(), calls[0].request->sampling_index, successor_role);
  GenerationStepAdapter<AfterSubmit, BatchObserver> adapter(model, calls, trace_states, continuation,
      after_submit, batch_observer, credits, drain_requested, resumed);
  (void)inference::drive_steps(adapter, {.resume = resumed, .defer_successor = defer_successor,
      .cursor = defer_successor ? &continuation.drive_cursor : nullptr});
  return true;
}

// The synchronous member of the same step protocol: packed prefill and
// recursive target verification supply their actual geometry and mathematics,
// while drive_steps owns prepare/submit/cancel/finish and failure recovery.
// All scratch and request borrows are call-scoped; Model/Graph retain execution.
template<class AfterSubmit, class BatchObserver>
struct SynchronousGenerationStepAdapter final {
  static constexpr bool kSupportsSuccessor = true;
  using Position = inference::StepPosition;
  Model &model;
  std::span<PreparedGenerationCall> calls;
  bool trace_states, packed_prefill;
  std::size_t packed_verification_capacity;
  AfterSubmit &after_submit;
  BatchObserver &batch_observer;
  GraphRole role;
  bool generation;
  hyle::container::SmallVector<EvaluationRequest, 2> inputs;
  hyle::container::SmallVector<EvaluationOutput, 2> outputs;
  hyle::container::SmallVector<MtpGenerationSelection, 2> selections;
  hyle::container::SmallVector<MtpRoundResult, 2> results;
  hyle::container::SmallVector<std::int32_t, 2> next;
  std::size_t input_rows = 0, active_rows = 0, physical_rows = 0;
  Clock::time_point episode_start{};
  GenerationContinuation *continuation = nullptr;
  using FollowingCalls = decltype(GenerationContinuation::pending_successor_calls);
  FollowingCalls local_following;
  FollowingCalls &following;
  std::shared_ptr<StepSources> source;
  bool resumed = false;

  SynchronousGenerationStepAdapter(Model &selected, std::span<PreparedGenerationCall> original,
      bool trace, bool packed, std::size_t verification_capacity, AfterSubmit &submitted, BatchObserver &observer,
      GenerationContinuation *pending = nullptr)
      : model(selected), calls(original), trace_states(trace), packed_prefill(packed),
        packed_verification_capacity(verification_capacity), after_submit(submitted), batch_observer(observer),
        role(calls[0].role), generation(calls[0].generation), continuation(pending),
        following(pending ? pending->pending_successor_calls : local_following) {
    resumed = continuation && calls.data() == continuation->pending_calls.data() &&
        !continuation->pending_calls.empty() && model.active_step && model.active_step->submitted;
    if (resumed && !generation)
      packed_prefill = packs_prefill(model.active_step->phase->family->packing);
    if (continuation && continuation->drive_cursor.awaiting_current) {
      if (!model.active_step || !continuation->matches_pending_successor(model.successor_step.get()) || !model.active_step->submitted ||
          model.active_step->members.size() != calls.size())
        throw std::logic_error("synchronous generation lost its original pending batch");
      submitted_geometry();
    } else { following.clear(); prepare_scratch(); }
  }
  void prepare_scratch() {
    if (!inputs.try_reserve(calls.size())) throw std::bad_alloc{};
    finish_scratch();
  }
  void finish_scratch() {
    if (!outputs.try_reserve(calls.size()) || !selections.try_reserve(calls.size()) ||
        !results.try_resize(generation && role != GraphRole::kText ? calls.size() : 0) || !next.try_resize(calls.size()))
      throw std::bad_alloc{};
  }
  void attach_outputs(std::span<PreparedGenerationCall> selected) {
    for (auto &call : selected) {
      auto &request = *call.request;
      outputs.push_back(evaluation_output(model, request.session, {call.name, &request.trace, trace_states},
          &call.step_seconds, generation ? &request.endpoint : &request.prefill_endpoint, nullptr, nullptr));
      outputs.back().disposition = &call.disposition;
      if (generation && role != GraphRole::kText)
        selections.push_back({request.session, call.pending, call.quota, call.draft_log_probabilities, call.maximum});
    }
  }
  void submitted_geometry() {
    input_rows = model.active_step->input_tokens;
    active_rows = model.active_step->active_input_rows;
    physical_rows = cache::multiply(model.active_step->phase->query_rows, calls.size());
    episode_start = model.active_step->step_start;
  }
  void prepare(Position) {
    if (resumed) { submitted_geometry(); return; }
    for (auto &call : calls) {
      auto &request = *call.request;
      // Recursive proposals may have shortened since the initial metadata pass.
      if (role == GraphRole::kVerify && !packed_verification_capacity && call.ids().size() != calls[0].ids().size())
        throw std::runtime_error("MTP batch requires equal role, kind and input rows without padding");
      prepare_generation_record(*request.ledger, *request.session, call.role,
          generation ? std::span<const std::int32_t>(&call.pending, 1) : call.ids(), generation,
          call.quota, request.progress.turn_index, generation ? std::shared_ptr<const void>{} : std::shared_ptr<const void>(request.warm_inputs.at(request.progress.turn_index)), call.maximum, call.completion, call.input_begin, call.prefill_complete);
      inputs.push_back({request.session, call.ids(), &request.ledger->stop_policy(), call.logits, call.conditional_logits, request.request_key});
      if(!generation && request.image_input){
        inputs.back().image_input=request.image_input.get();inputs.back().image_offset=call.input_begin;
      }
      if (!generation && packed_prefill) {
        inputs.back().prefill_packets = &request.prefill_packets;
        inputs.back().prefill_prompt = request.prompt;
        inputs.back().prefill_offset = call.input_begin;
        inputs.back().prefill_turn = request.progress.turn_index;
      }
    }
    attach_outputs(calls);
    if (calls.size() == 1 && calls[0].verification_prepared) {
      model.bind_device_verification_inputs(role, calls[0].ids());
      return;
    }
    if (!generation && packed_prefill) model.prepare_prefill_evaluations(std::span(inputs.data(), inputs.size()), nullptr, role, calls[0].completion);
    else if (packed_verification_capacity)
      model.prepare_verification_evaluations(std::span(inputs.data(), inputs.size()), packed_verification_capacity);
    else model.prepare_evaluations(std::span(inputs.data(), inputs.size()), nullptr, role, calls[0].completion);
  }
  bool prefill_pipeline() const {
    if (generation || !packed_prefill || role != GraphRole::kText || calls.size() != 1 ||
        !model.execution_workspace || calls[0].ids().size() != model.bounded_prefill_rows ||
        (trace_states && !calls[0].request->trace.empty())) return false;
    return model.supports_prefill_relay(calls[0].request->session->sampling_configuration().index);
  }
  void submit(Position position) {
    auto &step = position == Position::kCurrent ? *model.active_step : *model.successor_step;
    if (prefill_pipeline() && step.phase->family->completion == TextCompletion::kStateOnly)
      (void)model.prepare_submission_sources(step);
    model.submit_evaluations(step);
    submitted_geometry();
    if constexpr (!std::is_null_pointer_v<std::remove_cvref_t<AfterSubmit>>) after_submit();
  }
  inference::StepSuccessor prepare_successor() {
    if (!prefill_pipeline() || calls[0].prefill_complete || !model.active_step ||
        !model.active_step->execution.sources) return inference::StepSuccessor::kNone;
    auto &request = *calls[0].request;
    if (request.session->stopping || request.session->failed ||
        request.ledger->records.size() - request.ledger->committed_count < 2)
      return inference::StepSuccessor::kNone;
    const auto begin = calls[0].input_begin + calls[0].ids().size();
    if (begin >= request.prompt.size()) return inference::StepSuccessor::kNone;
    const auto count = (std::min)(calls[0].ids().size(), request.prompt.size() - begin);
    const bool complete = begin + count == request.prompt.size();
    const auto completion = complete ? TextCompletion::kEmitNext : TextCompletion::kStateOnly;
    const auto physical = model.prefill_capacity(count, completion, request.sampling_index, 1, role);
    const auto slot = model.prefill_successor_slot(count, physical, role, completion, request.sampling_index);
    if (!slot) return inference::StepSuccessor::kNone;
    following.push_back(calls[0]);
    auto &next_call = following.back();
    next_call.input_begin = begin; next_call.prompt = request.prompt.subspan(begin, count);
    next_call.prefill_complete = complete; next_call.completion = completion;
    next_call.step_seconds = next_call.proposal_seconds = 0;
    next_call.verification_prepared = false;
    next_call.disposition = EvaluationDisposition::kNotPublished;
    next_call.name = calls[0].name.substr(0, calls[0].name.find("_chunk")) + "_chunk" + std::to_string(begin);
    source = model.active_step->execution.sources;
    const std::array input{EvaluationRequest{request.session, next_call.prompt, &request.ledger->stop_policy(),
        next_call.logits, {}, request.request_key, &request.prefill_packets, request.prompt, begin, request.progress.turn_index}};
    model.prepare_evaluations(input, nullptr, role, completion, source, *slot, physical, RowPacking::kPrefill);
    return inference::StepSuccessor::kPrepared;
  }
  bool cancelled() const { return model.active_step && model.active_step->cancelled; }
  bool current_ready() { return take(model.active_step->execution.poll()) != c::CompletionPoll::kPending; }
  void defer_current(inference::StepSuccessor) { continuation->hold_current(calls, true); }
  void cancel(bool) {
    // A single/all-member or pre-submit stop cancels the whole episode.
    // Its original transactions still require actual native retirement.
    const bool requested = std::ranges::any_of(model.active_step->members,
        [](const auto &member) { return member.session->stopping; });
    // The original episodes and bound Source still retain all native readers.
    // Drop only this adapter's extra pin before their ordinary reverse cleanup.
    source.reset();
    auto retired = model.abort_episode();
    for (const auto &call : calls) if (call.request->session->stopping)
      inference::record_retirement_status(retired.status, call.request->session->retirement_status);
    if (!requested || !retired.retired || (!retired.status.ok() &&
        (!retired.status.IsCancelled() || retired.status.has_native_error()))) {
      auto status = retired.status;
      if (status.ok()) status = retired.retired
          ? hyle::Status::FailedPrecondition("cancelled generation episode has no stopped member")
          : hyle::Status::Unavailable("cancelled generation episode is still retiring");
      inference::EvaluationFailure failure(std::move(status));
      failure.observe_retirement(retired); throw failure;
    }
    const auto seconds = std::chrono::duration<double>(Clock::now() - episode_start).count();
    double proposal_seconds = 0;
    for (auto &call : calls) {
      call.request->ledger->discard_uncommitted();
      call.draft_log_probabilities.clear(); call.logits.counts = {}; call.conditional_logits = {};
      call.disposition = EvaluationDisposition::kCancelled;
      call.step_seconds = seconds + call.proposal_seconds;
      proposal_seconds += call.proposal_seconds;
    }
    observe_generation_batch(batch_observer, calls, input_rows, seconds + proposal_seconds, active_rows, physical_rows);
  }
  void finish(Position position) {
    const auto selected_calls = position == Position::kCurrent ? this->calls : std::span(following.data(), following.size());
    submitted_geometry();
    if (outputs.empty() || position == Position::kSuccessor) {
      outputs.clear(); selections.clear(); finish_scratch(); attach_outputs(selected_calls);
    }
    bool prepared_ahead = false;
    const auto &eos = model_eos_tokens;
    GenerationPublicationBatch publication(selected_calls.size(), [&](std::size_t index) {
      return GenerationPublicationInput{selected_calls[index].request->ledger.get(), nullptr};
    });
    const auto observe = [&](std::size_t j, const CompletedEvaluationView &view) {
      const auto &call = selected_calls[j];
      observe_generation_preparation(*call.request, view, call.generation);
      observe_completed_evaluation(model, outputs[j], {call.name, &call.request->trace, trace_states}, view);
    };
    // Controls retain their original post-submit ordering. Optional CPU-only
    // work follows them, and never turns its own allocation failure into a
    // failure or publication of this already submitted episode.
    if (generation && ((selected_calls.size() > 1 && role == GraphRole::kText) || verifies_prefix(role) || has_mtp(role)) &&
        selected_calls[0].completion == TextCompletion::kEmitNext) {
      const bool continues = std::ranges::all_of(selected_calls, [](const auto &call) {
        const auto &request = *call.request;
        return call.quota > 1 && request.progress.remaining_outputs() > 1 && request.session &&
            !request.session->stopping && !request.session->failed;
      });
      if (continues) {
        (void)model.prepare_decode_ahead();
        prepared_ahead = model.decode_ahead && model.decode_ahead->ready;
      }
    }
    if (generation && role == GraphRole::kVerify) {
      model.finish_recursive_mtp(std::span(selections.data(), selections.size()), eos,
          std::span(outputs.data(), outputs.size()), std::span(results.data(), results.size()), observe, publication);
      // The original verifier and prefix decision have retired and released
      // their reads. Completed call records no longer need these private q
      // leases; retaining them would charge the next episode's bounded pool.
      for (auto &call : selected_calls) call.draft_log_probabilities.clear();
    }
    else if (generation && role != GraphRole::kText) model.finish_mtp_rounds(std::span(selections.data(), selections.size()), eos,
        std::span(outputs.data(), outputs.size()), std::span(results.data(), results.size()), observe, publication);
    else model.finish_evaluations(std::span(outputs.data(), outputs.size()), std::span(next.data(), next.size()), observe, publication);
    double target_seconds = 0, proposal_seconds = 0;
    for (std::size_t j = 0; j < selected_calls.size(); ++j) {
      target_seconds = (std::max)(target_seconds, selected_calls[j].step_seconds);
      proposal_seconds += selected_calls[j].proposal_seconds;
      selected_calls[j].step_seconds += selected_calls[j].proposal_seconds;
      if (selected_calls[j].proposal_seconds && selected_calls[j].disposition == EvaluationDisposition::kPublished) {
        auto &ledger = *selected_calls[j].request->ledger;
        ledger.records[ledger.committed_count - 1].step_seconds += selected_calls[j].proposal_seconds;
      }
    }
    // Shared target work is counted once; recursive proposals were serial
    // per member and are all retained. Cancelled members may have no output
    // endpoint, so a fully withdrawn batch uses its actual retired call end.
    if (target_seconds == 0) target_seconds = std::chrono::duration<double>(Clock::now() - episode_start).count();
    observe_generation_batch(batch_observer, selected_calls, input_rows, target_seconds + proposal_seconds, active_rows, physical_rows);
    for (auto &call : selected_calls) { finish_generation_call(model, call); call.conditional_logits = {}; }
    if (prepared_ahead)
      for (const auto &call : selected_calls)
        if (call.disposition != EvaluationDisposition::kPublished || call.request->progress.turn_done ||
            !call.request->session || call.request->session->stopping) {
          model.discard_decode_ahead(call.request->session.get());
        }
  }
  bool keep_successor() const {
    return !following.empty() && calls[0].disposition == EvaluationDisposition::kPublished &&
        !calls[0].request->session->stopping && !calls[0].request->session->failed &&
        model.active_step && !model.active_step->cancelled;
  }
  void bind_successor() {
    auto &call = following.front(); auto &request = *call.request;
    // A's real publication/observation now supplies B's authoritative before
    // boundary. B consumes known prompt input, never a guessed output token.
    prepare_generation_record(*request.ledger, *request.session, call.role, call.ids(), false,
        call.quota, request.progress.turn_index, request.warm_inputs.at(request.progress.turn_index),
        call.maximum, call.completion, call.input_begin, call.prefill_complete);
  }
  void discard_successor() {
    const auto retired = model.discard_successful_episode();
    if (!retired.retired || !retired.status.ok()) {
      inference::EvaluationFailure failure(retired.status.ok()
          ? hyle::Status::Unavailable("prefill successor has not actually retired") : retired.status);
      failure.observe_retirement(retired); throw failure;
    }
    following.clear(); source.reset();
  }
  void carry(inference::StepCarry disposition) {
    if (disposition == inference::StepCarry::kDeferred) {
      if (!continuation) throw std::logic_error("deferred prefill lost its original request continuation");
      continuation->hold_current(std::span(following.data(), following.size()), true, true);
      following.clear();
    }
  }
  void finish_refusal() {}
  void finish_reuse() {}
  inference::RetirementObservation retire_failure(inference::EvaluationFailure *failure) noexcept {
    source.reset();
    return model.retire_failed_publication(
        failure && failure->retirement_observed() && !failure->retirement().retired);
  }
  void recover_records(bool) noexcept { observe_generation_commits_after_failure(calls); }
};

// The serialized owner may process controls after the original native submit
// and before completion. Static optional dispatch adds no callback table, native
// executor or cost for the ordinary synchronous caller. A throwing control
// observer aborts through the original episode owner, retaining unretired work.
template<class AfterSubmit = std::nullptr_t, class BatchObserver = std::nullptr_t>
void execute_generation_calls(Model &model, std::span<PreparedGenerationCall> calls, bool trace_states,
    cache::PrefixCache *prefix = nullptr, std::span<const GenerationRequest> requests = {},
    AfterSubmit &&after_submit = nullptr, GenerationContinuation *continuation = nullptr,
    BatchObserver &&batch_observer = nullptr, bool defer_successor = false,
    std::span<const inference::RequestCreditGrant> credits = {}) {
  if (calls.empty()) return;
  const bool packed_prefill = generation_prefill_capacity(model, calls) != 0;
  // Validate every request's cold metadata before prefix eviction, proposal
  // work or preparing even the first uncommitted ledger slot.
  for (std::size_t member = 0; member < calls.size(); ++member) {
    const auto &call = calls[member];
    if (!call.request || !call.request->ledger)
      throw std::runtime_error("generation request requires its cold-prepared commit ledger");
    const auto &request = *call.request;
    if (!request.session || !request.session->head || request.session->stopping ||
        request.session->failed || request.session->pending)
      throw std::runtime_error("generation batch requires live idle Session heads");
    if (drafts_token(call.role) && (!request.session->head->draft ||
        request.session->head->draft->lag != draft_lag(call.role)))
      throw std::runtime_error("generation K1 member role differs from its actual draft lag");
    for (const auto &prior : calls.first(member))
      if (prior.request->session.get() == request.session.get())
        throw std::runtime_error("generation batch requires distinct Session owners");
    const auto &sampling = model.sampling_configuration(request.sampling_index);
    if (&request.session->sampling_configuration() != &sampling)
      throw std::runtime_error("request sampling cannot change after Session admission");
    request.ledger->validate_sampling(sampling, model.resources_);
    if (call.completion == TextCompletion::kEmitNext) {
      auto required = sampling.configuration;
      if (call.full_hit || !call.logits.counts.lease.valid()) required.modifiers.history = {};
      validate_generation_logits_inputs(required, call, model.vocabulary);
    }
    if (request.ledger->stop_policy() != request.stop_policy)
      throw std::runtime_error("generation ledger stop policy differs from its admitted episode");
    if ((call.full_hit && calls.size() != 1) ||
        call.request->sampling_index != calls[0].request->sampling_index ||
        (call.role != calls[0].role && !(call.generation && calls[0].generation &&
            drafts_token(call.role) && drafts_token(calls[0].role))) ||
        call.completion != calls[0].completion || call.generation != calls[0].generation ||
        (!packed_prefill && call.role != GraphRole::kVerify && call.ids().size() != calls[0].ids().size()))
      throw std::runtime_error("MTP batch requires equal role, kind and input rows without padding");
    if (!call.full_hit) {
      if (!supports_commit_completion(call.role, call.completion, call.generation, call.prefill_complete))
        throw std::runtime_error("state-only ledger entry requires an incomplete teacher-forced prefill");
      if (call.ids().empty() || (call.role == GraphRole::kVerify && !call.generation))
        throw std::runtime_error("generation call requires its actual prepared input");
      (void)request.ledger->next();
      if (!call.generation) (void)request.warm_inputs.at(request.progress.turn_index);
    }
  }
  // C0 storage is bounded by the actual executing cohort, never the number
  // of queued requests. All original request metadata above is checked first.
  for (auto &call : calls) if (!call.full_hit && call.completion == TextCompletion::kEmitNext) {
    auto &request = *call.request;
    const auto &configuration = model.sampling_configuration(request.sampling_index).configuration;
    if (!configuration.modifiers.history.empty() && !call.logits.counts.lease.valid())
      call.logits = history_logits_inputs(model, *request.ledger,
          call.generation ? std::span<const std::int32_t>(&call.pending, 1) : call.prompt,
          call.logits, request.sampling_index);
    validate_generation_logits_inputs(configuration, call, model.vocabulary);
  }
  if (calls.size() == 1 && calls[0].full_hit) {
    if (continuation) continuation->reset();
    finish_generation_call(model, calls[0]); calls[0].disposition = EvaluationDisposition::kPublished; return;
  }
  if (!make_generation_batch_room(model, prefix, calls, requests)) throw std::runtime_error("MTP batch exceeds committed head budget");
  if (continuation) {
    if (execute_generation_continuation(model, calls, trace_states, *continuation,
        after_submit, batch_observer, defer_successor, credits)) return;
    continuation->reset();
  }
  const auto &eos = model_eos_tokens;
  std::size_t packed_verification_capacity = 0;
  if (calls[0].role == GraphRole::kVerify) {
    const auto discard_proposals = [&] {
      for (auto &peer : calls) {
        if (!peer.conditional_logits.empty()) {
          const auto &first = peer.conditional_logits.rows().front();
          peer.logits.bias = first.bias; peer.logits.allowed = first.allowed;
        }
        peer.draft_log_probabilities.clear(); peer.conditional_logits = {}; peer.verification_ids.clear();
        peer.verification_prepared = false;
      }
    };
    for (auto &call : calls) {
      if (!call.generation || call.role != GraphRole::kVerify)
        throw std::runtime_error("recursive MTP batch requires generation verifiers only");
      if (call.verification_ids.empty()) {
        call.verification_prepared = false;
        const auto begin = Clock::now();
        try {
          std::optional<ConditionalLogitsProposal> dynamic;
          if (call.logits_provider) {
            dynamic.emplace(ConditionalLogitsProposal{call.logits_provider, provider_committed_outputs(*call.request), {}});
            dynamic->rows.reserve(call.maximum + 1);
            dynamic->rows.push_back({call.logits.bias, call.logits.allowed});
          }
          const auto &sampling = call.request->session->sampling_configuration().configuration;
          const bool device_packet = calls.size() == 1 && !dynamic && call.maximum == model.max_drafts() &&
              model.draft_packet_enabled(sampling) && call.request->trace.empty();
          if (device_packet) {
            call.verification_ids = model.propose_recursive(*call.request->session, call.maximum, eos,
                &call.draft_log_probabilities, &call.request->ledger->stop_policy(), call.logits, {}, nullptr,
                [&](std::span<const std::int32_t> prefix_ids, const std::shared_ptr<StepSources> &source, bool ready) {
                  if (ready) { model.finish_device_verification_preparation(); return; }
                  // The last ID is produced by this Source, never by the host
                  // placeholder. Exact row count and all target states are known.
                  std::array<std::int32_t, 4> shape{};
                  if (prefix_ids.size() >= shape.size()) throw std::logic_error("device verifier packet exceeds its bounded K3 shape");
                  std::copy(prefix_ids.begin(), prefix_ids.end(), shape.begin());
                  const std::array input{EvaluationRequest{call.request->session, std::span(shape).first(prefix_ids.size() + 1),
                      &call.request->ledger->stop_policy(), call.logits, {}, call.request->request_key,
                      nullptr, {}, 0, 0, source}};
                  model.prepare_device_verifier(input);
                  call.verification_prepared = true;
                });
          } else call.verification_ids = model.propose_recursive(*call.request->session, call.maximum, eos,
              &call.draft_log_probabilities, &call.request->ledger->stop_policy(), call.logits, {}, dynamic ? &*dynamic : nullptr);
          if (dynamic) {
            call.conditional_logits = inference::ConditionalLogitsInputs(dynamic->rows);
            call.logits.bias = {}; call.logits.allowed = {};
          }
        } catch (const inference::EvaluationFailure &failure) {
          const auto key = call.request->request_key;
          if (failure.retirement_observed() && failure.retirement().retired) discard_proposals();
          // The original one-request q validator has no target participant.
          // Preserve unknown/native/unfinished failures without reclassifying.
          if (failure.status().IsInvalidArgument() && !failure.status().has_native_error() && !failure.participant() &&
              failure.retirement_observed() && failure.retirement().retired && !failure.retirement().status.has_native_error() &&
              key.generation && key.slot != (std::numeric_limits<std::size_t>::max)())
            throw inference::RequestExecutionFailure::from_request_evaluation(failure, key);
          throw;
        } catch (...) {
          const auto observed = model.try_retire_owned_resources();
          if (observed.retired) discard_proposals();
          throw;
        }
        call.proposal_seconds = std::chrono::duration<double>(Clock::now() - begin).count();
        // Preparation already overlapped the producer and is paid in proposal
        // wall time. The remaining target stage starts here, exactly once.
        if (call.verification_prepared) model.start_device_verification_timing();
      }
    }
    // Keep exact R1/uniform graphs. Only an unequal cohort selects the cold
    // capacity whose offsets express every member's real rows.
    const auto uniform_rows = calls[0].ids().size();
    const bool unequal = std::ranges::any_of(calls, [&](const auto &call) { return call.ids().size() != uniform_rows; });
    if (unequal) packed_verification_capacity = generation_verification_capacity(model, calls);
    // Missing provider capability keeps exact grouping, never retries native failure.
    if (unequal && !packed_verification_capacity) {
      std::map<std::size_t, std::vector<std::size_t>> groups;
      for (std::size_t i = 0; i < calls.size(); ++i) groups[calls[i].ids().size()].push_back(i);
      for (const auto &[rows, members] : groups) {
        (void)rows;
        hyle::container::SmallVector<PreparedGenerationCall, 2> selected;
        if (!selected.try_reserve(members.size())) throw std::bad_alloc{};
        inference::SelectedRequestCalls<PreparedGenerationCall, 2> borrowed(calls, members, selected);
        try {
          execute_generation_calls(model, borrowed.members(), trace_states, prefix, requests, after_submit, nullptr, batch_observer);
        } catch (const inference::EvaluationFailure &failure) {
          // A child index is not an index into the original request bucket.
          // Preserve the original error and actual retirement independently.
          const auto child = failure.participant();
          inference::EvaluationFailure mapped(failure.status(), child && *child < members.size()
              ? std::optional<std::size_t>(members[*child]) : std::nullopt);
          if (failure.retirement_observed()) mapped.observe_retirement(failure.retirement());
          throw mapped;
        }
      }
      return;
    }
  }
  SynchronousGenerationStepAdapter<AfterSubmit, BatchObserver> adapter(model, calls, trace_states,
      packed_prefill, packed_verification_capacity, after_submit, batch_observer,
      defer_successor ? continuation : nullptr);
  (void)inference::drive_steps(adapter, {.defer_successor = defer_successor,
      .cursor = defer_successor && continuation ? &continuation->drive_cursor : nullptr});
}

void publish_generation_prompt(Model &model, PreparedGenerationCall &call, cache::PrefixCache *prefix) {
  if (!call.initial_prefix || call.full_hit || !call.prefill_complete ||
      call.disposition != EvaluationDisposition::kPublished) return;
  auto &request = *call.request;
  if (!request.ledger) throw std::runtime_error("generation request requires its cold-prepared commit ledger");
  for (const auto &record : std::span(request.ledger->records).first(request.ledger->committed_count)) {
    if (record.imported || record.generation || record.turn != 0) continue;
    if (record.input_begin != request.history->length)
      throw std::runtime_error("complete prompt is missing its actual committed prefill chunk");
    request.history->append(request.input, record.input_begin, record.accepted, record.after.next_id,
        record.completion == TextCompletion::kStateOnly);
  }
  if (request.history->length != request.prompt.size())
    throw std::runtime_error("committed prefill chunks do not cover the complete prompt");
  if (request.session->head->draft && request.session->head->draft->lag)
    throw std::runtime_error("only a complete lag0 MTP prompt may enter the directory");
  prefix->publish(request.input, request.session->checkpoint(), {request.history, request.history->chunks.size(), request.prompt.size()}, request.cache_sharing_domain);
  (void)model;
}

void publish_generation_completed(Model &model, GenerationRequest &request, cache::PrefixCache *prefix) {
  if (!request.ledger) throw std::runtime_error("generation request requires its cold-prepared commit ledger");
  if (!prefix || !request.progress.turn_done || !request.session || !request.session->head)
    return;
  std::vector<cache::ConsumedChunk> chunks;
  chunks.reserve(request.ledger->committed_count);
  for (std::size_t i = 0; i < request.ledger->committed_count; ++i) {
    const auto &record = request.ledger->records[i];
    if (!record.committed) throw std::runtime_error("completed checkpoint has a gap in its original committed ledger");
    chunks.push_back({record.committed_ids(), record.after.next_id, record.completion == TextCompletion::kStateOnly});
  }
  request.completed_checkpoint = cache::consumed_checkpoint(chunks, *request.session->head, model.capacity);
  prefix->publish(request.completed_checkpoint.input, request.session->checkpoint(), request.completed_checkpoint.replay, request.cache_sharing_domain);
}

void advance_generation_request(Model &model, GenerationRequest &request,
    std::size_t steps, bool trace_states, cache::PrefixCache *prefix = nullptr,
    std::span<const GenerationRequest> requests = {}) {
  std::array calls{prepare_generation_call(model, request, steps, prefix, requests)};
  execute_generation_calls(model, calls, trace_states, prefix, requests);
  publish_generation_prompt(model, calls[0], prefix);
}

inference::RetirementObservation retire_generation_request(GenerationRequest &request) {
  // Stop/observe even on error; the runtime owner returns payload/key capacity
  // only after proof, before reporting the original observation status.
  (void)request.session->stop();
  return request.session->try_retire();
}

// Explicit synchronous consumer, including component drivers. The request
// adapter below has one wave contract for both synchronous and pipelined use.
template<class Execute>
struct SynchronousGenerationWave final {
  Model &model; Execute &execute;
  std::size_t pending_count() const noexcept { return 0; }
  inference::RequestWavePendingMember pending_member(std::size_t) const noexcept { return {}; }
  inference::RequestWaveAdvance advance(inference::RequestWaveAction action, inference::RequestWave wave) const {
    if (action == inference::RequestWaveAction::kDrain)
      return {.remaining = retire(inference::RequestWaveRetirement::kObserve)};
    execute(wave.keys);
    if (model.active_step || model.successor_step)
      throw std::logic_error("synchronous request wave retained an unfinished episode");
    return {.advanced = true, .remaining = {true, {}}};
  }
  inference::RetirementObservation retire(inference::RequestWaveRetirement action) const noexcept {
    if (action == inference::RequestWaveRetirement::kAbort) return model.retire_failed_publication();
    auto retired = model.try_retire_episode();
    retired.retired = retired.retired && model.private_publications_retired();
    return retired;
  }
  inference::RequestWaveResult operator()(inference::RequestWave wave,
      std::span<std::size_t> remaining_reserved) const {
    return inference::advance_request_wave(wave, remaining_reserved, *this);
  }
};

// Native work and deferred calls stay in the original Model/continuation.
// Common wave control owns only the cohort/credit and retirement transitions.
template<class Ready, class AfterSubmit = std::nullptr_t, class BatchObserver = std::nullptr_t>
inference::RequestWaveResult advance_generation_wave(Model &model, inference::RequestWave wave,
    std::span<std::size_t> remaining_reserved, bool trace_states, GenerationContinuation &continuation,
    Ready &&execute_ready, AfterSubmit &&after_submit = nullptr, BatchObserver &&batch_observer = nullptr,
    cache::PrefixCache *prefix = nullptr) {
  struct Stage final {
    Model &model;
    GenerationContinuation &continuation;
    bool trace_states;
    Ready &execute_ready;
    AfterSubmit &after_submit;
    BatchObserver &batch_observer;
    cache::PrefixCache *prefix;

    std::size_t pending_count() const noexcept { return continuation.pending_calls.size(); }
    inference::RequestWavePendingMember pending_member(std::size_t j) const noexcept {
      const auto &call = continuation.pending_calls[j];
      const auto outputs = [](const PreparedGenerationCall &member) noexcept {
        return member.completion == TextCompletion::kStateOnly ? std::size_t{0}
            : member.generation ? (std::min)(member.quota, member.maximum + 1) : std::size_t{1};
      };
      auto reserved = outputs(call);
      if (continuation.drive_cursor.awaiting_current &&
          continuation.drive_cursor.successor == inference::StepSuccessor::kPrepared)
        reserved += outputs(continuation.pending_successor_calls[j]);
      return {call.request->request_key, reserved};
    }
    inference::RequestWaveAdvance advance(inference::RequestWaveAction action, inference::RequestWave current) {
      if (action == inference::RequestWaveAction::kResume) {
        auto calls = std::span(continuation.pending_calls.data(), continuation.pending_calls.size());
        try {
          if (continuation.pending_synchronous) {
            SynchronousGenerationStepAdapter<AfterSubmit, BatchObserver> adapter(model, calls, trace_states,
                false, 0, after_submit, batch_observer, &continuation);
            const auto result = inference::drive_steps(adapter,
                {.resume = true, .defer_successor = true, .cursor = &continuation.drive_cursor});
            if (result == inference::StepDriveResult::kCompleted) {
              for (auto &call : calls) publish_generation_prompt(model, call, prefix);
              continuation.clear_pending();
            }
          } else if (!execute_generation_continuation(model, calls, trace_states, continuation,
              after_submit, batch_observer, true, current.credits, current.drain_requested))
            throw std::logic_error("pending generation wave lost its original continuation");
        } catch (const inference::EvaluationFailure &failure) {
          throw inference::RequestExecutionFailure::from_evaluation(failure, current.keys.size(),
              [&](std::size_t j) { return current.keys[j]; });
        }
      } else if (action == inference::RequestWaveAction::kStart) execute_ready(current);
      else continuation.reset();
      const bool pending = !continuation.pending_calls.empty();
      if (pending) {
        if (!model.active_step || !continuation.matches_pending_successor(model.successor_step.get()) ||
            !model.active_step->submitted || continuation.pending_calls.size() != current.keys.size())
          throw std::logic_error("generation wave retained metadata without its original submission");
      } else if (model.active_step || model.successor_step)
        throw std::logic_error("generation wave lost the host metadata for its original submission");
      return {.advanced = action != inference::RequestWaveAction::kResume || !continuation.drive_cursor.awaiting_current,
          .remaining = {!pending, {}}};
    }
    inference::RetirementObservation retire(inference::RequestWaveRetirement action) noexcept {
      auto retired = action == inference::RequestWaveRetirement::kObserve
          ? model.try_retire_episode() : model.retire_failed_publication();
      retired.retired = retired.retired && model.private_publications_retired();
      if (retired.retired) {
        continuation.clear_pending(); continuation.reset();
      }
      return retired;
    }
  } stage{model, continuation, trace_states, execute_ready, after_submit, batch_observer, prefix};
  return inference::advance_request_wave(wave, remaining_reserved, stage);
}

// Model-specific state and execution only. RequestLoop owns FIFO, idle
// selection and turn/key transitions; no native work is owned by this adapter.
template<class ExecuteWave>
struct GenerationRequestAdapter final {
  Model &fallback_model;
  cache::PrefixCache *fallback_prefix;
  std::span<GenerationRequest> requests;
  bool trace_states;
  std::vector<std::size_t> *evictions;
  ExecuteWave &execute;
  // Optional explicit policy; the default remains committed-token recovery.
  // Host storage has independent finite image-count and actual-byte limits.
  std::shared_ptr<inference::HostStateStorage> host_storage{};
  GenerationContinuation *continuation = nullptr;

  inference::RequestProgress &progress(inference::RequestKey key) const noexcept {
    return requests[key.slot].progress;
  }
  bool resident(inference::RequestKey key) const noexcept {
    return bool(requests[key.slot].session);
  }
  void admission_preflight(inference::RequestKey key) const {
    auto &request = requests[key.slot];
    auto &model = generation_model(this->fallback_model, request);
    // Release the previous request before any restoration/eviction can need
    // its state slots; execute_wave would be too late for this boundary.
    if (continuation) continuation->admission(key, request.progress.turn_index);
    preflight_generation_sampling(model, request);
    if (!request.progress.input_open() || request.admission_start == Clock::time_point{})
      request.admission_start = Clock::now();
    if (!request.session) model.restore_mtp_decode_recordings(request.sampling_index);
  }
  void admit(inference::RequestKey key, std::span<const inference::RequestKey>) const {
    auto &request = requests[key.slot];
    auto &model = generation_model(this->fallback_model, request);
    auto *prefix = generation_prefix(this->fallback_prefix, request);
    if (!request.session && request.host_image) {
      if (request.host_image->identity.sampling_index != request.sampling_index)
        throw std::runtime_error("parked request sampling configuration changed");
      make_generation_room(model, prefix, nullptr, requests);
      request.session = model.restore_session(request.host_image);
      request.host_image.reset(); // Only successful complete import releases it.
    }
    if (!request.session)
      restore_generation_session(model, request, prefix, trace_states, requests);
    if (request.progress.turn_index && !request.session)
      throw std::runtime_error("later generation turn requires its rebuilt complete head");
    if (!request.progress.turn_started()) {
      request.prompt = request.progress.begin_turn(*request.warm_inputs[request.progress.turn_index],
          request.progress.turn_index ? std::optional<std::int32_t>(request.session->head->next_id) : std::nullopt);
      request.turn_observations[request.progress.turn_index].output_begin = request.progress.turn_output_begin;
    }
  }
  inference::RequestWaveResult execute_wave(inference::RequestWave wave,
      std::span<std::size_t> remaining_reserved) const {
    if (remaining_reserved.size() != wave.keys.size())
      throw std::logic_error("request wave reservation extent differs from its original cohort");
    return execute(wave, remaining_reserved);
  }
  hyle::Status stop(inference::RequestKey key) const noexcept {
    if (continuation) continuation->release(key);
    const auto &request = requests[key.slot];
    return request.session ? request.session->stop() : hyle::Status::Ok();
  }
  void publish_turn(inference::RequestKey key) const {
    if (continuation) continuation->release(key);
    publish_generation_completed(generation_model(fallback_model, requests[key.slot]), requests[key.slot],
        generation_prefix(fallback_prefix, requests[key.slot]));
  }
  inference::RetirementObservation retire(inference::RequestKey key, inference::RequestRetirement reason) const {
    if (continuation) continuation->release(key);
    auto &request = requests[key.slot];
    auto &model = generation_model(this->fallback_model, request);
    if (!request.session) {
      if (reason != inference::RequestRetirement::kEvicted) request.host_image.reset();
      return {true, {}};
    }
    if (reason == inference::RequestRetirement::kEvicted && host_storage && !request.host_image)
      request.host_image = model.export_session(*request.session, host_storage);
    const auto retired = retire_generation_request(request);
    if (retired.retired) {
      request.retire_draft_rows();
      request.session.reset();
      if (reason == inference::RequestRetirement::kEvicted) {
        ++request.idle_evictions;
        if (evictions) evictions->push_back(key.slot);
      }
    }
    return retired;
  }
};

// Execute only an existing configured tier. This private Session is discarded
// through ordinary publication/retirement; no real request boundary is restored.
hyle::Result<StartupPrefillWarmup> prepare_cold_prefill(Model &model, bool bounded,
    bool mtp, std::size_t sampling_index) {
  const auto &sampling = model.sampling_configuration(sampling_index).configuration;
  if (!bounded || model.execution.kind() != BackendKind::kCuda || model.execution.has_cpu_assist() ||
      model.scheduler || sampling.enabled || !sampling.modifiers.empty() || !model.layer0_probe.empty())
    return std::unexpected(hyle::Status::Unimplemented("startup prefill warmup requires bounded pure-CUDA unmodified greedy generation"));
  const auto role = mtp ? GraphRole::kMtpPrefill : GraphRole::kText;
  const auto rows = model.prefill_capacity(mtp ? 2 : 1, TextCompletion::kEmitNext, sampling_index, 1, role);
  if (!rows)
    return std::unexpected(hyle::Status::Unimplemented("startup prefill warmup has no existing configured tier"));
  if (model.active_step || model.successor_step || model.carrier_borrowed || model.snapshot_versions.live)
    throw std::logic_error("startup prefill warmup requires an idle model before request admission");
  const auto started = Clock::now();
  const std::array chunks{rows};
  std::vector<std::int32_t> placeholder(rows, 0);
  auto session = model.create_session(chunks, {}, role, cache::SamplingCursor{}, false,
      TextCompletion::kEmitNext, sampling_index);
  hyle::util::ScopeGuard retire_session{[&]() noexcept {
    session->stop().IgnoreError(); (void)session->drain();
  }};
  if (session->length() || session->head->next_id != -1 || session->head->sampling != cache::SamplingCursor{})
    throw std::logic_error("startup prefill warmup did not receive its original empty Session");
  const std::array requests{EvaluationRequest{session, placeholder}};
  model.prepare_prefill_evaluations(requests, nullptr, role);
  StartupPrefillWarmup report;
  report.target_rows = model.active_step->active_input_rows;
  report.draft_rows = model.active_step->active_draft_rows;
  report.physical_rows = model.active_step->phase->query_rows;
  if (report.target_rows != rows || report.physical_rows != rows ||
      report.draft_rows != (mtp ? rows : 0))
    throw std::logic_error("startup prefill warmup differs from its fixed configured tier");
  model.submit_evaluation(*session);
  ++report.calls;
  (void)finish_evaluation(model, *session, "startup_prefill_warmup");
  if (session->pending || session->failed || session->length() != rows ||
      session->head->sampling != cache::SamplingCursor{})
    throw std::logic_error("startup prefill warmup did not publish its original private result");
  model.validate_mtp_snapshot(*session->head);
  check(session->stop());
  const auto retired = session->drain();
  check(retired.status);
  if (!retired.retired || session->pending || session->head || model.active_step || model.successor_step ||
      model.carrier_borrowed || model.snapshot_versions.live || !model.private_publications_retired())
    throw std::runtime_error("startup prefill warmup has not actually retired");
  const auto pages_released = [](const auto &inventory) {
    return std::ranges::none_of(inventory.owners, [](auto owners) { return owners != 0; });
  };
  if (!pages_released(model.page_inventory) || !pages_released(model.draft_inventory))
    throw std::runtime_error("startup prefill warmup retained private page references");
  for (const auto &pool : model.state_pools)
    if ((pool.history.valid() && pool.history.stats().active_uses) ||
        (pool.recurrent.valid() && pool.recurrent.stats().active_uses))
      throw std::runtime_error("startup prefill warmup retained native state uses");
  if (model.hidden_pool.valid() && model.hidden_pool.stats().active_uses)
    throw std::runtime_error("startup prefill warmup retained native hidden-state uses");
  for (const auto &[shape, phase] : model.phases) {
    (void)shape;
    for (const auto &slot : phase->slots.slots())
      if (!slot.carrier.retired())
        throw std::runtime_error("startup prefill warmup retained an active original carrier");
  }
  retire_session.dismiss();
  report.retired = true;
  report.seconds = std::chrono::duration<double>(Clock::now() - started).count();
  return report;
}

// Internal, serialized ownership of one Model's bounded cold preparation.
// A failed invocation poisons this handle; it never substitutes for native
// retirement or keeps the Model/resources alive.
struct ReusableGenerationPreparation final {
  // Keep the original identity owner stable so delayed keys never alias a new
  // request when successive calls reuse the same stack address.
  inference::RequestLoop request_loop;
  const Model *model = nullptr;
  std::weak_ptr<ColdResources> resources;
  std::size_t prefill_rows = 0;
  bool mtp = false, decode_overlap = false, ready = false, busy = false, failed = false;
  std::vector<const Model *> profile_models;
  ReusableGenerationPreparation() = default;
  ReusableGenerationPreparation(const ReusableGenerationPreparation &) = delete;
  ReusableGenerationPreparation &operator=(const ReusableGenerationPreparation &) = delete;
};

// Pure admission checks shared by the model adapter and actual reusable path.
void validate_image_turn(const Model &model, const inference::GenerationTurn &turn) {
  if(turn.attachments.empty())return;
  if(!model.vision || model.cache_plan.draft_cache)
    throw std::invalid_argument("image input requires a prepared ordinary vision execution profile");
  for(const auto &part:turn.attachments)model.vision->validate(part);
  (void)image_position_plan(model.vision->geometry(),turn.prompt,turn.attachments,0,0);
}

// Inspecting supplied turns never chooses a Graph shape or submits model work.
void validate_reusable_batch(Model &model, inference::GenerationBatch batch, bool mtp,
                             std::size_t configured_prefill, std::size_t profile = 0) {
  const auto schedules = batch.sessions;
  const auto sampling_indices = batch.sampling_indices;
  const auto stop_policies = batch.stop_policies;
  if (schedules.empty() || schedules.size() > model.cache_plan.resident_sessions ||
      (!sampling_indices.empty() && sampling_indices.size() != schedules.size()))
    throw std::invalid_argument("reusable preparation requires bounded requests with prepared sampling configurations");
  if (!configured_prefill || (mtp && configured_prefill < 2))
    throw std::invalid_argument("reusable request exceeds its configured input or output bounds");
  for (const auto &schedule : schedules) {
    if (schedule.execution_profile != profile)
      throw std::invalid_argument("execution profile was not prepared by this Qwen owner");
    const auto error = inference::check_request_schedule(schedule, model.capacity,
        static_cast<std::size_t>(model.vocabulary));
    if (error) throw std::invalid_argument(inference::request_admission_message(*error));
    for(const auto &turn:schedule.turns)validate_image_turn(model,turn);
    if (schedule.max_draft_tokens && *schedule.max_draft_tokens > (mtp ? model.max_drafts() : 0))
      throw std::invalid_argument("requested draft ceiling exceeds the cold-prepared execution plan");
  }
  for (const auto selected : sampling_indices)
    if (selected >= model.sampling_configuration_count())
      throw std::invalid_argument("sampling configuration was not cold prepared");
  if (!stop_policies.empty()) {
    if (stop_policies.size() != schedules.size())
      throw std::invalid_argument("one stop policy is required per reusable request");
    for (const auto &policy : stop_policies) policy.validate(model.vocabulary);
  }
}

struct PreparedGenerationProfile final {
  Model *model;
  bool overlap;
  cache::PrefixCache *prefix = nullptr;
};

Model &prepared_profile_model(Model &fallback, std::span<const PreparedGenerationProfile> profiles,
                              std::size_t index) {
  if (profiles.empty() && index == 0) return fallback;
  if (index >= profiles.size() || !profiles[index].model)
    throw std::invalid_argument("execution profile was not cold prepared");
  return *profiles[index].model;
}

void validate_profile_batch(Model &fallback, inference::GenerationBatch batch, std::size_t prefill,
                            std::span<const PreparedGenerationProfile> profiles) {
  if (profiles.empty()) {
    validate_reusable_batch(fallback, batch, fallback.cache_plan.draft_cache, prefill);
    return;
  }
  if (batch.sessions.empty() || batch.sessions.size() > fallback.cache_plan.resident_sessions ||
      (!batch.sampling_indices.empty() && batch.sampling_indices.size() != batch.sessions.size()) ||
      (!batch.stop_policies.empty() && batch.stop_policies.size() != batch.sessions.size()))
    throw std::invalid_argument("request cohort exceeds its prepared extent");
  for (std::size_t j = 0; j < batch.sessions.size(); ++j) {
    const auto &schedule = batch.sessions[j];
    auto &selected = prepared_profile_model(fallback, profiles, schedule.execution_profile);
    validate_reusable_batch(selected, {batch.sessions.subspan(j, 1),
        batch.sampling_indices.empty() ? batch.sampling_indices : batch.sampling_indices.subspan(j, 1),
        batch.stop_policies.empty() ? batch.stop_policies : batch.stop_policies.subspan(j, 1)},
        selected.cache_plan.draft_cache, prefill, schedule.execution_profile);
  }
}

// This cold admission boundary is shared by all generation-mode template
// instantiations. The original request loop remains the slot/retirement owner.
bool admit_waiting_generation(Model &model, inference::RequestLoop &loop,
    inference::GenerationInputRef &input, inference::TokenConsumerRef &consumer,
    std::vector<GenerationRequest> &requests,
    std::vector<inference::GenerationSessionSchedule> &schedules,
    inference::RequestOutputSlots &deliveries, std::size_t prefill,
    std::size_t bound, std::size_t record_bound,
    std::size_t &active, std::size_t &total_turns, std::size_t &outputs,
    std::span<const PreparedGenerationProfile> profiles = {}, cache::PrefixCache *prefix = nullptr) {
  const auto validate = [&](const inference::GenerationAdmission &next) {
    const std::span one_schedule(&next.schedule, 1);
    const std::span one_sampling(&next.sampling_index, 1);
    const std::span one_stop(&next.stop, 1);
    validate_profile_batch(model, {one_schedule, one_sampling, one_stop}, prefill, profiles);
  };
  const auto prepare = [&](std::size_t slot, inference::GenerationAdmission next) -> inference::RequestKey & {
    if (requests[slot].session || loop.matches(requests[slot].request_key))
      throw std::logic_error("new request slot still owns a model session");
    schedules[slot] = std::move(next.schedule);
    const auto &schedule = schedules[slot];
    auto &selected = prepared_profile_model(model, profiles, schedule.execution_profile);
    const auto visible_turns = std::span(schedule.turns).first(1);
    requests[slot] = install_generation_request({.schedule = schedule, .visible_turns = visible_turns,
        .model = selected, .prepared_model = profiles.empty() ? nullptr : &selected,
        .prefix = profiles.empty() ? prefix : profiles[schedule.execution_profile].prefix,
        .output_capacity = bound, .turn_capacity = bound, .record_capacity = record_bound,
        .open_turns = true, .sampling_index = next.sampling_index, .stop = std::move(next.stop),
        .arrival = Clock::now()});
    outputs = cache::add(outputs, visible_turns.front().max_outputs);
    return requests[slot].request_key;
  };
  const auto admitted = inference::admit_waiting_requests(loop, input, consumer, deliveries, bound, validate, prepare);
  active = cache::add(active, admitted); total_turns = cache::add(total_turns, admitted);
  return admitted != 0;
}

struct GenerationProgramPlan final {
  std::span<GenerationRequest> requests;
  const inference::PrefillPolicy & prefill;
  std::size_t max_prefill_rows;
  std::size_t configured_bound;
  std::size_t decode_bound;
  std::size_t restore_bound;
  bool reuse;
  bool reuse_ready;
  bool first_turn_only;
  bool prefix_enabled;
  bool warmup_planned_shapes;
  bool decode_overlap;
};

template<GenerationMode Mode>
StartupPrefillWarmup prepare_generation_programs(Model &model, const GenerationProgramPlan &plan) {
  constexpr bool mtp = Mode == GenerationMode::kMtp;
  const auto &requests = plan.requests;
  const auto &prefill = plan.prefill;
  const auto &max_prefill_rows = plan.max_prefill_rows;
  const auto &configured_bound = plan.configured_bound;
  const auto &decode_bound = plan.decode_bound;
  const auto &restore_bound = plan.restore_bound;
  const auto &reuse = plan.reuse;
  const auto &reuse_ready = plan.reuse_ready;
  const auto &first_turn_only = plan.first_turn_only;
  const auto &prefix_enabled = plan.prefix_enabled;
  const auto &warmup_planned_shapes = plan.warmup_planned_shapes;
  const auto &decode_overlap = plan.decode_overlap;
  const auto prompt_role = [](std::size_t rows, bool lag = false) {
    if constexpr (!mtp) { (void)rows; (void)lag; return GraphRole::kText; }
    else return rows == 1 ? (lag ? GraphRole::kMtpCatchupSingle : GraphRole::kMtpSingle)
        : (lag ? GraphRole::kMtpCatchupPrefill : GraphRole::kMtpPrefill);
  };
  const std::array pair{model.max_drafts()+1};
  const std::array roles{GraphRole::kMtpRound, GraphRole::kMtpCatchupRound,
      GraphRole::kMtpSingle, GraphRole::kMtpCatchupSingle};
  StartupPrefillWarmup startup_prefill_warmup;
  if (!reuse_ready) {
  using PromptKey = std::pair<GraphRole, TextCompletion>;
  using Demands = inference::ColdDemandInventory<PromptKey>;
  Demands prompt_demand, carrier_demand;
  const auto note_prompt = [&](GraphRole role, std::size_t tokens, std::size_t member,
                               TextCompletion completion = TextCompletion::kEmitNext, bool possible = false) {
    prompt_demand.note({role, completion}, tokens, member,
        possible ? inference::ColdDemandKind::kPossible : inference::ColdDemandKind::kCounted);
  };
  const auto note_carrier = [&](GraphRole role, std::size_t rows, std::size_t member, TextCompletion completion, bool possible = false) {
    carrier_demand.note({role, completion}, rows, member,
        possible ? inference::ColdDemandKind::kPossible : inference::ColdDemandKind::kCounted);
  };
  const auto note_prefill = [&](std::size_t tokens, std::size_t member, bool lag = false) {
    for (std::size_t consumed = 0; consumed < tokens;) {
      const auto part = prefill.next(tokens, consumed);
      const auto completion = part.complete ? TextCompletion::kEmitNext : TextCompletion::kStateOnly;
      // A committed prefill consumes an old catch-up row exactly once and
      // leaves a complete draft frontier for every later input chunk.
      const auto role = prompt_role(part.rows, lag && consumed == 0);
      note_prompt(role, part.rows, member, completion);
      if (role == GraphRole::kText || is_mtp_prefill(role))
        note_carrier(role, (std::min)(tokens, max_prefill_rows), member, completion);
      consumed = part.begin + part.rows;
    }
  };
  if (first_turn_only) {
    // Fixed model/service policy, independent of even the first input length.
    // Nonterminal chunks are exactly K; only the real final chunk emits an ID.
    constexpr std::array prefill_tiers{std::size_t{16}, std::size_t{32}, std::size_t{64},
        std::size_t{128}, std::size_t{256}, std::size_t{512}};
    const auto add_capacity = [&](GraphRole role) {
      const auto lag = static_cast<std::size_t>(draft_lag(role));
      if (model.cache_plan.max_query_rows <= lag) return;
      const auto maximum = (std::min)(max_prefill_rows, model.cache_plan.max_query_rows - lag);
      if (is_mtp_prefill(role) && maximum < 2) return; // T1 keeps Single/CatchupSingle.
      std::size_t previous = 0;
      for (const auto tier : prefill_tiers) {
        const auto rows = (std::min)(tier, maximum);
        if (rows != previous) for (std::size_t member = 0; member < (reuse ? model.cache_plan.active_sequences : 1); ++member)
          note_carrier(role, rows, member, TextCompletion::kEmitNext, true);
        previous = rows;
        if (rows == maximum) break; // Include a configured non-power-of-two maximum exactly once.
      }
      if (configured_bound > max_prefill_rows && maximum == max_prefill_rows)
        for (std::size_t member = 0; member < (reuse ? model.cache_plan.active_sequences : 1); ++member)
          note_carrier(role, maximum, member, TextCompletion::kStateOnly, true);
    };
    if constexpr (mtp) {
      add_capacity(GraphRole::kMtpPrefill);
      add_capacity(GraphRole::kMtpCatchupPrefill);
    } else add_capacity(GraphRole::kText);
  } else for (std::size_t i = 0; i < requests.size(); ++i) {
    std::vector<std::size_t> suffixes{requests[i].prompt.size()};
    if (prefix_enabled) for (std::size_t j = 0; j < i; ++j) {
      const auto length = requests[j].prompt.size();
      if (length < requests[i].prompt.size() && cache::same_prefix(*requests[i].input, *requests[j].input, length))
        suffixes.push_back(requests[i].prompt.size() - length);
    }
    std::sort(suffixes.begin(), suffixes.end()); suffixes.erase(std::unique(suffixes.begin(), suffixes.end()), suffixes.end());
    for (auto tokens : suffixes) note_prefill(tokens, i);
    if (prefix_enabled && prefill.bounded()) {
      // A generated checkpoint can leave any real suffix. Keep one capacity
      // per completion/role, including the original T1 path, rather than an
      // exact graph for every possible suffix length.
      prefill.for_each_capacity(requests[i].prompt.size(), [&](std::size_t rows, bool complete) {
        const auto completion = complete ? TextCompletion::kEmitNext : TextCompletion::kStateOnly;
        const auto note_suffix = [&](bool lag) {
          // A nonempty prefix already consumed at least one target row. Its
          // catch-up carrier need not exceed the complete prompt capacity.
          const auto suffix_rows = lag ? (std::min)(rows, requests[i].prompt.size() - 1) : rows;
          if (!suffix_rows) return;
          const auto role = prompt_role(suffix_rows, lag);
          if (role == GraphRole::kText || is_mtp_prefill(role)) note_carrier(role, suffix_rows, i, completion, true);
          else note_prompt(role, 1, i, completion, true);
        };
        note_suffix(false);
        if constexpr (mtp) note_suffix(true);
      });
      if constexpr (mtp) {
        note_prompt(GraphRole::kMtpSingle, 1, i, TextCompletion::kEmitNext, true);
        if (requests[i].prompt.size() > 1)
          note_prompt(GraphRole::kMtpCatchupSingle, 1, i, TextCompletion::kEmitNext, true);
      }
    }
    for (std::size_t turn = 1; turn < requests[i].progress.turns().size(); ++turn) {
      const auto tokens = cache::add(requests[i].progress.turns()[turn].prompt.size(), 1);
      note_prefill(tokens, i);
      if constexpr (mtp) note_prefill(tokens, i, true);
    }
  }
  const auto standalone_draft_limit = [&]([[maybe_unused]] std::size_t sampling_index) {
    // In this configured R1 policy, a full K always selects the compound
    // round. Only shorter quota/context tails reach standalone proposal and
    // verification programs; the bound needs no future turn information.
    const auto maximum = model.max_drafts();
    if constexpr (mtp) if (first_turn_only && requests.size() == 1 &&
        !requests.front().logits_provider && model.cache_plan.in_flight_batches >= 3 &&
        model.greedy_draft_chain_enabled(model.sampling_configuration(sampling_index).configuration))
      return maximum - 1;
    return maximum;
  };
  // Normal rebuild preserves original committed chunks, not original proposals.
  // Every consumed verifier prefix can become a teacher-forced prefill chunk.
  prompt_demand.inventory({prompt_role(1), TextCompletion::kEmitNext}, 1);
  if constexpr (mtp) for (std::size_t rows = 2; !first_turn_only &&
      rows <= (std::min)(model.cache_plan.candidate_boundaries, model.cache_plan.max_query_rows); ++rows)
    prompt_demand.inventory({GraphRole::kMtpPrefill, TextCompletion::kEmitNext}, rows);
  // Keep the group inside this cold preparation scope so exceptional exits
  // always join its jobs before Model/backend/weight owners can retire.
  struct ColdFamilyGuard final {
    Model &model;
    ~ColdFamilyGuard() { model.cancel_cold_families(); }
  } cold_family_guard{model};
  // Ordinary and MTP families borrow the same immutable weights. Auxiliary
  // preparation stays on this caller while independent families compile.
  if (first_turn_only && decode_bound && requests.size() == 1 &&
      !requests[0].logits_provider && model.sampling_configuration_count() == 1 &&
      !model.sampling_configuration(0).configuration.enabled &&
      model.sampling_configuration(0).configuration.modifiers.empty() &&
      model.execution.kind() == BackendKind::kCuda && !model.execution.has_cpu_assist() &&
      !model.scheduler && model.cache_plan.active_sequences == 1 && model.target_kv_layers &&
      model.layer0_probe.empty() &&
      (!mtp || (model.max_drafts() >= 1 && model.max_drafts() <= kMaximumDeviceMtpDrafts)))
    (void)model.begin_cold_families();
  for (std::size_t sampling_index = 0; sampling_index < model.sampling_configuration_count(); ++sampling_index) {
    if (!reuse && std::ranges::none_of(requests, [&](const auto &request) { return request.sampling_index == sampling_index; })) continue;
    const auto matches_member = [&](std::size_t member) { return reuse || requests[member].sampling_index == sampling_index; };
    const auto policy_decoding_requests = reuse ? model.cache_plan.active_sequences : static_cast<std::size_t>(std::count_if(requests.begin(), requests.end(), [&](const auto &request) {
      return request.sampling_index == sampling_index && (first_turn_only || std::ranges::any_of(request.progress.turns(), [](const auto &turn) { return turn.max_outputs > 1; }));
    }));
    const auto dynamic_decoding_requests = static_cast<std::size_t>(std::count_if(requests.begin(), requests.end(), [&](const auto &request) {
      return request.sampling_index == sampling_index && request.logits_provider &&
          std::ranges::any_of(request.progress.turns(), [](const auto &turn) { return turn.max_outputs > 1; });
    }));
    const auto recursive_requests = model.recursive_mtp() ? policy_decoding_requests : dynamic_decoding_requests;
  for (std::size_t width = 1; width <= model.cache_plan.active_sequences; ++width) {
    std::map<PromptKey, std::map<std::size_t, std::size_t>> shapes;
    for (const auto &[key, tokens] : prompt_demand.entries()) for (const auto &[count, demand] : tokens) {
      const auto eligible = demand.eligible_count(matches_member);
      if (width == 1 || width <= eligible) shapes[key][count] = cache::add(demand.uses, width == 1 ? restore_bound : 0);
    }
    if (decode_bound && width <= policy_decoding_requests) {
      if constexpr (mtp) if (width == 1 && model.device_mtp_successor_enabled() &&
          !model.sampling_configuration(sampling_index).configuration.enabled &&
          model.sampling_configuration(sampling_index).configuration.modifiers.empty())
        shapes[{GraphRole::kMtpContinuation, TextCompletion::kEmitNext}][model.max_drafts()+1] = decode_bound;
      if constexpr (mtp) for (auto role : roles) {
        const bool compound = width == 1 && model.cache_plan.in_flight_batches >= 3 &&
            model.greedy_draft_chain_enabled(model.sampling_configuration(sampling_index).configuration);
        if (drafts_token(role) && (((model.max_drafts() != 1 || model.recursive_mtp()) && !compound) ||
            width > policy_decoding_requests - dynamic_decoding_requests)) continue;
        auto &uses = shapes[{role, TextCompletion::kEmitNext}][drafts_token(role)
            ? (compound ? model.max_drafts() + 1 : 2) : 1]; uses = cache::add(uses, decode_bound);
      }
      else {
        auto &uses = shapes[{GraphRole::kText, TextCompletion::kEmitNext}][1];
        uses = cache::add(uses, decode_bound);
      }
    }
    if (mtp && decode_bound && width <= recursive_requests)
      for (std::size_t rows = standalone_draft_limit(sampling_index) < model.max_drafts() ? 2 : 1;
          rows <= standalone_draft_limit(sampling_index) + 1; ++rows)
        shapes[{GraphRole::kVerify, TextCompletion::kEmitNext}][rows] = decode_bound;
    // Committed-token reconstruction normalizes lag while retaining each
    // chunk's completion. Strict replay instead preserves its recorded role.
    if constexpr (mtp) if (width == 1)
      for (const auto completion : {TextCompletion::kEmitNext, TextCompletion::kStateOnly})
        for (const auto role : {GraphRole::kMtpCatchupPrefill, GraphRole::kMtpCatchupSingle}) {
          const auto normalized = role == GraphRole::kMtpCatchupPrefill ? GraphRole::kMtpPrefill : GraphRole::kMtpSingle;
          if (const auto *found = prompt_demand.find({role, completion}))
            for (const auto &[tokens, demand] : *found) {
              (void)demand; shapes[{normalized, completion}].try_emplace(tokens, restore_bound);
            }
        }
    if (width == 1 && !model.sampling_configuration(sampling_index).configuration.modifiers.empty()) {
      // A modifier history contains consumed IDs and the original pending/RNG,
      // not old bias/mask bytes. Prepare its state-only reconstruction before
      // storage freezes, including every bounded accepted verifier prefix.
      for (const auto &[key, entries] : prompt_demand.entries()) {
        const auto original = key.first;
        const auto role = original == GraphRole::kMtpCatchupPrefill ? GraphRole::kMtpPrefill
            : original == GraphRole::kMtpCatchupSingle ? GraphRole::kMtpSingle : original;
        if (!supports_state_only(role)) continue;
        for (const auto &[tokens, demand] : entries) {
          (void)demand;
          shapes[{role, TextCompletion::kStateOnly}].try_emplace(tokens, restore_bound);
        }
      }
    }
    for (const auto &[key, controls] : carrier_demand.entries())
      if (width <= Demands::eligible_members(controls, matches_member)) shapes.try_emplace(key);
    std::vector<decltype(shapes)::const_iterator> ordered_shapes;
    for (auto entry = shapes.cbegin(); entry != shapes.cend(); ++entry) ordered_shapes.push_back(entry);
    if constexpr (mtp) if (first_turn_only) {
      // Submit multi-variant capacity families before singleton rounds. Their
      // independent compiles can otherwise start in a second worker wave and
      // extend ready even while most workers have already finished. The work
      // estimate uses only the configured cold inventory, never future turns.
      const auto work = [&](const auto entry) {
        const auto *controls = carrier_demand.find(entry->first);
        const auto variants = entry->second.size() + (controls ? controls->size() : 0);
        const auto stages = drafts_token(entry->first.first) ? model.max_drafts() : std::size_t{1};
        return (std::max)(variants, stages);
      };
      std::stable_sort(ordered_shapes.begin(), ordered_shapes.end(), [&](const auto a, const auto b) {
        return work(a) > work(b);
      });
    }
    for (const auto entry : ordered_shapes) {
      const auto &[key, entries] = *entry;
      const auto [role, completion] = key;
      std::vector<std::size_t> chunks; chunks.reserve(entries.size());
      for (const auto &[tokens, uses] : entries) { (void)uses; chunks.push_back(tokens); }
      bool packed = false;
      // Known whole turns of one request already have exact cold variants.
      // Length-aware carriers are needed for batching, bounded chunks or an
      // unknown prefix suffix; do not replace uniform math merely because the
      // more general carrier is available.
      const bool variable_prefill = first_turn_only || width > 1 || prefill.bounded() || prefix_enabled;
      if (variable_prefill && (role == GraphRole::kText || is_mtp_prefill(role)) && model.layer0_probe.empty()) {
        std::vector<std::size_t> capacities;
        if (const auto *controls = carrier_demand.find(key))
          Demands::visit_support(*controls, matches_member,
              [&](std::size_t rows, const Demands::Entry &, std::size_t eligible) {
                const auto physical = is_mtp_prefill(role) ? cache::add(rows, draft_lag(role)) : rows;
                if (width <= eligible && physical <= model.cache_plan.max_query_rows / width)
                  capacities.push_back(rows);
              });
        if (width == 1 && completion == TextCompletion::kStateOnly &&
            !model.sampling_configuration(sampling_index).configuration.modifiers.empty()) {
          // A dynamic prefix suffix may have used any real length within its
          // original EmitNext carrier. Rebuild can reuse that cold capacity
          // without retaining historical modifier values or admitting hot L.
          for (const auto &[source_key, source_controls] : carrier_demand.entries()) {
            const auto source_role = source_key.first == GraphRole::kMtpCatchupPrefill
                ? GraphRole::kMtpPrefill : source_key.first;
            if (source_role != role) continue;
            for (const auto &[rows, demand] : source_controls)
              if (rows <= model.cache_plan.max_query_rows && demand.supports(matches_member))
                capacities.push_back(rows);
          }
          std::sort(capacities.begin(), capacities.end());
          capacities.erase(std::unique(capacities.begin(), capacities.end()), capacities.end());
        }
        if (!capacities.empty()) packed = model.prepare_prefill_family(capacities, completion, sampling_index, width, role);
        if (first_turn_only && !packed) {
          // A wider packed shape can be unavailable even though its original
          // single-request carrier is supported. Dispatch then splits the wave.
          if (reuse && width > 1) continue;
          throw inference::EvaluationFailure(hyle::Status::Unimplemented("configured bounded prefill has no original packed implementation"));
        }
        if (packed) {
          for (const auto rows : capacities) {
            const auto uses = first_turn_only ? configured_bound : carrier_demand.uses(key, rows);
            model.plan_replay_uses(rows, width, cache::add(uses, width == 1 ? restore_bound : 0),
                role, completion, sampling_index, RowPacking::kPrefill);
          }
          // Decode still has its original T1 mathematics and source ports.
          if (role == GraphRole::kText && completion == TextCompletion::kEmitNext && decode_bound && width <= policy_decoding_requests) {
            const std::array decode{std::size_t{1}};
            model.prepare_family(decode, false, width, role, false, completion, sampling_index);
            model.plan_replay_uses(1, width, cache::add(decode_bound, width == 1 ? restore_bound : 0), role, completion, sampling_index);
          }
        }
      }
      if (!packed && !chunks.empty()) {
        model.prepare_family(chunks, false, width, role, false, completion, sampling_index);
        for (const auto &[tokens, uses] : entries) model.plan_replay_uses(tokens, width,
            cache::add(uses, width == 1 ? restore_bound : 0), role, completion, sampling_index);
      }
    }
    if constexpr (mtp) if (decode_bound && width > 1 && width <= policy_decoding_requests &&
        width <= recursive_requests && model.target_kv_layers > 0) {
      const auto capacity = static_cast<std::size_t>(model.cache_plan.candidate_boundaries);
      const std::array capacities{capacity};
      if (model.prepare_family(capacities, false, width, GraphRole::kVerify, false, TextCompletion::kEmitNext,
          sampling_index, RowPacking::kVerify))
        model.plan_replay_uses(capacity, width, decode_bound, GraphRole::kVerify,
            TextCompletion::kEmitNext, sampling_index, RowPacking::kVerify);
    }
    if constexpr (mtp) if (decode_bound && width > 1 && width <= policy_decoding_requests &&
        model.max_drafts() == 1 && !model.recursive_mtp() && width <= policy_decoding_requests - dynamic_decoding_requests) {
      constexpr auto role = GraphRole::kMtpCatchupRound;
      if (model.prepare_family(pair, false, width, role, false, TextCompletion::kEmitNext,
          sampling_index, RowPacking::kDraft))
        model.plan_replay_uses(2, width, decode_bound, role, TextCompletion::kEmitNext, sampling_index, RowPacking::kDraft);
    }
  }
  }
  // Family jobs borrow only immutable model/cache facts and return private
  // plans. The caller can build its independent tail programs before joining;
  // it alone merges plans and mutates auxiliary carriers. ColdFamilyGuard joins
  // every worker before these owners retire on either success or failure.
  if (mtp && decode_bound)
    for (std::size_t index = 0; index < model.sampling_configuration_count(); ++index)
      for (const bool dynamic : {false, true})
        if ((dynamic || model.recursive_mtp()) && ((reuse && !dynamic) || std::ranges::any_of(requests, [index, dynamic, first_turn_only](const auto &request) {
            return request.sampling_index == index && bool(request.logits_provider) == dynamic &&
                (first_turn_only || std::ranges::any_of(request.progress.turns(), [](const auto &turn) { return turn.max_outputs > 1; }));
          }))) model.prepare_recursive_round_programs(index, dynamic, decode_bound,
              dynamic ? model.max_drafts() : standalone_draft_limit(index));
  model.finish_cold_families();
  model.freeze_shape_plan();
  if (first_turn_only) model.prepare_shared_execution_workspace();
  if (warmup_planned_shapes) {
    const auto begin = Clock::now();
    const auto shapes = model.prepare_planned_execution();
    std::cerr << "shape_warmup policy=planned shapes=" << shapes
              << " seconds=" << std::chrono::duration<double>(Clock::now() - begin).count() << '\n';
  }
  }
  const auto first_part = prefill.next(requests.front().prompt.size());
  const std::array first_prefill{first_part.rows};
  const auto first_completion = first_part.complete ? TextCompletion::kEmitNext : TextCompletion::kStateOnly;
  const auto first_role = prompt_role(first_prefill[0]);
  const auto first_capacity = model.prefill_capacity(first_prefill[0], first_completion, requests.front().sampling_index, 1, first_role);
  if (first_capacity) {
    const std::array carrier{first_capacity};
    model.prepare_prefill_execution(carrier, first_completion, requests.front().sampling_index, 1, first_role);
  } else model.prepare_execution(first_prefill, 1, first_role, first_completion, requests.front().sampling_index);
  if constexpr (!mtp) if (decode_bound) {
    const std::array decode{std::size_t{1}};
    for (std::size_t i = 0; i < model.sampling_configuration_count(); ++i)
      if (std::ranges::any_of(requests, [i, first_turn_only](const auto &request) { return request.sampling_index == i &&
          (first_turn_only || std::ranges::any_of(request.progress.turns(), [](const auto &turn) { return turn.max_outputs > 1; })); }))
      {
        model.prepare_execution(decode, 1, GraphRole::kText, TextCompletion::kEmitNext, i);
        if (decode_overlap && model.cache_plan.in_flight_batches >= 2 &&
            model.supports_device_relay(i)) {
          const auto eligible = static_cast<std::size_t>(std::count_if(requests.begin(), requests.end(), [i, first_turn_only](const auto &request) {
            return request.sampling_index == i && (first_turn_only || std::ranges::any_of(request.progress.turns(),
                [](const auto &turn) { return turn.max_outputs > 1; }));
          }));
          for (std::size_t width = 1; width <= (std::min)(eligible, model.cache_plan.active_sequences); ++width) {
            if (width != 1) model.prepare_execution(decode, width, GraphRole::kText, TextCompletion::kEmitNext, i);
            model.prepare_relay_execution(width, i);
          }
        }
      }
  }
  if (mtp && decode_bound && model.max_drafts() == 1 && !model.recursive_mtp())
    for (std::size_t index = 0; index < model.sampling_configuration_count(); ++index)
      if (std::ranges::any_of(requests, [index, first_turn_only](const auto &request) {
          return request.sampling_index == index && !request.logits_provider &&
              (first_turn_only || std::ranges::any_of(request.progress.turns(), [](const auto &turn) { return turn.max_outputs > 1; }));
        })) for (const auto role : roles) if (drafts_token(role))
          model.prepare_execution(pair, 1, role, TextCompletion::kEmitNext, index);
  if constexpr (mtp) if (decode_overlap && decode_bound && model.cache_plan.in_flight_batches >= 2 &&
      model.bounded_device_mtp_policy() && (model.max_drafts()==1 || model.device_mtp_successor_enabled()))
    for (std::size_t i = 0; i < model.sampling_configuration_count(); ++i) {
      if (!model.supports_device_relay(i) ||
          std::ranges::none_of(requests, [i](const auto &request) { return request.sampling_index == i; })) continue;
      for (const auto role : {GraphRole::kMtpRound, GraphRole::kMtpCatchupRound})
        model.prepare_execution(pair, 1, role, TextCompletion::kEmitNext, i);
      model.prepare_relay_execution(1, i, model.device_mtp_successor_enabled()
          ? GraphRole::kMtpContinuation : GraphRole::kMtpCatchupRound);
    }
  if (!reuse_ready) {
  auto warmed = prepare_cold_prefill(model, first_turn_only && !prefix_enabled, mtp, requests.front().sampling_index);
  if (warmed) startup_prefill_warmup = *warmed;
  else {
    if (!warmed.error().IsUnimplemented()) check(warmed.error());
    std::cerr << "startup_prefill_warmup skipped=unimplemented reason=" << warmed.error().message() << '\n';
  }
  if constexpr (!mtp) if (first_turn_only)
    (void)model.prepare_prefill_recordings(requests.front().sampling_index);
  }
  return startup_prefill_warmup;
}

template<GenerationMode Mode>
void prepare_generation_recordings(Model &model, const GenerationProgramPlan &plan) {
  if (!plan.reuse_ready && plan.decode_overlap && !plan.requests.empty() && plan.decode_bound && model.cache_plan.active_sequences == 1 &&
      (plan.requests.size() == 1 || model.cache_plan.resident_sessions == 1) &&
      (plan.first_turn_only || std::ranges::any_of(plan.requests.front().progress.turns(), [](const auto &turn) { return turn.max_outputs > 1; }))) {
    auto *prepared = plan.prefix_enabled ? nullptr : &plan.requests.front().session;
    if constexpr (Mode == GenerationMode::kMtp) (void)model.prepare_mtp_decode_recordings(plan.requests.front().sampling_index, prepared);
    else (void)model.prepare_ordinary_decode_recordings(plan.requests.front().sampling_index, prepared);
  }
}

struct GenerationExecutionOptions final {
  std::size_t prefill_rows = SIZE_MAX;
  bool decode_overlap = false;
  ReusableGenerationPreparation *preparation = nullptr;
  std::span<const PreparedGenerationProfile> profiles{};
  cache::PrefixCache *prefix = nullptr;
};

template<GenerationMode Mode, bool Diagnostic = false, class BatchObserver = std::nullptr_t, class OutputConsumer = std::nullptr_t>
int execute_generation_sessions(Model &model, inference::GenerationBatch batch, GenerationExecutionOptions options,
    OutputConsumer &&output_consumer = nullptr, BatchObserver &&batch_observer = nullptr,
    const GenerationDiagnostics *diagnostics = nullptr) {
  if constexpr (Diagnostic) if (!diagnostics) throw std::invalid_argument("diagnostic invocation requires its observer configuration");
  auto schedules = batch.sessions;
  const auto sampling_indices = batch.sampling_indices;
  const auto stop_policies = batch.stop_policies;
  auto max_prefill_rows = options.prefill_rows;
  const bool decode_overlap = options.decode_overlap;
  auto *reuse = options.preparation;
  auto *live_input = batch.input;
  const std::filesystem::path trace = Diagnostic ? diagnostics->trace : std::filesystem::path{};
  const bool trace_states = Diagnostic && diagnostics->trace_states;
  const auto startup = Diagnostic ? diagnostics->startup : Clock::now();
  const auto constructed = Diagnostic ? diagnostics->constructed : startup;
  const bool prefix_enabled = Diagnostic ? diagnostics->prefix_enabled : options.prefix ||
      std::ranges::any_of(options.profiles, [](const auto &profile) { return profile.prefix != nullptr; });
  const bool warmup_planned_shapes = Diagnostic && diagnostics->warmup_planned_shapes;
  const bool first_turn_only = bool(reuse) || (Diagnostic && diagnostics->first_turn_only);
  const auto token_plans = Diagnostic ? diagnostics->token_plans : std::span<const cache::TokenPlanRef>{};
  const auto logits_providers = Diagnostic ? diagnostics->logits_providers : std::span<const inference::LogitsInputProvider>{};
  constexpr bool mtp = Mode == GenerationMode::kMtp;
  constexpr bool mixed = Mode == GenerationMode::kMixed;
  const auto selected_model = [&](std::size_t index) -> Model & {
    if constexpr (mixed) return prepared_profile_model(model, options.profiles, index);
    else { (void)index; return model; }
  };
  const auto for_each_model = [&](auto &&visit) {
    if constexpr (mixed) { for (const auto &profile : options.profiles) visit(*profile.model); }
    else visit(model);
  };
  auto local_prefix = Diagnostic && prefix_enabled ? std::make_unique<cache::PrefixCache>(model.cache_plan.resident_sessions) : nullptr;
  auto *prefix = Diagnostic ? local_prefix.get() : options.prefix;
  const auto selected_prefix = [&](std::size_t index) -> cache::PrefixCache * {
    if constexpr (mixed) return options.profiles[index].prefix;
    else { (void)index; return prefix; }
  };
  const auto cached_versions = [&](const Model &selected) {
    if constexpr (mixed) {
      for (const auto &profile : options.profiles) if (profile.model == &selected)
        return profile.prefix ? profile.prefix->retained_versions() : std::size_t{0};
      throw std::logic_error("model lost its prepared prefix owner");
    } else { (void)selected; return prefix ? prefix->retained_versions() : std::size_t{0}; }
  };
  if (std::ranges::any_of(schedules, [](const auto &schedule) { return schedule.accepts_live_input(); }) &&
      (!first_turn_only || !reuse || !live_input))
    throw inference::RequestRejected("live input requires a reusable request owner and input source");
  inference::validate_turn_delivery(std::ranges::any_of(schedules,
      [](const auto &schedule) { return schedule.reports_turn_ends(); }), output_consumer);
  const auto configured_prefill = (std::min)({max_prefill_rows, std::size_t{512}, model.capacity});
  const bool reuse_ready = reuse && reuse->ready;
  if (reuse) {
    if (!first_turn_only || warmup_planned_shapes || !logits_providers.empty() ||
        !trace.empty() || trace_states)
      throw std::invalid_argument("reusable preparation requires bounded requests with a prepared sampling configuration");
    if constexpr (mixed) validate_profile_batch(model, {schedules, sampling_indices, stop_policies}, configured_prefill, options.profiles);
    else validate_reusable_batch(model, {schedules, sampling_indices, stop_policies}, mtp, configured_prefill);
    if (reuse->busy || reuse->failed) throw std::logic_error("reusable preparation is busy or failed");
    if (reuse_ready && (reuse->model != &model || reuse->resources.lock() != model.resources_ ||
        reuse->mtp != mtp || reuse->decode_overlap != decode_overlap || reuse->prefill_rows != configured_prefill))
      throw std::invalid_argument("reusable preparation differs from its original model or cold configuration");
    for_each_model([&](Model &selected) {
      if (selected.active_step || selected.successor_step || selected.carrier_borrowed ||
          selected.snapshot_versions.live != cached_versions(selected) || selected.resources_->pending_readbacks())
        throw std::logic_error("reusable preparation requires actual retirement of the preceding request");
    });
    if constexpr (mixed) {
      if (reuse_ready && (reuse->profile_models.size() != options.profiles.size() ||
          !std::equal(reuse->profile_models.begin(), reuse->profile_models.end(), options.profiles.begin(),
              [](const Model *expected, const auto &profile) { return expected == profile.model; })))
        throw std::invalid_argument("prepared execution profile identities changed");
      if (!reuse_ready) for (const auto &profile : options.profiles) reuse->profile_models.push_back(profile.model);
    }
    if (!reuse_ready) {
      reuse->model = &model; reuse->resources = model.resources_; reuse->mtp = mtp;
      reuse->decode_overlap = decode_overlap; reuse->prefill_rows = configured_prefill;
    }
    reuse->busy = true;
  }
  bool reuse_success = false;
  hyle::util::ScopeGuard release_reuse{[&]() noexcept {
    if (reuse) { reuse->busy = false; if (!reuse_success) reuse->failed = true; }
  }};
  // Model counters persist across requests; only the packed-prefill and
  // aggregate call totals below are converted to this request's interval.
  const std::array packed_before{reuse_ready ? model.packed_prefill_calls : 0,
      reuse_ready ? model.packed_prefill_active_rows : 0,
      reuse_ready ? model.packed_prefill_physical_rows : 0};
  const auto frontier_before = reuse_ready ? model.frontier_sampling_graph_calls : 0;
  const auto draft_before = reuse_ready ? model.draft_program_graph_calls : 0;
  const auto prefix_before = reuse_ready ? model.prefix_decision_graph_calls : 0;
  const auto frontier_seconds_before = reuse_ready ? model.frontier_sampling_seconds : 0.;
  const auto cold_prepare_begin = Clock::now();
  std::cerr << "cold_prepare_begin\n";
  if (first_turn_only && ((prefix_enabled && !reuse) || warmup_planned_shapes ||
      (!reuse && (schedules.size() != 1 || model.cache_plan.resident_sessions != 1 || model.cache_plan.active_sequences != 1)) ||
      model.cache_plan.max_query_rows / model.cache_plan.active_sequences < (reuse ? configured_prefill : model.capacity)))
    throw std::runtime_error("bounded natural generation requires configured query capacity, no prefix cache and lazy shape preparation");
  const auto configured_bound = model.capacity;
  const bool dynamic_admission = reuse && live_input;
  const auto initial_request_count = schedules.size();
  // Keep borrowed turn spans stable even when a retired slot is reused.
  std::vector<inference::GenerationSessionSchedule> slot_schedules;
  if (dynamic_admission) {
    slot_schedules.reserve(model.cache_plan.resident_sessions);
    for (const auto &schedule : schedules) slot_schedules.push_back(schedule);
    schedules = std::span(slot_schedules);
  }
  if (first_turn_only) {
    if (model.active_step || model.successor_step || model.carrier_borrowed ||
        (model.execution_workspace.valid() && !reuse_ready))
      throw std::logic_error("bounded prefill policy must be configured before original execution");
    max_prefill_rows = (std::min)({max_prefill_rows, std::size_t{512}, configured_bound});
    if constexpr (mtp) if (max_prefill_rows < 2)
      throw inference::EvaluationFailure(hyle::Status::Unimplemented("bounded MTP prefill requires a configured multirow carrier"));
    for_each_model([&](Model &selected) { selected.bounded_prefill_rows = max_prefill_rows; });
  }
  const auto configured_record_bound = first_turn_only ? cache::add(cache::multiply(configured_bound, 2), 2) : 0;
  if (first_turn_only)
    std::cerr << "cold_plan policy=configured-capacity visible_turns=1 configured_capacity=" << configured_bound
              << " configured_sessions=" << model.cache_plan.resident_sessions
              << " configured_active_sequences=" << model.cache_plan.active_sequences << " query_capacity=" << model.cache_plan.max_query_rows
              << " configured_prefill_rows=" << max_prefill_rows << " future_shape_hints=0\n";
  if constexpr (mixed) for (const auto &profile : options.profiles)
    profile.model->prepare_default_device_mtp_successor(profile.model->cache_plan.draft_cache && profile.overlap && !profile.prefix);
  else model.prepare_default_device_mtp_successor(mtp && decode_overlap && !prefix_enabled && logits_providers.empty());
  if (!max_prefill_rows) throw std::runtime_error("prefill rows require a positive quantum");
  const inference::PrefillPolicy prefill(max_prefill_rows);
  if (!stop_policies.empty() && stop_policies.size() != schedules.size())
    throw std::runtime_error("one immutable stop policy is required per request");
  for (const auto &policy : stop_policies) {
    policy.validate(model.vocabulary);
  }
  if (!sampling_indices.empty() && sampling_indices.size() != schedules.size())
    throw std::runtime_error("one cold sampling choice is required per request");
  for (const auto index : sampling_indices) {
    (void)model.sampling_configuration(index);
  }
  if (!logits_providers.empty() && logits_providers.size() != schedules.size())
    throw std::runtime_error("one optional logits provider is required per request");
  for (std::size_t i = 0; i < logits_providers.size(); ++i) if (logits_providers[i]) {
    const auto &policy = model.sampling_configuration(sampling_indices.empty() ? 0 : sampling_indices[i]).configuration.modifiers;
    if (!policy.bias && !policy.allowed)
      throw std::runtime_error("dynamic logits require configured bias or mask ports");
  }
  if (schedules.empty() || (!mixed && model.cache_plan.draft_cache != mtp))
    throw std::runtime_error("natural generation mode differs from its complete model/cache plan");
  if (Diagnostic && prefix_enabled && token_plans.size() != schedules.size())
    throw std::runtime_error("natural generation requires one validated prefix input per request");
  std::vector<GenerationRequest> requests;
  requests.reserve(dynamic_admission ? model.cache_plan.resident_sessions : schedules.size());
  std::size_t total_turns = 0, requested_outputs = 0, decode_bound = 0, restore_bound = 0, dispatch_bound = 0;
  for (const auto &schedule : schedules) {
    auto &request_model = selected_model(schedule.execution_profile);
    if (schedule.turns.empty()) throw std::runtime_error("natural request requires a nonempty turn list");
    if (schedule.turns.front().prompt.empty())
      throw std::runtime_error("natural request requires a nonempty initial prompt");
    const auto visible_turns = first_turn_only ? std::span(schedule.turns).first(1) : std::span(schedule.turns);
    std::size_t outputs = 0;
    for (const auto &turn : visible_turns) {
      if (!turn.max_outputs)
        throw std::runtime_error("natural turns require positive output quota");
      outputs = cache::add(outputs, turn.max_outputs);
    }
    const auto &first = schedule.turns.front().prompt;
    const auto ledger_capacity = first_turn_only ? configured_record_bound
        : generation_record_capacity(visible_turns, max_prefill_rows, first.size());
    requests.push_back(install_generation_request({.schedule = schedule, .visible_turns = visible_turns,
        .model = request_model, .prepared_model = mixed ? &request_model : nullptr,
        .prefix = selected_prefix(schedule.execution_profile), .output_capacity = first_turn_only ? configured_bound : outputs,
        .turn_capacity = (first_turn_only || schedule.accepts_live_input()) ? configured_bound : visible_turns.size(),
        .record_capacity = ledger_capacity, .open_turns = first_turn_only,
        .sampling_index = sampling_indices.empty() ? 0 : sampling_indices[requests.size()],
        .stop = stop_policies.empty() ? inference::RequestStopPolicy{} : stop_policies[requests.size()],
        .trace = trace.empty() ? trace : trace / ("session" + std::to_string(requests.size()) + "_" + schedule.id),
        .arrival = startup, .logits_provider = logits_providers.empty() ? inference::LogitsInputProvider{} : logits_providers[requests.size()],
        .token_plan = Diagnostic && prefix_enabled ? token_plans[requests.size()] : cache::TokenPlanRef{}}));
    dispatch_bound = cache::add(dispatch_bound, ledger_capacity);
    total_turns = cache::add(total_turns, visible_turns.size());
    requested_outputs = cache::add(requested_outputs, outputs);
    decode_bound = cache::add(decode_bound, first_turn_only ? configured_bound : outputs - visible_turns.size());
    if (model.cache_plan.resident_sessions < schedules.size())
      restore_bound = cache::add(restore_bound, cache::multiply(schedule.turns.size() - 1, ledger_capacity));
  }
  const GenerationProgramPlan program_plan{
      requests, prefill, max_prefill_rows, configured_bound, decode_bound, restore_bound,
      bool(reuse), reuse_ready, first_turn_only, prefix_enabled, warmup_planned_shapes, decode_overlap};
  StartupPrefillWarmup startup_prefill_warmup;
  if constexpr (mixed) {
    for (const auto &profile : options.profiles) {
      // Reusable preparation depends on configured bounds, never future turns.
      // Include every declared sampler so later requests select prepared work.
      std::vector<GenerationRequest> warm;
      for (std::size_t j = 0; j < profile.model->sampling_configuration_count(); ++j)
        warm.emplace_back("", requests.front().prompt, std::filesystem::path{}, configured_bound, j);
      auto plan = program_plan;
      plan.requests = warm; plan.decode_overlap = profile.overlap; plan.prefix_enabled = bool(profile.prefix);
      if (profile.model->cache_plan.draft_cache) {
        (void)prepare_generation_programs<GenerationMode::kMtp>(*profile.model, plan);
        prepare_generation_recordings<GenerationMode::kMtp>(*profile.model, plan);
      } else {
        (void)prepare_generation_programs<GenerationMode::kText>(*profile.model, plan);
        prepare_generation_recordings<GenerationMode::kText>(*profile.model, plan);
      }
    }
  } else startup_prefill_warmup = prepare_generation_programs<Mode>(model, program_plan);
  const std::array packed_prefill_offset{cache::add(packed_before[0], startup_prefill_warmup.calls),
      cache::add(packed_before[1], startup_prefill_warmup.target_rows),
      cache::add(packed_before[2], startup_prefill_warmup.physical_rows)};
  std::cerr << "startup_prefill_warmup calls=" << startup_prefill_warmup.calls
            << " target_rows=" << startup_prefill_warmup.target_rows
            << " draft_rows=" << startup_prefill_warmup.draft_rows
            << " physical_rows=" << startup_prefill_warmup.physical_rows
            << " retired=" << startup_prefill_warmup.retired
            << " seconds=" << startup_prefill_warmup.seconds << '\n';
  inference::RequestLoop transient_request_loop;
  auto &request_loop = reuse ? reuse->request_loop : transient_request_loop;
  if constexpr (!mixed) prepare_generation_recordings<Mode>(model, program_plan);
  if (dynamic_admission) {
    while (requests.size() < model.cache_plan.resident_sessions) {
      requests.emplace_back("", std::span<const std::int32_t>{}, std::filesystem::path{}, 1);
      slot_schedules.emplace_back();
    }
  }
  request_loop.prepare(requests.size(), model.cache_plan.resident_sessions);
  using OutputDeliveries = std::conditional_t<std::is_null_pointer_v<std::remove_cvref_t<OutputConsumer>>,
      std::nullptr_t, inference::RequestOutputSlots>;
  OutputDeliveries output_deliveries{};
  if constexpr (!std::is_null_pointer_v<std::remove_cvref_t<OutputConsumer>>) {
    static_assert(inference::TokenConsumer<std::remove_reference_t<OutputConsumer>>);
    output_deliveries.resize(requests.size());
    for (std::size_t index = 0; index < initial_request_count; ++index)
      output_deliveries.prepare(index, output_consumer.max_pending(),
          schedules[index].reports_turn_ends() ? (schedules[index].accepts_live_input() ? configured_bound : schedules[index].turns.size()) : 0);
  }
  for (std::size_t index = 0; index < initial_request_count; ++index) {
    if constexpr (std::is_null_pointer_v<std::remove_cvref_t<OutputConsumer>>)
      requests[index].request_key = request_loop.enqueue(index);
    else {
      output_deliveries.start(request_loop, index, requests[index].request_key, output_consumer);
    }
  }
  std::vector<std::size_t> dispatch, dispatch_turns;
  if constexpr (Diagnostic) { dispatch.reserve(dispatch_bound); dispatch_turns.reserve(dispatch_bound); }
  std::vector<std::size_t> evictions; evictions.reserve(first_turn_only ? cache::multiply(configured_bound, requests.size()) : total_turns);
  std::size_t completed = 0, admitted = 0, active_requests = initial_request_count;
  using BatchObservation = GenerationBatchObservation;
  std::vector<BatchObservation> batches;
  if constexpr (Diagnostic) batches.reserve(dispatch_bound);
  std::size_t actual_prefill_calls = 0, actual_decode_calls = 0;
  std::size_t actual_fresh_participations = 0, actual_fresh_target_rows = 0;
  double actual_step_sum = 0;
  std::size_t peak_resident = 0, peak_committed = 0;
  const auto resident_count = [&] {
    return static_cast<std::size_t>(std::count_if(requests.begin(), requests.end(), [](const auto &request) { return bool(request.session); }));
  };
  inference::RequestBatchWorkspace<PreparedGenerationCall> batch_workspace;
  batch_workspace.prepare(model.cache_plan.resident_sessions, model.cache_plan.active_sequences);
  GenerationContinuation continuation; continuation.enabled = decode_overlap;
  continuation.prepare(model.cache_plan.active_sequences);
  const auto submitted_draft_begin = model.submitted_draft_rows;
  const auto ready = Clock::now();
  const auto cold_prepare_seconds = std::chrono::duration<double>(ready - cold_prepare_begin).count();
  std::cerr << "cold_prepare_end seconds=" << cold_prepare_seconds << '\n'
            << "generation_shape_prepare_seconds=" << std::chrono::duration<double>(ready - constructed).count()
            << " startup_to_ready_seconds=" << std::chrono::duration<double>(ready - startup).count() << '\n'
            << "measure_memory name=ready_before_first\n";
  const auto start = Clock::now(); auto endpoint = start;
  double prefill_seconds = 0, decode_seconds = 0;
  const auto observe_capacity = [&] {
    std::size_t committed = 0;
    for_each_model([&](Model &selected) {
      std::size_t candidates = 0;
      for (const auto *step : {selected.active_step.get(), selected.successor_step.get()})
        if (step) for (const auto &member : step->members) candidates += bool(member.next);
      const auto live = std::size_t(selected.snapshot_versions.live);
      if (candidates > live) throw std::logic_error("generation candidates exceed their actual version ownership");
      if (live - candidates > selected.cache_plan.resident_sessions)
        throw std::runtime_error("generation committed version count exceeds its model's original budget");
      committed += live - candidates;
    });
    peak_resident = (std::max)(peak_resident, resident_count());
    peak_committed = (std::max)(peak_committed, committed);
  };
  const auto observe_batch = [&](const GenerationBatchFacts &facts) {
    const auto participations = cache::add(actual_fresh_participations, facts.members.size());
    auto rows = actual_fresh_target_rows;
    BatchObservation batch{facts.input_rows, facts.active_rows, facts.physical_rows, facts.role, facts.generation, {}, {}};
    if constexpr (Diagnostic)
      if (!batch.members.try_reserve(facts.members.size()) || !batch.input_rows.try_reserve(facts.members.size()))
        throw std::bad_alloc{};
    for (const auto &member : facts.members) {
      const auto key = member.request->request_key;
      if (!request_loop.matches(key) || key.slot >= requests.size() || &requests[key.slot] != member.request)
        throw std::runtime_error("actual batch fact lost its original request identity");
      if constexpr (Diagnostic) batch.members.push_back(key.slot);
      const auto member_rows = member.actual_target_rows();
      if constexpr (Diagnostic) batch.input_rows.push_back(member_rows);
      rows = cache::add(rows, member_rows);
    }
    if (rows - actual_fresh_target_rows != facts.active_rows || facts.active_rows > facts.physical_rows)
      throw std::logic_error("actual batch row facts differ from its submitted physical carrier");
    // All vectors have a cold bound from the maximum actual episode count.
    // Prepare the variable-width member storage before committing any report.
    if constexpr (Diagnostic) {
    if (!dynamic_admission && (batches.size() == batches.capacity() || facts.members.size() > dispatch.capacity() - dispatch.size() ||
        facts.members.size() > dispatch_turns.capacity() - dispatch_turns.size()))
      throw std::runtime_error("actual batch facts exceed their cold report capacity");
    if (!dynamic_admission) {
      batches.push_back(std::move(batch));
      for (const auto &member : facts.members) {
        dispatch.push_back(member.request->request_key.slot);
        dispatch_turns.push_back(member.request->progress.turn_index);
      }
    }
    }
    ++(facts.generation ? actual_decode_calls : actual_prefill_calls);
    actual_fresh_participations = participations; actual_fresh_target_rows = rows;
    actual_step_sum += facts.step_seconds;
    auto finished = endpoint;
    for (const auto &member : facts.members)
      finished = (std::max)(finished, facts.generation ? member.request->endpoint : member.request->prefill_endpoint);
    const auto seconds = std::chrono::duration<double>(finished - endpoint).count();
    (facts.generation ? decode_seconds : prefill_seconds) += seconds; endpoint = finished;
    observe_capacity();
    if constexpr (!std::is_null_pointer_v<std::remove_cvref_t<BatchObserver>>) batch_observer(facts);
  };
  const auto execute_ready = [&](inference::RequestWave wave) {
    const auto prepare_call = [&](inference::RequestKey key) {
      auto &request = requests[key.slot];
      auto &model = selected_model(request.execution_profile);
      auto *prefix = selected_prefix(request.execution_profile);
      std::cerr << "generation_session id=" << request.id << " turn=" << request.progress.turn_index + 1 << " episode=" << request.rounds.size() << '\n';
      return prepare_generation_call(model, request, request.progress.output_limit(), prefix, requests,
          request_loop.credit(key, request.progress.generated.size()), max_prefill_rows);
    };
    const auto batch_limit = [&](const PreparedGenerationCall &first) -> std::size_t {
      auto &model = selected_model(first.request->execution_profile);
      return first.full_hit ? std::size_t{1} : model.cache_plan.active_sequences;
    };
    const auto compatible_calls = [&](const PreparedGenerationCall &first, const PreparedGenerationCall &other) {
      if constexpr (mixed) if (first.request->execution_profile != other.request->execution_profile) return false;
      auto &model = selected_model(first.request->execution_profile);
      return other.full_hit == first.full_hit && other.request->sampling_index == first.request->sampling_index &&
          other.generation == first.generation &&
          (other.role == first.role || (first.generation && drafts_token(first.role) && drafts_token(other.role))) &&
          other.completion == first.completion &&
          (other.planned_rows() == first.planned_rows() ||
            (!first.full_hit && first.generation && first.role == GraphRole::kVerify &&
             model.verification_capacity((std::max)(other.planned_rows(), first.planned_rows()),
                 2, first.request->sampling_index)) ||
           (!first.full_hit && !first.generation && (first.role == GraphRole::kText || is_mtp_prefill(first.role)) &&
            model.prefill_capacity((std::max)(other.planned_rows(), first.planned_rows()),
                first.completion, first.request->sampling_index, 2, first.role)));
    };
    const auto fits = [&](std::span<PreparedGenerationCall> members) {
      auto &model = selected_model(members[0].request->execution_profile);
      auto *prefix = selected_prefix(members[0].request->execution_profile);
      if (members[0].full_hit) return true;
      if (reuse && members.size() > 1 && !members[0].generation &&
          (members[0].role == GraphRole::kText || is_mtp_prefill(members[0].role)) &&
          !generation_prefill_capacity(model, members)) return false;
      const bool unequal = std::ranges::any_of(members,
          [&](const auto &call) { return call.ids().size() != members[0].ids().size(); });
      const bool mixed_roles = std::ranges::any_of(members,
          [&](const auto &call) { return call.role != members[0].role; });
      if (mixed_roles && !model.find_child(members.size(), 2, GraphRole::kMtpCatchupRound, false,
          members[0].completion, members[0].request->sampling_index, RowPacking::kDraft)) return false;
      return (!unequal || generation_prefill_capacity(model, members) || generation_verification_capacity(model, members)) &&
          make_generation_batch_room(model, prefix, members, requests);
    };
    const auto execute_batch = [&](std::span<PreparedGenerationCall> bucket, bool entire_wave) {
      auto &model = selected_model(bucket[0].request->execution_profile);
      auto *prefix = selected_prefix(bucket[0].request->execution_profile);
      if constexpr (mixed) continuation.enabled = options.profiles[bucket[0].request->execution_profile].overlap;
      std::cerr << "generation_batch requests=" << bucket.size() << " T=" << bucket[0].planned_rows()
                << " role=" << static_cast<int>(bucket[0].role) << " full_hit=" << bucket[0].full_hit;
      if constexpr (mixed) std::cerr << " execution_profile=" << bucket[0].request->execution_profile;
      std::cerr << '\n';
      try {
        execute_generation_calls(model, bucket, trace_states, prefix, requests,
            nullptr, &continuation, observe_batch, entire_wave, wave.credits);
      } catch (const inference::EvaluationFailure &failure) {
        // Only the real admitted bucket may attribute a native member failure.
        throw inference::RequestExecutionFailure::from_evaluation(failure, bucket.size(),
            [&](std::size_t member) { return bucket[member].request->request_key; });
      }
      auto finished = endpoint;
      for (const auto &call : bucket) {
        const auto index = call.request->request_key.slot;
        if constexpr (Diagnostic) if (call.full_hit) {
          // A cache hit is a logical dispatch, never an invented Graph.
          dispatch.push_back(index); dispatch_turns.push_back(requests[index].progress.turn_index);
        }
        finished = (std::max)(finished, requests[index].endpoint);
      }
      // Real Graph batches report their own completion, including a successor
      // observed on a later pump. A full hit has no Graph.
      if (bucket[0].full_hit) {
        const auto seconds = std::chrono::duration<double>(finished - endpoint).count();
        (bucket[0].generation ? decode_seconds : prefill_seconds) += seconds; endpoint = finished;
      }
      observe_capacity();
    };
    const auto publish_call = [&](PreparedGenerationCall &call) {
      publish_generation_prompt(selected_model(call.request->execution_profile), call, selected_prefix(call.request->execution_profile));
    };
    // These typed callable references bind the original model stages directly;
    // they hold no Model interface, runtime callbacks or execution authority.
    struct Policy final {
      decltype(prepare_call) &prepare;
      decltype(batch_limit) &limit;
      decltype(compatible_calls) &compatible;
      decltype(fits) &fit;
      decltype(execute_batch) &execute;
      decltype(publish_call) &publish;
    } policy{prepare_call, batch_limit, compatible_calls, fits, execute_batch, publish_call};
    inference::execute_request_batches(batch_workspace, wave.keys, policy);
    if (decode_overlap && continuation.ready)
      for (const auto &member : continuation.cohort.members())
        if (!request_loop.credit(member.key, requests[member.key.slot].progress.generated.size())) {
          continuation.release_ready(); break;
        }
  };
  Model *wave_model = &model;
  const auto execute_wave = [&](inference::RequestWave wave, std::span<std::size_t> reserved) {
    if constexpr (mixed) {
      if (!continuation.pending_calls.empty()) wave_model = &selected_model(continuation.pending_calls.front().request->execution_profile);
      else if (!wave.keys.empty()) wave_model = &selected_model(requests[wave.keys.front().slot].execution_profile);
    }
    const auto profile = !continuation.pending_calls.empty() ? continuation.pending_calls.front().request->execution_profile :
        !wave.keys.empty() ? requests[wave.keys.front().slot].execution_profile : 0;
    return advance_generation_wave(*wave_model, wave, reserved, trace_states, continuation, execute_ready, nullptr, observe_batch, selected_prefix(profile));
  };
  GenerationRequestAdapter adapter{model, prefix, std::span(requests), trace_states, &evictions, execute_wave,
      {}, &continuation};
  const auto service_output = [&]() {
    if constexpr (!std::is_null_pointer_v<std::remove_cvref_t<OutputConsumer>>)
      return output_deliveries.service(request_loop, adapter, output_consumer);
    else return false;
  };
  const auto pump = [&] { return request_loop.pump_once(adapter); };
  const auto observe_pump = [&](const inference::RequestPumpResult &pumped) {
    completed = cache::add(completed, pumped.completed);
    if (pumped.completed > active_requests) throw std::logic_error("retired requests exceed active requests");
    active_requests -= pumped.completed;
    admitted = cache::add(admitted, pumped.admitted);
  };
  const auto await_retirement = [&](const inference::RequestPumpResult &pumped) {
    // This closed CLI has no host work while the current Graph is pending.
    // Any successor is already submitted; promoted B reaches the next pump first
    // so C can overlap it. The original next poll consumes any wait error.
    if (pumped.waiting_retirement && continuation.drive_cursor.awaiting_current &&
        wave_model->active_step && wave_model->active_step->execution.has_completion())
      wave_model->active_step->execution.wait().IgnoreError();
  };
  const auto poll_input = [&](const inference::RequestPumpResult &pumped) {
    bool input_arrived = false, live_waiting = false;
    if (first_turn_only && !pumped.waiting_retirement) for (std::size_t index = 0; index < requests.size(); ++index) {
      auto &request = requests[index];
      auto &selected = selected_model(request.execution_profile);
      if (!request_loop.matches(request.request_key) || request_loop.cancelled(request.request_key) ||
          !request.progress.waiting_for_input()) continue;
      if (selected.active_step || selected.successor_step || selected.carrier_borrowed || !request.session || request.session->pending ||
          (!reuse && !request_loop.active().empty()))
        throw std::logic_error("next input cannot arrive before its original published wave retires");
      const auto arrival = Clock::now();
      const auto visible = request.progress.turns().size();
      // This is workload-driver EOF/ingress, outside every cold-planning loop.
      // No future turn length or quota was supplied to the Model beforehand.
      const auto &schedule = dynamic_admission ? slot_schedules[index] : schedules[index];
      const auto &source = schedule.turns;
      if (schedule.accepts_live_input()) {
        auto received = live_input->poll_turn(request.request_key);
        if (received.state == inference::TurnInputState::kWaiting) { live_waiting = true; continue; }
        if (received.state == inference::TurnInputState::kClosed) request.progress.close_input();
        else {
          if (visible >= configured_bound) throw std::out_of_range("live input exceeds its configured turn bound");
          request.live_turns.push_back(std::move(received.turn));
        }
      } else if (visible == source.size()) request.progress.close_input();
      if (request.progress.input_open()) {
        const auto &next_source = schedule.accepts_live_input() ? request.live_turns : source;
        const auto &next = next_source[visible];
        const auto followup = inference::check_followup_turn(next, visible, configured_bound,
            request.session->length(), selected.capacity, static_cast<std::size_t>(selected.vocabulary));
        if (followup == inference::FollowupTurnError::kInvalidInput)
          throw std::out_of_range("input exceeds configured turn, prompt or output bound");
        std::exception_ptr input_error;
        try { validate_image_turn(selected,next); }
        catch(const std::invalid_argument &error){input_error=std::make_exception_ptr(inference::RequestRejected(error.what()));}
        catch(const std::out_of_range &error){input_error=std::make_exception_ptr(inference::RequestRejected(error.what()));}
        // The preceding turn may have stopped early. Check the next turn
        // against its actual committed state, before extending or submitting it.
        if (followup == inference::FollowupTurnError::kRemainingContext || input_error) {
          const auto error = input_error?input_error:std::make_exception_ptr(inference::RequestRejected(
              "complete turn input and output quota exceed the remaining context"));
          if constexpr (!std::is_null_pointer_v<std::remove_cvref_t<OutputConsumer>> &&
              requires { { output_consumer.fail(request.request_key, error) } noexcept -> std::same_as<bool>; }) {
            if (reuse && output_consumer.fail(request.request_key, error)) { input_arrived = true; continue; }
          }
          std::rethrow_exception(error);
        }
        auto owner = std::make_shared<std::vector<std::int32_t>>(cache::add(next.prompt.size(), 1));
        std::copy(next.prompt.begin(), next.prompt.end(), owner->begin() + 1);
        request.progress.extend_turns(std::span(next_source).first(visible + 1));
        request.warm_inputs.push_back(std::move(owner));
        request.turn_observations.emplace_back(); request.turn_observations.back().arrival = arrival;
        request.admission_start = arrival;
        ++total_turns; requested_outputs = cache::add(requested_outputs, next.max_outputs);
      }
      std::cerr << "input_arrival visible_turns=" << request.progress.turns().size()
                << " eof=" << !request.progress.input_open() << " elapsed_seconds="
                << std::chrono::duration<double>(arrival - startup).count() << '\n';
      request_loop.wake(request.request_key); input_arrived = true;
    }
    if constexpr (!std::is_null_pointer_v<std::remove_cvref_t<OutputConsumer>>) {
      if (dynamic_admission && !pumped.waiting_retirement) {
        const auto admit = [&](inference::TokenConsumerRef &consumer) {
          return admit_waiting_generation(model, request_loop, *live_input, consumer, requests,
              slot_schedules, output_deliveries, configured_prefill,
              configured_bound, configured_record_bound, active_requests, total_turns, requested_outputs,
              mixed ? options.profiles : std::span<const PreparedGenerationProfile>{}, prefix);
        };
        if constexpr (std::is_same_v<std::remove_cvref_t<OutputConsumer>, inference::TokenConsumerRef>)
          input_arrived = admit(output_consumer) || input_arrived;
        else {
          inference::TokenConsumerRef consumer(output_consumer);
          input_arrived = admit(consumer) || input_arrived;
        }
      }
    }
    return inference::RequestOwnerInputProgress{input_arrived, live_waiting};
  };
  const auto wait_idle = [&](const inference::RequestPumpResult &pumped,
                             inference::RequestOwnerInputProgress input) {
    (void)pumped; (void)input;
    if constexpr (!std::is_null_pointer_v<std::remove_cvref_t<OutputConsumer>>) {
      if (pumped.waiting_output || input.waiting) { output_consumer.wait(); return true; }
    }
    return false;
  };
  inference::drive_request_owner(active_requests, service_output, pump, observe_pump,
                                 await_retirement, poll_input, wait_idle);
  if (reuse) {
    for_each_model([&](Model &selected) {
      if (selected.active_step || selected.successor_step || selected.carrier_borrowed ||
          selected.snapshot_versions.live != cached_versions(selected) || selected.resources_->pending_readbacks())
        throw std::logic_error("reusable request has not actually retired its model resources");
    });
    reuse->ready = true; reuse_success = true;
  }
  if constexpr (Diagnostic) {
    if (!dynamic_admission && diagnostics->completed) {
      const inference::RequestOutputSlots *delivery_view = nullptr;
      if constexpr (!std::is_null_pointer_v<std::remove_cvref_t<OutputConsumer>>) delivery_view = &output_deliveries;
      const GenerationDiagnosticView view{
          .model = model,
          .requests = requests,
          .output_deliveries = delivery_view,
          .prefix = prefix,
          .dispatch = dispatch,
          .dispatch_turns = dispatch_turns,
          .evictions = evictions,
          .batches = batches,
          .startup = startup,
          .ready = ready,
          .start = start,
          .endpoint = endpoint,
          .total_turns = total_turns,
          .requested_outputs = requested_outputs,
          .admitted = admitted,
          .completed = completed,
          .peak_resident = peak_resident,
          .peak_committed = peak_committed,
          .actual_prefill_calls = actual_prefill_calls,
          .actual_decode_calls = actual_decode_calls,
          .actual_fresh_participations = actual_fresh_participations,
          .actual_fresh_target_rows = actual_fresh_target_rows,
          .frontier_before = frontier_before,
          .draft_before = draft_before,
          .prefix_before = prefix_before,
          .submitted_draft_begin = submitted_draft_begin,
          .prefill_seconds = prefill_seconds,
          .decode_seconds = decode_seconds,
          .actual_step_sum = actual_step_sum,
          .frontier_seconds_before = frontier_seconds_before,
          .cold_prepare_seconds = cold_prepare_seconds,
          .mtp = mtp,
          .reuse_ready = reuse_ready,
          .decode_overlap = decode_overlap,
          .prefix_enabled = prefix_enabled,
          .first_turn_only = first_turn_only,
          .startup_prefill_warmup = startup_prefill_warmup,
          .packed_prefill_offset = packed_prefill_offset,
          .continuation = {continuation.unavailable, continuation.successor_submissions, continuation.successor_publications, continuation.successor_discards, continuation.successor_refusals},
          .routes = *diagnostics->routes,
          .trace = trace,
      };
      diagnostics->completed(diagnostics->context, view);
    }
  }
  return 0;
}
template<GenerationMode Mode, class BatchObserver = std::nullptr_t, class OutputConsumer = std::nullptr_t>
int run_generation_sessions(Model &model, std::span<const inference::GenerationSessionSchedule> schedules,
    const std::filesystem::path &trace, bool trace_states, Clock::time_point startup,
    Clock::time_point constructed, bool prefix_enabled, std::span<const cache::TokenPlanRef> token_plans,
    std::span<const std::size_t> sampling_indices = {}, std::span<const inference::RequestStopPolicy> stop_policies = {},
    std::size_t max_prefill_rows = SIZE_MAX, bool warmup_planned_shapes = false,
    bool decode_overlap = false, std::span<const inference::LogitsInputProvider> logits_providers = {},
    BatchObserver &&batch_observer = nullptr, bool first_turn_only = false,
    const qwen35::RouteDecisionFacts &routes = {}, OutputConsumer &&output_consumer = nullptr,
    std::ostream &report_out = std::cout, ReusableGenerationPreparation *reuse = nullptr,
    inference::GenerationInputRef *live_input = nullptr) {
  GenerationDiagnostics diagnostic{trace, trace_states, prefix_enabled, warmup_planned_shapes, first_turn_only,
      startup, constructed, token_plans, logits_providers, &routes, &report_out,
      [](void *stream, const GenerationDiagnosticView &view) { write_generation_report(*static_cast<std::ostream *>(stream), view); }};
  return execute_generation_sessions<Mode, true>(model, {schedules, sampling_indices, stop_policies, live_input},
      {max_prefill_rows, decode_overlap, reuse}, std::forward<OutputConsumer>(output_consumer),
      std::forward<BatchObserver>(batch_observer), &diagnostic);
}

// Internal bridge from a prepared Qwen Model to the common finite-cohort
// contract. The original Model outlives this object and remains the sole
// owner of its weights, sessions, native work and retirement. It deliberately
// does not expose Qwen's state or legacy JSON report to the common interface.
class PreparedModelExecutor final : public inference::ModelExecutor {
public:
  PreparedModelExecutor(Model &model, bool mtp, std::size_t prefill_rows, bool overlap, cache::PrefixCache *prefix = nullptr)
      : model_(model), mtp_(mtp), overlap_(overlap), prefill_rows_(prefill_rows), prefix_(prefix) {}
  inference::ModelDescription describe() const override {
    inference::ModelDescription description{static_cast<std::size_t>(model_.vocabulary), model_.capacity,
        {model_eos_tokens.begin(), model_eos_tokens.end()}};
    description.execution_profiles[0].algorithm = mtp_ ? "mtp" : "ordinary";
    description.execution_profiles[0].max_draft_tokens = mtp_ ? model_.max_drafts() : 0;
    description.source_identity = model_.source_identity;
    return description;
  }
  bool supports_dynamic_admission() const noexcept override { return true; }
  bool supports_concurrent_sampling_profiles() const noexcept override { return true; }
  void validate(inference::GenerationBatch batch) override {
    if (running_.test(std::memory_order_acquire) || preparation_.busy || preparation_.failed)
      throw std::logic_error("prepared model executor is busy or failed");
    if (model_.cache_plan.draft_cache != mtp_)
      throw std::logic_error("prepared model mode differs from its cache plan");
    try {
      const auto prefill = (std::min)({prefill_rows_, std::size_t{512}, model_.capacity});
      validate_reusable_batch(model_, batch, mtp_, prefill);
    } catch (const std::invalid_argument &error) {
      throw inference::RequestRejected(error.what());
    } catch (const std::out_of_range &error) {
      throw inference::RequestRejected(error.what());
    }
  }
  void execute(inference::GenerationBatch batch, inference::TokenConsumerRef &consumer) override {
    // One original model/request owner is serialized. Do not hold a second
    // scheduler or infer that a failed invocation retired native work.
    if (running_.test_and_set(std::memory_order_acquire))
      throw std::logic_error("prepared model executor is already running");
    hyle::util::ScopeGuard release{[&]() noexcept { running_.clear(std::memory_order_release); }};
    const GenerationExecutionOptions options{prefill_rows_, overlap_, &preparation_, {}, prefix_};
    const auto result = mtp_
        ? execute_generation_sessions<GenerationMode::kMtp>(model_, batch, options, consumer)
        : execute_generation_sessions<GenerationMode::kText>(model_, batch, options, consumer);
    if (result != 0) throw std::runtime_error("prepared model execution returned a failure");
  }
private:
  Model &model_;
  ReusableGenerationPreparation preparation_;
  bool mtp_, overlap_;
  std::size_t prefill_rows_;
  cache::PrefixCache *prefix_;
  std::atomic_flag running_ = ATOMIC_FLAG_INIT;
};
std::unique_ptr<inference::ModelExecutor> make_prepared_executor(
    Model &model, bool mtp, std::size_t prefill_rows, bool overlap) {
  return std::make_unique<PreparedModelExecutor>(model, mtp, prefill_rows, overlap);
}

// All profiles borrow one immutable weight bank and endpoint. The original
// RequestLoop serializes their compatible native batches; no per-profile
// thread, key mapper or alternative retirement authority is introduced.
class LoadedModelExecutor final : public inference::ModelExecutor {
public:
  LoadedModelExecutor(std::vector<std::unique_ptr<Model>> models,
      std::vector<PreparedGenerationProfile> profiles, std::vector<std::string> names,
      std::size_t prefill_rows, std::size_t prefix_entries)
      : models_(std::move(models)), profiles_(std::move(profiles)), names_(std::move(names)), prefill_rows_(prefill_rows) {
    if (prefix_entries) for (auto &profile : profiles_) {
      if(profile.model->vision)continue; // Token-only prefix keys do not identify image content.
      prefixes_.push_back(std::make_unique<cache::PrefixCache>(prefix_entries));
      profile.prefix = prefixes_.back().get();
    }
    if (models_.size() == 1) single_ = std::make_unique<PreparedModelExecutor>(
        *models_[0], models_[0]->cache_plan.draft_cache, prefill_rows, profiles_[0].overlap, profiles_[0].prefix);
  }
  inference::ModelDescription describe() const override {
    inference::ModelDescription description{static_cast<std::size_t>(models_[0]->vocabulary), models_[0]->capacity,
        {model_eos_tokens.begin(), model_eos_tokens.end()}};
    description.execution_profiles.clear();
    description.source_identity = models_[0]->source_identity;
    for (std::size_t j = 0; j < models_.size(); ++j) {
      const auto &model = *models_[j];
      inference::ExecutionProfileDescription profile{names_[j], model.cache_plan.draft_cache ? "mtp" : "ordinary",
          model.cache_plan.draft_cache ? model.max_drafts() : 0};
      for (std::size_t k = 0; k < models_.size(); ++k) if (j != k) profile.co_resident_with.push_back(k);
      description.execution_profiles.push_back(std::move(profile));
    }
    return description;
  }
  bool supports_dynamic_admission() const noexcept override { return true; }
  bool supports_concurrent_sampling_profiles() const noexcept override { return true; }
  void validate(inference::GenerationBatch batch) override {
    if (single_) { single_->validate(batch); return; }
    if (running_.test(std::memory_order_acquire) || preparation_.busy || preparation_.failed)
      throw std::logic_error("prepared model executor is busy or failed");
    try {
      validate_profile_batch(*models_[0], batch,
          (std::min)({prefill_rows_, std::size_t{512}, models_[0]->capacity}), profiles_);
    } catch (const std::invalid_argument &error) { throw inference::RequestRejected(error.what()); }
      catch (const std::out_of_range &error) { throw inference::RequestRejected(error.what()); }
  }
  void execute(inference::GenerationBatch batch, inference::TokenConsumerRef &consumer) override {
    if (single_) { single_->execute(batch, consumer); return; }
    if (running_.test_and_set(std::memory_order_acquire))
      throw std::logic_error("prepared model executor is already running");
    hyle::util::ScopeGuard release{[&]() noexcept { running_.clear(std::memory_order_release); }};
    const bool overlap = std::ranges::any_of(profiles_, [](const auto &profile) { return profile.overlap; });
    if (execute_generation_sessions<GenerationMode::kMixed>(*models_[0], batch,
        {prefill_rows_, overlap, &preparation_, profiles_}, consumer))
      throw std::runtime_error("prepared model execution returned a failure");
  }
private:
  std::vector<std::unique_ptr<Model>> models_;
  // Complete cached states release before their original model resources.
  std::vector<std::unique_ptr<cache::PrefixCache>> prefixes_;
  std::vector<PreparedGenerationProfile> profiles_;
  std::vector<std::string> names_;
  std::size_t prefill_rows_;
  std::unique_ptr<PreparedModelExecutor> single_;
  ReusableGenerationPreparation preparation_;
  std::atomic_flag running_ = ATOMIC_FLAG_INIT;
};

bool matches_prepared_model(const std::filesystem::path &source) {
  std::error_code error;
  const bool directory = std::filesystem::is_directory(source, error);
  if (error && error != std::errc::no_such_file_or_directory && error != std::errc::not_a_directory)
    throw std::filesystem::filesystem_error("cannot inspect model input", source, error);
  if (!directory) return false;
  const auto manifest = source / "weights.tsv";
  const bool exists = std::filesystem::exists(manifest, error);
  if (error) throw std::filesystem::filesystem_error("cannot inspect model manifest", manifest, error);
  if (!exists) return false;
  std::ifstream input(manifest);
  if (!input) throw std::runtime_error("cannot open model manifest: " + manifest.string());
  std::string signature;
  char ch;
  while (input.get(ch) && ch != '\n') {
    if (signature.size() == 128) return false;
    signature.push_back(ch);
  }
  if (input.bad()) throw std::runtime_error("cannot read model manifest: " + manifest.string());
  return prepared::is_role_signature(signature);
}

struct ExecutionProfilePreparation final {
  PreparedWeightSource source;
  Backend endpoint;
  cache::Plan cache;
  Model::Sampling sampling;
  std::vector<SamplingConfiguration> additional_sampling;
  std::size_t draft_depth;
  bool overlap;
  std::filesystem::path vision_path;
};

ExecutionProfilePreparation prepare_execution_profile(const inference::ModelLoadConfig &config,
    std::size_t index, const Backend *shared_endpoint = nullptr) {
  const auto &profile = config.execution_profiles[index];
  const auto source_assets = inference::execution_profile_assets(config, index);
  inference::validate_model_assets(source_assets);
  for (const auto &asset : source_assets.auxiliary)
    if (asset.role != "draft.vocabulary" && asset.role != "draft.head" && asset.role != "vision.encoder")
      throw std::invalid_argument("Qwen model does not support asset role: " + asset.role);
  const auto *vocabulary_asset = inference::find_model_asset(source_assets, "draft.vocabulary");
  const auto *head_asset = inference::find_model_asset(source_assets, "draft.head");
  const auto *vision_asset=inference::find_model_asset(source_assets,"vision.encoder");
  if(vision_asset && (vision_asset->mode!=inference::ModelAssetMode::kFile || profile.overlap.value_or(false) ||
      (profile.draft_count && *profile.draft_count)))
    throw std::invalid_argument("the prepared image recipe requires a file asset and ordinary generation without device overlap");
  if (!config.execution.context || !config.execution.resident_sessions || !config.execution.active_sequences ||
      config.execution.active_sequences > config.execution.resident_sessions || !config.execution.prefill_rows ||
      (profile.draft_count && *profile.draft_count >= config.execution.context))
    throw std::invalid_argument("model loading requires valid context, sequence, prefill and draft capacities");
  const auto execution = select_execution(BackendSelection{.primary = config.execution.backend}, 0);
  const bool has_greedy_sampling = !config.sampling.random || std::ranges::any_of(config.additional_sampling,
      [](const auto &policy) { return !policy.random; });
  const inference::ColdWorkloadFacts workload{
      .natural_generation = true, .greedy = has_greedy_sampling,
      .max_sessions = config.execution.resident_sessions, .active_sequences = config.execution.active_sequences,
      .capacity = config.execution.context};
  namespace mtp_policy = inference::mtp;
  const bool explicit_depth = profile.draft_count.has_value();
  const auto requested_depth = profile.draft_count.value_or(1);
  MtpDefaultRequest request{
      .mode = vision_asset ? mtp_policy::RequestedMode::kOff : !explicit_depth ? mtp_policy::RequestedMode::kAuto :
          requested_depth ? mtp_policy::RequestedMode::kOn : mtp_policy::RequestedMode::kOff,
      .workload = workload, .automatic_backend_eligible = execution.primary == BackendKind::kCuda,
      .max_drafts = requested_depth,
      .automatic_vocabulary = vocabulary_asset ? vocabulary_asset->mode == inference::ModelAssetMode::kAutomatic : !explicit_depth};
  auto vocabulary = vocabulary_asset ? (vocabulary_asset->mode == inference::ModelAssetMode::kFile
      ? DraftAssetRequest::from_file(vocabulary_asset->path) : DraftAssetRequest::automatic()) :
      explicit_depth ? DraftAssetRequest::original() : DraftAssetRequest::automatic();
  auto head = head_asset ? (head_asset->mode == inference::ModelAssetMode::kFile
      ? DraftAssetRequest::from_file(head_asset->path) : DraftAssetRequest::automatic()) :
      !explicit_depth ? DraftAssetRequest::automatic() : DraftAssetRequest::original();
  auto assets = select_draft_assets(config.assets.inputs, request, std::move(vocabulary), std::move(head),
      Backend::supports_encoded_row_gather(execution));
  const bool mtp = assets.policy.enabled;
  const auto prefill = (std::min)({config.execution.prefill_rows, std::size_t{512}, config.execution.context});
  if (mtp && prefill < 2) throw std::invalid_argument("MTP preparation needs a multirow prefill capacity");
  auto source = prepare_weight_source(config.assets.inputs / "weights.tsv", mtp, assets.vocabulary,
      assets.head, assets.auto_head);
  const auto selected_depth = mtp ? mtp_policy::select_draft_depth(assets.policy, explicit_depth,
      requested_depth, source.draft_preferences.recommended_drafts).max_drafts : 0;
  // Recommendations describe a preferred maximum, not a requirement that a
  // short context can hold every proposed row.
  const auto depth = (std::min)(selected_depth, config.execution.context - 1);
  if (mtp && !depth) throw std::invalid_argument("model draft depth requires a remaining context row");
  const auto geometry = source.roles.geometry;
  validate_mtp_source(geometry, mtp);
  const auto valid_sampling = [&](const inference::PreparedSamplingPolicy &policy) {
    if (policy.top_k > static_cast<std::size_t>(geometry.vocabulary))
      throw std::invalid_argument("sampling top-k exceeds the source vocabulary");
  };
  valid_sampling(config.sampling);
  std::vector<SamplingConfiguration> additional_sampling;
  additional_sampling.reserve(config.additional_sampling.size());
  bool retains_random_frontier = config.sampling.random;
  for (const auto &policy : config.additional_sampling) {
    valid_sampling(policy);
    additional_sampling.push_back({policy.random, {policy.temperature, policy.top_k, policy.top_p}, {}});
    retains_random_frontier = retains_random_frontier || policy.random;
  }
  const auto precision = PrecisionProfile::from_roles(source.roles);
  const qwen35::DecodeRouteFacts facts{.workload = workload, .overlap_requested = true,
      .mtp_enabled = mtp, .max_drafts = depth, .max_device_drafts = kMaximumDeviceMtpDrafts,
      .stored_draft_head = supports_stored_draft_projection(source.roles),
      .has_decode = config.execution.context > 1, .resolved_active_sequences = config.execution.active_sequences};
  std::vector<qwen35::DecodeRouteFacts> sampling_routes;
  sampling_routes.reserve(config.additional_sampling.size() + 1);
  const auto append_sampling_route = [&](const inference::PreparedSamplingPolicy &policy) {
    auto selected = facts;
    selected.workload.greedy = !policy.random;
    const DeviceRelayCapabilities relay{execution.primary, false, false, policy.random && mtp, false};
    selected.relay_eligible = relay.eligible(); selected.relay_refusal = relay.refusal_reason();
    sampling_routes.push_back(selected);
  };
  append_sampling_route(config.sampling);
  for (const auto &policy : config.additional_sampling) append_sampling_route(policy);
  const auto routes = qwen35::plan_prepared_decode_routes(sampling_routes, profile.overlap);
  const bool overlap = !vision_asset && routes.overlap;
  // The verifier can consume a standalone draft chain; it does not require
  // an overlapping successor. Keep its independently selected capacity.
  const bool verification = routes.verification;
  const auto query_rows = (std::max)(prefill, mtp ? cache::add(depth, 1) : std::size_t{1});
  const auto recurrent = recurrent_bytes(state_schema_for(geometry, config.execution.context, precision), geometry);
  auto continuation_facts = sampling_routes.front();
  for (std::size_t i = 0; i < routes.by_sampling.size(); ++i)
    if (routes.by_sampling[i].overlap.enabled) { continuation_facts = sampling_routes[i]; break; }
  auto continuous = prepare_mtp_continuation({.routes = continuation_facts, .geometry = geometry, .source = source,
      .precision = precision, .backend = execution, .context = config.execution.context, .residents = config.execution.resident_sessions,
      .active_sequences = config.execution.active_sequences, .query_rows = query_rows, .recurrent_bytes = recurrent,
      .draft_depth = depth, .overlap = overlap, .verification = verification, .random = retains_random_frontier,
      .shared_endpoint = shared_endpoint});
  const auto plan = cache::plan(geometry, source.cache_row_bytes, config.execution.context, config.execution.resident_sessions,
      query_rows, recurrent, config.execution.active_sequences, mtp ? cache::add(depth, 1) : 1, mtp, retains_random_frontier,
      overlap ? (continuous.enabled ? 3 : 2) : 1, verification ? 2 : 0, precision, verification ? depth : 0);
  const Model::Sampling sampling{.enabled = config.sampling.random,
      .policy = {config.sampling.temperature, config.sampling.top_k, config.sampling.top_p}, .seed = 0};
  auto backend = shared_endpoint ? take(shared_endpoint->share_endpoint()) :
      continuous.endpoint ? std::move(*continuous.endpoint) : create_execution(geometry, plan, 0, execution);
  return {std::move(source), std::move(backend), plan, sampling, std::move(additional_sampling), depth, overlap,
      vision_asset?vision_asset->path:std::filesystem::path{}};
}

std::unique_ptr<Model> instantiate_execution_profile(ExecutionProfilePreparation preparation,
    std::size_t capacity, std::shared_ptr<WeightStorage> weights = {}, std::shared_ptr<VisionEncoder> shared_vision = {}) {
  const auto geometry = preparation.source.roles.geometry;
  const auto vision_path=preparation.vision_path;
  const auto vision_target=vision_path.empty()?prepared::Roles{}:preparation.source.roles;
  auto model = std::make_unique<Model>(std::move(preparation.endpoint), hyle::compute::ResourceBudget{}, std::move(preparation.source),
      geometry, capacity, preparation.cache, std::filesystem::path{}, false, ReplayPolicy::kAuto, SIZE_MAX,
      preparation.sampling, 0, preparation.additional_sampling, std::move(weights));
  if (preparation.cache.draft_cache) model->configure_draft_policy({preparation.draft_depth});
  if(!vision_path.empty())model->vision=shared_vision?std::move(shared_vision):
      std::make_shared<VisionEncoder>(model->execution,vision_target,vision_path,model->allocation_budget);
  return model;
}
std::unique_ptr<inference::ModelExecutor> load_model_executor(const inference::ModelLoadConfig &config) {
  inference::validate_execution_preparation(config);
  const auto count = config.execution_profiles.size();
  std::vector<std::unique_ptr<Model>> models(count);
  std::vector<PreparedGenerationProfile> profiles(count);
  std::vector<std::string> names;
  auto first = prepare_execution_profile(config, 0);
  if (count == 1) {
    // Preserve the ordinary inline load and single prepared executor. No union
    // inventory or separate asset pass is needed for one execution profile.
    const bool overlap = first.overlap;
    models[0] = instantiate_execution_profile(std::move(first), config.execution.context);
    profiles[0] = {models[0].get(), overlap};
  } else {
    std::vector<ExecutionProfilePreparation> prepared;
    prepared.reserve(count);
    prepared.push_back(std::move(first));
    for (std::size_t j = 1; j < count; ++j)
      prepared.push_back(prepare_execution_profile(config, j, &prepared.front().endpoint));
    std::vector<const PreparedWeightSource *> sources;
    sources.reserve(count);
    for (const auto &profile : prepared) sources.push_back(&profile.source);
    auto weights = prepare_shared_weights(prepared.front().endpoint, shared_weight_roles(sources));
    std::map<std::filesystem::path,std::shared_ptr<VisionEncoder>> shared_vision;
    for (std::size_t j = 0; j < count; ++j) {
      const bool overlap = prepared[j].overlap;
      const auto key=prepared[j].vision_path.empty()?std::filesystem::path{}:std::filesystem::canonical(prepared[j].vision_path);
      auto encoder=key.empty()?std::shared_ptr<VisionEncoder>{}:shared_vision[key];
      models[j] = instantiate_execution_profile(std::move(prepared[j]), config.execution.context, weights,std::move(encoder));
      if(!key.empty())shared_vision[key]=models[j]->vision;
      profiles[j] = {models[j].get(), overlap};
    }
  }
  for (const auto &profile : config.execution_profiles) names.push_back(profile.name);
  return std::make_unique<LoadedModelExecutor>(std::move(models), std::move(profiles), std::move(names),
      config.execution.prefill_rows, config.execution.prefix_cache_entries);
}
} // namespace qwen35
