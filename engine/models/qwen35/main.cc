#include <atomic>
#include <deque>

#include "model_core.h"
#include "prepared_executor.h"
#include "schedule.h"
#include "session_schedule.h"
#include "prefix_cache.h"
#include "batch_schedule.h"
#include "generation_schedule.h"
#include "generation_ledger.h"
#include "generation_publication.h"
#include "draft_asset_policy.h"
#include "../../inference/session_cache.h"
#include "../../inference/request_progress.h"
#include "../../inference/request_loop.h"
#include "../../inference/token_delivery.h"
#include "../../inference/model_executor.h"
#include "../../inference/request_wave_execution.h"
#include "../../inference/step_execution.h"
#include "../../inference/continuation.h"
#include "../../inference/decode_route_policy.h"
#include "../../inference/prefill_plan.h"
#include "../../inference/version_admission.h"
#include "../../inference/batch_dispatch.h"
#include "../../inference/request_batch_execution.h"
#include "../../inference/mtp_policy.h"
#include "../../reporting/execution_report.h"
#include "../../reporting/qwen35_cache_report.h"
#include "../../reporting/qwen35_weight_inventory.h"
#include "driver.h"

namespace qwen35 {

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
  const auto dump_bytes = [&](mem::ConstBufferView view, const std::filesystem::path &path) {
    std::vector<std::byte> bytes(view.size()); check(model.execution.read(view, bytes));
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!stream) throw std::runtime_error("cannot write Qwen trace: " + path.string());
  };
  if (io.trace && !io.trace->empty()) {
    const auto directory = *io.trace / io.name; std::filesystem::create_directories(directory);
    const auto dump_row = [&](c::ConstTensorView tensor, const char *file) {
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
    dump_row(completed.hidden, model.precision.is_f16() ? "hidden.f16" : "hidden.bf16");
    if (io.trace_states) {
      for (std::size_t i = 0; i < model.state_schema.size(); ++i) if (model.geometry.is_gdn(i)) {
        const auto prefix = "layer" + std::to_string(i);
        dump_bytes(take(completed.head.state[i].history.view()), directory / (prefix + (model.precision.is_f16() ? ".conv.f16" : ".conv.bf16")));
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
          directory / (std::string(layer0_probe_names[i]) + (model.precision.is_f16() ? ".f16" : ".bf16")));
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

struct GenerationRestoreStats final {
  std::size_t graph_calls = 0, target_rows = 0, draft_rows = 0;
  std::size_t submitted_draft_rows = 0;
  std::size_t normalized_lag_rows = 0;
  struct DecisionChange { std::size_t episode; std::int32_t historical, fresh; };
  std::vector<DecisionChange> decision_changes;
  double step_seconds = 0, seconds = 0;
};

enum class GenerationMode { kText, kMtp };

struct GenerationRequest {
  std::string id, memory_name;
  std::span<const std::int32_t> prompt;
  PrefillInputPackets prefill_packets;
  std::filesystem::path trace;
  inference::RequestProgress progress;
  inference::RequestKey request_key{};
  std::shared_ptr<Session> session;
  cache::TokenPlanRef input;
  std::shared_ptr<const cache::HostImage> host_image;
  cache::Checkpoint completed_checkpoint;
  std::shared_ptr<cache::ReplayHistory> history;
  std::vector<MtpRoundResult> rounds;
  std::vector<std::vector<std::int32_t>> round_emitted;
  Clock::time_point prefill_start{}, prefill_endpoint{}, endpoint{};
  double prefill_step_seconds = 0, decode_step_sum_seconds = 0;
  std::size_t verified_rows = 0, draft_rows = 0, draft_rounds = 0, agreements = 0, accepts = 0;
  std::size_t visible_tokens = 0, draft_frontier = 0, draft_lag = 0;
  std::size_t reused_tokens = 0, prefix_hits = 0, prefix_misses = 0, prefix_full_hits = 0;
  std::size_t prefill_graph_calls = 0, prefill_graph_tokens = 0, prefill_draft_rows = 0;
  std::size_t prefix_sampling_calls = 0;
  std::size_t auxiliary_graph_calls = 0;
  double prefix_sampling_seconds = 0;
  const inference::RequestStopPolicy stop_policy;
  std::unique_ptr<GenerationLedger> ledger;
  struct TurnObservation {
    Clock::time_point start{}, prefill_endpoint{}, endpoint{}, arrival{};
    double prefill_shape_admission_seconds = 0, prefill_phase_prepare_seconds = 0, prefill_slot_initialize_seconds = 0;
    std::size_t prefill_phase_creations = 0, prefill_slot_initializations = 0;
    std::size_t output_begin = 0, output_end = 0;
    std::size_t context_before = 0, context_after = 0;
  };
  std::vector<TurnObservation> turn_observations;
  std::vector<std::shared_ptr<std::vector<std::int32_t>>> warm_inputs;
  std::size_t last_idle = 0;
  const std::size_t sampling_index;
  LogitsInputs logits; // One [1,V] snapshot for every conditional row in an output episode.
  inference::LogitsInputProvider logits_provider;
  Clock::time_point admission_start{};
  GenerationRestoreStats restored;
  std::size_t restore_count = 0, idle_evictions = 0;
  // Retired Session submissions plus the live Session delta. Restore submits
  // have their own original call scope and are excluded from that live delta.
  std::size_t retired_draft_rows = 0, session_draft_rows_begin = 0;
  std::size_t actual_draft_rows() const noexcept {
    return retired_draft_rows + restored.submitted_draft_rows +
        (session ? session->submitted_draft_rows - session_draft_rows_begin : 0);
  }
  void retire_draft_rows() noexcept {
    if (session) retired_draft_rows += session->submitted_draft_rows - session_draft_rows_begin;
    session_draft_rows_begin = 0;
  }
  std::size_t checkpoint_resume_hits = 0, checkpoint_resume_misses = 0;
  std::int32_t fresh_next_id = -1;

  GenerationRequest(std::string name, std::span<const std::int32_t> ids,
                       std::filesystem::path output, std::size_t steps, std::size_t sampling = 0,
                       inference::RequestStopPolicy stop = {})
      : id(std::move(name)), memory_name(id.empty() ? "generation" : "generation_" + id),
        prompt(ids), trace(std::move(output)), progress(steps), stop_policy(std::move(stop)), sampling_index(sampling) {
    rounds.reserve(steps - 1);
  }
};

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
  const auto residents = requests | std::views::transform([](const auto &request) {
    return version_holder(request.session.get());
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
      auto hit = prefix->lookup(*request.completed_checkpoint.input, length, length, &model, model.math_identity());
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
        consumed = prefix->match_length(*request.input, request.prompt.size(), 1, &model, model.math_identity());
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
      auto hit = prefix->lookup(*request.input, request.prompt.size(), 1, &model, model.math_identity());
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
        ? (std::min)({model.max_drafts(), call.quota - 1, remaining - 1}) : 0;
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
  const auto residents = requests | std::views::transform([](const auto &request) {
    return version_holder(request.session.get());
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
  prefix->publish(request.input, request.session->checkpoint(), {request.history, request.history->chunks.size(), request.prompt.size()});
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
  prefix->publish(request.completed_checkpoint.input, request.session->checkpoint(), request.completed_checkpoint.replay);
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
  Model &model;
  cache::PrefixCache *prefix;
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
    publish_generation_completed(model, requests[key.slot], prefix);
  }
  inference::RetirementObservation retire(inference::RequestKey key, inference::RequestRetirement reason) const {
    if (continuation) continuation->release(key);
    auto &request = requests[key.slot];
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

void print_generation_rebuild(std::ostream &out, const GenerationRestoreStats &stats, bool draft_cache) {
  out << ",\"restore_contract\":\"recompute_committed_inputs\",\"restored_graph_calls\":" << stats.graph_calls
      << ",\"restored_target_rows\":" << stats.target_rows;
  if (draft_cache) out << ",\"restored_draft_rows\":" << stats.draft_rows
      << ",\"actual_restored_draft_rows\":" << stats.submitted_draft_rows
      << ",\"normalized_lag_rows\":" << stats.normalized_lag_rows;
  out << ",\"decision_changes\":[";
  for (std::size_t i = 0; i < stats.decision_changes.size(); ++i) {
    const auto &change = stats.decision_changes[i];
    out << (i ? "," : "") << "{\"episode\":" << change.episode << ",\"historical_next\":" << change.historical
        << ",\"fresh_next\":" << change.fresh << '}';
  }
  out << ']';
}

void write_request_sampling(std::ostream &out, const Model &model, const GenerationRequest &request) {
  const auto &configuration = model.sampling_configuration(request.sampling_index).configuration;
  if (model.sampling_configuration_count() == 1 && configuration.modifiers.history.empty()) return;
  out << ",\"sampling_configuration_index\":" << request.sampling_index
      << ",\"request_sampling\":\"" << (configuration.enabled ? "random" : "greedy") << '\"'
      << ",\"request_temperature\":" << configuration.policy.temperature
      << ",\"request_top_k\":" << configuration.policy.top_k << ",\"request_top_p\":" << configuration.policy.top_p
      << ",\"request_history_repetition\":" << configuration.modifiers.history.repetition
      << ",\"request_history_presence\":" << configuration.modifiers.history.presence
      << ",\"request_history_frequency\":" << configuration.modifiers.history.frequency
      << ",\"request_mtp_acceptance_policy\":\"" << (!model.cache_plan.draft_cache ? "disabled" : configuration.enabled ? "min_one_p_over_q" : "equal_greedy_ids") << '\"';
}

void print_generation_request(std::ostream &out, const Model &model, const GenerationRequest &request, std::size_t steps,
                       Clock::time_point startup, Clock::time_point ready,
                       std::array<std::size_t, 3> packed_prefill_offset = {},
                       const hyle::engine::reporting::RouteDecisionFacts &routes = {}) {
  const auto &prompt = request.prompt;
  const auto &generated = request.progress.generated;
  const auto &rounds = request.rounds;
  const auto prefill_start = request.turn_observations.empty() ? request.prefill_start : request.turn_observations.front().start;
  const auto prefill_endpoint = request.turn_observations.empty() ? request.prefill_endpoint : request.turn_observations.front().prefill_endpoint;
  const auto endpoint = request.endpoint;
  const auto prefill_step_seconds = request.prefill_step_seconds,
      decode_step_sum_seconds = request.decode_step_sum_seconds;
  const auto verified_rows = request.verified_rows, draft_rows = request.draft_rows,
      draft_rounds = request.draft_rounds, agreements = request.agreements, accepts = request.accepts;
  const auto &trace = request.trace;
  // The original finish endpoint follows publication and host token visibility,
  // before its trailing trace/log output. Inter-call work remains in elapsed.
  auto prefill_seconds = std::chrono::duration<double>(prefill_endpoint - prefill_start).count();
  auto decode_seconds = std::chrono::duration<double>(endpoint - prefill_endpoint).count();
  std::size_t logical_prompt = prompt.size(), turn_count = 1;
  if (!request.turn_observations.empty()) {
    prefill_seconds = 0; decode_seconds = 0; logical_prompt = request.progress.turns().size() - 1; turn_count = request.progress.turns().size();
    for (std::size_t i = 0; i < request.progress.turns().size(); ++i) {
      logical_prompt = cache::add(logical_prompt, request.progress.turns()[i].prompt.size());
      const auto &turn = request.turn_observations[i];
      prefill_seconds += std::chrono::duration<double>(turn.prefill_endpoint - turn.start).count();
      decode_seconds += std::chrono::duration<double>(turn.endpoint - turn.prefill_endpoint).count();
    }
  }
  const auto session_seconds = std::chrono::duration<double>(endpoint - prefill_start).count();
  out << std::setprecision(17)
            << "{\"source_id\":\"" << model.source_identity << "\",\"score_dtype\":\"F32\",\"score_policy\":\"" << model.score_precision_policy()
            << "\",\"mtp\":" << (model.cache_plan.draft_cache ? "true" : "false") << hyle::engine::reporting::qwen35_weight_inventory(model.weight_inventory_facts(), routes)
            << hyle::engine::reporting::qwen35_cache_report(model, packed_prefill_offset) << ",\"visible_tokens\":" << request.visible_tokens;
  write_request_sampling(out, model, request);
  if (model.cache_plan.draft_cache) out << ",\"draft_frontier\":" << request.draft_frontier
      << ",\"draft_lag\":" << request.draft_lag;
  out << ",\"generated_ids\":[";
  for (std::size_t i = 0; i < generated.size(); ++i) out << (i ? "," : "") << generated[i];
  out << "],\"prefill_seconds\":" << prefill_seconds << ",\"decode_seconds\":" << decode_seconds
            << ",\"session_seconds\":" << session_seconds
            << ",\"startup_total_seconds\":" << std::chrono::duration<double>(endpoint - startup).count()
            << ",\"startup_to_endpoint_seconds\":" << std::chrono::duration<double>(endpoint - startup).count()
            << ",\"startup_to_ready_seconds\":" << std::chrono::duration<double>(ready - startup).count()
            << ",\"first_token_elapsed_seconds\":" << std::chrono::duration<double>(prefill_endpoint - startup).count()
            << ",\"prefill_step_seconds\":" << prefill_step_seconds + request.prefix_sampling_seconds
            << ",\"decode_step_sum_seconds\":" << decode_step_sum_seconds
            << ",\"step_sum_seconds\":" << prefill_step_seconds + decode_step_sum_seconds + request.restored.step_seconds + request.prefix_sampling_seconds
            << ",\"prefill_tokens_per_second\":" << (prefill_seconds > 0 ? logical_prompt / prefill_seconds : 0)
            << ",\"decode_output_tokens_per_second\":" << (decode_seconds > 0 ? (generated.size() - turn_count) / decode_seconds : 0)
            << ",\"output_tokens_per_second\":" << (session_seconds > 0 ? generated.size() / session_seconds : 0)
            << ",\"requested_steps\":" << steps << ",\"generated_tokens\":" << generated.size()
            << ",\"generated_decode_tokens\":" << generated.size() - turn_count << ",\"verified_target_rows\":" << verified_rows
            << (request.turn_observations.empty() ? ",\"actual_graph_calls\":" : ",\"graph_participations\":")
            << rounds.size() + request.prefill_graph_calls + request.restored.graph_calls + request.prefix_sampling_calls + request.auxiliary_graph_calls
            << ",\"prefix_sampling_graph_calls\":" << request.prefix_sampling_calls
            << ",\"sampling_stream\":" << request.progress.initial_sampling.stream
            << ",\"sampling_draw_cursor\":" << (request.ledger && request.ledger->committed_count
                ? request.ledger->records[request.ledger->committed_count - 1].after.sampling.draw
                : request.session ? request.session->head->sampling.draw : 0)
            << ",\"prefill_graph_calls\":" << request.prefill_graph_calls << ",\"decode_graph_calls\":" << rounds.size()
            << ",\"prefill_target_rows\":" << request.prefill_graph_tokens
            << ",\"actual_target_rows\":" << request.prefill_graph_tokens + verified_rows + request.restored.target_rows
            << ",\"committed_target_rows\":" << request.visible_tokens
            << ",\"committed_target_decode_rows\":" << request.visible_tokens - logical_prompt;
  if (model.cache_plan.draft_cache) out << ",\"prefill_draft_rows\":" << request.prefill_draft_rows
      << ",\"actual_draft_rows\":" << request.actual_draft_rows()
      << ",\"actual_draft_rows_scope\":\"admitted_active_input_rows\""
      << ",\"draft_rows\":" << draft_rows << ",\"draft_rounds\":" << draft_rounds
      << ",\"agreements\":" << agreements << ",\"consumed_drafts\":" << accepts;
  out << ",\"diagnostic\":" << (trace.empty() ? "false" : "true");
  if (!request.turn_observations.empty()) {
    out << ",\"emitted_pending\":" << generated.back() << ",\"fresh_next_id\":" << request.fresh_next_id;
    print_generation_rebuild(out, request.restored, model.cache_plan.draft_cache);
  }
  if (!request.id.empty())
    out << ",\"logical_prompt_tokens\":" << logical_prompt << ",\"reused_tokens\":" << request.reused_tokens
              << ",\"prefix_hits\":" << request.prefix_hits << ",\"prefix_misses\":" << request.prefix_misses
              << ",\"prefix_full_hits\":" << request.prefix_full_hits;
  if (!request.turn_observations.empty()) {
    out << ",\"between_turn_seconds\":" << (std::max)(0.0, session_seconds - prefill_seconds - decode_seconds)
              << ",\"phase_timing\":\"sum_per_turn_latency_including_interleaving\",\"turns\":[";
    for (std::size_t i = 0; i < request.turn_observations.size(); ++i) {
      const auto &turn = request.turn_observations[i];
      out << (i ? "," : "") << "{\"turn\":" << i + 1 << ",\"requested_steps\":" << request.progress.turns()[i].max_outputs
          << ",\"new_user_tokens\":" << request.progress.turns()[i].prompt.size() << ",\"pending_flushed\":" << (i ? 1 : 0)
          << ",\"context_before\":" << turn.context_before << ",\"context_after\":" << turn.context_after
          << ",\"output_begin\":" << turn.output_begin << ",\"output_end\":" << turn.output_end
          << ",\"prefill_seconds\":" << std::chrono::duration<double>(turn.prefill_endpoint - turn.start).count()
          << ",\"decode_seconds\":" << std::chrono::duration<double>(turn.endpoint - turn.prefill_endpoint).count()
          << ",\"prefill_shape_admission_seconds\":" << turn.prefill_shape_admission_seconds
          << ",\"prefill_phase_prepare_seconds\":" << turn.prefill_phase_prepare_seconds
          << ",\"prefill_slot_initialize_seconds\":" << turn.prefill_slot_initialize_seconds
          << ",\"prefill_phase_creations\":" << turn.prefill_phase_creations
          << ",\"prefill_slot_initializations\":" << turn.prefill_slot_initializations;
      if (turn.arrival != Clock::time_point{})
        out << ",\"arrival_elapsed_seconds\":" << std::chrono::duration<double>(turn.arrival - startup).count()
                  << ",\"request_to_first_token_seconds\":" << std::chrono::duration<double>(turn.prefill_endpoint - turn.arrival).count();
      if (i == 0) out << ",\"after_ready_first_token_seconds\":"
                           << std::chrono::duration<double>(turn.prefill_endpoint - ready).count();
      out << ",\"input_ids\":[";
      for (std::size_t k = 0; k < request.warm_inputs[i]->size(); ++k)
        out << (k ? "," : "") << (*request.warm_inputs[i])[k];
      out << "]}";
    }
    out << ']';
  }
  out << ",\"rounds\":[";
  for (std::size_t i = 0; i < rounds.size(); ++i) {
    const auto &row = rounds[i];
    out << (i ? "," : "") << "{\"pending\":" << row.pending;
    if (!model.cache_plan.draft_cache) {
      out << ",\"next_id\":" << row.next_id << ",\"committed_target_rows\":" << row.processed_target << '}';
      continue;
    }
    out << ",\"draft\":" << row.draft
              << ",\"target_first\":" << row.target_first << ",\"target_second\":" << row.target_second
              << ",\"accepted_draft\":" << (row.accepted_draft ? "true" : "false")
              << ",\"proposals\":" << row.proposal_count << ",\"verified_rows\":" << row.verified_rows
              << ",\"consumed_drafts\":" << row.consumed_drafts << ",\"auxiliary_graph_calls\":" << row.auxiliary_graph_calls
              << ",\"committed_target_rows\":" << row.processed_target << ",\"draft_rows\":" << row.processed_draft << '}';
  }
  out << "]}";
}

struct StartupPrefillWarmup final {
  std::size_t calls = 0, target_rows = 0, draft_rows = 0, physical_rows = 0;
  double seconds = 0;
  bool retired = false;
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
  std::size_t prefill_rows = 0, sampling_index = 0;
  bool mtp = false, decode_overlap = false, ready = false, busy = false, failed = false;
  ReusableGenerationPreparation() = default;
  ReusableGenerationPreparation(const ReusableGenerationPreparation &) = delete;
  ReusableGenerationPreparation &operator=(const ReusableGenerationPreparation &) = delete;
};

// Pure admission checks shared by the model adapter and actual reusable path.
// Inspecting supplied turns never chooses a Graph shape or submits model work.
void validate_reusable_batch(Model &model, inference::GenerationBatch batch, bool mtp,
                             std::size_t configured_prefill) {
  const auto schedules = batch.sessions;
  const auto sampling_indices = batch.sampling_indices;
  const auto stop_policies = batch.stop_policies;
  const auto selected = sampling_indices.empty() ? 0 : sampling_indices.front();
  if (schedules.empty() || schedules.size() > model.cache_plan.resident_sessions ||
      (!sampling_indices.empty() && sampling_indices.size() != schedules.size()) ||
      std::ranges::any_of(sampling_indices, [&](auto index) { return index != selected; }))
    throw std::invalid_argument("reusable preparation requires bounded requests with a prepared sampling configuration");
  if (!configured_prefill || (mtp && configured_prefill < 2))
    throw std::invalid_argument("reusable request exceeds its configured input or output bounds");
  for (const auto &schedule : schedules) {
    if (schedule.turns.empty()) throw std::invalid_argument("reusable request requires an initial turn");
    for (std::size_t turn = 0; turn < schedule.turns.size(); ++turn) {
      const auto &input = schedule.turns[turn];
      if (!input.max_outputs || (turn == 0 && input.prompt.empty()) ||
          input.prompt.size() > model.capacity || (turn != 0 && input.prompt.size() >= model.capacity) ||
          (turn == 0 && input.max_outputs - 1 > model.capacity - input.prompt.size()) ||
          (turn != 0 && input.max_outputs > model.capacity))
        throw std::invalid_argument("reusable request exceeds its configured input or output bounds");
      if (std::ranges::any_of(input.prompt, [&](auto id) { return id < 0 || id >= model.vocabulary; }))
        throw std::invalid_argument("reusable request contains a token outside the model vocabulary");
    }
  }
  if (selected >= model.sampling_configuration_count())
    throw std::invalid_argument("sampling configuration was not cold prepared");
  if (!stop_policies.empty()) {
    if (stop_policies.size() != schedules.size())
      throw std::invalid_argument("one stop policy is required per reusable request");
    for (const auto &policy : stop_policies) policy.validate(model.vocabulary);
  }
}

template<GenerationMode Mode, class BatchObserver = std::nullptr_t, class OutputConsumer = std::nullptr_t>
int run_generation_sessions(Model &model, std::span<const inference::GenerationSessionSchedule> schedules,
    const std::filesystem::path &trace, bool trace_states, Clock::time_point startup,
    Clock::time_point constructed, bool prefix_enabled, std::span<const cache::TokenPlanRef> token_plans,
    std::span<const std::size_t> sampling_indices = {}, std::span<const inference::RequestStopPolicy> stop_policies = {},
    std::size_t max_prefill_rows = SIZE_MAX, bool warmup_planned_shapes = false,
    bool decode_overlap = false, std::span<const inference::LogitsInputProvider> logits_providers = {},
    BatchObserver &&batch_observer = nullptr, bool first_turn_only = false,
    const hyle::engine::reporting::RouteDecisionFacts &routes = {}, OutputConsumer &&output_consumer = nullptr,
    std::ostream &report_out = std::cout, ReusableGenerationPreparation *reuse = nullptr) {
  constexpr bool mtp = Mode == GenerationMode::kMtp;
  const auto selected_sampling = sampling_indices.empty() ? 0 : sampling_indices.front();
  const auto configured_prefill = (std::min)({max_prefill_rows, std::size_t{512}, model.capacity});
  const bool reuse_ready = reuse && reuse->ready;
  if (reuse) {
    if (!first_turn_only || prefix_enabled || warmup_planned_shapes || !logits_providers.empty() ||
        !trace.empty() || trace_states)
      throw std::invalid_argument("reusable preparation requires bounded requests with a prepared sampling configuration");
    validate_reusable_batch(model, {schedules, sampling_indices, stop_policies}, mtp, configured_prefill);
    if (reuse->busy || reuse->failed) throw std::logic_error("reusable preparation is busy or failed");
    if (reuse_ready && (reuse->model != &model || reuse->resources.lock() != model.resources_ ||
        reuse->mtp != mtp || reuse->decode_overlap != decode_overlap || reuse->prefill_rows != configured_prefill ||
        reuse->sampling_index != selected_sampling))
      throw std::invalid_argument("reusable preparation differs from its original model or cold configuration");
    if (model.active_step || model.successor_step || model.carrier_borrowed ||
        model.snapshot_versions.live || model.resources_->pending_readbacks())
      throw std::logic_error("reusable preparation requires actual retirement of the preceding request");
    if (!reuse_ready) {
      reuse->model = &model; reuse->resources = model.resources_; reuse->mtp = mtp;
      reuse->decode_overlap = decode_overlap; reuse->prefill_rows = configured_prefill;
      reuse->sampling_index = selected_sampling;
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
  if (first_turn_only && (prefix_enabled || warmup_planned_shapes ||
      (!reuse && (schedules.size() != 1 || model.cache_plan.resident_sessions != 1 || model.cache_plan.active_sequences != 1)) ||
      model.cache_plan.max_query_rows / model.cache_plan.active_sequences < (reuse ? configured_prefill : model.capacity)))
    throw std::runtime_error("bounded natural generation requires configured query capacity, no prefix cache and lazy shape preparation");
  const auto configured_bound = model.capacity;
  if (first_turn_only) {
    if (model.active_step || model.successor_step || model.carrier_borrowed ||
        (model.execution_workspace.valid() && !reuse_ready))
      throw std::logic_error("bounded prefill policy must be configured before original execution");
    max_prefill_rows = (std::min)({max_prefill_rows, std::size_t{512}, configured_bound});
    if constexpr (mtp) if (max_prefill_rows < 2)
      throw inference::EvaluationFailure(hyle::Status::Unimplemented("bounded MTP prefill requires a configured multirow carrier"));
    model.bounded_prefill_rows = max_prefill_rows;
  }
  const auto configured_record_bound = first_turn_only ? cache::add(cache::multiply(configured_bound, 2), 2) : 0;
  if (first_turn_only)
    std::cerr << "cold_plan policy=configured-capacity visible_turns=1 configured_capacity=" << configured_bound
              << " configured_sessions=" << model.cache_plan.resident_sessions
              << " configured_active_sequences=" << model.cache_plan.active_sequences << " query_capacity=" << model.cache_plan.max_query_rows
              << " configured_prefill_rows=" << max_prefill_rows << " future_shape_hints=0\n";
  model.prepare_default_device_mtp_successor(mtp && decode_overlap && !prefix_enabled && logits_providers.empty());
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
  if (schedules.empty() || model.cache_plan.draft_cache != mtp)
    throw std::runtime_error("natural generation mode differs from its complete model/cache plan");
  if (prefix_enabled && token_plans.size() != schedules.size())
    throw std::runtime_error("natural generation requires one validated prefix input per request");
  const auto prompt_role = [](std::size_t rows, bool lag = false) {
    if constexpr (!mtp) { (void)rows; (void)lag; return GraphRole::kText; }
    else return rows == 1 ? (lag ? GraphRole::kMtpCatchupSingle : GraphRole::kMtpSingle)
        : (lag ? GraphRole::kMtpCatchupPrefill : GraphRole::kMtpPrefill);
  };
  auto prefix = prefix_enabled ? std::make_unique<cache::PrefixCache>(model.cache_plan.resident_sessions) : nullptr;
  std::vector<GenerationRequest> requests;
  requests.reserve(schedules.size());
  std::size_t total_turns = 0, requested_outputs = 0, decode_bound = 0, restore_bound = 0, dispatch_bound = 0;
  for (const auto &schedule : schedules) {
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
    requests.emplace_back(schedule.id, first, trace.empty() ? trace : trace /
        ("session" + std::to_string(requests.size()) + "_" + schedule.id), first_turn_only ? configured_bound : outputs, sampling_indices.empty() ? 0 : sampling_indices[requests.size()],
        stop_policies.empty() ? inference::RequestStopPolicy{} : stop_policies[requests.size()]);
    auto &request = requests.back();
    if (first_turn_only) request.progress.bind_open_turns(visible_turns, configured_bound, configured_bound);
    else request.progress.bind_turns(visible_turns);
    if (!logits_providers.empty()) request.logits_provider = logits_providers[requests.size() - 1];
    request.progress.initial_sampling = {model.sampling_seed, static_cast<std::uint64_t>(requests.size() - 1), 0};
    request.turn_observations.reserve(first_turn_only ? configured_bound : visible_turns.size());
    request.turn_observations.resize(visible_turns.size());
    if (first_turn_only) request.turn_observations[0].arrival = startup;
    request.warm_inputs.reserve(first_turn_only ? configured_bound : visible_turns.size());
    for (std::size_t i = 0; i < visible_turns.size(); ++i) {
      const auto &ids = visible_turns[i].prompt;
      auto owner = std::make_shared<std::vector<std::int32_t>>(cache::add(ids.size(), i ? 1 : 0));
      std::copy(ids.begin(), ids.end(), owner->begin() + (i ? 1 : 0));
      request.warm_inputs.push_back(std::move(owner));
    }
    request.prompt = *request.warm_inputs[0];
    const auto ledger_capacity = first_turn_only ? configured_record_bound
        : generation_record_capacity(visible_turns, max_prefill_rows, first.size());
    request.ledger = std::make_unique<GenerationLedger>(ledger_capacity, model.cache_plan.candidate_boundaries);
    dispatch_bound = cache::add(dispatch_bound, ledger_capacity);
    request.round_emitted.resize((first_turn_only ? configured_bound : outputs) - 1);
    for (auto &emitted : request.round_emitted) emitted.reserve(model.cache_plan.candidate_boundaries);
    if (prefix) request.input = token_plans[requests.size() - 1];
    total_turns = cache::add(total_turns, visible_turns.size());
    requested_outputs = cache::add(requested_outputs, outputs);
    decode_bound = cache::add(decode_bound, first_turn_only ? configured_bound : outputs - visible_turns.size());
    if (model.cache_plan.resident_sessions < schedules.size())
      restore_bound = cache::add(restore_bound, cache::multiply(schedule.turns.size() - 1, ledger_capacity));
  }
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
    if (prefix) for (std::size_t j = 0; j < i; ++j) {
      const auto length = requests[j].prompt.size();
      if (length < requests[i].prompt.size() && cache::same_prefix(*requests[i].input, *requests[j].input, length))
        suffixes.push_back(requests[i].prompt.size() - length);
    }
    std::sort(suffixes.begin(), suffixes.end()); suffixes.erase(std::unique(suffixes.begin(), suffixes.end()), suffixes.end());
    for (auto tokens : suffixes) note_prefill(tokens, i);
    if (prefix && prefill.bounded()) {
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
    if (std::ranges::none_of(requests, [&](const auto &request) { return request.sampling_index == sampling_index; })) continue;
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
        if ((dynamic || model.recursive_mtp()) && std::ranges::any_of(requests, [index, dynamic, first_turn_only](const auto &request) {
            return request.sampling_index == index && bool(request.logits_provider) == dynamic &&
                (first_turn_only || std::ranges::any_of(request.progress.turns(), [](const auto &turn) { return turn.max_outputs > 1; }));
          })) model.prepare_recursive_round_programs(index, dynamic, decode_bound,
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
  auto warmed = prepare_cold_prefill(model, first_turn_only, mtp, requests.front().sampling_index);
  if (warmed) startup_prefill_warmup = *warmed;
  else {
    if (!warmed.error().IsUnimplemented()) check(warmed.error());
    std::cerr << "startup_prefill_warmup skipped=unimplemented reason=" << warmed.error().message() << '\n';
  }
  if constexpr (!mtp) if (first_turn_only)
    (void)model.prepare_prefill_recordings(requests.front().sampling_index);
  }
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
  if (!reuse_ready && decode_overlap && !requests.empty() && decode_bound && model.cache_plan.active_sequences == 1 &&
      (requests.size() == 1 || model.cache_plan.resident_sessions == 1) &&
      (first_turn_only || std::ranges::any_of(requests.front().progress.turns(), [](const auto &turn) { return turn.max_outputs > 1; }))) {
    auto *prepared = prefix_enabled ? nullptr : &requests.front().session;
    if constexpr (mtp) (void)model.prepare_mtp_decode_recordings(requests.front().sampling_index, prepared);
    else (void)model.prepare_ordinary_decode_recordings(requests.front().sampling_index, prepared);
  }
  request_loop.prepare(requests.size(), model.cache_plan.resident_sessions);
  using OutputDeliveries = std::conditional_t<std::is_null_pointer_v<std::remove_cvref_t<OutputConsumer>>,
      std::nullptr_t, std::deque<inference::TokenDelivery>>;
  OutputDeliveries output_deliveries{};
  if constexpr (!std::is_null_pointer_v<std::remove_cvref_t<OutputConsumer>>) {
    static_assert(inference::TokenConsumer<std::remove_reference_t<OutputConsumer>>);
    for (std::size_t index = 0; index < requests.size(); ++index)
      output_deliveries.emplace_back(output_consumer.max_pending());
  }
  for (std::size_t index = 0; index < requests.size(); ++index) {
    if constexpr (std::is_null_pointer_v<std::remove_cvref_t<OutputConsumer>>)
      requests[index].request_key = request_loop.enqueue(index);
    else {
      requests[index].request_key = request_loop.enqueue(index, &output_deliveries[index].window());
      output_deliveries[index].bind(request_loop, requests[index].request_key);
      if constexpr (requires { { output_consumer.start(requests[index].request_key) } noexcept -> std::same_as<bool>; })
        if (!output_consumer.start(requests[index].request_key))
          throw std::runtime_error("output consumer refused its original request identity");
    }
  }
  std::vector<std::size_t> dispatch, dispatch_turns;
  dispatch.reserve(dispatch_bound); dispatch_turns.reserve(dispatch_bound);
  std::vector<std::size_t> evictions; evictions.reserve(first_turn_only ? cache::multiply(configured_bound, requests.size()) : total_turns);
  std::size_t completed = 0, admitted = 0;
  struct BatchObservation {
    std::size_t tokens, active_rows, physical_rows; GraphRole role; bool generation;
    hyle::container::SmallVector<std::size_t, 2> members, input_rows;
  };
  std::vector<BatchObservation> batches; batches.reserve(dispatch_bound);
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
    std::size_t candidates = 0;
    for (const auto *step : {model.active_step.get(), model.successor_step.get()})
      if (step) for (const auto &member : step->members) candidates += bool(member.next);
    const auto live = std::size_t(model.snapshot_versions.live);
    if (candidates > live) throw std::logic_error("generation candidates exceed their actual version ownership");
    const auto committed = live - candidates;
    if (committed > model.cache_plan.resident_sessions)
      throw std::runtime_error("generation committed version count exceeds N after publication");
    peak_resident = (std::max)(peak_resident, resident_count());
    peak_committed = (std::max)(peak_committed, committed);
  };
  const auto observe_batch = [&](const GenerationBatchFacts &facts) {
    const auto participations = cache::add(actual_fresh_participations, facts.members.size());
    auto rows = actual_fresh_target_rows;
    BatchObservation batch{facts.input_rows, facts.active_rows, facts.physical_rows, facts.role, facts.generation, {}, {}};
    if (!batch.members.try_reserve(facts.members.size()) || !batch.input_rows.try_reserve(facts.members.size()))
      throw std::bad_alloc{};
    for (const auto &member : facts.members) {
      const auto key = member.request->request_key;
      if (!request_loop.matches(key) || key.slot >= requests.size() || &requests[key.slot] != member.request)
        throw std::runtime_error("actual batch fact lost its original request identity");
      batch.members.push_back(key.slot);
      const auto member_rows = member.actual_target_rows();
      batch.input_rows.push_back(member_rows); rows = cache::add(rows, member_rows);
    }
    if (rows - actual_fresh_target_rows != facts.active_rows || facts.active_rows > facts.physical_rows)
      throw std::logic_error("actual batch row facts differ from its submitted physical carrier");
    // All vectors have a cold bound from the maximum actual episode count.
    // Prepare the variable-width member storage before committing any report.
    if (batches.size() == batches.capacity() || facts.members.size() > dispatch.capacity() - dispatch.size() ||
        facts.members.size() > dispatch_turns.capacity() - dispatch_turns.size())
      throw std::runtime_error("actual batch facts exceed their cold report capacity");
    batches.push_back(std::move(batch));
    for (const auto &member : facts.members) {
      dispatch.push_back(member.request->request_key.slot);
      dispatch_turns.push_back(member.request->progress.turn_index);
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
      std::cerr << "generation_session id=" << request.id << " turn=" << request.progress.turn_index + 1 << " episode=" << request.rounds.size() << '\n';
      return prepare_generation_call(model, request, request.progress.output_limit(), prefix.get(), requests,
          request_loop.credit(key, request.progress.generated.size()), max_prefill_rows);
    };
    const auto batch_limit = [&](const PreparedGenerationCall &first) -> std::size_t {
      return first.full_hit ? std::size_t{1} : model.cache_plan.active_sequences;
    };
    const auto compatible_calls = [&](const PreparedGenerationCall &first, const PreparedGenerationCall &other) {
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
          make_generation_batch_room(model, prefix.get(), members, requests);
    };
    const auto execute_batch = [&](std::span<PreparedGenerationCall> bucket, bool entire_wave) {
      std::cerr << "generation_batch requests=" << bucket.size() << " T=" << bucket[0].planned_rows()
                << " role=" << static_cast<int>(bucket[0].role) << " full_hit=" << bucket[0].full_hit << '\n';
      try {
        execute_generation_calls(model, bucket, trace_states, prefix.get(), requests,
            nullptr, &continuation, observe_batch, entire_wave, wave.credits);
      } catch (const inference::EvaluationFailure &failure) {
        // Only the real admitted bucket may attribute a native member failure.
        throw inference::RequestExecutionFailure::from_evaluation(failure, bucket.size(),
            [&](std::size_t member) { return bucket[member].request->request_key; });
      }
      auto finished = endpoint;
      for (const auto &call : bucket) {
        const auto index = call.request->request_key.slot;
        if (call.full_hit) {
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
    const auto publish_call = [&](PreparedGenerationCall &call) { publish_generation_prompt(model, call, prefix.get()); };
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
  const auto execute_wave = [&](inference::RequestWave wave, std::span<std::size_t> reserved) {
    return advance_generation_wave(model, wave, reserved, trace_states, continuation, execute_ready, nullptr, observe_batch, prefix.get());
  };
  GenerationRequestAdapter adapter{model, prefix.get(), std::span(requests), trace_states, &evictions, execute_wave,
      {}, &continuation};
  const auto service_output = [&]() {
    bool advanced = false;
    if constexpr (!std::is_null_pointer_v<std::remove_cvref_t<OutputConsumer>>) {
      for (std::size_t index = 0; index < requests.size(); ++index)
        advanced = output_deliveries[index].service(request_loop, adapter, output_consumer) || advanced;
    }
    return advanced;
  };
  while (completed != requests.size()) {
    bool output_progress = service_output();
    const auto pumped = request_loop.pump_once(adapter);
    completed = cache::add(completed, pumped.completed);
    admitted = cache::add(admitted, pumped.admitted);
    output_progress = service_output() || output_progress;
    // This closed CLI has no host work while the current Graph is pending.
    // Any successor is already submitted; promoted B reaches the next pump first
    // so C can overlap it. The original next poll consumes any wait error.
    if (pumped.waiting_retirement && continuation.drive_cursor.awaiting_current &&
        model.active_step && model.active_step->execution.has_completion())
      model.active_step->execution.wait().IgnoreError();
    bool input_arrived = false;
    if (first_turn_only && !pumped.waiting_retirement) for (std::size_t index = 0; index < requests.size(); ++index) {
      auto &request = requests[index];
      if (!request_loop.matches(request.request_key) || request_loop.cancelled(request.request_key) ||
          !request.progress.waiting_for_input()) continue;
      if (model.active_step || model.successor_step || model.carrier_borrowed || !request.session || request.session->pending ||
          (!reuse && !request_loop.active().empty()))
        throw std::logic_error("next input cannot arrive before its original published wave retires");
      const auto arrival = Clock::now();
      const auto visible = request.progress.turns().size();
      // This is workload-driver EOF/ingress, outside every cold-planning loop.
      // No future turn length or quota was supplied to the Model beforehand.
      const auto &source = schedules[index].turns;
      if (visible == source.size()) request.progress.close_input();
      else {
        const auto &next = source[visible];
        if (visible >= configured_bound || next.prompt.size() >= configured_bound ||
            !next.max_outputs || next.max_outputs > configured_bound)
          throw std::out_of_range("input exceeds configured turn, prompt or output bound");
        auto owner = std::make_shared<std::vector<std::int32_t>>(cache::add(next.prompt.size(), 1));
        std::copy(next.prompt.begin(), next.prompt.end(), owner->begin() + 1);
        request.progress.extend_turns(std::span(source).first(visible + 1));
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
    if (!pumped.executed && !pumped.waiting_retirement && !input_arrived && !output_progress && completed != requests.size()) {
      if constexpr (!std::is_null_pointer_v<std::remove_cvref_t<OutputConsumer>>) {
        if (pumped.waiting_output) { output_consumer.wait(); continue; }
      }
      throw std::runtime_error("generation admission has no runnable resident request");
    }
  }
  if (reuse) {
    if (model.active_step || model.successor_step || model.carrier_borrowed ||
        model.snapshot_versions.live || model.resources_->pending_readbacks())
      throw std::logic_error("reusable request has not actually retired its model resources");
    reuse->ready = true; reuse_success = true;
  }
  if constexpr (!std::is_null_pointer_v<std::remove_cvref_t<OutputConsumer>>) {
    const auto cancelled = static_cast<std::size_t>(std::count_if(output_deliveries.begin(), output_deliveries.end(),
        [](const auto &delivery) { return delivery.cancelled(); }));
    if (cancelled) {
      // Planned-turn timing and decode totals are not meaningful for an
      // interrupted workload. Preserve actual outputs without inventing them.
      report_out << "{\"cancelled_requests\":" << cancelled << ",\"sessions\":[";
      for (std::size_t index = 0; index < requests.size(); ++index) {
        report_out << (index ? "," : "") << "{\"slot\":" << index << ",\"cancelled\":"
            << (output_deliveries[index].cancelled() ? "true" : "false") << ",\"generated_ids\":[";
        const auto &ids = requests[index].progress.generated;
        for (std::size_t at = 0; at < ids.size(); ++at) report_out << (at ? "," : "") << ids[at];
        report_out << "]}";
      }
      report_out << "]}\n";
      return 0;
    }
  }
  const auto post_endpoint_seconds = std::chrono::duration<double>(Clock::now() - endpoint).count();
  const auto session_seconds = std::chrono::duration<double>(endpoint - start).count();
  if ((actual_prefill_calls && !(prefill_seconds > 0)) || (actual_decode_calls && !(decode_seconds > 0)))
    throw std::logic_error("an actual generation phase has no observed completion interval");
  std::size_t logical_prompt = 0, user_prompt = 0, prefill_tokens = 0, prefill_draft = 0;
  std::size_t generated = 0, verified = 0, drafted = 0, committed = 0;
  std::size_t draft_rounds = 0, agreements = 0, accepts = 0, hits = 0, misses = 0, full_hits = 0, reused = 0;
  std::size_t restores = 0, recomputed_changes = 0; GenerationRestoreStats restored; double step_sum_seconds = 0;
  for (const auto &request : requests) {
    for (const auto &turn : request.progress.turns()) user_prompt = cache::add(user_prompt, turn.prompt.size());
    prefill_tokens = cache::add(prefill_tokens, request.prefill_graph_tokens);
    prefill_draft = cache::add(prefill_draft, request.prefill_draft_rows);
    generated = cache::add(generated, request.progress.generated.size());
    verified = cache::add(verified, request.verified_rows); drafted = cache::add(drafted, request.draft_rows);
    committed = cache::add(committed, request.visible_tokens); draft_rounds = cache::add(draft_rounds, request.draft_rounds);
    agreements = cache::add(agreements, request.agreements); accepts = cache::add(accepts, request.accepts);
    hits = cache::add(hits, request.prefix_hits); misses = cache::add(misses, request.prefix_misses);
    full_hits = cache::add(full_hits, request.prefix_full_hits); reused = cache::add(reused, request.reused_tokens);
    step_sum_seconds += request.prefill_step_seconds + request.decode_step_sum_seconds;
    restores = cache::add(restores, request.restore_count);
    restored.graph_calls = cache::add(restored.graph_calls, request.restored.graph_calls);
    restored.target_rows = cache::add(restored.target_rows, request.restored.target_rows);
    restored.draft_rows = cache::add(restored.draft_rows, request.restored.draft_rows);
    restored.submitted_draft_rows = cache::add(restored.submitted_draft_rows, request.restored.submitted_draft_rows);
    restored.normalized_lag_rows = cache::add(restored.normalized_lag_rows, request.restored.normalized_lag_rows);
    recomputed_changes = cache::add(recomputed_changes, request.restored.decision_changes.size());
    restored.step_seconds += request.restored.step_seconds; restored.seconds += request.restored.seconds;
  }
  logical_prompt = cache::add(user_prompt, total_turns - requests.size());
  const auto frontier_calls = model.frontier_sampling_graph_calls - frontier_before;
  const auto draft_calls = model.draft_program_graph_calls - draft_before;
  const auto prefix_calls = model.prefix_decision_graph_calls - prefix_before;
  const auto graph_calls = cache::add(cache::add(cache::add(actual_prefill_calls, actual_decode_calls), restored.graph_calls),
      cache::add(frontier_calls, cache::add(draft_calls, prefix_calls)));
  report_out << std::setprecision(17) << "{\"source_id\":\"" << model.source_identity << "\",\"score_dtype\":\"F32\",\"score_policy\":\"" << model.score_precision_policy()
      << "\",\"mtp\":" << (model.cache_plan.draft_cache ? "true" : "false") << hyle::engine::reporting::qwen35_weight_inventory(model.weight_inventory_facts(), routes)
      << (reuse_ready ? ",\"model_counters_scope\":\"lifetime_except_packed_prefill_and_aggregate_calls\"" : "")
      << ",\"session_count\":" << requests.size() << ",\"turn_count\":" << total_turns << ",\"active_sequences\":" << model.cache_plan.active_sequences << ",\"max_sessions\":" << model.cache_plan.resident_sessions
      << ",\"decode_overlap\":" << (decode_overlap ? "true" : "false")
      << ",\"decode_overlap_available\":" << (decode_overlap && !continuation.unavailable ? "true" : "false")
      << ",\"overlap_successor_submissions\":" << continuation.successor_submissions
      << ",\"overlap_successor_publications\":" << continuation.successor_publications
      << ",\"overlap_successor_discards\":" << continuation.successor_discards
      << ",\"overlap_successor_refusals\":" << continuation.successor_refusals
      << hyle::engine::reporting::qwen35_cache_report(model, packed_prefill_offset) << ",\"prefix_cache\":" << (prefix_enabled ? "true" : "false")
      << ",\"admitted_requests\":" << requests.size() << ",\"admitted_turns\":" << admitted << ",\"retired_requests\":" << completed
      << ",\"peak_resident_requests\":" << peak_resident << ",\"peak_committed_versions\":" << peak_committed
      << ",\"live_committed_versions\":" << model.snapshot_versions.live << ",\"idle_evictions\":" << evictions.size()
      << ",\"prefix_hits\":" << hits << ",\"prefix_misses\":" << misses << ",\"prefix_full_hits\":" << full_hits << ",\"reused_tokens\":" << reused
      << ",\"prefix_entries\":" << (prefix ? prefix->size() : 0) << ",\"prefix_evictions\":" << (prefix ? prefix->evictions() : 0)
      << ",\"requested_output_tokens\":" << requested_outputs << ",\"generated_tokens\":" << generated << ",\"generated_decode_tokens\":" << generated - total_turns
      << ",\"actual_graph_calls\":" << graph_calls << ",\"prefill_graph_calls\":" << actual_prefill_calls << ",\"decode_graph_calls\":" << actual_decode_calls
      << ",\"graph_participations\":" << actual_fresh_participations + restored.graph_calls + frontier_calls + draft_calls + prefix_calls
      << ",\"cold_planning\":\"" << (first_turn_only ? "configured_capacity" : "full_schedule_preknowledge_invalid")
      << "\",\"future_shape_hints\":" << (first_turn_only ? 0 : 1)
      << ",\"configured_prefill_rows\":" << model.bounded_prefill_rows
      << ",\"shared_execution_scratch_bytes\":" << model.execution_workspace.bytes()
      << ",\"shared_prefill_scratch_bytes\":" << model.execution_workspace.bytes()
      << ",\"cold_prepare_seconds\":" << cold_prepare_seconds
      << ",\"startup_prefill_warmup_calls\":" << startup_prefill_warmup.calls
      << ",\"startup_prefill_warmup_target_rows\":" << startup_prefill_warmup.target_rows
      << ",\"startup_prefill_warmup_draft_rows\":" << startup_prefill_warmup.draft_rows
      << ",\"startup_prefill_warmup_physical_rows\":" << startup_prefill_warmup.physical_rows
      << ",\"startup_prefill_warmup_retired\":" << (startup_prefill_warmup.retired ? "true" : "false")
      << ",\"startup_prefill_warmup_seconds\":" << startup_prefill_warmup.seconds
      << ",\"logical_dispatches\":" << dispatch.size() << ",\"logical_prompt_tokens\":" << logical_prompt << ",\"logical_user_prompt_tokens\":" << user_prompt
      << ",\"pending_flushed_tokens\":" << total_turns - requests.size() << ",\"prefill_target_rows\":" << prefill_tokens
      << ",\"verified_target_rows\":" << verified
      << ",\"actual_target_rows\":" << cache::add(actual_fresh_target_rows, restored.target_rows)
      << ",\"committed_target_rows\":" << committed << ",\"committed_target_decode_rows\":" << committed - logical_prompt
      << ",\"restore_count\":" << restores << ",\"restored_graph_calls\":" << restored.graph_calls << ",\"restored_target_rows\":" << restored.target_rows
      << ",\"restore_step_seconds\":" << restored.step_seconds << ",\"restore_seconds\":" << restored.seconds
      << ",\"restore_contract\":\"recompute_committed_inputs\"";
  if constexpr (mtp) report_out << ",\"prefill_draft_rows\":" << prefill_draft << ",\"draft_rows\":" << drafted
      << ",\"actual_draft_rows\":" << model.submitted_draft_rows - submitted_draft_begin
      << ",\"actual_draft_rows_scope\":\"admitted_active_input_rows\""
      << ",\"draft_rounds\":" << draft_rounds << ",\"agreements\":" << agreements << ",\"consumed_drafts\":" << accepts
      << ",\"restored_draft_rows\":" << restored.draft_rows
      << ",\"actual_restored_draft_rows\":" << restored.submitted_draft_rows
      << ",\"normalized_lag_rows\":" << restored.normalized_lag_rows;
  report_out << ",\"recomputed_decision_changes\":" << recomputed_changes
      << ",\"prefill_seconds\":" << prefill_seconds << ",\"decode_seconds\":" << decode_seconds << ",\"session_seconds\":" << session_seconds
      << ",\"step_sum_seconds\":" << actual_step_sum + restored.step_seconds + model.frontier_sampling_seconds - frontier_seconds_before
      << ",\"participation_step_sum_seconds\":" << step_sum_seconds << ",\"post_endpoint_seconds\":" << post_endpoint_seconds
      << ",\"phase_timing\":\"disjoint_continuous_episode_intervals\",\"startup_total_seconds\":" << std::chrono::duration<double>(endpoint - startup).count()
      << ",\"startup_to_endpoint_seconds\":" << std::chrono::duration<double>(endpoint - startup).count() << ",\"startup_to_ready_seconds\":" << std::chrono::duration<double>(ready - startup).count()
      << ",\"first_token_elapsed_seconds\":" << std::chrono::duration<double>(requests.front().turn_observations.front().prefill_endpoint - startup).count()
      << ",\"prefill_tokens_per_second\":" << (prefill_seconds > 0 ? logical_prompt / prefill_seconds : 0)
      << ",\"decode_output_tokens_per_second\":" << (decode_seconds > 0 ? (generated - total_turns) / decode_seconds : 0)
      << ",\"output_tokens_per_second\":" << (session_seconds > 0 ? generated / session_seconds : 0)
      << ",\"diagnostic\":" << (trace.empty() ? "false" : "true") << ",\"dispatch\":[";
  for (std::size_t i = 0; i < dispatch.size(); ++i)
    report_out << (i ? "," : "") << "{\"session_id\":\"" << requests[dispatch[i]].id << "\",\"turn\":" << dispatch_turns[i] + 1 << '}';
  report_out << "],\"batches\":[";
  for (std::size_t i = 0; i < batches.size(); ++i) {
    const auto &batch = batches[i];
    report_out << (i ? "," : "") << "{\"index\":" << i << ",\"requests\":" << batch.members.size()
        << ",\"tokens\":" << batch.tokens << ",\"role\":" << static_cast<int>(batch.role)
        << ",\"generation\":" << (batch.generation ? "true" : "false") << ",\"session_ids\":[";
    for (std::size_t j = 0; j < batch.members.size(); ++j)
      report_out << (j ? "," : "") << '"' << requests[batch.members[j]].id << '"';
    report_out << ']';
    report_out << ",\"input_rows\":[";
    for (std::size_t j = 0; j < batch.input_rows.size(); ++j) report_out << (j ? "," : "") << batch.input_rows[j];
    report_out << "],\"active_rows\":" << batch.active_rows << ",\"physical_rows\":" << batch.physical_rows << '}';
  }
  report_out << "],\"sessions\":[";
  for (std::size_t i = 0; i < requests.size(); ++i) {
    auto &request = requests[i]; std::size_t quota = 0;
    for (const auto &turn : request.progress.turns()) quota = cache::add(quota, turn.max_outputs);
    report_out << (i ? "," : "") << "{\"session_id\":\"" << request.id << "\",\"result\":";
    print_generation_request(report_out, model, request, quota, startup, ready, packed_prefill_offset, routes);
    report_out << ",\"restore_count\":" << request.restore_count << ",\"idle_evictions\":" << request.idle_evictions
        << ",\"checkpoint_resume_hits\":" << request.checkpoint_resume_hits
        << ",\"checkpoint_resume_misses\":" << request.checkpoint_resume_misses << ",\"ledger\":[";
    std::size_t round = 0;
    for (std::size_t j = 0; j < request.ledger->committed_count; ++j) {
      const auto &record = request.ledger->records[j]; if (record.generation) ++round;
      std::uint64_t hash = 1469598103934665603ULL;
      for (const auto token : record.ids()) hash = (hash ^ static_cast<std::uint32_t>(token)) * 1099511628211ULL;
      const auto prefix_name = record.turn ? "round" + std::to_string(record.turn + 1) + "_" : "";
      const auto name = prefix_name + (record.generation ? (mtp ? "mtp" : "decode") + std::to_string(round) : "prefill");
      report_out << (j ? "," : "") << "{\"index\":" << j << ",\"turn\":" << record.turn + 1 << ",\"role\":" << static_cast<int>(record.role)
          << ",\"requests\":" << record.requests << ",\"member_index\":" << record.member_index
          << ",\"generation\":" << (record.generation ? "true" : "false") << ",\"input_rows\":" << record.ids().size()
          << ",\"verified_rows\":" << record.verified_rows << ",\"draft_limit\":" << record.draft_limit
          << ",\"consumed_drafts\":" << record.consumed_drafts << ",\"auxiliary_graph_calls\":" << record.auxiliary_graph_calls
          << ",\"input_hash\":" << hash << ",\"first_input\":" << record.ids().front() << ",\"quota\":" << record.quota
          << ",\"before_length\":" << record.before.length << ",\"before_frontier\":" << record.before.frontier << ",\"before_lag\":" << record.before.lag
          << ",\"before_next_id\":" << record.before.next_id << ",\"draft\":" << record.draft << ",\"target_first\":" << record.target_first << ",\"target_second\":" << record.target_second
          << ",\"sampling_seed\":" << record.before.sampling.seed << ",\"sampling_stream\":" << record.before.sampling.stream
          << ",\"sampling_draw_before\":" << record.before.sampling.draw << ",\"sampling_draw_after\":" << record.after.sampling.draw
          << ",\"accepted\":" << record.accepted << ",\"draft_rows\":" << record.draft_rows << ",\"after_length\":" << record.after.length
          << ",\"after_frontier\":" << record.after.frontier << ",\"after_lag\":" << record.after.lag << ",\"next_id\":" << record.after.next_id
          << ",\"committed_input_ids\":[";
      if (record.generation) for (std::size_t k = 0; k < record.accepted; ++k)
        report_out << (k ? "," : "") << record.accepted_inputs[k];
      report_out << "],\"verification_input_ids\":[";
      for (std::size_t k = 0; k < record.verification_inputs.size(); ++k)
        report_out << (k ? "," : "") << record.verification_inputs[k];
      report_out << "],\"imported\":" << (record.imported ? "true" : "false")
          << ",\"source_geometry\":\"" << (record.imported ? "unknown" : "recorded")
          << "\",\"trace_name\":\"" << (record.imported ? "" : name) << "\"}";
    }
    report_out << "]}";
  }
  report_out << "]}\n"; return 0;
}
// Internal bridge from a prepared Qwen Model to the common finite-cohort
// contract. The original Model outlives this object and remains the sole
// owner of its weights, sessions, native work and retirement. It deliberately
// does not expose Qwen's state or legacy JSON report to the common interface.
class PreparedModelExecutor final : public inference::ModelExecutor {
public:
  PreparedModelExecutor(Model &model, bool mtp, std::size_t prefill_rows, bool overlap)
      : model_(model), mtp_(mtp), overlap_(overlap), prefill_rows_(prefill_rows) {}
  void validate(inference::GenerationBatch batch) override {
    if (running_.test(std::memory_order_acquire) || preparation_.busy || preparation_.failed)
      throw std::logic_error("prepared model executor is busy or failed");
    if (model_.cache_plan.draft_cache != mtp_)
      throw std::logic_error("prepared model mode differs from its cache plan");
    try {
      const auto prefill = (std::min)({prefill_rows_, std::size_t{512}, model_.capacity});
      validate_reusable_batch(model_, batch, mtp_, prefill);
      const auto selected = batch.sampling_indices.empty() ? 0 : batch.sampling_indices.front();
      if (preparation_.ready && preparation_.sampling_index != selected)
        throw std::invalid_argument("sampling configuration differs from the prepared executor");
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
    class NullBuffer final : public std::streambuf {
      int_type overflow(int_type ch) override { return traits_type::not_eof(ch); }
      std::streamsize xsputn(const char *, std::streamsize count) override { return count; }
    } buffer;
    std::ostream ignored_report(&buffer);
    const std::array<cache::TokenPlanRef, 0> plans{};
    const auto now = Clock::now();
    const auto result = mtp_
        ? run_generation_sessions<GenerationMode::kMtp>(model_, batch.sessions, {}, false, now, now, false,
            plans, batch.sampling_indices, batch.stop_policies, prefill_rows_, false, overlap_, {}, nullptr,
            true, {}, consumer, ignored_report, &preparation_)
        : run_generation_sessions<GenerationMode::kText>(model_, batch.sessions, {}, false, now, now, false,
            plans, batch.sampling_indices, batch.stop_policies, prefill_rows_, false, overlap_, {}, nullptr,
            true, {}, consumer, ignored_report, &preparation_);
    if (result != 0) throw std::runtime_error("prepared model execution returned a failure");
  }
private:
  Model &model_;
  ReusableGenerationPreparation preparation_;
  bool mtp_, overlap_;
  std::size_t prefill_rows_;
  std::atomic_flag running_ = ATOMIC_FLAG_INIT;
};
std::unique_ptr<inference::ModelExecutor> make_prepared_executor(
    Model &model, bool mtp, std::size_t prefill_rows, bool overlap) {
  return std::make_unique<PreparedModelExecutor>(model, mtp, prefill_rows, overlap);
}

class LoadedModelExecutor final : public inference::ModelExecutor {
public:
  LoadedModelExecutor(std::unique_ptr<Model> model, bool mtp, std::size_t prefill_rows, bool overlap)
      : model_(std::move(model)), prepared_(*model_, mtp, prefill_rows, overlap) {}
  void validate(inference::GenerationBatch batch) override { prepared_.validate(batch); }
  void execute(inference::GenerationBatch batch, inference::TokenConsumerRef &consumer) override {
    prepared_.execute(batch, consumer);
  }
private:
  // The adapter is destroyed before the sole original model/resource owner.
  std::unique_ptr<Model> model_;
  PreparedModelExecutor prepared_;
};

std::unique_ptr<inference::ModelExecutor> load_model_executor(const ModelLoadConfig &config) {
  if (config.inputs.empty() || !config.context || !config.resident_sessions || !config.active_sequences ||
      config.active_sequences > config.resident_sessions || !config.prefill_rows || config.draft_count >= config.context)
    throw std::invalid_argument("model loading requires valid context, sequence, prefill and draft capacities");
  const bool mtp = config.draft_count != 0;
  if (!mtp && (!config.draft_vocabulary.empty() || !config.draft_head.empty() || config.auto_draft_head))
    throw std::invalid_argument("draft assets require enabled drafting");
  const auto prefill = (std::min)({config.prefill_rows, std::size_t{512}, config.context});
  if (mtp && prefill < 2) throw std::invalid_argument("MTP preparation needs a multirow prefill capacity");
  auto source = prepare_weight_source(config.inputs / "weights.tsv", mtp, config.draft_vocabulary,
      config.draft_head, config.auto_draft_head);
  const auto geometry = source.roles.geometry;
  validate_mtp_source(geometry, mtp);
  if (config.top_k > static_cast<std::size_t>(geometry.vocabulary))
    throw std::invalid_argument("sampling top-k exceeds the source vocabulary");
  const auto precision = PrecisionProfile::from_roles(source.roles);
  const auto query_rows = (std::max)(prefill, mtp ? cache::add(config.draft_count, 1) : std::size_t{1});
  const auto plan = cache::plan(geometry, source.cache_row_bytes, config.context, config.resident_sessions,
      query_rows, recurrent_bytes(state_schema_for(geometry, config.context, precision), geometry),
      config.active_sequences, mtp ? cache::add(config.draft_count, 1) : 1, mtp, config.random,
      config.overlap ? 2 : 1, 0, precision);
  const Model::Sampling sampling{.enabled = config.random,
      .policy = {config.temperature, config.top_k, config.top_p}, .seed = config.seed};
  auto model = std::make_unique<Model>(std::move(source), geometry, config.context, plan,
      std::filesystem::path{}, false, ReplayPolicy::kAuto, SIZE_MAX, sampling, 0,
      BackendSelection{.primary = config.backend});
  if (mtp) model->configure_draft_policy({config.draft_count});
  return std::make_unique<LoadedModelExecutor>(std::move(model), mtp, config.prefill_rows, config.overlap);
}
} // namespace qwen35

int qwen35::run(int argc, char **argv) {
  if (argc == 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
    std::cout << "Hyle inference engine (Qwen text)\n"
        "Usage: hyle --inputs DIR [options]\n"
        "  --inputs DIR                Locally prepared checkpoint and token IDs\n"
        "  --backend cpu|cuda|apple|vulkan  Primary backend; required when multiple GPU routes are compiled\n"
        "  --mtp auto|off|on           Auto: prepared/explicit assets or eligible CUDA GGUF full-head K1\n"
        "  --mtp-max-drafts N          Maximum proposals; 0 disables drafting (default: artifact recommendation or 1)\n"
        "  --mtp-draft-vocabulary auto|full|FILE  Prepared draft IDs (default: auto); greedy only\n"
        "  --mtp-draft-head auto|original|FILE  Prepared proposal head (default: auto); greedy only\n"
        "  --sampling greedy|random    Token selection (default: greedy)\n"
        "  --temperature FLOAT         Random policy temperature > 0 (default: 1)\n"
        "  --top-k N                    Random policy top-k; 0 disables (default: 0)\n"
        "  --top-p FLOAT                Random policy nucleus in (0,1] (default: 1)\n"
        "  --seed N                     Unsigned 64-bit random seed (default: 0)\n"
        "  --steps N                   Maximum output tokens (default: 4)\n"
        "  --capacity N                Total context capacity (default: 128)\n"
        "  --prefill-chunk-tokens N    Maximum real rows per natural prefill episode\n"
        "  --cpu-ffn-layers N          First N target FFNs on CPU; needs CPU+GPU build (default: 0)\n"
        "  --schedule FILE             One session's prompt/generate turns\n"
        "  --session-schedules FILE    Multiple named session schedules\n"
        "  --max-sessions auto|N       Bounded retained session states\n"
        "  --active-sequences auto|N   Maximum actual execution batch width\n"
        "  --cache-bytes auto|N        Session cache backing budget\n"
        "  --prefix-cache off|auto     Reuse completed prompt/turn checkpoints\n"
        "Advanced execution and diagnostics:\n"
        "  --warmup-shapes lazy|planned  Prepare known carriers before ready (default: lazy)\n"
        "  --progress direct|scheduler  Request progress implementation (default: direct)\n"
        "  --replay auto|commands|prefer-capture\n"
        "  --require-bounded-generation on|off  Reject routes that need future request shapes (default: off)\n"
        "  --decode-overlap auto|off   Automatic route or disable overlap\n"
        "  --trace DIR                 Write generation trace to a local directory\n"
        "  --trace-level full|decisions  State and decision trace, or decisions only\n"
        "  --preparation-diagnostics 0|1  Report preparation diagnostics for fixed replay schedules\n"
        "  --layer0-probe DIR          Write layer-zero probe results\n"
        "  --continuation IDS          Append fixed token IDs before fixed replay\n"
        "  --decode-ids IDS            Fixed replay token IDs\n"
        "Prepare inputs with engine/models/qwen35/prepare.py; no model is downloaded at runtime.\n";
    return 0;
  }
  const auto startup = qwen35::Clock::now();
  try {
    std::filesystem::path input, continuation, trace, layer0_probe, decode_ids, schedule_file, session_file;
    std::filesystem::path draft_vocabulary = "auto";
    std::filesystem::path draft_head="auto";
    std::size_t steps = 4, capacity = 128, max_sessions = 0, active_sequences = 0, cache_bytes = 0, max_drafts = 1, cpu_ffn_layers = 0;
    std::size_t max_prefill_rows = SIZE_MAX;
    bool warmup_planned_shapes = false, decode_overlap = true, require_bounded_generation = false;
    bool progress = false, trace_states = true, explicit_steps = false;
    bool mtp_enabled = false;
    std::string mtp_policy = "auto";
    bool explicit_mtp_drafts = false;
    qwen35::Model::Sampling sampler;
    bool explicit_sampling_parameters = false;
    bool single_schedule = false, multiple_schedules = false;
    std::string prefix_policy;
    bool explicit_continuation = false, explicit_decode_ids = false, preparation_diagnostics = false;
    auto replay = qwen35::ReplayPolicy::kAuto;
    qwen35::BackendSelection backend_selection;
    const auto positive_size = [](const std::string &option, const std::string &value) {
      std::size_t result = 0;
      const auto converted = std::from_chars(value.data(), value.data() + value.size(), result);
      if (converted.ec != std::errc{} || converted.ptr != value.data() + value.size() || result == 0)
        throw std::runtime_error(option + " requires a positive integer");
      return result;
    };
    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      if (i + 1 >= argc) throw std::runtime_error("option needs a value");
      const std::string value = argv[++i];
      if (option == "--inputs") input = value;
      else if (option == "--prefill-chunk-tokens") max_prefill_rows = positive_size(option, value);
      else if (option == "--require-bounded-generation") {
        if (value != "on" && value != "off")
          throw std::runtime_error("--require-bounded-generation requires on or off");
        require_bounded_generation = value == "on";
      }
      else if (option == "--decode-overlap") {
        // Internal benchmark/diagnostic override. Normal use selects the
        // automatic policy and falls back using actual execution facts.
        if (value != "off" && value != "auto") throw std::runtime_error("--decode-overlap requires off or auto");
        decode_overlap = value == "auto";
      }
      else if (option == "--warmup-shapes") {
        if (value != "lazy" && value != "planned")
          throw std::runtime_error("--warmup-shapes requires lazy or planned");
        warmup_planned_shapes = value == "planned";
      }
      else if (option == "--backend") {
        if (value == "cpu") backend_selection.primary = qwen35::BackendKind::kCpu;
        else if (value == "cuda") backend_selection.primary = qwen35::BackendKind::kCuda;
        else if (value == "apple") backend_selection.primary = qwen35::BackendKind::kApple;
        else if (value == "vulkan") backend_selection.primary = qwen35::BackendKind::kVulkan;
        else throw std::runtime_error("--backend requires cpu, cuda, apple or vulkan");
      }
      else if (option == "--sampling") {
        if (value != "greedy" && value != "random") throw std::runtime_error("--sampling requires greedy or random");
        sampler.enabled = value == "random";
      }
      else if (option == "--temperature" || option == "--top-p") {
        float number = 0;
        std::istringstream parsed(value); parsed.imbue(std::locale::classic());
        parsed >> std::noskipws >> number;
        if (!parsed || parsed.peek() != std::char_traits<char>::eof() || !std::isfinite(number) ||
            number <= 0 || (option == "--top-p" && number > 1))
          throw std::runtime_error(option + " requires a finite positive number" + (option == "--top-p" ? " at most one" : ""));
        if (option == "--temperature") sampler.policy.temperature = number; else sampler.policy.top_p = number;
        explicit_sampling_parameters = true;
      }
      else if (option == "--top-k" || option == "--seed") {
        std::uint64_t number = 0;
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), number);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
            (option == "--top-k" && number > INT32_MAX))
          throw std::runtime_error(option + " requires an unsigned integer" + (option == "--top-k" ? " no greater than the vocabulary" : ""));
        if (option == "--seed") sampler.seed = number; else sampler.policy.top_k = static_cast<std::size_t>(number);
        explicit_sampling_parameters = true;
      }
      else if (option == "--mtp") {
        if (value != "auto" && value != "on" && value != "off") throw std::runtime_error("--mtp requires auto, on or off");
        mtp_policy = value;
      }
      else if (option == "--mtp-max-drafts") {
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), max_drafts);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
          throw std::runtime_error("--mtp-max-drafts requires a nonnegative integer within the supported size range");
        explicit_mtp_drafts = true;
      }
      else if (option == "--mtp-draft-vocabulary") {
        if (value.empty()) throw std::runtime_error("--mtp-draft-vocabulary requires a nonempty file path");
        draft_vocabulary = value;
      }
      else if(option=="--mtp-draft-head"){
        if(value.empty())throw std::runtime_error("--mtp-draft-head requires a nonempty file path");
        draft_head=value;
      }
      else if (option == "--cpu-ffn-layers") {
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), cpu_ffn_layers);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
          throw std::runtime_error("--cpu-ffn-layers requires a nonnegative integer within the supported size range");
      }
      else if (option == "--continuation") { continuation = value; explicit_continuation = true; }
      else if (option == "--schedule") { schedule_file = value; single_schedule = true; }
      else if (option == "--session-schedules") { session_file = value; multiple_schedules = true; }
      else if (option == "--max-sessions") max_sessions = value == "auto" ? 0 : positive_size(option, value);
      else if (option == "--active-sequences") active_sequences = value == "auto" ? 0 : positive_size(option, value);
      else if (option == "--cache-bytes") cache_bytes = value == "auto" ? 0 : positive_size(option, value);
      else if (option == "--prefix-cache") {
        if (value != "off" && value != "auto") throw std::runtime_error("--prefix-cache requires off or auto");
        prefix_policy = value;
      }
      else if (option == "--trace") trace = value;
      else if (option == "--preparation-diagnostics") {
        if (value != "0" && value != "1") throw std::runtime_error("--preparation-diagnostics requires 0 or 1");
        preparation_diagnostics = value == "1";
      }
      else if (option == "--trace-level") {
        if (value != "full" && value != "decisions")
          throw std::runtime_error("--trace-level requires full or decisions");
        trace_states = value == "full";
      }
      else if (option == "--decode-ids") { decode_ids = value; explicit_decode_ids = true; }
      else if (option == "--layer0-probe") layer0_probe = value;
      else if (option == "--steps") { steps = positive_size(option, value); explicit_steps = true; }
      else if (option == "--capacity") capacity = positive_size(option, value);
      else if (option == "--progress") {
        if (value != "direct" && value != "scheduler")
          throw std::runtime_error("--progress requires direct or scheduler");
        progress = value == "scheduler";
      }
      else if (option == "--replay") {
        if (value != "auto" && value != "commands" && value != "prefer-capture")
          throw std::runtime_error("--replay requires auto, commands or prefer-capture");
        replay = value == "auto" ? qwen35::ReplayPolicy::kAuto
            : value == "prefer-capture" ? qwen35::ReplayPolicy::kPreferCapture
                                         : qwen35::ReplayPolicy::kCommands;
      }
      else throw std::runtime_error("unknown option: " + option);
    }
    const bool scheduled = single_schedule || multiple_schedules;
    if (input.empty()) throw std::runtime_error("--inputs DIR is required");
    backend_selection = qwen35::select_execution(backend_selection, cpu_ffn_layers);
    const bool prefix_enabled = prefix_policy == "auto" || (prefix_policy.empty() && multiple_schedules);
    using Grammar = qwen35::benchmark::ScheduleGrammar;
    const auto grammar = single_schedule ? qwen35::benchmark::schedule_grammar(schedule_file)
        : multiple_schedules ? qwen35::benchmark::session_schedule_grammar(session_file,
            std::vector<qwen35::benchmark::SessionManifestEntry>{}.max_size()) : Grammar::kPromptOnly;
    const bool natural_schedule = scheduled && grammar != Grammar::kFixedInput;
    const bool natural_single = !scheduled && !explicit_continuation && !explicit_decode_ids &&
        layer0_probe.empty() && !preparation_diagnostics;
    const bool natural_generation = natural_schedule || natural_single;
    const bool unmodified_greedy = !sampler.enabled && sampler.modifiers.empty();
    const hyle::inference::ColdWorkloadFacts cold_workload{
        .natural_generation = natural_generation, .multiple_schedules = multiple_schedules,
        .prefix_enabled = prefix_enabled, .scheduler_progress = progress,
        .planned_shapes = warmup_planned_shapes, .cpu_assist = cpu_ffn_layers != 0,
        .greedy = unmodified_greedy,
        .state_trace = !trace.empty() && trace_states,
        .max_sessions = max_sessions, .active_sequences = active_sequences, .capacity = capacity};
    // A source-bound vocabulary opts a prepared directory into drafting.
    // The policy inspects current configuration, never future requests.
    namespace mtp = hyle::inference::mtp;
    const mtp::DefaultRequest default_request{
        .mode = mtp_policy == "off" ? mtp::RequestedMode::kOff :
            mtp_policy == "on" ? mtp::RequestedMode::kOn : mtp::RequestedMode::kAuto,
        .workload = cold_workload,
        .automatic_backend_eligible = backend_selection.primary == qwen35::BackendKind::kCuda,
        .max_drafts = max_drafts, .automatic_vocabulary = draft_vocabulary == "auto"};
    auto assets = qwen35::select_draft_assets(input, default_request,
        std::move(draft_vocabulary), std::move(draft_head),
        qwen35::Backend::supports_encoded_row_gather(backend_selection));
    const auto default_selection = assets.policy;
    mtp_enabled = default_selection.enabled;
    auto weight_source = qwen35::prepare_weight_source(input / "weights.tsv", mtp_enabled,
        assets.vocabulary, assets.head, assets.auto_head);
    const auto depth_selection = mtp::select_draft_depth(default_selection, explicit_mtp_drafts,
        max_drafts, weight_source.draft_preferences.recommended_drafts);
    max_drafts = depth_selection.max_drafts;
    const auto geometry = weight_source.roles.geometry;
    qwen35::validate_mtp_source(geometry, mtp_enabled);
    qwen35::FfnPlacement::validate(cpu_ffn_layers, geometry.layers.size());
    if (sampler.policy.top_k > static_cast<std::size_t>(geometry.vocabulary))
      throw std::runtime_error("--top-k exceeds the source vocabulary");
    if (explicit_mtp_drafts && max_drafts != 0 && !mtp_enabled)
      throw std::runtime_error("--mtp-max-drafts requires --mtp on or prepared automatic MTP");
    if (explicit_sampling_parameters && !sampler.enabled)
      throw std::runtime_error("--temperature, --top-k, --top-p and --seed require --sampling random");
    if (sampler.enabled && (!layer0_probe.empty() || preparation_diagnostics))
      throw std::runtime_error("random sampling does not support layer or preparation diagnostics");
    if (mtp_enabled) {
      if (explicit_continuation || explicit_decode_ids || !layer0_probe.empty())
        throw std::runtime_error("--mtp on uses natural generation; --continuation, --decode-ids and --layer0-probe are not supported");
      if (preparation_diagnostics)
        throw std::runtime_error("--preparation-diagnostics is not supported with --mtp on");

    }
    if (prefix_enabled && !scheduled) throw std::runtime_error("--prefix-cache auto requires a schedule");
    if (single_schedule && multiple_schedules)
      throw std::runtime_error("--schedule and --session-schedules are exclusive");
    // The default single-session route admits later turns only after the
    // previous published wave retires. Auto N/R are configuration bounds of
    // one here; future schedule lengths and quotas cannot select this policy.
    // Other backends, sampling, multi-session/prefix and explicit
    // planned workloads retain their finite route and invalid-performance label.
    const auto bounded_plan = hyle::inference::select_bounded_generation({
        .workload = cold_workload,
        .supported_backend = backend_selection.primary == qwen35::BackendKind::kCuda,
        .mtp_enabled = mtp_enabled, .max_drafts = max_drafts});
    const bool bounded_generation = bounded_plan.enabled;
    if (require_bounded_generation && !bounded_generation)
      throw std::runtime_error("bounded natural generation unavailable: " + std::string(inference::route_reason_name(bounded_plan.reason)));
    if (max_prefill_rows != SIZE_MAX && !natural_generation)
      throw std::runtime_error("--prefill-chunk-tokens requires natural generation; fixed replay is not supported");
    if (mtp_enabled && grammar == Grammar::kFixedInput)
      throw std::runtime_error("--mtp on requires natural prompt/generate requests; decode IDSFILE is fixed replay");
    if (natural_schedule && (!layer0_probe.empty() || preparation_diagnostics))
      throw std::runtime_error("natural requests do not support layer or preparation diagnostics");
    if (scheduled && (explicit_continuation || explicit_decode_ids || (explicit_steps && !natural_schedule)))
      throw std::runtime_error("schedule inputs are exclusive with continuation, decode IDs and fixed-replay --steps");
    if ((single_schedule && schedule_file.empty()) || (multiple_schedules && session_file.empty()))
      throw std::runtime_error("schedule requires a nonempty manifest path");
    if (preparation_diagnostics && !scheduled)
      throw std::runtime_error("--preparation-diagnostics requires a fixed replay schedule");
    std::vector<qwen35::benchmark::SessionSchedule> schedules;
    std::vector<hyle::inference::GenerationSessionSchedule> generation_schedules;
    if (natural_schedule && single_schedule)
      generation_schedules.push_back({"default", qwen35::benchmark::read_generation_schedule(schedule_file,
          geometry.vocabulary, capacity, steps, explicit_steps)});
    else if (natural_schedule && multiple_schedules)
      generation_schedules = qwen35::benchmark::read_generation_session_schedules(session_file, geometry.vocabulary,
          capacity, std::vector<hyle::inference::GenerationSessionSchedule>{}.max_size(), steps, explicit_steps);
    else if (single_schedule)
      schedules.push_back({"default", qwen35::benchmark::read_schedule(schedule_file, geometry.vocabulary, capacity, true)});
    else if (multiple_schedules)
      schedules = qwen35::benchmark::read_session_schedules(session_file, geometry.vocabulary, capacity,
          std::vector<qwen35::benchmark::SessionSchedule>{}.max_size(), true);
    if (bounded_generation && natural_schedule && generation_schedules.size() != 1)
      throw std::runtime_error("bounded workload driver accepts exactly one session");
    const auto input_sessions = bounded_generation ? std::size_t{1}
        : natural_schedule ? generation_schedules.size() : scheduled ? schedules.size() : std::size_t{1};
    if (bounded_generation && active_sequences == 0) active_sequences = 1;
    const auto source_precision = qwen35::PrecisionProfile::from_roles(weight_source.roles);
    const auto recurrent_payload = qwen35::recurrent_bytes(qwen35::state_schema_for(geometry, capacity, source_precision), geometry);
    // Configuration describes a maximum, never permission to overrun context.
    // Actual calls additionally clip against output quota and remaining rows.
    max_drafts = (std::min)(max_drafts, capacity - 1);
    if (max_sessions == 0) max_sessions = input_sessions;
    const auto cached_slots = (std::min)(max_sessions, input_sessions);
    if (active_sequences > max_sessions)
      throw std::runtime_error("--active-sequences cannot exceed --max-sessions");
    const auto forced = decode_ids.empty() ? std::vector<std::int32_t>{} : qwen35::read_ids(decode_ids, geometry.vocabulary, capacity);
    if (!decode_ids.empty()) {
      if (forced.empty()) throw std::runtime_error("--decode-ids requires a nonempty token tape");
      const auto forced_steps = qwen35::cache::add(forced.size(), 1);
      if (explicit_steps && steps != forced_steps)
        throw std::runtime_error("--steps must equal decode input count plus one");
      steps = forced_steps;
    }
    if (input.empty())
      throw std::runtime_error("usage: hyle --inputs DIR [--schedule FILE | --session-schedules FILE | [--continuation IDS] [--steps 4 | --decode-ids IDS]] [--mtp auto|off|on] [--capacity 128] [--max-sessions auto|N] [--active-sequences auto|N] [--cache-bytes auto|N] [--prefix-cache off|auto] [--progress direct|scheduler] [--replay auto|commands|prefer-capture] [--trace DIR] [--trace-level full|decisions] [--preparation-diagnostics 0|1] [--layer0-probe DIR]");
    const auto prompt = scheduled ? std::vector<std::int32_t>{} : qwen35::read_ids(input / "prompt.ids", geometry.vocabulary, capacity);
    const auto extra = continuation.empty() ? std::vector<std::int32_t>{} : qwen35::read_ids(continuation, geometry.vocabulary, capacity);
    if (!scheduled) {
      if (prompt.empty()) throw std::runtime_error("generation requires a nonempty prompt before loading weights");
      if (prompt.size() > capacity || extra.size() > capacity - prompt.size() ||
          steps - 1 > capacity - prompt.size() - extra.size())
        throw std::runtime_error("prompt, continuation and requested generation exceed --capacity");
    }
    if (natural_single) generation_schedules.push_back({"default", {{prompt, steps}}});
    const inference::PrefillPolicy prefill(max_prefill_rows);
    std::vector<std::size_t> planned_chunks;
    std::size_t mtp_schema_rows = 0;
    bool mtp_has_decode = !scheduled && steps > 1;
    if (bounded_generation) {
      const auto &first = generation_schedules.front().turns.front();
      planned_chunks.push_back(prefill.next(first.prompt.size()).rows);
      mtp_schema_rows = capacity;
      mtp_has_decode = mtp_enabled && capacity > 1;
    } else if (natural_generation) {
      for (const auto &request : generation_schedules) for (std::size_t i = 0; i < request.turns.size(); ++i) {
        const auto tokens = qwen35::cache::add(request.turns[i].prompt.size(), i ? 1 : 0);
        const auto chunk_rows = prefill.next(tokens).rows;
        planned_chunks.push_back(chunk_rows);
        // The first nonempty prefix leaves at most tokens-1 new rows. Later
        // turns can consume the full carrier plus their historical catch-up.
        if (mtp_enabled) mtp_schema_rows = (std::max)(mtp_schema_rows,
            qwen35::cache::add(chunk_rows, i || (prefix_enabled && tokens > chunk_rows) ? 1 : 0));
        mtp_has_decode = mtp_has_decode || request.turns[i].max_outputs > 1;
      }
    } else if (scheduled) {
      for (const auto &request : schedules) for (const auto &phase : request.phases) {
        const auto tokens = phase.kind == qwen35::benchmark::PhaseKind::kPrompt ? phase.ids.size() : std::size_t{1};
        if (std::find(planned_chunks.begin(), planned_chunks.end(), tokens) == planned_chunks.end())
          planned_chunks.push_back(tokens);
      }
    } else {
      planned_chunks.push_back(prompt.size());
      if (!extra.empty()) planned_chunks.push_back(extra.size());
      planned_chunks.insert(planned_chunks.end(), steps - 1, 1);
    }
    // Immutable host token owners are bounded by the validated input batch.
    // Only suffixes at real prompt/turn boundaries enter the cold shape union.
    std::vector<qwen35::cache::TokenPlanRef> token_plans;
    if (prefix_enabled) {
      token_plans.reserve(input_sessions);
      if (natural_generation) for (const auto &request : generation_schedules) {
        token_plans.push_back(qwen35::cache::token_plan(request.turns.front().prompt, capacity));
      } else for (const auto &request : schedules) {
        std::vector<std::span<const std::int32_t>> chunks;
        chunks.reserve(request.phases.size());
        for (const auto &phase : request.phases) chunks.push_back(phase.ids);
        token_plans.push_back(qwen35::cache::token_plan(chunks, capacity));
      }
    }
    const auto max_query_rows = bounded_generation ? capacity : (std::max)({*std::max_element(planned_chunks.begin(), planned_chunks.end()), mtp_schema_rows,
        mtp_enabled && mtp_has_decode ? max_drafts + 1 : std::size_t{1}});
    const auto requested_active = (std::min)(active_sequences, input_sessions);
    const bool stored_draft_head = qwen35::supports_stored_draft_projection(weight_source.roles);
    const qwen35::DeviceRelayCapabilities device_relay{
        backend_selection.primary, cpu_ffn_layers != 0, progress,
        sampler.enabled && mtp_enabled, !sampler.modifiers.empty()};
    const hyle::inference::DecodeRouteFacts route_facts{
        .workload = cold_workload, .overlap_requested = decode_overlap,
        .mtp_enabled = mtp_enabled, .max_drafts = max_drafts,
        .max_device_drafts = qwen35::kMaximumDeviceMtpDrafts,
        .stored_draft_head = stored_draft_head,
        .relay_eligible = device_relay.eligible(), .relay_refusal = device_relay.refusal_reason(),
        .commands_replay = replay == qwen35::ReplayPolicy::kCommands,
        .has_decode = bounded_generation ? capacity > 1 : std::ranges::any_of(generation_schedules, [](const auto &request) {
          return std::ranges::any_of(request.turns, [](const auto &turn) { return turn.max_outputs > 1; });
        }), .resolved_active_sequences = active_sequences == 0 ? cached_slots : requested_active};
    // A continuous conditional successor retains its completed producer until
    // the consumer really retires. Count that third original batch in the
    // existing page/state bound, independently of simultaneous GPU work.
    auto [overlap_plan, verification_plan] = hyle::inference::plan_decode_routes(route_facts);
    bool overlap_selected = overlap_plan.enabled;
    bool device_verification_selected = verification_plan.enabled;
    // Ask the actual backend about the provisional maximum batch and every
    // target cache storage. Unsupported implementations keep the original
    // two-batch finite route. The created endpoint is reused by Model below.
    std::optional<qwen35::Backend> prepared_execution;
    bool continuous_mtp_candidate = false;
    constexpr auto exact_token_domain = std::int64_t{1} << std::numeric_limits<float>::digits;
    const auto continuous_plan = hyle::inference::select_continuous_mtp(route_facts, overlap_selected,
        cached_slots, geometry.attention_layers() != 0, geometry.vocabulary <= exact_token_domain);
    auto overlap_depth_reason = continuous_plan.reason;
    if (continuous_plan.enabled) {
      const auto provisional = qwen35::cache::plan(geometry, weight_source.cache_row_bytes, capacity, cached_slots,
          mtp_has_decode ? (std::max)(max_query_rows, std::size_t{2}) : max_query_rows, recurrent_payload,
          active_sequences == 0 ? cached_slots : requested_active, max_drafts + 1, true, sampler.enabled, 3,
          device_verification_selected ? 2 : 0, source_precision, device_verification_selected ? max_drafts : 0);
      prepared_execution.emplace(qwen35::take(qwen35::Backend::create(backend_selection)));
      continuous_mtp_candidate = qwen35::supports_mtp_attention(*prepared_execution,
          qwen35::attention_capacity(geometry, source_precision, provisional), weight_source.cache_storage);
      overlap_depth_reason = continuous_mtp_candidate ? inference::RouteReason::kSelected :
          inference::RouteReason::kAttentionCapability;
    }
    auto overlap_depth = continuous_mtp_candidate ? std::size_t{3} : std::size_t{2};
    const auto plan_for_routes = [&](std::size_t residents, bool overlap, std::size_t depth, bool verification) {
      return qwen35::cache::plan(geometry, weight_source.cache_row_bytes, capacity, residents,
          mtp_enabled && mtp_has_decode ? (std::max)(max_query_rows, std::size_t{2}) : max_query_rows, recurrent_payload,
          active_sequences == 0 ? residents : requested_active,
          mtp_enabled ? max_drafts + 1 : 1, mtp_enabled, sampler.enabled, overlap ? depth : 1,
          verification && (active_sequences == 0 ? residents : requested_active) == 1 ? 2 : 0, source_precision,
          verification ? max_drafts : 0);
    };
    const auto minimum_residents = active_sequences == 0 ? std::size_t{1} : requested_active;
    const auto budget_routes = hyle::inference::select_decode_budget_routes(
        {overlap_plan, verification_plan, overlap_depth, overlap_depth_reason},
        [&](const auto &routes) {
          return !cache_bytes || plan_for_routes(minimum_residents, routes.overlap.enabled,
              routes.overlap_depth, routes.verification.enabled).backing_bytes <= cache_bytes;
        });
    overlap_plan = budget_routes.overlap;
    verification_plan = budget_routes.verification;
    overlap_selected = overlap_plan.enabled;
    device_verification_selected = verification_plan.enabled;
    overlap_depth = budget_routes.overlap_depth;
    overlap_depth_reason = budget_routes.overlap_depth_reason;
    const auto plan_for = [&](std::size_t residents) {
      return plan_for_routes(residents, overlap_selected, overlap_depth, device_verification_selected);
    };
    auto selected_cache = plan_for(cached_slots);
    // Caller-written pools have an exact cold formula. Native generated
    // results instead enter the same byte scope after real completion, before
    // any append is taken or head published. Do not estimate native N from the
    // CUDA pool formula or pretend auto bounds opaque execution allocations.
    // CPU cache metadata shares the payload's actual host resource. The GPU
    // payload formula is not an upper bound for that combined byte scope.
    // Auto keeps the configured version/batch bounds; an explicit byte cap
    // still accounts for and limits every cache allocation, including metadata.
    const bool native_generated_storage = backend_selection.primary == qwen35::BackendKind::kApple ||
        backend_selection.primary == qwen35::BackendKind::kVulkan;
    if (cache_bytes == 0) cache_bytes = (native_generated_storage || backend_selection.primary == qwen35::BackendKind::kCpu)
        ? SIZE_MAX : selected_cache.backing_bytes;
    else if (!native_generated_storage && selected_cache.backing_bytes > cache_bytes) {
      const auto minimum = plan_for(minimum_residents);
      if (minimum.backing_bytes > cache_bytes)
        throw std::runtime_error("--cache-bytes cannot hold the configured active sequences: requires " +
            std::to_string(minimum.backing_bytes) + " shared KV/RNN backing bytes");
      std::size_t low = minimum_residents, high = cached_slots;
      while (low < high) {
        const auto middle = low + (high - low + 1) / 2;
        if (plan_for(middle).backing_bytes <= cache_bytes) low = middle;
        else high = middle - 1;
      }
      selected_cache = plan_for(low);
    }
    const auto cache_slots = selected_cache.resident_sessions;
    const auto active_slots = selected_cache.active_sequences;
    const auto metadata_prepare_seconds = weight_source.metadata_prepare_seconds;
    const auto model_start = qwen35::Clock::now();
    auto execution = prepared_execution ? std::move(*prepared_execution)
        : qwen35::create_execution(geometry, selected_cache, cpu_ffn_layers, backend_selection);
    qwen35::Model model(std::move(execution), {}, std::move(weight_source), geometry, capacity,
        selected_cache, layer0_probe, progress, replay, cache_bytes, sampler, cpu_ffn_layers);
    const hyle::engine::reporting::RouteDecisionFacts routes{
        .mtp_selection_reason = default_selection.reason, .mtp_depth_source = depth_selection.source,
        .draft_vocabulary_selection_reason = assets.vocabulary_reason,
        .bounded_generation_plan_reason = bounded_plan.reason, .decode_overlap_plan_reason = overlap_plan.reason,
        .device_verification_plan_reason = verification_plan.reason, .overlap_depth_plan_reason = overlap_depth_reason};
    model.configure_draft_policy({max_drafts});
    const auto constructed = qwen35::Clock::now();
    if (!trace.empty()) model.trace_routes = trace / "routes";
    if (natural_generation) {
      if (mtp_enabled) return qwen35::run_generation_sessions<qwen35::GenerationMode::kMtp>(model,
          generation_schedules, trace, trace_states, startup, constructed, prefix_enabled, token_plans, {}, {}, max_prefill_rows, warmup_planned_shapes, overlap_selected, {}, nullptr, bounded_generation, routes);
      return qwen35::run_generation_sessions<qwen35::GenerationMode::kText>(model,
          generation_schedules, trace, trace_states, startup, constructed, prefix_enabled, token_plans, {}, {}, max_prefill_rows, warmup_planned_shapes, overlap_selected, {}, nullptr, bounded_generation, routes);
    }

    // These owners are declared after Model, including on exception paths.
    // The schedule is consumer policy; every call uses the same Model/Graph chain.
    struct PhaseSummary {
      std::size_t context_before = 0, context_after = 0;
      double seconds = 0, step_sum_seconds = 0;
      std::string name, memory_name;
      qwen35::Clock::time_point start{};
    };
    struct Call { std::size_t phase, local; };
    // Count/idle policy only. Actual shared arena and state-pool backings own
    // the byte reservations; charging this directory per Session would double
    // count shared pages and immutable snapshots.
    hyle::inference::SessionCache resident_cache({cache_slots, (std::numeric_limits<std::size_t>::max)()});
    // Destroy runs (and their real slots) before this directory and Model.
    auto prefix_cache = prefix_enabled ? std::make_unique<qwen35::cache::PrefixCache>(cache_slots) : nullptr;
    struct SessionRun {
      struct RecomputedDecision { std::size_t chunk; std::int32_t previous, actual; };
      std::shared_ptr<qwen35::Session> session;
      hyle::inference::SessionCache::Pin pin;
      std::vector<std::size_t> chunks;
      std::shared_ptr<qwen35::cache::ReplayHistory> history;
      std::vector<PhaseSummary> summaries;
      std::vector<Call> calls;
      std::vector<std::int32_t> generated, consumed;
      std::vector<inference::StepObservation> observations;
      std::vector<inference::PreparationObservation> preparation;
      qwen35::cache::SamplingCursor saved_sampling;
      std::int32_t saved_pending = -1;
      std::filesystem::path trace;
      std::string trace_subdirectory;
      qwen35::Clock::time_point start{}, endpoint{};
      std::size_t visible_tokens = 0, cache_hits = 0, cache_misses = 0;
      std::size_t restored_calls = 0, restored_tokens = 0;
      std::vector<RecomputedDecision> recomputed_decisions;
      std::size_t graph_participations = 0, actual_graph_tokens = 0;
      std::size_t prefix_hits = 0, prefix_misses = 0, prefix_full_hits = 0, reused_tokens = 0;
      std::size_t prefix_sampling_calls = 0;
      double step_sum_seconds = 0, first_token_elapsed_seconds = 0, cache_prepare_seconds = 0;
    };
    std::vector<SessionRun> runs(schedules.size());
    std::size_t peak_committed_versions = 0;
    const auto note_committed_versions = [&] {
      peak_committed_versions = (std::max)(peak_committed_versions, std::size_t(model.snapshot_versions.live));
    };
    std::size_t all_calls = 0;
    for (std::size_t s = 0; s < schedules.size(); ++s) {
      const auto &request = schedules[s];
      auto &run = runs[s];
      run.saved_sampling = {model.sampling_seed, static_cast<std::uint64_t>(s), 0};
      run.summaries.resize(request.phases.size());
      if (prefix_enabled) run.history = std::make_shared<qwen35::cache::ReplayHistory>(capacity);
      run.trace_subdirectory = "session" + std::to_string(s) + "_" + request.id;
      run.trace = trace.empty() || !multiple_schedules ? trace : trace / run.trace_subdirectory;
      for (std::size_t i = 0; i < request.phases.size(); ++i) {
        const auto &phase = request.phases[i];
        auto &summary = run.summaries[i];
        summary.name = "round" + std::to_string(phase.turn) +
            (phase.kind == qwen35::benchmark::PhaseKind::kPrompt ? "_prefill" : "_decode");
        summary.memory_name = multiple_schedules ? request.id + "_" + summary.name : summary.name;
        const bool is_prompt = phase.kind == qwen35::benchmark::PhaseKind::kPrompt;
        const auto count = is_prompt ? std::size_t{1} : phase.ids.size();
        for (std::size_t local = 0; local < count; ++local) {
          run.calls.push_back({i, local});
          run.chunks.push_back(is_prompt ? phase.ids.size() : std::size_t{1});
        }
      }
      if (run.calls.size() > (std::numeric_limits<std::size_t>::max)() - all_calls)
        throw std::runtime_error("total schedule call count overflows");
      all_calls += run.calls.size();
      run.generated.reserve(run.calls.size()); run.consumed.reserve(run.calls.size());
      run.observations.resize(run.calls.size());
      if (preparation_diagnostics) run.preparation.resize(run.calls.size());
      for (std::size_t i = 0; i < run.calls.size(); ++i) {
        const auto call = run.calls[i];
        run.observations[i].name = qwen35::benchmark::phase_name(request.phases[call.phase], call.local);
      }
    }
    const auto waves = scheduled ? qwen35::benchmark::make_schedule_waves(schedules, active_slots)
        : std::vector<qwen35::benchmark::ScheduleWave>{};
    std::vector<std::pair<std::size_t, std::size_t>> dispatch;
    dispatch.reserve(all_calls);
    std::vector<std::vector<std::size_t>> query_options(all_calls);
    std::vector<std::map<std::size_t, std::size_t>> published_at(schedules.size());
    std::vector<std::size_t> prior_tokens(schedules.size());
    for (std::size_t w = 0; w < waves.size(); ++w) for (const auto &call : waves[w].calls) {
      dispatch.emplace_back(call.session, call.call);
      prior_tokens[call.session] = qwen35::cache::add(prior_tokens[call.session], call.query_rows);
      const auto &phase = schedules[call.session].phases[call.phase];
      if (phase.kind == qwen35::benchmark::PhaseKind::kPrompt || call.local + 1 == phase.ids.size())
        published_at[call.session].emplace(prior_tokens[call.session], w);
    }
    std::fill(prior_tokens.begin(), prior_tokens.end(), 0);
    std::size_t restore_call_bound = 0;
    for (std::size_t w = 0; w < waves.size(); ++w) for (const auto &call : waves[w].calls) {
      auto &before = prior_tokens[call.session];
      const auto target = qwen35::cache::add(before, call.query_rows);
      if (call.kind == qwen35::benchmark::PhaseKind::kPrompt) {
        if (schedules.size() > cache_slots) restore_call_bound = qwen35::cache::add(restore_call_bound, before);
        if (prefix_enabled) for (std::size_t candidate = 0; candidate < token_plans.size(); ++candidate)
          for (const auto boundary : token_plans[candidate]->checkpoints) {
            const auto producer = published_at[candidate].find(boundary);
            if (boundary < before || boundary > target || producer == published_at[candidate].end() || producer->second >= w ||
                !qwen35::cache::same_prefix(*token_plans[candidate], *token_plans[call.session], boundary)) continue;
            query_options[call.ordinal].push_back(target - boundary);
          }
      }
      before = target;
    }
    const auto shape_uses = qwen35::benchmark::plan_batch_shape_uses(waves, query_options);
    if (!scheduled) {
      model.prepare_family(planned_chunks);
      if (!layer0_probe.empty()) model.prepare_family(planned_chunks, true);
      model.plan_execution(planned_chunks);
    } else {
      std::map<std::size_t, std::vector<std::size_t>> chunks_by_width;
      for (const auto &shape : shape_uses) chunks_by_width[shape.requests].push_back(shape.query_rows);
      for (const auto &[width, chunks] : chunks_by_width) {
        model.prepare_family(chunks, false, width);
        if (!layer0_probe.empty()) model.prepare_family(chunks, true, width);
      }
      if (!multiple_schedules && !prefix_enabled) model.plan_execution(runs.front().chunks);
      else for (const auto &shape : shape_uses)
        model.plan_replay_uses(shape.query_rows, shape.requests,
            qwen35::cache::add(shape.max_wave_calls, shape.requests == 1 ? restore_call_bound : 0));
    }
    struct BatchObservation {
      qwen35::benchmark::PhaseKind kind;
      std::size_t tokens;
      bool restore;
      double seconds;
      hyle::container::SmallVector<std::size_t, 4> sessions;
    };
    struct WaveObservation {
      int kind;
      std::size_t outputs = 0, logical_tokens = 0, actual_tokens = 0, graph_calls = 0;
      double seconds = 0;
    };
    std::vector<BatchObservation> batches;
    batches.reserve(all_calls);
    std::vector<WaveObservation> wave_observations;
    wave_observations.reserve(waves.size());
    std::size_t executed_graph_calls = 0, executed_graph_tokens = 0, max_batch_requests = 0;
    const auto record_batch = [&](qwen35::benchmark::PhaseKind kind, std::size_t tokens,
                                   std::span<const std::size_t> members, bool restore, double seconds) {
      BatchObservation value{kind, tokens, restore, seconds, {}};
      if (!value.sessions.try_reserve(members.size())) throw std::bad_alloc{};
      for (auto s : members) value.sessions.push_back(s);
      batches.push_back(std::move(value));
      executed_graph_calls = qwen35::cache::add(executed_graph_calls, 1);
      executed_graph_tokens = qwen35::cache::add(executed_graph_tokens, qwen35::cache::multiply(tokens, members.size()));
      max_batch_requests = (std::max)(max_batch_requests, members.size());
    };
    std::vector<std::size_t> evicted_sessions;
    evicted_sessions.reserve(all_calls);
    const auto make_initial_room = [&] {
      if (model.active_step) throw std::runtime_error("initial admission requires retired sessions");
      const std::array selected{qwen35::version_holder(nullptr)};
      const auto residents = runs | std::views::transform([](const auto &run) {
        return qwen35::version_holder(run.session.get());
      });
      const auto observe = [&] {
        const auto live = std::size_t(model.snapshot_versions.live);
        return hyle::inference::VersionRoomObservation{{live, cache_slots}, prefix_cache ? prefix_cache->size() : 0,
            hyle::inference::version_admission({live, cache_slots}, selected, residents,
                [&](const auto *head) { return prefix_cache && prefix_cache->retains(head); }).fits};
      };
      if (hyle::inference::make_version_room(observe, [&] { return prefix_cache && prefix_cache->evict_one(); }) !=
          hyle::inference::VersionRoomResult::kReady)
        throw std::runtime_error("prefix/session heads retain the committed version budget before initial admission");
    };
    const auto make_batch_room = [&](std::span<qwen35::Session *const> owners) {
      if (owners.empty() || model.active_step || model.snapshot_versions.live > cache_slots)
        throw std::runtime_error("batch advance requires retired sessions within the committed version budget");
      for (auto *owner : owners) if (!owner || !owner->head || owner->pending || owner->failed || owner->stopping)
        throw std::runtime_error("batch advance contains a stopped, failed or pending session");
      // With prefix reuse disabled, each resident Session has its own sole
      // committed head. Advancing it replaces one version; no directory scan.
      if (!prefix_cache) return;
      const auto selected = owners | std::views::transform([](const auto *owner) {
        return qwen35::version_holder(owner);
      });
      const auto residents = runs | std::views::transform([](const auto &run) {
        return qwen35::version_holder(run.session.get());
      });
      const auto observe = [&] {
        const auto live = std::size_t(model.snapshot_versions.live);
        return hyle::inference::VersionRoomObservation{{live, cache_slots}, prefix_cache->size(),
            hyle::inference::version_admission({live, cache_slots}, selected, residents,
                [&](const auto *head) { return prefix_cache->retains(head); }).fits};
      };
      // Dropping a directory alias need not free a version. Re-read both live
      // inventory and external retainers after EVERY eviction, including full hits.
      if (hyle::inference::make_version_room(observe, [&] { return prefix_cache->evict_one(); }) !=
          hyle::inference::VersionRoomResult::kReady)
        throw std::runtime_error("batch commit exceeds N retained snapshots; release a checkpoint/session or increase the cache");
    };
    const auto make_advance_room = [&](qwen35::Session &owner) {
      const std::array owners{&owner}; make_batch_room(owners);
    };
    const auto ensure_resident = [&](std::size_t s, bool defer_restore) {
      auto &run = runs[s];
      if (run.pin) return;
      const auto begin = qwen35::Clock::now();
      const auto admission = resident_cache.admit(s, 0, [&](std::size_t victim) {
        auto &old = runs[victim];
        if (old.pin) return hyle::inference::RetirementObservation{};
        hyle::inference::RetirementObservation retired{true, {}};
        if (old.session) {
          (void)old.session->stop();
          retired = old.session->try_retire();
          if (!retired.retired) return retired;
        }
        old.session.reset(); // Actual state leases retire before quota returns.
        evicted_sessions.push_back(victim);
        return retired;
      });
      check(admission.retirement_status);
      if (!admission.accepted())
        throw std::runtime_error(admission.status == hyle::inference::CacheAdmissionStatus::kRetirementUnproven
            ? "session cache cannot prove retirement; the original owner remains charged"
            : "session cache admission exhausted; reduce --active-sequences or increase --max-sessions/--cache-bytes");
      run.pin = resident_cache.pin(s);
      if (!run.pin) throw std::runtime_error("session cache could not pin the admitted owner");
      if (run.session) ++run.cache_hits;
      else {
        ++run.cache_misses;
        if (!defer_restore) {
          make_initial_room();
          run.session = model.create_session(run.chunks, {}, qwen35::GraphRole::kText,
              qwen35::cache::SamplingCursor{model.sampling_seed, static_cast<std::uint64_t>(s), 0});
          note_committed_versions();
          // Recompute the original token/chunk tape; never resample or replace
          // already returned tokens. A different legal batch reduction may
          // change logits/greedy decisions. The rebuilt real head is authority
          // for new work; old decisions remain diagnostics, not state payload.
          for (std::size_t i = 0; i < run.generated.size(); ++i) {
            const auto call = run.calls[i];
            const auto &phase = schedules[s].phases[call.phase];
            const auto ids = phase.kind == qwen35::benchmark::PhaseKind::kPrompt
                ? std::span<const std::int32_t>(phase.ids)
                : std::span<const std::int32_t>(&phase.ids[call.local], 1);
            inference::StepObservation restored;
            const auto name = "restore" + std::to_string(run.cache_misses) + "_call" + std::to_string(i);
            const auto restore_start = qwen35::Clock::now();
            make_advance_room(*run.session);
            const auto next = qwen35::evaluate(model, *run.session, ids, name,
                run.trace.empty() ? std::filesystem::path{} : run.trace / "restores", false, nullptr, nullptr,
                &restored, nullptr, multiple_schedules ? std::string_view(run.trace_subdirectory) : std::string_view{});
            if (next != run.generated[i]) run.recomputed_decisions.push_back({i, run.generated[i], next});
            ++run.restored_calls; run.restored_tokens += ids.size();
            ++run.graph_participations; run.actual_graph_tokens += ids.size();
            const std::array members{s};
            record_batch(phase.kind, ids.size(), members, true,
                std::chrono::duration<double>(qwen35::Clock::now() - restore_start).count());
            note_committed_versions();
          }
          if (run.session->length() != run.visible_tokens)
            throw std::runtime_error("restored cache context differs from retained consumer history");
          if (run.saved_pending >= 0) model.restore_pending_boundary(*run.session, run.saved_sampling, run.saved_pending);
        }
      }
      (void)model.cache_usage();
      run.cache_prepare_seconds += std::chrono::duration<double>(qwen35::Clock::now() - begin).count();
    };
    model.freeze_shape_plan();
    if (warmup_planned_shapes) {
      const auto begin = qwen35::Clock::now();
      const auto shapes = model.prepare_planned_execution();
      std::cerr << "shape_warmup policy=planned shapes=" << shapes
                << " seconds=" << std::chrono::duration<double>(qwen35::Clock::now() - begin).count() << '\n';
    }
    // Prefix mode admits at the timed request boundary, before any initial
    // state allocation. A full hit must not create or evaluate an empty state.
    if (scheduled && !prefix_enabled) {
      // Exact nominal buckets are known before admission. Suffix alternatives
      // stay cold metadata until an actual prefix hit selects one.
      if (!multiple_schedules) model.prepare_execution(runs.front().chunks);
      else for (const auto &wave : waves) {
        std::vector<std::size_t> rows;
        rows.reserve(wave.calls.size());
        for (const auto &call : wave.calls) rows.push_back(call.query_rows);
        for (const auto &bucket : qwen35::benchmark::bucket_schedule_wave(wave, rows)) {
          const std::array chunks{bucket.query_rows};
          model.prepare_execution(chunks, bucket.calls.size());
        }
      }
      if (schedules.size() > cache_slots) for (const auto &run : runs)
        model.prepare_execution(run.chunks);
    }
    if (!prefix_enabled) for (std::size_t s = 0; s < (std::min)(active_slots, runs.size()); ++s) ensure_resident(s, false);
    auto session = scheduled ? std::shared_ptr<qwen35::Session>{} : model.create_session(planned_chunks);
    if (session) model.prepare_execution(planned_chunks);
    inference::BatchIndices batch_indices;
    if (scheduled) batch_indices.prepare(active_slots, active_slots);
    std::vector<std::int32_t> generated, consumed;
    if (!scheduled) { generated.reserve(steps); consumed.reserve(steps - 1); }
    const auto ready = qwen35::Clock::now();
    std::cerr << "initialized model_seconds=" << metadata_prepare_seconds +
                  std::chrono::duration<double>(constructed - model_start).count()
              << " shape_prepare_seconds=" << std::chrono::duration<double>(ready - constructed).count()
              << " startup_to_ready_seconds=" << std::chrono::duration<double>(ready - startup).count() << '\n';
    std::cerr << "cache_limits context=" << capacity << " max_sessions=" << max_sessions
              << " active_sequences=" << active_slots << " effective_cache_sessions=" << cache_slots
              << " recurrent_payload_bytes_per_snapshot=" << recurrent_payload
              << " backend=" << model.execution.name()
              << " shared_cache_backing_bytes=" << model.known_cache_backing_bytes
              << " native_execution_private_bytes_unknown=" << model.execution.native_generated_storage()
              << " cache_budget_scope=shared_pools_and_retained_output_backing";
    if (cache_bytes == SIZE_MAX) std::cerr << " cache_requested_byte_limit=none";
    else std::cerr << " cache_requested_byte_limit=" << cache_bytes;
    std::cerr << " physical_vram_limit_known=false\n";
    if (scheduled) {
      qwen35::Clock::time_point journey_start{}, endpoint{};
      struct PreparedCall {
        std::span<const std::int32_t> ids, logical_ids;
        std::size_t before = 0, reused = 0;
        bool used_prefix = false;
        qwen35::Clock::time_point call_start{};
        double complete_seconds = 0;
        std::int32_t next = -1;
      };
      std::vector<PreparedCall> prepared;
      std::vector<std::size_t> actual_rows, members;
      std::vector<qwen35::Session *> owners;
      std::vector<qwen35::EvaluationRequest> requests;
      std::vector<qwen35::EvaluationOutput> outputs;
      std::vector<std::int32_t> next_ids;
      prepared.reserve(active_slots); actual_rows.reserve(active_slots); members.reserve(active_slots);
      owners.reserve(active_slots); requests.reserve(active_slots); outputs.reserve(active_slots); next_ids.reserve(active_slots);
      for (const auto &wave : waves) {
        const auto wave_start = qwen35::Clock::now();
        if (journey_start == qwen35::Clock::time_point{}) journey_start = wave_start;
        const auto graphs_before = executed_graph_calls, tokens_before = executed_graph_tokens;
        WaveObservation wave_observation{wave.calls.front().kind == qwen35::benchmark::PhaseKind::kPrompt ? 0 : 1};
        prepared.clear(); prepared.resize(wave.calls.size()); actual_rows.clear();
        // Resolve the whole wave against PREVIOUS publications. A full hit
        // stays pinned and charged while the other members are evaluated.
        for (std::size_t j = 0; j < wave.calls.size(); ++j) {
          const auto &scheduled_call = wave.calls[j];
          const auto s = scheduled_call.session, index = scheduled_call.call;
          auto &run = runs[s];
          const auto call = run.calls[index];
          const auto &phase = schedules[s].phases[call.phase];
          auto &summary = run.summaries[call.phase];
          if (call.local == 0) {
            summary.context_before = run.visible_tokens;
            std::cerr << "memory_phase=" << summary.memory_name << "_begin\n";
            summary.start = qwen35::Clock::now();
            if (index == 0) run.start = summary.start;
          }
          const bool is_prompt = phase.kind == qwen35::benchmark::PhaseKind::kPrompt;
          auto &work = prepared[j];
          work.call_start = qwen35::Clock::now();
          if (is_prompt) ensure_resident(s, prefix_enabled);
          const auto logical_ids = is_prompt ? std::span<const std::int32_t>(phase.ids)
              : std::span<const std::int32_t>(&phase.ids[call.local], 1);
          auto ids = logical_ids;
          const auto before = run.visible_tokens;
          const auto target = before + logical_ids.size();
          std::size_t reused = 0;
          bool used_prefix = false;
        if (prefix_enabled && is_prompt) {
          // An equal-length checkpoint is useful after eviction, but must not
          // replace a resident same-boundary Session and lose its recordings.
          auto hit = prefix_cache->lookup(*token_plans[s], target,
              before + (run.session ? std::size_t{1} : std::size_t{0}), &model, model.math_identity());
          if (hit) {
            if (hit.payload.snapshot->next_id >= geometry.vocabulary)
              throw std::runtime_error("prefix next_id is outside the model vocabulary");
            reused = hit.payload.snapshot->length - before;
            const auto suffix = target - hit.payload.snapshot->length;
            std::vector<std::size_t> remaining;
            remaining.reserve(run.chunks.size() - index);
            if (suffix) remaining.push_back(suffix);
            remaining.insert(remaining.end(), run.chunks.begin() + static_cast<std::ptrdiff_t>(index + 1), run.chunks.end());
            // All fallible host/IO preparation precedes withdrawal of the old
            // Session. These are real original Model slots, not shadow state.
            auto history = qwen35::cache::copy_history(hit.payload.replay, capacity);
            const auto cursor = run.session ? run.session->head->sampling : run.saved_sampling;
            auto adopted = model.create_session(remaining, hit.payload.snapshot, qwen35::GraphRole::kText, cursor, suffix == 0);
            // This fixed-input CLI tape has one default sampling configuration.
            if (model.default_sampler.enabled && !suffix) {
              history->chunks.back().next_id = adopted->head->next_id; ++run.prefix_sampling_calls;
            }
            if (run.session) {
              (void)run.session->stop();
              const auto retired = run.session->try_retire();
              if (retired.retired) run.session.reset();
              check(retired.status);
              if (!retired.retired)
                throw std::runtime_error("prefix adoption cannot retire the original session");
            }
            run.session = std::move(adopted);
            run.history = std::move(history);
            ids = logical_ids.subspan(reused);
            hit.reset(); // No temporary Snapshot owner may hide from advance admission.
            used_prefix = true; ++run.prefix_hits; run.reused_tokens += reused;
            if (ids.empty()) ++run.prefix_full_hits;
          } else ++run.prefix_misses;
        }
        if (prefix_enabled && !run.session) {
          make_initial_room();
          std::vector<std::size_t> remaining;
          remaining.reserve(run.history->chunks.size() + run.chunks.size() - index);
          for (const auto &part : run.history->chunks) remaining.push_back(part.count);
          remaining.insert(remaining.end(), run.chunks.begin() + static_cast<std::ptrdiff_t>(index), run.chunks.end());
          run.session = model.create_session(remaining, {}, qwen35::GraphRole::kText,
              qwen35::cache::SamplingCursor{model.sampling_seed, static_cast<std::uint64_t>(s), 0});
          note_committed_versions();
          // Recompute actual evaluated chunks, including imported prefixes.
          // Keep any old shared history immutable: new checkpoint metadata
          // must describe the REAL rebuilt head and its recomputed decisions.
          auto rebuilt_history = std::make_shared<qwen35::cache::ReplayHistory>(capacity);
          for (std::size_t i = 0; i < run.history->chunks.size(); ++i) {
            const auto &part = run.history->chunks[i];
            const auto restored_ids = std::span<const std::int32_t>(part.input->ids).subspan(part.first, part.count);
            inference::StepObservation restored;
            const auto name = "restore" + std::to_string(run.cache_misses) + "_chunk" + std::to_string(i);
            const auto restore_start = qwen35::Clock::now();
            make_advance_room(*run.session);
            const auto next = qwen35::evaluate(model, *run.session, restored_ids, name,
                run.trace.empty() ? std::filesystem::path{} : run.trace / "restores", false, nullptr, nullptr,
                &restored, nullptr, multiple_schedules ? std::string_view(run.trace_subdirectory) : std::string_view{});
            if (next != part.next_id) run.recomputed_decisions.push_back({i, part.next_id, next});
            rebuilt_history->append(part.input, part.first, part.count, next);
            ++run.restored_calls; run.restored_tokens += restored_ids.size();
            ++run.graph_participations; run.actual_graph_tokens += restored_ids.size();
            const std::array restored_members{s};
            record_batch(phase.kind, restored_ids.size(), restored_members, true,
                std::chrono::duration<double>(qwen35::Clock::now() - restore_start).count());
          }
          if (run.session->length() != before)
            throw std::runtime_error("actual-chunk recompute differs from the retained logical boundary");
          if (run.saved_pending >= 0) {
            model.restore_pending_boundary(*run.session, run.saved_sampling, run.saved_pending);
            // The fixed-input tape retains that same CLI-wide configuration.
            if (model.default_sampler.enabled && !rebuilt_history->chunks.empty())
              rebuilt_history->chunks.back().next_id = run.saved_pending;
          }
          run.history = std::move(rebuilt_history);
        }
          work.ids = ids; work.logical_ids = logical_ids;
          work.before = before; work.reused = reused; work.used_prefix = used_prefix;
          actual_rows.push_back(ids.size());
          wave_observation.logical_tokens += logical_ids.size();
          if (wave_observation.kind != (is_prompt ? 0 : 1)) wave_observation.kind = -1;
          if (ids.empty()) {
            work.next = run.session->head->next_id;
            run.endpoint = qwen35::Clock::now();
            auto &observation = run.observations[index];
            observation.graph_executed = false; observation.graph_input_tokens = 0;
            observation.visible = run.session->length(); observation.next = work.next;
            observation.scheduled_progress = progress;
          }
        }
        // Validate the entire resolved wave before any bucket can submit.
        // A later incompatible member must not turn an old preflight error
        // into a partially committed wave.
        if (actual_rows.size() != wave.calls.size())
          throw std::runtime_error("batch wave query lengths do not match its calls");
        for (std::size_t j = 0; j < wave.calls.size(); ++j)
          if (actual_rows[j] > wave.calls[j].query_rows)
            throw std::runtime_error("batch wave actual query exceeds its logical input");
        inference::dispatch_batches(batch_indices, wave.calls.size(),
            [&](std::size_t index) { return actual_rows[index] != 0; },
            [&](std::size_t) -> std::size_t { return active_slots; },
            [&](std::size_t first, std::size_t j) {
              return wave.calls[j].kind == wave.calls[first].kind && actual_rows[j] == actual_rows[first];
            }, [&](std::span<const std::size_t> selected) -> std::size_t {
              const auto kind = wave.calls[selected.front()].kind;
              const auto query_rows = actual_rows[selected.front()];
              requests.clear(); outputs.clear(); owners.clear(); members.clear(); next_ids.resize(selected.size());
              for (const auto j : selected) {
                const auto &call = wave.calls[j]; auto &run = runs[call.session]; auto &work = prepared[j];
                requests.push_back({run.session, work.ids, nullptr, {}});
                owners.push_back(run.session.get()); members.push_back(call.session);
                outputs.push_back(qwen35::evaluation_output(model, run.session,
                    {run.observations[call.call].name, &run.trace, trace_states,
                        multiple_schedules ? std::string_view(run.trace_subdirectory) : std::string_view{}},
                    &work.complete_seconds, &run.endpoint, &run.observations[call.call],
                    preparation_diagnostics ? &run.preparation[call.call] : nullptr));
              }
              const auto batch_start = qwen35::Clock::now();
              make_batch_room(owners);
              inference::PreparationObservation preparation;
              model.prepare_evaluations(requests, preparation_diagnostics ? &preparation : nullptr);
              if (preparation_diagnostics) for (const auto j : selected)
                runs[wave.calls[j].session].preparation[wave.calls[j].call] = preparation;
              model.submit_evaluations();
              model.finish_evaluations(outputs, next_ids, [&](std::size_t at, const qwen35::CompletedEvaluationView &view) {
                const auto &call = wave.calls[selected[at]]; const auto &run = runs[call.session];
                qwen35::observe_completed_evaluation(model, outputs[at],
                    {run.observations[call.call].name, &run.trace, trace_states,
                        multiple_schedules ? std::string_view(run.trace_subdirectory) : std::string_view{}}, view);
              });
              const auto batch_endpoint = runs[members.front()].endpoint;
              record_batch(kind, query_rows, members, false,
                  std::chrono::duration<double>(batch_endpoint - batch_start).count());
              for (std::size_t at = 0; at < selected.size(); ++at) {
                const auto j = selected[at]; const auto &call = wave.calls[j];
                auto &run = runs[call.session]; auto &work = prepared[j]; work.next = next_ids[at];
                ++run.graph_participations; run.actual_graph_tokens += work.ids.size();
                if (prefix_enabled) run.history->append(token_plans[call.session], work.before + work.reused, work.ids.size(), work.next);
              }
              // No temporary strong Session owner may retain an old head across
              // next-wave adoption or eviction outside the charged directory.
              requests.clear(); outputs.clear();
              if (model.snapshot_versions.live > cache_slots)
                throw std::runtime_error("committed batch versions exceed the finite cache budget");
              note_committed_versions();
              return selected.size();
            });
        for (std::size_t j = 0; j < wave.calls.size(); ++j) {
          const auto &scheduled_call = wave.calls[j];
          const auto s = scheduled_call.session, index = scheduled_call.call;
          auto &run = runs[s]; const auto call = run.calls[index];
          const auto &phase = schedules[s].phases[call.phase]; auto &summary = run.summaries[call.phase];
          const bool is_prompt = phase.kind == qwen35::benchmark::PhaseKind::kPrompt;
          auto &work = prepared[j]; auto &observation = run.observations[index];
          const auto logical_ids = work.logical_ids;
          const auto call_start = work.call_start;
          const auto next = work.next; const auto reused = work.reused; const auto used_prefix = work.used_prefix;
          // Per-session latency includes time waiting for its batch. Aggregate
          // throughput uses the wave wall once, never the sum of these spans.
          auto complete_seconds = std::chrono::duration<double>(run.endpoint - call_start).count();
          observation.step_seconds = complete_seconds;
          if (prefix_enabled) {
            observation.input_tokens = logical_ids.size();
            observation.prefix_hit = used_prefix; observation.reused_tokens = reused;
            if (model.snapshot_versions.live > cache_slots)
              throw std::runtime_error("committed prefix/session versions exceed the finite cache budget");
            if (!observation.graph_executed) {
              observation.seconds = complete_seconds;
              const auto write_hit_trace = [&](const std::filesystem::path &base) {
                if (base.empty()) return;
                const auto directory = base / observation.name;
                std::filesystem::create_directories(directory);
                std::ofstream output(directory / "cache-hit.json");
                output << "{\"prefix_hit\":true,\"prompt_graph_calls\":0,\"frontier_sampling_graph_calls\":" << (sampler.enabled ? 1 : 0)
                       << ",\"graph_calls\":" << (sampler.enabled ? 1 : 0) << ",\"tensor_payload_available\":false,\"visible_tokens\":"
                       << run.session->length() << ",\"next_id\":" << next << "}\n";
                if (!output) throw std::runtime_error("cannot write prefix-hit metadata trace");
              };
              write_hit_trace(run.trace);
              if (!layer0_probe.empty()) write_hit_trace(multiple_schedules ? layer0_probe / run.trace_subdirectory : layer0_probe);
            }
          }
          note_committed_versions();
          endpoint = (std::max)(endpoint, run.endpoint);
          run.visible_tokens = run.session->length();
          summary.step_sum_seconds += complete_seconds; run.step_sum_seconds += complete_seconds;
          run.generated.push_back(next);
          run.saved_sampling = run.session->head->sampling; run.saved_pending = next;
          if (!is_prompt) run.consumed.push_back(phase.ids[call.local]);
          if (index == 0) run.first_token_elapsed_seconds = std::chrono::duration<double>(run.endpoint - startup).count();
          const auto count = is_prompt ? std::size_t{1} : phase.ids.size();
          if (call.local + 1 == count) {
            summary.context_after = run.visible_tokens;
            summary.seconds = std::chrono::duration<double>(run.endpoint - summary.start).count();
            std::cerr << "memory_phase=" << summary.memory_name << "_end\n";
            // Terminal prompts are real completed requests too.
            if (!is_prompt || call.phase + 1 == schedules[s].phases.size()) run.pin.reset();
          }
        }
        // Publication is a wave boundary. Same-wave calls never consume each
        // other's new checkpoint, matching the dependency and cold shape plan.
        if (prefix_enabled) for (const auto &call : wave.calls) {
          const auto &phase = schedules[call.session].phases[call.phase];
          const auto count = phase.kind == qwen35::benchmark::PhaseKind::kPrompt ? std::size_t{1} : phase.ids.size();
          if (call.local + 1 != count) continue;
          auto &run = runs[call.session];
          const qwen35::cache::ReplayPrefix history{run.history, run.history->chunks.size(), run.history->length};
          (void)prefix_cache->publish(token_plans[call.session], run.session->checkpoint(), history);
        }
        wave_observation.outputs = wave.calls.size();
        wave_observation.actual_tokens = executed_graph_tokens - tokens_before;
        wave_observation.graph_calls = executed_graph_calls - graphs_before;
        endpoint = qwen35::Clock::now(); // Include all result/checkpoint publication once.
        wave_observation.seconds = std::chrono::duration<double>(endpoint - wave_start).count();
        wave_observations.push_back(wave_observation);
      }
      for (std::size_t s = 0; s < runs.size(); ++s) {
        const auto &run = runs[s];
        if (multiple_schedules) std::cerr << "session_steps id=" << schedules[s].id << '\n';
        for (const auto &observation : run.observations) hyle::engine::reporting::write_step_observation(std::cerr, observation);
        if (preparation_diagnostics) for (std::size_t i = 0; i < run.observations.size(); ++i)
          if (run.observations[i].graph_executed)
            hyle::engine::reporting::write_preparation_observation(std::cerr, run.observations[i], run.preparation[i]);
      }
      // Define after the execution observations. Call once in the single
      // schedule's top-level object, or once in the multi-schedule top-level
      // object; do not call for each nested Session. Existing output sets
      // setprecision(17). Session IDs were validated as ASCII [A-Za-z0-9_-].
      const auto print_execution = [&] {
        std::cout << ",\"actual_graph_calls\":" << executed_graph_calls + model.frontier_sampling_graph_calls
                  << ",\"actual_graph_tokens\":" << executed_graph_tokens
                  << ",\"max_batch_requests\":" << max_batch_requests
                  << ",\"batches\":[";
        for (std::size_t i = 0; i < batches.size(); ++i) {
          const auto &batch = batches[i];
          std::cout << (i ? "," : "") << "{\"kind\":\""
                    << (batch.kind == qwen35::benchmark::PhaseKind::kPrompt ? "prefill" : "decode")
                    << "\",\"restore\":" << (batch.restore ? "true" : "false")
                    << ",\"query_rows\":" << batch.tokens
                    << ",\"requests\":" << batch.sessions.size()
                    << ",\"seconds\":" << batch.seconds << ",\"session_ids\":[";
          for (std::size_t j = 0; j < batch.sessions.size(); ++j)
            std::cout << (j ? "," : "") << '"' << schedules[batch.sessions[j]].id << '"';
          std::cout << "]}";
        }
        std::cout << "],\"waves\":[";
        for (std::size_t i = 0; i < wave_observations.size(); ++i) {
          const auto &wave = wave_observations[i];
          std::cout << (i ? "," : "") << "{\"kind\":\""
                    << (wave.kind == 0 ? "prefill" : wave.kind == 1 ? "decode" : "mixed")
                    << "\",\"outputs\":" << wave.outputs
                    << ",\"logical_tokens\":" << wave.logical_tokens
                    << ",\"actual_tokens\":" << wave.actual_tokens
                    << ",\"graph_calls\":" << wave.graph_calls
                    << ",\"seconds\":" << wave.seconds << '}';
        }
        std::cout << ']';
      };
      const auto print_result = [&](std::size_t s) {
        const auto &run = runs[s];
        const auto &phases = schedules[s].phases;
        std::cout << "{\"source_id\":\"" << model.source_identity << "\",\"score_dtype\":\"F32\",\"score_policy\":\"" << model.score_precision_policy()
                  << '"' << hyle::engine::reporting::qwen35_weight_inventory(model.weight_inventory_facts(), routes) << ",\"visible_tokens\":"
                  << run.visible_tokens << ",\"generated_ids\":[";
        for (std::size_t i = 0; i < run.generated.size(); ++i) std::cout << (i ? "," : "") << run.generated[i];
        std::cout << "],\"fixed_decode\":true,\"decode_input_ids\":[";
        for (std::size_t i = 0; i < run.consumed.size(); ++i) std::cout << (i ? "," : "") << run.consumed[i];
        std::cout << "],\"schedule\":true,\"phase_count\":" << phases.size()
                  << ",\"turn_count\":" << phases.back().turn
                  << ",\"processed_tokens\":" << run.visible_tokens
                  << ",\"generated_tokens\":" << run.generated.size()
                  << ",\"session_seconds\":" << std::chrono::duration<double>(run.endpoint - run.start).count()
                  << ",\"startup_total_seconds\":" << std::chrono::duration<double>(run.endpoint - startup).count()
                  << ",\"step_sum_seconds\":" << run.step_sum_seconds
                  << ",\"cache_hits\":" << run.cache_hits << ",\"cache_misses\":" << run.cache_misses
                  << ",\"restored_calls\":" << run.restored_calls << ",\"restored_tokens\":" << run.restored_tokens
                  << ",\"restore_contract\":\"recompute_original_inputs\""
                  << ",\"cache_prepare_seconds\":" << run.cache_prepare_seconds
                  << ",\"prefix_cache\":\"" << (prefix_enabled ? "auto" : "off") << '"'
                  << ",\"prefix_hits\":" << run.prefix_hits << ",\"prefix_misses\":" << run.prefix_misses
                  << ",\"prefix_full_hits\":" << run.prefix_full_hits << ",\"reused_tokens\":" << run.reused_tokens
                  << ",\"graph_participations\":" << run.graph_participations + run.prefix_sampling_calls
                  << ",\"prefix_sampling_graph_calls\":" << run.prefix_sampling_calls
                  << ",\"sampling_stream\":" << run.saved_sampling.stream
                  << ",\"sampling_draw_cursor\":" << run.saved_sampling.draw
                  << ",\"graph_input_tokens\":" << run.actual_graph_tokens
                  << hyle::engine::reporting::qwen35_cache_report(model)
                  << ",\"cache_committed_version_limit\":" << cache_slots
                  << ",\"cache_live_snapshot_versions\":" << model.snapshot_versions.live
                  << ",\"cache_peak_committed_versions\":" << peak_committed_versions
                  << ",\"diagnostic\":" << (preparation_diagnostics || !trace.empty() || !layer0_probe.empty() ? "true" : "false");
        std::cout << ",\"recomputed_decision_changes\":[";
        for (std::size_t i = 0; i < run.recomputed_decisions.size(); ++i) {
          const auto &change = run.recomputed_decisions[i];
          std::cout << (i ? "," : "") << "{\"chunk\":" << change.chunk
                    << ",\"previous\":" << change.previous << ",\"actual\":" << change.actual << '}';
        }
        std::cout << ']';
        if (multiple_schedules) std::cout << ",\"session_id\":\"" << schedules[s].id
            << "\",\"trace_subdirectory\":\"" << run.trace_subdirectory << '"';
        if (!multiple_schedules) print_execution();
        std::cout << ",\"phases\":[";
        for (std::size_t i = 0; i < phases.size(); ++i) {
          const auto &phase = phases[i]; const auto &summary = run.summaries[i];
          const bool is_prompt = phase.kind == qwen35::benchmark::PhaseKind::kPrompt;
          const auto count = is_prompt ? std::size_t{1} : phase.ids.size();
          std::cout << (i ? "," : "") << "{\"turn\":" << phase.turn
                    << ",\"kind\":\"" << (is_prompt ? "prefill" : "decode") << "\",\"name\":\"" << summary.name
                    << "\",\"input_tokens\":" << phase.ids.size() << ",\"calls\":" << count << ",\"generated_tokens\":" << count
                    << ",\"context_before\":" << summary.context_before << ",\"context_after\":" << summary.context_after
                    << ",\"seconds\":" << summary.seconds << ",\"step_sum_seconds\":" << summary.step_sum_seconds << '}';
        }
        std::cout << "]}";
      };
      std::cerr << "first_token_elapsed_seconds=" << runs.front().first_token_elapsed_seconds << '\n';
      std::cout << std::setprecision(17);
      if (!multiple_schedules) print_result(0);
      else {
        std::size_t processed = 0, graph_participations = 0;
        std::size_t prefix_hits = 0, prefix_misses = 0, prefix_full_hits = 0, reused_tokens = 0;
        double step_sum = 0;
        for (const auto &run : runs) {
          processed += run.visible_tokens; step_sum += run.step_sum_seconds;
          graph_participations += run.graph_participations + run.prefix_sampling_calls;
          prefix_hits += run.prefix_hits; prefix_misses += run.prefix_misses;
          prefix_full_hits += run.prefix_full_hits; reused_tokens += run.reused_tokens;
        }
        std::cout << "{\"source_id\":\"" << model.source_identity << "\",\"score_dtype\":\"F32\",\"score_policy\":\"" << model.score_precision_policy()
                  << '"' << hyle::engine::reporting::qwen35_weight_inventory(model.weight_inventory_facts(), routes) << ",\"session_count\":" << runs.size()
                  << ",\"execution\":\"equal_T_batched\",\"processed_tokens\":" << processed
                  << ",\"generated_tokens\":" << all_calls
                  << ",\"graph_participations\":" << graph_participations
                  << ",\"prefix_hits\":" << prefix_hits << ",\"prefix_misses\":" << prefix_misses
                  << ",\"prefix_full_hits\":" << prefix_full_hits << ",\"reused_tokens\":" << reused_tokens
                  << ",\"interleaved_seconds\":" << std::chrono::duration<double>(endpoint - journey_start).count()
                  << ",\"startup_total_seconds\":" << std::chrono::duration<double>(endpoint - startup).count()
                  << ",\"step_sum_seconds\":" << step_sum
                  << ",\"max_cached_sessions\":" << max_sessions << ",\"active_sequences\":" << active_slots
                  << ",\"effective_cache_sessions\":" << cache_slots
                  << ",\"cache_recurrent_payload_bytes_per_snapshot\":" << recurrent_payload
                  << hyle::engine::reporting::qwen35_cache_report(model)
                  << ",\"cache_page_rows\":" << qwen35::cache::Plan::page_rows
                  << ",\"cache_max_physical_pages\":" << selected_cache.max_pages
                  << ",\"cache_live_snapshot_versions\":" << model.snapshot_versions.live
                  << ",\"cache_peak_committed_versions\":" << peak_committed_versions
                  << ",\"cache_committed_version_limit\":" << cache_slots
                  << ",\"cache_retained_sessions\":" << resident_cache.retained_sessions()
                  << ",\"prefix_cache\":\"" << (prefix_enabled ? "auto" : "off") << '"'
                  << ",\"prefix_entry_capacity\":" << (prefix_enabled ? cache_slots : 0)
                  << ",\"prefix_entries\":" << (prefix_cache ? prefix_cache->size() : 0)
                  << ",\"prefix_evictions\":" << (prefix_cache ? prefix_cache->evictions() : 0)
                  << ",\"diagnostic\":" << (preparation_diagnostics || !trace.empty() || !layer0_probe.empty() ? "true" : "false")
                  << ",\"logical_dispatch_session_ids\":[";
        for (std::size_t i = 0; i < dispatch.size(); ++i)
          std::cout << (i ? "," : "") << '"' << schedules[dispatch[i].first].id << '"';
        std::cout << "],\"evicted_session_ids\":[";
        for (std::size_t i = 0; i < evicted_sessions.size(); ++i)
          std::cout << (i ? "," : "") << '"' << schedules[evicted_sessions[i]].id << '"';
        std::cout << ']';
        print_execution();
        std::cout << ",\"sessions\":[";
        for (std::size_t s = 0; s < runs.size(); ++s) { if (s) std::cout << ','; print_result(s); }
        std::cout << "]}";
      }
      std::cout << '\n';
      return 0;
    }
    std::cerr << "measure_memory name=ready_before_first\nmemory_phase=generation_prefill_begin\n";
    double prefill_step_seconds = 0, continuation_step_seconds = 0, continuation_seconds = 0, decode_step_sum_seconds = 0;
    qwen35::Clock::time_point prefill_endpoint{};
    const auto prefill_start = qwen35::Clock::now();
    auto next = qwen35::evaluate(model, *session, prompt, "prefill", trace, trace_states,
        &prefill_step_seconds, &prefill_endpoint);
    auto endpoint = prefill_endpoint;
    std::cerr << "first_token_elapsed_seconds=" << std::chrono::duration<double>(prefill_endpoint - startup).count() << '\n';
    std::cerr << "memory_phase=generation_prefill_end\nmemory_phase=generation_decode_begin\n";
    if (!extra.empty()) {
      const auto continuation_start = qwen35::Clock::now();
      next = qwen35::evaluate(model, *session, extra, "continuation", trace, trace_states,
          &continuation_step_seconds, &endpoint);
      continuation_seconds = std::chrono::duration<double>(endpoint - continuation_start).count();
    }
    generated.push_back(next);
    for (std::size_t step = 1; step < steps && (!forced.empty() || !qwen35::is_end_of_generation(next)); ++step) {
      const auto previous = forced.empty() ? next : forced[step - 1];
      double step_seconds = 0;
      next = qwen35::evaluate(model, *session, std::span(&previous, 1), "decode" + std::to_string(step), trace, trace_states,
          &step_seconds, &endpoint);
      decode_step_sum_seconds += step_seconds;
      consumed.push_back(previous); generated.push_back(next);
    }
    std::cerr << "memory_phase=generation_decode_end\n";
    const auto prefill_seconds = std::chrono::duration<double>(prefill_endpoint - prefill_start).count();
    const auto decode_seconds = std::chrono::duration<double>(endpoint - prefill_endpoint).count();
    const auto session_seconds = std::chrono::duration<double>(endpoint - prefill_start).count();
    const auto graph_calls = 1 + std::size_t(!extra.empty()) + consumed.size();
    std::cout << std::setprecision(17)
              << "{\"source_id\":\"" << model.source_identity << "\",\"score_dtype\":\"F32\",\"score_policy\":\"" << model.score_precision_policy()
              << "\",\"mtp\":false" << hyle::engine::reporting::qwen35_weight_inventory(model.weight_inventory_facts(), routes) << hyle::engine::reporting::qwen35_cache_report(model)
              << ",\"sampling_stream\":" << session->head->sampling.stream
              << ",\"sampling_draw_cursor\":" << session->head->sampling.draw
              << ",\"visible_tokens\":" << session->length() << ",\"generated_ids\":[";
    for (std::size_t i = 0; i < generated.size(); ++i) std::cout << (i ? "," : "") << generated[i];
    std::cout << "],\"fixed_decode\":" << (forced.empty() ? "false" : "true") << ",\"decode_input_ids\":[";
    for (std::size_t i = 0; i < consumed.size(); ++i) std::cout << (i ? "," : "") << consumed[i];
    std::cout << "],\"prefill_seconds\":" << prefill_seconds << ",\"decode_seconds\":" << decode_seconds
              << ",\"session_seconds\":" << session_seconds
              << ",\"startup_total_seconds\":" << std::chrono::duration<double>(endpoint - startup).count()
              << ",\"startup_to_endpoint_seconds\":" << std::chrono::duration<double>(endpoint - startup).count()
              << ",\"startup_to_ready_seconds\":" << std::chrono::duration<double>(ready - startup).count()
              << ",\"first_token_elapsed_seconds\":" << std::chrono::duration<double>(prefill_endpoint - startup).count()
              << ",\"prefill_step_seconds\":" << prefill_step_seconds
              << ",\"continuation_seconds\":" << continuation_seconds << ",\"continuation_step_seconds\":" << continuation_step_seconds
              << ",\"decode_step_sum_seconds\":" << decode_step_sum_seconds
              << ",\"step_sum_seconds\":" << prefill_step_seconds + continuation_step_seconds + decode_step_sum_seconds
              << ",\"prefill_tokens_per_second\":" << (prefill_seconds > 0 ? prompt.size() / prefill_seconds : 0)
              << ",\"decode_output_tokens_per_second\":" << (decode_seconds > 0 ? consumed.size() / decode_seconds : 0)
              << ",\"output_tokens_per_second\":" << (session_seconds > 0 ? generated.size() / session_seconds : 0)
              << ",\"requested_steps\":" << steps << ",\"generated_tokens\":" << generated.size()
              << ",\"generated_decode_tokens\":" << consumed.size() << ",\"actual_graph_calls\":" << graph_calls
              << ",\"prefill_graph_calls\":1,\"decode_graph_calls\":" << consumed.size()
              << ",\"continuation_graph_calls\":" << std::size_t(!extra.empty())
              << ",\"prefill_target_rows\":" << prompt.size() << ",\"prefill_draft_rows\":0"
              << ",\"continuation_target_rows\":" << extra.size() << ",\"verified_target_rows\":" << consumed.size()
              << ",\"actual_target_rows\":" << prompt.size() + extra.size() + consumed.size() << ",\"actual_draft_rows\":0"
              << ",\"committed_target_rows\":" << session->length()
              << ",\"committed_target_decode_rows\":" << consumed.size()
              << ",\"draft_rows\":0,\"draft_rounds\":0,\"agreements\":0,\"consumed_drafts\":0"
              << ",\"decode_seconds_includes_continuation\":" << (extra.empty() ? "false" : "true")
              << ",\"diagnostic\":" << (!trace.empty() || !layer0_probe.empty() ? "true" : "false") << "}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Qwen3.5 text consumer: " << error.what() << '\n'; return 1;
  }
}
