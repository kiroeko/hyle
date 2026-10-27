#include "../../engine/models/qwen35/main.cc"
#include "qwen_generation_state_case.h"

// A finite RequestLoop consumer of the ordinary R1 greedy path. The original
// Model owns submission, candidate publication, cancellation and retirement.
// Extra stop IDs below are request policy from actual fixture outputs,
// not model EOS; this makes no throughput or continuous-pipeline claim.
namespace {
namespace q = qwen35;
namespace i = hyle::inference;
namespace c = hyle::compute;

// The real prompt is followed by at most thirteen consumed T1 inputs.
constexpr std::size_t kOutputLimit = 14;

void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}

using qwen_generation_state_case::StateBytes;
using qwen_generation_state_case::state_bytes;

struct CommitFact final {
  q::GenerationBoundary before, after;
  bool generation = false, stopped = false;
  std::size_t accepted = 0;
  std::vector<std::int32_t> inputs, emitted;
  friend bool operator==(const CommitFact &, const CommitFact &) = default;
};

std::vector<CommitFact> commit_facts(const q::GenerationRequest &request) {
  std::vector<CommitFact> result;
  result.reserve(request.ledger->committed_count);
  for (const auto &record : std::span(request.ledger->records).first(request.ledger->committed_count)) {
    require(record.committed && !record.input_deferred && record.role == q::GraphRole::kText &&
        record.user_output && record.requests == 1 && record.member_index == 0,
        "ordinary R1 ledger retained a deferred, non-output or foreign member fact");
    const auto ids = record.committed_ids();
    require(!ids.empty(), "generation ledger lost its actual consumed token prefix");
    result.push_back({record.before, record.after, record.generation, record.stopped,
        record.accepted, {ids.begin(), ids.end()},
        {record.emitted.begin(), record.emitted.begin() + record.emitted_count}});
  }
  return result;
}

enum class Arm { kCreditOne, kCreditTwoPause, kExtraStop, kAfterSubmitCancel, kLateExtraStop };
const char *arm_name(Arm arm) {
  switch (arm) {
  case Arm::kCreditOne: return "credit1";
  case Arm::kCreditTwoPause: return "credit2_pause_resume";
  case Arm::kExtraStop: return "extra_stop";
  case Arm::kAfterSubmitCancel: return "after_submit_cancel";
  case Arm::kLateExtraStop: return "late_extra_stop";
  }
  throw std::logic_error("unknown finite generation arm");
}

struct ArmResult final {
  std::vector<std::int32_t> ids;
  std::vector<CommitFact> commits;
  std::map<std::size_t, StateBytes> states;
  std::map<std::size_t, std::uint64_t> epochs;
  std::size_t waves = 0, credit_stalls = 0, cancelled = 0;
  std::size_t completed_steps = 0, successor_submissions = 0,
      successor_publications = 0, successor_discards = 0, successor_refusals = 0;
  bool manual_pause = false, cancel_after_submit = false, cancellation_exception = false;
};

ArmResult run_arm(q::Model &model, std::span<const std::int32_t> prompt, Arm arm, bool relay,
                  const ArmResult *oracle = nullptr, std::int32_t extra_stop = -1) {
  require(!model.active_step && !model.successor_step && !model.carrier_borrowed && !model.snapshot_versions.live,
      "preceding RequestLoop arm retained an episode or model version");
  const std::array turns{i::GenerationTurn{{prompt.begin(), prompt.end()}, kOutputLimit}};
  const std::array stop_ids{extra_stop};
  const auto stop = arm == Arm::kExtraStop || arm == Arm::kLateExtraStop
      ? i::RequestStopPolicy(stop_ids) : i::RequestStopPolicy{};
  std::vector<q::GenerationRequest> requests;
  requests.emplace_back(arm_name(arm), turns[0].prompt, std::filesystem::path{}, kOutputLimit, 0, stop);
  auto &request = requests.front();
  request.progress.bind_turns(turns);
  // Session creation canonicalizes a disabled sampler to the zero cursor.
  request.progress.initial_sampling = {};
  request.input = q::cache::token_plan(prompt, model.capacity);
  request.ledger = std::make_unique<q::GenerationLedger>(q::generation_record_capacity(turns), 1);
  request.round_emitted.resize(kOutputLimit - 1);
  for (auto &row : request.round_emitted) row.reserve(1);
  request.turn_observations.resize(1);
  request.warm_inputs.push_back(std::make_shared<std::vector<std::int32_t>>(prompt.begin(), prompt.end()));
  i::OutputDelivery delivery(arm == Arm::kCreditOne ? 1 : 2);
  i::RequestLoop loop;
  loop.prepare(1, 1);
  q::GenerationContinuation continuation;
  ArmResult result;
  std::vector<std::weak_ptr<q::StepSources>> source_lifetimes;
  std::optional<q::GenerationBoundary> cancelled_head;
  std::size_t cancelled_commits = 0;
  struct StopView final {
    q::GenerationRequest &request;
    hyle::Status stop(i::RequestKey) const noexcept {
      return request.session ? request.session->stop() : hyle::Status::Ok();
    }
  } stop_view{request};
  hyle::util::ScopeGuard cleanup{[&]() noexcept {
    if (model.successor_step && model.successor_step->has_completion())
      model.successor_step->wait().IgnoreError();
    if (model.active_step && model.active_step->has_completion())
      model.active_step->wait().IgnoreError();
    continuation.reset();
    (void)model.abort_episode();
    if (request.session) {
      request.session->stop().IgnoreError();
      (void)request.session->drain(); (void)request.session->try_retire();
      request.session.reset();
    }
  }};
  const auto execute = [&](std::span<const i::RequestKey> active) {
    require(active.size() == 1 && active[0] == request.request_key,
        "generation relay changed its original R1 request identity");
    const auto before_outputs = request.progress.generated.size();
    const auto available = loop.credit(request.request_key, before_outputs);
    require(available, "RequestLoop executed with zero real output credit");
    // Reach the same eight-output control boundary in both arms even when a
    // relay wave can commit two original steps. This only narrows real credit.
    const auto credit = arm == Arm::kAfterSubmitCancel && before_outputs < 8
        ? (std::min)(available, 8 - before_outputs) : available;
    std::array calls{q::prepare_generation_call(model, request, request.progress.output_limit(),
        nullptr, requests, credit)};
    const auto submitted = [&] {
      if (arm != Arm::kAfterSubmitCancel || result.cancel_after_submit || before_outputs < 8) return;
      require(model.active_step && model.active_step->submitted && model.active_step->has_completion(),
          "cancellation did not observe an original accepted submission");
      cancelled_head = q::generation_boundary(*request.session->head);
      cancelled_commits = request.ledger->committed_count;
      const auto status = loop.cancel(request.request_key, stop_view);
      require(status.ok() || (status.IsCancelled() && !status.has_native_error()),
          "original after-submit cancellation returned an unexpected status");
      result.cancel_after_submit = true;
      require(q::generation_boundary(*request.session->head) == *cancelled_head &&
          request.ledger->committed_count == cancelled_commits,
          "cancellation itself published a candidate or advanced the ledger");
    };
    try {
      q::execute_generation_calls(model, calls, false, nullptr, requests, submitted,
          relay ? &continuation : nullptr);
    } catch (const std::exception &error) {
      if (!result.cancel_after_submit) throw;
      std::cerr << "generation-cancel-finish: " << error.what() << '\n';
      const auto retired = model.abort_episode();
      require(retired.retired && (retired.status.ok() ||
          (retired.status.IsCancelled() && !retired.status.has_native_error())),
          "cancelled original episode did not actually retire");
      q::observe_generation_commits(request);
      require(request.ledger->committed_count == cancelled_commits &&
          request.progress.generated.size() == before_outputs,
          "cancelled episode published an output through its exceptional finish");
      result.cancellation_exception = true;
    }
    for (auto &call : calls) q::publish_generation_prompt(model, call, nullptr);
    ++result.waves;
    require(request.progress.generated.size() - before_outputs <= credit &&
        request.progress.observed_commits() == request.ledger->committed_count,
        "generation wave exceeded credit or lost committed observations");
    require(!model.active_step && !model.successor_step && !model.carrier_borrowed &&
        (!request.session || !request.session->pending),
        "RequestLoop wave returned before its original episodes retired");
    if (relay && continuation.ready) source_lifetimes.push_back(continuation.ready);
    if (request.session && request.session->head) {
      const auto count = request.progress.generated.size();
      if (count) {
        result.states.insert_or_assign(count, state_bytes(model, *request.session->head));
        result.epochs.insert_or_assign(count, request.session->head.epoch());
      }
      require(request.session->head->sampling == request.progress.initial_sampling,
          "ordinary greedy generation changed its original sampling cursor");
      if (oracle && count) {
        const auto found = oracle->states.find(count);
        require(found != oracle->states.end() && found->second == result.states.at(count) &&
            oracle->epochs.at(count) == result.epochs.at(count),
            "natural relay changed complete state at a committed serial prefix");
      }
    }
    // The owner is about to pause for real transport credit. A ready Source
    // must not keep the current version pinned across that idle boundary.
    if (relay && !loop.credit(request.request_key, request.progress.generated.size()))
      continuation.release_ready();
  };
  q::GenerationRequestAdapter adapter{model, static_cast<q::cache::PrefixCache *>(nullptr),
      std::span(requests), false, static_cast<std::vector<std::size_t> *>(nullptr), execute,
      {}, relay ? &continuation : nullptr};
  request.request_key = loop.enqueue(0, &delivery);
  const auto key = request.request_key;
  for (std::size_t pumps = 0; loop.matches(key); ++pumps) {
    require(pumps < 64, "finite RequestLoop journey failed to retire");
    const auto pumped = loop.pump_once(adapter);
    result.cancelled += pumped.cancelled;
    if (!loop.matches(key)) break;
    if (arm == Arm::kCreditTwoPause && !result.manual_pause &&
        request.progress.generated.size() == 1 && !loop.active().empty()) {
      const auto head = q::generation_boundary(*request.session->head);
      const auto facts = commit_facts(request);
      const auto waves = result.waves;
      if (relay) continuation.release_ready();
      (void)loop.pause(key);
      const auto paused = loop.pump_once(adapter);
      require(!paused.executed && result.waves == waves &&
          q::generation_boundary(*request.session->head) == head && commit_facts(request) == facts,
          "manual pause submitted work or changed committed pending/RNG");
      loop.wake(key); result.manual_pause = true;
    }
    const auto produced = request.progress.generated.size();
    if (delivery.pending(request.progress.generated).empty()) continue;
    if (!delivery.available(produced) || !request.session || request.progress.turn_done || loop.cancelled(key)) {
      const auto before = result.waves;
      const auto ledger = request.ledger->committed_count;
      const auto blocked = loop.pump_once(adapter);
      require(!blocked.executed && result.waves == before &&
          request.ledger->committed_count == ledger && request.progress.generated.size() == produced,
          "credit exhaustion or terminal delivery resubmitted native work");
      ++result.credit_stalls;
      require(loop.acknowledge(key, produced, adapter), "original delivery prefix ACK was rejected");
    } else if (arm != Arm::kCreditOne && result.credit_stalls) {
      // After proving one actual full-credit pause, deliver partial prefixes
      // promptly. Nonzero credit keeps the same ready Source across enough
      // imported warm steps; this ACK creates no token or device permission.
      require(loop.acknowledge(key, produced, adapter), "partial delivery prefix ACK was rejected");
    }
  }
  result.ids = request.progress.generated;
  result.commits = commit_facts(request);
  if (arm == Arm::kAfterSubmitCancel) {
    require(result.cancel_after_submit && result.cancelled == 1 && result.ids.size() == 8 &&
        request.ledger->committed_count == cancelled_commits,
        "after-submit cancellation lost its original committed eight-token prefix");
  } else {
    require(!result.cancelled && request.progress.request_done(), "ordinary request did not complete");
    if (arm == Arm::kExtraStop || arm == Arm::kLateExtraStop)
      require(!result.ids.empty() && result.ids.back() == extra_stop && result.ids.size() < kOutputLimit &&
          result.commits.back().stopped, "extra stop policy did not stop on its real selected token");
    else require(result.ids.size() == kOutputLimit, "ordinary request lost an output token");
  }
  if (arm == Arm::kCreditTwoPause) require(result.manual_pause, "manual pause/resume was not exercised");
  require(result.credit_stalls && !request.session && !model.active_step && !model.successor_step &&
      !model.carrier_borrowed && !model.snapshot_versions.live && !loop.waiting_front() && loop.active().empty(),
      "RequestLoop terminal cleanup retained an identity, source generation or candidate");
  if (oracle) require(result.ids == oracle->ids && result.commits == oracle->commits,
      "natural relay changed token/pending/ledger facts against the same-width serial path");
  result.completed_steps = continuation.completed_steps;
  result.successor_submissions = continuation.successor_submissions;
  result.successor_publications = continuation.successor_publications;
  result.successor_discards = continuation.successor_discards;
  result.successor_refusals = continuation.successor_refusals;
  if (relay) {
    require(!continuation.unavailable &&
        result.completed_steps + request.prefill_graph_calls == result.commits.size() &&
        !result.successor_refusals &&
        result.successor_publications + result.successor_discards == result.successor_submissions,
        "ordinary continuation silently fell back or lost an accepted/published step");
    if (arm == Arm::kLateExtraStop)
      require(result.successor_discards == 1,
          "late extra stop did not retire exactly one accepted, unpublished successor");
    else require(!result.successor_discards, "ordinary arm discarded an unexpected successor");
    if (arm == Arm::kCreditOne || arm == Arm::kExtraStop)
      require(!result.successor_submissions,
          "single-credit or early-stop arm unexpectedly accepted a successor");
    if (arm == Arm::kCreditTwoPause)
      require(result.successor_submissions,
          "natural credit-two RequestLoop never submitted its warmed successor");
  }
  continuation.reset();
  for (const auto &lifetime : source_lifetimes)
    require(lifetime.expired(), "terminal RequestLoop retained an original Source generation");
  cleanup.dismiss();
  return result;
}

int run(const std::filesystem::path &input) {
  auto source = q::prepare_weight_source(input / "weights.tsv", false);
  const auto geometry = source.roles.geometry;
  const auto context_limit = static_cast<std::size_t>(geometry.context_limit);
  const auto prompt = q::read_ids(input / "prompt.ids", geometry.vocabulary, context_limit);
  const auto journey_rows = q::cache::add(prompt.size(), kOutputLimit - 1);
  const auto capacity = (std::min)(context_limit, (std::max)(std::size_t{32}, journey_rows));
  require(!prompt.empty() && capacity >= journey_rows && geometry.vocabulary > 3 &&
      geometry.recurrent_layers() && geometry.attention_layers(),
      "generation relay fixture requires real recurrent/KV state and bounded context");
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, capacity, 3, prompt.size(),
      q::recurrent_bytes(q::state_schema_for(geometry, capacity), geometry), 1, 1, false, false, 2);
  q::Model model(std::move(source), geometry, capacity, plan, {}, false,
      q::ReplayPolicy::kPreferCapture, SIZE_MAX, {}, 0, {.primary = i::BackendKind::kCuda});
  const std::array shapes{prompt.size(), std::size_t{1}};
  model.prepare_family(shapes);
  model.plan_replay_uses(prompt.size(), 1, 10);
  model.plan_replay_uses(1, 1, kOutputLimit * 10);
  model.freeze_shape_plan(); model.prepare_execution(shapes); model.prepare_relay_execution();
  const auto serial_credit_one = run_arm(model, prompt, Arm::kCreditOne, false);
  const auto credit_one = run_arm(model, prompt, Arm::kCreditOne, true, &serial_credit_one);
  const auto serial_credit_two = run_arm(model, prompt, Arm::kCreditTwoPause, false);
  const auto credit_two = run_arm(model, prompt, Arm::kCreditTwoPause, true, &serial_credit_two);
  const auto stop_id = serial_credit_one.ids.at(2);
  const auto serial_stop = run_arm(model, prompt, Arm::kExtraStop, false, nullptr, stop_id);
  const auto stopped = run_arm(model, prompt, Arm::kExtraStop, true, &serial_stop, stop_id);
  const auto serial_cancel = run_arm(model, prompt, Arm::kAfterSubmitCancel, false);
  const auto cancelled = run_arm(model, prompt, Arm::kAfterSubmitCancel, true, &serial_cancel);
  std::optional<std::int32_t> late_stop;
  // With this bounded ACK schedule, the first pending pair follows six real
  // imported warm steps at outputs ten/eleven; another can end at fourteen.
  // A stop token must first occur at A, never earlier or only at its B result.
  for (const auto at : {std::size_t{9}, std::size_t{12}}) {
    const auto id = serial_credit_one.ids.at(at);
    if (std::find(serial_credit_one.ids.begin(), serial_credit_one.ids.begin() + at, id) ==
        serial_credit_one.ids.begin() + at) { late_stop = id; break; }
  }
  std::optional<ArmResult> late_stopped;
  if (late_stop) {
    const auto serial_late_stop = run_arm(model, prompt, Arm::kLateExtraStop, false, nullptr, *late_stop);
    late_stopped = run_arm(model, prompt, Arm::kLateExtraStop, true, &serial_late_stop, *late_stop);
  } else {
    std::cerr << "late-extra-stop UNRUN: no first-occurring token at a warmed predecessor boundary\n";
  }
  std::cout << "{\"status\":\"PASS\",\"backend\":\"cuda\",\"sampling\":\"greedy\","
      "\"request_loop\":true,\"prompt_tokens\":" << prompt.size()
      << ",\"capacity\":" << capacity << ",\"serial_and_relay_arms\":" << (late_stopped ? 10 : 8)
      << ",\"credit1\":true,\"credit2\":true,"
      "\"pause_resume\":true,\"extra_stop\":true,\"model_eos_run\":false,"
      "\"late_extra_stop_run\":" << (late_stopped ? "true" : "false")
      << ",\"late_extra_stop_missing_unique_token\":" << (late_stop ? "false" : "true") << ','
      << "\"after_submit_cancel\":true,\"complete_state_equal\":true,\"final_live_versions\":0,"
      "\"controlled_pending_gate\":false,\"unknown_capture_tuple_run\":false,"
      "\"performance_claim\":false,\"continuous_pipeline_claim\":false,\"waves\":["
      << credit_one.waves << ',' << credit_two.waves << ',' << stopped.waves << ',' << cancelled.waves
      << "],\"continuations\":[";
  const std::array<const ArmResult *, 5> results{&credit_one, &credit_two, &stopped, &cancelled,
      late_stopped ? &*late_stopped : nullptr};
  for (std::size_t index = 0; index < results.size(); ++index) {
    if (!results[index]) continue;
    const auto &result = *results[index];
    std::cout << (index ? "," : "") << "{\"completed_steps\":" << result.completed_steps
        << ",\"successor_submissions\":" << result.successor_submissions
        << ",\"successor_publications\":" << result.successor_publications
        << ",\"successor_discards\":" << result.successor_discards
        << ",\"successor_refusals\":" << result.successor_refusals << '}';
  }
  std::cout << "]}\n";
  return 0;
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 1) { std::cerr << "SKIP: pass --cuda PREPARED_LOCAL_INPUTS\n"; return 77; }
    if (argc != 3 || std::string_view(argv[1]) != "--cuda")
      throw std::runtime_error("usage: qwen_generation_relay_probe --cuda PREPARED_LOCAL_INPUTS");
    const auto compiled = q::Backend::compiled_backends();
    if (std::find(compiled.begin(), compiled.end(), q::BackendKind::kCuda) == compiled.end()) {
      std::cerr << "SKIP: CUDA route is not built\n"; return 77;
    }
    return run(argv[2]);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
