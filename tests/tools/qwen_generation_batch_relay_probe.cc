#include "../../engine/models/qwen35/main.cc"
#include "qwen_generation_state_case.h"
#include "qwen_continuous_generation_case.h"

// Finite CUDA RequestLoop consumer. The original Model owns all target work,
// Source permissions and retirement. No native gate or output is fabricated.
namespace {
namespace q = qwen35;
namespace i = hyle::inference;
using qwen_generation_state_case::require;
using qwen_generation_state_case::StateBytes;
using qwen_generation_state_case::state_bytes;
constexpr std::size_t kOutputs = 14;
enum class Mode { kNormal, kMemberCredit, kCancel, kModelEos, kColdTuple };

struct BatchCopy final {
  std::vector<i::RequestKey> keys;
  std::vector<q::EvaluationDisposition> dispositions;
  std::size_t rows = 0;
  bool generation = false;
};
struct ArmResult final {
  std::array<std::vector<std::int32_t>, 2> ids;
  std::array<std::vector<q::GenerationCommitRecord>, 2> records;
  std::array<std::map<std::size_t, StateBytes>, 2> states;
  std::array<std::map<std::size_t, std::uint64_t>, 2> epochs;
  std::array<i::RequestKey, 2> keys;
  std::vector<BatchCopy> batches;
  std::optional<std::array<std::size_t, 2>> first_successor_before, cancelled_before;
  std::size_t waves = 0, submits = 0, completed_steps = 0, successors = 0,
      publications = 0, discards = 0, refusals = 0, published_members = 0;
  std::size_t credit_blocked = 0, cancelled = 0, survivor_r1 = 0, discarded_batches = 0;
  bool cold_tuple_changed = false, cold_refusal_observed = false, accepted_eos_successor = false;
};

void compare(const ArmResult &control, const ArmResult &candidate) {
  require(control.ids == candidate.ids, "R2 relay changed original per-request generated IDs");
  for (std::size_t member = 0; member < 2; ++member) {
    require(control.records[member].size() == candidate.records[member].size(), "R2 relay changed ledger length");
    for (std::size_t row = 0; row < control.records[member].size(); ++row) {
      auto expected = control.records[member][row];
      const auto &actual = candidate.records[member][row];
      // A serial next wave can have two fresh delivery credits, whereas B in
      // the same quantum has one left. Each quota is checked against its own
      // real credit in the actual-batch observer below, not silently ignored.
      expected.quota = actual.quota;
      require(expected.same_result(actual) && expected.committed == actual.committed &&
          expected.user_output == actual.user_output && expected.imported == actual.imported &&
          expected.turn == actual.turn && std::ranges::equal(expected.ids(), actual.ids()),
          "R2 relay changed a complete non-time ledger or original member fact");
    }
    for (const auto &[count, bytes] : candidate.states[member]) {
      const auto found = control.states[member].find(count);
      require(found != control.states[member].end() && found->second == bytes &&
          control.epochs[member].at(count) == candidate.epochs[member].at(count),
          "R2 relay changed complete initialized state, epoch, pending or RNG at an original serial prefix");
    }
  }
}

ArmResult run_arm(q::Model &model, const std::array<std::vector<std::int32_t>, 2> &prompts,
    Mode mode, bool relay, std::optional<std::array<std::size_t, 2>> cancel_at = {}) {
  require(!model.active_step && !model.successor_step && !model.carrier_borrowed && !model.snapshot_versions.live,
      "preceding R2 relay arm retained original episodes or state");
  std::array<std::array<i::GenerationTurn, 1>, 2> turns{
      std::array{i::GenerationTurn{prompts[0], kOutputs}}, std::array{i::GenerationTurn{prompts[1], kOutputs}}};
  std::vector<q::GenerationRequest> requests;
  requests.reserve(2);
  for (std::size_t member = 0; member < 2; ++member) {
    requests.emplace_back(member ? "batch-B" : "batch-A", prompts[member], std::filesystem::path{}, kOutputs, 0,
        mode == Mode::kColdTuple ? i::RequestStopPolicy({}, true) : i::RequestStopPolicy{});
    auto &request = requests.back();
    request.progress.bind_turns(turns[member]); request.progress.initial_sampling = {};
    request.input = q::cache::token_plan(prompts[member], model.capacity);
    request.ledger = std::make_unique<q::GenerationLedger>(q::generation_record_capacity(turns[member]), 1);
    request.round_emitted.resize(kOutputs - 1);
    for (auto &values : request.round_emitted) values.reserve(1);
    request.turn_observations.resize(1);
    request.warm_inputs.push_back(std::make_shared<std::vector<std::int32_t>>(prompts[member]));
  }
  std::array deliveries{i::OutputDelivery(2), i::OutputDelivery(2)};
  i::RequestLoop loop; loop.prepare(2, 2);
  q::GenerationContinuation continuation; continuation.prepare(2);
  ArmResult result;
  std::vector<std::weak_ptr<q::StepSources>> sources;
  std::array<std::size_t, 2> observed_records{};
  bool cancel_sent = false, leave_one_pending = false, credit_done = false, eos_finished = false;
  std::optional<q::PreparedReadback> displaced_output;
  std::array<bool, q::execution_slot_count> imported_captures{};
  struct StopView final {
    std::span<q::GenerationRequest> requests;
    hyle::Status stop(i::RequestKey key) const noexcept {
      return requests[key.slot].session ? requests[key.slot].session->stop() : hyle::Status::Ok();
    }
  } stop_view{requests};
  hyle::util::ScopeGuard cleanup{[&]() noexcept {
    if (model.successor_step && model.successor_step->execution.has_completion()) model.successor_step->execution.wait().IgnoreError();
    if (model.active_step && model.active_step->execution.has_completion()) model.active_step->execution.wait().IgnoreError();
    continuation.reset(); (void)model.abort_episode();
    for (auto &request : requests) if (request.session) {
      request.session->stop().IgnoreError(); (void)request.session->drain();
      (void)request.session->try_retire(); request.session.reset();
    }
  }};
  const auto observed = [&](const q::GenerationBatchFacts &facts) {
    require(!facts.members.empty() && facts.members.size() <= 2 && facts.role == q::GraphRole::kText &&
        facts.completion == q::TextCompletion::kEmitNext && std::isfinite(facts.step_seconds) &&
        facts.step_seconds > 0 && facts.input_rows == (facts.generation ? 1 : prompts[0].size()),
        "R2 observer lost an actual original target batch");
    BatchCopy copy;
    copy.rows = facts.input_rows; copy.generation = facts.generation;
    bool all_discarded = true;
    for (std::size_t index = 0; index < facts.members.size(); ++index) {
      const auto &call = facts.members[index];
      require(call.request && call.request->request_key.slot < requests.size(), "R2 facts contain a foreign request");
      auto &request = *call.request; const auto member = request.request_key.slot;
      require(&request == &requests[member] && request.request_key == result.keys[member] &&
          loop.matches(request.request_key), "R2 facts changed original RequestKey ownership");
      require(std::ranges::find(copy.keys, request.request_key) == copy.keys.end(), "R2 facts repeated a member");
      copy.keys.push_back(request.request_key); copy.dispositions.push_back(call.disposition);
      if (call.disposition == q::EvaluationDisposition::kPublished) {
        require(request.ledger->committed_count == observed_records[member] + 1,
            "R2 facts duplicated or omitted an authoritative original commit");
        const auto &record = request.ledger->records[observed_records[member]];
        require(record.committed && !record.input_deferred && record.requests == facts.members.size() &&
            record.member_index == index && record.verified_rows == facts.input_rows &&
            record.quota == call.quota &&
            record.after == q::generation_boundary(*request.session->head),
            "R2 ledger membership/geometry differs from its actual completed batch");
        if (facts.generation)
          require(call.quota == (std::min)(request.progress.remaining_outputs(),
              loop.credit(request.request_key, request.progress.generated.size())),
              "R2 ledger quota differs from its real remaining output/delivery credit");
        if (mode == Mode::kModelEos && member == 0 && record.stopped) {
          require(request.stop_policy.is_default() && q::is_end_of_generation(record.after.next_id),
              "R2 stopped the first member through something other than real model EOS");
          eos_finished = true;
        }
        ++observed_records[member]; ++result.published_members; all_discarded = false;
      } else require(call.disposition == q::EvaluationDisposition::kCancelled,
          "R2 callback described an unfinished target as completed");
    }
    const bool terminal_member = (mode == Mode::kCancel && cancel_sent) || (mode == Mode::kModelEos && eos_finished);
    if (terminal_member && all_discarded && copy.keys.size() == 2) ++result.discarded_batches;
    if (terminal_member && copy.keys.size() == 1 &&
        copy.keys[0] == result.keys[1] && !all_discarded) ++result.survivor_r1;
    result.batches.push_back(std::move(copy)); // No facts.members/call pointer escapes.
  };
  const auto execute = [&](std::span<const i::RequestKey> active) {
    require(!active.empty() && active.size() <= 2, "R2 owner selected an invalid width");
    std::array<std::size_t, 2> before{requests[0].progress.generated.size(), requests[1].progress.generated.size()};
    std::array<std::size_t, 2> credits{};
    std::vector<q::PreparedGenerationCall> calls;
    calls.reserve(active.size());
    for (const auto key : active) {
      require(key.slot < requests.size() && key == result.keys[key.slot] && loop.matches(key),
          "R2 execution changed original active request identities");
      auto &request = requests[key.slot];
      credits[key.slot] = loop.credit(key, before[key.slot]);
      require(credits[key.slot] > 0, "R2 owner submitted a member with no output credit");
      calls.push_back(q::prepare_generation_call(model, request, kOutputs, nullptr, requests, credits[key.slot]));
    }
    const bool blocked = mode == Mode::kMemberCredit && leave_one_pending && !credit_done;
    const auto old_successors = continuation.successor_submissions;
    if (blocked) {
      require(active.size() == 2 && credits[0] == 1 && credits[1] == 2 &&
          (!relay || (continuation.ready && continuation.ready_steps >= 6)),
          "member-credit check did not reach a genuinely warmed R2 cohort with real credits1/2");
    }
    const auto refused_before = continuation.successor_refusals;
    if (mode == Mode::kColdTuple && relay && !result.cold_tuple_changed && active.size() == 2 &&
        credits[0] >= 2 && credits[1] >= 2 && continuation.matches(calls) && continuation.ready_steps >= 6 &&
        continuation.direction_warm[(continuation.next_slot + 1) % q::execution_slot_count]) {
      require(imported_captures[(continuation.next_slot + 1) % q::execution_slot_count],
          "cold R2 tuple replacement did not target an actually captured imported carrier");
      displaced_output = qwen_generation_state_case::replace_successor_output(model, continuation, 2);
      result.cold_tuple_changed = true;
    }
    const auto submitted = [&] {
      ++result.submits;
      require(model.active_step && model.active_step->submitted && model.active_step->execution.has_completion(),
          "R2 AfterSubmit did not follow an original accepted target");
      if (model.active_step->phase->imported_sources &&
          model.active_step->capture_recorded + model.active_step->capture_reused > 0)
        imported_captures[model.active_step->state_slot] = true;
      const bool successor = model.successor_step && model.successor_step->submitted;
      if (successor) {
        require(model.successor_step->members.size() == active.size(), "R2 successor silently changed cohort width");
        if (!result.first_successor_before) result.first_successor_before = before;
        if (mode == Mode::kModelEos) {
          require(model.successor_step->execution.has_completion() && model.successor_step->capture_reused > 0 &&
              model.successor_step->capture_recorded == 0,
              "R2 model EOS did not observe a real accepted B reusing its existing capture");
          result.accepted_eos_successor = true;
        }
      }
      if (model.active_step->execution.sources) sources.push_back(model.active_step->execution.sources);
      if (successor && model.successor_step->execution.sources) sources.push_back(model.successor_step->execution.sources);
      if (mode != Mode::kCancel || cancel_sent || !cancel_at || before != *cancel_at) return;
      if (relay && !successor) return;
      require(active.size() == 2 && (!relay || model.successor_step->execution.has_completion()),
          "cancellation did not observe the original accepted R2 predecessor/successor");
      const std::array heads{q::generation_boundary(*requests[0].session->head),
          q::generation_boundary(*requests[1].session->head)};
      const std::array committed{requests[0].ledger->committed_count, requests[1].ledger->committed_count};
      const auto status = loop.cancel(result.keys[0], stop_view);
      require(status.ok() || (status.IsCancelled() && !status.has_native_error()), "R2 cancellation returned a native failure");
      cancel_sent = true; result.cancelled_before = before;
      require(requests[0].session->stopping && !requests[1].session->stopping &&
          q::generation_boundary(*requests[0].session->head) == heads[0] &&
          q::generation_boundary(*requests[1].session->head) == heads[1] &&
          requests[0].ledger->committed_count == committed[0] && requests[1].ledger->committed_count == committed[1],
          "cancellation itself published a candidate or permanently stopped its peer");
    };
    q::execute_generation_calls(model, calls, false, nullptr, requests, submitted, relay ? &continuation : nullptr, observed);
    if (continuation.successor_refusals != refused_before) {
      require(mode == Mode::kColdTuple && result.cold_tuple_changed &&
          continuation.successor_refusals == refused_before + 1 && continuation.unavailable &&
          !continuation.successor_submissions && active.size() == 2,
          "natural R2 cold tuple refusal recorded or accepted B instead of retaining A");
      for (const auto &call : calls)
        require(call.disposition == q::EvaluationDisposition::kPublished &&
            call.request->progress.generated.size() == before[call.request->request_key.slot] + 1,
            "R2 cold tuple refusal failed to publish exactly the already accepted A members");
      result.cold_refusal_observed = true;
    }
    for (auto &call : calls) q::publish_generation_prompt(model, call, nullptr);
    ++result.waves;
    require(!model.active_step && !model.successor_step && !model.carrier_borrowed,
        "R2 owner wave returned before original episodes actually retired");
    if (blocked) {
      require(continuation.successor_submissions == old_successors, "one member's real credit1 failed to block B");
      credit_done = true; leave_one_pending = false; ++result.credit_blocked;
    }
    for (std::size_t member = 0; member < 2; ++member) {
      auto &request = requests[member];
      require(request.progress.generated.size() - before[member] <= credits[member] &&
          request.progress.observed_commits() == request.ledger->committed_count,
          "R2 wave exceeded per-member output credit or lost commit observation");
      if (request.session && request.session->head) {
        require(!request.session->pending && request.session->head->sampling == request.progress.initial_sampling,
            "R2 greedy wave retained pending work or changed RNG");
        if (member == 1 && (cancel_sent || eos_finished)) require(!request.session->stopping && !request.session->failed,
            "R2 whole-B retirement permanently stopped the surviving peer");
        const auto count = request.progress.generated.size();
        if (count) {
          result.states[member].insert_or_assign(count, state_bytes(model, *request.session->head));
          result.epochs[member].insert_or_assign(count, request.session->head.epoch());
        }
      }
    }
    if (relay && continuation.ready) sources.push_back(continuation.ready);
    // Ready device generations are not retained over a real transport pause.
    if (relay && std::ranges::any_of(active, [&](auto key) {
          return !loop.credit(key, requests[key.slot].progress.generated.size());
        })) continuation.release_ready();
  };
  q::SynchronousGenerationWave wave{model, execute};
  q::GenerationRequestAdapter adapter{model, static_cast<q::cache::PrefixCache *>(nullptr), std::span(requests),
      false, static_cast<std::vector<std::size_t> *>(nullptr), wave, {}, relay ? &continuation : nullptr};
  for (std::size_t member = 0; member < 2; ++member)
    requests[member].request_key = result.keys[member] = loop.enqueue(member, &deliveries[member]);
  require(result.keys[0] != result.keys[1], "R2 cohort did not receive distinct original request keys");
  for (std::size_t pumps = 0; loop.matches(result.keys[0]) || loop.matches(result.keys[1]); ++pumps) {
    require(pumps < 96, "finite R2 request journey failed to retire");
    result.cancelled += loop.pump_once(adapter).cancelled;
    // Reach the actual warm boundary first. Withholding just A's last ACK then
    // supplies real credit1 versus B's2 without altering the helper's quota.
    if (mode == Mode::kMemberCredit && !credit_done && !leave_one_pending &&
        requests[0].progress.generated.size() == 8 && requests[1].progress.generated.size() == 8) {
      if (relay) require(continuation.ready_steps >= 6 && continuation.ready,
          "credit fixture reached eight outputs before six real imported preparations");
      leave_one_pending = true;
    }
    for (std::size_t member = 0; member < 2; ++member) if (loop.matches(result.keys[member])) {
      const auto produced = requests[member].progress.generated.size();
      if (!deliveries[member].pending(requests[member].progress.generated).empty()) {
        const auto through = mode == Mode::kMemberCredit && leave_one_pending && member == 0
            ? produced - 1 : produced;
        require(loop.acknowledge(result.keys[member], through, adapter), "R2 original delivery ACK was rejected");
      }
    }
  }
  for (std::size_t member = 0; member < 2; ++member) {
    const auto &request = requests[member];
    result.ids[member] = request.progress.generated;
    result.records[member].assign(request.ledger->records.begin(),
        request.ledger->records.begin() + request.ledger->committed_count);
    require(observed_records[member] == request.ledger->committed_count && !request.session,
        "R2 actual-batch facts lost a ledger commit or terminal Session");
    if (mode == Mode::kModelEos && member == 0)
      require(result.ids[member].size() == 9 && request.progress.request_done() &&
          q::is_end_of_generation(result.ids[member].back()) && result.records[member].back().stopped,
          "R2 default model EOS did not publish its original ninth selected token");
    else if (mode != Mode::kCancel || member == 1)
      require(result.ids[member].size() == kOutputs && request.progress.request_done(),
          "R2 finite fixture ended before its fourteen real outputs");
  }
  require(result.submits == result.batches.size() && result.published_members ==
      result.records[0].size() + result.records[1].size(), "R2 confused physical batches with member participation");
  if (mode == Mode::kCancel) {
    require(cancel_sent && result.cancelled_before == cancel_at && result.cancelled == 1 && result.survivor_r1 &&
        result.ids[0].size() == (*cancel_at)[0] && result.ids[0].size() < kOutputs,
        "R2 cancellation failed to preserve its committed prefix and run the surviving R1 peer");
    if (relay) require(result.discarded_batches == 1 && continuation.successor_discards == 1,
        "R2 accepted B was not wholly discarded after real retirement");
  } else require(!result.cancelled, "non-cancellation R2 arm unexpectedly cancelled a key");
  if (mode == Mode::kMemberCredit) require(result.credit_blocked == 1, "R2 member-credit boundary was not checked");
  if (mode == Mode::kModelEos)
    require(eos_finished && result.survivor_r1 && (!relay || (result.accepted_eos_successor &&
        result.discarded_batches == 1 && continuation.successor_submissions == 1 &&
        continuation.successor_publications == 0 && continuation.successor_discards == 1)),
        "R2 EOS failed to retire accepted B as a whole and continue the surviving R1 peer");
  if (relay && mode == Mode::kColdTuple) {
    require(result.cold_tuple_changed && result.cold_refusal_observed && continuation.unavailable &&
        continuation.successor_refusals == 1 && !continuation.successor_submissions && !continuation.successor_discards,
        "R2 cold tuple was silently recorded, downgraded, retried or accepted");
  } else if (relay) {
    require(!continuation.unavailable && continuation.successor_refusals == 0 &&
        continuation.successor_publications + continuation.successor_discards == continuation.successor_submissions,
        "R2 relay silently fell back or lost an accepted successor");
    if (mode == Mode::kNormal) require(continuation.successor_submissions && result.first_successor_before,
        "R2 fourteen-output journey never accepted its warmed successor");
  }
  result.completed_steps = continuation.completed_steps; result.successors = continuation.successor_submissions;
  result.publications = continuation.successor_publications; result.discards = continuation.successor_discards;
  result.refusals = continuation.successor_refusals;
  continuation.reset();
  for (const auto &source : sources) require(source.expired(), "R2 terminal cleanup retained an original Source generation");
  require(!model.snapshot_versions.live && !model.active_step && !model.successor_step && !model.carrier_borrowed &&
      !loop.waiting_front() && loop.active().empty(), "R2 RequestLoop retained a key, candidate or original model version");
  cleanup.dismiss();
  return result;
}

int run(const std::filesystem::path &input, bool boundaries = false) {
  auto source = q::prepare_weight_source(input / "weights.tsv", false);
  const auto geometry = source.roles.geometry;
  const auto context = static_cast<std::size_t>(geometry.context_limit);
  const auto prompt = q::read_ids(input / "prompt.ids", geometry.vocabulary, context);
  const auto needed = q::cache::add(prompt.size(), kOutputs - 1);
  const auto capacity = (std::min)(context, (std::max)(std::size_t{32}, needed));
  require(prompt.size() > 1 && capacity >= needed && geometry.recurrent_layers() && geometry.attention_layers(),
      "R2 relay requires its actual prompt and complete recurrent/KV fixture");
  std::array prompts{prompt, prompt};
  const auto different = std::find_if(prompts[1].rbegin() + 1, prompts[1].rend(),
      [&](auto token) { return token != prompts[1].back(); });
  require(different != prompts[1].rend(), "R2 fixture needs two distinct original prompt tokens");
  std::iter_swap(different, prompts[1].rbegin()); // Same actual ID set/width; distinct Session payload.
  // Two actual resident heads plus two simultaneously retained R2 candidates:
  // N=2,R=2,D=2 => snapshot/state slots6; no immutable device checkpoint is held.
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, capacity, 2, prompt.size(),
      q::recurrent_bytes(q::state_schema_for(geometry, capacity), geometry), 2, 1, false, false, 2);
  require(plan.snapshot_slots == 6 && plan.state_slots == 6, "R2 relay changed the actual N+D*R capacity");
  q::Model model(std::move(source), geometry, capacity, plan, {}, false,
      q::ReplayPolicy::kPreferCapture, SIZE_MAX, {}, 0, {.primary = i::BackendKind::kCuda});
  const std::array shapes{prompt.size(), std::size_t{1}};
  for (const auto width : {std::size_t{1}, std::size_t{2}}) {
    model.prepare_family(shapes, false, width);
    model.plan_replay_uses(prompt.size(), width, 8); model.plan_replay_uses(1, width, kOutputs * 8);
  }
  model.freeze_shape_plan();
  for (const auto width : {std::size_t{1}, std::size_t{2}}) {
    model.prepare_execution(shapes, width); model.prepare_relay_execution(width);
  }
  if (boundaries) {
    const auto cold_serial = run_arm(model, prompts, Mode::kColdTuple, false);
    const auto cold = run_arm(model, prompts, Mode::kColdTuple, true);
    compare(cold_serial, cold);
    const auto eos_serial = run_arm(model, prompts, Mode::kModelEos, false);
    const auto eos = run_arm(model, prompts, Mode::kModelEos, true);
    compare(eos_serial, eos);
    std::cout << "{\"status\":\"PASS\",\"backend\":\"cuda\",\"requests\":2,"
        "\"serial_and_relay_arms\":4,\"fixture\":\"untrained_one_hot_GA_wide_vocab\","
        "\"model_eos_run\":true,\"extra_stop_ids\":false,\"eos_outputs\":" << eos.ids[0].size()
        << ",\"peer_outputs\":" << eos.ids[1].size() << ",\"survivor_r1\":" << eos.survivor_r1
        << ",\"eos_successor_submissions\":" << eos.successors
        << ",\"eos_successor_publications\":" << eos.publications
        << ",\"eos_successor_discards\":" << eos.discards
        << ",\"cold_ignores_model_eos\":true,\"natural_cold_tuple_run\":true,\"cold_successor_refusals\":" << cold.refusals
        << ",\"cold_successor_submissions\":" << cold.successors
        << ",\"complete_state_ledger_rng_equal\":true,\"actual_batch_keys_checked\":true,"
        "\"final_live_versions\":0,\"model_quality_claim\":false,\"performance_claim\":false}\n";
    return 0;
  }
  const auto normal_serial = run_arm(model, prompts, Mode::kNormal, false);
  const auto normal = run_arm(model, prompts, Mode::kNormal, true);
  compare(normal_serial, normal);
  const auto credit_serial = run_arm(model, prompts, Mode::kMemberCredit, false);
  const auto credit = run_arm(model, prompts, Mode::kMemberCredit, true);
  compare(credit_serial, credit);
  const auto cancel_serial = run_arm(model, prompts, Mode::kCancel, false, normal.first_successor_before);
  const auto cancelled = run_arm(model, prompts, Mode::kCancel, true, normal.first_successor_before);
  compare(cancel_serial, cancelled);
  std::cout << "{\"status\":\"PASS\",\"backend\":\"cuda\",\"sampling\":\"greedy\",\"requests\":2,"
      "\"prompt_tokens\":" << prompt.size() << ",\"capacity\":" << capacity
      << ",\"resident_sessions\":2,\"active_sequences\":2,\"in_flight_batches\":2,\"state_slots\":6,"
      "\"serial_and_relay_arms\":6,\"outputs_per_live_member\":14,"
      "\"distinct_prompt_payloads\":true,\"complete_initialized_state_equal\":true,\"ledger_and_rng_equal\":true,"
      "\"actual_batch_keys_checked\":true,\"real_member_credit1_blocks_b\":true,"
      "\"accepted_b_whole_discard\":true,\"survivor_r1_continues\":true,\"peer_not_permanently_stopped\":true,"
      "\"model_eos_run\":false,\"native_fault_injection\":false,\"performance_claim\":false,\"final_versions\":0,\"relay\":[";
  const std::array results{&normal, &credit, &cancelled};
  for (std::size_t arm = 0; arm < results.size(); ++arm) {
    const auto &value = *results[arm]; if (arm) std::cout << ',';
    std::cout << "{\"waves\":" << value.waves << ",\"actual_batches\":" << value.batches.size()
        << ",\"published_members\":" << value.published_members << ",\"completed_steps\":" << value.completed_steps
        << ",\"successor_submissions\":" << value.successors << ",\"successor_publications\":" << value.publications
        << ",\"successor_discards\":" << value.discards << ",\"successor_refusals\":" << value.refusals
        << ",\"survivor_r1\":" << value.survivor_r1 << ",\"batch_facts\":[";
    for (std::size_t batch = 0; batch < value.batches.size(); ++batch) {
      const auto &facts = value.batches[batch]; if (batch) std::cout << ',';
      std::cout << "{\"rows\":" << facts.rows << ",\"generation\":" << (facts.generation ? "true" : "false")
          << ",\"members\":[";
      for (std::size_t member = 0; member < facts.keys.size(); ++member) {
        if (member) std::cout << ',';
        std::cout << "{\"slot\":" << facts.keys[member].slot << ",\"key_generation\":" << facts.keys[member].generation
            << ",\"disposition\":" << static_cast<int>(facts.dispositions[member]) << '}';
      }
      std::cout << "]}";
    }
    std::cout << "]}";
  }
  std::cout << "]}\n";
  return 0;
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 1) { std::cerr << "SKIP: pass --cuda PREPARED_LOCAL_INPUTS\n"; return 77; }
    if (argc == 2 && std::string_view(argv[1]) == "--cuda-continuous") {
      std::cerr << "SKIP: pass --cuda-continuous CONTEXT32_TINY_INPUT\n"; return 77;
    }
    const bool continuous = argc == 3 && std::string_view(argv[1]) == "--cuda-continuous";
    const bool boundaries = argc == 4 && std::string_view(argv[1]) == "--cuda-boundaries";
    if (!continuous && !boundaries && (argc != 3 || std::string_view(argv[1]) != "--cuda"))
      throw std::runtime_error("usage: qwen_generation_batch_relay_probe --cuda INPUT | --cuda-continuous CONTEXT32_TINY_INPUT | --cuda-boundaries TINY_INPUT NEW_OUTPUT_DIR");
    const auto compiled = q::Backend::compiled_backends();
    if (std::find(compiled.begin(), compiled.end(), q::BackendKind::kCuda) == compiled.end()) {
      std::cerr << "SKIP: CUDA route is not built\n"; return 77;
    }
    return continuous ? qwen_continuous_generation_case::run(argv[2])
        : boundaries ? run(qwen_generation_state_case::model_eos_fixture(argv[2], argv[3], 9), true)
        : run(argv[2]);
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
