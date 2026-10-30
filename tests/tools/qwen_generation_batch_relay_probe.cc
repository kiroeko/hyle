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

void check_captured_sources(const q::Model &model, const q::StepState &step) {
  if (!step.execution.sources) return; // The existing legal non-Source route.
  const auto &sources = *step.execution.sources;
  require(step.submitted && !step.execution.completion.valid() && step.execution.has_completion() &&
      sources.submission.valid() && sources.primary_output.valid() && step.phase &&
      sources.states.size() == step.phase->state_sources.size() &&
      sources.pages.size() == model.draft_bank,
      "Source cohort lost its sole original completion, token or prepared bank");
  for (const auto &state : sources.states)
    require(state.valid() && state.destination_count() == step.members.size(),
        "Source cohort did not capture every original Store destination");
  for (const auto &pages : sources.pages)
    require(pages.valid(), "Source cohort did not capture every original KV append");
}

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
  std::array<bool, q::execution_slot_count> imported_r2_slots{};
  std::size_t maximum_ready_steps = 0, survivor_imported_batches = 0,
      survivor_first_imported_uses = 0, survivor_first_imported_captures = 0;
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
    Mode mode, bool relay, std::optional<std::array<std::size_t, 2>> cancel_at = {},
    q::GenerationContinuation *borrowed_continuation = nullptr) {
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
  q::GenerationContinuation local_continuation;
  auto &continuation = borrowed_continuation ? *borrowed_continuation : local_continuation;
  if (!borrowed_continuation) continuation.prepare(2);
  require(!continuation.ready && continuation.pending_calls.empty(),
      "R2 arm borrowed a continuation still retaining an original Source or pending call");
  const auto completed_before = continuation.completed_steps;
  const auto successors_before = continuation.successor_submissions;
  const auto publications_before = continuation.successor_publications;
  const auto discards_before = continuation.successor_discards;
  const auto refusals_before = continuation.successor_refusals;
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
      check_captured_sources(model, *model.active_step);
      if (model.active_step->phase->imported_sources &&
          model.active_step->capture_recorded + model.active_step->capture_reused > 0) {
        imported_captures[model.active_step->state_slot] = true;
        if (active.size() == 2) result.imported_r2_slots[model.active_step->state_slot] = true;
        else if (mode == Mode::kCancel && cancel_sent) {
          if (!result.survivor_imported_batches) {
            result.survivor_first_imported_uses =
                model.active_step->phase->slots[model.active_step->state_slot].preparation_uses;
            result.survivor_first_imported_captures = model.active_step->capture_recorded;
            if (borrowed_continuation)
              std::cerr << "warm_cohort first_survivor_imported slot=" << model.active_step->state_slot
                  << " preparation_uses=" << result.survivor_first_imported_uses
                  << " recorded=" << result.survivor_first_imported_captures << '\n';
          }
          ++result.survivor_imported_batches;
        }
      }
      const bool successor = model.successor_step && model.successor_step->submitted;
      if (successor) {
        check_captured_sources(model, *model.successor_step);
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
    result.maximum_ready_steps = (std::max)(result.maximum_ready_steps, continuation.ready_steps);
    if (continuation.successor_refusals != refused_before) {
      require(mode == Mode::kColdTuple && result.cold_tuple_changed &&
          continuation.successor_refusals == refused_before + 1 && !continuation.unavailable &&
          !continuation.successor_submissions && active.size() == 2,
          "natural R2 cold tuple refusal recorded or accepted B instead of retaining A");
      require(!model.active_step && !model.successor_step && !model.carrier_borrowed &&
          !continuation.ready && continuation.pending_calls.empty() && !continuation.ready_steps,
          "cold tuple refusal must retire A and release reuse hints before re-priming");
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
    if (relay) require(result.discarded_batches == 1 && continuation.successor_discards - discards_before == 1,
        "R2 accepted B was not wholly discarded after real retirement");
  } else require(!result.cancelled, "non-cancellation R2 arm unexpectedly cancelled a key");
  if (mode == Mode::kMemberCredit) require(result.credit_blocked == 1, "R2 member-credit boundary was not checked");
  if (mode == Mode::kModelEos)
    require(eos_finished && result.survivor_r1 && (!relay || (result.accepted_eos_successor &&
        result.discarded_batches == 1 && continuation.successor_submissions == 1 &&
        continuation.successor_publications == 0 && continuation.successor_discards == 1)),
        "R2 EOS failed to retire accepted B as a whole and continue the surviving R1 peer");
  if (relay && mode == Mode::kColdTuple) {
    require(result.cold_tuple_changed && result.cold_refusal_observed && !continuation.unavailable &&
        continuation.successor_refusals == 1 && !continuation.successor_submissions && !continuation.successor_discards,
        "R2 cold tuple lost its refusal, disabled later reuse or accepted the refused B");
  } else if (relay) {
    require(!continuation.unavailable && continuation.successor_refusals == 0 &&
        continuation.successor_publications + continuation.successor_discards == continuation.successor_submissions,
        "R2 relay silently fell back or lost an accepted successor");
    if (mode == Mode::kNormal) require(continuation.successor_submissions && result.first_successor_before,
        "R2 fourteen-output journey never accepted its warmed successor");
  }
  result.completed_steps = continuation.completed_steps - completed_before;
  result.successors = continuation.successor_submissions - successors_before;
  result.publications = continuation.successor_publications - publications_before;
  result.discards = continuation.successor_discards - discards_before;
  result.refusals = continuation.successor_refusals - refusals_before;
  continuation.reset();
  for (const auto &source : sources) require(source.expired(), "R2 terminal cleanup retained an original Source generation");
  require(!model.snapshot_versions.live && !model.active_step && !model.successor_step && !model.carrier_borrowed &&
      !loop.waiting_front() && loop.active().empty(), "R2 RequestLoop retained a key, candidate or original model version");
  cleanup.dismiss();
  return result;
}

struct SourceFailureArm final {
  std::array<std::int32_t, 2> ids{};
  std::array<StateBytes, 2> states;
  std::vector<q::GenerationCommitRecord> records;
  std::size_t submissions = 0, captured_stores = 0, prepared_pages = 0;
};

SourceFailureArm source_failure_arm(q::Model &model, bool fail_capture) {
  require(!model.active_step && !model.successor_step && !model.carrier_borrowed && !model.snapshot_versions.live,
      "Source failure arm retained a preceding episode or published version");
  static constexpr std::array<std::int32_t, 3> prompt{1, 2, 3};
  const std::array chunks{prompt.size(), std::size_t{1}};
  auto session = model.create_session(chunks);
  q::GenerationLedger ledger(2, 1);
  std::shared_ptr<q::StepSources> sources;
  hyle::util::ScopeGuard cleanup{[&]() noexcept {
    if (sources && sources->submission.valid()) sources->submission.wait().IgnoreError();
    sources.reset(); (void)model.abort_episode();
    if (session) {
      session->stop().IgnoreError(); (void)session->drain(); (void)session->try_retire(); session.reset();
    }
  }};
  SourceFailureArm result;
  const auto finish = [&](const char *name) {
    return q::finish_evaluation(model, *session, name, {}, false,
        nullptr, nullptr, nullptr, nullptr, {}, &ledger);
  };
  q::prepare_generation_record(ledger, *session, q::GraphRole::kText, prompt, false, 1, 0);
  model.prepare_evaluation(*session, prompt, nullptr);
  model.submit_evaluations(); ++result.submissions;
  result.ids[0] = finish("source-failure-prefill");
  result.states[0] = state_bytes(model, *session->head);
  auto checkpoint = session->checkpoint();
  const auto *head = session->head.get();
  const auto epoch = session->head.epoch();
  const auto boundary = q::generation_boundary(*session->head);
  const auto free_pages = model.page_inventory.free_ids.size();
  const auto page_owners = model.page_inventory.owners;
  const auto live_versions = model.snapshot_versions.live;
  const auto pending = session->head->next_id;
  const auto unchanged_head = [&] {
    require(session->head.get() == head && session->head.epoch() == epoch &&
        q::generation_boundary(*session->head) == boundary &&
        state_bytes(model, *session->head) == result.states[0] &&
        state_bytes(model, *checkpoint) == result.states[0],
        "Source capture failure changed a committed complete head, checkpoint, epoch or RNG");
  };
  if (fail_capture) {
    q::prepare_generation_record(ledger, *session, q::GraphRole::kText, std::span(&pending, 1), true, 1, 0);
    const auto before_records = ledger.records;
    const auto before_commits = ledger.committed_count;
    const auto unchanged_ledger = [&] {
      require(ledger.committed_count == before_commits && ledger.records.size() == before_records.size(),
          "Source capture failure published or resized the original ledger");
      for (std::size_t index = 0; index < before_records.size(); ++index) {
        const auto &before = before_records[index]; const auto &after = ledger.records[index];
        require(after.same_result(before) && after.committed == before.committed &&
            after.user_output == before.user_output && after.imported == before.imported && after.turn == before.turn &&
            after.input_owner == before.input_owner && std::ranges::equal(after.ids(), before.ids()) &&
            after.step_seconds == before.step_seconds && after.endpoint == before.endpoint,
            "Source capture failure changed an original committed or prepared ledger record");
      }
    };
    model.prepare_evaluation(*session, std::span(&pending, 1), nullptr);
    auto *const failed_step = model.active_step.get();
    require(failed_step && failed_step->prepared && !failed_step->submitted &&
        failed_step->execution.state_transactions && failed_step->execution.state_transactions->bound(),
        "Source failure did not use the original prepared Store/Append episode");
    auto *const transactions = failed_step->execution.state_transactions;
    sources = model.prepare_submission_sources(*failed_step);
    const auto stores = sources->states.size(); const auto pages = sources->pages.size();
    require(stores == model.geometry.recurrent_layers() * 2 && stores &&
        pages == model.geometry.attention_layers() * 2 && pages,
        "Source failure omitted an actual recurrent or KV bank");
    for (const auto &state : sources->states) require(state.prepared() && !state.valid(), "Store was not cold prepared");
    for (const auto &page : sources->pages) require(page.prepared() && !page.valid(), "KV Source was not cold prepared");
    // Deliberate invalid metadata at the end of the real Store cohort. The
    // accepted target, original Completion and preceding captures are real.
    // This is neither a native device fault nor a synthetic Completion.
    sources->states.emplace_back();
    bool rejected = false;
    try { model.submit_evaluations(*failed_step, true); }
    catch (const i::EvaluationFailure &failure) {
      require(failure.status().IsFailedPrecondition() && !failure.status().has_native_error() &&
          std::string_view(failure.what()).find("Store source requires cold preparation") != std::string_view::npos,
          "Source failure did not come from the appended unprepared Store metadata");
      rejected = true;
    }
    require(rejected && model.active_step.get() == failed_step && failed_step->submitted &&
        failed_step->execution.sources == sources && !failed_step->execution.completion.valid() &&
        failed_step->execution.has_completion() && sources->submission.valid() && sources->primary_output.valid(),
        "Optional Source export downgraded or lost an already accepted partial capture");
    ++result.submissions;
    for (std::size_t index = 0; index < stores; ++index)
      require(sources->states[index].valid() && sources->states[index].destination_count() == 1,
          "Source failure occurred before capturing the original valid Store prefix");
    require(!sources->states.back().prepared() && !sources->states.back().valid(),
        "Source failure silently prepared or captured the invalid trailing Store");
    for (const auto &page : sources->pages)
      require(page.prepared() && !page.valid(), "late Store rejection unexpectedly captured a later KV Source");
    result.captured_stores = stores; result.prepared_pages = pages;
    unchanged_head(); unchanged_ledger();
    q::check(sources->submission.wait()); // Isolate the external-owner retirement gate from native progress.
    const auto weak_sources = std::weak_ptr(sources);
    const auto weak_step = std::weak_ptr(model.active_step);
    const auto blocked = model.abort_episode(); q::check(blocked.status);
    require(!blocked.retired && model.active_step.get() == failed_step && session->pending == failed_step &&
        failed_step->execution.sources == sources && transactions->bound() && model.carrier_borrowed == 1 &&
        model.snapshot_versions.live > live_versions,
        "External Source ownership did not retain the original candidate and transactions after actual native retirement");
    unchanged_head(); unchanged_ledger();
    sources.reset();
    const auto retired = model.abort_episode(); q::check(retired.status);
    require(retired.retired && weak_sources.expired() && weak_step.expired() &&
        !model.active_step && !model.successor_step && !model.carrier_borrowed && !session->pending &&
        !transactions->bound() && model.snapshot_versions.live == live_versions &&
        model.page_inventory.free_ids.size() == free_pages && model.page_inventory.owners == page_owners &&
        !session->failed && !session->stopping && session->retirement_status.ok(),
        "Dropping the final external Source did not recover the original Store/Append and candidate/page resources");
    unchanged_head(); unchanged_ledger();
  }
  // Same original Session and committed pending token after the failed target;
  // the ordinary owner reuses its still-uncommitted ledger slot for retry.
  q::prepare_generation_record(ledger, *session, q::GraphRole::kText, std::span(&pending, 1), true, 1, 0);
  model.prepare_evaluation(*session, std::span(&pending, 1), nullptr);
  model.submit_evaluations(); ++result.submissions;
  result.ids[1] = finish("source-failure-retry");
  result.states[1] = state_bytes(model, *session->head);
  require(ledger.committed_count == 2 && session->head.epoch() == epoch + 1 &&
      state_bytes(model, *checkpoint) == result.states[0], "Source retry changed its old checkpoint or omitted the one real commit");
  result.records.assign(ledger.records.begin(), ledger.records.begin() + ledger.committed_count);
  checkpoint.reset();
  q::check(session->stop());
  const auto drained = session->drain(); q::check(drained.status);
  const auto retired = session->try_retire(); q::check(retired.status);
  require(retired.retired, "Source failure arm did not actually retire its original Session");
  session.reset();
  require(!model.active_step && !model.successor_step && !model.carrier_borrowed && !model.snapshot_versions.live &&
      model.page_inventory.free_ids.size() == model.page_inventory.owners.size() &&
      std::ranges::all_of(model.page_inventory.owners, [](auto owners) { return owners == 0; }),
      "Source failure arm retained a published version, candidate or original KV page");
  cleanup.dismiss();
  return result;
}

int run_source_failure(const std::filesystem::path &input) {
  auto source = q::prepare_weight_source(input / "weights.tsv", false);
  const auto geometry = source.roles.geometry;
  require(geometry.context_limit >= 16 && geometry.vocabulary > 3 &&
      geometry.recurrent_layers() && geometry.attention_layers(), "Source failure requires the complete cached C16 GA fixture");
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 16, 3, 3,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 1, 1, false, false);
  q::Model model(std::move(source), geometry, 16, plan, {}, false, q::ReplayPolicy::kPreferCapture,
      SIZE_MAX, {}, 0, {.primary = i::BackendKind::kCuda});
  const std::array shapes{std::size_t{3}, std::size_t{1}};
  model.prepare_family(shapes); model.plan_replay_uses(3, 1, 2); model.plan_replay_uses(1, 1, 3);
  model.freeze_shape_plan(); model.prepare_execution(shapes);
  const auto control = source_failure_arm(model, false);
  const auto recovered = source_failure_arm(model, true);
  require(control.ids == recovered.ids && control.states == recovered.states &&
      control.records.size() == recovered.records.size(), "Source capture failure/retry changed complete serial results");
  for (std::size_t index = 0; index < control.records.size(); ++index) {
    const auto &before = control.records[index]; const auto &after = recovered.records[index];
    require(before.same_result(after) && before.committed == after.committed &&
        before.user_output == after.user_output && before.imported == after.imported && before.turn == after.turn &&
        std::ranges::equal(before.ids(), after.ids()), "Source capture failure/retry changed a complete non-time serial ledger");
  }
  std::size_t compared_bytes = 0;
  for (const auto &state : control.states) for (const auto &payload : state.payloads) compared_bytes += payload.size();
  std::cout << "{\"status\":\"PASS\",\"backend\":\"cuda\",\"case\":\"source-post-submit-capture-failure\","
      "\"native_submissions\":" << control.submissions + recovered.submissions
      << ",\"control_submissions\":" << control.submissions << ",\"failure_and_retry_submissions\":" << recovered.submissions
      << ",\"captured_original_stores\":" << recovered.captured_stores << ",\"prepared_uncaptured_pages\":" << recovered.prepared_pages
      << ",\"complete_state_bytes_compared\":" << compared_bytes
      << ",\"unprepared_trailing_store_rejected\":true,\"optional_export_did_not_downgrade\":true,"
      "\"original_completion_transferred\":true,\"external_source_owner_blocks_recovery\":true,"
      "\"source_release_recovers_store_append\":true,\"committed_head_checkpoint_epoch_rng_ledger_unchanged\":true,"
      "\"same_session_retry_matches_serial\":true,\"all_source_session_page_owners_retired\":true,"
      "\"final_live_versions\":0,\"metadata_capture_failure\":true,\"native_fault_injection\":false,"
      "\"model_quality_claim\":false,\"performance_claim\":false}\n";
  return 0;
}

int run(const std::filesystem::path &input, bool boundaries = false, bool warm_cohort = false) {
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
  if (warm_cohort) {
    q::GenerationContinuation continuation; continuation.prepare(2);
    const auto imported_uses = [&](std::size_t width) {
      const auto found = model.phases.find({width, 1, q::GraphRole::kText,
          q::TextCompletion::kEmitNext, 0, true});
      require(found != model.phases.end() && found->second && found->second->imported_sources &&
          found->second->slots.size() == q::execution_slot_count,
          "warm-cohort case lost its original width-specific imported phase");
      std::array<std::size_t, q::execution_slot_count> uses{};
      for (std::size_t slot = 0; slot < uses.size(); ++slot)
        uses[slot] = found->second->slots[slot].preparation_uses;
      return uses;
    };
    const auto fresh_r1 = imported_uses(1);
    require(std::ranges::all_of(fresh_r1, [](auto uses) { return uses == 0; }),
        "warm-cohort case started with an already prepared R1 imported slot");
    const auto warm = run_arm(model, prompts, Mode::kNormal, true, {}, &continuation);
    require(warm.first_successor_before && warm.successors && warm.maximum_ready_steps >= 6 &&
        std::ranges::all_of(warm.imported_r2_slots, [](auto captured) { return captured; }) &&
        std::ranges::all_of(imported_uses(2), [](auto uses) { return uses != 0; }),
        "warm-cohort case did not really prepare all three R2 imported slots and accept a successor");
    const auto cancel_at = warm.first_successor_before;
    require((*cancel_at)[1] + 4 <= kOutputs,
        "warm-cohort cancellation leaves too few outputs to enter the fresh R1 imported phase");
    const auto control = run_arm(model, prompts, Mode::kCancel, false, cancel_at);
    require(imported_uses(1) == fresh_r1,
        "serial cancellation oracle warmed the R1 imported phase before the candidate");
    std::cerr << "warm_cohort r2_slots_prepared=3 cancel_at=" << (*cancel_at)[0] << ',' << (*cancel_at)[1]
        << " r1_imported_prior_uses=0,0,0 ready_steps=" << continuation.ready_steps << '\n';
    const auto candidate = run_arm(model, prompts, Mode::kCancel, true, cancel_at, &continuation);
    compare(control, candidate);
    require(candidate.survivor_imported_batches && candidate.survivor_first_imported_uses == 1 &&
        candidate.survivor_first_imported_captures && candidate.discards == 1 && !candidate.refusals,
        "warm-cohort cancellation did not capture a genuinely fresh R1 slot and retire the original R2 successor");
    std::cout << "{\"status\":\"PASS\",\"backend\":\"cuda\",\"case\":\"warm-r2-to-fresh-r1-imported\","
        "\"serial_and_relay_arms\":3,\"outputs_per_live_member\":" << kOutputs
        << ",\"actual_r2_imported_slots\":3,\"warmup_successors\":" << warm.successors
        << ",\"cancel_before\":[" << (*cancel_at)[0] << ',' << (*cancel_at)[1]
        << "],\"fresh_r1_imported_prior_uses\":[0,0,0],\"survivor_imported_batches\":" << candidate.survivor_imported_batches
        << ",\"first_survivor_slot_uses\":" << candidate.survivor_first_imported_uses
        << ",\"first_survivor_capture_recorded\":" << candidate.survivor_first_imported_captures
        << ",\"successor_submissions\":" << candidate.successors
        << ",\"successor_publications\":" << candidate.publications
        << ",\"successor_discards\":" << candidate.discards
        << ",\"successor_refusals\":" << candidate.refusals
        << ",\"complete_state_epoch_rng_ledger_equal\":true,\"all_source_session_owners_retired\":true,"
        "\"final_live_versions\":0,\"native_fault_injection\":false,\"performance_claim\":false}\n";
    return 0;
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
    const bool source_failure = argc == 3 && std::string_view(argv[1]) == "--cuda-source-failure";
    const bool warm_cohort = argc == 3 && std::string_view(argv[1]) == "--warm-cohort";
    const bool boundaries = argc == 4 && std::string_view(argv[1]) == "--cuda-boundaries";
    if (!continuous && !source_failure && !boundaries && !warm_cohort && (argc != 3 || std::string_view(argv[1]) != "--cuda"))
      throw std::runtime_error("usage: qwen_generation_batch_relay_probe --cuda INPUT | --cuda-continuous CONTEXT32_TINY_INPUT | --cuda-source-failure INPUT | --cuda-boundaries TINY_INPUT NEW_OUTPUT_DIR | --warm-cohort INPUT");
    const auto compiled = q::Backend::compiled_backends();
    if (std::find(compiled.begin(), compiled.end(), q::BackendKind::kCuda) == compiled.end()) {
      std::cerr << "SKIP: CUDA route is not built\n"; return 77;
    }
    return source_failure ? run_source_failure(argv[2]) : continuous ? qwen_continuous_generation_case::run(argv[2])
        : warm_cohort ? run(argv[2], false, true)
        : boundaries ? run(qwen_generation_state_case::model_eos_fixture(argv[2], argv[3], 9), true)
        : run(argv[2]);
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
