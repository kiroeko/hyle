#pragma once

// Finite original Model regression, included after Model and the state oracle.
// Same-build PlanState access below observes dormant records only after real
// retirement; it creates no Source, permission, observer or execution owner.
namespace qwen_short_turn_capture_case {
namespace q = qwen35;
namespace i = hyle::inference;
namespace gd = hyle::graph::detail;
using qwen_generation_state_case::require;
using qwen_generation_state_case::state_bytes;
using qwen_generation_state_case::StateBytes;
struct RefusalRecord final {
  std::size_t predecessor = 0, target = 0, uses = 0, keys_before = 0, keys_after = 0;
  bool reprimed = false, terminal_release = false;
};
struct SlotFacts final {
  std::uintptr_t invocation = 0;
  std::size_t uses = 0, keyed_variants = 0;
};
struct Result final {
  std::vector<std::int32_t> ids;
  std::vector<q::GenerationCommitRecord> commits;
  std::map<std::size_t, StateBytes> states;
  std::map<std::size_t, std::uint64_t> epochs;
  std::array<std::array<SlotFacts, q::execution_slot_count>, 3> turn_slots{};
  std::array<std::uintptr_t, 3> sessions{};
  std::size_t successors = 0, refusals = 0;
  bool forced_output_owner = false, retained_invocation = false, ordinary_reprime = false;
  std::size_t successors_after_reprime = 0, completed_reprimes = 0;
  std::vector<RefusalRecord> refusal_records;
};
inline auto slot_facts(q::Model &model) {
  std::array<SlotFacts, q::execution_slot_count> result{};
  const auto phase = model.phases.find({1, 1, q::GraphRole::kText,
      q::TextCompletion::kEmitNext, 0, true});
  require(phase != model.phases.end() && phase->second && phase->second->imported_sources,
      "short-turn regression has no original imported phase");
  for (std::size_t slot = 0; slot < result.size(); ++slot) {
    const auto &saved = phase->second->slots[slot];
    result[slot].uses = saved.preparation_uses;
    const auto original = phase->second->slots[slot].carrier.borrow();
    if (!original) continue;
    result[slot].invocation = reinterpret_cast<std::uintptr_t>((*original)->object_identity());
    const auto *state = gd::InvocationAccess::state(**original);
    if (!state) continue;
    for (const auto &range : state->capture_rebuilds)
      for (const auto &variant : range.variants)
        result[slot].keyed_variants += variant.key_valid;
  }
  return result;
}
struct SlotSnapshot final {
  SlotFacts slot;
  hyle::compute::TensorView fixed_output;
  std::array<std::uint8_t, 8> key_masks{};
  std::size_t ranges = 0;
};
inline SlotSnapshot snapshot_slot(q::Model &model, std::size_t slot) {
  auto &phase = *model.phases.at({1, 1, q::GraphRole::kText, q::TextCompletion::kEmitNext, 0, true});
  auto &run = *q::take(phase.slots[slot].carrier.borrow());
  const auto *state = gd::InvocationAccess::state(run);
  SlotSnapshot result; result.slot = slot_facts(model)[slot];
  require(state && state->capture_rebuilds.size() <= result.key_masks.size(),
      "finite capture observation exceeds its cold bound");
  result.ranges = state->capture_rebuilds.size();
  for (std::size_t index = 0; index < result.ranges; ++index) {
    const auto &range = state->capture_rebuilds[index];
    result.key_masks[index] = static_cast<std::uint8_t>(range.variants[0].key_valid |
        (range.variants[1].key_valid << 1));
  }
  require(phase.id_port.ordinal() < state->plan->external_instances.size(), "original fixed ID port missing");
  const auto physical = state->plan->external_instances[phase.id_port.ordinal()];
  require(physical < state->writable.size() && state->writable[physical].valid(),
      "original fixed output storage is not bound");
  // Passive descriptor only, not an extra writer/reader/permission owner.
  result.fixed_output = state->writable[physical]; return result;
}
inline bool same_fixed_output(hyle::compute::TensorView left, hyle::compute::TensorView right) {
  const auto a = left.storage(), b = right.storage();
  return left.valid() && right.valid() && left.description() == right.description() &&
      a.resource() == b.resource() && a.native_handle() == b.native_handle() &&
      a.offset() == b.offset() && a.size() == b.size();
}
inline SlotSnapshot verify_unsubmitted_refusal(q::Model &model, std::size_t target, const SlotSnapshot &before) {
  const auto after = snapshot_slot(model, target);
  const auto &carrier = model.phases.at({1, 1, q::GraphRole::kText,
      q::TextCompletion::kEmitNext, 0, true})->slots[target].carrier;
  require(carrier.initialized() && !carrier.closing() && carrier.retired() &&
      after.slot.invocation == before.slot.invocation && after.slot.uses == before.slot.uses &&
      after.slot.keyed_variants <= before.slot.keyed_variants &&
      same_fixed_output(before.fixed_output, after.fixed_output),
      "typed unsubmitted refusal rebuilt its original Invocation/fixedIO or recorded/submitted B");
  auto &run = *q::take(model.phases.at({1, 1, q::GraphRole::kText,
      q::TextCompletion::kEmitNext, 0, true})->slots[target].carrier.borrow());
  const auto *records = gd::InvocationAccess::state(run);
  require(after.ranges == before.ranges, "unsubmitted refusal changed the immutable range inventory");
  for (std::size_t index = 0; index < after.ranges; ++index) {
    const auto &range = records->capture_rebuilds[index];
    require((after.key_masks[index] & static_cast<std::uint8_t>(~before.key_masks[index])) == 0,
        "unsubmitted refusal created a recorded variant");
    for (const auto &variant : range.variants)
      require(!variant.commands.valid() || (variant.key_valid && range.qualification_valid &&
          range.preserves_private_arguments), "unsubmitted refusal retained unqualified dormant commands");
    for (const auto &active : records->captured_ranges) {
      if (active.step_begin != range.step_begin || active.step_count != range.step_count) continue;
      require(!active.commands.valid() || (range.qualification_valid && range.preserves_private_arguments &&
          range.active_variant < 2 && range.variants[range.active_variant].key_valid),
          "unsubmitted refusal left an unqualified active recording");
    }
  }
  return after;
}
inline Result arm(const std::filesystem::path &input, bool relay, bool force_output_tuple = false,
    std::size_t capacity = 32, std::size_t last_turn_outputs = 14, bool require_recovery_reuse = false) {
  const auto output_total = last_turn_outputs + 6;
  auto source = q::prepare_weight_source(input / "weights.tsv", false);
  const auto geometry = source.roles.geometry;
  const auto prompt = q::read_ids(input / "prompt.ids", geometry.vocabulary, 32);
  require(prompt.size() == 3 && geometry.context_limit >= static_cast<std::int64_t>(capacity) &&
      geometry.recurrent_layers() && geometry.attention_layers(),
      "cross-turn regression requires a complete explicitly declared GA source context");
  const std::array turns{i::GenerationTurn{prompt, 3}, i::GenerationTurn{prompt, 3},
      i::GenerationTurn{prompt, last_turn_outputs}};
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, capacity, 1, 4,
      q::recurrent_bytes(q::state_schema_for(geometry, capacity), geometry), 1, 1, false, false, 2);
  q::Model model(std::move(source), geometry, capacity, plan, {}, false,
      q::ReplayPolicy::kPreferCapture, SIZE_MAX, {}, 0, {.primary = i::BackendKind::kCuda});
  const std::array shapes{std::size_t{1}, std::size_t{3}, std::size_t{4}};
  model.prepare_family(shapes);
  for (auto rows : shapes) model.plan_replay_uses(rows, 1, 128);
  model.freeze_shape_plan(); model.prepare_execution(shapes); model.prepare_relay_execution();
  std::vector<q::GenerationRequest> requests;
  requests.emplace_back("short-short-long", prompt, std::filesystem::path{}, output_total);
  auto &request = requests.front();
  request.progress.bind_turns(turns); request.progress.initial_sampling = {};
  request.input = q::cache::token_plan(prompt, model.capacity);
  request.ledger = std::make_unique<q::GenerationLedger>(q::generation_record_capacity(turns), 1);
  request.round_emitted.resize(output_total - 1);
  for (auto &row : request.round_emitted) row.reserve(1);
  request.turn_observations.resize(turns.size());
  for (std::size_t turn = 0; turn < turns.size(); ++turn) {
    auto owner = std::make_shared<std::vector<std::int32_t>>(prompt.size() + (turn ? 1u : 0u));
    std::copy(prompt.begin(), prompt.end(), owner->begin() + (turn ? 1u : 0u));
    request.warm_inputs.push_back(std::move(owner));
  }
  i::OutputDelivery delivery(2); i::RequestLoop loop; loop.prepare(1, 1);
  q::GenerationContinuation continuation; continuation.prepare(1);
  Result result; result.refusal_records.reserve(32);
  std::optional<q::PreparedReadback> displaced_output;
  SlotFacts refused_slot_before{};
  bool injected = false, typed_refusal = false;
  std::size_t force_target = 0;
  std::optional<std::size_t> active_refusal, first_reprime_facts;
  std::uintptr_t re_prime_invocation = 0;
  struct SubmitFact final {
    std::size_t turn = 0, slot = 0, recorded = 0, reused = 0;
    bool imported = false, successor = false;
    std::uintptr_t invocation = 0;
  };
  std::array<SubmitFact, 128> submissions{};
  std::size_t submission_count = 0;
  std::vector<std::weak_ptr<q::StepSources>> lifetimes; lifetimes.reserve(128);
  hyle::util::ScopeGuard cleanup{[&]() noexcept {
    if (model.successor_step && model.successor_step->execution.has_completion()) model.successor_step->execution.wait().IgnoreError();
    if (model.active_step && model.active_step->execution.has_completion()) model.active_step->execution.wait().IgnoreError();
    continuation.reset(); (void)model.abort_episode();
    if (request.session) { request.session->stop().IgnoreError(); (void)request.session->drain();
      (void)request.session->try_retire(); request.session.reset(); }
  }};
  const auto observed = [&](const q::GenerationBatchFacts &facts) {
    require(facts.members.size() == 1 && facts.members[0].request == &request &&
        facts.members[0].disposition == q::EvaluationDisposition::kPublished,
        "short-turn callback lost its original published participant");
    result.states.insert_or_assign(request.ledger->committed_count, state_bytes(model, *request.session->head));
    result.epochs.insert_or_assign(request.ledger->committed_count, request.session->head.epoch());
  };
  // This hook retains weak metadata only. No logging, range scans or provider
  // callback is performed while the original predecessor is pending.
  const auto submitted = [&] {
    require(submission_count < submissions.size() && model.active_step && model.active_step->phase,
        "cold submit-fact capacity or original active phase missing");
    const bool successor = model.successor_step && model.successor_step->submitted;
    const auto &step = successor ? *model.successor_step : *model.active_step;
    submissions[submission_count++] = {request.progress.turn_index, step.state_slot,
        step.capture_recorded, step.capture_reused, step.phase->imported_sources, successor,
        reinterpret_cast<std::uintptr_t>(step.execution.run->object_identity())};
    require(lifetimes.size() + 2 <= lifetimes.capacity(), "cold Source observation capacity exhausted");
    if (model.active_step->execution.sources) lifetimes.push_back(model.active_step->execution.sources);
    if (model.successor_step && model.successor_step->submitted && model.successor_step->execution.sources)
      lifetimes.push_back(model.successor_step->execution.sources);
  };
  const auto execute = [&](std::span<const i::RequestKey> active) {
    require(active.size() == 1 && active[0] == request.request_key, "short-turn request identity changed");
    const auto credit = loop.credit(request.request_key, request.progress.generated.size());
    std::array calls{q::prepare_generation_call(model, request, request.progress.output_limit(),
        nullptr, requests, credit)};
    // A2 has completed, but this call has not prepared/submitted A0. B1's
    // token Source is still the original A0 owner; only its unrelated output
    // owner changes. Keep the old real pool alive to exclude address reuse.
    if (force_output_tuple && !injected && request.progress.turn_index == 2 &&
        calls[0].generation && calls[0].quota >= 2 && continuation.matches(calls) &&
        continuation.ready_slot == 2 && continuation.next_slot == 0) {
      auto &phase = *model.phases.at({1, 1, q::GraphRole::kText,
          q::TextCompletion::kEmitNext, 0, true});
      force_target = (continuation.next_slot + 1) % q::execution_slot_count;
      require(force_target == 1 && *continuation.ready_slot != force_target &&
          !model.active_step && !model.successor_step && !model.carrier_borrowed,
          "forced tuple must select retired nonzero B1, not a live Source");
      refused_slot_before = slot_facts(model)[force_target];
      require(refused_slot_before.uses && refused_slot_before.keyed_variants &&
          phase.slots[force_target].carrier.retired(), "forced tuple target has no real recorded history");
      auto &run = *q::take(phase.slots[force_target].carrier.borrow());
      auto &io = phase.slots[force_target].data;
      auto &readback = *q::take(phase.slots[force_target].io.readback(io.next_id));
      q::check(run.unbind(phase.id_port));
      displaced_output.emplace(std::move(readback));
      readback = q::take(q::PreparedReadback::prepare(phase.child->compiled, phase.id_port,
          model.execution.storage_domains(), model.execution_budget));
      q::check(readback.bind(run));
      const auto after = slot_facts(model)[force_target];
      require(after.invocation == refused_slot_before.invocation &&
          after.uses == refused_slot_before.uses && after.keyed_variants == refused_slot_before.keyed_variants,
          "ordinary retired output rebinding unexpectedly destroyed the qualified capture cache");
      injected = true;
    }
    const auto refused_before = continuation.successor_refusals;
    const auto submitted_before = continuation.successor_submissions;
    const auto facts_begin = submission_count;
    const auto route_before = continuation.reprime_slot;
    const auto predecessor_slot = continuation.next_slot;
    const auto target_slot = (predecessor_slot + 1) % q::execution_slot_count;
    std::optional<SlotSnapshot> candidate_before;
    if (force_output_tuple && continuation.matches(calls) && !route_before)
      candidate_before = snapshot_slot(model, target_slot);
    q::execute_generation_calls(model, calls, false, nullptr, requests, submitted,
        relay ? &continuation : nullptr, observed);
    if (continuation.successor_refusals != refused_before) {
      require(force_output_tuple && injected && candidate_before && !active_refusal &&
          continuation.successor_refusals == refused_before + 1 &&
          continuation.reprime_slot == target_slot && continuation.next_slot == predecessor_slot &&
          !continuation.ready && continuation.pending_calls.empty() && !continuation.unavailable &&
          continuation.successor_submissions == submitted_before,
          "fulltuple refusal lost its typed origin, ordered cohort or actual predecessor route");
      require(!model.active_step && !model.successor_step && !model.carrier_borrowed && !request.session->pending,
          "typed refusal did not wait for actual original predecessor retirement");
      const auto after = verify_unsubmitted_refusal(model, target_slot, *candidate_before);
      require(result.refusal_records.size() < result.refusal_records.capacity(),
          "finite refusal observation capacity exhausted");
      result.refusal_records.push_back({predecessor_slot, target_slot, candidate_before->slot.uses,
          candidate_before->slot.keyed_variants, after.slot.keyed_variants});
      active_refusal = result.refusal_records.size() - 1;
      re_prime_invocation = candidate_before->slot.invocation;
      typed_refusal = true; result.retained_invocation = true;
    }
    if (route_before) {
      require(active_refusal && typed_refusal && continuation.successor_submissions == submitted_before,
          "temporary ordinary re-prime route attempted speculative B");
      if (!continuation.reprime_slot) {
        auto &refusal = result.refusal_records[*active_refusal];
        if (request.progress.turn_done) {
          // Original terminal release is allowed to abandon future re-priming.
          // It is never counted as a successful target or reuse proof.
          refusal.terminal_release = true;
        } else {
          require(submission_count == facts_begin + 1 && submissions[facts_begin].imported &&
              !submissions[facts_begin].successor && submissions[facts_begin].slot == *route_before &&
              *route_before == refusal.target && submissions[facts_begin].invocation == re_prime_invocation &&
              submissions[facts_begin].recorded > 0 && continuation.ready_slot == *route_before,
              "re-prime route cleared without the original target's completed ordinary submission");
          refusal.reprimed = true; ++result.completed_reprimes; result.ordinary_reprime = true;
          if (!first_reprime_facts) first_reprime_facts = submission_count;
        }
        active_refusal.reset();
      }
    }
    for (auto &call : calls) q::publish_generation_prompt(model, call, nullptr);
    require(!model.active_step && !model.successor_step && !model.carrier_borrowed,
        "short-turn snapshot must follow actual original native retirement");
    if (request.progress.turn_done) {
      const auto turn = request.progress.turn_index;
      result.sessions[turn] = reinterpret_cast<std::uintptr_t>(request.session.get());
      result.turn_slots[turn] = slot_facts(model);
    }
  };
  q::SynchronousGenerationWave wave{model, execute};
  q::GenerationRequestAdapter adapter{model, static_cast<q::cache::PrefixCache *>(nullptr), std::span(requests),
      false, static_cast<std::vector<std::size_t> *>(nullptr), wave, {}, relay ? &continuation : nullptr};
  request.request_key = loop.enqueue(0, &delivery); const auto key = request.request_key;
  for (std::size_t pumps = 0; loop.matches(key); ++pumps) {
    require(pumps < 96, "short-turn request did not terminate");
    require(!loop.pump_once(adapter).cancelled, "short-turn request was cancelled");
    if (loop.matches(key) && !delivery.pending(request.progress.generated).empty())
      require(loop.acknowledge(key, request.progress.generated.size(), adapter), "short-turn ACK rejected");
  }
  result.ids = request.progress.generated;
  result.commits.assign(request.ledger->records.begin(), request.ledger->records.begin() + request.ledger->committed_count);
  result.successors = continuation.successor_submissions; result.refusals = continuation.successor_refusals;
  require(result.ids.size() == output_total && request.progress.request_done() && !request.session,
      "short-turn request lost output or final Session retirement");
  require(result.sessions[0] && result.sessions[0] == result.sessions[1] && result.sessions[1] == result.sessions[2],
      "short-turn fixture did not preserve the same original Session across turns");
  if (relay) {
    require(result.turn_slots[0][0].uses == 1 && result.turn_slots[1][0].uses == 2 &&
        result.turn_slots[0][0].invocation == result.turn_slots[1][0].invocation &&
        result.turn_slots[0][0].keyed_variants == 1 && result.turn_slots[1][0].keyed_variants == 1 &&
        result.turn_slots[1][1].uses == 0 && result.turn_slots[1][2].uses == 0,
        "two short turns did not construct bootstrap-only slot0 with count2");
    if (!force_output_tuple) require(!result.refusals && result.successors,
        "long third turn performed a blind destructive B miss or never used a warmed successor");
    else {
      require(injected && typed_refusal && result.refusals == result.refusal_records.size() &&
          result.completed_reprimes && !continuation.reprime_slot && !active_refusal,
          "forced output tuple did not safely complete its finite ordinary recovery routes");
      result.forced_output_owner = true;
      if (first_reprime_facts) for (const auto &fact : std::span(submissions).subspan(
          *first_reprime_facts, submission_count - *first_reprime_facts))
        if (fact.successor && fact.reused > 0 && fact.recorded == 0) ++result.successors_after_reprime;
      if (require_recovery_reuse && !result.successors_after_reprime) {
        std::cerr << "long_recovery_no_B recorded_refusals=" << result.refusal_records.size()
            << " completed_reprimes=" << result.completed_reprimes << " B_submits=" << result.successors << '\n';
        for (const auto &fact : result.refusal_records)
          std::cerr << "refusal pred=" << fact.predecessor << " target=" << fact.target << " uses=" << fact.uses
              << " keys=" << fact.keys_before << "->" << fact.keys_after << " reprimed=" << fact.reprimed
              << " terminal_release=" << fact.terminal_release << '\n';
        require(false, "long finite recovery never resumed B using an existing actual capture");
      }
    }
  }
  continuation.reset();
  for (const auto &weak : lifetimes) require(weak.expired(), "short-turn Source owner survived real retirement");
  require(!model.snapshot_versions.live && !model.active_step && !model.successor_step && !model.carrier_borrowed,
      "short-turn cleanup retained native candidate or Session versions");
  cleanup.dismiss(); return result;
}
inline void compare(const Result &control, const Result &candidate) {
  require(control.ids == candidate.ids && control.states == candidate.states && control.epochs == candidate.epochs &&
      control.commits.size() == candidate.commits.size(),
      "three-turn relay changed actual IDs, full initialized state or ledger size");
  for (std::size_t row = 0; row < control.commits.size(); ++row) {
    auto expected = control.commits[row]; const auto &actual = candidate.commits[row];
    expected.quota = actual.quota; // Each actual call independently obeys RequestLoop credit.
    require(expected.same_result(actual) && expected.committed == actual.committed &&
        expected.user_output == actual.user_output && expected.imported == actual.imported &&
        expected.turn == actual.turn && std::ranges::equal(expected.ids(), actual.ids()),
        "three-turn relay changed a complete non-time original commit");
  }
}
inline void report(std::string_view name, const Result &candidate) {
  std::cout << "{\"status\":\"PASS\",\"case\":\"" << name
      << "\",\"same_Session\":true,\"complete_state_epoch_rng_ledger_equal\":true,"
      "\"generated_outputs\":" << candidate.ids.size() << ",\"successor_refusals\":" << candidate.refusals
      << ",\"successor_submissions\":" << candidate.successors
      << ",\"completed_ordinary_reprimes\":" << candidate.completed_reprimes
      << ",\"successors_reused_after_reprime\":" << candidate.successors_after_reprime
      << ",\"refusal_facts\":[";
  for (std::size_t index = 0; index < candidate.refusal_records.size(); ++index) {
    const auto &f = candidate.refusal_records[index]; if (index) std::cout << ',';
    std::cout << "{\"pred\":" << f.predecessor << ",\"target\":" << f.target << ",\"uses\":" << f.uses
        << ",\"keys_before\":" << f.keys_before << ",\"keys_after\":" << f.keys_after
        << ",\"reprimed\":" << f.reprimed << ",\"terminal_release\":" << f.terminal_release << '}';
  }
  std::cout << "]}\n";
}
inline int run(const std::filesystem::path &context32, const std::filesystem::path &context64) {
  const auto control = arm(context32, false);
  const auto candidate = arm(context32, true);
  compare(control, candidate); report("context32-three-three-fourteen-basic", candidate);
  const auto short_forced = arm(context32, true, true);
  compare(control, short_forced); report("context32-generic-fulltuple-refusals", short_forced);
  const auto long_control = arm(context64, false, false, 64, 30);
  const auto long_forced = arm(context64, true, true, 64, 30, true);
  compare(long_control, long_forced); report("synthetic-context64-three-three-thirty-recovery", long_forced);
  return 0;
}
} // namespace qwen_short_turn_capture_case
