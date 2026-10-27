#pragma once

// Included after the real Model and qwen_generation_state_case oracle. This
// drives the existing RequestLoop/advance_generation_wave; it owns no executor.
namespace qwen_continuous_generation_case {
namespace q = qwen35;
namespace i = hyle::inference;
using qwen_generation_state_case::require;
using qwen_generation_state_case::StateBytes;
using qwen_generation_state_case::state_bytes;
constexpr std::size_t outputs = 14, prompt_rows = 18, capacity = 32, intervention = 11;
enum class Mode { kNormal, kCredit, kCancel, kYield };

struct Result final {
  std::array<std::vector<std::int32_t>, 2> ids;
  std::array<std::vector<q::GenerationCommitRecord>, 2> records;
  std::array<std::map<std::size_t, StateBytes>, 2> states;
  std::array<std::map<std::size_t, std::uint64_t>, 2> epochs;
  std::size_t pumps = 0, submissions = 0, batches = 0, published_members = 0;
  std::size_t pending_returns = 0, consecutive_pending = 0, submit_before_observe = 0;
  std::size_t renewals = 0, credit_drains = 0, cancellations = 0, yield_drains = 0;
  bool pending_length27 = false;
};

inline void compare(const Result &control, const Result &candidate, std::size_t width) {
  require(control.ids == candidate.ids, "continuous wave changed the per-request output IDs");
  for (std::size_t member = 0; member < width; ++member) {
    require(control.records[member].size() == candidate.records[member].size() &&
        control.states[member] == candidate.states[member] && control.epochs[member] == candidate.epochs[member],
        "continuous wave changed a complete initialized state, prefix epoch, pending or RNG");
    for (std::size_t row = 0; row < control.records[member].size(); ++row) {
      auto expected = control.records[member][row]; const auto &actual = candidate.records[member][row];
      // The held ordinary successor can have one original credit while the
      // serial next call has two. Each actual quota is checked against its own
      // approved wave below; all mathematical and physical-member facts match.
      expected.quota = actual.quota;
      require(expected.same_result(actual) && expected.committed == actual.committed &&
          expected.user_output == actual.user_output && expected.imported == actual.imported &&
          expected.turn == actual.turn && std::ranges::equal(expected.ids(), actual.ids()),
          "continuous wave changed its committed ledger or original physical member facts");
    }
  }
}

inline Result run_arm(q::Model &model, const std::array<std::vector<std::int32_t>, 2> &prompts,
                      std::size_t width, Mode mode, bool continuous) {
  require(!model.active_step && !model.successor_step && !model.carrier_borrowed && !model.snapshot_versions.live,
      "preceding continuous arm retained its original native/state owner");
  std::array<std::array<i::GenerationTurn, 1>, 2> turns{
      std::array{i::GenerationTurn{prompts[0], outputs}}, std::array{i::GenerationTurn{prompts[1], outputs}}};
  std::vector<q::GenerationRequest> requests; requests.reserve(width);
  for (std::size_t member = 0; member < width; ++member) {
    requests.emplace_back(member ? "continuous-B" : "continuous-A", prompts[member], std::filesystem::path{}, outputs);
    auto &request = requests.back(); request.progress.bind_turns(turns[member]);
    request.progress.initial_sampling = {}; request.input = q::cache::token_plan(prompts[member], capacity);
    request.ledger = std::make_unique<q::GenerationLedger>(q::generation_record_capacity(turns[member]), 1);
    request.round_emitted.resize(outputs - 1);
    for (auto &values : request.round_emitted) values.reserve(1);
    request.turn_observations.resize(1);
    request.warm_inputs.push_back(std::make_shared<std::vector<std::int32_t>>(prompts[member]));
  }
  std::array deliveries{i::OutputDelivery(2), i::OutputDelivery(2)};
  i::RequestLoop loop; loop.prepare(width, width);
  q::GenerationContinuation continuation; continuation.prepare(width); continuation.enabled = continuous;
  std::array<i::RequestKey, 2> keys{};
  std::array<std::size_t, 2> observed{}, before_commits{}, returned_reserved{};
  std::vector<std::weak_ptr<q::StepSources>> sources; sources.reserve(128);
  Result result;
  std::optional<i::RequestWave> current_wave;
  bool action_sent = false, hold_ack = false, yielded = false;
  bool submitted_successor_this_pump = false;
  std::size_t streak = 0;
  hyle::util::ScopeGuard cleanup{[&]() noexcept {
    if (model.successor_step && model.successor_step->has_completion()) model.successor_step->wait().IgnoreError();
    if (model.active_step && model.active_step->has_completion()) model.active_step->wait().IgnoreError();
    continuation.release_ready(); (void)model.abort_episode();
    for (auto &request : requests) if (request.session) {
      request.session->stop().IgnoreError(); (void)request.session->drain();
      (void)request.session->try_retire(); request.session.reset();
    }
  }};
  const auto on_batch = [&](const q::GenerationBatchFacts &facts) {
    require(current_wave && !facts.members.empty() && facts.members.size() <= width &&
        facts.role == q::GraphRole::kText && facts.completion == q::TextCompletion::kEmitNext &&
        facts.input_rows == (facts.generation ? 1 : prompt_rows), "continuous observer lost its actual target batch");
    ++result.batches;
    for (std::size_t index = 0; index < facts.members.size(); ++index) {
      const auto &call = facts.members[index]; auto &request = *call.request; const auto member = request.request_key.slot;
      require(member < width && &request == &requests[member] && request.request_key == keys[member] && loop.matches(keys[member]),
          "continuous observer changed its original RequestKey owner");
      if (call.disposition == q::EvaluationDisposition::kCancelled) {
        require(request.ledger->committed_count == observed[member], "cancelled member published an original candidate");
        continue;
      }
      require(call.disposition == q::EvaluationDisposition::kPublished && request.ledger->committed_count == observed[member] + 1,
          "continuous observer duplicated or omitted an authoritative commit");
      const auto &record = request.ledger->records[observed[member]];
      require(record.committed && !record.input_deferred && record.requests == facts.members.size() &&
          record.member_index == index && record.quota == call.quota && record.verified_rows == facts.input_rows &&
          record.after == q::generation_boundary(*request.session->head), "continuous ledger lost its actual batch geometry");
      if (facts.generation) {
        const auto found = std::find(current_wave->keys.begin(), current_wave->keys.end(), keys[member]);
        require(found != current_wave->keys.end(), "continuous quota has no original wave member");
        const auto &grant = current_wave->credits[static_cast<std::size_t>(found - current_wave->keys.begin())];
        require(call.quota && call.quota <= request.progress.remaining_outputs() &&
            call.quota <= grant.reserved + grant.additional,
            "continuous record exceeds its actual approved delivery/turn budget");
      }
      ++observed[member]; ++result.published_members;
      result.states[member].emplace(observed[member], state_bytes(model, *request.session->head));
      result.epochs[member].emplace(observed[member], request.session->head.epoch());
    }
  };
  // Cancellation goes through this same adapter's original Session stop. The
  // callback is assigned before any pump can invoke on_submit.
  const auto stop = [&](i::RequestKey key) {
    continuation.release(key);
    return requests[key.slot].session ? requests[key.slot].session->stop() : hyle::Status::Ok();
  };
  struct StopView final {
    decltype(stop) &invoke;
    hyle::Status stop(i::RequestKey key) const { return invoke(key); }
  } stop_view{stop};
  const auto cancel_member = [&] {
    require(width == 2 && !action_sent && requests[0].progress.generated.size() == intervention,
        "cross-pump cancellation missed the intended original prefix");
    const auto original = q::generation_boundary(*requests[0].session->head);
    const auto peer = q::generation_boundary(*requests[1].session->head);
    const auto status = loop.cancel(keys[0], stop_view);
    require(status.ok() || (status.IsCancelled() && !status.has_native_error()), "cross-pump cancellation returned a native failure");
    require(q::generation_boundary(*requests[0].session->head) == original &&
        q::generation_boundary(*requests[1].session->head) == peer && !requests[1].session->stopping,
        "cancellation itself published a candidate or permanently stopped its peer");
    action_sent = true;
  };
  const auto on_submit = [&] {
    ++result.submissions;
    require(model.active_step && model.active_step->submitted && model.active_step->has_completion(),
        "continuous submit hook did not follow the original accepted native work");
    if (model.active_step->sources) sources.push_back(model.active_step->sources);
    if (model.successor_step && model.successor_step->submitted) {
      require(continuous && model.successor_step->has_completion() &&
          model.successor_step->members.size() == current_wave->keys.size(), "continuous successor changed its original cohort");
      for (auto key : current_wave->keys)
        require(requests[key.slot].ledger->committed_count == before_commits[key.slot],
            "successor was submitted only after its original predecessor was observed");
      sources.push_back(model.successor_step->sources);
      submitted_successor_this_pump = true; ++result.submit_before_observe;
      for (const auto &member : model.successor_step->members)
        if (member.next->length >= 27) result.pending_length27 = true;
      require(model.snapshot_versions.live <= model.cache_plan.snapshot_slots &&
          model.carrier_borrowed <= 2 && model.cache_plan.in_flight_batches == 2,
          "continuous test escaped its actual D2 state/page budget");
    }
    // The serial control cancels after the corresponding real R2 batch has
    // submitted. Thus its surviving member keeps the same batch geometry as
    // the already submitted candidate cancelled between pumps.
    if (!continuous && mode == Mode::kCancel && !action_sent && requests[0].progress.generated.size() == intervention)
      cancel_member();
  };
  const auto execute_ready = [&](i::RequestWave wave) {
    std::vector<q::PreparedGenerationCall> calls; calls.reserve(wave.keys.size());
    for (std::size_t at = 0; at < wave.keys.size(); ++at) {
      const auto key = wave.keys[at]; auto &request = requests[key.slot];
      require(!wave.credits[at].reserved && wave.credits[at].additional, "fresh continuous call has no approved output credit");
      calls.push_back(q::prepare_generation_call(model, request, outputs, nullptr, requests, wave.credits[at].additional));
    }
    q::execute_generation_calls(model, calls, false, nullptr, requests, on_submit,
        continuous ? &continuation : nullptr, on_batch, continuous, wave.credits);
    for (auto &call : calls) q::publish_generation_prompt(model, call, nullptr);
  };
  const auto execute_wave = [&](i::RequestWave wave, std::span<std::size_t> reserved) {
    current_wave = wave;
    const bool resumed = !continuation.pending_calls.empty();
    const bool draining = wave.drain_requested || continuation.drain_requested ||
        std::ranges::any_of(wave.credits, [](const auto &grant) { return grant.cancelled; });
    const auto answer = q::advance_generation_wave(model, wave, reserved, false, continuation, execute_ready, on_submit, on_batch);
    for (std::size_t at = 0; at < wave.keys.size(); ++at) returned_reserved[wave.keys[at].slot] = reserved[at];
    if (resumed && !draining && !answer.failure &&
        std::ranges::any_of(wave.credits, [](const auto &grant) { return grant.additional != 0; })) {
      for (std::size_t at = 0; at < wave.keys.size(); ++at) {
        const auto member = wave.keys[at].slot; const auto &record = requests[member].ledger->records[before_commits[member]];
        require(record.committed && record.quota == wave.credits[at].reserved + wave.credits[at].additional,
            "resumed ordinary record did not adopt this pump's real renewed grant");
      }
      ++result.renewals;
    }
    current_wave.reset();
    return answer;
  };
  q::GenerationRequestAdapter adapter{model, static_cast<q::cache::PrefixCache *>(nullptr), std::span(requests), false,
      static_cast<std::vector<std::size_t> *>(nullptr), execute_wave, {}, continuous ? &continuation : nullptr};
  for (std::size_t member = 0; member < width; ++member)
    requests[member].request_key = keys[member] = loop.enqueue(member, &deliveries[member]);
  for (; std::ranges::any_of(std::span(keys).first(width), [&](auto key) { return loop.matches(key); }); ++result.pumps) {
    require(result.pumps < 80, "finite continuous request did not retire");
    for (std::size_t member = 0; member < width; ++member) before_commits[member] = requests[member].ledger->committed_count;
    submitted_successor_this_pump = false;
    const auto pumped = loop.pump_once(adapter); result.cancellations += pumped.cancelled;
    const bool pending = !continuation.pending_calls.empty();
    if (pending) {
      require(continuous && pumped.waiting_retirement && model.active_step && !model.successor_step &&
          model.active_step->submitted && continuation.pending_calls.size() == loop.active().size(),
          "continuous pump lost its original pending native owner or active cohort");
      ++result.pending_returns;
      streak = submitted_successor_this_pump ? streak + 1 : 0;
      result.consecutive_pending = (std::max)(result.consecutive_pending, streak);
      for (auto key : loop.active()) {
        const auto &request = requests[key.slot];
        require(key == keys[key.slot] && request.progress.turn_index == 0 && request.progress.turn_started() &&
            returned_reserved[key.slot] == 1 && request.session->pending == model.active_step.get() &&
            request.ledger->next().pending_input == request.progress.generated.back(),
            "pending wave changed its key/turn/ledger or released its actual output reservation");
      }
    } else streak = 0;
    for (std::size_t member = 0; member < width; ++member)
      require(requests[member].progress.observed_commits() == requests[member].ledger->committed_count,
          "continuous pump lost or duplicated a committed host observation");
    if (mode != Mode::kNormal && !action_sent && requests[0].progress.generated.size() == intervention) {
      if (continuous) require(pending && result.consecutive_pending >= 3, "control intent arrived before three genuine cross-pump successors");
      if (mode == Mode::kCredit) { hold_ack = true; action_sent = true; }
      else if (mode == Mode::kYield) { loop.request_yield(); action_sent = true; }
      else if (continuous) cancel_member();
    }
    if (mode == Mode::kCredit && hold_ack && pumped.waiting_output && !pumped.waiting_retirement) {
      require(!pending && !model.active_step && !model.successor_step && !model.carrier_borrowed && loop.active().empty(),
          "credit exhaustion freed an in-flight owner or submitted another successor");
      for (std::size_t member = 0; member < width; ++member)
        require(requests[member].progress.generated.size() == intervention + 1 &&
            deliveries[member].pending(requests[member].progress.generated).size() == 2 && loop.matches(keys[member]),
            "credit drain did not preserve exactly two original unacknowledged outputs");
      ++result.credit_drains; hold_ack = false;
    }
    if (mode == Mode::kYield && continuous && action_sent && !yielded && !pending) {
      require(requests[0].progress.generated.size() == intervention + 1 && loop.active().empty() &&
          loop.waiting_front() == keys[0] && !model.active_step && !model.carrier_borrowed,
          "explicit yield did not wait for actual retirement before returning survivors to FIFO");
      for (std::size_t member = 0; member < width; ++member)
        require(loop.matches(keys[member]) && requests[member].progress.turn_index == 0 &&
            requests[member].progress.turn_started() && !requests[member].progress.turn_done,
            "yield replaced a request key or restarted its original turn");
      yielded = true; ++result.yield_drains;
    }
    for (std::size_t member = 0; member < width; ++member) if (loop.matches(keys[member]) && !hold_ack) {
      const auto produced = requests[member].progress.generated.size();
      require(loop.acknowledge(keys[member], produced, adapter), "original continuous ACK was rejected");
      require(deliveries[member].available(produced) >= returned_reserved[member], "ACK released the pending candidate's charged credit");
    }
  }
  require(!continuous || (result.consecutive_pending >= 3 && result.submit_before_observe >= 3 &&
      result.renewals >= 2 && result.pending_length27 && !sources.empty()),
      "finite candidate never exercised three real continuous pumps and the length27 D2 boundary");
  if (mode == Mode::kCredit) require(result.credit_drains == 1, "actual bounded-credit drain was not checked");
  if (mode == Mode::kYield && continuous) require(result.yield_drains == 1, "actual cross-pump yield was not checked");
  if (mode == Mode::kCancel) require(action_sent && result.cancellations == 1, "cross-pump member cancellation was not observed once");
  else require(!result.cancellations, "uncancelled continuous arm cancelled a request");
  for (std::size_t member = 0; member < width; ++member) {
    const auto &request = requests[member]; result.ids[member] = request.progress.generated;
    result.records[member].assign(request.ledger->records.begin(), request.ledger->records.begin() + request.ledger->committed_count);
    const auto expected = mode == Mode::kCancel && member == 0 ? intervention : outputs;
    require(result.ids[member].size() == expected && observed[member] == expected && !request.session &&
        request.progress.turn_index == 0 && request.ledger->committed_count == expected,
        "continuous journey lost outputs, retained its Session or advanced an original turn");
  }
  require(result.submissions == result.batches, "continuous facts counted a pending native batch twice or omitted its retirement");
  continuation.reset();
  for (const auto &source : sources) require(source.expired(), "terminal continuous cleanup retained an old Source");
  require(!model.active_step && !model.successor_step && !model.carrier_borrowed && !model.snapshot_versions.live &&
      model.page_inventory.free_ids.size() == model.page_inventory.owners.size() &&
      std::ranges::all_of(model.page_inventory.owners, [](auto owners) { return owners == 0; }) &&
      loop.active().empty() && !loop.waiting_front(), "continuous journey retained an original episode, page reference, version or request key");
  cleanup.dismiss(); return result;
}

inline int run(const std::filesystem::path &input) {
  const auto original = q::read_weight_geometry(input / "weights.tsv");
  require(original.hidden == 64 && original.intermediate == 128 && original.vocabulary == 64 &&
      original.layers == "GA" && original.context_limit >= static_cast<std::int64_t>(capacity),
      "continuous case requires the cached untrained GA/H64 fixture with an explicitly prepared context32 manifest");
  const auto base_prompt = q::read_ids(input / "prompt.ids", original.vocabulary, capacity);
  require(!base_prompt.empty() && base_prompt.size() <= prompt_rows, "continuous source prompt cannot form its finite P18 input");
  std::array<std::vector<std::int32_t>, 2> prompts;
  for (std::size_t row = 0; row < prompt_rows; ++row) prompts[0].push_back(base_prompt[row % base_prompt.size()]);
  prompts[1] = prompts[0];
  const auto different = std::find_if(prompts[1].rbegin() + 1, prompts[1].rend(), [&](auto id) { return id != prompts[1].back(); });
  require(different != prompts[1].rend(), "continuous R2 needs distinct original prompt IDs");
  std::iter_swap(different, prompts[1].rbegin());
  struct ArmSummary { std::size_t width; Mode mode; Result result; };
  std::vector<ArmSummary> summaries;
  for (const auto width : {std::size_t{1}, std::size_t{2}}) {
    auto source = q::prepare_weight_source(input / "weights.tsv", false); const auto geometry = source.roles.geometry;
    const auto plan = q::cache::plan(geometry, source.cache_row_bytes, capacity, width, prompt_rows,
        q::recurrent_bytes(q::state_schema_for(geometry, capacity), geometry), width, 1, false, false, 2);
    require(plan.snapshot_slots == 3 * width && plan.state_slots == 3 * width && plan.in_flight_batches == 2,
        "continuous fixture silently enlarged its original D2 candidate budget");
    q::Model model(std::move(source), geometry, capacity, plan, {}, false, q::ReplayPolicy::kPreferCapture,
        SIZE_MAX, {}, 0, {.primary = i::BackendKind::kCuda});
    const std::array shapes{prompt_rows, std::size_t{1}};
    for (std::size_t active = 1; active <= width; ++active) {
      model.prepare_family(shapes, false, active);
      model.plan_replay_uses(prompt_rows, active, 16); model.plan_replay_uses(1, active, outputs * 24);
    }
    model.freeze_shape_plan();
    for (std::size_t active = 1; active <= width; ++active) {
      model.prepare_execution(shapes, active); model.prepare_relay_execution(active);
    }
    for (auto mode : {Mode::kNormal, Mode::kCredit, Mode::kCancel, Mode::kYield}) {
      if (width == 1 && mode != Mode::kNormal) continue;
      const auto control = run_arm(model, prompts, width, mode, false);
      auto candidate = run_arm(model, prompts, width, mode, true);
      compare(control, candidate, width);
      summaries.push_back({width, mode, std::move(candidate)});
    }
  }
  require(summaries.size() == 5, "continuous case did not finish all ten control/candidate arms");
  std::cout << "{\"status\":\"PASS\",\"backend\":\"cuda\",\"fixture\":\"untrained_GA_H64\","
      "\"prompt_tokens\":18,\"outputs_per_live_member\":14,\"capacity\":32,\"in_flight_batches\":2,"
      "\"control_and_candidate_arms\":10,\"same_original_executor\":true,\"complete_prefix_state_epoch_rng_equal\":true,"
      "\"actual_owner_credit_checked\":true,\"all_source_session_page_owners_retired\":true,"
      "\"model_eos_run\":false,\"model_quality_claim\":false,\"performance_claim\":false,\"candidates\":[";
  for (std::size_t at = 0; at < summaries.size(); ++at) {
    const auto &entry = summaries[at]; const auto &value = entry.result;
    constexpr std::array names{"normal", "credit_exhaustion", "member_cancel", "yield_resume"};
    std::cout << (at ? "," : "") << "{\"requests\":" << entry.width << ",\"mode\":\"" << names[static_cast<std::size_t>(entry.mode)] << '"'
        << ",\"pumps\":" << value.pumps << ",\"actual_batches\":" << value.batches
        << ",\"pending_returns\":" << value.pending_returns << ",\"max_consecutive_pending\":" << value.consecutive_pending
        << ",\"submit_before_observe\":" << value.submit_before_observe << ",\"renewals\":" << value.renewals
        << ",\"credit_drains\":" << value.credit_drains << ",\"cancelled_requests\":" << value.cancellations
        << ",\"yield_drains\":" << value.yield_drains << ",\"pending_length27\":" << (value.pending_length27 ? "true" : "false") << '}';
  }
  std::cout << "]}\n"; return 0;
}
} // namespace qwen_continuous_generation_case
