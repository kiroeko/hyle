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
enum class Mode { kNormal, kCredit, kCancel, kYield, kExtraStop, kControlFailure };

class SubmittedControlFailure final : public std::runtime_error {
public:
  SubmittedControlFailure() : std::runtime_error("continuous accepted-work control failure") {}
  std::size_t prefix = intervention;
};

struct Result final {
  std::array<std::vector<std::int32_t>, 2> ids;
  std::array<std::vector<q::GenerationCommitRecord>, 2> records;
  std::array<std::map<std::size_t, StateBytes>, 2> states;
  std::array<std::map<std::size_t, std::uint64_t>, 2> epochs;
  std::size_t pumps = 0, submissions = 0, batches = 0, published_members = 0;
  std::size_t pending_returns = 0, consecutive_pending = 0, submit_before_observe = 0;
  std::size_t renewals = 0, credit_drains = 0, cancellations = 0, yield_drains = 0;
  std::size_t stop_outputs = 0, discarded_batches = 0, survivor_r1 = 0;
  std::size_t control_failures = 0, aborted_submissions = 0;
  bool stop_successor_accepted = false;
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
                      std::size_t width, Mode mode, bool continuous, std::int32_t stop_id = -1) {
  require(!model.active_step && !model.successor_step && !model.carrier_borrowed && !model.snapshot_versions.live,
      "preceding continuous arm retained its original native/state owner");
  std::array<std::array<i::GenerationTurn, 1>, 2> turns{
      std::array{i::GenerationTurn{prompts[0], outputs}}, std::array{i::GenerationTurn{prompts[1], outputs}}};
  std::vector<q::GenerationRequest> requests; requests.reserve(width);
  for (std::size_t member = 0; member < width; ++member) {
    const std::array extra{stop_id};
    requests.emplace_back(member ? "continuous-B" : "continuous-A", prompts[member], std::filesystem::path{}, outputs, 0,
        mode == Mode::kExtraStop && member == 0 ? i::RequestStopPolicy(extra) : i::RequestStopPolicy{});
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
    if (model.successor_step && model.successor_step->execution.has_completion()) model.successor_step->execution.wait().IgnoreError();
    if (model.active_step && model.active_step->execution.has_completion()) model.active_step->execution.wait().IgnoreError();
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
    bool all_discarded = true;
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
      if (mode == Mode::kExtraStop && member == 0 && record.stopped) {
        require(record.after.next_id == stop_id && !q::is_end_of_generation(stop_id) &&
            request.stop_policy.extra_tokens().size() == 1 && observed[member] == intervention,
            "continuous extra-stop did not select its actual twelfth output through the original stop policy");
        ++result.stop_outputs;
      }
      all_discarded = false;
      ++observed[member]; ++result.published_members;
      result.states[member].emplace(observed[member], state_bytes(model, *request.session->head));
      result.epochs[member].emplace(observed[member], request.session->head.epoch());
    }
    if (all_discarded) {
      require(mode == Mode::kExtraStop && continuous && result.stop_outputs == 1 &&
          !model.active_step && !model.successor_step && !model.carrier_borrowed,
          "extra-stop reported a discarded successor before actual original native retirement");
      for (const auto &call : facts.members) {
        const auto &request = *call.request; const auto member = request.request_key.slot;
        require(state_bytes(model, *request.session->head) == result.states[member].at(observed[member]) &&
            request.session->head.epoch() == result.epochs[member].at(observed[member]),
            "discarding accepted B changed A's already published complete state, epoch or RNG");
      }
      ++result.discarded_batches;
    }
    if (((mode == Mode::kExtraStop && result.stop_outputs) || (mode == Mode::kControlFailure && result.control_failures)) &&
        width == 2 && facts.members.size() == 1 &&
        facts.members[0].request == &requests[1] && !all_discarded) {
      require(!requests[1].session->stopping && !requests[1].session->failed,
          "EOS-equivalent successor discard permanently stopped the surviving peer");
      ++result.survivor_r1;
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
    require(model.active_step && model.active_step->submitted && model.active_step->execution.has_completion(),
        "continuous submit hook did not follow the original accepted native work");
    if (model.active_step->execution.sources) sources.push_back(model.active_step->execution.sources);
    if (model.successor_step && model.successor_step->submitted) {
      require(continuous && model.successor_step->execution.has_completion() &&
          model.successor_step->members.size() == current_wave->keys.size(), "continuous successor changed its original cohort");
      for (auto key : current_wave->keys)
        require(requests[key.slot].ledger->committed_count == before_commits[key.slot],
            "successor was submitted only after its original predecessor was observed");
      sources.push_back(model.successor_step->execution.sources);
      submitted_successor_this_pump = true; ++result.submit_before_observe;
      if (mode == Mode::kExtraStop && requests[0].progress.generated.size() == intervention) {
        require(model.successor_step->capture_reused > 0 && !model.successor_step->capture_recorded,
            "extra-stop successor did not use its original already captured native route");
        result.stop_successor_accepted = true;
      }
      for (const auto &member : model.successor_step->members)
        if (member.next->length >= 27) result.pending_length27 = true;
      require(model.snapshot_versions.live <= model.cache_plan.snapshot_slots &&
          model.carrier_borrowed <= 2 && model.cache_plan.in_flight_batches == 2,
          "continuous test escaped its actual D2 state/page budget");
    }
    // The serial control cancels after the corresponding real R2 batch has
    // submitted. Thus its surviving member keeps the same batch geometry as
    // the already submitted candidate cancelled between pumps.
    if ((mode == Mode::kCancel || mode == Mode::kControlFailure) && !action_sent &&
        requests[0].progress.generated.size() == intervention &&
        (!continuous || (mode == Mode::kControlFailure && model.successor_step && model.successor_step->submitted))) {
      cancel_member();
      if (mode == Mode::kControlFailure) {
        require(current_wave->keys.size() == 2 && result.consecutive_pending >= (continuous ? 3u : 0u),
            "control failure did not follow the original accepted complete cohort");
        throw SubmittedControlFailure{};
      }
    }
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
    const bool draining = wave.drain_requested || continuation.cohort.drain_requested() ||
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
    i::RequestPumpResult pumped;
    try { pumped = loop.pump_once(adapter); }
    catch (const SubmittedControlFailure &failure) {
      require(mode == Mode::kControlFailure && action_sent && failure.prefix == intervention &&
          std::string_view(failure.what()) == "continuous accepted-work control failure" && !result.control_failures,
          "accepted-work control failure was changed, repeated or raised outside its intended arm");
      require(!model.active_step && !model.successor_step && !model.carrier_borrowed &&
          continuation.pending_calls.empty(), "control exception released its original cohort before native retirement");
      for (std::size_t member = 0; member < width; ++member) {
        const auto &request = requests[member];
        require(loop.matches(keys[member]) && request.progress.generated.size() == intervention &&
            request.progress.observed_commits() == observed[member] && request.ledger->committed_count == observed[member] &&
            !request.session->pending && state_bytes(model, *request.session->head) == result.states[member].at(observed[member]) &&
            request.session->head.epoch() == result.epochs[member].at(observed[member]),
            "aborted accepted work changed its previously committed prefix, state, epoch or RNG");
      }
      require(requests[0].session->stopping && !requests[1].session->stopping && !requests[1].session->failed,
          "control exception permanently stopped the uncancelled original peer");
      ++result.control_failures; result.aborted_submissions += continuous ? 2 : 1;
    }
    result.cancellations += pumped.cancelled;
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
    if ((mode == Mode::kCredit || mode == Mode::kCancel || mode == Mode::kYield) &&
        !action_sent && requests[0].progress.generated.size() == intervention) {
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
  if (mode == Mode::kCancel || mode == Mode::kControlFailure)
    require(action_sent && result.cancellations == 1, "cross-pump member cancellation was not observed once");
  else require(!result.cancellations, "uncancelled continuous arm cancelled a request");
  if (mode == Mode::kControlFailure) require(result.control_failures == 1 && result.survivor_r1,
      "actual accepted-work control exception was not observed exactly once");
  if (mode == Mode::kExtraStop) {
    require(result.stop_outputs == 1 && (!continuous || (result.stop_successor_accepted && result.discarded_batches == 1 &&
        continuation.successor_discards == 1 && continuation.successor_publications + 1 == continuation.successor_submissions)),
        "EOS-equivalent stop did not preserve A and actually retire all accepted B without publication");
    require(width == 1 || result.survivor_r1, "EOS-equivalent stop did not resume its original peer as R1");
  }
  for (std::size_t member = 0; member < width; ++member) {
    const auto &request = requests[member]; result.ids[member] = request.progress.generated;
    result.records[member].assign(request.ledger->records.begin(), request.ledger->records.begin() + request.ledger->committed_count);
    const auto expected = member == 0 && (mode == Mode::kCancel || mode == Mode::kControlFailure) ? intervention
        : member == 0 && mode == Mode::kExtraStop ? intervention + 1 : outputs;
    require(result.ids[member].size() == expected && observed[member] == expected && !request.session &&
        request.progress.turn_index == 0 && request.ledger->committed_count == expected,
        "continuous journey lost outputs, retained its Session or advanced an original turn");
  }
  require(result.submissions == result.batches + result.aborted_submissions,
      "continuous facts counted a pending native batch twice or omitted its completed/aborted retirement");
  continuation.reset();
  for (const auto &source : sources) require(source.expired(), "terminal continuous cleanup retained an old Source");
  require(!model.active_step && !model.successor_step && !model.carrier_borrowed && !model.snapshot_versions.live &&
      model.page_inventory.free_ids.size() == model.page_inventory.owners.size() &&
      std::ranges::all_of(model.page_inventory.owners, [](auto owners) { return owners == 0; }) &&
      loop.active().empty() && !loop.waiting_front(), "continuous journey retained an original episode, page reference, version or request key");
  cleanup.dismiss(); return result;
}

// The recorded original normal fixture settles on ID34 after output3, so no
// immutable extra-stop ID can first fire after three continuous pumps. Derive
// only small V64 embedding/head/residual roles for a finite greedy chain. The
// original GA state-producing roles, kernels and resources still execute.
// This tests extra-stop's shared EOS contract, not real Qwen model EOS/quality.
inline std::filesystem::path extra_stop_fixture(const std::filesystem::path &input) {
  const auto original = q::prepared::read_roles(input / "weights.tsv", false);
  auto geometry = original.geometry;
  require(geometry.hidden == 64 && geometry.vocabulary == 64 && geometry.layers == "GA",
      "continuous stop fixture requires the bounded original V64 GA inventory");
  const auto norm = std::ranges::find(original.selected, std::string("model.language_model.norm.weight"),
      &q::prepared::Role::name);
  require(norm != original.selected.end() && norm->dtype == "BF16" && norm->bytes == 128,
      "continuous stop fixture requires the original BF16 final norm");
  std::array<std::uint16_t, 64> gamma{};
  std::ifstream source_norm(norm->source, std::ios::binary);
  source_norm.seekg(static_cast<std::streamoff>(norm->offset));
  source_norm.read(reinterpret_cast<char *>(gamma.data()), static_cast<std::streamsize>(sizeof(gamma)));
  require(bool(source_norm) && std::ranges::all_of(gamma, [](auto value) { return value == 0; }),
      "continuous stop fixture requires original raw gamma0, hence effective norm gamma1");
  geometry.tied_head = false; geometry.mtp_layers = 0; geometry.context_limit = capacity; geometry.validate();
  const auto folder = input / "continuous-extra-stop-v64";
  std::filesystem::create_directories(folder);
  require(!std::filesystem::exists(folder / "weights.tsv") && !std::filesystem::exists(folder / "chain-weights.bin"),
      "continuous stop fixture already exists; preserve its previous attempt");
  const auto body = std::filesystem::absolute(folder / "chain-weights.bin");
  std::ofstream payload(body, std::ios::binary), manifest(folder / "weights.tsv");
  require(bool(payload) && bool(manifest), "cannot create finite continuous stop fixture");
  manifest << q::prepared::role_signature << '\n' << std::string(64, 'c') << '\n' << qwen_test::geometry_line(geometry) << '\n';
  std::size_t offset = 0;
  for (const auto &[name, parameter] : q::prepared::parameter_inventory(false, geometry)) {
    const bool embedding = name == "model.language_model.embed_tokens.weight", head = name == "lm_head.weight";
    const bool residual = name.ends_with(".mlp.down_proj.weight") ||
        name.ends_with(".linear_attn.out_proj.weight") || name.ends_with(".self_attn.o_proj.weight");
    if (!embedding && !head && !residual) {
      const auto found = std::ranges::find(original.selected, name, &q::prepared::Role::name);
      require(found != original.selected.end() && found->shape == parameter.shape,
          "continuous stop fixture cannot retain an original GA state-producing role");
      manifest << name << ' ' << found->dtype << ' ' << parameter.shape.size();
      for (auto dimension : parameter.shape) manifest << ' ' << dimension;
      manifest << ' ' << std::quoted(std::filesystem::absolute(found->source).generic_string())
          << ' ' << found->offset << ' ' << found->bytes << '\n';
      continue;
    }
    std::size_t elements = 1;
    for (auto dimension : parameter.shape) elements = q::cache::multiply(elements, static_cast<std::size_t>(dimension));
    std::vector<std::uint16_t> values(elements, 0);
    if (embedding) for (std::size_t token = 0; token < 64; ++token) values[token * 64 + token] = 0x3f80;
    else if (head) {
      values[32 * 64 + 2] = 0x3f80;
      for (std::size_t token = 3; token < 18; ++token) values[(token + 1) * 64 + token] = 0x3f80;
      for (std::size_t token = 32; token < 48; ++token) values[(token + 1) * 64 + token] = 0x3f80;
    }
    const auto bytes = q::cache::multiply(values.size(), sizeof(std::uint16_t));
    payload.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(bytes));
    manifest << name << " BF16 " << parameter.shape.size();
    for (auto dimension : parameter.shape) manifest << ' ' << dimension;
    manifest << ' ' << std::quoted(body.generic_string()) << ' ' << offset << ' ' << bytes << '\n';
    offset = q::cache::add(offset, bytes);
  }
  require(bool(payload) && bool(manifest) && offset <= (128u << 10),
      "continuous stop fixture write failed or exceeded its finite role-byte bound");
  payload.close(); manifest.close();
  require(bool(payload) && bool(manifest), "cannot flush finite continuous stop role bytes/manifest");
  std::ofstream prompt(folder / "prompt.ids"); prompt << "1 2 3\n";
  require(bool(prompt), "cannot write continuous stop fixture prompt");
  return folder;
}

inline int run(const std::filesystem::path &input) {
  const auto original = q::read_weight_geometry(input / "weights.tsv");
  require(original.hidden == 64 && original.intermediate == 128 && original.vocabulary == 64 &&
      original.layers == "GA" && original.context_limit >= static_cast<std::int64_t>(capacity),
      "continuous case requires the cached untrained GA/H64 fixture with an explicitly prepared context32 manifest");
  struct ArmSummary { std::size_t width; Mode mode; Result result; };
  std::vector<ArmSummary> summaries;
  const auto chain = extra_stop_fixture(input);
  for (bool extra_stop : {false, true}) {
  const auto &selected_input = extra_stop ? chain : input;
  const auto base_prompt = q::read_ids(selected_input / "prompt.ids", original.vocabulary, capacity);
  require(!base_prompt.empty() && base_prompt.size() <= prompt_rows, "continuous source prompt cannot form its finite P18 input");
  std::array<std::vector<std::int32_t>, 2> prompts;
  for (std::size_t row = 0; row < prompt_rows; ++row) prompts[0].push_back(base_prompt[row % base_prompt.size()]);
  prompts[1] = prompts[0];
  const auto different = std::find_if(prompts[1].rbegin() + 1, prompts[1].rend(), [&](auto id) { return id != prompts[1].back(); });
  require(different != prompts[1].rend(), "continuous R2 needs distinct original prompt IDs");
  std::iter_swap(different, prompts[1].rbegin());
  for (const auto width : {std::size_t{1}, std::size_t{2}}) {
    auto source = q::prepare_weight_source(selected_input / "weights.tsv", false); const auto geometry = source.roles.geometry;
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
    for (auto mode : {Mode::kNormal, Mode::kCredit, Mode::kCancel, Mode::kYield, Mode::kExtraStop, Mode::kControlFailure}) {
      if ((mode == Mode::kExtraStop) != extra_stop ||
          (width == 1 && mode != Mode::kNormal && mode != Mode::kExtraStop)) continue;
      constexpr std::int32_t stop_id = 15;
      const auto control = run_arm(model, prompts, width, mode, false, stop_id);
      if (extra_stop) {
        for (std::size_t at = 0; at < control.ids[0].size(); ++at)
          require(control.ids[0][at] == static_cast<std::int32_t>(4 + at),
              "finite V64 stop chain was not qualified by its actual original CUDA outputs");
        require(control.ids[0].back() == stop_id && control.records[0].back().stopped,
            "finite V64 control did not actually select its cold extra-stop ID");
      }
      auto candidate = run_arm(model, prompts, width, mode, true, stop_id);
      compare(control, candidate, width);
      summaries.push_back({width, mode, std::move(candidate)});
    }
  }
  }
  require(summaries.size() == 8, "continuous case did not finish all sixteen control/candidate arms");
  std::cout << "{\"status\":\"PASS\",\"backend\":\"cuda\",\"fixture\":\"untrained_GA_H64\","
      "\"prompt_tokens\":18,\"outputs_per_live_member\":14,\"capacity\":32,\"in_flight_batches\":2,"
      "\"control_and_candidate_arms\":16,\"same_original_executor\":true,\"complete_prefix_state_epoch_rng_equal\":true,"
      "\"actual_owner_credit_checked\":true,\"all_source_session_page_owners_retired\":true,"
      "\"model_eos_run\":false,\"synthetic_v64_extra_stop_shares_eos_contract\":true,"
      "\"synthetic_fixture_directory\":\"continuous-extra-stop-v64\",\"synthetic_role_payload_bytes\":"
      << std::filesystem::file_size(chain / "chain-weights.bin") << ',' <<
      "\"accepted_work_control_exception\":true,\"gpu_native_fault_injected\":false,"
      "\"model_quality_claim\":false,\"performance_claim\":false,\"candidates\":[";
  for (std::size_t at = 0; at < summaries.size(); ++at) {
    const auto &entry = summaries[at]; const auto &value = entry.result;
    constexpr std::array names{"normal", "credit_exhaustion", "member_cancel", "yield_resume", "extra_stop", "cancel_then_control_exception"};
    std::cout << (at ? "," : "") << "{\"requests\":" << entry.width << ",\"mode\":\"" << names[static_cast<std::size_t>(entry.mode)] << '"'
        << ",\"pumps\":" << value.pumps << ",\"actual_batches\":" << value.batches
        << ",\"pending_returns\":" << value.pending_returns << ",\"max_consecutive_pending\":" << value.consecutive_pending
        << ",\"submit_before_observe\":" << value.submit_before_observe << ",\"renewals\":" << value.renewals
        << ",\"credit_drains\":" << value.credit_drains << ",\"cancelled_requests\":" << value.cancellations
        << ",\"yield_drains\":" << value.yield_drains << ",\"pending_length27\":" << (value.pending_length27 ? "true" : "false")
        << ",\"extra_stop_outputs\":" << value.stop_outputs << ",\"extra_stop_accepted_successor\":" << (value.stop_successor_accepted ? "true" : "false")
        << ",\"discarded_batches\":" << value.discarded_batches << ",\"survivor_r1_steps\":" << value.survivor_r1
        << ",\"control_exceptions\":" << value.control_failures << ",\"aborted_actual_submissions\":" << value.aborted_submissions << '}';
  }
  std::cout << "]}\n"; return 0;
}
} // namespace qwen_continuous_generation_case
