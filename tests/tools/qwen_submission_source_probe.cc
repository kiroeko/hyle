#include "../../engine/models/qwen35/main.cc"
#include "../../src/graph/plan_state.h"
#include "hyle/compute/backends/cuda.h"

#include <atomic>
#include <condition_variable>
#include <mutex>

// A finite complete-Model owner/publication check. Source export does not
// change the original mathematical graph. The explicit --relay mode adds a
// finite three-carrier journey and one controlled pair of pending episodes.
namespace {
namespace q = qwen35;
namespace c = hyle::compute;
namespace g = hyle::graph;

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

struct StateBytes final {
  q::GenerationBoundary boundary, evaluated;
  std::vector<q::cache::HostPayloadDesc> layout;
  std::vector<std::vector<std::byte>> payloads;
  friend bool operator==(const StateBytes &, const StateBytes &) = default;
};

// Use the complete-state format's logical geometry, and read only its real
// bounded leases. Physical page IDs and pool padding are not an equality oracle.
StateBytes state_bytes(q::Model &model, const q::cache::State &state) {
  const auto &evaluated = *state.state_identity();
  const auto boundary = q::take(q::cache::host_image_boundary(state));
  StateBytes result{q::generation_boundary(state), q::generation_boundary(evaluated),
      q::take(q::cache::make_host_image_layout(model.host_image_schema(), boundary)), {}};
  result.payloads.reserve(result.layout.size());
  for (const auto &part : result.layout) {
    c::ReadLease lease;
    switch (part.role) {
    case q::cache::HostPayloadRole::kTargetKey:
    case q::cache::HostPayloadRole::kTargetValue: {
      const auto found = std::find(model.attention_layer_ids.begin(), model.attention_layer_ids.end(), part.layer);
      require(found != model.attention_layer_ids.end() && part.page < evaluated.pages.size(),
          "logical snapshot names an absent attention layer/page");
      const auto bank = static_cast<std::size_t>(found - model.attention_layer_ids.begin());
      const auto &frame = part.role == q::cache::HostPayloadRole::kTargetKey
          ? evaluated.keys[bank] : evaluated.values[bank];
      lease = q::take(q::take(frame.view()).page_prefix(evaluated.pages[part.page].page()));
      break;
    }
    case q::cache::HostPayloadRole::kConvHistory:
      require(part.layer < evaluated.state.size(), "logical snapshot names an absent history layer");
      lease = evaluated.state[part.layer].history;
      break;
    case q::cache::HostPayloadRole::kGdnState:
      require(part.layer < evaluated.state.size(), "logical snapshot names an absent recurrent layer");
      lease = evaluated.state[part.layer].recurrent;
      break;
    default:
      throw std::runtime_error("ordinary greedy snapshot unexpectedly contains draft/hidden state");
    }
    require(lease.valid() && lease.size() == part.bytes,
        "logical snapshot differs from the actual initialized lease extent");
    result.payloads.emplace_back(part.bytes);
    q::check(model.execution.read(q::take(lease.view()), result.payloads.back()));
  }
  return result;
}

// Read-only diagnostics of the exact original slot. This neither records a
// range nor changes the provider/runtime qualification used by Source.create.
void diagnose_replay(const q::StepState &episode, const char *point) {
  const auto *state = g::detail::InvocationAccess::state(*episode.execution.run);
  if (!state || !state->plan) {
    std::cerr << "source_replay_diagnostic point=" << point << " missing_state=true\n";
    return;
  }
  const auto &plan = *state->plan;
  const auto &report = state->replay_report;
  std::cerr << "source_replay_diagnostic point=" << point
      << " route=" << static_cast<unsigned>(state->replay_route)
      << " report_route=" << static_cast<unsigned>(report.route)
      << " classified=" << static_cast<unsigned>(g::detail::submitted_shared_replay_route(*state))
      << " replay_prepared=" << state->replay_prepared << " progress=" << static_cast<bool>(state->progress)
      << " steps=" << plan.steps.size() << " commands=" << plan.command_count
      << " segments=" << plan.segments.size() << " required_queues=" << plan.required_queue_count
      << " immediate_transfers=" << report.immediate_transfer_count
      << " host_wait_dependencies=" << report.host_wait_dependency_count
      << " recorded=" << report.recorded_capture_range_count << " reused=" << report.reused_capture_range_count
      << " candidate_ranges=" << plan.capture_ranges.size() << " retained_ranges=" << state->captured_ranges.size()
      << " spans=" << state->capture_spans.size() << '\n';
  for (std::size_t i = 0; i < plan.segments.size(); ++i) {
    const auto &segment = plan.segments[i];
    std::cerr << "source_replay_segment index=" << i << " lane=" << segment.lane.ordinal()
        << " begin=" << segment.step_begin << " count=" << segment.step_count << '\n';
  }
  for (std::size_t i = 0; i < plan.capture_ranges.size(); ++i) {
    const auto &range = plan.capture_ranges[i];
    std::cerr << "source_replay_candidate index=" << i << " begin=" << range.step_begin
        << " count=" << range.step_count << '\n';
  }
  for (std::size_t i = 0; i < state->capture_spans.size(); ++i) {
    const auto &span = state->capture_spans[i];
    std::cerr << "source_replay_span index=" << i << " begin=" << span.begin << " count=" << span.count << '\n';
    if (span.begin > state->captured_ranges.size() || span.count > state->captured_ranges.size() - span.begin)
      continue;
    for (std::size_t j = span.begin; j < span.begin + span.count; ++j) {
      const auto &range = state->captured_ranges[j];
      std::cerr << "source_replay_actual index=" << j << " begin=" << range.step_begin
          << " count=" << range.step_count << " valid=" << range.commands.valid() << '\n';
    }
  }
  const auto captured = [&](std::size_t step) {
    if (state->replay_route != g::ReplayRoute::kCaptured && state->replay_route != g::ReplayRoute::kHybrid)
      return false;
    for (const auto &span : state->capture_spans) {
      if (span.begin > state->captured_ranges.size() || span.count > state->captured_ranges.size() - span.begin)
        continue;
      for (std::size_t j = span.begin; j < span.begin + span.count; ++j) {
        const auto &range = state->captured_ranges[j];
        if (step >= range.step_begin && step - range.step_begin < range.step_count && range.commands.valid())
          return true;
      }
    }
    return false;
  };
  std::size_t bad_gaps = 0;
  for (std::size_t i = 0; i < plan.steps.size(); ++i) {
    const auto &step = plan.steps[i];
    if (!step.emits_command() || captured(i)) continue;
    const auto runtime = step.runtime_properties();
    if (runtime.mode == c::ProviderRuntimeMode::kSharedQueue &&
        runtime.replay_class == c::ProviderReplayClass::kPrepared && !runtime.blocks_host) continue;
    ++bad_gaps;
    if (bad_gaps > 64) continue;
    std::cerr << "source_replay_bad_gap step=" << i << " kind=" << static_cast<unsigned>(step.kind)
        << " provider=" << step.provider().name << " mode=" << static_cast<unsigned>(runtime.mode)
        << " replay_class=" << static_cast<unsigned>(runtime.replay_class)
        << " blocks_host=" << runtime.blocks_host << " capture_compatible=" << step.capture_compatible()
        << " inputs=" << step.input_count << " outputs=" << step.output_count << " nodes=";
    for (std::size_t j = 0; j < step.node_count; ++j) {
      const auto node = episode.phase->child->compiled.report.step_node(i, j);
      if (!node) { std::cerr << " unavailable"; continue; }
      std::cerr << ' ' << node->ordinal() << ':';
      const auto kind = episode.phase->family->graph.operation_kind(*node);
      if (!kind) { std::cerr << "unavailable"; continue; }
      if (*kind == c::OperationKind::kView) std::cerr << c::operation_name(*kind);
      else {
        const auto operation = episode.phase->family->graph.operation(*node);
        if (operation) std::cerr << c::operation_name(*operation);
        else std::cerr << c::operation_name(*kind);
      }
    }
    std::cerr << '\n';
  }
  std::cerr << "source_replay_bad_gap_count=" << bad_gaps << " printed=" << (std::min)(bad_gaps, std::size_t{64}) << '\n';
}

struct Arm final {
  std::array<std::int32_t, 4> ids{};
  std::array<std::uint64_t, 4> epochs{};
  std::array<StateBytes, 4> states;
  std::array<g::ReplayRoute, 3> routes{};
  std::size_t stores = 0, page_banks = 0, recorded = 0, reused = 0;
};

Arm run_arm(q::Model &model, bool exported) {
  require(!model.active_step && !model.carrier_borrowed && !model.snapshot_versions.live,
      "previous arm retained an episode or committed model version");
  const std::array calls{std::size_t{3}, std::size_t{1}, std::size_t{1}, std::size_t{1}};
  auto session = model.create_session(calls);
  const auto original_cursor = session->head->sampling;
  const std::array<std::int32_t, 3> prompt{1, 2, 3};
  hyle::inference::StepObservation observation;
  Arm result;
  result.ids[0] = q::evaluate(model, *session, prompt, "source-probe-prefill", {}, false,
      nullptr, nullptr, &observation);
  result.epochs[0] = session->head.epoch();
  result.states[0] = state_bytes(model, *session->head);
  auto checkpoint = session->checkpoint();
  const auto *checkpoint_identity = checkpoint->state_identity();
  const auto old = state_bytes(model, *checkpoint);
  require(checkpoint->length == prompt.size() && checkpoint->sampling == original_cursor,
      "prefill checkpoint advanced the greedy RNG cursor or wrong logical length");
  const auto expected_stores = model.geometry.recurrent_layers() * 2;
  const auto expected_pages = model.geometry.attention_layers() * 2;
  require(expected_stores && expected_pages,
      "source probe requires actual recurrent and full-attention layers");
  g::Invocation *decode_run = nullptr;
  for (std::size_t round = 0; round < result.routes.size(); ++round) {
    const auto before = q::generation_boundary(*session->head);
    const auto epoch = session->head.epoch();
    const auto *committed = session->head.get();
    const auto input = session->head->next_id;
    require(input == result.ids[round] && input >= 0 && input < model.vocabulary,
        "decode did not use the original committed pending token");
    model.prepare_evaluation(*session, std::span(&input, 1), nullptr);
    auto &step = *model.active_step;
    if (!decode_run) decode_run = step.execution.run;
    require(step.execution.run == decode_run && step.state_slot == 0 && session->pending == &step &&
        step.phase->state_sources.size() == expected_stores && step.execution.nodes.size() >= expected_pages,
        "ordinary decode changed the original slot or omitted a mathematical state bank");
    std::shared_ptr<q::StepSources> sources;
    std::weak_ptr<q::StepSources> source_lifetime;
    if (exported) {
      sources = model.prepare_submission_sources();
      source_lifetime = sources;
      require(sources && sources->states.size() == expected_stores && sources->pages.size() == expected_pages &&
          !sources->submission.valid() && !sources->token.valid(),
          "cold Source preparation omitted a state bank or accepted native work");
      for (const auto &source : sources->states) require(source.prepared(), "Store Source metadata was not prepared");
      for (const auto &source : sources->pages) require(source.prepared(), "KV Source metadata was not prepared");
    }
    require(session->head.get() == committed && session->head.epoch() == epoch &&
        q::generation_boundary(*session->head) == before,
        "episode/Source preparation published a candidate boundary or RNG cursor");
    if (exported && round == 0) diagnose_replay(step, "before-submit");
    try { model.submit_evaluation(*session); }
    catch (...) {
      if (exported && round == 0) diagnose_replay(step, "submit-failure");
      throw;
    }
    if (exported) {
      require(!step.execution.completion.valid() && sources->submission.valid() && sources->token.valid(),
          "exported episode retained a second original Completion or omitted its token Source");
      for (const auto &source : sources->states)
        require(source.valid() && source.destination_count() == 1, "Store Source did not capture the actual K1 destination");
      for (const auto &source : sources->pages) require(source.valid(), "KV Source did not capture its actual append");
      result.stores += sources->states.size();
      result.page_banks += sources->pages.size();
    }
    require(session->head.get() == committed && session->head.epoch() == epoch &&
        q::generation_boundary(*session->head) == before,
        "native submission published an unobserved candidate boundary or RNG cursor");
    result.routes[round] = step.replay_route;
    result.recorded += step.capture_recorded;
    result.reused += step.capture_reused;
    observation = {};
    result.ids[round + 1] = q::finish_evaluation(model, *session, "source-probe-decode", {}, false,
        nullptr, nullptr, &observation);
    require(!model.active_step && !model.carrier_borrowed && !session->pending &&
        session->head.epoch() == epoch + 1 && session->length() == before.length + 1 &&
        session->head->next_id == result.ids[round + 1] && session->head->sampling == original_cursor,
        "successful finish did not publish exactly one complete greedy boundary");
    if (exported) {
      q::check(sources->submission.wait());
      sources.reset();
      require(source_lifetime.expired(), "finished StepSources owner remained pinned before slot reuse");
    }
    result.epochs[round + 1] = session->head.epoch();
    result.states[round + 1] = state_bytes(model, *session->head);
    require(checkpoint->state_identity() == checkpoint_identity && state_bytes(model, *checkpoint) == old,
        "decode/Source publication changed the old immutable complete checkpoint");
  }
  checkpoint.reset();
  q::check(session->stop());
  const auto drained = session->drain(); q::check(drained.status);
  const auto retired = session->try_retire(); q::check(retired.status);
  require(retired.retired, "finite source arm did not actually retire its Session");
  session.reset();
  require(!model.active_step && !model.carrier_borrowed && !model.snapshot_versions.live,
      "finite source arm leaked an episode, ticket or committed version");
  return result;
}

const char *route_name(g::ReplayRoute route) noexcept {
  switch (route) {
  case g::ReplayRoute::kCommands: return "commands";
  case g::ReplayRoute::kCaptured: return "captured";
  case g::ReplayRoute::kHybrid: return "hybrid";
  default: return "unprepared";
  }
}

int run(const std::filesystem::path &input) {
  auto source = q::prepare_weight_source(input / "weights.tsv", false);
  const auto geometry = source.roles.geometry;
  require(geometry.context_limit >= 16 && geometry.vocabulary > 3,
      "source probe requires a prepared fixture with C16 and four tokens");
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 16, 3, 3,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 1, 1, false, false);
  q::Model model(std::move(source), geometry, 16, plan, {}, false, q::ReplayPolicy::kPreferCapture,
      SIZE_MAX, {}, 0, {.primary = q::BackendKind::kCuda});
  const std::array shapes{std::size_t{3}, std::size_t{1}};
  model.prepare_family(shapes);
  model.plan_replay_uses(3, 1, 2);
  model.plan_replay_uses(1, 1, 6);
  model.freeze_shape_plan();
  model.prepare_execution(shapes);
  const auto serial = run_arm(model, false);
  const auto exported = run_arm(model, true);
  require(serial.ids == exported.ids && serial.epochs == exported.epochs && serial.states == exported.states,
      "Source-aware finish changed token IDs, epochs, RNG, logical layout or complete initialized state bytes");
  require(exported.stores == geometry.recurrent_layers() * 2 * 3 &&
      exported.page_banks == geometry.attention_layers() * 2 * 3,
      "finite source arm did not export every state/KV bank in all three decodes");
  std::size_t compared_bytes = 0;
  for (const auto &state : serial.states) for (const auto &payload : state.payloads)
    compared_bytes = q::cache::add(compared_bytes, payload.size());
  std::cout << "{\"status\":\"PASS\",\"backend\":\"cuda\",\"sampling\":\"greedy\",\"prefill_tokens\":3,"
      "\"decode_steps_per_arm\":3,\"source_token_views\":3,\"source_store_views\":" << exported.stores
      << ",\"source_kv_views\":" << exported.page_banks << ",\"complete_state_bytes_compared\":" << compared_bytes
      << ",\"old_checkpoint_unchanged\":true,\"rng_cursor_unchanged\":true,\"sources_released_before_slot_reuse\":true,"
      "\"final_live_versions\":0,\"performance_claim\":false,\"routes\":[";
  for (std::size_t round = 0; round < exported.routes.size(); ++round) {
    if (round) std::cout << ',';
    std::cout << '"' << route_name(exported.routes[round]) << '"';
  }
  std::cout << "],\"capture_recorded\":" << exported.recorded << ",\"capture_reused\":" << exported.reused << ",\"ids\":[";
  for (std::size_t row = 0; row < exported.ids.size(); ++row) {
    if (row) std::cout << ',';
    std::cout << exported.ids[row];
  }
  std::cout << "]}\n";
  return 0;
}

constexpr std::size_t relay_decode_steps = 10;
constexpr std::size_t relay_ready_steps = 6;
struct RelayArm final {
  std::array<std::int32_t, relay_decode_steps + 1> ids{};
  std::array<std::uint64_t, relay_decode_steps + 1> epochs{};
  std::array<StateBytes, relay_decode_steps + 1> states;
  std::array<std::size_t, relay_decode_steps> slots{};
  std::array<g::ReplayRoute, relay_decode_steps> routes{};
  std::size_t recorded = 0, reused = 0;
  bool pending_prepare = false, pending_submit = false, publish_before_successor_retirement = false;
  bool rejected_successor_preserved_predecessor = false;
};

// A real callback on the original compute stream, never on the metadata
// transfer queue. A finite watchdog opens it after an unexpected blocking call
// so a failed qualification can still retire its accepted original work.
struct RelayGate final {
  std::atomic<bool> released{false}, timed_out{false};
  std::mutex mutex;
  std::condition_variable changed;
  std::thread watchdog;

  static void CUDART_CB wait(void *opaque) {
    static_cast<RelayGate *>(opaque)->released.wait(false, std::memory_order_acquire);
  }
  void enqueue(cudaStream_t stream) {
    require(!watchdog.joinable(), "a relay gate cannot be reused");
    watchdog = std::thread([this] {
      std::unique_lock lock(mutex);
      if (!changed.wait_for(lock, std::chrono::seconds(30),
              [this] { return released.load(std::memory_order_acquire); })) {
        timed_out.store(true, std::memory_order_release); release();
      }
    });
    q::check(c::cuda::status(cudaLaunchHostFunc(stream, &RelayGate::wait, this)));
  }
  void release() noexcept {
    released.store(true, std::memory_order_release);
    released.notify_all(); changed.notify_all();
  }
  ~RelayGate() {
    release();
    if (watchdog.joinable()) watchdog.join();
  }
};

RelayArm run_relay_arm(q::Model &model, bool relay, bool reject_successor = false) {
  require(!model.active_step && !model.successor_step && !model.carrier_borrowed && !model.snapshot_versions.live,
      "previous relay arm retained an original episode or model version");
  std::array<std::size_t, relay_decode_steps + 1> calls;
  calls.fill(1); calls[0] = 3;
  auto session = model.create_session(calls);
  q::cache::SnapshotRef checkpoint;
  std::shared_ptr<q::StepSources> previous, a_sources, b_sources;
  std::array<std::weak_ptr<q::StepSources>, relay_decode_steps + 1> lifetimes;
  RelayGate predecessor_gate, successor_gate;
  // On every exceptional exit open both callbacks before waiting. The Model
  // keeps the original graph/candidate owners; release external Source pins
  // only after their accepted work has retired, then use its normal abort.
  hyle::util::ScopeGuard cleanup{[&]() noexcept {
    predecessor_gate.release(); successor_gate.release();
    if (model.successor_step && model.successor_step->execution.has_completion()) {
      model.successor_step->execution.wait().IgnoreError();
      if (model.successor_step->execution.run) model.execution.drain_retired(*model.successor_step->execution.run).IgnoreError();
    }
    if (model.active_step && model.active_step->execution.has_completion()) {
      model.active_step->execution.wait().IgnoreError();
      if (model.active_step->execution.run) model.execution.drain_retired(*model.active_step->execution.run).IgnoreError();
    }
    b_sources.reset(); a_sources.reset(); previous.reset();
    (void)model.abort_episode(); checkpoint.reset();
    if (session) {
      session->stop().IgnoreError(); (void)session->drain(); (void)session->try_retire();
    }
  }};

  RelayArm result;
  const auto original_cursor = session->head->sampling;
  const std::array<std::int32_t, 3> prompt{1, 2, 3};
  model.prepare_evaluation(*session, prompt);
  if (relay) {
    previous = model.prepare_submission_sources(); lifetimes[0] = previous;
  }
  model.submit_evaluation(*session);
  result.ids[0] = q::finish_evaluation(model, *session, "relay-probe-prefill", {}, false);
  result.epochs[0] = session->head.epoch();
  result.states[0] = state_bytes(model, *session->head);
  checkpoint = session->checkpoint();
  const auto *checkpoint_identity = checkpoint->state_identity();
  const auto old = state_bytes(model, *checkpoint);
  const auto expected_stores = model.geometry.recurrent_layers() * 2;
  const auto expected_pages = model.geometry.attention_layers() * 2;
  require(expected_stores && expected_pages && checkpoint->sampling == original_cursor,
      "relay fixture requires real recurrent/full-attention state and unchanged greedy RNG");

  const auto check_after_finish = [&](std::size_t round, std::int32_t id, bool successor_pending) {
    require(session->head.epoch() == result.epochs[round] + 1 &&
        session->length() == prompt.size() + round + 1 && session->head->next_id == id &&
        session->head->sampling == original_cursor && !model.successor_step &&
        bool(model.active_step) == successor_pending && bool(session->pending) == successor_pending &&
        model.carrier_borrowed == (successor_pending ? 1u : 0u),
        "relay finish changed the greedy boundary or lost the original successor handoff");
    if (successor_pending)
      require(session->pending == model.active_step.get() &&
          model.active_step->members[0].base == session->head.get() &&
          model.active_step->members[0].base_epoch == session->head.epoch() &&
          model.active_step->members[0].first_input == id,
          "successor handoff lost its exact committed base or actual predecessor token");
    result.ids[round + 1] = id;
    result.epochs[round + 1] = session->head.epoch();
    result.states[round + 1] = state_bytes(model, *session->head);
    require(checkpoint->state_identity() == checkpoint_identity && state_bytes(model, *checkpoint) == old,
        "relay changed the retained prefill checkpoint or its initialized payloads");
  };
  const auto record_episode = [&](std::size_t round, const q::StepState &step, bool pending_input) {
    require(step.phase && step.state_slot == (relay ? round % 3 : 0) &&
        step.phase->state_sources.size() == expected_stores && step.execution.nodes.size() >= expected_pages,
        "relay episode omitted a state bank or selected another physical carrier");
    if (relay) {
      require(step.execution.incoming && step.execution.incoming->bound && step.execution.incoming->owner &&
          step.execution.incoming->bindings.size() == 1 + expected_stores + expected_pages,
          "relay omitted its complete original token/Store/KV effect bundle");
      const auto sources = q::take(step.execution.run->input_source_report());
      require(sources.source_count == 1 && sources.device_wait_count == 1 && !sources.host_wait_count,
          "relay did not deduplicate the complete original submission dependency");
    }
    if (pending_input)
      require(step.capture_recorded == 0 && step.capture_reused &&
          (step.replay_route == g::ReplayRoute::kCaptured || step.replay_route == g::ReplayRoute::kHybrid),
          "pending model input recorded a new capture or lost its real warmed tuple");
    result.slots[round] = step.state_slot; result.routes[round] = step.replay_route;
    result.recorded += step.capture_recorded; result.reused += step.capture_reused;
  };

  for (std::size_t round = 0; round < relay_decode_steps; ++round) {
    if (relay && round == relay_ready_steps) {
      require(previous && q::take(previous->submission.poll()) != c::CompletionPoll::kPending,
          "pending pair must begin with an actually retired ready predecessor");
      const auto base = q::generation_boundary(*session->head);
      const auto epoch = session->head.epoch();
      const auto *committed = session->head.get();
      const auto prior_lifetime = std::weak_ptr(previous);
      const std::array requests{q::EvaluationRequest{session, {}}};
      auto &a = model.prepare_relay_evaluations(requests, previous, round % 3);
      std::cerr << "relay-stage A-prepared\n";
      a_sources = model.prepare_submission_sources(a); lifetimes[round + 1] = a_sources;
      record_episode(round, a, false);
      const auto queues = a.phase->slots[a.state_slot].carrier.queues();
      require(queues.size() == 1 && queues[0].queue, "relay gate requires its original single compute Queue");
      const auto stream = q::take(c::cuda::stream(*queues[0].queue));
      predecessor_gate.enqueue(stream);
      model.submit_evaluations(a);
      std::cerr << "relay-stage A-submitted\n";
      require(q::take(a_sources->submission.poll()) == c::CompletionPoll::kPending &&
          session->head.get() == committed && session->head.epoch() == epoch &&
          q::generation_boundary(*session->head) == base,
          "controlled predecessor completed or published before successor preparation");

      bool rejected_live_carrier = false;
      try { (void)model.prepare_relay_evaluations(requests, a_sources, round % 3); }
      catch (const std::runtime_error &) { rejected_live_carrier = true; }
      require(rejected_live_carrier && !model.successor_step &&
          model.active_step.get() == &a && model.carrier_borrowed == 1 &&
          session->pending == &a && !predecessor_gate.released.load(std::memory_order_acquire) &&
          q::take(a_sources->submission.poll()) == c::CompletionPoll::kPending,
          "rejecting a live predecessor carrier damaged A or waited for its retirement");

      if (reject_successor) {
        bool rejected = false;
        // Slot 2 was warmed only from slot 1. A currently owns slot 0; this
        // exact token/source tuple has never been recorded for slot 2.
        try { (void)model.prepare_relay_evaluations(requests, a_sources, (round + 2) % 3); }
        catch (const std::runtime_error &) { rejected = true; }
        require(rejected && model.successor_step && model.successor_step->execution.cleanup_pending &&
            !model.successor_step->submitted && model.carrier_borrowed == 2 &&
            model.active_step.get() == &a && session->pending == &a &&
            !predecessor_gate.released.load(std::memory_order_acquire) &&
            q::take(a_sources->submission.poll()) == c::CompletionPoll::kPending,
            "a rejected cold tuple waited for A or returned its still-owned candidate early");
        predecessor_gate.release();
        const auto id = q::finish_evaluation(model, *session, "relay-probe-rejected-B", {}, false);
        check_after_finish(round, id, false);
        previous.reset(); a_sources.reset();
        require(prior_lifetime.expired() && !model.successor_step &&
            !predecessor_gate.timed_out.load(std::memory_order_acquire),
            "rejected successor was promoted or retained after original retirement");
        result.rejected_successor_preserved_predecessor = true;
        relay = false; // Remaining real steps use the committed host token.
        continue;
      }

      auto &b = model.prepare_relay_evaluations(requests, a_sources, (round + 1) % 3);
      std::cerr << "relay-stage B-prepared\n";
      b_sources = model.prepare_submission_sources(b); lifetimes[round + 2] = b_sources;
      record_episode(round + 1, b, true);
      result.pending_prepare = q::take(a_sources->submission.poll()) == c::CompletionPoll::kPending;
      require(result.pending_prepare && !predecessor_gate.released.load(std::memory_order_acquire) &&
          model.active_step.get() == &a && model.successor_step.get() == &b && model.carrier_borrowed == 2,
          "successor preparation waited for A or changed its committed publication boundary");
      successor_gate.enqueue(stream);
      model.submit_evaluations(b);
      std::cerr << "relay-stage B-submitted\n";
      result.pending_submit = q::take(a_sources->submission.poll()) == c::CompletionPoll::kPending;
      require(result.pending_submit && q::take(b_sources->submission.poll()) == c::CompletionPoll::kPending &&
          session->head.get() == committed && session->head.epoch() == epoch &&
          q::generation_boundary(*session->head) == base,
          "successor submission observed or published an unknown device-token boundary");

      predecessor_gate.release();
      const auto a_id = q::finish_evaluation(model, *session, "relay-probe-pending-A", {}, false);
      result.publish_before_successor_retirement =
          q::take(b_sources->submission.poll()) == c::CompletionPoll::kPending &&
          !successor_gate.released.load(std::memory_order_acquire);
      require(result.publish_before_successor_retirement,
          "A publication waited for its controlled successor's native retirement");
      check_after_finish(round, a_id, true);
      previous.reset();
      require(prior_lifetime.expired(), "A's retired incoming Source remained pinned after exact bundle release");
      const auto a_lifetime = std::weak_ptr(a_sources);
      a_sources.reset(); // B's own incoming bundle still keeps this source alive.
      require(!a_lifetime.expired(), "A source vanished while B still reads its original output/state");
      successor_gate.release();
      const auto b_id = q::finish_evaluation(model, *session, "relay-probe-pending-B", {}, false);
      check_after_finish(round + 1, b_id, false);
      require(a_lifetime.expired(), "retired B retained A's Source before physical carrier reuse");
      previous = std::move(b_sources);
      require(!predecessor_gate.timed_out.load(std::memory_order_acquire) &&
          !successor_gate.timed_out.load(std::memory_order_acquire),
          "watchdog released a blocked relay operation; pending qualification did not pass");
      ++round;
      continue;
    }

    const auto before = q::generation_boundary(*session->head);
    const auto epoch = session->head.epoch();
    const auto *committed = session->head.get();
    const auto prior_lifetime = std::weak_ptr(previous);
    std::shared_ptr<q::StepSources> outgoing;
    if (relay) {
      require(previous && q::take(previous->submission.poll()) != c::CompletionPoll::kPending,
          "first tuple capture requires a genuine successful ready predecessor");
      const std::array requests{q::EvaluationRequest{session, {}}};
      auto &step = model.prepare_relay_evaluations(requests, previous, round % 3);
      outgoing = model.prepare_submission_sources(step); lifetimes[round + 1] = outgoing;
      record_episode(round, step, false);
      model.submit_evaluations(step);
    } else {
      const auto pending = session->head->next_id;
      require(pending == result.ids[round], "serial decode lost its committed pending token");
      model.prepare_evaluation(*session, std::span(&pending, 1));
      record_episode(round, *model.active_step, false);
      model.submit_evaluations();
    }
    require(session->head.get() == committed && session->head.epoch() == epoch &&
        q::generation_boundary(*session->head) == before,
        "model preparation/submission published an unobserved decode candidate");
    const auto id = q::finish_evaluation(model, *session, "relay-probe-ready-decode", {}, false);
    check_after_finish(round, id, false);
    if (relay) {
      previous.reset();
      require(prior_lifetime.expired(), "ready successor retained a consumed host Source owner");
      previous = std::move(outgoing);
    }
  }

  previous.reset();
  for (const auto &lifetime : lifetimes) require(lifetime.expired(), "relay left a previous source generation retained");
  checkpoint.reset(); q::check(session->stop());
  const auto drained = session->drain(); q::check(drained.status);
  const auto retired = session->try_retire(); q::check(retired.status);
  require(retired.retired, "relay arm did not actually retire its original Session");
  session.reset();
  require(!model.active_step && !model.successor_step && !model.carrier_borrowed && !model.snapshot_versions.live,
      "relay arm leaked a candidate version, original episode or carrier admission");
  cleanup.dismiss();
  return result;
}

int run_relay(const std::filesystem::path &input) {
  auto source = q::prepare_weight_source(input / "weights.tsv", false);
  const auto geometry = source.roles.geometry;
  const auto context = (std::min)(static_cast<std::size_t>(geometry.context_limit), std::size_t{32});
  require(context >= 3 + relay_decode_steps && geometry.vocabulary > 3,
      "relay probe requires enough actual context for its prompt and ten decodes, and four tokens");
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, context, 3, 3,
      q::recurrent_bytes(q::state_schema_for(geometry, context), geometry), 1, 1, false, false, 2);
  q::Model model(std::move(source), geometry, context, plan, {}, false, q::ReplayPolicy::kPreferCapture,
      SIZE_MAX, {}, 0, {.primary = q::BackendKind::kCuda});
  const std::array shapes{std::size_t{3}, std::size_t{1}};
  model.prepare_family(shapes);
  model.plan_replay_uses(3, 1, 2); model.plan_replay_uses(1, 1, relay_decode_steps * 2);
  model.freeze_shape_plan(); model.prepare_execution(shapes); model.prepare_relay_execution();
  const auto serial = run_relay_arm(model, false);
  const auto relay = run_relay_arm(model, true);
  const auto rejected = run_relay_arm(model, true, true);
  require(serial.ids == relay.ids && serial.epochs == relay.epochs && serial.states == relay.states,
      "relay changed generated IDs, complete initialized KV/recurrent state, epochs or greedy RNG");
  require(rejected.rejected_successor_preserved_predecessor && serial.ids == rejected.ids &&
      serial.epochs == rejected.epochs && serial.states == rejected.states,
      "rejected successor changed the predecessor or subsequent committed inference");
  require(relay.pending_prepare && relay.pending_submit && relay.publish_before_successor_retirement,
      "finite model relay did not prove its controlled pair of actual pending episodes");
  std::size_t compared_bytes = 0;
  for (const auto &state : serial.states) for (const auto &payload : state.payloads)
    compared_bytes = q::cache::add(compared_bytes, payload.size());
  std::cout << "{\"status\":\"PASS\",\"mode\":\"relay\",\"backend\":\"cuda\",\"sampling\":\"greedy\","
      "\"prefill_tokens\":3,\"decode_steps_per_arm\":" << relay_decode_steps
      << ",\"ready_decode_steps_before_pending_pair\":" << relay_ready_steps
      << ",\"physical_carriers\":3,\"candidate_depth\":2,\"controlled_pending_pairs\":1,"
      "\"successor_prepared_while_predecessor_pending\":true,\"successor_submitted_while_predecessor_pending\":true,"
      "\"predecessor_published_while_successor_pending\":true,\"pending_capture_recorded\":0,"
      "\"rejected_successor_preserved_predecessor\":true,"
      "\"complete_state_bytes_compared\":" << compared_bytes
      << ",\"old_checkpoint_unchanged\":true,\"rng_cursor_unchanged\":true,\"all_source_generations_released\":true,"
      "\"final_live_versions\":0,\"continuous_pipeline\":false,\"request_scheduler_coverage\":false,\"performance_claim\":false,"
      "\"capture_recorded\":" << relay.recorded << ",\"capture_reused\":" << relay.reused << ",\"slots\":[";
  for (std::size_t round = 0; round < relay.slots.size(); ++round)
    std::cout << (round ? "," : "") << relay.slots[round];
  std::cout << "],\"routes\":[";
  for (std::size_t round = 0; round < relay.routes.size(); ++round)
    std::cout << (round ? "," : "") << '"' << route_name(relay.routes[round]) << '"';
  std::cout << "],\"ids\":[";
  for (std::size_t index = 0; index < relay.ids.size(); ++index)
    std::cout << (index ? "," : "") << relay.ids[index];
  std::cout << "]}\n";
  return 0;
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 1) { std::cerr << "SKIP: pass --cuda PREPARED_LOCAL_INPUTS\n"; return 77; }
    const bool relay = argc == 4 && std::string_view(argv[1]) == "--relay" && std::string_view(argv[2]) == "--cuda";
    if (!relay && (argc != 3 || std::string_view(argv[1]) != "--cuda"))
      throw std::runtime_error("usage: qwen_submission_source_probe [--relay] --cuda PREPARED_LOCAL_INPUTS");
    const auto compiled = q::Backend::compiled_backends();
    if (std::find(compiled.begin(), compiled.end(), q::BackendKind::kCuda) == compiled.end()) {
      std::cerr << "SKIP: CUDA route is not built\n"; return 77;
    }
    return relay ? run_relay(argv[3]) : run(argv[2]);
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
