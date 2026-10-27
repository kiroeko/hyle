#include "../../engine/models/qwen35/main.cc"
#include "../../src/graph/plan_state.h"

// A finite complete-Model owner/publication check. Source export does not
// change the original mathematical graph, and this probe does not pipeline it.
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
  const auto *state = g::detail::InvocationAccess::state(*episode.run);
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
    if (!decode_run) decode_run = step.run;
    require(step.run == decode_run && step.state_slot == 0 && session->pending == &step &&
        step.phase->state_sources.size() == expected_stores && step.nodes.size() >= expected_pages,
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
      require(!step.completion.valid() && sources->submission.valid() && sources->token.valid(),
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
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 1) { std::cerr << "SKIP: pass --cuda PREPARED_LOCAL_INPUTS\n"; return 77; }
    if (argc != 3 || std::string_view(argv[1]) != "--cuda")
      throw std::runtime_error("usage: qwen_submission_source_probe --cuda PREPARED_LOCAL_INPUTS");
    const auto compiled = q::Backend::compiled_backends();
    if (std::find(compiled.begin(), compiled.end(), q::BackendKind::kCuda) == compiled.end()) {
      std::cerr << "SKIP: CUDA route is not built\n"; return 77;
    }
    return run(argv[2]);
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
