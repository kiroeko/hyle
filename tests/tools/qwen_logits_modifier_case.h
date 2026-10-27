#pragma once

// The normal lifecycle --logits-modifiers mode and this round's thin CPU
// driver call this same real Model helper. One cached tiny fixture;
// no replacement executor, policy manager, framework or device fault injection.
namespace qwen_logits_modifier_case {
namespace q = qwen35;
namespace i = hyle::inference;

inline void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
inline void retire(q::GenerationRequest &request) {
  const auto observed = q::retire_generation_request(request);
  if (observed.retired) request.session.reset();
  q::check(observed.status);
  require(observed.retired, "modifier owner did not observe actual retirement");
}
template<class T>
q::LogitsInput ready(q::Model &model, q::Type type, std::span<const T> values) {
  const auto description = q::dense(type, {1, model.vocabulary});
  require(values.size_bytes() == q::take(description.required_bytes()), "modifier fixture extent differs");
  auto buffer = q::take(q::mem::Buffer::create(model.execution.device(), values.size_bytes(), 256));
  q::check(buffer.view().write(std::as_bytes(values)));
  // The Backend retains its actual StorageOwner, or its original process-
  // lifetime CPU system-resource domain; no fixture-only owner is introduced.
  return {description, q::take(model.execution.adopt_buffer(std::move(buffer)))};
}
inline q::LogitsInputs supplied(q::Model &model, std::int32_t allowed, std::int32_t bias = -1) {
  std::vector<float> offsets(static_cast<std::size_t>(model.vocabulary));
  std::vector<std::uint8_t> mask(offsets.size()); // Actual Bool carrier, never vector<bool>.
  if (allowed >= 0) mask.at(static_cast<std::size_t>(allowed)) = 1;
  if (bias >= 0) offsets.at(static_cast<std::size_t>(bias)) = 0x1p100F;
  return {ready<float>(model, q::Type::kFloat32, offsets),
      ready<std::uint8_t>(model, q::Type::kBool, mask)};
}
inline void prepare_request(q::GenerationRequest &request) {
  auto owner = std::make_shared<std::vector<std::int32_t>>(request.prompt.begin(), request.prompt.end());
  request.prompt = *owner; request.warm_inputs.push_back(std::move(owner));
  request.ledger = std::make_unique<q::GenerationLedger>(8, 1);
  request.round_emitted.resize(request.progress.maximum_outputs() - 1);
  for (auto &row : request.round_emitted) row.reserve(1);
}

// Same concrete Recipe/Graph mechanism, with explicit input rows to expose bad
// columns that a normal model fixture cannot independently control. This runs
// inside the one CPU process below, not a second model/backend/test matrix.
inline void modifier_domains(q::Model &model) {
  q::GraphAssembly assembly(model.execution, model.precision, model.allocation_budget);
  q::Recipe r(model.weights, model.projections, assembly, model.geometry, 1, 1, &model.weight_meanings);
  const auto raw_port = r.input(q::dense(q::Type::kFloat32, {1, model.vocabulary}));
  const auto input_ports = q::prepare_logits_inputs(r, {.bias = true, .allowed = true});
  const auto identity_port = r.input(q::dense(q::Type::kUInt32, {1, 4}));
  const auto greedy = q::sampling::ordinary_selection(r, raw_port, std::optional<q::V>{}, {}, input_ports);
  const auto random = q::sampling::ordinary_selection(r, raw_port, std::optional<q::V>{identity_port}, {}, input_ports);
  const std::array ports{greedy.id, r.host_predicate(*greedy.valid), random.id, r.host_predicate(*random.valid)};
  for (auto port : ports) r.publish(port);
  const auto graph = q::take(std::move(r.builder).finish());
  const std::array shape{q::g::ShapeBinding{r.last_row, 0}};
  const std::array variants{q::g::ExactShapeVariant{shape}};
  auto compiled = q::take(model.execution.compile(graph, assembly.constants,
      assembly.options({.shape_policy = q::g::CompileShapePolicy::kFiniteExact,
        .exact_variants = variants, .constants = assembly.constants, .resource_budget = model.execution_budget})));
  compiled.executable = q::take(compiled.executable.exact_variant(0));
  compiled.report = q::take(compiled.report.exact_variant(0));
  const auto queues = q::take(i::prepare_queue_bindings(model.execution.queue_bindings(), compiled.report));
  std::array<q::PreparedReadback, 4> outputs;
  for (std::size_t at = 0; at < outputs.size(); ++at)
    outputs[at] = q::take(q::PreparedReadback::prepare(compiled, ports[at], model.execution, model.allocation_budget));
  auto identity = model.caller_tensor(model.execution.device(), q::dense(q::Type::kUInt32, {1, 4}));
  const auto words = q::cache::SamplingCursor{123, 7, 0}.identity(); identity.write<std::uint32_t>(words);
  auto invocation = q::take(compiled.executable.create_invocation({.resources = q::g::ResourceOptions{}}));
  for (unsigned sample = 0; sample < 9; ++sample) {
    std::vector<float> scores(static_cast<std::size_t>(model.vocabulary));
    std::vector<float> bias(scores.size());
    std::vector<std::uint8_t> mask(scores.size()); mask[3] = 1;
    const auto nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
    if (sample == 1) scores.back() = nan;       // Invalid raw column is masked.
    if (sample == 2) scores.back() = inf;
    if (sample == 3) bias.back() = nan;         // Invalid bias column is masked.
    if (sample == 4) bias.back() = inf;
    if (sample == 5) std::fill(mask.begin(), mask.end(), std::uint8_t{0});
    if (sample == 6) bias.back() = -inf;
    if (sample == 7) std::fill(scores.begin(), scores.end(), -inf);
    if (sample == 8) scores.back() = -inf;      // Ordinary zero mass is legal.
    const auto raw = ready<float>(model, q::Type::kFloat32, scores);
    const q::LogitsInputs inputs{ready<float>(model, q::Type::kFloat32, bias),
        ready<std::uint8_t>(model, q::Type::kBool, mask)};
    q::check(invocation.reset());
    q::check(invocation.bind(raw_port, raw.description, raw.lease));
    q::check(invocation.bind(*input_ports.bias, inputs.bias.description, inputs.bias.lease));
    q::check(invocation.bind(*input_ports.allowed, inputs.allowed.description, inputs.allowed.lease));
    q::check(invocation.bind(identity_port, identity.read()));
    for (auto &output : outputs) q::check(output.bind(invocation));
    q::check(invocation.prepare_replay(i::queue_span(queues),
        {.mode = q::g::ReplayMode::kCommands, .host_wait = q::g::HostWaitPolicy::kAllow}));
    auto done = q::take(invocation.submit(i::queue_span(queues))); q::check(done.wait());
    const auto expected = std::uint8_t(sample == 0 || sample == 8);
    require(q::take(outputs[1].predicate(model.execution, invocation)) == expected &&
        q::take(outputs[3].predicate(model.execution, invocation)) == expected,
        "modifier validity hid an invalid column or rejected ordinary negative-infinite zero mass");
    if (expected) for (const auto port : {std::size_t{0}, std::size_t{2}}) {
      std::int32_t id = -1;
      q::check(outputs[port].read(model.execution, invocation, std::as_writable_bytes(std::span(&id, 1))));
      require(id == 3, "single allowed token was not selected by the original greedy/categorical endpoint");
    }
  }
}

inline int run(const std::filesystem::path &input) {
  const auto geometry = q::read_weight_geometry(input / "weights.tsv");
  require(geometry.hidden == 64 && geometry.intermediate == 128 && geometry.vocabulary == 64 && geometry.layers == "A",
      "logits modifiers require the existing untrained A/H64/FFN128/V64 CPU fixture");
  const q::cache::KvRowBytes rows{
      q::cache::multiply(q::cache::multiply(static_cast<std::size_t>(geometry.kv_heads), static_cast<std::size_t>(geometry.key_depth)), 2),
      q::cache::multiply(q::cache::multiply(static_cast<std::size_t>(geometry.kv_heads), static_cast<std::size_t>(geometry.value_depth)), 2)};
  const auto plan = q::cache::plan(geometry, rows, 16, 3, 2,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 2, 1, false, true);
  const std::array additional{
      q::SamplingConfiguration{false, {}, {.bias = true, .allowed = true}},
      q::SamplingConfiguration{true, {1.F, 0, 1.F}, {.bias = true, .allowed = true}}};
  q::Model model(input / "weights.tsv", geometry, 16, plan, {}, false,
      q::ReplayPolicy::kCommands, 32u << 20, {.enabled = false, .policy = {}, .seed = 123}, 0,
      {.primary = i::BackendKind::kCpu}, additional);
  modifier_domains(model);
  const std::array query_rows{std::size_t{1}, std::size_t{2}};
  for (std::size_t policy = 0; policy < 3; ++policy) {
    model.prepare_family(query_rows, false, 1, q::GraphRole::kText, false, q::TextCompletion::kEmitNext, policy);
    for (auto n : query_rows) model.plan_replay_uses(n, 1, 16, q::GraphRole::kText, q::TextCompletion::kEmitNext, policy);
  }
  model.prepare_family(query_rows, false, 1, q::GraphRole::kText, false, q::TextCompletion::kStateOnly, 1);
  for (auto n : query_rows) model.plan_replay_uses(n, 1, 16, q::GraphRole::kText, q::TextCompletion::kStateOnly, 1);
  const std::array two{std::size_t{2}};
  model.prepare_family(two, false, 2, q::GraphRole::kText, false, q::TextCompletion::kEmitNext, 1);
  model.plan_replay_uses(2, 2, 8, q::GraphRole::kText, q::TextCompletion::kEmitNext, 1);
  for (std::size_t policy = 0; policy < 3; ++policy)
    model.prepare_execution(query_rows, 1, q::GraphRole::kText, q::TextCompletion::kEmitNext, policy);
  model.prepare_execution(query_rows, 1, q::GraphRole::kText, q::TextCompletion::kStateOnly, 1);
  model.prepare_execution(two, 2, q::GraphRole::kText, q::TextCompletion::kEmitNext, 1);
  model.prepare_sampling_frontier();
  for (const auto &[key, family] : model.families) if (!key.sampling_index)
    require(family->logits_bias_ports.empty() && family->logits_allowed_ports.empty() && !family->sampling_valid_port,
        "default greedy acquired modifier ports or a validity readback");

  const std::array<std::int32_t, 2> prompt{1, 2};
  std::vector<q::GenerationRequest> requests; requests.reserve(4);
  requests.emplace_back("all-masked-A", prompt, std::filesystem::path{}, 1, 1);
  requests.emplace_back("same-key-B", prompt, std::filesystem::path{}, 1, 1);
  requests.emplace_back("dynamic-C", prompt, std::filesystem::path{}, 3, 1);
  for (auto &request : requests) prepare_request(request);
  auto &a = requests[0], &b = requests[1], &c = requests[2];
  a.logits = supplied(model, -1); b.logits = supplied(model, 4, 4);
  const auto prepare = [&](q::GenerationRequest &request, std::size_t quantum = SIZE_MAX,
                           q::cache::PrefixCache *prefix = nullptr) {
    return q::prepare_generation_call(model, request, request.progress.output_limit(), prefix, requests, 1, quantum);
  };
  const auto execute = [&](q::GenerationRequest &request, std::size_t quantum = SIZE_MAX) {
    std::array calls{prepare(request, quantum)}; q::execute_generation_calls(model, calls, false, nullptr, requests);
  };
  i::RequestWindow window; window.prepare(4, 2);
  const auto key_b = window.enqueue(0), key_a = window.enqueue(1);
  b.request_key = key_b; a.request_key = key_a;
  window.admitted(key_b); window.admitted(key_a);
  std::array first{prepare(a), prepare(b)};
  const auto *a_head = a.session->head.get(), *b_head = b.session->head.get();
  const auto b_before = q::generation_boundary(*b_head);
  bool failed = false;
  try { q::execute_generation_calls(model, first, false, nullptr, requests); }
  catch (const i::EvaluationFailure &failure) {
    failed = failure.participant() == 0 && failure.status().IsInvalidArgument() &&
        failure.retirement_observed() && failure.retirement().retired && failure.retirement().status.ok();
    std::cerr << "modifier_invalid_batch participant=" << (failure.participant() ? static_cast<long long>(*failure.participant()) : -1)
        << " status=" << failure.status().message() << " retirement_seen=" << failure.retirement_observed()
        << " retired=" << failure.retirement().retired << " cleanup=" << failure.retirement().status.message() << '\n';
  }
  std::cerr << "modifier_invalid_facts qualified=" << failed << " slot=" << key_a.slot
      << " same_A=" << (a.session->head.get() == a_head) << " same_B=" << (b.session->head.get() == b_head)
      << " B_boundary=" << (q::generation_boundary(*b.session->head) == b_before)
      << " outputs=" << a.progress.generated.size() << ',' << b.progress.generated.size()
      << " commits=" << a.ledger->committed_count << ',' << b.ledger->committed_count
      << " active=" << bool(model.active_step) << " carrier=" << model.carrier_borrowed << '\n';
  require(failed && key_a.slot != 0 && a.session->head.get() == a_head && b.session->head.get() == b_head &&
      q::generation_boundary(*b.session->head) == b_before &&
      a.progress.generated.empty() && b.progress.generated.empty() &&
      !a.ledger->committed_count && !b.ledger->committed_count && !model.active_step && !model.carrier_borrowed,
      "all-masked greedy published state/output/RNG or failed to retire the original R2 episode");
  window.retire_active(key_a); retire(a); window.release(key_a); window.requeue(key_b);
  require(window.matches(key_b) && b.request_key == key_b && window.waiting_front() == key_b,
      "qualified failure changed the survivor's key or FIFO membership");
  window.admitted(key_b); execute(b);
  require(b.progress.generated == std::vector<std::int32_t>{4} && b.session->head->sampling == q::cache::SamplingCursor{},
      "same-key survivor did not consume the original ready modifier input with zero draws");
  window.release_turn(key_b, false); retire(b); window.release(key_b);

  const auto key_c = window.enqueue(2); c.request_key = key_c; window.admitted(key_c);
  // First chunk is StateOnly: no modifier leases are supplied or bound.
  execute(c, 1);
  require(c.progress.generated.empty() && c.ledger->committed_count == 1 && c.session->head->next_id == -1 &&
      c.session->head->sampling == q::cache::SamplingCursor{}, "StateOnly consumed modifiers or RNG");
  // Bad rank must reject BEFORE any new request/ledger/Session state changes.
  c.logits = supplied(model, 3, 4);
  auto good = c.logits; c.logits.allowed.description = q::dense(q::Type::kBool, {model.vocabulary});
  const auto *partial = c.session->head.get(); const auto consumed = c.turn_input_consumed;
  bool rank_rejected = false;
  try { (void)prepare(c, 1); }
  catch (const i::RequestExecutionFailure &failure) {
    const auto &evaluation = failure.evaluation();
    rank_rejected = failure.request_key() == key_c && evaluation.status().IsInvalidArgument() &&
        !evaluation.status().has_native_error() && !evaluation.participant() && !evaluation.retirement_observed();
  }
  require(rank_rejected && c.session->head.get() == partial && c.turn_input_consumed == consumed &&
      c.ledger->committed_count == 1 && c.progress.generated.empty() && !model.active_step,
      "invalid typed modifier changed an admitted request before rejection");
  c.logits = std::move(good); execute(c, 1);
  require(c.progress.generated == std::vector<std::int32_t>{3}, "final prefill ignored the new mask or masked bias");
  c.logits = supplied(model, 5); execute(c);
  require(c.progress.generated == std::vector<std::int32_t>({3, 5}) && !c.progress.turn_done,
      "same cold policy did not accept a replacement next-step input lease");
  const auto boundary = q::generation_boundary(*c.session->head);
  const auto observed = c.progress.observed_commits();
  const auto emitted = c.progress.generated;
  c.last_idle = window.pause(key_c); retire(c);
  c.logits = supplied(model, 7);
  q::restore_generation_session(model, c, nullptr, false, requests);
  require(q::generation_boundary(*c.session->head) == boundary && c.progress.generated == emitted &&
      c.progress.observed_commits() == observed && c.restore_count == 1,
      "StateOnly reconstruction resampled historical outputs or used the newest mask");
  window.wake(key_c); window.admitted(key_c); execute(c);
  require(c.progress.generated == std::vector<std::int32_t>({3, 5, 7}) && c.progress.request_done() &&
      c.ledger->records[c.ledger->committed_count - 1].pending_input == 5 &&
      c.session->head->sampling == q::cache::SamplingCursor{},
      "restored decode lost its historical pending or consumed an extra draw");

  q::cache::PrefixCache prefix(1); q::publish_generation_completed(model, c, &prefix);
  const auto *physical = c.session->head->state_identity();
  const auto checkpoint = c.completed_checkpoint;
  window.release_turn(key_c, false); retire(c); window.release(key_c);
  requests.emplace_back("new-policy-full-hit", checkpoint.input->ids, std::filesystem::path{}, 1, 2);
  auto &d = requests[3]; prepare_request(d); d.input = checkpoint.input;
  d.progress.initial_sampling = {123, 19, 0}; d.logits = supplied(model, 6, 6);
  const auto key_d = window.enqueue(3); d.request_key = key_d; window.admitted(key_d);
  std::array hit{prepare(d, SIZE_MAX, &prefix)};
  require(hit[0].full_hit && d.prefix_full_hits == 1 && d.prefix_sampling_calls == 1,
      "new request did not use its own actual cached-frontier typed inputs");
  q::execute_generation_calls(model, hit, false, &prefix, requests);
  require(d.progress.generated == std::vector<std::int32_t>{6} && d.session->head->state_identity() == physical &&
      d.session->head->carry_hidden.valid() && d.session->head->sampling == d.progress.initial_sampling.advanced(1) &&
      d.ledger->records[d.ledger->committed_count - 1].after.sampling == d.session->head->sampling,
      "full hit inherited the publisher selection/cursor, copied state, or lost its one committed draw");
  window.release_turn(key_d, false); retire(d); window.release(key_d); prefix.evict_one();
  require(!model.snapshot_versions.live && !model.active_step && !model.carrier_borrowed &&
      window.active().empty() && !window.waiting_front(), "modifier case retained a final head/key/carrier");
  std::cout << "{\"status\":\"PASS\",\"backend\":\"cpu\",\"dynamic_inputs\":true,"
      "\"greedy_draws\":0,\"fullhit_random_draws\":1,\"allmasked_survivor\":true,"
      "\"state_only_without_inputs\":true,\"same_key_restore\":true,\"final_versions\":0,"
      "\"GPU\":false,\"HTTP\":false,\"quality_claim\":false}\n";
  return 0;
}
} // namespace qwen_logits_modifier_case
