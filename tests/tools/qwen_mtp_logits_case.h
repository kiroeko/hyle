#pragma once

#include "../compute_models/sampling_fixture.h"

// Finite cached GA/MTP consumers. Modifier inputs are one admitted [1,V]
// snapshot per episode, not per-prefix grammar or retained replay payloads.
namespace qwen_mtp_logits_case {
namespace q = qwen35;
namespace i = hyle::inference;
namespace h = qwen_host_image_case;
namespace p = qwen_packed_prefill_case;
namespace c = qwen_mtp_chunked_prefill_case;
namespace s = qwen_mtp_batch_state_case;
namespace t = qwen_mtp_stop_case;
namespace u = qwen_mtp_sampling_case;
namespace e = qwen_logits_preflight_case;
using h::require;

template<class T>
inline q::LogitsInput ready(q::Model &model, q::Type type, std::span<const T> values) {
  const auto desc = q::dense(type, {1, model.vocabulary});
  require(values.size_bytes() == q::take(desc.required_bytes()), "modifier carrier has the wrong actual extent");
  auto buffer = q::take(q::mem::Buffer::create(model.execution.device(), values.size_bytes(), 256));
  q::check(model.execution.write(buffer.view(), std::as_bytes(values)));
  return {desc, q::take(model.execution.adopt_buffer(std::move(buffer)))};
}
struct Payload final {
  q::LogitsInputs inputs;
  std::vector<float> bias;
  std::vector<std::uint8_t> allowed;
};
inline Payload payload(q::Model &model, unsigned set, bool all_masked = false, bool nan_bias = false) {
  Payload result; result.bias.resize(static_cast<std::size_t>(model.vocabulary)); result.allowed.resize(result.bias.size());
  for (std::size_t at = 0; at != 5; ++at) {
    const auto token = 1 + set + 3 * at;
    if (!all_masked) result.allowed.at(token) = 1;
    result.bias.at(token) = static_cast<float>(at) * .375F - .5F;
  }
  if (nan_bias) result.bias.back() = (std::numeric_limits<float>::quiet_NaN)();
  result.inputs = {ready<float>(model, q::Type::kFloat32, result.bias),
      ready<std::uint8_t>(model, q::Type::kBool, result.allowed)};
  return result;
}
inline void allowed(std::span<const std::int32_t> ids, const Payload &value) {
  for (auto id : ids) require(id >= 0 && static_cast<std::size_t>(id) < value.allowed.size() && value.allowed[static_cast<std::size_t>(id)],
      "MTP emitted an ID outside the original episode's multi-token support");
}
inline void q_rows(q::Model &model, std::span<const q::c::ReadLease> rows, const Payload &value) {
  for (const auto &row : rows) {
    std::vector<float> actual(value.allowed.size());
    require(row.valid() && row.size() == actual.size() * sizeof(float), "MTP proposal lost its complete normalized q owner");
    q::check(model.execution.read(q::take(row.view()), std::as_writable_bytes(std::span(actual))));
    double mass = 0; std::size_t support = 0;
    for (std::size_t at = 0; at != actual.size(); ++at) {
      if (!value.allowed[at]) require(actual[at] == -(std::numeric_limits<float>::infinity)(), "q ignored the original mask snapshot");
      else { require(std::isfinite(actual[at]), "q lost a finite allowed column"); mass += std::exp(double(actual[at])); ++support; }
    }
    require(support == 5 && std::abs(mass - 1) < 0.00003, "modified q is not its actual multi-token normalized distribution");
  }
}

inline std::vector<double> normalized(std::span<const float> raw, std::span<const float> bias,
    std::span<const std::uint8_t> mask, float temperature) {
  std::vector<double> result(raw.size()); double mass = 0;
  for (std::size_t at = 0; at != raw.size(); ++at) if (mask[at]) {
    const float processed = raw[at] + bias[at]; result[at] = std::exp(double(processed) / temperature); mass += result[at];
  }
  require(mass > 0 && std::isfinite(mass), "independent modifier reference has no finite mass");
  for (auto &entry : result) entry /= mass;
  return result;
}
struct DrawReference final { std::int32_t selected = -1; double margin = 0; };

// Independent host equations, not calls to a provider's sampling helper.
// CUDA/Metal use F32 log1p; Vulkan's F32 upper half uses its eight-term
// small-tail polynomial or corrected log(1-small). None uses the CPU CDF.
inline float gumbel(std::uint32_t hi, std::uint32_t lo, q::BackendKind kind) {
  const bool upper = (hi & 0x80000000u) != 0;
  if (upper) { hi = ~hi; lo = ~lo; }
  const float high = static_cast<float>(hi) * 0x1p-32F;
  const float low = (static_cast<float>(lo) + .5F) * 0x1p-64F;
  const float small = high + low;
  float exponential;
  if (!upper) exponential = -std::log(small);
  else if (kind != q::BackendKind::kVulkan) exponential = -std::log1p(-small);
  else if (small < .125F) {
    // Evaluate the shader's positive terms in F32, without silently replacing
    // its tail recipe by a double-precision CUDA/Gumbel model.
    float series = .125F;
    for (const float coefficient : {1.F / 7.F, 1.F / 6.F, .2F, .25F, 1.F / 3.F, .5F, 1.F}) {
      const float product = small * series;
      series = coefficient + product;
    }
    exponential = small * series;
  } else {
    const float complement = 1.F - small;
    const float delta = complement - 1.F;
    const float correction = std::log(complement) / delta;
    exponential = small * correction;
  }
  return -std::log(exponential);
}

inline DrawReference draw_reference(std::span<const double> weights,
    sampling_test::Words identity, q::BackendKind kind) {
  require(kind == q::BackendKind::kCpu || kind == q::BackendKind::kCuda ||
      kind == q::BackendKind::kApple || kind == q::BackendKind::kVulkan,
      "categorical reference requires an explicit native backend");
  double mass = 0;
  float maximum = -(std::numeric_limits<float>::infinity)();
  for (const double value : weights) {
    require(value >= 0 && std::isfinite(value), "independent categorical has an invalid weight");
    mass += value;
    if (value > 0) maximum = (std::max)(maximum, static_cast<float>(std::log(value)));
  }
  require(mass > 0 && std::isfinite(mass) && std::isfinite(maximum),
      "independent categorical has no positive finite support");
  if (kind == q::BackendKind::kCpu) {
    const auto words = sampling_test::philox({0, 0, identity[2], identity[3]}, {identity[0], identity[1]});
    const double uniform = (double(words[0]) * 0x1p20 + double(words[1] >> 12) + .5) * 0x1p-52;
    const double threshold = uniform * mass;
    double cumulative = 0;
    for (std::size_t at = 0; at != weights.size(); ++at) {
      const double before = cumulative; cumulative += weights[at];
      if (cumulative > threshold)
        return {static_cast<std::int32_t>(at), (std::min)(threshold - before, cumulative - threshold) / mass};
    }
    throw std::runtime_error("independent CPU categorical has no selected support");
  }
  float best = -(std::numeric_limits<float>::infinity)(), runner = best;
  std::int32_t selected = -1;
  for (std::size_t at = 0; at != weights.size(); ++at) if (weights[at] > 0) {
    const auto words = sampling_test::philox({static_cast<std::uint32_t>(at / 2), 0, identity[2], identity[3]}, {identity[0], identity[1]});
    const float shifted = static_cast<float>(std::log(weights[at])) - maximum;
    const float score = shifted + gumbel(words[(at % 2) * 2], words[(at % 2) * 2 + 1], kind);
    require(std::isfinite(score), "independent native F32 Gumbel produced a nonfinite score");
    // Native reductions resolve equal scores in favour of the lower column.
    if (score > best) { runner = best; best = score; selected = static_cast<std::int32_t>(at); }
    else runner = (std::max)(runner, score);
  }
  require(selected >= 0, "independent native F32 Gumbel has no selected support");
  return {selected, double(best) - double(runner)};
}
inline std::int32_t draw(std::span<const double> weights, sampling_test::Words identity, q::BackendKind kind) {
  return draw_reference(weights, identity, kind).selected;
}
inline bool separated_draw(std::span<const double> weights, sampling_test::Words identity, q::BackendKind kind) {
  // Select addresses before executing the fixed corpus. These are exclusion
  // margins, not output tolerances: all device IDs must still match exactly.
  // The four-column corpus keeps >=1e-3 probability distance from CPU CDF
  // edges, or >=.05 log-score distance between GPU winner and runner-up.
  // This is well above the native endpoint gate (2e-5 absolute Gumbel error)
  // and the corpus's F32 normalization/rounding, including small residuals.
  return draw_reference(weights, identity, kind).margin >= (kind == q::BackendKind::kCpu ? .001 : .05);
}

inline void finite_math(q::Model &model, q::BackendKind kind) {
  constexpr float temperature = .8F;
  const auto width = static_cast<std::size_t>(model.vocabulary);
  std::array<std::vector<float>, 3> raw; for (auto &row : raw) row.assign(width, -40.F);
  const std::array<std::array<double, 4>, 3> weights{{{0, .25, .25, .5}, {.5, .125, .25, .125}, {.25, .5, .125, .125}}};
  std::vector<float> bias(width); std::vector<std::uint8_t> mask(width);
  const std::array offsets{.5F, -.25F, 1.F, .3F};
  for (std::size_t at = 0; at != 4; ++at) {
    bias[at] = offsets[at]; mask[at] = 1;
    for (std::size_t row = 0; row != raw.size(); ++row) raw[row][at] = static_cast<float>(std::log(weights[row][at]));
  }
  std::array<std::vector<double>, 3> expected;
  for (std::size_t row = 0; row != expected.size(); ++row) expected[row] = normalized(raw[row], bias, mask, temperature);
  std::vector<float> q_log(width, -(std::numeric_limits<float>::infinity)()), bonus_log(q_log);
  for (std::size_t at = 0; at != width; ++at) {
    if (expected[1][at]) q_log[at] = static_cast<float>(std::log(expected[1][at]));
    if (expected[2][at]) bonus_log[at] = static_cast<float>(std::log(expected[2][at]));
  }
  const auto double_q = normalized(q_log, bias, mask, temperature), double_bonus = normalized(bonus_log, bias, mask, temperature);
  std::vector<double> residual(width), wrong_residual(width);
  for (std::size_t at = 0; at != width; ++at) {
    residual[at] = (std::max)(0., expected[0][at] - expected[1][at]);
    wrong_residual[at] = (std::max)(0., expected[0][at] - double_q[at]);
  }
  std::array<sampling_test::Words, 2> finals{};
  for (std::size_t branch = 0; branch != 2; ++branch) {
    bool found = false;
    for (std::uint32_t address = 0; address != 128 && !found; ++address) {
      const sampling_test::Words key{0xb7e15162u, 0x8aed2a6bu, address, 0};
      if (separated_draw(branch ? expected[2] : residual, key, kind) &&
          separated_draw(branch ? double_bonus : wrong_residual, key, kind) &&
          draw(branch ? expected[2] : residual, key, kind) != draw(branch ? double_bonus : wrong_residual, key, kind)) {
        finals[branch] = key; found = true;
      }
    }
    require(found, "finite modifier corpus failed to distinguish repeat transformation of q or bonus");
  }
  q::GraphAssembly assembly(model.execution, model.precision, model.allocation_budget);
  q::Recipe r(model.weights, model.projections, assembly, model.geometry, 1, 1, &model.weight_meanings);
  const auto target = r.input(q::dense(q::Type::kFloat32, {2, model.vocabulary}));
  const auto draft = r.input(q::dense(q::Type::kFloat32, {1, model.vocabulary}));
  const auto identities = r.input(q::dense(q::Type::kUInt32, {2, 4}));
  const auto proposals = r.input(q::dense(q::Type::kInt32, {1}));
  const auto inputs = q::prepare_logits_inputs(r, {.bias = true, .allowed = true});
  const q::sampling::Policy policy{temperature, 0, 1};
  const auto prefix = q::sampling::speculative_prefix(r, target, identities, proposals, policy, inputs);
  std::array<q::V, 9> ports;
  for (std::size_t row = 0; row != 3; ++row) {
    const auto logits = row == 1 ? draft : r.copy(r.slice(target, 0, q::f(row == 0 ? 0 : 1), q::f(1)));
    const auto value = q::sampling::processed_distribution(r, q::sampling::process_logits(r, logits, inputs), policy);
    ports[row] = value.log_probabilities; ports[row + 3] = r.host_predicate(value.valid);
  }
  ports[6] = r.host_predicate(prefix.accepted[0]); ports[7] = r.host_predicate(prefix.valid[0]); ports[8] = prefix.final_ids[0];
  for (const auto port : ports) r.publish(port);
  const auto graph = q::take(std::move(r.builder).finish()); const std::array shape{q::g::ShapeBinding{r.last_row, 0}};
  const std::array variants{q::g::ExactShapeVariant{shape}};
  auto compiled = q::take(model.execution.compile(graph, assembly.constants,
      assembly.options({.shape_policy = q::g::CompileShapePolicy::kFiniteExact, .exact_variants = variants,
        .constants = assembly.constants, .resource_budget = model.execution_budget})));
  compiled.executable = q::take(compiled.executable.exact_variant(0)); compiled.report = q::take(compiled.report.exact_variant(0));
  const auto queues = q::take(i::prepare_queue_bindings(model.execution.queue_bindings(), compiled.report));
  std::array<q::PreparedReadback, 9> outputs;
  for (std::size_t at = 0; at != outputs.size(); ++at) outputs[at] = q::take(q::PreparedReadback::prepare(compiled, ports[at], model.execution, model.allocation_budget));
  auto target_values = model.caller_tensor(model.execution.device(), q::dense(q::Type::kFloat32, {2, model.vocabulary}));
  auto draft_values = model.caller_tensor(model.execution.device(), q::dense(q::Type::kFloat32, {1, model.vocabulary}));
  auto q_values = model.caller_tensor(model.execution.device(), q::dense(q::Type::kFloat32, {1, model.vocabulary}));
  auto key_values = model.caller_tensor(model.execution.device(), q::dense(q::Type::kUInt32, {2, 4}));
  auto proposal_values = model.caller_tensor(model.execution.device(), q::dense(q::Type::kInt32, {1}));
  auto target_rows = raw[0]; target_rows.insert(target_rows.end(), raw[2].begin(), raw[2].end());
  target_values.write<float>(target_rows); draft_values.write<float>(raw[1]); q_values.write<float>(q_log);
  const q::LogitsInputs modifiers{ready<float>(model, q::Type::kFloat32, bias), ready<std::uint8_t>(model, q::Type::kBool, mask)};
  auto run = q::take(compiled.executable.create_invocation({.resources = q::g::ResourceOptions{}}));
  for (std::size_t branch = 0; branch != 2; ++branch) {
    q::check(run.reset()); const std::array<std::int32_t, 1> proposal{branch ? 3 : 0};
    const std::array<std::uint32_t, 8> keys{123, 7, 0, 0, finals[branch][0], finals[branch][1], finals[branch][2], finals[branch][3]};
    proposal_values.write<std::int32_t>(proposal); key_values.write<std::uint32_t>(keys);
    q::check(run.bind(target, target_values.read())); q::check(run.bind(draft, draft_values.read()));
    q::check(run.bind(prefix.q[0], q_values.read())); q::check(run.bind(identities, key_values.read()));
    q::check(run.bind(proposals, proposal_values.read()));
    q::check(run.bind(*inputs.bias, modifiers.bias.description, modifiers.bias.lease));
    q::check(run.bind(*inputs.allowed, modifiers.allowed.description, modifiers.allowed.lease));
    for (auto &output : outputs) q::check(output.bind(run));
    q::check(run.prepare_replay(i::queue_span(queues), {.mode = q::g::ReplayMode::kCommands, .host_wait = q::g::HostWaitPolicy::kAllow}));
    auto done = q::take(run.submit(i::queue_span(queues))); q::check(done.wait());
    for (std::size_t row = 0; row != 3; ++row) {
      require(q::take(outputs[row + 3].predicate(model.execution, run)) == 1, "finite modified p/q/bonus was invalid");
      std::vector<float> actual(width); q::check(outputs[row].read(model.execution, run, std::as_writable_bytes(std::span(actual))));
      for (std::size_t at = 0; at != width; ++at)
        require(expected[row][at] ? std::isfinite(actual[at]) && std::abs(std::exp(double(actual[at])) - expected[row][at]) < 0.00003
                                 : actual[at] == -(std::numeric_limits<float>::infinity)(),
            "native modified p/q/bonus differs from the independent finite distribution");
    }
    std::int32_t final_id = -1; q::check(outputs[8].read(model.execution, run, std::as_writable_bytes(std::span(&final_id, 1))));
    require(q::take(outputs[6].predicate(model.execution, run)) == branch && q::take(outputs[7].predicate(model.execution, run)) == 1 &&
        final_id == draw(branch ? expected[2] : residual, finals[branch], kind),
        "modified speculative residual/bonus changed q twice or lost its independent original final draw");
  }
}

inline void cold(q::Model &model) {
  const std::array prefills{std::size_t{2}, std::size_t{3}};
  const std::array one{std::size_t{1}};
  const std::array two{std::size_t{2}};
  const std::array three{std::size_t{3}};
  for (std::size_t index = 0; index != 2; ++index) {
    model.prepare_family(prefills, false, 1, q::GraphRole::kMtpPrefill, false, q::TextCompletion::kEmitNext, index);
    for (const auto rows : prefills) model.plan_replay_uses(rows, 1, 12, q::GraphRole::kMtpPrefill, q::TextCompletion::kEmitNext, index);
    for (const auto role : {q::GraphRole::kMtpSingle, q::GraphRole::kMtpCatchupSingle}) {
      model.prepare_family(one, false, 1, role, false, q::TextCompletion::kEmitNext, index);
      model.plan_replay_uses(1, 1, 16, role, q::TextCompletion::kEmitNext, index);
      if (index == 1) {
        model.prepare_family(one, false, 1, role, false, q::TextCompletion::kStateOnly, index);
        model.plan_replay_uses(1, 1, 12, role, q::TextCompletion::kStateOnly, index);
      }
    }
    for (const auto role : {q::GraphRole::kMtpRound, q::GraphRole::kMtpCatchupRound}) {
      model.prepare_family(two, false, 1, role, false, q::TextCompletion::kEmitNext, index);
      model.plan_replay_uses(2, 1, 4, role, q::TextCompletion::kEmitNext, index);
    }
    model.prepare_family(three, false, 1, q::GraphRole::kVerify, false, q::TextCompletion::kEmitNext, index);
    model.plan_replay_uses(3, 1, 12, q::GraphRole::kVerify, q::TextCompletion::kEmitNext, index);
    model.prepare_recursive_round_programs(index);
  }
  for (const auto role : {q::GraphRole::kMtpPrefill, q::GraphRole::kVerify}) {
    model.prepare_family(three, false, 2, role, false, q::TextCompletion::kEmitNext, 1);
    model.plan_replay_uses(3, 2, 2, role, q::TextCompletion::kEmitNext, 1);
  }
}

inline std::int32_t raw(q::Model &model, const std::shared_ptr<q::Session> &session,
    std::span<const std::int32_t> ids, q::GraphRole role, const q::LogitsInputs &inputs,
    q::TextCompletion completion = q::TextCompletion::kEmitNext) {
  const auto epoch = session->head.epoch(); const auto cursor = session->head->sampling;
  model.prepare_evaluation(*session, ids, nullptr, role, completion, nullptr, inputs);
  if (completion == q::TextCompletion::kStateOnly) c::state_only_ports(*model.active_step->phase);
  model.submit_evaluation(*session); const std::array outputs{q::EvaluationOutput{.session = session}};
  std::array<std::int32_t, 1> next{-1}; model.finish_evaluations(outputs, next, [](auto, const auto &) {});
  const bool samples = session->sampling_configuration().configuration.enabled && completion == q::TextCompletion::kEmitNext;
  require(session->head.epoch() == epoch + 1 && session->head->sampling == (samples ? cursor.advanced(1) : cursor),
      "modified teacher-forced input changed its original epoch or RNG boundary");
  t::idle(model); return next[0];
}

inline q::MtpRoundResult round(q::Model &model, const std::shared_ptr<q::Session> &session,
    bool recursive, const Payload &value) {
  const auto &selected = session->sampling_configuration(); const auto cursor = session->head->sampling;
  const auto epoch = session->head.epoch(); const auto before = s::image(model, *session->head);
  const auto pending = session->head->next_id; std::vector<q::c::ReadLease> probabilities;
  auto ids = recursive ? model.propose_recursive(*session, 2, q::model_eos_tokens, &probabilities, nullptr, value.inputs)
                       : std::vector<std::int32_t>{pending};
  require(s::image(model, *session->head) == before && session->head.epoch() == epoch,
      "modified private proposal published a head or RNG state");
  if (recursive) { allowed(std::span(ids).subspan(1), value); q_rows(model, probabilities, value); }
  const auto role = recursive ? q::GraphRole::kVerify
      : session->head->draft->lag ? q::GraphRole::kMtpCatchupRound : q::GraphRole::kMtpRound;
  q::GenerationLedger ledger(1, 3); ledger.bind_stop_policy({}, model.vocabulary);
  q::prepare_generation_record(ledger, *session, role, std::span(&pending, 1), true, 3, 0, {}, recursive ? 2 : 0);
  model.prepare_evaluation(*session, ids, nullptr, role, q::TextCompletion::kEmitNext, nullptr, value.inputs); model.submit_evaluation(*session);
  const auto output = q::evaluation_output(model, session, {.name = "modified-round", .trace_states = false},
      nullptr, nullptr, nullptr, nullptr, &ledger);
  const std::array outputs{output}; const std::array selections{q::MtpGenerationSelection{session, pending, 3, probabilities, recursive ? 2u : 1u}};
  std::array<q::MtpRoundResult, 1> result;
  if (recursive) model.finish_recursive_mtp(selections, q::model_eos_tokens, outputs, result, [](auto, const auto &) {});
  else model.finish_mtp_rounds(selections, q::model_eos_tokens, outputs, result, [](auto, const auto &) {});
  allowed(result[0].emitted, value);
  require(session->head->sampling == (selected.configuration.enabled ? cursor.advanced(recursive ? 5 : 3) : cursor) &&
      session->head.epoch() == epoch + 1 && session->length() == before.length + result[0].processed_target && ledger.committed_count == 1,
      "modified K1/recursive selection changed the accepted boundary or original RNG reservation");
  t::idle(model); return std::move(result[0]);
}

struct CohortState final { h::ImageBytes image; std::vector<std::int32_t> generated; };
inline std::array<CohortState, 2> cohort(q::Model &model, const std::shared_ptr<i::HostStateStorage> &storage,
    const Payload &a, const Payload &b, bool replace_owner) {
  const std::array<std::int32_t, 3> prompt{1, 2, 3}; const std::array<const Payload *, 2> original{&a, &b};
  std::vector<q::GenerationRequest> requests; requests.reserve(2);
  i::RequestWindow window; window.prepare(2, 2);
  std::array<i::OutputDelivery, 2> delivery{i::OutputDelivery(3), i::OutputDelivery(3)};
  for (std::size_t member = 0; member != 2; ++member) {
    requests.emplace_back("modifier-batch" + std::to_string(member), prompt, std::filesystem::path{}, 4, 1);
    auto &request = requests.back(); t::prepare_request(request, 16); request.progress.initial_sampling = {171, 301 + member, 0};
    request.logits = original[member]->inputs; request.request_key = window.enqueue(member); window.admitted(request.request_key);
  }
  const auto prepare = [&](std::size_t member) {
    auto &request = requests[member]; return q::prepare_generation_call(model, request, 4, nullptr, requests,
        delivery[member].available(request.progress.generated.size()));
  };
  std::array prefills{prepare(0), prepare(1)};
  if (replace_owner) for (std::size_t member = 0; member != 2; ++member) requests[member].logits = original[1 - member]->inputs;
  q::execute_generation_calls(model, prefills, false, nullptr, requests);
  std::array<q::cache::SnapshotRef, 2> checkpoints; std::array<s::StateImage, 2> before; std::array<std::uint64_t, 2> epochs;
  for (std::size_t member = 0; member != 2; ++member) {
    auto &request = requests[member]; allowed(request.progress.generated, *original[member]);
    checkpoints[member] = request.session->checkpoint(); before[member] = s::image(model, *checkpoints[member]);
    epochs[member] = request.session->head.epoch(); delivery[member].acknowledge_prefix(request.progress.generated.size(), 1);
    request.logits = original[member]->inputs;
  }
  std::array calls{prepare(0), prepare(1)};
  if (replace_owner) for (std::size_t member = 0; member != 2; ++member) requests[member].logits = original[1 - member]->inputs;
  bool observed_snapshot = false;
  q::execute_generation_calls(model, calls, false, nullptr, requests, [&] {
    require(model.active_step->submitted && model.active_step->members.size() == 2 && model.active_step->input_tokens == 3 &&
        model.active_step->phase->family->sampling == &model.sampling_configuration(1), "modified B2 was split or changed sampler");
    for (std::size_t member = 0; member != 2; ++member) {
      require(calls[member].draft_log_probabilities.size() == 2, "modified B2 did not own both original q rows");
      q_rows(model, calls[member].draft_log_probabilities, *original[member]);
    }
    observed_snapshot = true;
  });
  require(observed_snapshot, "modified B2 never reached its original native submission");
  std::array<CohortState, 2> result;
  for (std::size_t member = 0; member != 2; ++member) {
    auto &request = requests[member]; allowed(request.progress.generated, *original[member]);
    require(calls[member].draft_log_probabilities.empty() && request.session->head.epoch() == epochs[member] + 1 &&
        request.session->head->sampling == before[member].sampling.advanced(5) &&
        s::image(model, *checkpoints[member]) == before[member], "modified B2 changed an old checkpoint, q lifetime, RNG or epoch");
    result[member] = {p::capture(model, *request.session, storage), request.progress.generated};
    window.retire_active(request.request_key); delivery[member].acknowledge_prefix(request.progress.generated.size(), request.progress.generated.size());
    h::retire(request.session); window.release(request.request_key); checkpoints[member].reset();
  }
  require(!model.snapshot_versions.live && !storage->live_images() && window.active().empty() && !window.waiting_front(),
      "modified B2 retained a request/state/image owner"); t::idle(model); return result;
}

inline void unchanged(q::Model &model, q::GenerationRequest &request, const e::SavedRequest &before,
    const std::shared_ptr<i::HostStateStorage> &storage) {
  require(request.session->head.get() == before.head && request.session->head.epoch() == before.epoch &&
      request.progress.generated == before.generated && request.progress.observed_commits() == before.observed &&
      request.ledger->committed_count == before.commits && request.progress.input_consumed() == before.consumed &&
      request.progress.input_turn() == before.input_turn && request.history.get() == before.history &&
      (!request.history || request.history->length == before.history_length) && request.ledger->sampling == before.sampling &&
      request.ledger->records.size() == before.records.size(), "modifier failure changed the original head/RNG/progress/ledger");
  for (std::size_t at = 0; at != before.records.size(); ++at)
    require(e::same_record(before.records[at], request.ledger->records[at]), "modifier failure changed an original ledger record");
  h::same_image(before.image, p::capture(model, *request.session, storage));
  auto reader = model.create_session({}, before.checkpoint, q::GraphRole::kMtpSingle, {}, false,
      q::TextCompletion::kEmitNext, request.sampling_index);
  h::same_image(before.image, p::capture(model, *reader, storage)); h::retire(reader); t::idle(model);
}

inline void terminal_failure_unchanged(q::Model &model, q::GenerationRequest &request, const e::SavedRequest &before,
    const std::shared_ptr<i::HostStateStorage> &storage) {
  require(request.session->head.get() == before.head && request.session->head.epoch() == before.epoch &&
      request.progress.generated == before.generated && request.progress.observed_commits() == before.observed &&
      request.ledger->committed_count == before.commits && request.progress.input_consumed() == before.consumed &&
      request.progress.input_turn() == before.input_turn && request.history.get() == before.history &&
      (!request.history || request.history->length == before.history_length) && request.ledger->sampling == before.sampling &&
      !request.ledger->successor_ordinal() && request.ledger->records.size() == before.records.size(),
      "terminal failure changed the committed head/RNG/progress/ledger or retained a successor reservation");
  for (std::size_t at = 0; at != before.commits; ++at)
    require(e::same_record(before.records[at], request.ledger->records[at]),
        "terminal failure changed an authoritative committed ledger record");
  const auto &slot = request.ledger->records.at(before.commits), &old_slot = before.records.at(before.commits);
  // Submission legitimately fills this reserved slot. Retirement must clear
  // all proposal/publication metadata, not restore the initial {-1,-1} sizes.
  // same_record covers quota/pending/input owner, accepted/verified IDs,
  // before/after boundaries, result/commit bits and timing/publication facts.
  q::GenerationCommitRecord cleared;
  cleared.emitted.clear(); cleared.accepted_inputs.clear(); cleared.verification_inputs.clear();
  require(e::same_record(cleared, slot) && slot.emitted.capacity() >= old_slot.emitted.capacity() &&
      slot.accepted_inputs.capacity() >= old_slot.accepted_inputs.capacity() &&
      slot.verification_inputs.capacity() >= old_slot.verification_inputs.capacity(),
      "terminal failure retained uncommitted authority or shrank its cold metadata capacity");
  for (std::size_t at = before.commits + 1; at < before.records.size(); ++at)
    require(e::same_record(before.records[at], request.ledger->records[at]),
        "terminal failure changed another reserved ledger slot");
  h::same_image(before.image, p::capture(model, *request.session, storage));
  auto reader = model.create_session({}, before.checkpoint, q::GraphRole::kMtpSingle, {}, false,
      q::TextCompletion::kEmitNext, request.sampling_index);
  h::same_image(before.image, p::capture(model, *reader, storage)); h::retire(reader); t::idle(model);
}

inline void request_journey(q::Model &model, const std::shared_ptr<i::HostStateStorage> &storage,
    const Payload &a, const Payload &b, const Payload &next, const Payload &masked, const Payload &nan,
    std::vector<p::StepReport> &checks) {
  const std::array<std::int32_t, 3> prompt{1, 2, 3};
  std::vector<q::GenerationRequest> requests; requests.reserve(2);
  requests.emplace_back("modified-history", prompt, std::filesystem::path{}, 4, 1);
  auto &owner = requests[0]; t::prepare_request(owner, 16); owner.progress.initial_sampling = {511, 71, 0};
  owner.logits = nan.inputs; owner.logits.allowed.description = q::dense(q::Type::kBool, {model.vocabulary});
  i::RequestWindow window; window.prepare(2, 2); owner.request_key = window.enqueue(0); const auto key = owner.request_key; window.admitted(key);
  i::OutputDelivery delivery(3); q::cache::PrefixCache prefix(1);
  const auto prepare = [&](q::GenerationRequest &request, std::size_t credit, std::size_t quantum = SIZE_MAX,
      q::cache::PrefixCache *directory = nullptr) {
    return q::prepare_generation_call(model, request, request.progress.output_limit(), directory, requests, credit, quantum);
  };
  const auto chunk = [&](bool final) {
    std::array calls{prepare(owner, delivery.available(owner.progress.generated.size()), 1, &prefix)};
    require(calls[0].completion == (final ? q::TextCompletion::kEmitNext : q::TextCompletion::kStateOnly),
        "bounded modifier input changed its true prefill completion");
    q::execute_generation_calls(model, calls, false, &prefix, requests, [&] {
      if (!final) {
        c::state_only_ports(*model.active_step->phase);
        require(!model.active_step->members[0].logits.bias.lease.valid() && !model.active_step->members[0].logits.allowed.lease.valid(),
            "StateOnly retained an episode modifier lease");
      }
    });
    q::publish_generation_prompt(model, calls[0], &prefix);
  };
  chunk(false);
  require(owner.session->length() == 1 && owner.session->head->next_id == -1 && owner.session->head->sampling == owner.progress.initial_sampling &&
      owner.progress.generated.empty() && !prefix.size(), "first StateOnly read invalid modifiers or produced an early output/draw/prefix");
  auto first_checkpoint = owner.session->checkpoint(); const auto first_bytes = s::image(model, *first_checkpoint);
  auto image = model.export_session(*owner.session, storage); const auto mirror = h::read_image(*image);
  require(image->identity == model.host_image_identity(1), "partial modifier HostImage lost its original configuration identity");
  owner.last_idle = window.pause(key); h::retire(owner.session); owner.session = model.restore_session(image); image.reset();
  u::same(model, mirror, p::capture(model, *owner.session, storage), checks); window.wake(key); window.admitted(key);
  chunk(false);
  require(owner.session->length() == 2 && owner.session->head->next_id == -1 && owner.progress.generated.empty() &&
      owner.session->head->sampling == owner.progress.initial_sampling && !prefix.size(), "second StateOnly sampled or published invalid modifier input");
  owner.logits = a.inputs; chunk(true);
  allowed(owner.progress.generated, a);
  require(owner.ledger->committed_count == 3 && owner.progress.generated.size() == 1 && prefix.size() == 1 &&
      owner.session->head->sampling == owner.progress.initial_sampling.advanced(1), "final bounded input did not emit exactly its one original draw");
  auto prompt_checkpoint = owner.session->checkpoint(); const auto publisher = p::capture(model, *owner.session, storage);
  const auto publisher_bytes = s::image(model, *prompt_checkpoint);
  const auto drafts_before_replay = model.draft_program_graph_calls;
  bool strict_rejected = false;
  try { auto invalid = q::restore_mtp_session(model, *owner.ledger, {}, false); (void)invalid; }
  catch (const std::runtime_error &failure) {
    const std::string_view message(failure.what()); strict_rejected = message.find("strict") != message.npos && message.find("modifiers") != message.npos;
  }
  require(strict_rejected && model.draft_program_graph_calls == drafts_before_replay && s::image(model, *prompt_checkpoint) == publisher_bytes,
      "modifier strict replay pretended to retain historical per-episode input values");

  requests.emplace_back("modified-new-prefix", prompt, std::filesystem::path{}, 1, 0);
  auto &hit = requests[1]; t::prepare_request(hit, 16); hit.logits = masked.inputs;
  hit.request_key = window.enqueue(1); const auto hit_key = hit.request_key; window.admitted(hit_key);
  const auto frontiers_before = model.frontier_sampling_graph_calls; const auto versions_before = model.snapshot_versions.live;
  bool frontier_failed = false;
  try { (void)prepare(hit, 1, SIZE_MAX, &prefix); }
  catch (const i::RequestExecutionFailure &failure) {
    const auto &actual = failure.evaluation(); frontier_failed = failure.request_key() == hit_key && actual.status().IsInvalidArgument() &&
        !actual.status().has_native_error() && !actual.participant() && actual.retirement_observed() &&
        actual.retirement().retired && actual.retirement().status.ok();
  }
  require(frontier_failed && model.frontier_sampling_graph_calls == frontiers_before + 1 &&
      model.snapshot_versions.live == versions_before && !hit.session && !hit.ledger->committed_count && hit.progress.generated.empty() &&
      !hit.ledger->sampling && window.matches(hit_key) && hit.request_key == hit_key &&
      s::image(model, *prompt_checkpoint) == publisher_bytes, "failed prefix frontier changed a publisher/key/output/RNG or claimed unobserved retirement");
  hit.logits = b.inputs; std::array hit_call{prepare(hit, 1, SIZE_MAX, &prefix)};
  require(hit_call[0].full_hit, "same-key prefix retry did not retain its original full hit");
  q::execute_generation_calls(model, hit_call, false, &prefix, requests); allowed(hit.progress.generated, b);
  {
    auto control = model.create_session({}, prompt_checkpoint, q::GraphRole::kMtpPrefill, q::cache::SamplingCursor{}, true,
        q::TextCompletion::kEmitNext, 0, b.inputs);
    require(hit.prefix_sampling_calls == 1 && hit.session->head->state_identity() == prompt_checkpoint->state_identity() &&
        hit.session->head->sampling == q::cache::SamplingCursor{} && hit.ledger->sampling == &model.sampling_configuration(0),
        "new prefix request inherited its random publisher or historical modifier payload");
    u::same(model, p::capture(model, *control, storage), p::capture(model, *hit.session, storage), checks); h::retire(control);
  }
  i::OutputDelivery hit_delivery(1); hit_delivery.acknowledge_prefix(hit.progress.generated.size(), 1);
  window.release_turn(hit_key, false); h::retire(hit.session); window.release(hit_key);

  owner.logits = next.inputs; const auto historical = q::generation_boundary(*owner.session->head);
  const auto generated = owner.progress.generated; const auto observed = owner.progress.observed_commits();
  owner.last_idle = window.pause(key); h::retire(owner.session); q::restore_generation_session(model, owner, nullptr, false, requests);
  require(owner.request_key == key && window.matches(key) && q::generation_boundary(*owner.session->head) == historical &&
      owner.progress.generated == generated && owner.progress.observed_commits() == observed && owner.restore_count == 1,
      "modifier StateOnly history reconstruction resampled a historical pending/cursor");
  auto expected_rebuild = publisher; const auto rebuilt = p::capture(model, *owner.session, storage);
  require(!publisher.boundary.sampling_wrapper && rebuilt.boundary.sampling_wrapper && rebuilt.boundary.evaluated_next_id == -1 &&
      rebuilt.boundary.evaluated_sampling == owner.ledger->records[2].before.sampling,
      "modifier rebuild lost its explicit unsampled canonical versus historical boundary");
  expected_rebuild.boundary.sampling_wrapper = true; expected_rebuild.boundary.evaluated_next_id = -1;
  expected_rebuild.boundary.evaluated_sampling = owner.ledger->records[2].before.sampling;
  t::equal(model, expected_rebuild, rebuilt, checks); window.wake(key); window.admitted(key);
  delivery.acknowledge_prefix(owner.progress.generated.size(), 1);

  auto saved = e::save(model, owner, storage); const auto original_inputs = owner.logits;
  const auto proposals_before = model.draft_program_graph_calls; const auto decisions_before = model.prefix_decision_graph_calls;
  owner.logits.allowed.description = q::dense(q::Type::kBool, {model.vocabulary}); bool metadata_failed = false;
  try { (void)prepare(owner, 3); }
  catch (const i::RequestExecutionFailure &failure) {
    const auto &actual = failure.evaluation(); metadata_failed = failure.request_key() == key && actual.status().IsInvalidArgument() &&
        !actual.status().has_native_error() && !actual.participant() && !actual.retirement_observed();
  }
  require(metadata_failed && model.draft_program_graph_calls == proposals_before && model.prefix_decision_graph_calls == decisions_before,
      "bad known-key MTP metadata reached a private proposal or submitted target");
  qwen_mtp_logits_case::unchanged(model, owner, saved, storage); owner.logits = original_inputs;
  for (const auto *bad : {&masked, &nan}) {
    owner.logits = bad->inputs; const auto native_before = model.draft_program_graph_calls; bool target_submitted = false, numeric_failed = false;
    std::array bad_call{prepare(owner, 3)};
    try { q::execute_generation_calls(model, bad_call, false, nullptr, requests, [&] { target_submitted = true; }); }
    catch (const i::RequestExecutionFailure &failure) {
      const auto &actual = failure.evaluation(); numeric_failed = failure.request_key() == key && actual.status().IsInvalidArgument() &&
          !actual.status().has_native_error() && !actual.participant() && actual.retirement_observed() &&
          actual.retirement().retired && actual.retirement().status.ok();
    }
    require(numeric_failed && !target_submitted && model.draft_program_graph_calls == native_before + 1,
        "invalid MTP q was preflight-only, published work or lost actual original native retirement");
    qwen_mtp_logits_case::unchanged(model, owner, saved, storage);
  }
  owner.logits = original_inputs;
  // The original q failures above retire the draft before target submission.
  // Here the actual q is valid: only the target's admitted mask is changed.
  // The real Verify episode runs before the original private prefix sampler
  // rejects its target distribution; neither may publish tentative state,
  // consume RNG or change the request key.
  {
    const auto versions = model.snapshot_versions.live;
    const auto target_pages = model.page_inventory.free_ids.size(), draft_pages = model.draft_inventory.free_ids.size();
    const auto &proposal_program = *model.draft_programs.at({1, true, owner.sampling_index});
    const auto q_blocks = proposal_program.probability_pool.stats().live_blocks;
    std::array terminal_call{prepare(owner, 3)};
    auto &call = terminal_call[0];
    require(call.role == q::GraphRole::kVerify, "invalid terminal check did not select the real recursive verifier");
    call.verification_ids = model.propose_recursive(*owner.session, call.maximum, q::model_eos_tokens,
        &call.draft_log_probabilities, &owner.ledger->stop_policy(), original_inputs);
    require(call.verification_ids.size() == 3 && call.draft_log_probabilities.size() == 2,
        "invalid terminal check lost its two real valid proposal distributions");
    q_rows(model, call.draft_log_probabilities, next);
    const auto native_before = model.draft_program_graph_calls;
    const auto terminals_before = model.prefix_decision_graph_calls;
    call.logits = masked.inputs;
    bool target_submitted = false, numeric_failed = false;
    try {
      q::execute_generation_calls(model, terminal_call, false, nullptr, requests, [&] {
        const auto &step = *model.active_step;
        target_submitted = step.submitted && step.members.size() == 1 && step.input_tokens == 3 &&
            step.phase->family->role == q::GraphRole::kVerify &&
            step.phase->family->sampling == &owner.session->sampling_configuration();
      });
    } catch (const i::EvaluationFailure &failure) {
      numeric_failed = failure.status().IsInvalidArgument() && !failure.status().has_native_error() &&
          failure.participant() && *failure.participant() == 0 && failure.retirement_observed() &&
          failure.retirement().retired && failure.retirement().status.ok();
    }
    require(target_submitted && numeric_failed && model.draft_program_graph_calls == native_before &&
        model.prefix_decision_graph_calls == terminals_before + 1,
        "invalid private prefix terminal was preflight-only, rebuilt q or lost its original target retirement");
    terminal_failure_unchanged(model, owner, saved, storage);
    call.draft_log_probabilities.clear();
    require(owner.request_key == key && window.matches(key) && !owner.session->pending && model.snapshot_versions.live == versions &&
        model.page_inventory.free_ids.size() == target_pages && model.draft_inventory.free_ids.size() == draft_pages &&
        proposal_program.probability_pool.stats().live_blocks == q_blocks,
        "failed private prefix terminal retained candidate state, append pages or private proposal reads");
    t::idle(model);
  }
  h::ImageBytes expected_next; q::MtpRoundResult expected_outputs;
  {
    const std::array three{std::size_t{3}};
    auto control = model.create_session(three, owner.session->checkpoint(), q::GraphRole::kVerify, {}, false, q::TextCompletion::kEmitNext, 1);
    expected_outputs = round(model, control, true, next); expected_next = p::capture(model, *control, storage); h::retire(control);
  }
  const auto cursor = owner.session->head->sampling; std::array continuation{prepare(owner, 3)};
  q::execute_generation_calls(model, continuation, false, nullptr, requests);
  require(owner.request_key == key && window.matches(key) && continuation[0].draft_log_probabilities.empty() &&
      owner.session->head->sampling == cursor.advanced(5) && owner.ledger->committed_count == 4 &&
      std::ranges::equal(owner.rounds.back().emitted, expected_outputs.emitted), "same-key modifier retry lost its current snapshot/RNG or retained completed q");
  allowed(owner.rounds.back().emitted, next); u::same(model, expected_next, p::capture(model, *owner.session, storage), checks);
  require(s::image(model, *first_checkpoint) == first_bytes && s::image(model, *prompt_checkpoint) == publisher_bytes,
      "modifier failure/rebuild/continuation changed an old complete checkpoint");
  window.retire_active(key); delivery.acknowledge_prefix(owner.progress.generated.size(), owner.progress.generated.size());
  h::retire(owner.session); window.release(key); saved.checkpoint.reset(); first_checkpoint.reset(); prompt_checkpoint.reset(); prefix.evict_one();
  require(!model.snapshot_versions.live && !storage->live_images() && window.active().empty() && !window.waiting_front(),
      "modifier request journey did not really retire its state/image/key owners"); t::idle(model);
}

inline int run(const std::filesystem::path &input, q::BackendKind kind) {
  auto source = q::prepare_weight_source(input / "weights.tsv", true); const auto geometry = source.roles.geometry;
  require(geometry.hidden == 64 && geometry.vocabulary == 64 && geometry.context_limit >= 16 && geometry.mtp_layers == 1 &&
      geometry.recurrent_layers() && geometry.attention_layers(), "MTP modifiers require the existing cached GA/H64/V64/C16 fixture");
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 16, 12, 4,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 2, 3, true, true);
  const std::array additional{q::SamplingConfiguration{true, {.8F, 0, 1.F}, {.bias = true, .allowed = true}}};
  q::Model model(std::move(source), geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands, SIZE_MAX,
      {.enabled = false, .policy = {}, .seed = 123, .modifiers = {.bias = true, .allowed = true}}, 0, {.primary = kind}, additional);
  model.configure_draft_policy({.max_drafts = 2}); cold(model); finite_math(model, kind);
  const auto storage = q::take(i::HostStateStorage::create(model.execution, 1, 4u << 20));
  const auto a = payload(model, 0), b = payload(model, 1), next = payload(model, 2), masked = payload(model, 0, true), nan = payload(model, 0, false, true);
  std::vector<p::StepReport> checks; const std::array chunks{std::size_t{3}}; const std::array<std::int32_t, 3> prompt{1, 2, 3};
  for (std::size_t index = 0; index != 2; ++index) {
    const auto cursor = index ? q::cache::SamplingCursor{313, 17, 0} : q::cache::SamplingCursor{};
    auto control = model.create_session(chunks, {}, q::GraphRole::kMtpPrefill, cursor, false, q::TextCompletion::kEmitNext, index);
    auto candidate = model.create_session(chunks, {}, q::GraphRole::kMtpPrefill, cursor, false, q::TextCompletion::kEmitNext, index);
    require(raw(model, control, prompt, q::GraphRole::kMtpPrefill, a.inputs) == raw(model, candidate, prompt, q::GraphRole::kMtpPrefill, a.inputs),
        "identical modified prefill changed its sampled output");
    auto old = candidate->checkpoint(); const auto old_bytes = s::image(model, *old);
    for (const bool recursive : {false, true}) {
      const auto expected = round(model, control, recursive, a); const auto actual = round(model, candidate, recursive, a);
      h::same_round(expected, actual); u::same(model, p::capture(model, *control, storage), p::capture(model, *candidate, storage), checks);
    }
    require(s::image(model, *old) == old_bytes, "modified K1/recursive work changed a frozen checkpoint");
    old.reset(); h::retire(control); h::retire(candidate);
  }
  const auto original = cohort(model, storage, a, b, false); const auto replaced = cohort(model, storage, a, b, true);
  for (std::size_t member = 0; member != 2; ++member) {
    require(original[member].generated == replaced[member].generated, "B2 used replacement owner values instead of original episode inputs");
    u::same(model, original[member].image, replaced[member].image, checks);
  }
  request_journey(model, storage, a, b, next, masked, nan, checks);
  require(!model.snapshot_versions.live && !storage->live_images() && model.page_inventory.free_ids.size() == plan.max_pages &&
      model.draft_inventory.free_ids.size() == model.draft_inventory.owners.size(), "MTP modifier consumer left state/page/native owners live");
  t::idle(model);
  std::cout << "{\"status\":\"PASS\",\"case\":\"mtp-logits-modifiers\",\"backend\":\"" << h::backend_name(kind)
      << "\",\"models\":1,\"configurations\":2,\"K1_greedy_random\":true,\"recursive_greedy_random\":true,"
      "\"finite_math_branches\":2,\"already_normalized_q_not_reprocessed\":true,\"original_B2_snapshots\":true,"
      "\"StateOnly_no_input_or_draw\":true,\"partial_HostImage\":true,\"prefix_frontier_numeric_failure\":true,"
      "\"new_prefix_policy_payload\":true,\"strict_modifier_replay_rejected\":true,\"historical_StateOnly_rebuild\":true,"
      "\"known_key_metadata_zero_native\":true,\"numeric_q_failures\":2,\"target_terminal_numeric_failure\":true,"
      "\"target_terminal_retired_without_publish\":true,\"target_terminal_resources_returned\":true,\"same_key_retry\":true,"
      "\"synthetic_untrained\":true,\"quality_claim\":false,\"performance_claim\":false,\"final_versions\":0,\"state_checks\":[";
  for (std::size_t at = 0; at != checks.size(); ++at) { if (at) std::cout << ','; p::write_step(checks[at]); }
  std::cout << "]}\n"; return 0;
}

} // namespace qwen_mtp_logits_case
