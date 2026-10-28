#pragma once

#include "../compute_models/sampling_fixture.h"

// Finite public Graph mathematics inside the caller's already loaded Model.
// No token history owner, model execution route or production RNG reference.
namespace inference_history_math_case {
namespace q = qwen35;
namespace i = hyle::inference;
namespace m = qwen_mtp_logits_case;
using qwen_host_image_case::require;
using Row = std::array<float, 4>;
using Counts = std::array<std::int32_t, 4>;

inline Row processed(Row raw, const Row &bias, const Counts &counts,
    q::sampling::HistoryPolicy policy) {
  for (std::size_t at = 0; at != raw.size(); ++at) {
    raw[at] += bias[at];
    if (counts[at] && policy.repetition != 1)
      raw[at] = raw[at] > 0 ? raw[at] / policy.repetition : raw[at] * policy.repetition;
    raw[at] -= policy.frequency * static_cast<float>(counts[at]);
    raw[at] -= counts[at] ? policy.presence : 0;
  }
  return raw;
}
inline std::vector<double> probabilities(const Row &row) {
  const double maximum = *std::max_element(row.begin(), row.end());
  std::vector<double> result(row.size()); double total = 0;
  for (std::size_t at = 0; at != row.size(); ++at) { result[at] = std::exp(double(row[at]) - maximum); total += result[at]; }
  for (auto &entry : result) entry /= total;
  return result;
}
inline Row logarithms(std::span<const double> values) {
  require(values.size() == 4, "finite history reference width changed");
  Row result{}; for (std::size_t at = 0; at != result.size(); ++at) result[at] = static_cast<float>(std::log(values[at]));
  return result;
}
inline q::g::CompiledGraph compile(q::Model &model, q::GraphAssembly &assembly, q::Recipe &r) {
  const auto graph = q::take(std::move(r.builder).finish());
  const std::array shape{q::g::ShapeBinding{r.last_row, 0}};
  const std::array variants{q::g::ExactShapeVariant{shape}};
  auto result = q::take(model.execution.compile(graph, assembly.constants,
      assembly.options({.shape_policy = q::g::CompileShapePolicy::kFiniteExact, .exact_variants = variants,
        .constants = assembly.constants, .resource_budget = model.execution_budget})));
  result.executable = q::take(result.executable.exact_variant(0)); result.report = q::take(result.report.exact_variant(0));
  return result;
}
template<class T, std::size_t N>
inline std::array<T, N> read(q::Model &model, q::PreparedReadback &output, q::g::Invocation &run) {
  std::array<T, N> result{};
  q::check(output.read(model.execution, run, std::as_writable_bytes(std::span(result)))); return result;
}

inline void neutral(q::Model &model) {
  require(q::sampling::LogitsPolicy{}.empty() && q::sampling::HistoryPolicy{}.empty() &&
      q::sampling::valid_history_policy({}), "default history is not the original identity policy");
  for (const auto value : {q::sampling::HistoryPolicy{0, 0, 0},
      q::sampling::HistoryPolicy{-1, 0, 0},
      q::sampling::HistoryPolicy{(std::numeric_limits<float>::denorm_min)(), 0, 0},
      q::sampling::HistoryPolicy{1, (std::numeric_limits<float>::quiet_NaN)(), 0},
      q::sampling::HistoryPolicy{1, 0, (std::numeric_limits<float>::infinity)()}})
    require(!q::sampling::valid_history_policy(value), "history admitted a nonfinite coefficient or unusable repetition reciprocal");
  require(q::sampling::valid_history_policy({.5F, -.5F, -.25F}), "finite boosting history coefficients were rejected");
  q::GraphAssembly assembly(model.execution, model.precision, model.allocation_budget);
  q::Recipe r(model.weights, model.projections, assembly, model.geometry, 1, 1, &model.weight_meanings);
  const auto raw = r.input(q::dense(q::Type::kFloat32, {1, 4}));
  const auto counts = r.input(q::dense(q::Type::kInt32, {1, 4}));
  const auto value = q::sampling::process_logits(r, raw, {.counts = counts});
  require(value.logits == raw && !value.valid, "neutral history inserted a processing or publication proof node");
  bool rejected = false;
  try { (void)q::sampling::process_logits(r, raw, {.counts = counts}, {0, 0, 0}); }
  catch (const std::runtime_error &) { rejected = true; }
  require(rejected, "cold history accepted zero repetition");
  r.publish(value.logits); const auto graph = q::take(std::move(r.builder).finish());
  require(graph.node_count() == 0 && graph.constant_count() == 0,
      "default history added Graph nodes or literal storage");
}

inline void conditional_math(q::Model &model, q::BackendKind kind) {
  const q::sampling::HistoryPolicy history{2, .5F, .25F};
  const Row raw{4, -2, 3, 0}, bias{0, 1, 0, 0};
  const std::array<Counts, 4> counts{{{1, 2, 0, 0}, {1, 2, 1, 0}, {1, 2, 2, 0}, {1, 2, 2, 1}}};
  const std::array<Row, 4> hand{{{1.25F, -3, 3, 0}, {1.25F, -3, .75F, 0},
      {1.25F, -3, .5F, 0}, {1.25F, -3, .5F, -.75F}}};
  std::array<std::vector<double>, 4> target;
  for (std::size_t row = 0; row != target.size(); ++row) {
    require(processed(raw, bias, counts[row], history) == hand[row], "finite history hand calculation changed");
    target[row] = probabilities(hand[row]);
  }
  const std::array<Row, 2> draft_raw{{{2, 0, 6, -1}, {1, 5, 2, -2}}};
  std::array<std::vector<double>, 2> draft;
  std::array<Row, 2> normalized_q;
  for (std::size_t row = 0; row != draft.size(); ++row) {
    draft[row] = probabilities(processed(draft_raw[row], bias, counts[row], history));
    normalized_q[row] = logarithms(draft[row]);
  }
  std::vector<double> residual(4), wrong_residual(4);
  const auto wrong_q = probabilities(processed(normalized_q[0], bias, counts[0], history));
  for (std::size_t at = 0; at != residual.size(); ++at) {
    residual[at] = (std::max)(0., target[0][at] - draft[0][at]);
    wrong_residual[at] = (std::max)(0., target[0][at] - wrong_q[at]);
  }
  const double ratio = (std::min)(1., target[0][2] / draft[0][2]);
  require(ratio > .1 && ratio < .95 && target[1][2] > draft[1][2], "finite history no longer has both actual acceptance outcomes");
  std::array<sampling_test::Words, 2> accepts{}, finals{};
  for (std::size_t branch = 0; branch != 2; ++branch) {
    bool found_accept = false, found_final = false;
    for (std::uint32_t at = 0; at != 4096 && (!found_accept || !found_final); ++at) {
      const sampling_test::Words address{0x243f6a88U, 0x85a308d3U, at, 0};
      if (!found_accept && m::separated_draw(std::array{ratio, 1 - ratio}, address, kind) &&
          m::draw(std::array{ratio, 1 - ratio}, address, kind) == (branch ? 0 : 1)) {
        accepts[branch] = address; found_accept = true;
      }
      if (!found_final && m::separated_draw(branch ? target[2] : residual, address, kind) &&
          m::separated_draw(branch ? target[1] : wrong_residual, address, kind) &&
          m::separated_draw(target[2], address, kind) &&
          m::draw(branch ? target[2] : residual, address, kind) !=
          m::draw(branch ? target[1] : wrong_residual, address, kind)) {
        finals[branch] = address; found_final = true;
      }
    }
    require(found_accept && found_final, "finite history addresses do not distinguish q reprocessing or the bonus prefix");
  }

  q::GraphAssembly assembly(model.execution, model.precision, model.allocation_budget);
  q::Recipe r(model.weights, model.projections, assembly, model.geometry, 1, 1, &model.weight_meanings);
  const auto raw_port = r.input(q::dense(q::Type::kFloat32, {1, 4}));
  const auto target_port = r.input(q::dense(q::Type::kFloat32, {3, 4}));
  const auto q0_port = r.input(q::dense(q::Type::kFloat32, {1, 4}));
  const auto q1_port = r.input(q::dense(q::Type::kFloat32, {1, 4}));
  const auto bias_port = r.input(q::dense(q::Type::kFloat32, {1, 4}));
  const auto mask_port = r.input(q::dense(q::Type::kBool, {1, 4}));
  const auto counts_port = r.input(q::dense(q::Type::kInt32, {1, 4}));
  const auto ids_port = r.input(q::dense(q::Type::kInt32, {2}));
  const auto keys_port = r.input(q::dense(q::Type::kUInt32, {3, 4}));
  const auto one_shape = q::take(q::c::TensorShape::create(std::array<std::int64_t, 1>{1}));
  const auto token = [&](std::int32_t value) {
    return r.one(q::c::Fill{.output_shape = one_shape, .output_type = q::Type::kInt32, .value = q::c::ScalarValue(value)}, {});
  };
  std::array<q::V, 4> count_values{counts_port};
  for (std::size_t row = 1; row != count_values.size(); ++row)
    count_values[row] = q::sampling::advance_history_counts(r, count_values[row - 1], token(row == 3 ? 3 : 2));
  std::vector<q::V> ports, target_ports;
  for (std::size_t row = 0; row != count_values.size(); ++row) {
    const auto value = q::sampling::process_logits(r, raw_port,
        {.bias = bias_port, .allowed = mask_port, .counts = count_values[row]}, history);
    ports.push_back(value.logits); ports.push_back(r.host_predicate(*value.valid)); ports.push_back(count_values[row]);
    if (row < 3) {
      const auto normalized = q::sampling::processed_distribution(r, value);
      target_ports.push_back(normalized.log_probabilities); target_ports.push_back(r.host_predicate(normalized.valid));
    }
  }
  for (std::size_t row = 0; row != 2; ++row) {
    const auto value = q::sampling::processed_distribution(r, q::sampling::process_logits(r, row ? q1_port : q0_port,
        {.bias = bias_port, .allowed = mask_port, .counts = count_values[row]}, history));
    ports.push_back(value.log_probabilities); ports.push_back(r.host_predicate(value.valid));
  }
  const auto prefix = q::sampling::speculative_prefix(r, target_port, keys_port, ids_port, {},
      {.bias = bias_port, .allowed = mask_port, .counts = counts_port}, history);
  for (std::size_t row = 0; row != 2; ++row) {
    ports.push_back(r.host_predicate(prefix.accepted[row])); ports.push_back(prefix.final_ids[row]);
    ports.push_back(r.host_predicate(prefix.valid[row]));
  }
  ports.insert(ports.end(), target_ports.begin(), target_ports.end());
  for (auto value : ports) r.publish(value);
  auto compiled = compile(model, assembly, r);
  const auto queues = q::take(i::prepare_queue_bindings(model.execution.queue_bindings(), compiled.report));
  std::vector<q::PreparedReadback> outputs;
  for (auto value : ports) outputs.push_back(q::take(q::PreparedReadback::prepare(compiled, value, model.execution, model.allocation_budget)));
  auto raw_values = model.caller_tensor(model.execution.device(), q::dense(q::Type::kFloat32, {1, 4}));
  auto target_values = model.caller_tensor(model.execution.device(), q::dense(q::Type::kFloat32, {3, 4}));
  auto q0_values = model.caller_tensor(model.execution.device(), q::dense(q::Type::kFloat32, {1, 4}));
  auto q1_values = model.caller_tensor(model.execution.device(), q::dense(q::Type::kFloat32, {1, 4}));
  auto normalized0 = model.caller_tensor(model.execution.device(), q::dense(q::Type::kFloat32, {1, 4}));
  auto normalized1 = model.caller_tensor(model.execution.device(), q::dense(q::Type::kFloat32, {1, 4}));
  auto bias_values = model.caller_tensor(model.execution.device(), q::dense(q::Type::kFloat32, {1, 4}));
  auto mask_values = model.caller_tensor(model.execution.device(), q::dense(q::Type::kBool, {1, 4}));
  auto counter_values = model.caller_tensor(model.execution.device(), q::dense(q::Type::kInt32, {1, 4}));
  auto id_values = model.caller_tensor(model.execution.device(), q::dense(q::Type::kInt32, {2}));
  auto key_values = model.caller_tensor(model.execution.device(), q::dense(q::Type::kUInt32, {3, 4}));
  q0_values.write<float>(draft_raw[0]); q1_values.write<float>(draft_raw[1]);
  normalized0.write<float>(normalized_q[0]); normalized1.write<float>(normalized_q[1]);
  id_values.write<std::int32_t>(std::array<std::int32_t, 2>{2, 2});
  auto run = q::take(compiled.executable.create_invocation({.resources = q::g::ResourceOptions{}}));
  for (std::size_t arm = 0; arm != 6; ++arm) {
    q::check(run.reset());
    auto current_raw = raw; auto current_bias = bias; auto current_counts = counts[0];
    std::array<std::uint8_t, 4> mask{1, 1, 1, 1};
    if (arm == 2) { current_counts[3] = -1; mask[3] = 0; }
    if (arm == 3) { current_raw[3] = (std::numeric_limits<float>::quiet_NaN)(); mask[3] = 0; }
    if (arm == 4) { current_bias[3] = (std::numeric_limits<float>::quiet_NaN)(); mask[3] = 0; }
    if (arm == 5) { current_raw[3] = (std::numeric_limits<float>::infinity)(); mask[3] = 0; }
    raw_values.write<float>(current_raw); bias_values.write<float>(current_bias);
    mask_values.write<std::uint8_t>(mask); counter_values.write<std::int32_t>(current_counts);
    std::array<float, 12> target_rows{};
    for (std::size_t row = 0; row != 3; ++row) std::copy(current_raw.begin(), current_raw.end(), target_rows.begin() + row * 4);
    target_values.write<float>(target_rows);
    std::array<std::uint32_t, 12> keys{};
    const auto branch = arm == 1 ? 1U : 0U;
    std::copy(accepts[branch].begin(), accepts[branch].end(), keys.begin());
    const sampling_test::Words certain{0x13198a2eU, 0x03707344U, 0, 0};
    std::copy(certain.begin(), certain.end(), keys.begin() + 4);
    std::copy(finals[branch].begin(), finals[branch].end(), keys.begin() + 8); key_values.write<std::uint32_t>(keys);
    q::check(run.bind(raw_port, raw_values.read())); q::check(run.bind(target_port, target_values.read()));
    q::check(run.bind(q0_port, q0_values.read())); q::check(run.bind(q1_port, q1_values.read()));
    q::check(run.bind(bias_port, bias_values.read())); q::check(run.bind(mask_port, mask_values.read()));
    q::check(run.bind(counts_port, counter_values.read())); q::check(run.bind(ids_port, id_values.read()));
    q::check(run.bind(keys_port, key_values.read())); q::check(run.bind(prefix.q[0], normalized0.read()));
    q::check(run.bind(prefix.q[1], normalized1.read())); for (auto &output : outputs) q::check(output.bind(run));
    q::check(run.prepare_replay(i::queue_span(queues), {.mode = q::g::ReplayMode::kCommands, .host_wait = q::g::HostWaitPolicy::kAllow}));
    auto done = q::take(run.submit(i::queue_span(queues))); q::check(done.wait());
    Counts original{}; q::check(model.execution.read(counter_values.read().storage(), std::as_writable_bytes(std::span(original))));
    require(original == current_counts, "hypothesis updates changed the caller's original readonly counts");
    if (arm >= 2) {
      require(q::take(outputs[1].predicate(model.execution, run)) == 0 && q::take(outputs[18].predicate(model.execution, run)) == 0,
          "negative counts, nonfinite raw logits or bias NaN disappeared behind the allowed mask");
      continue;
    }
    for (std::size_t row = 0; row != 4; ++row) {
      require(read<float, 4>(model, outputs[row * 3], run) == hand[row] &&
          q::take(outputs[row * 3 + 1].predicate(model.execution, run)) == 1 &&
          read<std::int32_t, 4>(model, outputs[row * 3 + 2], run) == counts[row],
          "history hand logits, duplicate ID counts or complete validity changed");
    }
    for (std::size_t row = 0; row != 2; ++row) {
      const auto actual = read<float, 4>(model, outputs[12 + row * 2], run);
      require(q::take(outputs[13 + row * 2].predicate(model.execution, run)) == 1, "history proposal distribution is invalid");
      for (std::size_t at = 0; at != 4; ++at)
        require(std::abs(std::exp(double(actual[at])) - draft[row][at]) < .00003,
            "history-conditioned q differs from the independent F64 normalizer");
    }
    for (std::size_t row = 0; row != 3; ++row) {
      const auto actual = read<float, 4>(model, outputs[22 + row * 2], run);
      require(q::take(outputs[23 + row * 2].predicate(model.execution, run)) == 1, "history target or bonus distribution is invalid");
      for (std::size_t at = 0; at != 4; ++at)
        require(std::abs(std::exp(double(actual[at])) - target[row][at]) < .00003,
            "history-conditioned p/bonus differs from the independent F64 normalizer");
    }
    require(q::take(outputs[16].predicate(model.execution, run)) == branch &&
        q::take(outputs[18].predicate(model.execution, run)) == 1 &&
        q::take(outputs[19].predicate(model.execution, run)) == 1 && q::take(outputs[21].predicate(model.execution, run)) == 1,
        "history conditional acceptance changed or lost its complete validity proof");
    require(read<std::int32_t, 1>(model, outputs[17], run)[0] == m::draw(branch ? target[1] : residual, finals[branch], kind) &&
        read<std::int32_t, 1>(model, outputs[20], run)[0] == m::draw(target[2], finals[branch], kind),
        "history verifier transformed external q again or used the wrong residual/bonus history");
    Row original_q{}; q::check(model.execution.read(normalized0.read().storage(), std::as_writable_bytes(std::span(original_q))));
    require(original_q == normalized_q[0], "verifier rewrote its original normalized q input");
  }
  q::check(run.close()); q::check(model.execution.drain_retired(run));
  require(!run.in_flight(), "finite history Graph did not actually retire");
}

inline void integer_boundaries(q::Model &model) {
  q::GraphAssembly assembly(model.execution, model.precision, model.allocation_budget);
  q::Recipe r(model.weights, model.projections, assembly, model.geometry, 1, 1, &model.weight_meanings);
  const auto counts = r.input(q::dense(q::Type::kInt32, {1, 4}));
  const auto token = r.input(q::dense(q::Type::kInt32, {1}));
  auto next = counts;
  for (unsigned at = 0; at != 3; ++at) next = q::sampling::advance_history_counts(r, next, token);
  r.publish(next); auto compiled = compile(model, assembly, r);
  const auto queues = q::take(i::prepare_queue_bindings(model.execution.queue_bindings(), compiled.report));
  auto values = model.caller_tensor(model.execution.device(), q::dense(q::Type::kInt32, {1, 4}));
  auto id = model.caller_tensor(model.execution.device(), q::dense(q::Type::kInt32, {1}));
  const Counts original{16777217, (std::numeric_limits<std::int32_t>::max)() - 1, 0, 0};
  for (std::int32_t arm = 0; arm != 2; ++arm) {
    values.write<std::int32_t>(original); id.write<std::int32_t>(std::span(&arm, 1));
    auto output = q::take(q::PreparedReadback::prepare(compiled, next, model.execution, model.allocation_budget));
    auto run = q::take(compiled.executable.create_invocation({.resources = q::g::ResourceOptions{}}));
    q::check(run.bind(counts, values.read())); q::check(run.bind(token, id.read())); q::check(output.bind(run));
    q::check(run.prepare_replay(i::queue_span(queues), {.mode = q::g::ReplayMode::kCommands, .host_wait = q::g::HostWaitPolicy::kAllow}));
    // Managed CPU work may report checked overflow directly from submit;
    // an asynchronous backend may report the same failure from completion.
    auto submitted = run.submit(i::queue_span(queues));
    hyle::Status status;
    if (submitted) status = submitted->wait();
    else status = submitted.error();
    if (!arm) {
      q::check(status); auto expected = original; expected[0] += 3;
      require(read<std::int32_t, 4>(model, output, run) == expected,
          "history counters were rounded through F32 or artificially capped at 2^24");
    } else {
      require(status.IsOutOfRange() && !run.result(next), "INT32_MAX history overflow published a usable counter row");
      const auto retry = run.submit(i::queue_span(queues));
      require(!retry && retry.error().IsFailedPrecondition(), "failed checked Scan invocation was silently reused");
    }
    Counts untouched{}; q::check(model.execution.read(values.read().storage(), std::as_writable_bytes(std::span(untouched))));
    require(untouched == original, "checked count overflow changed the readonly source histogram");
    if (submitted) *submitted = {};
    (void)run.close(); const auto retired = model.execution.drain_retired(run);
    // Drain observes the original terminal error as well as native retirement.
    // An expected checked overflow is still an error after its work retires.
    if (!retired.ok() && (!arm || !retired.IsOutOfRange() || retired.has_native_error())) q::check(retired);
    require(!run.in_flight(), "count boundary work did not actually retire");
  }
}

inline void finite_math(q::Model &model, q::BackendKind kind) {
  neutral(model); conditional_math(model, kind); integer_boundaries(model);
  std::cout << "{\"case\":\"history-finite-math\",\"backend\":\"" << qwen_host_image_case::backend_name(kind)
      << "\",\"neutral_nodes\":0,\"hand_rows\":4,\"duplicate_prefix_ids\":2,\"p_q_bonus\":true,"
         "\"negative_counts_rejected\":true,\"masked_raw_nan_rejected\":true,\"masked_raw_inf_rejected\":true,\"masked_bias_nan_rejected\":true,"
         "\"readonly_counts\":true,\"normalized_q_unchanged\":true,\"count_above_2pow24_exact\":true,"
         "\"int32_overflow_rejected\":true,\"managed_commands\":true,\"result\":\"PASS\"}\n";
}
} // namespace inference_history_math_case
