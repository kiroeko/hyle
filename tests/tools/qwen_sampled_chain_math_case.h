#pragma once

#include "../compute_models/sampling_fixture.h"

// Independent Philox/Gumbel equations validate the K1/compound graph's proposal,
// accept and shared-final addresses. No model text or trajectory is its oracle.
namespace qwen_sampled_chain_math_case {
namespace q = qwen35;
namespace i = hyle::inference;
using qwen_host_image_case::require;
using Row = std::array<double, 4>;

inline std::pair<std::int32_t, double> draw(std::span<const double> weights,
    sampling_test::Words key) {
  float best = -INFINITY, runner = best;
  std::int32_t selected = -1;
  for (std::size_t at = 0; at < weights.size(); ++at) if (weights[at] > 0) {
    auto words = sampling_test::philox({static_cast<std::uint32_t>(at / 2), 0, key[2], key[3]}, {key[0], key[1]});
    auto hi = words[(at % 2) * 2], lo = words[(at % 2) * 2 + 1];
    const bool upper = (hi & 0x80000000U) != 0;
    if (upper) { hi = ~hi; lo = ~lo; }
    const float small = static_cast<float>(hi) * 0x1p-32F + (static_cast<float>(lo) + .5F) * 0x1p-64F;
    const float exponential = upper ? -std::log1p(-small) : -std::log(small);
    const float score = static_cast<float>(std::log(weights[at])) - std::log(exponential);
    if (score > best) { runner = best; best = score; selected = static_cast<std::int32_t>(at); }
    else runner = (std::max)(runner, score);
  }
  require(selected >= 0, "independent compound reference has no support");
  return {selected, double(best) - double(runner)};
}

inline void run(q::Model &model, std::size_t count) {
  require(count >= 1 && count <= q::kMaximumDeviceMtpDrafts,
      "sampled math witness requires the actual bounded K1/K2/K3 recipe");
  const std::array<Row, 4> p{{{.10, .20, .30, .40}, {.40, .30, .20, .10},
      {.20, .15, .25, .40}, {.10, .50, .15, .25}}};
  const std::array<Row, 3> qrows{{{.40, .10, .40, .10}, {.10, .50, .10, .30}, {.45, .25, .20, .10}}};
  q::GraphAssembly assembly(model.execution, model.precision, model.allocation_budget);
  q::Recipe r(model.weights, model.projections, assembly, model.geometry, 1, 1, &model.weight_meanings);
  const auto target = r.input(q::dense(q::Type::kFloat32, {static_cast<std::int64_t>(count + 1), 4}));
  const auto raw_q = r.input(q::dense(q::Type::kFloat32, {static_cast<std::int64_t>(count), 4}));
  const auto keys = r.input(q::dense(q::Type::kUInt32, {static_cast<std::int64_t>(2 * count + 1), 4}));
  std::vector<q::V> identities, proposals;
  std::vector<q::sampling::Distribution> distributions;
  for (std::size_t at = 0; at <= 2 * count; ++at)
    identities.push_back(r.copy(r.slice(keys, 0, q::f(static_cast<std::int64_t>(at)), q::f(1))));
  for (std::size_t at = 0; at < count; ++at) {
    distributions.push_back(q::sampling::distribution(r,
        r.copy(r.slice(raw_q, 0, q::f(static_cast<std::int64_t>(at)), q::f(1)))));
    proposals.push_back(r.one(q::c::sampling_ops::categorical(),
        {distributions.back().log_probabilities, identities[2 * at]}));
  }
  const auto decision = [&]() -> q::DraftChainSamplingValues {
    if (count != 1) return q::decide_draft_chain(r, target, distributions, proposals, identities, {});
    // Match the real K1 Family, not the recursive K2/K3 chain helper: q uses
    // draw0, accept uses draw1, and residual/bonus shares the final draw2.
    const auto p = q::sampling::distribution(r, r.copy(r.slice(target, 0, q::f(0), q::f(1))));
    const auto bonus = q::sampling::distribution(r, r.copy(r.slice(target, 0, q::f(1), q::f(1))));
    const auto selected = q::sampling::decision(r, p.log_probabilities,
        distributions[0].log_probabilities, bonus.log_probabilities, proposals[0], identities[1], identities[2]);
    const std::array ids{r.one(q::c::Select{}, {selected.accepted, proposals[0], selected.final_id}), selected.final_id};
    const auto no = r.cast(r.scalar(0), q::Type::kBool);
    auto valid = r.one(q::c::Select{}, {p.valid, distributions[0].valid, no});
    valid = r.one(q::c::Select{}, {valid, bonus.valid, no});
    valid = r.one(q::c::Select{}, {valid, selected.valid, no});
    const std::array accepts{selected.accepted}, validity{valid};
    return {r.reshape(r.concat(ids, 0, r.dense_spec(q::Type::kInt32, {q::f(2)})), {q::f(2), q::f(1)}),
        r.concat(accepts, 0, r.dense_spec(q::Type::kBool, {q::f(1)})),
        r.concat(validity, 0, r.dense_spec(q::Type::kBool, {q::f(1)}))};
  }();
  // Match the real Family: its token packet is materialized before publication.
  const std::array ports{r.copy(decision.ids), r.host_predicate(decision.accepted),
      r.host_predicate(decision.valid), r.concat(proposals, 0,
          r.dense_spec(q::Type::kInt32, {q::f(static_cast<std::int64_t>(count))}))};
  for (const auto port : ports) r.publish(port);
  const auto graph = q::take(std::move(r.builder).finish());
  const std::array shape{q::g::ShapeBinding{r.last_row, 0}};
  const std::array variants{q::g::ExactShapeVariant{shape}};
  auto compiled = q::take(model.execution.compile(graph, assembly.constants(), assembly.options({
      .shape_policy = q::g::CompileShapePolicy::kFiniteExact, .exact_variants = variants,
      .constants = assembly.constants(), .resource_budget = model.execution_budget})));
  compiled.executable = q::take(compiled.executable.exact_variant(0));
  compiled.report = q::take(compiled.report.exact_variant(0));
  const auto queues = q::take(i::prepare_queue_bindings(model.execution.queue_bindings(), compiled.report));
  std::array<q::PreparedReadback, 4> outputs;
  for (std::size_t at = 0; at < ports.size(); ++at)
    outputs[at] = q::take(q::PreparedReadback::prepare(compiled, ports[at], model.execution.storage_domains(), model.allocation_budget));
  auto target_values = model.caller_tensor(model.execution.device(), q::dense(q::Type::kFloat32, {static_cast<std::int64_t>(count + 1), 4}));
  auto q_values = model.caller_tensor(model.execution.device(), q::dense(q::Type::kFloat32, {static_cast<std::int64_t>(count), 4}));
  auto key_values = model.caller_tensor(model.execution.device(), q::dense(q::Type::kUInt32, {static_cast<std::int64_t>(2 * count + 1), 4}));
  std::vector<float> target_raw, q_raw;
  for (std::size_t at = 0; at <= count; ++at) for (const auto value : p[at]) target_raw.push_back(static_cast<float>(std::log(value)));
  for (std::size_t at = 0; at < count; ++at) for (const auto value : qrows[at]) q_raw.push_back(static_cast<float>(std::log(value)));
  target_values.write<float>(target_raw); q_values.write<float>(q_raw);
  auto invocation = q::take(compiled.executable.create_invocation({.resources = q::g::ResourceOptions{}}));
  std::array<std::size_t, 4> prefix_counts{};
  std::size_t cases = 0;
  for (std::size_t attempt = 0; attempt < 4096 && cases < 128; ++attempt) {
    const i::SamplingCursor cursor{0x123456780000002aULL + attempt, 7 + attempt % 17,
        i::SamplingCursor::maximum_coordinate - 2 * count - 1};
    std::vector<std::uint32_t> words;
    for (std::size_t at = 0; at <= 2 * count; ++at) {
      const auto key = cursor.identity(at); words.insert(words.end(), key.begin(), key.end());
    }
    std::vector<std::int32_t> expected_ids(count + 1), expected_proposals(count);
    std::vector<std::uint8_t> expected_accepts(count);
    bool separated = true;
    for (std::size_t at = 0; at < count; ++at) {
      const auto proposal = draw(qrows[at], cursor.identity(2 * at));
      expected_proposals[at] = proposal.first;
      const auto ratio = (std::min)(1., p[at][proposal.first] / qrows[at][proposal.first]);
      const std::array acceptance{ratio, 1 - ratio};
      const auto accepted = draw(acceptance, cursor.identity(2 * at + 1));
      expected_accepts[at] = accepted.first == 0;
      Row final_weights = p[at + 1];
      if (!expected_accepts[at]) for (std::size_t column = 0; column < final_weights.size(); ++column)
        final_weights[column] = (std::max)(0., p[at][column] - qrows[at][column]);
      const auto final = draw(final_weights, cursor.identity(2 * count));
      expected_ids[at] = expected_accepts[at] ? proposal.first : final.first;
      if (at + 1 == count) expected_ids[count] = final.first;
      separated = separated && proposal.second >= .05 && accepted.second >= .05 && final.second >= .05;
    }
    // Select the finite corpus before GPU execution, using the existing .05
    // score-margin rule; never discard a disagreeing device observation.
    if (!separated) continue;
    q::check(invocation.reset());
    q::check(invocation.prepare_resources());
    key_values.write<std::uint32_t>(words);
    q::check(invocation.bind(target, target_values.read()));
    q::check(invocation.bind(raw_q, q_values.read())); q::check(invocation.bind(keys, key_values.read()));
    for (auto &output : outputs) q::check(output.bind(invocation));
    q::check(invocation.prepare_replay(i::queue_span(queues), {.mode = q::g::ReplayMode::kCommands, .host_wait = q::g::HostWaitPolicy::kAllow}));
    auto done = q::take(invocation.submit(i::queue_span(queues))); q::check(done.wait());
    std::vector<std::int32_t> ids(count + 1), proposals_read(count);
    std::vector<std::uint8_t> accepts(count), valid(1); std::vector<float> scratch(count);
    q::check(outputs[0].read(invocation, std::as_writable_bytes(std::span(ids))));
    q::check(outputs[1].predicates(invocation, accepts, scratch));
    q::check(outputs[2].predicates(invocation, valid, scratch));
    q::check(outputs[3].read(invocation, std::as_writable_bytes(std::span(proposals_read))));
    require(valid[0] == 1 && ids == expected_ids && accepts == expected_accepts && proposals_read == expected_proposals,
        "compound q/accept/residual/bonus differs from independent Philox/Gumbel equations");
    ++prefix_counts[static_cast<std::size_t>(std::find(accepts.begin(), accepts.end(), 0) - accepts.begin())];
    ++cases;
  }
  require(cases == 128 && std::ranges::all_of(std::span(prefix_counts).first(count + 1),
      [](auto n) { return n != 0; }), "compound finite corpus did not cover every accepted prefix");
  std::cout << "PASS independent_sampled_chain_math proposals=" << count << " cases=" << cases
            << " every_prefix=covered proposal_rng=equal accept_rng=equal shared_final_rng=equal\n" << std::flush;
}
} // namespace qwen_sampled_chain_math_case
