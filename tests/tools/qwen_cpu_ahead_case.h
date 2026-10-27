#pragma once

// Included after the original product Model; shared finite ordinary consumer.
namespace qwen_cpu_ahead_case {

namespace q = qwen35;
namespace i = hyle::inference;

inline void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

struct Arm final {
  std::array<std::vector<std::int32_t>, 2> ids;
  std::array<std::vector<q::GenerationCommitRecord>, 2> records;
  std::array<q::GenerationBoundary, 2> final;
  std::size_t prepared = 0, hits = 0, discarded = 0;
};

inline Arm run_arm(q::Model &model, bool consume) {
  model.discard_decode_ahead();
  const auto prepared = model.cpu_ahead_prepared, hits = model.cpu_ahead_hits, discarded = model.cpu_ahead_discarded;
  const std::array turns{i::GenerationTurn{{1, 2}, 5}};
  std::vector<q::GenerationRequest> requests;
  requests.reserve(2);
  for (std::size_t j = 0; j < 2; ++j) {
    requests.emplace_back(j ? "B" : "A", turns[0].prompt, std::filesystem::path{}, 5);
    auto &request = requests.back();
    request.progress.bind_turns(turns);
    request.progress.initial_sampling = {123, 100 + j, 0};
    request.ledger = std::make_unique<q::GenerationLedger>(q::generation_record_capacity(turns), 1);
    request.round_emitted.resize(4);
    for (auto &values : request.round_emitted) values.reserve(1);
    request.turn_observations.resize(1);
    request.warm_inputs.push_back(std::make_shared<std::vector<std::int32_t>>(turns[0].prompt));
    request.prompt = request.progress.begin_turn(*request.warm_inputs[0]);
  }
  std::size_t calls = 0;
  for (std::size_t round = 0; round < 5; ++round) {
    std::vector<q::PreparedGenerationCall> batch;
    const auto add = [&](std::size_t j) {
      auto &request = requests[j];
      batch.push_back(q::prepare_generation_call(model, request, request.progress.output_limit(), nullptr, requests));
    };
    if (round == 3) { add(1); add(0); } // Same original owners, different member order.
    else if (round == 4) add(1);       // A is paused-cancelled after round 3.
    else { add(0); add(1); }
    q::execute_generation_calls(model, batch, false, nullptr, requests);
    ++calls;
    if (!consume) model.discard_decode_ahead(); // Control keeps the original synchronous path.
    if (round == 2 && consume)
      require(model.cpu_ahead_hits - hits == 1, "the real B2 successor did not consume prepared metadata");
    if (round == 3) {
      require(!model.active_step && !model.carrier_borrowed && requests[0].session->head->sampling.draw == 4,
          "member-change control did not reach the original successful boundary");
      q::check(requests[0].session->stop());
    }
  }
  Arm result;
  require(calls == 5 && !model.active_step && !model.carrier_borrowed,
      "ahead consumer changed the number or retirement of original Graph calls");
  for (std::size_t j = 0; j < 2; ++j) {
    const auto &request = requests[j];
    const auto expected = j ? std::size_t{5} : std::size_t{4};
    require(request.progress.generated.size() == expected && request.session->head->sampling.draw == expected &&
        request.ledger->committed_count == expected,
        "lookahead or cancellation advanced an unexpected output, RNG draw or ledger record");
    result.ids[j] = request.progress.generated;
    result.final[j] = q::generation_boundary(*request.session->head);
    result.records[j].assign(request.ledger->records.begin(), request.ledger->records.begin() + expected);
  }
  result.prepared = model.cpu_ahead_prepared - prepared;
  result.hits = model.cpu_ahead_hits - hits;
  result.discarded = model.cpu_ahead_discarded - discarded;
  require(result.prepared == 3 && result.hits == (consume ? 3u : 0u) && result.discarded == (consume ? 1u : 3u),
      "ahead metadata did not consume the real reordered/subset owners or discard the stopped member");
  require(!model.decode_ahead || !model.decode_ahead->ready, "last singleton retained an obsolete B2 preparation");
  requests.clear();
  require(!model.snapshot_versions.live, "weak metadata identities kept a completed Session/head alive");
  return result;
}

struct Report final {
  Arm control, candidate;
  std::size_t compared = 0;
  double prepare_seconds = 0;
};

inline Report exercise(const std::filesystem::path &input, i::BackendKind kind) {
  auto source = q::prepare_weight_source(input / "weights.tsv", false);
  const auto geometry = source.roles.geometry;
  require(geometry.context_limit >= 16 && geometry.vocabulary > 2,
      "ahead consumer requires a local fixture with C16 and at least three tokens");
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 16, 2, 2,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 2, 1, false, true);
  q::Model model(std::move(source), geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands, SIZE_MAX,
      {.enabled = true, .policy = {1.F, 8, .9F}, .seed = 123}, 0, {.primary = kind});
  const std::array shapes{std::size_t{1}, std::size_t{2}};
  for (const auto width : {std::size_t{1}, std::size_t{2}}) {
    model.prepare_family(shapes, false, width);
    for (const auto rows : shapes) model.plan_replay_uses(rows, width, 12);
  }
  model.freeze_shape_plan();
  for (const auto width : {std::size_t{1}, std::size_t{2}}) model.prepare_execution(shapes, width);
  auto control = run_arm(model, false);
  auto candidate = run_arm(model, true);
  require(control.ids == candidate.ids && control.final == candidate.final,
      "ahead changed original token output, visible boundary or RNG identity");
  std::size_t compared = 0;
  for (std::size_t j = 0; j < 2; ++j) {
    require(control.records[j].size() == candidate.records[j].size(), "ahead changed ledger length");
    for (std::size_t row = 0; row < control.records[j].size(); ++row) {
      const auto &a = control.records[j][row], &b = candidate.records[j][row];
      require(a.same_result(b) && a.committed == b.committed && a.user_output == b.user_output &&
          a.imported == b.imported && a.turn == b.turn && std::ranges::equal(a.ids(), b.ids()),
          "ahead changed a non-time input, decision, member or state publication fact");
      ++compared;
    }
  }
  return {std::move(control), std::move(candidate), compared, model.cpu_ahead_prepare_seconds};
}

inline void write_json(const Report &report, i::BackendKind kind) {
  const auto &control = report.control, &candidate = report.candidate;
  const auto compared = report.compared;
  std::cout << "{\"status\":\"PASS\",\"backend\":\"" << (kind == i::BackendKind::kCpu ? "cpu" : "cuda")
      << "\",\"physical_calls\":10,\"outputs\":18,"
      "\"records_compared\":" << compared << ",\"candidate_prepared\":" << candidate.prepared
      << ",\"candidate_hits\":" << candidate.hits << ",\"candidate_discarded\":" << candidate.discarded
      << ",\"control_hits\":" << control.hits << ",\"owner_epoch_length_order_checked\":true,"
      "\"reorder_consumed\":true,\"paused_cancel_subset_consumed\":true,\"final_versions\":0,"
      "\"cpu_prepare_seconds\":" << report.prepare_seconds << ",\"ids\":[";
  for (std::size_t j = 0; j < 2; ++j) {
    if (j) std::cout << ',';
    std::cout << '[';
    for (std::size_t n = 0; n < candidate.ids[j].size(); ++n) { if (n) std::cout << ','; std::cout << candidate.ids[j][n]; }
    std::cout << ']';
  }
  std::cout << "]}";
}
} // namespace qwen_cpu_ahead_case
