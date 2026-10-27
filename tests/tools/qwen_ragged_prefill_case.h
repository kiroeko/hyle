#pragma once

#include "qwen_packed_prefill_case.h"

// One finite real-Model journey, sharing the existing complete-state oracle.
namespace qwen_ragged_prefill_case {
namespace p = qwen_packed_prefill_case;
namespace q = qwen35;
namespace h = qwen_host_image_case;
using h::require;

struct BatchReport final {
  std::array<std::size_t, 2> order{}, lengths{};
  std::size_t active = 0, physical = 0, committed = 0;
  bool state_only = false, member_cancelled = false, retirement_deferred = false;
  std::array<p::StepReport, 2> members;
};

inline h::ImageBytes snapshot_image(q::Model &model, q::cache::SnapshotRef snapshot,
    const std::shared_ptr<hyle::inference::HostStateStorage> &storage) {
  auto reader = model.create_session({}, std::move(snapshot));
  auto result = p::capture(model, *reader, storage);
  h::retire(reader); return result;
}

inline int run(const std::filesystem::path &input, q::BackendKind kind) {
  auto exact_source = q::prepare_weight_source(input / "weights.tsv", false);
  auto packed_source = q::prepare_weight_source(input / "weights.tsv", false);
  const auto geometry = exact_source.roles.geometry;
  require(geometry.context_limit >= 16 && geometry.vocabulary >= 16 && geometry.recurrent_layers() &&
      geometry.attention_layers(), "ragged prefill requires the cached C16 GA fixture");
  const auto plan = q::cache::plan(geometry, exact_source.cache_row_bytes, 16, 4, 8,
      q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 2, 1, false, true);
  const q::Model::Sampling sampler{.enabled = true, .policy = {1.F, 8, .9F}, .seed = 123};
  q::Model exact(std::move(exact_source), geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands,
      SIZE_MAX, sampler, 0, {.primary = kind});
  q::Model packed(std::move(packed_source), geometry, 16, plan, {}, false, q::ReplayPolicy::kCommands,
      SIZE_MAX, sampler, 0, {.primary = kind});
  const std::array exact_rows{std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{8}};
  const std::array state_rows{std::size_t{1}, std::size_t{3}};
  const std::array capacities{std::size_t{8}};
  exact.prepare_family(exact_rows);
  exact.prepare_family(state_rows, false, 1, q::GraphRole::kText, false, q::TextCompletion::kStateOnly);
  for (const auto rows : exact_rows) exact.plan_replay_uses(rows, 1, 8);
  for (const auto rows : state_rows) exact.plan_replay_uses(rows, 1, 2, q::GraphRole::kText, q::TextCompletion::kStateOnly);
  for (const auto completion : {q::TextCompletion::kStateOnly, q::TextCompletion::kEmitNext}) {
    require(packed.prepare_prefill_family(capacities, completion, 0, 2),
        "selected route did not prepare its real B2 packed family");
    packed.plan_replay_uses(8, 2, 8, q::GraphRole::kText, completion, 0, true);
  }
  exact.prepare_execution(exact_rows);
  exact.prepare_execution(state_rows, 1, q::GraphRole::kText, q::TextCompletion::kStateOnly);
  for (const auto completion : {q::TextCompletion::kStateOnly, q::TextCompletion::kEmitNext})
    packed.prepare_prefill_execution(capacities, completion, 0, 2);
  auto exact_storage = q::take(hyle::inference::HostStateStorage::create(exact.execution, 1, 4u << 20));
  auto packed_storage = q::take(hyle::inference::HostStateStorage::create(packed.execution, 1, 4u << 20));
  std::array<std::shared_ptr<q::Session>, 2> a, b;
  for (std::size_t member = 0; member < a.size(); ++member) {
    const q::cache::SamplingCursor cursor{123, member + 7, 0};
    a[member] = exact.create_session(capacities, {}, q::GraphRole::kText, cursor);
    b[member] = packed.create_session(capacities, {}, q::GraphRole::kText, cursor);
  }
  std::array<const q::Phase *, 2> carriers{};
  std::vector<BatchReport> reports;
  const auto advance = [&](std::array<std::size_t, 2> order, std::array<std::size_t, 2> lengths,
      q::TextCompletion completion, bool cancel_first) {
    BatchReport report; report.order = order; report.lengths = lengths;
    report.active = lengths[0] + lengths[1]; report.physical = 16;
    report.state_only = completion == q::TextCompletion::kStateOnly; report.member_cancelled = cancel_first;
    std::array<q::cache::SnapshotRef, 2> old_a, old_b;
    std::array<h::ImageBytes, 2> before_a, before_b;
    std::array<std::vector<std::int32_t>, 2> ids;
    std::array<q::EvaluationRequest, 2> requests;
    std::array<q::EvaluationOutput, 2> outputs;
    std::array<q::EvaluationDisposition, 2> dispositions{};
    std::array<std::size_t, 2> observed{};
    for (std::size_t j = 0; j < order.size(); ++j) {
      const auto member = order[j]; auto &control = *a[member]; auto &candidate = *b[member];
      require(control.length() == candidate.length() && control.head.epoch() == candidate.head.epoch() &&
          control.head->sampling == candidate.head->sampling, "ragged members do not share their original boundary");
      old_a[j] = control.checkpoint(); old_b[j] = candidate.checkpoint();
      before_a[j] = p::capture(exact, control, exact_storage); before_b[j] = p::capture(packed, candidate, packed_storage);
      ids[j].resize(lengths[j]);
      for (std::size_t row = 0; row < lengths[j]; ++row)
        ids[j][row] = static_cast<std::int32_t>(1 + (control.length() + row + 7 * member) % 15);
      requests[j] = {b[member], ids[j]};
      outputs[j].session = b[member]; outputs[j].disposition = &dispositions[j];
      auto &facts = report.members[j];
      facts.sequence = member; facts.prefix = control.length(); facts.active = lengths[j]; facts.physical = 8;
      facts.prefill = true;
      if (!(cancel_first && j == 0)) {
        exact.prepare_evaluation(control, ids[j], nullptr, q::GraphRole::kText, completion);
        exact.submit_evaluation(control);
        facts.exact_next = q::finish_evaluation(exact, control, "ragged-exact", {}, false);
      } else facts.exact_next = control.head->next_id;
    }
    packed.prepare_prefill_evaluations(requests, nullptr, q::GraphRole::kText, completion);
    require(packed.active_step && packed.active_step->members.size() == 2 &&
        packed.active_step->phase->query_rows == 8 && packed.active_step->phase->family->requests == 2 &&
        packed.active_step->phase->family->packed_prefill && packed.active_step->active_input_rows == report.active,
        "ragged prefill did not submit its original B2/C8 carrier with the true sum of Li");
    const auto signature = report.state_only ? 0u : 1u;
    if (carriers[signature]) require(carriers[signature] == packed.active_step->phase,
        "ragged member lengths or order silently compiled another physical carrier");
    else carriers[signature] = packed.active_step->phase;
    for (std::size_t j = 0; j < order.size(); ++j)
      require(packed.active_step->members[j].session == b[order[j]] &&
          packed.active_step->members[j].input_tokens == lengths[j], "ragged member identity or real input length was lost");
    packed.submit_evaluations();
    if (cancel_first) {
      const auto stopped = b[order[0]]->stop();
      require(stopped.ok() || (stopped.IsCancelled() && !stopped.has_native_error()), "ragged member cancellation reported a native failure");
      require(packed.active_step && packed.active_step->submitted && packed.active_step->has_completion() &&
          packed.active_step->discard_stopped_members && !packed.active_step->cancelled && !b[order[1]]->stopping,
          "single ragged cancellation stopped the original peer or replaced the episode owner");
      const auto pending = b[order[0]]->try_retire(); q::check(pending.status);
      report.retirement_deferred = !pending.retired;
      require(report.retirement_deferred, "cancelled member escaped before original peer publication and retirement");
    }
    std::array<std::int32_t, 2> next{};
    const auto observe = [&](std::size_t j, const q::CompletedEvaluationView &view) {
      require(j < order.size() && !(cancel_first && j == 0) && view.member_index == j && view.requests == 2 &&
          view.observation.input_tokens == lengths[j] && view.observation.graph_input_tokens == lengths[j],
          "ragged observer published a cancelled member or counted another member's rows");
      ++observed[j];
    };
    packed.finish_evaluations(outputs, next, observe);
    require(!packed.active_step && !packed.carrier_borrowed && !b[0]->pending && !b[1]->pending,
        "ragged finish did not retire the original submitted batch");
    for (std::size_t j = 0; j < order.size(); ++j) {
      const auto member = order[j]; const bool cancelled = cancel_first && j == 0;
      auto &control = *a[member]; auto &candidate = *b[member]; auto &facts = report.members[j];
      const auto committed = cancelled ? 0u : lengths[j]; report.committed += committed;
      require(candidate.length() == facts.prefix + committed && control.length() == candidate.length() &&
          candidate.head.epoch() == control.head.epoch() &&
          candidate.head->sampling == before_b[j].boundary.sampling.advanced(!cancelled && !report.state_only ? 1u : 0u) &&
          candidate.head->sampling == control.head->sampling && observed[j] == (cancelled ? 0u : 1u) &&
          dispositions[j] == (cancelled ? q::EvaluationDisposition::kCancelled : q::EvaluationDisposition::kPublished),
          "ragged finish published padding/cancelled rows, wrong RNG/epoch or duplicate observations");
      facts.epoch = static_cast<std::size_t>(candidate.head.epoch()); facts.bucket_next = candidate.head->next_id;
      if (!cancelled) require(next[j] == facts.bucket_next, "ragged finish returned a different committed next ID");
      if (report.state_only) require(facts.exact_next == -1 && facts.bucket_next == -1,
          "ragged state-only prefill sampled or published a next token");
      const auto actual = cancelled ? snapshot_image(packed, candidate.checkpoint(), packed_storage)
                                    : p::capture(packed, candidate, packed_storage);
      p::compare(exact, p::capture(exact, control, exact_storage), actual, facts);
      p::check_old(exact, old_a[j], before_a[j], exact_storage); p::check_old(packed, old_b[j], before_b[j], packed_storage);
    }
    reports.push_back(std::move(report));
  };
  advance({0, 1}, {1, 3}, q::TextCompletion::kStateOnly, false);
  advance({1, 0}, {3, 1}, q::TextCompletion::kEmitNext, false);
  advance({0, 1}, {8, 2}, q::TextCompletion::kEmitNext, false);
  advance({1, 0}, {1, 3}, q::TextCompletion::kEmitNext, true);
  require(b[1]->stopping && !b[0]->stopping && b[0]->length() == 13 && b[1]->length() == 8,
      "ragged cancellation changed the wrong member's committed prefix");
  h::retire(b[1]); h::retire(a[1]);
  p::StepReport continued;
  {
  const auto old_a = a[0]->checkpoint(), old_b = b[0]->checkpoint();
  const auto before_a = p::capture(exact, *a[0], exact_storage), before_b = p::capture(packed, *b[0], packed_storage);
  const auto pending = a[0]->head->next_id;
  continued = p::StepReport{0, a[0]->length(), 1, 1};
  continued.exact_next = q::evaluate(exact, *a[0], std::span(&pending, 1), "ragged-peer-exact", {}, false);
  continued.bucket_next = q::evaluate(packed, *b[0], std::span(&pending, 1), "ragged-peer-next", {}, false);
  require(a[0]->length() == 14 && b[0]->length() == 14 && a[0]->head.epoch() == b[0]->head.epoch() &&
      a[0]->head->sampling == before_a.boundary.sampling.advanced(1) &&
      b[0]->head->sampling == a[0]->head->sampling && !b[0]->stopping && !b[0]->failed,
      "surviving ragged member could not continue through the original T1 path");
  continued.epoch = static_cast<std::size_t>(b[0]->head.epoch());
  p::compare(exact, p::capture(exact, *a[0], exact_storage), p::capture(packed, *b[0], packed_storage), continued);
  p::check_old(exact, old_a, before_a, exact_storage); p::check_old(packed, old_b, before_b, packed_storage);
  h::retire(a[0]); h::retire(b[0]);
  } // Drop the two explicit final checkpoints before checking zero live versions.
  require(!exact.snapshot_versions.live && !packed.snapshot_versions.live &&
      !exact.active_step && !packed.active_step && !exact.successor_step && !packed.successor_step &&
      !exact.carrier_borrowed && !packed.carrier_borrowed &&
      !exact_storage->live_images() && !packed_storage->live_images() &&
      exact.page_inventory.free_ids.size() == plan.max_pages && packed.page_inventory.free_ids.size() == plan.max_pages,
      "ragged member retirement retained a state version, native episode, host image or page permission");
  bool equal = continued.numerically_equivalent, byte_equal = continued.payloads_equal;
  std::size_t active = 0, physical = 0, committed = 0;
  for (const auto &report : reports) {
    active += report.active; physical += report.physical; committed += report.committed;
    for (const auto &member : report.members) { equal = equal && member.numerically_equivalent; byte_equal = byte_equal && member.payloads_equal; }
  }
  require(active == 22 && physical == 64 && committed == 21, "ragged actual/physical/cancelled work accounting drifted");
  std::cout << "{\"backend\":\"" << h::backend_name(kind)
      << "\",\"mode\":\"ragged_prefill\",\"synthetic_untrained\":true,\"cold_rows_per_member\":8,\"width\":2"
      << ",\"actual_prefill_rows\":" << active << ",\"physical_prefill_rows\":" << physical
      << ",\"committed_prefill_rows\":" << committed << ",\"cancelled_prefill_rows\":1,\"peer_decode_rows\":1,\"batches\":[";
  for (std::size_t at = 0; at < reports.size(); ++at) {
    const auto &report = reports[at];
    std::cout << (at ? "," : "") << "{\"order\":[" << report.order[0] << ',' << report.order[1]
        << "],\"Li\":[" << report.lengths[0] << ',' << report.lengths[1] << "],\"active_rows\":" << report.active
        << ",\"physical_rows\":" << report.physical << ",\"committed_rows\":" << report.committed
        << ",\"state_only\":" << (report.state_only ? "true" : "false")
        << ",\"member_cancelled\":" << (report.member_cancelled ? "true" : "false")
        << ",\"retirement_deferred\":" << (report.retirement_deferred ? "true" : "false") << ",\"members\":[";
    p::write_step(report.members[0]); std::cout << ','; p::write_step(report.members[1]); std::cout << "]}";
  }
  std::cout << "],\"peer_decode\":"; p::write_step(continued);
  std::cout << ",\"old_checkpoints_byte_unchanged\":true,\"rng_and_epochs_equal\":true"
      << ",\"final_versions\":0,\"final_images\":0,\"final_native_episodes\":0"
      << ",\"payloads_byte_equal\":" << (byte_equal ? "true" : "false")
      << ",\"numerically_equivalent\":" << (equal ? "true" : "false")
      << ",\"gdn_state_absolute_budget\":" << p::gdn_state_absolute_budget
      << ",\"numerical_budget_basis\":\"packed_sequence_case.h independent FP64 component oracle\",\"performance_claim\":false"
      << ",\"status\":\"" << (equal ? "PASS" : "VALUE_OR_CONTRACT_DIFFERENCE") << "\"}\n";
  require(equal, "ragged versus exact complete state exceeded the original numerical or byte/metadata contract");
  return 0;
}
} // namespace qwen_ragged_prefill_case
