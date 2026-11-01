#pragma once

#include <iomanip>
#include <limits>

#include "qwen_host_image_case.h"
#include "qwen_packed_prefill_case.h"

// Diagnostic only: identical teacher-forced inputs, two original Sessions and
// one Model/weight owner. Optional greedy-gap acceptance grants no payload
// tolerance and is never a benchmark or a general model-quality verdict.
namespace qwen_prefill_diagnostic_case {
namespace q = qwen35;
namespace h = qwen_host_image_case;
namespace p = qwen_packed_prefill_case;
using h::require;

struct Options final {
  std::size_t capacity = 2048, maximum = 512, outputs = 256;
  bool tiers = true, shared = true, acceptance = false;
};

struct Output final {
  std::int32_t next = -1;
  std::size_t physical = 0;
  std::vector<float> logits;
};

inline Output evaluate(q::Model &model, const std::shared_ptr<q::Session> &session,
    std::span<const std::int32_t> ids, bool packed) {
  if (packed) {
    const std::array requests{q::EvaluationRequest{session, ids}};
    model.prepare_prefill_evaluations(requests);
  } else model.prepare_evaluation(*session, ids);
  require(bool(model.active_step) && model.active_step->members.size() == 1 &&
      model.active_step->input_tokens == ids.size() &&
      q::packs_prefill(model.active_step->phase->family->packing) == packed,
      "diagnostic did not prepare the requested exact/packed R1 route");
  Output result;
  result.physical = model.active_step->phase->query_rows;
  try {
    model.submit_evaluation(*session);
    const std::array outputs{q::EvaluationOutput{.session = session, .capture_tensors = true}};
    model.finish_evaluations(outputs, std::span(&result.next, 1),
        [&](std::size_t member, const q::CompletedEvaluationView &view) {
          require(member == 0 && view.requests == 1 &&
              view.logits.description() == q::dense(q::Type::kFloat32, {1, model.vocabulary}),
              "diagnostic needs the original complete F32 last-row logits");
          result.logits.resize(static_cast<std::size_t>(model.vocabulary));
          q::check(model.execution.read(view.logits.storage(), std::as_writable_bytes(std::span(result.logits))));
        });
  } catch (...) { (void)model.abort_episode(); throw; }
  require(!model.active_step && !model.carrier_borrowed && !session->pending,
      "diagnostic output escaped its original native retirement");
  return result;
}

inline h::ImageBytes capture(q::Model &model, q::Session &session,
    const std::shared_ptr<hyle::inference::HostStateStorage> &storage) {
  const auto image = model.export_session(session, storage);
  return h::read_image(*image);
}

inline std::array<std::size_t, 2> top2(std::span<const float> values) {
  require(values.size() >= 2, "diagnostic requires at least two vocabulary logits");
  std::array<std::size_t, 2> result{SIZE_MAX, SIZE_MAX};
  for (std::size_t index = 0; index < values.size(); ++index) {
    require(std::isfinite(values[index]), "diagnostic observed nonfinite full-vocabulary logits");
    if (result[0] == SIZE_MAX || values[index] > values[result[0]]) {
      result[1] = result[0]; result[0] = index;
    } else if (result[1] == SIZE_MAX || values[index] > values[result[1]]) result[1] = index;
  }
  return result;
}

inline bool report_logits(std::string_view name, std::size_t output, const Output &a, const Output &b) {
  require(a.logits.size() == b.logits.size(), "diagnostic logit extents differ");
  const auto at = top2(a.logits), bt = top2(b.logits);
  require(a.next == static_cast<std::int32_t>(at[0]) && b.next == static_cast<std::int32_t>(bt[0]),
      "native greedy ID differs from the independent full-logit argmax");
  double maximum = 0, squared = 0;
  std::size_t unequal = 0;
  for (std::size_t index = 0; index < a.logits.size(); ++index) {
    unequal += std::memcmp(&a.logits[index], &b.logits[index], sizeof(float)) != 0;
    const auto delta = double(a.logits[index]) - double(b.logits[index]);
    maximum = (std::max)(maximum, std::abs(delta)); squared += delta * delta;
  }
  std::cout << "logits name=" << name << " output=" << output
      << " exact_top1=" << at[0] << " exact_score1=" << a.logits[at[0]]
      << " exact_top2=" << at[1] << " exact_score2=" << a.logits[at[1]]
      << " exact_margin=" << double(a.logits[at[0]]) - a.logits[at[1]]
      << " packed_top1=" << bt[0] << " packed_score1=" << b.logits[bt[0]]
      << " packed_top2=" << bt[1] << " packed_score2=" << b.logits[bt[1]]
      << " packed_margin=" << double(b.logits[bt[0]]) - b.logits[bt[1]]
      << " packed_at_exact_top1=" << b.logits[at[0]]
      << " exact_at_packed_top1=" << a.logits[bt[0]]
      << " unequal_f32=" << unequal << " max_absolute=" << maximum
      << " rms=" << std::sqrt(squared / a.logits.size()) << '\n';
  return a.next == b.next;
}

inline bool report_state(q::Model &model, std::string_view name, std::size_t output,
    const h::ImageBytes &a, const h::ImageBytes &b, bool detail) {
  require(a.identity == b.identity && a.layout == b.layout && a.total_bytes == b.total_bytes &&
      a.payloads.size() == b.payloads.size(), "diagnostic complete initialized state layouts differ");
  require(a.boundary.length == b.boundary.length && a.boundary.sampling == b.boundary.sampling &&
      a.boundary.evaluated_sampling == b.boundary.evaluated_sampling && a.boundary.draft == b.boundary.draft &&
      a.boundary.has_carry == b.boundary.has_carry && a.boundary.has_prior == b.boundary.has_prior,
      "diagnostic advanced padding, RNG or a different logical state extent");
  std::array<std::size_t, 8> changed_parts{}, changed_bytes{};
  std::array<double, 8> maxima{};
  bool equal = true;
  for (std::size_t index = 0; index < a.payloads.size(); ++index) {
    if (a.payloads[index] == b.payloads[index]) continue;
    equal = false;
    const auto difference = p::compare_payload(model, a.layout[index], a.payloads[index], b.payloads[index]);
    const auto role = static_cast<std::size_t>(a.layout[index].role);
    ++changed_parts[role]; changed_bytes[role] += difference.different_bytes;
    maxima[role] = (std::max)(maxima[role], difference.maximum_absolute);
    if (detail) std::cout << "payload name=" << name << " output=" << output << " role=" << role
        << " layer=" << a.layout[index].layer << " page=" << a.layout[index].page
        << " rows=" << a.layout[index].initialized_rows << " bytes=" << a.layout[index].bytes
        << " first_byte=" << difference.first_byte << " different_bytes=" << difference.different_bytes
        << " max_absolute=" << difference.maximum_absolute << " max_relative=" << difference.maximum_relative
        << " nonfinite_differences=" << difference.nonfinite_differences << '\n';
  }
  std::cout << "state name=" << name << " output=" << output << " length=" << a.boundary.length
      << " boundary_equal=" << h::same_boundary(a.boundary, b.boundary)
      << " payload_bytes_equal=" << equal << " total_bytes=" << a.total_bytes;
  for (std::size_t role = 0; role < changed_parts.size(); ++role) if (changed_parts[role])
    std::cout << " role" << role << "_parts=" << changed_parts[role]
        << " role" << role << "_different_bytes=" << changed_bytes[role]
        << " role" << role << "_max_absolute=" << maxima[role];
  std::cout << '\n';
  return equal;
}

inline int run(const std::filesystem::path &input, const std::filesystem::path &schedule,
    const std::filesystem::path &teacher_file, q::BackendKind kind, Options options = {}) {
  require(kind == q::BackendKind::kCuda, "prefill diagnostic currently requires the original CUDA route");
  auto source = q::prepare_weight_source(input / "weights.tsv", false);
  const auto geometry = source.roles.geometry;
  require(options.capacity && options.capacity <= static_cast<std::size_t>(geometry.context_limit) &&
      options.maximum && options.maximum <= options.capacity && options.maximum <= 512 && options.outputs,
      "invalid explicit diagnostic context, fixed capacity or output bound");
  const auto turns = q::benchmark::read_generation_schedule(schedule, geometry.vocabulary, options.capacity, 64, false);
  const auto teacher = q::read_ids(teacher_file, geometry.vocabulary, options.capacity);
  require(!options.acceptance || teacher.size() >= options.outputs,
      "acceptance requires the complete requested oracle token prefix");
  const auto limit = (std::min)(options.outputs, teacher.size());
  require(limit != 0, "diagnostic teacher sequence is empty");
  std::vector<std::size_t> exact_rows{1}, capacities;
  std::size_t planned_outputs = 0;
  for (std::size_t turn = 0; turn < turns.size() && planned_outputs < limit; ++turn) {
    const auto rows = turns[turn].prompt.size() + (turn ? 1u : 0u);
    require(rows <= options.maximum, "diagnostic needs one actual prefill chunk within its fixed capacity");
    exact_rows.push_back(rows); planned_outputs += turns[turn].max_outputs;
  }
  require(planned_outputs >= limit, "diagnostic teacher prefix exceeds the supplied workload");
  std::sort(exact_rows.begin(), exact_rows.end());
  exact_rows.erase(std::unique(exact_rows.begin(), exact_rows.end()), exact_rows.end());
  if (options.tiers) for (const auto tier : {16u, 32u, 64u, 128u, 256u, 512u}) {
    const auto rows = (std::min)(std::size_t(tier), options.maximum);
    if (capacities.empty() || capacities.back() != rows) capacities.push_back(rows);
    if (rows == options.maximum) break;
  }
  else capacities.push_back(options.maximum);
  const auto plan = q::cache::plan(geometry, source.cache_row_bytes, options.capacity, 2, options.capacity,
      q::recurrent_bytes(q::state_schema_for(geometry, options.capacity), geometry), 1, 1, false, true);
  q::Model model(std::move(source), geometry, options.capacity, plan, {}, false, q::ReplayPolicy::kAuto,
      SIZE_MAX, {}, 0, {.primary = kind});
  model.prepare_family(exact_rows);
  require(model.prepare_prefill_family(capacities), "diagnostic packed prefill route is unavailable");
  for (const auto rows : exact_rows) model.plan_replay_uses(rows, 1, 2 * limit + 4);
  for (const auto rows : capacities) model.plan_replay_uses(rows, 1, 2 * limit + 4, q::GraphRole::kText,
      q::TextCompletion::kEmitNext, 0, q::RowPacking::kPrefill);
  if (options.shared) model.prepare_shared_prefill_workspace();
  model.prepare_execution(exact_rows);
  if (!options.shared) model.prepare_prefill_execution(capacities);
  auto storage = q::take(hyle::inference::HostStateStorage::create(model.execution, 2, std::size_t{512} << 20));
  const std::array initial{turns.front().prompt.size()};
  auto exact = model.create_session(initial), packed = model.create_session(initial);
  std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
      << "diagnostic ordinary_teacher_forced=1 benchmark=0 numerical_pass_budget=none shared_weights=1"
      << " mode=" << (options.acceptance ? "complete_acceptance" : "first_difference_isolation")
      << " oracle_margin_limit=0.1"
      << " role0=target_key role1=target_value role2=conv_history role3=gdn_state role4=carry_hidden role5=prior_hidden"
      << " shared_prefill_scratch=" << options.shared << " capacities=";
  for (const auto rows : capacities) std::cout << rows << ',';
  std::cout << '\n';
  std::size_t output = 0, first_greedy = SIZE_MAX, first_payload = SIZE_MAX, first_teacher = SIZE_MAX;
  std::size_t disagreements = 0, large_margin = 0, exact_teacher_mismatches = 0, packed_teacher_mismatches = 0;
  std::size_t state_checkpoints = 0, metadata_checks = 0, candidate_restores = 0;
  const auto advance = [&](std::span<const std::int32_t> ids, bool prefill, std::size_t turn) {
    const auto before_length = exact->length();
    const auto before_epoch = exact->head.epoch();
    const auto before_sampling = exact->head->sampling;
    require(packed->length() == before_length && packed->head.epoch() == before_epoch &&
        packed->head->sampling == before_sampling, "teacher-forced sessions entered different context/epoch/RNG positions");
    std::shared_ptr<const q::cache::HostImage> before;
    if (!options.acceptance) before = model.export_session(*exact, storage);
    const auto a = evaluate(model, exact, ids, false);
    const auto b = evaluate(model, packed, ids, prefill);
    require(exact->length() == before_length + ids.size() && packed->length() == exact->length() &&
        exact->head.epoch() == before_epoch + 1 && packed->head.epoch() == exact->head.epoch() &&
        exact->head->sampling == before_sampling && packed->head->sampling == before_sampling,
        "teacher-forced step advanced padding, a different epoch or greedy RNG state");
    ++metadata_checks;
    std::cout << "step output=" << output << " turn=" << turn + 1 << " prefill=" << prefill
        << " before=" << before_length << " active=" << ids.size() << " physical=" << b.physical
        << " teacher=" << teacher[output] << " exact=" << a.next << " packed=" << b.next
        << " length=" << exact->length() << " epoch=" << exact->head.epoch()
        << " rng_draw=" << exact->head->sampling.draw << " metadata_equal=1\n";
    const bool same = report_logits("stream", output, a, b);
    exact_teacher_mismatches += a.next != teacher[output];
    packed_teacher_mismatches += b.next != teacher[output];
    if (!same) {
      ++disagreements;
      const auto oracle = top2(a.logits), candidate = top2(b.logits);
      const auto oracle_gap = double(a.logits[oracle[0]]) - a.logits[oracle[1]];
      const auto candidate_gap = double(b.logits[candidate[0]]) - b.logits[candidate[1]];
      const bool within = oracle_gap < 0.1;
      large_margin += !within;
      std::cout << "argmax_mismatch output=" << output << " turn=" << turn + 1 << " prefill=" << prefill
          << " oracle=" << a.next << " candidate=" << b.next << " teacher=" << teacher[output]
          << " oracle_matches_teacher=" << (a.next == teacher[output])
          << " oracle_top1_top2_gap=" << oracle_gap << " candidate_top1_top2_gap=" << candidate_gap
          << " candidate_preference_gap=" << double(b.logits[candidate[0]]) - b.logits[oracle[0]]
          << " oracle_gap_lt_0.1=" << within << '\n';
    }
    h::ImageBytes a_state;
    if (!options.acceptance || prefill || !same || a.next != teacher[output] || b.next != teacher[output]) {
      a_state = capture(model, *exact, storage);
      const auto b_state = capture(model, *packed, storage);
      ++state_checkpoints;
      if (!report_state(model, "stream", output, a_state, b_state, prefill || !same) && first_payload == SIZE_MAX)
        first_payload = output;
    }
    if ((a.next != teacher[output] || b.next != teacher[output]) && first_teacher == SIZE_MAX) first_teacher = output;
    if (!same && first_greedy == SIZE_MAX) {
      first_greedy = output;
      // A real published image, restored only after the original candidate
      // Session retires. This is a local numerical diagnostic, not a new head
      // permission or a rewrite of the streamed state under comparison.
      if (!options.acceptance && !prefill && before_length) {
        h::retire(packed); packed = model.restore_session(before);
        ++candidate_restores;
        const auto local = evaluate(model, packed, ids, false);
        (void)report_logits("same_exact_before", output, a, local);
        (void)report_state(model, "same_exact_before", output, a_state, capture(model, *packed, storage), true);
      } else std::cout << "same_exact_before output=" << output
          << " not_run=" << (options.acceptance ? "continuous_oracle_teacher_forcing" : "prefill_difference") << '\n';
    }
    ++output;
    return options.acceptance || same;
  };
  bool keep_running = true;
  for (std::size_t turn = 0; turn < turns.size() && output < limit && keep_running; ++turn) {
    std::vector<std::int32_t> prompt;
    if (turn) prompt.push_back(teacher[output - 1]);
    prompt.insert(prompt.end(), turns[turn].prompt.begin(), turns[turn].prompt.end());
    keep_running = advance(prompt, true, turn);
    for (std::size_t token = 1; token < turns[turn].max_outputs && output < limit && keep_running; ++token) {
      const auto pending = teacher[output - 1];
      keep_running = advance(std::span(&pending, 1), false, turn);
    }
  }
  h::retire(exact); h::retire(packed);
  require(!storage->live_images() && !model.snapshot_versions.live && !model.active_step && !model.carrier_borrowed &&
      model.page_inventory.free_ids.size() == plan.max_pages,
      "diagnostic retained a StateImage, Session, page or original native episode");
  const auto index = [](std::size_t value) { return value == SIZE_MAX ? -1LL : static_cast<long long>(value); };
  const bool complete = output == limit;
  const bool accepted = complete && !exact_teacher_mismatches && !large_margin;
  std::cout << "diagnostic_completed=1 compared_outputs=" << output << " first_greedy_difference=" << index(first_greedy)
      << " first_observed_payload_difference=" << index(first_payload) << " first_teacher_mismatch=" << index(first_teacher)
      << " argmax_disagreements=" << disagreements << " oracle_gap_ge_0.1=" << large_margin
      << " exact_teacher_mismatches=" << exact_teacher_mismatches << " packed_teacher_mismatches=" << packed_teacher_mismatches
      << " metadata_checks=" << metadata_checks << " state_checkpoints=" << state_checkpoints
      << " requested_outputs=" << options.outputs << " complete_requested_prefix=" << complete
      << " candidate_restores=" << candidate_restores
      << " live_versions=" << model.snapshot_versions.live << " numerical_pass_budget=none";
  if (options.acceptance) std::cout << " greedy_gap_acceptance=" << accepted;
  std::cout << '\n';
  return options.acceptance && !accepted ? 1 : 0;
}
} // namespace qwen_prefill_diagnostic_case
