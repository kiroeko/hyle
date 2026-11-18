#pragma once

#include <array>
#include <optional>
#include <string_view>

#include "../../engine/models/qwen35/generation_ledger.h"
#include "../../engine/models/qwen35/mtp_capabilities.h"

// Test-only legacy versus fixed-width generation semantics and the retained
// numerical witness. The same-physical oracle always uses strict_equal.
namespace qwen_legacy_ledger_case {
namespace q = qwen35;
using Record = q::GenerationCommitRecord;

inline bool strict_equal(const Record &expected, const Record &actual, bool compare_second = true) {
  // Default callers, including every same-physical oracle, keep the original
  // complete diagnostic equality. The optional second-row projection does
  // not omit proposal values or any other fact. Keep this list
  // aligned with GenerationCommitRecord::same_result; do not copy/mutate a
  // record to manufacture an equal random result.
  const bool same_result = compare_second ? expected.same_result(actual) :
      expected.requests == actual.requests && expected.member_index == actual.member_index &&
      expected.role == actual.role && expected.generation == actual.generation && expected.quota == actual.quota &&
      expected.input_deferred == actual.input_deferred &&
      expected.completion == actual.completion && expected.prefill_complete == actual.prefill_complete &&
      expected.input_begin == actual.input_begin && expected.before == actual.before && expected.after == actual.after &&
      expected.accepted == actual.accepted && expected.draft_limit == actual.draft_limit &&
      expected.verified_rows == actual.verified_rows && expected.consumed_drafts == actual.consumed_drafts &&
      expected.auxiliary_graph_calls == actual.auxiliary_graph_calls && expected.draft_rows == actual.draft_rows &&
      expected.draft == actual.draft && expected.target_first == actual.target_first &&
      expected.agreed == actual.agreed && expected.accepted_draft == actual.accepted_draft &&
      expected.stopped == actual.stopped && expected.emitted == actual.emitted &&
      expected.emitted_count == actual.emitted_count && expected.accepted_inputs == actual.accepted_inputs &&
      expected.verification_inputs == actual.verification_inputs;
  return same_result && expected.committed == actual.committed &&
      expected.user_output == actual.user_output && expected.imported == actual.imported &&
      expected.turn == actual.turn && std::ranges::equal(expected.ids(), actual.ids()) &&
      std::ranges::equal(expected.committed_ids(), actual.committed_ids());
}

// This is the existing compound K2/K3 record shape, not a universal MTP
// contract. accepted_inputs is the actually committed verifier prefix; later
// proposal rows can differ between the legacy and fixed physical recipes.
inline bool compound_record(bool sampled, const Record &record, std::int32_t vocabulary) {
  const auto valid_id = [vocabulary](auto id) { return id >= 0 && id < vocabulary; };
  if (vocabulary <= 0 || !record.generation || !record.committed || !record.user_output || record.imported ||
      record.input_deferred || record.completion != q::TextCompletion::kEmitNext || !q::drafts_token(record.role) ||
      record.requests != 1 || record.member_index || record.draft_limit < 2 ||
      record.draft_limit > q::kMaximumDeviceMtpDrafts || record.verified_rows != record.draft_limit + 1 ||
      !record.accepted || record.accepted > record.verified_rows || record.consumed_drafts != record.accepted - 1 ||
      record.accepted_draft != (record.consumed_drafts != 0) || (record.consumed_drafts && !record.agreed) ||
      (record.accepted == 1 && record.agreed && !record.stopped && record.quota != 1) ||
      record.verification_inputs.size() != record.verified_rows ||
      !std::ranges::all_of(record.verification_inputs, valid_id) ||
      record.verification_inputs[0] != record.pending_input || record.verification_inputs[1] != record.draft ||
      record.pending_input != record.before.next_id || !valid_id(record.target_first) || !valid_id(record.target_second) ||
      record.emitted_count != record.accepted || record.emitted_count > record.quota ||
      record.emitted.size() < record.emitted_count || record.accepted_inputs.size() < record.accepted ||
      !std::ranges::all_of(std::span(record.emitted).first(record.emitted_count), valid_id) ||
      !std::ranges::all_of(std::span(record.emitted).subspan(record.emitted_count), [](auto id) { return id == -1; }) ||
      !std::ranges::equal(std::span(record.accepted_inputs).first(record.accepted),
          std::span(record.verification_inputs).first(record.accepted)) ||
      !std::ranges::all_of(std::span(record.accepted_inputs).subspan(record.accepted), [](auto id) { return id == -1; }) ||
      !std::ranges::equal(std::span(record.emitted).first(record.consumed_drafts),
          std::span(record.verification_inputs).subspan(1, record.consumed_drafts)) ||
      record.emitted[0] != record.target_first || (record.agreed && record.emitted[0] != record.draft) ||
      (record.emitted_count > 1 && record.emitted[1] != record.target_second)) return false;
  const auto &before = record.before; const auto &after = record.after;
  const auto lag = record.accepted > 1 ? 1u : 0u;
  if (before.length > SIZE_MAX - record.accepted || before.frontier > before.length ||
      before.lag != q::draft_lag(record.role) || before.length - before.frontier != before.lag ||
      (before.lag ? !valid_id(before.catchup_token) : before.catchup_token != -1) ||
      after.length != before.length + record.accepted || after.frontier > after.length || after.lag != lag ||
      after.length - after.frontier != lag || after.next_id != record.emitted[record.emitted_count - 1] ||
      (lag ? !valid_id(after.catchup_token) : after.catchup_token != -1) ||
      after.rotary_offset != before.rotary_offset ||
      record.draft_rows != 2 * (record.draft_limit + before.lag)) return false;
  const auto draws = 2 * record.draft_limit + 1;
  return sampled ? before.sampling.can_advance(draws) && after.sampling == before.sampling.advanced(draws)
      : before.sampling == hyle::inference::SamplingCursor{} && after.sampling == before.sampling;
}

inline bool first_proposal_unselected(const Record &record) {
  return record.accepted == 1 && !record.consumed_drafts && !record.agreed &&
      !record.accepted_draft && record.emitted_count == 1;
}

inline std::size_t protected_verification_prefix(const Record &record) {
  auto count = record.accepted;
  // choose_prefix can publish an accepted proposal as the final pending token
  // on quota/stop, without consuming it. The ledger records first_agrees only,
  // so equality of a later proposal ID with that output is NOT proof that it
  // was selected. Conservatively protect that candidate too: this can be
  // stricter than generated-token equivalence, but never changes the product.
  // A real first rejection has an explicit false agreement and needs no guess.
  if (!first_proposal_unselected(record) && count < record.verified_rows &&
      record.verification_inputs[count] == record.emitted[record.emitted_count - 1]) ++count;
  return count;
}

// Cross-geometry semantic projection only. Keep all published tokens, committed
// input, complete boundaries, work counts and RNG exact. The two omissions are
// computed but unconsumed proposal rows and an unpublished raw second result.
// K1, product/replay same_result, and strict_equal are deliberately unchanged.
inline bool cross_geometry_compound_semantic_equal(bool sampled, std::int32_t vocabulary,
    const Record &expected, const Record &actual) {
  if (!compound_record(sampled, expected, vocabulary) || !compound_record(sampled, actual, vocabulary)) return false;
  const bool unselected_first = first_proposal_unselected(expected) && first_proposal_unselected(actual);
  return expected.requests == actual.requests && expected.member_index == actual.member_index &&
      expected.role == actual.role && expected.generation == actual.generation && expected.quota == actual.quota &&
      expected.input_deferred == actual.input_deferred && expected.completion == actual.completion &&
      expected.prefill_complete == actual.prefill_complete && expected.input_begin == actual.input_begin &&
      expected.before == actual.before && expected.after == actual.after && expected.accepted == actual.accepted &&
      expected.draft_limit == actual.draft_limit && expected.verified_rows == actual.verified_rows &&
      expected.consumed_drafts == actual.consumed_drafts && expected.auxiliary_graph_calls == actual.auxiliary_graph_calls &&
      expected.draft_rows == actual.draft_rows && expected.target_first == actual.target_first &&
      expected.agreed == actual.agreed && expected.accepted_draft == actual.accepted_draft &&
      expected.stopped == actual.stopped && expected.emitted == actual.emitted &&
      expected.emitted_count == actual.emitted_count && expected.accepted_inputs == actual.accepted_inputs &&
      expected.committed == actual.committed && expected.user_output == actual.user_output &&
      expected.imported == actual.imported && expected.turn == actual.turn &&
      std::ranges::equal(expected.ids(), actual.ids()) &&
      std::ranges::equal(expected.committed_ids(), actual.committed_ids()) &&
      std::ranges::equal(std::span(expected.verification_inputs).first((std::max)(
              protected_verification_prefix(expected), protected_verification_prefix(actual))),
          std::span(actual.verification_inputs).first((std::max)(
              protected_verification_prefix(expected), protected_verification_prefix(actual)))) &&
      (unselected_first || expected.draft == actual.draft) &&
      ((expected.emitted_count == 1 && actual.emitted_count == 1) || expected.target_second == actual.target_second);
}

inline std::size_t check_sampled_semantic_contract() {
  std::size_t checks = 0;
  const auto require = [&](bool value) {
    ++checks;
    if (!value) throw std::runtime_error("cross-geometry compound semantic contract failed");
  };
  constexpr std::int32_t vocabulary = 248320;
  for (const auto drafts : {2u, 3u}) for (const bool lag : {false, true}) for (const bool sampled : {false, true}) {
    const auto equivalent = [&](const Record &x, const Record &y) {
      return cross_geometry_compound_semantic_equal(sampled, vocabulary, x, y);
    };
    Record expected;
    expected.role = lag ? q::GraphRole::kMtpCatchupRound : q::GraphRole::kMtpRound;
    expected.generation = expected.committed = true; expected.quota = 16;
    expected.draft_limit = drafts; expected.verified_rows = drafts + 1; expected.accepted = 1;
    expected.draft_rows = 2 * (drafts + (lag ? 1 : 0));
    expected.pending_input = 81; expected.draft = 3312;
    expected.target_first = 11674; expected.target_second = 77;
    expected.before = {.length = 8, .frontier = lag ? 7u : 8u, .lag = lag ? 1u : 0u,
        .next_id = 81, .catchup_token = lag ? 63 : -1,
        .sampling = sampled ? hyle::inference::SamplingCursor{43, 8, 32} : hyle::inference::SamplingCursor{}};
    expected.after = {.length = 9, .frontier = 9, .next_id = 11674,
        .sampling = sampled ? expected.before.sampling.advanced(2 * drafts + 1) : expected.before.sampling};
    expected.verification_inputs = {81, 3312, 59012};
    if (drafts == 3) expected.verification_inputs.push_back(29);
    expected.accepted_inputs = {81}; expected.emitted = {11674}; expected.emitted_count = 1;
    auto actual = expected; actual.target_second = 6; ++actual.verification_inputs.back();
    require(equivalent(expected, actual)); require(equivalent(actual, expected));
    require(!strict_equal(expected, actual)); require(!expected.same_result(actual));
    require(expected.target_second == 77 && actual.target_second == 6); // Original records remain untouched.
    auto first_changed = actual; ++first_changed.draft; ++first_changed.verification_inputs[1];
    require(equivalent(expected, first_changed)); // First rejection publishes neither proposal value.
    require(!strict_equal(expected, first_changed));
    const auto rejects_change = [&](auto mutate) {
      auto changed = actual; mutate(changed); require(!equivalent(expected, changed));
    };
    rejects_change([](auto &x) { x.target_second = -1; });
    rejects_change([](auto &x) { x.target_second = vocabulary; });
    rejects_change([](auto &x) { x.verification_inputs.back() = vocabulary; });
    rejects_change([](auto &x) { x.verification_inputs.back() = -1; });
    rejects_change([](auto &x) { x.verification_inputs.pop_back(); });
    rejects_change([](auto &x) { ++x.verification_inputs[0]; });
    rejects_change([](auto &x) { x.role = q::GraphRole::kText; });
    rejects_change([](auto &x) { x.completion = q::TextCompletion::kStateOnly; });
    rejects_change([](auto &x) { x.generation = false; });
    rejects_change([](auto &x) { ++x.quota; });
    rejects_change([](auto &x) { ++x.requests; });
    rejects_change([](auto &x) { ++x.member_index; });
    rejects_change([](auto &x) { x.committed = false; });
    rejects_change([](auto &x) { x.user_output = false; });
    rejects_change([](auto &x) { x.imported = true; });
    rejects_change([](auto &x) { x.input_deferred = true; });
    rejects_change([](auto &x) { x.prefill_complete = false; });
    rejects_change([](auto &x) { ++x.input_begin; });
    rejects_change([](auto &x) { ++x.turn; });
    rejects_change([](auto &x) { --x.verified_rows; });
    rejects_change([](auto &x) { ++x.accepted; });
    rejects_change([](auto &x) { ++x.consumed_drafts; });
    rejects_change([](auto &x) { x.agreed = true; });
    rejects_change([](auto &x) { x.accepted_draft = true; });
    rejects_change([](auto &x) { ++x.emitted_count; });
    rejects_change([](auto &x) { ++x.auxiliary_graph_calls; });
    rejects_change([](auto &x) { ++x.draft_rows; });
    rejects_change([](auto &x) { ++x.draft; }); // Does not match its own verification input.
    rejects_change([](auto &x) { ++x.pending_input; });
    rejects_change([](auto &x) { ++x.target_first; });
    rejects_change([](auto &x) { ++x.emitted[0]; });
    rejects_change([](auto &x) { ++x.accepted_inputs[0]; });
    rejects_change([](auto &x) { x.stopped = true; });
    rejects_change([](auto &x) { ++x.before.length; });
    rejects_change([](auto &x) { ++x.before.frontier; });
    rejects_change([](auto &x) { ++x.before.rotary_offset; });
    rejects_change([](auto &x) { ++x.before.sampling.seed; });
    rejects_change([](auto &x) { ++x.before.sampling.stream; });
    rejects_change([](auto &x) { ++x.before.sampling.draw; });
    rejects_change([](auto &x) { ++x.after.length; });
    rejects_change([](auto &x) { ++x.after.frontier; });
    rejects_change([](auto &x) { ++x.after.lag; });
    rejects_change([](auto &x) { ++x.after.next_id; });
    rejects_change([](auto &x) { ++x.after.catchup_token; });
    rejects_change([](auto &x) { ++x.after.rotary_offset; });
    rejects_change([](auto &x) { ++x.after.sampling.seed; });
    rejects_change([](auto &x) { ++x.after.sampling.stream; });
    rejects_change([](auto &x) { ++x.after.sampling.draw; });
    auto x = expected, y = actual;
    for (auto *record : {&x, &y}) {
      record->draft_limit = 1; record->verified_rows = 2; record->verification_inputs.resize(2);
      record->draft_rows = 2 * (1 + record->before.lag);
      record->after.sampling = sampled ? record->before.sampling.advanced(3) : record->before.sampling;
    }
    require(!equivalent(x, y)); // K1 never gets this projection.
    x = expected; y = actual; ++x.after.sampling.draw; ++y.after.sampling.draw;
    require(!equivalent(x, y)); // Matching but incorrect RNG boundaries are not valid.
    if (sampled) {
      x = expected; y = actual;
      x.before.sampling.draw = y.before.sampling.draw = hyle::inference::SamplingCursor::maximum_coordinate;
      require(!equivalent(x, y));
    }
    require(!cross_geometry_compound_semantic_equal(sampled, 0, expected, actual));

    // Even a matching stop proposal is pending rather than consumed. It is
    // nevertheless selected and its first-proposal identity must stay strict.
    x = expected; x.agreed = x.stopped = true;
    x.target_first = x.draft; x.emitted[0] = x.draft; x.after.next_id = x.draft;
    y = x; ++y.verification_inputs.back();
    require(equivalent(x, y)); require(!first_proposal_unselected(x));
    auto changed = y; ++changed.draft; ++changed.verification_inputs[1];
    require(!equivalent(x, changed));
    x.stopped = y.stopped = false;
    require(!compound_record(sampled, x, vocabulary)); // Agreement cannot end row0 without stop/quota.
    x.quota = y.quota = 1;
    require(equivalent(x, y)); // A selected first proposal can be pending at quota1 too.
    changed = y; ++changed.draft; ++changed.verification_inputs[1];
    require(!equivalent(x, changed));
    x = expected; y = actual; x.quota = y.quota = 1;
    require(equivalent(x, y)); // Explicit first rejection still leaves its proposal unselected.

    // A partial acceptance consumes the first proposal. Its remaining tail
    // can vary, but neither that proposal nor the now-published second output can.
    x = expected;
    x.accepted = x.emitted_count = 2; x.consumed_drafts = 1;
    x.agreed = x.accepted_draft = true;
    x.target_first = x.draft; x.emitted = {x.draft, x.target_second};
    x.accepted_inputs = {x.pending_input, x.draft};
    x.after.length = 10; x.after.frontier = 9; x.after.lag = 1;
    x.after.next_id = x.target_second; x.after.catchup_token = x.draft;
    y = x; ++y.verification_inputs.back();
    require(equivalent(x, y)); require(!strict_equal(x, y));
    changed = y; ++changed.verification_inputs[1]; ++changed.draft; ++changed.accepted_inputs[1];
    require(!equivalent(x, changed)); // Consumed row cannot change, even consistently.
    changed = y; ++changed.target_second;
    require(!equivalent(x, changed));
    changed = y; ++changed.target_second; ++changed.emitted[1]; ++changed.after.next_id;
    require(!equivalent(x, changed)); // Consistent published second still cannot change.
    changed = y; ++changed.emitted[0];
    require(!equivalent(x, changed));

    // At quota2 or a stop, the second proposal may itself be the public pending
    // token while only the first proposal entered committed state. Protect it,
    // even though all emitted IDs and the next_id could otherwise stay equal.
    for (const bool stopped : {false, true}) {
      auto boundary = x;
      boundary.quota = stopped ? 16u : 2u; boundary.stopped = stopped;
      boundary.verification_inputs[2] = boundary.emitted[1];
      require(compound_record(sampled, boundary, vocabulary));
      require(protected_verification_prefix(boundary) == 3);
      auto tail = boundary;
      if (drafts == 3) ++tail.verification_inputs[3]; // This third proposal is genuinely later.
      require(equivalent(boundary, tail));
      auto pending_changed = tail; ++pending_changed.verification_inputs[2];
      require(compound_record(sampled, pending_changed, vocabulary));
      require(!equivalent(boundary, pending_changed)); // Conservative when later acceptance is not recorded.
      require(!equivalent(pending_changed, boundary));
      auto invalid_quota = boundary; invalid_quota.quota = 1;
      require(!compound_record(sampled, invalid_quota, vocabulary));
    }
  }
  return checks;
}

inline constexpr std::string_view teacher_source =
    "4a76decdfa58a5fd6e15acea7ff410e84fd3e81cb6c39db8a658eeb23d869fbe";
inline constexpr std::array<std::int32_t, 6> teacher_prefix{19, 20, 21, 18, 14, 16};
// Original b12063/2cbca389f BF16 logits at this exact prefix, position 5.
// Apply the existing same-prefix rank-2/raw-logit-gap<0.1 budget, not a new
// backend tolerance. This second row is computed; it is not undefined.
inline constexpr double teacher_first_score = 16.963973999023438;
inline constexpr double teacher_second_score = 16.955802917480469;
static_assert(teacher_first_score > teacher_second_score &&
    teacher_first_score - teacher_second_score < 0.1);

inline bool teacher_prefix_matches(std::span<const Record> records, std::size_t ordinal) {
  if (ordinal >= records.size()) return false;
  std::size_t length = 0;
  const auto append = [&](std::span<const std::int32_t> ids) {
    if (ids.size() > teacher_prefix.size() - length ||
        !std::ranges::equal(ids, std::span(teacher_prefix).subspan(length, ids.size()))) return false;
    length += ids.size();
    return true;
  };
  for (const auto &record : records.first(ordinal)) {
    if (!record.committed || record.input_deferred || record.before.rotary_offset ||
        record.after.rotary_offset || record.before.length != length ||
        record.committed_ids().empty() || !append(record.committed_ids()) || record.after.length != length)
      return false;
  }
  const auto &record = records[ordinal];
  // target_second is the second causal row. Later speculative rows are not
  // part of its mathematical prefix, regardless of the complete packet width.
  return !record.before.rotary_offset && !record.after.rotary_offset && record.before.length == length &&
      record.verification_inputs.size() >= 2 &&
      append(std::span(record.verification_inputs).first(2)) && length == teacher_prefix.size();
}

inline bool unconsumed_greedy_second(const Record &record) {
  return record.generation && record.committed && record.user_output && !record.input_deferred &&
      record.completion == q::TextCompletion::kEmitNext && !record.stopped && q::drafts_token(record.role) &&
      record.requests == 1 && !record.member_index && record.draft_limit >= 1 &&
      record.draft_limit <= q::kMaximumDeviceMtpDrafts && record.verified_rows == record.draft_limit + 1 &&
      record.accepted == 1 && !record.consumed_drafts && !record.agreed && !record.accepted_draft &&
      record.verification_inputs.size() == record.verified_rows && record.verification_inputs[0] == record.pending_input &&
      record.verification_inputs[1] == record.draft && record.pending_input == record.before.next_id &&
      record.target_first >= 0 && record.target_first != record.draft && record.emitted_count == 1 &&
      !record.emitted.empty() && record.emitted[0] == record.target_first &&
      std::ranges::all_of(std::span(record.emitted).subspan(1), [](auto id) { return id == -1; }) &&
      record.after.length == record.before.length + 1 && record.after.next_id == record.target_first &&
      record.before.sampling == hyle::inference::SamplingCursor{} && record.after.sampling == record.before.sampling;
}

inline std::optional<Record> legacy_rejected_second(std::string_view source, bool unmodified_greedy,
    std::span<const Record> expected, std::span<const Record> actual, std::size_t ordinal) {
  if (!unmodified_greedy || source != teacher_source || ordinal >= expected.size() || ordinal >= actual.size()) return {};
  const auto &x = expected[ordinal]; const auto &y = actual[ordinal];
  const auto in_teacher_top2 = [](auto token) { return token == 24 || token == 17; };
  if (x.target_second == y.target_second || !in_teacher_top2(x.target_second) || !in_teacher_top2(y.target_second) ||
      !unconsumed_greedy_second(x) || !unconsumed_greedy_second(y) ||
      !teacher_prefix_matches(expected, ordinal) || !teacher_prefix_matches(actual, ordinal)) return {};
  auto normalized = y;
  normalized.target_second = x.target_second;
  if (!strict_equal(x, normalized)) return {};
  return normalized;
}

inline std::size_t check_contract() {
  std::size_t checks = 0;
  const auto require = [&](bool value) {
    ++checks;
    if (!value) throw std::runtime_error("legacy second-row teacher contract failed");
  };
  const std::array<std::int32_t, 3> prompt{19, 20, 21}, wrong_prompt{19, 20, 22};
  std::array<Record, 3> expected;
  auto &prefill = expected[0];
  prefill.committed = true; prefill.prompt_ids = prompt;
  prefill.after = {.length = 3, .frontier = 3, .next_id = 18};
  auto &first = expected[1];
  first.generation = first.committed = true; first.pending_input = 18; first.accepted = 1;
  first.accepted_inputs = {18, -1}; first.before = prefill.after;
  first.after = {.length = 4, .frontier = 3, .lag = 1, .next_id = 14, .catchup_token = 18};
  auto &second = expected[2];
  second.generation = second.committed = true; second.role = q::GraphRole::kMtpCatchupRound;
  second.pending_input = 14; second.draft = 16; second.target_first = 20; second.target_second = 24;
  second.accepted = second.draft_limit = second.emitted_count = 1; second.verified_rows = 2;
  second.before = first.after;
  second.after = {.length = 5, .frontier = 4, .lag = 1, .next_id = 20, .catchup_token = 14};
  second.accepted_inputs = {14, -1}; second.verification_inputs = {14, 16}; second.emitted = {20, -1};
  auto actual = expected; actual[2].target_second = 17;
  const auto allows = [&](const auto &x, const auto &y, bool unmodified_greedy = true,
                          std::string_view source = teacher_source) {
    return legacy_rejected_second(source, unmodified_greedy, x, y, 2).has_value();
  };
  require(strict_equal(second, second));
  require(!strict_equal(second, actual[2])); // Same-physical has no numerical exception.
  require(allows(expected, actual)); require(allows(actual, expected));
  require(expected[2].target_second == 24 && actual[2].target_second == 17); // Never mutate real records.
  require(!allows(expected, actual, false)); // Random or modified logits have no exception.
  require(!allows(expected, actual, true, "another-source"));
  auto x = expected, y = actual; x[0].prompt_ids = y[0].prompt_ids = wrong_prompt;
  require(!allows(x, y));
  y = actual; y[2].target_second = 18; require(!allows(expected, y));
  y = actual; ++y[2].quota; require(!allows(expected, y));
  y = actual; ++y[2].after.sampling.draw; require(!allows(expected, y));
  x = expected; y = actual;
  for (auto *record : {&x[2], &y[2]}) {
    record->accepted = 2; record->consumed_drafts = 1; record->agreed = record->accepted_draft = true;
    record->emitted_count = 2; record->emitted[1] = record->target_second;
    ++record->after.length; record->after.next_id = record->target_second;
  }
  require(!allows(x, y)); // The accepted/published second row is always strict.
  x = expected; y = actual; x[2].verified_rows = y[2].verified_rows = 1; require(!allows(x, y));
  x = expected; y = actual; x[2].before.rotary_offset = y[2].before.rotary_offset = 1; require(!allows(x, y));
  y = actual; y[1].accepted_inputs[0] = 19; require(!allows(expected, y));
  x = expected; y = actual; x[2].after.rotary_offset = y[2].after.rotary_offset = 1;
  require(!allows(x, y));
  const std::array<std::int32_t, 4> full_input{14, 16, 22, 23};
  const std::array<std::int32_t, 4> full_emitted{20, -1, -1, -1}, full_accepted{14, -1, -1, -1};
  for (std::size_t proposals = 2; proposals <= q::kMaximumDeviceMtpDrafts; ++proposals) {
    x = expected; y = actual;
    for (auto *record : {&x[2], &y[2]}) {
      record->draft_limit = proposals; record->verified_rows = proposals + 1;
      q::write_generation_tokens(record->verification_inputs, std::span(full_input).first(proposals + 1));
      q::write_generation_tokens(record->emitted, std::span(full_emitted).first(proposals + 1));
      q::write_generation_tokens(record->accepted_inputs, std::span(full_accepted).first(proposals + 1));
    }
    require(allows(x, y)); require(allows(y, x));
    require(!strict_equal(x[2], y[2])); // Same-physical comparison stays strict.
    ++x[2].verification_inputs[2]; ++y[2].verification_inputs[2];
    require(allows(x, y)); // The common future suffix cannot affect causal row1.
    ++y[2].verification_inputs[2]; require(!allows(x, y));
    y = x; y[2].target_second = 17; --y[2].verified_rows;
    require(!allows(x, y)); // A partial or inconsistent packet is not this witness.
  }
  x = expected; y = actual; x[2].draft_limit = y[2].draft_limit = q::kMaximumDeviceMtpDrafts + 1;
  require(!allows(x, y));
  return checks;
}
} // namespace qwen_legacy_ledger_case
