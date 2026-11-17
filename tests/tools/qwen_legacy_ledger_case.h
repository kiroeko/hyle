#pragma once

#include <array>
#include <optional>
#include <string_view>

#include "../../engine/models/qwen35/generation_ledger.h"

// Test-only numerical witness for the legacy T1/T2 versus fixed-T2 journey.
// The same-physical oracle must always use strict_equal without this witness.
namespace qwen_legacy_ledger_case {
namespace q = qwen35;
using Record = q::GenerationCommitRecord;

inline bool strict_equal(const Record &expected, const Record &actual) {
  return expected.same_result(actual) && expected.committed == actual.committed &&
      expected.user_output == actual.user_output && expected.imported == actual.imported &&
      expected.turn == actual.turn && std::ranges::equal(expected.ids(), actual.ids()) &&
      std::ranges::equal(expected.committed_ids(), actual.committed_ids());
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
  return !record.before.rotary_offset && record.before.length == length &&
      append(record.verification_inputs) && length == teacher_prefix.size();
}

inline bool unconsumed_greedy_second(const Record &record) {
  return record.generation && record.committed && record.user_output && !record.input_deferred &&
      record.completion == q::TextCompletion::kEmitNext && !record.stopped && q::drafts_token(record.role) &&
      record.requests == 1 && !record.member_index && record.draft_limit == 1 && record.verified_rows == 2 &&
      record.accepted == 1 && !record.consumed_drafts && !record.agreed && !record.accepted_draft &&
      record.verification_inputs.size() == 2 && record.verification_inputs[0] == record.pending_input &&
      record.verification_inputs[1] == record.draft && record.pending_input == record.before.next_id &&
      record.target_first >= 0 && record.target_first != record.draft && record.emitted_count == 1 &&
      !record.emitted.empty() && record.emitted[0] == record.target_first &&
      std::ranges::all_of(std::span(record.emitted).subspan(1), [](auto id) { return id == -1; }) &&
      record.after.length == record.before.length + 1 && record.after.next_id == record.target_first &&
      record.before.sampling == hyle::inference::SamplingCursor{} && record.after.sampling == record.before.sampling;
}

inline std::optional<Record> legacy_rejected_second(std::string_view source, bool sampled,
    std::span<const Record> expected, std::span<const Record> actual, std::size_t ordinal) {
  if (sampled || source != teacher_source || ordinal >= expected.size() || ordinal >= actual.size()) return {};
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
  const auto allows = [&](const auto &x, const auto &y, bool sampled = false,
                          std::string_view source = teacher_source) {
    return legacy_rejected_second(source, sampled, x, y, 2).has_value();
  };
  require(strict_equal(second, second));
  require(!strict_equal(second, actual[2])); // Same-physical has no numerical exception.
  require(allows(expected, actual)); require(allows(actual, expected));
  require(expected[2].target_second == 24 && actual[2].target_second == 17); // Never mutate real records.
  require(!allows(expected, actual, true)); // Random rejection may publish its second-row residual.
  require(!allows(expected, actual, false, "another-source"));
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
  return checks;
}
} // namespace qwen_legacy_ledger_case
