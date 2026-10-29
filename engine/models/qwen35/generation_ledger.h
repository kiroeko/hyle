#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "execution_mode.h"
#include "../../inference/request_stop_policy.h"
#include "../../inference/sampling_cursor.h"
#include "../../inference/history_counts.h"
#include "../../inference/continuation.h"
#include "hyle/container/small_vector.h"

namespace qwen35 {

struct PreparedSampling;
struct ColdResources;

struct GenerationBoundary final {
  std::size_t length = 0, frontier = 0;
  std::uint32_t lag = 0;
  std::int32_t next_id = -1, catchup_token = -1;
  hyle::inference::SamplingCursor sampling{};
  friend bool operator==(const GenerationBoundary &, const GenerationBoundary &) = default;
};

struct GenerationCommitRecord final {
  GraphRole role = GraphRole::kMtpPrefill;
  bool generation = false, committed = false, user_output = true;
  bool imported = false, prefill_complete = true;
  // A reserved successor has no host input fact until its exact
  // predecessor commits and the request owner observes that committed output.
  bool input_deferred = false;
  TextCompletion completion = TextCompletion::kEmitNext;
  std::size_t input_begin = 0;
  std::shared_ptr<const void> input_owner; // Immutable host data, never a Snapshot/lease.
  std::span<const std::int32_t> prompt_ids;
  std::int32_t pending_input = -1;
  std::size_t quota = 0, turn = 0, accepted = 0, draft_rows = 0;
  std::size_t draft_limit = 0, verified_rows = 0, consumed_drafts = 0;
  std::size_t auxiliary_graph_calls = 0;
  std::int32_t draft = -1, target_first = -1, target_second = -1;
  bool agreed = false, accepted_draft = false, stopped = false;
  hyle::container::SmallVector<std::int32_t, 2> emitted{-1, -1};
  hyle::container::SmallVector<std::int32_t, 2> accepted_inputs{-1, -1};
  hyle::container::SmallVector<std::int32_t, 2> verification_inputs;
  std::size_t requests = 1, member_index = 0;
  std::size_t emitted_count = 0;
  GenerationBoundary before, after;
  double step_seconds = 0;
  std::chrono::steady_clock::time_point endpoint{};
  std::span<const std::int32_t> ids() const noexcept {
    if (input_deferred) return {};
    return generation ? std::span(&pending_input, 1) : prompt_ids;
  }
  std::span<const std::int32_t> committed_ids() const noexcept {
    if (input_deferred || (generation && (accepted == 0 || accepted > accepted_inputs.size()))) return {};
    return generation ? std::span<const std::int32_t>(accepted_inputs).first(accepted) : prompt_ids;
  }
  bool same_result(const GenerationCommitRecord &other) const noexcept {
    return requests == other.requests && member_index == other.member_index &&
        role == other.role && generation == other.generation && quota == other.quota &&
        input_deferred == other.input_deferred &&
        completion == other.completion && prefill_complete == other.prefill_complete && input_begin == other.input_begin &&
        before == other.before && after == other.after && accepted == other.accepted &&
        draft_limit == other.draft_limit && verified_rows == other.verified_rows && consumed_drafts == other.consumed_drafts &&
        auxiliary_graph_calls == other.auxiliary_graph_calls &&
        draft_rows == other.draft_rows && draft == other.draft && target_first == other.target_first &&
        target_second == other.target_second && agreed == other.agreed && accepted_draft == other.accepted_draft &&
        stopped == other.stopped && emitted == other.emitted && emitted_count == other.emitted_count &&
        accepted_inputs == other.accepted_inputs && verification_inputs == other.verification_inputs;
  }
};

struct GenerationLedger final {
  std::vector<GenerationCommitRecord> records;
  std::size_t committed_count = 0;
private:
  // Value-owned immutable configuration survives GenerationRequest relocation.
  // This is copied only at initial cold binding, never per token/record.
  std::optional<hyle::inference::RequestStopPolicy> stop_policy_;
  std::optional<std::size_t> successor_ordinal_;
  std::optional<hyle::inference::HistoryCounts> history_counts_;

  static void reset_record(GenerationCommitRecord &record) noexcept {
    auto emitted = std::move(record.emitted), accepted = std::move(record.accepted_inputs);
    auto verified = std::move(record.verification_inputs);
    record = GenerationCommitRecord{};
    record.emitted = std::move(emitted); record.accepted_inputs = std::move(accepted);
    record.verification_inputs = std::move(verified);
    record.emitted.clear(); record.accepted_inputs.clear(); record.verification_inputs.clear();
  }
  GenerationCommitRecord &uncommitted_record() {
    if (committed_count >= records.size() || records[committed_count].committed)
      throw std::runtime_error("MTP replay ledger has no uncommitted reserved record");
    return records[committed_count];
  }
  // A K1 successor needs the original greedy R1 pending boundary. The Model
  // still proves its sampler/capability; a zero cursor is only a ledger fact.
  static bool k1_pending_boundary(const GenerationCommitRecord &record) noexcept {
    return drafts_token(record.role) && record.draft_limit == 1 &&
        record.requests == 1 && record.member_index == 0 &&
        record.before.lag == draft_lag(record.role) &&
        record.before.frontier <= record.before.length &&
        record.before.length - record.before.frontier == record.before.lag &&
        record.pending_input >= 0 && record.pending_input == record.before.next_id &&
        record.before.sampling == hyle::inference::SamplingCursor{};
  }

public:
  hyle::inference::HistoryCounts &history_counts(std::size_t vocabulary,
      const hyle::compute::ResourceBudget &budget, hyle::mem::AllocatorRef allocator) {
    if (!history_counts_) history_counts_.emplace(vocabulary, budget, allocator);
    auto &cache = *history_counts_;
    if (cache.vocabulary() != vocabulary || cache.ordinal() > committed_count || committed_count > records.size())
      throw std::runtime_error("history count cache differs from its original committed ledger");
    while (cache.ordinal() < committed_count) {
      const auto &record = records[cache.ordinal()];
      const auto ids = record.committed_ids();
      if (!record.committed || ids.empty() || record.before.length != cache.consumed_length() ||
          record.after.length < record.before.length || ids.size() != record.after.length - record.before.length)
        throw std::runtime_error("history count source is not a continuous committed token prefix");
      cache.append_committed(ids);
    }
    return cache;
  }
  const hyle::inference::RequestStopPolicy &stop_policy() const noexcept {
    return stop_policy_ ? *stop_policy_ : hyle::inference::default_request_stop_policy;
  }
  void validate_stop_policy(const hyle::inference::RequestStopPolicy &value) const {
    if ((stop_policy_ || committed_count) && stop_policy() != value)
      throw std::runtime_error("request stop policy differs from its committed generation ledger");
  }
  void bind_stop_policy(const hyle::inference::RequestStopPolicy &value, std::int64_t vocabulary) {
    value.validate(vocabulary); validate_stop_policy(value);
    if (!stop_policy_) stop_policy_.emplace(value);
  }
  const PreparedSampling *sampling = nullptr; // Borrowed immutable cold policy, never a cursor.
  std::weak_ptr<ColdResources> sampling_owner; // Identity only: cannot pin weights or admit an address-reused Model.
  void validate_sampling(const PreparedSampling &value, const std::weak_ptr<ColdResources> &owner) const {
    if (sampling && (sampling_owner.expired() || sampling_owner.owner_before(owner) || owner.owner_before(sampling_owner) || sampling != &value))
      throw std::runtime_error("generation ledger sampling belongs to another model or configuration");
  }
  void bind_sampling(const PreparedSampling &value, const std::weak_ptr<ColdResources> &owner) {
    validate_sampling(value, owner); sampling = &value; sampling_owner = owner;
  }
  explicit GenerationLedger(std::size_t capacity, std::size_t maximum_width = 2) : records(capacity) {
    for (auto &record : records)
      if (!record.emitted.try_reserve(maximum_width) || !record.accepted_inputs.try_reserve(maximum_width) ||
          !record.verification_inputs.try_reserve(maximum_width))
        throw std::bad_alloc{};
  }
  GenerationCommitRecord &next() {
    auto &record = uncommitted_record();
    if (record.input_deferred)
      throw std::runtime_error("generation successor input has not been bound to its observed predecessor");
    return record;
  }

  // The original request owner may renew the budget of an already bound,
  // still-pending T1 or greedy K1 record when a later pump grants fresh credit.
  // It supplies its actual current boundary/observation and RequestLoop grant;
  // this ledger neither owns delivery credit nor creates a new reservation.
  // The caller has checked the same submitted Step/cohort and may not renew
  // cancelled, draining or completed turns. No committed result is changed.
  GenerationCommitRecord &renew_pending_quota(const GenerationBoundary &actual,
      std::size_t observed_commits, std::size_t turn, std::size_t remaining_outputs,
      std::size_t reserved_outputs, std::size_t additional_outputs) {
    auto &record = next();
    const bool ordinary = record.role == GraphRole::kText && !record.draft_limit;
    const bool k1 = k1_pending_boundary(record) && record.quota >= 2;
    const auto width = ordinary ? std::size_t{1} : std::size_t{2};
    const auto quota = hyle::inference::continuation_quota(
        {reserved_outputs, additional_outputs}, width, remaining_outputs);
    if (successor_ordinal_ || observed_commits != committed_count || (!ordinary && !k1) ||
        !record.generation || record.completion != TextCompletion::kEmitNext ||
        !record.user_output || record.imported || record.ids().size() != 1 ||
        record.pending_input < 0 || record.pending_input != actual.next_id ||
        record.before != actual || record.turn != turn || !record.quota ||
        record.accepted || record.verified_rows || record.emitted_count || record.stopped ||
        (k1 && (record.consumed_drafts || record.draft_rows)) ||
        !quota)
      throw std::runtime_error("quota renewal requires its bound pending record and actual owner grant");
    record.quota = *quota;
    return record;
  }

  // One unknown-input successor of the current T1 or greedy K1 episode. These
  // counters are facts supplied by the serialized request owner, not a second
  // progress/RNG/credit authority. All actual Model capacity, base, epoch,
  // device-Source and output-credit checks remain with their original owners.
  GenerationCommitRecord &reserve_successor(std::size_t observed_commits,
      std::size_t quota, std::size_t turn, std::size_t requests = 1,
      std::size_t member_index = 0) {
    const auto &previous = next();
    const bool ordinary = previous.role == GraphRole::kText;
    const bool k1 = k1_pending_boundary(previous) && requests == 1 && member_index == 0 &&
        !previous.accepted && !previous.verified_rows && !previous.emitted_count &&
        !previous.stopped && !previous.consumed_drafts && !previous.draft_rows;
    if (successor_ordinal_ || observed_commits != committed_count ||
        records.size() - committed_count < 2 || !quota || !requests || member_index >= requests ||
        (!ordinary && !k1) || !previous.generation ||
        previous.completion != TextCompletion::kEmitNext || !previous.user_output || previous.imported ||
        previous.ids().size() != 1 || previous.pending_input < 0 || previous.turn != turn ||
        (ordinary ? quota >= previous.quota
                  : previous.quota < 4 || quota < 2 || quota > previous.quota - 2) ||
        records[committed_count + 1].committed)
      throw std::runtime_error("generation successor requires one bounded pending predecessor");
    const auto ordinal = committed_count + 1;
    auto &record = records[ordinal];
    reset_record(record);
    record.role = ordinary ? GraphRole::kText : GraphRole::kMtpCatchupRound;
    record.draft_limit = ordinary ? 0 : 1;
    record.generation = true; record.input_deferred = true;
    record.completion = TextCompletion::kEmitNext; record.quota = quota; record.turn = turn;
    record.requests = requests; record.member_index = member_index;
    // Inline capacity is already present, including maximum_width=1. Retain
    // any larger cold capacity rather than allocating a second record owner.
    record.emitted.resize(2, -1); record.accepted_inputs.resize(2, -1);
    successor_ordinal_ = ordinal;
    return record;
  }
  std::optional<std::size_t> successor_ordinal() const noexcept { return successor_ordinal_; }

  // Bind only after the exact predecessor's publication AND its host output
  // observation. A complete boundary, rather than just a token, ties this
  // deferred record to that predecessor's actual length and sampling facts.
  // A rejection preserves the unknown input and every reserved record.
  GenerationCommitRecord &bind_successor(const GenerationBoundary &actual,
      std::size_t observed_commits, std::size_t turn, std::int32_t actual_last_token) {
    if (!successor_ordinal_ || *successor_ordinal_ != committed_count ||
        observed_commits != committed_count || !committed_count || committed_count >= records.size())
      throw std::runtime_error("generation successor requires its committed and observed predecessor");
    const auto &previous = records[committed_count - 1];
    auto &record = records[committed_count];
    const bool ordinary = previous.role == GraphRole::kText &&
        previous.accepted == 1 && previous.verified_rows == 1 &&
        previous.emitted_count == 1 && !previous.emitted.empty() &&
        previous.emitted[0] == actual.next_id && record.role == GraphRole::kText;
    const bool k1 = k1_pending_boundary(previous) &&
        previous.quota >= 4 && record.quota >= 2 && record.quota <= previous.quota - 2 &&
        previous.accepted == 2 && previous.verified_rows == 2 && previous.consumed_drafts == 1 &&
        previous.agreed && previous.accepted_draft && previous.draft >= 0 &&
        previous.target_first == previous.draft && previous.target_second == actual.next_id &&
        previous.emitted_count == 2 && previous.emitted.size() >= 2 &&
        previous.emitted[0] == previous.draft && previous.emitted[1] == actual.next_id &&
        previous.accepted_inputs.size() >= 2 && previous.accepted_inputs[0] == previous.pending_input &&
        previous.accepted_inputs[1] == previous.draft && previous.verification_inputs.size() == 2 &&
        previous.verification_inputs[0] == previous.pending_input && previous.verification_inputs[1] == previous.draft &&
        actual.length >= previous.before.length && actual.length - previous.before.length == 2 &&
        actual.frontier >= previous.before.frontier &&
        actual.frontier - previous.before.frontier == previous.before.lag + 1 &&
        previous.draft_rows == previous.before.lag + 1 && actual.lag == 1 &&
        actual.frontier <= actual.length && actual.length - actual.frontier == 1 &&
        actual.catchup_token == previous.draft && actual.sampling == previous.before.sampling &&
        record.role == GraphRole::kMtpCatchupRound && record.draft_limit == 1;
    if (!previous.committed || (!ordinary && !k1) || !previous.generation ||
        previous.completion != TextCompletion::kEmitNext || !previous.user_output || previous.imported ||
        previous.input_deferred || previous.ids().size() != 1 || previous.stopped ||
        previous.after != actual || actual.next_id < 0 || actual_last_token != actual.next_id ||
        previous.turn != turn || record.turn != turn ||
        previous.requests != record.requests || previous.member_index != record.member_index ||
        record.committed || !record.input_deferred || !record.generation ||
        record.completion != TextCompletion::kEmitNext || !record.user_output || record.imported)
      throw std::runtime_error("generation successor input differs from its actual predecessor");
    record.before = actual; record.pending_input = actual.next_id; record.input_deferred = false;
    // B is now the known-input current record; it may reserve C while B is
    // still pending. Only one unknown input exists beyond the committed prefix.
    successor_ordinal_.reset();
    return record;
  }

  void discard_successor() noexcept {
    if (!successor_ordinal_) return;
    const auto ordinal = *std::exchange(successor_ordinal_, std::nullopt);
    // Never erase an authoritative commit, even if an owner is recovering
    // after an error. A bound current episode is cleared by the normal path.
    if (ordinal < records.size() && !records[ordinal].committed) reset_record(records[ordinal]);
  }
  void discard_uncommitted() {
    auto &record = uncommitted_record();
    discard_successor();
    reset_record(record);
  }
};

} // namespace qwen35
