#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "../../engine/inference/commit_observation.h"

// Finite host-contract consumer. The publishers below represent checked host
// head operations only; this case performs no model work or native submission.
namespace inference_commit_observation_case {
namespace i = hyle::inference;

inline void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
template<class T> T take(hyle::Result<T> result) {
  if (!result) throw std::runtime_error("commit observation case could not prepare its original record");
  return std::move(*result);
}

struct Record final {
  bool committed = false;
  std::array<std::int32_t, 1> emitted{};
};
struct OtherRecord final {
  std::uint64_t payload = 0x123456789abcdef0ULL;
  bool committed = false;
  std::array<std::int32_t, 1> emitted{23};
};
using Ticket = i::PreparedRecordCommit;
static_assert(!std::is_copy_constructible_v<Ticket> && !std::is_copy_assignable_v<Ticket>);
static_assert(std::is_nothrow_move_constructible_v<Ticket> && std::is_nothrow_move_assignable_v<Ticket>);
static_assert(std::is_same_v<decltype(i::prepare_record_commit(
    std::declval<std::span<OtherRecord>>(), std::declval<std::size_t &>(),
    std::declval<OtherRecord &>())), hyle::Result<Ticket>>);

inline int run() {
  std::array<Record, 2> records{{{false, {11}}, {false, {17}}}};
  std::size_t committed_count = 0, publisher_calls = 0, head_publications = 0;
  Record foreign;
  const auto foreign_slot = i::prepare_record_commit(std::span<Record>(records), committed_count, foreign);
  const auto future_slot = i::prepare_record_commit(std::span<Record>(records), committed_count, records[1]);
  std::size_t exhausted_count = records.size();
  const auto exhausted = i::prepare_record_commit(std::span<Record>(records), exhausted_count, records[0]);
  require(!foreign_slot && foreign_slot.error().IsInvalidArgument() &&
      !future_slot && future_slot.error().IsInvalidArgument() &&
      !exhausted && exhausted.error().IsOutOfRange() && !committed_count &&
      !records[0].committed && !records[1].committed,
      "record-independent tickets weakened original factory slot validation");
  {
    std::array<OtherRecord, 1> other;
    std::size_t other_count = 0, other_publications = 0;
    auto transferred = take(i::prepare_record_commit(std::span<Record>(records), committed_count, records[0]));
    transferred = take(i::prepare_record_commit(std::span<OtherRecord>(other), other_count, other[0]));
    require(!committed_count && !records[0].committed && transferred.valid(),
        "move assignment committed the abandoned original record");
    require(transferred.publish([&]() noexcept -> hyle::Status {
      ++other_publications; return hyle::Status::Ok();
    }).ok() && !transferred.valid() && other_publications == 1 && other_count == 1 &&
        other[0].committed && other[0].payload == 0x123456789abcdef0ULL && other[0].emitted[0] == 23 &&
        !committed_count && !records[0].committed,
        "one ticket type did not mark only the original heterogeneous record witness");
    std::size_t already_committed_count = 0;
    const auto authoritative = i::prepare_record_commit(std::span<OtherRecord>(other), already_committed_count, other[0]);
    require(!authoritative && authoritative.error().IsFailedPrecondition() &&
        !already_committed_count && other_count == 1 && other[0].committed,
        "factory accepted an already authoritative original record");
  }
  auto prepared = take(i::prepare_record_commit(std::span<Record>(records), committed_count, records[0]));
  auto peer_ticket = take(i::prepare_record_commit(std::span<Record>(records), committed_count, records[0]));
  auto moved = std::move(prepared);
  require(!prepared.valid() && moved.valid(), "moving a record ticket preserved two publication rights");
  const auto moved_rejected = prepared.publish([&]() noexcept -> hyle::Status {
    ++publisher_calls; return hyle::Status::Ok();
  });
  require(moved_rejected.IsFailedPrecondition() && !publisher_calls,
      "a moved-from ticket invoked its publisher");
  const auto expected_refusal = hyle::Status::Unavailable("original expected-base refusal");
  const auto refused = moved.publish([&]() noexcept -> hyle::Status {
    ++publisher_calls; return expected_refusal;
  });
  require(refused == expected_refusal && moved.valid() && publisher_calls == 1 &&
      !head_publications && !committed_count && !records[0].committed,
      "a refused original publisher advanced the commit witness or consumed its retry");
  const auto accepted = moved.publish([&]() noexcept -> hyle::Status {
    ++publisher_calls; ++head_publications; return hyle::Status::Ok();
  });
  require(accepted.ok() && !moved.valid() && publisher_calls == 2 && head_publications == 1 &&
      committed_count == 1 && records[0].committed,
      "successful head publication did not immediately mark its original record");
  const auto duplicate_publication = moved.publish([&]() noexcept -> hyle::Status {
    ++publisher_calls; ++head_publications; return hyle::Status::Ok();
  });
  require(duplicate_publication.IsFailedPrecondition() && publisher_calls == 2 && head_publications == 1,
      "a consumed ticket invoked the original head a second time");
  const auto peer_rejected = peer_ticket.publish([&]() noexcept -> hyle::Status {
    ++publisher_calls; ++head_publications; return hyle::Status::Ok();
  });
  require(peer_rejected.IsFailedPrecondition() && publisher_calls == 2 && head_publications == 1,
      "another prepared ticket republished an already authoritative original slot");
  auto next = take(i::prepare_record_commit(std::span<Record>(records), committed_count, records[1]));
  require(next.publish([&]() noexcept -> hyle::Status {
    ++publisher_calls; ++head_publications; return hyle::Status::Ok();
  }).ok(), "the next original record could not publish");
  require(committed_count == 2 && records[1].committed && head_publications == 2,
      "successive original records did not form one authoritative prefix");

  i::RequestProgress progress(2);
  (void)progress.begin_turn();
  std::size_t projections = 0, family_checks = 0, family_applications = 0;
  bool refuse_observation = false;
  const auto project = [&](const Record &record) -> i::CommitObservation {
    ++projections;
    return {.emitted = record.emitted, .user_output = true, .prefill_complete = true};
  };
  const auto check_family = [&](const Record &, const i::CommitObservation &) {
    ++family_checks;
    if (refuse_observation) throw std::runtime_error("late family observation refusal");
  };
  const auto apply_family = [&](const Record &, const i::CommitObservation &) noexcept {
    ++family_applications;
  };
  const auto observe = [&](std::size_t ordinal) {
    return i::observe_record_commit(std::span<const Record>(records), committed_count,
        ordinal, progress, project, check_family, apply_family);
  };
  bool gap_rejected = false;
  try { (void)observe(1); }
  catch (const std::out_of_range &) { gap_rejected = true; }
  require(gap_rejected && !projections && !family_checks && !family_applications &&
      !progress.observed_commits() && progress.generated.empty(),
      "an ordinal gap projected payload or mutated request progress");

  const auto original_publications = head_publications, original_publisher_calls = publisher_calls;
  refuse_observation = true;
  bool observation_failed = false;
  try { (void)observe(0); }
  catch (const std::runtime_error &) { observation_failed = true; }
  require(observation_failed && records[0].committed && committed_count == 2 &&
      !progress.observed_commits() && progress.generated.empty() && !family_applications &&
      head_publications == original_publications && publisher_calls == original_publisher_calls,
      "late observation failure lost a commit or republished an original head");
  refuse_observation = false;
  require(observe(0) && progress.observed_commits() == 1 && progress.generated.size() == 1 &&
      progress.generated[0] == 11 && family_applications == 1,
      "retry did not observe the already committed original result once");

  const auto duplicate_projection = [](const Record &) -> i::CommitObservation {
    throw std::runtime_error("duplicate payload must not be projected");
  };
  const auto prior_projections = projections, prior_checks = family_checks;
  require(!i::observe_record_commit(std::span<const Record>(records), committed_count,
      0, progress, duplicate_projection, check_family, apply_family),
      "duplicate observation was treated as new progress");
  require(projections == prior_projections && family_checks == prior_checks && family_applications == 1,
      "duplicate observation inspected its payload or repeated family bookkeeping");
  require(observe(1) && progress.observed_commits() == 2 && progress.generated.size() == 2 &&
      progress.generated[1] == 17 && progress.turn_done && family_applications == 2,
      "next ordinal did not consume its own original result");
  require(head_publications == original_publications && publisher_calls == original_publisher_calls &&
      committed_count == 2 && records[0].committed && records[1].committed,
      "observation/retry changed original publication or authority facts");

  std::cout << "{\"status\":\"PASS\",\"case\":\"commit-observation\",\"models\":0,\"native_submissions\":0,"
      "\"publisher_refusal_preserved\":true,\"original_commit_witness\":true,\"moved_ticket_invalid\":true,"
      "\"heterogeneous_record_witness\":true,\"factory_slot_validation_preserved\":true,"
      "\"duplicate_publication_rejected\":true,\"contiguous_observation\":true,"
      "\"duplicate_payload_not_projected\":true,\"observation_retry_without_republication\":true}\n";
  return 0;
}
} // namespace inference_commit_observation_case
