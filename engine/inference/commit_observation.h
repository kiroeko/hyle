#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "hyle/core/assert.h"
#include "hyle/error/result.h"
#include "request_progress.h"

namespace hyle::inference {

// Call-scoped projection of one original authoritative record. Token/input
// storage remains with that record and its admitted input owner. Model roles,
// accepted-prefix state, pending leases and RNG publication are not redefined.
struct CommitObservation final {
  std::span<const std::int32_t> emitted{};
  bool user_output = true, prefill_complete = false, stopped = false;
  std::optional<RequestInputCommit> consumed_input{};
};

template<class Record> class PreparedRecordCommit;
template<class Record>
Result<PreparedRecordCommit<Record>> prepare_record_commit(
    std::span<Record>, std::size_t &, Record &) noexcept;

// A synchronous borrow of the actual cold-reserved current record/count, not
// another ledger, reservation owner or persistent callback. Neither the record
// nor its count may move/reset while this ticket exists. One serialized owner
// prepares all participant tickets before publishing its cohort. Independent
// requests keep their own publication outcomes; no cross-head CAS is implied.
template<class Record>
class PreparedRecordCommit final {
  static_assert(std::is_same_v<decltype(Record::committed), bool>,
      "an original commit record must expose its bool committed witness");

 public:
  PreparedRecordCommit() noexcept = default;
  PreparedRecordCommit(const PreparedRecordCommit &) = delete;
  PreparedRecordCommit &operator=(const PreparedRecordCommit &) = delete;
  PreparedRecordCommit(PreparedRecordCommit &&other) noexcept
      : record_(std::exchange(other.record_, nullptr)),
        count_(std::exchange(other.count_, nullptr)), ordinal_(other.ordinal_) {}
  PreparedRecordCommit &operator=(PreparedRecordCommit &&other) noexcept {
    if (this != &other) {
      record_ = std::exchange(other.record_, nullptr);
      count_ = std::exchange(other.count_, nullptr);
      ordinal_ = other.ordinal_;
    }
    return *this;
  }

  bool valid() const noexcept { return record_ && count_; }

  // Publish is the original model head operation, called directly and never
  // stored. A refusal preserves this ticket and the already computed result;
  // it does not authorize a new decision or a different expected base. Success
  // immediately makes the original record authoritative before any later
  // observation, catch-up or diagnostic work. Abandonment changes no record.
  template<class Publish>
  Status publish(Publish &&publish_head) noexcept {
    static_assert(std::is_nothrow_invocable_v<Publish &&> &&
        std::is_same_v<std::invoke_result_t<Publish &&>, Status>,
        "the original head publisher must synchronously return Status without throwing");
    if (!valid())
      return Status::FailedPrecondition("record commit ticket is not prepared");
    // A second ticket or an intervening original commit may have consumed this
    // slot. Reject before touching the head; do not rescan the cohort or IO.
    if (*count_ != ordinal_ || record_->committed)
      return Status::FailedPrecondition("record commit no longer names its original current slot");
    auto accepted = std::forward<Publish>(publish_head)();
    if (!accepted.ok()) return accepted;
    // The head operation owns model state only. It must not independently
    // mutate the very ledger witness this success is about to publish.
    HYLE_FATAL_ASSERT(*count_ == ordinal_ && !record_->committed);
    record_->committed = true;
    ++*count_;
    record_ = nullptr; count_ = nullptr;
    return Status::Ok();
  }

 private:
  template<class R>
  friend Result<PreparedRecordCommit<R>> prepare_record_commit(
      std::span<R>, std::size_t &, R &) noexcept;
  PreparedRecordCommit(Record &record, std::size_t &count) noexcept
      : record_(&record), count_(&count), ordinal_(count) {}
  Record *record_ = nullptr;
  std::size_t *count_ = nullptr;
  std::size_t ordinal_ = 0;
};

// Check the actual reserved slot once before model publication. Model-specific
// record completeness, expected-base/epoch and resource preflight stay with the
// original owner. count < records.size() also proves increment cannot overflow.
template<class Record>
Result<PreparedRecordCommit<Record>> prepare_record_commit(
    std::span<Record> records, std::size_t &count, Record &record) noexcept {
  if (count >= records.size())
    return std::unexpected(Status::OutOfRange("record commit exceeds its original reserved prefix"));
  if (&records[count] != &record)
    return std::unexpected(Status::InvalidArgument("record commit is not the original current slot"));
  if (record.committed)
    return std::unexpected(Status::FailedPrecondition("record commit slot is already authoritative"));
  return PreparedRecordCommit<Record>(record, count);
}

// Observe only an authoritative original record. A duplicate returns before
// projecting its payload or touching family bookkeeping. CheckFamily is a
// side-effect-free capacity/precondition check; ApplyFamily must not throw.
// Neither callback survives the call or changes RequestProgress itself.
//
// Use the existing observe_commit exactly once: it already performs its own
// preflight. There is no additional progress-preflight pass here. Successful
// host progress is followed only by the family's already prepared noexcept
// bookkeeping, so a late failure can retry without republishing the head.
template<class Record, class Project, class CheckFamily, class ApplyFamily>
bool observe_record_commit(std::span<const Record> records, std::size_t count,
    std::size_t ordinal, RequestProgress &progress, Project &&project,
    CheckFamily &&check_family, ApplyFamily &&apply_family) {
  static_assert(std::is_same_v<std::invoke_result_t<CheckFamily &&,
      const Record &, const CommitObservation &>, void>,
      "family observation preflight must return void or throw before mutation");
  static_assert(std::is_nothrow_invocable_v<ApplyFamily &&,
      const Record &, const CommitObservation &> &&
      std::is_same_v<std::invoke_result_t<ApplyFamily &&,
          const Record &, const CommitObservation &>, void>,
      "prepared family observation bookkeeping must return void without throwing");
  if (count > records.size() || ordinal >= count)
    throw std::out_of_range("observation is outside the original committed record prefix");
  if (ordinal < progress.observed_commits()) return false;
  if (ordinal != progress.observed_commits())
    throw std::out_of_range("record observations must follow the original contiguous prefix");
  const auto &record = records[ordinal];
  if (!record.committed)
    throw std::logic_error("record observation has no authoritative original commit witness");
  const CommitObservation &observation = std::forward<Project>(project)(record);
  std::forward<CheckFamily>(check_family)(record, observation);
  if (!progress.observe_commit(ordinal, observation.emitted, observation.user_output,
          observation.prefill_complete, observation.stopped, observation.consumed_input))
    return false;
  std::forward<ApplyFamily>(apply_family)(record, observation);
  return true;
}

} // namespace hyle::inference
