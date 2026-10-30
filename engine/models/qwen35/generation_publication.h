#pragma once

#include <initializer_list>
#include <optional>

#include "generation_ledger.h"
#include "publication_facts.h"
#include "../../inference/evaluation_failure.h"

namespace qwen35 {

struct GenerationPublicationInput final {
  GenerationLedger *ledger = nullptr;
  const GenerationCommitRecord *expected = nullptr;
};

// A call-scoped borrow of the original request records. No Model/Step/native
// owner retains this object. Every member keeps its exact attempted slot, even
// if a peer commits or an observer later fails; count-1 is never a cleanup key.
class GenerationPublicationBatch final {
  struct Member final {
    GenerationPublicationInput input;
    GenerationCommitRecord *record = nullptr;
    std::optional<hyle::inference::PreparedRecordCommit> ticket;
  };
  hyle::container::SmallVector<Member, 2> members_;

public:
  template<class Project>
  GenerationPublicationBatch(std::size_t count, Project project) {
    if (!members_.try_reserve(count)) throw std::bad_alloc{};
    for (std::size_t index = 0; index < count; ++index)
      members_.push_back(Member{project(index), nullptr, {}});
  }
  explicit GenerationPublicationBatch(std::span<const GenerationPublicationInput> inputs) {
    if (!members_.try_reserve(inputs.size())) throw std::bad_alloc{};
    for (const auto &input : inputs) members_.push_back(Member{input, nullptr, {}});
  }
  GenerationPublicationBatch(std::initializer_list<GenerationPublicationInput> inputs)
      : GenerationPublicationBatch(std::span<const GenerationPublicationInput>(inputs.begin(), inputs.size())) {}
  GenerationPublicationBatch(const GenerationPublicationBatch &) = delete;
  GenerationPublicationBatch &operator=(const GenerationPublicationBatch &) = delete;

  GenerationCommitRecord *record(std::size_t index) const noexcept { return members_[index].record; }
  hyle::inference::PreparedRecordCommit *ticket(std::size_t index) noexcept {
    return members_[index].ticket ? &*members_[index].ticket : nullptr;
  }
  hyle::inference::PreparedRecordCommit *prepare(std::size_t index, const EvaluationAttemptFacts &facts) {
    if (index >= members_.size()) throw std::runtime_error("publication has no original batch member");
    auto &member = members_[index];
    if (!member.input.ledger) {
      if (member.input.expected) throw std::runtime_error("expected episode record requires a fresh commit witness");
      return nullptr;
    }
    if (!facts.complete_state)
      throw std::runtime_error("generation role differs from its complete target/draft participant");
    for (std::size_t prior = 0; prior < index; ++prior)
      if (members_[prior].input.ledger == member.input.ledger)
        throw std::runtime_error("batch participants cannot share a mutable generation ledger");
    auto &ledger = *member.input.ledger;
    if (ledger.stop_policy() != facts.stop_policy)
      throw std::runtime_error("generation ledger stop policy differs from its admitted episode");
    auto &record = ledger.next(); member.record = &record;
    if (record.role != facts.role || record.completion != facts.completion || record.before != facts.before ||
        record.ids().size() != (record.generation ? 1u : facts.input_tokens) ||
        record.ids().empty() || record.ids().front() != facts.first_input)
      throw std::runtime_error("generation ledger entry differs from its actual episode");
    if (member.input.expected && !member.input.expected->committed)
      throw std::runtime_error("expected episode record is not committed");
    if (facts.withdrawn) {
      member.record = nullptr; ledger.discard_uncommitted(); return nullptr;
    }
    auto prepared = hyle::inference::prepare_record_commit(std::span(ledger.records), ledger.committed_count, record);
    if (!prepared) throw hyle::inference::EvaluationFailure(prepared.error());
    member.ticket.emplace(std::move(*prepared));
    return &*member.ticket;
  }

  // The model's selector forms its actual result first. This only projects it
  // into the request record; it does not select, repair or rerun mathematics.
  template<class Result>
  void selected(std::size_t index, const Result &result, std::size_t quota) {
    auto *record = members_[index].record;
    if (!record) return;
    if (!record->generation || record->quota != quota)
      throw std::runtime_error("MTP ledger quota differs from its original selection");
    record->draft = result.draft; record->target_first = result.target_first;
    record->target_second = result.target_second; record->agreed = result.agreed;
    record->accepted_draft = result.accepted_draft; record->stopped = result.stopped;
    record->consumed_drafts = result.consumed_drafts;
    if (record->role == GraphRole::kVerify) {
      record->auxiliary_graph_calls = result.auxiliary_graph_calls;
      write_generation_tokens(record->emitted, result.emitted);
    } else write_generation_tokens(record->emitted,
        {result.emitted[0], result.emitted.size() == 2 ? result.emitted[1] : -1});
    record->emitted_count = result.emitted.size();
  }

  void complete(std::size_t index, const EvaluationResultFacts &facts) {
    auto *record = members_[index].record;
    if (!record) return;
    record->requests = facts.requests; record->member_index = facts.member_index;
    record->accepted = facts.accepted; record->draft_rows = facts.draft_rows;
    record->after = facts.after;
    write_generation_tokens(record->accepted_inputs, facts.accepted_inputs);
    if (record->generation) write_generation_tokens(record->verification_inputs, facts.verification_inputs);
    record->verified_rows = facts.verified_rows;
    if (!record->generation || facts.role == GraphRole::kText) {
      record->draft = -1; record->target_first = facts.next_id; record->target_second = -1;
      record->agreed = false; record->accepted_draft = false; record->stopped = facts.stopped;
      if (facts.state_only) { record->emitted.clear(); record->emitted_count = 0; }
      else { write_generation_tokens(record->emitted, {facts.next_id, -1}); record->emitted_count = 1; }
    }
    if (members_[index].input.expected && !record->same_result(*members_[index].input.expected))
      throw std::runtime_error("MTP replay natural decision or state boundary diverged from its recorded episode");
  }

  // Stamp all authoritative peers before any fallible report/observer.
  void finished(std::chrono::steady_clock::time_point endpoint, double seconds) noexcept {
    for (auto &member : members_) if (member.record && member.record->committed) {
      member.record->endpoint = endpoint; member.record->step_seconds = seconds;
    }
  }
  void retired(const hyle::inference::RetirementObservation &observed) noexcept {
    for (auto &member : members_) {
      member.ticket.reset();
      if (observed.retired && member.input.ledger && member.record)
        (void)member.input.ledger->discard_attempt(member.record);
      member.record = nullptr;
    }
  }
};

} // namespace qwen35
