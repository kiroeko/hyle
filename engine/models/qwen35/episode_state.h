#pragma once

// Original Qwen episode layout, shared only so its embedded private DraftProgram
// can be defined without including main.cc. This is runtime state, not a new
// cold owner; Model/Session admission, methods and retirement are unchanged.
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "phase.h"
#include "session_state.h"
#include "../../inference/execution_episode.h"
#include "../../inference/logits_inputs.h"
#include "../../inference/request_stop_policy.h"

namespace qwen35 {
using Clock = std::chrono::steady_clock;
using inference::LogitsInputs;
struct ColdResources;

class Session;
struct StepMember final {
  std::shared_ptr<Session> session;
  cache::WorkingState *owner_state = nullptr;
  const cache::State *base = nullptr;
  std::uint64_t base_epoch = 0;
  // Synchronous episode borrow, like its original request input. No native
  // callback reads this policy; the original owner keeps it through finish.
  const inference::RequestStopPolicy *stop_policy = nullptr;
  LogitsInputs logits{}; // Original admitted episode snapshot, never history.
  inference::ConditionalLogitsInputs conditional_logits{};
  std::unique_ptr<cache::State> next{};
  hyle::container::SmallVector<c::PagedArenaCommit, 32> target_updates{};
  std::array<c::PagedArenaCommit, 2> draft_updates{};
  // Armed only after members and ticket storage stop moving, then consumed
  // before publication recycles their original candidate or previous state.
  inference::PagedSequencePublication target_publication, draft_publication;
  hyle::container::SmallVector<c::PagedPageAppender, 32> next_tails{};
  std::array<c::PagedPageAppender, 2> next_draft_tails{};
  c::BatchLayoutVersion layout{};
  c::BatchLayoutVersion draft_layout{};
  std::size_t input_tokens = 0, input_row_begin = 0, accepted_tokens = 0;
  std::uint32_t verified_last_page = 0;
  std::int32_t first_input = -1, draft_token = -1;
  std::vector<std::int32_t> verification_inputs{};
  std::vector<c::ReadLease> carry_prefix{};
  // Frozen active inputs of this member's actual draft Append, not accepted rows.
  std::size_t draft_input_rows = 0, discarded_draft_rows = 0;
  bool sampled_accept = false;


  inference::PagedSequenceEdit target_edit() noexcept {
    return {base->paged, next->paged, {target_updates.data(), target_updates.size()}};
  }
  inference::PagedSequenceEdit draft_edit() noexcept {
    return {base->draft->paged, next->draft->paged, draft_updates};
  }
  void preflight_frames() {
    target_publication = {}; draft_publication = {};
    target_publication = take(target_edit().prepare_publication());
    if (draft_updates[0].valid() || draft_updates[1].valid())
      draft_publication = take(draft_edit().prepare_publication());
  }
  void publish_frames(cache::State *previous, cache::State &candidate) noexcept {
    HYLE_FATAL_ASSERT(target_publication.candidate() == &candidate.paged);
    target_publication.publish(previous ? &previous->paged : nullptr);
    if (draft_publication.valid()) {
      HYLE_FATAL_ASSERT(candidate.draft && draft_publication.candidate() == &candidate.draft->paged &&
          (!previous || previous->draft));
      draft_publication.publish(previous ? &previous->draft->paged : nullptr);
    }
  }
};
// Typed views of one original Graph submission. They contain no executor,
// model-state snapshot or observer independent of the original proxy. A
// successor may keep these views after its producer's host episode retires.
struct StepSources final : inference::SubmissionViews {
  struct Boundary final {
    std::weak_ptr<Session> session;
    const cache::State *candidate = nullptr;
    std::uint64_t epoch = 0;
    std::size_t length = 0, draft_frontier = 0;
  };
  std::vector<Boundary> boundaries;
  std::vector<c::ReadLease> published_states;
  // K1 keeps the producer's full two-prefix publication rights in states.
  // These independent original final-prefix views serve the next Gather.
  std::vector<g::StoreSource> successor_states;
  bool mtp_full_candidate = false;
  void release_publication_views() noexcept {
    if (!mtp_full_candidate) return;
    // Model-owned full-prefix views have no consumer binding. Once A has
    // published, B needs only its independent last-prefix views. Keeping the
    // obsolete prefix tickets would exceed the original N + K*D*R pool bound.
    // Preserve hidden's two rows and the identities of every successor view.
    for (std::size_t bank = 0; bank < successor_states.size(); ++bank) states[bank] = {};
  }
};
using IncomingStepSources = inference::IncomingExecutionSources<StepSources>;
struct StepState final {
  // Keep the model's original participant pins alive until after the common
  // resource episode is destroyed. Session pending still borrows this Step.
  hyle::container::SmallVector<StepMember, 1> members;
  inference::ExecutionEpisode<ColdResources, StepSources, 34> execution;
  c::BatchLayoutVersion draft_layout;
  Phase *phase = nullptr;
  Clock::time_point step_start{}, start{};
  double submit_seconds = 0, cold_batch_seconds = 0, shape_admission_seconds = 0;
  std::size_t input_tokens = 0, active_input_rows = 0, active_draft_rows = 0,
      capture_recorded = 0, capture_reused = 0;
  g::ReplayRoute replay_route = g::ReplayRoute::kUnprepared;
  bool prepared = false, submitted = false, cancelled = false;
  std::size_t state_slot = 0;
  bool discard_stopped_members = false;
  bool withdrawn(const StepMember &member) const noexcept;

  void recycle_unpublished() noexcept {
    for (auto &member : members) {
      member.target_publication = {}; member.draft_publication = {};
      member.owner_state->recycle_candidate(std::move(member.next)); member.base = nullptr;
    }
    phase = nullptr;
  }
  hyle::Result<bool> try_abort() noexcept {
    if (execution.unsubmitted_reuse_declined()) {
      if (submitted || !phase) {
        inference::record_retirement_status(execution.terminal,
            hyle::Status::FailedPrecondition("reuse refusal lost its unsubmitted original Phase"));
        return execution.recover([&]() noexcept { recycle_unpublished(); });
      }
      return execution.recover_unsubmitted_reuse(phase->state_resource_ports,
          [&]() noexcept { recycle_unpublished(); });
    }
    return execution.recover([&]() noexcept { recycle_unpublished(); });
  }
  hyle::Result<bool> try_discard_successful() noexcept {
    if (!phase)
      return std::unexpected(hyle::Status::FailedPrecondition("unpublished episode lost its original program"));
    return execution.discard_successful(phase->state_resource_ports,
        [&]() noexcept { recycle_unpublished(); });
  }
};

} // namespace qwen35
