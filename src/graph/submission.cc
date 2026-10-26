#include "submission_internal.h"

#include <array>
#include <mutex>
#include <utility>

#include "../compute/completion_internal.h"
#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/transfer.h"
#include "hyle/core/assert.h"
#include "hyle/graph/detail/region_access.h"
#include "hyle/obs/trace.h"
#include "plan_state.h"

namespace hyle::graph {
using detail::delegated_step_index;
using detail::enqueue_segment_steps;
using detail::shared_lane;

namespace {

constexpr std::uint8_t kSegmentBlocked = 0;
constexpr std::uint8_t kSegmentReady = 1;
constexpr std::uint8_t kSegmentClaimed = 2;
constexpr std::uint8_t kSegmentCommitted = 3;

void enqueue_ready(detail::InvocationState &state, std::uint32_t segment) noexcept {
  auto &progress = state.segment_progress[segment];
  auto &lane = state.lane_progress[state.plan->segments[segment].lane.ordinal()];
  progress.state = kSegmentReady;
  progress.next = detail::kNoNode;
  if (lane.tail == detail::kNoNode)
    lane.head = segment;
  else
    state.segment_progress[lane.tail].next = segment;
  lane.tail = segment;
}

#if defined(HYLE_TESTING)
thread_local detail::AcceptedCompletionFn accepted_observation_hook = nullptr;
#endif

void merge_aggregate_terminal(detail::AggregateCompletionState &aggregate,
                              const Status &terminal) noexcept {
  std::lock_guard lock(aggregate.lock);
  aggregate.terminal.Update(terminal);
}

void release_aggregate_episode(
    detail::AggregateCompletionState *aggregate) noexcept {
  if (aggregate->remaining.fetch_sub(1, std::memory_order_acq_rel) != 1)
    return;
  compute::detail::CompletionLifetime outer;
  Status terminal;
  {
    std::lock_guard lock(aggregate->lock);
    outer = aggregate->outer_lifetime;
    aggregate->outer_lifetime = {};
    terminal = aggregate->terminal;
  }
  if (outer.valid()) {
    outer.terminal(outer.state, terminal);
    // release() may destroy the InvocationState containing aggregate. Do not
    // touch aggregate after this call.
    outer.release(outer.state);
  }
}

void aggregate_child_terminal(void *opaque, const Status &terminal) noexcept {
  auto *child = static_cast<detail::AggregateChildState *>(opaque);
  merge_aggregate_terminal(*child->aggregate, terminal);
}

void aggregate_child_release(void *opaque) noexcept {
  auto *child = static_cast<detail::AggregateChildState *>(opaque);
  auto *invocation = child->invocation;
  child->invocation = nullptr;
  release_aggregate_episode(child->aggregate);
  // The child owns this reference independently of the outer Completion:
  // even an unreturnable prefix cannot destroy aggregate before native proof.
  detail::release(invocation);
}

compute::Completion &
aggregate_terminal_completion(detail::AggregateCompletionState &aggregate,
                              std::size_t index) noexcept {
  return aggregate.completions[aggregate.completion_segments[index]];
}

Result<compute::CompletionPoll> poll_aggregate(void *opaque) noexcept {
  auto &aggregate = *static_cast<detail::AggregateCompletionState *>(opaque);
  Status terminal = Status::Ok();
  bool pending = false;
  for (std::size_t index = 0; index < aggregate.completion_count; ++index) {
    compute::Completion &completion =
        aggregate_terminal_completion(aggregate, index);
    if (!completion.valid()) {
      terminal.Update(Status::Internal("graph terminal completion is invalid"));
      continue;
    }
    auto result = completion.poll();
    if (!result)
      terminal.Update(std::move(result).error());
    else
      pending = pending || *result == compute::CompletionPoll::kPending;
  }
  merge_aggregate_terminal(aggregate, terminal);
  if (pending)
    return compute::CompletionPoll::kPending;
  {
    std::lock_guard lock(aggregate.lock);
    terminal = aggregate.terminal;
  }
  return terminal.ok()
             ? Result<compute::CompletionPoll>(
                   compute::CompletionPoll::kComplete)
             : Result<compute::CompletionPoll>(std::unexpected(terminal));
}

Status wait_aggregate(void *opaque) noexcept {
  auto &aggregate = *static_cast<detail::AggregateCompletionState *>(opaque);
  Status terminal = Status::Ok();
  for (std::size_t index = 0; index < aggregate.completion_count; ++index) {
    compute::Completion &completion =
        aggregate_terminal_completion(aggregate, index);
    if (!completion.valid()) {
      terminal.Update(Status::Internal("graph terminal completion is invalid"));
      continue;
    }
    terminal.Update(completion.wait());
  }
  merge_aggregate_terminal(aggregate, terminal);
  std::lock_guard lock(aggregate.lock);
  return aggregate.terminal;
}

Status cancel_aggregate(void *opaque) noexcept {
  auto &aggregate = *static_cast<detail::AggregateCompletionState *>(opaque);
  Status terminal = Status::Ok();
  bool unsupported = false;
  for (std::size_t index = 0; index < aggregate.completion_count; ++index) {
    compute::Completion &completion =
        aggregate_terminal_completion(aggregate, index);
    if (!completion.valid())
      continue;
    Status result = completion.request_cancel();
    if (result.IsUnimplemented())
      unsupported = true;
    else
      terminal.Update(std::move(result));
  }
  if (!terminal.ok())
    return terminal;
  return unsupported
             ? Status::Unimplemented(
                   "one or more graph terminal segments cannot be cancelled")
             : Status::Ok();
}

void destroy_aggregate(void *opaque,
                       compute::detail::CompletionLifetime lifetime,
                       Status terminal_hint) noexcept {
  auto &aggregate = *static_cast<detail::AggregateCompletionState *>(opaque);
  {
    std::lock_guard lock(aggregate.lock);
    aggregate.terminal.Update(terminal_hint);
    aggregate.outer_lifetime = lifetime;
  }
  // Every selected child was protected before any public poll/wait. An
  // observation failure may already have handed its guard to the native
  // reaper; a cached error cannot be used to decide whether to attach it now.
  for (std::size_t index = 0; index < aggregate.completion_count; ++index) {
    aggregate_terminal_completion(aggregate, index) = {};
  }
  release_aggregate_episode(&aggregate);
}

const compute::detail::CompletionVTable kAggregateCompletionVTable{
    .destroy = &destroy_aggregate,
    .poll = &poll_aggregate,
    .wait = &wait_aggregate,
    .request_cancel = &cancel_aggregate,
};

compute::Completion make_aggregate_completion(
    detail::AggregateCompletionState &aggregate,
    detail::InvocationState *invocation) noexcept {
  // Keep a sentinel until the outer Completion hands off its lifetime.
  aggregate.remaining.store(1, std::memory_order_relaxed);
  {
    std::lock_guard lock(aggregate.lock);
    aggregate.terminal = Status::Ok();
    aggregate.outer_lifetime = {};
  }
  for (std::size_t index = 0; index < aggregate.completion_count; ++index) {
    auto &child = aggregate.children[index];
    HYLE_FATAL_ASSERT(!child.retention.own.valid());
    child.aggregate = &aggregate;
    child.invocation = invocation;
    child.retention.own = {&child, &aggregate_child_terminal, &aggregate_child_release};
    detail::retain(invocation);
    aggregate.remaining.fetch_add(1, std::memory_order_relaxed);
    const auto attached = compute::detail::CompletionAccess::attach_retention(
        aggregate_terminal_completion(aggregate, index), child.retention);
    HYLE_FATAL_ASSERT(attached.ok());
  }
  return compute::detail::CompletionAccess::make(&kAggregateCompletionVTable,
                                                 &aggregate);
}

void retire_invocation(detail::InvocationState *state, const Status &terminal,
                       bool accepted) noexcept {
  // A successful constant/View-only graph can publish without native work.
  state->outputs_ready.store(terminal.ok(), std::memory_order_release);
  if (!state->plan->supports_concurrent_invocations) {
    state->plan->exclusive_in_flight.store(false, std::memory_order_release);
  }
  state->phase.store(accepted && !terminal.ok() ? detail::InvocationPhase::kFailed
                                               : detail::InvocationPhase::kIdle,
                     std::memory_order_release);
  state->phase.notify_all();
}

void mark_invocation_terminal(void *opaque, const Status &terminal) noexcept {
  retire_invocation(static_cast<detail::InvocationState *>(opaque), terminal, true);
}

void record_invocation_terminal(void *opaque, const Status &terminal) noexcept {
  static_cast<detail::InvocationState *>(opaque)->completion_terminal = terminal;
}

void release_invocation_retention(void *opaque) noexcept {
  auto *state = static_cast<detail::InvocationState *>(opaque);
  // attach_retention clears its node and releases the inherited native result
  // first. Publish reuse only afterwards, so a new run cannot overwrite an
  // active retention node while the previous retirement is still using it.
  mark_invocation_terminal(state, state->completion_terminal);
  detail::release(state);
}

void release_preparation_retention(void *opaque) noexcept {
  auto *state = static_cast<detail::InvocationState *>(opaque);
  // Recording does not publish new results or erase an earlier successful
  // output. A failed accepted prefix makes existing output state unusable.
  if (!state->completion_terminal.ok())
    state->outputs_ready.store(false, std::memory_order_release);
  detail::release_preparation(*state);
  state->phase.store(state->completion_terminal.ok()
                         ? detail::InvocationPhase::kIdle
                         : detail::InvocationPhase::kFailed,
                     std::memory_order_release);
  state->phase.notify_all();
  detail::release(state);
}

Result<compute::Completion>
submit_delegated_segment(detail::InvocationState &invocation,
                         const detail::PlanSegment &segment,
                         std::span<const compute::detail::CompletionDependency>
                             dependencies) noexcept {
  auto selected = delegated_step_index(*invocation.plan, segment);
  if (!selected)
    return std::unexpected(std::move(selected).error());
  const detail::PlanStep &step = invocation.plan->steps[*selected];
  const auto all_inputs = std::span<const compute::ConstTensorView>(
      invocation.submission_inputs.data(), invocation.submission_inputs.size());
  const auto all_outputs = std::span<const compute::TensorView>(
      invocation.submission_outputs.data(),
      invocation.submission_outputs.size());
  const auto inputs = all_inputs.subspan(step.input_begin, step.input_count);
  const auto outputs =
      all_outputs.subspan(step.output_begin, step.output_count);
  if (step.region.valid()) {
    return invocation.invocations[*selected].region.submit_delegated(
        dependencies, inputs, outputs);
  }
  if (step.kind == detail::PlanStepKind::kOperation) {
    return compute::detail::OperationInvocationAccess::
        submit_delegated_borrowed_prevalidated(
            invocation.invocations[*selected].operation, dependencies, inputs,
            outputs);
  }
  return std::unexpected(
      Status::Internal("graph delegated step kind is invalid"));
}

} // namespace

Status detail::begin_preparation(InvocationState &state) noexcept {
  HYLE_RETURN_IF_ERROR(resource_control_ready(&state));
  const bool exclusive = !state.plan->supports_concurrent_invocations;
  bool expected = false;
  if (exclusive && !state.plan->exclusive_in_flight.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel, std::memory_order_acquire))
    return Status::FailedPrecondition(
        "a provider in this executable requires serialized invocations");
  std::size_t idle = 0;
  if (!state.preparation_uses.compare_exchange_strong(
          idle, 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
    if (exclusive)
      state.plan->exclusive_in_flight.store(false, std::memory_order_release);
    return Status::Unavailable("graph bindings are already being prepared");
  }
  return Status::Ok();
}

void detail::release_preparation(InvocationState &state) noexcept {
  const auto previous = state.preparation_uses.fetch_sub(1, std::memory_order_acq_rel);
  HYLE_FATAL_ASSERT(previous != 0);
  if (previous == 1 && !state.plan->supports_concurrent_invocations)
    state.plan->exclusive_in_flight.store(false, std::memory_order_release);
}

Status detail::retain_preparation_completion(
    InvocationState &state, compute::Completion &completion) noexcept {
  HYLE_FATAL_ASSERT(!state.completion_retention.own.valid());
  HYLE_FATAL_ASSERT(state.preparation_uses.load(std::memory_order_acquire) != 0);
  state.preparation_uses.fetch_add(1, std::memory_order_relaxed);
  state.completion_terminal = Status::Ok();
  state.phase.store(InvocationPhase::kRunning, std::memory_order_release);
  retain(&state);
  state.completion_retention.own = {
      &state, &record_invocation_terminal, &release_preparation_retention};
  const Status attached = compute::detail::CompletionAccess::attach_retention(
      completion, state.completion_retention);
  // An idle node and a valid sequence Completion always support attachment.
  // Failure after acceptance cannot safely return borrowed state to the caller.
  HYLE_FATAL_ASSERT(attached.ok());
  return attached;
}

bool detail::shared_lane(const detail::PlanState &plan, ExecutionLaneId lane) noexcept {
  return lane.valid() && lane.ordinal() < plan.lanes.size() &&
         plan.lanes[lane.ordinal()].runtime_mode ==
             compute::ProviderRuntimeMode::kSharedQueue;
}

Result<std::size_t>
detail::delegated_step_index(const detail::PlanState &plan,
                     const detail::PlanSegment &segment) noexcept {
  if (!segment.lane.valid() || segment.lane.ordinal() >= plan.lanes.size() ||
      plan.lanes[segment.lane.ordinal()].runtime_mode !=
          compute::ProviderRuntimeMode::kDelegated) {
    return std::unexpected(
        Status::Internal("graph delegated segment has an invalid lane"));
  }
  std::size_t selected = SIZE_MAX;
  const std::size_t end = segment.step_begin + segment.step_count;
  for (std::size_t step_index = segment.step_begin; step_index < end;
       ++step_index) {
    const detail::PlanStep &step = plan.steps[step_index];
    if (!step.emits_command())
      continue;
    if (step.lane != segment.lane ||
        step.runtime_properties().mode !=
            compute::ProviderRuntimeMode::kDelegated ||
        selected != SIZE_MAX) {
      return std::unexpected(Status::Internal(
          "graph delegated segment does not own exactly one command"));
    }
    selected = step_index;
  }
  if (selected == SIZE_MAX) {
    return std::unexpected(
        Status::Internal("graph delegated segment has no command"));
  }
  return selected;
}

Status
detail::enqueue_segment_steps(detail::InvocationState &invocation,
                      const detail::PlanSegment &segment,
                      compute::detail::QueueSequence &sequence) noexcept {
  const detail::PlanState &plan = *invocation.plan;
  const auto all_inputs = std::span<const compute::ConstTensorView>(
      invocation.submission_inputs.data(), invocation.submission_inputs.size());
  const auto all_outputs = std::span<const compute::TensorView>(
      invocation.submission_outputs.data(),
      invocation.submission_outputs.size());
  const std::size_t end = segment.step_begin + segment.step_count;
  for (std::size_t step_index = segment.step_begin; step_index < end;
       ++step_index) {
    const detail::PlanStep &step = plan.steps[step_index];
    if (!step.emits_command())
      continue;
    const auto step_inputs =
        all_inputs.subspan(step.input_begin, step.input_count);
    const auto step_outputs =
        all_outputs.subspan(step.output_begin, step.output_count);
    compute::detail::EnqueueResult enqueued;
    if (step.region.valid()) {
      enqueued = invocation.invocations[step_index].region.enqueue_prevalidated(
          sequence, step_inputs, step_outputs);
    } else if (step.kind == detail::PlanStepKind::kOperation) {
      enqueued =
          compute::detail::OperationInvocationAccess::enqueue_prevalidated(
              invocation.invocations[step_index].operation, sequence,
              step_inputs, step_outputs);
    } else if (step.kind == detail::PlanStepKind::kCopy) {
      const auto enqueue = [&]() noexcept -> compute::detail::EnqueueResult {
        auto source_bytes = step_inputs[0].description().required_bytes();
        if (!source_bytes) return compute::detail::EnqueueResult::rejected(std::move(source_bytes).error());
        auto source = step_inputs[0].storage().subview(0, *source_bytes);
        if (!source) return compute::detail::EnqueueResult::rejected(std::move(source).error());
        auto destination_bytes = step_outputs[0].description().required_bytes();
        if (!destination_bytes) return compute::detail::EnqueueResult::rejected(std::move(destination_bytes).error());
        auto destination = step_outputs[0].storage().subview(0, *destination_bytes);
        if (!destination) return compute::detail::EnqueueResult::rejected(std::move(destination).error());
        return compute::detail::TransferAccess::enqueue_prevalidated_copy(sequence, *destination, *source);
      };
      enqueued = enqueue();
    } else {
      enqueued = compute::detail::EnqueueResult::rejected(
          Status::Internal("compiled graph step kind is invalid"));
    }
    const bool keep_recording = enqueued.status.ok();
    Status status = enqueued.status;
    sequence.record(std::move(enqueued));
    if (!keep_recording)
      return status;
  }
  return Status::Ok();
}

Result<compute::Completion> Invocation::submit(compute::Queue &queue) noexcept {
  if (state_ == nullptr) {
    return std::unexpected(Status::FailedPrecondition("invocation is invalid"));
  }
  if (state_->plan->required_queue_count != 1) {
    return std::unexpected(Status::InvalidArgument(
        "single-queue submit requires exactly one active execution lane"));
  }
  std::uint32_t active_lane = 0;
  while (state_->plan->lane_segment_counts[active_lane] == 0 ||
         !shared_lane(*state_->plan, ExecutionLaneId{active_lane}))
    ++active_lane;
  const std::array bindings{
      QueueBinding{.lane = ExecutionLaneId{active_lane}, .queue = &queue}};
  if (!state_->replay_prepared ||
      (state_->replay_route == ReplayRoute::kCommands &&
       !state_->queues[active_lane].same_owner(queue))) {
    Status prepared = prepare_replay(bindings);
    if (!prepared.ok())
      return std::unexpected(std::move(prepared));
  }
  return submit(bindings);
}

Result<compute::Completion> Invocation::submit() noexcept {
  if (state_ == nullptr) {
    return std::unexpected(Status::FailedPrecondition("invocation is invalid"));
  }
  if (state_->plan->required_queue_count != 0) {
    return std::unexpected(Status::InvalidArgument(
        "queue-free submit requires a graph with no caller Queue lanes"));
  }
  constexpr std::span<const QueueBinding> no_queues{};
  if (!state_->replay_prepared) {
    Status prepared = prepare_replay(no_queues);
    if (!prepared.ok())
      return std::unexpected(std::move(prepared));
  }
  return submit(no_queues);
}

Result<compute::Completion>
Invocation::submit(std::span<const QueueBinding> queues) noexcept {
  HYLE_RETURN_IF_ERROR(detail::resource_control_ready(state_));
  if (state_->progress)
    return state_->progress.submit(*this, queues, {});
  if (state_->resources || state_->input_sources)
    return detail::RegionAccess::submit(*this, queues, {});
  HYLE_ASSIGN_OR_RETURN(auto submission,
      detail::SubmissionCursor::begin(*this, queues, false));
  while (!submission.done()) {
    HYLE_ASSIGN_OR_RETURN(const bool advanced, submission.advance(true));
    if (!advanced && submission.pending_dependency() != nullptr)
      submission.pending_dependency()->wait().IgnoreError();
  }
  HYLE_ASSIGN_OR_RETURN(auto completion, submission.finish());
  return state_->observation.publish(std::move(completion));
}

Result<detail::SubmissionCursor> detail::SubmissionCursor::begin(
    Invocation &invocation, std::span<const QueueBinding> queues,
    bool retain_invocation) noexcept {
  HYLE_TRACE_ZONE_NAMED("hyle.graph.submit");
  auto *state_ = InvocationAccess::state(invocation);
  if (state_ == nullptr) {
    return std::unexpected(Status::FailedPrecondition("invocation is invalid"));
  }
  HYLE_RETURN_IF_ERROR(resource_control_ready(state_));
  auto idle = detail::InvocationPhase::kIdle;
  if (!state_->phase.compare_exchange_strong(
          idle, detail::InvocationPhase::kRunning, std::memory_order_acq_rel,
          std::memory_order_acquire)) {
    return std::unexpected(Status::FailedPrecondition(
        idle == detail::InvocationPhase::kFailed
            ? "failed graph invocation must be recreated"
            : "invocation already has work in flight"));
  }
  const detail::PlanState &plan = *state_->plan;
  bool exclusive = false;
  if (!plan.supports_concurrent_invocations &&
      !state_->plan->exclusive_in_flight.compare_exchange_strong(
          exclusive, true, std::memory_order_acq_rel,
          std::memory_order_acquire)) {
    state_->phase.store(detail::InvocationPhase::kIdle,
                        std::memory_order_release);
    return std::unexpected(Status::FailedPrecondition(
        "a provider in this executable requires serialized invocations"));
  }
  state_->outputs_ready.store(false, std::memory_order_release);
  struct RunRollback final {
    detail::InvocationState *state = nullptr;
    bool plan_exclusive = false;
    bool armed = true;
    ~RunRollback() {
      if (armed) {
        if (plan_exclusive) {
          state->plan->exclusive_in_flight.store(false,
                                                 std::memory_order_release);
        }
        state->phase.store(detail::InvocationPhase::kIdle,
                           std::memory_order_release);
      }
    }
    void disarm() noexcept { armed = false; }
  } rollback{state_, !plan.supports_concurrent_invocations};

  if (!state_->replay_prepared) {
    return std::unexpected(Status::FailedPrecondition(
        "multi-domain invocation requires prepare_replay()"));
  }
  if (queues.size() != plan.required_queue_count) {
    return std::unexpected(Status::InvalidArgument(
        "submit requires one queue for every active execution domain"));
  }
  for (std::size_t index = 0; index < state_->queue_seen.size(); ++index)
    state_->queue_seen[index] = 0;
  for (const QueueBinding binding : queues) {
    if (!binding.lane.valid() || binding.lane.ordinal() >= plan.lanes.size() ||
        binding.queue == nullptr) {
      return std::unexpected(
          Status::InvalidArgument("graph submit queue binding is invalid"));
    }
    const std::size_t lane = binding.lane.ordinal();
    if (plan.lane_segment_counts[lane] == 0 ||
        !shared_lane(plan, binding.lane) || state_->queue_seen[lane] != 0 ||
        !state_->queues[lane].same_owner(*binding.queue)) {
      return std::unexpected(Status::InvalidArgument(
          "graph submit queues differ from prepared replay queues"));
    }
    // Compare the actual retained Queue state, not the public handle address.
    // A moved handle may name this same Queue from a different address.
    const ExecutionTargetId target = plan.lanes[lane].target;
    if (!detail::ExecutionSourceAccess::owns(
            plan.targets[target.ordinal()].source, state_->queues[lane])) {
      return std::unexpected(Status::InvalidArgument(
          "graph submit queue belongs to a different compute context"));
    }
    state_->queue_seen[lane] = 1;
  }
  for (std::size_t lane = 0; lane < state_->queue_seen.size(); ++lane) {
    if (plan.lane_segment_counts[lane] != 0 &&
        shared_lane(plan, ExecutionLaneId{static_cast<std::uint32_t>(lane)}) &&
        state_->queue_seen[lane] == 0) {
      return std::unexpected(
          Status::InvalidArgument("graph submit queue binding is missing"));
    }
  }
  if (!state_->bindings_prepared || !state_->transfers_prepared ||
      !state_->dependencies_prepared) {
    return std::unexpected(Status::Internal(
        "prepared graph replay lost a binding, transfer, or dependency route"));
  }


  for (compute::Completion &completion : state_->segment_completions)
    completion = {};
  for (auto &lane : state_->lane_progress) {
    lane.head = lane.tail = detail::kNoNode;
    lane.claimed = false;
  }
  for (std::size_t segment = 0; segment < plan.segments.size(); ++segment) {
    auto &progress = state_->segment_progress[segment];
    progress = {};
    progress.remaining = plan.segments[segment].dependency_count;
    progress.state = kSegmentBlocked;
    if (progress.remaining == 0)
      enqueue_ready(*state_, static_cast<std::uint32_t>(segment));
  }
  SubmissionCursor result(state_, retain_invocation);
  rollback.disarm();
  return result;
}

detail::SubmissionCursor::SubmissionCursor(InvocationState *state,
                                           bool retained) noexcept
    : state_(state), retained_(retained) {
  if (retained_)
    retain(state_);
}

detail::SubmissionCursor::SubmissionCursor(SubmissionCursor &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)),
      completed_segments_(other.completed_segments_),
      claimed_segments_(other.claimed_segments_), next_lane_(other.next_lane_),
      accepted_order_(other.accepted_order_),
      tail_error_(std::move(other.tail_error_)), pending_(other.pending_),
      pending_segment_(other.pending_segment_),
      finished_(other.finished_), retained_(other.retained_) {
  other.pending_ = nullptr;
}

detail::SubmissionCursor &detail::SubmissionCursor::operator=(
    SubmissionCursor &&other) noexcept {
  if (this != &other) {
    reset();
    state_ = std::exchange(other.state_, nullptr);
    completed_segments_ = other.completed_segments_;
    claimed_segments_ = other.claimed_segments_;
    next_lane_ = other.next_lane_;
    accepted_order_ = other.accepted_order_;
    tail_error_ = std::move(other.tail_error_);
    pending_ = std::exchange(other.pending_, nullptr);
    pending_segment_ = other.pending_segment_;
    finished_ = other.finished_;
    retained_ = other.retained_;
  }
  return *this;
}

detail::SubmissionCursor::~SubmissionCursor() { reset(); }

void detail::SubmissionCursor::reset() noexcept {
  if (state_ != nullptr && !finished_) {
    // Private emergency cleanup only. Public asynchronous composition retains
    // the cursor until finish() transfers this accepted prefix to Completion.
    for (auto &completion : state_->segment_completions) {
      if (completion.valid())
        completion.wait().IgnoreError();
    }
    mark_invocation_terminal(state_, Status::Cancelled(
        "graph submission cursor abandoned before completion handoff"));
  }
  auto *state = std::exchange(state_, nullptr);
  if (retained_)
    release(state);
}

bool detail::SubmissionCursor::done() const noexcept {
  return state_ == nullptr || finished_ || !tail_error_.ok() ||
         completed_segments_ == state_->plan->segments.size();
}

std::size_t detail::SubmissionCursor::claim_ready(ExecutionTargetId target) noexcept {
  if (done()) return SIZE_MAX;
  const auto &plan = *state_->plan;
  for (std::size_t offset = 0; offset < plan.lanes.size(); ++offset) {
    const auto index = (next_lane_ + offset) % plan.lanes.size();
    auto &lane = state_->lane_progress[index];
    if ((target.valid() && plan.lanes[index].target != target) ||
        lane.head == detail::kNoNode ||
        state_->lane_progress[lane.producer_group].claimed)
      continue;
    const auto segment = lane.head;
    auto &progress = state_->segment_progress[segment];
    HYLE_FATAL_ASSERT(progress.state == kSegmentReady && progress.remaining == 0);
    lane.head = progress.next;
    if (lane.head == detail::kNoNode) lane.tail = detail::kNoNode;
    progress.next = detail::kNoNode;
    progress.state = kSegmentClaimed;
    state_->lane_progress[lane.producer_group].claimed = true;
    ++claimed_segments_;
    next_lane_ = (index + 1) % plan.lanes.size();
    return segment;
  }
  return SIZE_MAX;
}

void detail::SubmissionCursor::defer_segment(std::size_t segment) noexcept {
  HYLE_FATAL_ASSERT(state_ && !finished_ && segment < state_->segment_progress.size());
  auto &progress = state_->segment_progress[segment];
  HYLE_FATAL_ASSERT(progress.state == kSegmentClaimed && claimed_segments_ != 0);
  const auto lane = state_->plan->segments[segment].lane.ordinal();
  state_->lane_progress[state_->lane_progress[lane].producer_group].claimed = false;
  --claimed_segments_;
  enqueue_ready(*state_, static_cast<std::uint32_t>(segment));
}

bool detail::SubmissionCursor::has_parallel_targets() const noexcept {
  return state_ && state_->plan->has_parallel_targets;
}

const compute::detail::QueueRef *
detail::SubmissionCursor::pending_queue() const noexcept {
  if (state_ == nullptr || pending_ == nullptr ||
      pending_segment_ >= state_->plan->segments.size())
    return nullptr;
  const auto lane = state_->plan->segments[pending_segment_].lane;
  return shared_lane(*state_->plan, lane)
             ? &state_->queues[lane.ordinal()]
             : nullptr;
}

void detail::SubmissionCursor::request_cancel() noexcept {
  reject_successors(Status::Cancelled("graph successors cancelled"));
}

void detail::SubmissionCursor::reject_successors(Status terminal) noexcept {
  if (!finished_ && tail_error_.ok() && !terminal.ok())
    tail_error_ = std::move(terminal);
}

Result<bool> detail::SubmissionCursor::dependencies_ready(std::size_t index) noexcept {
  if (!state_ || finished_ || index >= state_->plan->segments.size() ||
      state_->segment_progress[index].state != kSegmentClaimed)
    return std::unexpected(Status::FailedPrecondition("graph segment is not claimed"));
  const auto &segment = state_->plan->segments[index];
  for (std::size_t i = segment.dependency_begin;
       i < segment.dependency_begin + segment.dependency_count; ++i) {
    if (state_->dependency_routes[i] != ReplayDependencyRoute::kHostWait) continue;
    const auto source = state_->plan->segment_dependencies[i];
    std::lock_guard guard(state_->completion_observation_lock);
    auto ready = state_->segment_completions[source].poll();
    if (!ready) return std::unexpected(std::move(ready).error());
    if (*ready == compute::CompletionPoll::kPending) return false;
  }
  return input_sources_ready(*state_, index);
}

detail::SegmentSubmission detail::SubmissionCursor::submit_segment(
    std::size_t segment_index, bool wait_dependencies, SharedSequenceFn admission,
    void *admission_owner, bool dependencies_prechecked) noexcept {
  SegmentSubmission outcome;
  if (state_ == nullptr || finished_ || segment_index >= state_->plan->segments.size() ||
      state_->segment_progress[segment_index].state != kSegmentClaimed) {
    outcome.status = Status::FailedPrecondition("graph segment submission is invalid or finished");
    return outcome;
  }
  const auto &plan = *state_->plan;
  auto &tail_error = outcome.status;
  do {
    const detail::PlanSegment &segment = plan.segments[segment_index];
    const bool is_shared = shared_lane(plan, segment.lane);
    const compute::detail::QueueRef *queue =
        is_shared ? &state_->queues[segment.lane.ordinal()] : nullptr;
    bool continue_segment = true;
    const std::size_t dependency_end =
        segment.dependency_begin + segment.dependency_count;

    if (!dependencies_prechecked) {
      if (wait_dependencies) {
        const auto ready = wait_input_sources(*state_, segment_index);
        if (!ready.ok()) { tail_error = ready; break; }
      } else {
        auto ready = input_sources_ready(*state_, segment_index);
        if (!ready) { tail_error = ready.error(); break; }
        if (!*ready) {
          // The source is external, so it has no original Plan segment cell.
          // The asynchronous owner polls it before dispatching this segment.
          outcome.pending_input = true;
          return outcome;
        }
      }
    }

    // A host fallback is a cold-selected, public replay fact. Complete it
    // before taking the destination Queue's single-producer admission so a
    // slow upstream device cannot monopolize an otherwise idle Queue.
    for (std::size_t dependency_index = segment.dependency_begin;
         dependency_index < dependency_end; ++dependency_index) {
      const std::uint32_t source_segment =
          plan.segment_dependencies[dependency_index];
      if (source_segment >= segment_index ||
          !state_->segment_completions[source_segment].valid()) {
        tail_error =
            Status::Internal("graph segment dependency is not available");
        continue_segment = false;
        break;
      }
      compute::Completion &source = state_->segment_completions[source_segment];
      const ReplayDependencyRoute route =
          state_->dependency_routes[dependency_index];
      if (route == ReplayDependencyRoute::kOrderedQueue ||
          route == ReplayDependencyRoute::kSynchronousSource)
        continue;
      if (route == ReplayDependencyRoute::kHostWait) {
        if (dependencies_prechecked) continue;
        if (wait_dependencies) {
          Status host = source.wait();
          if (!host.ok()) {
            tail_error = std::move(host);
            continue_segment = false;
            break;
          }
          continue;
        }
        auto host = source.poll();
        if (!host) {
          tail_error = std::move(host).error();
          continue_segment = false;
          break;
        }
        if (*host == compute::CompletionPoll::kPending) {
          outcome.pending = &source;
          outcome.pending_segment = source_segment;
          return outcome;
        }
      } else if (route != ReplayDependencyRoute::kDeviceWait) {
        tail_error =
            Status::Internal("prepared graph dependency route is invalid");
        continue_segment = false;
        break;
      }
    }
    if (!continue_segment)
      break;

    if (is_shared) {
      if (queue == nullptr || !queue->valid()) {
        tail_error =
            Status::Internal("prepared graph shared lane has no Queue binding");
        break;
      }
      auto sequence = compute::detail::QueueSequence::begin(*queue);
      if (!sequence) {
        tail_error = std::move(sequence).error();
        break;
      }
      const auto inputs = enqueue_input_sources(*state_, segment_index, *sequence);
      if (!inputs.ok()) {
        tail_error = inputs;
        continue_segment = false;
      }
      if (continue_segment && admission) {
        const auto admitted = admission(admission_owner, segment.lane, *sequence);
        if (!admitted.ok()) {
          tail_error = admitted;
          sequence->record(compute::detail::EnqueueResult::rejected(admitted));
          continue_segment = false;
        }
      }
      for (std::size_t dependency_index = segment.dependency_begin;
           continue_segment && dependency_index < dependency_end; ++dependency_index) {
        if (state_->dependency_routes[dependency_index] ==
            ReplayDependencyRoute::kDeviceWait) {
          const std::uint32_t source_segment =
              plan.segment_dependencies[dependency_index];
          compute::Completion &source =
              state_->segment_completions[source_segment];
          compute::detail::EnqueueResult waited;
          if (state_->progress) {
            std::lock_guard guard(state_->completion_observation_lock);
            waited = sequence->enqueue_wait(source);
          } else {
            waited = sequence->enqueue_wait(source);
          }
          const bool wait_succeeded = waited.status.ok();
          if (!wait_succeeded) {
            tail_error =
                waited.status.IsUnimplemented()
                    ? Status::Internal(
                          "prepared graph completion-wait route was rejected")
                    : waited.status;
          }
          sequence->record(std::move(waited));
          if (!wait_succeeded) {
            continue_segment = false;
            break;
          }
        }
      }

      if (continue_segment) {
        const auto span = state_->capture_spans[segment_index];
        std::size_t cursor = segment.step_begin;
        const std::size_t end = segment.step_begin + segment.step_count;
        for (std::size_t index = span.begin;
             continue_segment && index < span.begin + span.count; ++index) {
          auto &range = state_->captured_ranges[index];
          if (cursor < range.step_begin) {
            auto gap = segment;
            gap.step_begin = static_cast<std::uint32_t>(cursor);
            gap.step_count = range.step_begin - gap.step_begin;
            tail_error = enqueue_segment_steps(*state_, gap, *sequence);
            continue_segment = tail_error.ok();
          }
          if (continue_segment) {
            auto launched = range.commands.enqueue(*sequence);
            continue_segment = launched.status.ok();
            if (!continue_segment)
              tail_error = launched.status;
            sequence->record(std::move(launched));
          }
          cursor = range.step_begin + range.step_count;
        }
        if (continue_segment && cursor < end) {
          auto gap = segment;
          gap.step_begin = static_cast<std::uint32_t>(cursor);
          gap.step_count = static_cast<std::uint32_t>(end - cursor);
          tail_error = enqueue_segment_steps(*state_, gap, *sequence);
          continue_segment = tail_error.ok();
        }
      }
      auto completion = sequence->finish();
      if (!completion) {
        tail_error = std::move(completion).error();
        break;
      }
      outcome.completion = std::move(*completion);
    } else {
      // A delegated import consumes live source identities in its enqueue
      // callback. Actual owner-runtime math stays in that runtime's job.
      std::unique_lock<util::SpinLock> imports(state_->completion_observation_lock,
                                             std::defer_lock);
      if (state_->progress) {
        for (std::size_t i = segment.dependency_begin; i < dependency_end; ++i)
          if (state_->dependency_routes[i] == ReplayDependencyRoute::kDeviceWait) {
            imports.lock();
            break;
          }
      }
      std::size_t imported_count = 0;
      for (std::size_t dependency_index = segment.dependency_begin;
           dependency_index < dependency_end; ++dependency_index) {
        if (state_->dependency_routes[dependency_index] !=
            ReplayDependencyRoute::kDeviceWait) {
          continue;
        }
        const std::uint32_t source_segment =
            plan.segment_dependencies[dependency_index];
        compute::Completion &source =
            state_->segment_completions[source_segment];
        const compute::detail::CompletionDependency dependency =
            compute::detail::CompletionAccess::dependency(source);
        if (!dependency.pending()) {
          if (dependency.terminal_hint != nullptr &&
              !dependency.terminal_hint->ok()) {
            tail_error = *dependency.terminal_hint;
            continue_segment = false;
            break;
          }
          continue;
        }
        state_->completion_dependencies[segment.dependency_begin +
                                        imported_count] = dependency;
        ++imported_count;
      }
      if (continue_segment) {
        const auto dependencies =
            imported_count == 0
                ? std::span<const compute::detail::CompletionDependency>{}
                : std::span<const compute::detail::CompletionDependency>(
                      state_->completion_dependencies.data() +
                          segment.dependency_begin,
                      imported_count);
        auto completion =
            submit_delegated_segment(*state_, segment, dependencies);
        if (!completion) {
          tail_error = std::move(completion).error();
          continue_segment = false;
        } else {
          outcome.completion = std::move(*completion);
        }
      }
    }
    if (!continue_segment)
      break;

  } while (false);
  return outcome;
}

Status detail::SubmissionCursor::commit_segment(std::size_t segment_index,
    SegmentSubmission outcome, AcceptedCompletionFn accepted, void *owner) noexcept {
  if (state_ == nullptr || finished_ || segment_index >= state_->segment_completions.size() ||
      state_->segment_progress[segment_index].state != kSegmentClaimed ||
      state_->segment_completions[segment_index].valid() || outcome.pending != nullptr || outcome.pending_input)
    return Status::FailedPrecondition("graph segment outcome is not publishable");
  auto &progress = state_->segment_progress[segment_index];
  if (outcome.completion.valid()) {
    auto &completion = state_->segment_completions[segment_index];
    completion = std::move(outcome.completion);
    // One producer group remains claimed until this publication, so this is
    // the actual Queue prefix order even across differently numbered lanes.
    progress.accepted_order = ++accepted_order_;
    if (accepted) accepted(owner, segment_index, completion);
#if defined(HYLE_TESTING)
    if (accepted && accepted_observation_hook)
      accepted_observation_hook(nullptr, segment_index, completion);
#endif
  } else if (outcome.status.ok()) {
    outcome.status = Status::Internal("graph segment accepted no completion");
  }
  const auto &segment = state_->plan->segments[segment_index];
  auto &lane = state_->lane_progress[segment.lane.ordinal()];
  state_->lane_progress[lane.producer_group].claimed = false;
  HYLE_FATAL_ASSERT(claimed_segments_ != 0);
  --claimed_segments_;
  progress.state = kSegmentCommitted;
  ++completed_segments_;
  tail_error_.Update(outcome.status);
  if (outcome.status.ok()) {
    const auto &plan = *state_->plan;
    const auto end = segment.successor_begin + segment.successor_count;
    for (auto edge = segment.successor_begin; edge < end; ++edge) {
      const auto successor = plan.segment_successors[edge];
      auto &destination = state_->segment_progress[successor];
      HYLE_FATAL_ASSERT(destination.state == kSegmentBlocked && destination.remaining != 0);
      if (--destination.remaining == 0) enqueue_ready(*state_, successor);
    }
  }
  return outcome.status;
}

void detail::SubmissionCursor::finish_submissions(Status terminal) noexcept {
  if (state_ == nullptr || finished_) return;
  reject_successors(std::move(terminal));
  if (tail_error_.ok() && completed_segments_ != state_->plan->segments.size())
    reject_successors(Status::Internal("graph progress stopped before all ready segments were committed"));
  pending_ = nullptr;
  pending_segment_ = SIZE_MAX;
}

Result<bool> detail::SubmissionCursor::advance(bool wait_dependencies,
    AcceptedCompletionFn accepted, void *owner, SharedSequenceFn admission,
    void *admission_owner) noexcept {
  if (state_ == nullptr || finished_)
    return std::unexpected(Status::FailedPrecondition(
        "graph submission cursor is invalid or finished"));
  if (done()) return true;
  pending_ = nullptr;
  pending_segment_ = SIZE_MAX;
  std::size_t first_deferred = SIZE_MAX;
  std::size_t pending_input_segment = SIZE_MAX;
  for (;;) {
    const auto segment = claim_ready();
    if (segment == SIZE_MAX) break;
    if (segment == first_deferred) {
      defer_segment(segment);
      break; // One bounded pass through the ready lanes, without busy-spin.
    }
    auto outcome = submit_segment(segment, false, admission, admission_owner);
    if (outcome.pending != nullptr || outcome.pending_input) {
      if (first_deferred == SIZE_MAX) first_deferred = segment;
      if (outcome.pending_input && pending_input_segment == SIZE_MAX)
        pending_input_segment = segment;
      if (outcome.pending != nullptr && pending_ == nullptr) {
        pending_ = outcome.pending;
        pending_segment_ = outcome.pending_segment;
      }
      defer_segment(segment);
      continue;
    }
    const auto committed = commit_segment(segment, std::move(outcome), accepted, owner);
    tail_error_.Update(committed);
    return true;
  }
  if (pending_ != nullptr) {
    if (!wait_dependencies) return false;
    const auto terminal = pending_->wait();
    if (!terminal.ok()) {
      reject_successors(terminal);
      return true;
    }
    // wait() settled the original dependency. Retry the same ready index;
    // independent work was already offered before taking this host barrier.
    return advance(false, accepted, owner, admission, admission_owner);
  }
  if (pending_input_segment != SIZE_MAX) {
    if (!wait_dependencies) return false;
    const auto terminal = wait_input_sources(*state_, pending_input_segment);
    if (!terminal.ok()) {
      reject_successors(terminal);
      return true;
    }
    // An immutable external source has no Plan Completion cell. Settle its
    // actual producer only after independent ready segments were offered.
    return advance(false, accepted, owner, admission, admission_owner);
  }
  if (claimed_segments_ != 0) return false; // The composition owner has work.
  reject_successors(Status::Internal("graph dependency frontier made no progress"));
  return true;
}

Result<compute::Completion> detail::SubmissionCursor::finish(bool defer_retirement) noexcept {
  if (state_ == nullptr || finished_ || !done() || claimed_segments_ != 0)
    return std::unexpected(Status::FailedPrecondition(
        "graph submission cursor cannot finish while successors remain"));
  auto &tail_error = tail_error_;
  compute::Completion final_completion;

  auto assemble_completion = [&](std::span<const std::uint32_t> segments,
                                 const Status &terminal) noexcept -> Status {
    if (segments.empty())
      return terminal;
    for (std::uint32_t segment : segments) {
      if (segment >= state_->segment_completions.size() ||
          !state_->segment_completions[segment].valid()) {
        return Status::Internal(
            "graph completion segment is unavailable after submission");
      }
    }
    if (segments.size() == 1) {
      final_completion =
          std::move(state_->segment_completions[segments.front()]);
      compute::detail::CompletionAccess::append_terminal(final_completion,
                                                         terminal);
      return Status::Ok();
    }
    for (std::size_t index = 0; index < segments.size(); ++index)
      state_->aggregate_segments[index] = segments[index];
    state_->aggregate_completion.completion_count = segments.size();
    final_completion = make_aggregate_completion(state_->aggregate_completion, state_);
    if (!terminal.ok()) {
      merge_aggregate_terminal(state_->aggregate_completion, terminal);
      compute::detail::CompletionAccess::append_terminal(final_completion,
                                                         terminal);
    }
    return Status::Ok();
  };

  // Only used for an internal assembly failure after acceptance. No borrowed
  // run storage can escape while its Completion is unavailable to the caller.
  auto retire_unreturnable_prefix = [&](const Status &failure) noexcept {
    for (auto &completion : state_->segment_completions) {
      if (completion.valid())
        completion.wait().IgnoreError();
    }
    if (!defer_retirement) mark_invocation_terminal(state_, failure);
    finished_ = true;
  };

  // begin() cleared every previous episode's Completion. Keep every accepted
  // child from this episode, including earlier prefixes of the same Queue.
  // A later prefix proves command ordering but does not release an earlier
  // Completion's own retention; all of those guards must retire before the
  // leased invocation can publish results. This also preserves every accepted
  // child when a synchronous tail rejection prevents later submission.
  std::size_t accepted_count = 0;
  for (std::size_t index = 0; index < state_->segment_completions.size(); ++index) {
    if (state_->segment_completions[index].valid())
      state_->aggregate_segments[accepted_count++] = static_cast<std::uint32_t>(index);
  }
  if (!tail_error.ok() && accepted_count == 0) {
    // An opt-in owner still has an admission/driver guard even with no
    // native acceptance. It alone ends the last possible access to the slot.
    if (!defer_retirement) retire_invocation(state_, tail_error, false);
    finished_ = true;
    return std::unexpected(std::move(tail_error));
  }
  const Status assembled = assemble_completion(
      std::span<const std::uint32_t>(state_->aggregate_segments.data(), accepted_count),
      tail_error);
  if (!assembled.ok()) {
    retire_unreturnable_prefix(assembled);
    return std::unexpected(assembled);
  }
  tail_error = Status::Ok();
  if (!final_completion.valid())
    final_completion = compute::Completion::completed();

  if (defer_retirement) {
    // The opt-in owner has retained Invocation before admission and attached
    // every accepted child before observation. It publishes the actual final
    // terminal, so a cached observation error cannot release plan exclusion.
    finished_ = true;
    return final_completion;
  }

  if (!compute::detail::CompletionAccess::has_async_state(final_completion)) {
    mark_invocation_terminal(state_, final_completion.wait());
    finished_ = true;
    return final_completion;
  }

  detail::retain(state_);
  state_->completion_retention.own = compute::detail::CompletionLifetime{
      .state = state_,
      .terminal = &record_invocation_terminal,
      .release = &release_invocation_retention,
  };
  Status attached = compute::detail::CompletionAccess::attach_retention(
      final_completion, state_->completion_retention);
  if (!attached.ok()) {
    // This is an invariant/error path after work may have been accepted.
    // Drain before returning without a Completion, then retire the slot.
    final_completion.wait().IgnoreError();
    state_->completion_retention = {};
    mark_invocation_terminal(state_, attached);
    detail::release(state_); // attach_retention consumes own only on success.
    finished_ = true;
    return std::unexpected(std::move(attached));
  }
  finished_ = true;
  return final_completion;
}

void detail::InvocationAccess::retire(InvocationState &invocation,
                                    const Status &terminal,
                                    bool accepted) noexcept {
  retire_invocation(&invocation, terminal, accepted);
}
#if defined(HYLE_TESTING)
detail::AcceptedCompletionFn detail::set_accepted_observation_hook(AcceptedCompletionFn hook) noexcept {
  return std::exchange(accepted_observation_hook, hook);
}
#endif

} // namespace hyle::graph
