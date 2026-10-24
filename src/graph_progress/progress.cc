#include "hyle/graph/progress.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <new>
#include <utility>

#include "hyle/compute/detail/retained_completion.h"
#include "hyle/core/assert.h"
#include "hyle/sched/blocking.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/sync.h"
#include "hyle/util/ref_counted.h"
#include "../graph/plan_state.h"
#include "../graph/progress_internal.h"
#include "../graph/resource_episode.h"
#include "../graph/submission_internal.h"
#include "../graph/submission_lifetime.h"

namespace hyle::graph {
namespace {

namespace gd = graph::detail;
namespace cd = compute::detail;

struct Control;
struct JobSlot final {
  sched::detail::BlockingJob job{};
  Control *owner = nullptr;
  std::size_t segment = SIZE_MAX;
  gd::SegmentSubmission outcome{};
  bool busy = false;
  bool ready = false;
};

// One cold attachment and notification owner, not another execution slot.
// The Invocation owns the idle reference. Pending observation hooks and posted
// jobs retain this control independently, without retaining the Invocation.
struct Control final : util::RefCounted<Control> {
  Control(gd::InvocationState &state, sched::Scheduler &selected_scheduler,
          sched::BlockingPool &selected_pool) noexcept
      : allocator(state.allocator), invocation(&state),
        scheduler(&selected_scheduler), pool(&selected_pool) {}

  ~Control() {
    for (std::size_t i = 0; i < job_count; ++i) jobs[i].~JobSlot();
    if (jobs) allocator.deallocate(jobs, job_count * sizeof(JobSlot), alignof(JobSlot));
  }
  void destroy() noexcept {
    const auto selected = allocator;
    this->~Control();
    selected.deallocate(this, sizeof(Control), alignof(Control));
  }

  mem::AllocatorRef allocator;
  gd::InvocationState *invocation;
  sched::Scheduler *scheduler;
  sched::BlockingPool *pool;
  std::mutex mutex;
  sched::ConditionVariable changed;
  // Only permission/admission metadata uses this lock. Provider math and waits
  // remain outside it, including the synchronous body of CPU Queue submission.
  std::mutex admission_mutex;
  std::atomic<bool> cancel_requested{false};
  std::uint64_t revision = 0;
  bool running = false;
  JobSlot *jobs = nullptr;
  std::size_t job_count = 0;
  gd::SubmissionCursor cursor;
  cd::RetainedCompletionHandoff ticket;
};

void notify_control(void *opaque) noexcept {
  auto &control = *static_cast<Control *>(opaque);
  std::lock_guard guard(control.mutex);
  ++control.revision;
  control.changed.notify_all();
}

void release_control(void *opaque) noexcept {
  static_cast<Control *>(opaque)->decref();
}

void cancel_control(void *opaque) noexcept {
  auto &control = *static_cast<Control *>(opaque);
  control.cancel_requested.store(true, std::memory_order_release);
  notify_control(&control);
}

void wait_for_handoff(void *opaque, const std::atomic<std::uint64_t> &revision,
                      std::uint64_t observed) noexcept {
  auto &control = *static_cast<Control *>(opaque);
  std::unique_lock lock(control.mutex);
  control.changed.wait(lock, [&] {
    return revision.load(std::memory_order_acquire) != observed;
  });
}

Status admit_sequence(void *opaque, ExecutionLaneId lane,
                      cd::QueueSequence &sequence) noexcept {
  auto &control = *static_cast<Control *>(opaque);
  std::lock_guard guard(control.admission_mutex);
  return gd::admit_submission_sequence(control.invocation, lane, sequence);
}

void run_job(void *opaque) noexcept {
  auto &slot = *static_cast<JobSlot *>(opaque);
  auto &control = *slot.owner;
  if (control.cancel_requested.load(std::memory_order_acquire)) {
    slot.outcome.status = Status::Cancelled("graph segment cancelled before admission");
    return;
  }
  const auto before = gd::submission_admission_enabled(*control.invocation)
      ? &admit_sequence : nullptr;
  slot.outcome = control.cursor.submit_segment(slot.segment, false, before, &control, true);
}

void cancel_job(void *opaque) noexcept {
  auto &slot = *static_cast<JobSlot *>(opaque);
  slot.outcome.status = Status::Cancelled("graph blocking pool admission is closed");
}

void destroy_job(void *opaque) noexcept {
  auto &slot = *static_cast<JobSlot *>(opaque);
  auto *control = slot.owner;
  {
    std::lock_guard guard(control->mutex);
    // run/cancel returning alone is insufficient: the pool still reads this
    // embedded job header to call destroy. Publish only after its last access.
    slot.ready = true;
    ++control->revision;
    control->changed.notify_all();
  }
  // The coordinator may already reuse the slot. Do not access it again.
  control->decref();
}

void complete_driver(Control &control) noexcept {
  auto *state = control.invocation;
  auto completion = gd::finish_submission(*state, control.cursor);
  const Status terminal = completion ? Status::Ok() : Status{completion.error()};
  auto actual = completion ? std::move(*completion)
                           : compute::Completion::completed(terminal);
  auto ticket = std::move(control.ticket);
  control.cursor = {};
  {
    std::lock_guard guard(control.mutex);
    control.running = false;
    ++control.revision;
    control.changed.notify_all();
  }
  // The public observer must not report success before the slot can publish
  // its result. Return the driver sentinel before exposing the actual native
  // completion. This short reference protects synchronous retirement and an
  // abandoned public Invocation through handoff, without delaying its phase.
  gd::retain(state);
  gd::finish_submission_lifetime(*state, terminal);
  HYLE_FATAL_ASSERT(ticket.handoff(std::move(actual)).ok());
  gd::release(state);
}

void drive(void *opaque) noexcept {
  auto &control = *static_cast<Control *>(opaque);
  auto &cursor = control.cursor;
  auto *state = control.invocation;
  std::size_t busy = 0;
  for (;;) {
    std::uint64_t observed;
    {
      std::lock_guard guard(control.mutex);
      observed = control.revision;
    }
    if (control.cancel_requested.load(std::memory_order_acquire) ||
        sched::shutdown_requested()) {
      control.cancel_requested.store(true, std::memory_order_release);
      cursor.request_cancel();
    }

    bool progressed = false;
    for (std::size_t lane = 0; lane < control.job_count; ++lane) {
      gd::SegmentSubmission outcome;
      std::size_t segment = SIZE_MAX;
      {
        std::lock_guard guard(control.mutex);
        auto &slot = control.jobs[lane];
        if (!slot.ready) continue;
        segment = slot.segment;
        outcome = std::move(slot.outcome);
        slot.segment = SIZE_MAX;
        slot.ready = false;
        slot.busy = false;
      }
      --busy;
      if (outcome.pending != nullptr || outcome.pending_input) {
        cursor.defer_segment(segment);
      } else {
        cursor.commit_segment(segment, std::move(outcome),
            &gd::retain_submission_completion, state).IgnoreError();
      }
      progressed = true;
    }

    bool pending_host = false;
    if (!cursor.done()) {
      // A bounded pass preserves the original lane rotation even if every
      // ready segment currently depends on an unretired host prefix.
      const auto limit = state->plan->segments.size();
      for (std::size_t attempt = 0; attempt < limit && !cursor.done(); ++attempt) {
        if (control.cancel_requested.load(std::memory_order_acquire) ||
            sched::shutdown_requested()) {
          control.cancel_requested.store(true, std::memory_order_release);
          cursor.request_cancel();
          break;
        }
        const auto segment = cursor.claim_ready();
        if (segment == SIZE_MAX) break;
        auto ready = cursor.dependencies_ready(segment);
        if (!ready) {
          gd::SegmentSubmission rejected;
          rejected.status = ready.error();
          cursor.commit_segment(segment, std::move(rejected),
              &gd::retain_submission_completion, state).IgnoreError();
          progressed = true;
          continue;
        }
        if (!*ready) {
          pending_host = true;
          cursor.defer_segment(segment);
          continue;
        }
        const auto lane = state->plan->segments[segment].lane.ordinal();
        if (state->segment_blocks_host[segment]) {
          auto &slot = control.jobs[lane];
          {
            std::lock_guard guard(control.mutex);
            HYLE_FATAL_ASSERT(!slot.busy && !slot.ready);
            slot.busy = true;
            slot.segment = segment;
            slot.outcome = {};
          }
          ++busy;
          control.incref(); // pool-side control, retained through destroy
          if (!control.pool->submit(&slot.job)) {
            slot.job.cancel();
            slot.job.destroy();
          }
          progressed = true;
        } else {
          const auto before = gd::submission_admission_enabled(*state)
              ? &admit_sequence : nullptr;
          auto outcome = cursor.submit_segment(segment, false, before, &control, true);
          if (outcome.pending != nullptr || outcome.pending_input) {
            pending_host = true;
            cursor.defer_segment(segment);
          } else {
            cursor.commit_segment(segment, std::move(outcome),
                &gd::retain_submission_completion, state).IgnoreError();
            progressed = true;
          }
        }
      }
    }

    if (cursor.done() && busy == 0) {
      if (!cursor.submission_error().ok()) break;
      auto ready = gd::input_results_ready(*state);
      if (!ready) { cursor.reject_successors(ready.error()); break; }
      if (*ready) break;
      pending_host = true;
    }
    if (progressed) continue;
    if (busy == 0 && !pending_host) {
      cursor.reject_successors(Status::Internal("graph progress has no ready or pending segment"));
      continue;
    }
    std::unique_lock lock(control.mutex);
    // Generic Completion has no readiness subscription. Use the existing
    // fiber-aware timed wait for host fallbacks; job tails also wake it early.
    control.changed.wait_for(lock, std::chrono::milliseconds(1), [&] {
      return control.revision != observed;
    });
  }
  complete_driver(control);
  // DetachedDriver/Fiber are separate Scheduler allocations and do not read
  // their raw callback context after return. No Invocation access follows.
  control.decref();
}

Result<compute::Completion> submit_progress(void *opaque, Invocation &invocation,
    std::span<const QueueBinding> queues, gd::OrderedGraphSubmission ordered) noexcept {
  auto &control = *static_cast<Control *>(opaque);
  auto *state = gd::InvocationAccess::state(invocation);
  if (state != control.invocation)
    return std::unexpected(Status::InvalidArgument("graph progress belongs to another invocation"));
  if (state->segment_blocks_host.size() != state->plan->segments.size())
    return std::unexpected(Status::FailedPrecondition("graph progress requires prepared replay cost facts"));
  {
    std::lock_guard guard(control.mutex);
    if (control.running)
      return std::unexpected(Status::Unavailable("graph progress is still running"));
  }
  HYLE_ASSIGN_OR_RETURN(auto cursor, gd::begin_submission(invocation, queues, ordered));
  control.cursor = std::move(cursor);
  control.cancel_requested.store(false, std::memory_order_release);
  {
    std::lock_guard guard(control.mutex);
    control.running = true;
  }
  control.incref(); // transferred notification reference
  auto pending = state->observation.publish_pending(
      cd::RetainedCompletionControl{&control, &cancel_control, &wait_for_handoff,
                                   &notify_control, &release_control}, true);
  control.ticket = std::move(pending.ticket);
  control.incref(); // posted driver reference, also covers posting failure
  const auto posted = control.scheduler->spawn_detached(&drive, &control, mem::kFiberStackLarge);
  if (!posted.ok()) {
    control.cursor.reject_successors(posted);
    complete_driver(control);
    control.decref();
    return std::unexpected(posted);
  }
  return std::move(pending.completion);
}

Status wait_progress(void *opaque, gd::InvocationState &state) noexcept {
  auto &control = *static_cast<Control *>(opaque);
  Status terminal;
  while (state.phase.load(std::memory_order_acquire) == gd::InvocationPhase::kRunning) {
    auto observed = state.observation.poll();
    if (!observed) terminal.Update(observed.error());
    if (state.phase.load(std::memory_order_acquire) != gd::InvocationPhase::kRunning) break;
    std::unique_lock lock(control.mutex);
    const auto revision = control.revision;
    control.changed.wait_for(lock, std::chrono::milliseconds(1), [&] {
      return control.revision != revision ||
          state.phase.load(std::memory_order_acquire) != gd::InvocationPhase::kRunning;
    });
  }
  terminal.Update(gd::submission_lifetime_status(state));
  return terminal;
}

const gd::ProgressVTable kProgressVTable{
    &release_control, &submit_progress, &wait_progress, &notify_control};

} // namespace

Status prepare_progress(Invocation &invocation, sched::Scheduler &scheduler,
                        sched::BlockingPool &pool) noexcept {
  auto *state = detail::InvocationAccess::state(invocation);
  HYLE_RETURN_IF_ERROR(detail::resource_control_ready(state));
  if (!scheduler.valid() || !pool.valid())
    return Status::InvalidArgument("graph progress requires valid Scheduler and BlockingPool");
  if (state->progress) {
    if (state->progress.table() != &kProgressVTable)
      return Status::FailedPrecondition("graph invocation has another progress implementation");
    const auto &control = *static_cast<Control *>(state->progress.get());
    return control.scheduler == &scheduler && control.pool == &pool ? Status::Ok()
        : Status::FailedPrecondition("graph progress is already prepared with other runtimes");
  }
  const auto count = state->plan->lanes.size();
  if (count > SIZE_MAX / sizeof(JobSlot))
    return Status::ResourceExhausted("graph progress lane metadata overflow");
  auto *control = state->allocator.construct<Control>(*state, scheduler, pool);
  if (!control) return Status::ResourceExhausted("graph progress control metadata");
  detail::ProgressOwner candidate(control, &kProgressVTable);
  if (count != 0) {
    control->jobs = static_cast<JobSlot *>(state->allocator.allocate(
        count * sizeof(JobSlot), alignof(JobSlot)));
    if (!control->jobs) return Status::ResourceExhausted("graph progress lane metadata");
    for (std::size_t i = 0; i < count; ++i) {
      auto &slot = *::new (control->jobs + i) JobSlot();
      slot.owner = control;
      slot.job.ctx = &slot;
      slot.job.run_fn = &run_job;
      slot.job.cancel_fn = &cancel_job;
      slot.job.destroy_fn = &destroy_job;
      ++control->job_count;
    }
  }
  HYLE_RETURN_IF_ERROR(detail::prepare_submission_lifetime(*state, state->allocator));
  state->progress = std::move(candidate);
  return Status::Ok();
}

} // namespace hyle::graph
