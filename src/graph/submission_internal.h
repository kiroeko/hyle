#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "hyle/graph/executable.h"
#include "hyle/compute/resource_leases.h"
#include "hyle/compute/detail/queue_sequence.h"
#include "hyle/compute/detail/queue_ref.h"
#include "hyle/mem/std_allocator.h"

namespace hyle::compute::detail { class AppendLease; }

namespace hyle::graph::detail {

struct InvocationState;
struct PlanState;
struct PlanSegment;

// Cold replay preparation and strict submission share these exact lane and
// command operations; capture records the same commands used by replay.
bool shared_lane(const PlanState &, ExecutionLaneId) noexcept;
Result<std::size_t> delegated_step_index(const PlanState &,
                                       const PlanSegment &) noexcept;
Status enqueue_segment_steps(InvocationState &, const PlanSegment &,
                             compute::detail::QueueSequence &) noexcept;
// Cold bindings/capture share the same nonconcurrent artifact gate as submit.
// A function sentinel prevents a completed range from releasing admission while
// later ranges still prepare; unknown native retirement keeps its own count.
Status begin_preparation(InvocationState &) noexcept;
void release_preparation(InvocationState &) noexcept;

// Uses the original Invocation's idle retention node for a cold capture prefix.
// Actual retirement releases the same slots/workspace without publishing results.
Status retain_preparation_completion(InvocationState &, compute::Completion &) noexcept;

// Exact-page owners reserve once and borrow their current bindings only during
// admission. The original F1 checks and retirement still own every native use.
class LeasePermissionSet final {
public:
  explicit LeasePermissionSet(mem::AllocatorRef allocator) noexcept
      : entries_(mem::StdAllocator<Entry>(allocator)) {}
  Status reserve(std::size_t capacity) noexcept;
  bool enabled() const noexcept { return entries_.capacity() != 0; }
  void reset() noexcept { entries_.clear(); }
  Status add(const compute::ReadLease &) noexcept;
  Status add(const compute::WriteLease &) noexcept;
  Status add(const compute::detail::AppendLease &) noexcept;
  Status admit(compute::LeaseUseBatch &) noexcept;

private:
  struct Entry final {
    const compute::ReadLease *read = nullptr;
    const compute::WriteLease *write = nullptr;
    const compute::detail::AppendLease *append = nullptr;
  };
  std::vector<Entry, mem::StdAllocator<Entry>> entries_;
};

using AcceptedCompletionFn = void (*)(void *owner, std::size_t segment,
                                      compute::Completion &completion) noexcept;
// Opt-in composition admission inside the original ordered sequence. The
// callback records its real prefix before admitting externally ordered reads.
// It must retain every borrowed source through the final mapped completion.
using SharedSequenceFn = Status (*)(void *, ExecutionLaneId,
                                    compute::detail::QueueSequence &) noexcept;
// One uniquely claimed segment's local result. Keep an accepted prefix even
// when a later command rejects. Publication and final aggregation belong to
// the original run owner, never a concurrently executing provider callback.
struct SegmentSubmission final {
  compute::Completion completion;
  Status status;
  const compute::Completion *pending = nullptr;
  // A retained external input has no producer segment in this Plan. The
  // coordinator checks its own readiness rather than borrowing a raw cell.
  bool pending_input = false;
  std::size_t pending_segment = SIZE_MAX;
};
#if defined(HYLE_TESTING)
// Native observation decoration only, after opt-in retention admission.
HYLE_GRAPH_API AcceptedCompletionFn set_accepted_observation_hook(AcceptedCompletionFn hook) noexcept;
#endif

// Same-build coarse submission cursor. One admitted run retains its original
// Invocation and fixed segment storage. A native composition owner drives it;
// no scheduler or new per-kernel interface enters ordinary Graph objects.
class SubmissionCursor final {
public:
  SubmissionCursor() noexcept = default;
  SubmissionCursor(const SubmissionCursor &) = delete;
  SubmissionCursor &operator=(const SubmissionCursor &) = delete;
  HYLE_GRAPH_API SubmissionCursor(SubmissionCursor &&other) noexcept;
  HYLE_GRAPH_API SubmissionCursor &operator=(SubmissionCursor &&other) noexcept;
  HYLE_GRAPH_API ~SubmissionCursor();

  HYLE_GRAPH_API static Result<SubmissionCursor>
  begin(Invocation &invocation, std::span<const QueueBinding> queues,
        bool retain_invocation = true) noexcept;
  // Advance one ready segment, passing pending host dependencies to visit
  // independent lanes first. false means this pass could only find pending
  // dependencies. Waiting is explicit; native math keeps its declared cost.
  HYLE_GRAPH_API Result<bool> advance(bool wait_dependencies = false,
                                     AcceptedCompletionFn accepted = nullptr,
                                     void *owner = nullptr,
                                     SharedSequenceFn admission = nullptr,
                                     void *admission_owner = nullptr) noexcept;
  // Same-build joint composition only. Unique segment/step state and actual
  // Queue single-producer ordering must be established before execution.
  // Other segments' published Completion cells stay stable until all workers
  // return. No bind/prepare/reset/budget growth or final finish runs alongside.
  HYLE_GRAPH_API SegmentSubmission submit_segment(std::size_t segment,
      bool wait_dependencies = false, SharedSequenceFn admission = nullptr,
      void *admission_owner = nullptr, bool dependencies_prechecked = false) noexcept;
  // The unique progress owner settles HostWait cells before handing a claimed
  // segment to a worker. Imports and observations share the slot's short lock.
  HYLE_GRAPH_API Result<bool> dependencies_ready(std::size_t segment) noexcept;
  // Sole coordinator calls these after receiving a fully published outcome.
  // Neither function encloses provider math or a native wait in a total lock.
  HYLE_GRAPH_API Status commit_segment(std::size_t segment,
      SegmentSubmission outcome, AcceptedCompletionFn accepted = nullptr,
      void *owner = nullptr) noexcept;
  // The unique progress owner claims from the original plan's ready index.
  // A default target accepts any target. SIZE_MAX means none is available.
  // One actual Queue producer group stays claimed through commit or defer.
  HYLE_GRAPH_API std::size_t claim_ready(ExecutionTargetId target = {}) noexcept;
  HYLE_GRAPH_API void defer_segment(std::size_t segment) noexcept;
  HYLE_GRAPH_API bool has_parallel_targets() const noexcept;
  HYLE_GRAPH_API void finish_submissions(Status terminal) noexcept;
  HYLE_GRAPH_API bool done() const noexcept;
  const compute::Completion *pending_dependency() const noexcept {
    return pending_;
  }
  HYLE_GRAPH_API const compute::detail::QueueRef *pending_queue() const noexcept;
  HYLE_GRAPH_API void request_cancel() noexcept;
  // Stop unaccepted successors while preserving the actual failing endpoint.
  HYLE_GRAPH_API void reject_successors(Status terminal) noexcept;
  // Emits the actual final accepted prefix, including tail failure. This
  // retains all native owners even when its public handle is abandoned.
  // Opt-in owners can defer terminal publication until their preattached
  // native-use guards prove retirement, including already-observed errors.
  // Such an owner MUST retain Invocation and publish its actual terminal.
  HYLE_GRAPH_API Result<compute::Completion>
  finish(bool defer_retirement = false) noexcept;
  const Status &submission_error() const noexcept { return tail_error_; }

private:
  SubmissionCursor(InvocationState *state, bool retained) noexcept;
  void reset() noexcept;
  InvocationState *state_ = nullptr;
  std::size_t completed_segments_ = 0;
  std::size_t claimed_segments_ = 0;
  std::size_t next_lane_ = 0;
  std::uint64_t accepted_order_ = 0;
  Status tail_error_{};
  const compute::Completion *pending_ = nullptr;
  std::size_t pending_segment_ = SIZE_MAX;
  bool finished_ = false;
  bool retained_ = false;
};

} // namespace hyle::graph::detail
