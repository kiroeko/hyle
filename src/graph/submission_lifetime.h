#pragma once

#include <cstddef>
#include <span>

#include "hyle/error/status.h"
#include "hyle/error/result.h"
#include "hyle/graph/visibility.h"
#include "hyle/mem/allocator.h"

namespace hyle::compute {
class Completion;
namespace detail { class QueueSequence; }
} // namespace hyle::compute

namespace hyle::graph {
class ExecutionLaneId;
class Invocation;
struct QueueBinding;
namespace detail {
struct InvocationState;
struct OrderedGraphSubmission;
struct SubmissionLifetime;
class SubmissionCursor;

// Cold, optional retirement storage in the original Invocation. Idle storage
// does not retain its owner; one admitted submission retains it until its
// admission sentinel and every accepted native guard have been released.
class SubmissionLifetimeOwner final {
public:
  SubmissionLifetimeOwner() noexcept = default;
  explicit SubmissionLifetimeOwner(SubmissionLifetime *state) noexcept : state_(state) {}
  SubmissionLifetimeOwner(const SubmissionLifetimeOwner &) = delete;
  SubmissionLifetimeOwner &operator=(const SubmissionLifetimeOwner &) = delete;
  HYLE_GRAPH_API SubmissionLifetimeOwner(SubmissionLifetimeOwner &&) noexcept;
  HYLE_GRAPH_API SubmissionLifetimeOwner &operator=(SubmissionLifetimeOwner &&) noexcept;
  HYLE_GRAPH_API ~SubmissionLifetimeOwner();
  SubmissionLifetime *get() const noexcept { return state_; }
  explicit operator bool() const noexcept { return state_ != nullptr; }
  HYLE_GRAPH_API void reset() noexcept;

private:
  SubmissionLifetime *state_ = nullptr;
};

// Preparation is cold and does not prepare or require owned resource bindings.
HYLE_GRAPH_API Status prepare_submission_lifetime(
    InvocationState &, mem::AllocatorRef) noexcept;
// Called after the original cursor and optional lease batch are admitted. This
// only initializes preallocated storage and retains the original Invocation.
HYLE_GRAPH_API Status begin_submission_lifetime(
    InvocationState &, OrderedGraphSubmission) noexcept;
HYLE_GRAPH_API void retain_submission_completion(
    void *, std::size_t segment, compute::Completion &) noexcept;
// Admission and outcome publication are serialized by the original cursor
// owner. Stable storage does not make shared lease metadata thread-safe.
HYLE_GRAPH_API Status admit_submission_sequence(
    void *, ExecutionLaneId, compute::detail::QueueSequence &) noexcept;
HYLE_GRAPH_API void record_submission_error(
    InvocationState &, const Status &) noexcept;
// The unique submission owner returns its admission sentinel once. Any later
// slot access needs an independent owner; accepted native guards remain active.
HYLE_GRAPH_API void finish_submission_lifetime(
    InvocationState &, const Status &) noexcept;
HYLE_GRAPH_API Status submission_lifetime_status(
    const InvocationState &) noexcept;
// Shared admission/finalization for synchronous and optional asynchronous
// progress. The caller ends cursor/job access before returning the sentinel.
// Any later handoff holds its own short reference, so public success cannot
// precede the original slot's result publication.
HYLE_GRAPH_API Result<SubmissionCursor> begin_submission(
    Invocation &, std::span<const QueueBinding>, OrderedGraphSubmission) noexcept;
HYLE_GRAPH_API Result<compute::Completion> finish_submission(
    InvocationState &, SubmissionCursor &) noexcept;
HYLE_GRAPH_API bool submission_admission_enabled(const InvocationState &) noexcept;
} // namespace detail
} // namespace hyle::graph
