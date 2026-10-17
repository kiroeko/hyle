#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>

#include "hyle/compute/resource_budget.h"
#include "hyle/compute/resource_leases.h"

namespace hyle::compute {

struct CopyResourceId final {
  std::uint32_t value = (std::numeric_limits<std::uint32_t>::max)();
  constexpr bool valid() const noexcept {
    return value != (std::numeric_limits<std::uint32_t>::max)();
  }
  constexpr bool operator==(const CopyResourceId &) const noexcept = default;
};

struct CopyPhaseId final {
  std::uint32_t value = (std::numeric_limits<std::uint32_t>::max)();
  constexpr bool valid() const noexcept {
    return value != (std::numeric_limits<std::uint32_t>::max)();
  }
  constexpr bool operator==(const CopyPhaseId &) const noexcept = default;
};

// Independent copies within one phase. Native padding is explicit: zero means
// exactly bytes, otherwise physical_bytes must be at least bytes and fit both
// owned ranges. It is charged and participates in every alias check.
struct CopyRange final {
  CopyResourceId source{}, destination{};
  CopyPhaseId phase{};
  std::size_t source_offset = 0, destination_offset = 0, bytes = 0;
  std::size_t physical_bytes = 0;
};

// A bit identifies a preceding phase, never a submission-array ordering hint.
// The caller supplies all causal edges; no model/layer interpretation exists.
struct CopyPhase final {
  std::uint64_t dependency_mask = 0;
  std::uint64_t group_tag = 0;
};

struct CopyResourceSpec final {
  mem::MemoryResourceRef resource{};
  std::size_t capacity_bytes = 0;
};

struct CopyBatchOptions final {
  std::uint32_t max_resources = 16, max_copies = 64, max_phases = 4;
  std::size_t max_bytes = 64u * 1024u * 1024u;
  mem::AllocatorRef allocator = mem::system_allocator();
  ResourceBudget resource_budget{};
};

// Cold snapshot of actual work and charged control/native costs. A replay
// allocator count is not a claim that an opaque driver allocates no memory.
struct CopyBatchReport final {
  std::uint32_t resources = 0, copies = 0, active_copies = 0, phases = 0;
  std::uint32_t no_op_copies = 0;
  std::size_t logical_bytes = 0, physical_bytes = 0, staging_bytes = 0;
  std::size_t metadata_bytes = 0;
  // Copy entry-point calls, accepted native dependency waits, actual phase
  // completion boundaries and explicit host waits are charged separately.
  std::uint64_t native_calls = 0, native_waits = 0, phase_completions = 0, host_waits = 0;
  bool native_metadata_unknown = false;
};

class Context;
class Queue;
class CopyBatchInvocation;
class CopyBatchSubmission;
namespace detail { struct CopyProgramState; struct CopyInvocationState; struct CopyStreamAccess; struct CopyUpdateAccess; }

// Immutable prepared CPU/CUDA copy contract. Resources and allocator remain
// borrowed; invocation/native retirement retain this program's cold metadata.
class CopyBatchProgram final {
public:
  CopyBatchProgram() noexcept = default;
  HYLE_COMPUTE_API CopyBatchProgram(const CopyBatchProgram &) noexcept;
  HYLE_COMPUTE_API CopyBatchProgram &operator=(const CopyBatchProgram &) noexcept;
  HYLE_COMPUTE_API CopyBatchProgram(CopyBatchProgram &&) noexcept;
  HYLE_COMPUTE_API CopyBatchProgram &operator=(CopyBatchProgram &&) noexcept;
  HYLE_COMPUTE_API ~CopyBatchProgram();
  HYLE_COMPUTE_API static Result<CopyBatchProgram> prepare(
      const Context &context, std::span<const CopyResourceSpec> resources,
      CopyBatchOptions options = {}) noexcept;
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_COMPUTE_API void reset() noexcept;
private:
  friend class CopyBatchInvocation;
  explicit CopyBatchProgram(detail::CopyProgramState *state) noexcept : state_(state) {}
  detail::CopyProgramState *state_ = nullptr;
};

// Thread-compatible, independent fixed-capacity replay slot. The original
// Queue wrapper is borrowed and must stay at its address until this slot and
// every submission/native use retire. Failed bindings do not consume leases.
class CopyBatchInvocation final {
public:
  CopyBatchInvocation() noexcept = default;
  CopyBatchInvocation(const CopyBatchInvocation &) = delete;
  CopyBatchInvocation &operator=(const CopyBatchInvocation &) = delete;
  HYLE_COMPUTE_API CopyBatchInvocation(CopyBatchInvocation &&) noexcept;
  HYLE_COMPUTE_API CopyBatchInvocation &operator=(CopyBatchInvocation &&) noexcept;
  HYLE_COMPUTE_API ~CopyBatchInvocation();
  HYLE_COMPUTE_API static Result<CopyBatchInvocation> create(const CopyBatchProgram &, Queue &) noexcept;
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_COMPUTE_API Status bind_source(CopyResourceId, const ReadLease &) noexcept;
  HYLE_COMPUTE_API Status bind_mutable(CopyResourceId, WriteLease &&) noexcept;
  HYLE_COMPUTE_API Status bind_producer(CopyResourceId, Completion &&) noexcept;
  // Return the original mutable binding after all accepted phases actually
  // succeed and retire, or withdraw it before submission. Retained submissions,
  // pending/failed/unknown work and published outputs reject without consuming.
  // Nonblocking retirement observation, no wait/publication; untouched bytes
  // remain caller-owned content. Abandonment alone does not retire a writer.
  HYLE_COMPUTE_API Result<WriteLease> unbind_mutable(CopyResourceId) noexcept;
  HYLE_COMPUTE_API Status prepare_replay(std::span<const CopyPhase>, std::span<const CopyRange>) noexcept;
  HYLE_COMPUTE_API Result<CopyBatchSubmission> submit() noexcept;
  // Reset is explicit after successful retirement or failure; a retained
  // submission or unproven retirement rejects reset without changing bindings.
  HYLE_COMPUTE_API Status reset() noexcept;
  HYLE_COMPUTE_API Status close() noexcept;
  // Permanently closes admission, then waits for this slot's submitted work.
  // To wait between replays, observe CopyBatchSubmission instead.
  HYLE_COMPUTE_API Status drain() noexcept;
  HYLE_COMPUTE_API bool in_flight() const noexcept;
  // Metadata counts the shared program and all its live cold slots once.
  HYLE_COMPUTE_API CopyBatchReport report() const noexcept;
private:
  friend struct detail::CopyStreamAccess;
  friend struct detail::CopyUpdateAccess;
  explicit CopyBatchInvocation(detail::CopyInvocationState *state) noexcept : state_(state) {}
  detail::CopyInvocationState *state_ = nullptr;
};

// Retains one invocation slot. Destruction abandons observation without a host
// wait; actual native retirement continues to own descriptors and resources.
class CopyBatchSubmission final {
public:
  CopyBatchSubmission() noexcept = default;
  CopyBatchSubmission(const CopyBatchSubmission &) = delete;
  CopyBatchSubmission &operator=(const CopyBatchSubmission &) = delete;
  HYLE_COMPUTE_API CopyBatchSubmission(CopyBatchSubmission &&) noexcept;
  HYLE_COMPUTE_API CopyBatchSubmission &operator=(CopyBatchSubmission &&) noexcept;
  HYLE_COMPUTE_API ~CopyBatchSubmission();
  bool valid() const noexcept { return state_ != nullptr; }
  // Nonblocking observation; output only covers successfully copied logical
  // bytes of the block's last-write phase. Pending/unknown never publish data.
  HYLE_COMPUTE_API Result<ReadLease> output(CopyPhaseId, CopyResourceId,
                                          std::size_t offset, std::size_t bytes) noexcept;
  HYLE_COMPUTE_API Status wait() noexcept;
  HYLE_COMPUTE_API Status request_cancel() noexcept;
  HYLE_COMPUTE_API bool in_flight() const noexcept;
  HYLE_COMPUTE_API CopyBatchReport report() const noexcept;
private:
  friend class CopyBatchInvocation;
  friend struct detail::CopyUpdateAccess;
  explicit CopyBatchSubmission(detail::CopyInvocationState *state) noexcept : state_(state) {}
  detail::CopyInvocationState *state_ = nullptr;
};

namespace detail {
// Ownership/metadata shell only. Native observation and identity stay compiled
// in the original factor. No completion, execution or allocation lives here.
template<class Invocation, class Submission, class Access>
class CopyUpdateOwner final {
public:
  CopyUpdateOwner() noexcept = default;
  CopyUpdateOwner(const CopyUpdateOwner &) = delete;
  CopyUpdateOwner &operator=(const CopyUpdateOwner &) = delete;
  CopyUpdateOwner(CopyUpdateOwner &&) noexcept = default;
  CopyUpdateOwner &operator=(CopyUpdateOwner &&) noexcept = default;
  ~CopyUpdateOwner() = default;
  // Qualification fails without consuming either original control. Success
  // moves both: the caller must not retain another observer of their episode.
  static Result<CopyUpdateOwner> take(Invocation &&invocation,
                                     Submission &&submission) noexcept {
    HYLE_RETURN_IF_ERROR(Access::qualify(invocation, submission));
    return CopyUpdateOwner(std::move(invocation), std::move(submission));
  }
  bool valid() const noexcept { return invocation_.valid(); }
  bool in_flight() const noexcept { return invocation_.in_flight(); }
  CopyBatchReport report() const noexcept { return invocation_.report(); }
  // Nonblocking original status, never reinterpret an unavailable/error code
  // as success or infer retirement from an observed error.
  Status observe() noexcept { return Access::observe(invocation_); }
  Status wait() noexcept {
    return submission_.valid() ? submission_.wait() : observe();
  }
  Status request_cancel() noexcept {
    return submission_.valid() ? submission_.request_cancel() : invocation_.close();
  }
  // Return caller-owned mutable content after real success. Untouched bytes
  // are still that caller's responsibility; this does not expand F2 output.
  Result<WriteLease> take_mutable(CopyResourceId resource) noexcept {
    HYLE_RETURN_IF_ERROR(observe());
    submission_ = {};
    return invocation_.unbind_mutable(resource);
  }
  // Control recovery is not data publication. A failed/closed control retains
  // those states. Borrowed native prefixes must retire at their receiver first.
  Result<Invocation> release_invocation() noexcept {
    if (!valid()) return std::unexpected(Status::FailedPrecondition("copy update is empty"));
    observe().IgnoreError();
    if (in_flight()) return std::unexpected(Status::Unavailable("copy update native work still retires"));
    submission_ = {};
    return std::move(invocation_);
  }
private:
  friend Access;
  CopyUpdateOwner(Invocation invocation, Submission submission) noexcept
      : invocation_(std::move(invocation)), submission_(std::move(submission)) {}
  Invocation invocation_;
  Submission submission_;
};
} // namespace detail

namespace detail {
struct CopyUpdateAccess final {
  static HYLE_COMPUTE_API Status qualify(const CopyBatchInvocation &,
                              const CopyBatchSubmission &) noexcept;
  static HYLE_COMPUTE_API Status observe(CopyBatchInvocation &) noexcept;
  // Trusted receiver/copy seams only; no ordinary public pending byte view.
  template<class Update> static auto &invocation(Update &owner) noexcept { return owner.invocation_; }
  template<class Update> static auto &submission(Update &owner) noexcept { return owner.submission_; }
};
} // namespace detail
// Thread-compatible single observation owner for an accepted original copy
// episode. Borrowed Context/Queue/resources/allocator outlive all late work.
using CopyBatchUpdate = compute::detail::CopyUpdateOwner<
    CopyBatchInvocation, CopyBatchSubmission, detail::CopyUpdateAccess>;

} // namespace hyle::compute
