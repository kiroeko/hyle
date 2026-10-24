#pragma once

#include <cstdint>
#include <span>
#include <utility>

#include "hyle/compute/completion.h"
#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/detail/operation_access.h"
#include "hyle/compute/detail/queue_ref.h"
#include "hyle/compute/device.h"
#include "hyle/compute/tensor.h"

namespace hyle::compute {

class OperationInvocation;
struct OperationId;

namespace detail {

enum class QueueDependencyRoute : std::uint8_t {
  // The source Queue completes its work before finish_sequence() returns.
  kNoWait = 0,
  // The destination backend can consume the source Queue's prefix completion
  // without blocking the submitting host thread.
  kDeviceWait,
  // No compatible device route exists; a caller may explicitly wait on host.
  kHostWait,
};

// Provider/backend command recording result. `accepted` describes whether
// this command placed any work into the ordered Queue. A non-OK status with an
// accepted command is a terminal hint for the already-recorded prefix; a
// non-OK rejected command remains a synchronous Result error when the prefix
// is empty.
struct EnqueueResult final {
  bool accepted = false;
  Status status{};

  static EnqueueResult
  accepted_command(Status terminal = Status::Ok()) noexcept {
    return EnqueueResult{true, std::move(terminal)};
  }
  static EnqueueResult rejected(Status error) noexcept {
    return EnqueueResult{false, std::move(error)};
  }
  static EnqueueResult no_op() noexcept { return {}; }
};

// Internal factor bridge for one uninterrupted ordered-Queue submission
// sequence.  It holds Context/Queue admission and excludes a second producer
// only while commands are being submitted; asynchronous execution continues
// after this guard is destroyed. Each shared Graph segment uses one guard so
// a close or another producer cannot split/interleave that admitted sequence.
class QueueSequence final {
public:
  QueueSequence() noexcept = default;
  QueueSequence(const QueueSequence &) = delete;
  QueueSequence &operator=(const QueueSequence &) = delete;
  HYLE_COMPUTE_API QueueSequence(QueueSequence &&other) noexcept;
  HYLE_COMPUTE_API QueueSequence &operator=(QueueSequence &&other) noexcept;
  HYLE_COMPUTE_API ~QueueSequence();

  HYLE_COMPUTE_API static Result<QueueSequence> begin(Queue &queue) noexcept;
  HYLE_COMPUTE_API static Result<QueueSequence> begin(const QueueRef &queue) noexcept;
  // Cold route query for the prefix Completion produced by source. It does
  // not begin a sequence or create an event and is stable for these Queue
  // objects. Same-Queue ordering is normally handled by Graph before calling.
  HYLE_COMPUTE_API static Result<QueueDependencyRoute>
  dependency_route(const Queue &destination, const Queue &source) noexcept;
  HYLE_COMPUTE_API static Result<QueueDependencyRoute>
  dependency_route(const QueueRef &destination, const QueueRef &source) noexcept;
  // Cold endpoint bridge shared by Queue and delegated-provider islands.
  // An invalid endpoint is valid only for a synchronous Queue whose prefix is
  // complete before finish_sequence() returns.
  HYLE_COMPUTE_API static Result<CompletionEndpoint>
  completion_endpoint(const Queue &queue) noexcept;
  HYLE_COMPUTE_API static Result<CompletionEndpoint>
  completion_endpoint(const QueueRef &queue) noexcept;
  HYLE_COMPUTE_API static Result<bool>
  can_import_endpoint(const Queue &destination,
                      CompletionEndpoint source) noexcept;
  HYLE_COMPUTE_API static Result<bool>
  can_import_endpoint(const QueueRef &destination,
                      CompletionEndpoint source) noexcept;

  // Accumulate one provider/backend enqueue result. Once an error is recorded,
  // callers must stop recording and finish the sequence.
  HYLE_COMPUTE_API void record(EnqueueResult result) noexcept;
  // Enqueue a device-side wait for a still-live source prefix. Immediate source
  // completions become a no-op (or propagate their terminal error). A backend
  // that cannot consume the exact source completion returns Unimplemented;
  // callers may then choose an explicit, reported host barrier.
  HYLE_COMPUTE_API EnqueueResult
  enqueue_wait(const Completion &source) noexcept;
  // F2 owner-resolved dependency, with the original producer held through use.
  // Known error is rejected before native acceptance; no implicit host wait.
  HYLE_COMPUTE_API EnqueueResult enqueue_resolved_wait(CompletionDependency source) noexcept;
  // Finalize exactly once. An empty rejected prefix is returned as a Result
  // error after the backend releases its reserved terminal slot; any accepted
  // prefix is represented by a valid Completion carrying the first tail error.
  HYLE_COMPUTE_API Result<Completion> finish() noexcept;

  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  bool accepted() const noexcept { return accepted_; }
  // Actual recording state of this admitted sequence, never inferred from a
  // selected provider. CommandCaptureAccess owns its begin/end/abort changes.
  bool recording() const noexcept { return recording_; }
  const Status &terminal() const noexcept { return terminal_; }

private:
  friend struct ContextAccess;
  friend struct CommandCaptureAccess;
  QueueSequence(QueueState *state, void *backend_sequence) noexcept
      : state_(state), backend_sequence_(backend_sequence) {}
  static Result<QueueSequence> begin(QueueState *) noexcept;
  static Result<QueueDependencyRoute>
  dependency_route(QueueState *destination, QueueState *source) noexcept;
  static Result<CompletionEndpoint> completion_endpoint(QueueState *) noexcept;
  static Result<bool> can_import_endpoint(QueueState *, CompletionEndpoint) noexcept;
  void reset() noexcept;

  QueueState *state_ = nullptr;
  void *backend_sequence_ = nullptr;
  Status terminal_{};
  bool accepted_ = false;
  bool recording_ = false;
};

// Validation and provider dispatch for a command inside an already-admitted
// QueueSequence.  A successful Completion represents the entire ordered Queue
// prefix through this command, not merely the command in isolation.  This
// prefix rule lets Graph discard intermediate handles without losing earlier
// terminal errors or inserting host waits.
struct OperationInvocationAccess final {
  // Cold only, while the original slot is idle and unretained. Revalidates
  // descriptions/resources/aliases before its optional preparation callback.
  // No Queue admission, payload execution, or implicit submit is performed.
  HYLE_COMPUTE_API static Status prepare_bindings(
      OperationInvocation &, std::span<const ConstTensorView> inputs,
      std::span<const TensorView> outputs) noexcept;
  // Same-build typed adapter only. False means this selected implementation
  // needs no layout update. Native state never leaves object-local dispatch.
  HYLE_COMPUTE_API static Result<bool> configure_cold(
      OperationInvocation &, ProviderExtensionId,
      const void *configuration) noexcept;
  HYLE_COMPUTE_API static EnqueueResult
  enqueue(OperationInvocation &invocation, QueueSequence &sequence,
          std::span<const ConstTensorView> inputs,
          std::span<const TensorView> outputs) noexcept;
  // Graph-only hot route. The owning plan has already checked descriptions,
  // resources and aliases for these stable spans during prepare_replay(). It
  // retains only O(1) state/context invariants before the provider callback.
  HYLE_COMPUTE_API static EnqueueResult
  enqueue_prevalidated(OperationInvocation &invocation, QueueSequence &sequence,
                       std::span<const ConstTensorView> inputs,
                       std::span<const TensorView> outputs) noexcept;
  // Delegated Graph route. Endpoint compatibility and stable bindings are
  // qualified during replay preparation; the provider owns execution and
  // returns its one real Completion rather than recording into a HYLE Queue.
  HYLE_COMPUTE_API static CompletionEndpoint
  delegated_endpoint(const OperationInvocation &invocation) noexcept;
  HYLE_COMPUTE_API static bool
  can_import_endpoint(const OperationInvocation &invocation,
                      CompletionEndpoint source) noexcept;
  HYLE_COMPUTE_API static Result<Completion> submit_delegated_prevalidated(
      OperationInvocation &invocation,
      std::span<const CompletionDependency> dependencies,
      std::span<const ConstTensorView> inputs,
      std::span<const TensorView> outputs) noexcept;
  // Graph-only borrowed route. The caller serializes this invocation and
  // retains it, every binding, and every dependency until the returned
  // Completion reaches terminal state. Unlike the standalone route above it
  // deliberately does not consume Completion's single lifetime attachment;
  // Graph uses that attachment to retain its whole Invocation and workspace.
  HYLE_COMPUTE_API static Result<Completion>
  submit_delegated_borrowed_prevalidated(
      OperationInvocation &invocation,
      std::span<const CompletionDependency> dependencies,
      std::span<const ConstTensorView> inputs,
      std::span<const TensorView> outputs) noexcept;
};

} // namespace detail
} // namespace hyle::compute
