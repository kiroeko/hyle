#pragma once

#include <span>
#include <utility>

#include "hyle/compute/completion.h"
#include "hyle/compute/device.h"
#include "hyle/compute/tensor.h"

namespace hyle::compute {

class OperationInvocation;

namespace detail {

// Provider/backend command recording result. `accepted` describes whether
// this command placed any work into the ordered Queue. A non-OK status with an
// accepted command is a terminal hint for the already-recorded prefix; a
// non-OK rejected command remains a synchronous Result error when the prefix
// is empty.
struct EnqueueResult final {
  bool accepted = false;
  Status status{};

  static EnqueueResult accepted_command(Status terminal = Status::Ok()) noexcept {
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
// after this guard is destroyed.  Graph uses one guard for its whole plan so a
// close or another producer cannot split/interleave the sequence.
class QueueSequence final {
public:
  QueueSequence() noexcept = default;
  QueueSequence(const QueueSequence &) = delete;
  QueueSequence &operator=(const QueueSequence &) = delete;
  HYLE_COMPUTE_API QueueSequence(QueueSequence &&other) noexcept;
  HYLE_COMPUTE_API QueueSequence &operator=(QueueSequence &&other) noexcept;
  HYLE_COMPUTE_API ~QueueSequence();

  HYLE_COMPUTE_API static Result<QueueSequence> begin(Queue &queue) noexcept;

  // Accumulate one provider/backend enqueue result. Once an error is recorded,
  // callers must stop recording and finish the sequence.
  HYLE_COMPUTE_API void record(EnqueueResult result) noexcept;
  // Enqueue a device-side wait for a still-live source prefix. Immediate source
  // completions become a no-op (or propagate their terminal error). A backend
  // that cannot consume the exact source completion returns Unimplemented;
  // callers may then choose an explicit, reported host barrier.
  HYLE_COMPUTE_API EnqueueResult enqueue_wait(
      const Completion &source) noexcept;
  // Finalize exactly once. An empty rejected prefix is returned as a Result
  // error after the backend releases its reserved terminal slot; any accepted
  // prefix is represented by a valid Completion carrying the first tail error.
  HYLE_COMPUTE_API Result<Completion> finish() noexcept;

  bool valid() const noexcept { return queue_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  Queue &queue() const noexcept { return *queue_; }
  bool accepted() const noexcept { return accepted_; }
  const Status &terminal() const noexcept { return terminal_; }

private:
  QueueSequence(Queue *queue, void *backend_sequence) noexcept
      : queue_(queue), backend_sequence_(backend_sequence) {}
  void reset() noexcept;

  Queue *queue_ = nullptr;
  void *backend_sequence_ = nullptr;
  Status terminal_{};
  bool accepted_ = false;
};

// Validation and provider dispatch for a command inside an already-admitted
// QueueSequence.  A successful Completion represents the entire ordered Queue
// prefix through this command, not merely the command in isolation.  This
// prefix rule lets Graph discard intermediate handles without losing earlier
// terminal errors or inserting host waits.
struct OperationInvocationAccess final {
  HYLE_COMPUTE_API static EnqueueResult
  enqueue(OperationInvocation &invocation, QueueSequence &sequence,
          std::span<const ConstTensorView> inputs,
          std::span<const TensorView> outputs) noexcept;
};

} // namespace detail
} // namespace hyle::compute
