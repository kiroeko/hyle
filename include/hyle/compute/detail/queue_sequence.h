#pragma once

#include <span>

#include "hyle/compute/completion.h"
#include "hyle/compute/device.h"
#include "hyle/compute/tensor.h"

namespace hyle::compute {

class OperationInvocation;

namespace detail {

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

  bool valid() const noexcept { return queue_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  Queue &queue() const noexcept { return *queue_; }

private:
  explicit QueueSequence(Queue *queue) noexcept : queue_(queue) {}
  void reset() noexcept;

  Queue *queue_ = nullptr;
};

// Validation and provider dispatch for a command inside an already-admitted
// QueueSequence.  A successful Completion represents the entire ordered Queue
// prefix through this command, not merely the command in isolation.  This
// prefix rule lets Graph discard intermediate handles without losing earlier
// terminal errors or inserting host waits.
struct OperationInvocationAccess final {
  HYLE_COMPUTE_API static Result<Completion>
  submit(OperationInvocation &invocation, QueueSequence &sequence,
         std::span<const ConstTensorView> inputs,
         std::span<const TensorView> outputs) noexcept;
};

} // namespace detail
} // namespace hyle::compute
