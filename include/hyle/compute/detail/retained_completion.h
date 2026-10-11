#pragma once

#include "hyle/compute/completion.h"
#include "hyle/mem/allocator.h"

namespace hyle::compute::detail {
struct RetainedCompletionState;

// One cold, thread-compatible observation slot for opt-in lease adapters.
// Control and public Completion observations are externally serialized. This
// owns the original Completion; it neither executes work nor owns a Queue.
class RetainedCompletion final {
public:
  RetainedCompletion() noexcept = default;
  RetainedCompletion(const RetainedCompletion &) = delete;
  RetainedCompletion &operator=(const RetainedCompletion &) = delete;
  HYLE_COMPUTE_API RetainedCompletion(RetainedCompletion &&) noexcept;
  HYLE_COMPUTE_API RetainedCompletion &operator=(RetainedCompletion &&) noexcept;
  HYLE_COMPUTE_API ~RetainedCompletion();
  HYLE_COMPUTE_API static Result<RetainedCompletion> create(mem::AllocatorRef allocator) noexcept;
  HYLE_COMPUTE_API bool available() const noexcept;
  // The caller checks available before accepting native work. A public proxy
  // retains this slot until observation/abandonment; no hot allocation occurs.
  HYLE_COMPUTE_API Completion publish(Completion completion) noexcept;
  HYLE_COMPUTE_API Status wait() noexcept;
  HYLE_COMPUTE_API void reset() noexcept;
private:
  explicit RetainedCompletion(RetainedCompletionState *state) noexcept : state_(state) {}
  RetainedCompletionState *state_ = nullptr;
};
} // namespace hyle::compute::detail
