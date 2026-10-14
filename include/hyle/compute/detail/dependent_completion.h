#pragma once

#include "hyle/compute/completion.h"
#include "hyle/compute/detail/completion_access.h"
#include "hyle/mem/allocator.h"

namespace hyle::compute::detail {

// One trusted source owner, with externally serialized observations. Its
// preallocated subscription consumes the lifetime and delivers terminal then
// release only after all accepted source work actually retires. Neither
// subscribe_retirement nor abandon may wait or allocate. Abandon transfers
// progress to the original native owners, including cached observation errors.
// release consumes the single source-owner reference transferred to publish;
// it follows both actual retirement callbacks and public proxy destruction.
struct OutcomeSources final {
  void *state = nullptr;
  Result<CompletionPoll> (*poll)(void *) noexcept = nullptr;
  Status (*wait)(void *) noexcept = nullptr;
  Status (*request_cancel)(void *) noexcept = nullptr;
  void (*abandon)(void *) noexcept = nullptr;
  void (*subscribe_retirement)(void *, CompletionLifetime) noexcept = nullptr;
  void (*release)(void *) noexcept = nullptr;

  constexpr bool valid() const noexcept {
    return state && poll && wait && request_cancel && abandon &&
           subscribe_retirement && release;
  }
};

struct DependentCompletionState;

// Opt-in, reusable cold storage. This owns the original suffix Completion and
// one source-owner reference, without another executor or result slot. The
// source is first observed after actual suffix retirement, preserving borrowed
// native dependencies. Errors alone never release attached execution owners.
class DependentCompletion final {
public:
  DependentCompletion() noexcept = default;
  DependentCompletion(const DependentCompletion &) = delete;
  DependentCompletion &operator=(const DependentCompletion &) = delete;
  HYLE_COMPUTE_API DependentCompletion(DependentCompletion &&) noexcept;
  HYLE_COMPUTE_API DependentCompletion &operator=(DependentCompletion &&) noexcept;
  HYLE_COMPUTE_API ~DependentCompletion();
  HYLE_COMPUTE_API static Result<DependentCompletion>
  create(mem::AllocatorRef allocator) noexcept;
  HYLE_COMPUTE_API bool available() const noexcept;
  // Check available and validate sources before accepting work. No allocation
  // occurs here. Consumes the original suffix and one source-owner reference.
  HYLE_COMPUTE_API Completion publish(Completion &&suffix,
                                      OutcomeSources sources) noexcept;
  HYLE_COMPUTE_API void reset() noexcept;

private:
  explicit DependentCompletion(DependentCompletionState *state) noexcept
      : state_(state) {}
  DependentCompletionState *state_ = nullptr;
};

} // namespace hyle::compute::detail
