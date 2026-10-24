#pragma once

#include "hyle/compute/device.h"

namespace hyle::compute::detail {
class QueueSequence;

// Internal owner of one actual ordered Queue state. Copies preserve identity
// across public-handle moves; releasing this reference never closes admission.
// It cannot create a Queue or authorize work after the original Queue closes.
class QueueRef final {
public:
  QueueRef() noexcept = default;
  HYLE_COMPUTE_API QueueRef(const QueueRef &) noexcept;
  HYLE_COMPUTE_API QueueRef &operator=(const QueueRef &) noexcept;
  HYLE_COMPUTE_API QueueRef(QueueRef &&) noexcept;
  HYLE_COMPUTE_API QueueRef &operator=(QueueRef &&) noexcept;
  HYLE_COMPUTE_API ~QueueRef();

  HYLE_COMPUTE_API static Result<QueueRef> acquire(const Queue &) noexcept;
  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  const void *identity_key() const noexcept { return state_; }
  HYLE_COMPUTE_API bool is_open() const noexcept;
  HYLE_COMPUTE_API Device device() const noexcept;
  HYLE_COMPUTE_API bool same_owner(const QueueRef &) const noexcept;
  HYLE_COMPUTE_API bool same_owner(const Queue &) const noexcept;
  HYLE_COMPUTE_API bool same_owner(const QueueSequence &) const noexcept;
  HYLE_COMPUTE_API Status reserve_completions(std::size_t minimum) const noexcept;
  HYLE_COMPUTE_API void reset() noexcept;

private:
  friend struct ContextAccess;
  explicit QueueRef(QueueState *state) noexcept : state_(state) {}
  QueueState *state_ = nullptr;
};

} // namespace hyle::compute::detail
