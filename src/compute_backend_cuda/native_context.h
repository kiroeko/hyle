#pragma once

#include <cuda.h>

#include "hyle/compute/backends/cuda.h"

namespace hyle::compute::cuda::detail {

// Same-build Provider bridge. The caller must first establish that the
// ProviderVTable backend_state belongs to this CUDA backend. Compute keeps
// that state alive through every provider artifact and invocation.
HYLE_COMPUTE_BACKEND_CUDA_API CUcontext
borrow_primary_context(void *backend_state) noexcept;

// Preserve the caller's current driver context across native SDK work. This
// is the backend's existing scope, shared by native providers without giving
// them a second context owner or access to the backend's mutable state.
class ScopedPrimaryContext final {
public:
  HYLE_COMPUTE_BACKEND_CUDA_API
  explicit ScopedPrimaryContext(CUcontext primary) noexcept;
  ScopedPrimaryContext(const ScopedPrimaryContext &) = delete;
  ScopedPrimaryContext &operator=(const ScopedPrimaryContext &) = delete;
  HYLE_COMPUTE_BACKEND_CUDA_API ~ScopedPrimaryContext();

  const Status &status() const noexcept { return status_; }
  explicit operator bool() const noexcept { return status_.ok(); }

private:
  CUcontext expected_ = nullptr;
  bool pushed_ = false;
  Status status_{};
};

} // namespace hyle::compute::cuda::detail
