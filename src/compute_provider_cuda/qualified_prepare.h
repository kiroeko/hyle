#pragma once
#include "hyle/compute/providers/cuda.h"
#include "provider_internal.h"

namespace hyle::compute::cuda_native::internal {
// Same-build cold seam. Reuses the native artifact/table and raises the
// existing binding checks; no extra submit dispatch or mutable runtime state.
HYLE_COMPUTE_PROVIDER_CUDA_API ProviderPrepareOutcome
prepare_qualified(void *, const detail::ProviderPrepareRequest &,
                  mem::AllocatorRef, detail::ProviderPrepared *) noexcept;
} // namespace hyle::compute::cuda_native::internal
