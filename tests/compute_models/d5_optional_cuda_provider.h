#pragma once

#include "hyle/compute/provider.h"
#if defined(HYLE_COMPUTE_FLASHINFER)
#include "hyle/compute/providers/flashinfer.h"
#endif

namespace d5_optional_cuda_provider {

// Installed Windows SDKs intentionally omit the Linux-only Provider header.
// A consumer must not obtain it accidentally from the producer source tree.
inline bool is_flashinfer(hyle::compute::ProviderId provider) noexcept {
#if defined(HYLE_COMPUTE_FLASHINFER)
  return provider == hyle::compute::flashinfer::kProviderId;
#else
  (void)provider;
  return false;
#endif
}

} // namespace d5_optional_cuda_provider
