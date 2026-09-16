#pragma once

#include "hyle/compute/provider.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_COMPUTE_PROVIDER_CUDA_API
#elif defined(HYLE_COMPUTE_PROVIDER_CUDA_BUILD)
#define HYLE_COMPUTE_PROVIDER_CUDA_API HYLE_EXPORT
#else
#define HYLE_COMPUTE_PROVIDER_CUDA_API HYLE_IMPORT
#endif

namespace hyle::compute::cuda_native {

inline constexpr ProviderId kProviderId{0x48594c4500000001ULL,
                                        0x435544414b455201ULL};

// Expert composition factory for HYLE's native CUDA SwiGLU fragment.
// Supports dense FP16/FP32 separate inputs or [..., gate | value] packed
// halves. Arithmetic is FP32 with one final conversion to the output type.
// Exact nonempty shapes are prepared once; output element count <= INT32_MAX.
// No tensor aliasing, per-invocation native storage, or caller workspace.
// Payload owners must cover the real Completion. Kernel/configuration binding
// is cold; the shared CUDA stream also permits native Graph capture.
#if defined(HYLE_COMPUTE_CUDA)
HYLE_COMPUTE_PROVIDER_CUDA_API Provider provider() noexcept;
#endif

} // namespace hyle::compute::cuda_native
