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
inline constexpr ProviderId kSglangProviderId{0x48594c4500000001ULL,
                                              0x53474c4143540001ULL};

// Expert composition factory for HYLE's native CUDA gated activation fragment.
// Supports SiLU/GELU/GELU-tanh, dense FP16/BF16/FP32 separate inputs or [...,
// gate | value] packed halves. Arithmetic is FP32 with one final conversion to
// the output type. Exact shapes are prepared once; output element count <=
// INT32_MAX. Empty outputs require no device payload allocation or kernel
// launch. No tensor aliasing, per-invocation native storage, or caller
// workspace. Payload owners must cover the real Completion.
// Kernel/configuration binding is cold; the shared CUDA stream also permits
// native Graph capture.
#if defined(HYLE_COMPUTE_CUDA)
HYLE_COMPUTE_PROVIDER_CUDA_API Provider provider() noexcept;

// Pinned precise SGLang implementation under the same native ownership and
// completion contract and three activation modes. Packed halves only, 16-byte
// tensor alignment, FP16/BF16 hidden dimension divisible by 8 or FP32 by 4.
// FP32 intermediates and one final output conversion; no fast-math, PDL,
// filtering, or input reuse. The ordinary CUDA route selects it in this domain
// during preparation.
HYLE_COMPUTE_PROVIDER_CUDA_API Provider sglang_provider() noexcept;
#endif

} // namespace hyle::compute::cuda_native
