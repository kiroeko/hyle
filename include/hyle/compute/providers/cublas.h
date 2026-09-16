#pragma once

#include "hyle/compute/provider.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_COMPUTE_PROVIDER_CUBLAS_API
#elif defined(HYLE_COMPUTE_PROVIDER_CUBLAS_BUILD)
#define HYLE_COMPUTE_PROVIDER_CUBLAS_API HYLE_EXPORT
#else
#define HYLE_COMPUTE_PROVIDER_CUBLAS_API HYLE_IMPORT
#endif

namespace hyle::compute::cublas {

inline constexpr ProviderId kProviderId{0x48594c4500000001ULL,
                                        0x4355424c41530001ULL};

// Expert composition factory for native CUDA MatMul.
// MatMul accepts plain rank-2 FP16/FP32 row/column-major inputs and a dense
// row-major output, with matching types and 16-byte-aligned storage.
// Accumulation is FP32, without TF32 input
// conversion. Descriptors/algorithm are fixed in prepare; each Invocation owns
// a separate vendor handle and borrows its caller-supplied CUDA workspace.
// Context/artifacts retain the backend; payload and workspace owners must cover
// every real Completion. Vendor-private storage has an unknown byte bound.
#if defined(HYLE_COMPUTE_CUDA)
HYLE_COMPUTE_PROVIDER_CUBLAS_API Provider provider() noexcept;
#endif

} // namespace hyle::compute::cublas
