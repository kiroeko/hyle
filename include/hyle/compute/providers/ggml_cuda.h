#pragma once

#include "hyle/compute/provider.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_COMPUTE_PROVIDER_GGML_CUDA_API
#elif defined(HYLE_COMPUTE_PROVIDER_GGML_CUDA_BUILD)
#define HYLE_COMPUTE_PROVIDER_GGML_CUDA_API HYLE_EXPORT
#else
#define HYLE_COMPUTE_PROVIDER_GGML_CUDA_API HYLE_IMPORT
#endif

namespace hyle::compute::ggml_cuda {

inline constexpr ProviderId kProviderId{0x48594c4500000001ULL,
                                        0x4747435544410001ULL};

// Original ggml CUDA programs on qualified SM89 devices. Each invocation owns
// its backend, native streams, scratch and capture cache. Payload is borrowed
// through Completion. Managed replay may allocate in upstream pools and may
// block the host; it cannot be captured by an external CUDA graph. A failed
// invocation retires permanently; instantiate another from the same program.
#if defined(HYLE_COMPUTE_GGML_CUDA)
HYLE_COMPUTE_PROVIDER_GGML_CUDA_API Provider provider() noexcept;
#endif

} // namespace hyle::compute::ggml_cuda
