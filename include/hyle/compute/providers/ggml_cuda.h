#pragma once

#include "hyle/compute/provider.h"
#include "hyle/visibility.h"

namespace hyle::compute::ggml_cuda {

inline constexpr ProviderId kProviderId{0x48594c4500000001ULL,
                                        0x4747435544410001ULL};

// Original ggml CUDA programs. Native per-node support and the compiled device
// program determine availability; an SM89 timing certificate is not an
// all-provider capability gate. Each invocation owns
// its backend, native streams, scratch and capture cache. Payload is borrowed
// through Completion. SDPA keeps probability and accumulation in F32, with
// only the final port conversion; large standalone queries reuse row scratch.
// Managed replay may allocate in upstream pools and may
// block the host; it cannot be captured by an external CUDA graph. A failed
// invocation retires permanently; instantiate another from the same program.
#if defined(HYLE_COMPUTE_GGML_CUDA)
HYLE_COMPUTE_PROVIDER_GGML_CUDA_API Provider provider() noexcept;
#endif

} // namespace hyle::compute::ggml_cuda
