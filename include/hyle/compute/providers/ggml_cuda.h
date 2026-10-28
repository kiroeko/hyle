#pragma once

#include "hyle/compute/provider.h"
#include "hyle/visibility.h"

namespace hyle::compute::ggml_cuda {

inline constexpr ProviderId kProviderId{0x48594c4500000001ULL,
                                        0x4747435544410001ULL};

// Original ggml CUDA programs. Native per-node support and the compiled device
// program determine availability; an SM89 timing certificate is not an
// all-provider capability gate. Each invocation owns its backend, native
// pools, persistent inputs and capture cache. Graph temporaries use the exact
// caller workspace requirement; that backing stays borrowed through Completion
// and any subsequent replay. SDPA keeps probability and accumulation in F32, with
// only the final port conversion; large standalone queries reuse row scratch.
// Managed replay may allocate in upstream pools or block the host. External
// CUDA graph recording is admitted only when the actual lowered native graph
// is record-compatible; managed replay alone does not imply that capability.
// A failed invocation retires permanently; instantiate another from the same
// program.
#if defined(HYLE_COMPUTE_GGML_CUDA)
HYLE_COMPUTE_PROVIDER_GGML_CUDA_API Provider provider() noexcept;
#endif

} // namespace hyle::compute::ggml_cuda
