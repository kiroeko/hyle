#pragma once

#include "hyle/compute/provider.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_COMPUTE_PROVIDER_TORCH_CUDA_API
#elif defined(HYLE_COMPUTE_PROVIDER_TORCH_CUDA_BUILD)
#define HYLE_COMPUTE_PROVIDER_TORCH_CUDA_API HYLE_EXPORT
#else
#define HYLE_COMPUTE_PROVIDER_TORCH_CUDA_API HYLE_IMPORT
#endif

namespace hyle::compute::torch_cuda {
inline constexpr ProviderId kProviderId{0x48594c4500000001ULL,
                                       0x544f524355444101ULL};
// Original NF4/nested, AutoAWQ and compressed-tensors weight preparation on
// the caller's CUDA stream. Private ATen allocations are managed replay;
// strict caller budgets and external capture are unsupported. Borrowed input
// and output resources must cover Completion, including a failed prefix.
#if defined(HYLE_COMPUTE_TORCH_CUDA)
HYLE_COMPUTE_PROVIDER_TORCH_CUDA_API Provider provider() noexcept;
#endif
} // namespace hyle::compute::torch_cuda
