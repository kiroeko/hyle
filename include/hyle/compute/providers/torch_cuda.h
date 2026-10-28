#pragma once

#include "hyle/compute/provider.h"
#include "hyle/visibility.h"

namespace hyle::compute::torch_cuda {
inline constexpr ProviderId kProviderId{0x48594c4500000001ULL,
                                       0x544f524355444101ULL};
// Optional ATen adapter: original NF4/nested, AutoAWQ and compressed-tensors
// decode, stable Softplus and fused BF16 BSHD attention
// run on the caller's CUDA stream. Attention uses F32 accumulation, GQA and
// bottom-right causal ordering without a mask port; it requires SM80-SM121
// and head depth 8..256 in steps of 8. Probability tiles retain BF16.
//
// Ordinary NF4/AutoAWQ/CT decode and attention use managed private ATen
// allocations; strict caller budgets and external capture are unsupported
// for those allocation routes. Softplus writes the caller's destination and
// advertises capture compatibility. Native MLX/GPTQ decoding and contracted
// packed-linear/FP8 graphs belong to cuda_native::provider() and do not require
// this adapter. HYLE_WITH_ATEN=OFF omits the ATen SDK and this provider.
// Borrowed input/output resources must cover Completion, including a failed prefix.
#if defined(HYLE_COMPUTE_TORCH_CUDA)
HYLE_COMPUTE_PROVIDER_TORCH_CUDA_API Provider provider() noexcept;
#endif
} // namespace hyle::compute::torch_cuda
