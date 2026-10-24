#pragma once

#include "hyle/compute/provider.h"
#include "hyle/visibility.h"

namespace hyle::compute::torch_cuda {
inline constexpr ProviderId kProviderId{0x48594c4500000001ULL,
                                       0x544f524355444101ULL};
// Original NF4/nested, AutoAWQ and compressed-tensors decode, native GPTQ-v1
// decode, MLX packed equations, stable Softplus and fused BF16 BSHD attention
// run on the caller's CUDA stream. Attention uses F32 accumulation, GQA and
// bottom-right causal ordering without a mask port; it requires SM80-SM121
// and head depth 8..256 in steps of 8. Probability tiles retain BF16.
//
// Ordinary NF4/AutoAWQ/CT decode and attention use managed private ATen
// allocations; strict caller budgets and external capture are unsupported
// for those allocation routes. Softplus writes the caller's destination and
// advertises capture compatibility. MLX retains F32 accumulation, original
// group-scale order and device-checked IDs; it reads current packed weights
// without an unpacked bank or host IDs check, but does not advertise capture.
//
// Checked native GPTQ-v1 decode and closed ordinary CT/GPTQ linear recipes
// support capture. Closed stored-scale FP8 recipes also use native capture
// with caller-owned activation-code workspace. The contracted linear carriers
// are same-build private implementations of ordinary graphs, not new public
// operations; their preparation reports retain operation-specific storage and
// replay requirements. This does not make all artifact decoders allocation-free.
// Borrowed input/output resources must cover Completion, including a failed prefix.
#if defined(HYLE_COMPUTE_TORCH_CUDA)
HYLE_COMPUTE_PROVIDER_TORCH_CUDA_API Provider provider() noexcept;
#endif
} // namespace hyle::compute::torch_cuda
