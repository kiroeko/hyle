#pragma once

#include "hyle/compute/provider.h"
#include "hyle/visibility.h"

namespace hyle::compute::torch_cuda {
inline constexpr ProviderId kProviderId{0x48594c4500000001ULL,
                                       0x544f524355444101ULL};
// Original NF4/nested, AutoAWQ and compressed-tensors weight preparation plus
// MLX native packed equations (F32 accumulation, original group-scale order
// and device-checked IDs) on the caller's CUDA stream, plus stable Softplus
// and fused BF16 BSHD attention (F32 accumulation, GQA, bottom-right causal,
// no mask port). Attention requires SM80-SM121 and head depth 8..256 in steps
// of 8; probability tiles retain BF16 rather than fully F32 intermediates.
// MLX reads current bound weights without unpacked-bank temporaries or a host
// IDs check. Artifact decode/Softplus/attention use managed private ATen allocations;
// strict caller budgets and external capture are unsupported. Borrowed input
// and output resources must cover Completion, including a failed prefix.
#if defined(HYLE_COMPUTE_TORCH_CUDA)
HYLE_COMPUTE_PROVIDER_TORCH_CUDA_API Provider provider() noexcept;
#endif
} // namespace hyle::compute::torch_cuda
