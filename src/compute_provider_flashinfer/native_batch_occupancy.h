#pragma once
#include <cuda_runtime_api.h>

namespace hyle::compute::flashinfer::native {
// Cold device query for the selected BF16/HD256/CTA64 paged prefill kernel.
cudaError_t paged_prefill_bf16_d256_occupancy(int &blocks_per_sm) noexcept;
}
