#pragma once
#include "program.h"

namespace hyle::compute::ggml_cuda::internal {
// Trusted cold Graph bridge, after the original invocation retirement and
// opaque Batch layout/mirror/offset ValueId qualification. The ID is a local
// Recipe value ID, not a boundary ordinal. No lease or execution is created.
// Configuration covers every PackedRowMatMul using that exact input, and
// expires on clear/rebinding. Native storage and all physical C headers stay.
HYLE_COMPUTE_PROVIDER_GGML_CUDA_API Status configure_rows(
    const detail::ProviderPrepared &, void *invocation,
    std::uint32_t recipe_offset_input, ConstTensorView offsets,
    std::size_t rows) noexcept;
HYLE_COMPUTE_PROVIDER_GGML_CUDA_API Status clear_rows(
    const detail::ProviderPrepared &, void *invocation) noexcept;
} // namespace hyle::compute::ggml_cuda::internal
