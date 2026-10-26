#pragma once

#include <cstdint>

// Same-build private native contract. A marker alone cannot qualify an old
// linked CUDA runtime; cold lowering checks the actual native symbol as well.
extern "C" std::int32_t ggml_hyle_cuda_gdn_indexed_state_contract();

namespace hyle::compute::ggml_cuda::internal {
inline constexpr std::int32_t kIndexedStateContract = 0x47444931;
inline constexpr int kIndexedStateParameter = 4;
} // namespace hyle::compute::ggml_cuda::internal
