#pragma once

#include <cstdint>

// Same-build private native contract. Cold lowering checks the real linked
// symbol as well as this marker; a header alone cannot qualify an old SDK.
extern "C" std::int32_t ggml_hyle_cuda_conv_indexed_state_contract();

namespace hyle::compute::ggml_cuda::internal {
inline constexpr std::int32_t kIndexedConvContract = 0x43444931;
inline constexpr int kIndexedConvParameter = 4;
} // namespace hyle::compute::ggml_cuda::internal
