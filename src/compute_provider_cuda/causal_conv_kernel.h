#pragma once

#include "native_kernel.h"
#include <cstdint>

namespace hyle::compute::cuda_native::native {

struct CausalConvLayout final {
  std::uint64_t elements = 0;
  std::uint64_t window = 0;
  std::uint64_t channels = 0;
  std::uint64_t taps = 0;
  std::uint64_t tokens = 0;
};

struct CausalConvProgram final {
  const void *kernel = nullptr;
  unsigned blocks = 0;
  CausalConvLayout layout{};
};

// Dense F32 samples [B,W,C] and weights [C,K] produce [B,W-K+1,C].
// Preparation validates all element/byte offsets and binds launch metadata.
cudaError_t causal_conv_prepare(const CausalConvLayout &layout,
                               CausalConvProgram *output) noexcept;
cudaError_t causal_conv_launch(const CausalConvProgram &program,
                              const void *samples, const void *weights,
                              void *output, cudaStream_t stream) noexcept;

} // namespace hyle::compute::cuda_native::native
