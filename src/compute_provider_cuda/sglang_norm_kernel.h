#pragma once

#include "native_kernel.h"

namespace hyle::compute::cuda_native::native {

constexpr bool sglang_norm_width(unsigned hidden) noexcept {
  return hidden == 64 || hidden == 128 || hidden == 256 ||
         (hidden > 256 && hidden <= 8192 && hidden % 256 == 0) ||
         (hidden >= 8192 && hidden <= 16384 && hidden % 512 == 0);
}

struct SglangNormProgram final {
  const void *kernel = nullptr;
  unsigned rows = 0;
  unsigned hidden = 0;
  float epsilon = 0;
  unsigned threads = 0;
  unsigned blocks = 0;
};

cudaError_t sglang_norm_prepare(ElementType type, unsigned rows,
                                unsigned hidden, float epsilon,
                                SglangNormProgram *output) noexcept;
cudaError_t sglang_norm_launch(const SglangNormProgram &program,
                               const void *input, const void *scale,
                               void *output, cudaStream_t stream) noexcept;

} // namespace hyle::compute::cuda_native::native
