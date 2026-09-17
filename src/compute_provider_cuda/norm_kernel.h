#pragma once

#include "native_kernel.h"

namespace hyle::compute::cuda_native::native {

struct NormProgram final {
  const void *kernel = nullptr;
  unsigned rows = 0;
  unsigned hidden = 0;
  float epsilon = 0;
};

cudaError_t norm_prepare(ElementType type, bool layer, bool scale_f32,
                         bool bias_f32, unsigned rows, unsigned hidden,
                         float epsilon, NormProgram *output) noexcept;
cudaError_t norm_launch(const NormProgram &program, const void *input,
                        const void *scale, const void *bias, void *output,
                        cudaStream_t stream) noexcept;

} // namespace hyle::compute::cuda_native::native
