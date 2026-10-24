#pragma once

#include <cuda_runtime_api.h>

namespace hyle::compute::cuda_native::deepseek_fp4::native {
struct Program final {
  const void *kernel=nullptr;
  int experts=0,columns=0,depth=0,slots=0,activation_slots=0,rows=0;
};
cudaError_t prepare(Program *program) noexcept;
cudaError_t launch(const Program &program,const unsigned char *activation,
    const unsigned char *weight,void *output,const float *activation_scales,
    const float *weight_scales,const int *ids,unsigned *status,cudaStream_t stream) noexcept;
} // namespace hyle::compute::cuda_native::deepseek_fp4::native
