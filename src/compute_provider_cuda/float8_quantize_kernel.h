#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace hyle::compute::cuda_native::float8_quantize::native {
struct Program final {
  std::uint64_t groups = 0;
  float floor = 1.0e-10F;
  bool bfloat = false, power2 = false, multiply_reciprocal = false;
};
cudaError_t prepare(Program) noexcept;
cudaError_t launch(Program, const void *input, unsigned char *codes,
                   float *scales, unsigned *status, cudaStream_t) noexcept;
} // namespace hyle::compute::cuda_native::float8_quantize::native
