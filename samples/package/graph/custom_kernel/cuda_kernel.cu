#include "cuda_kernel.h"

namespace {
__global__ void affine(const float *input, float *output, std::size_t count,
                       float scale, float bias) {
  const auto index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
    output[index] = __fmaf_rn(input[index], scale, bias);
}
} // namespace

cudaError_t launch_affine(cudaStream_t stream, const float *input,
                          float *output, std::size_t count, float scale,
                          float bias, unsigned block_size) noexcept {
  const auto blocks =
      static_cast<unsigned>((count + block_size - 1) / block_size);
  affine<<<blocks, block_size, 0, stream>>>(input, output, count, scale, bias);
  return cudaPeekAtLastError();
}
