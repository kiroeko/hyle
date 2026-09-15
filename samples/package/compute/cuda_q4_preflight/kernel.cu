#include "kernel.h"

#include <cstddef>
#include <cuda_fp16.h>

namespace q4_preflight {
namespace {

__global__ void matmul_kernel(const float *input, const Block *weights,
                              float *output, int columns, int inner) {
  const int lane = threadIdx.x % 32;
  const int column = blockIdx.x * 4 + threadIdx.x / 32;
  if (column >= columns)
    return; // Uniform within the entire warp.
  const int row = blockIdx.y;
  const int blocks = inner / 32;
  float sum = 0.0F;
  for (int block = 0; block < blocks; ++block) {
    const auto &weight = weights[column * blocks + block];
    const unsigned packed = weight.quants[lane % 16];
    const int quant =
        static_cast<int>((packed >> (lane < 16 ? 0 : 4)) & 15) - 8;
    const float scale = __half2float(__ushort_as_half(weight.scale));
    sum = __fmaf_rn(
        input[static_cast<std::size_t>(row) * inner + block * 32 + lane],
        static_cast<float>(quant) * scale, sum);
  }
  for (int offset = 16; offset != 0; offset /= 2)
    sum += __shfl_down_sync(0xffffffffU, sum, offset);
  if (lane == 0)
    output[static_cast<std::size_t>(row) * columns + column] = sum;
}

} // namespace

cudaError_t matmul(cudaStream_t stream, const float *input,
                   const Block *weights, float *output, int rows, int columns,
                   int inner) noexcept {
  if (rows < 0 || columns < 0 || inner <= 0 || inner % 32 != 0 ||
      rows > 65535 || columns > 65535 || inner > 65535)
    return cudaErrorInvalidValue;
  if (rows == 0 || columns == 0)
    return cudaSuccess;
  if (!input || !weights || !output)
    return cudaErrorInvalidValue;
  const dim3 grid(static_cast<unsigned>((columns + 3) / 4),
                  static_cast<unsigned>(rows));
  matmul_kernel<<<grid, 128, 0, stream>>>(input, weights, output, columns,
                                          inner);
  return cudaPeekAtLastError();
}

} // namespace q4_preflight
