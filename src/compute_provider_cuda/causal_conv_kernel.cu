#include "causal_conv_kernel.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <limits>

namespace hyle::compute::cuda_native::native {
namespace {
constexpr unsigned kThreads = 256;
constexpr std::uint64_t kMaxElements =
    static_cast<std::uint64_t>((std::numeric_limits<std::ptrdiff_t>::max)()) /
    sizeof(float);

bool multiply_elements(std::uint64_t first, std::uint64_t second,
                       std::uint64_t *output) noexcept {
  if (second != 0 && first > kMaxElements / second)
    return false;
  *output = first * second;
  return true;
}

bool valid_layout(const CausalConvLayout &layout) noexcept {
  // The public operation requires positive B/W/C/K and therefore positive T.
  if (layout.elements == 0 || layout.elements > kMaxElements ||
      layout.channels == 0 || layout.taps == 0 ||
      layout.window < layout.taps ||
      layout.tokens != layout.window - layout.taps + 1)
    return false;
  std::uint64_t output_row = 0, source_row = 0, source_elements = 0,
                weight_elements = 0;
  return multiply_elements(layout.tokens, layout.channels, &output_row) &&
         layout.elements % output_row == 0 &&
         multiply_elements(layout.window, layout.channels, &source_row) &&
         multiply_elements(layout.elements / output_row, source_row,
                           &source_elements) &&
         multiply_elements(layout.channels, layout.taps, &weight_elements);
}

__global__ void causal_conv_f32(const float *samples, const float *weights,
                                float *output, CausalConvLayout layout) {
  const auto stride = std::uint64_t(gridDim.x) * blockDim.x;
  for (auto index = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < layout.elements; index += stride) {
    const auto channel = index % layout.channels;
    const auto row = index / layout.channels;
    const auto token = row % layout.tokens;
    const auto batch = row / layout.tokens;
    const auto sample = (batch * layout.window + token) * layout.channels +
                        channel;
    const auto weight = channel * layout.taps;
    float sum = 0.0f;
    for (std::uint64_t tap = 0; tap < layout.taps; ++tap)
      sum = __fmaf_rn(samples[sample + tap * layout.channels],
                      weights[weight + tap], sum);
    output[index] = sum;
  }
}
} // namespace

cudaError_t causal_conv_prepare(const CausalConvLayout &layout,
                               CausalConvProgram *output) noexcept {
  if (output == nullptr || !valid_layout(layout))
    return cudaErrorInvalidValue;
  const auto *kernel = reinterpret_cast<const void *>(causal_conv_f32);
  cudaFuncAttributes attributes{};
  auto status = cudaFuncGetAttributes(&attributes, kernel);
  if (status != cudaSuccess)
    return status;
  if (attributes.maxThreadsPerBlock < static_cast<int>(kThreads))
    return cudaErrorNotSupported;
  int device = 0, grid_limit = 0;
  status = cudaGetDevice(&device);
  if (status == cudaSuccess)
    status = cudaDeviceGetAttribute(&grid_limit, cudaDevAttrMaxGridDimX, device);
  if (status != cudaSuccess)
    return status;
  if (grid_limit <= 0)
    return cudaErrorNotSupported;
  const auto work_blocks = layout.elements / kThreads +
                           (layout.elements % kThreads != 0);
  const auto blocks = static_cast<unsigned>((std::min)(
      work_blocks, (std::min)(std::uint64_t{65535},
                              static_cast<std::uint64_t>(grid_limit))));
  *output = {kernel, blocks, layout};
  return cudaSuccess;
}

cudaError_t causal_conv_launch(const CausalConvProgram &program,
                              const void *samples, const void *weights,
                              void *output, cudaStream_t stream) noexcept {
  if (program.kernel == nullptr || program.blocks == 0 || samples == nullptr ||
      weights == nullptr || output == nullptr)
    return cudaErrorInvalidValue;
  auto layout = program.layout;
  void *arguments[]{&samples, &weights, &output, &layout};
  return cudaLaunchKernel(program.kernel, dim3(program.blocks), dim3(kThreads),
                          arguments, 0, stream);
}

} // namespace hyle::compute::cuda_native::native
