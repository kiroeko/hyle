#include "norm_kernel.h"

#include <cstddef>

#include <cooperative_groups/reduce.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace hyle::compute::cuda_native::native {
namespace {
namespace cg = cooperative_groups;
constexpr unsigned kThreads = 256;

template <class T> __device__ float decode(T value) { return value; }
template <> __device__ float decode(__half value) {
  return __half2float(value);
}
template <> __device__ float decode(__nv_bfloat16 value) {
  return __bfloat162float(value);
}
template <class T> __device__ T encode(float value) { return value; }
template <> __device__ __half encode(float value) {
  return __float2half_rn(value);
}
template <> __device__ __nv_bfloat16 encode(float value) {
  return __float2bfloat16_rn(value);
}

template <class T, class Weight, class Bias, bool Layer>
__global__ void norm(const T *input, const Weight *scale, const Bias *bias,
                     T *output, unsigned hidden, float epsilon) {
  // CUDA SDK owns the multi-warp collective and its on-chip scratch. There is
  // one complete tile per row; all threads participate, including tail lanes.
  const auto tile = cg::tiled_partition<kThreads>(cg::this_thread_block());
  const std::size_t row = static_cast<std::size_t>(blockIdx.x) * hidden;
  float local = 0;
  for (unsigned col = threadIdx.x; col < hidden; col += kThreads) {
    const float value = decode(input[row + col]);
    if constexpr (Layer)
      local += value;
    else
      local += value * value;
  }
  const float first = cg::reduce(tile, local, cg::plus<float>()) / hidden;
  float mean = 0;
  float variance = first;
  if constexpr (Layer) {
    mean = first;
    local = 0;
    for (unsigned col = threadIdx.x; col < hidden; col += kThreads) {
      const float centered = decode(input[row + col]) - mean;
      local += centered * centered;
    }
    variance = cg::reduce(tile, local, cg::plus<float>()) / hidden;
  }
  const float inverse = rsqrtf(variance + epsilon);
  for (unsigned col = threadIdx.x; col < hidden; col += kThreads) {
    float value = (decode(input[row + col]) - mean) * inverse;
    if (scale != nullptr)
      value *= decode(scale[col]);
    if (bias != nullptr)
      value += decode(bias[col]);
    output[row + col] = encode<T>(value);
  }
}

template <class T, bool Layer>
const void *bind(bool scale_f32, bool bias_f32) noexcept {
  if (scale_f32)
    return bias_f32
               ? reinterpret_cast<const void *>(norm<T, float, float, Layer>)
               : reinterpret_cast<const void *>(norm<T, float, T, Layer>);
  return bias_f32 ? reinterpret_cast<const void *>(norm<T, T, float, Layer>)
                  : reinterpret_cast<const void *>(norm<T, T, T, Layer>);
}
template <class T>
const void *bind(bool layer, bool scale_f32, bool bias_f32) noexcept {
  return layer ? bind<T, true>(scale_f32, bias_f32)
               : bind<T, false>(scale_f32, bias_f32);
}
} // namespace

cudaError_t norm_prepare(ElementType type, bool layer, bool scale_f32,
                         bool bias_f32, unsigned rows, unsigned hidden,
                         float epsilon, NormProgram *output) noexcept {
  if (output == nullptr || (rows != 0 && hidden == 0))
    return cudaErrorInvalidValue;
  const void *kernel = nullptr;
  switch (type) {
  case ElementType::kFloat32:
    kernel = bind<float>(layer, scale_f32, bias_f32);
    break;
  case ElementType::kFloat16:
    kernel = bind<__half>(layer, scale_f32, bias_f32);
    break;
  case ElementType::kBFloat16:
    kernel = bind<__nv_bfloat16>(layer, scale_f32, bias_f32);
    break;
  }
  if (kernel == nullptr)
    return cudaErrorInvalidValue;
  if (rows != 0) {
    cudaFuncAttributes attributes{};
    const auto status = cudaFuncGetAttributes(&attributes, kernel);
    if (status != cudaSuccess)
      return status;
    if (attributes.maxThreadsPerBlock < static_cast<int>(kThreads))
      return cudaErrorNotSupported;
  }
  *output = {kernel, rows, hidden, epsilon};
  return cudaSuccess;
}

cudaError_t norm_launch(const NormProgram &program, const void *input,
                        const void *scale, const void *bias, void *output,
                        cudaStream_t stream) noexcept {
  if (program.rows == 0)
    return cudaSuccess;
  unsigned hidden = program.hidden;
  float epsilon = program.epsilon;
  void *arguments[]{&input, &scale, &bias, &output, &hidden, &epsilon};
  return cudaLaunchKernel(program.kernel, dim3(program.rows), dim3(kThreads),
                          arguments, 0, stream);
}
} // namespace hyle::compute::cuda_native::native
