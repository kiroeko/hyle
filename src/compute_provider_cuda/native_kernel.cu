#include "native_kernel.h"

#include <cstddef>

#include <cuda_fp16.h>

namespace hyle::compute::cuda_native::native {
namespace {

constexpr unsigned kThreads = 256;

template <class T> __device__ float load(T value) { return value; }
template <> __device__ float load(__half value) { return __half2float(value); }
template <class T> __device__ T store(float value) { return value; }
template <> __device__ __half store(float value) {
  return __float2half_rn(value);
}

template <class T, bool Packed>
__global__ void swiglu(const T *first, const T *second, T *output,
                       unsigned elements, unsigned hidden) {
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements)
    return;
  std::size_t gate_index = index;
  if constexpr (Packed)
    gate_index += static_cast<std::size_t>(index / hidden) * hidden;
  const float gate = load(first[gate_index]);
  const float value =
      Packed ? load(first[gate_index + hidden]) : load(second[index]);
  // Match SiLU(gate) * value with FP32 intermediates. Do not enable CUDA's
  // fast-math mode or round the activation to FP16 before the multiply.
  output[index] = store<T>((gate / (1.0F + expf(-gate))) * value);
}

template <class T> const void *kernel(bool packed) noexcept {
  return packed ? reinterpret_cast<const void *>(swiglu<T, true>)
                : reinterpret_cast<const void *>(swiglu<T, false>);
}

} // namespace

cudaError_t prepare(bool half, bool packed, unsigned elements, unsigned hidden,
                    Program *output) noexcept {
  if (output == nullptr || elements == 0 || hidden == 0)
    return cudaErrorInvalidValue;
  const void *function = half ? kernel<__half>(packed) : kernel<float>(packed);
  // Resolve/load this compiled image during preparation, before capture or
  // replay. The driver owns its code storage; no per-slot native scratch.
  cudaFuncAttributes attributes{};
  const auto result = cudaFuncGetAttributes(&attributes, function);
  if (result != cudaSuccess)
    return result;
  if (attributes.maxThreadsPerBlock < static_cast<int>(kThreads))
    return cudaErrorNotSupported;
  *output = {function, elements, hidden, (elements + kThreads - 1) / kThreads};
  return cudaSuccess;
}

cudaError_t launch(const Program &program, const void *first,
                   const void *second, void *output,
                   cudaStream_t stream) noexcept {
  unsigned elements = program.elements;
  unsigned hidden = program.hidden;
  void *arguments[]{&first, &second, &output, &elements, &hidden};
  return cudaLaunchKernel(program.kernel, dim3(program.blocks), dim3(kThreads),
                          arguments, 0, stream);
}

} // namespace hyle::compute::cuda_native::native
