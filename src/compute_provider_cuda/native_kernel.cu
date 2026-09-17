#include "native_kernel.h"

#include <cstddef>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace hyle::compute::cuda_native::native {
namespace {

constexpr unsigned kThreads = 256;

template <class T> __device__ float load(T value) { return value; }
template <> __device__ float load(__half value) { return __half2float(value); }
template <> __device__ float load(__nv_bfloat16 value) {
  return __bfloat162float(value);
}
template <class T> __device__ T store(float value) { return value; }
template <> __device__ __half store(float value) {
  return __float2half_rn(value);
}
template <> __device__ __nv_bfloat16 store(float value) {
  return __float2bfloat16_rn(value);
}

template <Activation Kind> __device__ float activate(float value) {
  if constexpr (Kind == Activation::kSilu)
    return value / (1.0F + expf(-value));
  else if constexpr (Kind == Activation::kGelu)
    return value * (0.5F * (1.0F + erff(value * 0.7071067811865475F)));
  else
    return value *
           (0.5F * (1.0F + tanhf(0.7978845608028654F *
                                 (value + 0.044715F * value * value * value))));
}

template <class T, bool Packed, Activation Kind>
__global__ void gated(const T *first, const T *second, T *output,
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
  // Match activation(gate) * value with FP32 intermediates. Do not enable
  // CUDA's fast-math mode or round the activation to FP16 before the multiply.
  output[index] = store<T>(activate<Kind>(gate) * value);
}

template <class T, Activation Kind> const void *kernel(bool packed) noexcept {
  return packed ? reinterpret_cast<const void *>(gated<T, true, Kind>)
                : reinterpret_cast<const void *>(gated<T, false, Kind>);
}

template <class T>
const void *kernel(Activation activation, bool packed) noexcept {
  switch (activation) {
  case Activation::kSilu:
    return kernel<T, Activation::kSilu>(packed);
  case Activation::kGelu:
    return kernel<T, Activation::kGelu>(packed);
  case Activation::kGeluTanh:
    return kernel<T, Activation::kGeluTanh>(packed);
  }
  return nullptr;
}

} // namespace

cudaError_t prepare(ElementType type, Activation activation, bool packed,
                    unsigned elements, unsigned hidden,
                    Program *output) noexcept {
  if (output == nullptr || (elements != 0 && hidden == 0))
    return cudaErrorInvalidValue;
  const void *function = nullptr;
  switch (type) {
  case ElementType::kFloat32:
    function = kernel<float>(activation, packed);
    break;
  case ElementType::kFloat16:
    function = kernel<__half>(activation, packed);
    break;
  case ElementType::kBFloat16:
    function = kernel<__nv_bfloat16>(activation, packed);
    break;
  }
  if (function == nullptr)
    return cudaErrorInvalidValue;
  if (elements == 0) {
    *output = {nullptr, 0, hidden, 0};
    return cudaSuccess;
  }
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
  if (program.elements == 0)
    return cudaSuccess;
  unsigned elements = program.elements;
  unsigned hidden = program.hidden;
  void *arguments[]{&first, &second, &output, &elements, &hidden};
  return cudaLaunchKernel(program.kernel, dim3(program.blocks), dim3(kThreads),
                          arguments, 0, stream);
}

} // namespace hyle::compute::cuda_native::native
