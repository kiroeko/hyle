#include "native_kernel.h"

#include "activation.cuh"

namespace hyle::compute::cuda_native::native {
namespace {

constexpr unsigned kThreads = 256;

template <class T, sglang_vendor::ActivationKind Kind>
const void *kernel() noexcept {
  return reinterpret_cast<const void *>(
      sglang_vendor::act_and_mul_kernel<T, Kind, false, false, false, false>);
}

template <class T> const void *kernel(Activation activation) noexcept {
  switch (activation) {
  case Activation::kSilu:
    return kernel<T, sglang_vendor::ActivationKind::kSiLU>();
  case Activation::kGelu:
    return kernel<T, sglang_vendor::ActivationKind::kGELU>();
  case Activation::kGeluTanh:
    return kernel<T, sglang_vendor::ActivationKind::kGELUTanh>();
  }
  return nullptr;
}

} // namespace

cudaError_t sglang_prepare(ElementType type, Activation activation, bool packed,
                           unsigned elements, unsigned hidden,
                           Program *output) noexcept {
  if (output == nullptr || (elements != 0 && hidden == 0))
    return cudaErrorInvalidValue;
  const unsigned vector_elements = type == ElementType::kFloat32 ? 4U : 8U;
  if (!packed || hidden % vector_elements != 0 ||
      (elements != 0 && elements % hidden != 0))
    return cudaErrorNotSupported;
  const void *function = nullptr;
  switch (type) {
  case ElementType::kFloat32:
    function = kernel<float>(activation);
    break;
  case ElementType::kFloat16:
    function = kernel<__half>(activation);
    break;
  case ElementType::kBFloat16:
    function = kernel<__nv_bfloat16>(activation);
    break;
  }
  if (function == nullptr)
    return cudaErrorInvalidValue;
  if (elements == 0) {
    *output = {nullptr, 0, hidden, 0};
    return cudaSuccess;
  }
  cudaFuncAttributes attributes{};
  const auto result = cudaFuncGetAttributes(&attributes, function);
  if (result != cudaSuccess)
    return result;
  if (attributes.maxThreadsPerBlock < static_cast<int>(kThreads))
    return cudaErrorNotSupported;
  const auto vectors = elements / vector_elements;
  *output = {function, elements, hidden, (vectors + kThreads - 1) / kThreads};
  return cudaSuccess;
}

cudaError_t sglang_launch(const Program &program, const void *first,
                          const void *, void *output,
                          cudaStream_t stream) noexcept {
  if (program.elements == 0)
    return cudaSuccess;
  auto params = sglang_vendor::ActivationParams{
      first,   output, program.hidden, program.elements / program.hidden,
      nullptr, 1};
  void *arguments[]{&params};
  return cudaLaunchKernel(program.kernel, dim3(program.blocks), dim3(kThreads),
                          arguments, 0, stream);
}

} // namespace hyle::compute::cuda_native::native
