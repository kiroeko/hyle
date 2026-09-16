#include "native_kernel.h"
#include "sglang_activation.cuh"

namespace hyle::compute::sglang_reference::native {
namespace {
constexpr unsigned kThreads = 256;
template <class T> const void *kernel() noexcept {
  return reinterpret_cast<const void *>(
      sglang::act_and_mul_kernel<T, sglang::ActivationKind::kSiLU, false, false,
                                 false, false>);
}
} // namespace

cudaError_t prepare(bool half, bool packed, unsigned elements, unsigned hidden,
                    Program *output) noexcept {
  if (output == nullptr || elements == 0 || hidden == 0)
    return cudaErrorInvalidValue;
  const unsigned vector_elements = half ? 8U : 4U;
  if (!packed || hidden % vector_elements != 0 || elements % hidden != 0)
    return cudaErrorNotSupported;
  const void *function = half ? kernel<__half>() : kernel<float>();
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

cudaError_t launch(const Program &program, const void *first, const void *,
                   void *output, cudaStream_t stream) noexcept {
  auto params = sglang::ActivationParams{
      first,   output, program.hidden, program.elements / program.hidden,
      nullptr, 1};
  void *arguments[]{&params};
  return cudaLaunchKernel(program.kernel, dim3(program.blocks), dim3(kThreads),
                          arguments, 0, stream);
}
} // namespace hyle::compute::sglang_reference::native
