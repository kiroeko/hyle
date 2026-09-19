#include "sglang_norm_kernel.h"

#include <algorithm>

#include "rmsnorm.cuh"

namespace hyle::compute::cuda_native::native {
namespace {

template <class T, unsigned Width> const void *select_width() noexcept {
  if constexpr (Width <= 256)
    return reinterpret_cast<const void *>(
        sglang_vendor::rmsnorm_warp<Width, false, T>);
  else if constexpr (Width == 512 || (Width >= 2048 && Width % 512 == 0))
    return reinterpret_cast<const void *>(
        sglang_vendor::rmsnorm_cta_double<Width, false, T>);
  else
    return reinterpret_cast<const void *>(
        sglang_vendor::rmsnorm_cta<Width, false, T>);
}

template <class T, unsigned Width = 512>
const void *select_large(unsigned hidden) noexcept {
  if (hidden == Width)
    return select_width<T, Width>();
  if constexpr (Width < 16384)
    return select_large<T, Width + (Width < 8192 ? 256 : 512)>(hidden);
  return nullptr;
}

template <class T> const void *select(unsigned hidden) noexcept {
  switch (hidden) {
  case 64:
    return select_width<T, 64>();
  case 128:
    return select_width<T, 128>();
  case 256:
    return select_width<T, 256>();
  default:
    return select_large<T>(hidden);
  }
}
} // namespace

cudaError_t sglang_norm_prepare(ElementType type, unsigned rows,
                                unsigned hidden, float epsilon,
                                SglangNormProgram *output) noexcept {
  if (output == nullptr)
    return cudaErrorInvalidValue;
  if (!sglang_norm_width(hidden) || type == ElementType::kFloat32)
    return cudaErrorNotSupported;
  const auto kernel = type == ElementType::kFloat16
                          ? select<__half>(hidden)
                          : select<__nv_bfloat16>(hidden);
  const bool half = hidden == 512 || (hidden >= 2048 && hidden % 512 == 0);
  const unsigned threads = hidden <= 256 ? 32 : half ? hidden / 16 : hidden / 8;
  unsigned blocks = rows;
  if (rows != 0) {
    cudaFuncAttributes attributes{};
    auto status = cudaFuncGetAttributes(&attributes, kernel);
    if (status != cudaSuccess)
      return status;
    if (attributes.maxThreadsPerBlock < static_cast<int>(threads))
      return cudaErrorNotSupported;
    if (!half) {
      int occupancy = 0, device = 0, sms = 0;
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, kernel, static_cast<int>(threads), 0);
      if (status == cudaSuccess)
        status = cudaGetDevice(&device);
      if (status == cudaSuccess)
        status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                                        device);
      if (status != cudaSuccess)
        return status;
      if (occupancy <= 0 || sms <= 0)
        return cudaErrorNotSupported;
      blocks = std::min(rows, static_cast<unsigned>(occupancy * sms));
    }
  }
  *output = {kernel, rows, hidden, epsilon, threads, blocks};
  return cudaSuccess;
}

cudaError_t sglang_norm_launch(const SglangNormProgram &program,
                               const void *input, const void *scale,
                               void *output, cudaStream_t stream) noexcept {
  if (program.rows == 0)
    return cudaSuccess;
  sglang_vendor::RMSNormParams params{
      input,          scale,        output,         program.hidden,
      program.hidden, program.rows, program.epsilon};
  void *arguments[]{&params};
  return cudaLaunchKernel(program.kernel, dim3(program.blocks),
                          dim3(program.threads), arguments, 0, stream);
}
} // namespace hyle::compute::cuda_native::native
