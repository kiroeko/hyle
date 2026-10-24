#include "row_cache_indexer_kernel.h"
#include "device_status_kernel.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <type_traits>

namespace hyle::compute::cuda_native::row_cache_indexer::native {
namespace {
constexpr unsigned kThreads = 256;
__global__ void initialize(float *output, std::uint64_t count,
                           const unsigned *status) {
  if (*status)
    return;
  for (auto i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += std::uint64_t(gridDim.x) * blockDim.x)
    output[i] = -CUDART_INF_F;
}
__global__ void validate(const std::int32_t *ranges, std::uint32_t tokens,
                         std::uint32_t rows, unsigned *status) {
  for (auto t = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       t < tokens; t += std::uint64_t(gridDim.x) * blockDim.x) {
    const auto begin = ranges[t * 2], end = ranges[t * 2 + 1];
    if (begin < 0 || end < begin || std::uint64_t(end) > rows)
      atomicCAS(status, 0U,
                unsigned(cuda::detail::DeviceStatusCode::kIndexOutOfRange));
  }
}
__global__ void active(const std::int32_t *ranges, std::uint32_t tokens,
                       std::uint32_t begin, unsigned width, unsigned char *keys,
                       const unsigned *status) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= width)
    return;
  bool used = false;
  if (*status == 0)
    for (std::uint32_t t = 0; t < tokens; ++t)
      if (std::uint64_t(ranges[t * 2]) <= std::uint64_t(begin) + i &&
          std::uint64_t(begin) + i < std::uint64_t(ranges[t * 2 + 1])) {
        used = true;
        break;
      }
  keys[i] = static_cast<unsigned char>(used);
}
__global__ void write_scores(const __nv_bfloat16 *scores,
                             const std::int32_t *ranges, std::uint32_t tokens,
                             std::uint32_t rows, std::uint32_t begin,
                             unsigned width, float *output,
                             const unsigned *status) {
  if (*status != 0)
    return;
  for (auto i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < std::uint64_t(tokens) * width;
       i += std::uint64_t(gridDim.x) * blockDim.x) {
    const auto key = i / tokens, t = i % tokens;
    const auto column = std::uint64_t(begin) + key;
    output[t * rows + column] =
        std::uint64_t(ranges[t * 2]) <= column &&
                column < std::uint64_t(ranges[t * 2 + 1])
            ? __bfloat162float(scores[i])
            : -CUDART_INF_F;
  }
}
unsigned blocks(std::uint64_t count) {
  return unsigned(count / kThreads + (count % kThreads != 0) > 65535
                      ? 65535
                      : count / kThreads + (count % kThreads != 0));
}
template <bool Masked, class Out>
__global__ void write_dense_scores(const __nv_bfloat16 *scores,
                                   const __half *mask, std::uint32_t tokens,
                                   std::uint32_t rows, std::uint32_t begin,
                                   unsigned width, Out *output) {
  for (auto i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < std::uint64_t(tokens) * width;
       i += std::uint64_t(gridDim.x) * blockDim.x) {
    const auto key = i / tokens, token = i % tokens;
    const auto index = token * rows + begin + key;
    auto rounded = scores[i]; // The head sum has already rounded to BF16.
    if constexpr (Masked) {
      // The public F16 mask Cast is an independent BF16 RNE boundary.
      const auto rounded_mask = __float2bfloat16_rn(__half2float(mask[index]));
      rounded = __float2bfloat16_rn(__bfloat162float(rounded) +
                                    __bfloat162float(rounded_mask));
    }
    if constexpr (std::is_same_v<Out, float>)
      output[index] = __bfloat162float(rounded);
    else
      output[index] = rounded;
  }
}
template <class Out>
cudaError_t prepare_dense_output(bool masked) {
  cudaFuncAttributes attributes{};
  return masked
      ? cudaFuncGetAttributes(&attributes, write_dense_scores<true, Out>)
      : cudaFuncGetAttributes(&attributes, write_dense_scores<false, Out>);
}
template <class Out>
void launch_dense_output(const void *scores, const void *mask,
                         std::uint32_t tokens, std::uint32_t rows,
                         std::uint32_t begin, unsigned width, void *output,
                         bool masked, cudaStream_t stream) {
  const auto grid = blocks(std::uint64_t(tokens) * width);
  if (masked)
    write_dense_scores<true, Out><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(scores),
        static_cast<const __half *>(mask), tokens, rows, begin, width,
        static_cast<Out *>(output));
  else
    write_dense_scores<false, Out><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(scores), nullptr,
        tokens, rows, begin, width, static_cast<Out *>(output));
}
} // namespace
cudaError_t prepare_dense(bool masked, bool float_output) noexcept {
  return float_output ? prepare_dense_output<float>(masked)
                      : prepare_dense_output<__nv_bfloat16>(masked);
}
cudaError_t publish_dense(const void *scores, const void *mask,
                          std::uint32_t tokens, std::uint32_t rows,
                          std::uint32_t begin, unsigned width, void *output,
                          bool masked, bool float_output,
                          cudaStream_t stream) noexcept {
  if (!tokens || !width)
    return cudaSuccess;
  if (float_output)
    launch_dense_output<float>(scores, mask, tokens, rows, begin, width,
                                output, masked, stream);
  else
    launch_dense_output<__nv_bfloat16>(scores, mask, tokens, rows, begin, width,
                                        output, masked, stream);
  return cudaGetLastError();
}
cudaError_t prepare() noexcept {
  cudaFuncAttributes attributes{};
  auto error = cudaFuncGetAttributes(&attributes, validate);
  if (error == cudaSuccess)
    error = cudaFuncGetAttributes(&attributes, active);
  if (error == cudaSuccess)
    error = cudaFuncGetAttributes(&attributes, write_scores);
  if (error == cudaSuccess)
    error = cudaFuncGetAttributes(&attributes, initialize);
  return error;
}
cudaError_t initialize_scores(float *output, std::uint64_t count,
                              const unsigned *status,
                              cudaStream_t stream) noexcept {
  if (!count)
    return cudaSuccess;
  initialize<<<blocks(count), kThreads, 0, stream>>>(output, count, status);
  return cudaGetLastError();
}
cudaError_t check_ranges(const std::int32_t *ranges, std::uint32_t tokens,
                         std::uint32_t rows, unsigned *status,
                         cudaStream_t stream) noexcept {
  if (!tokens)
    return cudaSuccess;
  validate<<<blocks(tokens), kThreads, 0, stream>>>(ranges, tokens, rows,
                                                    status);
  return cudaGetLastError();
}
cudaError_t active_keys(const std::int32_t *ranges, std::uint32_t tokens,
                        std::uint32_t begin, unsigned width,
                        unsigned char *keys, const unsigned *status,
                        cudaStream_t stream) noexcept {
  if (!width)
    return cudaSuccess;
  active<<<blocks(width), kThreads, 0, stream>>>(ranges, tokens, begin, width,
                                                 keys, status);
  return cudaGetLastError();
}
cudaError_t publish(const void *scores, const std::int32_t *ranges,
                    std::uint32_t tokens, std::uint32_t rows,
                    std::uint32_t begin, unsigned width, float *output,
                    const unsigned *status, cudaStream_t stream) noexcept {
  if (!tokens || !width)
    return cudaSuccess;
  write_scores<<<blocks(std::uint64_t(tokens) * width), kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(scores), ranges, tokens, rows, begin,
      width, output, status);
  return cudaGetLastError();
}
} // namespace hyle::compute::cuda_native::row_cache_indexer::native
