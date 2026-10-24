#include "index_guard.h"

#include <algorithm>

#include "device_status_kernel.h"

namespace hyle::compute::ggml_cuda::internal {
namespace {
constexpr unsigned kThreads = 256;
unsigned blocks(std::size_t count) noexcept {
  return static_cast<unsigned>(std::min<std::size_t>(
      (count - 1) / kThreads + 1, 65535));
}

__global__ void check(const std::int32_t *source, std::int32_t *safe,
                      std::size_t count, std::int64_t extent,
                      unsigned *status) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(gridDim.x) * blockDim.x) {
    const auto id = source[i];
    const bool valid = id >= 0 && std::int64_t(id) < extent;
    if (!valid)
      atomicCAS(status, 0U, static_cast<unsigned>(
          cuda::detail::DeviceStatusCode::kIndexOutOfRange));
    safe[i] = valid ? id : 0;
  }
}

__global__ void publish(const unsigned char *source, unsigned char *destination,
                        std::size_t bytes, const unsigned *status) {
  if (*status != 0)
    return;
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < bytes; i += std::size_t(gridDim.x) * blockDim.x)
    destination[i] = source[i];
}
} // namespace

cudaError_t check_indices(const std::int32_t *source, std::int32_t *safe,
                          std::size_t count, std::int64_t extent,
                          unsigned *status, cudaStream_t stream) noexcept {
  if (count == 0)
    return cudaSuccess;
  check<<<blocks(count), kThreads, 0, stream>>>(source, safe, count, extent, status);
  return cudaGetLastError();
}

cudaError_t publish_checked(const void *source, void *destination,
                            std::size_t bytes, const unsigned *status,
                            cudaStream_t stream) noexcept {
  if (bytes == 0)
    return cudaSuccess;
  publish<<<blocks(bytes), kThreads, 0, stream>>>(
      static_cast<const unsigned char *>(source),
      static_cast<unsigned char *>(destination), bytes, status);
  return cudaGetLastError();
}
} // namespace hyle::compute::ggml_cuda::internal
