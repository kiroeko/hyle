#include <cstddef>

#include "../../src/compute_backend_cuda/device_status_kernel.h"
#include <cuda_runtime_api.h>

namespace {

__global__ void increment_kernel(unsigned char *data, std::size_t size) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < size)
    ++data[index];
}

} // namespace

cudaError_t launch_increment_kernel(cudaStream_t stream, void *data,
                                    std::size_t size) noexcept {
  constexpr unsigned int kBlockSize = 256;
  const auto blocks =
      static_cast<unsigned int>((size + kBlockSize - 1) / kBlockSize);
  increment_kernel<<<blocks, kBlockSize, 0, stream>>>(
      static_cast<unsigned char *>(data), size);
  return cudaPeekAtLastError();
}

namespace {

__global__ void delayed_store_kernel(unsigned int *destination,
                                     unsigned long long delay_clocks,
                                     unsigned int value) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  const unsigned long long begin = clock64();
  while (clock64() - begin < delay_clocks) {
  }
  *destination = value;
}

} // namespace

cudaError_t launch_delayed_store_kernel(cudaStream_t stream,
                                        unsigned int *destination,
                                        unsigned long long delay_clocks,
                                        unsigned int value) noexcept {
  delayed_store_kernel<<<1, 1, 0, stream>>>(destination, delay_clocks, value);
  return cudaPeekAtLastError();
}

namespace {

__global__ void delay_kernel(unsigned long long delay_clocks) {
  const unsigned long long begin = clock64();
  while (clock64() - begin < delay_clocks) {
  }
}

} // namespace

cudaError_t launch_delay_kernel(cudaStream_t stream,
                                unsigned long long delay_clocks) noexcept {
  delay_kernel<<<1, 1, 0, stream>>>(delay_clocks);
  return cudaPeekAtLastError();
}

namespace {
__global__ void validate_indices_kernel(const int *indices, std::size_t count,
                                        int extent, unsigned *status) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count && (indices[index] < 0 || indices[index] >= extent))
    atomicCAS(
        status, 0U,
        static_cast<unsigned>(
            hyle::compute::cuda::detail::DeviceStatusCode::kIndexOutOfRange));
}
__global__ void validate_scan_kernel(const int *values, std::size_t count,
                                     unsigned *status) {
  long long sum = 0;
  for (std::size_t index = 0; index < count; ++index) {
    sum += values[index];
    if (sum > 2147483647LL || sum < -2147483648LL) {
      atomicCAS(
          status, 0U,
          static_cast<unsigned>(
              hyle::compute::cuda::detail::DeviceStatusCode::kIntegerOverflow));
      return;
    }
  }
}
} // namespace

cudaError_t launch_index_validation(cudaStream_t stream, const int *indices,
                                    std::size_t count, int extent,
                                    unsigned *status) noexcept {
  if (count == 0)
    return cudaSuccess;
  validate_indices_kernel<<<static_cast<unsigned>((count + 127) / 128), 128, 0,
                            stream>>>(indices, count, extent, status);
  return cudaPeekAtLastError();
}
cudaError_t launch_scan_validation(cudaStream_t stream, const int *values,
                                   std::size_t count,
                                   unsigned *status) noexcept {
  validate_scan_kernel<<<1, 1, 0, stream>>>(values, count, status);
  return cudaPeekAtLastError();
}

cudaError_t prepare_validation_kernels() noexcept {
  cudaFuncAttributes attributes{};
  const auto first =
      cudaFuncGetAttributes(&attributes, validate_indices_kernel);
  if (first != cudaSuccess)
    return first;
  return cudaFuncGetAttributes(&attributes, validate_scan_kernel);
}
