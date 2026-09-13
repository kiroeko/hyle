#include <cstddef>

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
