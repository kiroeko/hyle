#include "device_status_kernel.h"

#include <cuda_runtime.h>

namespace hyle::compute::cuda::detail {
namespace {
__global__ void merge_status(const unsigned *source, unsigned *destination) {
  const unsigned error = *source;
  if (error != 0)
    atomicCAS(destination, 0U, error);
}
} // namespace

const void *device_status_merge_kernel() noexcept {
  return reinterpret_cast<const void *>(merge_status);
}
cudaError_t merge_device_status(const void *kernel, const unsigned *source,
                                unsigned *destination,
                                cudaStream_t stream) noexcept {
  void *arguments[]{&source, &destination};
  return cudaLaunchKernel(kernel, dim3(1), dim3(1), arguments, 0, stream);
}
} // namespace hyle::compute::cuda::detail
