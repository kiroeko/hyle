#pragma once

#include <cuda_runtime_api.h>

namespace hyle::compute::cuda::detail {
// Private same-build device protocol; never CUDA's thread-local last-error.
enum class DeviceStatusCode : unsigned {
  kOk = 0,
  kIndexOutOfRange = 1,
  kIntegerOverflow = 2,
};

const void *device_status_merge_kernel() noexcept;
cudaError_t merge_device_status(const void *kernel, const unsigned *source,
                                unsigned *destination,
                                cudaStream_t stream) noexcept;
} // namespace hyle::compute::cuda::detail
