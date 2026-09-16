#pragma once

#include <climits>

#include "native_context.h"
#include "native_kernel.h"

namespace hyle::compute::marlin::internal {

struct Hardware final {
  CUcontext primary = nullptr;
  int multiprocessors = 0;
  int shared_bytes = 0;
};

inline Result<Hardware> hardware(Device device, CUcontext primary) noexcept {
  if (device.backend_id() != cuda::kBackendId || primary == nullptr)
    return std::unexpected(
        Status::Unimplemented("Marlin requires the CUDA backend"));
  HYLE_ASSIGN_OR_RETURN(auto ordinal, cuda::device_ordinal(device));
  cuda::detail::ScopedPrimaryContext scope(primary);
  HYLE_RETURN_IF_ERROR(scope.status());
  int major = 0;
  HYLE_RETURN_IF_ERROR(cuda::status(cudaDeviceGetAttribute(
      &major, cudaDevAttrComputeCapabilityMajor, ordinal)));
  if (major < 8)
    return std::unexpected(
        Status::Unimplemented("Marlin requires SM80 or newer"));
  Hardware result{.primary = primary};
  HYLE_RETURN_IF_ERROR(cuda::status(cudaDeviceGetAttribute(
      &result.multiprocessors, cudaDevAttrMultiProcessorCount, ordinal)));
  HYLE_RETURN_IF_ERROR(cuda::status(cudaDeviceGetAttribute(
      &result.shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, ordinal)));
  return result;
}

inline Status weight_shape(std::int64_t channels, std::int64_t inner) noexcept {
  if (channels <= 0 || channels > INT32_MAX || channels % 64 != 0 ||
      inner <= 0 || inner > INT32_MAX || inner % 128 != 0)
    return Status::Unimplemented(
        "Marlin requires N%64=0, K%128=0 and positive int32 dimensions");
  // The borrowed kernels use 32-bit packed offsets and intermediate strides.
  // With M<=64, N>=64 and K>=128 this also bounds activation/output indexing.
  if (channels > INT32_MAX / inner)
    return Status::Unimplemented(
        "Marlin logical N*K exceeds the supported 32-bit indexing domain");
  return Status::Ok();
}

} // namespace hyle::compute::marlin::internal
