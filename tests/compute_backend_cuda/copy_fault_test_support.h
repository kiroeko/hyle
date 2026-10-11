#pragma once

#include <utility>

#include <gtest/gtest.h>

#include "cuda_internal.h"

#if defined(HYLE_TESTING)
namespace copy_fault_test {
namespace d = hyle::compute::cuda::detail;
inline thread_local unsigned probes = 0, copies = 0, fail_copy = 0;
inline thread_local bool fail_query = false;

inline CUresult absent_batch(const char *name, void **entry, int version,
                            cuuint64_t flags, CUdriverProcAddressQueryResult *query) noexcept {
  ++probes;
  EXPECT_STREQ(name, "cuMemcpyBatchAsync");
  EXPECT_EQ(version, 13000);
  EXPECT_EQ(flags, CU_GET_PROC_ADDRESS_LEGACY_STREAM);
  const auto real = cuGetProcAddress(name, entry, version, flags, query);
  if (real != CUDA_SUCCESS && real != CUDA_ERROR_NOT_FOUND && real != CUDA_ERROR_NOT_SUPPORTED)
    return real;
  // Decorate only the capability verdict; all actual fallback copies remain
  // original driver work on the original non-default stream.
  *entry = nullptr;
  *query = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
  return CUDA_ERROR_NOT_FOUND;
}
inline cudaError_t observed_copy(void *destination, const void *source, std::size_t bytes,
                                 cudaMemcpyKind kind, cudaStream_t stream) noexcept {
  ++copies;
  const auto real = cudaMemcpyAsync(destination, source, bytes, kind, stream);
  if (real == cudaSuccess && fail_copy == copies)
    return cudaErrorMemoryAllocation; // Preserve the command already accepted.
  return real;
}
inline cudaError_t observed_query(cudaEvent_t event) noexcept {
  const auto real = cudaEventQuery(event);
  if (std::exchange(fail_query, false)) {
    EXPECT_EQ(real, cudaErrorNotReady);
    return cudaErrorMemoryAllocation;
  }
  return real;
}
struct NativeScope final {
  d::CopyNativeHooks previous;
  NativeScope() : previous(d::set_copy_native_hooks({&absent_batch, &observed_copy})) {
    probes = copies = fail_copy = 0;
  }
  ~NativeScope() { d::set_copy_native_hooks(previous); fail_copy = 0; }
};
struct ObservationScope final {
  d::CompletionObservationHooks previous;
  ObservationScope() : previous(d::set_completion_observation_hooks({&observed_query, nullptr})) {}
  ~ObservationScope() { d::set_completion_observation_hooks(previous); fail_query = false; }
};
} // namespace copy_fault_test
#endif
