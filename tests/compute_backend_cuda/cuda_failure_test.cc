#include <cstdint>

#include <gtest/gtest.h>

#include "hyle/compute/backends/cuda.h"

cudaError_t launch_illegal_address_kernel(cudaStream_t stream) noexcept;

namespace {

namespace cuda = hyle::compute::cuda;

// An asynchronous execution fault poisons the CUDA context, so this probe is
// deliberately isolated in its own process. It proves that a successful host
// enqueue can still become a typed terminal Completion failure without a
// hidden wait in submit().
TEST(CudaBackendLateFailure, SurfacesNativeFailureAtCompletionBoundary) {
  auto device = cuda::device(0);
  ASSERT_TRUE(device.has_value()) << device.error().message();
  auto context = hyle::compute::Context::create(*device);
  ASSERT_TRUE(context.has_value()) << context.error().message();
  auto queue = context->create_queue();
  ASSERT_TRUE(queue.has_value()) << queue.error().message();
  ASSERT_TRUE(queue->reserve_completions(1).ok());

  auto completion = cuda::submit(*queue, [](cudaStream_t stream) noexcept {
    return launch_illegal_address_kernel(stream);
  });
  ASSERT_TRUE(completion.has_value()) << completion.error().message();
  const hyle::Status terminal = completion->wait();
  EXPECT_FALSE(terminal.ok());
  EXPECT_EQ(terminal.native_error().domain,
            hyle::NativeErrorDomain::CudaRuntime);
  EXPECT_EQ(terminal.native_error().code,
            static_cast<std::int64_t>(cudaErrorIllegalAddress));
}

} // namespace
