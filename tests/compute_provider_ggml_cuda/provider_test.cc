#include "test_support.h"

#if defined(HYLE_GGML_CUDA_PROVIDER_FAULTS)
namespace {
thread_local bool fail_capture = false;
}
extern "C" cudaError_t __real_cudaStreamBeginCapture(cudaStream_t,
                                                     cudaStreamCaptureMode);
extern "C" cudaError_t
__wrap_cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) {
  if (fail_capture) {
    fail_capture = false;
    return cudaErrorStreamCaptureUnsupported;
  }
  return __real_cudaStreamBeginCapture(stream, mode);
}
#endif

namespace {
using namespace ggml_cuda_test;
TEST(GgmlCuda, IndependentInvocationsReplayAndRetainBorrowedPayload) {
  auto context = compute::Context::create(cuda::device(0).value(),
                                          compute::ggml_cuda::provider())
                     .value();
  exercise_direct(context);
}

#if defined(HYLE_GGML_CUDA_PROVIDER_FAULTS)
TEST(GgmlCuda, NativeCaptureFailureRetiresOnlyItsInvocation) {
  auto context = compute::Context::create(cuda::device(0).value(),
                                          compute::ggml_cuda::provider())
                     .value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  Tensor input(memory.ref()), output(memory.ref());
  const auto values = input.write(2);
  auto queue = context.create_queue().value();
  auto prepared = compute::prepare(context, compute::RmsNorm{},
                                   input.description, memory.ref())
                      .value();
  auto invocation = prepared.create_invocation().value();
  ASSERT_TRUE(invocation.submit(queue, input.view.as_const(), output.view)
                  ->wait()
                  .ok());
  fail_capture = true;
  auto failed = invocation.submit(queue, input.view.as_const(), output.view);
  ASSERT_TRUE(failed) << failed.error().message();
  auto terminal = failed->wait();
  EXPECT_FALSE(terminal.ok());
  EXPECT_EQ(terminal.native_error().code, cudaErrorStreamCaptureUnsupported);
  EXPECT_FALSE(fail_capture);
  auto reuse = invocation.submit(queue, input.view.as_const(), output.view);
  EXPECT_FALSE(reuse);
  auto recovered = prepared.create_invocation().value();
  auto done = recovered.submit(queue, input.view.as_const(), output.view);
  ASSERT_TRUE(done) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  check_norm(values, output.read(), false);
}
#endif
} // namespace
