#include "test_support.h"
#include "../compute_provider_flashinfer/attention_test_support.h"

#include <atomic>
#include <future>
#include <thread>

#include "hyle/compute/detail/command_capture.h"
#include "hyle/compute/detail/queue_sequence.h"

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

TEST(GgmlCuda, OriginalCaptureRetirementDoesNotBorrowALaterCaptureOwner) {
  namespace cd = compute::detail;
  auto context = compute::Context::create(cuda::device(0).value(),
                                          compute::ggml_cuda::provider()).value();
  auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  Tensor input(memory.ref()), first_output(memory.ref()), second_output(memory.ref());
  const auto values = input.write(3);
  auto queue = context.create_queue().value();
  auto prepared = compute::prepare(context, compute::RmsNorm{}, input.description,
                                    memory.ref()).value();
  auto first_workspace = workspace_buffer(prepared, memory.ref());
  auto second_workspace = workspace_buffer(prepared, memory.ref());
  auto first = prepared.create_invocation(first_workspace.view()).value();
  auto second = prepared.create_invocation(second_workspace.view()).value();
  auto extension = cd::CommandCaptureAccess::query(queue).value();
  const std::array inputs{input.view.as_const()};
  const std::array first_outputs{first_output.view}, second_outputs{second_output.view};
  for (bool abort_original : {false, true}) {
    SCOPED_TRACE(abort_original);
    auto recording = cd::QueueSequence::begin(queue).value();
    ASSERT_TRUE(cd::CommandCaptureAccess::begin(extension, recording).ok());
    recording.record(cd::OperationInvocationAccess::enqueue(first, recording, inputs, first_outputs));
    recording.record(cd::OperationInvocationAccess::enqueue(second, recording, inputs, second_outputs));
    const auto encoded = recording.terminal();
    cd::CapturedCommands captured;
    hyle::Status ended;
    if (abort_original || !encoded.ok()) {
      ended = cd::CommandCaptureAccess::abort(extension, recording);
    } else {
      auto result = cd::CommandCaptureAccess::end(extension, recording, hyle::mem::system_allocator());
      if (result) captured = std::move(*result);
      else ended = result.error();
    }
    auto original = recording.finish().value();
    if (!encoded.ok() || !ended.ok()) {
      original.wait().IgnoreError();
      FAIL() << encoded.message() << "; " << ended.message();
    }
    // Complete the real root event without observing the retained native nodes.
    ASSERT_EQ(cudaStreamSynchronize(cuda::stream(queue).value()), cudaSuccess);
    std::promise<hyle::Status> started;
    auto active = started.get_future();
    std::promise<void> release;
    auto finish_later = release.get_future();
    hyle::Status later_status;
    std::thread later([&] {
      auto sequence = cd::QueueSequence::begin(queue);
      if (!sequence) { started.set_value(sequence.error()); return; }
      auto begun = cd::CommandCaptureAccess::begin(extension, *sequence);
      started.set_value(begun);
      if (!begun.ok()) return;
      finish_later.wait();
      later_status = cd::CommandCaptureAccess::abort(extension, *sequence);
      auto done = sequence->finish();
      if (done) later_status.Update(done->wait());
      else later_status.Update(done.error());
    });
    const auto begun = active.get();
    auto pending = original.poll();
    release.set_value();
    later.join(); // Never leave a test assertion while the other owner is blocked.
    const auto original_status = original.wait();
    ASSERT_TRUE(begun.ok()) << begun.message();
    ASSERT_TRUE(pending) << pending.error().message();
    EXPECT_EQ(*pending, compute::CompletionPoll::kPending);
    ASSERT_TRUE(later_status.ok()) << later_status.message();
    ASSERT_TRUE(original_status.ok()) << original_status.message();
    if (captured.valid()) {
      auto replay = cd::QueueSequence::begin(queue).value();
      replay.record(captured.enqueue(replay));
      ASSERT_TRUE(replay.finish()->wait().ok());
      check_norm(values, first_output.read(), false);
      check_norm(values, second_output.read(), false);
    } else {
      ASSERT_TRUE(first.submit(queue, input.view.as_const(), first_output.view)->wait().ok());
      check_norm(values, first_output.read(), false);
    }
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
  }
}

TEST(GgmlCuda, SameAttentionInvocationAlternatesQueuesAfterNativeWarmup) {
  auto context = compute::Context::create(cuda::device(0).value(),
                                          compute::ggml_cuda::provider()).value();
  auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  using AttentionTensor = flashinfer_test::Tensor;
  constexpr auto type = compute::DataType::kFloat32;
  AttentionTensor query(type, {1, 64, 2, 128}, memory.ref());
  AttentionTensor key(type, {1, 64, 2, 128}, memory.ref());
  AttentionTensor value(type, {1, 64, 2, 128}, memory.ref());
  AttentionTensor output(type, {1, 64, 2, 128}, memory.ref());
  const std::array inputs{query.view.as_const(), key.view.as_const(), value.view.as_const()};
  const std::array outputs{output.view};
  const compute::ScaledDotProductAttention operation{.scale = 0.0625F};
  auto prepared = compute::prepare(context, compute::Operation(operation), inputs, outputs, memory.ref());
  ASSERT_TRUE(prepared) << prepared.error().message();
  auto first_workspace =
      ggml_cuda_test::workspace_buffer(*prepared, memory.ref());
  auto first =
      prepared->create_invocation(first_workspace.view()).value();
  auto second_workspace =
      ggml_cuda_test::workspace_buffer(*prepared, memory.ref());
  auto second =
      prepared->create_invocation(second_workspace.view()).value();
  std::array queues{context.create_queue().value(), context.create_queue().value()};
  for (int round = 0; round != 8; ++round) {
    const auto q = flashinfer_test::payload<float>(query, round);
    const auto k = flashinfer_test::payload<float>(key, round + 5);
    const auto v = flashinfer_test::payload<float>(value, round + 13);
    for (auto *invocation : {&first, &second}) {
      auto done = invocation->submit(queues[static_cast<std::size_t>(round) % 2], inputs, outputs);
      ASSERT_TRUE(done) << done.error().message();
      ASSERT_TRUE(done->wait().ok());
      flashinfer_test::check_output(output, q, k, v, 64, 64, 1e-5,
                                    operation.scale, operation.causal);
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
    }
  }
}

TEST(GgmlCuda, PendingBorrowedStreamSurvivesAbandonAndClose) {
  auto context = compute::Context::create(cuda::device(0).value(),
                                          compute::ggml_cuda::provider()).value();
  auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  Tensor input(memory.ref()), output(memory.ref());
  const auto values = input.write(5);
  auto queue = context.create_queue().value();
  auto prepared = compute::prepare(context, compute::RmsNorm{}, input.description, memory.ref()).value();
  auto invocation_workspace =
      ggml_cuda_test::workspace_buffer(prepared, memory.ref());
  auto invocation =
      prepared.create_invocation(invocation_workspace.view()).value();
  for (int warm = 0; warm != 3; ++warm)
    ASSERT_TRUE(invocation.submit(queue, input.view.as_const(), output.view)->wait().ok());
  struct Gate final {
    compute::Context &context;
    std::atomic<bool> released{false};
    static void CUDART_CB wait(void *opaque) {
      static_cast<Gate *>(opaque)->released.wait(false, std::memory_order_acquire);
    }
    void release() {
      released.store(true, std::memory_order_release);
      released.notify_all();
    }
    ~Gate() { release(); context.drain().IgnoreError(); }
  } gate{context};
  auto blocked = cuda::submit(queue, [&](cudaStream_t stream) noexcept {
    return cudaLaunchHostFunc(stream, &Gate::wait, &gate);
  });
  ASSERT_TRUE(blocked);
  auto pending = invocation.submit(queue, input.view.as_const(), output.view);
  ASSERT_TRUE(pending) << pending.error().message();
  auto observed = pending->poll();
  ASSERT_TRUE(observed);
  EXPECT_EQ(*observed, compute::CompletionPoll::kPending);
  *pending = {}; // The original CUDA reaper retains the provider invocation.
  invocation = {};
  prepared = {};
  ASSERT_TRUE(queue.close().ok());
  ASSERT_TRUE(context.close().ok());
  gate.release();
  ASSERT_TRUE(blocked->wait().ok());
  ASSERT_TRUE(queue.drain().ok());
  check_norm(values, output.read(), false);
}

TEST(GgmlCuda, CallerCaptureIsRejectedBeforeNativeWorkAndSlotRemainsUsable) {
  auto context = compute::Context::create(cuda::device(0).value(),
                                          compute::ggml_cuda::provider()).value();
  auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  Tensor input(memory.ref()), output(memory.ref());
  const auto values = input.write(4);
  auto queue = context.create_queue().value();
  auto prepared = compute::prepare(context, compute::RmsNorm{}, input.description, memory.ref()).value();
  auto invocation_workspace =
      ggml_cuda_test::workspace_buffer(prepared, memory.ref());
  auto invocation =
      prepared.create_invocation(invocation_workspace.view()).value();
  ASSERT_TRUE(invocation.submit(queue, input.view.as_const(), output.view)->wait().ok());
  const auto stream = cuda::stream(queue).value();
  ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), cudaSuccess);
  auto rejected = invocation.submit(queue, input.view.as_const(), output.view);
  cudaGraph_t graph = nullptr;
  const auto ended = cudaStreamEndCapture(stream, &graph);
  if (graph != nullptr) EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
  ASSERT_EQ(ended, cudaSuccess);
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsFailedPrecondition());
  EXPECT_EQ(rejected.error().message(),
            "ggml CUDA capture requires the original recording sequence");
  auto done = invocation.submit(queue, input.view.as_const(), output.view);
  ASSERT_TRUE(done) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  check_norm(values, output.read(), false);
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
  auto invocation_workspace =
      ggml_cuda_test::workspace_buffer(prepared, memory.ref());
  auto invocation =
      prepared.create_invocation(invocation_workspace.view()).value();
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
  auto recovered_workspace =
      ggml_cuda_test::workspace_buffer(prepared, memory.ref());
  auto recovered =
      prepared.create_invocation(recovered_workspace.view()).value();
  auto done = recovered.submit(queue, input.view.as_const(), output.view);
  ASSERT_TRUE(done) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  check_norm(values, output.read(), false);
}
#endif
} // namespace
