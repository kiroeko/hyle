#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include <gtest/gtest.h>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/transfer.h"
#include "hyle/error/status.h"

cudaError_t launch_increment_kernel(cudaStream_t stream, void *data,
                                    std::size_t size) noexcept;
cudaError_t launch_delayed_store_kernel(cudaStream_t stream,
                                        unsigned int *destination,
                                        unsigned long long delay_clocks,
                                        unsigned int value) noexcept;

namespace {

using hyle::compute::CompletionPoll;
using hyle::compute::Context;
using hyle::compute::copy_async;
using hyle::compute::fill_async;
namespace cuda = hyle::compute::cuda;
namespace mem = hyle::mem;

using hyle::StatusCode;

TEST(CudaBackend, EnumeratesExplicitDevices) {
  const auto count = cuda::backend().device_count();
  ASSERT_TRUE(count.has_value()) << count.error().message();
  ASSERT_GT(*count, 0u);

  const auto device = cuda::backend().device(0);
  ASSERT_TRUE(device.has_value()) << device.error().message();
  EXPECT_EQ(device->backend_id(), cuda::kBackendId);
  EXPECT_EQ(device->backend_name(), "cuda");
  EXPECT_EQ(cuda::device_ordinal(*device).value(), 0);
  const auto properties = device->properties();
  ASSERT_TRUE(properties.has_value()) << properties.error().message();
  EXPECT_FALSE(properties->name.empty());
  EXPECT_TRUE(properties->asynchronous_queues);

  const auto invalid = cuda::device(static_cast<std::size_t>(-1));
  ASSERT_FALSE(invalid.has_value());
  EXPECT_EQ(invalid.error().code(), StatusCode::OutOfRange);

  const auto not_cuda = cuda::device_ordinal(hyle::compute::Device{});
  ASSERT_FALSE(not_cuda.has_value());
  EXPECT_EQ(not_cuda.error().code(), StatusCode::InvalidArgument);
}

TEST(CudaBackend, QueueTransferNativeSubmitAndMemoryRoundTrip) {
  auto device = cuda::device(0);
  ASSERT_TRUE(device.has_value()) << device.error().message();
  auto context = Context::create(*device);
  ASSERT_TRUE(context.has_value()) << context.error().message();
  auto queue = context->create_queue();
  ASSERT_TRUE(queue.has_value()) << queue.error().message();
  ASSERT_TRUE(queue->reserve_completions(4).ok());

  auto pinned =
      cuda::MemoryResource::create(*context, cuda::MemoryKind::kPinnedHost);
  auto resident =
      cuda::MemoryResource::create(*context, cuda::MemoryKind::kDevice);
  ASSERT_TRUE(pinned.has_value()) << pinned.error().message();
  ASSERT_TRUE(resident.has_value()) << resident.error().message();

  constexpr std::size_t kBytes = 4096;
  auto input = mem::Buffer::create(pinned->ref(), kBytes);
  auto device_buffer = mem::Buffer::create(resident->ref(), kBytes);
  auto output = mem::Buffer::create(pinned->ref(), kBytes);
  ASSERT_TRUE(input.has_value()) << input.error().message();
  ASSERT_TRUE(device_buffer.has_value()) << device_buffer.error().message();
  ASSERT_TRUE(output.has_value()) << output.error().message();

  std::array<std::byte, kBytes> expected{};
  for (std::size_t index = 0; index < expected.size(); ++index)
    expected[index] = static_cast<std::byte>((index * 29u + 7u) & 0xffu);
  ASSERT_TRUE(input->write(0, expected).ok());

  auto uploaded = copy_async(*queue, device_buffer->view(), input->view());
  ASSERT_TRUE(uploaded.has_value()) << uploaded.error().message();
  const auto first_poll = uploaded->poll();
  ASSERT_TRUE(first_poll.has_value()) << first_poll.error().message();
  EXPECT_TRUE(*first_poll == CompletionPoll::kPending ||
              *first_poll == CompletionPoll::kComplete);
  ASSERT_TRUE(uploaded->wait().ok());

  auto downloaded = copy_async(*queue, output->view(), device_buffer->view());
  ASSERT_TRUE(downloaded.has_value()) << downloaded.error().message();
  ASSERT_TRUE(downloaded->wait().ok());
  std::array<std::byte, kBytes> observed{};
  ASSERT_TRUE(output->read(0, observed).ok());
  EXPECT_EQ(observed, expected);

  auto resident_pointer = cuda::pointer<std::byte>(device_buffer->view());
  ASSERT_TRUE(resident_pointer.has_value())
      << resident_pointer.error().message();

  auto kernel = cuda::submit(*queue, [&](cudaStream_t stream) noexcept {
    return launch_increment_kernel(stream, *resident_pointer, kBytes);
  });
  ASSERT_TRUE(kernel.has_value()) << kernel.error().message();
  ASSERT_TRUE(kernel->wait().ok());
  auto kernel_download =
      copy_async(*queue, output->view(), device_buffer->view());
  ASSERT_TRUE(kernel_download.has_value()) << kernel_download.error().message();
  ASSERT_TRUE(kernel_download->wait().ok());
  ASSERT_TRUE(output->read(0, observed).ok());
  for (std::size_t index = 0; index < observed.size(); ++index) {
    EXPECT_EQ(observed[index],
              static_cast<std::byte>(
                  (std::to_integer<unsigned int>(expected[index]) + 1u) &
                  0xffu));
  }

  auto native = cuda::submit(*queue, [&](cudaStream_t stream) noexcept {
    return cudaMemsetAsync(*resident_pointer, 0x5a, kBytes, stream);
  });
  ASSERT_TRUE(native.has_value()) << native.error().message();
  ASSERT_TRUE(native->wait().ok());
  auto second_download =
      copy_async(*queue, output->view(), device_buffer->view());
  ASSERT_TRUE(second_download.has_value()) << second_download.error().message();
  ASSERT_TRUE(second_download->wait().ok());
  ASSERT_TRUE(output->read(0, observed).ok());
  for (std::byte value : observed)
    EXPECT_EQ(value, std::byte{0x5a});

  auto filled = fill_async(*queue, device_buffer->view(), std::byte{0xa5});
  ASSERT_TRUE(filled.has_value()) << filled.error().message();
  ASSERT_TRUE(filled->wait().ok());
  EXPECT_TRUE(queue->drain().ok());
}

TEST(CudaBackend, BorrowedStreamUsesExactContextAndIsNotDestroyed) {
  auto device = cuda::device(0);
  ASSERT_TRUE(device.has_value()) << device.error().message();
  auto context = Context::create(*device);
  ASSERT_TRUE(context.has_value()) << context.error().message();
  auto execution = cuda::execution_context(*context);
  ASSERT_TRUE(execution.has_value()) << execution.error().message();

  cudaStream_t external = nullptr;
  ASSERT_EQ(cudaExecutionCtxStreamCreate(&external, *execution,
                                         cudaStreamNonBlocking, 0),
            cudaSuccess);
  {
    auto queue = cuda::borrow_stream(*context, external);
    ASSERT_TRUE(queue.has_value()) << queue.error().message();
    EXPECT_EQ(cuda::stream(*queue).value(), external);
    auto completion = cuda::submit(*queue, [](cudaStream_t) noexcept {});
    ASSERT_TRUE(completion.has_value()) << completion.error().message();
    ASSERT_TRUE(completion->wait().ok());
  }
  EXPECT_EQ(cudaStreamQuery(external), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(external), cudaSuccess);
}

TEST(CudaBackend, StreamImportRejectsDefaultsAndBlockingStreamsWithoutSteal) {
  auto device = cuda::device(0);
  ASSERT_TRUE(device.has_value()) << device.error().message();
  auto context = Context::create(*device);
  ASSERT_TRUE(context.has_value()) << context.error().message();

  auto default_stream = cuda::borrow_stream(*context, nullptr);
  ASSERT_FALSE(default_stream.has_value());
  EXPECT_EQ(default_stream.error().code(), StatusCode::InvalidArgument);

  auto execution = cuda::execution_context(*context);
  ASSERT_TRUE(execution.has_value()) << execution.error().message();
  cudaStream_t blocking = nullptr;
  ASSERT_EQ(cudaExecutionCtxStreamCreate(&blocking, *execution,
                                         cudaStreamDefault, 0),
            cudaSuccess);
  auto rejected = cuda::adopt_stream(*context, blocking);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(rejected.error().code(), StatusCode::FailedPrecondition);
  // Failed adopt never transfers ownership.
  EXPECT_EQ(cudaStreamDestroy(blocking), cudaSuccess);
}

TEST(CudaBackend, AdoptedStreamIsReleasedAfterItsQueueAndCompletion) {
  auto device = cuda::device(0);
  ASSERT_TRUE(device.has_value()) << device.error().message();
  auto context = Context::create(*device);
  ASSERT_TRUE(context.has_value()) << context.error().message();
  auto execution = cuda::execution_context(*context);
  ASSERT_TRUE(execution.has_value()) << execution.error().message();

  cudaStream_t external = nullptr;
  ASSERT_EQ(cudaExecutionCtxStreamCreate(&external, *execution,
                                         cudaStreamNonBlocking, 0),
            cudaSuccess);
  {
    auto queue = cuda::adopt_stream(*context, external);
    ASSERT_TRUE(queue.has_value()) << queue.error().message();
    auto completion = cuda::submit(*queue, [](cudaStream_t) noexcept {});
    ASSERT_TRUE(completion.has_value()) << completion.error().message();
    ASSERT_TRUE(completion->wait().ok());
  }
  EXPECT_EQ(cudaStreamQuery(external), cudaErrorInvalidResourceHandle);
  (void)cudaGetLastError();
}

TEST(CudaBackend, MemoryKindsExposeTruthfulSyncAndTypedPointerSemantics) {
  auto device = cuda::device(0);
  ASSERT_TRUE(device.has_value()) << device.error().message();
  auto context = Context::create(*device);
  ASSERT_TRUE(context.has_value()) << context.error().message();

  constexpr std::size_t kBytes = 256;
  std::array<std::byte, kBytes> expected{};
  std::array<std::byte, kBytes> observed{};
  for (std::size_t index = 0; index < expected.size(); ++index)
    expected[index] = static_cast<std::byte>((index * 17u + 3u) & 0xffu);

  for (cuda::MemoryKind kind : {cuda::MemoryKind::kDevice,
                                cuda::MemoryKind::kPinnedHost,
                                cuda::MemoryKind::kManaged}) {
    auto resource = cuda::MemoryResource::create(*context, kind);
    ASSERT_TRUE(resource.has_value()) << resource.error().message();
    EXPECT_EQ(resource->kind(), kind);
    EXPECT_EQ(resource->device(), *device);
    auto buffer = mem::Buffer::create(resource->ref(), kBytes);
    ASSERT_TRUE(buffer.has_value()) << buffer.error().message();
    ASSERT_TRUE(buffer->write(0, expected).ok());
    ASSERT_TRUE(buffer->read(0, observed).ok());
    EXPECT_EQ(observed, expected);
    ASSERT_TRUE(buffer->fill(0, kBytes, std::byte{0x6d}).ok());
    ASSERT_TRUE(buffer->read(0, observed).ok());
    for (std::byte value : observed)
      EXPECT_EQ(value, std::byte{0x6d});

    auto pointer = cuda::pointer<std::uint32_t>(buffer->view());
    ASSERT_TRUE(pointer.has_value()) << pointer.error().message();
    auto misaligned = buffer->view().subview(1, sizeof(std::uint32_t));
    ASSERT_TRUE(misaligned.has_value()) << misaligned.error().message();
    auto rejected = cuda::pointer<std::uint32_t>(*misaligned);
    ASSERT_FALSE(rejected.has_value());
    EXPECT_EQ(rejected.error().code(), StatusCode::InvalidArgument);

    if (kind == cuda::MemoryKind::kDevice) {
      auto mapping = buffer->view().map_host();
      ASSERT_FALSE(mapping.has_value());
      EXPECT_EQ(mapping.error().code(), StatusCode::FailedPrecondition);
    } else {
      auto mapping = buffer->view().map_host();
      ASSERT_TRUE(mapping.has_value()) << mapping.error().message();
      ASSERT_TRUE(mapping->close().ok());
    }
  }
}

TEST(CudaBackend, ExternalBufferBorrowAndAdoptHaveDistinctOwnership) {
  auto device = cuda::device(0);
  ASSERT_TRUE(device.has_value()) << device.error().message();
  auto context = Context::create(*device);
  ASSERT_TRUE(context.has_value()) << context.error().message();
  auto primary = cuda::primary_context(*context);
  ASSERT_TRUE(primary.has_value()) << primary.error().message();
  ASSERT_EQ(cuCtxPushCurrent(*primary), CUDA_SUCCESS);

  constexpr std::size_t kBytes = 1024;
  CUdeviceptr borrowed_pointer = 0;
  ASSERT_EQ(cuMemAlloc(&borrowed_pointer, kBytes), CUDA_SUCCESS);
  {
    auto external = cuda::ExternalBuffer::borrow(
        *context, reinterpret_cast<void *>(borrowed_pointer), kBytes);
    ASSERT_TRUE(external.has_value()) << external.error().message();
    EXPECT_EQ(external->size(), kBytes);
    EXPECT_EQ(external->device(), *device);
  }
  EXPECT_EQ(cuMemFree(borrowed_pointer), CUDA_SUCCESS);

  CUdeviceptr adopted_pointer = 0;
  ASSERT_EQ(cuMemAlloc(&adopted_pointer, kBytes), CUDA_SUCCESS);
  {
    auto external = cuda::ExternalBuffer::adopt(
        *context, reinterpret_cast<void *>(adopted_pointer), kBytes);
    ASSERT_TRUE(external.has_value()) << external.error().message();
  }
  CUdeviceptr base = 0;
  std::size_t allocation_size = 0;
  EXPECT_NE(cuMemGetAddressRange(&base, &allocation_size, adopted_pointer),
            CUDA_SUCCESS);

  CUcontext popped = nullptr;
  ASSERT_EQ(cuCtxPopCurrent(&popped), CUDA_SUCCESS);
  EXPECT_EQ(popped, *primary);
}

TEST(CudaBackend, RejectsExternalCaptureWithoutInvokingCommand) {
  auto device = cuda::device(0);
  ASSERT_TRUE(device.has_value()) << device.error().message();
  auto context = Context::create(*device);
  ASSERT_TRUE(context.has_value()) << context.error().message();
  auto queue = context->create_queue();
  ASSERT_TRUE(queue.has_value()) << queue.error().message();
  auto native_stream = cuda::stream(*queue);
  ASSERT_TRUE(native_stream.has_value()) << native_stream.error().message();

  ASSERT_EQ(cudaStreamBeginCapture(*native_stream,
                                   cudaStreamCaptureModeThreadLocal),
            cudaSuccess);
  bool invoked = false;
  auto rejected = cuda::submit(*queue, [&](cudaStream_t) noexcept {
    invoked = true;
  });
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(rejected.error().code(), StatusCode::FailedPrecondition);
  EXPECT_FALSE(invoked);
  cudaGraph_t graph = nullptr;
  ASSERT_EQ(cudaStreamEndCapture(*native_stream, &graph), cudaSuccess);
  if (graph != nullptr)
    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
}

TEST(CudaBackend, AbandonedPendingCompletionRetainsQueueAndStorage) {
  auto device = cuda::device(0);
  ASSERT_TRUE(device.has_value()) << device.error().message();
  int clock_rate_khz = 0;
  ASSERT_EQ(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0),
            cudaSuccess);
  ASSERT_GT(clock_rate_khz, 0);
  auto context = Context::create(*device);
  ASSERT_TRUE(context.has_value()) << context.error().message();
  auto queue = context->create_queue();
  ASSERT_TRUE(queue.has_value()) << queue.error().message();
  ASSERT_TRUE(queue->reserve_completions(2).ok());

  auto resident =
      cuda::MemoryResource::create(*context, cuda::MemoryKind::kDevice);
  auto pinned =
      cuda::MemoryResource::create(*context, cuda::MemoryKind::kPinnedHost);
  ASSERT_TRUE(resident.has_value()) << resident.error().message();
  ASSERT_TRUE(pinned.has_value()) << pinned.error().message();
  auto device_word = mem::Buffer::create(resident->ref(), sizeof(unsigned int));
  auto host_word = mem::Buffer::create(pinned->ref(), sizeof(unsigned int));
  ASSERT_TRUE(device_word.has_value()) << device_word.error().message();
  ASSERT_TRUE(host_word.has_value()) << host_word.error().message();
  auto pointer = cuda::pointer<unsigned int>(device_word->view());
  ASSERT_TRUE(pointer.has_value()) << pointer.error().message();

  constexpr unsigned int kExpected = 0x51a7b00fu;
  const unsigned long long delay_clocks =
      static_cast<unsigned long long>(clock_rate_khz) * 4u;
  {
    auto pending = cuda::submit(*queue, [&](cudaStream_t stream) noexcept {
      return launch_delayed_store_kernel(stream, *pointer, delay_clocks,
                                         kExpected);
    });
    ASSERT_TRUE(pending.has_value()) << pending.error().message();
    const auto cancel = pending->request_cancel();
    EXPECT_EQ(cancel.code(), StatusCode::Unimplemented);
    // Destruction deliberately exercises the non-blocking abandonment path.
  }

  auto downloaded = copy_async(*queue, host_word->view(), device_word->view());
  ASSERT_TRUE(downloaded.has_value()) << downloaded.error().message();
  ASSERT_TRUE(downloaded->wait().ok());
  unsigned int observed = 0;
  ASSERT_TRUE(host_word->read(
      0, std::as_writable_bytes(std::span<unsigned int>(&observed, 1))).ok());
  EXPECT_EQ(observed, kExpected);
  EXPECT_TRUE(queue->drain().ok());
}

} // namespace
