#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/detail/generated_result.h"
#include "hyle/compute/detail/queue_sequence.h"
#include "hyle/compute/transfer.h"
#include "hyle/error/status.h"
#include "hyle/mem/stats.h"

cudaError_t launch_increment_kernel(cudaStream_t stream, void *data,
                                    std::size_t size) noexcept;
cudaError_t launch_delayed_store_kernel(cudaStream_t stream,
                                        unsigned int *destination,
                                        unsigned long long delay_clocks,
                                        unsigned int value) noexcept;
cudaError_t launch_delay_kernel(cudaStream_t stream,
                                unsigned long long delay_clocks) noexcept;

namespace {

using hyle::compute::CompletionPoll;
using hyle::compute::Context;
using hyle::compute::copy_async;
using hyle::compute::fill_async;
namespace cuda = hyle::compute::cuda;
namespace mem = hyle::mem;

using hyle::StatusCode;

struct HostStreamGate final {
  std::atomic<bool> entered{false};
  std::atomic<bool> released{false};
  std::atomic<bool> finished{false};

  ~HostStreamGate() {
    // Native stream/event completion remains the actual lifetime boundary.
    // Observe the callback's final access explicitly for host race detectors.
    if (entered.load(std::memory_order_acquire)) {
      EXPECT_TRUE(finished.load(std::memory_order_acquire));
    }
  }

  static void CUDART_CB wait(void *raw) {
    auto &gate = *static_cast<HostStreamGate *>(raw);
    gate.entered.store(true, std::memory_order_release);
    gate.entered.notify_all();
    gate.released.wait(false, std::memory_order_acquire);
    gate.finished.store(true, std::memory_order_release);
  }
};

struct OwnedCudaBuffer final {
  mem::Buffer buffer;
  std::shared_ptr<std::atomic<int>> destroyed;
  OwnedCudaBuffer(mem::Buffer value,
                  std::shared_ptr<std::atomic<int>> counter) noexcept
      : buffer(std::move(value)), destroyed(std::move(counter)) {}
  OwnedCudaBuffer(OwnedCudaBuffer &&other) noexcept
      : buffer(std::move(other.buffer)),
        destroyed(std::move(other.destroyed)) {}
  ~OwnedCudaBuffer() {
    if (destroyed) {
      buffer = {};
      destroyed->fetch_add(1, std::memory_order_release);
      destroyed->notify_all();
    }
  }
};

TEST(CudaBackend, OwnedKernelResultWaitAndAbandonmentPreserveNativeStorage) {
  using hyle::compute::detail::ResultSlot;
  auto context = Context::create(cuda::device(0).value()).value();
  auto queue = context.create_queue().value();
  ASSERT_TRUE(queue.reserve_completions(1).ok());
  auto resident =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  for (bool abandon : {false, true}) {
    auto destroyed = std::make_shared<std::atomic<int>>(0);
    HostStreamGate gate;
    struct ReleaseGate final {
      HostStreamGate &gate;
      Context &context;
      std::shared_ptr<std::atomic<int>> destroyed;
      bool has_owner = false;
      void release() noexcept {
        gate.released.store(true, std::memory_order_release);
        gate.released.notify_all();
      }
      ~ReleaseGate() {
        release();
        context.drain().IgnoreError();
        if (has_owner)
          destroyed->wait(0, std::memory_order_acquire);
      }
    } release{gate, context, destroyed};
    {
      auto slot =
          ResultSlot<OwnedCudaBuffer>::create(mem::system_allocator()).value();
      auto write = slot.begin().value();
      auto buffer = mem::Buffer::create(resident.ref(), 256).value();
      ASSERT_TRUE(buffer.view().fill(std::byte{41}).ok());
      auto pointer = cuda::pointer<std::byte>(buffer.view()).value();
      // Finish first-use kernel preparation before placing a blocking host
      // callback on this stream. The controlled gate covers result retirement.
      auto prepared = cuda::submit(queue, [&](cudaStream_t stream) noexcept {
        return launch_increment_kernel(stream, pointer, 256);
      });
      ASSERT_TRUE(prepared.has_value());
      ASSERT_TRUE(prepared->wait().ok());
      ASSERT_TRUE(buffer.view().fill(std::byte{41}).ok());
      ASSERT_TRUE(
          write.emplace(OwnedCudaBuffer(std::move(buffer), destroyed)).ok());
      release.has_owner = true;
      auto completion = cuda::submit(queue, [&](cudaStream_t stream) noexcept {
        const auto gated =
            cudaLaunchHostFunc(stream, &HostStreamGate::wait, &gate);
        if (gated != cudaSuccess)
          return gated;
        return launch_increment_kernel(stream, pointer, 256);
      });
      ASSERT_TRUE(completion.has_value());
      auto submitted = write.publish(std::move(*completion));
      ASSERT_TRUE(submitted.has_value());
      EXPECT_EQ(submitted->poll().value(), CompletionPoll::kPending);
      EXPECT_FALSE(slot.begin().has_value());
      if (!abandon) {
        release.release();
        auto ready = submitted->wait();
        ASSERT_TRUE(ready.has_value());
        std::array<std::byte, 256> observed{};
        ASSERT_TRUE((*ready)->buffer.read(0, observed).ok());
        for (auto value : observed)
          EXPECT_EQ(value, std::byte{42});
      }
    }
    EXPECT_EQ(destroyed->load(), abandon ? 0 : 1);
    release.release();
    ASSERT_TRUE(context.drain().ok());
    // Device drain and background owner retirement are distinct boundaries.
    // Await the owner's explicit signal before destroying its resource owner.
    destroyed->wait(0, std::memory_order_acquire);
    EXPECT_EQ(destroyed->load(), 1);
  }
}

TEST(CudaBackend, SynchronousDeviceFillAndCopyDoNotReturnBeforeCompletion) {
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  auto device = cuda::device(0).value();
  auto context = Context::create(device).value();
  auto resource =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  auto source = mem::Buffer::create(resource.ref(), 4096).value();
  auto destination = mem::Buffer::create(resource.ref(), 4096).value();
  ASSERT_TRUE(source.view().fill(std::byte{0x5a}).ok());
  // Fixture initialization must also work with the deliberately broken
  // baseline, so drain it explicitly before installing the controlled gate.
  ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);

  for (bool copy : {false, true}) {
    HostStreamGate gate;
    ASSERT_EQ(cudaLaunchHostFunc(nullptr, &HostStreamGate::wait, &gate),
              cudaSuccess);
    gate.entered.wait(false, std::memory_order_acquire);
    std::atomic<bool> started{false};
    std::atomic<bool> finished{false};
    hyle::Status result;
    std::thread worker([&] {
      started.store(true, std::memory_order_release);
      started.notify_all();
      result = copy ? destination.view().copy_from(source.view())
                    : destination.view().fill(std::byte{0x5a});
      finished.store(true, std::memory_order_release);
    });
    started.wait(false, std::memory_order_acquire);
    // No throughput assertion: a return while the prerequisite stream is
    // gated is a correctness violation, however fast or slow the GPU is.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const bool returned_while_gated = finished.load(std::memory_order_acquire);
    gate.released.store(true, std::memory_order_release);
    gate.released.notify_all();
    worker.join();
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
    EXPECT_FALSE(returned_while_gated) << (copy ? "device copy" : "device fill");
    ASSERT_TRUE(result.ok()) << result.message();
    std::array<std::byte, 4096> observed{};
    ASSERT_TRUE(destination.view().read(observed).ok());
    for (const auto value : observed)
      EXPECT_EQ(value, std::byte{0x5a});
  }
}

TEST(CudaBackend, PreservesPortableAndNativeErrorIdentity) {
  const auto runtime_oom = cuda::status(cudaErrorMemoryAllocation);
  EXPECT_EQ(runtime_oom.code(), StatusCode::ResourceExhausted);
  EXPECT_EQ(runtime_oom.native_error().domain,
            hyle::NativeErrorDomain::CudaRuntime);
  EXPECT_EQ(runtime_oom.native_error().code,
            static_cast<std::int64_t>(cudaErrorMemoryAllocation));

  const auto driver_oom = cuda::status(CUDA_ERROR_OUT_OF_MEMORY);
  EXPECT_EQ(driver_oom.code(), StatusCode::ResourceExhausted);
  EXPECT_EQ(driver_oom.native_error().domain,
            hyle::NativeErrorDomain::CudaDriver);
  EXPECT_EQ(driver_oom.native_error().code,
            static_cast<std::int64_t>(CUDA_ERROR_OUT_OF_MEMORY));

  const auto late_failure = cuda::status(cudaErrorIllegalAddress);
  EXPECT_FALSE(late_failure.ok());
  EXPECT_EQ(late_failure.native_error().domain,
            hyle::NativeErrorDomain::CudaRuntime);
  EXPECT_EQ(late_failure.native_error().code,
            static_cast<std::int64_t>(cudaErrorIllegalAddress));
}

TEST(CudaBackend, SynchronousResourceInitializationPrecedesIndependentQueue) {
  auto device = cuda::device(0).value();
  auto context = Context::create(device).value();
  auto queue = context.create_queue().value();
  ASSERT_TRUE(queue.reserve_completions(1).ok());
  auto resource =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  constexpr std::size_t kBytes = 4099;
  auto destination = mem::Buffer::create(resource.ref(), kBytes).value();
  auto source = mem::Buffer::create(resource.ref(), kBytes).value();
  auto host = mem::Buffer::create(mem::system_resource(), kBytes).value();
  auto pointer = cuda::pointer<std::byte>(destination.view()).value();
  std::array<std::byte, kBytes> expected{};
  std::array<std::byte, kBytes> observed{};
  for (int operation = 0; operation < 4; ++operation) {
    for (std::size_t i = 0; i < kBytes; ++i)
      expected[i] = operation == 1 ? std::byte{0x51}
                                   : static_cast<std::byte>((i * 17 + 3) & 255);
    switch (operation) {
    case 0:
      ASSERT_TRUE(destination.view().write(expected).ok());
      break;
    case 1:
      ASSERT_TRUE(destination.view().fill(std::byte{0x51}).ok());
      break;
    case 2:
      ASSERT_TRUE(source.view().write(expected).ok());
      ASSERT_TRUE(destination.view().copy_from(source.view()).ok());
      break;
    case 3:
      ASSERT_TRUE(host.view().write(expected).ok());
      ASSERT_TRUE(destination.view().copy_from(host.view()).ok());
      break;
    }
    // No caller-side default-stream wait: the resource methods are complete.
    auto done = cuda::submit(queue, [&](cudaStream_t stream) noexcept {
      return launch_increment_kernel(stream, pointer, kBytes);
    });
    ASSERT_TRUE(done.has_value()) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    ASSERT_TRUE(destination.view().read(observed).ok());
    for (std::size_t i = 0; i < kBytes; ++i)
      EXPECT_EQ(observed[i], static_cast<std::byte>(
                                 (std::to_integer<unsigned>(expected[i]) + 1) &
                                 255));
  }
}

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

TEST(CudaBackend, CompletionEndpointIdentifiesQueueNotContext) {
  using hyle::compute::detail::QueueSequence;
  auto device = cuda::device(0).value();
  auto context = Context::create(device).value();
  auto producer = context.create_queue().value();
  auto consumer = context.create_queue().value();
  const auto source = QueueSequence::completion_endpoint(producer).value();
  const auto destination = QueueSequence::completion_endpoint(consumer).value();
  EXPECT_TRUE(source.valid());
  EXPECT_TRUE(destination.valid());
  // Compatibility is not execution order: separate streams on one device
  // must still import an event before consuming the producer's writes.
  EXPECT_NE(source, destination);
  EXPECT_EQ(source, QueueSequence::completion_endpoint(producer).value());
  EXPECT_TRUE(QueueSequence::can_import_endpoint(consumer, source).value());
  auto moved = std::move(producer);
  EXPECT_EQ(source, QueueSequence::completion_endpoint(moved).value());
  EXPECT_TRUE(QueueSequence::can_import_endpoint(consumer, source).value());

  auto other_context = Context::create(device).value();
  auto other_queue = other_context.create_queue().value();
  EXPECT_FALSE(QueueSequence::can_import_endpoint(other_queue, source).value());
  EXPECT_FALSE(QueueSequence::can_import_endpoint(consumer, {}).value());
  const int foreign_authority = 0;
  EXPECT_FALSE(QueueSequence::can_import_endpoint(
                   consumer, {&foreign_authority, &foreign_authority})
                   .value());
}

TEST(CudaBackend, TwoQueuesImportACompletionWithoutAHostBarrier) {
  auto device = cuda::device(0);
  ASSERT_TRUE(device.has_value()) << device.error().message();
  int clock_rate_khz = 0;
  ASSERT_EQ(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0),
            cudaSuccess);
  ASSERT_GT(clock_rate_khz, 0);
  auto context = Context::create(*device);
  ASSERT_TRUE(context.has_value()) << context.error().message();
  auto producer = context->create_queue();
  auto consumer = context->create_queue();
  ASSERT_TRUE(producer.has_value()) << producer.error().message();
  ASSERT_TRUE(consumer.has_value()) << consumer.error().message();
  ASSERT_TRUE(producer->reserve_completions(1).ok());
  ASSERT_TRUE(consumer->reserve_completions(2).ok());

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

  constexpr unsigned int kExpected = 0x71c0ffeeu;
  const unsigned long long delay_clocks =
      static_cast<unsigned long long>(clock_rate_khz) * 4u;
  auto source = cuda::submit(*producer, [&](cudaStream_t stream) noexcept {
    return launch_delayed_store_kernel(stream, *pointer, delay_clocks,
                                       kExpected);
  });
  ASSERT_TRUE(source.has_value()) << source.error().message();

  using hyle::compute::detail::QueueDependencyRoute;
  using hyle::compute::detail::QueueSequence;
  auto route = QueueSequence::dependency_route(*consumer, *producer);
  ASSERT_TRUE(route.has_value()) << route.error().message();
  EXPECT_EQ(*route, QueueDependencyRoute::kDeviceWait);
  auto endpoint = QueueSequence::completion_endpoint(*producer);
  ASSERT_TRUE(endpoint.has_value()) << endpoint.error().message();
  auto importable = QueueSequence::can_import_endpoint(*consumer, *endpoint);
  ASSERT_TRUE(importable.has_value()) << importable.error().message();
  EXPECT_TRUE(*importable);

  auto sequence = QueueSequence::begin(*consumer);
  ASSERT_TRUE(sequence.has_value()) << sequence.error().message();
  auto waited = sequence->enqueue_wait(*source);
  ASSERT_TRUE(waited.status.ok()) << waited.status.message();
  EXPECT_TRUE(waited.accepted);
  sequence->record(std::move(waited));
  auto imported = sequence->finish();
  ASSERT_TRUE(imported.has_value()) << imported.error().message();

  // The host has not waited for either prefix. Queue ordering makes this copy
  // observe the producer store only if the imported event was recorded on the
  // consumer stream correctly.
  auto downloaded =
      copy_async(*consumer, host_word->view(), device_word->view());
  ASSERT_TRUE(downloaded.has_value()) << downloaded.error().message();
  ASSERT_TRUE(downloaded->wait().ok());
  unsigned int observed = 0;
  ASSERT_TRUE(host_word
                  ->read(0, std::as_writable_bytes(
                                std::span<unsigned int>(&observed, 1)))
                  .ok());
  EXPECT_EQ(observed, kExpected);
  EXPECT_TRUE(source->wait().ok());
  EXPECT_TRUE(imported->wait().ok());
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
  auto tracked = mem::StatsAllocator::create(mem::system_allocator());
  ASSERT_TRUE(tracked.has_value()) << tracked.error().message();
  auto &metadata = *tracked;
  auto device = cuda::device(0);
  ASSERT_TRUE(device.has_value()) << device.error().message();
  auto context = Context::create(*device, {.metadata_allocator = metadata.get()});
  ASSERT_TRUE(context.has_value()) << context.error().message();
  auto execution = cuda::execution_context(*context);
  ASSERT_TRUE(execution.has_value()) << execution.error().message();

  cudaStream_t external = nullptr;
  ASSERT_EQ(cudaExecutionCtxStreamCreate(&external, *execution,
                                         cudaStreamNonBlocking, 0),
            cudaSuccess);
  const auto before_queue = metadata.snapshot();
  hyle::compute::Completion completion;
  {
    auto queue = cuda::adopt_stream(*context, external);
    ASSERT_TRUE(queue.has_value()) << queue.error().message();
    auto submitted = cuda::submit(*queue, [](cudaStream_t stream) noexcept {
      return cuda::status(launch_delay_kernel(stream, 1000000));
    });
    ASSERT_TRUE(submitted.has_value()) << submitted.error().message();
    completion = std::move(*submitted);
  }
  // The Completion retains the adopted stream after the public Queue dies.
  // Querying it after releasing both owners would itself be CUDA API UB.
  EXPECT_GT(metadata.snapshot().live_bytes, before_queue.live_bytes);
  const auto ready = cudaStreamQuery(external);
  EXPECT_TRUE(ready == cudaSuccess || ready == cudaErrorNotReady);
  ASSERT_TRUE(completion.wait().ok());
  completion = {};
  EXPECT_EQ(metadata.snapshot().live_bytes, before_queue.live_bytes);
  EXPECT_EQ(metadata.snapshot().live_count, before_queue.live_count);
  *context = {};
  EXPECT_EQ(metadata.snapshot().live_bytes, 0u);
  EXPECT_EQ(metadata.snapshot().live_count, 0u);
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
  if (graph != nullptr) {
    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
  }
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

TEST(CudaBackend, ConcurrentContextsRetirePendingPrefixesWithoutRaces) {
  auto device = cuda::device(0);
  ASSERT_TRUE(device.has_value()) << device.error().message();
  int clock_rate_khz = 0;
  ASSERT_EQ(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0),
            cudaSuccess);
  ASSERT_GT(clock_rate_khz, 0);

  constexpr std::size_t kThreads = 4;
  constexpr std::size_t kIterations = 16;
  std::atomic<bool> failed{false};
  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (std::size_t thread = 0; thread < kThreads; ++thread) {
    workers.emplace_back([&, thread] {
      for (std::size_t iteration = 0; iteration < kIterations; ++iteration) {
        auto context = Context::create(*device);
        if (!context) {
          failed.store(true, std::memory_order_relaxed);
          return;
        }
        auto queue = context->create_queue();
        if (!queue || !queue->reserve_completions(2).ok()) {
          failed.store(true, std::memory_order_relaxed);
          return;
        }
        const auto delay = static_cast<unsigned long long>(clock_rate_khz);
        auto pending = cuda::submit(*queue, [&](cudaStream_t stream) noexcept {
          return launch_delay_kernel(stream, delay);
        });
        if (!pending) {
          failed.store(true, std::memory_order_relaxed);
          return;
        }
        if (((thread + iteration) & 1U) == 0U) {
          if (!pending->wait().ok()) {
            failed.store(true, std::memory_order_relaxed);
            return;
          }
        } else {
          // Exercise handoff to the one process-wide reaper while other
          // contexts are polling, waiting, draining and being destroyed.
          pending = hyle::compute::Completion{};
        }
        if (!queue->drain().ok()) {
          failed.store(true, std::memory_order_relaxed);
          return;
        }
      }
    });
  }
  for (auto &worker : workers)
    worker.join();
  EXPECT_FALSE(failed.load(std::memory_order_relaxed));
}

} // namespace
