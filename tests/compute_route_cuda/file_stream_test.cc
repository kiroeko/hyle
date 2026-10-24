#include "../compute_stream/native_journey.h"
#include "hyle/compute/cuda.h"
#include "hyle/compute/cuda_file_stream.h"
#include "hyle/compute/backends/cuda.h"
#include <chrono>
#include <thread>

namespace {
namespace f = file_stream_test;
namespace c = hyle::compute;
namespace cu = c::cuda;
struct Native final {
  c::Context context = c::Context::create(cu::device(0).value()).value();
  c::Queue queue = context.create_queue().value();
  cu::MemoryResource host = cu::MemoryResource::create(context, cu::MemoryKind::kPinnedHost).value();
  cu::MemoryResource device = cu::MemoryResource::create(context, cu::MemoryKind::kDevice).value();
  ~Native() { context.drain().IgnoreError(); }
  c::FileResourcePipeline create(f::Host &runtime, unsigned slots = 2) {
    const std::array specs{c::CopyResourceSpec{device.ref(), 1024}};
    return cu::create_file_pipeline(runtime.io, context, queue, host.ref(), specs, runtime.options(slots)).value();
  }
};
TEST(CudaFileStream, OriginalGpuTwoGroupsTwoSlotsThreeReplaysAndFileWriteback) {
  Native native; f::Host host; auto pool = c::LeasePool::create(native.device.ref(), 2048).value();
  auto pipeline = native.create(host); f::roundtrip(host, pipeline, pool);
  host.scheduler.shutdown(); EXPECT_TRUE(host.metadata.balanced());
}
TEST(CudaFileStream, ActualLaterFileEofKeepsEarlierGpuReady) {
  Native native; f::Host host; auto pool = c::LeasePool::create(native.device.ref(), 2048).value();
  auto pipeline = native.create(host, 1); f::late_eof(host, pipeline, pool);
  host.scheduler.shutdown(); EXPECT_TRUE(host.metadata.balanced());
}
struct ProducerGate final {
  std::atomic<bool> open{false};
  static void CUDART_CB wait(void *opaque) {
    auto &gate = *static_cast<ProducerGate *>(opaque);
    for (bool ready = gate.open.load(std::memory_order_acquire); !ready;
         ready = gate.open.load(std::memory_order_acquire)) gate.open.wait(false, std::memory_order_acquire);
  }
  void release() noexcept { open.store(true, std::memory_order_release); open.notify_all(); }
};
struct ReleaseGate final {
  ProducerGate &gate; c::Context &context;
  ~ReleaseGate() { gate.release(); context.drain().IgnoreError(); }
};
void pending_producer(bool cancel, bool abandon) {
  Native native; f::Host host;
  auto pool = c::LeasePool::create(native.device.ref(), 2048).value();
  auto writer = pool.allocate(256).value(); auto pointer = cu::pointer<std::byte>(writer.view().value()).value();
  auto producer_queue = native.context.create_queue().value();
  auto uses = c::LeaseUsePool::create().value(); auto use = uses.begin().value();
  ASSERT_TRUE(use.write(writer).ok());
  ProducerGate gate; ReleaseGate release{gate, native.context};
  auto producer = cu::submit(producer_queue, [&gate, pointer](cudaStream_t stream) noexcept -> cudaError_t {
    auto blocked = cudaLaunchHostFunc(stream, &ProducerGate::wait, &gate);
    return blocked == cudaSuccess ? cudaMemsetAsync(pointer, 0x5d, 256, stream) : blocked;
  }).value();
  ASSERT_TRUE(use.attach(producer).ok());
  auto sink = host.file({}, 256);
  auto pipeline = native.create(host, 1);
  std::array<c::StreamWriteRange, 1> ranges{c::StreamWriteRange::produced(std::move(writer), std::move(producer), 0, 0, 256)};
  std::array<c::StreamWriteGroup, 1> groups{{{81, ranges}}};
  auto ticket = pipeline.write(sink, groups).value();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (pipeline.report().active_slots == 0 && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
  ASSERT_EQ(pipeline.report().active_slots, 1u);
  EXPECT_FALSE(ticket.result().complete); EXPECT_EQ(ticket.result().written_bytes, 0u);
  EXPECT_FALSE(ticket.next_ready()); EXPECT_GT(pool.stats().active_uses, 0u);
  if (cancel) ticket.request_cancel();
  if (abandon) {
    std::fprintf(stderr, "F3 abandon: dropping ticket\n"); std::fflush(stderr); ticket = {};
    std::fprintf(stderr, "F3 abandon: dropping pipeline\n"); std::fflush(stderr); pipeline = {};
    std::fprintf(stderr, "F3 abandon: controls dropped\n"); std::fflush(stderr);
  }
  else EXPECT_FALSE(pipeline.close().ok());
  EXPECT_GT(pool.stats().active_uses, 0u); // requesting cancel/drop is not GPU completion
  gate.release();
  if (abandon) { std::fprintf(stderr, "F3 abandon: gate released, draining Context\n"); std::fflush(stderr); }
  ASSERT_TRUE(native.context.drain().ok());
  if (abandon) { std::fprintf(stderr, "F3 abandon: Context drained, shutting Scheduler down\n"); std::fflush(stderr); }
  if (!abandon) {
    auto finished = ticket.wait();
    if (cancel) { EXPECT_TRUE(finished.status.IsCancelled()); EXPECT_EQ(finished.ready_groups, 0u); }
    else {
      ASSERT_TRUE(finished.status.ok()); EXPECT_EQ(finished.written_bytes, 256u);
      auto stored = hyle::platform::open_random_access_file(host.paths.back().c_str(), hyle::platform::kOpenRead).value();
      std::array<std::byte, 256> bytes; ASSERT_EQ(stored.read_at(0, bytes).value(), bytes.size());
      for (auto byte : bytes) EXPECT_EQ(byte, std::byte{0x5d});
    }
    pipeline.drain().IgnoreError(); ticket = {}; pipeline = {};
  }
  host.scheduler.shutdown();
  if (abandon) { std::fprintf(stderr, "F3 abandon: Scheduler stopped, checking native guards\n"); std::fflush(stderr); }
  // Original CUDA stream drain does not join its abandoned-event reaper.
  // Keep all borrowed resources/allocators alive until the actual guard drops.
  const auto retired = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while ((pool.stats().active_uses != 0 || !host.metadata.balanced()) && std::chrono::steady_clock::now() < retired)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  EXPECT_EQ(pool.stats().active_uses, 0u); EXPECT_TRUE(host.metadata.balanced());
}
TEST(CudaFileStream, PendingOriginalProducerOrdersReadbackBeforeAnyFileWrite) { pending_producer(false, false); }
TEST(CudaFileStream, CancelledPendingProducerRetiresAcceptedGpuUseWithoutReady) { pending_producer(true, false); }
TEST(CudaFileStream, AbandonedTicketAndPipelineRetainOriginalPendingGpuUse) { pending_producer(true, true); }
TEST(CudaFileStream, FailedProducerNeverPublishesNormalOrEmptyWriteGroup) {
  for (auto bytes : {std::size_t{256}, std::size_t{0}}) {
    Native native; f::Host host;
    auto pool = c::LeasePool::create(native.device.ref(), 2048).value();
    auto writer = pool.allocate(256).value(); auto sink = host.file({}, 256);
    auto pipeline = native.create(host, 1);
    std::array<c::StreamWriteRange, 1> ranges{c::StreamWriteRange::produced(std::move(writer),
        c::Completion::completed(hyle::Status::Internal("original producer failed")), 0, 0, bytes)};
    std::array<c::StreamWriteGroup, 1> groups{{{91, ranges}}};
    auto admitted = pipeline.write(sink, groups);
    if (admitted) {
      auto ticket = std::move(*admitted); auto result = ticket.wait();
      EXPECT_TRUE(result.status.IsInternal()); EXPECT_EQ(result.ready_groups, 0u);
      EXPECT_EQ(result.written_bytes, 0u); EXPECT_FALSE(ticket.next_ready());
    } else { EXPECT_TRUE(admitted.error().IsInternal()); EXPECT_TRUE(ranges[0].pending_source.valid()); }
    pipeline.drain().IgnoreError(); pipeline = {}; host.scheduler.shutdown(); EXPECT_TRUE(host.metadata.balanced());
  }
}
TEST(CudaFileStream, FailedFirstGroupRetiresAdmittedProducerOfUnstartedSuffix) {
  Native native; f::Host host;
  auto pool = c::LeasePool::create(native.device.ref(), 2048).value();
  auto writer = pool.allocate(256).value(); auto pointer = cu::pointer<std::byte>(writer.view().value()).value();
  auto producer_queue = native.context.create_queue().value();
  auto uses = c::LeaseUsePool::create().value(); auto use = uses.begin().value();
  ASSERT_TRUE(use.write(writer).ok());
  ProducerGate gate; ReleaseGate release{gate, native.context};
  auto producer = cu::submit(producer_queue, [&gate, pointer](cudaStream_t stream) noexcept -> cudaError_t {
    auto blocked = cudaLaunchHostFunc(stream, &ProducerGate::wait, &gate);
    return blocked == cudaSuccess ? cudaMemsetAsync(pointer, 0x5d, 256, stream) : blocked;
  }).value();
  ASSERT_TRUE(use.attach(producer).ok());
  auto sink = host.file({}, 512);
  auto readonly = hyle::platform::open_file(host.paths.back().c_str(), hyle::platform::kOpenRead).value();
  auto imported = hyle::io::File::adopt(host.io, 0, std::move(readonly)).value();
  auto readonly_sink = c::StreamFile::adopt(std::move(imported), 512).value();
  auto ready_writer = pool.allocate(256).value(); auto ready = ready_writer.publish().value();
  std::array<c::StreamWriteRange, 2> ranges{
      c::StreamWriteRange::ready(ready, 0, 0, 256),
      c::StreamWriteRange::produced(std::move(writer), std::move(producer), 256, 0, 256)};
  std::array<c::StreamWriteGroup, 2> groups{{{101, std::span(ranges).first(1)}, {102, std::span(ranges).subspan(1)}}};
  auto pipeline = native.create(host, 1); auto ticket = pipeline.write(readonly_sink, groups).value();
  auto result = ticket.wait(); EXPECT_FALSE(result.status.ok()); EXPECT_EQ(result.ready_groups, 0u);
  EXPECT_EQ(result.written_bytes, 0u); EXPECT_FALSE(result.retirement_proven);
  EXPECT_GT(pool.stats().active_uses, 0u);
  ticket = {}; pipeline = {}; gate.release(); ASSERT_TRUE(native.context.drain().ok());
  host.scheduler.shutdown();
  const auto retired = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while ((pool.stats().active_uses != 0 || !host.metadata.balanced()) && std::chrono::steady_clock::now() < retired)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  EXPECT_EQ(pool.stats().active_uses, 0u); EXPECT_TRUE(host.metadata.balanced());
}
} // namespace
