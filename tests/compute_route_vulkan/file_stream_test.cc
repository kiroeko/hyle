#include "../compute_stream/native_journey.h"
#include "fixture.h"
#include "hyle/compute/vulkan_memory.h"
#include "vulkan_copy_internal.h"
#include "hyle/compute/detail/retained_completion.h"
#include <chrono>
#include <thread>

namespace {
namespace f = file_stream_test;
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace d = v::detail;
struct Native final {
  v::Context context = vulkan_test::context().value();
  v::MemoryResource host = v::MemoryResource::create(context, v::MemoryKind::kStaging).value();
  v::MemoryResource device = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  ~Native() { context.close().IgnoreError(); }
  c::FileResourcePipeline create(f::Host &runtime, unsigned slots = 2) {
    const std::array specs{c::CopyResourceSpec{device.ref(), 1024}};
    return v::create_file_pipeline(runtime.io, context, host.ref(), specs, runtime.options(slots)).value();
  }
};
TEST(VulkanFileStream, OriginalGpuTwoGroupsTwoSlotsThreeReplaysAndFileWriteback) {
  Native native; ASSERT_FALSE(native.context.environment().software_device);
  RecordProperty("device_name", native.context.environment().device_name);
  f::Host host; auto pool = c::LeasePool::create(native.device.ref(), 2048).value();
  auto pipeline = native.create(host); f::roundtrip(host, pipeline, pool);
  host.scheduler.shutdown(); EXPECT_TRUE(host.metadata.balanced());
}
TEST(VulkanFileStream, ActualLaterFileEofKeepsEarlierGpuReady) {
  Native native; ASSERT_FALSE(native.context.environment().software_device);
  f::Host host; auto pool = c::LeasePool::create(native.device.ref(), 2048).value();
  auto pipeline = native.create(host, 1); f::late_eof(host, pipeline, pool);
  host.scheduler.shutdown(); EXPECT_TRUE(host.metadata.balanced());
}
struct SubmitFailure final {
  d::Device &device; PFN_vkQueueSubmit original;
  std::atomic<unsigned> count{0}; unsigned fail;
  static inline SubmitFailure *active = nullptr;
  SubmitFailure(const v::Context &context, unsigned target)
      : device(*d::Access::runtime(context)->device), original(device.fn.QueueSubmit), fail(target) {
    active = this; device.fn.QueueSubmit = invoke;
  }
  ~SubmitFailure() { device.fn.QueueSubmit = original; active = nullptr; }
  static VKAPI_ATTR VkResult VKAPI_CALL invoke(VkQueue queue, std::uint32_t count, const VkSubmitInfo *info, VkFence fence) {
    if (++active->count == active->fail) return VK_ERROR_OUT_OF_HOST_MEMORY;
    return active->original(queue, count, info, fence);
  }
};
TEST(VulkanFileStream, RejectedNativeSuffixPreservesEarlierGroupAndActualFileProgress) {
  Native native; ASSERT_FALSE(native.context.environment().software_device);
  f::Host host; auto pool = c::LeasePool::create(native.device.ref(), 2048).value();
  auto pipeline = native.create(host, 1); const auto bytes = f::input();
  auto source = host.file(bytes, bytes.size()); auto ranges = f::ranges(pool); auto groups = f::groups(ranges);
  c::StreamTicket ticket; c::StreamReadyGroup early;
  {
    // Four 64-byte chunks retire the first group. An eligible upload uses
    // transfer submit plus compute handoff, otherwise one original submit.
    SubmitFailure failure(native.context, native.context.environment().distinct_transfer_queue ? 9 : 5);
    ticket = pipeline.read(c::StreamSource::file(source), groups).value();
    early = ticket.wait_next_ready().value(); auto result = ticket.wait();
    EXPECT_TRUE(result.status.IsResourceExhausted()); EXPECT_TRUE(result.retirement_proven);
    EXPECT_EQ(result.ready_groups, 1u); EXPECT_EQ(result.copied_bytes, 256u); EXPECT_EQ(result.read_bytes, 320u);
    EXPECT_FALSE(ticket.next_ready());
  }
  auto desc = c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 1>{64}).value();
  auto tensor = native.device.binding(early.output(0).value(), desc).value();
  std::array<std::byte, 256> output;
  ASSERT_TRUE(native.context.download(tensor, output).ok());
  EXPECT_TRUE(std::equal(output.begin(), output.end(), bytes.begin()));
  pipeline.drain().IgnoreError(); tensor = {}; early = {}; ticket = {}; pipeline = {};
  host.scheduler.shutdown(); EXPECT_EQ(pool.stats().active_uses, 0u); EXPECT_TRUE(host.metadata.balanced());
}
struct UnknownObservation final {
  d::Device &device; d::DeviceFunctions original;
  std::atomic<bool> armed{true}; std::atomic<unsigned> waits{0}, drains{0};
  static inline UnknownObservation *active = nullptr;
  explicit UnknownObservation(const v::Context &context)
      : device(*d::Access::runtime(context)->device), original(device.fn) {
    active = this; device.fn.GetFenceStatus = poll; device.fn.WaitForFences = wait; device.fn.QueueWaitIdle = drain;
  }
  ~UnknownObservation() { armed = false; device.fn = original; active = nullptr; }
  static VKAPI_ATTR VkResult VKAPI_CALL poll(VkDevice device, VkFence fence) {
    return active->armed ? VK_ERROR_OUT_OF_HOST_MEMORY : active->original.GetFenceStatus(device, fence);
  }
  static VKAPI_ATTR VkResult VKAPI_CALL wait(VkDevice device, std::uint32_t count,
      const VkFence *fences, VkBool32 all, std::uint64_t timeout) {
    ++active->waits;
    return active->armed ? VK_ERROR_OUT_OF_HOST_MEMORY : active->original.WaitForFences(device, count, fences, all, timeout);
  }
  static VKAPI_ATTR VkResult VKAPI_CALL drain(VkQueue queue) {
    ++active->drains;
    return active->armed ? VK_ERROR_OUT_OF_HOST_MEMORY : active->original.QueueWaitIdle(queue);
  }
};
TEST(VulkanFileStream, UnknownActualNativeRetirementCannotPublishOrFreeCancelledAndAbandonedControl) {
  for (bool abandon : {false, true}) {
    SCOPED_TRACE(abandon); Native native; ASSERT_FALSE(native.context.environment().software_device);
    f::Host host; auto pool = c::LeasePool::create(native.device.ref(), 2048).value();
    auto pipeline = native.create(host, 1); const auto bytes = f::input(); auto source = host.file(bytes, bytes.size());
    auto ranges = f::ranges(pool); auto groups = f::groups(ranges);
    UnknownObservation boundary(native.context);
    struct Release {
      UnknownObservation &boundary; c::FileResourcePipeline &pipeline; f::Host &host; v::Context &context;
      ~Release() {
        boundary.armed = false; pipeline.drain().IgnoreError(); host.scheduler.shutdown(); context.close().IgnoreError();
      }
    } release{boundary, pipeline, host, native.context};
    auto ticket = pipeline.read(c::StreamSource::file(source), groups).value();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (boundary.drains.load() == 0 && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    EXPECT_GT(boundary.waits.load(), 0u); EXPECT_GT(boundary.drains.load(), 0u);
    EXPECT_FALSE(ticket.result().complete); EXPECT_FALSE(ticket.next_ready()); EXPECT_GT(pool.stats().active_uses, 0u);
    ticket.request_cancel(); EXPECT_FALSE(pipeline.close().ok());
    if (abandon) { ticket = {}; pipeline = {}; }
    EXPECT_GT(pool.stats().active_uses, 0u); EXPECT_FALSE(host.metadata.balanced());
    boundary.armed = false;
    if (!abandon) {
      auto result = ticket.wait(); EXPECT_TRUE(result.status.IsResourceExhausted());
      EXPECT_TRUE(result.retirement_proven); EXPECT_EQ(result.ready_groups, 0u); EXPECT_EQ(result.copied_bytes, 0u);
      pipeline.drain().IgnoreError(); ticket = {}; pipeline = {};
    }
    host.scheduler.shutdown(); native.context.close().IgnoreError();
    EXPECT_EQ(pool.stats().active_uses, 0u); EXPECT_TRUE(host.metadata.balanced());
  }
}
TEST(VulkanFileStream, OriginalNativeProducerOrdersResourceReadbackAndFileBytes) {
  Native native; ASSERT_FALSE(native.context.environment().software_device);
  f::Host host; auto pool = c::LeasePool::create(native.device.ref(), 2048).value();
  auto staging = hyle::mem::Buffer::create(native.host.ref(), 256).value();
  ASSERT_TRUE(staging.fill(0, 256, std::byte{0x5d}).ok());
  auto writer = pool.allocate(256).value(); auto destination = writer.view().value();
  auto recording = d::prepare_copy_recording(native.context, 1).value();
  const std::array<c::detail::CopyBackendRange, 1> copies{{{destination, staging.view()}}};
  ASSERT_TRUE(d::bind_copy_recording(*recording, copies, true).ok());
  auto uses = c::LeaseUsePool::create().value(); auto use = uses.begin().value(); ASSERT_TRUE(use.write(writer).ok());
  auto original = d::submit_copy_recording(*recording).value(); ASSERT_TRUE(use.attach(original).ok());
  auto retained = c::detail::RetainedCompletion::create(hyle::mem::system_allocator()).value();
  auto producer = retained.publish(std::move(original));
  auto sink = host.file({}, 256); auto pipeline = native.create(host, 1);
  std::array<c::StreamWriteRange, 1> ranges{c::StreamWriteRange::produced(std::move(writer), std::move(producer), 0, 0, 256)};
  std::array<c::StreamWriteGroup, 1> groups{{{81, ranges}}};
  auto ticket = pipeline.write(sink, groups).value(); ASSERT_TRUE(ticket.wait().status.ok());
  auto stored = hyle::platform::open_random_access_file(host.paths.back().c_str(), hyle::platform::kOpenRead).value();
  std::array<std::byte, 256> output; ASSERT_EQ(stored.read_at(0, output).value(), output.size());
  for (auto byte : output) EXPECT_EQ(byte, std::byte{0x5d});
  ticket = {}; pipeline.drain().IgnoreError(); pipeline = {}; host.scheduler.shutdown();
  EXPECT_EQ(uses.active_batches(), 0u); EXPECT_TRUE(retained.available()); EXPECT_TRUE(host.metadata.balanced());
}
} // namespace
