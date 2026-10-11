#include "../compute_stream/native_journey.h"
#include "fixture.h"

namespace {
namespace f = file_stream_test;
namespace c = hyle::compute;
namespace v = c::vulkan;
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
} // namespace
