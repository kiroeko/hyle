#include "../compute_stream/native_journey.h"
#include "hyle/compute/cuda.h"
#include "hyle/compute/backends/cuda.h"

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
} // namespace
