#include "../compute_stream/native_journey.h"
#include "hyle/compute/apple_memory.h"

namespace {
namespace f = file_stream_test;
namespace c = hyle::compute;
namespace a = c::apple;
struct Native final {
  a::Context context = a::Context::create().value();
  a::MemoryResource memory = a::MemoryResource::create(context).value();
  ~Native() { context.close().IgnoreError(); }
  c::FileResourcePipeline create(f::Host &runtime, unsigned slots = 2) {
    const std::array specs{c::CopyResourceSpec{memory.ref(), 1024}};
    return a::create_file_pipeline(runtime.io, context, memory.ref(), specs, runtime.options(slots)).value();
  }
};
TEST(AppleFileStream, OriginalMetalTwoGroupsTwoSlotsThreeReplaysAndFileWriteback) {
  Native native; f::Host host; auto pool = c::LeasePool::create(native.memory.ref(), 2048).value();
  auto pipeline = native.create(host); f::roundtrip(host, pipeline, pool);
  host.scheduler.shutdown(); EXPECT_TRUE(host.metadata.balanced());
}
TEST(AppleFileStream, ActualLaterFileEofKeepsEarlierMetalReady) {
  Native native; f::Host host; auto pool = c::LeasePool::create(native.memory.ref(), 2048).value();
  auto pipeline = native.create(host, 1); f::late_eof(host, pipeline, pool);
  host.scheduler.shutdown(); EXPECT_TRUE(host.metadata.balanced());
}
} // namespace
