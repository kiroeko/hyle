#include "../compute_stream/native_journey.h"
#include "hyle/compute/apple_memory.h"

namespace {
namespace f = file_stream_test;
namespace c = hyle::compute;
namespace a = c::apple;
struct Native final {
  a::Context context = f::checked(a::Context::create(), "Apple Context create");
  a::MemoryResource memory = f::checked(a::MemoryResource::create(context), "Apple memory create");
  ~Native() { context.close().IgnoreError(); }
  hyle::Result<c::FileResourcePipeline> create(f::Host &runtime, unsigned slots = 2) {
    const std::array specs{c::CopyResourceSpec{memory.ref(), 1024}};
    return a::create_file_pipeline(runtime.io, context, memory.ref(), specs, runtime.options(slots));
  }
};
TEST(AppleFileStream, OriginalMetalTwoGroupsTwoSlotsThreeReplaysAndFileWriteback) {
  Native native; f::Host host; auto pool = f::checked(c::LeasePool::create(native.memory.ref(), 2048), "Apple target lease pool");
  auto created = native.create(host); char failure[512]{};
  if (!created) created.error().ToString(failure, sizeof(failure));
  ASSERT_TRUE(created.has_value()) << failure;
  auto pipeline = std::move(*created); f::roundtrip(host, pipeline, pool);
  host.scheduler.shutdown(); EXPECT_TRUE(host.metadata.balanced());
}
TEST(AppleFileStream, ActualLaterFileEofKeepsEarlierMetalReady) {
  Native native; f::Host host; auto pool = f::checked(c::LeasePool::create(native.memory.ref(), 2048), "Apple target lease pool");
  auto created = native.create(host, 1); char failure[512]{};
  if (!created) created.error().ToString(failure, sizeof(failure));
  ASSERT_TRUE(created.has_value()) << failure;
  auto pipeline = std::move(*created); f::late_eof(host, pipeline, pool);
  host.scheduler.shutdown(); EXPECT_TRUE(host.metadata.balanced());
}
} // namespace
