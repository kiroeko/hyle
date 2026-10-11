#include "../compute_stream/native_journey.h"
#include "hyle/compute/apple_memory.h"
#include "apple_copy_internal.h"
#include "hyle/compute/detail/retained_completion.h"
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <mlx/ops.h>
#include <mlx/primitives.h>

namespace {
namespace f = file_stream_test;
namespace c = hyle::compute;
namespace a = c::apple;
namespace mx = mlx::core;
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
struct Gate final {
  std::mutex mutex; std::condition_variable condition;
  bool armed = false, entered = false, released = false, fail = false;
  void wait() {
    std::unique_lock lock(mutex); if (!armed) return;
    entered = true; condition.notify_all(); condition.wait(lock, [&] { return released; });
    if (fail) throw std::runtime_error("injected original MLX prefix failure for F3");
  }
  void release() { std::lock_guard lock(mutex); released = true; condition.notify_all(); }
};
struct ReleaseGate final {
  std::shared_ptr<Gate> gate; a::Context &context;
  ~ReleaseGate() { gate->release(); context.close().IgnoreError(); }
};
class Checkpoint final : public mx::Primitive {
public:
  Checkpoint(mx::Stream stream, std::shared_ptr<Gate> gate) : Primitive(stream), gate_(std::move(gate)) {}
  void eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) override { throw std::runtime_error("native GPU checkpoint cannot use CPU"); }
  void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override {
    gate_->wait(); outputs[0].copy_shared_buffer(inputs[0]);
  }
  const char *name() const override { return "HyleF3NativeCheckpoint"; }
private: std::shared_ptr<Gate> gate_;
};
void pending_producer(bool cancel, bool abandon, bool fail) {
  Native native; f::Host host; auto pool = c::LeasePool::create(native.memory.ref(), 2048).value();
  auto staging = hyle::mem::Buffer::create(native.memory.ref(), 256).value();
  ASSERT_TRUE(staging.fill(0, 256, std::byte{0x5d}).ok());
  auto writer = pool.allocate(256).value(); auto destination = writer.view().value();
  auto recording = a::detail::prepare_copy_recording(native.context, 1).value();
  const std::array<c::detail::CopyBackendRange, 1> copies{{{destination, staging.view()}}};
  ASSERT_TRUE(a::detail::bind_copy_recording(*recording, copies, true).ok());
  auto uses = c::LeaseUsePool::create().value(); auto use = uses.begin().value(); ASSERT_TRUE(use.write(writer).ok());
  auto gate = std::make_shared<Gate>(); ReleaseGate release{gate, native.context};
  const std::array specs{a::InputSpec{{4}, mx::float32}};
  auto program = native.context.compile([gate](const auto &args, mx::Stream stream) {
    auto value = mx::square(args[0], stream);
    return std::vector{mx::array(value.shape(), value.dtype(), std::make_shared<Checkpoint>(stream, gate), {value})};
  }, specs).value();
  auto invocation = program.create_invocation().value();
  const std::array values{mx::array({1.f, 2.f, 3.f, 4.f})}; ASSERT_TRUE(invocation.bind(values).ok());
  { std::lock_guard lock(gate->mutex); gate->armed = true; gate->fail = fail; }
  auto blocked = invocation.submit().value();
  { std::unique_lock lock(gate->mutex);
    ASSERT_TRUE(gate->condition.wait_for(lock, std::chrono::seconds(3), [&] { return gate->entered; })); }
  // An original Metal producer is accepted behind the actual blocked MLX job.
  auto original = a::detail::submit_copy_recording(*recording).value(); ASSERT_TRUE(use.attach(original).ok());
  auto retained = c::detail::RetainedCompletion::create(hyle::mem::system_allocator()).value();
  auto producer = retained.publish(std::move(original));
  auto sink = host.file({}, 256); auto pipeline = native.create(host, 1).value();
  std::array<c::StreamWriteRange, 1> ranges{c::StreamWriteRange::produced(std::move(writer), std::move(producer), 0, 0, 256)};
  std::array<c::StreamWriteGroup, 1> groups{{{81, ranges}}};
  auto ticket = pipeline.write(sink, groups).value();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (pipeline.report().active_slots == 0 && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
  EXPECT_EQ(pipeline.report().active_slots, 1u); EXPECT_FALSE(ticket.result().complete);
  EXPECT_EQ(ticket.result().written_bytes, 0u); EXPECT_FALSE(ticket.next_ready()); EXPECT_GT(pool.stats().active_uses, 0u);
  if (cancel) ticket.request_cancel();
  if (abandon) { ticket = {}; pipeline = {}; } else EXPECT_FALSE(pipeline.close().ok());
  EXPECT_GT(pool.stats().active_uses, 0u); gate->release(); auto prefix = blocked.wait();
  if (fail) { EXPECT_FALSE(prefix); if (!prefix) EXPECT_TRUE(prefix.error().IsInternal()); }
  else EXPECT_TRUE(prefix.has_value());
  if (!abandon) {
    auto result = ticket.wait();
    if (fail) { EXPECT_TRUE(result.status.IsInternal()); EXPECT_EQ(result.ready_groups, 0u); EXPECT_EQ(result.written_bytes, 0u); }
    else if (cancel) { EXPECT_TRUE(result.status.IsCancelled()); EXPECT_EQ(result.ready_groups, 0u); }
    else {
      ASSERT_TRUE(result.status.ok()); EXPECT_TRUE(result.retirement_proven); EXPECT_EQ(result.written_bytes, 256u);
      auto stored = hyle::platform::open_random_access_file(host.paths.back().c_str(), hyle::platform::kOpenRead).value();
      std::array<std::byte, 256> bytes; ASSERT_EQ(stored.read_at(0, bytes).value(), bytes.size());
      for (auto byte : bytes) EXPECT_EQ(byte, std::byte{0x5d});
    }
    pipeline.drain().IgnoreError(); ticket = {}; pipeline = {};
  }
  native.context.close().IgnoreError(); host.scheduler.shutdown();
  EXPECT_EQ(pool.stats().active_uses, 0u); EXPECT_EQ(uses.active_batches(), 0u); EXPECT_TRUE(host.metadata.balanced());
}
TEST(AppleFileStream, PendingOriginalMetalProducerOrdersFileReadback) { pending_producer(false, false, false); }
TEST(AppleFileStream, CancelledPendingOriginalProducerCannotPublishBeforeRetirement) { pending_producer(true, false, false); }
TEST(AppleFileStream, AbandonedControlsKeepOriginalPendingMetalProducerAlive) { pending_producer(true, true, false); }
TEST(AppleFileStream, LateOriginalNativeFailureCannotTurnPendingSourceIntoReady) { pending_producer(false, false, true); }
} // namespace
