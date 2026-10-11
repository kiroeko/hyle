#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <mlx/ops.h>
#include <mlx/primitives.h>

#include "apple_copy_internal.h"
#include "hyle/compute/apple_memory.h"
#include "hyle/compute/detail/retained_completion.h"
#include "hyle/mem/quota_allocator.h"

namespace {
namespace c = hyle::compute;
namespace a = c::apple;
namespace d = a::detail;
namespace mx = mlx::core;
namespace mem = hyle::mem;
constexpr std::array<c::CopyPhase, 3> kPhases{{{}, {1, 1}, {2, 2}}};
constexpr std::array<c::CopyRange, 4> kRanges{{
    {{0}, {1}, {0}, 0, 0, 16}, {{0}, {1}, {0}, 16, 16, 16},
    {{1}, {2}, {1}, 0, 0, 32}, {{2}, {3}, {2}, 0, 0, 32}}};
struct DrainGuard final { a::Context &context; ~DrainGuard() { context.close().IgnoreError(); } };
std::array<c::CopyResourceSpec, 4> specs(mem::MemoryResourceRef resource) {
  return {{{resource, 32}, {resource, 32}, {resource, 32}, {resource, 32}}};
}
struct Data final {
  std::array<c::LeasePool, 4> pools;
  explicit Data(mem::MemoryResourceRef resource) {
    for (auto &pool : pools) pool = c::LeasePool::create(resource, 1024, {.max_blocks = 2}).value();
  }
  hyle::Status bind(a::CopyBatchInvocation &invocation, std::byte value) noexcept {
    HYLE_ASSIGN_OR_RETURN(auto writer, pools[0].allocate(32)); HYLE_RETURN_IF_ERROR(writer.view()->fill(value));
    HYLE_ASSIGN_OR_RETURN(auto ready, writer.publish()); HYLE_RETURN_IF_ERROR(invocation.bind_source({0}, ready));
    for (std::uint32_t id = 1; id < pools.size(); ++id) {
      HYLE_ASSIGN_OR_RETURN(auto destination, pools[id].allocate(32));
      HYLE_RETURN_IF_ERROR(invocation.bind_mutable({id}, std::move(destination)));
    }
    return hyle::Status::Ok();
  }
};
TEST(AppleCopyProgram, TwoSlotsPhaseOutputsAndThreeReplaysKeepActualMetadataFixed) {
  auto context = a::Context::create().value(); auto memory = a::MemoryResource::create(context).value(); Data data(memory.ref());
  mem::QuotaAllocator metadata(mem::system_allocator()); DrainGuard drain{context};
  auto program = a::CopyBatchProgram::prepare(context, specs(memory.ref()), {.allocator = metadata.get()}).value();
  auto first = a::CopyBatchInvocation::create(program).value(), second = a::CopyBatchInvocation::create(program).value();
  const auto cold_bytes = metadata.used();
  for (unsigned replay = 0; replay < 3; ++replay) {
    ASSERT_TRUE(data.bind(first, std::byte(17 + replay)).ok()); ASSERT_TRUE(data.bind(second, std::byte(79 + replay)).ok());
    ASSERT_TRUE(first.prepare_replay(kPhases, kRanges).ok()); ASSERT_TRUE(second.prepare_replay(kPhases, kRanges).ok());
    auto one = first.submit().value(), two = second.submit().value();
    EXPECT_FALSE(first.reset().ok()); EXPECT_FALSE(first.submit().has_value());
    ASSERT_TRUE(one.wait().ok()); ASSERT_TRUE(two.wait().ok());
    auto left = one.output({2}, {3}, 0, 32).value(), right = two.output({2}, {3}, 0, 32).value();
    std::array<std::byte, 32> x, y; ASSERT_TRUE(left.view()->read(x).ok()); ASSERT_TRUE(right.view()->read(y).ok());
    for (auto byte : x) EXPECT_EQ(byte, std::byte(17 + replay)); for (auto byte : y) EXPECT_EQ(byte, std::byte(79 + replay));
    EXPECT_EQ(one.report().native_calls, 4u); EXPECT_EQ(one.report().phase_completions, 3u);
    EXPECT_GE(one.report().host_waits, 3u); EXPECT_EQ(one.report().native_waits, 0u);
    EXPECT_EQ(one.report().metadata_bytes, metadata.used()); EXPECT_EQ(metadata.used(), cold_bytes);
    EXPECT_FALSE(left.try_write().has_value()); one = {}; two = {};
    ASSERT_TRUE(first.reset().ok()); ASSERT_TRUE(second.reset().ok()); EXPECT_TRUE(left.try_write().has_value()); right.reset();
    for (auto &pool : data.pools) EXPECT_EQ(pool.stats().live_blocks, 0u);
  }
}

TEST(AppleCopyProgram, RetainedOriginalBlitProducerOrdersMutableSourceAndRetiresWriteGuard) {
  auto context = a::Context::create().value(); auto memory = a::MemoryResource::create(context).value(); Data data(memory.ref());
  auto input = mem::Buffer::create(memory.ref(), 32).value(); ASSERT_TRUE(input.fill(0, 32, std::byte{41}).ok());
  auto writer = data.pools[0].allocate(32).value(); auto destination = writer.view().value();
  auto native = d::prepare_copy_recording(context, 1).value();
  const std::array<c::detail::CopyBackendRange, 1> producer_ranges{{{destination, input.view()}}};
  ASSERT_TRUE(d::bind_copy_recording(*native, producer_ranges).ok());
  auto uses = c::LeaseUsePool::create({.max_batches = 1, .max_resources_per_batch = 1}).value();
  auto use = uses.begin().value(); ASSERT_TRUE(use.write(writer).ok());
  auto produced = d::submit_copy_recording(*native).value(); ASSERT_TRUE(use.attach(produced).ok());
  auto owner = c::detail::RetainedCompletion::create(mem::system_allocator()).value(); auto producer = owner.publish(std::move(produced));
  auto program = a::CopyBatchProgram::prepare(context, specs(memory.ref())).value(); auto invocation = a::CopyBatchInvocation::create(program).value();
  EXPECT_FALSE(writer.view().has_value()); ASSERT_TRUE(invocation.bind_mutable({0}, std::move(writer)).ok());
  ASSERT_TRUE(invocation.bind_producer({0}, std::move(producer)).ok());
  for (std::uint32_t id = 1; id < 4; ++id) {
    auto target = data.pools[id].allocate(32).value(); ASSERT_TRUE(invocation.bind_mutable({id}, std::move(target)).ok());
  }
  ASSERT_TRUE(invocation.prepare_replay(kPhases, kRanges).ok()); auto submitted = invocation.submit().value();
  ASSERT_TRUE(submitted.wait().ok()); auto ready = submitted.output({2}, {3}, 0, 32).value();
  std::array<std::byte, 32> bytes; ASSERT_TRUE(ready.view()->read(bytes).ok()); for (auto byte : bytes) EXPECT_EQ(byte, std::byte{41});
  EXPECT_EQ(uses.active_batches(), 0u); EXPECT_TRUE(owner.available());
}

// Decorate actual MLX GPU evaluation after an original square prefix. Warmup
// is unarmed; only this exact native invocation can hold the original owner.
struct Gate final {
  std::mutex mutex; std::condition_variable condition; bool armed = false, entered = false, released = false, fail = false;
  void wait() {
    std::unique_lock lock(mutex); if (!armed) return; entered = true; condition.notify_all();
    condition.wait(lock, [&] { return released; });
    if (fail) throw std::runtime_error("injected F2 MLX native prefix failure");
  }
  void release() { std::lock_guard lock(mutex); released = true; condition.notify_all(); }
};
struct GateGuard final { std::shared_ptr<Gate> gate; a::Context &context; ~GateGuard() { gate->release(); context.close().IgnoreError(); } };
class Checkpoint final : public mx::Primitive {
public:
  Checkpoint(mx::Stream stream, std::shared_ptr<Gate> gate) : Primitive(stream), gate_(std::move(gate)) {}
  void eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) override { throw std::runtime_error("GPU checkpoint cannot run on CPU"); }
  void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override { gate_->wait(); outputs[0].copy_shared_buffer(inputs[0]); }
  const char *name() const override { return "HyleF2NativeCheckpoint"; }
private: std::shared_ptr<Gate> gate_;
};
TEST(AppleCopyProgram, EarlyIndependentGroupStaysReadyDuringQueuedCancellationAndAbandonment) {
  auto context = a::Context::create().value(); auto memory = a::MemoryResource::create(context).value(); Data data(memory.ref());
  auto gate = std::make_shared<Gate>(); GateGuard release{gate, context};
  const std::array inputs{a::InputSpec{{4}, mx::float32}};
  auto native_program = context.compile([gate](const auto &args, mx::Stream stream) {
    auto value = mx::square(args[0], stream);
    return std::vector{mx::array(value.shape(), value.dtype(), std::make_shared<Checkpoint>(stream, gate), {value})};
  }, inputs).value();
  auto native = native_program.create_invocation().value();
  auto program = a::CopyBatchProgram::prepare(context, specs(memory.ref())).value();
  auto first = a::CopyBatchInvocation::create(program).value(), second = a::CopyBatchInvocation::create(program).value();
  ASSERT_TRUE(data.bind(first, std::byte{23}).ok()); ASSERT_TRUE(data.bind(second, std::byte{61}).ok());
  ASSERT_TRUE(first.prepare_replay(kPhases, kRanges).ok()); ASSERT_TRUE(second.prepare_replay(kPhases, kRanges).ok());
  auto completed = first.submit().value(); ASSERT_TRUE(completed.wait().ok()); auto ready = completed.output({2}, {3}, 0, 32).value();
  const std::array values{mx::array({1.f, 2.f, 3.f, 4.f})}; ASSERT_TRUE(native.bind(values).ok());
  { std::lock_guard lock(gate->mutex); gate->armed = true; }
  auto blocked = native.submit().value();
  { std::unique_lock lock(gate->mutex); ASSERT_TRUE(gate->condition.wait_for(lock, std::chrono::seconds(3), [&] { return gate->entered; })); }
  auto queued = second.submit().value(); EXPECT_TRUE(queued.in_flight()); ASSERT_TRUE(queued.request_cancel().ok());
  EXPECT_FALSE(queued.output({2}, {3}, 0, 32).has_value()); EXPECT_FALSE(second.reset().ok());
  std::array<std::byte, 32> bytes; ASSERT_TRUE(ready.view()->read(bytes).ok()); for (auto byte : bytes) EXPECT_EQ(byte, std::byte{23});
  queued = {}; second = {}; gate->release(); ASSERT_TRUE(blocked.wait().has_value());
  ASSERT_TRUE(context.close().ok());
  for (auto &pool : data.pools) EXPECT_EQ(pool.stats().active_uses, 0u);
  EXPECT_EQ(data.pools[3].stats().live_blocks, 1u); // first retained old result only
}

TEST(AppleCopyProgram, EmptyDependencyNoConsumeAndAbandonedMetadataFollowActualJobs) {
  auto context = a::Context::create().value(); auto memory = a::MemoryResource::create(context).value(); Data data(memory.ref());
  mem::QuotaAllocator metadata(mem::system_allocator()); DrainGuard drain{context};
  {
    auto program = a::CopyBatchProgram::prepare(context, specs(memory.ref()), {.allocator = metadata.get()}).value();
    auto invocation = a::CopyBatchInvocation::create(program).value();
    auto oversized = data.pools[1].allocate(64).value();
    EXPECT_FALSE(invocation.bind_mutable({1}, std::move(oversized)).ok()); EXPECT_TRUE(oversized.valid()); oversized.reset();
    ASSERT_TRUE(data.bind(invocation, std::byte{31}).ok());
    auto error = c::Completion::completed(hyle::Status::Cancelled("producer"));
    EXPECT_FALSE(invocation.bind_producer({0}, std::move(error)).ok()); EXPECT_TRUE(error.valid());
    const std::array<c::CopyPhase, 2> phases{{{}, {1, 1}}};
    const std::array<c::CopyRange, 1> ranges{{{{0}, {1}, {0}, 0, 0, 32}}};
    ASSERT_TRUE(invocation.prepare_replay(phases, ranges).ok()); auto submitted = invocation.submit().value();
    ASSERT_TRUE(submitted.wait().ok()); EXPECT_EQ(submitted.report().phase_completions, 2u);
    submitted = {}; ASSERT_TRUE(invocation.reset().ok());
    ASSERT_TRUE(data.bind(invocation, std::byte{43}).ok()); ASSERT_TRUE(invocation.prepare_replay(kPhases, kRanges).ok());
    auto abandoned = invocation.submit().value(); EXPECT_GT(metadata.used(), 0u);
  }
  ASSERT_TRUE(context.close().ok()); EXPECT_EQ(metadata.used(), 0u);
  for (auto &pool : data.pools) { EXPECT_EQ(pool.stats().active_uses, 0u); EXPECT_EQ(pool.stats().live_blocks, 0u); }
  mem::QuotaAllocator no_metadata(mem::system_allocator(), 1);
  // Closed context rejects before admission; the allocator remains untouched.
  EXPECT_FALSE(a::CopyBatchProgram::prepare(context, specs(memory.ref()), {.allocator = no_metadata.get()}).has_value());
  EXPECT_EQ(no_metadata.used(), 0u);
}

TEST(AppleCopyProgram, LateOriginalNativeFailureBlocksQueuedCopiesAndPreservesOldReady) {
  auto context = a::Context::create().value(); auto memory = a::MemoryResource::create(context).value(); Data data(memory.ref());
  mem::QuotaAllocator metadata(mem::system_allocator());
  auto gate = std::make_shared<Gate>(); GateGuard release{gate, context};
  {
    const std::array inputs{a::InputSpec{{4}, mx::float32}};
    auto native_program = context.compile([gate](const auto &args, mx::Stream stream) {
      auto value = mx::square(args[0], stream);
      return std::vector{mx::array(value.shape(), value.dtype(), std::make_shared<Checkpoint>(stream, gate), {value})};
    }, inputs).value();
    auto native = native_program.create_invocation().value();
    auto program = a::CopyBatchProgram::prepare(context, specs(memory.ref()), {.allocator = metadata.get()}).value();
    auto first = a::CopyBatchInvocation::create(program).value(), second = a::CopyBatchInvocation::create(program).value();
    ASSERT_TRUE(data.bind(first, std::byte{29}).ok()); ASSERT_TRUE(first.prepare_replay(kPhases, kRanges).ok());
    auto completed = first.submit().value(); ASSERT_TRUE(completed.wait().ok());
    auto ready = completed.output({2}, {3}, 0, 32).value();
    ASSERT_TRUE(data.bind(second, std::byte{83}).ok()); ASSERT_TRUE(second.prepare_replay(kPhases, kRanges).ok());
    const std::array values{mx::array({1.f, 2.f, 3.f, 4.f})}; ASSERT_TRUE(native.bind(values).ok());
    { std::lock_guard lock(gate->mutex); gate->armed = true; gate->fail = true; }
    auto producer = native.submit().value();
    { std::unique_lock lock(gate->mutex); ASSERT_TRUE(gate->condition.wait_for(lock, std::chrono::seconds(3), [&] { return gate->entered; })); }
    auto queued = second.submit().value(); EXPECT_TRUE(queued.in_flight());
    EXPECT_FALSE(queued.output({2}, {3}, 0, 32)); EXPECT_FALSE(second.reset().ok());
    for (auto &pool : data.pools) EXPECT_GT(pool.stats().active_uses, 0u);
    gate->release(); auto failed = producer.wait(); ASSERT_FALSE(failed); EXPECT_TRUE(failed.error().IsInternal());
    EXPECT_TRUE(queued.wait().IsInternal()); EXPECT_FALSE(queued.output({2}, {3}, 0, 32));
    EXPECT_FALSE(queued.in_flight()); EXPECT_EQ(queued.report().native_calls, 0u);
    EXPECT_FALSE(second.prepare_replay(kPhases, kRanges).ok());
    std::array<std::byte, 32> bytes; ASSERT_TRUE(ready.view()->read(bytes).ok());
    for (auto byte : bytes) EXPECT_EQ(byte, std::byte{29});
    queued = {}; second = {}; completed = {}; first = {}; program = {};
    EXPECT_TRUE(context.close().IsInternal());
    EXPECT_EQ(metadata.used(), 0u);
    for (auto &pool : data.pools) EXPECT_EQ(pool.stats().active_uses, 0u);
    EXPECT_EQ(data.pools[3].stats().live_blocks, 1u);
    ready.reset();
  }
  EXPECT_EQ(metadata.used(), 0u);
  for (auto &pool : data.pools) EXPECT_EQ(pool.stats().live_blocks, 0u);
}
} // namespace
