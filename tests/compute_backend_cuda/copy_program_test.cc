#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <thread>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/batch_transfer.h"
#include "hyle/compute/detail/retained_completion.h"
#include "hyle/mem/quota_allocator.h"

namespace {
using namespace hyle::compute;
using hyle::Status;
namespace cuda = hyle::compute::cuda;
namespace mem = hyle::mem;
constexpr std::array<CopyPhase, 3> kPhases{{{}, {1, 1}, {2, 2}}};
constexpr std::array<CopyRange, 4> kRanges{{
    {{0}, {1}, {0}, 0, 0, 16}, {{0}, {1}, {0}, 16, 16, 16},
    {{1}, {2}, {1}, 0, 0, 32}, {{2}, {3}, {2}, 0, 0, 32}}};

struct Data final {
  std::array<LeasePool, 4> pools;
  explicit Data(std::span<const CopyResourceSpec> specs) {
    for (std::size_t id = 0; id < specs.size(); ++id)
      pools[id] = LeasePool::create(specs[id].resource, 1024, {.max_blocks = 2}).value();
  }
  hyle::Status bind(CopyBatchInvocation &invocation, std::byte value) noexcept {
    auto writer = pools[0].allocate(32).value();
    HYLE_RETURN_IF_ERROR(writer.view()->fill(value));
    HYLE_ASSIGN_OR_RETURN(auto source, writer.publish());
    HYLE_RETURN_IF_ERROR(invocation.bind_source({0}, source));
    for (std::uint32_t id = 1; id < pools.size(); ++id) {
      HYLE_ASSIGN_OR_RETURN(auto destination, pools[id].allocate(32));
      HYLE_RETURN_IF_ERROR(invocation.bind_mutable({id}, std::move(destination)));
    }
    return Status::Ok();
  }
};
struct Gate final {
  std::atomic<bool> released{false};
  static void CUDART_CB wait(void *opaque) {
    auto &gate = *static_cast<Gate *>(opaque);
    for (auto ready = gate.released.load(std::memory_order_acquire); !ready;
         ready = gate.released.load(std::memory_order_acquire))
      gate.released.wait(false, std::memory_order_acquire);
  }
  void release() noexcept { released.store(true, std::memory_order_release); released.notify_all(); }
};
struct GateGuard final {
  Gate &gate; Context &context;
  ~GateGuard() { gate.release(); EXPECT_TRUE(context.drain().ok()); }
};
std::array<CopyResourceSpec, 4> specs(mem::MemoryResourceRef host, mem::MemoryResourceRef device) {
  return {{{host, 32}, {device, 32}, {device, 32}, {host, 32}}};
}

TEST(CudaCopyProgram, TwoIndependentSlotsPhaseBytesOldResultsAndNoHotMetadataGrowth) {
  auto context = Context::create(cuda::device(0).value()).value();
  auto queue = context.create_queue().value(), other_queue = context.create_queue().value();
  auto host = cuda::MemoryResource::create(context, cuda::MemoryKind::kPinnedHost).value();
  auto device = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  mem::QuotaAllocator metadata(mem::system_allocator());
  auto resources = specs(host.ref(), device.ref());
  Data data(resources);
  auto program = CopyBatchProgram::prepare(context, resources, {.allocator = metadata.get()}).value();
  auto first = CopyBatchInvocation::create(program, queue).value();
  auto second = CopyBatchInvocation::create(program, other_queue).value();
  ASSERT_TRUE(data.bind(first, std::byte{17}).ok());
  ASSERT_TRUE(data.bind(second, std::byte{83}).ok());
  const auto cold_bytes = metadata.used();
  ASSERT_TRUE(first.prepare_replay(kPhases, kRanges).ok());
  ASSERT_TRUE(second.prepare_replay(kPhases, kRanges).ok());
  EXPECT_EQ(metadata.used(), cold_bytes);
  auto one = first.submit().value(), two = second.submit().value();
  EXPECT_FALSE(first.submit().has_value()); EXPECT_FALSE(first.reset().ok());
  ASSERT_TRUE(one.wait().ok()); ASSERT_TRUE(two.wait().ok());
  auto a = one.output({2}, {3}, 0, 32).value(), b = two.output({2}, {3}, 0, 32).value();
  std::array<std::byte, 32> left, right;
  ASSERT_TRUE(a.view()->read(left).ok()); ASSERT_TRUE(b.view()->read(right).ok());
  for (auto byte : left) EXPECT_EQ(byte, std::byte{17});
  for (auto byte : right) EXPECT_EQ(byte, std::byte{83});
  EXPECT_EQ(metadata.used(), cold_bytes);
  EXPECT_EQ(one.report().native_calls, 3u); EXPECT_EQ(one.report().phase_completions, 3u);
  EXPECT_EQ(one.report().logical_bytes, 96u); EXPECT_EQ(one.report().physical_bytes, 96u);
  EXPECT_EQ(one.report().metadata_bytes, metadata.used()); EXPECT_TRUE(one.report().native_metadata_unknown);
  EXPECT_FALSE(a.try_write().has_value()); // invocation still retains its published block
  one = {}; ASSERT_TRUE(first.reset().ok());
  EXPECT_TRUE(a.try_write().has_value());
  EXPECT_EQ(data.pools[3].stats().live_blocks, 1u); // second slot and its old result still own one block
  two = {}; ASSERT_TRUE(second.reset().ok());
  b.reset(); EXPECT_EQ(data.pools[3].stats().live_blocks, 0u);
  ASSERT_TRUE(data.bind(first, std::byte{29}).ok());
  ASSERT_TRUE(first.prepare_replay(kPhases, kRanges).ok());
  auto replay = first.submit().value(); ASSERT_TRUE(replay.wait().ok());
  auto latest = replay.output({2}, {3}, 0, 32).value();
  ASSERT_TRUE(latest.view()->read(left).ok());
  for (auto byte : left) EXPECT_EQ(byte, std::byte{29});
  EXPECT_EQ(metadata.used(), cold_bytes);
}

TEST(CudaCopyProgram, EarlyReadyRetainsPendingReadonlySuccessorAndAbandonedNativeWork) {
  auto context = Context::create(cuda::device(0).value()).value();
  auto queue = context.create_queue().value(), producer_queue = context.create_queue().value();
  auto host = cuda::MemoryResource::create(context, cuda::MemoryKind::kPinnedHost).value();
  auto device = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  auto resources = specs(host.ref(), device.ref());
  Data data(resources);
  auto program = CopyBatchProgram::prepare(context, resources).value();
  auto invocation = CopyBatchInvocation::create(program, queue).value();
  ASSERT_TRUE(data.bind(invocation, std::byte{41}).ok());
  Gate gate; GateGuard release{gate, context};
  auto owner = detail::RetainedCompletion::create(mem::system_allocator()).value();
  auto native = cuda::submit(producer_queue, [&gate](cudaStream_t stream) noexcept {
    return cudaLaunchHostFunc(stream, &Gate::wait, &gate);
  }).value();
  auto producer = owner.publish(std::move(native));
  ASSERT_TRUE(invocation.bind_producer({2}, std::move(producer)).ok());
  ASSERT_TRUE(invocation.prepare_replay(kPhases, kRanges).ok());
  auto submitted = invocation.submit().value();
  EXPECT_TRUE(submitted.in_flight()); EXPECT_FALSE(invocation.reset().ok());
  EXPECT_FALSE(submitted.output({2}, {3}, 0, 32).has_value());
  ReadLease early;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < deadline) {
    auto ready = submitted.output({0}, {1}, 0, 32);
    if (ready) { early = std::move(*ready); break; }
    ASSERT_TRUE(ready.error().IsUnavailable());
    std::this_thread::yield();
  }
  ASSERT_TRUE(early.valid());
  EXPECT_FALSE(early.try_write().has_value());
  EXPECT_EQ(submitted.report().host_waits, 0u);
  EXPECT_EQ(data.pools[1].stats().active_uses, 1u);
  EXPECT_FALSE(invocation.close().ok());
  submitted = {}; invocation = {}; // abandon both controls; original retirement must keep the data/indices
  EXPECT_EQ(data.pools[1].stats().active_uses, 1u);
  EXPECT_FALSE(early.try_write().has_value());
  gate.release();
  ASSERT_TRUE(context.drain().ok());
  for (auto &pool : data.pools) ASSERT_TRUE(pool.drain().ok());
  EXPECT_EQ(data.pools[1].stats().active_uses, 0u);
  // Native attachment release may trail the use counter: original context
  // owner and all pool resources stay alive until this finite observation.
  bool reclaimed = false;
  const auto retired_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < retired_deadline) {
    auto writer = early.try_write();
    if (writer) { reclaimed = true; break; }
    std::this_thread::yield();
  }
  EXPECT_TRUE(reclaimed);
}

TEST(CudaCopyProgram, CapacityAliasErrorProducerAndZeroBatchDoNotConsumeOrAcceptWork) {
  auto context = Context::create(cuda::device(0).value()).value();
  auto queue = context.create_queue().value();
  auto host = cuda::MemoryResource::create(context, cuda::MemoryKind::kPinnedHost).value();
  auto device = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  auto resources = specs(host.ref(), device.ref());
  mem::QuotaAllocator no_metadata(mem::system_allocator(), 1);
  EXPECT_FALSE(CopyBatchProgram::prepare(context, resources, {.allocator = no_metadata.get()}).has_value());
  EXPECT_EQ(no_metadata.used(), 0u);
  auto program = CopyBatchProgram::prepare(context, resources).value();
  auto invocation = CopyBatchInvocation::create(program, queue).value();
  Data data(resources);
  auto oversized = data.pools[1].allocate(64).value();
  EXPECT_FALSE(invocation.bind_mutable({1}, std::move(oversized)).ok());
  EXPECT_TRUE(oversized.valid()); oversized.reset();
  ASSERT_TRUE(data.bind(invocation, std::byte{7}).ok());
  auto failed_producer = Completion::completed(hyle::Status::Cancelled("producer"));
  EXPECT_FALSE(invocation.bind_producer({0}, std::move(failed_producer)).ok());
  EXPECT_TRUE(failed_producer.valid());
  auto invalid = kRanges; invalid[0].source.value = 99;
  EXPECT_FALSE(invocation.prepare_replay(kPhases, invalid).ok());
  EXPECT_FALSE(invocation.submit().has_value());
  const std::array<CopyPhase, 1> phase{{{}}};
  const std::array<CopyRange, 2> overlap{{{{0}, {1}, {0}, 0, 0, 16}, {{0}, {1}, {0}, 16, 8, 16}}};
  EXPECT_FALSE(invocation.prepare_replay(phase, overlap).ok());
  EXPECT_EQ(invocation.report().native_calls, 0u);
  ASSERT_TRUE(invocation.prepare_replay(phase, {}).ok());
  auto empty = invocation.submit().value();
  EXPECT_FALSE(empty.in_flight()); EXPECT_EQ(empty.report().native_calls, 0u);
  EXPECT_EQ(empty.report().phase_completions, 1u); ASSERT_TRUE(empty.wait().ok());
  EXPECT_FALSE(empty.output({0}, {1}, 0, 1).has_value());
  EXPECT_FALSE(invocation.reset().ok());
  empty = {}; ASSERT_TRUE(invocation.reset().ok());
  EXPECT_TRUE(invocation.close().ok()); EXPECT_TRUE(invocation.close().ok());
  EXPECT_FALSE(invocation.prepare_replay({}, {}).ok());
  auto moved = std::move(invocation); EXPECT_FALSE(invocation.valid());
  EXPECT_TRUE(invocation.drain().ok()); EXPECT_TRUE(moved.drain().ok());
}
} // namespace
