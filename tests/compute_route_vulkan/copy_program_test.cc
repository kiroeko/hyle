#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <string>
#include <thread>

#include "fixture.h"
#include "vulkan_copy_internal.h"
#include "hyle/compute/detail/retained_completion.h"
#include "hyle/mem/quota_allocator.h"

namespace {
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace d = v::detail;
namespace mem = hyle::mem;
constexpr std::array<c::CopyPhase, 3> kPhases{{{}, {1, 1}, {2, 2}}};
constexpr std::array<c::CopyRange, 4> kRanges{{
    {{0}, {1}, {0}, 0, 0, 16}, {{0}, {1}, {0}, 16, 16, 16},
    {{1}, {2}, {1}, 0, 0, 32}, {{2}, {3}, {2}, 0, 0, 32}}};
struct DrainGuard final { v::Context &context; ~DrainGuard() { context.close().IgnoreError(); } };
struct Data final {
  std::array<c::LeasePool, 4> pools;
  explicit Data(std::span<const c::CopyResourceSpec> specs) {
    for (std::size_t id = 0; id < specs.size(); ++id)
      pools[id] = c::LeasePool::create(specs[id].resource, 1024, {.max_blocks = 2}).value();
  }
  hyle::Status bind(v::CopyBatchInvocation &invocation, std::byte value) noexcept {
    HYLE_ASSIGN_OR_RETURN(auto writer, pools[0].allocate(32));
    HYLE_RETURN_IF_ERROR(writer.view()->fill(value));
    HYLE_ASSIGN_OR_RETURN(auto read, writer.publish());
    HYLE_RETURN_IF_ERROR(invocation.bind_source({0}, read));
    for (std::uint32_t id = 1; id < pools.size(); ++id) {
      HYLE_ASSIGN_OR_RETURN(auto destination, pools[id].allocate(32));
      HYLE_RETURN_IF_ERROR(invocation.bind_mutable({id}, std::move(destination)));
    }
    return hyle::Status::Ok();
  }
};
std::array<c::CopyResourceSpec, 4> specs(mem::MemoryResourceRef host, mem::MemoryResourceRef device) {
  return {{{host, 32}, {device, 32}, {device, 32}, {host, 32}}};
}
TEST(VulkanCopyProgram, TwoSlotsPhaseReadyOldResultsAndReplayKeepActualMetadataFixed) {
  auto context = vulkan_test::context().value(); ASSERT_FALSE(context.environment().software_device);
  RecordProperty("device_name", context.environment().device_name);
  auto host = v::MemoryResource::create(context, v::MemoryKind::kStaging).value();
  auto device = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  auto resources = specs(host.ref(), device.ref()); Data data(resources);
  mem::QuotaAllocator metadata(mem::system_allocator()); DrainGuard drain{context};
  auto program = v::CopyBatchProgram::prepare(context, resources, {.allocator = metadata.get()}).value();
  auto first = v::CopyBatchInvocation::create(program).value(), second = v::CopyBatchInvocation::create(program).value();
  const auto cold_bytes = metadata.used();
  for (unsigned replay = 0; replay < 3; ++replay) {
    ASSERT_TRUE(data.bind(first, std::byte(13 + replay)).ok());
    ASSERT_TRUE(data.bind(second, std::byte(79 + replay)).ok());
    ASSERT_TRUE(first.prepare_replay(kPhases, kRanges).ok()); ASSERT_TRUE(second.prepare_replay(kPhases, kRanges).ok());
    auto one = first.submit().value(), two = second.submit().value();
    EXPECT_FALSE(first.reset().ok()); EXPECT_FALSE(first.submit().has_value());
    ASSERT_TRUE(one.wait().ok()); ASSERT_TRUE(two.wait().ok());
    auto a = one.output({2}, {3}, 0, 32).value(), b = two.output({2}, {3}, 0, 32).value();
    std::array<std::byte, 32> left, right;
    ASSERT_TRUE(a.view()->read(left).ok()); ASSERT_TRUE(b.view()->read(right).ok());
    for (auto byte : left) EXPECT_EQ(byte, std::byte(13 + replay));
    for (auto byte : right) EXPECT_EQ(byte, std::byte(79 + replay));
    EXPECT_EQ(one.report().native_calls, 3u); EXPECT_EQ(one.report().phase_completions, 3u);
    EXPECT_EQ(one.report().logical_bytes, 96u); EXPECT_EQ(one.report().physical_bytes, 96u);
    EXPECT_EQ(one.report().native_waits, 0u); // same original queue, explicit barriers
    EXPECT_EQ(one.report().metadata_bytes, metadata.used()); EXPECT_EQ(metadata.used(), cold_bytes);
    EXPECT_FALSE(a.try_write().has_value()); one = {}; two = {};
    ASSERT_TRUE(first.reset().ok()); ASSERT_TRUE(second.reset().ok());
    EXPECT_TRUE(a.try_write().has_value()); b.reset();
    for (auto &pool : data.pools) EXPECT_EQ(pool.stats().live_blocks, 0u);
  }
}

TEST(VulkanCopyProgram, OriginalRetainedProducerOrdersPendingMutableSourceAndGuards) {
  auto context = vulkan_test::context().value(); ASSERT_FALSE(context.environment().software_device);
  auto host = v::MemoryResource::create(context, v::MemoryKind::kStaging).value();
  auto device = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  auto resources = specs(host.ref(), device.ref()); resources[0].resource = device.ref();
  Data data(resources);
  auto input = mem::Buffer::create(host.ref(), 32).value(); ASSERT_TRUE(input.fill(0, 32, std::byte{37}).ok());
  auto writer = data.pools[0].allocate(32).value(); auto destination = writer.view().value();
  auto producer_recording = d::prepare_copy_recording(context, 1).value();
  const std::array<c::detail::CopyBackendRange, 1> producer_ranges{{{destination, input.view()}}};
  ASSERT_TRUE(d::bind_copy_recording(*producer_recording, producer_ranges, true).ok());
  auto uses = c::LeaseUsePool::create({.max_batches = 1, .max_resources_per_batch = 1}).value();
  auto use = uses.begin().value(); ASSERT_TRUE(use.write(writer).ok());
  auto original = d::submit_copy_recording(*producer_recording).value(); ASSERT_TRUE(use.attach(original).ok());
  auto retained = c::detail::RetainedCompletion::create(mem::system_allocator()).value();
  auto producer = retained.publish(std::move(original));
  auto program = v::CopyBatchProgram::prepare(context, resources).value();
  auto invocation = v::CopyBatchInvocation::create(program).value();
  ASSERT_FALSE(writer.view().has_value());
  ASSERT_TRUE(invocation.bind_mutable({0}, std::move(writer)).ok());
  ASSERT_TRUE(invocation.bind_producer({0}, std::move(producer)).ok());
  for (std::uint32_t id = 1; id < resources.size(); ++id) {
    auto target = data.pools[id].allocate(32).value(); ASSERT_TRUE(invocation.bind_mutable({id}, std::move(target)).ok());
  }
  ASSERT_TRUE(invocation.prepare_replay(kPhases, kRanges).ok());
  auto submitted = invocation.submit().value(); ASSERT_TRUE(submitted.wait().ok());
  auto ready = submitted.output({2}, {3}, 0, 32).value(); std::array<std::byte, 32> bytes;
  ASSERT_TRUE(ready.view()->read(bytes).ok()); for (auto byte : bytes) EXPECT_EQ(byte, std::byte{37});
  EXPECT_EQ(uses.active_batches(), 0u);
  EXPECT_EQ(submitted.report().native_waits, 0u); EXPECT_TRUE(retained.available());
}

struct SubmitFailure final {
  d::Device &device; PFN_vkQueueSubmit original; unsigned count = 0, fail = 0;
  static inline SubmitFailure *active = nullptr;
  explicit SubmitFailure(const v::Context &context) : device(*d::Access::runtime(context)->device), original(device.fn.QueueSubmit) {
    active = this; device.fn.QueueSubmit = invoke;
  }
  ~SubmitFailure() { device.fn.QueueSubmit = original; active = nullptr; }
  static VKAPI_ATTR VkResult VKAPI_CALL invoke(VkQueue queue, std::uint32_t count, const VkSubmitInfo *info, VkFence fence) {
    if (++active->count == active->fail) return VK_ERROR_OUT_OF_HOST_MEMORY;
    return active->original(queue, count, info, fence);
  }
};
TEST(VulkanCopyProgram, RejectedNativeSuffixPreservesSuccessfulEarlierGroupAndFailedSlot) {
  auto context = vulkan_test::context().value(); ASSERT_FALSE(context.environment().software_device);
  auto host = v::MemoryResource::create(context, v::MemoryKind::kStaging).value();
  auto device = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  auto resources = specs(host.ref(), device.ref()); Data data(resources);
  auto program = v::CopyBatchProgram::prepare(context, resources).value();
  auto invocation = v::CopyBatchInvocation::create(program).value(); ASSERT_TRUE(data.bind(invocation, std::byte{43}).ok());
  ASSERT_TRUE(invocation.prepare_replay(kPhases, kRanges).ok());
  SubmitFailure fault(context);
  // Eligible upload is transfer + original compute handoff; otherwise it is
  // one compute submit. Fail the first native command of the next phase.
  fault.fail = context.environment().distinct_transfer_queue ? 3u : 2u;
  auto submitted = invocation.submit(); ASSERT_TRUE(submitted.has_value());
  auto terminal = submitted->wait(); EXPECT_TRUE(terminal.IsResourceExhausted());
  auto ready = submitted->output({0}, {1}, 0, 32); ASSERT_TRUE(ready.has_value());
  EXPECT_FALSE(submitted->output({2}, {3}, 0, 32).has_value());
  EXPECT_FALSE(invocation.prepare_replay(kPhases, kRanges).ok());
  EXPECT_EQ(submitted->report().phase_completions, 1u);
  submitted = v::CopyBatchSubmission{}; ASSERT_TRUE(invocation.reset().ok());
}

TEST(VulkanCopyProgram, EmptyDependencyUsesActualPrefixAndBindingErrorsDoNotConsume) {
  auto context = vulkan_test::context().value(); ASSERT_FALSE(context.environment().software_device);
  auto host = v::MemoryResource::create(context, v::MemoryKind::kStaging).value();
  auto device = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  auto resources = specs(host.ref(), device.ref()); Data data(resources);
  mem::QuotaAllocator no_metadata(mem::system_allocator(), 1);
  EXPECT_FALSE(v::CopyBatchProgram::prepare(context, resources, {.allocator = no_metadata.get()}).has_value());
  EXPECT_EQ(no_metadata.used(), 0u);
  auto program = v::CopyBatchProgram::prepare(context, resources).value();
  auto invocation = v::CopyBatchInvocation::create(program).value();
  auto oversized = data.pools[1].allocate(64).value();
  EXPECT_FALSE(invocation.bind_mutable({1}, std::move(oversized)).ok()); EXPECT_TRUE(oversized.valid()); oversized.reset();
  ASSERT_TRUE(data.bind(invocation, std::byte{59}).ok());
  auto error = c::Completion::completed(hyle::Status::Cancelled("producer"));
  EXPECT_FALSE(invocation.bind_producer({0}, std::move(error)).ok()); EXPECT_TRUE(error.valid());
  const std::array<c::CopyPhase, 2> phases{{{}, {1, 1}}};
  const std::array<c::CopyRange, 1> ranges{{{{0}, {1}, {0}, 0, 0, 32}}};
  ASSERT_TRUE(invocation.prepare_replay(phases, ranges).ok());
  auto submitted = invocation.submit().value(); ASSERT_TRUE(submitted.wait().ok());
  EXPECT_EQ(submitted.report().phase_completions, 2u); EXPECT_EQ(submitted.report().native_calls, 1u);
  EXPECT_TRUE(submitted.output({0}, {1}, 0, 32).has_value());
  submitted = {}; ASSERT_TRUE(invocation.reset().ok());
  ASSERT_TRUE(invocation.prepare_replay({}, {}).ok()); auto empty = invocation.submit().value();
  EXPECT_FALSE(empty.in_flight()); EXPECT_EQ(empty.report().phase_completions, 0u); ASSERT_TRUE(empty.wait().ok());
  empty = {}; EXPECT_TRUE(invocation.close().ok()); EXPECT_TRUE(invocation.close().ok());
}

TEST(VulkanCopyProgram, AbandonedHandlesKeepOriginalRunsUntilActualRetirementAndFreeMetadata) {
  auto context = vulkan_test::context().value(); ASSERT_FALSE(context.environment().software_device);
  auto host = v::MemoryResource::create(context, v::MemoryKind::kStaging).value();
  auto device = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  auto resources = specs(host.ref(), device.ref()); Data data(resources);
  mem::QuotaAllocator metadata(mem::system_allocator()); DrainGuard drain{context};
  {
    auto program = v::CopyBatchProgram::prepare(context, resources, {.allocator = metadata.get()}).value();
    auto invocation = v::CopyBatchInvocation::create(program).value();
    ASSERT_TRUE(data.bind(invocation, std::byte{61}).ok());
    ASSERT_TRUE(invocation.prepare_replay(kPhases, kRanges).ok());
    auto submitted = invocation.submit().value();
    EXPECT_GT(metadata.used(), 0u);
  }
  // No public observation has retired these original Runs, even if the GPU
  // has executed them already. Abandonment must retain descriptors and uses.
  EXPECT_GT(metadata.used(), 0u);
  bool retired = false;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (std::chrono::steady_clock::now() < deadline) {
    ASSERT_TRUE(context.collect().ok());
    if (metadata.used() == 0) { retired = true; break; }
    std::this_thread::yield();
  }
  EXPECT_TRUE(retired);
  for (auto &pool : data.pools) {
    EXPECT_EQ(pool.stats().active_uses, 0u); EXPECT_EQ(pool.stats().live_blocks, 0u);
  }
}
} // namespace
