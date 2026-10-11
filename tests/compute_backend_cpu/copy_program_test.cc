#include <gtest/gtest.h>

#include <array>

#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/batch_transfer.h"
#include "hyle/mem/quota_allocator.h"

namespace {
using namespace hyle::compute;
namespace mem = hyle::mem;
constexpr std::array<CopyPhase, 3> kPhases{{{}, {1, 1}, {2, 2}}};
constexpr std::array<CopyRange, 4> kRanges{{
    {{0}, {1}, {0}, 0, 0, 16}, {{0}, {1}, {0}, 16, 16, 16},
    {{1}, {2}, {1}, 0, 0, 32}, {{2}, {3}, {2}, 0, 0, 32}}};
std::array<CopyResourceSpec, 4> specs() {
  return {{{mem::system_resource(), 32}, {mem::system_resource(), 32},
           {mem::system_resource(), 32}, {mem::system_resource(), 32}}};
}
hyle::Status bind(std::array<LeasePool, 4> &pools, CopyBatchInvocation &invocation, std::byte value) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto writer, pools[0].allocate(32));
  HYLE_RETURN_IF_ERROR(writer.view()->fill(value));
  HYLE_ASSIGN_OR_RETURN(auto ready, writer.publish());
  HYLE_RETURN_IF_ERROR(invocation.bind_source({0}, ready));
  for (std::uint32_t id = 1; id < pools.size(); ++id) {
    HYLE_ASSIGN_OR_RETURN(auto destination, pools[id].allocate(32));
    HYLE_RETURN_IF_ERROR(invocation.bind_mutable({id}, std::move(destination)));
  }
  return hyle::Status::Ok();
}

TEST(CpuCopyProgram, TwoSlotsUseOriginalSynchronousCopyAndFixedMetadataReplay) {
  auto context = Context::create(cpu::device()).value();
  auto queue = context.create_queue().value();
  mem::QuotaAllocator metadata(mem::system_allocator());
  auto program = CopyBatchProgram::prepare(context, specs(), {.allocator = metadata.get()}).value();
  auto first = CopyBatchInvocation::create(program, queue).value();
  auto second = CopyBatchInvocation::create(program, queue).value();
  std::array<LeasePool, 4> pools;
  for (auto &pool : pools) pool = LeasePool::create(mem::system_resource(), 1024, {.max_blocks = 2}).value();
  const auto cold_bytes = metadata.used();
  for (unsigned replay = 0; replay < 3; ++replay) {
    ASSERT_TRUE(bind(pools, first, std::byte(11 + replay)).ok());
    ASSERT_TRUE(bind(pools, second, std::byte(71 + replay)).ok());
    ASSERT_TRUE(first.prepare_replay(kPhases, kRanges).ok());
    ASSERT_TRUE(second.prepare_replay(kPhases, kRanges).ok());
    auto one = first.submit().value(), two = second.submit().value();
    EXPECT_FALSE(one.in_flight()); EXPECT_FALSE(two.in_flight());
    auto left = one.output({2}, {3}, 0, 32).value(), right = two.output({2}, {3}, 0, 32).value();
    std::array<std::byte, 32> a, b;
    ASSERT_TRUE(left.view()->read(a).ok()); ASSERT_TRUE(right.view()->read(b).ok());
    for (auto byte : a) EXPECT_EQ(byte, std::byte(11 + replay));
    for (auto byte : b) EXPECT_EQ(byte, std::byte(71 + replay));
    EXPECT_EQ(one.report().native_calls, 4u); EXPECT_EQ(one.report().phase_completions, 3u);
    EXPECT_EQ(one.report().native_waits, 0u); EXPECT_EQ(one.report().host_waits, 0u);
    EXPECT_FALSE(one.report().native_metadata_unknown);
    EXPECT_EQ(one.report().metadata_bytes, cold_bytes); EXPECT_EQ(metadata.used(), cold_bytes);
    EXPECT_FALSE(first.reset().ok()); EXPECT_FALSE(left.try_write().has_value());
    one = {}; two = {}; ASSERT_TRUE(first.reset().ok()); ASSERT_TRUE(second.reset().ok());
    left.reset(); right.reset();
    for (auto &pool : pools) EXPECT_EQ(pool.stats().live_blocks, 0u);
  }
}

TEST(CpuCopyProgram, LaterWriteBlocksEarlierOutputAndAliasHazardsRequireExplicitPhases) {
  auto context = Context::create(cpu::device()).value();
  auto queue = context.create_queue().value();
  auto program = CopyBatchProgram::prepare(context, specs()).value();
  auto invocation = CopyBatchInvocation::create(program, queue).value();
  std::array<LeasePool, 4> pools;
  for (auto &pool : pools) pool = LeasePool::create(mem::system_resource(), 512).value();
  ASSERT_TRUE(bind(pools, invocation, std::byte{39}).ok());
  const std::array<CopyRange, 2> independent{{{{0}, {1}, {0}, 0, 0, 32}, {{1}, {2}, {0}, 0, 0, 32}}};
  const std::array<CopyPhase, 1> one_phase{{{}}};
  EXPECT_FALSE(invocation.prepare_replay(one_phase, independent).ok());
  EXPECT_FALSE(invocation.submit().has_value());
  EXPECT_EQ(invocation.report().native_calls, 0u);
  const std::array<CopyRange, 3> dependent{{{{0}, {1}, {0}, 0, 0, 32},
      {{1}, {2}, {1}, 0, 0, 32}, {{0}, {1}, {2}, 0, 0, 16}}};
  ASSERT_TRUE(invocation.prepare_replay(kPhases, dependent).ok());
  auto submitted = invocation.submit().value();
  EXPECT_FALSE(submitted.output({0}, {1}, 0, 32).has_value());
  auto latest = submitted.output({2}, {1}, 0, 32).value();
  std::array<std::byte, 32> bytes;
  ASSERT_TRUE(latest.view()->read(bytes).ok());
  for (auto byte : bytes) EXPECT_EQ(byte, std::byte{39});
  ASSERT_TRUE(submitted.wait().ok());
}

TEST(CpuCopyProgram, PhysicalSelfCopyAcrossReadSubviewsEmptyInvalidAndOomHaveNoNativeWork) {
  auto context = Context::create(cpu::device()).value();
  auto queue = context.create_queue().value();
  mem::QuotaAllocator no_metadata(mem::system_allocator(), 1);
  EXPECT_FALSE(CopyBatchProgram::prepare(context, specs(), {.allocator = no_metadata.get()}).has_value());
  EXPECT_EQ(no_metadata.used(), 0u);
  auto program = CopyBatchProgram::prepare(context, specs()).value();
  auto invocation = CopyBatchInvocation::create(program, queue).value();
  auto pool = LeasePool::create(mem::system_resource(), 512).value();
  auto writer = pool.allocate(32).value(); ASSERT_TRUE(writer.view()->fill(std::byte{23}).ok());
  auto ready = writer.publish().value();
  ASSERT_TRUE(invocation.bind_source({0}, ready).ok());
  ASSERT_TRUE(invocation.bind_source({1}, ready.subview(16, 16).value()).ok());
  const std::array<CopyPhase, 1> phase{{{}}};
  const std::array<CopyRange, 1> self{{{{0}, {1}, {0}, 16, 0, 16}}};
  ASSERT_TRUE(invocation.prepare_replay(phase, self).ok());
  auto noop = invocation.submit().value(); ASSERT_TRUE(noop.wait().ok());
  EXPECT_EQ(noop.report().no_op_copies, 1u); EXPECT_EQ(noop.report().native_calls, 0u);
  EXPECT_EQ(noop.report().logical_bytes, 16u); EXPECT_EQ(noop.report().physical_bytes, 0u);
  noop = {}; ASSERT_TRUE(invocation.reset().ok());
  ASSERT_TRUE(invocation.prepare_replay({}, {}).ok());
  auto empty = invocation.submit().value(); ASSERT_TRUE(empty.wait().ok());
  EXPECT_EQ(empty.report().phase_completions, 0u); EXPECT_FALSE(empty.in_flight());
  empty = {}; ASSERT_TRUE(invocation.reset().ok());
  auto oversized = pool.allocate(64).value();
  EXPECT_FALSE(invocation.bind_mutable({0}, std::move(oversized)).ok()); EXPECT_TRUE(oversized.valid());
  auto cancelled = Completion::completed(hyle::Status::Cancelled("producer"));
  ASSERT_TRUE(invocation.bind_source({0}, ready).ok());
  EXPECT_FALSE(invocation.bind_producer({0}, std::move(cancelled)).ok()); EXPECT_TRUE(cancelled.valid());
  EXPECT_TRUE(invocation.close().ok()); EXPECT_TRUE(invocation.drain().ok());
}
} // namespace
