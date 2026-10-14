#include <gtest/gtest.h>

#include <array>

#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/batch_transfer.h"
#include "hyle/compute/detail/copy_stream_access.h"
#include "hyle/compute/detail/copy_outcome_source.h"
#include "hyle/compute/transfer.h"
#include "hyle/compute/detail/lease_access.h"
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

TEST(CpuCopyProgram, StreamChunksKeepWholeTargetMutableUntilGroupPublication) {
  auto context = Context::create(cpu::device()).value();
  auto queue = context.create_queue().value();
  mem::QuotaAllocator metadata(mem::system_allocator());
  const std::array resources{CopyResourceSpec{mem::system_resource(), 16},
                            CopyResourceSpec{mem::system_resource(), 64}};
  auto program = CopyBatchProgram::prepare(context, resources, {.allocator = metadata.get()}).value();
  auto invocation = CopyBatchInvocation::create(program, queue).value();
  auto stage_pool = LeasePool::create(mem::system_resource(), 256).value();
  auto target_pool = LeasePool::create(mem::system_resource(), 256).value();
  auto stage_writer = stage_pool.allocate(16).value();
  auto target_writer = target_pool.allocate(64).value();
  const auto target_generation = target_writer.generation();
  const auto cold_metadata = metadata.used();
  const std::array phases{CopyPhase{}};
  for (std::size_t offset = 0; offset < 64; offset += 16) {
    std::array<std::byte, 16> bytes;
    for (std::size_t index = 0; index < bytes.size(); ++index)
      bytes[index] = std::byte(offset + index);
    ASSERT_TRUE(stage_writer.view()->write(bytes).ok());
    auto stage = stage_writer.publish().value();
    ASSERT_TRUE(invocation.bind_source({0}, stage).ok());
    ASSERT_TRUE(invocation.bind_mutable({1}, std::move(target_writer)).ok());
    const std::array ranges{CopyRange{{0}, {1}, {0}, 0, offset, bytes.size()}};
    ASSERT_TRUE(invocation.prepare_replay(phases, ranges).ok());
    auto submitted = invocation.submit().value();
    ASSERT_TRUE(submitted.wait().ok());
    // Current replay coverage cannot certify the whole group. A live result
    // still owns the invocation even for synchronous CPU retirement.
    EXPECT_FALSE(submitted.output({0}, {1}, 0, 64).has_value());
    EXPECT_FALSE(hyle::compute::detail::CopyStreamAccess::take_mutable(invocation, {1}).has_value());
    EXPECT_FALSE(stage.try_write().has_value());
    submitted = {};
    EXPECT_FALSE(hyle::compute::detail::CopyStreamAccess::take_mutable(invocation, {0}).has_value());
    target_writer = hyle::compute::detail::CopyStreamAccess::take_mutable(invocation, {1}).value();
    EXPECT_EQ(target_writer.generation(), target_generation);
    EXPECT_FALSE(invocation.submit().has_value());
    ASSERT_TRUE(invocation.reset().ok());
    stage_writer = stage.try_write().value();
    EXPECT_EQ(metadata.used(), cold_metadata);
  }
  auto ready = target_writer.publish().value();
  std::array<std::byte, 64> output;
  ASSERT_TRUE(ready.view()->read(output).ok());
  for (std::size_t index = 0; index < output.size(); ++index)
    EXPECT_EQ(output[index], std::byte(index));
}

TEST(CpuCopyProgram, StreamWithdrawalAndBackingAccountingPreserveOriginalOwners) {
  auto context = Context::create(cpu::device()).value();
  auto queue = context.create_queue().value();
  auto program = CopyBatchProgram::prepare(context, specs()).value();
  auto invocation = CopyBatchInvocation::create(program, queue).value();
  auto pool = LeasePool::create(mem::system_resource(), 512).value();
  auto first = pool.allocate(32).value(), second = pool.allocate(32).value();
  const auto one = hyle::compute::detail::LeaseAccess::backing(first).value();
  const auto two = hyle::compute::detail::LeaseAccess::backing(second).value();
  EXPECT_EQ(one.identity, two.identity); EXPECT_EQ(one.bytes, 512u);
  EXPECT_EQ(one.resource, mem::system_resource());
  ASSERT_TRUE(invocation.bind_mutable({1}, std::move(first)).ok());
  EXPECT_FALSE(hyle::compute::detail::CopyStreamAccess::take_mutable(invocation, {9}).has_value());
  first = hyle::compute::detail::CopyStreamAccess::take_mutable(invocation, {1}).value();
  auto ready = first.publish().value();
  EXPECT_EQ(hyle::compute::detail::LeaseAccess::backing(ready)->identity, one.identity);
  EXPECT_TRUE(hyle::compute::detail::LeaseAccess::unique_read_owner(ready));
  auto reader = ready.subview(8, 8).value();
  EXPECT_FALSE(hyle::compute::detail::LeaseAccess::unique_read_owner(ready));
  reader.reset();
  auto uses = LeaseUsePool::create().value();
  {
    auto admitted = uses.begin().value();
    ASSERT_TRUE(admitted.read(ready).ok());
    EXPECT_FALSE(hyle::compute::detail::LeaseAccess::unique_read_owner(ready));
  }
  ASSERT_TRUE(pool.close().ok());
  // Accounting can release a last reader of a closed native owner without
  // promoting it to a writer or replacing the owner's retention callback.
  EXPECT_TRUE(hyle::compute::detail::LeaseAccess::unique_read_owner(ready));
  EXPECT_FALSE(ready.try_write().has_value());
  EXPECT_FALSE(hyle::compute::detail::LeaseAccess::backing(ReadLease{}).has_value());
}

TEST(CpuCopyUpdate, InvalidUnacceptedAndPublishedControlsDoNotCreatePendingAuthority) {
  CopyBatchInvocation empty_invocation;
  CopyBatchSubmission empty_submission;
  EXPECT_FALSE(CopyBatchUpdate::take(std::move(empty_invocation), std::move(empty_submission)));
  auto context = Context::create(cpu::device()).value();
  auto queue = context.create_queue().value();
  auto program = CopyBatchProgram::prepare(context, specs()).value();
  auto invocation = CopyBatchInvocation::create(program, queue).value();
  EXPECT_FALSE(CopyBatchUpdate::take(std::move(invocation), std::move(empty_submission)));
  EXPECT_TRUE(invocation.valid());
  std::array<LeasePool, 4> pools;
  for (auto &pool : pools) pool = LeasePool::create(mem::system_resource(), 1024).value();
  ASSERT_TRUE(bind(pools, invocation, std::byte{61}).ok());
  ASSERT_TRUE(invocation.prepare_replay(kPhases, kRanges).ok());
  auto submission = invocation.submit().value();
  auto published = submission.output({2}, {3}, 0, 32).value();
  auto update = CopyBatchUpdate::take(std::move(invocation), std::move(submission)).value();
  EXPECT_FALSE(invocation.valid()); EXPECT_FALSE(submission.valid());
  EXPECT_FALSE(detail::CopyStreamAccess::input(update, {0}));
  EXPECT_FALSE(detail::CopyStreamAccess::input(update, {3}));
  EXPECT_FALSE(detail::CopyStreamAccess::input(update, {4}));
  auto input = detail::CopyStreamAccess::input(update, {1}).value();
  EXPECT_NE(input.root, nullptr); EXPECT_NE(input.writer, nullptr);
  EXPECT_EQ(input.last_write.value, 0u); EXPECT_EQ(input.view.size(), 32u);
  auto dependency = detail::CopyStreamAccess::dependency(update).value();
  EXPECT_FALSE(dependency.pending()); // Original CPU copy is synchronously retired.
  auto uses = LeaseUsePool::create().value();
  auto use = uses.begin().value();
  ASSERT_TRUE(detail::CopyStreamAccess::admit_read(use, update, {1}).ok());
  EXPECT_EQ(pools[1].stats().active_uses, 1u);
  use = {};
  auto writer = update.take_mutable({1}).value();
  EXPECT_FALSE(update.take_mutable({1}));
  auto recovered = update.release_invocation().value();
  EXPECT_TRUE(recovered.valid()); EXPECT_FALSE(update.valid());
  ASSERT_TRUE(recovered.reset().ok());
  std::array<std::byte,32> bytes;
  ASSERT_TRUE(writer.view()->read(bytes).ok());
  for (auto byte : bytes) EXPECT_EQ(byte, std::byte{61});
  ASSERT_TRUE(published.view()->read(bytes).ok());
  for (auto byte : bytes) EXPECT_EQ(byte, std::byte{61});
}
TEST(CpuCopyUpdate, RetirementSourceBorrowCountsAcceptedPhasesAndDistinctUsedProducers) {
  auto context = Context::create(cpu::device()).value();
  auto queue = context.create_queue().value();
  std::array<CopyResourceSpec, 5> resources;
  for (auto &resource : resources) resource = {mem::system_resource(), 32};
  auto program = CopyBatchProgram::prepare(context, resources).value();
  auto invocation = CopyBatchInvocation::create(program, queue).value();
  std::array<LeasePool, 4> pools;
  for (auto &pool : pools) pool = LeasePool::create(mem::system_resource(), 512).value();
  ASSERT_TRUE(bind(pools, invocation, std::byte{37}).ok());
  auto unused_pool = LeasePool::create(mem::system_resource(), 512).value();
  auto unused_writer = unused_pool.allocate(32).value();
  ASSERT_TRUE(invocation.bind_mutable({4}, std::move(unused_writer)).ok());
  ASSERT_TRUE(invocation.bind_producer({0}, Completion::completed()).ok());
  ASSERT_TRUE(invocation.bind_producer({4}, Completion::completed()).ok());
  const std::array<CopyRange, 3> ranges{{{{0}, {1}, {0}, 0, 0, 32},
      {{1}, {2}, {1}, 0, 0, 32}, {{0}, {3}, {2}, 0, 0, 32}}};
  ASSERT_TRUE(invocation.prepare_replay(kPhases, ranges).ok());
  auto submitted = invocation.submit().value();
  auto update = CopyBatchUpdate::take(std::move(invocation), std::move(submitted)).value();
  ASSERT_EQ(detail::CopyStreamAccess::retirement_sources(update, {}).value(), 4u);
  std::array<Completion *, 4> original{};
  ASSERT_EQ(detail::CopyStreamAccess::retirement_sources(update, original).value(), 4u);
  for (auto *completion : original) ASSERT_NE(completion, nullptr);
  std::array<Completion *, 3> short_buffer{original[0], original[1], original[2]};
  const auto before = short_buffer;
  EXPECT_TRUE(detail::CopyStreamAccess::retirement_sources(update, short_buffer)
      .error().IsResourceExhausted());
  EXPECT_EQ(short_buffer, before);
  std::array<Completion *, 4> repeated{};
  ASSERT_EQ(detail::CopyStreamAccess::retirement_sources(update, repeated).value(), 4u);
  EXPECT_EQ(repeated, original);
  EXPECT_TRUE(update.valid());
  EXPECT_TRUE(update.wait().ok());
  CopyBatchUpdate invalid;
  EXPECT_FALSE(detail::CopyStreamAccess::retirement_sources(invalid, {}));
}
TEST(CpuCopyUpdate, ColdSourcePreservesOriginalTwoInvocationsFor64WarmReplays) {
  auto context = Context::create(cpu::device()).value();
  auto queue = context.create_queue().value();
  mem::QuotaAllocator metadata(mem::system_allocator());
  struct Counted final { mem::QuotaAllocator *metadata; std::size_t allocations = 0; } counts{&metadata};
  mem::AllocatorRef counted{
      [](void *p, std::size_t bytes, std::size_t alignment) noexcept -> void * {
        auto &state = *static_cast<Counted *>(p); ++state.allocations;
        return state.metadata->get().allocate(bytes, alignment);
      },
      [](void *p, void *pointer, std::size_t bytes, std::size_t alignment) noexcept {
        static_cast<Counted *>(p)->metadata->get().deallocate(pointer, bytes, alignment);
      },
      [](void *p, void *pointer, std::size_t before, std::size_t bytes,
          std::size_t alignment) noexcept -> void * {
        ++static_cast<Counted *>(p)->allocations;
        return static_cast<Counted *>(p)->metadata->get().reallocate(pointer, before, bytes, alignment);
      }, &counts};
  auto program = CopyBatchProgram::prepare(context, specs(), {.allocator = counted}).value();
  std::array<CopyBatchInvocation, 2> invocations;
  using Source = detail::CopyOutcomeSource<CopyBatchUpdate, detail::CopyStreamAccess>;
  std::array<Source, 2> sources;
  std::array<detail::DependentCompletion, 2> observers;
  std::array<LeasePool, 4> pools;
  for (auto &pool : pools) pool = LeasePool::create(mem::system_resource(), 1024, {.max_blocks = 2}).value();
  auto output_pool = LeasePool::create(mem::system_resource(), 512).value();
  auto output = output_pool.allocate(32).value();
  const auto output_view = output.view().value();
  auto uses = LeaseUsePool::create().value();
  for (unsigned slot = 0; slot < 2; ++slot) {
    invocations[slot] = CopyBatchInvocation::create(program, queue).value();
    sources[slot] = Source::create(3, {.allocator = counted}).value();
    observers[slot] = detail::DependentCompletion::create(counted).value();
  }
  const auto cold_bytes = metadata.used();
  const auto cold_allocations = counts.allocations;
  for (unsigned step = 0; step < 64; ++step) {
    const auto slot = step % 2;
    auto &invocation = invocations[slot];
    ASSERT_TRUE(bind(pools, invocation, std::byte(step + 1)).ok());
    ASSERT_TRUE(invocation.prepare_replay(kPhases, kRanges).ok());
    auto submitted = invocation.submit().value();
    auto update = CopyBatchUpdate::take(std::move(invocation), std::move(submitted)).value();
    auto protocol = sources[slot].take(std::move(update)).value();
    auto pending = detail::CopyStreamAccess::input(sources[slot].update(), {3}).value();
    auto reading = uses.begin().value();
    ASSERT_TRUE(detail::CopyStreamAccess::admit_read(reading, sources[slot].update(), {3}).ok());
    ASSERT_TRUE(reading.write(output).ok());
    auto suffix = copy_async(queue, output_view, pending.view).value();
    ASSERT_TRUE(reading.attach(suffix).ok());
    auto completion = observers[slot].publish(std::move(suffix), protocol);
    ASSERT_TRUE(completion.wait().ok());
    ASSERT_TRUE(sources[slot].available());
    EXPECT_TRUE(sources[slot].update().valid()); // Original controls remain recoverable.
    std::array<std::byte, 32> bytes{};
    ASSERT_TRUE(output.view()->read(bytes).ok());
    for (auto byte : bytes) EXPECT_EQ(byte, std::byte(step + 1));
    invocation = sources[slot].update().release_invocation().value();
    ASSERT_TRUE(invocation.reset().ok());
    EXPECT_EQ(metadata.used(), cold_bytes);
    EXPECT_EQ(counts.allocations, cold_allocations);
    EXPECT_EQ(uses.active_batches(), 0u);
  }
}
} // namespace
