#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <barrier>
#include <thread>
#include <limits>
#include <type_traits>

#include "completion_internal.h"
#include "hyle/compute/detail/append_lease.h"
#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/detail/lease_access.h"
#include "hyle/compute/leased_operation.h"
#include "hyle/compute/lease_slots.h"
#include "hyle/compute/resource_leases.h"

namespace {
using namespace hyle::compute;
namespace mem = hyle::mem;

// Writer bindings borrow a named permission; ready input leases are owned.
static_assert(std::is_constructible_v<LeasedInput, TensorDesc, WriteLease &>);
static_assert(std::is_constructible_v<LeasedInput, TensorDesc, const WriteLease &>);
static_assert(std::is_constructible_v<LeasedOutput, TensorDesc, WriteLease &>);
static_assert(std::is_constructible_v<LeasedOutput, TensorDesc, const WriteLease &>);
static_assert(!std::is_constructible_v<LeasedInput, TensorDesc, WriteLease &&>);
static_assert(!std::is_constructible_v<LeasedInput, TensorDesc, const WriteLease &&>);
static_assert(!std::is_constructible_v<LeasedOutput, TensorDesc, WriteLease &&>);
static_assert(!std::is_constructible_v<LeasedOutput, TensorDesc, const WriteLease &&>);
static_assert(std::is_constructible_v<LeasedInput, TensorDesc, ReadLease>);

struct PendingUse final {
  bool done = false;
  hyle::compute::detail::CompletionLifetime retained{};
  hyle::Status hint{};
};
const hyle::compute::detail::CompletionVTable kPending{
    .destroy = [](void *opaque, hyle::compute::detail::CompletionLifetime lifetime,
                  hyle::Status hint) noexcept {
      auto &use = *static_cast<PendingUse *>(opaque);
      if (use.done && lifetime.valid()) {
        lifetime.terminal(lifetime.state, hint);
        lifetime.release(lifetime.state);
      } else {
        use.retained = lifetime;
        use.hint = hint;
      }
    },
    .poll = [](void *opaque) noexcept -> hyle::Result<CompletionPoll> {
      return static_cast<PendingUse *>(opaque)->done ? CompletionPoll::kComplete : CompletionPoll::kPending;
    },
    .wait = [](void *opaque) noexcept {
      static_cast<PendingUse *>(opaque)->done = true;
      return hyle::Status::Ok();
    }};

static_assert(!std::is_copy_constructible_v<detail::AppendLease>);
static_assert(!std::is_copy_assignable_v<detail::AppendLease>);
static_assert(std::is_nothrow_move_constructible_v<detail::AppendLease>);

TEST(ResourceLeases, FixedSlotsBindExactOriginalBlocksAndSharedSources) {
  const auto item = TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{16}).value();
  auto pool = LeasePool::create(mem::system_resource(), 577, {.max_blocks = 3}).value();
  auto layout = LeaseSlotLayout::create(pool, item, 3, 256).value();
  EXPECT_EQ(layout.description(), item);
  EXPECT_EQ(layout.slot_count(), 3u); EXPECT_EQ(layout.stride_bytes(), 256u);
  auto first = pool.allocate(64).value();
  ASSERT_TRUE(first.view()->fill(std::byte{23}).ok());
  auto ready = first.publish().value();
  const auto alias = ready;
  std::array<ReadLease, 2> sources{ready, alias};
  std::array<WriteLease, 2> destinations;
  destinations[0] = pool.allocate(64).value(); destinations[1] = pool.allocate(64).value();
  const auto before = pool.stats();
  std::array<LeaseSlotAddress, 2> inputs{}, outputs{};
  ASSERT_TRUE(layout.bind_indices(sources, destinations, inputs, outputs).ok());
  EXPECT_EQ(inputs[0], inputs[1]); EXPECT_EQ(inputs[0].index, 0u);
  EXPECT_EQ(inputs[0].generation, ready.generation());
  EXPECT_EQ(outputs[0].index, 1u); EXPECT_EQ(outputs[1].index, 2u);
  EXPECT_EQ(outputs[0].generation, destinations[0].generation());
  EXPECT_EQ(outputs[1].generation, destinations[1].generation());
  EXPECT_FALSE(ready.try_write());
  EXPECT_EQ(pool.stats().live_blocks, before.live_blocks);
  EXPECT_EQ(pool.stats().allocated_bytes, before.allocated_bytes);
  // The carrier is bounded by the final actual item, excluding the pool's
  // sentinel byte. It is not an initialized tensor or a whole-bank lease.
  const auto carrier = detail::LeaseAccess::slot_capacity(layout).value();
  EXPECT_EQ(carrier.size(), 576u);
  EXPECT_EQ(carrier.native_handle(), ready.view()->native_handle());
  EXPECT_EQ(carrier.offset() + 256u, destinations[0].view()->offset());
  std::array<std::byte, 64> retained{};
  ASSERT_TRUE(alias.view()->read(retained).ok());
  for (const auto byte : retained) EXPECT_EQ(byte, std::byte{23});
}

TEST(ResourceLeases, FixedSlotsRejectPartialForeignPaddingAndAppendAuthorities) {
  const auto item = TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{16}).value();
  auto pool = LeasePool::create(mem::system_resource(), 1024, {.max_blocks = 4}).value();
  auto layout = LeaseSlotLayout::create(pool, item, 4, 256).value();
  auto oversized = pool.allocate(128).value();
  auto root = oversized.publish().value();
  auto partial = root.subview(0, 64).value();
  EXPECT_TRUE(layout.locate(partial).error().IsInvalidArgument());
  root.reset(); partial.reset();
  auto first = pool.allocate(64).value();
  // A legal allocation in the stride's padding is not a slot, even though
  // its resource, domain, exact size and alignment are otherwise valid.
  auto padding = pool.allocate(64, 64).value();
  EXPECT_TRUE(layout.locate(padding).error().IsOutOfRange());
  padding.reset();
  auto foreign_pool = LeasePool::create(mem::system_resource(), 1024, {.max_blocks = 1}).value();
  auto foreign = foreign_pool.allocate(64).value();
  EXPECT_TRUE(layout.locate(foreign).error().IsInvalidArgument());
  auto appendable = detail::LeaseAccess::publish_appendable(first, 64).value();
  EXPECT_EQ(appendable.prefix.size(), 64u);
  EXPECT_TRUE(layout.locate(appendable.prefix).error().IsFailedPrecondition());
  EXPECT_TRUE(appendable.prefix.valid()); EXPECT_TRUE(foreign.valid());
  EXPECT_TRUE(foreign.view()->fill(std::byte{9}).ok());
}

TEST(ResourceLeases, FixedSlotsValidateWholeBatchBeforeChangingAnyAddress) {
  const auto item = TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{16}).value();
  auto pool = LeasePool::create(mem::system_resource(), 1024, {.max_blocks = 4}).value();
  auto layout = LeaseSlotLayout::create(pool, item, 4, 256).value();
  auto source_writer = pool.allocate(64).value();
  const auto source = source_writer.publish().value();
  const std::array sources{source};
  std::array<WriteLease, 2> destinations;
  destinations[0] = pool.allocate(64).value();
  auto foreign_pool = LeasePool::create(mem::system_resource(), 256, {.max_blocks = 1}).value();
  destinations[1] = foreign_pool.allocate(64).value();
  const LeaseSlotAddress sentinel{91, 12345};
  std::array<LeaseSlotAddress, 1> inputs{sentinel};
  std::array<LeaseSlotAddress, 2> outputs{sentinel, sentinel};
  EXPECT_TRUE(layout.bind_indices(sources, destinations, inputs, outputs).IsInvalidArgument());
  EXPECT_EQ(inputs[0], sentinel); EXPECT_EQ(outputs[0], sentinel); EXPECT_EQ(outputs[1], sentinel);
  EXPECT_TRUE(destinations[0].valid()); EXPECT_TRUE(destinations[1].valid());
  destinations[1] = pool.allocate(64).value();
  std::array<LeaseSlotAddress, 3> overlap{sentinel, sentinel, sentinel};
  EXPECT_TRUE(layout.bind_indices(sources, destinations, std::span(overlap).first(1),
      std::span(overlap).first(2)).IsInvalidArgument());
  for (const auto address : overlap) EXPECT_EQ(address, sentinel);
  EXPECT_TRUE(layout.bind_indices(sources, destinations, {}, outputs).IsInvalidArgument());
  EXPECT_EQ(outputs[0], sentinel); EXPECT_EQ(outputs[1], sentinel);
  auto uses = LeaseUsePool::create({.max_batches = 1, .max_resources_per_batch = 1}).value();
  auto episode = uses.begin().value();
  ASSERT_TRUE(episode.write(destinations[1]).ok());
  PendingUse pending;
  auto done = detail::CompletionAccess::make(&kPending, &pending);
  ASSERT_TRUE(episode.attach(done).ok());
  EXPECT_TRUE(layout.bind_indices(sources, destinations, inputs, outputs).IsUnavailable());
  EXPECT_EQ(inputs[0], sentinel); EXPECT_EQ(outputs[0], sentinel); EXPECT_EQ(outputs[1], sentinel);
  ASSERT_TRUE(done.wait().ok()); done = {};
  ASSERT_TRUE(layout.bind_indices(sources, destinations, inputs, outputs).ok());
  // Native failure is not made ready by knowing its address or generation.
  auto failed_episode = uses.begin().value();
  ASSERT_TRUE(failed_episode.write(destinations[1]).ok());
  auto failed = Completion::completed(hyle::Status::Internal("intentional slot output error"));
  ASSERT_TRUE(failed_episode.attach(failed).ok());
  failed.wait().IgnoreError(); failed = {};
  inputs.fill(sentinel); outputs.fill(sentinel);
  EXPECT_TRUE(layout.bind_indices(sources, destinations, inputs, outputs).IsInternal());
  EXPECT_EQ(inputs[0], sentinel); EXPECT_EQ(outputs[0], sentinel); EXPECT_EQ(outputs[1], sentinel);
}

TEST(ResourceLeases, FixedSlotsPreserveDomainAcrossMoveDropAndGenerationReuse) {
  const auto item = TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{16}).value();
  // Includes the original pool's range/block metadata, not only 512B payload.
  const std::array limits{ResourceBudgetLimit{mem::system_resource(), 65536}};
  auto budget_result = ResourceBudget::create({.limits = limits});
  ASSERT_TRUE(budget_result) << budget_result.error().message();
  auto budget = std::move(*budget_result);
  const auto initial_bytes = budget.usage(mem::system_resource())->live_bytes;
  LeaseSlotLayout retained;
  ReadLease alias;
  {
    auto pool_result = LeasePool::create(mem::system_resource(), 512,
        {.max_blocks = 2, .resource_budget = budget});
    ASSERT_TRUE(pool_result) << pool_result.error().message();
    auto pool = std::move(*pool_result);
    auto layout_result = LeaseSlotLayout::create(pool, item, 2, 256);
    ASSERT_TRUE(layout_result) << layout_result.error().message();
    auto layout = std::move(*layout_result);
    auto writer_result = pool.allocate(64);
    ASSERT_TRUE(writer_result) << writer_result.error().message();
    auto writer = std::move(*writer_result);
    const auto old_result = layout.locate(writer);
    ASSERT_TRUE(old_result) << old_result.error().message();
    const auto old = *old_result;
    writer.reset();
    auto replacement_result = pool.allocate(64);
    ASSERT_TRUE(replacement_result) << replacement_result.error().message();
    writer = std::move(*replacement_result);
    const auto next_result = layout.locate(writer);
    ASSERT_TRUE(next_result) << next_result.error().message();
    const auto next = *next_result;
    EXPECT_EQ(old.index, next.index); EXPECT_GT(next.generation, old.generation);
    auto moved = std::move(pool);
    EXPECT_FALSE(pool.valid());
    const auto moved_pool_address = layout.locate(writer);
    ASSERT_TRUE(moved_pool_address) << moved_pool_address.error().message();
    EXPECT_EQ(*moved_pool_address, next);
    retained = layout;
    LeaseSlotLayout moved_layout(std::move(layout));
    EXPECT_FALSE(layout.valid());
    const auto moved_layout_address = moved_layout.locate(writer);
    ASSERT_TRUE(moved_layout_address) << moved_layout_address.error().message();
    EXPECT_EQ(*moved_layout_address, next);
    auto published = writer.publish();
    ASSERT_TRUE(published) << published.error().message();
    alias = std::move(*published);
    ASSERT_TRUE(moved.close().ok());
    EXPECT_TRUE(LeaseSlotLayout::create(moved, item, 2, 256).error().IsFailedPrecondition());
    const auto closed_address = retained.locate(alias);
    ASSERT_TRUE(closed_address) << closed_address.error().message();
    EXPECT_EQ(*closed_address, next); // Address query, not admission.
  }
  EXPECT_TRUE(retained.locate(alias));
  auto uses_result = LeaseUsePool::create({.max_batches = 1, .max_resources_per_batch = 1});
  ASSERT_TRUE(uses_result) << uses_result.error().message();
  auto uses = std::move(*uses_result);
  auto episode_result = uses.begin();
  ASSERT_TRUE(episode_result) << episode_result.error().message();
  auto episode = std::move(*episode_result);
  EXPECT_TRUE(episode.read(alias).IsFailedPrecondition());
  auto replacement_pool = LeasePool::create(mem::system_resource(), 512, {.max_blocks = 1});
  ASSERT_TRUE(replacement_pool) << replacement_pool.error().message();
  auto replacement = std::move(*replacement_pool);
  auto other_result = replacement.allocate(64);
  ASSERT_TRUE(other_result) << other_result.error().message();
  auto other = std::move(*other_result);
  EXPECT_TRUE(retained.locate(other).error().IsInvalidArgument());
  alias.reset();
  EXPECT_GT(budget.usage(mem::system_resource())->live_bytes, initial_bytes);
  retained = {};
  EXPECT_EQ(budget.usage(mem::system_resource())->live_bytes, initial_bytes);
}

TEST(ResourceLeases, FixedSlotGeometryRejectsOverflowAndInsufficientBacking) {
  const auto item = TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{16}).value();
  auto pool = LeasePool::create(mem::system_resource(), 512, {.max_blocks = 2}).value();
  EXPECT_TRUE(LeaseSlotLayout::create({}, item, 1, 256).error().IsFailedPrecondition());
  EXPECT_TRUE(LeaseSlotLayout::create(pool, item, 0, 256).error().IsInvalidArgument());
  EXPECT_TRUE(LeaseSlotLayout::create(pool, item, 2, 32).error().IsInvalidArgument());
  EXPECT_TRUE(LeaseSlotLayout::create(pool, item, 2, 65).error().IsInvalidArgument());
  EXPECT_TRUE(LeaseSlotLayout::create(pool, item, 3, 256).error().IsOutOfRange());
  const auto large_stride = (std::numeric_limits<std::size_t>::max)() & ~std::size_t{3};
  EXPECT_TRUE(LeaseSlotLayout::create(pool, item, 3, large_stride).error().IsOutOfRange());
  EXPECT_EQ(pool.stats().live_blocks, 0u);
}

TEST(ResourceLeases, FreezeFullPoolKeepsBudgetAndFutureReadonlyAdmission) {
  const std::array limits{ResourceBudgetLimit{mem::system_resource(), 65536}};
  auto budget = ResourceBudget::create({.limits = limits}).value();
  const auto baseline = budget.usage(mem::system_resource())->live_bytes;
  auto pool = LeasePool::create(mem::system_resource(), 512,
      {.max_blocks = 1, .resource_budget = budget}).value();
  auto writer = pool.allocate(512).value();
  ASSERT_TRUE(writer.view()->fill(std::byte{37}).ok());
  const auto generation = writer.generation();
  auto frozen = pool.freeze(std::move(writer)); ASSERT_TRUE(frozen);
  EXPECT_FALSE(pool.valid()); EXPECT_FALSE(writer.valid());
  EXPECT_EQ(frozen->generation(), generation);
  EXPECT_FALSE(frozen->try_write());
  EXPECT_EQ(budget.usage(mem::system_resource())->live_by_category[
      static_cast<std::size_t>(ResourceBudgetCategory::kExternal)], 512u);
  auto uses = LeaseUsePool::create({.max_batches = 1, .max_resources_per_batch = 1}).value();
  auto episode = uses.begin().value();
  ASSERT_TRUE(episode.read(*frozen).ok()); // New use after pool-root relinquishment.
  auto complete = Completion::completed();
  ASSERT_TRUE(episode.attach(complete).ok());
  ASSERT_TRUE(complete.wait().ok());
  std::array<std::byte, 512> bytes{};
  ASSERT_TRUE(frozen->view()->read(bytes).ok());
  for (auto byte : bytes) EXPECT_EQ(byte, std::byte{37});
  frozen->reset();
  EXPECT_EQ(budget.usage(mem::system_resource())->live_bytes, baseline);
}

TEST(ResourceLeases, FreezeRejectsForeignAndPartialWritersWithoutMutation) {
  auto pool = LeasePool::create(mem::system_resource(), 512, {.max_blocks = 2}).value();
  auto partial = pool.allocate(256).value();
  auto other = LeasePool::create(mem::system_resource(), 512, {.max_blocks = 1}).value();
  auto foreign = other.allocate(512).value();
  const auto partial_generation = partial.generation(), foreign_generation = foreign.generation();
  EXPECT_FALSE(pool.freeze(std::move(foreign)));
  EXPECT_TRUE(pool.valid()); EXPECT_TRUE(other.valid()); EXPECT_TRUE(foreign.valid());
  EXPECT_EQ(foreign.generation(), foreign_generation);
  EXPECT_FALSE(pool.freeze(std::move(partial)));
  EXPECT_TRUE(partial.valid()); EXPECT_EQ(partial.generation(), partial_generation);
  EXPECT_TRUE(partial.view()->fill(std::byte{19}).ok());
  auto legacy = partial.publish(); ASSERT_TRUE(legacy);
  EXPECT_TRUE(legacy->try_write()); // Rejection did not permanently freeze it.
}

TEST(ResourceLeases, FreezeWaitsForActualWritesAndPreservesFailure) {
  auto pool = LeasePool::create(mem::system_resource(), 512, {.max_blocks = 1}).value();
  auto writer = pool.allocate(512).value();
  auto uses = LeaseUsePool::create({.max_batches = 1, .max_resources_per_batch = 1}).value();
  auto episode = uses.begin().value(); ASSERT_TRUE(episode.write(writer).ok());
  PendingUse pending;
  auto completion = detail::CompletionAccess::make(&kPending, &pending);
  ASSERT_TRUE(episode.attach(completion).ok());
  auto rejected = pool.freeze(std::move(writer));
  ASSERT_FALSE(rejected); EXPECT_TRUE(rejected.error().IsUnavailable());
  EXPECT_TRUE(pool.valid()); EXPECT_TRUE(writer.valid());
  ASSERT_TRUE(completion.wait().ok());
  auto frozen = pool.freeze(std::move(writer)); ASSERT_TRUE(frozen);
  EXPECT_FALSE(frozen->try_write());

  auto failed_pool = LeasePool::create(mem::system_resource(), 512, {.max_blocks = 1}).value();
  auto failed_writer = failed_pool.allocate(512).value();
  auto failed_use = uses.begin().value(); ASSERT_TRUE(failed_use.write(failed_writer).ok());
  auto failed = Completion::completed(hyle::Status::Internal("test original write failure"));
  ASSERT_TRUE(failed_use.attach(failed).ok());
  auto failed_freeze = failed_pool.freeze(std::move(failed_writer));
  ASSERT_FALSE(failed_freeze); EXPECT_TRUE(failed_freeze.error().IsInternal());
  EXPECT_TRUE(failed_pool.valid()); EXPECT_TRUE(failed_writer.valid());
}

TEST(AppendLeases, BoundedPrefixCannotWidenOrPromoteAndFullPageHasNoTail) {
  auto pool = LeasePool::create(mem::system_resource(), 1024, {.max_blocks = 3}).value();
  auto writer = pool.allocate(256).value();
  ASSERT_TRUE(writer.view()->fill(std::byte{37}).ok());
  const auto generation = writer.generation();
  EXPECT_FALSE(detail::LeaseAccess::publish_appendable(writer, 257).has_value());
  ASSERT_TRUE(writer.valid()); // rejected publication cannot consume its owner
  auto page = detail::LeaseAccess::publish_appendable(writer, 64).value();
  EXPECT_FALSE(writer.valid());
  EXPECT_EQ(page.prefix.size(), 64u);
  EXPECT_EQ(page.prefix.generation(), generation);
  EXPECT_EQ(page.appender.generation(), generation);
  EXPECT_EQ(page.appender.committed_bytes(), 64u);
  EXPECT_EQ(page.appender.capacity_bytes(), 256u);
  EXPECT_TRUE(detail::LeaseAccess::is_appendable_prefix(page.prefix));
  EXPECT_TRUE(detail::LeaseAccess::validate_prefix(page.appender, page.prefix).ok());
  auto retained = page.prefix;
  EXPECT_FALSE(retained.subview(0, 65).has_value());
  EXPECT_FALSE(retained.subview(63, 2).has_value());
  EXPECT_FALSE(retained.try_write().has_value());
  page.prefix.reset();
  page.appender.reset();
  EXPECT_FALSE(retained.try_write().has_value()); // sole reader still cannot widen authority

  auto full_writer = pool.allocate(256).value();
  auto full = detail::LeaseAccess::publish_appendable(full_writer, 256).value();
  EXPECT_EQ(full.prefix.size(), 256u);
  EXPECT_FALSE(full.appender.valid());
  EXPECT_TRUE(detail::LeaseAccess::is_appendable_prefix(full.prefix));
  EXPECT_FALSE(full.prefix.try_write().has_value());

  auto legacy_writer = pool.allocate(256).value();
  auto legacy = legacy_writer.publish().value();
  EXPECT_FALSE(detail::LeaseAccess::is_appendable_prefix(legacy));
  auto legacy_piece = legacy.subview(0, 64).value();
  EXPECT_FALSE(detail::LeaseAccess::is_appendable_prefix(legacy_piece));
}

TEST(AppendLeases, ReservedSuffixRequiresQualifiedUseAndKeepsOldReaderBounded) {
  auto pool = LeasePool::create(mem::system_resource(), 256, {.max_blocks = 1}).value();
  auto writer = pool.allocate(256).value();
  ASSERT_TRUE(writer.view()->fill(std::byte{37}).ok());
  auto page = detail::LeaseAccess::publish_appendable(writer, 64).value();
  auto uses = LeaseUsePool::create({.max_batches = 2, .max_resources_per_batch = 1}).value();
  auto prior = uses.begin().value();
  ASSERT_TRUE(prior.read(page.prefix).ok());
  EXPECT_TRUE(page.appender.reserve(32).IsUnavailable());
  auto prior_done = Completion::completed();
  ASSERT_TRUE(prior.attach(prior_done).ok());
  EXPECT_TRUE(page.appender.reserve(0).IsOutOfRange());
  EXPECT_TRUE(page.appender.reserve(193).IsOutOfRange());
  EXPECT_EQ(page.appender.epoch(), 0u);
  ASSERT_TRUE(page.appender.reserve(32).ok());
  const auto suffix = page.appender.view().value(); // view before F1 native admission
  EXPECT_EQ(suffix.size(), 32u);
  EXPECT_EQ(suffix.offset(), page.prefix.view()->offset() + 64);
  EXPECT_EQ(suffix.native_handle(), page.prefix.view()->native_handle());
  ASSERT_TRUE(suffix.fill(std::byte{9}).ok());
  auto rejected = uses.begin().value();
  EXPECT_TRUE(rejected.read(page.prefix).IsUnavailable());
  EXPECT_TRUE(detail::LeaseAccess::read_unique(rejected, page.prefix).IsUnavailable());
  auto qualified = uses.begin().value();
  ASSERT_TRUE(detail::LeaseAccess::append_unique(qualified, page.appender).ok());
  EXPECT_TRUE(page.appender.rollback().IsUnavailable());
  EXPECT_FALSE(page.appender.view().has_value());
  EXPECT_FALSE(page.appender.publish().has_value());
  PendingUse pending;
  auto completion = detail::CompletionAccess::make(&kPending, &pending);
  ASSERT_TRUE(qualified.attach(completion).ok());
  EXPECT_EQ(completion.poll().value(), CompletionPoll::kPending);
  ASSERT_TRUE(completion.wait().ok());
  auto widened = page.appender.publish().value();
  EXPECT_EQ(widened.size(), 96u);
  EXPECT_EQ(page.prefix.size(), 64u);
  EXPECT_FALSE(page.prefix.subview(0, 96).has_value());
  EXPECT_EQ(widened.generation(), page.prefix.generation());
  EXPECT_EQ(page.appender.epoch(), 1u);
  std::array<std::byte, 96> actual{};
  ASSERT_TRUE(widened.view()->read(actual).ok());
  for (std::size_t i = 0; i < actual.size(); ++i)
    EXPECT_EQ(actual[i], i < 64 ? std::byte{37} : std::byte{9});
  EXPECT_TRUE(rejected.read(page.prefix).ok()); // reservation ended only after successful publish
}

TEST(AppendLeases, FailedEpochWaitsForActualReleaseThenRollbackCanRetryAndClose) {
  auto pool = LeasePool::create(mem::system_resource(), 256, {.max_blocks = 1}).value();
  auto writer = pool.allocate(256).value();
  ASSERT_TRUE(writer.view()->fill(std::byte{19}).ok());
  auto page = detail::LeaseAccess::publish_appendable(writer, 64).value();
  const auto generation = page.prefix.generation();
  auto uses = LeaseUsePool::create({.max_batches = 1, .max_resources_per_batch = 1}).value();
  ASSERT_TRUE(page.appender.reserve(32).ok());
  ASSERT_TRUE(page.appender.view()->fill(std::byte{47}).ok());
  PendingUse pending;
  {
    auto batch = uses.begin().value();
    ASSERT_TRUE(detail::LeaseAccess::append_unique(batch, page.appender).ok());
    auto completion = detail::CompletionAccess::make(&kPending, &pending);
    ASSERT_TRUE(batch.attach(completion).ok());
    detail::CompletionAccess::append_terminal(completion, hyle::Status::Cancelled("later graph node failed"));
    // Dropping a failed public handle transfers its original retention, not retirement.
  }
  ASSERT_TRUE(pending.retained.valid());
  EXPECT_TRUE(pending.hint.IsCancelled());
  EXPECT_EQ(uses.active_batches(), 1u);
  EXPECT_EQ(pool.stats().active_uses, 1u);
  EXPECT_TRUE(page.appender.rollback().IsUnavailable());
  EXPECT_EQ(page.appender.committed_bytes(), 64u);
  std::array<std::byte, 64> prefix{};
  EXPECT_TRUE(page.prefix.view()->read(prefix).ok());
  for (auto value : prefix) EXPECT_EQ(value, std::byte{19});
  pending.done = true;
  pending.retained.terminal(pending.retained.state, pending.hint);
  // Even terminal failure notification alone cannot release the admitted use.
  EXPECT_TRUE(page.appender.rollback().IsUnavailable());
  pending.retained.release(pending.retained.state);
  pending.retained = {};
  EXPECT_EQ(uses.active_batches(), 0u);
  EXPECT_EQ(pool.stats().active_uses, 0u);
  auto poisoned = page.appender.publish();
  ASSERT_FALSE(poisoned.has_value());
  EXPECT_TRUE(poisoned.error().IsCancelled());
  ASSERT_TRUE(page.appender.rollback().ok());
  auto prefix_use = uses.begin().value();
  ASSERT_TRUE(prefix_use.read(page.prefix).ok()); // failed tail did not poison the committed root
  auto prefix_done = Completion::completed();
  ASSERT_TRUE(prefix_use.attach(prefix_done).ok());
  ASSERT_TRUE(page.appender.reserve(32).ok());
  EXPECT_EQ(page.appender.epoch(), 2u);
  ASSERT_TRUE(page.appender.view()->fill(std::byte{71}).ok());
  auto retry = uses.begin().value();
  ASSERT_TRUE(detail::LeaseAccess::append_unique(retry, page.appender).ok());
  auto retry_done = Completion::completed();
  ASSERT_TRUE(retry.attach(retry_done).ok());
  auto committed = page.appender.publish().value();
  EXPECT_EQ(committed.size(), 96u);
  EXPECT_EQ(committed.generation(), generation);
  EXPECT_EQ(page.prefix.size(), 64u);
  std::array<std::byte, 96> actual{};
  ASSERT_TRUE(committed.view()->read(actual).ok());
  for (std::size_t i = 0; i < actual.size(); ++i)
    EXPECT_EQ(actual[i], i < 64 ? std::byte{19} : std::byte{71});
  ASSERT_TRUE(page.appender.reserve(16).ok());
  ASSERT_TRUE(pool.close().ok());
  EXPECT_FALSE(page.appender.view().has_value());
  EXPECT_FALSE(page.appender.publish().has_value());
  ASSERT_TRUE(page.appender.rollback().ok()); // closed domain permits only retired cleanup
  EXPECT_TRUE(page.appender.reserve(16).IsFailedPrecondition());
  EXPECT_TRUE(page.prefix.view().has_value());
}

TEST(AppendLeases, AbandonedTailWaitsForOriginalNativeRetirement) {
  auto pool = LeasePool::create(mem::system_resource(), 256, {.max_blocks = 1}).value();
  auto writer = pool.allocate(256).value();
  auto page = detail::LeaseAccess::publish_appendable(writer, 64).value();
  auto uses = LeaseUsePool::create({.max_batches = 2, .max_resources_per_batch = 1}).value();
  ASSERT_TRUE(page.appender.reserve(32).ok());
  PendingUse pending;
  {
    auto batch = uses.begin().value();
    ASSERT_TRUE(detail::LeaseAccess::append_unique(batch, page.appender).ok());
    auto completion = detail::CompletionAccess::make(&kPending, &pending);
    ASSERT_TRUE(batch.attach(completion).ok());
    page.appender.reset();
  }
  ASSERT_TRUE(pending.retained.valid());
  auto reader = uses.begin().value();
  EXPECT_TRUE(reader.read(page.prefix).IsUnavailable());
  EXPECT_EQ(pool.stats().live_blocks, 1u);
  EXPECT_EQ(pool.stats().active_uses, 1u);
  pending.done = true;
  pending.retained.terminal(pending.retained.state, hyle::Status::Ok());
  pending.retained.release(pending.retained.state);
  pending.retained = {};
  EXPECT_EQ(pool.stats().active_uses, 0u);
  EXPECT_TRUE(reader.read(page.prefix).ok());
  EXPECT_EQ(page.prefix.size(), 64u);
  EXPECT_FALSE(page.prefix.try_write().has_value());
}

// These are resource handoff protocol tests, not a native ordering proof.
// The original F2 producer/consumer still supplies the real device dependency.
static_assert(std::is_nothrow_copy_constructible_v<detail::PendingReadTicket>);
static_assert(std::is_nothrow_move_constructible_v<detail::PendingReadTicket>);
static_assert(!std::is_convertible_v<detail::PendingReadTicket, ReadLease>);
static_assert(!std::is_convertible_v<detail::PendingReadTicket, WriteLease>);

TEST(PendingReadTickets, ExactRangeSurvivesWriterObjectAndGenuinePublication) {
  auto pool = LeasePool::create(mem::system_resource(), 1024, {.max_blocks = 3}).value();
  auto uses = LeaseUsePool::create({.max_batches = 3, .max_resources_per_batch = 1}).value();
  detail::PendingReadTicket ticket;
  WriteLease returned;
  mem::ConstBufferView original;
  PendingUse producing, consuming;
  Completion producer;
  {
    auto writer = pool.allocate(64).value();
    ASSERT_TRUE(writer.view()->fill(std::byte{37}).ok());
    original = mem::ConstBufferView(*writer.view());
    auto batch = uses.begin().value();
    ASSERT_TRUE(batch.write(writer).ok());
    producer = detail::CompletionAccess::make(&kPending, &producing);
    ASSERT_TRUE(batch.attach(producer).ok());
    auto exported = detail::LeaseAccess::pending_read(writer);
    ASSERT_TRUE(exported) << exported.error().message();
    ticket = std::move(*exported);
    EXPECT_EQ(ticket.size(), 64u);
    EXPECT_EQ(ticket.generation(), writer.generation());
    returned = std::move(writer);
  } // The original WriteLease object no longer exists.
  auto copy = ticket;
  detail::PendingReadTicket moved(std::move(copy));
  EXPECT_FALSE(copy.valid()); EXPECT_EQ(copy.size(), 0u); EXPECT_EQ(copy.generation(), 0u);
  ASSERT_TRUE(detail::LeaseAccess::matches_pending_read(moved, original).ok());
  auto other = pool.allocate(64).value();
  const auto other_view = mem::ConstBufferView(*other.view());
  EXPECT_TRUE(detail::LeaseAccess::matches_pending_read(ticket, other_view).IsInvalidArgument());
  EXPECT_TRUE(detail::LeaseAccess::matches_pending_read(ticket,
      original.subview(0, 63).value()).IsInvalidArgument());
  EXPECT_TRUE(detail::LeaseAccess::matches_pending_read(ticket,
      detail::LeaseAccess::backing_span(pool).value()).IsInvalidArgument());
  EXPECT_TRUE(detail::LeaseAccess::matches_pending_read({}, original).IsFailedPrecondition());
  auto reader = uses.begin().value();
  ASSERT_TRUE(detail::LeaseAccess::ordered_read_unique(reader, moved).ok());
  auto consumer = detail::CompletionAccess::make(&kPending, &consuming);
  ASSERT_TRUE(reader.attach(consumer).ok());
  EXPECT_TRUE(detail::LeaseAccess::publish_after_writes(returned).error().IsUnavailable());
  ASSERT_TRUE(producer.wait().ok());
  EXPECT_EQ(pool.stats().active_uses, 1u);
  EXPECT_FALSE(returned.publish()); // The ordinary all-reference rule remains.
  auto published = detail::LeaseAccess::publish_after_writes(returned);
  ASSERT_TRUE(published) << published.error().message();
  auto ready = std::move(*published);
  EXPECT_FALSE(returned.valid());
  EXPECT_TRUE(detail::LeaseAccess::matches_pending_read(ticket, original).ok());
  // The same ticket also works after genuine publication; it never consults a
  // moved/taken writer object or synthesizes another ready permission.
  auto later_reader = uses.begin().value();
  ASSERT_TRUE(detail::LeaseAccess::ordered_read_unique(later_reader, ticket).ok());
  auto later = Completion::completed();
  ASSERT_TRUE(later_reader.attach(later).ok());
  ASSERT_TRUE(later.wait().ok());
  EXPECT_FALSE(ready.try_write());
  EXPECT_EQ(*consumer.poll(), CompletionPoll::kPending);
  ASSERT_TRUE(consumer.wait().ok());
  EXPECT_FALSE(ready.try_write()); // Tickets themselves still retain this block.
  ticket.reset(); moved.reset();
  std::array<std::byte, 64> bytes{};
  ASSERT_TRUE(ready.view()->read(bytes).ok());
  for (const auto byte : bytes) EXPECT_EQ(byte, std::byte{37});
  auto writable = ready.try_write();
  ASSERT_TRUE(writable) << writable.error().message();
  EXPECT_TRUE(writable->view());
}

TEST(PendingReadTickets, ExportSealsNewWritesUntilUniqueReadyPromotionWithoutAllocation) {
  const std::array limits{ResourceBudgetLimit{mem::system_resource(), 65536}};
  auto budget = ResourceBudget::create({.limits = limits}).value();
  auto pool = LeasePool::create(mem::system_resource(), 256,
      {.max_blocks = 1, .resource_budget = budget}).value();
  auto writer = pool.allocate(64).value();
  auto uses = LeaseUsePool::create({.max_batches = 1, .max_resources_per_batch = 1}).value();
  auto original_write = uses.begin().value();
  ASSERT_TRUE(original_write.write(writer).ok());
  auto producer = Completion::completed();
  ASSERT_TRUE(original_write.attach(producer).ok());
  ASSERT_TRUE(producer.wait().ok()); // Fast-complete producers must also export.
  const auto usage_before = budget.usage(mem::system_resource())->live_bytes;
  const auto blocks_before = pool.stats().live_blocks;
  auto exported = detail::LeaseAccess::pending_read(writer);
  ASSERT_TRUE(exported) << exported.error().message();
  auto ticket = std::move(*exported);
  auto copy = ticket;
  detail::PendingReadTicket assigned;
  assigned = copy;
  EXPECT_EQ(budget.usage(mem::system_resource())->live_bytes, usage_before);
  EXPECT_EQ(pool.stats().live_blocks, blocks_before);
  auto rejected = uses.begin().value();
  EXPECT_TRUE(rejected.write(writer).IsFailedPrecondition());
  EXPECT_TRUE(detail::LeaseAccess::write_unique(rejected, writer, false).IsFailedPrecondition());
  EXPECT_TRUE(detail::LeaseAccess::write_unique(rejected, writer, true).IsFailedPrecondition());
  EXPECT_FALSE(writer.view());
  EXPECT_FALSE(detail::LeaseAccess::ordered_view(writer));
  ticket.reset(); copy.reset(); assigned.reset();
  // Even after every ticket disappears, the original writer cannot silently
  // start a new native write or hand out a mutable address in the sealed epoch.
  EXPECT_TRUE(rejected.write(writer).IsFailedPrecondition());
  EXPECT_TRUE(detail::LeaseAccess::write_unique(rejected, writer, true).IsFailedPrecondition());
  EXPECT_FALSE(writer.view());
  EXPECT_FALSE(detail::LeaseAccess::ordered_view(writer));
  EXPECT_EQ(pool.stats().active_uses, 0u);
  auto ready = writer.publish();
  ASSERT_TRUE(ready) << ready.error().message();
  auto next = ready->try_write();
  ASSERT_TRUE(next) << next.error().message();
  EXPECT_TRUE(next->view());
  ASSERT_TRUE(rejected.write(*next).ok());
  rejected = {}; // Unsubmitted use restores the original normal mutable state.
  auto whole = next->publish().value();
  auto slice = whole.subview(16, 32).value(); whole.reset();
  auto partial_writer = slice.try_write().value();
  const auto rejected_partial = detail::LeaseAccess::pending_read(partial_writer);
  ASSERT_FALSE(rejected_partial);
  EXPECT_TRUE(rejected_partial.error().IsInvalidArgument());
  EXPECT_TRUE(partial_writer.view()); // Rejection did not seal or consume it.
}

TEST(PendingReadTickets, ReaderFailureDoesNotPoisonProducerAndClosedDomainsRejectNewUse) {
  auto pool = LeasePool::create(mem::system_resource(), 512, {.max_blocks = 2}).value();
  auto writer = pool.allocate(64).value();
  auto bad_writer = pool.allocate(64).value();
  ASSERT_TRUE(writer.view()->fill(std::byte{19}).ok());
  const auto original = mem::ConstBufferView(*writer.view());
  auto uses = LeaseUsePool::create({.max_batches = 3, .max_resources_per_batch = 1}).value();
  PendingUse producing, bad_producing, consuming;
  auto produced = uses.begin().value(), bad_produced = uses.begin().value();
  ASSERT_TRUE(produced.write(writer).ok()); ASSERT_TRUE(bad_produced.write(bad_writer).ok());
  auto producer = detail::CompletionAccess::make(&kPending, &producing);
  auto bad_producer = detail::CompletionAccess::make(&kPending, &bad_producing);
  ASSERT_TRUE(produced.attach(producer).ok()); ASSERT_TRUE(bad_produced.attach(bad_producer).ok());
  auto ticket = detail::LeaseAccess::pending_read(writer).value();
  auto bad_ticket = detail::LeaseAccess::pending_read(bad_writer).value();
  auto reading = uses.begin().value();
  ASSERT_TRUE(detail::LeaseAccess::ordered_read_unique(reading, ticket).ok());
  auto consumer = detail::CompletionAccess::make(&kPending, &consuming);
  ASSERT_TRUE(reading.attach(consumer).ok());
  detail::CompletionAccess::append_terminal(consumer, hyle::Status::Cancelled("ticket reader cancelled"));
  ASSERT_TRUE(producer.wait().ok());
  auto published = detail::LeaseAccess::publish_after_writes(writer);
  ASSERT_TRUE(published) << published.error().message();
  auto ready = std::move(*published);
  EXPECT_EQ(*consumer.poll(), CompletionPoll::kPending);
  EXPECT_TRUE(consumer.wait().IsCancelled()); // A read failure is not a write failure.
  std::array<std::byte, 64> actual{};
  ASSERT_TRUE(ready.view()->read(actual).ok());
  for (const auto byte : actual) EXPECT_EQ(byte, std::byte{19});
  detail::CompletionAccess::append_terminal(bad_producer, hyle::Status::Internal("ticket producer failed"));
  EXPECT_TRUE(detail::LeaseAccess::publish_after_writes(bad_writer).error().IsUnavailable());
  EXPECT_TRUE(bad_producer.wait().IsInternal());
  auto failed_publication = detail::LeaseAccess::publish_after_writes(bad_writer);
  ASSERT_FALSE(failed_publication);
  EXPECT_TRUE(failed_publication.error().IsInternal());
  auto rejected = uses.begin().value();
  EXPECT_TRUE(detail::LeaseAccess::ordered_read_unique(rejected, bad_ticket).IsInternal());
  EXPECT_TRUE(detail::LeaseAccess::pending_read(bad_writer).error().IsInternal());
  ASSERT_TRUE(pool.close().ok());
  EXPECT_TRUE(detail::LeaseAccess::matches_pending_read(ticket, original).ok()); // Identity, not admission.
  EXPECT_TRUE(detail::LeaseAccess::ordered_read_unique(rejected, ticket).IsFailedPrecondition());
  EXPECT_EQ(pool.stats().active_uses, 0u);
}

// Append tickets prove the original epoch and bounded prefix, not readiness.
// Native dependency ordering remains the responsibility of the F2 owner.
static_assert(std::is_nothrow_copy_constructible_v<detail::PendingAppendReadTicket>);
static_assert(std::is_nothrow_move_constructible_v<detail::PendingAppendReadTicket>);
static_assert(!std::is_convertible_v<detail::PendingAppendReadTicket, ReadLease>);

TEST(PendingAppendReadTickets, ExactEpochSurvivesMovedOwnerAndPublishesWithReaderPending) {
  auto pool = LeasePool::create(mem::system_resource(), 512, {.max_blocks = 2}).value();
  auto writer = pool.allocate(256).value();
  ASSERT_TRUE(writer.view()->fill(std::byte{19}).ok());
  const auto capacity = mem::ConstBufferView(*writer.view());
  const auto exact = capacity.subview(0, 96).value();
  auto page = detail::LeaseAccess::publish_appendable(writer, 64).value();
  ASSERT_TRUE(page.appender.reserve(32).ok());
  ASSERT_TRUE(page.appender.view()->fill(std::byte{47}).ok());
  auto uses = LeaseUsePool::create({.max_batches = 3, .max_resources_per_batch = 1}).value();
  auto writing = uses.begin().value();
  ASSERT_TRUE(detail::LeaseAccess::append_unique(writing, page.appender).ok());
  PendingUse producing, consuming;
  auto producer = detail::CompletionAccess::make(&kPending, &producing);
  ASSERT_TRUE(writing.attach(producer).ok());
  auto exported = detail::LeaseAccess::pending_append_read(page.appender, exact);
  ASSERT_TRUE(exported) << exported.error().message();
  auto ticket = std::move(*exported);
  EXPECT_EQ(ticket.size(), 96u); EXPECT_EQ(ticket.committed_bytes(), 64u);
  EXPECT_EQ(ticket.reserved_bytes(), 32u); EXPECT_EQ(ticket.epoch(), 1u);
  auto copy = ticket;
  detail::PendingAppendReadTicket moved(std::move(copy));
  EXPECT_FALSE(copy.valid()); EXPECT_EQ(copy.size(), 0u);
  auto returned = std::move(page.appender);
  EXPECT_TRUE(detail::LeaseAccess::matches_pending_append_read(moved, returned).ok());
  EXPECT_TRUE(detail::LeaseAccess::matches_pending_append_read(ticket, exact).ok());
  EXPECT_FALSE(detail::LeaseAccess::matches_pending_append_read(ticket, capacity).ok());
  EXPECT_FALSE(detail::LeaseAccess::matches_pending_append_read(ticket,
      capacity.subview(0, 64).value()).ok());
  auto other_writer = pool.allocate(256).value();
  const auto other_range = mem::ConstBufferView(*other_writer.view()).subview(0, 96).value();
  EXPECT_FALSE(detail::LeaseAccess::matches_pending_append_read(ticket, other_range).ok());
  auto other = detail::LeaseAccess::publish_appendable(other_writer, 64).value();
  ASSERT_TRUE(other.appender.reserve(32).ok());
  EXPECT_FALSE(detail::LeaseAccess::matches_pending_append_read(ticket, other.appender).ok());
  auto reading = uses.begin().value();
  ASSERT_TRUE(detail::LeaseAccess::ordered_read_unique(reading, moved).ok());
  auto consumer = detail::CompletionAccess::make(&kPending, &consuming);
  ASSERT_TRUE(reading.attach(consumer).ok());
  EXPECT_TRUE(detail::LeaseAccess::publish_append_after_writes(returned, ticket).error().IsUnavailable());
  ASSERT_TRUE(producer.wait().ok());
  EXPECT_EQ(pool.stats().active_uses, 1u);
  EXPECT_FALSE(returned.publish()); // The public all-native-use gate remains.
  auto publication = detail::LeaseAccess::publish_append_after_writes(returned, ticket);
  ASSERT_TRUE(publication) << publication.error().message();
  auto ready = std::move(*publication);
  EXPECT_EQ(ready.size(), 96u); EXPECT_EQ(page.prefix.size(), 64u);
  EXPECT_EQ(returned.committed_bytes(), 96u); EXPECT_EQ(returned.reserved_bytes(), 0u);
  EXPECT_TRUE(returned.reserve(16).IsUnavailable());
  EXPECT_TRUE(returned.rollback().IsUnavailable());
  EXPECT_EQ(*consumer.poll(), CompletionPoll::kPending);
  ASSERT_TRUE(consumer.wait().ok());
  std::array<std::byte, 96> bytes{};
  ASSERT_TRUE(ready.view()->read(bytes).ok());
  for (std::size_t i = 0; i < bytes.size(); ++i)
    EXPECT_EQ(bytes[i], i < 64 ? std::byte{19} : std::byte{47});
  auto later = uses.begin().value();
  ASSERT_TRUE(detail::LeaseAccess::ordered_read_unique(later, ticket).ok());
  later = {}; // No accepted native work; normal admission rollback.
  ASSERT_TRUE(returned.reserve(16).ok());
  EXPECT_EQ(returned.epoch(), 2u);
  EXPECT_FALSE(detail::LeaseAccess::matches_pending_append_read(ticket, exact).ok());
  auto stale = uses.begin().value();
  EXPECT_FALSE(detail::LeaseAccess::ordered_read_unique(stale, moved).ok());
  EXPECT_FALSE(detail::LeaseAccess::publish_append_after_writes(returned, ticket));
  EXPECT_EQ(page.prefix.size(), 64u); EXPECT_EQ(ready.size(), 96u);
  ASSERT_TRUE(returned.rollback().ok());
}

TEST(PendingAppendReadTickets, RejectedExportDoesNotSealAndDroppingTicketsDoesNotReopenEpoch) {
  const std::array limits{ResourceBudgetLimit{mem::system_resource(), 65536}};
  auto budget = ResourceBudget::create({.limits = limits}).value();
  auto pool = LeasePool::create(mem::system_resource(), 256,
      {.max_blocks = 1, .resource_budget = budget}).value();
  auto writer = pool.allocate(256).value();
  const auto capacity = mem::ConstBufferView(*writer.view());
  auto page = detail::LeaseAccess::publish_appendable(writer, 64).value();
  ASSERT_TRUE(page.appender.reserve(32).ok());
  EXPECT_FALSE(detail::LeaseAccess::pending_append_read(page.appender, capacity));
  EXPECT_TRUE(page.appender.view());
  auto uses = LeaseUsePool::create({.max_batches = 1, .max_resources_per_batch = 1}).value();
  auto writing = uses.begin().value();
  ASSERT_TRUE(detail::LeaseAccess::append_unique(writing, page.appender).ok());
  auto producer = Completion::completed();
  ASSERT_TRUE(writing.attach(producer).ok()); ASSERT_TRUE(producer.wait().ok());
  const auto before = budget.usage(mem::system_resource())->live_bytes;
  auto ticket = detail::LeaseAccess::pending_append_read(page.appender,
      capacity.subview(0, 96).value()).value();
  auto copy = ticket;
  detail::PendingAppendReadTicket assigned; assigned = copy;
  EXPECT_EQ(budget.usage(mem::system_resource())->live_bytes, before);
  ticket.reset(); copy.reset(); assigned.reset();
  EXPECT_FALSE(page.appender.view());
  auto rejected = uses.begin().value();
  EXPECT_FALSE(detail::LeaseAccess::append_unique(rejected, page.appender).ok());
  // Genuine public publication still works once every native use retired.
  auto ready = page.appender.publish();
  ASSERT_TRUE(ready) << ready.error().message();
  ASSERT_TRUE(page.appender.reserve(16).ok());
  EXPECT_TRUE(page.appender.view());
  ASSERT_TRUE(page.appender.rollback().ok());
}

TEST(PendingAppendReadTickets, FailedProducerCannotPublishAndReaderFailureDoesNotPoisonAppend) {
  auto pool = LeasePool::create(mem::system_resource(), 256, {.max_blocks = 1}).value();
  auto writer = pool.allocate(256).value();
  ASSERT_TRUE(writer.view()->fill(std::byte{19}).ok());
  const auto capacity = mem::ConstBufferView(*writer.view());
  auto page = detail::LeaseAccess::publish_appendable(writer, 64).value();
  auto uses = LeaseUsePool::create({.max_batches = 2, .max_resources_per_batch = 1}).value();
  for (const bool fail_producer : {false, true}) {
    ASSERT_TRUE(page.appender.reserve(32).ok());
    const auto boundary = page.appender.committed_bytes() + 32;
    ASSERT_TRUE(page.appender.view()->fill(std::byte{47}).ok());
    auto writing = uses.begin().value();
    ASSERT_TRUE(detail::LeaseAccess::append_unique(writing, page.appender).ok());
    PendingUse producing, consuming;
    auto producer = detail::CompletionAccess::make(&kPending, &producing);
    ASSERT_TRUE(writing.attach(producer).ok());
    auto ticket = detail::LeaseAccess::pending_append_read(page.appender,
        capacity.subview(0, boundary).value()).value();
    auto reading = uses.begin().value();
    ASSERT_TRUE(detail::LeaseAccess::ordered_read_unique(reading, ticket).ok());
    auto consumer = detail::CompletionAccess::make(&kPending, &consuming);
    ASSERT_TRUE(reading.attach(consumer).ok());
    detail::CompletionAccess::append_terminal(consumer, hyle::Status::Cancelled("append reader cancelled"));
    if (fail_producer)
      detail::CompletionAccess::append_terminal(producer, hyle::Status::Internal("append producer failed"));
    const auto produced = producer.wait();
    EXPECT_EQ(produced.IsInternal(), fail_producer);
    auto published = detail::LeaseAccess::publish_append_after_writes(page.appender, ticket);
    if (fail_producer) {
      ASSERT_FALSE(published); EXPECT_TRUE(published.error().IsInternal());
      EXPECT_EQ(page.appender.committed_bytes(), boundary - 32);
      EXPECT_TRUE(page.appender.rollback().IsUnavailable());
    } else {
      ASSERT_TRUE(published) << published.error().message();
      EXPECT_EQ(published->size(), boundary);
    }
    EXPECT_TRUE(consumer.wait().IsCancelled());
    if (fail_producer) ASSERT_TRUE(page.appender.rollback().ok());
    EXPECT_EQ(page.prefix.size(), 64u);
    std::array<std::byte, 64> old{};
    ASSERT_TRUE(page.prefix.view()->read(old).ok());
    for (auto byte : old) EXPECT_EQ(byte, std::byte{19});
  }
  ASSERT_TRUE(page.appender.reserve(16).ok());
  const auto exact = capacity.subview(0, page.appender.committed_bytes() + 16).value();
  auto last_write = uses.begin().value();
  ASSERT_TRUE(detail::LeaseAccess::append_unique(last_write, page.appender).ok());
  auto last_producer = Completion::completed();
  ASSERT_TRUE(last_write.attach(last_producer).ok()); ASSERT_TRUE(last_producer.wait().ok());
  auto ticket = detail::LeaseAccess::pending_append_read(page.appender, exact).value();
  ASSERT_TRUE(pool.close().ok());
  EXPECT_TRUE(detail::LeaseAccess::matches_pending_append_read(ticket, exact).ok());
  auto rejected = uses.begin().value();
  EXPECT_TRUE(detail::LeaseAccess::ordered_read_unique(rejected, ticket).IsFailedPrecondition());
  EXPECT_FALSE(detail::LeaseAccess::publish_append_after_writes(page.appender, ticket));
  ASSERT_TRUE(page.appender.rollback().ok());
  EXPECT_EQ(pool.stats().active_uses, 0u);
}

// F2 control protocol only. These tests do not certify native ordering; the
// copy owner must separately prove the original device/phase dependency.
TEST(CopyLeaseHandoff, SuccessfulProducerCanPublishWhileOrderedReaderRemainsPending) {
  auto pool = LeasePool::create(mem::system_resource(), 512, {.max_blocks = 1}).value();
  auto writer = pool.allocate(256).value();
  ASSERT_TRUE(writer.view()->fill(std::byte{37}).ok());
  auto uses = LeaseUsePool::create({.max_batches = 2, .max_resources_per_batch = 1}).value();
  auto produced = uses.begin().value(), consumed = uses.begin().value();
  ASSERT_TRUE(detail::LeaseAccess::write_unique(produced, writer, false).ok());
  ASSERT_TRUE(detail::LeaseAccess::ordered_read_unique(consumed, writer).ok());
  PendingUse producing, consuming;
  auto producer = detail::CompletionAccess::make(&kPending, &producing);
  auto consumer = detail::CompletionAccess::make(&kPending, &consuming);
  ASSERT_TRUE(produced.attach(producer).ok()); ASSERT_TRUE(consumed.attach(consumer).ok());
  EXPECT_FALSE(detail::LeaseAccess::publish_after_writes(writer).has_value());
  ASSERT_TRUE(producer.wait().ok());
  EXPECT_EQ(pool.stats().active_uses, 1u);
  EXPECT_FALSE(writer.view().has_value()); // even readonly native use excludes mutable access
  EXPECT_FALSE(writer.publish().has_value()); // unchanged public rule
  auto ready = detail::LeaseAccess::publish_after_writes(writer).value();
  EXPECT_FALSE(writer.valid());
  std::array<std::byte, 8> observed{};
  ASSERT_TRUE(ready.subview(0, observed.size())->view()->read(observed).ok());
  for (auto value : observed) EXPECT_EQ(value, std::byte{37});
  EXPECT_FALSE(ready.try_write().has_value());
  EXPECT_EQ(consumer.poll().value(), CompletionPoll::kPending);
  ASSERT_TRUE(consumer.wait().ok());
  EXPECT_EQ(pool.stats().active_uses, 0u);
  EXPECT_TRUE(ready.try_write().has_value());
}

TEST(CopyLeaseHandoff, AllOrderedWritersMustActuallyRetireBeforeReady) {
  auto pool = LeasePool::create(mem::system_resource(), 512, {.max_blocks = 1}).value();
  auto writer = pool.allocate(256).value();
  auto uses = LeaseUsePool::create({.max_batches = 2, .max_resources_per_batch = 1}).value();
  auto first = uses.begin().value(), second = uses.begin().value();
  ASSERT_TRUE(detail::LeaseAccess::write_unique(first, writer, false).ok());
  ASSERT_TRUE(detail::LeaseAccess::write_unique(second, writer, true).ok());
  PendingUse a, b;
  auto one = detail::CompletionAccess::make(&kPending, &a);
  auto two = detail::CompletionAccess::make(&kPending, &b);
  ASSERT_TRUE(first.attach(one).ok()); ASSERT_TRUE(second.attach(two).ok());
  ASSERT_TRUE(one.wait().ok());
  EXPECT_FALSE(detail::LeaseAccess::publish_after_writes(writer).has_value());
  EXPECT_EQ(pool.stats().active_uses, 1u);
  ASSERT_TRUE(two.wait().ok());
  EXPECT_TRUE(detail::LeaseAccess::publish_after_writes(writer).has_value());
}

TEST(CopyLeaseHandoff, ReadFailurePreservesReadyBytesButWriterFailureCannotPublish) {
  auto pool = LeasePool::create(mem::system_resource(), 512, {.max_blocks = 2}).value();
  auto writer = pool.allocate(256).value(), bad_writer = pool.allocate(256).value();
  ASSERT_TRUE(writer.view()->fill(std::byte{19}).ok());
  auto uses = LeaseUsePool::create({.max_batches = 2, .max_resources_per_batch = 1}).value();
  auto read = uses.begin().value();
  ASSERT_TRUE(detail::LeaseAccess::ordered_read_unique(read, writer).ok());
  auto ready = detail::LeaseAccess::publish_after_writes(writer).value();
  auto cancelled_read = Completion::completed(hyle::Status::Cancelled("reader cancelled"));
  ASSERT_TRUE(read.attach(cancelled_read).ok());
  std::array<std::byte, 8> observed{};
  ASSERT_TRUE(ready.subview(0, observed.size())->view()->read(observed).ok());
  for (auto value : observed) EXPECT_EQ(value, std::byte{19});
  EXPECT_TRUE(ready.try_write().has_value());
  auto write = uses.begin().value();
  ASSERT_TRUE(detail::LeaseAccess::write_unique(write, bad_writer, false).ok());
  auto failed_write = Completion::completed(hyle::Status::Unavailable("producer failed"));
  ASSERT_TRUE(write.attach(failed_write).ok());
  EXPECT_FALSE(bad_writer.publish().has_value());
  EXPECT_FALSE(detail::LeaseAccess::publish_after_writes(bad_writer).has_value());
}

TEST(CopyLeaseHandoff, OrdinaryWriterCapacityAndClosedSourceChecksRemain) {
  auto pool = LeasePool::create(mem::system_resource(), 512, {.max_blocks = 2}).value();
  auto first = pool.allocate(256).value(), second = pool.allocate(256).value();
  auto uses = LeaseUsePool::create({.max_batches = 2, .max_resources_per_batch = 1}).value();
  auto one = uses.begin().value(), rejected = uses.begin().value();
  ASSERT_TRUE(detail::LeaseAccess::write_unique(one, first, false).ok());
  EXPECT_TRUE(detail::LeaseAccess::write_unique(rejected, first, false).IsUnavailable());
  EXPECT_TRUE(detail::LeaseAccess::write_unique(one, second, false).IsResourceExhausted());
  one = {};
  auto ready = first.publish().value();
  ASSERT_TRUE(pool.close().ok());
  EXPECT_TRUE(detail::LeaseAccess::read_unique(rejected, ready).IsFailedPrecondition());
  EXPECT_EQ(pool.stats().active_uses, 0u);
}

TEST(ResourceLeases, SharedReadersAndSubviewsPreventMutationAndHoldRange) {
  auto pool = LeasePool::create(mem::system_resource(), 1024, {.max_blocks = 2}).value();
  auto write = pool.allocate(256).value();
  ASSERT_TRUE(write.view()->fill(std::byte{42}).ok());
  auto read = write.publish().value();
  EXPECT_FALSE(write.valid());
  EXPECT_EQ(write.size(), 0u);
  const auto generation = read.generation();
  auto shared = read;
  auto piece = read.subview(16, 32).value();
  EXPECT_FALSE(read.try_write().has_value());
  read = read;
  EXPECT_TRUE(read.valid());
  std::array<std::byte, 32> data{};
  ASSERT_TRUE(piece.view()->read(data).ok());
  for (auto value : data) EXPECT_EQ(value, std::byte{42});
  shared.reset(); piece.reset();
  auto mutable_again = read.try_write().value();
  EXPECT_FALSE(read.valid());
  EXPECT_EQ(mutable_again.generation(), generation);
  mutable_again.reset();
  EXPECT_EQ(pool.stats().allocated_bytes, 0u);
  EXPECT_GT(pool.allocate(256)->generation(), generation);
}

TEST(ResourceLeases, MetadataAndBackingCapsDoNotGrowAndOldOwnerCanOutlivePool) {
  ReadLease retained;
  {
    auto pool = LeasePool::create(mem::system_resource(), 512, {.max_blocks = 1}).value();
    auto write = pool.allocate(256).value();
    EXPECT_FALSE(pool.allocate(1).has_value());
    EXPECT_FALSE(pool.allocate(0).has_value());
    EXPECT_FALSE(pool.allocate(1, 512).has_value());
    retained = write.publish().value();
    EXPECT_TRUE(pool.close().ok());
    EXPECT_FALSE(pool.allocate(256).has_value());
  }
  EXPECT_TRUE(retained.view().has_value());
  retained.reset();
  EXPECT_FALSE(LeasePool::create({}, 256).has_value());
  EXPECT_FALSE(LeasePool::create(mem::system_resource(), 0).has_value());
}

TEST(ResourceLeases, AbandonedCompletionKeepsBothPoolsUntilRealRetirement) {
  auto first = LeasePool::create(mem::system_resource(), 512, {.max_blocks = 1}).value();
  auto second = LeasePool::create(mem::system_resource(), 512, {.max_blocks = 1}).value();
  auto source_write = first.allocate(256).value();
  auto source = source_write.publish().value();
  auto destination = second.allocate(256).value();
  auto uses = LeaseUsePool::create({.max_batches = 1, .max_resources_per_batch = 2}).value();
  PendingUse pending;
  {
    auto batch = uses.begin().value();
    EXPECT_FALSE(uses.begin().has_value());
    ASSERT_TRUE(batch.read(source).ok());
    ASSERT_TRUE(batch.read(source).ok()); // same block uses one control slot
    ASSERT_TRUE(batch.write(destination).ok());
    EXPECT_FALSE(destination.view().has_value());
    EXPECT_FALSE(destination.publish().has_value());
    auto completion = detail::CompletionAccess::make(&kPending, &pending);
    ASSERT_TRUE(batch.attach(completion).ok());
    EXPECT_FALSE(batch.valid());
    source.reset(); destination.reset();
  }
  EXPECT_EQ(first.stats().live_blocks, 1u);
  EXPECT_EQ(second.stats().live_blocks, 1u);
  EXPECT_EQ(uses.active_batches(), 1u);
  EXPECT_TRUE(pending.retained.valid());
  EXPECT_TRUE(first.close().IsUnavailable());
  pending.done = true;
  pending.retained.terminal(pending.retained.state, hyle::Status::Ok());
  pending.retained.release(pending.retained.state);
  pending.retained = {};
  EXPECT_EQ(first.stats().live_blocks, 0u);
  EXPECT_EQ(second.stats().live_blocks, 0u);
  EXPECT_EQ(uses.active_batches(), 0u);
  EXPECT_TRUE(first.drain().ok());
  EXPECT_TRUE(uses.drain().ok());
}

TEST(ResourceLeases, WriteFailureCannotPublishAndUnsubmittedUseRollsBack) {
  auto pool = LeasePool::create(mem::system_resource(), 512, {.max_blocks = 1}).value();
  auto write = pool.allocate(256).value();
  auto uses = LeaseUsePool::create({.max_batches = 2, .max_resources_per_batch = 1}).value();
  {
    auto batch = uses.begin().value();
    ASSERT_TRUE(batch.write(write).ok());
    auto other = uses.begin().value();
    EXPECT_TRUE(other.write(write).IsUnavailable());
  }
  EXPECT_EQ(pool.stats().active_uses, 0u);
  auto batch = uses.begin().value();
  ASSERT_TRUE(batch.write(write).ok());
  auto failed = Completion::completed(hyle::Status::Unavailable("copy prefix failed"));
  ASSERT_TRUE(batch.attach(failed).ok());
  EXPECT_FALSE(write.publish().has_value());
  EXPECT_FALSE(write.view().has_value());
  write.reset();
  EXPECT_EQ(pool.stats().live_blocks, 0u);
  EXPECT_TRUE(pool.allocate(256).has_value());
}

TEST(ResourceLeases, RejectedAttachmentRetainsTicketAndQuotaIncludesMetadata) {
  auto pool = LeasePool::create(mem::system_resource(), 512, {.max_blocks = 1}).value();
  auto write = pool.allocate(256).value();
  auto uses = LeaseUsePool::create().value();
  auto batch = uses.begin().value();
  ASSERT_TRUE(batch.write(write).ok());
  Completion invalid;
  EXPECT_TRUE(batch.attach(invalid).IsFailedPrecondition());
  EXPECT_TRUE(batch.valid());
  auto done = Completion::completed();
  EXPECT_TRUE(batch.attach(done).ok());
  EXPECT_TRUE(write.publish().has_value());
  const std::array limits{ResourceBudgetLimit{mem::system_resource(), 16384}};
  auto budget = ResourceBudget::create({.limits = limits}).value();
  const auto before = budget.usage(mem::system_resource())->live_bytes;
  EXPECT_FALSE(LeasePool::create(mem::system_resource(), 32768,
      {.max_blocks = 1, .resource_budget = budget}).has_value());
  EXPECT_EQ(budget.usage(mem::system_resource())->live_bytes, before);
}
TEST(ResourceLeases, OwnedImportPreservesBackingAndReadonlyCannotBecomeWriter) {
  auto backing = mem::Buffer::create(mem::system_resource(), 512, 16).value();
  const auto token = backing.native_handle();
  ASSERT_TRUE(backing.view().fill(std::byte{23}).ok());
  auto readonly = ReadLease::adopt(std::move(backing)).value();
  EXPECT_FALSE(backing.valid());
  EXPECT_EQ(readonly.view()->native_handle(), token);
  auto piece = readonly.subview(16, 32).value();
  readonly.reset();
  EXPECT_FALSE(piece.try_write().has_value());
  std::array<std::byte, 32> observed{};
  ASSERT_TRUE(piece.view()->read(observed).ok());
  for (auto value : observed) EXPECT_EQ(value, std::byte{23});
  auto uses = LeaseUsePool::create({.max_batches = 1, .max_resources_per_batch = 1}).value();
  auto use = uses.begin().value();
  ASSERT_TRUE(use.read(piece).ok());
  auto done = Completion::completed();
  ASSERT_TRUE(use.attach(done).ok());
  EXPECT_FALSE(piece.try_write().has_value());

  auto writable_backing = mem::Buffer::create(mem::system_resource(), 512, 16).value();
  const auto mutable_token = writable_backing.native_handle();
  auto pool = LeasePool::adopt(std::move(writable_backing), {.max_blocks = 2}).value();
  EXPECT_FALSE(writable_backing.valid());
  auto write = pool.allocate(32, 16).value();
  EXPECT_EQ(write.view()->native_handle(), mutable_token);
  EXPECT_FALSE(pool.allocate(32, 256).has_value());
}

TEST(ResourceLeases, RejectedOwnedImportDoesNotConsumeBackingOrLeakBudget) {
  auto backing = mem::Buffer::create(mem::system_resource(), 32 * 1024, 256).value();
  const auto token = backing.native_handle();
  EXPECT_FALSE(LeasePool::adopt(std::move(backing), {.max_blocks = 0}).has_value());
  ASSERT_TRUE(backing.valid());
  EXPECT_EQ(backing.native_handle(), token);
  EXPECT_FALSE(ReadLease::adopt(std::move(backing), {.allocator = {}}).has_value());
  ASSERT_TRUE(backing.valid());
  const std::array limits{ResourceBudgetLimit{mem::system_resource(), 16 * 1024}};
  auto budget = ResourceBudget::create({.limits = limits}).value();
  const auto before = budget.usage(mem::system_resource())->live_bytes;
  EXPECT_FALSE(ReadLease::adopt(std::move(backing), {.resource_budget = budget}).has_value());
  EXPECT_EQ(budget.usage(mem::system_resource())->live_bytes, before);
  ASSERT_TRUE(backing.valid());
  EXPECT_EQ(backing.native_handle(), token);
  EXPECT_FALSE(LeasePool::adopt(std::move(backing), {.resource_budget = budget}).has_value());
  EXPECT_EQ(budget.usage(mem::system_resource())->live_bytes, before);
  EXPECT_TRUE(backing.valid());
}

TEST(ResourceLeases, CheckedCapacitySubviewAndMetadataArithmeticRejectBeforeMutation) {
  EXPECT_FALSE(LeasePool::create(mem::system_resource(), 512, {.backing_alignment = 3}).has_value());
  EXPECT_FALSE(LeaseUsePool::create({.max_batches = 0}).has_value());
  EXPECT_FALSE(LeaseUsePool::create({.max_resources_per_batch = 0}).has_value());
  auto pool = LeasePool::create(mem::system_resource(), 512, {.max_blocks = 2}).value();
  auto write = pool.allocate(16).value();
  EXPECT_FALSE(pool.allocate((std::numeric_limits<std::size_t>::max)()).has_value());
  EXPECT_FALSE(pool.allocate(16, 3).has_value());
  ASSERT_TRUE(write.view()->fill(std::byte{17}).ok());
  auto read = write.publish().value();
  EXPECT_FALSE(read.subview((std::numeric_limits<std::size_t>::max)(), 1).has_value());
  EXPECT_FALSE(read.subview(15, 2).has_value());
  std::array<std::byte, 16> actual;
  ASSERT_TRUE(read.view()->read(actual).ok());
  for (auto value : actual) EXPECT_EQ(value, std::byte{17});
  EXPECT_EQ(pool.stats().live_blocks, 1u);
}

TEST(ResourceLeases, CrossThreadReadersHoldGenerationWhileIndependentBlockRetires) {
  auto pool = LeasePool::create(mem::system_resource(), 1024, {.max_blocks = 2}).value();
  auto first_write = pool.allocate(256).value(), other_write = pool.allocate(256).value();
  ASSERT_TRUE(first_write.view()->fill(std::byte{29}).ok());
  auto first = first_write.publish().value();
  const auto generation = first.generation();
  std::barrier ready(3), release(3);
  std::atomic<bool> content_ok{true};
  auto reader = [&](ReadLease owned) {
    ready.arrive_and_wait();
    std::array<std::byte, 16> data{};
    auto window = owned.subview(0, data.size());
    if (!window || !window->view()->read(data).ok()) content_ok.store(false);
    for (auto value : data) if (value != std::byte{29}) content_ok.store(false);
    release.arrive_and_wait();
  };
  std::thread first_reader(reader, first), second_reader(reader, first.subview(16, 16).value());
  ready.arrive_and_wait();
  EXPECT_FALSE(first.try_write().has_value());
  other_write.reset();
  EXPECT_EQ(pool.stats().live_blocks, 1u);
  auto independent = pool.allocate(256).value();
  EXPECT_GT(independent.generation(), generation);
  EXPECT_TRUE(independent.view()->fill(std::byte{41}).ok());
  independent.reset();
  release.arrive_and_wait();
  first_reader.join(); second_reader.join();
  EXPECT_TRUE(content_ok.load());
  auto exclusive = first.try_write();
  ASSERT_TRUE(exclusive.has_value());
  EXPECT_EQ(exclusive->generation(), generation);
  exclusive->reset();
  EXPECT_EQ(pool.stats().allocated_bytes, 0u);
}

} // namespace
