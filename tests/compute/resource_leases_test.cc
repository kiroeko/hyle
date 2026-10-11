#include <gtest/gtest.h>

#include <array>
#include <barrier>
#include <thread>
#include <limits>

#include "completion_internal.h"
#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/resource_leases.h"

namespace {
using namespace hyle::compute;
namespace mem = hyle::mem;

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
    if (!owned.view()->read(data).ok()) content_ok.store(false);
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
  ASSERT_TRUE(independent.view()->fill(std::byte{41}).ok());
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
