#include <gtest/gtest.h>

#include <array>

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
} // namespace
