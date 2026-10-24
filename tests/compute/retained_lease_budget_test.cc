#include <gtest/gtest.h>

#include <array>
#include <utility>

#include "completion_internal.h"
#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/detail/lease_access.h"

namespace {
namespace c = hyle::compute;
namespace d = hyle::compute::detail;
namespace m = hyle::mem;

std::size_t external(const c::ResourceBudget &budget) {
  const auto usage = budget.usage(m::system_resource());
  return usage ? usage->live_by_category[static_cast<std::size_t>(c::ResourceBudgetCategory::kExternal)] : SIZE_MAX;
}

// The existing F1 completion attachment retains accepted uses after the
// public Completion is dropped. Only this controlled native terminal releases.
struct PendingNative final {
  bool retired = false;
  d::CompletionLifetime retained{};
  hyle::Status hint{};
  ~PendingNative() { retire(); }
  void retire() noexcept {
    retired = true;
    const auto lifetime = std::exchange(retained, {});
    if (lifetime.valid()) {
      lifetime.terminal(lifetime.state, hint);
      lifetime.release(lifetime.state);
    }
  }
};
const d::CompletionVTable pending_native{
    .destroy = [](void *opaque, d::CompletionLifetime lifetime, hyle::Status hint) noexcept {
      auto &native = *static_cast<PendingNative *>(opaque);
      native.retained = lifetime;
      native.hint = hint;
      if (native.retired) native.retire();
    },
    .poll = [](void *opaque) noexcept -> hyle::Result<c::CompletionPoll> {
      return static_cast<PendingNative *>(opaque)->retired
          ? c::CompletionPoll::kComplete : c::CompletionPoll::kPending;
    },
    .wait = [](void *opaque) noexcept {
      static_cast<PendingNative *>(opaque)->retire();
      return hyle::Status::Ok();
    }};

TEST(RetainedLeaseBudget, ExistingPoolChargeAndSubviewSurviveUntilActualUseRetirement) {
  const std::array limits{c::ResourceBudgetLimit{m::system_resource(), 65536}};
  auto made = c::ResourceBudget::create({.limits = limits}); ASSERT_TRUE(made);
  auto &budget = *made;
  const auto baseline = budget.usage(m::system_resource())->live_bytes;
  auto created = c::LeasePool::create(m::system_resource(), 4096,
      {.max_blocks = 1, .resource_budget = budget});
  ASSERT_TRUE(created);
  auto pool = std::move(*created);
  auto allocated = pool.allocate(256); ASSERT_TRUE(allocated);
  auto writer = std::move(*allocated);
  ASSERT_TRUE(writer.view()->fill(std::byte{37}).ok());
  auto published = writer.publish(); ASSERT_TRUE(published);
  auto piece_result = published->subview(16, 32); ASSERT_TRUE(piece_result);
  auto piece = std::move(*piece_result);
  published->reset();
  const auto generation = piece.generation();
  EXPECT_EQ(external(budget), 4096u);
  const auto remaining = *budget.usage(m::system_resource());
  auto blocker = budget.reserve(m::system_resource(),
      remaining.limit_bytes - remaining.live_bytes - 2048, c::ResourceBudgetCategory::kWorkspace);
  ASSERT_TRUE(blocker); // Metadata fits; a second 4096-byte payload charge cannot.
  ASSERT_TRUE(d::LeaseAccess::retain_budget(piece, m::system_resource(), budget).ok());
  EXPECT_EQ(external(budget), 4096u); // Adopt original pool charge, no new payload quota.
  blocker->reset();
  const auto admitted = budget.usage(m::system_resource())->live_bytes;
  ASSERT_TRUE(d::LeaseAccess::retain_budget(piece, m::system_resource(), budget).ok());
  EXPECT_EQ(budget.usage(m::system_resource())->live_bytes, admitted);
  EXPECT_EQ(piece.size(), 32u);
  EXPECT_EQ(piece.generation(), generation);

  auto uses = c::LeaseUsePool::create({.max_batches = 1, .max_resources_per_batch = 1});
  ASSERT_TRUE(uses);
  auto episode = uses->begin(); ASSERT_TRUE(episode);
  ASSERT_TRUE(episode->read(piece).ok());
  PendingNative native;
  auto completion = d::CompletionAccess::make(&pending_native, &native);
  ASSERT_TRUE(episode->attach(completion).ok());
  EXPECT_FALSE(piece.try_write());
  EXPECT_EQ(completion.poll().value(), c::CompletionPoll::kPending);
  piece.reset();
  EXPECT_TRUE(pool.close().IsUnavailable());
  pool = {};
  completion = {}; // Accepted work, not the observer, still owns the backing.
  EXPECT_TRUE(native.retained.valid());
  EXPECT_EQ(external(budget), 4096u);
  EXPECT_EQ(uses->active_batches(), 1u);
  native.retire();
  EXPECT_EQ(uses->active_batches(), 0u);
  EXPECT_EQ(external(budget), 0u);
  EXPECT_EQ(budget.usage(m::system_resource())->live_bytes, baseline);
}

TEST(RetainedLeaseBudget, QuotaRejectionPreservesReadonlyLeaseAndRetryChargesWholeBacking) {
  const std::array limits{c::ResourceBudgetLimit{m::system_resource(), 65536}};
  auto made = c::ResourceBudget::create({.limits = limits}); ASSERT_TRUE(made);
  auto &budget = *made;
  const auto baseline = budget.usage(m::system_resource())->live_bytes;
  auto allocated = m::Buffer::create(m::system_resource(), 512, 16); ASSERT_TRUE(allocated);
  ASSERT_TRUE(allocated->view().fill(std::byte{91}).ok());
  auto original = c::ReadLease::adopt(std::move(*allocated)); ASSERT_TRUE(original);
  auto selected = original->subview(16, 32); ASSERT_TRUE(selected);
  auto lease = std::move(*selected);
  original->reset();
  const auto generation = lease.generation();
  const auto before = lease.view().value();
  const auto usage = *budget.usage(m::system_resource());
  auto blocker = budget.reserve(m::system_resource(), usage.limit_bytes - usage.live_bytes - 1);
  ASSERT_TRUE(blocker);
  const auto blocked = budget.usage(m::system_resource())->live_bytes;
  EXPECT_FALSE(d::LeaseAccess::retain_budget(lease, m::system_resource(), budget).ok());
  EXPECT_EQ(budget.usage(m::system_resource())->live_bytes, blocked);
  EXPECT_TRUE(lease.valid());
  EXPECT_EQ(lease.generation(), generation);
  EXPECT_EQ(lease.size(), 32u);
  EXPECT_EQ(lease.view()->native_handle(), before.native_handle());
  EXPECT_EQ(lease.view()->offset(), before.offset());
  EXPECT_FALSE(lease.subview(0, 33));
  EXPECT_FALSE(lease.try_write()); // A budget cannot grant writer permission.
  std::array<std::byte, 32> observed{};
  ASSERT_TRUE(lease.view()->read(observed).ok());
  for (auto byte : observed) EXPECT_EQ(byte, std::byte{91});
  blocker->reset();
  ASSERT_TRUE(d::LeaseAccess::retain_budget(lease, m::system_resource(), budget).ok());
  EXPECT_EQ(external(budget), 512u); // Selected view is only 32 bytes.
  EXPECT_FALSE(lease.try_write());
  lease.reset();
  EXPECT_EQ(budget.usage(m::system_resource())->live_bytes, baseline);
}

// These keys describe each test's one real complete system Buffer. They do
// not pretend that independent allocations share a native root; the actual
// Vulkan consumer covers fresh export domains of one owned native root.
const unsigned annotation_authority = 0;
const unsigned other_annotation_authority = 0;

TEST(RetainedLeaseBudget, CanonicalAnnotationValidatesWholeWrapperAndPreservesF1Contract) {
  const std::array limits{c::ResourceBudgetLimit{m::system_resource(), 65536}};
  auto made = c::ResourceBudget::create({.limits = limits}); ASSERT_TRUE(made);
  auto &budget = *made;
  const auto baseline = budget.usage(m::system_resource())->live_bytes;
  auto allocated = m::Buffer::create(m::system_resource(), 512, 16); ASSERT_TRUE(allocated);
  const auto canonical_owner = allocated->native_handle();
  ASSERT_NE(canonical_owner, nullptr);
  ASSERT_TRUE(allocated->view().fill(std::byte{73}).ok());
  auto original = c::ReadLease::adopt(std::move(*allocated)); ASSERT_TRUE(original);
  auto selected = original->subview(16, 32); ASSERT_TRUE(selected);
  auto lease = std::move(*selected); original->reset();
  const auto block = d::LeaseAccess::identity(lease);
  const auto generation = lease.generation();
  const auto before = lease.view().value();
  const auto backing = d::LeaseAccess::backing(lease).value();
  const auto unchanged = [&] {
    EXPECT_TRUE(lease.valid()); EXPECT_EQ(lease.size(), 32u);
    EXPECT_EQ(lease.generation(), generation);
    EXPECT_EQ(d::LeaseAccess::identity(lease), block);
    auto now = lease.view(); ASSERT_TRUE(now);
    EXPECT_EQ(now->resource(), before.resource());
    EXPECT_EQ(now->native_handle(), before.native_handle());
    EXPECT_EQ(now->offset(), before.offset()); EXPECT_EQ(now->size(), before.size());
    auto domain = d::LeaseAccess::backing(lease); ASSERT_TRUE(domain);
    EXPECT_EQ(domain->identity, backing.identity); EXPECT_EQ(domain->resource, backing.resource);
    EXPECT_EQ(domain->bytes, backing.bytes); EXPECT_EQ(domain->bytes, 512u);
    EXPECT_FALSE(lease.subview(0, 33)); EXPECT_FALSE(lease.try_write());
    std::array<std::byte, 32> bytes{};
    ASSERT_TRUE(now->read(bytes).ok());
    for (auto value : bytes) EXPECT_EQ(value, std::byte{73});
  };
  c::ReadLease empty;
  EXPECT_TRUE(d::LeaseAccess::set_retained_backing(empty, m::system_resource(),
      &annotation_authority, canonical_owner, 512).IsInvalidArgument());
  EXPECT_TRUE(d::LeaseAccess::set_retained_backing(lease, {},
      &annotation_authority, canonical_owner, 512).IsInvalidArgument());
  EXPECT_TRUE(d::LeaseAccess::set_retained_backing(lease, m::system_resource(),
      nullptr, canonical_owner, 512).IsInvalidArgument());
  EXPECT_TRUE(d::LeaseAccess::set_retained_backing(lease, m::system_resource(),
      &annotation_authority, nullptr, 512).IsInvalidArgument());
  for (const auto bytes : {std::size_t{0}, std::size_t{511}, std::size_t{SIZE_MAX}})
    EXPECT_TRUE(d::LeaseAccess::set_retained_backing(lease, m::system_resource(),
        &annotation_authority, canonical_owner, bytes).IsInvalidArgument());
  for (const auto offset : {std::size_t{1}, std::size_t{513}, std::size_t{SIZE_MAX}})
    EXPECT_TRUE(d::LeaseAccess::set_retained_backing(lease, m::system_resource(),
        &annotation_authority, canonical_owner, 512, offset).IsInvalidArgument());
  // The selected read range fits at offset16, but the actual 512-byte wrapper
  // does not. Annotation validates the wrapper, not only the 32-byte sublease.
  EXPECT_TRUE(d::LeaseAccess::set_retained_backing(lease, m::system_resource(),
      &annotation_authority, canonical_owner, 512, 16).IsInvalidArgument());
  unchanged();
  ASSERT_TRUE(d::LeaseAccess::set_retained_backing(lease, m::system_resource(),
      &annotation_authority, canonical_owner, 512).ok());
  ASSERT_TRUE(d::LeaseAccess::set_retained_backing(lease, m::system_resource(),
      &annotation_authority, canonical_owner, 512).ok());
  EXPECT_EQ(budget.usage(m::system_resource())->live_bytes, baseline);
  EXPECT_TRUE(d::LeaseAccess::set_retained_backing(lease, m::system_resource(),
      &other_annotation_authority, canonical_owner, 512).IsFailedPrecondition());
  ASSERT_TRUE(d::LeaseAccess::retain_budget(lease, m::system_resource(), budget).ok());
  EXPECT_EQ(external(budget), 512u);
  const auto admitted = budget.usage(m::system_resource())->live_bytes;
  // Idempotent proof is allowed after admission; a different proof is not.
  ASSERT_TRUE(d::LeaseAccess::set_retained_backing(lease, m::system_resource(),
      &annotation_authority, canonical_owner, 512).ok());
  EXPECT_TRUE(d::LeaseAccess::set_retained_backing(lease, m::system_resource(),
      &other_annotation_authority, canonical_owner, 512).IsFailedPrecondition());
  ASSERT_TRUE(d::LeaseAccess::retain_budget(lease, m::system_resource(), budget).ok());
  EXPECT_EQ(budget.usage(m::system_resource())->live_bytes, admitted);
  unchanged();
  lease.reset();
  EXPECT_EQ(external(budget), 0u);
  EXPECT_EQ(budget.usage(m::system_resource())->live_bytes, baseline);
}

TEST(RetainedLeaseBudget, UnannotatedRetainedDomainCannotChangeItsAccountingKey) {
  const std::array limits{c::ResourceBudgetLimit{m::system_resource(), 65536}};
  auto made = c::ResourceBudget::create({.limits = limits}); ASSERT_TRUE(made);
  auto &budget = *made;
  const auto baseline = budget.usage(m::system_resource())->live_bytes;
  auto allocated = m::Buffer::create(m::system_resource(), 512, 16); ASSERT_TRUE(allocated);
  const auto canonical_owner = allocated->native_handle();
  ASSERT_TRUE(allocated->view().fill(std::byte{29}).ok());
  auto original = c::ReadLease::adopt(std::move(*allocated)); ASSERT_TRUE(original);
  auto selected = original->subview(16, 32); ASSERT_TRUE(selected);
  auto lease = std::move(*selected); original->reset();
  const auto block = d::LeaseAccess::identity(lease);
  const auto generation = lease.generation();
  const auto before = lease.view().value();
  const auto backing = d::LeaseAccess::backing(lease).value();
  ASSERT_TRUE(d::LeaseAccess::retain_budget(lease, m::system_resource(), budget).ok());
  EXPECT_EQ(external(budget), 512u);
  const auto admitted = budget.usage(m::system_resource())->live_bytes;
  EXPECT_TRUE(d::LeaseAccess::set_retained_backing(lease, m::system_resource(),
      &annotation_authority, canonical_owner, 512).IsFailedPrecondition());
  EXPECT_EQ(budget.usage(m::system_resource())->live_bytes, admitted);
  EXPECT_EQ(d::LeaseAccess::identity(lease), block); EXPECT_EQ(lease.generation(), generation);
  ASSERT_TRUE(lease.valid()); EXPECT_EQ(lease.size(), 32u);
  const auto now = lease.view().value();
  EXPECT_EQ(now.resource(), before.resource()); EXPECT_EQ(now.native_handle(), before.native_handle());
  EXPECT_EQ(now.offset(), before.offset()); EXPECT_EQ(now.size(), before.size());
  const auto after = d::LeaseAccess::backing(lease).value();
  EXPECT_EQ(after.identity, backing.identity); EXPECT_EQ(after.resource, backing.resource);
  EXPECT_EQ(after.bytes, backing.bytes); EXPECT_EQ(after.bytes, 512u);
  EXPECT_FALSE(lease.subview(0, 33)); EXPECT_FALSE(lease.try_write());
  std::array<std::byte, 32> bytes{};
  ASSERT_TRUE(now.read(bytes).ok());
  for (auto value : bytes) EXPECT_EQ(value, std::byte{29});
  ASSERT_TRUE(d::LeaseAccess::retain_budget(lease, m::system_resource(), budget).ok());
  EXPECT_EQ(budget.usage(m::system_resource())->live_bytes, admitted);
  lease.reset();
  EXPECT_EQ(external(budget), 0u);
  EXPECT_EQ(budget.usage(m::system_resource())->live_bytes, baseline);
}
} // namespace
