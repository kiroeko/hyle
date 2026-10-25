#include <gtest/gtest.h>

#include <array>
#include <barrier>
#include <thread>
#include <utility>

#include "hyle/compute/detail/resource_budget_access.h"

namespace {
namespace c = hyle::compute;
namespace d = hyle::compute::detail;
namespace m = hyle::mem;

constexpr std::size_t limit = 8192;
int authority, other_authority;
auto make_budget() {
  const std::array limits{c::ResourceBudgetLimit{m::system_resource(), limit}};
  return c::ResourceBudget::create({.limits = limits});
}

class HostResource final : public m::MemoryResource {
public:
  std::string_view name() const noexcept override { return "shared-budget-test-host"; }
  m::MemoryProperties properties() const noexcept override { return m::system_resource().properties(); }
  hyle::Result<m::ResourceAllocation> allocate(std::size_t bytes, std::size_t alignment) noexcept override {
    return m::system_resource().resource().allocate(bytes, alignment);
  }
  void deallocate(m::ResourceAllocation allocation) noexcept override {
    m::system_resource().resource().deallocate(allocation);
  }
};
struct BackingProbe final {
  std::size_t allocations = 0, reallocations = 0, frees = 0;
  bool fail = false;
  m::AllocatorRef ref() noexcept { return {allocate, free, reallocate, this}; }
  static void *allocate(void *p, std::size_t bytes, std::size_t alignment) noexcept {
    auto &self = *static_cast<BackingProbe *>(p); ++self.allocations;
    return self.fail ? nullptr : m::system_allocator().allocate(bytes, alignment);
  }
  static void free(void *p, void *storage, std::size_t bytes, std::size_t alignment) noexcept {
    ++static_cast<BackingProbe *>(p)->frees;
    m::system_allocator().deallocate(storage, bytes, alignment);
  }
  static void *reallocate(void *p, void *storage, std::size_t old_bytes,
      std::size_t bytes, std::size_t alignment) noexcept {
    auto &self = *static_cast<BackingProbe *>(p); ++self.reallocations;
    return self.fail ? nullptr : m::system_allocator().reallocate(storage, old_bytes, bytes, alignment);
  }
};
std::size_t external(const c::ResourceBudget &budget) {
  const auto usage = budget.usage(m::system_resource());
  return usage ? usage->live_by_category[static_cast<std::size_t>(c::ResourceBudgetCategory::kExternal)] : SIZE_MAX;
}

TEST(SharedBackingBudget, AliasesKeepOneChargeUntilLastToken) {
  auto made = make_budget(); ASSERT_TRUE(made);
  auto &budget = *made;
  const auto baseline = budget.usage(m::system_resource())->live_bytes;
  int allocation;
  const d::BackingIdentity identity{&authority, &allocation};
  auto first = d::ResourceBudgetAccess::reserve_backing(budget, m::system_resource(), identity, 256);
  ASSERT_TRUE(first);
  auto alias = d::ResourceBudgetAccess::reserve_backing(budget, m::system_resource(), identity, 256);
  ASSERT_TRUE(alias);
  d::SharedBackingReservation copy = *first;
  d::SharedBackingReservation moved = std::move(copy);
  EXPECT_FALSE(copy.valid());
  EXPECT_TRUE(moved.same_scope(budget));
  EXPECT_EQ(external(budget), 256u);
  EXPECT_FALSE(d::ResourceBudgetAccess::reserve_backing(budget, m::system_resource(), identity, 128));
  EXPECT_FALSE(d::ResourceBudgetAccess::reserve_backing(budget, m::system_resource(), {}, 256));
  EXPECT_EQ(external(budget), 256u);
  first->reset(); alias->reset();
  EXPECT_EQ(external(budget), 256u);
  moved.reset();
  EXPECT_EQ(budget.usage(m::system_resource())->live_bytes, baseline);
}

TEST(SharedBackingBudget, QuotaFailureAndIdentityNamespacesDoNotMerge) {
  auto made = make_budget(); ASSERT_TRUE(made);
  auto &budget = *made;
  const auto baseline = budget.usage(m::system_resource())->live_bytes;
  int allocation, second;
  auto first = d::ResourceBudgetAccess::reserve_backing(
      budget, m::system_resource(), {&authority, &allocation}, 128);
  ASSERT_TRUE(first);
  auto other_namespace = d::ResourceBudgetAccess::reserve_backing(
      budget, m::system_resource(), {&other_authority, &allocation}, 128);
  ASSERT_TRUE(other_namespace);
  EXPECT_EQ(external(budget), 256u);
  const auto usage = *budget.usage(m::system_resource());
  EXPECT_FALSE(d::ResourceBudgetAccess::reserve_backing(
      budget, m::system_resource(), {&authority, &second}, usage.limit_bytes - usage.live_bytes + 1));
  EXPECT_EQ(external(budget), 256u);
  first->reset(); other_namespace->reset();
  EXPECT_EQ(budget.usage(m::system_resource())->live_bytes, baseline);
  auto retry = d::ResourceBudgetAccess::reserve_backing(
      budget, m::system_resource(), {&authority, &second}, 256);
  EXPECT_TRUE(retry);
}

TEST(SharedBackingBudget, AdoptPreservesRejectedChargeAndConsumesOnlyOnSuccess) {
  auto made = make_budget(); ASSERT_TRUE(made);
  auto &budget = *made;
  const auto baseline = budget.usage(m::system_resource())->live_bytes;
  int allocation;
  const d::BackingIdentity identity{&authority, &allocation};
  auto existing = budget.reserve(m::system_resource(), 512); ASSERT_TRUE(existing);
  EXPECT_TRUE(d::ResourceBudgetAccess::same_scope(*existing, budget));
  EXPECT_FALSE(d::ResourceBudgetAccess::adopt_backing(
      budget, m::system_resource(), identity, 511, *existing));
  EXPECT_TRUE(d::ResourceBudgetAccess::same_scope(*existing, budget));
  EXPECT_EQ(external(budget), 512u);
  auto token = d::ResourceBudgetAccess::adopt_backing(
      budget, m::system_resource(), identity, 512, *existing);
  ASSERT_TRUE(token);
  EXPECT_FALSE(d::ResourceBudgetAccess::same_scope(*existing, budget));
  EXPECT_EQ(external(budget), 512u);
  auto duplicate = budget.reserve(m::system_resource(), 512); ASSERT_TRUE(duplicate);
  EXPECT_EQ(external(budget), 1024u);
  auto second = d::ResourceBudgetAccess::adopt_backing(
      budget, m::system_resource(), identity, 512, *duplicate);
  ASSERT_TRUE(second);
  EXPECT_FALSE(d::ResourceBudgetAccess::same_scope(*duplicate, budget));
  EXPECT_EQ(external(budget), 512u);
  auto wrong_category = budget.reserve(m::system_resource(), 512, c::ResourceBudgetCategory::kWorkspace);
  ASSERT_TRUE(wrong_category);
  EXPECT_FALSE(d::ResourceBudgetAccess::adopt_backing(
      budget, m::system_resource(), identity, 512, *wrong_category));
  EXPECT_TRUE(d::ResourceBudgetAccess::same_scope(*wrong_category, budget));
  wrong_category->reset(); token->reset();
  EXPECT_EQ(external(budget), 512u);
  second->reset();
  EXPECT_EQ(budget.usage(m::system_resource())->live_bytes, baseline);
}

TEST(SharedBackingBudget, ConcurrentLookupCreatesOneLiveCharge) {
  auto made = make_budget(); ASSERT_TRUE(made);
  auto &budget = *made;
  const auto baseline = budget.usage(m::system_resource())->live_bytes;
  int allocation;
  std::array<d::SharedBackingReservation, 2> holders;
  std::array<bool, 2> succeeded{};
  std::barrier start(3);
  const auto acquire = [&](std::size_t i) {
    start.arrive_and_wait();
    auto token = d::ResourceBudgetAccess::reserve_backing(
        budget, m::system_resource(), {&authority, &allocation}, 256);
    succeeded[i] = token.has_value();
    if (token) holders[i] = std::move(*token);
  };
  std::thread first(acquire, 0), second(acquire, 1);
  start.arrive_and_wait();
  first.join(); second.join();
  ASSERT_TRUE(succeeded[0]); ASSERT_TRUE(succeeded[1]);
  EXPECT_EQ(external(budget), 256u);
  holders[0].reset();
  EXPECT_EQ(external(budget), 256u);
  holders[1].reset();
  EXPECT_EQ(budget.usage(m::system_resource())->live_bytes, baseline);
}

TEST(SharedRequestedBudget, SelectedRowsAndHandleCopiesUseOneTotal) {
  HostResource first, second;
  const m::MemoryResourceRef a(first), b(second);
  const std::array rows{c::ResourceBudgetLimit{a, 1024}, c::ResourceBudgetLimit{b, 1024},
      c::ResourceBudgetLimit{m::system_resource(), 65536}};
  const std::array selected{a, b};
  auto made = c::ResourceBudget::create({.limits = rows}, selected, 256); ASSERT_TRUE(made);
  auto copy = *made;
  EXPECT_EQ(copy.shared_usage()->live_bytes, 0u); // CPU control is not nominated.
  auto left = made->reserve(a, 160); ASSERT_TRUE(left);
  EXPECT_FALSE(copy.reserve(b, 97));
  EXPECT_EQ(copy.usage(b)->live_bytes, 0u);
  auto right = copy.reserve(b, 96); ASSERT_TRUE(right);
  EXPECT_EQ(made->shared_usage()->live_bytes, 256u);
  EXPECT_EQ(made->shared_usage()->peak_bytes, 256u);
  EXPECT_EQ(made->shared_usage()->limit_bytes, 256u);
  left->reset(); right->reset();
  EXPECT_EQ(copy.shared_usage()->live_bytes, 0u);
  EXPECT_FALSE(copy.reserve(a, 900));
  EXPECT_EQ(copy.usage(a)->live_bytes, 0u);
  auto retry = copy.reserve(a, 256); EXPECT_TRUE(retry);
  auto legacy = c::ResourceBudget::create({.limits = rows}); ASSERT_TRUE(legacy);
  EXPECT_FALSE(legacy->shared_usage());
}

TEST(SharedRequestedBudget, NominatedCpuControlIsAdmittedBeforeBacking) {
  BackingProbe probe;
  const std::array rows{c::ResourceBudgetLimit{m::system_resource(), 65536}};
  const std::array selected{m::system_resource()};
  EXPECT_FALSE(c::ResourceBudget::create({.limits = rows, .allocator = probe.ref()}, selected, 0));
  EXPECT_EQ(probe.allocations, 0u);
  HostResource absent;
  const std::array bad{m::MemoryResourceRef(absent)};
  const std::array duplicate{m::system_resource(), m::system_resource()};
  EXPECT_FALSE(c::ResourceBudget::create({.limits = rows, .allocator = probe.ref()}, bad, 65536));
  EXPECT_FALSE(c::ResourceBudget::create({.limits = rows, .allocator = probe.ref()}, duplicate, 65536));
  EXPECT_FALSE(c::ResourceBudget::create({.limits = rows, .allocator = probe.ref()}, {}, 65536));
  EXPECT_EQ(probe.allocations, 0u);
  auto budget = c::ResourceBudget::create({.limits = rows, .allocator = probe.ref()}, selected, 65536);
  ASSERT_TRUE(budget);
  EXPECT_GT(budget->shared_usage()->live_bytes, 0u);
  EXPECT_EQ(budget->shared_usage()->live_bytes, budget->usage(m::system_resource())->live_bytes);
}

TEST(SharedRequestedBudget, CpuMetadataReallocationAndFailuresRollbackBothGates) {
  BackingProbe probe;
  const std::array rows{c::ResourceBudgetLimit{m::system_resource(), 65536}};
  const std::array selected{m::system_resource()};
  auto made = c::ResourceBudget::create({.limits = rows, .allocator = probe.ref()}, selected, 32768);
  ASSERT_TRUE(made);
  const auto base = made->shared_usage()->live_bytes;
  auto allocator = made->metadata_allocator().value();
  auto *p = static_cast<unsigned char *>(allocator.allocate(32, 16)); ASSERT_NE(p, nullptr);
  p[0] = 91;
  EXPECT_EQ(made->shared_usage()->live_bytes, base + 32);
  p = static_cast<unsigned char *>(allocator.reallocate(p, 32, 96, 16)); ASSERT_NE(p, nullptr);
  EXPECT_EQ(p[0], 91); EXPECT_EQ(made->shared_usage()->live_bytes, base + 96);
  const auto calls = probe.reallocations;
  EXPECT_EQ(allocator.reallocate(p, 96, 32768, 16), nullptr);
  EXPECT_EQ(probe.reallocations, calls); EXPECT_EQ(p[0], 91);
  EXPECT_EQ(made->shared_usage()->live_bytes, base + 96);
  probe.fail = true;
  EXPECT_EQ(allocator.reallocate(p, 96, 192, 16), nullptr);
  EXPECT_EQ(allocator.allocate(64, 16), nullptr);
  EXPECT_EQ(p[0], 91);
  EXPECT_EQ(made->shared_usage()->live_bytes, base + 96);
  EXPECT_EQ(made->usage(m::system_resource())->live_bytes, base + 96);
  probe.fail = false;
  p = static_cast<unsigned char *>(allocator.reallocate(p, 96, 16, 16)); ASSERT_NE(p, nullptr);
  EXPECT_EQ(p[0], 91); EXPECT_EQ(made->shared_usage()->live_bytes, base + 16);
  EXPECT_EQ(allocator.reallocate(p, 16, 0, 16), nullptr);
  EXPECT_EQ(made->shared_usage()->live_bytes, base);
  // The successful metadata allocation, not this borrowed allocator ref,
  // keeps the scope and its shared gate alive after its public handle dies.
  p = static_cast<unsigned char *>(allocator.allocate(32, 16)); ASSERT_NE(p, nullptr);
  made = c::ResourceBudget{}; allocator.deallocate(p, 32, 16);
}

TEST(SharedRequestedBudget, LocalRejectionReturnsSharedCapacityAndOpaqueStaysUnattributed) {
  HostResource resource;
  const m::MemoryResourceRef payload(resource);
  const std::array rows{c::ResourceBudgetLimit{payload, 16}, c::ResourceBudgetLimit{m::system_resource(), 65536}};
  const std::array selected{payload};
  auto made = c::ResourceBudget::create({.limits = rows, .allow_unattributed = true}, selected, 256);
  ASSERT_TRUE(made);
  EXPECT_FALSE(made->reserve(payload, 17));
  EXPECT_EQ(made->shared_usage()->live_bytes, 0u);
  auto known = made->reserve_unattributed(100); ASSERT_TRUE(known);
  auto unknown = made->reserve_unattributed(std::nullopt); ASSERT_TRUE(unknown);
  EXPECT_EQ(made->unattributed_usage().private_upper_bound_bytes, 100u);
  EXPECT_EQ(made->unattributed_usage().unknown_private_owners, 1u);
  EXPECT_EQ(made->shared_usage()->live_bytes, 0u);
  auto retry = made->reserve(payload, 16); EXPECT_TRUE(retry);
  EXPECT_EQ(made->shared_usage()->live_bytes, 16u);
}

TEST(SharedRequestedBudget, CpuLocalAllocatorLimitRollsBackTheSharedGrowth) {
  BackingProbe probe;
  const std::array rows{c::ResourceBudgetLimit{m::system_resource(), 65536}};
  const std::array selected{m::system_resource()};
  auto made = c::ResourceBudget::create({.limits = rows, .allocator = probe.ref()}, selected, 131072);
  ASSERT_TRUE(made); auto allocator = made->metadata_allocator().value();
  const auto base = made->usage(m::system_resource())->live_bytes;
  auto blocker = made->reserve(m::system_resource(), 65536 - base - 32); ASSERT_TRUE(blocker);
  const auto before = made->shared_usage()->live_bytes; const auto calls = probe.allocations;
  EXPECT_EQ(allocator.allocate(64, 16), nullptr);
  EXPECT_EQ(probe.allocations, calls); EXPECT_EQ(made->shared_usage()->live_bytes, before);
  auto *p = allocator.allocate(16, 16); ASSERT_NE(p, nullptr);
  const auto resizes = probe.reallocations;
  EXPECT_EQ(allocator.reallocate(p, 16, 96, 16), nullptr);
  EXPECT_EQ(probe.reallocations, resizes); EXPECT_EQ(made->shared_usage()->live_bytes, before + 16);
  allocator.deallocate(p, 16, 16); blocker->reset(); EXPECT_EQ(made->shared_usage()->live_bytes, base);
}

TEST(SharedRequestedBudget, OriginalBackingDedupRequiresTheSameScope) {
  auto rows = std::array{c::ResourceBudgetLimit{m::system_resource(), 65536}};
  const std::array selected{m::system_resource()};
  auto made = c::ResourceBudget::create({.limits = rows}, selected, 32768); ASSERT_TRUE(made);
  auto copy = *made; int allocation;
  auto existing = made->reserve(m::system_resource(), 256); ASSERT_TRUE(existing);
  auto first = d::ResourceBudgetAccess::adopt_backing(copy, m::system_resource(), {&authority, &allocation}, 256, *existing);
  ASSERT_TRUE(first); const auto used = made->shared_usage()->live_bytes;
  auto second = d::ResourceBudgetAccess::reserve_backing(*made, m::system_resource(), {&authority, &allocation}, 256);
  ASSERT_TRUE(second); EXPECT_EQ(made->shared_usage()->live_bytes, used);
  auto independent = c::ResourceBudget::create({.limits = rows}, selected, 32768); ASSERT_TRUE(independent);
  EXPECT_FALSE(first->same_scope(*independent));
  auto separate = d::ResourceBudgetAccess::reserve_backing(*independent, m::system_resource(), {&authority, &allocation}, 256);
  ASSERT_TRUE(separate); EXPECT_EQ(external(*independent), 256u);
  EXPECT_EQ(made->shared_usage()->live_bytes, used);
}

TEST(SharedRequestedBudget, ConcurrentDifferentRowsCannotExceedTheSharedTotal) {
  HostResource first, second;
  const m::MemoryResourceRef a(first), b(second);
  const std::array rows{c::ResourceBudgetLimit{a, 1024}, c::ResourceBudgetLimit{b, 1024},
      c::ResourceBudgetLimit{m::system_resource(), 65536}};
  const std::array selected{a, b};
  auto made = c::ResourceBudget::create({.limits = rows}, selected, 1024); ASSERT_TRUE(made);
  std::array<c::ResourceBudget::Reservation, 2> holders;
  std::array<bool, 2> accepted{}; std::barrier start(3);
  const auto reserve = [&](std::size_t i) {
    start.arrive_and_wait(); auto result = made->reserve(selected[i], 768);
    accepted[i] = result.has_value(); if (result) holders[i] = std::move(*result);
  };
  std::thread left(reserve, 0), right(reserve, 1); start.arrive_and_wait(); left.join(); right.join();
  EXPECT_NE(accepted[0], accepted[1]); EXPECT_EQ(made->shared_usage()->live_bytes, 768u);
  holders[0].reset(); holders[1].reset(); EXPECT_EQ(made->shared_usage()->live_bytes, 0u);
}
} // namespace
