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
} // namespace
