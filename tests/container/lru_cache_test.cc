#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "hyle/container/lru_cache.h"

namespace {

using hyle::container::LruCache;
using hyle::container::LruCacheLimits;
using hyle::container::LruInsertStatus;

struct FailingAllocator {
  bool* fail = nullptr;

  void* allocate(std::size_t size, std::size_t alignment) const noexcept {
    if (*fail) return nullptr;
    return hyle::mem::SystemAllocator::allocate(size, alignment);
  }
  void deallocate(void* pointer, std::size_t size,
                  std::size_t alignment) const noexcept {
    hyle::mem::SystemAllocator::deallocate(pointer, size, alignment);
  }
};

TEST(LruCache, HitPromotesAndEntryLimitEvictsLeastRecent) {
  LruCache<int, std::string> cache({.max_entries = 2});
  ASSERT_TRUE(cache.try_insert_or_assign(1, "one").stored());
  ASSERT_TRUE(cache.try_insert_or_assign(2, "two").stored());
  ASSERT_NE(cache.get(1), nullptr);  // 2 is now least recent.

  std::vector<int> evicted;
  auto result = cache.try_insert_or_assign(
      3, "three", 1,
      [&](int&& key, std::string&&, std::size_t) noexcept {
        evicted.push_back(key);
      });
  ASSERT_EQ(result.status, LruInsertStatus::kInserted);
  EXPECT_EQ(result.evicted, 1u);
  ASSERT_EQ(evicted.size(), 1u);
  EXPECT_EQ(evicted[0], 2);
  EXPECT_TRUE(cache.contains(1));
  EXPECT_TRUE(cache.contains(3));
  EXPECT_FALSE(cache.contains(2));
}

TEST(LruCache, WeightAndAssignmentShareOneNaturalContract) {
  LruCache<int, int> cache({.max_entries = 4, .max_weight = 5});
  ASSERT_TRUE(cache.try_insert_or_assign(1, 10, 2).stored());
  ASSERT_TRUE(cache.try_insert_or_assign(2, 20, 3).stored());

  auto assigned = cache.try_insert_or_assign(1, 11, 4);
  EXPECT_EQ(assigned.status, LruInsertStatus::kAssigned);
  EXPECT_EQ(assigned.evicted, 1u);
  ASSERT_NE(assigned.value, nullptr);
  EXPECT_EQ(*assigned.value, 11);
  EXPECT_EQ(cache.total_weight(), 4u);
  EXPECT_FALSE(cache.contains(2));

  const auto rejected = cache.try_insert_or_assign(3, 30, 6);
  EXPECT_EQ(rejected.status, LruInsertStatus::kRejected);
  EXPECT_FALSE(rejected.stored());
  EXPECT_FALSE(cache.contains(3));
}

TEST(LruCache, PeekDoesNotPromoteAndIterationIsMruFirst) {
  LruCache<int, int> cache({.max_entries = 3});
  cache.insert_or_assign(1, 10);
  cache.insert_or_assign(2, 20);
  cache.insert_or_assign(3, 30);
  ASSERT_NE(cache.peek(1), nullptr);

  std::vector<int> order;
  cache.for_each_mru(
      [&](const int& key, const int&, std::size_t) { order.push_back(key); });
  EXPECT_EQ(order, (std::vector<int>{3, 2, 1}));
}

TEST(LruCache, ReportsAllocatorFailureWithoutMutation) {
  bool fail = true;
  LruCache<int, int, hyle::container::Hash<int>, std::equal_to<>,
           FailingAllocator>
      cache({.max_entries = 2}, FailingAllocator{&fail});
  const auto result = cache.try_insert_or_assign(1, 2);
  EXPECT_EQ(result.status, LruInsertStatus::kOutOfMemory);
  EXPECT_TRUE(cache.empty());
}

}  // namespace
