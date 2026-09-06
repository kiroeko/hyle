#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "hyle/container/flat_sorted.h"
#include "hyle/mem/allocator.h"

namespace {

using hyle::container::FlatInsertResult;
using hyle::container::FlatMap;
using hyle::container::FlatSet;
using hyle::container::sorted_unique;

struct ToggleAllocatorState {
  hyle::mem::AllocatorRef backing = hyle::mem::system_allocator();
  bool fail = false;
  std::size_t allocation_calls = 0;
  std::size_t live_allocations = 0;
};

void *toggle_allocate(void *raw, std::size_t size,
                      std::size_t alignment) noexcept {
  auto *state = static_cast<ToggleAllocatorState *>(raw);
  ++state->allocation_calls;
  if (state->fail)
    return nullptr;
  void *result = state->backing.allocate(size, alignment);
  if (result != nullptr)
    ++state->live_allocations;
  return result;
}

void toggle_deallocate(void *raw, void *pointer, std::size_t size,
                       std::size_t alignment) noexcept {
  auto *state = static_cast<ToggleAllocatorState *>(raw);
  if (pointer != nullptr)
    --state->live_allocations;
  state->backing.deallocate(pointer, size, alignment);
}

void *toggle_reallocate(void *raw, void *pointer, std::size_t old_size,
                        std::size_t new_size,
                        std::size_t alignment) noexcept {
  auto *state = static_cast<ToggleAllocatorState *>(raw);
  if (state->fail)
    return nullptr;
  void *result = state->backing.reallocate(pointer, old_size, new_size,
                                            alignment);
  if (pointer == nullptr && result != nullptr)
    ++state->live_allocations;
  if (pointer != nullptr && new_size == 0)
    --state->live_allocations;
  return result;
}

hyle::mem::AllocatorRef toggle_allocator(ToggleAllocatorState &state) {
  return hyle::mem::AllocatorRef(toggle_allocate, toggle_deallocate,
                              toggle_reallocate, &state);
}

struct MoveOnly {
  explicit MoveOnly(int input = 0) noexcept : value(input) { ++alive; }
  MoveOnly(const MoveOnly &) = delete;
  MoveOnly &operator=(const MoveOnly &) = delete;
  MoveOnly(MoveOnly &&other) noexcept
      : value(std::exchange(other.value, -1)) {
    ++alive;
  }
  MoveOnly &operator=(MoveOnly &&) = delete;
  ~MoveOnly() noexcept { --alive; }

  static inline int alive = 0;
  int value;
};

struct MoveOnlyLess {
  using is_transparent = void;
  bool operator()(const MoveOnly &left,
                  const MoveOnly &right) const noexcept {
    return left.value < right.value;
  }
  bool operator()(const MoveOnly &left, int right) const noexcept {
    return left.value < right;
  }
  bool operator()(int left, const MoveOnly &right) const noexcept {
    return left < right.value;
  }
};

struct ThrowingMapped {
  explicit ThrowingMapped(int input) : value(input) {
    if (input == 13)
      throw std::runtime_error("requested construction failure");
    ++alive;
  }
  ThrowingMapped(const ThrowingMapped &other) : value(other.value) {
    if (copies_before_throw == 0)
      throw std::runtime_error("requested copy failure");
    --copies_before_throw;
    ++alive;
  }
  ThrowingMapped &operator=(const ThrowingMapped &) = delete;
  ThrowingMapped(ThrowingMapped &&other) noexcept
      : value(std::exchange(other.value, -1)) {
    ++alive;
  }
  ThrowingMapped &operator=(ThrowingMapped &&) = delete;
  ~ThrowingMapped() noexcept { --alive; }

  static inline int alive = 0;
  static inline int copies_before_throw = 100;
  int value;
};

struct CountingLess {
  std::size_t *calls = nullptr;
  bool operator()(int left, int right) const noexcept {
    ++*calls;
    return left < right;
  }
};

struct ThrowingLess {
  bool *fail = nullptr;
  bool operator()(int left, int right) const {
    if (*fail)
      throw std::runtime_error("requested comparator failure");
    return left < right;
  }
};

struct LargeFlatKey {
  int order = 0;
  std::array<std::byte, 512> payload{};

  friend bool operator==(const LargeFlatKey &, const LargeFlatKey &) = default;
};

struct LargeFlatValue {
  std::array<std::byte, 512> payload{};

  friend bool operator==(const LargeFlatValue &,
                         const LargeFlatValue &) = default;
};

struct LargeFlatLess {
  bool operator()(const LargeFlatKey &left,
                  const LargeFlatKey &right) const noexcept {
    return left.order < right.order;
  }
};

template <class Map>
void expect_same_map(const Map &actual,
                     const std::map<int, int> &expected) {
  ASSERT_EQ(actual.size(), expected.size());
  auto left = actual.begin();
  auto right = expected.begin();
  while (right != expected.end()) {
    ASSERT_NE(left, actual.end());
    EXPECT_EQ(left->key(), right->first);
    EXPECT_EQ(left->value(), right->second);
    ++left;
    ++right;
  }
  EXPECT_EQ(left, actual.end());
}

} // namespace

static_assert(std::random_access_iterator<FlatMap<int, int>::iterator>);
static_assert(!std::contiguous_iterator<FlatMap<int, int>::iterator>);
static_assert(std::contiguous_iterator<FlatSet<int>::iterator>);
static_assert(
    std::is_const_v<std::remove_reference_t<decltype(
        std::declval<FlatMap<int, int>::iterator>()->key())>>);
static_assert(!std::is_const_v<std::remove_reference_t<decltype(
                  std::declval<FlatMap<int, int>::iterator>()->value())>>);
static_assert(std::is_const_v<std::remove_reference_t<decltype(
                  *std::declval<FlatSet<int>::iterator>())>>);
static_assert(sizeof(FlatSet<LargeFlatKey, LargeFlatLess>) <
              sizeof(LargeFlatKey));
static_assert(sizeof(FlatMap<LargeFlatKey, LargeFlatValue, LargeFlatLess>) <
              sizeof(LargeFlatKey));

TEST(FlatSorted, EmptyObjectDoesNotEmbedKeyOrMappedValue) {
  FlatSet<LargeFlatKey, LargeFlatLess> set;
  FlatMap<LargeFlatKey, LargeFlatValue, LargeFlatLess> map;
  EXPECT_EQ(set.data(), nullptr);
  EXPECT_EQ(map.keys().data(), nullptr);
  EXPECT_EQ(map.values().data(), nullptr);
  EXPECT_EQ(set, (FlatSet<LargeFlatKey, LargeFlatLess>{}));
  EXPECT_EQ(map, (FlatMap<LargeFlatKey, LargeFlatValue, LargeFlatLess>{}));

  EXPECT_TRUE(set.insert(LargeFlatKey{.order = 7}).second);
  EXPECT_TRUE(map.insert(LargeFlatKey{.order = 9}, LargeFlatValue{}).second);
  EXPECT_EQ(set.begin()->order, 7);
  EXPECT_EQ(map.begin()->key().order, 9);
}

TEST(FlatSorted, IncrementalInsertUsesGeometricStorageGrowth) {
  ToggleAllocatorState state;
  {
    FlatMap<int, int, std::less<>, hyle::mem::AllocatorRef> map(
        std::less<>{}, toggle_allocator(state));
    for (int value = 0; value < 64; ++value)
      ASSERT_TRUE(map.insert(value, value).second);
    EXPECT_GE(map.capacity(), 64u);
    EXPECT_LE(state.allocation_calls, 16u);
    EXPECT_EQ(state.live_allocations, 2u);
  }
  EXPECT_EQ(state.live_allocations, 0u);
}

TEST(FlatMap, ProvidesOrderedSoaMapOperations) {
  FlatMap<std::string, int> map;
  EXPECT_TRUE(map.empty());
  EXPECT_TRUE(map.insert("gamma", 3).second);
  EXPECT_TRUE(map.insert(std::pair<std::string, int>{"zeta", 6}).second);
  EXPECT_TRUE(map.emplace("alpha", 1).second);
  EXPECT_TRUE(map.emplace("beta", 2).second);
  EXPECT_FALSE(map.emplace("beta", 200).second);
  EXPECT_EQ(map.find(std::string_view("beta"))->value(), 2);

  EXPECT_EQ(map.try_insert_or_assign("beta", 20),
            FlatInsertResult::kAssigned);
  EXPECT_TRUE(map.insert_or_assign("delta", 4));
  map["epsilon"] = 5;
  EXPECT_EQ(map.find("beta")->value(), 20);
  EXPECT_EQ(map.find("beta")->first, "beta");
  EXPECT_EQ(map.find("beta")->second, 20);
  EXPECT_TRUE(map.contains(std::string_view("delta")));
  EXPECT_EQ(map.count("absent"), 0u);

  std::vector<std::string> keys;
  for (const auto &[key, value] : map) {
    keys.push_back(key);
    EXPECT_EQ(value, map.find(key)->second);
  }
  EXPECT_EQ(keys, (std::vector<std::string>{"alpha", "beta", "delta",
                                             "epsilon", "gamma", "zeta"}));
  EXPECT_EQ(map.lower_bound("charlie")->key(), "delta");
  EXPECT_EQ(map.upper_bound("delta")->key(), "epsilon");
  auto [first, last] = map.equal_range(std::string_view("beta"));
  ASSERT_NE(first, last);
  EXPECT_EQ(first->value(), 20);
  EXPECT_EQ(++first, last);

  EXPECT_TRUE(map.erase(std::string_view("epsilon")));
  EXPECT_FALSE(map.erase("epsilon"));
  map.erase(map.begin() + 1);
  EXPECT_EQ(map.size(), 4u);
  map.erase(map.begin() + 1, map.end());
  ASSERT_EQ(map.size(), 1u);
  EXPECT_EQ(map.begin()->key(), "alpha");
  ASSERT_EQ(map.keys().size(), map.values().size());
  EXPECT_EQ(map.keys().front(), "alpha");
  map.values().front() = 10;
  EXPECT_EQ(map.begin()->second, 10);
  EXPECT_EQ(map.rbegin()->first, "alpha");
}

TEST(FlatSet, KeepsKeysConstAndSupportsHeterogeneousRanges) {
  FlatSet<std::string> set{"gamma", "alpha", "beta", "beta"};
  EXPECT_EQ(set.size(), 3u);
  EXPECT_TRUE(set.contains(std::string_view("alpha")));
  EXPECT_EQ(*set.lower_bound("aardvark"), "alpha");
  EXPECT_EQ(*set.upper_bound("alpha"), "beta");
  auto [first, last] = set.equal_range(std::string_view("gamma"));
  ASSERT_NE(first, last);
  EXPECT_EQ(*first, "gamma");
  EXPECT_EQ(++first, last);
  EXPECT_TRUE(set.erase(std::string_view("beta")));
  EXPECT_EQ((std::vector<std::string>(set.begin(), set.end())),
            (std::vector<std::string>{"alpha", "gamma"}));
}

TEST(FlatSorted, ComparatorExceptionsPropagateWithoutMutation) {
  bool fail = false;
  FlatMap<int, int, ThrowingLess> map(ThrowingLess{&fail});
  map.insert(1, 10);
  map.insert(3, 30);

  fail = true;
  EXPECT_THROW((void)map.contains(2), std::runtime_error);
  EXPECT_THROW((void)map.erase(1), std::runtime_error);

  fail = false;
  ASSERT_EQ(map.size(), 2u);
  EXPECT_EQ(map.find(1)->value(), 10);
  EXPECT_EQ(map.find(3)->value(), 30);
}

TEST(FlatMap, RandomDifferentialMatchesStdMap) {
  FlatMap<int, int> actual;
  std::map<int, int> expected;
  std::mt19937 generator(0xF1A7u);
  std::uniform_int_distribution<int> keys(-600, 600);

  for (int operation = 0; operation < 40'000; ++operation) {
    const int key = keys(generator);
    const int value = static_cast<int>(generator());
    switch (generator() % 7) {
    case 0:
    case 1:
      EXPECT_EQ(actual.insert_or_assign(key, value),
                expected.insert_or_assign(key, value).second);
      break;
    case 2:
      EXPECT_EQ(actual.insert(key, value).second,
                expected.insert({key, value}).second);
      break;
    case 3:
      EXPECT_EQ(actual.erase(key), expected.erase(key) != 0);
      break;
    case 4: {
      const auto left = actual.find(key);
      const auto right = expected.find(key);
      EXPECT_EQ(left == actual.end(), right == expected.end());
      if (right != expected.end()) {
        EXPECT_EQ(left->value(), right->second);
      }
      break;
    }
    case 5:
      EXPECT_EQ(static_cast<std::size_t>(actual.lower_bound(key) -
                                         actual.begin()),
                static_cast<std::size_t>(
                    std::distance(expected.begin(), expected.lower_bound(key))));
      break;
    default:
      if (!expected.empty()) {
        const std::size_t first = generator() % expected.size();
        const std::size_t count = generator() % (expected.size() - first + 1);
        actual.erase(actual.begin() + first, actual.begin() + first + count);
        auto expected_first = std::next(expected.begin(), first);
        auto expected_last = std::next(expected_first, count);
        expected.erase(expected_first, expected_last);
      }
      break;
    }
    if ((operation & 255) == 0) {
      expect_same_map(actual, expected);
    }
  }
  expect_same_map(actual, expected);
}

TEST(FlatSet, RandomDifferentialMatchesStdSet) {
  FlatSet<int> actual;
  std::set<int> expected;
  std::mt19937 generator(0x5E7u);
  std::uniform_int_distribution<int> keys(-900, 900);

  for (int operation = 0; operation < 35'000; ++operation) {
    const int key = keys(generator);
    if ((generator() & 1) == 0) {
      EXPECT_EQ(actual.insert(key).second, expected.insert(key).second);
    } else {
      EXPECT_EQ(actual.erase(key), expected.erase(key) != 0);
    }
    if ((operation & 255) == 0) {
      EXPECT_TRUE(std::equal(actual.begin(), actual.end(), expected.begin(),
                             expected.end()));
    }
  }
  EXPECT_TRUE(std::equal(actual.begin(), actual.end(), expected.begin(),
                         expected.end()));
}

TEST(FlatSorted, SortedUniqueBulkConstructionAndAssignAreLinear) {
  std::vector<std::pair<int, int>> map_values;
  std::vector<int> set_values;
  for (int value = 0; value < 4096; ++value) {
    map_values.emplace_back(value * 2, value);
    set_values.push_back(value * 3);
  }

  std::size_t map_calls = 0;
  FlatMap<int, int, CountingLess> map(
      sorted_unique, map_values.begin(), map_values.end(),
      CountingLess{&map_calls});
  EXPECT_EQ(map.size(), map_values.size());
  EXPECT_LT(map_calls, map_values.size() * 2);
  EXPECT_EQ(map.keys()[2048], 4096);
  EXPECT_EQ(map.values()[2048], 2048);

  std::size_t set_calls = 0;
  FlatSet<int, CountingLess> set(sorted_unique, set_values.begin(),
                                 set_values.end(), CountingLess{&set_calls});
  EXPECT_EQ(set.size(), set_values.size());
  EXPECT_LT(set_calls, set_values.size() * 2);

  const std::vector<std::pair<int, int>> replacement{{10, 1}, {20, 2},
                                                      {30, 3}};
  map.assign(sorted_unique, replacement.begin(), replacement.end());
  EXPECT_EQ(map.size(), 3u);
  EXPECT_EQ(map.find(20)->value(), 2);
}

TEST(FlatSorted, MoveOnlyKeysAndValuesHaveBalancedLifetime) {
  EXPECT_EQ(MoveOnly::alive, 0);
  {
    FlatMap<MoveOnly, MoveOnly, MoveOnlyLess> map;
    EXPECT_TRUE(map.emplace(MoveOnly(3), 30).second);
    EXPECT_TRUE(map.emplace(MoveOnly(1), 10).second);
    EXPECT_TRUE(map.emplace(MoveOnly(2), 20).second);
    EXPECT_FALSE(map.emplace(MoveOnly(2), 200).second);
    ASSERT_NE(map.find(3), map.end());
    EXPECT_EQ(map.find(3)->value().value, 30);
    EXPECT_TRUE(map.erase(1));

    FlatSet<MoveOnly, MoveOnlyLess> set;
    EXPECT_TRUE(set.insert(MoveOnly(8)).second);
    EXPECT_TRUE(set.insert(MoveOnly(5)).second);
    EXPECT_FALSE(set.insert(MoveOnly(8)).second);
    EXPECT_TRUE(set.contains(5));
    EXPECT_TRUE(set.erase(8));

    std::vector<std::pair<MoveOnly, MoveOnly>> ordered_map;
    ordered_map.emplace_back(MoveOnly(10), MoveOnly(100));
    ordered_map.emplace_back(MoveOnly(20), MoveOnly(200));
    FlatMap<MoveOnly, MoveOnly, MoveOnlyLess> bulk_map(
        sorted_unique, std::make_move_iterator(ordered_map.begin()),
        std::make_move_iterator(ordered_map.end()));
    ASSERT_EQ(bulk_map.size(), 2u);
    EXPECT_EQ(bulk_map.find(20)->value().value, 200);

    std::vector<MoveOnly> ordered_set;
    ordered_set.emplace_back(30);
    ordered_set.emplace_back(40);
    FlatSet<MoveOnly, MoveOnlyLess> bulk_set(
        sorted_unique, std::make_move_iterator(ordered_set.begin()),
        std::make_move_iterator(ordered_set.end()));
    EXPECT_TRUE(bulk_set.contains(30));
    EXPECT_TRUE(bulk_set.contains(40));
  }
  EXPECT_EQ(MoveOnly::alive, 0);
}

TEST(FlatMap, GrowthPreservesAliasedMappedArguments) {
  FlatMap<std::string, std::string> map;
  map.reserve(2);
  map.insert("a", "shared payload");
  map.insert("b", "other");
  ASSERT_EQ(map.size(), map.capacity());

  const std::string &borrowed = map.find("a")->value();
  EXPECT_TRUE(map.emplace("c", borrowed).second);
  EXPECT_EQ(map.find("c")->value(), "shared payload");

  const std::string &aliased_key = map.find("a")->key();
  EXPECT_FALSE(map.insert(aliased_key, "replacement").second);
  EXPECT_EQ(map.find("a")->value(), "shared payload");
}

TEST(FlatSorted, AllocationFailurePreservesExistingContainer) {
  ToggleAllocatorState state;
  {
    FlatMap<int, int, std::less<>, hyle::mem::AllocatorRef> singleton(
        std::less<>{}, toggle_allocator(state));
    state.fail = true;
    EXPECT_EQ(singleton.try_insert(1, 10), FlatInsertResult::kOutOfMemory);
    EXPECT_TRUE(singleton.empty());
    EXPECT_EQ(state.live_allocations, 0u);
    state.fail = false;
    EXPECT_EQ(singleton.try_insert(1, 10), FlatInsertResult::kInserted);
    EXPECT_EQ(singleton.size(), 1u);
  }
  {
    FlatMap<int, int, std::less<>, hyle::mem::AllocatorRef> map(
        std::less<>{}, toggle_allocator(state));
    ASSERT_TRUE(map.try_reserve(2));
    EXPECT_EQ(map.try_insert(1, 10), FlatInsertResult::kInserted);
    EXPECT_EQ(map.try_insert(2, 20), FlatInsertResult::kInserted);
    ASSERT_EQ(map.size(), map.capacity());

    state.fail = true;
    EXPECT_EQ(map.try_insert(3, 30), FlatInsertResult::kOutOfMemory);
    EXPECT_EQ(map.try_insert_or_assign(4, 40),
              FlatInsertResult::kOutOfMemory);
    EXPECT_EQ(map.size(), 2u);
    EXPECT_EQ(map.find(1)->value(), 10);
    EXPECT_EQ(map.find(2)->value(), 20);

    const std::vector<std::pair<int, int>> replacement{{5, 50}, {6, 60},
                                                        {7, 70}};
    EXPECT_FALSE(map.try_assign(sorted_unique, replacement.begin(),
                                replacement.end()));
    EXPECT_EQ(map.size(), 2u);
    EXPECT_TRUE(map.contains(1));
    EXPECT_TRUE(map.contains(2));
  }
  EXPECT_EQ(state.live_allocations, 0u);
}

TEST(FlatMap, ThrowingConstructionLeavesOrderingAndLifetimeUnchanged) {
  ThrowingMapped::alive = 0;
  ThrowingMapped::copies_before_throw = 100;
  {
    FlatMap<int, ThrowingMapped> map;
    map.emplace(1, 1);
    map.emplace(3, 3);
    ASSERT_EQ(ThrowingMapped::alive, 2);

    EXPECT_THROW(map.emplace(2, 13), std::runtime_error);
    EXPECT_EQ(map.size(), 2u);
    EXPECT_TRUE(map.contains(1));
    EXPECT_TRUE(map.contains(3));
    EXPECT_FALSE(map.contains(2));
    EXPECT_EQ(ThrowingMapped::alive, 2);

    ThrowingMapped::copies_before_throw = 0;
    std::vector<std::pair<int, ThrowingMapped>> replacement;
    replacement.emplace_back(7, ThrowingMapped(7));
    EXPECT_THROW(map.try_assign(sorted_unique, replacement.begin(),
                                replacement.end()),
                 std::runtime_error);
    EXPECT_EQ(map.size(), 2u);
    EXPECT_TRUE(map.contains(1));
    EXPECT_TRUE(map.contains(3));
  }
  EXPECT_EQ(ThrowingMapped::alive, 0);
}
