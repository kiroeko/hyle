#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "hyle/container/btree.h"
#include "hyle/mem/allocator.h"

namespace {

using hyle::container::BTreeMap;
using hyle::container::BTreeSet;

struct ToggleAllocatorState {
  hyle::mem::AllocatorRef backing = hyle::mem::system_allocator();
  bool fail = false;
};

void *toggle_allocate(void *raw, std::size_t size,
                      std::size_t alignment) noexcept {
  auto *state = static_cast<ToggleAllocatorState *>(raw);
  return state->fail ? nullptr : state->backing.allocate(size, alignment);
}

void toggle_deallocate(void *raw, void *pointer, std::size_t size,
                       std::size_t alignment) noexcept {
  static_cast<ToggleAllocatorState *>(raw)->backing.deallocate(pointer, size,
                                                               alignment);
}

void *toggle_reallocate(void *raw, void *pointer, std::size_t old_size,
                        std::size_t new_size, std::size_t alignment) noexcept {
  auto *state = static_cast<ToggleAllocatorState *>(raw);
  return state->fail ? nullptr
                     : state->backing.reallocate(pointer, old_size, new_size,
                                                 alignment);
}

struct MoveOnly {
  explicit MoveOnly(int input = 0) noexcept : value(input) {}
  MoveOnly(const MoveOnly &) = delete;
  MoveOnly &operator=(const MoveOnly &) = delete;
  MoveOnly(MoveOnly &&other) noexcept : value(std::exchange(other.value, -1)) {}
  MoveOnly &operator=(MoveOnly &&other) noexcept {
    value = std::exchange(other.value, -1);
    return *this;
  }
  friend bool operator<(const MoveOnly &left, const MoveOnly &right) noexcept {
    return left.value < right.value;
  }
  int value;
};

struct ConstructionCounted {
  static inline int constructions = 0;
  explicit ConstructionCounted(int input) noexcept : value(input) {
    ++constructions;
  }
  ConstructionCounted(const ConstructionCounted &) = delete;
  ConstructionCounted &operator=(const ConstructionCounted &) = delete;
  ConstructionCounted(ConstructionCounted &&other) noexcept
      : value(std::exchange(other.value, -1)) {
    ++constructions;
  }
  ConstructionCounted &operator=(ConstructionCounted &&other) noexcept {
    value = std::exchange(other.value, -1);
    return *this;
  }
  int value;
};

struct CountingLess {
  std::size_t *comparisons;
  bool operator()(int left, int right) const noexcept {
    ++*comparisons;
    return left < right;
  }
};

template <class HyleMap>
void expect_same_map(const HyleMap &actual,
                     const std::map<int, int> &expected) {
  ASSERT_EQ(actual.size(), expected.size());
  auto actual_it = actual.begin();
  auto expected_it = expected.begin();
  while (expected_it != expected.end()) {
    ASSERT_NE(actual_it, actual.end());
    EXPECT_EQ(actual_it->key(), expected_it->first);
    EXPECT_EQ(actual_it->value(), expected_it->second);
    ++actual_it;
    ++expected_it;
  }
  EXPECT_EQ(actual_it, actual.end());
}

template <class HyleSet>
void expect_same_set(const HyleSet &actual, const std::set<int> &expected) {
  ASSERT_EQ(actual.size(), expected.size());
  EXPECT_TRUE(std::equal(actual.begin(), actual.end(), expected.begin(),
                         expected.end()));
}

} // namespace

static_assert(std::bidirectional_iterator<BTreeMap<int, int>::iterator>);
static_assert(std::bidirectional_iterator<BTreeMap<int, int>::const_iterator>);
static_assert(std::bidirectional_iterator<BTreeSet<int>::iterator>);
static_assert(std::is_trivially_copyable_v<BTreeMap<int, int>::value_type>);
static_assert(!std::is_assignable_v<BTreeMap<int, int>::value_type &,
                                    BTreeMap<int, int>::value_type &&>);

TEST(BTreeMap, ProvidesOrderedIndexOperationsAndMutableValues) {
  BTreeMap<int, std::string> map;
  EXPECT_TRUE(map.empty());
  EXPECT_TRUE(map.insert(7, "seven").second);
  EXPECT_TRUE(map.try_emplace(2, "two").second);
  EXPECT_TRUE(map.try_emplace(11, "eleven").second);
  EXPECT_FALSE(map.try_emplace(7, "replacement").second);
  EXPECT_EQ(map.find(7)->value(), "seven");

  EXPECT_FALSE(map.insert_or_assign(7, "SEVEN"));
  EXPECT_TRUE(map.insert_or_assign(5, "five"));
  map[3] = "three";
  EXPECT_EQ(map.find(7)->value(), "SEVEN");
  EXPECT_EQ(map.count(99), 0u);
  EXPECT_TRUE(map.contains(11L));

  std::vector<int> keys;
  for (const auto &entry : map)
    keys.push_back(entry.key());
  EXPECT_EQ(keys, (std::vector<int>{2, 3, 5, 7, 11}));

  EXPECT_EQ(map.lower_bound(4)->key(), 5);
  EXPECT_EQ(map.upper_bound(7)->key(), 11);
  EXPECT_EQ(map.lower_bound(12), map.end());
  auto [first, last] = map.equal_range(7);
  ASSERT_NE(first, last);
  EXPECT_EQ(first->key(), 7);
  EXPECT_EQ(++first, last);

  std::vector<int> reverse_keys;
  for (auto it = map.rbegin(); it != map.rend(); ++it)
    reverse_keys.push_back(it->key());
  EXPECT_EQ(reverse_keys, (std::vector<int>{11, 7, 5, 3, 2}));

  // The owning-key erase overload must remain safe when the argument aliases
  // a slot that can move during B-tree rebalancing.
  const auto aliased = map.find(5);
  ASSERT_NE(aliased, map.end());
  EXPECT_TRUE(map.erase(aliased->key()));
  EXPECT_FALSE(map.erase(5));
}

TEST(BTreeMap, RandomMutableIndexMatchesStdMapAcrossSplitMergeAndBorrow) {
  BTreeMap<int, int> actual;
  std::map<int, int> expected;
  std::mt19937 generator(0xB7EEu);
  std::uniform_int_distribution<int> key_distribution(-1200, 1200);

  for (int operation = 0; operation < 80'000; ++operation) {
    const int key = key_distribution(generator);
    const int value = static_cast<int>(generator());
    switch (generator() % 6) {
    case 0:
    case 1: {
      const bool actual_inserted = actual.insert_or_assign(key, value);
      const bool expected_inserted =
          expected.insert_or_assign(key, value).second;
      EXPECT_EQ(actual_inserted, expected_inserted);
      break;
    }
    case 2:
      EXPECT_EQ(actual.erase(key), expected.erase(key) != 0);
      break;
    case 3: {
      const auto actual_it = actual.find(key);
      const auto expected_it = expected.find(key);
      ASSERT_EQ(actual_it == actual.end(), expected_it == expected.end());
      if (actual_it != actual.end()) {
        EXPECT_EQ(actual_it->value(), expected_it->second);
      }
      break;
    }
    case 4: {
      const auto actual_it = actual.lower_bound(key);
      const auto expected_it = expected.lower_bound(key);
      ASSERT_EQ(actual_it == actual.end(), expected_it == expected.end());
      if (actual_it != actual.end()) {
        EXPECT_EQ(actual_it->key(), expected_it->first);
      }
      break;
    }
    default: {
      const auto actual_it = actual.upper_bound(key);
      const auto expected_it = expected.upper_bound(key);
      ASSERT_EQ(actual_it == actual.end(), expected_it == expected.end());
      if (actual_it != actual.end()) {
        EXPECT_EQ(actual_it->key(), expected_it->first);
      }
    }
    }
    if ((operation % 251) == 0)
      expect_same_map(actual, expected);
  }
  expect_same_map(actual, expected);

  std::vector<int> remaining;
  remaining.reserve(expected.size());
  for (const auto &[key, value] : expected) {
    (void)value;
    remaining.push_back(key);
  }
  std::shuffle(remaining.begin(), remaining.end(), generator);
  for (int key : remaining)
    ASSERT_TRUE(actual.erase(key));
  EXPECT_TRUE(actual.empty());
  EXPECT_EQ(actual.begin(), actual.end());
}

TEST(BTreeSet, RandomUniqueIndexMatchesStdSet) {
  BTreeSet<int> actual;
  std::set<int> expected;
  std::mt19937 generator(0x5E7u);
  std::uniform_int_distribution<int> key_distribution(-1800, 1800);
  for (int operation = 0; operation < 60'000; ++operation) {
    const int key = key_distribution(generator);
    if ((generator() & 3u) != 0) {
      EXPECT_EQ(actual.insert(key).second, expected.insert(key).second);
    } else {
      EXPECT_EQ(actual.erase(key), expected.erase(key) != 0);
    }
    if ((operation % 257) == 0)
      expect_same_set(actual, expected);
  }
  expect_same_set(actual, expected);
}

TEST(BTreeMap, SupportsDescendingOrderCopyMoveAndMoveOnlyMappedValues) {
  BTreeMap<int, int, std::greater<>> descending;
  descending.insert({{1, 10}, {3, 30}, {2, 20}});
  std::vector<int> keys;
  for (const auto &entry : descending)
    keys.push_back(entry.key());
  EXPECT_EQ(keys, (std::vector<int>{3, 2, 1}));

  BTreeMap<int, int, std::greater<>> copy = descending;
  expect_same_map(BTreeMap<int, int>{{1, 10}, {2, 20}, {3, 30}},
                  std::map<int, int>{{1, 10}, {2, 20}, {3, 30}});
  EXPECT_EQ(copy.size(), descending.size());
  BTreeMap<int, int, std::greater<>> moved = std::move(copy);
  EXPECT_TRUE(copy.empty());
  EXPECT_EQ(moved.begin()->key(), 3);

  BTreeMap<int, MoveOnly> move_only;
  EXPECT_TRUE(move_only.try_emplace(4, 44).second);
  EXPECT_TRUE(move_only.insert_or_assign(4, MoveOnly{45}) == false);
  EXPECT_EQ(move_only.find(4)->value().value, 45);
}

TEST(BTreeMap, CopyPreservesTreeShapeInLinearTime) {
  std::size_t comparisons = 0;
  BTreeMap<int, int, CountingLess> original(CountingLess{&comparisons});
  for (int key = 0; key < 10'000; ++key)
    original.insert(key, key * 2);

  comparisons = 0;
  BTreeMap<int, int, CountingLess> copy(original);
  EXPECT_EQ(comparisons, 0u);
  EXPECT_EQ(copy.size(), original.size());
  EXPECT_EQ(copy.begin()->key(), 0);
  EXPECT_EQ(copy.rbegin()->key(), 9'999);

  EXPECT_TRUE(copy.erase(5'000));
  EXPECT_TRUE(original.contains(5'000));
}

TEST(BTreeSet, ErasesAliasedMoveOnlyKeyWithoutSnapshot) {
  BTreeSet<MoveOnly> set;
  EXPECT_TRUE(set.insert(MoveOnly{7}).second);
  EXPECT_TRUE(set.insert(MoveOnly{11}).second);
  EXPECT_TRUE(set.erase(*set.begin()));
  EXPECT_EQ(set.size(), 1u);
  EXPECT_EQ(set.begin()->value, 11);
  EXPECT_TRUE(set.insert(MoveOnly{13}).second);
  EXPECT_TRUE(set.insert(MoveOnly{17}).second);
  auto suffix = set.begin();
  ++suffix;
  EXPECT_EQ(set.erase(suffix, set.end()), set.end());
  EXPECT_EQ(set.size(), 1u);
  EXPECT_EQ(set.begin()->value, 11);
}

TEST(BTreeMap, IteratorAndRangeEraseReturnFreshOrderedPositions) {
  BTreeMap<int, int> map;
  for (int key = 0; key < 1'000; ++key)
    map.insert(key, key * 2);

  auto next = map.erase(map.find(200));
  ASSERT_NE(next, map.end());
  EXPECT_EQ(next->key(), 201);
  EXPECT_FALSE(map.contains(200));

  next = map.erase(map.lower_bound(300), map.lower_bound(700));
  ASSERT_NE(next, map.end());
  EXPECT_EQ(next->key(), 700);
  EXPECT_EQ(map.size(), 599u);
  for (int key = 300; key < 700; ++key)
    EXPECT_FALSE(map.contains(key));

  next = map.erase(map.lower_bound(900), map.end());
  EXPECT_EQ(next, map.end());
  EXPECT_EQ(map.rbegin()->key(), 899);
}

TEST(BTreeMap, RandomRangeExpiryMatchesStdMap) {
  BTreeMap<int, int> actual;
  std::map<int, int> expected;
  std::mt19937 generator(0xE8A5Eu);
  for (int key = 0; key < 5'000; ++key) {
    actual.insert(key, key * 2);
    expected.emplace(key, key * 2);
  }

  for (int operation = 0; operation < 2'000; ++operation) {
    const int lower = static_cast<int>(generator() % 7'000);
    const int upper = lower + static_cast<int>(generator() % 200);
    const auto actual_next =
        actual.erase(actual.lower_bound(lower), actual.lower_bound(upper));
    const auto expected_next = expected.erase(expected.lower_bound(lower),
                                              expected.lower_bound(upper));
    ASSERT_EQ(actual_next == actual.end(), expected_next == expected.end());
    if (actual_next != actual.end()) {
      EXPECT_EQ(actual_next->key(), expected_next->first);
    }

    for (int refill = 0; refill < 8; ++refill) {
      const int key = static_cast<int>(generator() % 7'000);
      actual.insert_or_assign(key, key * 2);
      expected.insert_or_assign(key, key * 2);
    }
    if ((operation % 37) == 0)
      expect_same_map(actual, expected);
  }
  expect_same_map(actual, expected);
}

TEST(BTreeMap, RepeatedFrontExpiryAndRefillMatchesStdMap) {
  BTreeMap<int, int> actual;
  std::map<int, int> expected;
  constexpr int kLive = 4'096;
  constexpr int kExpire = 17;
  for (int key = 0; key < kLive; ++key) {
    actual.insert(key, key * 2);
    expected.emplace(key, key * 2);
  }

  int next = kLive;
  for (int round = 0; round < 1'000; ++round) {
    const int cutoff = actual.begin()->key() + kExpire;
    const auto actual_next = actual.erase(actual.begin(),
                                          actual.lower_bound(cutoff));
    const auto expected_next = expected.erase(expected.begin(),
                                              expected.lower_bound(cutoff));
    ASSERT_NE(actual_next, actual.end());
    ASSERT_NE(expected_next, expected.end());
    EXPECT_EQ(actual_next->key(), expected_next->first);

    for (int index = 0; index < kExpire; ++index, ++next) {
      actual.insert(next, next * 2);
      expected.emplace(next, next * 2);
    }
    if ((round % 31) == 0)
      expect_same_map(actual, expected);
  }
  expect_same_map(actual, expected);
}

TEST(BTreeMap, RepeatedBackExpiryAndRefillMatchesStdMap) {
  BTreeMap<int, int> actual;
  std::map<int, int> expected;
  constexpr int kLive = 4'096;
  constexpr int kExpire = 17;
  for (int key = kLive - 1; key >= 0; --key) {
    actual.insert(key, key * 2);
    expected.emplace(key, key * 2);
  }

  int next = -1;
  for (int round = 0; round < 1'000; ++round) {
    const int cutoff = std::prev(actual.end())->key() - kExpire + 1;
    const auto actual_next = actual.erase(actual.lower_bound(cutoff),
                                          actual.end());
    const auto expected_next = expected.erase(expected.lower_bound(cutoff),
                                              expected.end());
    EXPECT_EQ(actual_next, actual.end());
    EXPECT_EQ(expected_next, expected.end());

    for (int index = 0; index < kExpire; ++index, --next) {
      actual.insert(next, next * 2);
      expected.emplace(next, next * 2);
    }
    if ((round % 31) == 0)
      expect_same_map(actual, expected);
  }
  expect_same_map(actual, expected);
}

TEST(BTreeMap, DuplicateTryEmplaceDoesNotConstructMappedValue) {
  ConstructionCounted::constructions = 0;
  BTreeMap<int, ConstructionCounted> map;
  EXPECT_TRUE(map.try_emplace(7, 70).second);
  const int after_insert = ConstructionCounted::constructions;
  EXPECT_FALSE(map.try_emplace(7, 700).second);
  EXPECT_EQ(ConstructionCounted::constructions, after_insert);
  EXPECT_EQ(map.find(7)->value().value, 70);
}

TEST(BTreeMap, FallibleInsertPreservesContentsOnAllocationFailure) {
  ToggleAllocatorState state;
  hyle::mem::AllocatorRef allocator(toggle_allocate, toggle_deallocate,
                                 toggle_reallocate, &state);
  using Map = BTreeMap<int, int, std::less<>, hyle::mem::AllocatorRef>;
  Map map(std::less<>{}, allocator);
  state.fail = true;
  EXPECT_EQ(map.try_insert(1, 10), Map::InsertResult::kOutOfMemory);
  EXPECT_TRUE(map.empty());

  state.fail = false;
  for (int key = 0; key < 500; ++key)
    ASSERT_EQ(map.try_insert(key, key * 2), Map::InsertResult::kInserted);
  const std::size_t size_before = map.size();

  state.fail = true;
  bool observed_failure = false;
  for (int key = 500; key < 5'000; ++key) {
    const auto *first_entry_before = std::addressof(*map.begin());
    const auto result = map.try_insert(key, key * 2);
    if (result == Map::InsertResult::kOutOfMemory) {
      observed_failure = true;
      EXPECT_EQ(std::addressof(*map.begin()), first_entry_before);
      break;
    }
  }
  EXPECT_TRUE(observed_failure);
  EXPECT_GE(map.size(), size_before);
  for (int key = 0; key < 500; ++key) {
    const auto it = map.find(key);
    ASSERT_NE(it, map.end());
    EXPECT_EQ(it->value(), key * 2);
  }
}
