#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "hyle/container/static_vector.h"

using hyle::container::InplaceVector;
using hyle::container::StaticVector;

namespace {

struct LifetimeValue {
  explicit LifetimeValue(int input = 0) noexcept : value(input) { ++alive; }
  LifetimeValue(const LifetimeValue &other) noexcept : value(other.value) {
    ++alive;
  }
  LifetimeValue(LifetimeValue &&other) noexcept : value(other.value) {
    other.value = -1;
    ++alive;
    ++moves;
  }
  LifetimeValue &operator=(const LifetimeValue &) = delete;
  LifetimeValue &operator=(LifetimeValue &&) = delete;
  ~LifetimeValue() noexcept { --alive; }

  friend bool operator==(const LifetimeValue &left,
                         const LifetimeValue &right) noexcept {
    return left.value == right.value;
  }

  static inline int alive = 0;
  static inline int moves = 0;
  int value;
};

struct ThrowingCopy {
  explicit ThrowingCopy(int input = 0) noexcept : value(input) { ++alive; }
  ThrowingCopy(const ThrowingCopy &other) : value(other.value) {
    if (copies_before_throw == 0)
      throw std::runtime_error("copy failed");
    --copies_before_throw;
    ++alive;
  }
  ThrowingCopy(ThrowingCopy &&other) noexcept : value(other.value) { ++alive; }
  ThrowingCopy &operator=(const ThrowingCopy &) = delete;
  ThrowingCopy &operator=(ThrowingCopy &&) = delete;
  ~ThrowingCopy() noexcept { --alive; }

  static inline int alive = 0;
  static inline int copies_before_throw = 0;
  int value;
};

struct alignas(64) OverAlignedValue {
  int value = 0;
};

struct PinnedValue {
  explicit PinnedValue(int input) noexcept : value(input) {}
  PinnedValue(const PinnedValue &) = delete;
  PinnedValue(PinnedValue &&) = delete;
  PinnedValue &operator=(const PinnedValue &) = delete;
  PinnedValue &operator=(PinnedValue &&) = delete;
  ~PinnedValue() noexcept = default;

  int value;
};

struct CopyOnlyValue {
  explicit CopyOnlyValue(int input) noexcept : value(input) {}
  CopyOnlyValue(const CopyOnlyValue &other) noexcept : value(other.value) {}
  CopyOnlyValue(CopyOnlyValue &&) = delete;
  CopyOnlyValue &operator=(const CopyOnlyValue &) = delete;
  CopyOnlyValue &operator=(CopyOnlyValue &&) = delete;
  ~CopyOnlyValue() noexcept = default;

  int value;
};

struct ThrowingMoveValue {
  explicit ThrowingMoveValue(int input) noexcept : value(input) { ++alive; }
  ThrowingMoveValue(const ThrowingMoveValue &) = delete;
  ThrowingMoveValue(ThrowingMoveValue &&other) : value(other.value) {
    if (moves_before_throw == 0)
      throw std::runtime_error("move failed");
    --moves_before_throw;
    other.value = -1;
    ++alive;
  }
  ThrowingMoveValue &operator=(const ThrowingMoveValue &) = delete;
  ThrowingMoveValue &operator=(ThrowingMoveValue &&) = delete;
  ~ThrowingMoveValue() noexcept { --alive; }

  static inline int alive = 0;
  static inline int moves_before_throw = 0;
  int value;
};

template <class Vector>
concept HasIntegerPositionalEmplace =
    requires(Vector &values) { values.emplace(values.begin(), 1); };

template <class Vector>
concept HasIntegerBackEmplace =
    requires(Vector &values) { values.emplace_back(1); };

} // namespace

static_assert(std::same_as<InplaceVector<int, 4>, StaticVector<int, 4>>);
static_assert(StaticVector<int, 4>::capacity() == 4);
static_assert(sizeof(StaticVector<std::uint8_t, 255>) == 256,
              "capacity-sized metadata should use one byte when possible");
static_assert(sizeof(StaticVector<OverAlignedValue, 0>) == 1,
              "zero capacity should not retain dummy aligned storage");
static_assert(HasIntegerBackEmplace<StaticVector<PinnedValue, 2>>);
static_assert(!HasIntegerPositionalEmplace<StaticVector<PinnedValue, 2>>);
static_assert(HasIntegerBackEmplace<StaticVector<ThrowingMoveValue, 2>>);
static_assert(!HasIntegerPositionalEmplace<StaticVector<ThrowingMoveValue, 2>>);

TEST(StaticVector, StorageIsAlwaysInsideObjectAndNeverRelocates) {
  StaticVector<std::string, 4> values;
  const auto object_begin = reinterpret_cast<std::uintptr_t>(&values);
  const auto object_end = object_begin + sizeof(values);
  const auto storage = reinterpret_cast<std::uintptr_t>(values.data());
  ASSERT_GE(storage, object_begin);
  ASSERT_LT(storage, object_end);
  ASSERT_NE(values.data(), nullptr);
  ASSERT_EQ(values.begin(), values.end());

  auto *const original_data = values.data();
  values.emplace_back("alpha");
  values.emplace_back("beta");
  values.insert(values.begin() + 1, "between");
  values.erase(values.begin());
  values.resize(4, "tail");
  EXPECT_EQ(values.data(), original_data);
  EXPECT_EQ(values.capacity(), 4u);

  StaticVector<OverAlignedValue, 1> aligned;
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(aligned.data()) % 64, 0u);
}

TEST(StaticVector, RepeatedSlotLifetimeRestartKeepsTheArrayAddress) {
  LifetimeValue::alive = 0;
  StaticVector<LifetimeValue, 2> values;
  auto *const storage = values.data();
  for (int cycle = 0; cycle < 1000; ++cycle) {
    values.emplace_back(cycle);
    values.emplace_back(cycle + 1);
    EXPECT_EQ(values.data(), storage);
    values.clear();
    EXPECT_EQ(LifetimeValue::alive, 0);
    EXPECT_EQ(values.data(), storage);
  }
}

TEST(StaticVector, CountRangeInitializerCopyAndMoveHaveValueSemantics) {
  StaticVector<int, 5> defaults(3);
  EXPECT_EQ(std::accumulate(defaults.begin(), defaults.end(), 0), 0);

  StaticVector<std::string, 5> filled(3, std::string("x"));
  StaticVector<std::string, 5> copied(filled);
  copied[0] = "changed";
  EXPECT_EQ(filled.front(), "x");
  EXPECT_EQ(copied.front(), "changed");

  StaticVector<std::string, 5> copy_assigned;
  copy_assigned = filled;
  EXPECT_EQ(copy_assigned, filled);
  StaticVector<std::string, 5> move_assigned;
  move_assigned = std::move(copied);
  EXPECT_TRUE(copied.empty());
  EXPECT_EQ(move_assigned.front(), "changed");

  const std::array<int, 4> source = {1, 2, 3, 4};
  StaticVector<int, 5> ranged(source.begin(), source.end());
  EXPECT_TRUE(std::equal(ranged.begin(), ranged.end(), source.begin()));

  StaticVector<int, 5> listed({7, 8, 9});
  StaticVector<int, 5> moved(std::move(listed));
  EXPECT_TRUE(listed.empty());
  EXPECT_EQ(moved, (StaticVector<int, 5>{7, 8, 9}));
}

TEST(StaticVector, TryCapacityFailuresPreserveOriginalValue) {
  StaticVector<std::string, 3> values({"a", "b", "c"});
  const auto original = values;

  EXPECT_FALSE(values.try_push_back("d"));
  EXPECT_FALSE(values.try_emplace(values.begin() + 1, "d"));
  EXPECT_FALSE(values.try_resize(4, std::string("d")));
  EXPECT_FALSE(values.try_assign(4, std::string("d")));

  const std::array<std::string, 2> extra = {"x", "y"};
  EXPECT_FALSE(
      values.try_insert(values.begin() + 1, extra.begin(), extra.end()));
  EXPECT_EQ(values, original);
}

TEST(StaticVector, OrdinaryCapacityOverflowFailsFast) {
  EXPECT_DEATH(([] {
                 StaticVector<int, 1> values;
                 values.push_back(1);
                 values.push_back(2);
               }()),
               "HYLE fatal");
}

TEST(StaticVector, AliasedValuesAndSelfRangesAreSafe) {
  StaticVector<std::string, 8> values({"a", "b", "c"});
  values.push_back(values.front());
  values.emplace(values.begin() + 1, values.back());
  EXPECT_EQ(values, (StaticVector<std::string, 8>{"a", "a", "b", "c", "a"}));

  values.assign(values.begin() + 1, values.end() - 1);
  EXPECT_EQ(values, (StaticVector<std::string, 8>{"a", "b", "c"}));

  values.insert(values.begin() + 1, values.begin(), values.end());
  EXPECT_EQ(values,
            (StaticVector<std::string, 8>{"a", "a", "b", "c", "b", "c"}));
}

TEST(StaticVector, SinglePassInputRangeIsConsumedOnce) {
  std::istringstream construction_stream("1 2 3");
  StaticVector<int, 6> values((std::istream_iterator<int>(construction_stream)),
                              std::istream_iterator<int>());
  EXPECT_EQ(values, (StaticVector<int, 6>{1, 2, 3}));

  std::istringstream insertion_stream("8 9");
  ASSERT_TRUE(values.try_insert(values.begin() + 1,
                                std::istream_iterator<int>(insertion_stream),
                                std::istream_iterator<int>()));
  EXPECT_EQ(values, (StaticVector<int, 6>{1, 8, 9, 2, 3}));

  const auto before = values;
  std::istringstream overflow_stream("4 5");
  EXPECT_TRUE(values.try_assign(std::istream_iterator<int>(overflow_stream),
                                std::istream_iterator<int>()));
  // The range itself fits capacity, so replacement succeeds.
  EXPECT_EQ(values, (StaticVector<int, 6>{4, 5}));
  EXPECT_NE(values, before);
}

TEST(StaticVector, OverflowingInputRangeLeavesDestinationUntouched) {
  StaticVector<int, 3> values({7, 8});
  std::istringstream stream("1 2 3 4");
  EXPECT_FALSE(values.try_assign(std::istream_iterator<int>(stream),
                                 std::istream_iterator<int>()));
  EXPECT_EQ(values, (StaticVector<int, 3>{7, 8}));
}

TEST(StaticVector, SupportsMoveOnlyValuesAndExactLifetime) {
  StaticVector<std::unique_ptr<int>, 4> values;
  values.emplace_back(std::make_unique<int>(1));
  values.push_back(std::make_unique<int>(3));
  values.emplace(values.begin() + 1, std::make_unique<int>(2));
  ASSERT_EQ(values.size(), 3u);
  EXPECT_EQ(*values[0], 1);
  EXPECT_EQ(*values[1], 2);
  EXPECT_EQ(*values[2], 3);
  values.erase(values.begin() + 1);
  EXPECT_EQ(*values[1], 3);

  LifetimeValue::alive = 0;
  LifetimeValue::moves = 0;
  {
    StaticVector<LifetimeValue, 6> tracked;
    tracked.emplace_back(1);
    tracked.emplace_back(3);
    tracked.emplace(tracked.begin() + 1, 2);
    EXPECT_EQ(LifetimeValue::alive, 3);
    tracked.erase(tracked.begin(), tracked.begin() + 2);
    EXPECT_EQ(LifetimeValue::alive, 1);
    tracked.clear();
    EXPECT_EQ(LifetimeValue::alive, 0);
  }
  EXPECT_EQ(LifetimeValue::alive, 0);
}

TEST(StaticVector, ImmovableValuesCanUseInPlaceBackOperations) {
  StaticVector<PinnedValue, 2> values;
  ASSERT_TRUE(values.try_emplace_back(1));
  ASSERT_TRUE(values.try_emplace_back(2));
  EXPECT_FALSE(values.try_emplace_back(3));
  EXPECT_EQ(values.front().value, 1);
  EXPECT_EQ(values.back().value, 2);
  values.pop_back();
  EXPECT_EQ(values.back().value, 1);
  values.clear();
  EXPECT_TRUE(values.empty());
}

TEST(StaticVector, CopyOnlyValuesRetainCopyAndAssignOperations) {
  CopyOnlyValue first(1);
  CopyOnlyValue second(2);
  const std::array<CopyOnlyValue, 2> input = {first, second};
  StaticVector<CopyOnlyValue, 3> source(input.begin(), input.end());
  StaticVector<CopyOnlyValue, 3> copied(source);
  ASSERT_EQ(copied.size(), 2u);
  EXPECT_EQ(copied[1].value, 2);

  StaticVector<CopyOnlyValue, 3> assigned;
  assigned = source;
  ASSERT_EQ(assigned.size(), 2u);
  EXPECT_EQ(assigned[0].value, 1);
  assigned.assign(3, first);
  EXPECT_EQ(assigned.size(), 3u);
  EXPECT_EQ(assigned.back().value, 1);
}

TEST(StaticVector, ThrowingMoveConstructionCleansDestinationPrefix) {
  ThrowingMoveValue::alive = 0;
  {
    StaticVector<ThrowingMoveValue, 4> source;
    source.emplace_back(1);
    source.emplace_back(2);
    source.emplace_back(3);
    EXPECT_EQ(ThrowingMoveValue::alive, 3);

    ThrowingMoveValue::moves_before_throw = 1;
    EXPECT_THROW(([](auto &input) {
                   StaticVector<ThrowingMoveValue, 4> failed(std::move(input));
                   (void)failed;
                 }(source)),
                 std::runtime_error);
    EXPECT_EQ(ThrowingMoveValue::alive, 3);
    EXPECT_EQ(source.size(), 3u);

    ThrowingMoveValue::moves_before_throw = 4;
    StaticVector<ThrowingMoveValue, 4> moved(std::move(source));
    EXPECT_TRUE(source.empty());
    EXPECT_EQ(moved.size(), 3u);
    EXPECT_EQ(ThrowingMoveValue::alive, 3);
  }
  EXPECT_EQ(ThrowingMoveValue::alive, 0);
}

TEST(StaticVector, ThrowingMoveAssignmentKeepsBothVectorsDestructible) {
  ThrowingMoveValue::alive = 0;
  {
    StaticVector<ThrowingMoveValue, 4> source;
    source.emplace_back(1);
    source.emplace_back(2);
    source.emplace_back(3);
    StaticVector<ThrowingMoveValue, 4> destination;
    destination.emplace_back(9);

    ThrowingMoveValue::moves_before_throw = 1;
    EXPECT_THROW(destination = std::move(source), std::runtime_error);
    EXPECT_EQ(destination.size(), 1u);
    EXPECT_EQ(source.size(), 3u);
    EXPECT_EQ(ThrowingMoveValue::alive, 4);
  }
  EXPECT_EQ(ThrowingMoveValue::alive, 0);
}

TEST(StaticVector, ThrowingGrowthRollsBackCompletedSuffix) {
  ThrowingCopy::alive = 0;
  {
    StaticVector<ThrowingCopy, 4> values;
    values.emplace_back(7);
    ThrowingCopy fill(9);
    ThrowingCopy::copies_before_throw = 1;
    EXPECT_THROW(values.try_resize(4, fill), std::runtime_error);
    ASSERT_EQ(values.size(), 1u);
    EXPECT_EQ(values.front().value, 7);
    EXPECT_EQ(ThrowingCopy::alive, 2);
  }
  EXPECT_EQ(ThrowingCopy::alive, 0);
}

TEST(StaticVector, PositionMutationMovesOnlySuffixAndKeepsStorageAddress) {
  StaticVector<LifetimeValue, 8> values;
  values.emplace_back(0);
  values.emplace_back(1);
  values.emplace_back(2);
  values.emplace_back(3);
  auto *const data_before = values.data();
  LifetimeValue::moves = 0;

  values.emplace(values.begin() + 2, 9);
  EXPECT_EQ(values.data(), data_before);
  EXPECT_EQ(LifetimeValue::moves, 3); // saved value + two-element suffix
  EXPECT_EQ(values[0].value, 0);
  EXPECT_EQ(values[1].value, 1);
  EXPECT_EQ(values[2].value, 9);

  LifetimeValue::moves = 0;
  values.erase(values.begin() + 3);
  EXPECT_EQ(LifetimeValue::moves, 1); // only the following suffix
  EXPECT_EQ(values.data(), data_before);
}

TEST(StaticVector, ResizeAssignEraseSwapAndReverseIteration) {
  StaticVector<int, 6> left({1, 2, 3});
  left.resize(5, 4);
  EXPECT_EQ(left, (StaticVector<int, 6>{1, 2, 3, 4, 4}));
  left.pop_back();
  left.erase(left.begin() + 1, left.begin() + 3);
  EXPECT_EQ(left, (StaticVector<int, 6>{1, 4}));
  left.insert(left.begin() + 1, 2, 6);
  EXPECT_EQ(left, (StaticVector<int, 6>{1, 6, 6, 4}));
  left.assign({1, 4});

  StaticVector<int, 6> right({9, 8, 7});
  swap(left, right);
  EXPECT_EQ(left, (StaticVector<int, 6>{9, 8, 7}));
  EXPECT_EQ(right, (StaticVector<int, 6>{1, 4}));
  EXPECT_LT(right, left);

  std::array<int, 3> reversed{};
  std::copy(left.rbegin(), left.rend(), reversed.begin());
  EXPECT_EQ(reversed, (std::array<int, 3>{7, 8, 9}));
}

TEST(StaticVector, RandomPositionalOperationsMatchStdVector) {
  StaticVector<int, 16> actual;
  std::vector<int> oracle;
  std::uint32_t state = 0x12345678u;
  auto next = [&] {
    state = state * 1664525u + 1013904223u;
    return state;
  };

  for (int step = 0; step < 10000; ++step) {
    const std::uint32_t value = next();
    switch (value % 5) {
    case 0:
      if (oracle.size() < actual.capacity()) {
        ASSERT_TRUE(actual.try_push_back(static_cast<int>(value)));
        oracle.push_back(static_cast<int>(value));
      } else {
        EXPECT_FALSE(actual.try_push_back(static_cast<int>(value)));
      }
      break;
    case 1:
      if (!oracle.empty()) {
        actual.pop_back();
        oracle.pop_back();
      }
      break;
    case 2: {
      const std::size_t position =
          oracle.empty() ? 0 : next() % (oracle.size() + 1);
      if (oracle.size() < actual.capacity()) {
        ASSERT_TRUE(actual.try_insert(actual.begin() + position,
                                      static_cast<int>(value)));
        oracle.insert(oracle.begin() + static_cast<std::ptrdiff_t>(position),
                      static_cast<int>(value));
      }
      break;
    }
    case 3:
      if (!oracle.empty()) {
        const std::size_t position = next() % oracle.size();
        actual.erase(actual.begin() + position);
        oracle.erase(oracle.begin() + static_cast<std::ptrdiff_t>(position));
      }
      break;
    case 4: {
      const std::size_t requested = next() % 18;
      const bool resized = actual.try_resize(requested, 7);
      EXPECT_EQ(resized, requested <= actual.capacity());
      if (requested <= actual.capacity())
        oracle.resize(requested, 7);
      break;
    }
    }
    ASSERT_TRUE(
        std::equal(actual.begin(), actual.end(), oracle.begin(), oracle.end()));
  }
}

TEST(StaticVector, ZeroCapacityIsAUsableEmptyContainer) {
  StaticVector<int, 0> values;
  EXPECT_TRUE(values.empty());
  EXPECT_TRUE(values.full());
  EXPECT_EQ(values.size(), 0u);
  EXPECT_EQ(values.capacity(), 0u);
  EXPECT_EQ(values.data(), nullptr);
  EXPECT_EQ(values.begin(), values.end());
  EXPECT_FALSE(values.try_push_back(1));
  EXPECT_TRUE(values.try_resize(0));
  EXPECT_TRUE(values.try_assign(0, 3));

  const std::array<int, 0> empty{};
  EXPECT_TRUE(values.try_assign(empty.begin(), empty.end()));
  EXPECT_TRUE(values.try_insert(values.begin(), empty.begin(), empty.end()));
}
