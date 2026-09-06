#include <gtest/gtest.h>

#include <array>
#include <iterator>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

#include "hyle/container/small_vector.h"
#include "hyle/mem/allocator.h"

using hyle::container::SmallVector;
using hyle::mem::AllocatorRef;
using hyle::mem::system_allocator;

namespace {

struct MoveConstructOnly {
  explicit MoveConstructOnly(int input) noexcept : value(input) {}
  MoveConstructOnly(const MoveConstructOnly&) = default;
  MoveConstructOnly(MoveConstructOnly&&) noexcept = default;
  ~MoveConstructOnly() noexcept {}
  MoveConstructOnly& operator=(const MoveConstructOnly&) = delete;
  MoveConstructOnly& operator=(MoveConstructOnly&&) = delete;

  int value;
};

struct ThrowingCopy {
  explicit ThrowingCopy(int input) noexcept : value(input) { ++alive; }
  ThrowingCopy(const ThrowingCopy& other) : value(other.value) {
    if (copies_before_throw == 0) throw std::runtime_error("copy failed");
    --copies_before_throw;
    ++alive;
  }
  ThrowingCopy(ThrowingCopy&& other) noexcept : value(other.value) { ++alive; }
  ~ThrowingCopy() noexcept { --alive; }
  ThrowingCopy& operator=(const ThrowingCopy&) = delete;
  ThrowingCopy& operator=(ThrowingCopy&&) = delete;

  static inline int alive = 0;
  static inline int copies_before_throw = 0;
  int value;
};

struct ThrowingMoveCopyable {
  explicit ThrowingMoveCopyable(int input = 0) noexcept : value(input) {}
  ThrowingMoveCopyable(const ThrowingMoveCopyable&) noexcept = default;
  ThrowingMoveCopyable& operator=(const ThrowingMoveCopyable&) noexcept =
      default;
  ThrowingMoveCopyable(ThrowingMoveCopyable&& other) noexcept(false) {
    if (throw_on_move) throw std::runtime_error("move failed");
    value = other.value;
    other.value = -1;
  }
  ThrowingMoveCopyable& operator=(ThrowingMoveCopyable&& other) noexcept(false) {
    if (throw_on_move) throw std::runtime_error("move assignment failed");
    value = other.value;
    other.value = -1;
    return *this;
  }

  static inline bool throw_on_move = false;
  int value = 0;
};

struct ThrowingMoveOnly {
  explicit ThrowingMoveOnly(int input = 0) noexcept : value(input) {}
  ThrowingMoveOnly(const ThrowingMoveOnly&) = delete;
  ThrowingMoveOnly& operator=(const ThrowingMoveOnly&) = delete;
  ThrowingMoveOnly(ThrowingMoveOnly&& other) noexcept(false) {
    if (throw_on_move) throw std::runtime_error("move failed");
    value = other.value;
    other.value = -1;
  }
  ThrowingMoveOnly& operator=(ThrowingMoveOnly&&) = delete;

  static inline bool throw_on_move = false;
  int value = 0;
};

struct ToggleAllocatorState {
  AllocatorRef backing = system_allocator();
  bool fail = false;
};

void* toggle_allocate(void* raw_state, std::size_t size,
                      std::size_t alignment) noexcept {
  auto* state = static_cast<ToggleAllocatorState*>(raw_state);
  return state->fail ? nullptr : state->backing.allocate(size, alignment);
}

void toggle_deallocate(void* raw_state, void* pointer, std::size_t size,
                       std::size_t alignment) noexcept {
  auto* state = static_cast<ToggleAllocatorState*>(raw_state);
  state->backing.deallocate(pointer, size, alignment);
}

void* toggle_reallocate(void* raw_state, void* pointer, std::size_t old_size,
                        std::size_t new_size,
                        std::size_t alignment) noexcept {
  auto* state = static_cast<ToggleAllocatorState*>(raw_state);
  return state->fail ? nullptr : state->backing.reallocate(
                                     pointer, old_size, new_size, alignment);
}

}  // namespace

// ---- SmallVector ------------------------------------------------------------

using LargeZeroInlineValue = std::array<std::byte, 512>;
static_assert(SmallVector<LargeZeroInlineValue, 0>::inline_capacity() == 0);
static_assert(sizeof(SmallVector<LargeZeroInlineValue, 0>) <
              sizeof(LargeZeroInlineValue));

TEST(SmallVector, ZeroInlineStartsEmptyAndAllocatesOnDemand) {
  SmallVector<int, 0> empty_trivial;
  SmallVector<int, 0> moved_empty_trivial(std::move(empty_trivial));
  SmallVector<int, 0> swapped_empty_trivial;
  swap(moved_empty_trivial, swapped_empty_trivial);
  EXPECT_TRUE(empty_trivial.empty());
  EXPECT_TRUE(moved_empty_trivial.empty());
  EXPECT_TRUE(swapped_empty_trivial.empty());

  SmallVector<std::string, 0> values;
  EXPECT_TRUE(values.empty());
  EXPECT_TRUE(values.inline_storage());
  EXPECT_EQ(values.capacity(), 0u);
  EXPECT_EQ(values.data(), nullptr);
  EXPECT_EQ(values.begin(), values.end());
  EXPECT_EQ(values, (SmallVector<std::string, 0>{}));

  values.emplace_back("alpha");
  values.emplace_back("beta");
  EXPECT_FALSE(values.inline_storage());
  EXPECT_GE(values.capacity(), 2u);
  EXPECT_EQ(values.front(), "alpha");
  EXPECT_EQ(values.back(), "beta");

  SmallVector<std::string, 0> moved(std::move(values));
  EXPECT_TRUE(values.inline_storage());
  EXPECT_EQ(values.data(), nullptr);
  EXPECT_EQ(moved.size(), 2u);
  moved.clear();
  moved.shrink_to_fit();
  EXPECT_TRUE(moved.inline_storage());
  EXPECT_EQ(moved.capacity(), 0u);
  EXPECT_EQ(moved.data(), nullptr);
}

TEST(SmallVector, ZeroInlineAllocationFailurePreservesEmptyState) {
  ToggleAllocatorState state;
  state.fail = true;
  AllocatorRef allocator(toggle_allocate, toggle_deallocate, toggle_reallocate,
                      &state);
  SmallVector<int, 0, AllocatorRef> values(allocator);
  EXPECT_FALSE(values.try_push_back(7));
  EXPECT_TRUE(values.empty());
  EXPECT_TRUE(values.inline_storage());
  EXPECT_EQ(values.capacity(), 0u);
}

TEST(SmallVector, ThrowingMoveTypesAreNotRejectedClassWide) {
  static_assert(!std::is_nothrow_move_constructible_v<ThrowingMoveCopyable>);
  SmallVector<ThrowingMoveCopyable, 2> values;
  values.emplace_back(1);
  values.emplace_back(2);
  ThrowingMoveCopyable::throw_on_move = true;
  EXPECT_NO_THROW(values.reserve(8));  // copy fallback, strong guarantee
  EXPECT_EQ(values[0].value, 1);
  EXPECT_EQ(values[1].value, 2);
  EXPECT_NO_THROW(values.emplace(values.begin() + 1, 3));
  EXPECT_EQ(values[1].value, 3);
  EXPECT_NO_THROW(values.erase(values.begin() + 1));
  EXPECT_EQ(values.size(), 2u);
  ThrowingMoveCopyable::throw_on_move = false;

  SmallVector<ThrowingMoveOnly, 2> move_only;
  move_only.emplace_back(7);
  move_only.emplace_back(9);
  ThrowingMoveOnly::throw_on_move = true;
  EXPECT_THROW(move_only.reserve(8), std::runtime_error);
  EXPECT_EQ(move_only.size(), 2u);  // valid basic-guarantee state
  ThrowingMoveOnly::throw_on_move = false;
}

TEST(SmallVector, StaysInlineUnderN) {
  SmallVector<int, 4> v;  // default = compile-time SystemAllocator (zero-overhead path)
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);
  v.push_back(4);
  EXPECT_EQ(v.size(), 4u);
  EXPECT_TRUE(v.inline_storage());  // no heap touched at capacity N
  for (int i = 0; i < 4; ++i) EXPECT_EQ(v[i], i + 1);
}

TEST(SmallVector, SpillsToHeapPastN) {
  // Runtime-injection path (opt-in mem::AllocatorRef): exercises the vtable allocator + spill.
  SmallVector<int, 4, AllocatorRef> v(system_allocator());
  for (int i = 0; i < 100; ++i) v.push_back(i);
  EXPECT_EQ(v.size(), 100u);
  EXPECT_FALSE(v.inline_storage());  // spilled
  for (int i = 0; i < 100; ++i) EXPECT_EQ(v[i], i);  // data preserved across spill
}

TEST(SmallVector, PopClearIterate) {
  SmallVector<int, 2> v({10, 20, 30});  // default alloc, forces spill
  EXPECT_EQ(v.size(), 3u);
  EXPECT_EQ(std::accumulate(v.begin(), v.end(), 0), 60);
  v.pop_back();
  EXPECT_EQ(v.back(), 20);
  v.clear();
  EXPECT_TRUE(v.empty());
}

TEST(SmallVector, MoveStealsHeap) {
  SmallVector<int, 2> a;
  for (int i = 0; i < 50; ++i) a.push_back(i);  // on heap
  int* p = a.data();
  SmallVector<int, 2> b(std::move(a));
  EXPECT_EQ(b.data(), p);  // heap buffer stolen, not copied
  EXPECT_EQ(b.size(), 50u);
  EXPECT_EQ(b[49], 49);
}

TEST(SmallVector, MoveInlineCopies) {
  SmallVector<int, 4> a;
  a.push_back(7);
  a.push_back(8);
  SmallVector<int, 4> b(std::move(a));  // inline -> element copy
  EXPECT_TRUE(b.inline_storage());
  EXPECT_EQ(b.size(), 2u);
  EXPECT_EQ(b[0], 7);
  EXPECT_EQ(b[1], 8);
}

TEST(SmallVector, MoveAssignmentHandlesInlineAndHeap) {
  SmallVector<int, 2> heap;
  heap.push_back(1);
  heap.push_back(2);
  heap.push_back(3);
  SmallVector<int, 2> inline_values;
  inline_values.push_back(9);

  inline_values = std::move(heap);
  ASSERT_EQ(inline_values.size(), 3u);
  EXPECT_EQ(inline_values[2], 3);
  EXPECT_TRUE(heap.empty());

  SmallVector<int, 2> other;
  other.push_back(7);
  inline_values = std::move(other);
  ASSERT_EQ(inline_values.size(), 1u);
  EXPECT_EQ(inline_values[0], 7);
}

TEST(SmallVector, PushBackMayAliasElementAcrossSpill) {
  SmallVector<int, 2> v;
  v.push_back(4);
  v.push_back(5);
  v.push_back(v[0]);
  ASSERT_EQ(v.size(), 3u);
  EXPECT_EQ(v[2], 4);
}

TEST(SmallVector, OwnsNonTrivialValuesAcrossInlineAndHeapStorage) {
  SmallVector<std::string, 2> values;
  values.emplace_back("alpha");
  values.push_back(std::string("beta"));
  values.push_back(values[0]);  // aliased copy across spill

  ASSERT_FALSE(values.inline_storage());
  ASSERT_EQ(values.size(), 3u);
  EXPECT_EQ(values[0], "alpha");
  EXPECT_EQ(values[1], "beta");
  EXPECT_EQ(values[2], "alpha");

  values.erase(values.begin() + 1);
  ASSERT_EQ(values.size(), 2u);
  EXPECT_EQ(values[1], "alpha");
  values.resize(4, std::string("tail"));
  EXPECT_EQ(values[3], "tail");
  values.clear();
  EXPECT_TRUE(values.empty());
}

TEST(SmallVector, EmplaceArgumentsMayAliasAcrossSpill) {
  SmallVector<std::string, 1> values;
  values.emplace_back("kept");
  values.emplace_back(values.front());
  ASSERT_EQ(values.size(), 2u);
  EXPECT_EQ(values[0], "kept");
  EXPECT_EQ(values[1], "kept");
}

TEST(SmallVector, SupportsMoveOnlyValues) {
  SmallVector<std::unique_ptr<int>, 1> values;
  values.emplace_back(std::make_unique<int>(7));
  values.push_back(std::make_unique<int>(9));
  ASSERT_EQ(values.size(), 2u);
  EXPECT_EQ(*values[0], 7);
  EXPECT_EQ(*values[1], 9);
}

TEST(SmallVector, CopyHasValueSemantics) {
  SmallVector<std::string, 2> original({"a", "b", "c"});
  SmallVector<std::string, 2> copy(original);
  copy[0] = "changed";
  EXPECT_EQ(original[0], "a");
  EXPECT_EQ(copy[0], "changed");

  SmallVector<std::string, 2> assigned;
  assigned = original;
  EXPECT_EQ(assigned.size(), original.size());
  EXPECT_EQ(assigned[2], "c");
}

TEST(SmallVector, PositionalInsertAndEmplaceMatchVectorShape) {
  SmallVector<std::string, 3> values({"alpha", "gamma"});
  auto inserted = values.insert(values.begin() + 1, std::string("beta"));
  ASSERT_EQ(inserted, values.begin() + 1);
  EXPECT_EQ(*inserted, "beta");

  auto first = values.emplace(values.begin(), 3, 'x');
  ASSERT_EQ(first, values.begin());
  EXPECT_EQ(*first, "xxx");
  ASSERT_EQ(values.size(), 4u);
  EXPECT_EQ(values[0], "xxx");
  EXPECT_EQ(values[1], "alpha");
  EXPECT_EQ(values[2], "beta");
  EXPECT_EQ(values[3], "gamma");
}

TEST(SmallVector, PositionalInsertMayAliasAcrossSpill) {
  SmallVector<std::string, 2> values({"alpha", "omega"});
  values.insert(values.begin() + 1, values.front());
  ASSERT_FALSE(values.inline_storage());
  ASSERT_EQ(values.size(), 3u);
  EXPECT_EQ(values[0], "alpha");
  EXPECT_EQ(values[1], "alpha");
  EXPECT_EQ(values[2], "omega");
}

TEST(SmallVector, EraseRangeMovesOnlyTheTail) {
  SmallVector<std::string, 3> values({"zero", "one", "two", "three", "four"});
  auto next = values.erase(values.begin() + 1, values.begin() + 4);
  ASSERT_EQ(next, values.begin() + 1);
  ASSERT_EQ(values.size(), 2u);
  EXPECT_EQ(values[0], "zero");
  EXPECT_EQ(values[1], "four");

  next = values.erase(values.begin(), values.begin());
  EXPECT_EQ(next, values.begin());
  EXPECT_EQ(values.size(), 2u);
}

TEST(SmallVector, PositionalMutationNeedsNoMoveAssignment) {
  SmallVector<MoveConstructOnly, 2> values;
  values.emplace_back(1);
  values.emplace_back(3);
  values.emplace(values.begin() + 1, 2);
  ASSERT_EQ(values.size(), 3u);
  EXPECT_EQ(values[0].value, 1);
  EXPECT_EQ(values[1].value, 2);
  EXPECT_EQ(values[2].value, 3);

  values.erase(values.begin(), values.begin() + 2);
  ASSERT_EQ(values.size(), 1u);
  EXPECT_EQ(values[0].value, 3);
}

TEST(SmallVector, CountAndRangeConstructionMatchSequenceShape) {
  SmallVector<int, 2> defaults(3);
  ASSERT_EQ(defaults.size(), 3u);
  EXPECT_EQ(std::accumulate(defaults.begin(), defaults.end(), 0), 0);

  SmallVector<std::string, 2> filled(3, std::string("x"));
  ASSERT_EQ(filled.size(), 3u);
  EXPECT_EQ(filled[2], "x");

  const std::array<int, 4> source = {1, 2, 3, 4};
  SmallVector<int, 2> forward(source.begin(), source.end());
  EXPECT_TRUE(std::equal(forward.begin(), forward.end(), source.begin()));

  std::istringstream stream("5 6 7");
  SmallVector<int, 2> input((std::istream_iterator<int>(stream)),
                            std::istream_iterator<int>());
  ASSERT_EQ(input.size(), 3u);
  EXPECT_EQ(input[0], 5);
  EXPECT_EQ(input[2], 7);
}

TEST(SmallVector, RangeAndCountInsertAreSelfAliasSafe) {
  SmallVector<std::string, 3> values({"a", "b", "c"});
  auto inserted = values.insert(values.begin() + 1, values.begin(),
                                values.end());
  ASSERT_EQ(inserted, values.begin() + 1);
  EXPECT_EQ(values,
            (SmallVector<std::string, 3>{"a", "a", "b", "c", "b", "c"}));

  inserted = values.insert(values.begin() + 2, 2, values.back());
  ASSERT_EQ(inserted, values.begin() + 2);
  EXPECT_EQ(values, (SmallVector<std::string, 3>{
                        "a", "a", "c", "c", "b", "c", "b", "c"}));

  std::istringstream stream("8 9");
  SmallVector<int, 2> numbers({1, 2});
  numbers.insert(numbers.begin() + 1, std::istream_iterator<int>(stream),
                 std::istream_iterator<int>());
  EXPECT_EQ(numbers, (SmallVector<int, 2>{1, 8, 9, 2}));
}

TEST(SmallVector, AssignBuildsReplacementBeforeMutation) {
  SmallVector<std::string, 2> values({"zero", "one", "two", "three"});
  values.assign(values.begin() + 1, values.end() - 1);
  EXPECT_EQ(values, (SmallVector<std::string, 2>{"one", "two"}));

  values.assign(3, values.front());
  EXPECT_EQ(values,
            (SmallVector<std::string, 2>{"one", "one", "one"}));
  values.assign({"fresh", "state"});
  EXPECT_EQ(values, (SmallVector<std::string, 2>{"fresh", "state"}));
}

TEST(SmallVector, ShrinkSwapAndComparisonMatchValueContainerShape) {
  SmallVector<int, 2> shrunk({1, 2, 3, 4, 5});
  shrunk.reserve(16);
  shrunk.erase(shrunk.begin() + 2, shrunk.end());
  ASSERT_FALSE(shrunk.inline_storage());
  shrunk.shrink_to_fit();
  EXPECT_TRUE(shrunk.inline_storage());
  EXPECT_EQ(shrunk.capacity(), 2u);

  SmallVector<int, 2> exact(5, 7);
  exact.reserve(16);
  ASSERT_TRUE(exact.try_shrink_to_fit());
  EXPECT_FALSE(exact.inline_storage());
  EXPECT_EQ(exact.capacity(), exact.size());

  SmallVector<int, 2> left({1});
  SmallVector<int, 2> right({2, 3, 4});
  swap(left, right);
  EXPECT_EQ(left, (SmallVector<int, 2>{2, 3, 4}));
  EXPECT_EQ(right, (SmallVector<int, 2>{1}));
  EXPECT_LT(right, left);
}

TEST(SmallVector, FailedShrinkPreservesStorageAndValues) {
  ToggleAllocatorState state;
  AllocatorRef allocator(toggle_allocate, toggle_deallocate, toggle_reallocate,
                      &state);
  SmallVector<int, 2, AllocatorRef> values(allocator, 5, 7);
  values.reserve(16);
  int* const data_before = values.data();
  const std::size_t capacity_before = values.capacity();

  state.fail = true;
  EXPECT_FALSE(values.try_shrink_to_fit());
  EXPECT_EQ(values.data(), data_before);
  EXPECT_EQ(values.capacity(), capacity_before);
  ASSERT_EQ(values.size(), 5u);
  for (int value : values) EXPECT_EQ(value, 7);
}

TEST(SmallVector, ThrowingRangeConstructionDestroysCompletedPrefix) {
  ThrowingCopy::alive = 0;
  {
    const std::array<ThrowingCopy, 3> source = {
        ThrowingCopy(1), ThrowingCopy(2), ThrowingCopy(3)};
    ThrowingCopy::copies_before_throw = 1;
    using ThrowVector = SmallVector<ThrowingCopy, 1>;
    EXPECT_THROW(
        {
          ThrowVector copy(source.begin(), source.end());
          (void)copy;
        },
        std::runtime_error);
    EXPECT_EQ(ThrowingCopy::alive, 3);
  }
  EXPECT_EQ(ThrowingCopy::alive, 0);
}

TEST(SmallVector, ThrowingRangeMutationPreservesOriginal) {
  ThrowingCopy::alive = 0;
  {
    const std::array<ThrowingCopy, 3> source = {
        ThrowingCopy(1), ThrowingCopy(2), ThrowingCopy(3)};
    SmallVector<ThrowingCopy, 1> values;
    values.emplace_back(9);

    ThrowingCopy::copies_before_throw = 1;
    EXPECT_THROW(values.assign(source.begin(), source.end()),
                 std::runtime_error);
    ASSERT_EQ(values.size(), 1u);
    EXPECT_EQ(values.front().value, 9);
    EXPECT_EQ(ThrowingCopy::alive, 4);

    ThrowingCopy::copies_before_throw = 1;
    EXPECT_THROW(values.insert(values.begin(), source.begin(), source.end()),
                 std::runtime_error);
    ASSERT_EQ(values.size(), 1u);
    EXPECT_EQ(values.front().value, 9);
    EXPECT_EQ(ThrowingCopy::alive, 4);
  }
  EXPECT_EQ(ThrowingCopy::alive, 0);
}
