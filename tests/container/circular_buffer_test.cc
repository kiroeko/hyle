#include "hyle/container/circular_buffer.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

using hyle::container::CircularBuffer;

void* fail_alloc(void*, std::size_t, std::size_t) noexcept { return nullptr; }
void fail_free(void*, void*, std::size_t, std::size_t) noexcept {}
void* fail_realloc(void*, void*, std::size_t, std::size_t,
                   std::size_t) noexcept {
  return nullptr;
}

struct Lifetime {
  static inline int live = 0;
  int value = 0;

  explicit Lifetime(int input = 0) noexcept : value(input) { ++live; }
  Lifetime(const Lifetime& other) noexcept : value(other.value) { ++live; }
  Lifetime(Lifetime&& other) noexcept : value(other.value) {
    other.value = -1;
    ++live;
  }
  ~Lifetime() noexcept { --live; }
};

static_assert(std::random_access_iterator<CircularBuffer<int>::iterator>);
static_assert(
    std::random_access_iterator<CircularBuffer<int>::const_iterator>);

TEST(CircularBuffer, CapacityIsExactAndFullPushOverwritesOldest) {
  CircularBuffer<int> values(3);
  EXPECT_EQ(values.capacity(), 3u);
  values.push_back(1);
  values.push_back(2);
  values.push_back(3);
  EXPECT_TRUE(values.full());

  values.push_back(4);
  EXPECT_EQ((std::vector<int>(values.begin(), values.end())),
            (std::vector<int>{2, 3, 4}));
  values.push_front(1);
  EXPECT_EQ((std::vector<int>(values.begin(), values.end())),
            (std::vector<int>{1, 2, 3}));
}

TEST(CircularBuffer, NonPowerOfTwoCapacityWrapsWithoutChangingLogicalLimit) {
  CircularBuffer<std::uint32_t> values(60);
  for (std::uint32_t i = 0; i < 200; ++i) values.push_back(i);

  ASSERT_EQ(values.size(), 60u);
  EXPECT_EQ(values.capacity(), 60u);
  EXPECT_EQ(values.front(), 140u);
  EXPECT_EQ(values.back(), 199u);
  for (std::uint32_t i = 0; i < 60; ++i) {
    EXPECT_EQ(values[i], i + 140);
  }
  const auto segments = values.segments();
  EXPECT_EQ(segments.first.size() + segments.second.size(), 60u);
}

TEST(CircularBuffer, ShrinkKeepsNewestAndZeroCapacityRejectsPush) {
  CircularBuffer<int> values(5);
  for (int i = 1; i <= 5; ++i) values.push_back(i);
  ASSERT_TRUE(values.try_set_capacity(3));
  EXPECT_EQ((std::vector<int>(values.begin(), values.end())),
            (std::vector<int>{3, 4, 5}));

  ASSERT_TRUE(values.try_set_capacity(0));
  EXPECT_TRUE(values.empty());
  EXPECT_EQ(values.capacity(), 0u);
  EXPECT_FALSE(values.try_emplace_back(6));
}

TEST(CircularBuffer, AliasedOverwriteAndObjectLifetimeAreSafe) {
  CircularBuffer<std::string> values(2);
  values.push_back("alpha");
  values.push_back("beta");
  values.push_back(values.front());
  EXPECT_EQ(values.front(), "beta");
  EXPECT_EQ(values.back(), "alpha");

  values.push_front(values.back());
  EXPECT_EQ(values.front(), "alpha");
  EXPECT_EQ(values.back(), "beta");
}

TEST(CircularBuffer, AllocationFailurePreservesCapacityAndElements) {
  hyle::mem::AllocatorRef failing(&fail_alloc, &fail_free, &fail_realloc, nullptr);
  CircularBuffer<int> values(failing);
  EXPECT_FALSE(values.try_set_capacity(8));
  EXPECT_EQ(values.capacity(), 0u);
  EXPECT_TRUE(values.empty());
  EXPECT_FALSE(values.try_push_back(1));
}

TEST(CircularBuffer, OverwriteShrinkAndClearBalanceObjectLifetime) {
  EXPECT_EQ(Lifetime::live, 0);
  {
    CircularBuffer<Lifetime> values(3);
    values.emplace_back(1);
    values.emplace_back(2);
    values.emplace_back(3);
    values.emplace_back(4);
    EXPECT_EQ(Lifetime::live, 3);
    EXPECT_EQ(values.front().value, 2);

    ASSERT_TRUE(values.try_set_capacity(1));
    EXPECT_EQ(Lifetime::live, 1);
    EXPECT_EQ(values.front().value, 4);
    values.clear();
    EXPECT_EQ(Lifetime::live, 0);
  }
  EXPECT_EQ(Lifetime::live, 0);
}

}  // namespace
