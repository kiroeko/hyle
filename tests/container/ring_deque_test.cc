#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "hyle/container/ring_deque.h"
#include "hyle/mem/allocator.h"

namespace {
using hyle::container::RingDeque;

void *fail_alloc(void *, std::size_t, std::size_t) noexcept { return nullptr; }
void fail_free(void *, void *, std::size_t, std::size_t) noexcept {}
void *fail_realloc(void *, void *, std::size_t, std::size_t,
                   std::size_t) noexcept {
  return nullptr;
}

struct Lifetime {
  static inline int live = 0;
  int value = 0;
  explicit Lifetime(int input = 0) noexcept : value(input) { ++live; }
  Lifetime(const Lifetime &other) noexcept : value(other.value) { ++live; }
  Lifetime(Lifetime &&other) noexcept : value(other.value) {
    other.value = -1;
    ++live;
  }
  ~Lifetime() noexcept { --live; }
};
} // namespace

static_assert(std::random_access_iterator<RingDeque<int>::iterator>);
static_assert(std::random_access_iterator<RingDeque<int>::const_iterator>);

TEST(RingDeque, WrapsAndExposesTwoReadableSegments) {
  RingDeque<std::uint32_t> ring(4);
  for (std::uint32_t value = 0; value < 4; ++value)
    EXPECT_TRUE(ring.try_push_back_bounded(value));
  EXPECT_FALSE(ring.try_push_back_bounded(4));
  ring.pop_front();
  ring.pop_front();
  EXPECT_TRUE(ring.try_push_back_bounded(4));
  EXPECT_TRUE(ring.try_push_back_bounded(5));

  const auto segments = ring.segments();
  ASSERT_EQ(segments.first.size(), 2u);
  ASSERT_EQ(segments.second.size(), 2u);
  EXPECT_EQ((std::vector<std::uint32_t>(ring.begin(), ring.end())),
            (std::vector<std::uint32_t>{2, 3, 4, 5}));
}

TEST(RingDeque, BothEndsAndRandomAccessKeepLogicalOrder) {
  RingDeque<int> ring;
  ring.push_back(2);
  ring.push_front(1);
  ring.push_back(3);
  ring.push_front(0);
  ASSERT_EQ(ring.size(), 4u);
  for (std::size_t i = 0; i < ring.size(); ++i)
    EXPECT_EQ(ring[i], static_cast<int>(i));
  EXPECT_EQ(ring.end() - ring.begin(), 4);
  EXPECT_EQ(ring.begin()[2], 2);
  ring.pop_back();
  ring.pop_front();
  EXPECT_EQ(ring.front(), 1);
  EXPECT_EQ(ring.back(), 2);
}

TEST(RingDeque, GrowthPreservesWrappedOrderAndAliasedInput) {
  RingDeque<std::string> ring(2);
  ring.push_back("alpha");
  ring.push_back("beta");
  ring.pop_front();
  ring.push_back("gamma");
  ASSERT_TRUE(ring.full());
  ASSERT_TRUE(ring.try_push_back(ring.front()));
  EXPECT_GE(ring.capacity(), 3u);
  EXPECT_EQ((std::vector<std::string>(ring.begin(), ring.end())),
            (std::vector<std::string>{"beta", "gamma", "beta"}));
  EXPECT_TRUE(ring.is_linearized());
}

TEST(RingDeque, CopyMoveAndClearManageObjectLifetime) {
  EXPECT_EQ(Lifetime::live, 0);
  {
    RingDeque<Lifetime> source;
    source.emplace_back(7);
    source.emplace_front(3);
    RingDeque<Lifetime> copy = source;
    EXPECT_EQ(copy.front().value, 3);
    EXPECT_EQ(copy.back().value, 7);
    RingDeque<Lifetime> moved = std::move(copy);
    EXPECT_TRUE(copy.empty());
    moved.clear();
    EXPECT_TRUE(moved.empty());
  }
  EXPECT_EQ(Lifetime::live, 0);
}

TEST(RingDeque, AllocationFailurePreservesExistingElements) {
  hyle::mem::Allocator failing(&fail_alloc, &fail_free, &fail_realloc, nullptr);
  RingDeque<int> ring(failing);
  EXPECT_FALSE(ring.try_reserve(8));
  EXPECT_FALSE(ring.try_push_back(1));
  EXPECT_TRUE(ring.empty());
}

TEST(RingDeque, RandomOperationsMatchStdDequeModel) {
  RingDeque<int> ring;
  std::deque<int> model;
  std::mt19937 random(0x48594c45u);
  for (int step = 0; step < 20000; ++step) {
    switch (random() % 6) {
    case 0:
      ring.push_back(step);
      model.push_back(step);
      break;
    case 1:
      ring.push_front(step);
      model.push_front(step);
      break;
    case 2:
      if (!model.empty()) {
        ring.pop_back();
        model.pop_back();
      }
      break;
    case 3:
      if (!model.empty()) {
        ring.pop_front();
        model.pop_front();
      }
      break;
    case 4:
      EXPECT_TRUE(ring.try_reserve(model.size() + random() % 32));
      break;
    default:
      if (!model.empty()) {
        EXPECT_EQ(ring.front(), model.front());
        EXPECT_EQ(ring.back(), model.back());
      }
      break;
    }
    ASSERT_EQ(ring.size(), model.size());
    for (std::size_t i = 0; i < model.size(); ++i)
      ASSERT_EQ(ring[i], model[i]);
  }
}
