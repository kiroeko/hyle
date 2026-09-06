#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <utility>

#include "hyle/container/spsc_queue.h"

using hyle::container::SpscQueue;

namespace {

struct NonAssignable {
  explicit NonAssignable(int init) : value(init) {}
  NonAssignable(const NonAssignable&) = delete;
  NonAssignable& operator=(const NonAssignable&) = delete;
  NonAssignable(NonAssignable&&) = delete;
  NonAssignable& operator=(NonAssignable&&) = delete;
  int value;
};

struct Lifetime {
  explicit Lifetime(int init) : value(init) { ++alive; }
  Lifetime(const Lifetime& other) : value(other.value) { ++alive; }
  Lifetime(Lifetime&& other) noexcept : value(other.value) { ++alive; }
  Lifetime& operator=(Lifetime&& other) noexcept {
    value = other.value;
    return *this;
  }
  ~Lifetime() noexcept { --alive; }
  int value;
  static inline int alive = 0;
};

}  // namespace

TEST(SpscQueue, ExactCapacityAndWrap) {
  SpscQueue<int> queue(4);
  ASSERT_TRUE(queue.valid());
  EXPECT_EQ(queue.capacity(), 4u);
  EXPECT_TRUE(queue.empty());
  EXPECT_TRUE(queue.try_push(1));
  EXPECT_TRUE(queue.try_push(2));
  EXPECT_TRUE(queue.try_push(3));
  EXPECT_TRUE(queue.try_push(4));
  EXPECT_FALSE(queue.try_push(5));
  EXPECT_TRUE(queue.full());
  EXPECT_EQ(queue.size_approx(), 4u);

  int value = 0;
  ASSERT_TRUE(queue.try_pop(value));
  EXPECT_EQ(value, 1);
  ASSERT_TRUE(queue.try_pop(value));
  EXPECT_EQ(value, 2);
  EXPECT_TRUE(queue.try_push(5));
  EXPECT_TRUE(queue.try_push(6));
  for (int expected = 3; expected <= 6; ++expected) {
    ASSERT_TRUE(queue.try_pop(value));
    EXPECT_EQ(value, expected);
  }
  EXPECT_FALSE(queue.try_pop(value));
  EXPECT_TRUE(queue.empty());
}

TEST(SpscQueue, CapacityNeedNotBePowerOfTwo) {
  SpscQueue<int> queue(3);
  ASSERT_TRUE(queue.valid());
  EXPECT_EQ(queue.capacity(), 3u);
  EXPECT_TRUE(queue.try_push(1));
  EXPECT_TRUE(queue.try_push(2));
  EXPECT_TRUE(queue.try_push(3));
  EXPECT_FALSE(queue.try_push(4));
  int value = 0;
  for (int expected = 1; expected <= 3; ++expected) {
    ASSERT_TRUE(queue.try_pop(value));
    EXPECT_EQ(value, expected);
  }
}

TEST(SpscQueue, SupportsNonAssignableValuesThroughConsumerCallback) {
  SpscQueue<NonAssignable> queue(2);
  ASSERT_TRUE(queue.try_emplace(42));
  ASSERT_NE(queue.try_front(), nullptr);
  EXPECT_EQ(queue.try_front()->value, 42);
  int observed = 0;
  ASSERT_TRUE(queue.try_consume(
      [&](NonAssignable& value) { observed = value.value; }));
  EXPECT_EQ(observed, 42);
  EXPECT_EQ(queue.try_front(), nullptr);
}

TEST(SpscQueue, OwnsAndDestroysOnlyLiveValues) {
  Lifetime::alive = 0;
  {
    SpscQueue<Lifetime> queue(4);
    ASSERT_TRUE(queue.try_emplace(1));
    ASSERT_TRUE(queue.try_emplace(2));
    EXPECT_EQ(Lifetime::alive, 2);
    Lifetime output(0);
    ASSERT_TRUE(queue.try_pop(output));
    EXPECT_EQ(output.value, 1);
    EXPECT_EQ(Lifetime::alive, 2);
  }
  EXPECT_EQ(Lifetime::alive, 0);
}

TEST(SpscQueue, TransfersSingleProducerSingleConsumerWithoutLoss) {
  constexpr std::uint64_t kCount = 200000;
  SpscQueue<std::uint64_t> queue(1024);
  ASSERT_TRUE(queue.valid());
  std::atomic<bool> start{false};
  std::uint64_t sum = 0;

  std::thread producer([&] {
    while (!start.load(std::memory_order_acquire)) {}
    for (std::uint64_t value = 1; value <= kCount; ++value)
      while (!queue.try_push(value)) std::this_thread::yield();
  });
  std::thread consumer([&] {
    start.store(true, std::memory_order_release);
    for (std::uint64_t index = 0; index < kCount; ++index) {
      std::uint64_t value = 0;
      while (!queue.try_pop(value)) std::this_thread::yield();
      sum += value;
    }
  });
  producer.join();
  consumer.join();

  EXPECT_EQ(sum, kCount * (kCount + 1) / 2);
  EXPECT_TRUE(queue.empty());
}

TEST(SpscQueue, MovesOwningValues) {
  SpscQueue<std::unique_ptr<int>> queue(2);
  ASSERT_TRUE(queue.try_push(std::make_unique<int>(7)));
  std::unique_ptr<int> value;
  ASSERT_TRUE(queue.try_pop(value));
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(*value, 7);
}
