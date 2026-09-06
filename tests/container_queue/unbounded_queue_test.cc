#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <memory>
#include <thread>
#include <vector>

#include "hyle/container/unbounded_queue.h"

namespace {

using hyle::container::UnboundedQueue;

struct MoveOnly {
  explicit MoveOnly(int v) noexcept : value(v) {}
  MoveOnly(const MoveOnly&) = delete;
  MoveOnly& operator=(const MoveOnly&) = delete;
  MoveOnly(MoveOnly&&) noexcept = default;
  MoveOnly& operator=(MoveOnly&&) = delete;
  int value;
};

struct FailingAllocator {
  bool* fail = nullptr;

  void* allocate(std::size_t size, std::size_t alignment) const noexcept {
    if (*fail) return nullptr;
    return hyle::mem::SystemAllocator::allocate(size, alignment);
  }
  void deallocate(void* ptr, std::size_t size,
                  std::size_t alignment) const noexcept {
    hyle::mem::SystemAllocator::deallocate(ptr, size, alignment);
  }
};

TEST(UnboundedQueue, PreservesFifoAcrossBlocks) {
  UnboundedQueue<int, hyle::mem::SystemAllocator, 2> queue;
  ASSERT_TRUE(queue.valid());
  for (int value = 0; value != 7; ++value)
    ASSERT_TRUE(queue.try_push(value));
  EXPECT_EQ(queue.approximate_size(), 7u);

  for (int expected = 0; expected != 7; ++expected) {
    int actual = -1;
    ASSERT_TRUE(queue.try_pop(actual));
    EXPECT_EQ(actual, expected);
  }
  EXPECT_FALSE(queue.try_pop([](int&&) noexcept {}));
  EXPECT_TRUE(queue.empty());
}

TEST(UnboundedQueue, CallbackSupportsNonAssignablePayload) {
  UnboundedQueue<MoveOnly, hyle::mem::SystemAllocator, 2> queue;
  ASSERT_TRUE(queue.try_emplace(42));
  int observed = 0;
  ASSERT_TRUE(queue.try_pop(
      [&](MoveOnly&& value) noexcept { observed = value.value; }));
  EXPECT_EQ(observed, 42);
}

TEST(UnboundedQueue, AllocationFailureDoesNotConsumeFifoPosition) {
  bool fail = false;
  UnboundedQueue<int, FailingAllocator, 2> queue(FailingAllocator{&fail});
  ASSERT_TRUE(queue.valid());
  ASSERT_TRUE(queue.try_push(1));
  ASSERT_TRUE(queue.try_push(2));
  fail = true;
  EXPECT_FALSE(queue.try_push(3));
  fail = false;
  EXPECT_TRUE(queue.try_push(3));

  for (int expected : {1, 2, 3}) {
    int actual = 0;
    ASSERT_TRUE(queue.try_pop(actual));
    EXPECT_EQ(actual, expected);
  }
}

TEST(UnboundedQueue, MultipleProducersAndConsumersLoseNoItems) {
  constexpr int kProducers = 4;
  constexpr int kConsumers = 4;
  constexpr int kPerProducer = 4000;
  constexpr int kTotal = kProducers * kPerProducer;

  UnboundedQueue<int> queue;
  ASSERT_TRUE(queue.valid());
  std::vector<std::atomic<unsigned char>> seen(kTotal);
  for (auto& value : seen) value.store(0, std::memory_order_relaxed);
  std::atomic<int> consumed{0};
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;

  for (int producer = 0; producer != kProducers; ++producer) {
    threads.emplace_back([&, producer] {
      while (!start.load(std::memory_order_acquire)) {}
      for (int i = 0; i != kPerProducer; ++i) {
        const int value = producer * kPerProducer + i;
        while (!queue.try_push(value)) {}
      }
    });
  }
  for (int consumer = 0; consumer != kConsumers; ++consumer) {
    threads.emplace_back([&] {
      while (!start.load(std::memory_order_acquire)) {}
      while (consumed.load(std::memory_order_relaxed) < kTotal) {
        int value = -1;
        if (queue.try_pop(value)) {
          ASSERT_GE(value, 0);
          ASSERT_LT(value, kTotal);
          EXPECT_EQ(seen[value].fetch_add(1, std::memory_order_relaxed), 0);
          consumed.fetch_add(1, std::memory_order_relaxed);
        } else {
          std::this_thread::yield();
        }
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (auto& thread : threads) thread.join();
  EXPECT_EQ(consumed.load(), kTotal);
  for (const auto& value : seen) EXPECT_EQ(value.load(), 1);
  EXPECT_TRUE(queue.empty());
}

}  // namespace
