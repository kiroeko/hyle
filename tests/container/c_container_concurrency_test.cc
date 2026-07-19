#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include "hyle/c/container.h"

TEST(ContainerCAbi, BoundedQueueMpmcConservesPointers) {
  constexpr std::uintptr_t kProducers = 4;
  constexpr std::uintptr_t kConsumers = 4;
  constexpr std::uintptr_t kPerProducer = 20000;
  constexpr std::uintptr_t kTotal = kProducers * kPerProducer;
  hyle_container_bounded_queue* queue =
      hyle_container_bounded_queue_create_system(1024);
  ASSERT_NE(queue, nullptr);

  std::atomic<std::uintptr_t> produced{0};
  std::atomic<std::uintptr_t> consumed{0};
  std::atomic<std::uint64_t> sum{0};
  std::vector<std::thread> threads;
  for (std::uintptr_t producer = 0; producer < kProducers; ++producer) {
    threads.emplace_back([&, producer] {
      const std::uintptr_t base = producer * kPerProducer;
      for (std::uintptr_t i = 1; i <= kPerProducer; ++i) {
        void* value = reinterpret_cast<void*>(base + i);
        while (!hyle_container_bounded_queue_push(queue, value))
          std::this_thread::yield();
        produced.fetch_add(1, std::memory_order_release);
      }
    });
  }
  for (std::uintptr_t consumer = 0; consumer < kConsumers; ++consumer) {
    threads.emplace_back([&] {
      for (;;) {
        void* value = nullptr;
        if (hyle_container_bounded_queue_pop(queue, &value)) {
          sum.fetch_add(reinterpret_cast<std::uintptr_t>(value),
                        std::memory_order_relaxed);
          consumed.fetch_add(1, std::memory_order_relaxed);
        } else if (produced.load(std::memory_order_acquire) == kTotal &&
                   consumed.load(std::memory_order_relaxed) == kTotal) {
          break;
        } else {
          std::this_thread::yield();
        }
      }
    });
  }
  for (std::thread& thread : threads) thread.join();

  EXPECT_EQ(consumed.load(), kTotal);
  EXPECT_EQ(sum.load(), kTotal * (kTotal + 1) / 2);
  hyle_container_bounded_queue_destroy(queue);
}

TEST(ContainerCAbi, ConcurrentVectorPublishesStableIndices) {
  constexpr std::size_t kWriters = 4;
  constexpr std::size_t kPerWriter = 10000;
  hyle_container_concurrent_vector* vector =
      hyle_container_concurrent_vector_create_system(1);
  ASSERT_NE(vector, nullptr);

  std::vector<std::thread> threads;
  for (std::size_t writer = 0; writer < kWriters; ++writer) {
    threads.emplace_back([&, writer] {
      for (std::size_t i = 0; i < kPerWriter; ++i) {
        void* value = reinterpret_cast<void*>(
            static_cast<std::uintptr_t>(writer * kPerWriter + i + 1));
        EXPECT_NE(hyle_container_concurrent_vector_push(vector, value),
                  SIZE_MAX);
      }
    });
  }
  for (std::thread& thread : threads) thread.join();

  ASSERT_EQ(hyle_container_concurrent_vector_size(vector),
            kWriters * kPerWriter);
  std::uint64_t sum = 0;
  for (std::size_t i = 0; i < kWriters * kPerWriter; ++i) {
    sum += reinterpret_cast<std::uintptr_t>(
        hyle_container_concurrent_vector_at(vector, i));
  }
  const std::uint64_t count = kWriters * kPerWriter;
  EXPECT_EQ(sum, count * (count + 1) / 2);
  hyle_container_concurrent_vector_destroy(vector);
}
