#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include "hyle/c/container.h"

namespace {

void increment_int(void* value, void*) {
  ++*static_cast<int*>(value);
}

int sum_int_pair(const void* key, const void* value, void* raw_sum) {
  auto* sum = static_cast<std::uint64_t*>(raw_sum);
  *sum += static_cast<std::uint64_t>(*static_cast<const int*>(key));
  *sum += static_cast<std::uint64_t>(*static_cast<const int*>(value));
  return 1;
}

}  // namespace

TEST(ContainerCAbi, BoundedQueueMpmcConservesPointers) {
  constexpr std::uintptr_t kProducers = 4;
  constexpr std::uintptr_t kConsumers = 4;
  constexpr std::uintptr_t kPerProducer = 20000;
  constexpr std::uintptr_t kTotal = kProducers * kPerProducer;
  hyle_container_bounded_queue* queue = nullptr;
  hyle_status create =
      hyle_container_bounded_queue_create_system(&queue, 1024);
  ASSERT_TRUE(hyle_status_is_ok(create));
  hyle_status_free(&create);
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
        while (!hyle_container_bounded_queue_try_push(queue, value))
          std::this_thread::yield();
        produced.fetch_add(1, std::memory_order_release);
      }
    });
  }
  for (std::uintptr_t consumer = 0; consumer < kConsumers; ++consumer) {
    threads.emplace_back([&] {
      for (;;) {
        void* value = nullptr;
        if (hyle_container_bounded_queue_try_pop(queue, &value)) {
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

TEST(ContainerCAbi, SpscQueuePreservesFifoAcrossThreads) {
  constexpr std::uintptr_t kCount = 100000;
  hyle_container_spsc_queue* queue = nullptr;
  hyle_status create = hyle_container_spsc_queue_create_system(&queue, 1024);
  ASSERT_TRUE(hyle_status_is_ok(create));
  hyle_status_free(&create);
  ASSERT_NE(queue, nullptr);

  std::atomic<bool> start{false};
  std::uint64_t sum = 0;
  std::thread producer([&] {
    while (!start.load(std::memory_order_acquire)) {}
    for (std::uintptr_t value = 1; value <= kCount; ++value) {
      void* item = reinterpret_cast<void*>(value);
      while (!hyle_container_spsc_queue_try_push(queue, item))
        std::this_thread::yield();
    }
  });
  std::thread consumer([&] {
    start.store(true, std::memory_order_release);
    for (std::uintptr_t expected = 1; expected <= kCount; ++expected) {
      void* item = nullptr;
      while (!hyle_container_spsc_queue_try_pop(queue, &item))
        std::this_thread::yield();
      EXPECT_EQ(reinterpret_cast<std::uintptr_t>(item), expected);
      sum += reinterpret_cast<std::uintptr_t>(item);
    }
  });
  producer.join();
  consumer.join();

  EXPECT_EQ(sum, kCount * (kCount + 1) / 2);
  EXPECT_EQ(hyle_container_spsc_queue_size_approx(queue), 0u);
  hyle_container_spsc_queue_destroy(queue);
}

TEST(ContainerCAbi, ConcurrentVectorPublishesStableIndices) {
  constexpr std::size_t kWriters = 4;
  constexpr std::size_t kPerWriter = 10000;
  hyle_container_concurrent_ptr_vector* vector = nullptr;
  hyle_status create =
      hyle_container_concurrent_ptr_vector_create_system(&vector, 1);
  ASSERT_TRUE(hyle_status_is_ok(create));
  hyle_status_free(&create);
  ASSERT_NE(vector, nullptr);
  hyle_status reserve = hyle_container_concurrent_ptr_vector_reserve(
      vector, kWriters * kPerWriter);
  ASSERT_TRUE(hyle_status_is_ok(reserve));
  hyle_status_free(&reserve);

  std::vector<std::thread> threads;
  for (std::size_t writer = 0; writer < kWriters; ++writer) {
    threads.emplace_back([&, writer] {
      for (std::size_t i = 0; i < kPerWriter; ++i) {
        void* value = reinterpret_cast<void*>(
            static_cast<std::uintptr_t>(writer * kPerWriter + i + 1));
        std::size_t index = SIZE_MAX;
        hyle_status status =
            hyle_container_concurrent_ptr_vector_try_push_back(
                &index, vector, value);
        EXPECT_TRUE(hyle_status_is_ok(status));
        EXPECT_NE(index, SIZE_MAX);
        hyle_status_free(&status);
      }
    });
  }
  for (std::thread& thread : threads) thread.join();

  ASSERT_EQ(hyle_container_concurrent_ptr_vector_size(vector),
            kWriters * kPerWriter);
  std::uint64_t sum = 0;
  for (std::size_t i = 0; i < kWriters * kPerWriter; ++i) {
    void* value = nullptr;
    ASSERT_TRUE(
        hyle_container_concurrent_ptr_vector_try_get(vector, i, &value));
    sum += reinterpret_cast<std::uintptr_t>(value);
  }
  const std::uint64_t count = kWriters * kPerWriter;
  EXPECT_EQ(sum, count * (count + 1) / 2);
  hyle_container_concurrent_ptr_vector_destroy(vector);
}

TEST(ContainerCAbi, ConcurrentHashMapSupportsSafeCopyAndUpdate) {
  constexpr int kThreads = 8;
  constexpr int kPerThread = 3000;
  constexpr int kCount = kThreads * kPerThread;
  hyle_container_hash_map_options options =
      HYLE_CONTAINER_HASH_MAP_OPTIONS_FOR(int, int);
  hyle_container_concurrent_hash_map* map = nullptr;
  hyle_status create = hyle_container_concurrent_hash_map_create(
      &map, &options, 64);
  ASSERT_TRUE(hyle_status_is_ok(create));
  hyle_status_free(&create);
  ASSERT_NE(map, nullptr);
  EXPECT_EQ(hyle_container_concurrent_hash_map_shard_count(map), 64u);

  hyle_status reserve =
      hyle_container_concurrent_hash_map_reserve(map, kCount);
  ASSERT_TRUE(hyle_status_is_ok(reserve));
  hyle_status_free(&reserve);

  std::vector<std::thread> threads;
  for (int thread = 0; thread < kThreads; ++thread) {
    threads.emplace_back([&, thread] {
      for (int i = 0; i < kPerThread; ++i) {
        const int key = thread * kPerThread + i;
        const int value = 1;
        hyle_container_insert_result result =
            HYLE_CONTAINER_ALREADY_PRESENT;
        hyle_status status = hyle_container_concurrent_hash_map_insert(
            &result, map, &key, &value);
        EXPECT_TRUE(hyle_status_is_ok(status));
        EXPECT_EQ(result, HYLE_CONTAINER_INSERTED);
        hyle_status_free(&status);
      }
    });
  }
  for (std::thread& thread : threads) thread.join();
  ASSERT_EQ(hyle_container_concurrent_hash_map_size(map),
            static_cast<std::size_t>(kCount));

  threads.clear();
  for (int thread = 0; thread < kThreads; ++thread) {
    threads.emplace_back([&] {
      for (int key = 0; key < kCount; ++key)
        ASSERT_TRUE(hyle_container_concurrent_hash_map_update(
            map, &key, &increment_int, nullptr));
    });
  }
  for (std::thread& thread : threads) thread.join();

  for (int key = 0; key < kCount; ++key) {
    int value = -1;
    ASSERT_TRUE(hyle_container_concurrent_hash_map_find_copy(
        &value, map, &key));
    EXPECT_EQ(value, kThreads + 1);
  }
  std::uint64_t sum = 0;
  EXPECT_TRUE(hyle_container_concurrent_hash_map_visit(
      map, &sum_int_pair, &sum));
  const std::uint64_t key_sum =
      static_cast<std::uint64_t>(kCount) * (kCount - 1) / 2;
  EXPECT_EQ(sum, key_sum +
                     static_cast<std::uint64_t>(kCount) * (kThreads + 1));

  for (int key = 0; key < kCount; key += 2)
    EXPECT_TRUE(hyle_container_concurrent_hash_map_erase(map, &key));
  EXPECT_EQ(hyle_container_concurrent_hash_map_size(map),
            static_cast<std::size_t>(kCount / 2));
  hyle_container_concurrent_hash_map_destroy(map);
}
