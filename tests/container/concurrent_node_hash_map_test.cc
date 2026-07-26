#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <thread>
#include <type_traits>
#include <vector>

#include "hyle/container/concurrent_node_hash_map.h"
#include "hyle/mem/allocator.h"

using hyle::container::ConcurrentNodeHashMap;
using hyle::mem::Allocator;
using hyle::mem::system_allocator;

static_assert(
    !std::is_copy_constructible_v<ConcurrentNodeHashMap<int, int>::Accessor>);

namespace {

struct NodeToggleAllocatorState {
  Allocator backing = system_allocator();
  std::atomic<bool> fail{false};
};

void *node_toggle_allocate(void *raw, std::size_t size,
                           std::size_t alignment) noexcept {
  auto *state = static_cast<NodeToggleAllocatorState *>(raw);
  return state->fail.load(std::memory_order_relaxed)
             ? nullptr
             : state->backing.allocate(size, alignment);
}
void node_toggle_deallocate(void *raw, void *pointer, std::size_t size,
                            std::size_t alignment) noexcept {
  static_cast<NodeToggleAllocatorState *>(raw)->backing.deallocate(
      pointer, size, alignment);
}
void *node_toggle_reallocate(void *, void *, std::size_t, std::size_t,
                             std::size_t) noexcept {
  return nullptr;
}

} // namespace

TEST(ConcurrentNodeHashMap, BasicAccessorsInsertAssignAndErase) {
  using Map = ConcurrentNodeHashMap<int, int>;
  Map map(16);
  ASSERT_TRUE(map.valid());
  Map::Accessor writer;
  EXPECT_EQ(map.try_emplace(writer, 3, 30), Map::InsertResult::kInserted);
  *writer = 31;
  writer.release();
  EXPECT_EQ(map.try_emplace(writer, 3, 99), Map::InsertResult::kAlreadyPresent);
  EXPECT_EQ(*writer, 31);
  writer.release();
  EXPECT_EQ(map.try_insert_or_assign(writer, 3, 32),
            Map::InsertResult::kAssigned);
  writer.release();

  Map::ConstAccessor reader;
  ASSERT_TRUE(map.find(reader, 3));
  EXPECT_EQ(reader.key(), 3);
  EXPECT_EQ(*reader, 32);
  reader.release();
  EXPECT_TRUE(map.erase(3));
  EXPECT_FALSE(map.contains(3));
}

TEST(ConcurrentNodeHashMap, ExistingKeyAtCapacityDoesNotTriggerGrowth) {
  using Map = ConcurrentNodeHashMap<int, int>;
  Map map(16);
  Map::Accessor accessor;
  for (int key = 0; key < 16; ++key) {
    ASSERT_EQ(map.try_emplace(accessor, key, key),
              Map::InsertResult::kInserted);
    accessor.release();
  }
  ASSERT_EQ(map.bucket_count(), 16u);

  EXPECT_EQ(map.try_emplace(accessor, 7, 70),
            Map::InsertResult::kAlreadyPresent);
  accessor.release();
  EXPECT_EQ(map.try_insert_or_assign(accessor, 7, 71),
            Map::InsertResult::kAssigned);
  EXPECT_EQ(*accessor, 71);
  accessor.release();
  EXPECT_EQ(map.bucket_count(), 16u);
}

TEST(ConcurrentNodeHashMap, RehashPreservesValuesAndWaitsForAccessor) {
  using Map = ConcurrentNodeHashMap<int, int>;
  Map map(16);
  Map::Accessor writer;
  for (int i = 0; i < 2000; ++i) {
    ASSERT_TRUE(map.emplace(writer, i, i * 2));
    writer.release();
  }
  EXPECT_GE(map.bucket_count(), 2048u);

  Map::ConstAccessor held;
  ASSERT_TRUE(map.find(held, 7));
  const int *stable_address = &*held;
  std::future<bool> reserved =
      std::async(std::launch::async, [&] { return map.reserve(8192); });
  EXPECT_EQ(reserved.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  EXPECT_EQ(*stable_address, 14);
  held.release();
  EXPECT_TRUE(reserved.get());

  Map::ConstAccessor found;
  for (int i = 0; i < 2000; ++i) {
    ASSERT_TRUE(map.find(found, i));
    EXPECT_EQ(*found, i * 2);
    found.release();
  }
}

TEST(ConcurrentNodeHashMap, ConcurrentMixedReadUpdateAndGrowth) {
  using Map = ConcurrentNodeHashMap<int, int>;
  constexpr int kThreads = 8;
  constexpr int kKeys = 16000;
  Map map(64);
  std::vector<std::thread> threads;
  for (int thread = 0; thread < kThreads; ++thread) {
    threads.emplace_back([&, thread] {
      Map::Accessor accessor;
      for (int key = thread; key < kKeys; key += kThreads) {
        ASSERT_EQ(map.try_emplace(accessor, key, 1),
                  Map::InsertResult::kInserted);
        accessor.release();
      }
    });
  }
  for (std::thread &thread : threads)
    thread.join();
  ASSERT_EQ(map.size(), static_cast<std::size_t>(kKeys));

  threads.clear();
  for (int thread = 0; thread < kThreads; ++thread) {
    threads.emplace_back([&, thread] {
      Map::ConstAccessor reader;
      Map::Accessor writer;
      for (int i = 0; i < 50000; ++i) {
        const int key = (i * 17 + thread * 101) % kKeys;
        if (i % 10 == 0) {
          ASSERT_TRUE(map.find(writer, key));
          ++*writer;
          writer.release();
        } else {
          ASSERT_TRUE(map.find(reader, key));
          EXPECT_GE(*reader, 1);
          reader.release();
        }
      }
    });
  }
  for (std::thread &thread : threads)
    thread.join();
  EXPECT_EQ(map.size(), static_cast<std::size_t>(kKeys));
}

TEST(ConcurrentNodeHashMap, FallibleNodeAllocationLeavesMapUsable) {
  NodeToggleAllocatorState state;
  Allocator allocator(&node_toggle_allocate, &node_toggle_deallocate,
                      &node_toggle_reallocate, &state);
  using Map = ConcurrentNodeHashMap<int, int, hyle::container::Hash<int>,
                                    std::equal_to<>, Allocator>;
  Map map(16, allocator);
  ASSERT_TRUE(map.valid());
  state.fail.store(true);
  Map::Accessor accessor;
  EXPECT_EQ(map.try_emplace(accessor, 1, 2), Map::InsertResult::kOutOfMemory);
  EXPECT_TRUE(map.empty());
  state.fail.store(false);
  EXPECT_EQ(map.try_emplace(accessor, 1, 2), Map::InsertResult::kInserted);
}

TEST(ConcurrentNodeHashMap, ForEachVisitsEveryStableNode) {
  ConcurrentNodeHashMap<int, int> map(16);
  ConcurrentNodeHashMap<int, int>::Accessor accessor;
  for (int i = 0; i < 1000; ++i) {
    map.emplace(accessor, i, i);
    accessor.release();
  }
  std::uint64_t sum = 0;
  map.for_each([&](int key, int value) {
    EXPECT_EQ(key, value);
    sum += static_cast<std::uint64_t>(value);
  });
  EXPECT_EQ(sum, 999u * 1000u / 2u);
  map.clear();
  EXPECT_TRUE(map.empty());
}
