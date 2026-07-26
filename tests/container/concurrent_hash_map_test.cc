#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

#include "hyle/container/concurrent_hash_map.h"
#include "hyle/mem/allocator.h"

using hyle::container::ConcurrentHashMap;
using hyle::mem::Allocator;
using hyle::mem::system_allocator;

static_assert(
    !std::is_copy_constructible_v<ConcurrentHashMap<int, int>::Accessor>);
static_assert(
    std::is_move_constructible_v<ConcurrentHashMap<int, int>::Accessor>);
static_assert(
    !std::is_copy_constructible_v<ConcurrentHashMap<int, int>::ConstAccessor>);

namespace {

struct ToggleAllocatorState {
  Allocator backing = system_allocator();
  std::atomic<bool> fail{false};
};

void *toggle_allocate(void *raw, std::size_t size,
                      std::size_t alignment) noexcept {
  auto *state = static_cast<ToggleAllocatorState *>(raw);
  return state->fail.load(std::memory_order_relaxed)
             ? nullptr
             : state->backing.allocate(size, alignment);
}

void toggle_deallocate(void *raw, void *pointer, std::size_t size,
                       std::size_t alignment) noexcept {
  static_cast<ToggleAllocatorState *>(raw)->backing.deallocate(pointer, size,
                                                               alignment);
}

void *toggle_reallocate(void *raw, void *pointer, std::size_t old_size,
                        std::size_t new_size, std::size_t alignment) noexcept {
  auto *state = static_cast<ToggleAllocatorState *>(raw);
  return state->fail.load(std::memory_order_relaxed)
             ? nullptr
             : state->backing.reallocate(pointer, old_size, new_size,
                                         alignment);
}

struct CountingStringHash {
  std::atomic<std::size_t> *calls;

  std::uint64_t operator()(const std::string &value) const noexcept {
    return operator()(std::string_view(value));
  }
  std::uint64_t operator()(std::string_view value) const noexcept {
    calls->fetch_add(1, std::memory_order_relaxed);
    return hyle::container::hash_bytes(value.data(), value.size());
  }
};

struct TransparentStringEq {
  bool operator()(const std::string &left,
                  const std::string &right) const noexcept {
    return left == right;
  }
  bool operator()(const std::string &left,
                  std::string_view right) const noexcept {
    return left == right;
  }
};

} // namespace

TEST(ConcurrentHashMap, ValidatesPowerOfTwoShardCountAndAllocation) {
  ConcurrentHashMap<int, int> invalid(3);
  EXPECT_FALSE(invalid.valid());

  ToggleAllocatorState state;
  state.fail.store(true);
  Allocator failing(&toggle_allocate, &toggle_deallocate, &toggle_reallocate,
                    &state);
  ConcurrentHashMap<int, int, hyle::container::Hash<int>, std::equal_to<>,
                    Allocator>
      no_memory(8, failing);
  EXPECT_FALSE(no_memory.valid());
}

TEST(ConcurrentHashMap, AccessorsProtectLookupAndMutation) {
  using Map = ConcurrentHashMap<int, int>;
  Map map(8);
  ASSERT_TRUE(map.valid());

  Map::Accessor writer;
  EXPECT_EQ(map.try_emplace(writer, 7, 70), Map::InsertResult::kInserted);
  EXPECT_EQ(writer.key(), 7);
  *writer = 71;
  writer.release();

  Map::ConstAccessor reader;
  ASSERT_TRUE(map.find(reader, 7));
  EXPECT_EQ(reader.key(), 7);
  EXPECT_EQ(*reader, 71);
  reader.release();

  EXPECT_EQ(map.try_emplace(writer, 7, 99), Map::InsertResult::kAlreadyPresent);
  EXPECT_EQ(*writer, 71);
  writer.release();
  EXPECT_EQ(map.try_insert_or_assign(writer, 7, 72),
            Map::InsertResult::kAssigned);
  EXPECT_EQ(*writer, 72);
}

TEST(ConcurrentHashMap, ConstAccessorKeepsEraseWaiting) {
  ConcurrentHashMap<int, int> map(8);
  ConcurrentHashMap<int, int>::Accessor inserted;
  ASSERT_TRUE(map.emplace(inserted, 1, 2));
  inserted.release();

  ConcurrentHashMap<int, int>::ConstAccessor held;
  ASSERT_TRUE(map.find(held, 1));
  std::promise<void> started;
  std::future<void> started_future = started.get_future();
  std::future<bool> erased = std::async(std::launch::async, [&] {
    started.set_value();
    return map.erase(1);
  });
  started_future.wait();
  EXPECT_EQ(erased.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  EXPECT_EQ(*held, 2);
  held.release();
  EXPECT_TRUE(erased.get());
}

TEST(ConcurrentHashMap, ConcurrentDisjointInsertAndSharedUpdate) {
  using Map = ConcurrentHashMap<int, int>;
  constexpr int kThreads = 8;
  constexpr int kPerThread = 4000;
  Map map(64);
  ASSERT_TRUE(map.reserve(kThreads * kPerThread));

  std::vector<std::thread> threads;
  for (int thread = 0; thread < kThreads; ++thread) {
    threads.emplace_back([&, thread] {
      Map::Accessor accessor;
      for (int i = 0; i < kPerThread; ++i) {
        const int key = thread * kPerThread + i;
        const auto result = map.try_emplace(accessor, key, 1);
        ASSERT_EQ(result, Map::InsertResult::kInserted);
        accessor.release();
      }
    });
  }
  for (std::thread &thread : threads)
    thread.join();
  EXPECT_EQ(map.size(), static_cast<std::size_t>(kThreads * kPerThread));

  threads.clear();
  for (int thread = 0; thread < kThreads; ++thread) {
    threads.emplace_back([&] {
      Map::Accessor accessor;
      for (int key = 0; key < kThreads * kPerThread; ++key) {
        ASSERT_TRUE(map.find(accessor, key));
        ++*accessor;
        accessor.release();
      }
    });
  }
  for (std::thread &thread : threads)
    thread.join();

  std::size_t visited = 0;
  map.for_each([&](int, int value) {
    EXPECT_EQ(value, kThreads + 1);
    ++visited;
  });
  EXPECT_EQ(visited, map.size());
}

TEST(ConcurrentHashMap, FallibleInsertDoesNotStrandLockOrEntry) {
  ToggleAllocatorState state;
  Allocator allocator(&toggle_allocate, &toggle_deallocate, &toggle_reallocate,
                      &state);
  using Map = ConcurrentHashMap<int, int, hyle::container::Hash<int>,
                                std::equal_to<>, Allocator>;
  Map map(8, allocator);
  ASSERT_TRUE(map.valid());
  state.fail.store(true);

  Map::Accessor accessor;
  EXPECT_EQ(map.try_emplace(accessor, 1, 2), Map::InsertResult::kOutOfMemory);
  EXPECT_FALSE(accessor);
  EXPECT_TRUE(map.empty());

  state.fail.store(false);
  EXPECT_EQ(map.try_emplace(accessor, 1, 2), Map::InsertResult::kInserted);
  EXPECT_EQ(*accessor, 2);
}

TEST(ConcurrentHashMap, LongKeyLookupHashesOnlyOnce) {
  std::atomic<std::size_t> calls{0};
  using Map = ConcurrentHashMap<std::string, int, CountingStringHash,
                                TransparentStringEq>;
  Map map(8, hyle::mem::SystemAllocator{}, CountingStringHash{&calls},
          TransparentStringEq{});
  ASSERT_TRUE(map.reserve(32));
  Map::Accessor inserted;
  EXPECT_EQ(map.try_emplace(inserted, std::string(200, 'x'), 3),
            Map::InsertResult::kInserted);
  inserted.release();

  calls.store(0);
  Map::ConstAccessor found;
  ASSERT_TRUE(map.find(found, std::string_view(std::string(200, 'x'))));
  EXPECT_EQ(*found, 3);
  EXPECT_EQ(calls.load(), 1u);
}
