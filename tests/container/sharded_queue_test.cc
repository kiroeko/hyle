#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "hyle/container/sharded_queue.h"
#include "hyle/mem/allocator.h"
#include "hyle/platform/thread.h"

using hyle::container::ShardedQueue;
using hyle::mem::Allocator;
using hyle::mem::system_allocator;

namespace {

struct ShardedMoveOnly {
  explicit ShardedMoveOnly(int input) noexcept : value(input) { ++alive; }
  ShardedMoveOnly(const ShardedMoveOnly&) = delete;
  ShardedMoveOnly& operator=(const ShardedMoveOnly&) = delete;
  ShardedMoveOnly(ShardedMoveOnly&& other) noexcept
      : value(std::exchange(other.value, -1)) {
    ++alive;
  }
  ShardedMoveOnly& operator=(ShardedMoveOnly&&) = delete;
  ~ShardedMoveOnly() noexcept { --alive; }

  static inline int alive = 0;
  int value;
};

}  // namespace

// ---- single-thread structure ------------------------------------------------

TEST(ShardedQueue, CapacityIsSumOfShards) {
  ShardedQueue<int> q(4, 8);  // default = compile-time SystemAllocator
  EXPECT_EQ(q.num_shards(), 4u);
  EXPECT_EQ(q.capacity(), 32u);  // 4 shards * 8
}

TEST(ShardedQueue, PerShardFifoSameHint) {
  ShardedQueue<int> q(4, 8);
  // Same hint -> same shard -> FIFO within it.
  EXPECT_TRUE(q.try_push(0, 10));
  EXPECT_TRUE(q.try_push(0, 20));
  EXPECT_TRUE(q.try_push(0, 30));
  std::size_t cur = 0;
  int v = 0;
  ASSERT_TRUE(q.try_pop(cur, v));
  EXPECT_EQ(v, 10);
  ASSERT_TRUE(q.try_pop(cur, v));
  EXPECT_EQ(v, 20);
  ASSERT_TRUE(q.try_pop(cur, v));
  EXPECT_EQ(v, 30);
}

TEST(ShardedQueue, PopScansAllShards) {
  // Three shards exercises the non-power-of-two modulo fallback; an arbitrary
  // large cursor proves try_pop normalizes caller-owned cursors before its
  // division-free increment/wrap scan.
  ShardedQueue<int> q(3, 8);
  // Push to distinct shards via distinct hints; a single cursor must still find
  // them all (round-robin scan), regardless of where it starts.
  EXPECT_TRUE(q.try_push(0, 100));
  EXPECT_TRUE(q.try_push(1, 101));
  EXPECT_TRUE(q.try_push(2, 102));
  std::size_t cur = 1024;
  std::unordered_set<int> got;
  int v = 0;
  while (q.try_pop(cur, v)) got.insert(v);
  EXPECT_EQ(got.size(), 3u);
  EXPECT_TRUE(got.count(100) && got.count(101) && got.count(102));
}

TEST(ShardedQueue, SuccessfulPopAdvancesPastBusyShard) {
  ShardedQueue<int> q(2, 8);
  EXPECT_TRUE(q.try_push(0, 10));
  EXPECT_TRUE(q.try_push(0, 11));
  EXPECT_TRUE(q.try_push(1, 20));
  std::size_t cursor = 0;
  int value = 0;
  ASSERT_TRUE(q.try_pop(cursor, value));
  EXPECT_EQ(value, 10);
  EXPECT_EQ(cursor, 1u);
  ASSERT_TRUE(q.try_pop(cursor, value));
  EXPECT_EQ(value, 20);  // shard 1 is not starved by continuously busy shard 0
  EXPECT_EQ(cursor, 0u);
}

TEST(ShardedQueue, ShardFullRejectsButOthersHaveRoom) {
  ShardedQueue<int> q(2, 2);  // 2 shards, cap 2 each
  // Fill shard chosen by hint 0 (=shard 0).
  EXPECT_TRUE(q.try_push(0, 1));
  EXPECT_TRUE(q.try_push(0, 2));
  EXPECT_FALSE(q.try_push(0, 3));  // shard 0 full
  // hint 1 -> shard 1 still has room.
  EXPECT_TRUE(q.try_push(1, 4));
  EXPECT_TRUE(q.try_push(1, 5));
  EXPECT_FALSE(q.try_push(1, 6));  // shard 1 full now too
}

TEST(ShardedQueue, SingleShardDegeneratesToBoundedQueue) {
  // num_shards==1 must behave exactly like one BoundedQueue (FIFO, bounded).
  ShardedQueue<int> q(1, 4);
  EXPECT_EQ(q.capacity(), 4u);
  for (int i = 0; i < 4; ++i) EXPECT_TRUE(q.try_push(0, i));
  EXPECT_FALSE(q.try_push(0, 99));  // full
  std::size_t cur = 0;
  int v = 0;
  for (int i = 0; i < 4; ++i) {
    ASSERT_TRUE(q.try_pop(cur, v));
    EXPECT_EQ(v, i);  // FIFO
  }
  EXPECT_FALSE(q.try_pop(cur, v));  // empty
}

TEST(ShardedQueue, PropagatesMoveOnlyPlacementAndCallbackConsumption) {
  EXPECT_EQ(ShardedMoveOnly::alive, 0);
  {
    ShardedQueue<ShardedMoveOnly> queue(2, 2);
    ASSERT_TRUE(queue.try_emplace(0, 10));
    ASSERT_TRUE(queue.try_emplace(1, 20));
    std::size_t cursor = 0;
    int sum = 0;
    ASSERT_TRUE(queue.try_consume(
        cursor, [&sum](ShardedMoveOnly& item) noexcept { sum += item.value; }));
    ASSERT_TRUE(queue.try_consume(
        cursor, [&sum](ShardedMoveOnly& item) noexcept { sum += item.value; }));
    EXPECT_EQ(sum, 30);
    EXPECT_EQ(ShardedMoveOnly::alive, 0);

    ASSERT_TRUE(queue.try_emplace(0, 30));
    // A live item left in any shard is owned and destroyed at quiescent teardown.
  }
  EXPECT_EQ(ShardedMoveOnly::alive, 0);
}

TEST(ShardedQueue, MoveAssignablePayloadUsesNaturalPopShape) {
  ShardedQueue<std::unique_ptr<int>> queue(2, 2);
  ASSERT_TRUE(queue.try_push(1, std::make_unique<int>(42)));
  std::size_t cursor = 0;
  std::unique_ptr<int> output;
  ASSERT_TRUE(queue.try_pop(cursor, output));
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(*output, 42);
}

// ---- MPMC conservation stress (the lock-free correctness gate) --------------
// Each producer pushes a disjoint id range INTO ITS OWN shard (hint = producer
// index); consumers round-robin-drain. The union of popped values must equal
// exactly the pushed set -- no loss, no duplication. Mirrors the BoundedQueue
// stress test but exercises the sharded fan-in/fan-out. Uses the OPT-IN runtime
// mem::Allocator variant (covers the scheduler's global-inject-queue path).
TEST(ShardedQueue, MpmcStressConservation) {
  constexpr int kProducers = 4;
  constexpr int kConsumers = 4;
  constexpr int kPerProducer = 100000;
  constexpr int kTotal = kProducers * kPerProducer;

  // One shard per producer so producers never contend; cap small to force lots
  // of wrap-around + full/empty races between producers and consumers.
  ShardedQueue<std::uint64_t, Allocator> q(system_allocator(), kProducers, 1024);
  std::atomic<int> producers_done{0};
  std::vector<std::vector<std::uint64_t>> got(kConsumers);

  struct ProdArgs {
    ShardedQueue<std::uint64_t, Allocator>* q;
    int hint;
    int base;
    std::atomic<int>* done;
  };
  struct ConsArgs {
    ShardedQueue<std::uint64_t, Allocator>* q;
    std::vector<std::uint64_t>* out;
    std::atomic<int>* done_producers;
    int start_cursor;
  };

  std::vector<ProdArgs> pa(kProducers);
  std::vector<ConsArgs> ca(kConsumers);
  std::vector<hyle::platform::Thread> threads;

  for (int i = 0; i < kProducers; ++i) {
    pa[i] = ProdArgs{&q, i, 1 + i * kPerProducer, &producers_done};
    auto r = hyle::platform::Thread::spawn(
        [](void* a) {
          auto* x = static_cast<ProdArgs*>(a);
          for (int k = 0; k < kPerProducer; ++k) {
            std::uint64_t id = static_cast<std::uint64_t>(x->base + k);
            while (!x->q->try_push(static_cast<std::size_t>(x->hint), id))
              hyle::platform::this_thread::yield();
          }
          x->done->fetch_add(1, std::memory_order_acq_rel);
        },
        &pa[i]);
    ASSERT_TRUE(r.has_value());
    threads.push_back(std::move(*r));
  }

  for (int i = 0; i < kConsumers; ++i) {
    got[i].reserve(kTotal / kConsumers + kPerProducer);
    ca[i] = ConsArgs{&q, &got[i], &producers_done, i};
    auto r = hyle::platform::Thread::spawn(
        [](void* a) {
          auto* x = static_cast<ConsArgs*>(a);
          std::size_t cursor = static_cast<std::size_t>(x->start_cursor);
          for (;;) {
            std::uint64_t v = 0;
            if (x->q->try_pop(cursor, v)) {
              x->out->push_back(v);
            } else if (x->done_producers->load(std::memory_order_acquire) ==
                       kProducers) {
              break;  // all produced and every shard read empty -> drained
            } else {
              hyle::platform::this_thread::yield();
            }
          }
        },
        &ca[i]);
    ASSERT_TRUE(r.has_value());
    threads.push_back(std::move(*r));
  }

  for (auto& t : threads) t.join();

  // Conservation: every produced id appears exactly once across all consumers.
  std::unordered_set<std::uint64_t> seen;
  seen.reserve(kTotal * 2);
  std::size_t count = 0;
  for (auto& v : got) {
    for (std::uint64_t id : v) {
      EXPECT_TRUE(seen.insert(id).second) << "duplicate id " << id;
      ++count;
    }
  }
  EXPECT_EQ(count, static_cast<std::size_t>(kTotal));
  EXPECT_EQ(seen.size(), static_cast<std::size_t>(kTotal));
}
