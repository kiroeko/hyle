#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <thread>
#include <unordered_set>
#include <vector>

#include "hyle/container/ws_fifo_queue.h"
#include "hyle/mem/allocator.h"

using hyle::container::WsFifoQueue;
using hyle::mem::AllocatorRef;
using hyle::mem::system_allocator;

// ---- single-thread owner FIFO -----------------------------------------------

TEST(WsFifoQueue, OwnerFifoPushPop) {
  WsFifoQueue<int> q(8);  // default = compile-time SystemAllocator
  EXPECT_TRUE(q.try_push_back(10));
  EXPECT_TRUE(q.try_push_back(20));
  EXPECT_TRUE(q.try_push_back(30));
  // FIFO: pop returns oldest first (unlike the Chase-Lev deque's LIFO)
  int value = -1;
  ASSERT_TRUE(q.try_pop_front(value));
  EXPECT_EQ(value, 10);
  ASSERT_TRUE(q.try_pop_front(value));
  EXPECT_EQ(value, 20);
  ASSERT_TRUE(q.try_pop_front(value));
  EXPECT_EQ(value, 30);
  EXPECT_FALSE(q.try_pop_front(value));
  EXPECT_EQ(value, 30);
}

TEST(WsFifoQueue, FullPushRejected) {
  WsFifoQueue<int> q(2);
  EXPECT_TRUE(q.try_push_back(1));
  EXPECT_TRUE(q.try_push_back(2));
  EXPECT_FALSE(q.try_push_back(3));  // capacity 2 -> full
}

TEST(WsFifoQueue, WrapsAroundManyLaps) {
  WsFifoQueue<int> q(4);
  for (int i = 0; i < 1000; ++i) {
    ASSERT_TRUE(q.try_push_back(i));
    int value = -1;
    ASSERT_TRUE(q.try_pop_front(value));
    EXPECT_EQ(value, i);
  }
}

// ---- try_steal_into semantics (single-thread, deterministic) ----------------

TEST(WsFifoQueue, StealMovesAboutHalfOldestFirst) {
  WsFifoQueue<int> victim(16);
  WsFifoQueue<int> thief(16);
  for (int i = 0; i < 8; ++i) victim.try_push_back(i);  // 0..7

  // Steal ~half (4): one returned directly, rest land in `thief`, oldest first.
  int value = -1;
  ASSERT_TRUE(victim.try_steal_into(thief, value));
  // tokio takes ceil(8/2)=4; returns the LAST of the batch, leaves first 3 in thief.
  // The batch is the oldest 4 (0,1,2,3); returned = index dst_tail+(n-1).
  EXPECT_EQ(value, 3);
  // thief now holds 0,1,2 (FIFO)
  ASSERT_TRUE(thief.try_pop_front(value));
  EXPECT_EQ(value, 0);
  ASSERT_TRUE(thief.try_pop_front(value));
  EXPECT_EQ(value, 1);
  ASSERT_TRUE(thief.try_pop_front(value));
  EXPECT_EQ(value, 2);
  EXPECT_FALSE(thief.try_pop_front(value));
  // victim retains the newer half (4,5,6,7)
  ASSERT_TRUE(victim.try_pop_front(value));
  EXPECT_EQ(value, 4);
}

TEST(WsFifoQueue, StealFromEmptyReturnsNotOk) {
  WsFifoQueue<int> victim(8);
  WsFifoQueue<int> thief(8);
  int value = 99;
  EXPECT_FALSE(victim.try_steal_into(thief, value));
  EXPECT_EQ(value, 99);
}

// ---- concurrent conservation: 1 owner + N thieves (each own dst) ------------
// Every pushed id is consumed EXACTLY once -- by the owner's pop or by a thief
// (directly or via its own dst queue). No loss, no duplication. TSan in WSL is
// the real gate; unlike the Chase-Lev deque this uses CAS (not standalone
// fences), so TSan coverage here is sound (no fence blind spot).
TEST(WsFifoQueue, ConcurrentConservation) {
  constexpr std::uint64_t kN = 200000;
  constexpr int kThieves = 4;
  WsFifoQueue<std::uint64_t, AllocatorRef> q(system_allocator(), 256);

  std::atomic<bool> done_pushing{false};
  std::vector<std::uint64_t> owner_got;
  std::vector<std::vector<std::uint64_t>> thief_got(kThieves);

  struct OwnerArgs {
    WsFifoQueue<std::uint64_t, AllocatorRef>* q;
    std::atomic<bool>* done;
    std::vector<std::uint64_t>* got;
  } oargs{&q, &done_pushing, &owner_got};

  struct ThiefArgs {
    WsFifoQueue<std::uint64_t, AllocatorRef>* victim;
    std::atomic<bool>* done;
    std::vector<std::uint64_t>* got;
  };
  std::vector<ThiefArgs> targs(kThieves);
  std::vector<std::thread> threads;

  threads.emplace_back(
      [](OwnerArgs* a) {
        for (std::uint64_t id = 1; id <= kN; ++id) {
          while (!a->q->try_push_back(id)) {
            std::uint64_t value = 0;
            if (a->q->try_pop_front(value))  // full: drain own end
              a->got->push_back(value);
          }
        }
        a->done->store(true, std::memory_order_release);
        for (;;) {  // drain remaining
          std::uint64_t value = 0;
          if (a->q->try_pop_front(value)) {
            a->got->push_back(value);
          } else {
            break;
          }
        }
      },
      &oargs);

  for (int i = 0; i < kThieves; ++i) {
    targs[i] = ThiefArgs{&q, &done_pushing, &thief_got[i]};
    threads.emplace_back(
        [](ThiefArgs* a) {
          WsFifoQueue<std::uint64_t, AllocatorRef> local(system_allocator(), 256);
          // Record a successful steal: the directly-returned task + everything
          // it dumped into our local queue. (One helper so no return value is
          // ever dropped -- that would silently lose an id.)
          auto absorb_steal = [&](std::uint64_t first) {
            a->got->push_back(first);
            std::uint64_t value = 0;
            while (local.try_pop_front(value)) a->got->push_back(value);
          };
          for (;;) {
            std::uint64_t value = 0;
            if (a->victim->try_steal_into(local, value)) {
              absorb_steal(value);
            } else if (a->done->load(std::memory_order_acquire)) {
              // empty/contended AND owner done producing -> a few more tries,
              // then exit if still nothing.
              bool more = false;
              for (int k = 0; k < 64; ++k) {
                if (a->victim->try_steal_into(local, value)) {
                  absorb_steal(value);
                  more = true;
                  break;
                }
              }
              if (!more) break;
            } else {
              std::this_thread::yield();
            }
          }
        },
        &targs[i]);
  }

  for (auto& t : threads) t.join();

  std::unordered_set<std::uint64_t> seen;
  seen.reserve(kN * 2);
  std::size_t total = 0;
  auto absorb = [&](const std::vector<std::uint64_t>& v) {
    for (std::uint64_t id : v) {
      EXPECT_TRUE(seen.insert(id).second) << "duplicate id " << id;
      ++total;
    }
  };
  absorb(owner_got);
  for (auto& v : thief_got) absorb(v);
  EXPECT_EQ(total, kN) << "lost or extra items";
  EXPECT_EQ(seen.size(), kN);
}
