#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include "hyle/container/ws_fifo_queue.h"
#include "hyle/mem/allocator.h"
#include "hyle/platform/thread.h"

using hyle::container::WsFifoQueue;
using hyle::mem::system_allocator;

// ---- single-thread owner FIFO -----------------------------------------------

TEST(WsFifoQueue, OwnerFifoPushPop) {
  WsFifoQueue<int> q(system_allocator(), 8);
  EXPECT_TRUE(q.push_back(10));
  EXPECT_TRUE(q.push_back(20));
  EXPECT_TRUE(q.push_back(30));
  // FIFO: pop returns oldest first (unlike the Chase-Lev deque's LIFO)
  auto a = q.pop();
  ASSERT_TRUE(a.ok);
  EXPECT_EQ(a.value, 10);
  auto b = q.pop();
  ASSERT_TRUE(b.ok);
  EXPECT_EQ(b.value, 20);
  auto c = q.pop();
  ASSERT_TRUE(c.ok);
  EXPECT_EQ(c.value, 30);
  EXPECT_FALSE(q.pop().ok);
}

TEST(WsFifoQueue, FullPushRejected) {
  WsFifoQueue<int> q(system_allocator(), 2);
  EXPECT_TRUE(q.push_back(1));
  EXPECT_TRUE(q.push_back(2));
  EXPECT_FALSE(q.push_back(3));  // capacity 2 -> full
}

TEST(WsFifoQueue, WrapsAroundManyLaps) {
  WsFifoQueue<int> q(system_allocator(), 4);
  for (int i = 0; i < 1000; ++i) {
    ASSERT_TRUE(q.push_back(i));
    auto r = q.pop();
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.value, i);
  }
}

// ---- steal_into semantics (single-thread, deterministic) --------------------

TEST(WsFifoQueue, StealMovesAboutHalfOldestFirst) {
  WsFifoQueue<int> victim(system_allocator(), 16);
  WsFifoQueue<int> thief(system_allocator(), 16);
  for (int i = 0; i < 8; ++i) victim.push_back(i);  // 0..7

  // Steal ~half (4): one returned directly, rest land in `thief`, oldest first.
  auto r = victim.steal_into(thief);
  ASSERT_TRUE(r.ok);
  // tokio takes ceil(8/2)=4; returns the LAST of the batch, leaves first 3 in thief.
  // The batch is the oldest 4 (0,1,2,3); returned = index dst_tail+(n-1).
  EXPECT_EQ(r.value, 3);
  // thief now holds 0,1,2 (FIFO)
  EXPECT_EQ(thief.pop().value, 0);
  EXPECT_EQ(thief.pop().value, 1);
  EXPECT_EQ(thief.pop().value, 2);
  EXPECT_FALSE(thief.pop().ok);
  // victim retains the newer half (4,5,6,7)
  EXPECT_EQ(victim.pop().value, 4);
}

TEST(WsFifoQueue, StealFromEmptyReturnsNotOk) {
  WsFifoQueue<int> victim(system_allocator(), 8);
  WsFifoQueue<int> thief(system_allocator(), 8);
  EXPECT_FALSE(victim.steal_into(thief).ok);
}

// ---- concurrent conservation: 1 owner + N thieves (each own dst) ------------
// Every pushed id is consumed EXACTLY once -- by the owner's pop or by a thief
// (directly or via its own dst queue). No loss, no duplication. TSan in WSL is
// the real gate; unlike the Chase-Lev deque this uses CAS (not standalone
// fences), so TSan coverage here is sound (no fence blind spot).
TEST(WsFifoQueue, ConcurrentConservation) {
  constexpr std::uint64_t kN = 200000;
  constexpr int kThieves = 4;
  WsFifoQueue<std::uint64_t> q(system_allocator(), 256);

  std::atomic<bool> done_pushing{false};
  std::vector<std::uint64_t> owner_got;
  std::vector<std::vector<std::uint64_t>> thief_got(kThieves);

  struct OwnerArgs {
    WsFifoQueue<std::uint64_t>* q;
    std::atomic<bool>* done;
    std::vector<std::uint64_t>* got;
  } oargs{&q, &done_pushing, &owner_got};

  struct ThiefArgs {
    WsFifoQueue<std::uint64_t>* victim;
    std::atomic<bool>* done;
    std::vector<std::uint64_t>* got;
  };
  std::vector<ThiefArgs> targs(kThieves);
  std::vector<hyle::platform::Thread> threads;

  auto owner_r = hyle::platform::Thread::spawn(
      [](void* p) {
        auto* a = static_cast<OwnerArgs*>(p);
        for (std::uint64_t id = 1; id <= kN; ++id) {
          while (!a->q->push_back(id)) {
            auto r = a->q->pop();  // full: drain own end to make room
            if (r.ok) a->got->push_back(r.value);
          }
        }
        a->done->store(true, std::memory_order_release);
        for (;;) {  // drain remaining
          auto r = a->q->pop();
          if (r.ok) {
            a->got->push_back(r.value);
          } else {
            break;
          }
        }
      },
      &oargs);
  ASSERT_TRUE(owner_r.has_value());
  threads.push_back(std::move(*owner_r));

  for (int i = 0; i < kThieves; ++i) {
    targs[i] = ThiefArgs{&q, &done_pushing, &thief_got[i]};
    auto r = hyle::platform::Thread::spawn(
        [](void* p) {
          auto* a = static_cast<ThiefArgs*>(p);
          WsFifoQueue<std::uint64_t> local(system_allocator(), 256);
          // Record a successful steal: the directly-returned task + everything
          // it dumped into our local queue. (One helper so no return value is
          // ever dropped -- that would silently lose an id.)
          auto absorb_steal = [&](std::uint64_t first) {
            a->got->push_back(first);
            for (auto r = local.pop(); r.ok; r = local.pop()) {
              a->got->push_back(r.value);
            }
          };
          for (;;) {
            auto s = a->victim->steal_into(local);
            if (s.ok) {
              absorb_steal(s.value);
            } else if (a->done->load(std::memory_order_acquire)) {
              // empty/contended AND owner done producing -> a few more tries,
              // then exit if still nothing.
              bool more = false;
              for (int k = 0; k < 64; ++k) {
                auto s2 = a->victim->steal_into(local);
                if (s2.ok) {
                  absorb_steal(s2.value);
                  more = true;
                  break;
                }
              }
              if (!more) break;
            } else {
              hyle::platform::this_thread::yield();
            }
          }
        },
        &targs[i]);
    ASSERT_TRUE(r.has_value());
    threads.push_back(std::move(*r));
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
