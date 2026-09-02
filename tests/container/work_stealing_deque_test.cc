#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include "hyle/container/work_stealing_deque.h"
#include "hyle/mem/allocator.h"
#include "hyle/platform/thread.h"

using hyle::container::WorkStealingDeque;
using hyle::mem::AllocatorRef;
using hyle::mem::system_allocator;
using Status = hyle::container::StealStatus;

// ---- single-thread owner semantics ------------------------------------------
TEST(WorkStealingDeque, OwnerLifoPushPop) {
  WorkStealingDeque<int> d(8);
  EXPECT_TRUE(d.try_push_bottom(10));
  EXPECT_TRUE(d.try_push_bottom(20));
  EXPECT_TRUE(d.try_push_bottom(30));
  // owner pops its own end LIFO
  int value = -1;
  ASSERT_TRUE(d.try_pop_bottom(value));
  EXPECT_EQ(value, 30);
  ASSERT_TRUE(d.try_pop_bottom(value));
  EXPECT_EQ(value, 20);
  ASSERT_TRUE(d.try_pop_bottom(value));
  EXPECT_EQ(value, 10);
}

TEST(WorkStealingDeque, EmptyPopReturnsNotOk) {
  WorkStealingDeque<int> d(8);
  int value = 99;
  EXPECT_FALSE(d.try_pop_bottom(value));
  EXPECT_EQ(value, 99);
  // and stays usable afterwards (bottom was restored correctly)
  EXPECT_TRUE(d.try_push_bottom(1));
  ASSERT_TRUE(d.try_pop_bottom(value));
  EXPECT_EQ(value, 1);
}

TEST(WorkStealingDeque, FullPushRejected) {
  WorkStealingDeque<int> d(2);
  EXPECT_TRUE(d.try_push_bottom(1));
  EXPECT_TRUE(d.try_push_bottom(2));
  EXPECT_FALSE(d.try_push_bottom(3));  // capacity 2 -> full
}

TEST(WorkStealingDeque, StealTakesOldestThenOwnerNewest) {
  WorkStealingDeque<int> d(8);
  d.try_push_bottom(10);
  d.try_push_bottom(20);
  d.try_push_bottom(30);
  int value = -1;
  ASSERT_EQ(d.try_steal_top(value), Status::kOk);  // oldest
  EXPECT_EQ(value, 10);
  ASSERT_TRUE(d.try_pop_bottom(value));  // owner end takes the newest
  EXPECT_EQ(value, 30);
}

TEST(WorkStealingDeque, StealEmptyReportsEmpty) {
  WorkStealingDeque<int> d(8);
  int value = 99;
  EXPECT_EQ(d.try_steal_top(value), Status::kEmpty);
  EXPECT_EQ(value, 99);
}

TEST(WorkStealingDeque, WrapsAroundManyLaps) {
  WorkStealingDeque<int> d(4);
  for (int i = 0; i < 1000; ++i) {
    ASSERT_TRUE(d.try_push_bottom(i));
    int value = -1;
    ASSERT_TRUE(d.try_pop_bottom(value));
    EXPECT_EQ(value, i);
  }
}

// ---- concurrent conservation: 1 owner + N thieves ---------------------------
// Every pushed id is taken EXACTLY once -- by the owner's pop or by one thief.
// No loss, no duplication. This is the real correctness gate (run under TSan in
// WSL; on x64 strong-ordering can hide bugs that TSan catches).
TEST(WorkStealingDeque, ConcurrentConservation) {
  constexpr std::uint64_t kN = 200000;
  constexpr int kThieves = 4;
  WorkStealingDeque<std::uint64_t, AllocatorRef> d(system_allocator(), 1024);

  std::atomic<bool> done_pushing{false};
  std::vector<std::uint64_t> owner_got;
  std::vector<std::vector<std::uint64_t>> thief_got(kThieves);

  struct OwnerArgs {
    WorkStealingDeque<std::uint64_t, AllocatorRef>* d;
    std::atomic<bool>* done;
    std::vector<std::uint64_t>* got;
  } oargs{&d, &done_pushing, &owner_got};

  struct ThiefArgs {
    WorkStealingDeque<std::uint64_t, AllocatorRef>* d;
    std::atomic<bool>* done;
    std::vector<std::uint64_t>* got;
  };
  std::vector<ThiefArgs> targs(kThieves);
  std::vector<hyle::platform::Thread> threads;

  // Owner: push 1..kN; when full, pop one (record it) to make room. Then drain.
  auto owner_r = hyle::platform::Thread::spawn(
      [](void* p) {
        auto* a = static_cast<OwnerArgs*>(p);
        for (std::uint64_t id = 1; id <= kN; ++id) {
          while (!a->d->try_push_bottom(id)) {
            std::uint64_t value = 0;
            if (a->d->try_pop_bottom(value))  // reclaim from own end
              a->got->push_back(value);
          }
        }
        a->done->store(true, std::memory_order_release);
        for (;;) {  // drain remaining owner-side
          std::uint64_t value = 0;
          if (a->d->try_pop_bottom(value)) {
            a->got->push_back(value);
          } else {
            break;
          }
        }
      },
      &oargs);
  ASSERT_TRUE(owner_r.has_value());
  threads.push_back(std::move(*owner_r));

  for (int i = 0; i < kThieves; ++i) {
    targs[i] = ThiefArgs{&d, &done_pushing, &thief_got[i]};
    auto r = hyle::platform::Thread::spawn(
        [](void* p) {
          auto* a = static_cast<ThiefArgs*>(p);
          for (;;) {
            std::uint64_t value = 0;
            const Status status = a->d->try_steal_top(value);
            if (status == Status::kOk) {
              a->got->push_back(value);
            } else if (status == Status::kEmpty &&
                       a->done->load(std::memory_order_acquire)) {
              // Empty AND owner finished producing: try a few more in case the
              // owner is mid-drain, then exit.
              bool more = false;
              for (int k = 0; k < 64; ++k) {
                if (a->d->try_steal_top(value) == Status::kOk) {
                  a->got->push_back(value);
                  more = true;
                  break;
                }
              }
              if (!more) break;
            } else {
              hyle::platform::this_thread::yield();  // kRetry or transient empty
            }
          }
        },
        &targs[i]);
    ASSERT_TRUE(r.has_value());
    threads.push_back(std::move(*r));
  }

  for (auto& t : threads) t.join();

  // Conservation: union == {1..kN}, each exactly once.
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
