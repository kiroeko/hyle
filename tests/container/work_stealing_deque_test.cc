#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include "hyle/container/work_stealing_deque.h"
#include "hyle/mem/allocator.h"
#include "hyle/platform/thread.h"

using hyle::container::WorkStealingDeque;
using hyle::mem::Allocator;
using hyle::mem::system_allocator;
using Status = hyle::container::StealStatus;

// ---- single-thread owner semantics ------------------------------------------

TEST(WorkStealingDeque, OwnerLifoPushPop) {
  WorkStealingDeque<int> d(8);
  EXPECT_TRUE(d.push_bottom(10));
  EXPECT_TRUE(d.push_bottom(20));
  EXPECT_TRUE(d.push_bottom(30));
  // owner pops its own end LIFO
  auto a = d.pop_bottom();
  ASSERT_TRUE(a.ok);
  EXPECT_EQ(a.value, 30);
  auto b = d.pop_bottom();
  ASSERT_TRUE(b.ok);
  EXPECT_EQ(b.value, 20);
  auto c = d.pop_bottom();
  ASSERT_TRUE(c.ok);
  EXPECT_EQ(c.value, 10);
}

TEST(WorkStealingDeque, EmptyPopReturnsNotOk) {
  WorkStealingDeque<int> d(8);
  EXPECT_FALSE(d.pop_bottom().ok);
  // and stays usable afterwards (bottom was restored correctly)
  EXPECT_TRUE(d.push_bottom(1));
  auto r = d.pop_bottom();
  ASSERT_TRUE(r.ok);
  EXPECT_EQ(r.value, 1);
}

TEST(WorkStealingDeque, FullPushRejected) {
  WorkStealingDeque<int> d(2);
  EXPECT_TRUE(d.push_bottom(1));
  EXPECT_TRUE(d.push_bottom(2));
  EXPECT_FALSE(d.push_bottom(3));  // capacity 2 -> full
}

TEST(WorkStealingDeque, StealTakesOldestThenOwnerNewest) {
  WorkStealingDeque<int> d(8);
  d.push_bottom(10);
  d.push_bottom(20);
  d.push_bottom(30);
  auto s = d.steal();  // thief end takes the OLDEST
  ASSERT_EQ(s.status, Status::kOk);
  EXPECT_EQ(s.value, 10);
  auto p = d.pop_bottom();  // owner end takes the NEWEST
  ASSERT_TRUE(p.ok);
  EXPECT_EQ(p.value, 30);
}

TEST(WorkStealingDeque, StealEmptyReportsEmpty) {
  WorkStealingDeque<int> d(8);
  EXPECT_EQ(d.steal().status, Status::kEmpty);
}

TEST(WorkStealingDeque, WrapsAroundManyLaps) {
  WorkStealingDeque<int> d(4);
  for (int i = 0; i < 1000; ++i) {
    ASSERT_TRUE(d.push_bottom(i));
    auto r = d.pop_bottom();
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.value, i);
  }
}

// ---- concurrent conservation: 1 owner + N thieves ---------------------------
// Every pushed id is taken EXACTLY once -- by the owner's pop or by one thief.
// No loss, no duplication. This is the real correctness gate (run under TSan in
// WSL; on x64 strong-ordering can hide bugs that TSan catches).
TEST(WorkStealingDeque, ConcurrentConservation) {
  constexpr std::uint64_t kN = 200000;
  constexpr int kThieves = 4;
  WorkStealingDeque<std::uint64_t, Allocator> d(system_allocator(), 1024);

  std::atomic<bool> done_pushing{false};
  std::vector<std::uint64_t> owner_got;
  std::vector<std::vector<std::uint64_t>> thief_got(kThieves);

  struct OwnerArgs {
    WorkStealingDeque<std::uint64_t, Allocator>* d;
    std::atomic<bool>* done;
    std::vector<std::uint64_t>* got;
  } oargs{&d, &done_pushing, &owner_got};

  struct ThiefArgs {
    WorkStealingDeque<std::uint64_t, Allocator>* d;
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
          while (!a->d->push_bottom(id)) {
            auto r = a->d->pop_bottom();  // full: reclaim from own end
            if (r.ok) a->got->push_back(r.value);
          }
        }
        a->done->store(true, std::memory_order_release);
        for (;;) {  // drain remaining owner-side
          auto r = a->d->pop_bottom();
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
    targs[i] = ThiefArgs{&d, &done_pushing, &thief_got[i]};
    auto r = hyle::platform::Thread::spawn(
        [](void* p) {
          auto* a = static_cast<ThiefArgs*>(p);
          for (;;) {
            auto s = a->d->steal();
            if (s.status == Status::kOk) {
              a->got->push_back(s.value);
            } else if (s.status == Status::kEmpty &&
                       a->done->load(std::memory_order_acquire)) {
              // Empty AND owner finished producing: try a few more in case the
              // owner is mid-drain, then exit.
              bool more = false;
              for (int k = 0; k < 64; ++k) {
                auto s2 = a->d->steal();
                if (s2.status == Status::kOk) {
                  a->got->push_back(s2.value);
                  more = true;
                  break;
                }
              }
              if (!more) break;
            } else {
              hyle::platform::this_thread::yield();  // kAbort or transient empty
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
