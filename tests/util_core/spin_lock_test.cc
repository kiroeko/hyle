#include <gtest/gtest.h>

#include <mutex>
#include <thread>
#include <vector>

#include "hyle/util/spin_lock.h"

using hyle::util::SpinLock;

static_assert(sizeof(SpinLock) <= sizeof(void*),
              "ordinary SpinLock must remain compact; pad at the owner");

TEST(SpinLock, BasicLockUnlock) {
  SpinLock m;
  EXPECT_FALSE(m.is_locked());
  m.lock();
  EXPECT_TRUE(m.is_locked());
  m.unlock();
  EXPECT_FALSE(m.is_locked());
}

TEST(SpinLock, TryLock) {
  SpinLock m;
  EXPECT_TRUE(m.try_lock());   // free -> acquired
  EXPECT_FALSE(m.try_lock());  // already held -> fails
  m.unlock();
  EXPECT_TRUE(m.try_lock());   // free again
  m.unlock();
}

TEST(SpinLock, ComposesWithLockGuard) {
  SpinLock m;
  {
    std::lock_guard<SpinLock> g(m);  // Lockable -> works with std RAII
    EXPECT_TRUE(m.is_locked());
  }
  EXPECT_FALSE(m.is_locked());
}

// The correctness gate: mutual exclusion. Many threads increment a shared
// counter under the lock; with real exclusion the total is exact. A broken lock
// drops updates. (Also the WSL TSan data-race gate.)
TEST(SpinLock, MutualExclusionUnderContention) {
  SpinLock m;
  std::uint64_t counter = 0;
  constexpr int kThreads = 8;
  constexpr int kIters = 100000;
  std::vector<std::thread> ts;
  for (int t = 0; t < kThreads; ++t) {
    ts.emplace_back([&] {
      for (int i = 0; i < kIters; ++i) {
        std::lock_guard<SpinLock> g(m);
        ++counter;  // protected by the spinlock
      }
    });
  }
  for (auto& th : ts) th.join();
  EXPECT_EQ(counter, static_cast<std::uint64_t>(kThreads) * kIters);
}

// try_lock from many threads: exactly the holders that succeed may enter; the
// critical region is never concurrently occupied.
TEST(SpinLock, TryLockExclusion) {
  SpinLock m;
  std::atomic<int> in_section{0};
  std::atomic<int> max_seen{0};
  constexpr int kThreads = 8;
  std::vector<std::thread> ts;
  for (int t = 0; t < kThreads; ++t) {
    ts.emplace_back([&] {
      for (int i = 0; i < 50000; ++i) {
        if (m.try_lock()) {
          int cur = ++in_section;
          int prev = max_seen.load();
          while (cur > prev && !max_seen.compare_exchange_weak(prev, cur)) {}
          --in_section;
          m.unlock();
        }
      }
    });
  }
  for (auto& th : ts) th.join();
  EXPECT_EQ(max_seen.load(), 1);  // never more than one holder inside
}
