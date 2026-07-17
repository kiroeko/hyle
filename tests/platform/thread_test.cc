#include <gtest/gtest.h>

#include <atomic>

#include "hyle/platform/thread.h"

using hyle::platform::Thread;
using hyle::platform::ThreadOpts;
using hyle::platform::ThreadPriority;
namespace this_thread = hyle::platform::this_thread;

TEST(Thread, SpawnRunsAndJoins) {
  std::atomic<int> ran{0};
  auto r = Thread::spawn([](void* a) { static_cast<std::atomic<int>*>(a)->store(
                             42, std::memory_order_release); },
                         &ran);
  ASSERT_TRUE(r.has_value());
  Thread t = std::move(*r);
  EXPECT_TRUE(t.joinable());
  t.join();
  EXPECT_FALSE(t.joinable());
  EXPECT_EQ(ran.load(std::memory_order_acquire), 42);
}

TEST(Thread, SpawnWithNameAndAffinity) {
  // name + affinity are best-effort; the contract here is "applies without
  // crashing and the thread still runs". (Value verification needs OS probing.)
  std::atomic<bool> ran{false};
  ThreadOpts opts;
  opts.name = "hyle-worker";
  opts.affinity_cpu = 0u;
  auto r = Thread::spawn(
      [](void* a) { static_cast<std::atomic<bool>*>(a)->store(true); }, &ran,
      opts);
  ASSERT_TRUE(r.has_value());
  std::move(*r).join();
  EXPECT_TRUE(ran.load());
}

TEST(Thread, MoveTransfersOwnership) {
  std::atomic<int> ran{0};
  auto r = Thread::spawn(
      [](void* a) { static_cast<std::atomic<int>*>(a)->fetch_add(1); }, &ran);
  ASSERT_TRUE(r.has_value());
  Thread a = std::move(*r);
  Thread b = std::move(a);  // a is now empty, b owns the thread
  EXPECT_FALSE(a.joinable());
  EXPECT_TRUE(b.joinable());
  b.join();
  EXPECT_EQ(ran.load(), 1);
}

TEST(Thread, HardwareConcurrencyIsPositive) {
  EXPECT_GE(hyle::platform::hardware_concurrency(), 1u);
}

TEST(ThisThread, IdAndHelpers) {
  EXPECT_NE(this_thread::get_id(), 0u);
  this_thread::yield();             // must not crash
  this_thread::set_name("hyle-main");  // best-effort, must not crash
  SUCCEED();
}

TEST(ThisThread, DistinctThreadsHaveDistinctIds) {
  std::atomic<std::uint64_t> child_id{0};
  auto r = Thread::spawn(
      [](void* a) {
        static_cast<std::atomic<std::uint64_t>*>(a)->store(
            this_thread::get_id());
      },
      &child_id);
  ASSERT_TRUE(r.has_value());
  std::move(*r).join();
  EXPECT_NE(child_id.load(), 0u);
  EXPECT_NE(child_id.load(), this_thread::get_id());
}

TEST(Thread, DefaultConstructedNotJoinable) {
  Thread t;
  EXPECT_FALSE(t.joinable());
}

TEST(Thread, SpawnWithStackSizeRuns) {
  // Contract: stack_size is honored by the OS attrs path and does not prevent
  // the thread from running. We do not probe the actual stack limit.
  std::atomic<int> ran{0};
  ThreadOpts opts;
  opts.stack_size = 2u * 1024u * 1024u;  // 2 MiB
  auto r = Thread::spawn(
      [](void* a) {
        static_cast<std::atomic<int>*>(a)->store(7, std::memory_order_release);
      },
      &ran, opts);
  ASSERT_TRUE(r.has_value()) << "spawn with stack_size failed";
  std::move(*r).join();
  EXPECT_EQ(ran.load(std::memory_order_acquire), 7);
}

TEST(Thread, SpawnWithNameOnly) {
  std::atomic<bool> ran{false};
  ThreadOpts opts;
  opts.name = "hyle-named";
  auto r = Thread::spawn(
      [](void* a) { static_cast<std::atomic<bool>*>(a)->store(true); }, &ran,
      opts);
  ASSERT_TRUE(r.has_value());
  std::move(*r).join();
  EXPECT_TRUE(ran.load());
}

TEST(Thread, AffinityOutOfRangeDoesNotCrash) {
  // Best-effort: absurd CPU index must not abort spawn/join.
  std::atomic<bool> ran{false};
  ThreadOpts opts;
  opts.affinity_cpu = 4096u;
  auto r = Thread::spawn(
      [](void* a) { static_cast<std::atomic<bool>*>(a)->store(true); }, &ran,
      opts);
  ASSERT_TRUE(r.has_value());
  std::move(*r).join();
  EXPECT_TRUE(ran.load());
}

TEST(Thread, PriorityBestEffortRuns) {
  // Priority is best-effort (raising needs privilege on POSIX). Every level must
  // spawn + run + join cleanly regardless of whether the OS actually grants it.
  for (ThreadPriority p : {ThreadPriority::kLowest, ThreadPriority::kLow,
                           ThreadPriority::kNormal, ThreadPriority::kHigh,
                           ThreadPriority::kHighest}) {
    std::atomic<bool> ran{false};
    ThreadOpts opts;
    opts.priority = p;
    auto r = Thread::spawn(
        [](void* a) { static_cast<std::atomic<bool>*>(a)->store(true); }, &ran,
        opts);
    ASSERT_TRUE(r.has_value());
    std::move(*r).join();
    EXPECT_TRUE(ran.load());
  }
}

TEST(Thread, ManyThreadsJoinClean) {
  constexpr int kN = 32;
  std::atomic<int> count{0};
  Thread threads[kN];
  for (int i = 0; i < kN; ++i) {
    auto r = Thread::spawn(
        [](void* a) {
          static_cast<std::atomic<int>*>(a)->fetch_add(1,
                                                       std::memory_order_acq_rel);
        },
        &count);
    ASSERT_TRUE(r.has_value());
    threads[i] = std::move(*r);
  }
  for (int i = 0; i < kN; ++i) threads[i].join();
  EXPECT_EQ(count.load(), kN);
}

TEST(Thread, MoveAssignIntoEmpty) {
  std::atomic<int> ran{0};
  auto r = Thread::spawn(
      [](void* a) { static_cast<std::atomic<int>*>(a)->store(1); }, &ran);
  ASSERT_TRUE(r.has_value());
  Thread empty;
  empty = std::move(*r);  // empty is not joinable -> allowed
  EXPECT_TRUE(empty.joinable());
  empty.join();
  EXPECT_EQ(ran.load(), 1);
}
