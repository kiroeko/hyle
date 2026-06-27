#include <gtest/gtest.h>

#include <atomic>

#include "hyle/platform/thread.h"

using hyle::platform::Thread;
using hyle::platform::ThreadOpts;
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
