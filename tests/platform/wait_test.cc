#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "hyle/platform/thread.h"
#include "hyle/platform/wait.h"

using hyle::platform::Parker;
using hyle::platform::Thread;
using hyle::platform::ThreadOpts;

// ---- Parker contract --------------------------------------------------------

TEST(Parker, UnparkBeforeParkDoesNotBlock) {
  // The token must be remembered: an unpark that lands before park() means the
  // following park() returns immediately rather than sleeping forever.
  Parker p;
  p.unpark();
  p.park();  // would hang if the token were lost
  SUCCEED();
}

TEST(Parker, RepeatedUnparkCollapsesToOneToken) {
  Parker p;
  p.unpark();
  p.unpark();  // idempotent: still a single token
  p.park();    // consumes it, returns
  SUCCEED();
}

TEST(Parker, ParkWakesFromAnotherThread) {
  Parker p;
  std::atomic<bool> parked_returned{false};

  void* ctx = nullptr;
  struct Args {
    Parker* p;
    std::atomic<bool>* flag;
  } args{&p, &parked_returned};
  ctx = &args;

  auto r = Thread::spawn(
      [](void* a) {
        auto* x = static_cast<Args*>(a);
        x->p->park();
        x->flag->store(true, std::memory_order_release);
      },
      ctx);
  ASSERT_TRUE(r.has_value());
  Thread t = std::move(*r);

  // Give the worker time to actually park, then wake it.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_FALSE(parked_returned.load(std::memory_order_acquire));
  p.unpark();
  t.join();
  EXPECT_TRUE(parked_returned.load(std::memory_order_acquire));
}

// Key concurrency gate: N distinct park<->unpark handoffs, no lost wakeup.
// Main waits for the worker to observe each unpark before issuing the next, so
// tokens never collapse -- the counter must reach exactly N.
TEST(Parker, HandoffStressNoLostWakeup) {
  constexpr int kRounds = 2000;
  Parker p;
  std::atomic<int> observed{0};

  struct Args {
    Parker* p;
    std::atomic<int>* observed;
  } args{&p, &observed};

  auto r = Thread::spawn(
      [](void* a) {
        auto* x = static_cast<Args*>(a);
        for (int i = 0; i < kRounds; ++i) {
          x->p->park();
          x->observed->fetch_add(1, std::memory_order_acq_rel);
        }
      },
      &args);
  ASSERT_TRUE(r.has_value());
  Thread t = std::move(*r);

  for (int i = 0; i < kRounds; ++i) {
    p.unpark();
    // Wait for the worker to consume this round before the next unpark.
    while (observed.load(std::memory_order_acquire) <= i) {
      hyle::platform::this_thread::yield();
    }
  }
  t.join();
  EXPECT_EQ(observed.load(), kRounds);
}

// Milestone acceptance: M workers, each with its own Parker, N rounds. Main
// unparks all M each round and waits for every worker to consume before the
// next round -- every per-worker counter must hit exactly N (no lost wakeup,
// no cross-parker interference).
TEST(Parker, MultiWorkerHandoffNoLostWakeup) {
  constexpr int kWorkers = 4;
  constexpr int kRounds = 500;
  std::array<Parker, kWorkers> parkers;        // non-movable: in-place
  std::array<std::atomic<int>, kWorkers> observed{};

  struct Args {
    Parker* p;
    std::atomic<int>* obs;
  };
  std::array<Args, kWorkers> args;
  std::vector<Thread> workers;
  for (int i = 0; i < kWorkers; ++i) {
    args[i] = Args{&parkers[i], &observed[i]};
    auto r = Thread::spawn(
        [](void* a) {
          auto* x = static_cast<Args*>(a);
          for (int k = 0; k < kRounds; ++k) {
            x->p->park();
            x->obs->fetch_add(1, std::memory_order_acq_rel);
          }
        },
        &args[i]);
    ASSERT_TRUE(r.has_value());
    workers.push_back(std::move(*r));
  }

  for (int k = 0; k < kRounds; ++k) {
    for (int i = 0; i < kWorkers; ++i) parkers[i].unpark();
    for (int i = 0; i < kWorkers; ++i) {
      while (observed[i].load(std::memory_order_acquire) <= k) {
        hyle::platform::this_thread::yield();
      }
    }
  }
  for (auto& t : workers) t.join();
  for (int i = 0; i < kWorkers; ++i) EXPECT_EQ(observed[i].load(), kRounds);
}
