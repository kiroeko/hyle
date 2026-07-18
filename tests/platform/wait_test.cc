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

// ---- timed park (backs the timer wheel's "sleep to next deadline") ----------

TEST(Parker, ParkTimeoutReturnsFalseOnTimeout) {
  Parker p;
  auto t0 = std::chrono::steady_clock::now();
  bool got = p.park_timeout(20'000'000);  // 20ms, no unpark coming
  auto elapsed = std::chrono::steady_clock::now() - t0;
  EXPECT_FALSE(got);  // timed out, no token
  // Slept roughly the requested time (allow generous slack for CI scheduling)...
  EXPECT_GE(elapsed, std::chrono::milliseconds(10));
  // ...but bounded above: a 20ms request must not block for seconds (catches a
  // broken timeout that degrades to an ~infinite wait). Wide slack for CI.
  EXPECT_LT(elapsed, std::chrono::seconds(5));
}

TEST(Parker, ParkTimeoutConsumesExistingToken) {
  Parker p;
  p.unpark();  // token already available
  auto t0 = std::chrono::steady_clock::now();
  bool got = p.park_timeout(1'000'000'000);  // 1s deadline we should NOT wait
  auto elapsed = std::chrono::steady_clock::now() - t0;
  EXPECT_TRUE(got);  // consumed the token immediately
  EXPECT_LT(elapsed, std::chrono::milliseconds(500));  // did not block
}

TEST(Parker, ParkTimeoutZeroIsNonBlockingPoll) {
  Parker p;
  EXPECT_FALSE(p.park_timeout(0));  // no token -> immediate false
  p.unpark();
  EXPECT_TRUE(p.park_timeout(0));  // token present -> immediate true
}

TEST(Parker, ParkTimeoutWokenByUnpark) {
  Parker p;
  std::atomic<bool> returned{false};
  std::atomic<bool> got_token{false};

  struct Args {
    Parker* p;
    std::atomic<bool>* returned;
    std::atomic<bool>* got;
  } args{&p, &returned, &got_token};

  auto r = Thread::spawn(
      [](void* a) {
        auto* x = static_cast<Args*>(a);
        bool g = x->p->park_timeout(10'000'000'000ull);  // 10s -- unpark first
        x->got->store(g, std::memory_order_release);
        x->returned->store(true, std::memory_order_release);
      },
      &args);
  ASSERT_TRUE(r.has_value());
  Thread t = std::move(*r);

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_FALSE(returned.load(std::memory_order_acquire));  // still parked
  p.unpark();
  t.join();
  EXPECT_TRUE(returned.load(std::memory_order_acquire));
  EXPECT_TRUE(got_token.load(std::memory_order_acquire));  // woke with token
}
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

// ---- low-level wait_on_address + Unparker -----------------------------------

TEST(WaitOnAddress, ValueAlreadyChangedDoesNotBlock) {
  // If *addr != expected, wait returns immediately (futex/WaitOnAddress contract).
  std::atomic<std::uint32_t> word{1};
  auto t0 = std::chrono::steady_clock::now();
  hyle::platform::detail::wait_on_address(&word, /*expected=*/0);
  auto elapsed = std::chrono::steady_clock::now() - t0;
  EXPECT_LT(elapsed, std::chrono::milliseconds(200));
}

TEST(WaitOnAddress, TimedZeroIsPoll) {
  std::atomic<std::uint32_t> word{0};
  // value matches expected but timeout 0 -> poll: must return PROMPTLY, not sleep.
  auto t0 = std::chrono::steady_clock::now();
  bool r = hyle::platform::detail::wait_on_address_for(&word, 0, 0);
  auto elapsed = std::chrono::steady_clock::now() - t0;
  (void)r;  // timed out (false) or spurious wake (true) -- both fine
  // The real assertion: a zero timeout does NOT block for any meaningful time.
  EXPECT_LT(elapsed, std::chrono::milliseconds(200));
}

TEST(WaitOnAddress, WakeOneUnblocksWaiter) {
  std::atomic<std::uint32_t> word{0};
  std::atomic<bool> done{false};
  struct Args {
    std::atomic<std::uint32_t>* word;
    std::atomic<bool>* done;
  } args{&word, &done};

  auto r = Thread::spawn(
      [](void* a) {
        auto* x = static_cast<Args*>(a);
        hyle::platform::detail::wait_on_address(x->word, 0);
        x->done->store(true, std::memory_order_release);
      },
      &args);
  ASSERT_TRUE(r.has_value());
  Thread t = std::move(*r);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_FALSE(done.load(std::memory_order_acquire));
  word.store(1, std::memory_order_release);
  hyle::platform::detail::wake_one(&word);
  t.join();
  EXPECT_TRUE(done.load(std::memory_order_acquire));
}

TEST(Unparker, UnparkThroughHandle) {
  Parker p;
  hyle::platform::Unparker u(&p);
  u.unpark();
  p.park();  // must not hang
  EXPECT_EQ(u.get(), &p);
}

TEST(Parker, TimeoutThenNeedsFreshUnpark) {
  // After a timed-out park_timeout the state is EMPTY; next park needs a token.
  Parker p;
  EXPECT_FALSE(p.park_timeout(5'000'000));  // 5ms, no unpark
  p.unpark();
  p.park();  // consumes the fresh token
  SUCCEED();
}

TEST(Parker, SequentialParkUnparkRounds) {
  Parker p;
  for (int i = 0; i < 100; ++i) {
    p.unpark();
    p.park();
  }
  SUCCEED();
}

// wake_all must release EVERY waiter blocked on the address (not just one). N
// threads wait on the same word; after they're parked we change the value and
// wake_all -- all N must observe it and finish (a missed waiter would hang the
// join and surface as a test timeout).
TEST(WaitOnAddress, WakeAllReleasesAllWaiters) {
  constexpr int kWaiters = 4;
  std::atomic<std::uint32_t> word{0};
  std::atomic<int> woke{0};
  struct Args {
    std::atomic<std::uint32_t>* word;
    std::atomic<int>* woke;
  } args{&word, &woke};

  std::vector<Thread> threads;
  for (int i = 0; i < kWaiters; ++i) {
    auto r = Thread::spawn(
        [](void* a) {
          auto* x = static_cast<Args*>(a);
          // Loop over spurious wakes: only exit once the value really changed.
          while (x->word->load(std::memory_order_acquire) == 0)
            hyle::platform::detail::wait_on_address(x->word, 0);
          x->woke->fetch_add(1, std::memory_order_release);
        },
        &args);
    ASSERT_TRUE(r.has_value());
    threads.push_back(std::move(*r));
  }
  // Give them time to reach the wait, then release all at once.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  word.store(1, std::memory_order_release);
  hyle::platform::detail::wake_all(&word);
  for (auto& t : threads) t.join();  // hangs (=> timeout) if any waiter was missed
  EXPECT_EQ(woke.load(std::memory_order_acquire), kWaiters);
}

// ---- park_timeout under concurrency: the timeout-vs-unpark race -------------
// The functional park_timeout tests above cover the sequential contracts
// (immediate token, zero-poll, woken-by-unpark, timeout-then-fresh-unpark). The
// remaining gap is CONCURRENT: park_timeout is single-shot and resets the state
// word with exchange(kEmpty) after its timed wait, so an unpark landing right at
// that reset boundary must still not lose the token -- it is either consumed by
// this call (returns true) or left as NOTIFIED for the next call, never dropped.
// Both stressors below reuse HandoffStressNoLostWakeup's handshake: main unparks
// every worker each round and waits for each to consume before the next round,
// so at most one token is ever outstanding per worker. A lost wakeup would stall
// a worker forever (join hangs -> test timeout); a double-count is impossible.
// The worker rides park_timeout in a re-check loop (single-shot: a false return
// is a spurious/expired wake, so it just re-parks) and only a real token, i.e. a
// true return, advances its counter -- which must end at exactly kRounds.

// Long deadline (10s, never fires here): isolates the unpark-vs-timed-reset race.
// The wake always comes from unpark, exercised through the TIMED wait path rather
// than park(), proving the single-shot reset never drops a concurrent token.
TEST(Parker, ParkTimeoutHandoffStressNoLostWakeup) {
  constexpr int kWorkers = 4;
  constexpr int kRounds = 500;
  constexpr std::uint64_t kLongNs = 10'000'000'000ull;  // 10s: must NOT expire
  std::array<Parker, kWorkers> parkers;  // non-movable: in place
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
          int consumed = 0;
          while (consumed < kRounds) {
            if (x->p->park_timeout(kLongNs)) {  // only a real token counts
              ++consumed;
              x->obs->fetch_add(1, std::memory_order_acq_rel);
            }
            // false == spurious low-level wake (deadline is far off): re-park.
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

// Tiny deadline (200us, fires constantly): the worker is perpetually cycling
// through the timeout -> exchange(kEmpty) reset window, so main's unpark lands on
// that exact boundary far more often -- the interleave the three-state machine
// must get right. Same invariant: every per-worker counter still reaches EXACTLY
// kRounds, because a token racing the expiry is consumed-or-left, never lost.
TEST(Parker, ParkTimeoutExpiryVsUnparkRace) {
  constexpr int kWorkers = 4;
  constexpr int kRounds = 500;
  constexpr std::uint64_t kShortNs = 200'000ull;  // 200us: expires constantly
  std::array<Parker, kWorkers> parkers;
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
          int consumed = 0;
          while (consumed < kRounds) {
            if (x->p->park_timeout(kShortNs)) {  // times out often -> re-park
              ++consumed;
              x->obs->fetch_add(1, std::memory_order_acq_rel);
            }
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
