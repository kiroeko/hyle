#include <gtest/gtest.h>

#include <atomic>
#include <array>
#include <chrono>
#include <cstdint>
#include <thread>
#include <type_traits>

#include "hyle/time/clock.h"

namespace {

using hyle::time::Duration;
using hyle::time::Instant;
using hyle::time::ManualClock;
using hyle::time::MonotonicClockRef;
using hyle::time::MonotonicTimePoint;
using hyle::time::SystemMonotonicClock;

static_assert(hyle::time::MonotonicClockSource<SystemMonotonicClock>);
static_assert(hyle::time::MonotonicClockSource<ManualClock>);
static_assert(hyle::time::MonotonicClockSource<MonotonicClockRef>);
static_assert(!std::is_copy_constructible_v<ManualClock>);
static_assert(!std::is_move_constructible_v<ManualClock>);
static_assert(std::is_copy_constructible_v<MonotonicClockRef>);
static_assert(!std::is_constructible_v<MonotonicClockRef,
                                       SystemMonotonicClock&&>);

MonotonicTimePoint point(std::int64_t nanoseconds) {
  return MonotonicTimePoint(Duration(nanoseconds));
}

TEST(ManualClock, StartsAtZeroAndMovesOnlyForward) {
  ManualClock clock;
  EXPECT_EQ(clock.now(), point(0));
  EXPECT_TRUE(clock.try_set(point(17)));
  EXPECT_EQ(clock.now(), point(17));
  EXPECT_TRUE(clock.try_set(point(17)));
  EXPECT_FALSE(clock.try_set(point(16)));
  EXPECT_FALSE(clock.try_set(point(-1)));
  EXPECT_EQ(clock.now(), point(17));
  EXPECT_EQ(clock.advance(Duration(5)), point(22));
  EXPECT_EQ(clock.advance(Duration::zero()), point(22));
}

TEST(ManualClock, AdvanceSaturatesAtDurationMaximum) {
  ManualClock clock(point((Duration::max)().count() - 2));
  EXPECT_EQ(clock.advance(Duration(10)), MonotonicTimePoint(Duration::max()));
  EXPECT_EQ(clock.advance(Duration(1)), MonotonicTimePoint(Duration::max()));
}

TEST(ManualClock, NegativeConstructionAndAdvanceFailFast) {
  EXPECT_DEATH_IF_SUPPORTED((void)ManualClock(point(-1)), "");
  ManualClock clock;
  EXPECT_DEATH_IF_SUPPORTED((void)clock.advance(Duration(-1)), "");
}

TEST(ManualClock, ConcurrentAdvancesAccumulate) {
  ManualClock clock;
  constexpr std::size_t kThreads = 4;
  constexpr std::size_t kAdvances = 10'000;
  std::array<std::thread, kThreads> threads;
  for (std::thread& thread : threads) {
    thread = std::thread([&clock] {
      for (std::size_t index = 0; index < kAdvances; ++index) {
        (void)clock.advance(Duration(1));
      }
    });
  }
  for (std::thread& thread : threads) thread.join();
  EXPECT_EQ(clock.now(), point(kThreads * kAdvances));
}

TEST(ManualClock, ConcurrentReadsSetAndAdvanceNeverRegress) {
  ManualClock clock;
  std::atomic<bool> stop_reader{false};
  std::atomic<bool> regressed{false};
  std::thread reader([&] {
    MonotonicTimePoint previous = clock.now();
    while (!stop_reader.load(std::memory_order_relaxed)) {
      const MonotonicTimePoint current = clock.now();
      if (current < previous) regressed.store(true, std::memory_order_relaxed);
      previous = current;
    }
  });
  std::thread advancing([&] {
    for (std::size_t index = 0; index < 20'000; ++index) {
      (void)clock.advance(Duration(1));
    }
  });
  std::thread setting([&] {
    for (std::int64_t value = 10; value <= 100'000; value += 10) {
      (void)clock.try_set(point(value));
    }
  });
  advancing.join();
  setting.join();
  stop_reader.store(true, std::memory_order_relaxed);
  reader.join();
  EXPECT_FALSE(regressed.load(std::memory_order_relaxed));
  EXPECT_GE(clock.now(), point(100'000));
}

TEST(ManualClock, DeadlineRemainingAndInstantUseExplicitSource) {
  using namespace std::chrono_literals;
  ManualClock clock(point(100));
  const MonotonicTimePoint deadline = hyle::time::deadline_after(clock, 50ns);
  EXPECT_EQ(deadline, point(150));
  EXPECT_EQ(hyle::time::deadline_after_ns(clock, 25), 125u);
  EXPECT_EQ(hyle::time::remaining(clock, deadline), 50ns);
  EXPECT_EQ(hyle::time::remaining_ns(clock, 150), 50u);

  const Instant started = Instant::now(clock);
  (void)clock.advance(20ns);
  EXPECT_EQ(started.elapsed(clock), 20ns);
  EXPECT_EQ(started.elapsed_ns(clock), 20u);
  EXPECT_EQ(hyle::time::remaining(clock, deadline), 30ns);
}

TEST(MonotonicClockRef, DefaultsToSystemAndBorrowsManualSource) {
  const MonotonicClockRef system;
  const auto before = hyle::time::monotonic_now();
  const auto sampled = system.now();
  const auto after = hyle::time::monotonic_now();
  EXPECT_GE(sampled, before);
  EXPECT_LE(sampled, after);

  ManualClock manual(point(41));
  const MonotonicClockRef selected(manual);
  EXPECT_EQ(selected.now(), point(41));
  (void)manual.advance(Duration(1));
  EXPECT_EQ(selected.now(), point(42));
  EXPECT_EQ(hyle::time::deadline_after_ns(selected, 8), 50u);
}

}  // namespace
