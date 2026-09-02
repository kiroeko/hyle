#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <random>
#include <vector>

#include "hyle/time/common_timeout_queue.h"

using hyle::time::CommonTimeoutQueue;
using hyle::time::TimerEntry;
using hyle::time::TimerList;

namespace {

// Drain to now_ns, return fired payload ids in fire order.
std::vector<int> advance_collect(CommonTimeoutQueue& q, std::uint64_t now_ns) {
  TimerList out;
  q.advance(now_ns, out);
  std::vector<int> ids;
  while (TimerEntry* e = out.pop_front()) {
    ids.push_back(static_cast<int>(reinterpret_cast<std::uintptr_t>(e->data())));
  }
  return ids;
}

TimerEntry make_entry(int id) {
  return TimerEntry{
      reinterpret_cast<void*>(static_cast<std::uintptr_t>(id))};
}

}  // namespace

// Same-duration arms (deadlines monotonic) fire in arm order (== deadline order).
TEST(CommonTimeoutQueue, FiresInArmOrder) {
  CommonTimeoutQueue q(1000);
  TimerEntry a = make_entry(1), b = make_entry(2), c = make_entry(3);
  // deadlines 100, 200, 300 (non-decreasing, as same-duration arms would be)
  q.schedule(&a, 100);
  q.schedule(&b, 200);
  q.schedule(&c, 300);
  EXPECT_EQ(q.size(), 3u);

  EXPECT_TRUE(advance_collect(q, 50).empty());  // nothing due
  auto f1 = advance_collect(q, 150);            // only a
  ASSERT_EQ(f1.size(), 1u);
  EXPECT_EQ(f1[0], 1);
  auto f2 = advance_collect(q, 300);            // b then c
  ASSERT_EQ(f2.size(), 2u);
  EXPECT_EQ(f2[0], 2);
  EXPECT_EQ(f2[1], 3);
  EXPECT_TRUE(q.empty());
}

TEST(CommonTimeoutQueue, ExplicitClockSourceSchedulesFixedDuration) {
  using namespace std::chrono_literals;
  hyle::time::ManualClock clock(
      hyle::time::MonotonicTimePoint(hyle::time::Duration(10)));
  CommonTimeoutQueue queue(25ns);
  TimerEntry first = make_entry(1);
  TimerEntry second = make_entry(2);

  EXPECT_TRUE(queue.try_schedule_now(&first, clock));
  EXPECT_EQ(first.deadline_ns(), 35u);
  (void)clock.advance(5ns);
  queue.schedule(&second, clock);
  EXPECT_EQ(second.deadline_ns(), 40u);

  TimerList due;
  (void)clock.advance(20ns);
  EXPECT_EQ(queue.advance(clock.now(), due), 1u);
  EXPECT_EQ(due.pop_front(), &first);
  (void)clock.advance(5ns);
  EXPECT_EQ(queue.advance(clock.now(), due), 1u);
  EXPECT_EQ(due.pop_front(), &second);
}

TEST(CommonTimeoutQueue, NormalScheduleSamplesClockAndTypedNextIsOptional) {
  using namespace std::chrono_literals;
  CommonTimeoutQueue q(2ms);
  EXPECT_FALSE(q.next_deadline().has_value());
  TimerEntry e = make_entry(1);
  q.schedule(&e);
  const auto next = q.next_deadline();
  ASSERT_TRUE(next.has_value());
  EXPECT_EQ(hyle::time::to_monotonic_ns(*next), e.deadline_ns());
  EXPECT_TRUE(q.cancel(&e));
}

// Equal deadlines keep FIFO among themselves.
TEST(CommonTimeoutQueue, EqualDeadlineFifo) {
  CommonTimeoutQueue q(500);
  TimerEntry a = make_entry(1), b = make_entry(2), c = make_entry(3);
  q.schedule(&a, 500);
  q.schedule(&b, 500);
  q.schedule(&c, 500);
  auto f = advance_collect(q, 500);
  ASSERT_EQ(f.size(), 3u);
  EXPECT_EQ(f[0], 1);
  EXPECT_EQ(f[1], 2);
  EXPECT_EQ(f[2], 3);
}

// O(1) cancel removes and does not fire; no leak in count.
TEST(CommonTimeoutQueue, CancelRemovesAndDoesNotFire) {
  CommonTimeoutQueue q(100);
  TimerEntry a = make_entry(1), b = make_entry(2), c = make_entry(3);
  q.schedule(&a, 10);
  q.schedule(&b, 20);
  q.schedule(&c, 30);
  EXPECT_TRUE(q.cancel(&b));
  EXPECT_FALSE(q.cancel(&b));
  EXPECT_EQ(q.size(), 2u);
  EXPECT_FALSE(b.scheduled());
  auto f = advance_collect(q, 100);
  ASSERT_EQ(f.size(), 2u);
  EXPECT_EQ(f[0], 1);
  EXPECT_EQ(f[1], 3);  // b cancelled
}

// Cancel the FRONT (next to fire) and the BACK -- both O(1) unlink, no corruption.
TEST(CommonTimeoutQueue, CancelFrontAndBack) {
  CommonTimeoutQueue q(100);
  TimerEntry a = make_entry(1), b = make_entry(2), c = make_entry(3);
  q.schedule(&a, 10);
  q.schedule(&b, 20);
  q.schedule(&c, 30);
  q.cancel(&a);  // front
  q.cancel(&c);  // back
  EXPECT_EQ(q.size(), 1u);
  auto f = advance_collect(q, 100);
  ASSERT_EQ(f.size(), 1u);
  EXPECT_EQ(f[0], 2);
}

TEST(CommonTimeoutQueue, CancelIdempotentAndAfterFire) {
  CommonTimeoutQueue q(100);
  TimerEntry e = make_entry(1);
  q.schedule(&e, 10);
  q.cancel(&e);
  q.cancel(&e);  // double cancel: no underflow
  EXPECT_TRUE(q.empty());
  // Re-arm works after cancel.
  q.schedule(&e, 10);
  auto f = advance_collect(q, 10);
  ASSERT_EQ(f.size(), 1u);
  q.cancel(&e);  // cancel after fire: no-op
  EXPECT_TRUE(q.empty());
}

TEST(CommonTimeoutQueue, CancelAllPreservesDeadlineOrder) {
  CommonTimeoutQueue q(100);
  TimerEntry a = make_entry(1), b = make_entry(2), c = make_entry(3);
  q.schedule(&a, 10);
  q.schedule(&b, 20);
  q.schedule(&c, 30);

  TimerList cancelled;
  EXPECT_EQ(q.cancel_all(cancelled), 3u);
  EXPECT_TRUE(q.empty());
  EXPECT_EQ(cancelled.pop_front(), &a);
  EXPECT_EQ(cancelled.pop_front(), &b);
  EXPECT_EQ(cancelled.pop_front(), &c);
  EXPECT_EQ(cancelled.pop_front(), nullptr);
}

TEST(CommonTimeoutQueue, ChronoDurationAndAdvanceFacade) {
  using namespace std::chrono_literals;
  CommonTimeoutQueue q(1500us);
  EXPECT_EQ(q.duration(), 1500us);
  TimerEntry e = make_entry(1);
  const hyle::time::MonotonicTimePoint origin{};
  ASSERT_TRUE(q.try_schedule_after(
      &e, hyle::time::to_monotonic_ns(origin + 1ms)));
  EXPECT_EQ(e.deadline_ns(), 2'500'000u);
  TimerList fired;
  EXPECT_EQ(q.advance(origin + 2499us, fired), 0u);
  EXPECT_EQ(q.advance(origin + 2500us, fired), 1u);
  EXPECT_EQ(fired.pop_front(), &e);
}

// Reschedule an armed entry to a later (still monotonic) deadline.
TEST(CommonTimeoutQueue, Reschedule) {
  CommonTimeoutQueue q(100);
  TimerEntry a = make_entry(1), b = make_entry(2);
  q.schedule(&a, 10);
  q.schedule(&b, 20);
  q.schedule(&a, 30);  // move a after b (cancel + re-append; still sorted)
  EXPECT_EQ(q.size(), 2u);
  auto f = advance_collect(q, 100);
  ASSERT_EQ(f.size(), 2u);
  EXPECT_EQ(f[0], 2);  // b now first
  EXPECT_EQ(f[1], 1);
}

TEST(CommonTimeoutQueue, TryScheduleRejectsOutOfOrderWithoutCorruption) {
  CommonTimeoutQueue q(100);
  TimerEntry a = make_entry(1), b = make_entry(2);
  ASSERT_TRUE(q.try_schedule(&a, 20));
  EXPECT_FALSE(q.try_schedule(&b, 10));
  EXPECT_FALSE(b.scheduled());
  EXPECT_EQ(q.size(), 1u);
  q.cancel(&a);
}

TEST(CommonTimeoutQueue, RejectedReschedulePreservesExistingArm) {
  CommonTimeoutQueue q(100);
  auto a = make_entry(1);
  auto b = make_entry(2);
  q.schedule(&a, 100);
  q.schedule(&b, 200);

  EXPECT_FALSE(q.try_schedule(&b, 50));
  EXPECT_TRUE(b.scheduled());
  EXPECT_EQ(b.deadline_ns(), 200u);
  EXPECT_EQ(q.size(), 2u);

  EXPECT_FALSE(q.try_schedule(&a, 50));
  EXPECT_TRUE(a.scheduled());
  EXPECT_EQ(a.deadline_ns(), 100u);
  EXPECT_EQ(q.size(), 2u);

  EXPECT_EQ(advance_collect(q, 200), (std::vector<int>{1, 2}));
}

TEST(CommonTimeoutQueue, ScheduleAfterUsesDeclaredDurationAndSaturates) {
  CommonTimeoutQueue q(30);
  TimerEntry a = make_entry(1), b = make_entry(2);
  ASSERT_TRUE(q.try_schedule_after(&a, 100));
  EXPECT_EQ(a.deadline_ns(), 130u);
  ASSERT_TRUE(q.try_schedule_after(&b, UINT64_MAX - 10));
  EXPECT_EQ(b.deadline_ns(), UINT64_MAX);
  q.cancel(&a);
  q.cancel(&b);
}

TEST(CommonTimeoutQueue, ScheduleNowUsesMonotonicClock) {
  CommonTimeoutQueue q(1'000'000);
  TimerEntry e = make_entry(1);
  const std::uint64_t before = hyle::time::monotonic_ns();
  ASSERT_TRUE(q.try_schedule_now(&e));
  const std::uint64_t after = hyle::time::monotonic_ns();
  EXPECT_GE(e.deadline_ns(), before + 1'000'000);
  EXPECT_LE(e.deadline_ns(), after + 1'000'000);
  q.cancel(&e);
}

TEST(CommonTimeoutQueue, RejectsWrongOwner) {
  CommonTimeoutQueue first(100);
  CommonTimeoutQueue second(100);
  TimerEntry e = make_entry(1);
  first.schedule(&e, 100);
  EXPECT_DEATH_IF_SUPPORTED(second.cancel(&e), "");
  EXPECT_DEATH_IF_SUPPORTED(second.schedule(&e, 200), "");
  first.cancel(&e);
}

TEST(CommonTimeoutQueue, NonEmptySourceTeardownDetachesWithoutFiring) {
  int callbacks = 0;
  TimerEntry a(&callbacks, [](TimerEntry* entry) noexcept {
    ++*static_cast<int*>(entry->data());
  });
  TimerEntry b;
  {
    CommonTimeoutQueue queue(100);
    queue.schedule(&a, 100);
    queue.schedule(&b, 200);
  }
  EXPECT_EQ(callbacks, 0);
  EXPECT_FALSE(a.scheduled());
  EXPECT_FALSE(b.scheduled());

  CommonTimeoutQueue reuse(100);
  reuse.schedule(&a, 300);
  EXPECT_TRUE(reuse.cancel(&a));
}

TEST(CommonTimeoutQueue, DestroyingEntryBeforeOwningSourceFailsFast) {
  EXPECT_DEATH_IF_SUPPORTED(
      {
        CommonTimeoutQueue queue(100);
        TimerEntry armed;
        queue.schedule(&armed, 100);
      },
      "");
}

TEST(CommonTimeoutQueue, NextDeadline) {
  CommonTimeoutQueue q(100);
  EXPECT_EQ(q.next_deadline_ns(), UINT64_MAX);
  TimerEntry a = make_entry(1), b = make_entry(2);
  q.schedule(&a, 100);
  q.schedule(&b, 200);
  EXPECT_EQ(q.next_deadline_ns(), 100u);  // front
  advance_collect(q, 100);                // fire a
  EXPECT_EQ(q.next_deadline_ns(), 200u);  // now b
  q.cancel(&b);
}

// Scale: a million arms (monotonic deadlines) with random cancels, oracle-checked.
TEST(CommonTimeoutQueue, MillionArmCancelChurnMatchesOracle) {
  constexpr int kN = 1'000'000;
  CommonTimeoutQueue q(30'000'000);  // e.g. a 30ms common timeout
  std::vector<TimerEntry> entries(kN);
  std::vector<char> cancelled(kN, 0);

  // Monotonic deadlines: i-th armed at deadline i+1 (models same-duration arrival).
  for (int i = 0; i < kN; ++i) {
    entries[i].set_data(
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(i)));
    q.schedule(&entries[i], static_cast<std::uint64_t>(i) + 1);
  }
  EXPECT_EQ(q.size(), static_cast<std::size_t>(kN));

  std::mt19937_64 rng(999);
  std::uniform_int_distribution<int> coin(0, 9);
  std::size_t live = kN;
  for (int i = 0; i < kN; ++i) {
    if (coin(rng) < 7) {  // cancel ~70%
      q.cancel(&entries[i]);
      cancelled[i] = 1;
      --live;
    }
  }
  EXPECT_EQ(q.size(), live);

  // Drain everything; verify exactly the live set fired, once each, in order.
  TimerList out;
  q.advance(static_cast<std::uint64_t>(kN) + 1, out);
  std::size_t fired = 0;
  std::uint64_t last_id = 0;
  bool first = true;
  while (TimerEntry* e = out.pop_front()) {
    int id = static_cast<int>(reinterpret_cast<std::uintptr_t>(e->data()));
    EXPECT_FALSE(cancelled[id]) << "cancelled fired: " << id;
    // ids are the deadlines-1, so a sorted list fires in ascending id order.
    if (!first) {
      EXPECT_GT(static_cast<std::uint64_t>(id), last_id);
    }
    last_id = static_cast<std::uint64_t>(id);
    first = false;
    ++fired;
  }
  EXPECT_EQ(fired, live);
  EXPECT_TRUE(q.empty());
}
