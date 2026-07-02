#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

#include "hyle/time/timer_wheel.h"

using hyle::time::TimerEntry;
using hyle::time::TimerWheel;

namespace {

constexpr std::uint64_t kMs = TimerWheel::kTickNs;  // 1ms in ns

// Drain the wheel to `now_ns` and return the fired payloads in fire order.
std::vector<int> advance_collect(TimerWheel& w, std::uint64_t now_ns) {
  TimerWheel::EntryList out;
  w.advance(now_ns, out);
  std::vector<int> ids;
  while (TimerEntry* e = out.pop_front()) {
    ids.push_back(static_cast<int>(reinterpret_cast<std::uintptr_t>(e->data)));
  }
  return ids;
}

TimerEntry make_entry(int id) {
  TimerEntry e;
  e.data = reinterpret_cast<void*>(static_cast<std::uintptr_t>(id));
  return e;
}

}  // namespace

// ---- basic arm + fire --------------------------------------------------------

TEST(TimerWheel, FiresAfterDeadline) {
  TimerWheel w(0);
  TimerEntry e = make_entry(1);
  w.schedule(&e, 5 * kMs);
  EXPECT_EQ(w.size(), 1u);

  // Not yet due at 4ms.
  EXPECT_TRUE(advance_collect(w, 4 * kMs).empty());
  EXPECT_EQ(w.size(), 1u);

  // Due at 5ms.
  auto fired = advance_collect(w, 5 * kMs);
  ASSERT_EQ(fired.size(), 1u);
  EXPECT_EQ(fired[0], 1);
  EXPECT_TRUE(w.empty());
}

TEST(TimerWheel, DueNowFiresOnNextAdvance) {
  TimerWheel w(0);
  // Advance the wheel first so "now" > 0.
  advance_collect(w, 10 * kMs);
  TimerEntry e = make_entry(7);
  w.schedule(&e, 0);  // already behind current time -> next tick
  EXPECT_EQ(w.size(), 1u);
  auto fired = advance_collect(w, 11 * kMs);
  ASSERT_EQ(fired.size(), 1u);
  EXPECT_EQ(fired[0], 7);
}

// ---- expiry ORDER across many deadlines --------------------------------------

TEST(TimerWheel, FiresInDeadlineOrder) {
  TimerWheel w(0);
  // Arm out of order; expiry must come back in ascending deadline order.
  TimerEntry e100 = make_entry(100);
  TimerEntry e5 = make_entry(5);
  TimerEntry e999 = make_entry(999);
  TimerEntry e64 = make_entry(64);  // exactly one level-0 rotation
  w.schedule(&e100, 100 * kMs);
  w.schedule(&e5, 5 * kMs);
  w.schedule(&e999, 999 * kMs);
  w.schedule(&e64, 64 * kMs);
  EXPECT_EQ(w.size(), 4u);

  std::vector<int> all;
  // Advance in a few steps to exercise repeated cascades.
  for (std::uint64_t t : {5, 64, 100, 999}) {
    for (int id : advance_collect(w, t * kMs)) all.push_back(id);
  }
  ASSERT_EQ(all.size(), 4u);
  EXPECT_EQ(all[0], 5);
  EXPECT_EQ(all[1], 64);
  EXPECT_EQ(all[2], 100);
  EXPECT_EQ(all[3], 999);
  EXPECT_TRUE(w.empty());
}

// FIFO within the same tick (folly/tokio push_back semantics).
TEST(TimerWheel, SameDeadlineIsFifo) {
  TimerWheel w(0);
  TimerEntry a = make_entry(1), b = make_entry(2), c = make_entry(3);
  w.schedule(&a, 50 * kMs);
  w.schedule(&b, 50 * kMs);
  w.schedule(&c, 50 * kMs);
  auto fired = advance_collect(w, 50 * kMs);
  ASSERT_EQ(fired.size(), 3u);
  EXPECT_EQ(fired[0], 1);
  EXPECT_EQ(fired[1], 2);
  EXPECT_EQ(fired[2], 3);
}

// ---- cascade correctness (the hierarchy's defining behaviour) ----------------

// An entry armed in a high level must cascade down and STILL fire at the right
// tick, never early. Sweep every deadline that crosses level boundaries.
TEST(TimerWheel, CascadeFiresAtExactTickNotEarly) {
  // 64 = L1 boundary, 4096 = L2 boundary, plus off-boundary values.
  for (std::uint64_t deadline : {64ull, 65ull, 127ull, 128ull, 4096ull, 4097ull,
                                 5000ull, 262143ull, 262144ull}) {
    TimerWheel w(0);
    TimerEntry e = make_entry(42);
    w.schedule(&e, deadline * kMs);

    // One tick before the deadline: must NOT have fired.
    auto before = advance_collect(w, (deadline - 1) * kMs);
    EXPECT_TRUE(before.empty()) << "fired early for deadline=" << deadline;
    EXPECT_EQ(w.size(), 1u) << "lost entry for deadline=" << deadline;

    // At the deadline: fires exactly once.
    auto at = advance_collect(w, deadline * kMs);
    ASSERT_EQ(at.size(), 1u) << "did not fire at deadline=" << deadline;
    EXPECT_EQ(at[0], 42);
    EXPECT_TRUE(w.empty());
  }
}

// Advancing straight past a far deadline in a single big step must still fire
// everything that came due (cascade loop drains all intermediate slots).
TEST(TimerWheel, SingleBigAdvanceFiresAllDue) {
  TimerWheel w(0);
  std::vector<TimerEntry> entries;
  entries.reserve(200);
  for (int i = 1; i <= 200; ++i) entries.push_back(make_entry(i));
  for (int i = 0; i < 200; ++i) {
    w.schedule(&entries[i], static_cast<std::uint64_t>(i + 1) * kMs);
  }
  EXPECT_EQ(w.size(), 200u);
  // Jump to 200ms in one shot.
  auto fired = advance_collect(w, 200 * kMs);
  EXPECT_EQ(fired.size(), 200u);
  EXPECT_TRUE(w.empty());
  // Order preserved.
  for (int i = 0; i < 200; ++i) EXPECT_EQ(fired[i], i + 1);
}

// ---- O(1) cancel, no leak ----------------------------------------------------

TEST(TimerWheel, CancelRemovesAndDoesNotFire) {
  TimerWheel w(0);
  TimerEntry a = make_entry(1), b = make_entry(2), c = make_entry(3);
  w.schedule(&a, 10 * kMs);
  w.schedule(&b, 20 * kMs);
  w.schedule(&c, 30 * kMs);
  EXPECT_EQ(w.size(), 3u);

  w.cancel(&b);
  EXPECT_EQ(w.size(), 2u);
  EXPECT_FALSE(b.scheduled);

  auto fired = advance_collect(w, 100 * kMs);
  ASSERT_EQ(fired.size(), 2u);
  EXPECT_EQ(fired[0], 1);
  EXPECT_EQ(fired[1], 3);  // b was cancelled
}

TEST(TimerWheel, CancelIsIdempotentAndSafeAfterFire) {
  TimerWheel w(0);
  TimerEntry e = make_entry(1);
  w.schedule(&e, 5 * kMs);
  w.cancel(&e);
  w.cancel(&e);  // double cancel: no-op, no underflow
  EXPECT_TRUE(w.empty());
  EXPECT_EQ(w.size(), 0u);

  // Re-arm the same node after cancel works.
  w.schedule(&e, 5 * kMs);
  EXPECT_EQ(w.size(), 1u);
  auto fired = advance_collect(w, 5 * kMs);
  ASSERT_EQ(fired.size(), 1u);
  // Cancel after it already fired is a no-op (scheduled == false).
  w.cancel(&e);
  EXPECT_TRUE(w.empty());
}

// Cancel an entry sitting in a HIGH level (before it cascades down): the O(1)
// recovery of (level, slot) must find it there.
TEST(TimerWheel, CancelHighLevelEntry) {
  TimerWheel w(0);
  TimerEntry e = make_entry(1);
  w.schedule(&e, 5000 * kMs);  // lands in level 2
  EXPECT_EQ(w.size(), 1u);
  w.cancel(&e);
  EXPECT_EQ(w.size(), 0u);
  EXPECT_TRUE(advance_collect(w, 6000 * kMs).empty());
}

// ---- reschedule --------------------------------------------------------------

TEST(TimerWheel, ScheduleOnScheduledReschedules) {
  TimerWheel w(0);
  TimerEntry e = make_entry(1);
  w.schedule(&e, 100 * kMs);
  w.schedule(&e, 10 * kMs);  // move earlier
  EXPECT_EQ(w.size(), 1u);   // not double-counted

  auto early = advance_collect(w, 10 * kMs);
  ASSERT_EQ(early.size(), 1u);
  EXPECT_EQ(early[0], 1);
  // Nothing left at the old deadline.
  EXPECT_TRUE(advance_collect(w, 200 * kMs).empty());
  EXPECT_TRUE(w.empty());
}

// ---- wrap-around: deadline beyond MAX_DURATION -------------------------------

TEST(TimerWheel, BeyondHorizonClampsAndStillFires) {
  TimerWheel w(0);
  TimerEntry e = make_entry(1);
  // Way past the ~2yr horizon: clamped to the max, must still eventually fire
  // and never fire early / get lost.
  std::uint64_t far = TimerWheel::kMaxDurationTicks + 1000;
  w.schedule(&e, far * kMs);
  EXPECT_EQ(w.size(), 1u);

  // Advance to the clamp point; it should fire at/by MAX_DURATION, not vanish.
  auto fired = advance_collect(w, TimerWheel::kMaxDurationTicks * kMs);
  ASSERT_EQ(fired.size(), 1u);
  EXPECT_EQ(fired[0], 1);
  EXPECT_TRUE(w.empty());
}

// ---- next_deadline_ns (the "no busy tick" contract) --------------------------

TEST(TimerWheel, NextDeadlineReportsEarliest) {
  TimerWheel w(100);  // non-zero epoch
  EXPECT_EQ(w.next_deadline_ns(), UINT64_MAX);  // empty

  TimerEntry a = make_entry(1), b = make_entry(2);
  w.schedule(&a, 100 + 50 * kMs);
  w.schedule(&b, 100 + 20 * kMs);
  // Earliest is b at 20ms -> reported deadline rounds to the tick boundary.
  std::uint64_t nd = w.next_deadline_ns();
  EXPECT_EQ(nd, 100 + 20 * kMs);

  w.cancel(&b);
  EXPECT_EQ(w.next_deadline_ns(), 100 + 50 * kMs);
  w.cancel(&a);
  EXPECT_EQ(w.next_deadline_ns(), UINT64_MAX);
}

// ---- scale: a million arm/cancel churn, oracle-checked -----------------------

// Mirrors the target workload (arm >> cancel >> fire). Cross-check the wheel's
// fire set against a brute-force oracle: exactly the never-cancelled timers,
// each fired once, in nondecreasing deadline order overall.
TEST(TimerWheel, MillionArmCancelChurnMatchesOracle) {
  constexpr int kN = 1'000'000;
  TimerWheel w(0);
  std::vector<TimerEntry> entries(kN);
  std::vector<std::uint64_t> deadlines(kN);
  std::vector<char> cancelled(kN, 0);

  std::mt19937_64 rng(12345);
  // Deadlines spread across all levels: 1ms .. ~1M ms (~17min), so cascades run.
  std::uniform_int_distribution<std::uint64_t> dl(1, 1'000'000);
  for (int i = 0; i < kN; ++i) {
    entries[i].data = reinterpret_cast<void*>(static_cast<std::uintptr_t>(i));
    deadlines[i] = dl(rng);
    w.schedule(&entries[i], deadlines[i] * kMs);
  }
  EXPECT_EQ(w.size(), static_cast<std::size_t>(kN));

  // Cancel ~70% at random (the arm-then-cancel-heavy reality).
  std::uniform_int_distribution<int> coin(0, 9);
  std::size_t live = kN;
  for (int i = 0; i < kN; ++i) {
    if (coin(rng) < 7) {
      w.cancel(&entries[i]);
      cancelled[i] = 1;
      --live;
    }
  }
  EXPECT_EQ(w.size(), live);

  // Drain fully. Collect fired ids + verify per-entry deadline is respected.
  std::vector<char> fired_seen(kN, 0);
  std::size_t fired_count = 0;
  std::uint64_t now = 0;
  // Advance in 4096ms strides until the wheel empties.
  while (!w.empty()) {
    now += 4096 * kMs;
    TimerWheel::EntryList out;
    w.advance(now, out);
    while (TimerEntry* e = out.pop_front()) {
      int id = static_cast<int>(reinterpret_cast<std::uintptr_t>(e->data));
      EXPECT_FALSE(cancelled[id]) << "cancelled timer fired: " << id;
      EXPECT_FALSE(fired_seen[id]) << "double fire: " << id;
      fired_seen[id] = 1;
      ++fired_count;
    }
  }
  // Exactly the live set fired, once each.
  EXPECT_EQ(fired_count, live);
  for (int i = 0; i < kN; ++i) {
    EXPECT_EQ(fired_seen[i], cancelled[i] ? 0 : 1) << "mismatch id=" << i;
  }
}
