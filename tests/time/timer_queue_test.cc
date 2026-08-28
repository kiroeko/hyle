#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <random>
#include <type_traits>
#include <vector>

#include "hyle/time/timer_queue.h"

using hyle::time::TimerEntry;
using hyle::time::TimerQueue;

namespace {

constexpr std::uint64_t kMs = 100'000;  // compact deterministic test unit

static_assert(!std::is_copy_constructible_v<TimerEntry>);
static_assert(!std::is_copy_assignable_v<TimerEntry>);
static_assert(!std::is_move_constructible_v<TimerEntry>);
static_assert(!std::is_move_assignable_v<TimerEntry>);

// Drain the queue to `now_ns` and return the fired payloads.
std::vector<int> advance_collect(TimerQueue& w, std::uint64_t now_ns) {
  TimerQueue::EntryList out;
  w.advance(now_ns, out);
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

// ---- basic arm + fire --------------------------------------------------------

TEST(TimerQueue, FiresAfterDeadline) {
  TimerQueue w(0);
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

TEST(TimerQueue, DefaultConstructionAndTypedNextDeadlineFitNormalUse) {
  using namespace std::chrono_literals;
  TimerQueue w;
  EXPECT_FALSE(w.next_deadline().has_value());
  TimerEntry e = make_entry(1);
  w.schedule_after(&e, 2ms);
  const auto next = w.next_deadline();
  ASSERT_TRUE(next.has_value());
  EXPECT_GT(*next, hyle::time::MonotonicTimePoint{});
  EXPECT_TRUE(w.cancel(&e));
}

TEST(TimerQueue, DueNowFiresOnNextAdvance) {
  TimerQueue w(0);
  // Advance first so "now" > 0.
  advance_collect(w, 10 * kMs);
  TimerEntry e = make_entry(7);
  w.schedule(&e, 0);  // already behind current time -> due on next advance
  EXPECT_EQ(w.size(), 1u);
  auto fired = advance_collect(w, 10 * kMs);
  ASSERT_EQ(fired.size(), 1u);
  EXPECT_EQ(fired[0], 7);
}

// ---- incremental expiry across many deadlines ------------------------------

TEST(TimerQueue, IncrementalAdvancesFireOnlyTheReachedDeadline) {
  TimerQueue w(0);
  // Arm out of order. Each incremental advance must expose only the deadline
  // reached by that call; a single batched advance deliberately has no
  // cross-bucket total-order guarantee.
  TimerEntry e100 = make_entry(100);
  TimerEntry e5 = make_entry(5);
  TimerEntry e999 = make_entry(999);
  TimerEntry e64 = make_entry(64);
  w.schedule(&e100, 100 * kMs);
  w.schedule(&e5, 5 * kMs);
  w.schedule(&e999, 999 * kMs);
  w.schedule(&e64, 64 * kMs);
  EXPECT_EQ(w.size(), 4u);

  std::vector<int> all;
  // Advance in a few steps to exercise repeated radix rebucketing.
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

// FIFO for equal deadlines.
TEST(TimerQueue, SameDeadlineIsFifo) {
  TimerQueue w(0);
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

// ---- radix boundary correctness ---------------------------------------------

// Sweep deadlines around binary boundaries: rebucketing must never fire early.
TEST(TimerQueue, RadixBoundariesFireExactlyAndNeverEarly) {
  // 64 = L1 boundary, 4096 = L2 boundary, plus off-boundary values.
  for (std::uint64_t deadline : {64ull, 65ull, 127ull, 128ull, 4096ull, 4097ull,
                                 5000ull, 262143ull, 262144ull}) {
    TimerQueue w(0);
    TimerEntry e = make_entry(42);
    w.schedule(&e, deadline * kMs);

    // One test unit before the deadline: must NOT have fired.
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
// everything that came due in one call.
TEST(TimerQueue, SingleBigAdvanceFiresAllDue) {
  TimerQueue w(0);
  std::vector<TimerEntry> entries(200);
  for (int i = 1; i <= 200; ++i) {
    entries[static_cast<std::size_t>(i - 1)].set_data(
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(i)));
  }
  for (int i = 0; i < 200; ++i) {
    w.schedule(&entries[i], static_cast<std::uint64_t>(i + 1) * kMs);
  }
  EXPECT_EQ(w.size(), 200u);
  // Jump to 200ms in one shot.
  auto fired = advance_collect(w, 200 * kMs);
  EXPECT_EQ(fired.size(), 200u);
  EXPECT_TRUE(w.empty());
  // Radix delivery order across different buckets is intentionally unspecified.
  std::vector<bool> seen(201, false);
  for (int id : fired) {
    ASSERT_GE(id, 1);
    ASSERT_LE(id, 200);
    EXPECT_FALSE(seen[static_cast<std::size_t>(id)]);
    seen[static_cast<std::size_t>(id)] = true;
  }
}

// ---- O(1) cancel, no leak ----------------------------------------------------

TEST(TimerQueue, CancelRemovesAndDoesNotFire) {
  TimerQueue w(0);
  TimerEntry a = make_entry(1), b = make_entry(2), c = make_entry(3);
  w.schedule(&a, 10 * kMs);
  w.schedule(&b, 20 * kMs);
  w.schedule(&c, 30 * kMs);
  EXPECT_EQ(w.size(), 3u);

  EXPECT_TRUE(w.cancel(&b));
  EXPECT_FALSE(w.cancel(&b));
  EXPECT_EQ(w.size(), 2u);
  EXPECT_FALSE(b.scheduled());

  auto fired = advance_collect(w, 100 * kMs);
  ASSERT_EQ(fired.size(), 2u);
  EXPECT_TRUE((fired[0] == 1 && fired[1] == 3) ||
              (fired[0] == 3 && fired[1] == 1));
}

TEST(TimerQueue, CancelIsIdempotentAndSafeAfterFire) {
  TimerQueue w(0);
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

TEST(TimerQueue, CancelAllDetachesEveryEntryForImmediateReuse) {
  TimerQueue w(0);
  TimerEntry a = make_entry(1), b = make_entry(2), c = make_entry(3);
  w.schedule(&a, 100 * kMs);
  w.schedule(&b, 2 * kMs);
  w.schedule(&c, 10'000 * kMs);

  TimerQueue::EntryList cancelled;
  EXPECT_EQ(w.cancel_all(cancelled), 3u);
  EXPECT_TRUE(w.empty());

  std::size_t seen = 0;
  while (TimerEntry* entry = cancelled.pop_front()) {
    EXPECT_FALSE(entry->scheduled());
    ++seen;
  }
  EXPECT_EQ(seen, 3u);

  w.schedule(&a, kMs);
  EXPECT_EQ(advance_collect(w, kMs), (std::vector<int>{1}));
}

TEST(TimerQueue, ChronoFacadeKeepsExactNanosecondDeadline) {
  using namespace std::chrono_literals;
  const hyle::time::MonotonicTimePoint origin{};
  TimerQueue w(origin);
  TimerEntry e = make_entry(1);
  w.schedule_after(&e, 150us, origin);

  TimerQueue::EntryList fired;
  EXPECT_EQ(w.advance(origin + 149us, fired), 0u);
  EXPECT_EQ(w.advance(origin + 150us, fired), 1u);
  EXPECT_EQ(fired.pop_front(), &e);
}

// Cancel an entry in a far radix bucket.
TEST(TimerQueue, CancelFarBucketEntry) {
  TimerQueue w(0);
  TimerEntry e = make_entry(1);
  w.schedule(&e, 5000 * kMs);
  EXPECT_EQ(w.size(), 1u);
  w.cancel(&e);
  EXPECT_EQ(w.size(), 0u);
  EXPECT_TRUE(advance_collect(w, 6000 * kMs).empty());
}

// Cancel after advance rebuckets a far entry relative to the new radix origin.
TEST(TimerQueue, CancelAfterRebucket) {
  TimerQueue w(0);
  TimerEntry e = make_entry(1);
  w.schedule(&e, 5000 * kMs);
  // Advance near it without firing; the boundary bucket is reclassified.
  EXPECT_TRUE(advance_collect(w, 4100 * kMs).empty());
  EXPECT_EQ(w.size(), 1u);
  w.cancel(&e);
  EXPECT_EQ(w.size(), 0u);
  EXPECT_FALSE(e.scheduled());
  // Advancing past the original deadline must fire nothing.
  EXPECT_TRUE(advance_collect(w, 6000 * kMs).empty());
}

// ---- reschedule --------------------------------------------------------------

TEST(TimerQueue, ScheduleOnScheduledReschedules) {
  TimerQueue w(0);
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

TEST(TimerQueue, ScheduleAfterUsesSaturatingDeadlineArithmetic) {
  TimerQueue w(0);
  TimerEntry e = make_entry(1);
  w.schedule_after(&e, 50, 100);
  EXPECT_EQ(e.deadline_ns(), 150u);
  w.cancel(&e);

  w.schedule_after(&e, 100, UINT64_MAX - 10);
  EXPECT_EQ(e.deadline_ns(), UINT64_MAX);
  w.cancel(&e);
}

// ---- complete uint64 deadline domain -----------------------------------------

TEST(TimerQueue, FormerHorizonNoLongerFiresEarly) {
  TimerQueue w(0);
  TimerEntry e = make_entry(1);
  // Regression for the old fixed-wheel horizon; radix timestamps cover uint64.
  const std::uint64_t far = (1ull << 36) + 1000;
  w.schedule(&e, far * kMs);
  EXPECT_EQ(w.size(), 1u);

  EXPECT_TRUE(advance_collect(w, (far - 1) * kMs).empty());
  auto fired = advance_collect(w, far * kMs);
  ASSERT_EQ(fired.size(), 1u);
  EXPECT_EQ(fired[0], 1);
  EXPECT_TRUE(w.empty());
}

// ---- next_deadline_ns (the "no busy tick" contract) --------------------------

TEST(TimerQueue, NextDeadlineReportsEarliest) {
  TimerQueue w(100);  // non-zero epoch
  EXPECT_EQ(w.next_deadline_ns(), UINT64_MAX);  // empty

  TimerEntry a = make_entry(1), b = make_entry(2);
  w.schedule(&a, 100 + 50 * kMs);
  w.schedule(&b, 100 + 20 * kMs);
  // Earliest is b at 20 units and is reported exactly.
  std::uint64_t nd = w.next_deadline_ns();
  EXPECT_EQ(nd, 100 + 20 * kMs);

  w.cancel(&b);
  // Cancelling the minimum deliberately keeps a conservative stale wakeup so
  // cancel stays O(1) without scanning its bucket.
  EXPECT_EQ(w.next_deadline_ns(), 100 + 20 * kMs);
  EXPECT_TRUE(advance_collect(w, 100 + 20 * kMs).empty());
  EXPECT_EQ(w.next_deadline_ns(), 100 + 50 * kMs);
  w.cancel(&a);
  EXPECT_EQ(w.next_deadline_ns(), UINT64_MAX);
}

// A far deadline remains exact and the complete uint64 range never wraps.
TEST(TimerQueue, FarDeadlineAndUint64RangeStayExact) {
  TimerQueue w(0);
  TimerEntry e = make_entry(1);
  w.schedule(&e, 100'000 * kMs);
  std::uint64_t nd = w.next_deadline_ns();
  EXPECT_NE(nd, UINT64_MAX);
  EXPECT_EQ(nd, 100'000 * kMs);
  w.cancel(&e);

  // An absolute deadline near u64 max must not wrap or fire early.
  TimerQueue w2(0);
  TimerEntry big = make_entry(2);
  w2.schedule(&big, UINT64_MAX - 100);
  EXPECT_EQ(w2.size(), 1u);
  EXPECT_EQ(w2.next_deadline_ns(), UINT64_MAX - 100);
  // Far below the deadline: must not have fired.
  EXPECT_TRUE(advance_collect(w2, 1000 * kMs).empty());
  EXPECT_EQ(w2.size(), 1u);
  TimerQueue::EntryList final;
  w2.advance(UINT64_MAX, final);
  EXPECT_TRUE(w2.empty());
  TimerEntry* fired = final.pop_front();
  ASSERT_EQ(fired, &big);
  EXPECT_EQ(final.pop_front(), nullptr);

  TimerQueue w3(0);
  TimerEntry last = make_entry(3);
  w3.schedule(&last, UINT64_MAX);
  EXPECT_TRUE(w3.next_deadline().has_value());
  EXPECT_TRUE(w3.cancel(&last));
}

TEST(TimerQueue, RejectsBackwardAdvance) {
  EXPECT_DEATH_IF_SUPPORTED(
      {
        TimerQueue queue(100);
        TimerQueue::EntryList out;
        queue.advance(99, out);
      },
      "");
}

TEST(TimerQueue, RejectsWrongOwnerAndLinkedOutputReuse) {
  TimerQueue first(0);
  TimerQueue second(0);
  TimerEntry e = make_entry(1);
  first.schedule(&e, 10 * kMs);

  EXPECT_DEATH_IF_SUPPORTED(second.cancel(&e), "");
  EXPECT_DEATH_IF_SUPPORTED(second.schedule(&e, 20 * kMs), "");
  first.cancel(&e);

  TimerEntry linked = make_entry(2);
  first.schedule(&linked, 20 * kMs);
  TimerQueue::EntryList out;
  first.advance(20 * kMs, out);
  EXPECT_DEATH_IF_SUPPORTED(first.schedule(&linked, 30 * kMs), "");
  EXPECT_EQ(out.pop_front(), &linked);
}

TEST(TimerQueue, NonEmptySourceTeardownDetachesWithoutFiring) {
  int callbacks = 0;
  TimerEntry a(&callbacks, [](TimerEntry* entry) noexcept {
    ++*static_cast<int*>(entry->data());
  });
  TimerEntry b;
  {
    TimerQueue w(0);
    w.schedule(&a, 10 * kMs);
    w.schedule(&b, 20 * kMs);
  }
  EXPECT_EQ(callbacks, 0);
  EXPECT_FALSE(a.scheduled());
  EXPECT_FALSE(b.scheduled());

  TimerQueue reuse(0);
  reuse.schedule(&a, kMs);
  EXPECT_TRUE(reuse.cancel(&a));
}

TEST(TimerQueue, DestroyingEntryBeforeOwningSourceFailsFast) {
  EXPECT_DEATH_IF_SUPPORTED(
      {
        TimerQueue w(0);
        TimerEntry e;
        w.schedule(&e, 10 * kMs);
      },
      "");
}

TEST(TimerQueue, DestroyingFiredListDetachesEntriesForReuse) {
  TimerEntry e;
  {
    TimerQueue::EntryList fired;
    TimerQueue w(0);
    w.schedule(&e, kMs);
    w.advance(kMs, fired);
  }
  TimerQueue reuse(kMs);
  reuse.schedule(&e, 2 * kMs);
  EXPECT_TRUE(reuse.cancel(&e));
}

// ---- scale: a million arm/cancel churn, oracle-checked -----------------------

// Mirrors the target workload (arm >> cancel >> fire). Cross-check the queue's
// fire set against a brute-force oracle: exactly the never-cancelled timers,
// each fired once, in nondecreasing deadline order overall.
TEST(TimerQueue, MillionArmCancelChurnMatchesOracle) {
  constexpr int kN = 1'000'000;
  TimerQueue w(0);
  std::vector<TimerEntry> entries(kN);
  std::vector<std::uint64_t> deadlines(kN);
  std::vector<char> cancelled(kN, 0);

  std::mt19937_64 rng(12345);
  // Deadlines spread across the radix range: 1 .. ~1M test units.
  std::uniform_int_distribution<std::uint64_t> dl(1, 1'000'000);
  for (int i = 0; i < kN; ++i) {
    entries[i].set_data(
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(i)));
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
  // Advance in 4096-unit strides until the queue empties.
  while (!w.empty()) {
    now += 4096 * kMs;
    TimerQueue::EntryList out;
    w.advance(now, out);
    while (TimerEntry* e = out.pop_front()) {
      int id = static_cast<int>(reinterpret_cast<std::uintptr_t>(e->data()));
      EXPECT_FALSE(cancelled[id]) << "cancelled timer fired: " << id;
      EXPECT_FALSE(fired_seen[id]) << "double fire: " << id;
      EXPECT_GE(now, e->deadline_ns()) << "early fire: " << id;
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
