#include <gtest/gtest.h>

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

// Same-duration arms (deadlines monotonic) fire in arm order (== deadline order).
TEST(CommonTimeoutQueue, FiresInArmOrder) {
  CommonTimeoutQueue q(0, 1000);
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

// Equal deadlines keep FIFO among themselves.
TEST(CommonTimeoutQueue, EqualDeadlineFifo) {
  CommonTimeoutQueue q(0, 500);
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
  CommonTimeoutQueue q(0, 100);
  TimerEntry a = make_entry(1), b = make_entry(2), c = make_entry(3);
  q.schedule(&a, 10);
  q.schedule(&b, 20);
  q.schedule(&c, 30);
  q.cancel(&b);
  EXPECT_EQ(q.size(), 2u);
  EXPECT_FALSE(b.scheduled);
  auto f = advance_collect(q, 100);
  ASSERT_EQ(f.size(), 2u);
  EXPECT_EQ(f[0], 1);
  EXPECT_EQ(f[1], 3);  // b cancelled
}

// Cancel the FRONT (next to fire) and the BACK -- both O(1) unlink, no corruption.
TEST(CommonTimeoutQueue, CancelFrontAndBack) {
  CommonTimeoutQueue q(0, 100);
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
  CommonTimeoutQueue q(0, 100);
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

// Reschedule an armed entry to a later (still monotonic) deadline.
TEST(CommonTimeoutQueue, Reschedule) {
  CommonTimeoutQueue q(0, 100);
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

TEST(CommonTimeoutQueue, NextDeadline) {
  CommonTimeoutQueue q(0, 100);
  EXPECT_EQ(q.next_deadline_ns(), UINT64_MAX);
  TimerEntry a = make_entry(1), b = make_entry(2);
  q.schedule(&a, 100);
  q.schedule(&b, 200);
  EXPECT_EQ(q.next_deadline_ns(), 100u);  // front
  advance_collect(q, 100);                // fire a
  EXPECT_EQ(q.next_deadline_ns(), 200u);  // now b
}

// Scale: a million arms (monotonic deadlines) with random cancels, oracle-checked.
TEST(CommonTimeoutQueue, MillionArmCancelChurnMatchesOracle) {
  constexpr int kN = 1'000'000;
  CommonTimeoutQueue q(0, 30'000'000);  // e.g. a 30ms common timeout
  std::vector<TimerEntry> entries(kN);
  std::vector<char> cancelled(kN, 0);

  // Monotonic deadlines: i-th armed at deadline i+1 (models same-duration arrival).
  for (int i = 0; i < kN; ++i) {
    entries[i].data = reinterpret_cast<void*>(static_cast<std::uintptr_t>(i));
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
    int id = static_cast<int>(reinterpret_cast<std::uintptr_t>(e->data));
    EXPECT_FALSE(cancelled[id]) << "cancelled fired: " << id;
    // ids are the deadlines-1, so a sorted list fires in ascending id order.
    if (!first) EXPECT_GT(static_cast<std::uint64_t>(id), last_id);
    last_id = static_cast<std::uint64_t>(id);
    first = false;
    ++fired;
  }
  EXPECT_EQ(fired, live);
  EXPECT_TRUE(q.empty());
}
