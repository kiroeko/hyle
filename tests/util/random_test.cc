#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

#include "hyle/util/random.h"

using hyle::util::Rng;
using hyle::util::thread_rng;

TEST(Rng, DeterministicForSameSeed) {
  Rng a(12345), b(12345);
  for (int i = 0; i < 1000; ++i) EXPECT_EQ(a.next_u64(), b.next_u64());
}

TEST(Rng, DifferentSeedsDiverge) {
  Rng a(1), b(2);
  int same = 0;
  for (int i = 0; i < 1000; ++i)
    if (a.next_u64() == b.next_u64()) ++same;
  EXPECT_LT(same, 5);  // essentially never collide
}

TEST(Rng, NotStuckOrZero) {
  Rng r(0);  // seed 0 must still produce nonzero stream (SplitMix64 expands it)
  std::uint64_t acc = 0;
  for (int i = 0; i < 100; ++i) acc |= r.next_u64();
  EXPECT_NE(acc, 0u);
}

TEST(Rng, RangeWithinBounds) {
  Rng r(42);
  for (int i = 0; i < 100000; ++i) {
    std::uint64_t v = r.range_u64(10, 20);
    ASSERT_GE(v, 10u);
    ASSERT_LE(v, 20u);
  }
}

TEST(Rng, RangeSingletonAndFull) {
  Rng r(7);
  EXPECT_EQ(r.range_u64(5, 5), 5u);  // degenerate range -> the only value
  // full range must not loop forever / must stay in type
  std::uint64_t v = r.range_u64(0, UINT64_MAX);
  (void)v;
  SUCCEED();
}

// Bounded output should be ~uniform (chi-square-ish smoke): each of N buckets
// gets roughly equal counts. A biased modulo map would skew low buckets.
TEST(Rng, RangeApproximatelyUniform) {
  Rng r(99);
  constexpr int kBuckets = 16;
  constexpr int kDraws = 1'600'000;
  std::array<int, kBuckets> hist{};
  for (int i = 0; i < kDraws; ++i)
    ++hist[r.range_u64(0, kBuckets - 1)];
  const int expected = kDraws / kBuckets;  // 100000
  for (int c : hist) {
    // within 5% of expected -- loose but catches gross bias
    EXPECT_NEAR(c, expected, expected / 20);
  }
}

TEST(Rng, DoubleInUnitInterval) {
  Rng r(3);
  for (int i = 0; i < 100000; ++i) {
    double d = r.next_double();
    ASSERT_GE(d, 0.0);
    ASSERT_LT(d, 1.0);
  }
}

TEST(Rng, FloatInUnitInterval) {
  Rng r(3);
  for (int i = 0; i < 100000; ++i) {
    float f = r.next_float();
    ASSERT_GE(f, 0.0f);
    ASSERT_LT(f, 1.0f);
  }
}

TEST(Rng, BoolRoughlyBalanced) {
  Rng r(5);
  int trues = 0;
  constexpr int kN = 100000;
  for (int i = 0; i < kN; ++i) trues += r.next_bool();
  EXPECT_NEAR(trues, kN / 2, kN / 20);  // within 5%
}

// jump() must produce a state that diverges from the un-jumped generator.
TEST(Rng, JumpDiverges) {
  Rng a(777);
  Rng b(777);
  b.jump();
  int same = 0;
  for (int i = 0; i < 1000; ++i)
    if (a.next_u64() == b.next_u64()) ++same;
  EXPECT_LT(same, 5);
}

// Fisher-Yates shuffle: a permutation (no element lost/duplicated), and not the
// identity for a decent-sized array.
TEST(Rng, ShufflePermutes) {
  Rng r(2024);
  constexpr int kN = 1000;
  std::vector<int> v(kN);
  for (int i = 0; i < kN; ++i) v[i] = i;
  r.shuffle(kN, [&](std::uint64_t i, std::uint64_t j) { std::swap(v[i], v[j]); });

  std::vector<bool> seen(kN, false);
  int fixed = 0;
  for (int i = 0; i < kN; ++i) {
    ASSERT_FALSE(seen[v[i]]) << "duplicate " << v[i];
    seen[v[i]] = true;
    if (v[i] == i) ++fixed;
  }
  EXPECT_LT(fixed, kN / 2);  // a real shuffle moves most elements
}

TEST(Rng, ThreadRngWorks) {
  std::uint64_t a = thread_rng().next_u64();
  std::uint64_t b = thread_rng().next_u64();
  EXPECT_NE(a, b);  // advances on the same thread
}
