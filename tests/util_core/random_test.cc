#include <gtest/gtest.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <set>
#include <thread>
#include <vector>

#include "hyle/util/random.h"

using hyle::util::Rng;
using hyle::util::thread_rng;

static_assert(std::uniform_random_bit_generator<Rng>);

TEST(Rng, DeterministicForSameSeed) {
  Rng a(12345), b(12345);
  for (int i = 0; i < 1000; ++i) EXPECT_EQ(a.next_u64(), b.next_u64());
}

TEST(Rng, SeedOneMatchesPinnedXoshiroSequence) {
  Rng rng(1);
  constexpr std::uint64_t expected[] = {
      0xcfc5d07f6f03c29bull, 0xbf424132963fe08dull,
      0x19a37d5757aaf520ull, 0xbf08119f05cd56d6ull,
      0x2f47184b86186fa4ull, 0x97299fcae7202345ull,
      0xfca3c79508f41507ull, 0x85fea5c90363f221ull,
  };
  for (std::uint64_t value : expected) EXPECT_EQ(rng.next_u64(), value);
}

TEST(Rng, StandardUrbgCompositionAndDiscard) {
  Rng direct(55);
  Rng discarded(55);
  for (int i = 0; i < 17; ++i) (void)direct.next_u64();
  discarded.discard(17);
  EXPECT_EQ(discarded(), direct.next_u64());

  std::array<int, 64> values{};
  for (int i = 0; i < 64; ++i) values[static_cast<std::size_t>(i)] = i;
  std::shuffle(values.begin(), values.end(), discarded);
  std::sort(values.begin(), values.end());
  for (int i = 0; i < 64; ++i) {
    EXPECT_EQ(values[static_cast<std::size_t>(i)], i);
  }

  std::uniform_int_distribution<int> distribution(3, 9);
  for (int i = 0; i < 1000; ++i) {
    const int value = distribution(discarded);
    EXPECT_GE(value, 3);
    EXPECT_LE(value, 9);
  }
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

TEST(Rng, RangeU32WithinBoundsSingletonAndFull) {
  Rng rng(17);
  for (int i = 0; i < 100000; ++i) {
    const std::uint32_t value = rng.range_u32(100, 1000);
    ASSERT_GE(value, 100u);
    ASSERT_LE(value, 1000u);
  }
  EXPECT_EQ(rng.range_u32(9, 9), 9u);
  (void)rng.range_u32(0, UINT32_MAX);
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

TEST(Rng, RangeDoubleHandlesExtremeAndAdjacentBounds) {
  Rng rng(123);
  const double low = -std::numeric_limits<double>::max();
  const double high = std::numeric_limits<double>::max();
  for (int i = 0; i < 10000; ++i) {
    const double value = rng.range_double(low, high);
    ASSERT_TRUE(std::isfinite(value));
    ASSERT_GE(value, low);
    ASSERT_LT(value, high);
  }

  const double adjacent = std::nextafter(1.0, 2.0);
  for (int i = 0; i < 1000; ++i) {
    EXPECT_EQ(rng.range_double(1.0, adjacent), 1.0);
  }
  EXPECT_EQ(rng.range_double(4.0, 4.0), 4.0);
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

TEST(Rng, LongJumpDivergesFromJumpAndOriginal) {
  Rng original(777);
  Rng jumped(777);
  Rng long_jumped(777);
  jumped.jump();
  long_jumped.long_jump();
  EXPECT_NE(original.next_u64(), jumped.next_u64());
  EXPECT_NE(original.next_u64(), long_jumped.next_u64());
  EXPECT_NE(jumped.next_u64(), long_jumped.next_u64());
}

TEST(Rng, StateSnapshotRestoresExactContinuation) {
  Rng first(91);
  for (int i = 0; i < 13; ++i) (void)first.next_u64();
  const hyle::util::RngState state = first.state();
  Rng restored = Rng::from_state(state);
  for (int i = 0; i < 1000; ++i) {
    EXPECT_EQ(first.next_u64(), restored.next_u64());
  }
  restored.reseed(91);
  Rng fresh(91);
  EXPECT_EQ(restored.next_u64(), fresh.next_u64());
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

TEST(Rng, IteratorShufflePermutes) {
  Rng rng(2025);
  std::array<int, 128> values{};
  for (int i = 0; i < 128; ++i) values[static_cast<std::size_t>(i)] = i;
  rng.shuffle(values.begin(), values.end());
  std::sort(values.begin(), values.end());
  for (int i = 0; i < 128; ++i) {
    EXPECT_EQ(values[static_cast<std::size_t>(i)], i);
  }
}

TEST(Rng, ThreadRngWorks) {
  std::uint64_t a = thread_rng().next_u64();
  std::uint64_t b = thread_rng().next_u64();
  EXPECT_NE(a, b);  // advances on the same thread
}

TEST(Rng, SequentialThreadsReceiveDistinctDefaultStreams) {
  std::set<std::uint64_t> first_values;
  for (int i = 0; i < 32; ++i) {
    std::uint64_t value = 0;
    std::thread thread([&] { value = thread_rng().next_u64(); });
    thread.join();  // encourages the OS to reuse the same stack address
    first_values.insert(value);
  }
  EXPECT_EQ(first_values.size(), 32u);
}

TEST(RngDeathTest, RejectsInvalidBoundsAndAllZeroState) {
  EXPECT_DEATH(
      ([] {
        Rng rng;
        (void)rng.range_u64(2, 1);
      }()),
      "");
  EXPECT_DEATH(
      ([] {
        Rng rng;
        (void)rng.range_u32(2, 1);
      }()),
      "");
  EXPECT_DEATH(
      ([] {
        Rng rng;
        (void)rng.range_double(2.0, 1.0);
      }()),
      "");
  EXPECT_DEATH(
      ([] {
        Rng rng;
        rng.set_state(hyle::util::RngState{});
      }()),
      "");
}
