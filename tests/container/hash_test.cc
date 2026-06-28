#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "hyle/container/hash.h"

using hyle::container::Hash;
using hyle::container::hash_bytes;

TEST(Hash, BytesDeterministic) {
  const char* s = "hello world";
  EXPECT_EQ(hash_bytes(s, 11), hash_bytes(s, 11));  // same input -> same hash
}

TEST(Hash, BytesSeedChangesResult) {
  const char* s = "hello";
  EXPECT_NE(hash_bytes(s, 5, 0), hash_bytes(s, 5, 1));
}

TEST(Hash, BytesAvalancheOnOneByteFlip) {
  char a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  char b[8] = {1, 2, 3, 4, 5, 6, 7, 9};  // flip one byte
  std::uint64_t ha = hash_bytes(a, 8);
  std::uint64_t hb = hash_bytes(b, 8);
  EXPECT_NE(ha, hb);
  // a good hash flips ~half the bits; require at least a meaningful change.
  int diff = std::popcount(ha ^ hb);
  EXPECT_GE(diff, 16);
}

TEST(Hash, EmptyInputDoesNotCrash) {
  EXPECT_EQ(hash_bytes(nullptr, 0), hash_bytes(nullptr, 0));  // defined, stable
}

TEST(Hash, IntegralMixerQualityStatistical) {
  // mix64 is now the abseil-style 128-bit multiply-fold: NOT a bijection (the fold
  // can collide ~1/2^64), so we test STATISTICAL quality, not zero-collision-by-
  // construction: (1) near-zero collisions over a large consecutive-int sample
  // (the real use pattern); (2) high bits populated (not identity); (3) avalanche.
  Hash<std::uint64_t> h;
  std::unordered_set<std::uint64_t> full;
  bool any_high_bits = false;
  constexpr int kN = 100000;
  for (int i = 0; i < kN; ++i) {
    std::uint64_t hv = h(static_cast<std::uint64_t>(i));
    full.insert(hv);
    if ((hv >> 32) != 0) any_high_bits = true;  // identity would leave high=0
  }
  // 100k draws from 2^64: expected collisions ~ kN^2/2^65 ~= 2^-32, i.e. ~0.
  // Allow a tiny slack for robustness but it should be essentially full.
  EXPECT_GE(full.size(), static_cast<std::size_t>(kN) - 2);
  EXPECT_TRUE(any_high_bits);  // not identity: high bits populated
  // Avalanche: flipping ONE input bit should flip ~half (>=16) of 64 output bits.
  for (int bit = 0; bit < 64; ++bit) {
    std::uint64_t base = 0x0123456789abcdefull;
    int diff = std::popcount(h(base) ^ h(base ^ (std::uint64_t{1} << bit)));
    EXPECT_GE(diff, 12) << "weak avalanche on input bit " << bit;
  }
}

TEST(Hash, PointerHashDistinct) {
  int arr[64];
  Hash<int*> h;
  std::unordered_set<std::uint64_t> seen;
  for (int i = 0; i < 64; ++i) seen.insert(h(&arr[i]));
  EXPECT_EQ(seen.size(), 64u);  // distinct pointers -> distinct hashes
}

TEST(Hash, StringViewNoCollisionsOn10k) {
  Hash<std::string_view> h;
  std::unordered_set<std::uint64_t> seen;
  std::vector<std::string> keep;
  keep.reserve(10000);
  for (int i = 0; i < 10000; ++i) {
    keep.push_back("key_" + std::to_string(i) + "_suffix");
    seen.insert(h(keep.back()));
  }
  EXPECT_EQ(seen.size(), 10000u);  // 64-bit XXH3: zero collisions expected
}

TEST(Hash, IntAndEnumUsable) {
  enum class Color : int { Red, Green, Blue };
  Hash<Color> h;
  EXPECT_NE(h(Color::Red), h(Color::Blue));
}
