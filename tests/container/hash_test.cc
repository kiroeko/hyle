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

TEST(Hash, IntegralBijectiveAndSpreadsHighBits) {
  // mix64 (splitmix64) is a bijection: distinct ints -> distinct full hashes.
  // (Testing low-N-bit bucket spread would be wrong -- pigeonhole caps distinct
  // 12-bit buckets at ~4096*(1-1/e)~=2590 for ANY good mixer; only identity
  // hashing hits 4096, which is exactly what we DON'T want.)
  Hash<int> h;
  std::unordered_set<std::uint64_t> full;
  bool any_high_bits = false;
  for (int i = 0; i < 4096; ++i) {
    std::uint64_t hv = h(i);
    full.insert(hv);
    if ((hv >> 32) != 0) any_high_bits = true;  // identity would leave high=0
  }
  EXPECT_EQ(full.size(), 4096u);   // bijection: zero collisions
  EXPECT_TRUE(any_high_bits);      // not identity: high bits populated
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
