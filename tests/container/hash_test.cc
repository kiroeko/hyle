#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <random>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "hyle/container/hash.h"
#include "hyle/container/hash_inline.h"  // opt-in inline variant (pulls xxhash/wyhash)

using hyle::container::Hash;
using hyle::container::hash_bytes;
using hyle::container::hash_bytes_inline;

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

// LONG strings (> kWyhashThreshold=128B) take the wyhash branch of hash_bytes
// (best-of-breed by length). Verify that branch: deterministic, seed-sensitive,
// and collision-free over 10k distinct long strings. The short test above only
// exercises the XXH3 branch, so without this the wyhash path is untested.
TEST(Hash, LongStringWyhashPathNoCollisions) {
  std::unordered_set<std::uint64_t> seen;
  std::vector<std::string> keep;
  keep.reserve(10000);
  const std::string filler(200, 'x');  // 200B base -> well past the 128B threshold
  for (int i = 0; i < 10000; ++i) {
    keep.push_back(filler + "_key_" + std::to_string(i) + "_" + filler);
    seen.insert(hyle::container::hash_bytes(keep.back().data(), keep.back().size()));
  }
  EXPECT_EQ(seen.size(), 10000u);  // wyhash 64-bit: zero collisions expected
  // Deterministic + seed-sensitive on the long path.
  const char* s = keep[0].data();
  const std::size_t n = keep[0].size();
  EXPECT_EQ(hyle::container::hash_bytes(s, n, 0), hyle::container::hash_bytes(s, n, 0));
  EXPECT_NE(hyle::container::hash_bytes(s, n, 0), hyle::container::hash_bytes(s, n, 1));
}

// The XXH3<->wyhash crossover at kWyhashThreshold must not break determinism: both
// sides of the boundary (128B XXH3, 129B wyhash) hash consistently to themselves.
TEST(Hash, LengthDispatchBoundaryDeterministic) {
  std::string a(128, 'a'), b(129, 'b');  // 128 -> XXH3, 129 -> wyhash
  EXPECT_EQ(hyle::container::hash_bytes(a.data(), 128), hyle::container::hash_bytes(a.data(), 128));
  EXPECT_EQ(hyle::container::hash_bytes(b.data(), 129), hyle::container::hash_bytes(b.data(), 129));
  EXPECT_NE(hyle::container::hash_bytes(a.data(), 128), hyle::container::hash_bytes(b.data(), 129));
}

// The opt-in header-inline variant MUST be bit-identical to the default out-of-line
// hash_bytes across ALL length dispatch bands (<=16 inline / 17..128 XXH3 / >128
// wyhash) and seeds -- otherwise a key hashed one way and looked up the other breaks.
TEST(Hash, InlineVariantBitIdenticalToOutOfLine) {
  std::mt19937_64 rng(12345);
  // Cover every band + boundaries + the two crossover edges (16/17, 128/129).
  for (std::size_t len : {std::size_t{0}, std::size_t{1}, std::size_t{7},
                          std::size_t{8}, std::size_t{9}, std::size_t{15},
                          std::size_t{16}, std::size_t{17}, std::size_t{31},
                          std::size_t{64}, std::size_t{96}, std::size_t{127},
                          std::size_t{128}, std::size_t{129}, std::size_t{192},
                          std::size_t{256}, std::size_t{512}, std::size_t{1000}}) {
    std::string s(len, '\0');
    for (auto& c : s) c = static_cast<char>(rng() & 0xFF);
    for (std::uint64_t seed : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{0xDEADBEEF}}) {
      EXPECT_EQ(hash_bytes(s.data(), len, seed),
                hash_bytes_inline(s.data(), len, seed))
          << "mismatch at len=" << len << " seed=" << seed;
    }
  }
}

TEST(Hash, IntAndEnumUsable) {
  enum class Color : int { Red, Green, Blue };
  Hash<Color> h;
  EXPECT_NE(h(Color::Red), h(Color::Blue));
}

// Small trivially-copyable struct key (<=16 bytes) goes through the inline hash_small
// path. Verify it has good quality: distinct keys -> distinct hashes, and avalanche on a
// single-field change.
TEST(Hash, SmallStructInlineQuality) {
  struct Key16 { std::uint64_t a, b; };  // 16 bytes -> inline mix path
  static_assert(sizeof(Key16) == 16);
  Hash<Key16> h;
  // No collisions over a large sweep of distinct keys (vary both words).
  std::unordered_set<std::uint64_t> seen;
  constexpr int kN = 200;
  for (int i = 0; i < kN; ++i)
    for (int j = 0; j < kN; ++j)
      seen.insert(h(Key16{static_cast<std::uint64_t>(i), static_cast<std::uint64_t>(j)}));
  EXPECT_EQ(seen.size(), static_cast<std::size_t>(kN) * kN);  // 40000 distinct
  // Avalanche: flip one bit in either word -> ~half the output bits change.
  Key16 base{0x0123456789abcdefull, 0xfedcba9876543210ull};
  for (int bit = 0; bit < 64; ++bit) {
    Key16 fa = base; fa.a ^= (std::uint64_t{1} << bit);
    Key16 fb = base; fb.b ^= (std::uint64_t{1} << bit);
    EXPECT_GE(std::popcount(h(base) ^ h(fa)), 12) << "weak avalanche, word a bit " << bit;
    EXPECT_GE(std::popcount(h(base) ^ h(fb)), 12) << "weak avalanche, word b bit " << bit;
  }
}

// A small struct <=8 bytes also uses the inline path (single mix).
TEST(Hash, SmallStruct8BInline) {
  struct Key8 { std::uint32_t x, y; };  // 8 bytes
  Hash<Key8> h;
  std::unordered_set<std::uint64_t> seen;
  for (std::uint32_t i = 0; i < 1000; ++i) seen.insert(h(Key8{i, i * 2654435761u}));
  EXPECT_EQ(seen.size(), 1000u);  // distinct
}
