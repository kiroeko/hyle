#include <gtest/gtest.h>

#include <bit>
#include <array>
#include <bitset>
#include <cstdint>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "hyle/container/hash.h"

using hyle::container::Hash;
using hyle::container::Hashable;
using hyle::container::BytewiseHash;
using hyle::container::HardenedStringHash;
using hyle::container::StringHash;
using hyle::container::hash_bytes;
using hyle::container::siphash24;

namespace {

struct PaddedKey {
  std::uint8_t kind;
  std::uint32_t value;

  bool operator==(const PaddedKey&) const = default;
};

static_assert(sizeof(PaddedKey) > sizeof(std::uint8_t) + sizeof(std::uint32_t));
static_assert(!std::has_unique_object_representations_v<PaddedKey>);
static_assert(!Hashable<PaddedKey>);

// Even a padding-free type is not safe to hash by representation when its
// equality relation deliberately ignores a field. The standard trait only
// describes representation uniqueness; it cannot discover Eq semantics.
struct SemanticKey {
  std::uint32_t id;
  std::uint32_t diagnostic_tag;

  bool operator==(const SemanticKey& other) const noexcept {
    return id == other.id;
  }
};

static_assert(sizeof(SemanticKey) == 8);
static_assert(std::has_unique_object_representations_v<SemanticKey>);
static_assert(!Hashable<SemanticKey>);

struct SemanticKeyHash {
  std::uint64_t operator()(const SemanticKey& key) const noexcept {
    return Hash<std::uint32_t>{}(key.id);
  }
};

}  // namespace

TEST(Hash, BytesDeterministic) {
  const char* s = "hello world";
  EXPECT_EQ(hash_bytes(s, 11), hash_bytes(s, 11));  // same input -> same hash
}

TEST(Hash, CompositeKeysUseSemanticFieldHashes) {
  using Pair = std::pair<std::string, std::string>;
  const Pair pair_a{"token", "rank"};
  const Pair pair_b{"rank", "token"};
  EXPECT_EQ(Hash<Pair>{}(pair_a), Hash<Pair>{}(pair_a));
  EXPECT_NE(Hash<Pair>{}(pair_a), Hash<Pair>{}(pair_b));

  using Tuple = std::tuple<int, std::string_view, std::uint64_t>;
  using Array = std::array<int, 3>;
  const Tuple tuple{7, "layer", 3};
  const Array array{1, 2, 3};
  EXPECT_EQ(Hash<Tuple>{}(tuple), Hash<Tuple>{}(tuple));
  EXPECT_EQ(Hash<Array>{}(array), Hash<Array>{}(array));
}

TEST(Hash, ValidNoexceptStdHashTypesHaveANaturalBridge) {
  const std::bitset<64> mask{0x12345678u};
  static_assert(Hashable<std::bitset<64>>);
  EXPECT_EQ(Hash<std::bitset<64>>{}(mask), Hash<std::bitset<64>>{}(mask));
}

TEST(Hash, BytesSeedChangesResult) {
  const char* s = "hello";
  EXPECT_NE(hash_bytes(s, 5, 0), hash_bytes(s, 5, 1));
}

TEST(Hash, DefaultStringPolicyCachesTheProcessSeedAndAllowsDeterminism) {
  const std::string_view text = "protocol-key";
  const StringHash first;
  const StringHash second;
  const StringHash deterministic_zero{0};
  const StringHash deterministic_one{1};

  EXPECT_NE(hyle::container::detail::process_hash_seed(), 0u);
  EXPECT_EQ(first(text), second(text));
  EXPECT_EQ(deterministic_zero(text), hash_bytes(text.data(), text.size(), 0));
  EXPECT_EQ(deterministic_one(text), hash_bytes(text.data(), text.size(), 1));
  EXPECT_NE(deterministic_zero(text), deterministic_one(text));
}

TEST(Hash, SipHash24MatchesReferenceVectors) {
  constexpr std::uint64_t k0 = 0x0706050403020100ull;
  constexpr std::uint64_t k1 = 0x0f0e0d0c0b0a0908ull;
  constexpr std::uint64_t expected[] = {
      0x726fdb47dd0e0e31ull, 0x74f839c593dc67fdull,
      0x0d6c8009d9a94f5aull, 0x85676696d7fb7e2dull,
      0xcf2794e0277187b7ull, 0x18765564cd99a68dull,
      0xcbc9466e58fee3ceull, 0xab0200f58b01d137ull,
      0x93f5f5799a932462ull, 0x9e0082df0ba9e4b0ull,
      0x7a5dbbc594ddb9f3ull, 0xf4b32f46226bada7ull,
      0x751e8fbc860ee5fbull, 0x14ea5627c0843d90ull,
      0xf723ca908e7af2eeull, 0xa129ca6149be45e5ull,
      0x3f2acc7f57c29bdbull, 0x699ae9f52cbe4794ull,
      0x4bc1b3f0968dd39cull, 0xbb6dc91da77961bdull,
      0xbed65cf21aa2ee98ull, 0xd0f2cbb02e3b67c7ull,
      0x93536795e3a33e88ull, 0xa80c038ccd5ccec8ull,
      0xb8ad50c6f649af94ull, 0xbce192de8a85b8eaull,
      0x17d835b85bbb15f3ull, 0x2f2e6163076bcfadull,
      0xde4daaaca71dc9a5ull, 0xa6a2506687956571ull,
      0xad87a3535c49ef28ull, 0x32d892fad841c342ull,
      0x7127512f72f27cceull, 0xa7f32346f95978e3ull,
      0x12e0b01abb051238ull, 0x15e034d40fa197aeull,
      0x314dffbe0815a3b4ull, 0x027990f029623981ull,
      0xcadcd4e59ef40c4dull, 0x9abfd8766a33735cull,
      0x0e3ea96b5304a7d0ull, 0xad0c42d6fc585992ull,
      0x187306c89bc215a9ull, 0xd4a60abcf3792b95ull,
      0xf935451de4f21df2ull, 0xa9538f0419755787ull,
      0xdb9acddff56ca510ull, 0xd06c98cd5c0975ebull,
      0xe612a3cb9ecba951ull, 0xc766e62cfcadaf96ull,
      0xee64435a9752fe72ull, 0xa192d576b245165aull,
      0x0a8787bf8ecb74b2ull, 0x81b3e73d20b49b6full,
      0x7fa8220ba3b2eceaull, 0x245731c13ca42499ull,
      0xb78dbfaf3a8d83bdull, 0xea1ad565322a1a0bull,
      0x60e61c23a3795013ull, 0x6606d7e446282b93ull,
      0x6ca4ecb15c5f91e1ull, 0x9f626da15c9625f3ull,
      0xe51b38608ef25f57ull, 0x958a324ceb064572ull};
  std::uint8_t message[sizeof(expected) / sizeof(expected[0])]{};
  for (std::size_t i = 0; i < sizeof(expected) / sizeof(expected[0]); ++i) {
    message[i] = static_cast<std::uint8_t>(i);
    EXPECT_EQ(siphash24(message, i, k0, k1), expected[i]) << "length " << i;
  }
  EXPECT_EQ(siphash24(nullptr, 0, k0, k1), expected[0]);
}

TEST(Hash, HardenedStringPolicySharesOneHeterogeneousEqualityClass) {
  constexpr std::uint64_t k0 = 0x0123456789abcdefull;
  constexpr std::uint64_t k1 = 0xfedcba9876543210ull;
  const HardenedStringHash hash{k0, k1};
  const std::string owned = "attacker-controlled";
  const std::string_view view = owned;

  EXPECT_EQ(hash(owned), hash(view));
  EXPECT_EQ(hash(owned), hash("attacker-controlled"));
  EXPECT_EQ(hash(owned), siphash24(owned.data(), owned.size(), k0, k1));
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
  EXPECT_EQ(seen.size(), 10000u);  // 64-bit table hash: zero collisions expected
}

TEST(Hash, MediumBytePathsAreSeededAndWellDistributed) {
  for (std::size_t length : {17u, 24u, 32u, 48u, 96u, 128u}) {
    std::unordered_set<std::uint64_t> seen;
    for (int i = 0; i < 10000; ++i) {
      std::string value(length, static_cast<char>('a' + i % 23));
      const std::string suffix = std::to_string(i);
      std::copy(suffix.begin(), suffix.end(), value.end() - suffix.size());
      seen.insert(hash_bytes(value.data(), value.size()));
    }
    EXPECT_EQ(seen.size(), 10000u) << "length " << length;

    std::string probe(length, 'm');
    EXPECT_NE(hash_bytes(probe.data(), probe.size(), 0),
              hash_bytes(probe.data(), probe.size(), 1));
  }
}

TEST(Hash, Medium17To32PathAvalanchesEveryInputBit) {
  std::string input(24, 'h');
  const std::uint64_t baseline = hash_bytes(input.data(), input.size());
  for (std::size_t byte = 0; byte < input.size(); ++byte) {
    for (int bit = 0; bit < 8; ++bit) {
      input[byte] ^= static_cast<char>(1u << bit);
      const int changed = std::popcount(
          baseline ^ hash_bytes(input.data(), input.size()));
      EXPECT_GE(changed, 12) << "byte " << byte << ", bit " << bit;
      input[byte] ^= static_cast<char>(1u << bit);
    }
  }
}

// LONG strings take the runtime best-of branch (wyhash, or AVX2 XXH3 past the
// shipping platform's measured crossover).
// Verify its public contract rather than pinning an implementation-specific value:
// process-local repeatability, seed sensitivity, and quality over distinct inputs.
TEST(Hash, LongStringRuntimePathNoCollisions) {
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

TEST(Hash, WideRuntimePathIsRepeatableAndSeedSensitive) {
  const std::string input(4096, 'w');
  const auto first = hash_bytes(input.data(), input.size(), 0);
  EXPECT_EQ(first, hash_bytes(input.data(), input.size(), 0));
  EXPECT_NE(first, hash_bytes(input.data(), input.size(), 1));
}

// The inline-medium <-> long-dispatch boundary must not break determinism: both
// sides (128B inline, 129B dispatcher) hash consistently to themselves.
TEST(Hash, LengthDispatchBoundaryDeterministic) {
  std::string a(128, 'a'), b(129, 'b');
  EXPECT_EQ(hyle::container::hash_bytes(a.data(), 128), hyle::container::hash_bytes(a.data(), 128));
  EXPECT_EQ(hyle::container::hash_bytes(b.data(), 129), hyle::container::hash_bytes(b.data(), 129));
  EXPECT_NE(hyle::container::hash_bytes(a.data(), 128), hyle::container::hash_bytes(b.data(), 129));
}

TEST(Hash, IntAndEnumUsable) {
  enum class Color : int { Red, Green, Blue };
  Hash<Color> h;
  EXPECT_NE(h(Color::Red), h(Color::Blue));
}

TEST(Hash, EqualFloatingPointZerosHaveTheSameHash) {
  Hash<float> hash_float;
  Hash<double> hash_double;
  Hash<long double> hash_long_double;

  EXPECT_EQ(hash_float(0.0f), hash_float(-0.0f));
  EXPECT_EQ(hash_double(0.0), hash_double(-0.0));
  EXPECT_EQ(hash_long_double(0.0L), hash_long_double(-0.0L));
}

TEST(Hash, FloatAndDoubleUseTheInlineCanonicalBitMix) {
  constexpr float float_key = -17.25f;
  constexpr double double_key = 0x1.23456789abcp+42;

  EXPECT_EQ(Hash<float>{}(float_key),
            Hash<std::uint32_t>{}(std::bit_cast<std::uint32_t>(float_key)));
  EXPECT_EQ(Hash<double>{}(double_key),
            Hash<std::uint64_t>{}(std::bit_cast<std::uint64_t>(double_key)));
}

TEST(Hash, PaddingFreeCustomEqualityStillRequiresSemanticHasher) {
  const SemanticKey first{7, 11};
  const SemanticKey second{7, 99};

  EXPECT_EQ(first, second);
  EXPECT_NE(std::bit_cast<std::uint64_t>(first),
            std::bit_cast<std::uint64_t>(second));
  EXPECT_EQ(SemanticKeyHash{}(first), SemanticKeyHash{}(second));
}

// A caller that explicitly chooses BytewiseHash for a representation-equality
// key gets the <=16-byte inline path. Verify its distribution and avalanche.
TEST(Hash, SmallStructInlineQuality) {
  struct Key16 { std::uint64_t a, b; };  // 16 bytes -> inline mix path
  static_assert(sizeof(Key16) == 16);
  BytewiseHash<Key16> h;
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

// The explicit bytewise path for an <=8-byte key uses one inline mix.
TEST(Hash, SmallStruct8BInline) {
  struct Key8 { std::uint32_t x, y; };  // 8 bytes
  BytewiseHash<Key8> h;
  std::unordered_set<std::uint64_t> seen;
  for (std::uint32_t i = 0; i < 1000; ++i) seen.insert(h(Key8{i, i * 2654435761u}));
  EXPECT_EQ(seen.size(), 1000u);  // distinct
}
