#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "hyle/simd/bitpack.h"
#include "hyle/simd/dispatch.h"
#include "hyle/simd/runtime.h"

// bitpack_test -- the borrowed Highway bit_pack (whatever tier is live: avx512/avx2/neon/scalar)
// must ROUND-TRIP: pack `count` values (each masked to `bits` low bits) then unpack must recover
// every value exactly. This is the bit-layout self-consistency proof -- the block region uses
// Highway's Pack<kBits> and the sub-block tail uses the shared scalar bit-stream codec, so a same-
// host round trip pins both halves. Coverage: every bits (1..8 / 1..16 / 1..32), counts spanning
// 0 / 1 / tiny / exact-block / block+tail / many-blocks, all three lane types.
//
// Also a cross-check that packed_size<T> is exactly the number of packed words the codec uses:
// unpack reads packed[0 .. packed_size) so an under-count would read OOB (ASan would catch it) --
// we size the packed buffer to packed_size and rely on ASan/valgrind in CI for the OOB check.

using hyle::simd::Isa;
using hyle::simd::pack;
using hyle::simd::packed_size;
using hyle::simd::unpack;

namespace {

template <typename T>
void round_trip(std::size_t count, unsigned bits, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<std::uint64_t> dist;
  const std::uint64_t mask =
      (bits >= 64) ? ~std::uint64_t{0} : ((std::uint64_t{1} << bits) - 1);

  std::vector<T> raw(count);
  for (std::size_t i = 0; i < count; ++i)
    raw[i] = static_cast<T>(dist(rng) & mask);

  const std::size_t words = packed_size<T>(count, bits);
  std::vector<T> packed(words, T{0});
  std::vector<T> out(count, T{0});

  pack(raw.data(), count, packed.data(), bits);
  unpack(packed.data(), count, out.data(), bits);

  for (std::size_t i = 0; i < count; ++i)
    ASSERT_EQ(raw[i], out[i]) << "type" << (8 * sizeof(T)) << " bits=" << bits
                              << " count=" << count << " i=" << i;
}

// Count ladder: 0, tiny (sub-block), exact block multiples, block+tail, many blocks. The vector
// block size is Lanes(d)*B (B=8*sizeof(T)); we cannot know Lanes at compile time, but these counts
// (up to a few thousand) exercise both the block loop and the tail for any real vector width.
const std::vector<std::size_t> kCounts = {0,   1,   2,    3,    7,    8,   15,  16,
                                          17,  31,  63,   64,   100,  127, 128, 200,
                                          255, 256, 257,  511,  512,  1000, 2048, 4097};

template <typename T>
void run_all_for_type(unsigned max_bits) {
  std::uint64_t seed = 0xD1B54A32D192ED03ULL;
  for (unsigned bits = 1; bits <= max_bits; ++bits)
    for (std::size_t count : kCounts) round_trip<T>(count, bits, ++seed);
}

}  // namespace

TEST(BitPack, U8) { run_all_for_type<std::uint8_t>(8); }
TEST(BitPack, U16) { run_all_for_type<std::uint16_t>(16); }
TEST(BitPack, U32) { run_all_for_type<std::uint32_t>(32); }

// Boundary values: all-zero, all-ones (masked), alternating -- pin the mask/straddle edges.
TEST(BitPack, ExtremesU32) {
  for (unsigned bits = 1; bits <= 32; ++bits) {
    const std::uint32_t hi =
        (bits >= 32) ? ~std::uint32_t{0} : ((std::uint32_t{1} << bits) - 1);
    std::vector<std::uint32_t> patterns;
    const std::size_t n = 300;
    for (std::size_t i = 0; i < n; ++i) {
      std::uint32_t v = (i % 3 == 0) ? 0u : (i % 3 == 1) ? hi : (0xAAAAAAAAu & hi);
      patterns.push_back(v);
    }
    std::vector<std::uint32_t> packed(packed_size<std::uint32_t>(n, bits), 0);
    std::vector<std::uint32_t> out(n, 0);
    pack(patterns.data(), n, packed.data(), bits);
    unpack(packed.data(), n, out.data(), bits);
    for (std::size_t i = 0; i < n; ++i)
      ASSERT_EQ(patterns[i], out[i]) << "extremes bits=" << bits << " i=" << i;
  }
}

// packed_size is the standard ceil(count*bits / (8*sizeof(T))).
TEST(BitPack, PackedSizeFormula) {
  EXPECT_EQ(packed_size<std::uint8_t>(0, 3), 0u);
  EXPECT_EQ(packed_size<std::uint8_t>(8, 1), 1u);   // 8*1/8
  EXPECT_EQ(packed_size<std::uint8_t>(8, 8), 8u);   // identity
  EXPECT_EQ(packed_size<std::uint8_t>(3, 3), 2u);   // 9 bits -> 2 bytes
  EXPECT_EQ(packed_size<std::uint16_t>(16, 5), 5u);  // 80 bits / 16
  EXPECT_EQ(packed_size<std::uint32_t>(32, 1), 1u);  // 32 bits / 32
  EXPECT_EQ(packed_size<std::uint32_t>(10, 32), 10u);
}

std::vector<std::uint16_t> pack_lane_oracle(const std::uint16_t* raw, unsigned bits) {
  std::vector<std::uint16_t> packed(bits, 0);
  for (std::size_t pair = 0; pair < 8; ++pair) {
    packed[pair] = static_cast<std::uint16_t>((raw[2 * pair] & 0xFFu) |
                                              ((raw[2 * pair + 1] & 0xFFu) << 8));
  }
  if (bits == 11) {
    for (std::size_t column = 0; column < 3; ++column) {
      std::uint16_t word = 0;
      for (std::size_t group = 0; group < 5; ++group) {
        const std::size_t raw_index = column + 3 * group;
        word = static_cast<std::uint16_t>(word |
            (((raw[raw_index] >> 8) & 0x7u) << (3 * group)));
      }
      word = static_cast<std::uint16_t>(word | (((raw[15] >> (8 + column)) & 1u) << 15));
      packed[8 + column] = word;
    }
  } else {
    for (std::size_t column = 0; column < 5; ++column) {
      std::uint16_t word = static_cast<std::uint16_t>(
          ((raw[column] >> 8) & 0x1Fu) |
          (((raw[column + 5] >> 8) & 0x1Fu) << 5) |
          (((raw[column + 10] >> 8) & 0x1Fu) << 10) |
          (((raw[15] >> (8 + column)) & 1u) << 15));
      packed[8 + column] = word;
    }
  }
  return packed;
}

void verify_u16_block_layout(unsigned bits) {
  const Isa tier = hyle::simd::runtime().bitpack.isa;
  const std::size_t lanes = tier == Isa::kAvx512 ? 32 : tier == Isa::kAvx2 ? 16 :
                            tier == Isa::kNeon ? 8 : 1;
  const std::size_t count = lanes * 16;
  const std::uint16_t mask = static_cast<std::uint16_t>((std::uint32_t{1} << bits) - 1);
  std::vector<std::uint16_t> raw(count);
  for (std::size_t group = 0; group < 16; ++group)
    for (std::size_t lane = 0; lane < lanes; ++lane)
      raw[group * lanes + lane] = static_cast<std::uint16_t>(
          (group * 0x6D5u + lane * 0x9E3u + (group ^ lane) * 0x101u) & mask);

  std::vector<std::uint16_t> expected(bits * lanes);
  std::vector<std::uint16_t> lane_raw(16);
  for (std::size_t lane = 0; lane < lanes; ++lane) {
    for (std::size_t group = 0; group < 16; ++group)
      lane_raw[group] = raw[group * lanes + lane];
    const auto lane_packed = pack_lane_oracle(lane_raw.data(), bits);
    for (std::size_t word = 0; word < bits; ++word)
      expected[word * lanes + lane] = lane_packed[word];
  }

  std::vector<std::uint16_t> actual(bits * lanes);
  std::vector<std::uint16_t> decoded(count);
  pack(raw.data(), count, actual.data(), bits);
  ASSERT_EQ(actual, expected) << "bits=" << bits;
  unpack(expected.data(), count, decoded.data(), bits);
  ASSERT_EQ(decoded, raw) << "bits=" << bits;
}

TEST(BitPack, U16IndependentLayout11) { verify_u16_block_layout(11); }
TEST(BitPack, U16IndependentLayout13) { verify_u16_block_layout(13); }

// Full legal domains for the two widths whose upstream Highway inverse is wrong. Offset every
// buffer by one element and guard both ends so this also pins unaligned access and exact bounds.
void exhaustive_u16(unsigned bits) {
  const std::size_t count = std::size_t{1} << bits;
  const std::size_t words = packed_size<std::uint16_t>(count, bits);
  constexpr std::uint16_t kGuard = 0xA55Au;
  std::vector<std::uint16_t> raw(count + 2, kGuard);
  std::vector<std::uint16_t> packed(words + 2, kGuard);
  std::vector<std::uint16_t> out(count + 2, kGuard);
  for (std::size_t i = 0; i < count; ++i) raw[i + 1] = static_cast<std::uint16_t>(i);

  pack(raw.data() + 1, count, packed.data() + 1, bits);
  unpack(packed.data() + 1, count, out.data() + 1, bits);

  EXPECT_EQ(raw.front(), kGuard);
  EXPECT_EQ(raw.back(), kGuard);
  EXPECT_EQ(packed.front(), kGuard);
  EXPECT_EQ(packed.back(), kGuard);
  EXPECT_EQ(out.front(), kGuard);
  EXPECT_EQ(out.back(), kGuard);
  for (std::size_t i = 0; i < count; ++i)
    ASSERT_EQ(out[i + 1], static_cast<std::uint16_t>(i)) << "bits=" << bits << " i=" << i;
}

TEST(BitPack, U16Exhaustive11) { exhaustive_u16(11); }
TEST(BitPack, U16Exhaustive13) { exhaustive_u16(13); }

// The dispatcher reports a sane tier for this host (smoke check that it is wired).
TEST(BitPack, ActiveVariantIsSane) {
  const Isa v = hyle::simd::runtime().bitpack.isa;
  EXPECT_TRUE(v == Isa::kAvx512 || v == Isa::kAvx2 || v == Isa::kNeon || v == Isa::kScalar);
}
