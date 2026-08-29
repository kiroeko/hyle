#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

#include "hyle/simd/bitpack.h"
#include "hyle/simd/dispatch.h"
#include "src/simd/runtime_internal.h"
#include "tests/simd/bitpack_golden.h"

namespace hs = hyle::simd;
namespace bt = hyle::simd::test;

namespace {

constexpr std::array<std::size_t, 24> kCounts = {
    0,   1,   2,   3,   7,   8,   15,  16,  17,  31,  63,   64,
    100, 127, 128, 129, 200, 255, 256, 257, 511, 512, 1000, 4097};

template <typename T> void check_round_trip(std::size_t count, unsigned bits) {
  constexpr T kGuard = static_cast<T>(0xA5A5A5A5u);
  const T mask = bt::low_mask<T>(bits);
  std::vector<T> raw(count + 2, kGuard);
  for (std::size_t i = 0; i < count; ++i)
    raw[i + 1] = static_cast<T>(bt::sample_value<T>(i) & mask);

  const std::size_t words = hs::packed_size<T>(count, bits);
  std::vector<T> packed(words + 2, kGuard);
  std::vector<T> decoded(count + 2, kGuard);
  hs::pack(std::span<const T>(raw).subspan(1, count),
           std::span<T>(packed).subspan(1, words), bits);
  hs::unpack(std::span<const T>(packed).subspan(1, words),
             std::span<T>(decoded).subspan(1, count), bits);

  EXPECT_EQ(raw.front(), kGuard);
  EXPECT_EQ(raw.back(), kGuard);
  EXPECT_EQ(packed.front(), kGuard);
  EXPECT_EQ(packed.back(), kGuard);
  EXPECT_EQ(decoded.front(), kGuard);
  EXPECT_EQ(decoded.back(), kGuard);
  for (std::size_t i = 0; i < count; ++i) {
    ASSERT_EQ(decoded[i + 1], static_cast<T>(raw[i + 1] & mask))
        << "lane=" << (8 * sizeof(T)) << " bits=" << bits << " count=" << count
        << " i=" << i;
  }

  if (words != 0) {
    constexpr unsigned W = 8 * sizeof(T);
    const unsigned used = static_cast<unsigned>((count * bits) % W);
    if (used != 0) {
      EXPECT_EQ(static_cast<std::uint64_t>(packed[words]) >> used, 0u)
          << "non-zero final padding: bits=" << bits << " count=" << count;
    }
  }
}

template <typename T> void check_all_round_trips() {
  constexpr unsigned W = 8 * sizeof(T);
  for (unsigned bits = 1; bits <= W; ++bits) {
    for (std::size_t count : kCounts)
      check_round_trip<T>(count, bits);
  }
}

template <typename T> struct RuntimeBitpack;

template <> struct RuntimeBitpack<std::uint8_t> {
  static auto pack(const hs::Runtime &runtime) noexcept {
    return runtime.bitpack_ops.pack_u8;
  }
  static auto unpack(const hs::Runtime &runtime) noexcept {
    return runtime.bitpack_ops.unpack_u8;
  }
};

template <> struct RuntimeBitpack<std::uint16_t> {
  static auto pack(const hs::Runtime &runtime) noexcept {
    return runtime.bitpack_ops.pack_u16;
  }
  static auto unpack(const hs::Runtime &runtime) noexcept {
    return runtime.bitpack_ops.unpack_u16;
  }
};

template <> struct RuntimeBitpack<std::uint32_t> {
  static auto pack(const hs::Runtime &runtime) noexcept {
    return runtime.bitpack_ops.pack_u32;
  }
  static auto unpack(const hs::Runtime &runtime) noexcept {
    return runtime.bitpack_ops.unpack_u32;
  }
};

std::vector<const hs::Runtime *> available_runtimes() {
  std::vector<const hs::Runtime *> runtimes;
  for (hs::Isa isa :
       {hs::Isa::kScalar, hs::Isa::kSse42, hs::Isa::kAvx2, hs::Isa::kAvx512,
        hs::Isa::kNeon, hs::Isa::kSve, hs::Isa::kRvv}) {
    if (const hs::Runtime *runtime = hs::runtime_for(isa)) {
      runtimes.push_back(runtime);
    }
  }
  if (runtimes.empty())
    runtimes.push_back(&hs::runtime());
  return runtimes;
}

template <typename T> void check_cross_tier_bytes() {
  constexpr unsigned W = 8 * sizeof(T);
  const auto runtimes = available_runtimes();
  const hs::Runtime &reference = *runtimes.front();
  for (unsigned bits = 1; bits <= W; ++bits) {
    const T mask = bt::low_mask<T>(bits);
    for (std::size_t count :
         {std::size_t{1}, std::size_t{127}, std::size_t{128}, std::size_t{129},
          std::size_t{255}, std::size_t{256}, std::size_t{257},
          std::size_t{1000}}) {
      std::vector<T> raw(count);
      for (std::size_t i = 0; i < count; ++i)
        raw[i] = static_cast<T>(bt::sample_value<T>(i + count) & mask);
      std::vector<T> expected(hs::packed_size<T>(count, bits));
      RuntimeBitpack<T>::pack(reference)(raw.data(), count, expected.data(),
                                         bits);

      for (const hs::Runtime *runtime : runtimes) {
        std::vector<T> actual(expected.size());
        std::vector<T> decoded(count);
        RuntimeBitpack<T>::pack (*runtime)(raw.data(), count, actual.data(),
                                           bits);
        ASSERT_EQ(actual, expected)
            << "lane=" << W << " bits=" << bits << " count=" << count
            << " tier=" << static_cast<unsigned>(runtime->isa);
        RuntimeBitpack<T>::unpack (*runtime)(expected.data(), count,
                                             decoded.data(), bits);
        for (std::size_t i = 0; i < count; ++i) {
          ASSERT_EQ(decoded[i], static_cast<T>(raw[i] & mask))
              << "lane=" << W << " bits=" << bits << " count=" << count
              << " i=" << i << " tier=" << static_cast<unsigned>(runtime->isa);
        }
      }
    }
  }
}

std::vector<std::uint16_t> pack_u16_lane_oracle(const std::uint16_t *raw,
                                                unsigned bits) {
  std::vector<std::uint16_t> packed(bits, 0);
  for (std::size_t pair = 0; pair < 8; ++pair) {
    packed[pair] = static_cast<std::uint16_t>(
        (raw[2 * pair] & 0xFFu) | ((raw[2 * pair + 1] & 0xFFu) << 8));
  }
  const unsigned high_bits = bits - 8;
  for (std::size_t column = 0; column < high_bits; ++column) {
    std::uint16_t word = 0;
    const std::size_t groups = 15 / high_bits;
    for (std::size_t group = 0; group < groups; ++group) {
      const std::size_t raw_index = column + high_bits * group;
      word = static_cast<std::uint16_t>(
          word |
          (((raw[raw_index] >> 8) & bt::low_mask<std::uint16_t>(high_bits))
           << (high_bits * group)));
    }
    word = static_cast<std::uint16_t>(word |
                                      (((raw[15] >> (8 + column)) & 1u) << 15));
    packed[8 + column] = word;
  }
  return packed;
}

void check_u16_corrected_layout(unsigned bits) {
  constexpr std::size_t kLogicalLanes = 8;
  constexpr std::size_t kCount = 128;
  const std::uint16_t mask = bt::low_mask<std::uint16_t>(bits);
  std::vector<std::uint16_t> raw(kCount);
  for (std::size_t row = 0; row < 16; ++row) {
    for (std::size_t lane = 0; lane < kLogicalLanes; ++lane) {
      raw[row * kLogicalLanes + lane] = static_cast<std::uint16_t>(
          (row * 0x6D5u + lane * 0x9E3u + (row ^ lane) * 0x101u) & mask);
    }
  }

  std::vector<std::uint16_t> expected(bits * kLogicalLanes);
  std::array<std::uint16_t, 16> lane_raw{};
  for (std::size_t lane = 0; lane < kLogicalLanes; ++lane) {
    for (std::size_t row = 0; row < 16; ++row)
      lane_raw[row] = raw[row * kLogicalLanes + lane];
    const auto lane_packed = pack_u16_lane_oracle(lane_raw.data(), bits);
    for (std::size_t word = 0; word < bits; ++word)
      expected[word * kLogicalLanes + lane] = lane_packed[word];
  }

  std::vector<std::uint16_t> actual(expected.size());
  std::vector<std::uint16_t> decoded(kCount);
  hs::pack(raw.data(), raw.size(), actual.data(), bits);
  ASSERT_EQ(actual, expected) << "bits=" << bits;
  hs::unpack(expected.data(), raw.size(), decoded.data(), bits);
  EXPECT_EQ(decoded, raw) << "bits=" << bits;
}

template <typename T, std::size_t N>
void check_wire_digests(const std::array<std::uint64_t, N> &expected) {
  static_assert(N == 8 * sizeof(T));
  for (unsigned bits = 1; bits <= N; ++bits) {
    EXPECT_EQ(bt::wire_digest<T>(bits), expected[bits - 1])
        << "lane=" << N << " bits=" << bits;
  }
}

} // namespace

TEST(BitPack, U8AllWidthsAndBoundaries) {
  check_all_round_trips<std::uint8_t>();
}
TEST(BitPack, U16AllWidthsAndBoundaries) {
  check_all_round_trips<std::uint16_t>();
}
TEST(BitPack, U32AllWidthsAndBoundaries) {
  check_all_round_trips<std::uint32_t>();
}

TEST(BitPack, PackedSizeIsOverflowSafeCeil) {
  EXPECT_EQ(hs::packed_size<std::uint8_t>(0, 3), 0u);
  EXPECT_EQ(hs::packed_size<std::uint8_t>(8, 1), 1u);
  EXPECT_EQ(hs::packed_size<std::uint8_t>(3, 3), 2u);
  EXPECT_EQ(hs::packed_size<std::uint16_t>(16, 5), 5u);
  EXPECT_EQ(hs::packed_size<std::uint32_t>(32, 1), 1u);
  const auto max = std::numeric_limits<std::size_t>::max();
  EXPECT_EQ(hs::packed_size<std::uint8_t>(max, 8), max);
  EXPECT_EQ(hs::packed_size<std::uint16_t>(max, 16), max);
  EXPECT_EQ(hs::packed_size<std::uint32_t>(max, 32), max);
}

TEST(BitPack, EveryAvailableTierUsesCanonicalBytesU8) {
  check_cross_tier_bytes<std::uint8_t>();
}
TEST(BitPack, EveryAvailableTierUsesCanonicalBytesU16) {
  check_cross_tier_bytes<std::uint16_t>();
}
TEST(BitPack, EveryAvailableTierUsesCanonicalBytesU32) {
  check_cross_tier_bytes<std::uint32_t>();
}

TEST(BitPack, IndependentU16Layout11) { check_u16_corrected_layout(11); }
TEST(BitPack, IndependentU16Layout13) { check_u16_corrected_layout(13); }

TEST(BitPack, WireFormatVersionAndGoldenDigests) {
  EXPECT_EQ(hs::kBitpackFormatVersion, 1u);
  check_wire_digests<std::uint8_t>(bt::kU8WireDigest);
  check_wire_digests<std::uint16_t>(bt::kU16WireDigest);
  check_wire_digests<std::uint32_t>(bt::kU32WireDigest);
}

TEST(BitPack, ActiveVariantMetadataIsSane) {
  const auto target = hs::selected_target(hs::KernelFamily::kBitpack);
  EXPECT_TRUE(target.isa == hs::Isa::kScalar || target.isa == hs::Isa::kSse42 ||
              target.isa == hs::Isa::kAvx2 || target.isa == hs::Isa::kAvx512 ||
              target.isa == hs::Isa::kNeon || target.isa == hs::Isa::kSve ||
              target.isa == hs::Isa::kRvv);
  EXPECT_TRUE(target.has(hs::TargetFeature::kBitpackLogical128));
}
