#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "hyle/simd/bitpack.h"
#include "src/simd/bitpack/kernels.h"
#include "src/simd/runtime_internal.h"
#include "tests/simd/bitpack_golden.h"

namespace hs = hyle::simd;
namespace bt = hyle::simd::test;

#if defined(HYLE_SIMD_TARGET_SVE)
#define HYLE_SCALABLE_BITPACK(name) hs::detail::name##_sve
constexpr hs::Isa kExpectedBitpackIsa = hs::Isa::kSve;
#elif defined(HYLE_SIMD_TARGET_RVV)
#define HYLE_SCALABLE_BITPACK(name) hs::detail::name##_rvv
constexpr hs::Isa kExpectedBitpackIsa = hs::Isa::kRvv;
#endif

namespace {

template <typename T> struct DirectBitpack;

template <> struct DirectBitpack<std::uint8_t> {
  static void pack(const std::uint8_t *raw, std::size_t count,
                   std::uint8_t *packed, unsigned bits) noexcept {
    HYLE_SCALABLE_BITPACK(pack_u8)(raw, count, packed, bits);
  }
  static void unpack(const std::uint8_t *packed, std::size_t count,
                     std::uint8_t *raw, unsigned bits) noexcept {
    HYLE_SCALABLE_BITPACK(unpack_u8)(packed, count, raw, bits);
  }
};

template <> struct DirectBitpack<std::uint16_t> {
  static void pack(const std::uint16_t *raw, std::size_t count,
                   std::uint16_t *packed, unsigned bits) noexcept {
    HYLE_SCALABLE_BITPACK(pack_u16)(raw, count, packed, bits);
  }
  static void unpack(const std::uint16_t *packed, std::size_t count,
                     std::uint16_t *raw, unsigned bits) noexcept {
    HYLE_SCALABLE_BITPACK(unpack_u16)(packed, count, raw, bits);
  }
};

template <> struct DirectBitpack<std::uint32_t> {
  static void pack(const std::uint32_t *raw, std::size_t count,
                   std::uint32_t *packed, unsigned bits) noexcept {
    HYLE_SCALABLE_BITPACK(pack_u32)(raw, count, packed, bits);
  }
  static void unpack(const std::uint32_t *packed, std::size_t count,
                     std::uint32_t *raw, unsigned bits) noexcept {
    HYLE_SCALABLE_BITPACK(unpack_u32)(packed, count, raw, bits);
  }
};

template <typename T> void check_all_widths() {
  constexpr unsigned W = 8 * sizeof(T);
  for (unsigned bits = 1; bits <= W; ++bits) {
    const T mask = bt::low_mask<T>(bits);
    for (std::size_t count :
         {std::size_t{127}, std::size_t{128}, std::size_t{129},
          std::size_t{263}, std::size_t{4169}}) {
      std::vector<T> raw(count);
      for (std::size_t i = 0; i < raw.size(); ++i) {
        raw[i] = static_cast<T>(bt::sample_value<T>(i + count) & mask);
      }
      std::vector<T> public_packed(hs::packed_size<T>(count, bits));
      std::vector<T> direct_packed(public_packed.size());
      std::vector<T> decoded(count);
      hs::pack(raw.data(), count, public_packed.data(), bits);
      DirectBitpack<T>::pack(raw.data(), count, direct_packed.data(), bits);
      DirectBitpack<T>::unpack(public_packed.data(), count, decoded.data(),
                               bits);
      ASSERT_EQ(direct_packed, public_packed)
          << "lane=" << W << " bits=" << bits << " count=" << count;
      ASSERT_EQ(decoded, raw)
          << "lane=" << W << " bits=" << bits << " count=" << count;
    }
  }
}

template <typename T, std::size_t N>
void check_golden(const std::array<std::uint64_t, N> &expected) {
  static_assert(N == 8 * sizeof(T));
  for (unsigned bits = 1; bits <= N; ++bits) {
    EXPECT_EQ(bt::wire_digest<T>(bits), expected[bits - 1])
        << "lane=" << N << " bits=" << bits;
  }
}

} // namespace

TEST(ScalableBitpackProduct, RuntimeTableOwnsDirectScalableEntries) {
  const hs::Runtime &active = hs::runtime();
  EXPECT_EQ(active.bitpack.isa, kExpectedBitpackIsa);
  EXPECT_NE(active.bitpack.features & hs::Runtime::kFeatureBitpackLogical128,
            0u);
  const hs::Runtime *exact = hs::runtime_for(kExpectedBitpackIsa);
  ASSERT_NE(exact, nullptr);
  EXPECT_EQ(exact->bitpack_ops.pack_u8, &HYLE_SCALABLE_BITPACK(pack_u8));
  EXPECT_EQ(exact->bitpack_ops.unpack_u16, &HYLE_SCALABLE_BITPACK(unpack_u16));
  EXPECT_EQ(exact->bitpack_ops.pack_u32, &HYLE_SCALABLE_BITPACK(pack_u32));
}

TEST(ScalableBitpackProduct, DirectAndPublicAllWidthsPreserveCanonicalBytes) {
  check_all_widths<std::uint8_t>();
  check_all_widths<std::uint16_t>();
  check_all_widths<std::uint32_t>();
}

TEST(ScalableBitpackProduct, CanonicalGoldenBytesIgnoreVlOrVlen) {
  check_golden<std::uint8_t>(bt::kU8WireDigest);
  check_golden<std::uint16_t>(bt::kU16WireDigest);
  check_golden<std::uint32_t>(bt::kU32WireDigest);
}

#undef HYLE_SCALABLE_BITPACK
