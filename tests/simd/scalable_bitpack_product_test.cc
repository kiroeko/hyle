#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "hyle/simd/bitpack.h"
#include "hyle/simd/runtime.h"
#include "src/simd/bitpack/kernels.h"

namespace hs = hyle::simd;

#if defined(HYLE_SIMD_TARGET_SVE)
#define HYLE_SCALABLE_BITPACK(name) hs::detail::name##_sve
constexpr hs::Isa kExpectedBitpackIsa = hs::Isa::kSve;
#elif defined(HYLE_SIMD_TARGET_RVV)
#define HYLE_SCALABLE_BITPACK(name) hs::detail::name##_rvv
constexpr hs::Isa kExpectedBitpackIsa = hs::Isa::kRvv;
#endif

namespace hyle::simd::detail {
void pack_u8_ref128(const std::uint8_t*, std::size_t, std::uint8_t*, unsigned) noexcept;
void pack_u16_ref128(const std::uint16_t*, std::size_t, std::uint16_t*, unsigned) noexcept;
void pack_u32_ref128(const std::uint32_t*, std::size_t, std::uint32_t*, unsigned) noexcept;
}  // namespace hyle::simd::detail

namespace {

template <typename T>
void direct_pack(const T* raw, std::size_t count, T* packed, unsigned bits);
template <>
void direct_pack<std::uint8_t>(const std::uint8_t* raw, std::size_t count,
                               std::uint8_t* packed, unsigned bits) {
  HYLE_SCALABLE_BITPACK(pack_u8)(raw, count, packed, bits);
}
template <>
void direct_pack<std::uint16_t>(const std::uint16_t* raw, std::size_t count,
                                std::uint16_t* packed, unsigned bits) {
  HYLE_SCALABLE_BITPACK(pack_u16)(raw, count, packed, bits);
}
template <>
void direct_pack<std::uint32_t>(const std::uint32_t* raw, std::size_t count,
                                std::uint32_t* packed, unsigned bits) {
  HYLE_SCALABLE_BITPACK(pack_u32)(raw, count, packed, bits);
}

template <typename T>
void direct_unpack(const T* packed, std::size_t count, T* raw, unsigned bits);
template <>
void direct_unpack<std::uint8_t>(const std::uint8_t* packed, std::size_t count,
                                 std::uint8_t* raw, unsigned bits) {
  HYLE_SCALABLE_BITPACK(unpack_u8)(packed, count, raw, bits);
}
template <>
void direct_unpack<std::uint16_t>(const std::uint16_t* packed, std::size_t count,
                                  std::uint16_t* raw, unsigned bits) {
  HYLE_SCALABLE_BITPACK(unpack_u16)(packed, count, raw, bits);
}
template <>
void direct_unpack<std::uint32_t>(const std::uint32_t* packed, std::size_t count,
                                  std::uint32_t* raw, unsigned bits) {
  HYLE_SCALABLE_BITPACK(unpack_u32)(packed, count, raw, bits);
}

template <typename T>
void reference_pack(const T* raw, std::size_t count, T* packed, unsigned bits);
template <>
void reference_pack<std::uint8_t>(const std::uint8_t* raw, std::size_t count,
                                  std::uint8_t* packed, unsigned bits) {
  hs::detail::pack_u8_ref128(raw, count, packed, bits);
}
template <>
void reference_pack<std::uint16_t>(const std::uint16_t* raw, std::size_t count,
                                   std::uint16_t* packed, unsigned bits) {
  hs::detail::pack_u16_ref128(raw, count, packed, bits);
}
template <>
void reference_pack<std::uint32_t>(const std::uint32_t* raw, std::size_t count,
                                   std::uint32_t* packed, unsigned bits) {
  hs::detail::pack_u32_ref128(raw, count, packed, bits);
}

template <typename T>
void check_all_widths(unsigned max_bits, bool public_api) {
  constexpr std::size_t kCount = 256;  // two fixed 128-value logical blocks
  for (unsigned bits = 1; bits <= max_bits; ++bits) {
    const std::uint64_t mask = bits == 8 * sizeof(T)
                                   ? static_cast<std::uint64_t>(~T{0})
                                   : (std::uint64_t{1} << bits) - 1;
    std::vector<T> raw(kCount);
    for (std::size_t i = 0; i < raw.size(); ++i)
      raw[i] = static_cast<T>((i * 0x9E3779B1u + bits * 0x85EBCA77u + (i >> 3)) & mask);
    std::vector<T> expected(hs::packed_size<T>(raw.size(), bits));
    reference_pack(raw.data(), raw.size(), expected.data(), bits);
    std::vector<T> packed(expected.size());
    std::vector<T> decoded(raw.size());
    if (public_api) {
      hs::pack(raw.data(), raw.size(), packed.data(), bits);
      hs::unpack(packed.data(), raw.size(), decoded.data(), bits);
    } else {
      direct_pack(raw.data(), raw.size(), packed.data(), bits);
      direct_unpack(packed.data(), raw.size(), decoded.data(), bits);
    }
    ASSERT_EQ(packed, expected) << "lane bits=" << 8 * sizeof(T) << " bits=" << bits;
    ASSERT_EQ(decoded, raw) << "lane bits=" << 8 * sizeof(T) << " bits=" << bits;
  }
}

}  // namespace

TEST(ScalableBitpackProduct, DirectKernelsUseFixed128Layout) {
  check_all_widths<std::uint8_t>(8, false);
  check_all_widths<std::uint16_t>(16, false);
  check_all_widths<std::uint32_t>(32, false);
}

TEST(ScalableBitpackProduct, RuntimeAndPublicApiUseScalableTier) {
  EXPECT_EQ(hs::runtime().bitpack.isa, kExpectedBitpackIsa);
  const hs::Runtime* exact = hs::runtime_for(kExpectedBitpackIsa);
  ASSERT_NE(exact, nullptr);
  EXPECT_EQ(exact->bitpack.isa, kExpectedBitpackIsa);
  check_all_widths<std::uint8_t>(8, true);
  check_all_widths<std::uint16_t>(16, true);
  check_all_widths<std::uint32_t>(32, true);
}

#undef HYLE_SCALABLE_BITPACK
