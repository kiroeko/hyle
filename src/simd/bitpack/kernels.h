#pragma once

#include <cstddef>
#include <cstdint>

namespace hyle::simd::detail {

#define HYLE_BITPACK_TYPES(X)                                                  \
  X(std::uint8_t, u8)                                                          \
  X(std::uint16_t, u16)                                                        \
  X(std::uint32_t, u32)

#define HYLE_BITPACK_DECLARE_ONE(T, tag, suffix)                               \
  void pack_##tag##_##suffix(const T *, std::size_t, T *, unsigned) noexcept;  \
  void unpack_##tag##_##suffix(const T *, std::size_t, T *, unsigned) noexcept;

#define HYLE_BITPACK_DECLARE_scalar(T, tag)                                    \
  HYLE_BITPACK_DECLARE_ONE(T, tag, scalar)
#define HYLE_BITPACK_DECLARE_sse4(T, tag) HYLE_BITPACK_DECLARE_ONE(T, tag, sse4)
#define HYLE_BITPACK_DECLARE_avx2(T, tag) HYLE_BITPACK_DECLARE_ONE(T, tag, avx2)
#define HYLE_BITPACK_DECLARE_avx512(T, tag)                                    \
  HYLE_BITPACK_DECLARE_ONE(T, tag, avx512)
#define HYLE_BITPACK_DECLARE_neon(T, tag) HYLE_BITPACK_DECLARE_ONE(T, tag, neon)
#define HYLE_BITPACK_DECLARE_sve(T, tag) HYLE_BITPACK_DECLARE_ONE(T, tag, sve)
#define HYLE_BITPACK_DECLARE_rvv(T, tag) HYLE_BITPACK_DECLARE_ONE(T, tag, rvv)

HYLE_BITPACK_TYPES(HYLE_BITPACK_DECLARE_scalar)
HYLE_BITPACK_TYPES(HYLE_BITPACK_DECLARE_sse4)
HYLE_BITPACK_TYPES(HYLE_BITPACK_DECLARE_avx2)
HYLE_BITPACK_TYPES(HYLE_BITPACK_DECLARE_avx512)
HYLE_BITPACK_TYPES(HYLE_BITPACK_DECLARE_neon)
HYLE_BITPACK_TYPES(HYLE_BITPACK_DECLARE_sve)
HYLE_BITPACK_TYPES(HYLE_BITPACK_DECLARE_rvv)

#undef HYLE_BITPACK_DECLARE_rvv
#undef HYLE_BITPACK_DECLARE_sve
#undef HYLE_BITPACK_DECLARE_neon
#undef HYLE_BITPACK_DECLARE_avx512
#undef HYLE_BITPACK_DECLARE_avx2
#undef HYLE_BITPACK_DECLARE_sse4
#undef HYLE_BITPACK_DECLARE_scalar
#undef HYLE_BITPACK_DECLARE_ONE

} // namespace hyle::simd::detail
