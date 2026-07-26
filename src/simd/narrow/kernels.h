#pragma once

#include <cstddef>

#include "hyle/simd/dtype.h"

namespace hyle::simd::detail {

#define HYLE_NARROW_DECL_ONE(T, tag, tier)                                     \
  std::size_t narrow_count_equal_##tag##_##tier(const T *, std::size_t,        \
                                                T) noexcept;                   \
  std::size_t narrow_find_equal_##tag##_##tier(const T *, std::size_t,         \
                                               T) noexcept;                    \
  std::size_t narrow_count_equal_bits_##tag##_##tier(const T *, std::size_t,   \
                                                     T) noexcept;              \
  std::size_t narrow_find_equal_bits_##tag##_##tier(const T *, std::size_t,    \
                                                    T) noexcept;               \
  float narrow_sum_##tag##_##tier(const T *, std::size_t) noexcept;            \
  bool narrow_is_sorted_total_asc_##tag##_##tier(const T *,                    \
                                                 std::size_t) noexcept;        \
  bool narrow_is_sorted_total_desc_##tag##_##tier(const T *,                   \
                                                  std::size_t) noexcept;

#define HYLE_NARROW_DECL_TIER(tier)                                            \
  HYLE_NARROW_DECL_ONE(float16_t, f16, tier)                                   \
  HYLE_NARROW_DECL_ONE(bfloat16_t, bf16, tier)

HYLE_NARROW_DECL_TIER(scalar)
HYLE_NARROW_DECL_TIER(sse4)
HYLE_NARROW_DECL_TIER(avx2)
HYLE_NARROW_DECL_TIER(avx512)
HYLE_NARROW_DECL_TIER(neon)
HYLE_NARROW_DECL_TIER(sve)
HYLE_NARROW_DECL_TIER(rvv)

#undef HYLE_NARROW_DECL_TIER
#undef HYLE_NARROW_DECL_ONE

} // namespace hyle::simd::detail
