#include <cstddef>

#include "hyle/simd/narrow_algorithm.h"
#include "src/simd/narrow/kernels.h"

#ifndef HYLE_SIMD_VARIANT_SUFFIX
#error "define HYLE_SIMD_VARIANT_SUFFIX before including narrow/kernel_impl.h"
#endif

#define HYLE_NARROW_CAT_(a, b) a##b
#define HYLE_NARROW_CAT(a, b) HYLE_NARROW_CAT_(a, b)

namespace hyle::simd::detail {

#define HYLE_NARROW_DEF(T, tag, NarrowTag)                                     \
  std::size_t HYLE_NARROW_CAT(narrow_count_equal_##tag##_,                     \
                              HYLE_SIMD_VARIANT_SUFFIX)(                       \
      const T *data, std::size_t n, T value) noexcept {                        \
    const NarrowTag<ScalableTag<float>> d;                                     \
    return CountEqualNarrow(d, data, n, value);                                \
  }                                                                            \
  std::size_t HYLE_NARROW_CAT(narrow_find_equal_##tag##_,                      \
                              HYLE_SIMD_VARIANT_SUFFIX)(                       \
      const T *data, std::size_t n, T value) noexcept {                        \
    const NarrowTag<ScalableTag<float>> d;                                     \
    return FindEqualNarrow(d, data, n, value);                                 \
  }                                                                            \
  std::size_t HYLE_NARROW_CAT(narrow_count_equal_bits_##tag##_,                \
                              HYLE_SIMD_VARIANT_SUFFIX)(                       \
      const T *data, std::size_t n, T value) noexcept {                        \
    const NarrowTag<ScalableTag<float>> d;                                     \
    return CountBitsNarrow(d, data, n, value);                                 \
  }                                                                            \
  std::size_t HYLE_NARROW_CAT(narrow_find_equal_bits_##tag##_,                 \
                              HYLE_SIMD_VARIANT_SUFFIX)(                       \
      const T *data, std::size_t n, T value) noexcept {                        \
    const NarrowTag<ScalableTag<float>> d;                                     \
    return FindBitsNarrow(d, data, n, value);                                  \
  }                                                                            \
  float HYLE_NARROW_CAT(narrow_sum_##tag##_, HYLE_SIMD_VARIANT_SUFFIX)(        \
      const T *data, std::size_t n) noexcept {                                 \
    const NarrowTag<ScalableTag<float>> d;                                     \
    return SumNarrow(d, data, n);                                              \
  }                                                                            \
  bool HYLE_NARROW_CAT(narrow_is_sorted_total_asc_##tag##_,                    \
                       HYLE_SIMD_VARIANT_SUFFIX)(const T *data,                \
                                                 std::size_t n) noexcept {     \
    const NarrowTag<ScalableTag<float>> d;                                     \
    return IsSortedTotalOrderNarrow(d, data, n, true);                         \
  }                                                                            \
  bool HYLE_NARROW_CAT(narrow_is_sorted_total_desc_##tag##_,                   \
                       HYLE_SIMD_VARIANT_SUFFIX)(const T *data,                \
                                                 std::size_t n) noexcept {     \
    const NarrowTag<ScalableTag<float>> d;                                     \
    return IsSortedTotalOrderNarrow(d, data, n, false);                        \
  }

HYLE_NARROW_DEF(float16_t, f16, F16Tag)
HYLE_NARROW_DEF(bfloat16_t, bf16, BF16Tag)

#undef HYLE_NARROW_DEF
} // namespace hyle::simd::detail

#undef HYLE_NARROW_CAT
#undef HYLE_NARROW_CAT_
