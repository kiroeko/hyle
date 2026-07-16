#include <cstddef>
#include <cstdint>

#include "hyle/simd/native/search.h"
#include "src/simd/search/kernels.h"

#ifndef HYLE_SIMD_VARIANT_SUFFIX
#error "define HYLE_SIMD_VARIANT_SUFFIX before including search/kernel_impl.h"
#endif

#define HYLE_SEARCH_CAT_(a, b) a##b
#define HYLE_SEARCH_CAT(a, b) HYLE_SEARCH_CAT_(a, b)

namespace hyle::simd::detail {

#define HYLE_SEARCH_DEF(T, tag, unused)                                      \
  std::size_t HYLE_SEARCH_CAT(count_equal_##tag##_,                          \
                              HYLE_SIMD_VARIANT_SUFFIX)(                     \
      const T* data, std::size_t n, T value) noexcept {                      \
    return native::count_equal(data, n, value);                              \
  }                                                                          \
  std::size_t HYLE_SEARCH_CAT(find_equal_##tag##_,                           \
                              HYLE_SIMD_VARIANT_SUFFIX)(                     \
      const T* data, std::size_t n, T value) noexcept {                      \
    return native::find_equal(data, n, value);                               \
  }                                                                          \
  T HYLE_SEARCH_CAT(min_value_##tag##_, HYLE_SIMD_VARIANT_SUFFIX)(           \
      const T* data, std::size_t n) noexcept {                               \
    return native::min_value(data, n);                                       \
  }                                                                          \
  T HYLE_SEARCH_CAT(max_value_##tag##_, HYLE_SIMD_VARIANT_SUFFIX)(           \
      const T* data, std::size_t n) noexcept {                               \
    return native::max_value(data, n);                                       \
  }                                                                          \
  void HYLE_SEARCH_CAT(minmax_value_##tag##_, HYLE_SIMD_VARIANT_SUFFIX)(     \
      const T* data, std::size_t n, T* out_min, T* out_max) noexcept {       \
    native::minmax_value(data, n, out_min, out_max);                         \
  }                                                                          \
  bool HYLE_SEARCH_CAT(is_sorted_asc_##tag##_, HYLE_SIMD_VARIANT_SUFFIX)(    \
      const T* data, std::size_t n) noexcept {                               \
    return native::is_sorted_ascending(data, n);                             \
  }                                                                          \
  bool HYLE_SEARCH_CAT(is_sorted_desc_##tag##_, HYLE_SIMD_VARIANT_SUFFIX)(   \
      const T* data, std::size_t n) noexcept {                               \
    return native::is_sorted_descending(data, n);                            \
  }

HYLE_SEARCH_DEF(std::int8_t, i8, unused)
HYLE_SEARCH_DEF(std::uint8_t, u8, unused)
HYLE_SEARCH_DEF(std::int16_t, i16, unused)
HYLE_SEARCH_DEF(std::uint16_t, u16, unused)
HYLE_SEARCH_DEF(std::int32_t, i32, unused)
HYLE_SEARCH_DEF(std::uint32_t, u32, unused)
HYLE_SEARCH_DEF(std::int64_t, i64, unused)
HYLE_SEARCH_DEF(std::uint64_t, u64, unused)
HYLE_SEARCH_DEF(float, f32, unused)
HYLE_SEARCH_DEF(double, f64, unused)

#undef HYLE_SEARCH_DEF
}  // namespace hyle::simd::detail

#undef HYLE_SEARCH_CAT
#undef HYLE_SEARCH_CAT_
