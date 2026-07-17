#include "hyle/simd/search.h"

#include "hyle/simd/runtime.h"
#include "src/simd/dispatch_internal.h"
#include "src/simd/search/kernels.h"

#if defined(HYLE_SIMD_EXACT)
#define HYLE_SEARCH_CALL(name, ...) HYLE_SIMD_EXACT_SYMBOL(name)(__VA_ARGS__)
#else
#define HYLE_SEARCH_CALL(name, ...) runtime().search.name(__VA_ARGS__)
#endif

namespace hyle::simd {

#define HYLE_SEARCH_DISPATCH(T, tag)                                          \
  std::size_t count_equal(const T* data, std::size_t n, T value) noexcept {   \
    return HYLE_SEARCH_CALL(count_equal_##tag, data, n, value);              \
  }                                                                           \
  std::size_t find_equal(const T* data, std::size_t n, T value) noexcept {   \
    return HYLE_SEARCH_CALL(find_equal_##tag, data, n, value);               \
  }                                                                           \
  T min_value(const T* data, std::size_t n) noexcept {                       \
    return HYLE_SEARCH_CALL(min_value_##tag, data, n);                       \
  }                                                                           \
  T max_value(const T* data, std::size_t n) noexcept {                       \
    return HYLE_SEARCH_CALL(max_value_##tag, data, n);                       \
  }                                                                           \
  void minmax_value(const T* data, std::size_t n, T* out_min,                \
                    T* out_max) noexcept {                                    \
    HYLE_SEARCH_CALL(minmax_value_##tag, data, n, out_min, out_max);         \
  }                                                                           \
  bool is_sorted(const T* data, std::size_t n, bool ascending) noexcept {    \
    return ascending ? HYLE_SEARCH_CALL(is_sorted_asc_##tag, data, n)        \
                     : HYLE_SEARCH_CALL(is_sorted_desc_##tag, data, n);      \
  }

HYLE_SEARCH_DISPATCH(std::int8_t, i8)
HYLE_SEARCH_DISPATCH(std::uint8_t, u8)
HYLE_SEARCH_DISPATCH(std::int16_t, i16)
HYLE_SEARCH_DISPATCH(std::uint16_t, u16)
HYLE_SEARCH_DISPATCH(std::int32_t, i32)
HYLE_SEARCH_DISPATCH(std::uint32_t, u32)
HYLE_SEARCH_DISPATCH(std::int64_t, i64)
HYLE_SEARCH_DISPATCH(std::uint64_t, u64)
HYLE_SEARCH_DISPATCH(float, f32)
HYLE_SEARCH_DISPATCH(double, f64)

#undef HYLE_SEARCH_DISPATCH
}  // namespace hyle::simd

#undef HYLE_SEARCH_CALL
