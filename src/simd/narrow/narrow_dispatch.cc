#include "hyle/simd/narrow_ops.h"

#include "src/simd/runtime_internal.h"
#include "src/simd/dispatch_internal.h"
#include "src/simd/narrow/kernels.h"

#if defined(HYLE_SIMD_EXACT)
#define HYLE_NARROW_CALL(name, ...)                                            \
  HYLE_SIMD_EXACT_SYMBOL(narrow_##name)(__VA_ARGS__)
#else
#define HYLE_NARROW_CALL(name, ...) runtime().narrow.name(__VA_ARGS__)
#endif

namespace hyle::simd {

#define HYLE_NARROW_DISPATCH(T, tag)                                           \
  std::size_t count_equal(const T *data, std::size_t n, T value) noexcept {    \
    return HYLE_NARROW_CALL(count_equal_##tag, data, n, value);                \
  }                                                                            \
  std::size_t find_equal(const T *data, std::size_t n, T value) noexcept {     \
    return HYLE_NARROW_CALL(find_equal_##tag, data, n, value);                 \
  }                                                                            \
  std::size_t count_equal_bits(const T *data, std::size_t n,                   \
                               T value) noexcept {                             \
    return HYLE_NARROW_CALL(count_equal_bits_##tag, data, n, value);           \
  }                                                                            \
  std::size_t find_equal_bits(const T *data, std::size_t n,                    \
                              T value) noexcept {                              \
    return HYLE_NARROW_CALL(find_equal_bits_##tag, data, n, value);            \
  }                                                                            \
  float sum(const T *data, std::size_t n) noexcept {                           \
    return HYLE_NARROW_CALL(sum_##tag, data, n);                               \
  }                                                                            \
  bool is_sorted_total(const T *data, std::size_t n,                           \
                       bool ascending) noexcept {                              \
    return ascending ? HYLE_NARROW_CALL(is_sorted_total_asc_##tag, data, n)    \
                     : HYLE_NARROW_CALL(is_sorted_total_desc_##tag, data, n);  \
  }

HYLE_NARROW_DISPATCH(float16_t, f16)
HYLE_NARROW_DISPATCH(bfloat16_t, bf16)

#undef HYLE_NARROW_DISPATCH
} // namespace hyle::simd

#undef HYLE_NARROW_CALL
