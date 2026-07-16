// Shared batch-loop helpers for HYLE-ported xsimd product math TUs.
// Include after: HYLE_SIMD_TARGET_*, hyle/simd/vec.h, math_xsimd_hyle-inl.h.
// Caller must already be outside any namespace (this file opens hyle::simd::detail).

#pragma once

#include <cmath>
#include <cstddef>

namespace hyle::simd::detail {
namespace xsimd_body {
using namespace hyle::simd;
using namespace xsimd_hyle;

template <class T, class Fn, class Sc>
HYLE_FORCE_INLINE void unary_loop(const T* src, T* dst, std::size_t n, Fn fn, Sc sc) noexcept {
  const ScalableTag<T> d;
  const std::size_t w = Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w) StoreU(fn(d, LoadU(d, src + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = sc(src[i]);
}

template <class T, class Fn, class Sc>
HYLE_FORCE_INLINE void binary_loop(const T* a, const T* b, T* dst, std::size_t n, Fn fn,
                                   Sc sc) noexcept {
  const ScalableTag<T> d;
  const std::size_t w = Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w)
    StoreU(fn(d, LoadU(d, a + i), LoadU(d, b + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = sc(a[i], b[i]);
}

}  // namespace xsimd_body
}  // namespace hyle::simd::detail
