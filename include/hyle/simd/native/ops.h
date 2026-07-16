#pragma once

#include <cstddef>

#include "hyle/simd/vec.h"

// Target-bound whole-buffer operations. These functions have internal linkage on purpose: the
// implementation selected by vec.h is different in each target TU, so no target-dependent symbol
// crosses a translation-unit or shared-library boundary.

namespace hyle::simd::native {
namespace detail {

template <typename T>
static HYLE_FORCE_INLINE T dot_impl(const T* x, const T* y, std::size_t n) noexcept {
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  auto acc0 = Zero(d);
  auto acc1 = Zero(d);
  auto acc2 = Zero(d);
  auto acc3 = Zero(d);
  std::size_t i = 0;
  const std::size_t width4 = width * 4;
  for (; i + width4 <= n; i += width4) {
    acc0 = MulAdd(LoadU(d, x + i), LoadU(d, y + i), acc0);
    acc1 = MulAdd(LoadU(d, x + i + width), LoadU(d, y + i + width), acc1);
    acc2 = MulAdd(LoadU(d, x + i + width * 2), LoadU(d, y + i + width * 2), acc2);
    acc3 = MulAdd(LoadU(d, x + i + width * 3), LoadU(d, y + i + width * 3), acc3);
  }
  for (; i + width <= n; i += width)
    acc0 = MulAdd(LoadU(d, x + i), LoadU(d, y + i), acc0);
  T sum = ReduceSum(d, Add(Add(acc0, acc1), Add(acc2, acc3)));
  for (; i < n; ++i) sum += x[i] * y[i];
  return sum;
}

template <typename T>
static HYLE_FORCE_INLINE void saxpy_impl(T a, const T* x, T* y,
                                         std::size_t n) noexcept {
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  const auto va = Set(d, a);
  std::size_t i = 0;
  for (; i + width <= n; i += width) {
    StoreU(MulAdd(va, LoadU(d, x + i), LoadU(d, y + i)), d, y + i);
  }
  for (; i < n; ++i) y[i] = a * x[i] + y[i];
}

template <typename T>
static HYLE_FORCE_INLINE void add_impl(const T* x, const T* y, T* out,
                                       std::size_t n) noexcept {
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  std::size_t i = 0;
  for (; i + width <= n; i += width)
    StoreU(Add(LoadU(d, x + i), LoadU(d, y + i)), d, out + i);
  for (; i < n; ++i) out[i] = x[i] + y[i];
}

}  // namespace detail

static HYLE_FORCE_INLINE float dot(const float* x, const float* y,
                                   std::size_t n) noexcept {
  return detail::dot_impl(x, y, n);
}
static HYLE_FORCE_INLINE double dot(const double* x, const double* y,
                                    std::size_t n) noexcept {
  return detail::dot_impl(x, y, n);
}
static HYLE_FORCE_INLINE void saxpy(float a, const float* x, float* y,
                                    std::size_t n) noexcept {
  detail::saxpy_impl(a, x, y, n);
}
static HYLE_FORCE_INLINE void saxpy(double a, const double* x, double* y,
                                    std::size_t n) noexcept {
  detail::saxpy_impl(a, x, y, n);
}
static HYLE_FORCE_INLINE void add(const float* x, const float* y, float* out,
                                  std::size_t n) noexcept {
  detail::add_impl(x, y, out, n);
}
static HYLE_FORCE_INLINE void add(const double* x, const double* y, double* out,
                                  std::size_t n) noexcept {
  detail::add_impl(x, y, out, n);
}

}  // namespace hyle::simd::native
