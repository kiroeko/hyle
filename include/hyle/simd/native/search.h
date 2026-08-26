#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "hyle/core/compiler.h"
#include "hyle/simd/vec.h"

namespace hyle::simd::native {
namespace detail {

template <typename T>
constexpr T min_identity() noexcept {
  if constexpr (std::numeric_limits<T>::has_infinity)
    return std::numeric_limits<T>::infinity();
  else
    return std::numeric_limits<T>::max();
}

template <typename T>
constexpr T max_identity() noexcept {
  if constexpr (std::numeric_limits<T>::has_infinity)
    return -std::numeric_limits<T>::infinity();
  else
    return std::numeric_limits<T>::lowest();
}

template <typename T>
static HYLE_FORCE_INLINE std::size_t count_equal_impl(
    const T* data, std::size_t n, T value) noexcept {
  using D = ScalableTag<T>;
  const D d;
  const std::size_t width = Lanes(d);
  const auto needle = Set(d, value);
  std::size_t total = 0;
  std::size_t i = 0;
  if constexpr (sizeof(T) >= 4) {
    // Accumulate compare masks in vector lanes and reduce once per large
    // chunk. CountTrue per input vector would serialize movemask+popcount on
    // x86 and was ~20% behind Highway for f32. Compare masks are -1 in the
    // signed lane domain, so subtracting them adds one for each match.
    const RebindToSigned<D> di;
    constexpr std::size_t kVectorsPerChunk = 1u << 20;
    while (i + 4 * width <= n) {
      auto acc0 = Zero(di);
      auto acc1 = Zero(di);
      auto acc2 = Zero(di);
      auto acc3 = Zero(di);
      std::size_t vectors = 0;
      for (; i + 4 * width <= n && vectors < kVectorsPerChunk;
           i += 4 * width, vectors += 4) {
        acc0 = Sub(acc0, BitCast(di, VecFromMask(
                                      d, Eq(needle, LoadU(d, data + i)))));
        acc1 = Sub(acc1, BitCast(di, VecFromMask(
                                      d, Eq(needle, LoadU(d, data + i + width)))));
        acc2 = Sub(acc2, BitCast(
                              di, VecFromMask(
                                      d, Eq(needle, LoadU(d, data + i + 2 * width)))));
        acc3 = Sub(acc3, BitCast(
                              di, VecFromMask(
                                      d, Eq(needle, LoadU(d, data + i + 3 * width)))));
      }
      total += static_cast<std::size_t>(
          ReduceSum(di, Add(Add(acc0, acc1), Add(acc2, acc3))));
    }
  } else {
    for (; i + 4 * width <= n; i += 4 * width) {
      total += CountTrue(d, Eq(needle, LoadU(d, data + i)));
      total += CountTrue(d, Eq(needle, LoadU(d, data + i + width)));
      total += CountTrue(d, Eq(needle, LoadU(d, data + i + 2 * width)));
      total += CountTrue(d, Eq(needle, LoadU(d, data + i + 3 * width)));
    }
  }
  for (; i + width <= n; i += width)
    total += CountTrue(d, Eq(needle, LoadU(d, data + i)));
  for (; i < n; ++i) total += data[i] == value;
  return total;
}

template <typename T>
static HYLE_FORCE_INLINE std::size_t find_equal_impl(
    const T* data, std::size_t n, T value) noexcept {
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  const auto needle = Set(d, value);
  std::size_t i = 0;
  for (; i + 2 * width <= n; i += 2 * width) {
    const auto m0 = Eq(needle, LoadU(d, data + i));
    const auto m1 = Eq(needle, LoadU(d, data + i + width));
    if (AllFalse(d, Or(m0, m1))) continue;
    const auto p0 = FindFirstTrue(d, m0);
    if (p0 >= 0) return i + static_cast<std::size_t>(p0);
    return i + width + FindKnownFirstTrue(d, m1);
  }
  for (; i + width <= n; i += width) {
    const auto m = Eq(needle, LoadU(d, data + i));
    const auto pos = FindFirstTrue(d, m);
    if (pos >= 0) return i + static_cast<std::size_t>(pos);
  }
  for (; i < n; ++i)
    if (data[i] == value) return i;
  return n;
}

template <typename T>
static HYLE_FORCE_INLINE T min_value_impl(const T* data,
                                          std::size_t n) noexcept {
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  const auto identity = Set(d, min_identity<T>());
  auto acc0 = identity;
  auto acc1 = identity;
  auto acc2 = identity;
  auto acc3 = identity;
  auto nan_mask = FirstN(d, 0);
  std::size_t i = 0;
  for (; i + 4 * width <= n; i += 4 * width) {
    const auto v0 = LoadU(d, data + i);
    const auto v1 = LoadU(d, data + i + width);
    const auto v2 = LoadU(d, data + i + 2 * width);
    const auto v3 = LoadU(d, data + i + 3 * width);
    if constexpr (std::is_floating_point_v<T>)
      nan_mask = Or(nan_mask, Or(Or(IsNaN(v0), IsNaN(v1)),
                                 Or(IsNaN(v2), IsNaN(v3))));
    acc0 = Min(acc0, v0);
    acc1 = Min(acc1, v1);
    acc2 = Min(acc2, v2);
    acc3 = Min(acc3, v3);
  }
  for (; i + width <= n; i += width) {
    const auto v = LoadU(d, data + i);
    if constexpr (std::is_floating_point_v<T>)
      nan_mask = Or(nan_mask, IsNaN(v));
    acc0 = Min(acc0, v);
  }
  T result = ReduceMin(d, Min(Min(acc0, acc1), Min(acc2, acc3)));
  bool has_nan = false;
  if constexpr (std::is_floating_point_v<T>) has_nan = !AllFalse(d, nan_mask);
  for (; i < n; ++i) {
    if constexpr (std::is_floating_point_v<T>) {
      if (std::isnan(data[i])) has_nan = true;
    }
    if (data[i] < result) result = data[i];
  }
  if constexpr (std::is_floating_point_v<T>)
    if (has_nan) return std::numeric_limits<T>::quiet_NaN();
  return result;
}

template <typename T>
static HYLE_FORCE_INLINE T max_value_impl(const T* data,
                                          std::size_t n) noexcept {
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  const auto identity = Set(d, max_identity<T>());
  auto acc0 = identity;
  auto acc1 = identity;
  auto acc2 = identity;
  auto acc3 = identity;
  auto nan_mask = FirstN(d, 0);
  std::size_t i = 0;
  for (; i + 4 * width <= n; i += 4 * width) {
    const auto v0 = LoadU(d, data + i);
    const auto v1 = LoadU(d, data + i + width);
    const auto v2 = LoadU(d, data + i + 2 * width);
    const auto v3 = LoadU(d, data + i + 3 * width);
    if constexpr (std::is_floating_point_v<T>)
      nan_mask = Or(nan_mask, Or(Or(IsNaN(v0), IsNaN(v1)),
                                 Or(IsNaN(v2), IsNaN(v3))));
    acc0 = Max(acc0, v0);
    acc1 = Max(acc1, v1);
    acc2 = Max(acc2, v2);
    acc3 = Max(acc3, v3);
  }
  for (; i + width <= n; i += width) {
    const auto v = LoadU(d, data + i);
    if constexpr (std::is_floating_point_v<T>)
      nan_mask = Or(nan_mask, IsNaN(v));
    acc0 = Max(acc0, v);
  }
  T result = ReduceMax(d, Max(Max(acc0, acc1), Max(acc2, acc3)));
  bool has_nan = false;
  if constexpr (std::is_floating_point_v<T>) has_nan = !AllFalse(d, nan_mask);
  for (; i < n; ++i) {
    if constexpr (std::is_floating_point_v<T>) {
      if (std::isnan(data[i])) has_nan = true;
    }
    if (data[i] > result) result = data[i];
  }
  if constexpr (std::is_floating_point_v<T>)
    if (has_nan) return std::numeric_limits<T>::quiet_NaN();
  return result;
}

template <typename T>
static HYLE_FORCE_INLINE void minmax_value_impl(const T* data, std::size_t n,
                                                T* out_min,
                                                T* out_max) noexcept {
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  const auto min_identity_v = Set(d, min_identity<T>());
  const auto max_identity_v = Set(d, max_identity<T>());
  auto min0 = min_identity_v;
  auto min1 = min_identity_v;
  auto min2 = min_identity_v;
  auto min3 = min_identity_v;
  auto max0 = max_identity_v;
  auto max1 = max_identity_v;
  auto max2 = max_identity_v;
  auto max3 = max_identity_v;
  auto nan_mask = FirstN(d, 0);
  std::size_t i = 0;
  for (; i + 4 * width <= n; i += 4 * width) {
    const auto v0 = LoadU(d, data + i);
    const auto v1 = LoadU(d, data + i + width);
    const auto v2 = LoadU(d, data + i + 2 * width);
    const auto v3 = LoadU(d, data + i + 3 * width);
    if constexpr (std::is_floating_point_v<T>)
      nan_mask = Or(nan_mask, Or(Or(IsNaN(v0), IsNaN(v1)),
                                 Or(IsNaN(v2), IsNaN(v3))));
    min0 = Min(min0, v0);
    min1 = Min(min1, v1);
    min2 = Min(min2, v2);
    min3 = Min(min3, v3);
    max0 = Max(max0, v0);
    max1 = Max(max1, v1);
    max2 = Max(max2, v2);
    max3 = Max(max3, v3);
  }
  for (; i + width <= n; i += width) {
    const auto v = LoadU(d, data + i);
    if constexpr (std::is_floating_point_v<T>)
      nan_mask = Or(nan_mask, IsNaN(v));
    min0 = Min(min0, v);
    max0 = Max(max0, v);
  }
  T min_result = ReduceMin(d, Min(Min(min0, min1), Min(min2, min3)));
  T max_result = ReduceMax(d, Max(Max(max0, max1), Max(max2, max3)));
  bool has_nan = false;
  if constexpr (std::is_floating_point_v<T>) has_nan = !AllFalse(d, nan_mask);
  for (; i < n; ++i) {
    if constexpr (std::is_floating_point_v<T>) {
      if (std::isnan(data[i])) has_nan = true;
    }
    if (data[i] < min_result) min_result = data[i];
    if (data[i] > max_result) max_result = data[i];
  }
  if constexpr (std::is_floating_point_v<T>) {
    if (has_nan) min_result = max_result = std::numeric_limits<T>::quiet_NaN();
  }
  *out_min = min_result;
  *out_max = max_result;
}

template <bool kAscending, typename T>
static HYLE_FORCE_INLINE bool is_sorted_impl(const T* data,
                                             std::size_t n) noexcept {
  if (n < 2) return true;
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  const std::size_t pairs = n - 1;
  std::size_t i = 0;
  for (; i + 2 * width <= pairs; i += 2 * width) {
    const auto current0 = LoadU(d, data + i);
    const auto next0 = LoadU(d, data + i + 1);
    const auto current1 = LoadU(d, data + i + width);
    const auto next1 = LoadU(d, data + i + width + 1);
    const auto bad0 = kAscending ? Lt(next0, current0) : Gt(next0, current0);
    const auto bad1 = kAscending ? Lt(next1, current1) : Gt(next1, current1);
    if (!AllFalse(d, Or(bad0, bad1))) return false;
  }
  if (i + width <= pairs) {
    const auto current = LoadU(d, data + i);
    const auto next = LoadU(d, data + i + 1);
    const auto bad = kAscending ? Lt(next, current) : Gt(next, current);
    if (!AllFalse(d, bad)) return false;
    i += width;
  }
  for (; i < pairs; ++i) {
    if (kAscending ? data[i + 1] < data[i] : data[i + 1] > data[i])
      return false;
  }
  return true;
}

}  // namespace detail

#define HYLE_NATIVE_SEARCH_OVERLOAD(T)                                         \
  static HYLE_FORCE_INLINE std::size_t count_equal(const T* data,              \
                                                    std::size_t n,              \
                                                    T value) noexcept {         \
    return detail::count_equal_impl(data, n, value);                           \
  }                                                                             \
  static HYLE_FORCE_INLINE std::size_t find_equal(const T* data,               \
                                                   std::size_t n,               \
                                                   T value) noexcept {          \
    return detail::find_equal_impl(data, n, value);                            \
  }                                                                             \
  static HYLE_FORCE_INLINE T min_value(const T* data, std::size_t n) noexcept { \
    return detail::min_value_impl(data, n);                                    \
  }                                                                             \
  static HYLE_FORCE_INLINE T max_value(const T* data, std::size_t n) noexcept { \
    return detail::max_value_impl(data, n);                                    \
  }                                                                             \
  static HYLE_FORCE_INLINE void minmax_value(const T* data, std::size_t n,      \
                                              T* out_min,                       \
                                              T* out_max) noexcept {            \
    detail::minmax_value_impl(data, n, out_min, out_max);                       \
  }                                                                             \
  static HYLE_FORCE_INLINE bool is_sorted_ascending(const T* data,              \
                                                     std::size_t n) noexcept {  \
    return detail::is_sorted_impl<true>(data, n);                              \
  }                                                                             \
  static HYLE_FORCE_INLINE bool is_sorted_descending(const T* data,             \
                                                      std::size_t n) noexcept { \
    return detail::is_sorted_impl<false>(data, n);                             \
  }                                                                             \
  static HYLE_FORCE_INLINE bool is_sorted(const T* data, std::size_t n,         \
                                           bool ascending = true) noexcept {    \
    return ascending ? is_sorted_ascending(data, n)                            \
                     : is_sorted_descending(data, n);                          \
  }

HYLE_NATIVE_SEARCH_OVERLOAD(std::int8_t)
HYLE_NATIVE_SEARCH_OVERLOAD(std::uint8_t)
HYLE_NATIVE_SEARCH_OVERLOAD(std::int16_t)
HYLE_NATIVE_SEARCH_OVERLOAD(std::uint16_t)
HYLE_NATIVE_SEARCH_OVERLOAD(std::int32_t)
HYLE_NATIVE_SEARCH_OVERLOAD(std::uint32_t)
HYLE_NATIVE_SEARCH_OVERLOAD(std::int64_t)
HYLE_NATIVE_SEARCH_OVERLOAD(std::uint64_t)
HYLE_NATIVE_SEARCH_OVERLOAD(float)
HYLE_NATIVE_SEARCH_OVERLOAD(double)

#undef HYLE_NATIVE_SEARCH_OVERLOAD

}  // namespace hyle::simd::native
