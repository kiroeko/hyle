#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/simd/vec.h"

namespace hyle::simd::native {
namespace detail {

template <int k, class D, class V>
static HYLE_FORCE_INLINE V scan_slide_up(D d, V v) noexcept {
  static_assert(k >= 1 && (k & (k - 1)) == 0);
  if constexpr (k == 1) return Slide1Up(d, v);
  else return scan_slide_up<k / 2>(d, scan_slide_up<k / 2>(d, v));
}

template <int k, class D, class V>
static HYLE_FORCE_INLINE V scan_step(D d, V v) noexcept {
  using T = TFromD<D>;
  if constexpr (k >= D::kMaxLanes) return v;
  else {
    if (static_cast<std::size_t>(k) >= Lanes(d)) return v;
    if constexpr (sizeof(T) == 4 && D::kMaxLanes >= 8)
      return Add(v, SlideUpLanes(d, v, static_cast<std::size_t>(k)));
    else
      return Add(v, scan_slide_up<k>(d, v));
  }
}

template <class D, class V>
static HYLE_FORCE_INLINE V scan_vector(D d, V v) noexcept {
  v = scan_step<1>(d, v);
  v = scan_step<2>(d, v);
  v = scan_step<4>(d, v);
  v = scan_step<8>(d, v);
  v = scan_step<16>(d, v);
  v = scan_step<32>(d, v);
  return v;
}

template <class D, class V>
static HYLE_FORCE_INLINE TFromD<D> scan_last(D d, V v) noexcept {
  alignas(64) TFromD<D> lanes[D::kMaxLanes];
  StoreU(v, d, lanes);
  return lanes[Lanes(d) - 1];
}

template <typename T>
static HYLE_FORCE_INLINE void inclusive_scan_impl(T* data, std::size_t n) noexcept {
  if (n == 0) return;
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  T carry{};
  std::size_t i = 0;
  for (; i + width <= n; i += width) {
    auto v = Add(scan_vector(d, LoadU(d, data + i)), Set(d, carry));
    StoreU(v, d, data + i);
    carry = scan_last(d, v);
  }
  for (; i < n; ++i) {
    carry = static_cast<T>(carry + data[i]);
    data[i] = carry;
  }
}

template <typename T>
static HYLE_FORCE_INLINE void exclusive_scan_impl(T* data, std::size_t n) noexcept {
  if (n == 0) return;
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  T carry{};
  std::size_t i = 0;
  for (; i + width <= n; i += width) {
    const auto raw = LoadU(d, data + i);
    const auto inc = scan_vector(d, raw);
    const auto exc = Add(Slide1Up(d, inc), Set(d, carry));
    StoreU(exc, d, data + i);
    carry = static_cast<T>(scan_last(d, exc) + scan_last(d, raw));
  }
  for (; i < n; ++i) {
    const T value = data[i];
    data[i] = carry;
    carry = static_cast<T>(carry + value);
  }
}

}  // namespace detail

#define HYLE_NATIVE_SCAN_OVERLOAD(T)                                                   \
  static HYLE_FORCE_INLINE void inclusive_scan(T* data, std::size_t n) noexcept {      \
    detail::inclusive_scan_impl(data, n);                                               \
  }                                                                                     \
  static HYLE_FORCE_INLINE void exclusive_scan(T* data, std::size_t n) noexcept {      \
    detail::exclusive_scan_impl(data, n);                                               \
  }

HYLE_NATIVE_SCAN_OVERLOAD(std::uint32_t)
HYLE_NATIVE_SCAN_OVERLOAD(std::int32_t)
HYLE_NATIVE_SCAN_OVERLOAD(std::uint64_t)
HYLE_NATIVE_SCAN_OVERLOAD(std::int64_t)
HYLE_NATIVE_SCAN_OVERLOAD(float)
HYLE_NATIVE_SCAN_OVERLOAD(double)

#undef HYLE_NATIVE_SCAN_OVERLOAD

}  // namespace hyle::simd::native
