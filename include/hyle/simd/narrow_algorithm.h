#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "hyle/simd/narrow_float.h"

// Target-bound whole-range algorithms for f16/bf16 storage. These keep the
// D-tag explicit, so a source-built kernel can inline one chosen backend
// without exposing a target-specific vector ABI. Use the dispatched dtype/ops
// APIs when a prebuilt unknown-CPU library boundary is required.

namespace hyle::simd {
namespace detail {

template <typename Narrow>
HYLE_FORCE_INLINE float narrow_to_float(Narrow value) noexcept {
  if constexpr (std::is_same_v<Narrow, float16_t>)
    return f16_to_f32(value);
  else
    return bf16_to_f32(value);
}

} // namespace detail

// IEEE numeric equality: NaN never matches and either signed zero matches the
// other.
template <class DN>
HYLE_FORCE_INLINE std::size_t
CountEqualNarrow(DN dn, const typename DN::T *data, std::size_t n,
                 typename DN::T value) noexcept {
  using DF = typename DN::FloatTag;
  const DF df;
  const std::size_t width = Lanes(dn);
  const float needle_scalar = detail::narrow_to_float(value);
  const auto needle = Set(df, needle_scalar);
  std::size_t total = 0;
  std::size_t i = 0;
  for (; i + 4 * width <= n; i += 4 * width) {
    total += CountTrue(
        df, Eq(needle, PromoteNarrowTo(df, dn, LoadNarrowU(dn, data + i))));
    total += CountTrue(
        df,
        Eq(needle, PromoteNarrowTo(df, dn, LoadNarrowU(dn, data + i + width))));
    total += CountTrue(
        df, Eq(needle,
               PromoteNarrowTo(df, dn, LoadNarrowU(dn, data + i + 2 * width))));
    total += CountTrue(
        df, Eq(needle,
               PromoteNarrowTo(df, dn, LoadNarrowU(dn, data + i + 3 * width))));
  }
  for (; i + width <= n; i += width)
    total += CountTrue(
        df, Eq(needle, PromoteNarrowTo(df, dn, LoadNarrowU(dn, data + i))));
  for (; i < n; ++i)
    total += detail::narrow_to_float(data[i]) == needle_scalar;
  return total;
}

template <class DN>
HYLE_FORCE_INLINE std::size_t FindEqualNarrow(DN dn, const typename DN::T *data,
                                              std::size_t n,
                                              typename DN::T value) noexcept {
  using DF = typename DN::FloatTag;
  const DF df;
  const std::size_t width = Lanes(dn);
  const float needle_scalar = detail::narrow_to_float(value);
  const auto needle = Set(df, needle_scalar);
  std::size_t i = 0;
  for (; i + 2 * width <= n; i += 2 * width) {
    const auto m0 =
        Eq(needle, PromoteNarrowTo(df, dn, LoadNarrowU(dn, data + i)));
    const auto m1 =
        Eq(needle, PromoteNarrowTo(df, dn, LoadNarrowU(dn, data + i + width)));
    if (AllFalse(df, Or(m0, m1)))
      continue;
    const auto p0 = FindFirstTrue(df, m0);
    if (p0 >= 0)
      return i + static_cast<std::size_t>(p0);
    return i + width + FindKnownFirstTrue(df, m1);
  }
  for (; i + width <= n; i += width) {
    const auto match =
        Eq(needle, PromoteNarrowTo(df, dn, LoadNarrowU(dn, data + i)));
    const auto pos = FindFirstTrue(df, match);
    if (pos >= 0)
      return i + static_cast<std::size_t>(pos);
  }
  for (; i < n; ++i)
    if (detail::narrow_to_float(data[i]) == needle_scalar)
      return i;
  return n;
}

// Exact representation search: distinguishes signed zero and individual NaN
// payloads.
template <class DN>
HYLE_FORCE_INLINE std::size_t CountBitsNarrow(DN dn, const typename DN::T *data,
                                              std::size_t n,
                                              typename DN::T value) noexcept {
  const typename DN::StorageTag db;
  const std::size_t width = Lanes(dn);
  const auto needle = Set(db, value.bits);
  std::size_t total = 0;
  std::size_t i = 0;
  for (; i + 4 * width <= n; i += 4 * width) {
    total += CountTrue(db, Eq(needle, LoadNarrowU(dn, data + i)));
    total += CountTrue(db, Eq(needle, LoadNarrowU(dn, data + i + width)));
    total += CountTrue(db, Eq(needle, LoadNarrowU(dn, data + i + 2 * width)));
    total += CountTrue(db, Eq(needle, LoadNarrowU(dn, data + i + 3 * width)));
  }
  for (; i + width <= n; i += width)
    total += CountTrue(db, Eq(needle, LoadNarrowU(dn, data + i)));
  for (; i < n; ++i)
    total += data[i].bits == value.bits;
  return total;
}

template <class DN>
HYLE_FORCE_INLINE std::size_t FindBitsNarrow(DN dn, const typename DN::T *data,
                                             std::size_t n,
                                             typename DN::T value) noexcept {
  const typename DN::StorageTag db;
  const std::size_t width = Lanes(dn);
  const auto needle = Set(db, value.bits);
  std::size_t i = 0;
  for (; i + 2 * width <= n; i += 2 * width) {
    const auto m0 = Eq(needle, LoadNarrowU(dn, data + i));
    const auto m1 = Eq(needle, LoadNarrowU(dn, data + i + width));
    if (AllFalse(db, Or(m0, m1)))
      continue;
    const auto p0 = FindFirstTrue(db, m0);
    if (p0 >= 0)
      return i + static_cast<std::size_t>(p0);
    return i + width + FindKnownFirstTrue(db, m1);
  }
  for (; i + width <= n; i += width) {
    const auto match = Eq(needle, LoadNarrowU(dn, data + i));
    const auto pos = FindFirstTrue(db, match);
    if (pos >= 0)
      return i + static_cast<std::size_t>(pos);
  }
  for (; i < n; ++i)
    if (data[i].bits == value.bits)
      return i;
  return n;
}

template <class DN>
HYLE_FORCE_INLINE bool
IsSortedTotalOrderNarrow(DN dn, const typename DN::T *data, std::size_t n,
                         bool ascending = true) noexcept {
  if (n < 2)
    return true;
  const typename DN::StorageTag db;
  const std::size_t width = Lanes(dn);
  const std::size_t pairs = n - 1;
  std::size_t i = 0;
  for (; i + 2 * width <= pairs; i += 2 * width) {
    const auto current0 = TotalOrderKeyNarrow(dn, LoadNarrowU(dn, data + i));
    const auto next0 = TotalOrderKeyNarrow(dn, LoadNarrowU(dn, data + i + 1));
    const auto current1 =
        TotalOrderKeyNarrow(dn, LoadNarrowU(dn, data + i + width));
    const auto next1 =
        TotalOrderKeyNarrow(dn, LoadNarrowU(dn, data + i + width + 1));
    const auto bad0 = ascending ? Lt(next0, current0) : Gt(next0, current0);
    const auto bad1 = ascending ? Lt(next1, current1) : Gt(next1, current1);
    if (!AllFalse(db, Or(bad0, bad1)))
      return false;
  }
  for (; i + width <= pairs; i += width) {
    const auto current = TotalOrderKeyNarrow(dn, LoadNarrowU(dn, data + i));
    const auto next = TotalOrderKeyNarrow(dn, LoadNarrowU(dn, data + i + 1));
    const auto bad = ascending ? Lt(next, current) : Gt(next, current);
    if (!AllFalse(db, bad))
      return false;
  }
  for (; i < pairs; ++i) {
    const auto current = narrow_total_order_key(data[i]);
    const auto next = narrow_total_order_key(data[i + 1]);
    if (ascending ? next < current : next > current)
      return false;
  }
  return true;
}

// f32 accumulation. Like other SIMD reductions, the exact association depends
// on vector width.
template <class DN>
HYLE_FORCE_INLINE float SumNarrow(DN dn, const typename DN::T *data,
                                  std::size_t n) noexcept {
  using DF = typename DN::FloatTag;
  const DF df;
  const std::size_t width = Lanes(dn);
  auto acc0 = Zero(df);
  auto acc1 = Zero(df);
  auto acc2 = Zero(df);
  auto acc3 = Zero(df);
  std::size_t i = 0;
  for (; i + 4 * width <= n; i += 4 * width) {
    acc0 = Add(acc0, PromoteNarrowTo(df, dn, LoadNarrowU(dn, data + i)));
    acc1 =
        Add(acc1, PromoteNarrowTo(df, dn, LoadNarrowU(dn, data + i + width)));
    acc2 = Add(acc2,
               PromoteNarrowTo(df, dn, LoadNarrowU(dn, data + i + 2 * width)));
    acc3 = Add(acc3,
               PromoteNarrowTo(df, dn, LoadNarrowU(dn, data + i + 3 * width)));
  }
  for (; i + width <= n; i += width)
    acc0 = Add(acc0, PromoteNarrowTo(df, dn, LoadNarrowU(dn, data + i)));
  if (i != n)
    acc0 = Add(acc0, PromoteNarrowTo(df, dn, LoadNarrowN(dn, data + i, n - i)));
  return ReduceSum(df, Add(Add(acc0, acc1), Add(acc2, acc3)));
}

} // namespace hyle::simd
