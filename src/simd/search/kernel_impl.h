// Direct-Highway per-target contiguous search kernels. HYLE owns the public
// pointer/count and floating-point contracts; Highway owns target lowering.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "hwy/highway.h"
#include "hwy/contrib/algo/count-inl.h"
#include "hwy/contrib/algo/find-inl.h"
#include "hwy/contrib/algo/is_sorted-inl.h"
#include "hwy/contrib/algo/minmax-inl.h"

#include "src/simd/search/kernels.h"

#ifndef HYLE_SIMD_VARIANT_SUFFIX
#error "define HYLE_SIMD_VARIANT_SUFFIX before including search/kernel_impl.h"
#endif

#define HYLE_SEARCH_CAT_(a, b) a##b
#define HYLE_SEARCH_CAT(a, b) HYLE_SEARCH_CAT_(a, b)
#define HYLE_SEARCH_FN(name) HYLE_SEARCH_CAT(name, HYLE_SIMD_VARIANT_SUFFIX)

namespace hyle::simd::detail {
namespace {

namespace hn = hwy::HWY_NAMESPACE;

template <bool kMin, class V>
HWY_INLINE auto preferred_zero(V value) noexcept {
  const hn::DFromV<V> d;
  const auto zero = hn::Eq(value, hn::Zero(d));
  if constexpr (kMin) {
    return hn::And(zero, hn::IsNegative(value));
  } else {
    return hn::And(zero, hn::Not(hn::IsNegative(value)));
  }
}

struct ZeroSigns {
  bool negative;
  bool positive;
};

template <typename T>
HWY_INLINE ZeroSigns zero_signs(const T* data, std::size_t n) noexcept {
  static_assert(std::is_floating_point_v<T>);
  const hn::ScalableTag<T> d;
  const std::size_t lanes = hn::Lanes(d);
  auto negative = hn::FirstN(d, 0);
  auto positive = hn::FirstN(d, 0);
  std::size_t i = 0;
  for (; lanes <= n - i; i += lanes) {
    const auto value = hn::LoadU(d, data + i);
    negative = hn::Or(negative, preferred_zero<true>(value));
    positive = hn::Or(positive, preferred_zero<false>(value));
  }
  bool has_negative = !hn::AllFalse(d, negative);
  bool has_positive = !hn::AllFalse(d, positive);
  for (; i < n; ++i) {
    if (data[i] != T{0}) continue;
    has_negative = has_negative || std::signbit(data[i]);
    has_positive = has_positive || !std::signbit(data[i]);
  }
  return {has_negative, has_positive};
}

template <bool kMin, class V>
HWY_INLINE auto better(V candidate, V current) noexcept {
  if constexpr (kMin) {
    return hn::Lt(candidate, current);
  } else {
    return hn::Gt(candidate, current);
  }
}

template <bool kMin, typename T>
HWY_INLINE bool scalar_better(T candidate, T current) noexcept {
  if constexpr (kMin) {
    return candidate < current ||
           (candidate == T{0} && current == T{0} &&
            std::signbit(candidate) && !std::signbit(current));
  } else {
    return candidate > current ||
           (candidate == T{0} && current == T{0} &&
            !std::signbit(candidate) && std::signbit(current));
  }
}

template <bool kMin, typename T>
HWY_INLINE T propagating_extreme(const T* data, std::size_t n) noexcept {
  static_assert(std::is_floating_point_v<T>);
  const hn::ScalableTag<T> d;
  const std::size_t lanes = hn::Lanes(d);
  const T identity = kMin ? hwy::PositiveInfOrHighestValue<T>()
                          : hwy::NegativeInfOrLowestValue<T>();
  const auto identity_v = hn::Set(d, identity);
  auto acc0 = identity_v;
  auto acc1 = identity_v;
  auto acc2 = identity_v;
  auto acc3 = identity_v;
  auto nan = hn::FirstN(d, 0);

  std::size_t i = 0;
  const std::size_t lanes4 = 4 * lanes;
  for (; lanes4 <= n - i; i += lanes4) {
    const auto v0 = hn::LoadU(d, data + i);
    const auto v1 = hn::LoadU(d, data + i + lanes);
    const auto v2 = hn::LoadU(d, data + i + 2 * lanes);
    const auto v3 = hn::LoadU(d, data + i + 3 * lanes);
    nan = hn::Or(nan, hn::Or(hn::Or(hn::IsNaN(v0), hn::IsNaN(v1)),
                            hn::Or(hn::IsNaN(v2), hn::IsNaN(v3))));
    if constexpr (kMin) {
      acc0 = hn::Min(acc0, v0);
      acc1 = hn::Min(acc1, v1);
      acc2 = hn::Min(acc2, v2);
      acc3 = hn::Min(acc3, v3);
    } else {
      acc0 = hn::Max(acc0, v0);
      acc1 = hn::Max(acc1, v1);
      acc2 = hn::Max(acc2, v2);
      acc3 = hn::Max(acc3, v3);
    }
  }

  auto acc = kMin ? hn::Min(hn::Min(acc0, acc1), hn::Min(acc2, acc3))
                  : hn::Max(hn::Max(acc0, acc1), hn::Max(acc2, acc3));
  for (; lanes <= n - i; i += lanes) {
    const auto v = hn::LoadU(d, data + i);
    nan = hn::Or(nan, hn::IsNaN(v));
    if constexpr (kMin) {
      acc = hn::Min(acc, v);
    } else {
      acc = hn::Max(acc, v);
    }
  }

  T result = kMin ? hn::ReduceMin(d, acc) : hn::ReduceMax(d, acc);
  for (; i < n; ++i) {
    if (std::isnan(data[i])) {
      return std::numeric_limits<T>::quiet_NaN();
    }
    if (scalar_better<kMin>(data[i], result)) result = data[i];
  }
  if (!hn::AllFalse(d, nan)) return std::numeric_limits<T>::quiet_NaN();
  if (result == T{0}) {
    // Signed zero is relevant only when the numeric result is zero. Keep the
    // ordinary non-zero reduction free of per-vector sign-mask work and pay
    // for this second linear pass only on that uncommon semantic edge.
    const ZeroSigns signs = zero_signs(data, n);
    const bool has_preferred = kMin ? signs.negative : signs.positive;
    return std::copysign(T{0}, kMin ? (has_preferred ? T{-1} : T{1})
                                    : (has_preferred ? T{1} : T{-1}));
  }
  return result;
}

template <bool kMin, typename T>
HWY_INLINE std::size_t floating_index_of_extreme(const T* data,
                                                 std::size_t n) noexcept {
  static_assert(std::is_floating_point_v<T>);
  if (n == 0) return n;

  const hn::ScalableTag<T> d;
  const hn::RebindToUnsigned<decltype(d)> du;
  using TU = hn::TFromD<decltype(du)>;
  const std::size_t lanes = hn::Lanes(d);

  const std::uint64_t lane_counter_limit =
      static_cast<std::uint64_t>(hwy::LimitsMax<TU>());
  const std::size_t max_blocks = static_cast<std::size_t>(
      lane_counter_limit < (std::uint64_t{1} << 20)
          ? lane_counter_limit
          : (std::uint64_t{1} << 20));
  const TU inverse_base = static_cast<TU>(max_blocks);
  const T identity = kMin ? hwy::PositiveInfOrHighestValue<T>()
                          : hwy::NegativeInfOrLowestValue<T>();
  const auto identity_v = hn::Set(d, identity);

  T best = identity;
  std::size_t best_index = 0;
  const std::size_t segment_capacity = max_blocks * lanes;
  for (std::size_t segment = 0; segment < n;
       segment += segment_capacity) {
    const std::size_t segment_size =
        (n - segment < segment_capacity) ? n - segment : segment_capacity;
    auto acc0 = identity_v;
    auto acc1 = identity_v;
    auto blocks0 = hn::Set(du, inverse_base);
    auto blocks1 = hn::Set(du, inverse_base);
    std::size_t i = 0;
    TU inverse_block = inverse_base;

    for (; 2 * lanes <= segment_size - i;
         i += 2 * lanes,
         inverse_block = static_cast<TU>(inverse_block - TU{2})) {
      const auto v0 = hn::LoadU(d, data + segment + i);
      const auto v1 = hn::LoadU(d, data + segment + i + lanes);
      const auto nan0 = hn::IsNaN(v0);
      if (!hn::AllFalse(d, nan0)) {
        return segment + i + hn::FindKnownFirstTrue(d, nan0);
      }
      const auto nan1 = hn::IsNaN(v1);
      if (!hn::AllFalse(d, nan1)) {
        return segment + i + lanes + hn::FindKnownFirstTrue(d, nan1);
      }

      const auto better0 = better<kMin>(v0, acc0);
      const auto better1 = better<kMin>(v1, acc1);
      acc0 = hn::IfThenElse(better0, v0, acc0);
      acc1 = hn::IfThenElse(better1, v1, acc1);
      blocks0 = hn::IfThenElse(hn::RebindMask(du, better0),
                               hn::Set(du, inverse_block), blocks0);
      blocks1 = hn::IfThenElse(
          hn::RebindMask(du, better1),
          hn::Set(du, static_cast<TU>(inverse_block - TU{1})), blocks1);
    }

    const auto value1_wins = better<kMin>(acc1, acc0);
    const auto earlier1_wins = hn::And(
        hn::Eq(acc0, acc1),
        hn::RebindMask(d, hn::Gt(blocks1, blocks0)));
    const auto take1 = hn::Or(value1_wins, earlier1_wins);
    acc0 = hn::IfThenElse(take1, acc1, acc0);
    blocks0 = hn::IfThenElse(hn::RebindMask(du, take1), blocks1, blocks0);

    for (; i < segment_size; i += lanes, --inverse_block) {
      const std::size_t remaining =
          (segment_size - i < lanes) ? segment_size - i : lanes;
      const auto valid = hn::FirstN(d, remaining);
      const auto v = hn::LoadNOr(identity_v, d, data + segment + i, remaining);
      const auto nan = hn::And(valid, hn::IsNaN(v));
      if (!hn::AllFalse(d, nan)) {
        return segment + i + hn::FindKnownFirstTrue(d, nan);
      }
      const auto is_better = better<kMin>(v, acc0);
      acc0 = hn::IfThenElse(is_better, v, acc0);
      blocks0 = hn::IfThenElse(hn::RebindMask(du, is_better),
                               hn::Set(du, inverse_block), blocks0);
    }

    const T segment_best =
        kMin ? hn::ReduceMin(d, acc0) : hn::ReduceMax(d, acc0);
    const auto has_best = hn::Eq(acc0, hn::Set(d, segment_best));
    const TU best_inverse = hn::ReduceMax(
        du, hn::IfThenElseZero(hn::RebindMask(du, has_best), blocks0));
    const auto winner = hn::RebindMask(
        d, hn::MaskedEq(hn::RebindMask(du, has_best), blocks0,
                        hn::Set(du, best_inverse)));
    const std::size_t block =
        max_blocks - static_cast<std::size_t>(best_inverse);
    const std::size_t index =
        segment + block * lanes + hn::FindKnownFirstTrue(d, winner);

    if constexpr (kMin) {
      if (segment_best < best) {
        best = segment_best;
        best_index = index;
      }
    } else {
      if (segment_best > best) {
        best = segment_best;
        best_index = index;
      }
    }
  }
  return best_index;
}

template <typename T>
HWY_INLINE void minmax_once(const T* data, std::size_t n, T* out_min,
                            T* out_max) noexcept {
  const hn::ScalableTag<T> d;
  const std::size_t lanes = hn::Lanes(d);
  const auto min_identity =
      hn::Set(d, hwy::PositiveInfOrHighestValue<T>());
  const auto max_identity =
      hn::Set(d, hwy::NegativeInfOrLowestValue<T>());
  auto min0 = min_identity;
  auto min1 = min_identity;
  auto max0 = max_identity;
  auto max1 = max_identity;
  auto nan = hn::FirstN(d, 0);

  std::size_t i = 0;
  for (; 2 * lanes <= n - i; i += 2 * lanes) {
    const auto v0 = hn::LoadU(d, data + i);
    const auto v1 = hn::LoadU(d, data + i + lanes);
    min0 = hn::Min(min0, v0);
    min1 = hn::Min(min1, v1);
    max0 = hn::Max(max0, v0);
    max1 = hn::Max(max1, v1);
    if constexpr (std::is_floating_point_v<T>) {
      nan = hn::Or(nan, hn::Or(hn::IsNaN(v0), hn::IsNaN(v1)));
    }
  }
  for (; lanes <= n - i; i += lanes) {
    const auto v = hn::LoadU(d, data + i);
    min0 = hn::Min(min0, v);
    max0 = hn::Max(max0, v);
    if constexpr (std::is_floating_point_v<T>) {
      nan = hn::Or(nan, hn::IsNaN(v));
    }
  }

  T minimum = hn::ReduceMin(d, hn::Min(min0, min1));
  T maximum = hn::ReduceMax(d, hn::Max(max0, max1));
  bool has_nan = false;
  if constexpr (std::is_floating_point_v<T>) {
    has_nan = !hn::AllFalse(d, nan);
  }
  for (; i < n; ++i) {
    if constexpr (std::is_floating_point_v<T>) {
      has_nan = has_nan || std::isnan(data[i]);
    }
    if constexpr (std::is_floating_point_v<T>) {
      if (scalar_better<true>(data[i], minimum)) minimum = data[i];
      if (scalar_better<false>(data[i], maximum)) maximum = data[i];
    } else {
      if (data[i] < minimum) minimum = data[i];
      if (data[i] > maximum) maximum = data[i];
    }
  }
  if constexpr (std::is_floating_point_v<T>) {
    if (has_nan) {
      minimum = std::numeric_limits<T>::quiet_NaN();
      maximum = std::numeric_limits<T>::quiet_NaN();
    } else {
      const ZeroSigns signs =
          (minimum == T{0} || maximum == T{0})
              ? zero_signs(data, n)
              : ZeroSigns{false, false};
      if (minimum == T{0})
        minimum = std::copysign(T{0}, signs.negative ? T{-1} : T{1});
      if (maximum == T{0})
        maximum = std::copysign(T{0}, signs.positive ? T{1} : T{-1});
    }
  }
  *out_min = minimum;
  *out_max = maximum;
}

struct Descending {
  template <class D, class V>
  HWY_INLINE auto operator()(D, V a, V b) const noexcept {
    return hn::Gt(a, b);
  }
};

template <typename T>
HWY_INLINE T min_value_direct(const T* data, std::size_t n) noexcept {
  const hn::ScalableTag<T> d;
  if constexpr (std::is_floating_point_v<T>) {
    return propagating_extreme<true>(data, n);
  } else {
    return hn::MinValue(d, data, n);
  }
}

template <typename T>
HWY_INLINE T max_value_direct(const T* data, std::size_t n) noexcept {
  const hn::ScalableTag<T> d;
  if constexpr (std::is_floating_point_v<T>) {
    return propagating_extreme<false>(data, n);
  } else {
    return hn::MaxValue(d, data, n);
  }
}

template <bool kMin, typename T>
HWY_INLINE std::size_t index_of_extreme_direct(const T* data,
                                               std::size_t n) noexcept {
  const hn::ScalableTag<T> d;
  if constexpr (std::is_floating_point_v<T>) {
    return floating_index_of_extreme<kMin>(data, n);
  } else if constexpr (kMin) {
    return hn::IndexOfMin(d, data, n);
  } else {
    return hn::IndexOfMax(d, data, n);
  }
}

}  // namespace

#define HYLE_SEARCH_DEF(T, tag, unused)                                      \
  std::size_t HYLE_SEARCH_FN(count_equal_##tag##_)(                          \
      const T* data, std::size_t n, T value) noexcept {                      \
    return hn::Count(hn::ScalableTag<T>(), value, data, n);                  \
  }                                                                          \
  std::size_t HYLE_SEARCH_FN(find_equal_##tag##_)(                           \
      const T* data, std::size_t n, T value) noexcept {                      \
    return hn::Find(hn::ScalableTag<T>(), value, data, n);                   \
  }                                                                          \
  T HYLE_SEARCH_FN(min_value_##tag##_)(const T* data,                        \
                                       std::size_t n) noexcept {             \
    return min_value_direct(data, n);                                       \
  }                                                                          \
  T HYLE_SEARCH_FN(max_value_##tag##_)(const T* data,                        \
                                       std::size_t n) noexcept {             \
    return max_value_direct(data, n);                                       \
  }                                                                          \
  std::size_t HYLE_SEARCH_FN(index_of_min_##tag##_)(                         \
      const T* data, std::size_t n) noexcept {                               \
    return index_of_extreme_direct<true>(data, n);                           \
  }                                                                          \
  std::size_t HYLE_SEARCH_FN(index_of_max_##tag##_)(                         \
      const T* data, std::size_t n) noexcept {                               \
    return index_of_extreme_direct<false>(data, n);                          \
  }                                                                          \
  void HYLE_SEARCH_FN(minmax_value_##tag##_)(                                \
      const T* data, std::size_t n, T* out_min, T* out_max) noexcept {       \
    minmax_once(data, n, out_min, out_max);                                  \
  }                                                                          \
  bool HYLE_SEARCH_FN(is_sorted_asc_##tag##_)(                               \
      const T* data, std::size_t n) noexcept {                               \
    return hn::IsSorted(hn::ScalableTag<T>(), data, n);                      \
  }                                                                          \
  bool HYLE_SEARCH_FN(is_sorted_desc_##tag##_)(                              \
      const T* data, std::size_t n) noexcept {                               \
    return hn::IsSorted(hn::ScalableTag<T>(), data, n, Descending{});        \
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

#undef HYLE_SEARCH_FN
#undef HYLE_SEARCH_CAT
#undef HYLE_SEARCH_CAT_
