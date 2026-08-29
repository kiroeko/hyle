// Direct-Highway per-target f16/bf16 range kernels. HYLE owns the public
// pointer/count and IEEE contracts; target-dependent vectors stay in this TU.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "hwy/highway.h"

#include "hyle/simd/dtype.h"
#include "src/simd/dtype/highway_narrow.h"
#include "src/simd/narrow/kernels.h"

#ifndef HYLE_SIMD_VARIANT_SUFFIX
#error "define HYLE_SIMD_VARIANT_SUFFIX before including narrow/kernel_impl.h"
#endif

#define HYLE_NARROW_CAT_(a, b) a##b
#define HYLE_NARROW_CAT(a, b) HYLE_NARROW_CAT_(a, b)
#define HYLE_NARROW_FN(name) HYLE_NARROW_CAT(name, HYLE_SIMD_VARIANT_SUFFIX)

namespace hyle::simd::detail {
namespace {

namespace hn = hwy::HWY_NAMESPACE;

template <typename HNarrow, typename Storage, class D>
HWY_INLINE auto load_as_f32(D df, const Storage* data) noexcept {
  return highway_narrow::load_as_f32<HNarrow>(df, data);
}

template <typename Storage, class D>
HWY_INLINE auto load_bits(D du16, const Storage* data) noexcept {
  static_assert(sizeof(Storage) == sizeof(std::uint16_t));
#if HWY_TARGET == HWY_SCALAR
  std::uint16_t bits;
  std::memcpy(&bits, data, sizeof(bits));
  return hn::Set(du16, bits);
#else
  const hn::Repartition<unsigned char, D> du8;
  return hn::BitCast(
      du16, hn::LoadU(du8, reinterpret_cast<const unsigned char*>(data)));
#endif
}

template <class D, class V>
HWY_INLINE auto total_order_key(D du16, V bits) noexcept {
  const auto negative =
      hn::Ne(hn::And(bits, hn::Set(du16, std::uint16_t{0x8000})),
             hn::Zero(du16));
  const auto flip = hn::IfThenElse(
      negative, hn::Set(du16, std::uint16_t{0xFFFF}),
      hn::Set(du16, std::uint16_t{0x8000}));
  return hn::Xor(bits, flip);
}

template <typename HNarrow, typename Storage, class Decode>
HWY_INLINE std::size_t count_numeric(const Storage* data, std::size_t n,
                                     Storage value,
                                     Decode decode) noexcept {
  const hn::ScalableTag<float> df;
  const std::size_t lanes = hn::Lanes(df);
  const auto expected = hn::Set(df, decode(value));
  std::size_t count = 0;
  std::size_t i = 0;
  for (; lanes <= n - i; i += lanes) {
    count += hn::CountTrue(
        df, hn::Eq(load_as_f32<HNarrow>(df, data + i), expected));
  }
  for (; i < n; ++i) count += decode(data[i]) == decode(value);
  return count;
}

template <typename HNarrow, typename Storage, class Decode>
HWY_INLINE std::size_t find_numeric(const Storage* data, std::size_t n,
                                    Storage value,
                                    Decode decode) noexcept {
  const hn::ScalableTag<float> df;
  const std::size_t lanes = hn::Lanes(df);
  const auto expected = hn::Set(df, decode(value));
  std::size_t i = 0;
  for (; lanes <= n - i; i += lanes) {
    const std::intptr_t lane = hn::FindFirstTrue(
        df, hn::Eq(load_as_f32<HNarrow>(df, data + i), expected));
    if (lane >= 0) return i + static_cast<std::size_t>(lane);
  }
  for (; i < n; ++i)
    if (decode(data[i]) == decode(value)) return i;
  return n;
}

template <typename Storage>
HWY_INLINE std::size_t count_bits(const Storage* data, std::size_t n,
                                  Storage value) noexcept {
  const hn::Rebind<std::uint16_t, hn::ScalableTag<float>> du16;
  const std::size_t lanes = hn::Lanes(du16);
  const auto expected = hn::Set(du16, value.bits);
  std::size_t count = 0;
  std::size_t i = 0;
  for (; lanes <= n - i; i += lanes)
    count += hn::CountTrue(du16,
                           hn::Eq(load_bits(du16, data + i), expected));
  for (; i < n; ++i) count += data[i].bits == value.bits;
  return count;
}

template <typename Storage>
HWY_INLINE std::size_t find_bits(const Storage* data, std::size_t n,
                                 Storage value) noexcept {
  const hn::Rebind<std::uint16_t, hn::ScalableTag<float>> du16;
  const std::size_t lanes = hn::Lanes(du16);
  const auto expected = hn::Set(du16, value.bits);
  std::size_t i = 0;
  for (; lanes <= n - i; i += lanes) {
    const std::intptr_t lane = hn::FindFirstTrue(
        du16, hn::Eq(load_bits(du16, data + i), expected));
    if (lane >= 0) return i + static_cast<std::size_t>(lane);
  }
  for (; i < n; ++i)
    if (data[i].bits == value.bits) return i;
  return n;
}

template <typename HNarrow, typename Storage, class Decode>
HWY_INLINE float sum_narrow(const Storage* data, std::size_t n,
                            Decode decode) noexcept {
  const hn::ScalableTag<float> df;
  const std::size_t lanes = hn::Lanes(df);
  const std::size_t lanes4 = lanes * 4;
  auto sum0 = hn::Zero(df);
  auto sum1 = hn::Zero(df);
  auto sum2 = hn::Zero(df);
  auto sum3 = hn::Zero(df);
  std::size_t i = 0;
  for (; lanes4 <= n - i; i += lanes4) {
    sum0 = hn::Add(sum0, load_as_f32<HNarrow>(df, data + i));
    sum1 = hn::Add(sum1, load_as_f32<HNarrow>(df, data + i + lanes));
    sum2 = hn::Add(sum2, load_as_f32<HNarrow>(df, data + i + lanes * 2));
    sum3 = hn::Add(sum3, load_as_f32<HNarrow>(df, data + i + lanes * 3));
  }
  for (; lanes <= n - i; i += lanes)
    sum0 = hn::Add(sum0, load_as_f32<HNarrow>(df, data + i));
  float result =
      hn::ReduceSum(df, hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3)));
  for (; i < n; ++i) result += decode(data[i]);
  return result;
}

template <typename Storage>
HWY_INLINE bool sorted_total(const Storage* data, std::size_t n,
                             bool ascending) noexcept {
  if (n < 2) return true;
  const hn::Rebind<std::uint16_t, hn::ScalableTag<float>> du16;
  const std::size_t lanes = hn::Lanes(du16);
  std::size_t i = 1;
  for (; lanes <= n - i; i += lanes) {
    const auto previous = total_order_key(du16, load_bits(du16, data + i - 1));
    const auto current = total_order_key(du16, load_bits(du16, data + i));
    const auto ordered = ascending ? hn::Le(previous, current)
                                   : hn::Ge(previous, current);
    if (!hn::AllTrue(du16, ordered)) return false;
  }
  auto previous = static_cast<std::uint16_t>(
      data[i - 1].bits ^ ((data[i - 1].bits & 0x8000u) ? 0xFFFFu : 0x8000u));
  for (; i < n; ++i) {
    const auto current = static_cast<std::uint16_t>(
        data[i].bits ^ ((data[i].bits & 0x8000u) ? 0xFFFFu : 0x8000u));
    if (ascending ? previous > current : previous < current) return false;
    previous = current;
  }
  return true;
}

}  // namespace

#define HYLE_NARROW_DEFINE(T, tag, htype, decode)                              \
  std::size_t HYLE_NARROW_FN(narrow_count_equal_##tag##_)(                    \
      const T* data, std::size_t n, T value) noexcept {                       \
    return count_numeric<htype>(data, n, value, decode);                      \
  }                                                                            \
  std::size_t HYLE_NARROW_FN(narrow_find_equal_##tag##_)(                     \
      const T* data, std::size_t n, T value) noexcept {                       \
    return find_numeric<htype>(data, n, value, decode);                       \
  }                                                                            \
  std::size_t HYLE_NARROW_FN(narrow_count_equal_bits_##tag##_)(               \
      const T* data, std::size_t n, T value) noexcept {                       \
    return count_bits(data, n, value);                                        \
  }                                                                            \
  std::size_t HYLE_NARROW_FN(narrow_find_equal_bits_##tag##_)(                \
      const T* data, std::size_t n, T value) noexcept {                       \
    return find_bits(data, n, value);                                         \
  }                                                                            \
  float HYLE_NARROW_FN(narrow_sum_##tag##_)(const T* data,                    \
                                             std::size_t n) noexcept {         \
    return sum_narrow<htype>(data, n, decode);                                \
  }                                                                            \
  bool HYLE_NARROW_FN(narrow_is_sorted_total_asc_##tag##_)(                   \
      const T* data, std::size_t n) noexcept {                                \
    return sorted_total(data, n, true);                                       \
  }                                                                            \
  bool HYLE_NARROW_FN(narrow_is_sorted_total_desc_##tag##_)(                  \
      const T* data, std::size_t n) noexcept {                                \
    return sorted_total(data, n, false);                                      \
  }

HYLE_NARROW_DEFINE(float16_t, f16, hwy::float16_t, f16_to_f32)
HYLE_NARROW_DEFINE(bfloat16_t, bf16, hwy::bfloat16_t, bf16_to_f32)

#undef HYLE_NARROW_DEFINE

}  // namespace hyle::simd::detail

#undef HYLE_NARROW_FN
#undef HYLE_NARROW_CAT
#undef HYLE_NARROW_CAT_
