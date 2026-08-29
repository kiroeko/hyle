// Direct-Highway per-target prefix-sum kernels. HYLE owns the range, aliasing
// and modulo-arithmetic contracts; Highway owns target lowering.

#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "hwy/highway.h"

#include "src/simd/scan/kernels.h"

#ifndef HYLE_SIMD_VARIANT_SUFFIX
#error "define HYLE_SIMD_VARIANT_SUFFIX before including scan/kernel_impl.h"
#endif

#define HYLE_SCAN_CAT_(a, b) a##b
#define HYLE_SCAN_CAT(a, b) HYLE_SCAN_CAT_(a, b)
#define HYLE_SCAN_FN(name) HYLE_SCAN_CAT(name, HYLE_SIMD_VARIANT_SUFFIX)

namespace hyle::simd::detail {
namespace {

namespace hn = hwy::HWY_NAMESPACE;

template <std::size_t kLanes, class D, class V>
HWY_INLINE V scan_step(D d, V value) noexcept {
  if constexpr (kLanes >= HWY_MAX_LANES_D(D)) {
    return value;
  } else {
    if (kLanes >= hn::Lanes(d)) return value;
    return hn::Add(value, hn::SlideUpLanes(d, value, kLanes));
  }
}

template <class D, class V>
HWY_INLINE V scan_vector(D d, V value) noexcept {
  value = scan_step<1>(d, value);
  value = scan_step<2>(d, value);
  value = scan_step<4>(d, value);
  value = scan_step<8>(d, value);
  value = scan_step<16>(d, value);
  value = scan_step<32>(d, value);
  return value;
}

template <typename Storage, typename Accumulator>
HWY_INLINE Accumulator to_accumulator(Storage value) noexcept {
  if constexpr (std::is_same_v<Storage, Accumulator>) {
    return value;
  } else {
    return std::bit_cast<Accumulator>(value);
  }
}

template <typename Storage, typename Accumulator>
HWY_INLINE Storage from_accumulator(Accumulator value) noexcept {
  if constexpr (std::is_same_v<Storage, Accumulator>) {
    return value;
  } else {
    return std::bit_cast<Storage>(value);
  }
}

template <typename Storage, class DS, class DA>
HWY_INLINE auto load_accumulator(DS ds, DA da, const Storage* src) noexcept {
  if constexpr (std::is_same_v<hn::TFromD<DS>, hn::TFromD<DA>>) {
    (void)ds;
    return hn::LoadU(da, src);
  } else {
    return hn::BitCast(da, hn::LoadU(ds, src));
  }
}

template <typename Storage, class DS, class DA, class V>
HWY_INLINE void store_accumulator(V value, DS ds, DA da,
                                  Storage* dst) noexcept {
  if constexpr (std::is_same_v<hn::TFromD<DS>, hn::TFromD<DA>>) {
    (void)ds;
    hn::StoreU(value, da, dst);
  } else {
    hn::StoreU(hn::BitCast(ds, value), ds, dst);
  }
}

template <bool kExclusive, typename Storage, class DS, class DA>
HWY_INLINE void scan_core(const Storage* src, Storage* dst, std::size_t n,
                          Storage init, DS ds, DA da) noexcept {
  using Accumulator = hn::TFromD<DA>;
  if (n == 0) return;

  const std::size_t lanes = hn::Lanes(da);
  Accumulator carry = to_accumulator<Storage, Accumulator>(init);
  std::size_t i = 0;
  for (; lanes <= n - i; i += lanes) {
    const auto raw = load_accumulator(ds, da, src + i);
    const auto inclusive = scan_vector(da, raw);
    if constexpr (kExclusive) {
      const auto exclusive =
          hn::Add(hn::Slide1Up(da, inclusive), hn::Set(da, carry));
      store_accumulator(exclusive, ds, da, dst + i);
      carry = static_cast<Accumulator>(
          carry + hn::ExtractLane(inclusive, lanes - 1));
    } else {
      const auto with_carry = hn::Add(inclusive, hn::Set(da, carry));
      store_accumulator(with_carry, ds, da, dst + i);
      carry = hn::ExtractLane(with_carry, lanes - 1);
    }
  }

  for (; i < n; ++i) {
    const Accumulator value =
        to_accumulator<Storage, Accumulator>(src[i]);
    if constexpr (kExclusive) {
      dst[i] = from_accumulator<Storage>(carry);
      carry = static_cast<Accumulator>(carry + value);
    } else {
      carry = static_cast<Accumulator>(carry + value);
      dst[i] = from_accumulator<Storage>(carry);
    }
  }
}

template <bool kExclusive, typename T>
HWY_INLINE void scan(const T* src, T* dst, std::size_t n, T init) noexcept {
  const hn::ScalableTag<T> storage;
  if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
    const hn::RebindToUnsigned<decltype(storage)> accumulator;
    scan_core<kExclusive>(src, dst, n, init, storage, accumulator);
  } else {
    scan_core<kExclusive>(src, dst, n, init, storage, storage);
  }
}

}  // namespace

#define HYLE_SCAN_DEF(T, tag, unused)                                        \
  void HYLE_SCAN_FN(inclusive_scan_##tag##_)(                                \
      const T* src, T* dst, std::size_t n) noexcept {                        \
    scan<false>(src, dst, n, T{});                                           \
  }                                                                          \
  void HYLE_SCAN_FN(exclusive_scan_##tag##_)(                                \
      const T* src, T* dst, std::size_t n, T init) noexcept {                \
    scan<true>(src, dst, n, init);                                           \
  }

HYLE_SCAN_DEF(std::uint32_t, u32, unused)
HYLE_SCAN_DEF(std::int32_t, i32, unused)
HYLE_SCAN_DEF(std::uint64_t, u64, unused)
HYLE_SCAN_DEF(std::int64_t, i64, unused)
HYLE_SCAN_DEF(float, f32, unused)
HYLE_SCAN_DEF(double, f64, unused)

#undef HYLE_SCAN_DEF
}  // namespace hyle::simd::detail

#undef HYLE_SCAN_FN
#undef HYLE_SCAN_CAT
#undef HYLE_SCAN_CAT_
