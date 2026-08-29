// Direct-Highway per-target conversion kernels for f32/f16/bf16 storage.
// Included once by every generated target TU. HYLE owns the storage, RNE and
// NaN contracts; Highway owns target registers and conversion lowering.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "hwy/highway.h"

#include "hyle/simd/dtype.h"
#include "src/simd/dtype/conversion_kernels.h"
#include "src/simd/dtype/highway_narrow.h"

#ifndef HYLE_SIMD_VARIANT_SUFFIX
#error "define HYLE_SIMD_VARIANT_SUFFIX before including dtype/conversion_kernel_impl.h"
#endif

#define HYLE_SIMD_DTYPE_CONVERT_CAT_(a, b) a##b
#define HYLE_SIMD_DTYPE_CONVERT_CAT(a, b) HYLE_SIMD_DTYPE_CONVERT_CAT_(a, b)
#define HYLE_SIMD_DTYPE_CONVERT_FN(name) \
  HYLE_SIMD_DTYPE_CONVERT_CAT(name, HYLE_SIMD_VARIANT_SUFFIX)

namespace hyle::simd::detail {
namespace {

namespace hn = hwy::HWY_NAMESPACE;

template <class D, class V, typename Destination>
HWY_INLINE void store_narrow_bits(D d, V value,
                                  Destination* destination) noexcept {
  static_assert(sizeof(Destination) == sizeof(std::uint16_t));
  (void)d;
  const hn::Rebind<std::uint16_t, D> du16;
#if HWY_TARGET == HWY_SCALAR
  const std::uint16_t bits = hn::GetLane(hn::BitCast(du16, value));
  std::memcpy(destination, &bits, sizeof(bits));
#else
  const hn::Repartition<unsigned char, decltype(du16)> du8;
  hn::StoreU(hn::BitCast(du8, hn::BitCast(du16, value)), du8,
             reinterpret_cast<unsigned char*>(destination));
#endif
}

template <typename HSource, typename Source, class ScalarConvert>
HWY_INLINE void convert_to_f32(const Source* src, float* dst, std::size_t n,
                               ScalarConvert scalar_convert) noexcept {
  static_assert(sizeof(HSource) == sizeof(Source));
  static_assert(sizeof(Source) == sizeof(std::uint16_t));

  const hn::ScalableTag<float> df;
  const std::size_t lanes = hn::Lanes(df);
  std::size_t i = 0;
  for (; lanes <= n - i; i += lanes) {
    hn::StoreU(highway_narrow::load_as_f32<HSource>(df, src + i), df,
               dst + i);
  }
  for (; i < n; ++i) dst[i] = scalar_convert(src[i]);
}

template <typename HNarrow, typename Destination, class ScalarConvert>
HWY_INLINE void convert_from_f32(const float* src, Destination* dst,
                                 std::size_t n,
                                 ScalarConvert scalar_convert) noexcept {
  static_assert(sizeof(HNarrow) == sizeof(Destination));
  static_assert(sizeof(Destination) == sizeof(std::uint16_t));

  const hn::ScalableTag<float> df;
  // The full tag stores 2 * lanes narrow values and lets Highway use its
  // production OrderedDemote2To packing shape. The half tag handles one
  // remaining f32 vector without reading beyond the caller's range.
#if HWY_TARGET != HWY_SCALAR
  const hn::Repartition<HNarrow, decltype(df)> dn_full;
#endif
  const hn::Rebind<HNarrow, decltype(df)> dn_half;
  const std::size_t lanes = hn::Lanes(df);

  std::size_t i = 0;
#if HWY_TARGET != HWY_SCALAR
  // Highway currently exposes the full-width ordered packing primitive for
  // bf16, but not for f16 on every target. Use it where it is an upstream
  // capability; f16 remains direct DemoteTo one vector at a time below.
  if constexpr (std::is_same_v<HNarrow, hwy::bfloat16_t>) {
    for (; lanes <= (n - i) / 2; i += 2 * lanes) {
      const auto a = hn::LoadU(df, src + i);
      const auto b = hn::LoadU(df, src + i + lanes);
      // HYLE's f16 contract canonicalizes every NaN mantissa and its bf16
      // contract explicitly quiets SNaN. Hardware/vendor payload choices may
      // differ, so restore only exceptional blocks through the scalar oracle.
      if (!hn::AllFalse(df, hn::IsNaN(a)) ||
          !hn::AllFalse(df, hn::IsNaN(b))) {
        for (std::size_t lane = 0; lane < 2 * lanes; ++lane)
          dst[i + lane] = scalar_convert(src[i + lane]);
        continue;
      }
      store_narrow_bits(
          dn_full, hn::OrderedDemote2To(dn_full, a, b), dst + i);
    }
  }
#endif
  for (; lanes <= n - i; i += lanes) {
    const auto value = hn::LoadU(df, src + i);
    if (!hn::AllFalse(df, hn::IsNaN(value))) {
      for (std::size_t lane = 0; lane < lanes; ++lane)
        dst[i + lane] = scalar_convert(src[i + lane]);
    } else {
      store_narrow_bits(dn_half, hn::DemoteTo(dn_half, value), dst + i);
    }
  }
  for (; i < n; ++i) dst[i] = scalar_convert(src[i]);
}

template <typename HSource, typename HDestination, typename Source,
          typename Destination, class ScalarConvert>
HWY_INLINE void convert_narrow(const Source* src, Destination* dst,
                               std::size_t n,
                               ScalarConvert scalar_convert) noexcept {
  static_assert(sizeof(HSource) == sizeof(Source));
  static_assert(sizeof(HDestination) == sizeof(Destination));
  static_assert(sizeof(Source) == sizeof(std::uint16_t));
  static_assert(sizeof(Destination) == sizeof(std::uint16_t));

  const hn::ScalableTag<float> df;
  const hn::Rebind<HDestination, decltype(df)> dd;
  const std::size_t lanes = hn::Lanes(df);

  std::size_t i = 0;
  for (; lanes <= n - i; i += lanes) {
    const auto wide = highway_narrow::load_as_f32<HSource>(df, src + i);
    if (!hn::AllFalse(df, hn::IsNaN(wide))) {
      for (std::size_t lane = 0; lane < lanes; ++lane)
        dst[i + lane] = scalar_convert(src[i + lane]);
      continue;
    }
    store_narrow_bits(dd, hn::DemoteTo(dd, wide), dst + i);
  }
  for (; i < n; ++i) dst[i] = scalar_convert(src[i]);
}

}  // namespace

void HYLE_SIMD_DTYPE_CONVERT_FN(f16_to_f32_)(const float16_t* src, float* dst,
                                              std::size_t n) noexcept {
  convert_to_f32<hwy::float16_t>(src, dst, n, f16_to_f32);
}

void HYLE_SIMD_DTYPE_CONVERT_FN(f32_to_f16_)(const float* src, float16_t* dst,
                                              std::size_t n) noexcept {
  convert_from_f32<hwy::float16_t>(src, dst, n, f32_to_f16);
}

void HYLE_SIMD_DTYPE_CONVERT_FN(bf16_to_f32_)(const bfloat16_t* src, float* dst,
                                               std::size_t n) noexcept {
  convert_to_f32<hwy::bfloat16_t>(src, dst, n, bf16_to_f32);
}

void HYLE_SIMD_DTYPE_CONVERT_FN(f32_to_bf16_)(const float* src, bfloat16_t* dst,
                                               std::size_t n) noexcept {
  convert_from_f32<hwy::bfloat16_t>(src, dst, n, f32_to_bf16);
}

void HYLE_SIMD_DTYPE_CONVERT_FN(f16_to_bf16_)(const float16_t* src,
                                               bfloat16_t* dst,
                                               std::size_t n) noexcept {
  convert_narrow<hwy::float16_t, hwy::bfloat16_t>(
      src, dst, n,
      [](float16_t value) { return f32_to_bf16(f16_to_f32(value)); });
}

void HYLE_SIMD_DTYPE_CONVERT_FN(bf16_to_f16_)(const bfloat16_t* src,
                                               float16_t* dst,
                                               std::size_t n) noexcept {
  convert_narrow<hwy::bfloat16_t, hwy::float16_t>(
      src, dst, n,
      [](bfloat16_t value) { return f32_to_f16(bf16_to_f32(value)); });
}

}  // namespace hyle::simd::detail

#undef HYLE_SIMD_DTYPE_CONVERT_FN
#undef HYLE_SIMD_DTYPE_CONVERT_CAT
#undef HYLE_SIMD_DTYPE_CONVERT_CAT_
