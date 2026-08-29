#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "hwy/highway.h"

namespace hyle::simd::detail::highway_narrow {

namespace hn = hwy::HWY_NAMESPACE;

// Highway's portable pre-F16C promotion in the locked revision preserves
// ordinary f16 values but does not map exponent 31 to the f32 special exponent.
// Keep Highway's target lowering and patch only that semantic gap. This also
// keeps infinities and NaN sign/payload consistent with HYLE's scalar contract.
template <typename HNarrow, typename Storage, class D>
HWY_INLINE auto load_as_f32(D df, const Storage* data) noexcept {
  static_assert(sizeof(HNarrow) == sizeof(Storage));
  static_assert(sizeof(Storage) == sizeof(std::uint16_t));
  const hn::Rebind<HNarrow, D> dn;
  const hn::Rebind<std::uint16_t, D> du16;
#if HWY_TARGET == HWY_SCALAR
  std::uint16_t source_bits;
  std::memcpy(&source_bits, data, sizeof(source_bits));
  const auto packed = hn::Set(du16, source_bits);
#else
  const hn::Repartition<unsigned char, decltype(du16)> du8;
  const auto packed = hn::BitCast(
      du16, hn::LoadU(du8, reinterpret_cast<const unsigned char*>(data)));
#endif
  const auto promoted = hn::PromoteTo(df, hn::BitCast(dn, packed));
  if constexpr (!std::is_same_v<HNarrow, hwy::float16_t>) {
    return promoted;
  } else {
    const hn::Rebind<std::uint32_t, D> du32;
    const auto source = hn::PromoteTo(du32, packed);
    const auto exponent =
        hn::And(hn::ShiftRight<10>(source), hn::Set(du32, 0x1Fu));
    const auto mantissa = hn::ShiftLeft<13>(
        hn::And(source, hn::Set(du32, 0x03FFu)));
#if HWY_TARGET <= HWY_AVX2
    // AVX2 and newer x86 targets lower PromoteTo to VCVTPH2PS. It already
    // handles finite values and infinities exactly; only signaling NaNs are
    // quieted by hardware. Keep the common finite inference path as the raw
    // Highway result and enter the semantic repair only for a NaN block.
    const auto is_nan = hn::And(
        hn::Eq(exponent, hn::Set(du32, 0x1Fu)),
        hn::Ne(mantissa, hn::Zero(du32)));
    if (hn::AllFalse(du32, is_nan)) return promoted;
    const auto sign =
        hn::ShiftLeft<16>(hn::And(source, hn::Set(du32, 0x8000u)));
    const auto special = hn::Or(
        sign, hn::Or(hn::Set(du32, 0x7F800000u), mantissa));
    const auto bits = hn::IfThenElse(
        is_nan, special, hn::BitCast(du32, promoted));
#else
    // Highway's portable integer fallback in the locked revision maps normal
    // values correctly but does not map exponent 31 to the f32 special
    // exponent. Repair infinities and NaNs on those targets.
    const auto sign =
        hn::ShiftLeft<16>(hn::And(source, hn::Set(du32, 0x8000u)));
    const auto special = hn::Or(
        sign, hn::Or(hn::Set(du32, 0x7F800000u), mantissa));
    const auto bits = hn::IfThenElse(
        hn::Eq(exponent, hn::Set(du32, 0x1Fu)), special,
        hn::BitCast(du32, promoted));
#endif
    return hn::BitCast(df, bits);
  }
}

}  // namespace hyle::simd::detail::highway_narrow
