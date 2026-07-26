#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "hyle/simd/dtype.h"
#include "hyle/simd/vec.h"

// First-class f16/bf16 lane facade.
//
// HYLE's public narrow values are stable two-byte POD carriers. They deliberately are not
// compiler-native `_Float16`/`__bf16`: those types would raise the baseline ISA and change ABI
// availability across compilers. A narrow tag therefore retains the numeric FORMAT while its
// vector value reuses the backend's u16 carrier. Keeping the format in every operation matters:
// on SVE/RVV the native u16 vector type itself cannot distinguish f16, bf16 and integer bits.
// Untagged operations on the carrier have integer semantics and are not narrow-float operations.
//
// The tag is rebound from an f32 tag, so both sides have the SAME number of lanes. On fixed-width
// backends the carrier is a half-width register; on SVE/RVV Rebind's kPow2 keeps the active lane
// count equal. This is the mature Highway-style promote-compute-demote shape and avoids requiring
// optional AVX-512-FP16, ARM BF16 or RISC-V Zvfh/Zvfbfmin extensions.

namespace hyle::simd {

template <typename T>
inline constexpr bool kIsNarrowFloat =
    std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>;

template <typename Narrow, class FloatD>
struct NarrowTag {
  static_assert(kIsNarrowFloat<Narrow>, "NarrowTag supports f16/bf16");
  static_assert(std::is_same_v<TFromD<FloatD>, float>,
                "NarrowTag must be rebound from an f32 tag");

  using T = Narrow;
  using FloatTag = FloatD;
  using BitsTag = Rebind<std::uint16_t, FloatD>;
#if HYLE_SIMD_HAVE_SCALABLE
  using StorageTag = BitsTag;
#else
  // Fixed-width Store/half-vector overloads encode their physical band in the type and require
  // kPow2=0. BitsTag retains Rebind's -1 for the generic Promote/Demote compositions; both tags
  // name the same physical half-width XMM/D-register value because their N is identical.
  using StorageTag = Simd<std::uint16_t, FloatD::kMaxLanes, 0>;
#endif

  constexpr NarrowTag() = default;
};

template <class FloatD>
using F16Tag = NarrowTag<float16_t, FloatD>;
template <class FloatD>
using BF16Tag = NarrowTag<bfloat16_t, FloatD>;

// The backend-native carrier type for a narrow tag. Treat it as opaque: use the tagged overloads
// below, not ordinary u16 arithmetic.
template <class DN>
using VFromNarrowD = VFromD<typename DN::BitsTag>;

template <class DN>
HYLE_FORCE_INLINE std::size_t Lanes(DN) noexcept
  requires kIsNarrowFloat<typename DN::T>
{
  return Lanes(typename DN::FloatTag{});
}

template <class DN>
constexpr std::size_t MaxLanes(DN) noexcept
  requires kIsNarrowFloat<typename DN::T>
{
  return DN::FloatTag::kMaxLanes;
}

namespace detail {

template <class DN>
HYLE_FORCE_INLINE const std::uint16_t* narrow_bits(const typename DN::T* p) noexcept {
  static_assert(sizeof(typename DN::T) == sizeof(std::uint16_t));
  static_assert(std::is_trivially_copyable_v<typename DN::T>);
  return reinterpret_cast<const std::uint16_t*>(p);
}

template <class DN>
HYLE_FORCE_INLINE std::uint16_t* narrow_bits(typename DN::T* p) noexcept {
  static_assert(sizeof(typename DN::T) == sizeof(std::uint16_t));
  static_assert(std::is_trivially_copyable_v<typename DN::T>);
  return reinterpret_cast<std::uint16_t*>(p);
}

template <class DF, class V>
HYLE_FORCE_INLINE VFromD<DF> promote_f16_portable(DF df, V packed) noexcept {
  const Rebind<std::uint32_t, DF> du32;
  const Rebind<std::int32_t, DF> di32;
  const auto h = PromoteTo(du32, packed);
  const auto sign = ShiftLeft<16>(And(h, Set(du32, 0x8000u)));
  const auto be = And(ShiftRight<10>(h), Set(du32, 0x1Fu));
  const auto mant = And(h, Set(du32, 0x3FFu));
  const auto be32 = IfThenElse(Eq(be, Set(du32, 31u)), Set(du32, 255u),
                               Add(be, Set(du32, 112u)));
  const auto normal = Or(sign, Or(ShiftLeft<23>(be32), ShiftLeft<13>(mant)));

  // Half subnormals are mantissa * 2^-24, exact in f32. Applying sign as bits also preserves -0.
  // mant <= 1023, so signed i32 conversion is exact and avoids requiring a u32->f32 operation
  // (not available on SSE4, matching the underlying hardware).
  const auto sub = Mul(ConvertTo(df, BitCast(di32, mant)), Set(df, 0x1p-24f));
  const auto sub_bits = Or(BitCast(du32, sub), sign);
  return BitCast(df, IfThenElse(Eq(be, Zero(du32)), sub_bits, normal));
}

template <class DN, class V>
HYLE_FORCE_INLINE VFromNarrowD<DN> demote_f16_portable(DN, V value) noexcept {
  using DF = typename DN::FloatTag;
  const Rebind<std::uint32_t, DF> du32;
  const auto u = BitCast(du32, value);
  const auto be = And(ShiftRight<23>(u), Set(du32, 0xFFu));
  const auto mant = And(u, Set(du32, 0x7FFFFFu));

  // Scalar oracle's RNE expressed lane-wise. idx is clamped before every variable shift, so
  // underflow lanes never execute an out-of-range shift even though their final result is zero.
  const auto raw_idx = Sub(Set(du32, 126u), be);
  const auto idx = IfThenElse(Gt(be, Set(du32, 113u)), Set(du32, 13u),
                              Max(Set(du32, 13u), Min(raw_idx, Set(du32, 24u))));
  const auto odd = And(Shr(Or(mant, Set(du32, 0x800000u)), idx), Set(du32, 1u));
  const auto bias = Sub(Shl(Set(du32, 1u), Sub(idx, Set(du32, 1u))), Set(du32, 1u));
  const auto rounded = Add(mant, Add(odd, bias));
  const auto carry_mask = Ge(rounded, Set(du32, 1u << 23));
  const auto biased = Add(be, IfThenElse(carry_mask, Set(du32, 1u), Zero(du32)));

  const auto below_normal = Lt(biased, Set(du32, 113u));
  const auto raw_sub_exp = Sub(Set(du32, 113u), biased);
  const auto sub_exp = IfThenElse(below_normal, Min(raw_sub_exp, Set(du32, 10u)), Zero(du32));
  const auto shifted_mant = Shr(And(rounded, Set(du32, 0x7FFFFFu)),
                                Add(Set(du32, 13u), sub_exp));
  const auto leading = IfThenElse(Eq(sub_exp, Zero(du32)), Zero(du32),
                                  Shr(Set(du32, 1024u), sub_exp));

  const auto is_nan = And(Eq(be, Set(du32, 255u)), Ne(mant, Zero(du32)));
  const auto overflow = Ge(biased, Set(du32, 143u));
  const auto mant16 = IfThenElse(is_nan, Set(du32, 0x3FFu),
                                 IfThenElse(overflow, Zero(du32), Add(leading, shifted_mant)));
  const auto rebased = IfThenElse(Lt(biased, Set(du32, 112u)), Zero(du32),
                                  Min(Sub(biased, Set(du32, 112u)), Set(du32, 31u)));
  const auto sign = ShiftRight<16>(And(u, Set(du32, 0x80000000u)));
  const auto encoded = Or(sign, Or(ShiftLeft<10>(rebased), mant16));
  const auto underflow = Lt(biased, Set(du32, 103u));
  const auto result = IfThenElse(underflow, sign, encoded);
  // Encoded lanes are in [0, 65535]. x86's u32->u16 pack is expressed as signed-i32
  // PACKUSDW, so reinterpret before the established saturating demote composition.
  const Rebind<std::int32_t, DF> di32;
  return DemoteTo(typename DN::BitsTag{}, BitCast(di32, result));
}

template <class DF, class V>
HYLE_FORCE_INLINE VFromD<DF> promote_f16(DF df, V packed) noexcept {
#if defined(HYLE_SIMD_TARGET_AVX512)
  if constexpr (HYLE_SIMD_BYTES_D(DF) > 32) {
    VFromD<DF> result;
    result.raw = _mm512_cvtph_ps(packed.raw);
    return result;
  } else if constexpr (HYLE_SIMD_BYTES_D(DF) > 16) {
    VFromD<DF> result;
    result.raw = _mm256_cvtph_ps(packed.raw);
    return result;
  } else {
    return promote_f16_portable(df, packed);
  }
#elif defined(HYLE_SIMD_TARGET_AVX2)
  if constexpr (HYLE_SIMD_BYTES_D(DF) > 16) {
    VFromD<DF> result;
    result.raw = _mm256_cvtph_ps(packed.raw);
    return result;
  } else {
    return promote_f16_portable(df, packed);
  }
#else
  return promote_f16_portable(df, packed);
#endif
}

template <class DN, class V>
HYLE_FORCE_INLINE VFromNarrowD<DN> demote_f16(DN dn, V value) noexcept {
  using DF = typename DN::FloatTag;
#if defined(HYLE_SIMD_TARGET_AVX512)
  if constexpr (HYLE_SIMD_BYTES_D(DF) > 32) {
    VFromNarrowD<DN> result;
#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable : 4556)
#endif
    result.raw = _mm512_cvtps_ph(value.raw, _MM_FROUND_NO_EXC);
#if defined(_MSC_VER)
#  pragma warning(pop)
#endif
    return result;
  } else if constexpr (HYLE_SIMD_BYTES_D(DF) > 16) {
    VFromNarrowD<DN> result;
#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable : 4556)
#endif
    result.raw = _mm256_cvtps_ph(value.raw, _MM_FROUND_NO_EXC);
#if defined(_MSC_VER)
#  pragma warning(pop)
#endif
    return result;
  } else {
    return demote_f16_portable(dn, value);
  }
#elif defined(HYLE_SIMD_TARGET_AVX2)
  if constexpr (HYLE_SIMD_BYTES_D(DF) > 16) {
    VFromNarrowD<DN> result;
#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable : 4556)
#endif
    result.raw = _mm256_cvtps_ph(value.raw, _MM_FROUND_NO_EXC);
#if defined(_MSC_VER)
#  pragma warning(pop)
#endif
    return result;
  } else {
    return demote_f16_portable(dn, value);
  }
#else
  return demote_f16_portable(dn, value);
#endif
}

template <class DF, class V>
HYLE_FORCE_INLINE VFromD<DF> promote_bf16(DF df, V packed) noexcept {
  const Rebind<std::uint32_t, DF> du32;
  return BitCast(df, ShiftLeft<16>(PromoteTo(du32, packed)));
}

template <class DN, class V>
HYLE_FORCE_INLINE VFromNarrowD<DN> demote_bf16(DN, V value) noexcept {
  using DF = typename DN::FloatTag;
  const Rebind<std::uint32_t, DF> du32;
  const Rebind<std::int32_t, DF> di32;
  const auto u = BitCast(du32, value);
  const auto abs = And(u, Set(du32, 0x7FFFFFFFu));
  const auto finite = Lt(abs, Set(du32, 0x7F800000u));
  const auto is_nan = Gt(abs, Set(du32, 0x7F800000u));
  const auto lsb = And(ShiftRight<16>(u), Set(du32, 1u));
  const auto incr = IfThenElseZero(finite, Add(Set(du32, 0x7FFFu), lsb));
  const auto sticky = IfThenElseZero(is_nan, Set(du32, 1u << 6));
  const auto out = Or(sticky, ShiftRight<16>(Add(u, incr)));
  return DemoteTo(typename DN::BitsTag{}, BitCast(di32, out));
}

}  // namespace detail

// Storage operations. They touch exactly Lanes(d) values; aligned forms inherit the underlying
// carrier tag's alignment contract.
template <class DN>
HYLE_FORCE_INLINE VFromNarrowD<DN> LoadNarrowU(DN, const typename DN::T* p) noexcept {
  return LoadU(typename DN::StorageTag{}, detail::narrow_bits<DN>(p));
}
template <class DN>
HYLE_FORCE_INLINE VFromNarrowD<DN> LoadNarrow(DN, const typename DN::T* p) noexcept {
  return Load(typename DN::StorageTag{}, detail::narrow_bits<DN>(p));
}
template <class DN>
HYLE_FORCE_INLINE void StoreNarrowU(VFromNarrowD<DN> v, DN, typename DN::T* p) noexcept {
  StoreU(v, typename DN::StorageTag{}, detail::narrow_bits<DN>(p));
}
template <class DN>
HYLE_FORCE_INLINE void StoreNarrow(VFromNarrowD<DN> v, DN, typename DN::T* p) noexcept {
  Store(v, typename DN::StorageTag{}, detail::narrow_bits<DN>(p));
}
template <class DN>
HYLE_FORCE_INLINE VFromNarrowD<DN> LoadNarrowN(
    DN, const typename DN::T* p, std::size_t max_lanes) noexcept {
  return LoadN(typename DN::StorageTag{}, detail::narrow_bits<DN>(p), max_lanes);
}
template <class DN>
HYLE_FORCE_INLINE void StoreNarrowN(VFromNarrowD<DN> v, DN,
                                    typename DN::T* p,
                                    std::size_t max_lanes) noexcept {
  StoreN(v, typename DN::StorageTag{}, detail::narrow_bits<DN>(p), max_lanes);
}
template <class DN>
HYLE_FORCE_INLINE VFromNarrowD<DN> ZeroNarrow(DN) noexcept {
  return Zero(typename DN::StorageTag{});
}
template <class DN>
HYLE_FORCE_INLINE VFromNarrowD<DN> SetNarrow(DN, typename DN::T value) noexcept {
  return Set(typename DN::StorageTag{}, value.bits);
}
template <class DN>
HYLE_FORCE_INLINE VFromNarrowD<DN> SetNarrow(DN d, float value) noexcept {
  if constexpr (std::is_same_v<typename DN::T, float16_t>)
    return SetNarrow(d, f32_to_f16(value));
  else
    return SetNarrow(d, f32_to_bf16(value));
}

// Explicit-format conversion. The source tag is intentionally present on PromoteNarrowTo: f16 and bf16
// share the same native carrier type, especially on SVE/RVV, so the value alone cannot select the
// conversion. DemoteNarrowTo is unambiguous because its destination tag carries the format.
template <class DF, class DN>
HYLE_FORCE_INLINE VFromD<DF> PromoteNarrowTo(DF df, DN,
                                             VFromNarrowD<DN> packed) noexcept {
  static_assert(std::is_same_v<DF, typename DN::FloatTag>,
                "PromoteNarrowTo destination must match NarrowTag::FloatTag");
  if constexpr (std::is_same_v<typename DN::T, float16_t>)
    return detail::promote_f16(df, packed);
  else
    return detail::promote_bf16(df, packed);
}

template <class DN>
HYLE_FORCE_INLINE VFromNarrowD<DN> DemoteNarrowTo(
    DN dn, VFromD<typename DN::FloatTag> value) noexcept {
  if constexpr (std::is_same_v<typename DN::T, float16_t>)
    return detail::demote_f16(dn, value);
  else
    return detail::demote_bf16(dn, value);
}

// Minimal compute set: promote to f32, compute in the backend's native f32 vector, then demote.
// Add/Sub/Mul/Div round once to the destination narrow format. MulAdd performs the backend f32 FMA
// (or its established fallback) and narrows once, which is the useful mixed-precision contract.
#define HYLE_SIMD_NARROW_BINARY(NAME, WIDE_NAME)                                        \
  template <class DN>                                                                  \
  HYLE_FORCE_INLINE VFromNarrowD<DN> NAME(                                              \
      DN dn, VFromNarrowD<DN> a, VFromNarrowD<DN> b) noexcept {                         \
    using DF = typename DN::FloatTag;                                                   \
    const DF df;                                                                        \
    return DemoteNarrowTo(                                                              \
        dn, WIDE_NAME(PromoteNarrowTo(df, dn, a), PromoteNarrowTo(df, dn, b)));         \
  }
HYLE_SIMD_NARROW_BINARY(AddNarrow, Add)
HYLE_SIMD_NARROW_BINARY(SubNarrow, Sub)
HYLE_SIMD_NARROW_BINARY(MulNarrow, Mul)
HYLE_SIMD_NARROW_BINARY(DivNarrow, Div)
#undef HYLE_SIMD_NARROW_BINARY

template <class DN>
HYLE_FORCE_INLINE VFromNarrowD<DN> MulAddNarrow(DN dn, VFromNarrowD<DN> a,
                                                VFromNarrowD<DN> b,
                                                VFromNarrowD<DN> c) noexcept {
  using DF = typename DN::FloatTag;
  const DF df;
  return DemoteNarrowTo(dn, MulAdd(PromoteNarrowTo(df, dn, a),
                                   PromoteNarrowTo(df, dn, b),
                                   PromoteNarrowTo(df, dn, c)));
}

}  // namespace hyle::simd
