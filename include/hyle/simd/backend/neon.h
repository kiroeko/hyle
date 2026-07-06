#pragma once

#include <cstddef>
#include <cstdint>
#include <bit>
#include <type_traits>

#include <arm_neon.h>

#include "hyle/macros.h"

// backend/neon.h -- 128-bit aarch64 NEON backend (simd-design landing list #2, ARM half).
//
// ONE backend per TU: pulled in by vec.h only when -DHYLE_SIMD_TARGET_NEON (aarch64 only).
// Same op contract as scalar/sse4/avx2, so the unified test/kernel source compiles
// unchanged. Lane types f32/f64 + all integer widths i8/u8/i16/u16/i32/u32/i64/u64;
// NATIVE FULL WIDTH (128-bit = 16 i8 / 8 i16 / 4 i32|f32 / 2 i64|f64).
//
// STATUS: written + cross-compiled (aarch64-linux-gnu-g++-14) + objdump-verified to emit
// real NEON instructions, but NOT executed (no ARM hardware / no qemu here). Same honesty
// bar as container-04b's GroupNeon: "written + compiled + instruction-verified, only unrun".
//
// Intrinsics follow Highway arm_neon-inl.h. f64 NEON ops (vaddq_f64 etc.) are aarch64-only
// (not ARMv7); this backend targets aarch64. Masks are an unsigned vector of all-1s/all-0s
// per lane (vceqq_* returns uintNxM), matching the SSE/AVX mask=vector model.

namespace hyle::simd {

namespace detail {
// NEON, unlike x86 (one __m256i for every integer width), has a DISTINCT vector type
// per lane width AND per signedness: int8x16_t vs uint8x16_t vs int16x8_t vs ... So the
// T->raw trait must be specialized for all 10 supported lane types (not just a scalar
// float/double override on a shared integer type).
template <typename T>
struct RawNeon;
template <> struct RawNeon<float> { using type = float32x4_t; };
template <> struct RawNeon<double> { using type = float64x2_t; };
template <> struct RawNeon<std::int8_t> { using type = int8x16_t; };
template <> struct RawNeon<std::uint8_t> { using type = uint8x16_t; };
template <> struct RawNeon<std::int16_t> { using type = int16x8_t; };
template <> struct RawNeon<std::uint16_t> { using type = uint16x8_t; };
template <> struct RawNeon<std::int32_t> { using type = int32x4_t; };
template <> struct RawNeon<std::uint32_t> { using type = uint32x4_t; };
template <> struct RawNeon<std::int64_t> { using type = int64x2_t; };
template <> struct RawNeon<std::uint64_t> { using type = uint64x2_t; };
template <typename T> using RawNeonT = typename RawNeon<T>::type;

// mask raw is always an unsigned lane vector of the SAME width as T (compare results).
// Named *RawNeon (paralleling RawNeon above) so the value struct MaskNeon<T,N> below owns
// the natural name symmetric with VecNeon<T,N>.
template <typename T> struct MaskRawNeon;
template <> struct MaskRawNeon<float> { using type = uint32x4_t; };
template <> struct MaskRawNeon<double> { using type = uint64x2_t; };
template <> struct MaskRawNeon<std::int8_t> { using type = uint8x16_t; };
template <> struct MaskRawNeon<std::uint8_t> { using type = uint8x16_t; };
template <> struct MaskRawNeon<std::int16_t> { using type = uint16x8_t; };
template <> struct MaskRawNeon<std::uint16_t> { using type = uint16x8_t; };
template <> struct MaskRawNeon<std::int32_t> { using type = uint32x4_t; };
template <> struct MaskRawNeon<std::uint32_t> { using type = uint32x4_t; };
template <> struct MaskRawNeon<std::int64_t> { using type = uint64x2_t; };
template <> struct MaskRawNeon<std::uint64_t> { using type = uint64x2_t; };
template <typename T> using MaskRawNeonT = typename MaskRawNeon<T>::type;

template <typename T>
inline constexpr bool kSupported =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::uint32_t> ||
    std::is_same_v<T, std::int16_t> || std::is_same_v<T, std::uint16_t> ||
    std::is_same_v<T, std::int8_t> || std::is_same_v<T, std::uint8_t> ||
    std::is_same_v<T, std::int64_t> || std::is_same_v<T, std::uint64_t>;

// Integer-width helpers (mirror avx2.h): NEON int intrinsics pick by lane WIDTH and
// signedness, so per-op constexpr chains branch on these.
template <typename T>
inline constexpr bool kI8 = std::is_same_v<T, std::int8_t> || std::is_same_v<T, std::uint8_t>;
template <typename T>
inline constexpr bool kI16 = std::is_same_v<T, std::int16_t> || std::is_same_v<T, std::uint16_t>;
template <typename T>
inline constexpr bool kI32 = std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::uint32_t>;
template <typename T>
inline constexpr bool kI64 = std::is_same_v<T, std::int64_t> || std::is_same_v<T, std::uint64_t>;
template <typename T>
inline constexpr bool kSigned = std::is_signed_v<T>;
}  // namespace detail

template <typename T, std::size_t N>
struct VecNeon {
  static_assert(detail::kSupported<T>, "neon backend lane type unsupported");
  typename detail::RawNeonT<T> raw;
  using PrivateT = T;
  static constexpr std::size_t kPrivateN = N;
};
template <typename T, std::size_t N>
struct MaskNeon {
  typename detail::MaskRawNeonT<T> raw;
  using PrivateT = T;
  static constexpr std::size_t kPrivateN = N;
};

template <class V>
using DFromV = Simd<typename V::PrivateT, V::kPrivateN, 0>;
template <class V>
using TFromV = typename V::PrivateT;

namespace detail {
template <class D>
using VN = VecNeon<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)>;
template <class D>
using MN = MaskNeon<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)>;
template <class D>
inline constexpr bool kNativeWidth = HYLE_SIMD_MAX_LANES_D(D) * sizeof(TFromD<D>) == 16;
}  // namespace detail

// ---- initialization -----------------------------------------------------------------
template <class D, typename T2>
HYLE_FORCE_INLINE detail::VN<D> Set(D, T2 t) noexcept {
  using T = TFromD<D>;
  static_assert(detail::kNativeWidth<D>, "neon backend is native-width-only (128-bit)");
  detail::VN<D> v;
  if constexpr (std::is_same_v<T, float>) v.raw = vdupq_n_f32(static_cast<float>(t));
  else if constexpr (std::is_same_v<T, double>) v.raw = vdupq_n_f64(static_cast<double>(t));
  else if constexpr (std::is_same_v<T, std::int8_t>) v.raw = vdupq_n_s8(static_cast<std::int8_t>(t));
  else if constexpr (std::is_same_v<T, std::uint8_t>) v.raw = vdupq_n_u8(static_cast<std::uint8_t>(t));
  else if constexpr (std::is_same_v<T, std::int16_t>) v.raw = vdupq_n_s16(static_cast<std::int16_t>(t));
  else if constexpr (std::is_same_v<T, std::uint16_t>) v.raw = vdupq_n_u16(static_cast<std::uint16_t>(t));
  else if constexpr (std::is_same_v<T, std::int32_t>) v.raw = vdupq_n_s32(static_cast<std::int32_t>(t));
  else if constexpr (std::is_same_v<T, std::uint32_t>) v.raw = vdupq_n_u32(static_cast<std::uint32_t>(t));
  else if constexpr (std::is_same_v<T, std::int64_t>) v.raw = vdupq_n_s64(static_cast<std::int64_t>(t));
  else v.raw = vdupq_n_u64(static_cast<std::uint64_t>(t));
  return v;
}
template <class D>
HYLE_FORCE_INLINE detail::VN<D> Zero(D d) noexcept {
  return Set(d, static_cast<TFromD<D>>(0));
}
template <class D>
using VFromD = decltype(Zero(D()));

// ---- load / store (vld1q/vst1q; NEON has no aligned-vs-unaligned distinction) --------
template <class D>
HYLE_FORCE_INLINE detail::VN<D> LoadU(D, const TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  detail::VN<D> v;
  if constexpr (std::is_same_v<T, float>) v.raw = vld1q_f32(p);
  else if constexpr (std::is_same_v<T, double>) v.raw = vld1q_f64(p);
  else if constexpr (std::is_same_v<T, std::int8_t>) v.raw = vld1q_s8(p);
  else if constexpr (std::is_same_v<T, std::uint8_t>) v.raw = vld1q_u8(p);
  else if constexpr (std::is_same_v<T, std::int16_t>) v.raw = vld1q_s16(p);
  else if constexpr (std::is_same_v<T, std::uint16_t>) v.raw = vld1q_u16(p);
  else if constexpr (std::is_same_v<T, std::int32_t>) v.raw = vld1q_s32(p);
  else if constexpr (std::is_same_v<T, std::uint32_t>) v.raw = vld1q_u32(p);
  else if constexpr (std::is_same_v<T, std::int64_t>) v.raw = vld1q_s64(p);
  else v.raw = vld1q_u64(p);
  return v;
}
template <class D>
HYLE_FORCE_INLINE detail::VN<D> Load(D d, const TFromD<D>* p) noexcept { return LoadU(d, p); }

template <typename T, std::size_t N>
HYLE_FORCE_INLINE void StoreU(VecNeon<T, N> v, Simd<T, N, 0>, T* p) noexcept {
  if constexpr (std::is_same_v<T, float>) vst1q_f32(p, v.raw);
  else if constexpr (std::is_same_v<T, double>) vst1q_f64(p, v.raw);
  else if constexpr (std::is_same_v<T, std::int8_t>) vst1q_s8(p, v.raw);
  else if constexpr (std::is_same_v<T, std::uint8_t>) vst1q_u8(p, v.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) vst1q_s16(p, v.raw);
  else if constexpr (std::is_same_v<T, std::uint16_t>) vst1q_u16(p, v.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) vst1q_s32(p, v.raw);
  else if constexpr (std::is_same_v<T, std::uint32_t>) vst1q_u32(p, v.raw);
  else if constexpr (std::is_same_v<T, std::int64_t>) vst1q_s64(p, v.raw);
  else vst1q_u64(p, v.raw);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE void Store(VecNeon<T, N> v, Simd<T, N, 0> d, T* p) noexcept { StoreU(v, d, p); }

// ---- arithmetic ---------------------------------------------------------------------
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Add(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = vaddq_f32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = vaddq_f64(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int8_t>) r.raw = vaddq_s8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint8_t>) r.raw = vaddq_u8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) r.raw = vaddq_s16(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint16_t>) r.raw = vaddq_u16(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) r.raw = vaddq_s32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint32_t>) r.raw = vaddq_u32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int64_t>) r.raw = vaddq_s64(a.raw, b.raw);
  else r.raw = vaddq_u64(a.raw, b.raw);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Sub(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = vsubq_f32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = vsubq_f64(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int8_t>) r.raw = vsubq_s8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint8_t>) r.raw = vsubq_u8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) r.raw = vsubq_s16(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint16_t>) r.raw = vsubq_u16(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) r.raw = vsubq_s32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint32_t>) r.raw = vsubq_u32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int64_t>) r.raw = vsubq_s64(a.raw, b.raw);
  else r.raw = vsubq_u64(a.raw, b.raw);
  return r;
}
namespace detail {
// i64/u64 low-64 multiply: NEON has NO vmulq_s64/vmulq_u64 (any width). Emulate via
// 32x32->64 partial products (mirror avx2.h's approach): with a = a_hi:a_lo, b likewise,
//   low64(a*b) = a_lo*b_lo + ((a_hi*b_lo + a_lo*b_hi) << 32).
// The top-32-bit cross term and the hi*hi term only affect bits >= 64, so they're dropped.
// Unsigned arithmetic yields the correct 2's-complement low 64 bits for signed too, so the
// signed path reinterprets to u64, computes here, reinterprets back. vmovn_u64 extracts the
// low 32 of each 64-bit lane; vshrn_n_u64(.,32) extracts the high 32; vmull_u32 widens 32x32.
HYLE_FORCE_INLINE uint64x2_t mul_u64(uint64x2_t au, uint64x2_t bu) noexcept {
  const uint32x2_t a_lo = vmovn_u64(au);
  const uint32x2_t b_lo = vmovn_u64(bu);
  const uint32x2_t a_hi = vshrn_n_u64(au, 32);
  const uint32x2_t b_hi = vshrn_n_u64(bu, 32);
  const uint64x2_t ll = vmull_u32(a_lo, b_lo);
  const uint64x2_t cross = vaddq_u64(vmull_u32(a_hi, b_lo), vmull_u32(a_lo, b_hi));
  return vaddq_u64(ll, vshlq_n_u64(cross, 32));
}
}  // namespace detail

template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Mul(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = vmulq_f32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = vmulq_f64(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int8_t>) r.raw = vmulq_s8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint8_t>) r.raw = vmulq_u8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) r.raw = vmulq_s16(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint16_t>) r.raw = vmulq_u16(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) r.raw = vmulq_s32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint32_t>) r.raw = vmulq_u32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int64_t>)
    r.raw = vreinterpretq_s64_u64(detail::mul_u64(vreinterpretq_u64_s64(a.raw),
                                                  vreinterpretq_u64_s64(b.raw)));
  else r.raw = detail::mul_u64(a.raw, b.raw);  // u64
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Div(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  static_assert(std::is_floating_point_v<T>, "Div is float/double only");
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = vdivq_f32(a.raw, b.raw);  // aarch64
  else r.raw = vdivq_f64(a.raw, b.raw);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Min(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = vminq_f32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = vminq_f64(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int8_t>) r.raw = vminq_s8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint8_t>) r.raw = vminq_u8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) r.raw = vminq_s16(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint16_t>) r.raw = vminq_u16(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) r.raw = vminq_s32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint32_t>) r.raw = vminq_u32(a.raw, b.raw);
  else r = IfThenElse(Gt(a, b), b, a);  // i64/u64: no vminq_[su]64 -> select (mirror avx2)
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Max(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = vmaxq_f32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = vmaxq_f64(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int8_t>) r.raw = vmaxq_s8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint8_t>) r.raw = vmaxq_u8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) r.raw = vmaxq_s16(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint16_t>) r.raw = vmaxq_u16(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) r.raw = vmaxq_s32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint32_t>) r.raw = vmaxq_u32(a.raw, b.raw);
  else r = IfThenElse(Gt(a, b), a, b);  // i64/u64: no vmaxq_[su]64 -> select (mirror avx2)
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Neg(VecNeon<T, N> a) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = vnegq_f32(a.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = vnegq_f64(a.raw);
  else if constexpr (std::is_same_v<T, std::int8_t>) r.raw = vnegq_s8(a.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) r.raw = vnegq_s16(a.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) r.raw = vnegq_s32(a.raw);
  else if constexpr (std::is_same_v<T, std::int64_t>) r.raw = vnegq_s64(a.raw);  // aarch64
  // unsigned: Neg = 0 - a (2's-complement wraparound = scalar T(-x)); vsubq from zero.
  else if constexpr (std::is_same_v<T, std::uint8_t>) r.raw = vsubq_u8(vdupq_n_u8(0), a.raw);
  else if constexpr (std::is_same_v<T, std::uint16_t>) r.raw = vsubq_u16(vdupq_n_u16(0), a.raw);
  else if constexpr (std::is_same_v<T, std::uint32_t>) r.raw = vsubq_u32(vdupq_n_u32(0), a.raw);
  else r.raw = vsubq_u64(vdupq_n_u64(0), a.raw);  // u64
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Abs(VecNeon<T, N> a) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = vabsq_f32(a.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = vabsq_f64(a.raw);
  else if constexpr (std::is_same_v<T, std::int8_t>) r.raw = vabsq_s8(a.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) r.raw = vabsq_s16(a.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) r.raw = vabsq_s32(a.raw);
  else if constexpr (std::is_same_v<T, std::int64_t>) r.raw = vabsq_s64(a.raw);  // aarch64 native
  else r.raw = a.raw;  // unsigned (u8/u16/u32/u64): abs is identity
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Sqrt(VecNeon<T, N> a) noexcept {
  static_assert(std::is_floating_point_v<T>, "Sqrt is float/double only");
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = vsqrtq_f32(a.raw);
  else r.raw = vsqrtq_f64(a.raw);
  return r;
}

// Float rounding to an integral value (result stays float): Round=nearest-even (vrndnq),
// Floor=toward -inf (vrndmq), Ceil=toward +inf (vrndpq), Trunc=toward zero (vrndq). All
// native ARMv8 (FRINTN/M/P/Z). Float/double only.
#define HYLE_SIMD_NEON_ROUND(NAME, FF, FD)                                   \
  template <typename T, std::size_t N>                                       \
  HYLE_FORCE_INLINE VecNeon<T, N> NAME(VecNeon<T, N> a) noexcept {           \
    static_assert(std::is_floating_point_v<T>, #NAME " is float/double only"); \
    VecNeon<T, N> r;                                                         \
    if constexpr (std::is_same_v<T, float>) r.raw = FF(a.raw);              \
    else r.raw = FD(a.raw);                                                  \
    return r;                                                                \
  }
HYLE_SIMD_NEON_ROUND(Round, vrndnq_f32, vrndnq_f64)
HYLE_SIMD_NEON_ROUND(Floor, vrndmq_f32, vrndmq_f64)
HYLE_SIMD_NEON_ROUND(Ceil, vrndpq_f32, vrndpq_f64)
HYLE_SIMD_NEON_ROUND(Trunc, vrndq_f32, vrndq_f64)
#undef HYLE_SIMD_NEON_ROUND

// FMA: NEON vfmaq computes c + a*b (single rounding). NegMulAdd = c - a*b via vfmsq.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> MulAdd(VecNeon<T, N> a, VecNeon<T, N> b, VecNeon<T, N> c) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = vfmaq_f32(c.raw, a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = vfmaq_f64(c.raw, a.raw, b.raw);
  else r = Add(Mul(a, b), c);  // no integer FMA
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> NegMulAdd(VecNeon<T, N> a, VecNeon<T, N> b, VecNeon<T, N> c) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = vfmsq_f32(c.raw, a.raw, b.raw);   // c - a*b
  else if constexpr (std::is_same_v<T, double>) r.raw = vfmsq_f64(c.raw, a.raw, b.raw);
  else r = Sub(c, Mul(a, b));
  return r;
}

// ---- bitwise (operate on the integer reinterpretation for float lanes) --------------
namespace detail {
// reinterpret a lane vector to u32x4 for bitwise ops, and back. All lane types are the
// same 128 bits, so this is a pure type pun (no data movement); u32x4 is the common hub.
template <typename T>
HYLE_FORCE_INLINE uint32x4_t to_u32(RawNeonT<T> v) noexcept {
  if constexpr (std::is_same_v<T, float>) return vreinterpretq_u32_f32(v);
  else if constexpr (std::is_same_v<T, double>) return vreinterpretq_u32_f64(v);
  else if constexpr (std::is_same_v<T, std::int8_t>) return vreinterpretq_u32_s8(v);
  else if constexpr (std::is_same_v<T, std::uint8_t>) return vreinterpretq_u32_u8(v);
  else if constexpr (std::is_same_v<T, std::int16_t>) return vreinterpretq_u32_s16(v);
  else if constexpr (std::is_same_v<T, std::uint16_t>) return vreinterpretq_u32_u16(v);
  else if constexpr (std::is_same_v<T, std::int32_t>) return vreinterpretq_u32_s32(v);
  else if constexpr (std::is_same_v<T, std::int64_t>) return vreinterpretq_u32_s64(v);
  else if constexpr (std::is_same_v<T, std::uint64_t>) return vreinterpretq_u32_u64(v);
  else return v;  // uint32_t
}
template <typename T>
HYLE_FORCE_INLINE RawNeonT<T> from_u32(uint32x4_t v) noexcept {
  if constexpr (std::is_same_v<T, float>) return vreinterpretq_f32_u32(v);
  else if constexpr (std::is_same_v<T, double>) return vreinterpretq_f64_u32(v);
  else if constexpr (std::is_same_v<T, std::int8_t>) return vreinterpretq_s8_u32(v);
  else if constexpr (std::is_same_v<T, std::uint8_t>) return vreinterpretq_u8_u32(v);
  else if constexpr (std::is_same_v<T, std::int16_t>) return vreinterpretq_s16_u32(v);
  else if constexpr (std::is_same_v<T, std::uint16_t>) return vreinterpretq_u16_u32(v);
  else if constexpr (std::is_same_v<T, std::int32_t>) return vreinterpretq_s32_u32(v);
  else if constexpr (std::is_same_v<T, std::int64_t>) return vreinterpretq_s64_u32(v);
  else if constexpr (std::is_same_v<T, std::uint64_t>) return vreinterpretq_u64_u32(v);
  else return v;  // uint32_t
}
}  // namespace detail

template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> And(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  VecNeon<T, N> r;
  r.raw = detail::from_u32<T>(vandq_u32(detail::to_u32<T>(a.raw), detail::to_u32<T>(b.raw)));
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Or(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  VecNeon<T, N> r;
  r.raw = detail::from_u32<T>(vorrq_u32(detail::to_u32<T>(a.raw), detail::to_u32<T>(b.raw)));
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Xor(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  VecNeon<T, N> r;
  r.raw = detail::from_u32<T>(veorq_u32(detail::to_u32<T>(a.raw), detail::to_u32<T>(b.raw)));
  return r;
}
// AndNot(a,b) = ~a & b (Highway order). NEON vbicq(x,y) = x & ~y, so swap args: vbicq(b,a).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> AndNot(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  VecNeon<T, N> r;
  r.raw = detail::from_u32<T>(vbicq_u32(detail::to_u32<T>(b.raw), detail::to_u32<T>(a.raw)));
  return r;
}

// ---- shifts (integer lanes, compile-time immediate count) ---------------------------
// NEON is easier than x86 here: vshlq_n_* / vshrq_n_* exist for EVERY integer width
// (8/16/32/64) as compile-time-immediate intrinsics, so NO emulation (unlike avx2's i8/i64
// masking + sign fixup). ShiftLeft is logical for all widths (vshlq_n_*). ShiftRight picks
// the SIGNED intrinsic (vshrq_n_s* = arithmetic, sign-extending) for signed T and the
// UNSIGNED intrinsic (vshrq_n_u* = logical, zero-fill) for unsigned T -- matching scalar >>.
// kBits is the template param, passed directly as the immediate; the caller guarantees a
// valid count (same contract as avx2/scalar). Dispatch per exact lane type (Add idiom).
template <int kBits, typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> ShiftLeft(VecNeon<T, N> a) noexcept {
  static_assert(std::is_integral_v<T>, "shifts are integer-lane only");
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, std::int8_t>) r.raw = vshlq_n_s8(a.raw, kBits);
  else if constexpr (std::is_same_v<T, std::uint8_t>) r.raw = vshlq_n_u8(a.raw, kBits);
  else if constexpr (std::is_same_v<T, std::int16_t>) r.raw = vshlq_n_s16(a.raw, kBits);
  else if constexpr (std::is_same_v<T, std::uint16_t>) r.raw = vshlq_n_u16(a.raw, kBits);
  else if constexpr (std::is_same_v<T, std::int32_t>) r.raw = vshlq_n_s32(a.raw, kBits);
  else if constexpr (std::is_same_v<T, std::uint32_t>) r.raw = vshlq_n_u32(a.raw, kBits);
  else if constexpr (std::is_same_v<T, std::int64_t>) r.raw = vshlq_n_s64(a.raw, kBits);
  else r.raw = vshlq_n_u64(a.raw, kBits);  // u64
  return r;
}
template <int kBits, typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> ShiftRight(VecNeon<T, N> a) noexcept {
  static_assert(std::is_integral_v<T>, "shifts are integer-lane only");
  VecNeon<T, N> r;
  // Signed -> vshrq_n_s* (arithmetic); unsigned -> vshrq_n_u* (logical). Both native per width.
  if constexpr (std::is_same_v<T, std::int8_t>) r.raw = vshrq_n_s8(a.raw, kBits);
  else if constexpr (std::is_same_v<T, std::uint8_t>) r.raw = vshrq_n_u8(a.raw, kBits);
  else if constexpr (std::is_same_v<T, std::int16_t>) r.raw = vshrq_n_s16(a.raw, kBits);
  else if constexpr (std::is_same_v<T, std::uint16_t>) r.raw = vshrq_n_u16(a.raw, kBits);
  else if constexpr (std::is_same_v<T, std::int32_t>) r.raw = vshrq_n_s32(a.raw, kBits);
  else if constexpr (std::is_same_v<T, std::uint32_t>) r.raw = vshrq_n_u32(a.raw, kBits);
  else if constexpr (std::is_same_v<T, std::int64_t>) r.raw = vshrq_n_s64(a.raw, kBits);
  else r.raw = vshrq_n_u64(a.raw, kBits);  // u64
  return r;
}

// ---- saturated arithmetic + rounding average (8/16-bit integer lanes) ---------------
// Native NEON: vqaddq/vqsubq (saturating), vrhaddq (rounding halving add = (a+b+1)>>1).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> SaturatedAdd(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  static_assert(std::is_integral_v<T> && sizeof(T) <= 2, "SaturatedAdd is 8/16-bit int only");
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, std::int8_t>) r.raw = vqaddq_s8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint8_t>) r.raw = vqaddq_u8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) r.raw = vqaddq_s16(a.raw, b.raw);
  else r.raw = vqaddq_u16(a.raw, b.raw);  // u16
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> SaturatedSub(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  static_assert(std::is_integral_v<T> && sizeof(T) <= 2, "SaturatedSub is 8/16-bit int only");
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, std::int8_t>) r.raw = vqsubq_s8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint8_t>) r.raw = vqsubq_u8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) r.raw = vqsubq_s16(a.raw, b.raw);
  else r.raw = vqsubq_u16(a.raw, b.raw);  // u16
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> AverageRound(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  static_assert(std::is_unsigned_v<T> && sizeof(T) <= 2, "AverageRound is u8/u16 only");
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, std::uint8_t>) r.raw = vrhaddq_u8(a.raw, b.raw);
  else r.raw = vrhaddq_u16(a.raw, b.raw);  // u16
  return r;
}

// ---- type conversion (same lane count: f32<->i32) -----------------------------------
// ConvertTo: f32->i32 truncate toward zero (vcvtq_s32_f32), i32->f32 (vcvtq_f32_s32).
// RoundToInt: f32->i32 round to nearest-even (vcvtnq_s32_f32). 4 lanes both sides = one
// float32x4_t <-> int32x4_t. Width-changing PromoteTo/DemoteTo deferred (see scalar.h).
template <typename DstT, std::size_t N, typename SrcT>
HYLE_FORCE_INLINE VecNeon<DstT, N> ConvertTo(Simd<DstT, N, 0>, VecNeon<SrcT, N> v) noexcept {
  VecNeon<DstT, N> r;
  if constexpr (std::is_same_v<SrcT, float> && std::is_same_v<DstT, std::int32_t>)
    r.raw = vcvtq_s32_f32(v.raw);        // truncate toward zero
  else if constexpr (std::is_same_v<SrcT, std::int32_t> && std::is_same_v<DstT, float>)
    r.raw = vcvtq_f32_s32(v.raw);
  else static_assert(sizeof(DstT) == 0, "neon ConvertTo: only f32<->i32 (same width) supported");
  return r;
}
template <typename DstT, std::size_t N, typename SrcT>
HYLE_FORCE_INLINE VecNeon<DstT, N> RoundToInt(Simd<DstT, N, 0>, VecNeon<SrcT, N> v) noexcept {
  static_assert(std::is_same_v<SrcT, float> && std::is_same_v<DstT, std::int32_t>,
                "neon RoundToInt: f32->i32");
  VecNeon<DstT, N> r;
  r.raw = vcvtnq_s32_f32(v.raw);         // round to nearest even
  return r;
}

// ---- comparison -> mask (vceqq/vcgtq/... return all-1s/all-0s unsigned lanes) --------
// NEON has NATIVE per-width compares for every lane type, signed AND unsigned AND 64-bit
// (aarch64) -- no x86-style sign-flip trick needed. Each intrinsic already returns the
// MaskRawNeonT<T> (unsigned same-width) that MaskNeon<T,N>::raw expects, so we just pick
// by type. Ne = bitwise-not of Eq: vmvnq_* exists for 8/16/32 but NOT 64-bit, so the
// 64-bit widths reinterpret to u32, vmvnq_u32, reinterpret back.
namespace detail {
template <typename T>
HYLE_FORCE_INLINE MaskRawNeonT<T> mask_not(MaskRawNeonT<T> m) noexcept {
  if constexpr (std::is_same_v<T, std::int8_t> || std::is_same_v<T, std::uint8_t>)
    return vmvnq_u8(m);
  else if constexpr (std::is_same_v<T, std::int16_t> || std::is_same_v<T, std::uint16_t>)
    return vmvnq_u16(m);
  else if constexpr (sizeof(T) == 8)  // double / i64 / u64: no vmvnq_u64
    return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(m)));
  else return vmvnq_u32(m);  // f32 / i32 / u32
}
}  // namespace detail

template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskNeon<T, N> Eq(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  MaskNeon<T, N> m;
  if constexpr (std::is_same_v<T, float>) m.raw = vceqq_f32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) m.raw = vceqq_f64(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int8_t>) m.raw = vceqq_s8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint8_t>) m.raw = vceqq_u8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) m.raw = vceqq_s16(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint16_t>) m.raw = vceqq_u16(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) m.raw = vceqq_s32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint32_t>) m.raw = vceqq_u32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int64_t>) m.raw = vceqq_s64(a.raw, b.raw);  // aarch64
  else m.raw = vceqq_u64(a.raw, b.raw);  // u64 (aarch64)
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskNeon<T, N> Ne(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  MaskNeon<T, N> m = Eq(a, b);
  m.raw = detail::mask_not<T>(m.raw);
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskNeon<T, N> Gt(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  MaskNeon<T, N> m;
  if constexpr (std::is_same_v<T, float>) m.raw = vcgtq_f32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) m.raw = vcgtq_f64(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int8_t>) m.raw = vcgtq_s8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint8_t>) m.raw = vcgtq_u8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) m.raw = vcgtq_s16(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint16_t>) m.raw = vcgtq_u16(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) m.raw = vcgtq_s32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint32_t>) m.raw = vcgtq_u32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int64_t>) m.raw = vcgtq_s64(a.raw, b.raw);  // aarch64
  else m.raw = vcgtq_u64(a.raw, b.raw);  // u64 (aarch64)
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskNeon<T, N> Lt(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  MaskNeon<T, N> m;
  if constexpr (std::is_same_v<T, float>) m.raw = vcltq_f32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) m.raw = vcltq_f64(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int8_t>) m.raw = vcltq_s8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint8_t>) m.raw = vcltq_u8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) m.raw = vcltq_s16(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint16_t>) m.raw = vcltq_u16(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) m.raw = vcltq_s32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint32_t>) m.raw = vcltq_u32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int64_t>) m.raw = vcltq_s64(a.raw, b.raw);  // aarch64
  else m.raw = vcltq_u64(a.raw, b.raw);  // u64 (aarch64)
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskNeon<T, N> Ge(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  MaskNeon<T, N> m;
  if constexpr (std::is_same_v<T, float>) m.raw = vcgeq_f32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) m.raw = vcgeq_f64(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int8_t>) m.raw = vcgeq_s8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint8_t>) m.raw = vcgeq_u8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) m.raw = vcgeq_s16(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint16_t>) m.raw = vcgeq_u16(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) m.raw = vcgeq_s32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint32_t>) m.raw = vcgeq_u32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int64_t>) m.raw = vcgeq_s64(a.raw, b.raw);  // aarch64
  else m.raw = vcgeq_u64(a.raw, b.raw);  // u64 (aarch64)
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskNeon<T, N> Le(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  MaskNeon<T, N> m;
  if constexpr (std::is_same_v<T, float>) m.raw = vcleq_f32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) m.raw = vcleq_f64(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int8_t>) m.raw = vcleq_s8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint8_t>) m.raw = vcleq_u8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) m.raw = vcleq_s16(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint16_t>) m.raw = vcleq_u16(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) m.raw = vcleq_s32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint32_t>) m.raw = vcleq_u32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int64_t>) m.raw = vcleq_s64(a.raw, b.raw);  // aarch64
  else m.raw = vcleq_u64(a.raw, b.raw);  // u64 (aarch64)
  return m;
}

// ---- mask ops -----------------------------------------------------------------------
// IfThenElse via vbslq(mask, yes, no): bit-select where mask bits set -> yes.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> IfThenElse(MaskNeon<T, N> m, VecNeon<T, N> yes, VecNeon<T, N> no) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = vbslq_f32(m.raw, yes.raw, no.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = vbslq_f64(m.raw, yes.raw, no.raw);
  else if constexpr (std::is_same_v<T, std::int8_t>) r.raw = vbslq_s8(m.raw, yes.raw, no.raw);
  else if constexpr (std::is_same_v<T, std::uint8_t>) r.raw = vbslq_u8(m.raw, yes.raw, no.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) r.raw = vbslq_s16(m.raw, yes.raw, no.raw);
  else if constexpr (std::is_same_v<T, std::uint16_t>) r.raw = vbslq_u16(m.raw, yes.raw, no.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) r.raw = vbslq_s32(m.raw, yes.raw, no.raw);
  else if constexpr (std::is_same_v<T, std::uint32_t>) r.raw = vbslq_u32(m.raw, yes.raw, no.raw);
  else if constexpr (std::is_same_v<T, std::int64_t>) r.raw = vbslq_s64(m.raw, yes.raw, no.raw);
  else r.raw = vbslq_u64(m.raw, yes.raw, no.raw);  // u64
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> IfThenElseZero(MaskNeon<T, N> m, VecNeon<T, N> yes) noexcept {
  return IfThenElse(m, yes, Zero(Simd<T, N, 0>()));
}

namespace detail {
// Fold a per-lane all-1s/all-0s mask to one bit per lane (LSB = lane 0). NEON has no
// movemask, so the idiom is: AND with per-lane bit-weights {1,2,4,...}, then horizontal-add.
// The sum's set bits ARE the lane bitmask (each lane contributes its weight or 0, weights
// are disjoint powers of two). Generalized per lane width; the 8-bit case is the trap.
HYLE_FORCE_INLINE unsigned mask_bits_4(uint32x4_t m) noexcept {  // 4x 32-bit lanes
  const uint32x4_t w = {1u, 2u, 4u, 8u};
  return static_cast<unsigned>(vaddvq_u32(vandq_u32(m, w)));  // aarch64 vaddvq
}
HYLE_FORCE_INLINE unsigned mask_bits_2(uint64x2_t m) noexcept {  // 2x 64-bit lanes
  const uint64x2_t w = {1u, 2u};
  return static_cast<unsigned>(vaddvq_u64(vandq_u64(m, w)));
}
HYLE_FORCE_INLINE unsigned mask_bits_8(uint16x8_t m) noexcept {  // 8x 16-bit lanes
  // weights 1..128 sum to <=255, fits a u16 accumulator -> single vaddvq_u16.
  const uint16x8_t w = {1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u};
  return static_cast<unsigned>(vaddvq_u16(vandq_u16(m, w)));
}
HYLE_FORCE_INLINE unsigned mask_bits_16(uint8x16_t m) noexcept {  // 16x 8-bit lanes
  // THE trap: 16 weights (up to 32768) can't live in u8 lanes, and a single vaddvq_u8 would
  // sum all 16 into one wraparound byte. So use per-8-lane weights {1..128} (repeated in
  // both halves), then reduce each 8-lane half separately (vaddv_u8 -> u8, max 255 fits),
  // giving low byte = bits 0..7 and high byte = bits 8..15; combine as lo | (hi<<8).
  const uint8x16_t w = {1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u,
                        1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u};
  const uint8x16_t masked = vandq_u8(m, w);
  const unsigned lo = vaddv_u8(vget_low_u8(masked));
  const unsigned hi = vaddv_u8(vget_high_u8(masked));
  return lo | (hi << 8);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE unsigned mask_bits(MaskNeon<T, N> m) noexcept {
  if constexpr (sizeof(T) == 1) return mask_bits_16(m.raw);
  else if constexpr (sizeof(T) == 2) return mask_bits_8(m.raw);
  else if constexpr (sizeof(T) == 8) return mask_bits_2(m.raw);  // f64 / i64 / u64
  else return mask_bits_4(m.raw);  // f32 / i32 / u32
}
}  // namespace detail

template <typename T, std::size_t N>
HYLE_FORCE_INLINE bool AllTrue(Simd<T, N, 0>, MaskNeon<T, N> m) noexcept {
  return detail::mask_bits(m) == (1u << N) - 1u;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE bool AllFalse(Simd<T, N, 0>, MaskNeon<T, N> m) noexcept {
  return detail::mask_bits(m) == 0u;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE std::size_t CountTrue(Simd<T, N, 0>, MaskNeon<T, N> m) noexcept {
  return static_cast<std::size_t>(std::popcount(detail::mask_bits(m)));
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE std::ptrdiff_t FindFirstTrue(Simd<T, N, 0>, MaskNeon<T, N> m) noexcept {
  const unsigned b = detail::mask_bits(m);
  return b ? static_cast<std::ptrdiff_t>(std::countr_zero(b)) : -1;
}

// ---- reductions (aarch64 horizontal vaddvq/vminvq/vmaxvq) ----------------------------
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T GetLane(VecNeon<T, N> v) noexcept {
  if constexpr (std::is_same_v<T, float>) return vgetq_lane_f32(v.raw, 0);
  else if constexpr (std::is_same_v<T, double>) return vgetq_lane_f64(v.raw, 0);
  else if constexpr (std::is_same_v<T, std::int8_t>) return vgetq_lane_s8(v.raw, 0);
  else if constexpr (std::is_same_v<T, std::uint8_t>) return vgetq_lane_u8(v.raw, 0);
  else if constexpr (std::is_same_v<T, std::int16_t>) return vgetq_lane_s16(v.raw, 0);
  else if constexpr (std::is_same_v<T, std::uint16_t>) return vgetq_lane_u16(v.raw, 0);
  else if constexpr (std::is_same_v<T, std::int32_t>) return vgetq_lane_s32(v.raw, 0);
  else if constexpr (std::is_same_v<T, std::uint32_t>) return vgetq_lane_u32(v.raw, 0);
  else if constexpr (std::is_same_v<T, std::int64_t>) return vgetq_lane_s64(v.raw, 0);
  else return vgetq_lane_u64(v.raw, 0);  // u64
}
namespace detail {
// Scalar fallback for reductions that lack a clean across-vector NEON reduce for the lane
// type (i64/u64 Min/Max: aarch64 has vaddvq_[su]64 but NO vminvq/vmaxvq at 64-bit width).
// Store to a stack array and scalar-reduce -- correct + simple, and horizontal reduce is
// not the hot path (mirror avx2.h's reduce_scalar).
enum class RedOp { kMin, kMax };
template <typename T, std::size_t N, RedOp op>
HYLE_FORCE_INLINE T reduce_scalar(VecNeon<T, N> v) noexcept {
  alignas(16) T tmp[N];
  StoreU(v, Simd<T, N, 0>(), tmp);
  T acc = tmp[0];
  for (std::size_t i = 1; i < N; ++i) {
    if constexpr (op == RedOp::kMin) acc = tmp[i] < acc ? tmp[i] : acc;
    else acc = tmp[i] > acc ? tmp[i] : acc;
  }
  return acc;
}
}  // namespace detail

template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceSum(Simd<T, N, 0>, VecNeon<T, N> v) noexcept {
  if constexpr (std::is_same_v<T, float>) return vaddvq_f32(v.raw);
  else if constexpr (std::is_same_v<T, double>) return vaddvq_f64(v.raw);
  else if constexpr (std::is_same_v<T, std::int8_t>) return vaddvq_s8(v.raw);
  else if constexpr (std::is_same_v<T, std::uint8_t>) return vaddvq_u8(v.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) return vaddvq_s16(v.raw);
  else if constexpr (std::is_same_v<T, std::uint16_t>) return vaddvq_u16(v.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) return vaddvq_s32(v.raw);
  else if constexpr (std::is_same_v<T, std::uint32_t>) return vaddvq_u32(v.raw);
  else if constexpr (std::is_same_v<T, std::int64_t>) return vaddvq_s64(v.raw);  // aarch64
  else return vaddvq_u64(v.raw);  // u64 (aarch64)
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceMin(Simd<T, N, 0>, VecNeon<T, N> v) noexcept {
  if constexpr (std::is_same_v<T, float>) return vminvq_f32(v.raw);
  else if constexpr (std::is_same_v<T, double>) return vgetq_lane_f64(vpminq_f64(v.raw, v.raw), 0);
  else if constexpr (std::is_same_v<T, std::int8_t>) return vminvq_s8(v.raw);
  else if constexpr (std::is_same_v<T, std::uint8_t>) return vminvq_u8(v.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) return vminvq_s16(v.raw);
  else if constexpr (std::is_same_v<T, std::uint16_t>) return vminvq_u16(v.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) return vminvq_s32(v.raw);
  else if constexpr (std::is_same_v<T, std::uint32_t>) return vminvq_u32(v.raw);
  else return detail::reduce_scalar<T, N, detail::RedOp::kMin>(v);  // i64/u64: no vminvq_64
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceMax(Simd<T, N, 0>, VecNeon<T, N> v) noexcept {
  if constexpr (std::is_same_v<T, float>) return vmaxvq_f32(v.raw);
  else if constexpr (std::is_same_v<T, double>) return vgetq_lane_f64(vpmaxq_f64(v.raw, v.raw), 0);
  else if constexpr (std::is_same_v<T, std::int8_t>) return vmaxvq_s8(v.raw);
  else if constexpr (std::is_same_v<T, std::uint8_t>) return vmaxvq_u8(v.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) return vmaxvq_s16(v.raw);
  else if constexpr (std::is_same_v<T, std::uint16_t>) return vmaxvq_u16(v.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) return vmaxvq_s32(v.raw);
  else if constexpr (std::is_same_v<T, std::uint32_t>) return vmaxvq_u32(v.raw);
  else return detail::reduce_scalar<T, N, detail::RedOp::kMax>(v);  // i64/u64: no vmaxvq_64
}

// ---- masked / partial load-store (tail handling) ------------------------------------
// FirstN(d,n) -> mask (per-width all-1s/all-0s unsigned lanes, matching MaskRawNeon<T>);
// built via a stack array + vld1q of the matching width. MaskedLoad(Or) reuse IfThenElse
// (like Highway's NEON path -- no native maskload). LoadN/StoreN touch only the first k
// lanes (fault-safe); BlendedStore writes only on-lanes via mask_bits (fault-safe).
template <class D>
HYLE_FORCE_INLINE MaskNeon<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> FirstN(D, std::size_t n) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  MaskNeon<T, kN> m;
  if constexpr (sizeof(T) == 1) {
    std::uint8_t tmp[16];
    for (std::size_t i = 0; i < kN; ++i) tmp[i] = (i < n) ? 0xFFu : 0u;
    m.raw = vld1q_u8(tmp);
  } else if constexpr (sizeof(T) == 2) {
    std::uint16_t tmp[8];
    for (std::size_t i = 0; i < kN; ++i) tmp[i] = (i < n) ? 0xFFFFu : 0u;
    m.raw = vld1q_u16(tmp);
  } else if constexpr (sizeof(T) == 4) {
    std::uint32_t tmp[4];
    for (std::size_t i = 0; i < kN; ++i) tmp[i] = (i < n) ? 0xFFFFFFFFu : 0u;
    m.raw = vld1q_u32(tmp);
  } else {
    std::uint64_t tmp[2];
    for (std::size_t i = 0; i < kN; ++i) tmp[i] = (i < n) ? ~0ull : 0ull;
    m.raw = vld1q_u64(tmp);
  }
  return m;
}
template <class D>
HYLE_FORCE_INLINE detail::VN<D> MaskedLoad(MaskNeon<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m,
                                           D d, const TFromD<D>* p) noexcept {
  return IfThenElseZero(m, LoadU(d, p));
}
template <class D>
HYLE_FORCE_INLINE detail::VN<D> MaskedLoadOr(detail::VN<D> no,
                                             MaskNeon<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m,
                                             D d, const TFromD<D>* p) noexcept {
  return IfThenElse(m, LoadU(d, p), no);
}
template <class D>
HYLE_FORCE_INLINE void BlendedStore(detail::VN<D> v,
                                    MaskNeon<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m, D d,
                                    TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  alignas(16) T tmp[kN];
  Store(v, d, tmp);
  const unsigned bits = detail::mask_bits(m);
  for (std::size_t i = 0; i < kN; ++i)
    if ((bits >> i) & 1u) p[i] = tmp[i];
}
template <class D>
HYLE_FORCE_INLINE detail::VN<D> LoadN(D d, const TFromD<D>* p, std::size_t n) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  const std::size_t k = n < kN ? n : kN;
  alignas(16) T tmp[kN] = {};
  for (std::size_t i = 0; i < k; ++i) tmp[i] = p[i];
  return Load(d, tmp);
}
template <class D>
HYLE_FORCE_INLINE void StoreN(detail::VN<D> v, D d, TFromD<D>* p, std::size_t n) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  const std::size_t k = n < kN ? n : kN;
  alignas(16) T tmp[kN];
  Store(v, d, tmp);
  for (std::size_t i = 0; i < k; ++i) p[i] = tmp[i];
}

}  // namespace hyle::simd
