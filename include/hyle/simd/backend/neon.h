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
// unchanged. Lane types f32/f64/i32/u32; NATIVE FULL WIDTH (128-bit = 4 f32 / 2 f64 / 4 i32).
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
template <typename T>
struct RawNeon;
template <> struct RawNeon<float> { using type = float32x4_t; };
template <> struct RawNeon<double> { using type = float64x2_t; };
template <> struct RawNeon<std::int32_t> { using type = int32x4_t; };
template <> struct RawNeon<std::uint32_t> { using type = uint32x4_t; };
template <typename T> using RawNeonT = typename RawNeon<T>::type;

// mask raw is always an unsigned lane vector (compare results)
template <typename T> struct MaskNeon;
template <> struct MaskNeon<float> { using type = uint32x4_t; };
template <> struct MaskNeon<double> { using type = uint64x2_t; };
template <> struct MaskNeon<std::int32_t> { using type = uint32x4_t; };
template <> struct MaskNeon<std::uint32_t> { using type = uint32x4_t; };
template <typename T> using MaskNeonT = typename MaskNeon<T>::type;

template <typename T>
inline constexpr bool kSupported =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::uint32_t>;
}  // namespace detail

template <typename T, std::size_t N>
struct VecNeon {
  static_assert(detail::kSupported<T>, "neon backend lane type unsupported");
  typename detail::RawNeonT<T> raw;
  using PrivateT = T;
  static constexpr std::size_t kPrivateN = N;
};
template <typename T, std::size_t N>
struct MaskN {
  typename detail::MaskNeonT<T> raw;
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
using MN = MaskN<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)>;
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
  else if constexpr (std::is_same_v<T, std::int32_t>) v.raw = vdupq_n_s32(static_cast<std::int32_t>(t));
  else v.raw = vdupq_n_u32(static_cast<std::uint32_t>(t));
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
  else if constexpr (std::is_same_v<T, std::int32_t>) v.raw = vld1q_s32(p);
  else v.raw = vld1q_u32(p);
  return v;
}
template <class D>
HYLE_FORCE_INLINE detail::VN<D> Load(D d, const TFromD<D>* p) noexcept { return LoadU(d, p); }

template <typename T, std::size_t N>
HYLE_FORCE_INLINE void StoreU(VecNeon<T, N> v, Simd<T, N, 0>, T* p) noexcept {
  if constexpr (std::is_same_v<T, float>) vst1q_f32(p, v.raw);
  else if constexpr (std::is_same_v<T, double>) vst1q_f64(p, v.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) vst1q_s32(p, v.raw);
  else vst1q_u32(p, v.raw);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE void Store(VecNeon<T, N> v, Simd<T, N, 0> d, T* p) noexcept { StoreU(v, d, p); }

// ---- arithmetic ---------------------------------------------------------------------
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Add(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = vaddq_f32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = vaddq_f64(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) r.raw = vaddq_s32(a.raw, b.raw);
  else r.raw = vaddq_u32(a.raw, b.raw);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Sub(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = vsubq_f32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = vsubq_f64(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) r.raw = vsubq_s32(a.raw, b.raw);
  else r.raw = vsubq_u32(a.raw, b.raw);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Mul(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = vmulq_f32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = vmulq_f64(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) r.raw = vmulq_s32(a.raw, b.raw);
  else r.raw = vmulq_u32(a.raw, b.raw);
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
  else if constexpr (std::is_same_v<T, std::int32_t>) r.raw = vminq_s32(a.raw, b.raw);
  else r.raw = vminq_u32(a.raw, b.raw);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Max(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = vmaxq_f32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = vmaxq_f64(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) r.raw = vmaxq_s32(a.raw, b.raw);
  else r.raw = vmaxq_u32(a.raw, b.raw);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Neg(VecNeon<T, N> a) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = vnegq_f32(a.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = vnegq_f64(a.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) r.raw = vnegq_s32(a.raw);
  else r.raw = vreinterpretq_u32_s32(vnegq_s32(vreinterpretq_s32_u32(a.raw)));
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> Abs(VecNeon<T, N> a) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = vabsq_f32(a.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = vabsq_f64(a.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) r.raw = vabsq_s32(a.raw);
  else r.raw = a.raw;  // u32: identity
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
// reinterpret a lane vector to u32x4 for bitwise ops, and back
template <typename T>
HYLE_FORCE_INLINE uint32x4_t to_u32(RawNeonT<T> v) noexcept {
  if constexpr (std::is_same_v<T, float>) return vreinterpretq_u32_f32(v);
  else if constexpr (std::is_same_v<T, double>) return vreinterpretq_u32_f64(v);
  else if constexpr (std::is_same_v<T, std::int32_t>) return vreinterpretq_u32_s32(v);
  else return v;
}
template <typename T>
HYLE_FORCE_INLINE RawNeonT<T> from_u32(uint32x4_t v) noexcept {
  if constexpr (std::is_same_v<T, float>) return vreinterpretq_f32_u32(v);
  else if constexpr (std::is_same_v<T, double>) return vreinterpretq_f64_u32(v);
  else if constexpr (std::is_same_v<T, std::int32_t>) return vreinterpretq_s32_u32(v);
  else return v;
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

// ---- comparison -> mask (vceqq/vcgtq/... return all-1s/all-0s unsigned lanes) --------
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskN<T, N> Eq(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  MaskN<T, N> m;
  if constexpr (std::is_same_v<T, float>) m.raw = vceqq_f32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) m.raw = vceqq_f64(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) m.raw = vceqq_s32(a.raw, b.raw);
  else m.raw = vceqq_u32(a.raw, b.raw);
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskN<T, N> Ne(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  MaskN<T, N> m = Eq(a, b);
  if constexpr (std::is_same_v<T, double>) m.raw = vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(m.raw)));
  else m.raw = vmvnq_u32(m.raw);
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskN<T, N> Gt(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  MaskN<T, N> m;
  if constexpr (std::is_same_v<T, float>) m.raw = vcgtq_f32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) m.raw = vcgtq_f64(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) m.raw = vcgtq_s32(a.raw, b.raw);
  else m.raw = vcgtq_u32(a.raw, b.raw);
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskN<T, N> Lt(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  MaskN<T, N> m;
  if constexpr (std::is_same_v<T, float>) m.raw = vcltq_f32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) m.raw = vcltq_f64(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) m.raw = vcltq_s32(a.raw, b.raw);
  else m.raw = vcltq_u32(a.raw, b.raw);
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskN<T, N> Ge(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  MaskN<T, N> m;
  if constexpr (std::is_same_v<T, float>) m.raw = vcgeq_f32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) m.raw = vcgeq_f64(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) m.raw = vcgeq_s32(a.raw, b.raw);
  else m.raw = vcgeq_u32(a.raw, b.raw);
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE MaskN<T, N> Le(VecNeon<T, N> a, VecNeon<T, N> b) noexcept {
  MaskN<T, N> m;
  if constexpr (std::is_same_v<T, float>) m.raw = vcleq_f32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) m.raw = vcleq_f64(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) m.raw = vcleq_s32(a.raw, b.raw);
  else m.raw = vcleq_u32(a.raw, b.raw);
  return m;
}

// ---- mask ops -----------------------------------------------------------------------
// IfThenElse via vbslq(mask, yes, no): bit-select where mask bits set -> yes.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> IfThenElse(MaskN<T, N> m, VecNeon<T, N> yes, VecNeon<T, N> no) noexcept {
  VecNeon<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = vbslq_f32(m.raw, yes.raw, no.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = vbslq_f64(m.raw, yes.raw, no.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) r.raw = vbslq_s32(m.raw, yes.raw, no.raw);
  else r.raw = vbslq_u32(m.raw, yes.raw, no.raw);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE VecNeon<T, N> IfThenElseZero(MaskN<T, N> m, VecNeon<T, N> yes) noexcept {
  return IfThenElse(m, yes, Zero(Simd<T, N, 0>()));
}

namespace detail {
// Fold a 4-lane (32-bit) mask to one bit per lane (low bit = lane 0). 2-lane (64-bit,
// f64) handled separately. AND with per-lane bit weights then horizontal-add (vaddvq).
HYLE_FORCE_INLINE unsigned mask_bits_4(uint32x4_t m) noexcept {
  const uint32x4_t w = {1u, 2u, 4u, 8u};
  return static_cast<unsigned>(vaddvq_u32(vandq_u32(m, w)));  // aarch64 vaddvq
}
HYLE_FORCE_INLINE unsigned mask_bits_2(uint64x2_t m) noexcept {
  const uint64x2_t w = {1u, 2u};
  return static_cast<unsigned>(vaddvq_u64(vandq_u64(m, w)));
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE unsigned mask_bits(MaskN<T, N> m) noexcept {
  if constexpr (std::is_same_v<T, double>) return mask_bits_2(m.raw);
  else return mask_bits_4(m.raw);
}
}  // namespace detail

template <typename T, std::size_t N>
HYLE_FORCE_INLINE bool AllTrue(Simd<T, N, 0>, MaskN<T, N> m) noexcept {
  return detail::mask_bits(m) == (1u << N) - 1u;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE bool AllFalse(Simd<T, N, 0>, MaskN<T, N> m) noexcept {
  return detail::mask_bits(m) == 0u;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE std::size_t CountTrue(Simd<T, N, 0>, MaskN<T, N> m) noexcept {
  return static_cast<std::size_t>(std::popcount(detail::mask_bits(m)));
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE std::ptrdiff_t FindFirstTrue(Simd<T, N, 0>, MaskN<T, N> m) noexcept {
  const unsigned b = detail::mask_bits(m);
  return b ? static_cast<std::ptrdiff_t>(std::countr_zero(b)) : -1;
}

// ---- reductions (aarch64 horizontal vaddvq/vminvq/vmaxvq) ----------------------------
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T GetLane(VecNeon<T, N> v) noexcept {
  if constexpr (std::is_same_v<T, float>) return vgetq_lane_f32(v.raw, 0);
  else if constexpr (std::is_same_v<T, double>) return vgetq_lane_f64(v.raw, 0);
  else if constexpr (std::is_same_v<T, std::int32_t>) return vgetq_lane_s32(v.raw, 0);
  else return vgetq_lane_u32(v.raw, 0);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceSum(Simd<T, N, 0>, VecNeon<T, N> v) noexcept {
  if constexpr (std::is_same_v<T, float>) return vaddvq_f32(v.raw);
  else if constexpr (std::is_same_v<T, double>) return vaddvq_f64(v.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) return vaddvq_s32(v.raw);
  else return vaddvq_u32(v.raw);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceMin(Simd<T, N, 0>, VecNeon<T, N> v) noexcept {
  if constexpr (std::is_same_v<T, float>) return vminvq_f32(v.raw);
  else if constexpr (std::is_same_v<T, double>) return vgetq_lane_f64(vpminq_f64(v.raw, v.raw), 0);
  else if constexpr (std::is_same_v<T, std::int32_t>) return vminvq_s32(v.raw);
  else return vminvq_u32(v.raw);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceMax(Simd<T, N, 0>, VecNeon<T, N> v) noexcept {
  if constexpr (std::is_same_v<T, float>) return vmaxvq_f32(v.raw);
  else if constexpr (std::is_same_v<T, double>) return vgetq_lane_f64(vpmaxq_f64(v.raw, v.raw), 0);
  else if constexpr (std::is_same_v<T, std::int32_t>) return vmaxvq_s32(v.raw);
  else return vmaxvq_u32(v.raw);
}

}  // namespace hyle::simd
