#pragma once

// backend/sve.h -- ARM SVE (Scalable Vector Extension) backend. The FIRST scalable-width backend
// (SSE4/AVX2/AVX512/NEON/scalar are all fixed-width). SVE's vector length is a RUNTIME property
// (128..2048 bit, implementation-defined): the true lane count is svcntw()/svcntb() at run time,
// NOT a compile-time constant. This forces two departures from the fixed-width backends, both
// SELF-CONTAINED here (they do not touch vec.h's shared layer or any other backend -- see
// project_simd_sve_rvv feasibility note):
//
//   1. Vec is NOT a `struct { raw; PrivateT; kPrivateN; }`. SVE's svfloat32_t etc. are SIZELESS
//      types (ACLE): they cannot be struct/array members, cannot be sizeof'd, cannot be globals.
//      They CAN be function parameters, return values, and locals. So `VFromD<D>` maps the lane
//      type to the native sizeless type via a traits table (detail::SveVecT<T>), and `DFromV<V>`
//      reverses it via a specialization table (detail::DFromV_sve<V>) -- NOT via a `V::PrivateT`
//      member (there is no member). Every op takes/returns the sizeless type by value, which the
//      ABI allows. This is exactly Highway's arm_sve-inl.h approach.
//
//   2. Lanes(d) is RUNTIME (svcntw() etc.), gated by vec.h's HYLE_SIMD_HAVE_SCALABLE=1 (which
//      the SVE target flips). kMaxLanes (from the tag) is only the UPPER BOUND for stack-buffer
//      sizing; loops iterate to the runtime Lanes(d).
//
// Predicate model: SVE masks are svbool_t (also sizeless). Most arithmetic uses the "_z" (zeroing)
// or "_x" (don't-care) merge forms under an all-true predicate svptrue_b*(); the mask ops build
// real svbool_t predicates. All-true is the full-vector predicate for the current VL.

#include <arm_sve.h>

#include <cstddef>
#include <cstdint>

#include "hyle/macros.h"

namespace hyle::simd {

namespace detail {

// ---- lane type -> native sizeless vector type (replaces the struct-member `raw`) ----
template <typename T> struct SveVecT;
template <> struct SveVecT<float> { using type = svfloat32_t; };
template <> struct SveVecT<double> { using type = svfloat64_t; };
template <> struct SveVecT<std::int8_t> { using type = svint8_t; };
template <> struct SveVecT<std::uint8_t> { using type = svuint8_t; };
template <> struct SveVecT<std::int16_t> { using type = svint16_t; };
template <> struct SveVecT<std::uint16_t> { using type = svuint16_t; };
template <> struct SveVecT<std::int32_t> { using type = svint32_t; };
template <> struct SveVecT<std::uint32_t> { using type = svuint32_t; };
template <> struct SveVecT<std::int64_t> { using type = svint64_t; };
template <> struct SveVecT<std::uint64_t> { using type = svuint64_t; };

// ---- native sizeless vector type -> tag (reverse map for DFromV; replaces `V::PrivateT`) ----
// Keyed on the sizeless vector type. The tag's N is the UPPER-BOUND lane count for T (the same
// kMaxLanes ScalableTag<T> carries): kNativeVectorBytes / sizeof(T).
template <typename V> struct DFromV_sve;
template <> struct DFromV_sve<svfloat32_t> { using T = float; };
template <> struct DFromV_sve<svfloat64_t> { using T = double; };
template <> struct DFromV_sve<svint8_t> { using T = std::int8_t; };
template <> struct DFromV_sve<svuint8_t> { using T = std::uint8_t; };
template <> struct DFromV_sve<svint16_t> { using T = std::int16_t; };
template <> struct DFromV_sve<svuint16_t> { using T = std::uint16_t; };
template <> struct DFromV_sve<svint32_t> { using T = std::int32_t; };
template <> struct DFromV_sve<svuint32_t> { using T = std::uint32_t; };
template <> struct DFromV_sve<svint64_t> { using T = std::int64_t; };
template <> struct DFromV_sve<svuint64_t> { using T = std::uint64_t; };

// All-true predicate for a given lane WIDTH. SVE predicates are per-byte; the b8/b16/b32/b64
// variants set every element of that width. Parameterized by sizeof(T).
template <typename T> HYLE_FORCE_INLINE svbool_t ptrue() noexcept {
  if constexpr (sizeof(T) == 1) return svptrue_b8();
  else if constexpr (sizeof(T) == 2) return svptrue_b16();
  else if constexpr (sizeof(T) == 4) return svptrue_b32();
  else return svptrue_b64();
}

}  // namespace detail

// ---- VFromD / DFromV / TFromV -------------------------------------------------------
// VFromD: the sizeless vector type for tag D. (Fixed-width backends define this as
// decltype(Zero(D())); here we map directly -- Zero still works, but the traits map avoids
// depending on Zero's definition order and is clearer for the sizeless case.)
template <class D>
using VFromD = typename detail::SveVecT<TFromD<D>>::type;

// DFromV: recover the tag from a vector type. Uses the reverse traits table (no V::PrivateT).
// The recovered tag is ScalableTag<T> == Simd<T, kNativeVectorBytes/sizeof(T)>.
template <class V>
using DFromV = Simd<typename detail::DFromV_sve<V>::T,
                    kNativeVectorBytes / sizeof(typename detail::DFromV_sve<V>::T), 0>;
template <class V>
using TFromV = typename detail::DFromV_sve<V>::T;

// ---- Lanes(d): RUNTIME vector length (the scalable hook; HAVE_SCALABLE=1 disables the
// constexpr generic Lanes in vec.h so this one is THE Lanes). svcnt{b,h,w,d} return the
// element count for the current VL at that width. ----
template <class D>
HYLE_FORCE_INLINE std::size_t Lanes(D) noexcept {
  using T = TFromD<D>;
  if constexpr (sizeof(T) == 1) return svcntb();
  else if constexpr (sizeof(T) == 2) return svcnth();
  else if constexpr (sizeof(T) == 4) return svcntw();
  else return svcntd();
}

// ---- initialization -----------------------------------------------------------------
template <class D, typename T2>
HYLE_FORCE_INLINE VFromD<D> Set(D, T2 t) noexcept {
  using T = TFromD<D>;
  if constexpr (std::is_same_v<T, float>) return svdup_n_f32(static_cast<float>(t));
  else if constexpr (std::is_same_v<T, double>) return svdup_n_f64(static_cast<double>(t));
  else if constexpr (std::is_same_v<T, std::int8_t>) return svdup_n_s8(static_cast<std::int8_t>(t));
  else if constexpr (std::is_same_v<T, std::uint8_t>) return svdup_n_u8(static_cast<std::uint8_t>(t));
  else if constexpr (std::is_same_v<T, std::int16_t>) return svdup_n_s16(static_cast<std::int16_t>(t));
  else if constexpr (std::is_same_v<T, std::uint16_t>) return svdup_n_u16(static_cast<std::uint16_t>(t));
  else if constexpr (std::is_same_v<T, std::int32_t>) return svdup_n_s32(static_cast<std::int32_t>(t));
  else if constexpr (std::is_same_v<T, std::uint32_t>) return svdup_n_u32(static_cast<std::uint32_t>(t));
  else if constexpr (std::is_same_v<T, std::int64_t>) return svdup_n_s64(static_cast<std::int64_t>(t));
  else return svdup_n_u64(static_cast<std::uint64_t>(t));
}

template <class D>
HYLE_FORCE_INLINE VFromD<D> Zero(D d) noexcept {
  return Set(d, static_cast<TFromD<D>>(0));
}
// NOTE: Undefined / Iota / SignBit etc. are defined GENERICALLY in vec.h's shared layer (built on
// Zero/Set/iota primitives) -- the backend must NOT redefine them or it's an ambiguous redefinition.

// ---- load / store -------------------------------------------------------------------
// svld1/svst1 under an all-true predicate move the full current-VL vector. SVE has no separate
// aligned/unaligned form (the instruction handles any alignment), so Load==LoadU here.
template <class D>
HYLE_FORCE_INLINE VFromD<D> LoadU(D, const TFromD<D>* p) noexcept {
  return svld1(detail::ptrue<TFromD<D>>(), p);
}
template <class D>
HYLE_FORCE_INLINE VFromD<D> Load(D d, const TFromD<D>* p) noexcept {
  return LoadU(d, p);
}
template <class D>
HYLE_FORCE_INLINE void StoreU(VFromD<D> v, D, TFromD<D>* p) noexcept {
  svst1(detail::ptrue<TFromD<D>>(), p, v);
}
template <class D>
HYLE_FORCE_INLINE void Store(VFromD<D> v, D d, TFromD<D>* p) noexcept {
  StoreU(v, d, p);
}

// ---- arithmetic ---------------------------------------------------------------------
// SVE arithmetic uses merge forms; under an all-true predicate the _x (don't-care) form is the
// cheapest and semantically an unconditional op on every lane. TFromV<V> recovers the lane type
// so ptrue picks the right element width.
template <class V> HYLE_FORCE_INLINE V Add(V a, V b) noexcept {
  return svadd_x(detail::ptrue<TFromV<V>>(), a, b);
}
template <class V> HYLE_FORCE_INLINE V Sub(V a, V b) noexcept {
  return svsub_x(detail::ptrue<TFromV<V>>(), a, b);
}
template <class V> HYLE_FORCE_INLINE V Mul(V a, V b) noexcept {
  return svmul_x(detail::ptrue<TFromV<V>>(), a, b);
}
template <class V> HYLE_FORCE_INLINE V Neg(V a) noexcept {
  return svneg_x(detail::ptrue<TFromV<V>>(), a);
}
template <class V> HYLE_FORCE_INLINE V Abs(V a) noexcept {
  return svabs_x(detail::ptrue<TFromV<V>>(), a);
}
template <class V> HYLE_FORCE_INLINE V Min(V a, V b) noexcept {
  return svmin_x(detail::ptrue<TFromV<V>>(), a, b);
}
template <class V> HYLE_FORCE_INLINE V Max(V a, V b) noexcept {
  return svmax_x(detail::ptrue<TFromV<V>>(), a, b);
}
// Floating-point only: Div / Sqrt / FMA.
template <class V> HYLE_FORCE_INLINE V Div(V a, V b) noexcept {
  return svdiv_x(detail::ptrue<TFromV<V>>(), a, b);
}
template <class V> HYLE_FORCE_INLINE V Sqrt(V a) noexcept {
  return svsqrt_x(detail::ptrue<TFromV<V>>(), a);
}
// MulAdd(a,b,c) = a*b + c; NegMulAdd(a,b,c) = c - a*b. SVE svmad_x(p, a, b, c) = a*b + c.
template <class V> HYLE_FORCE_INLINE V MulAdd(V a, V b, V c) noexcept {
  return svmad_x(detail::ptrue<TFromV<V>>(), a, b, c);
}
template <class V> HYLE_FORCE_INLINE V NegMulAdd(V a, V b, V c) noexcept {
  return svmsb_x(detail::ptrue<TFromV<V>>(), a, b, c);  // c - a*b
}

// ---- bitwise (integer lanes) --------------------------------------------------------
template <class V> HYLE_FORCE_INLINE V And(V a, V b) noexcept {
  return svand_x(detail::ptrue<TFromV<V>>(), a, b);
}
template <class V> HYLE_FORCE_INLINE V Or(V a, V b) noexcept {
  return svorr_x(detail::ptrue<TFromV<V>>(), a, b);
}
template <class V> HYLE_FORCE_INLINE V Xor(V a, V b) noexcept {
  return sveor_x(detail::ptrue<TFromV<V>>(), a, b);
}
template <class V> HYLE_FORCE_INLINE V AndNot(V a, V b) noexcept {  // ~a & b (Highway order)
  return svbic_x(detail::ptrue<TFromV<V>>(), b, a);                 // svbic(p, x, y) = x & ~y
}
template <class V> HYLE_FORCE_INLINE V Not(V a) noexcept {
  return svnot_x(detail::ptrue<TFromV<V>>(), a);
}

// ---- comparisons -> svbool_t predicate mask -----------------------------------------
// Mask is svbool_t for every lane type (sizeless). The compare intrinsics take the governing
// predicate + two vectors and yield a predicate with the active lanes set where the relation holds.
template <class V> HYLE_FORCE_INLINE svbool_t Eq(V a, V b) noexcept {
  return svcmpeq(detail::ptrue<TFromV<V>>(), a, b);
}
template <class V> HYLE_FORCE_INLINE svbool_t Ne(V a, V b) noexcept {
  return svcmpne(detail::ptrue<TFromV<V>>(), a, b);
}
template <class V> HYLE_FORCE_INLINE svbool_t Lt(V a, V b) noexcept {
  return svcmplt(detail::ptrue<TFromV<V>>(), a, b);
}
template <class V> HYLE_FORCE_INLINE svbool_t Le(V a, V b) noexcept {
  return svcmple(detail::ptrue<TFromV<V>>(), a, b);
}
template <class V> HYLE_FORCE_INLINE svbool_t Gt(V a, V b) noexcept {
  return svcmpgt(detail::ptrue<TFromV<V>>(), a, b);
}
template <class V> HYLE_FORCE_INLINE svbool_t Ge(V a, V b) noexcept {
  return svcmpge(detail::ptrue<TFromV<V>>(), a, b);
}

// ---- select / mask <-> vec ----------------------------------------------------------
// IfThenElse(m, yes, no): svsel picks yes where m active, else no.
template <class V> HYLE_FORCE_INLINE V IfThenElse(svbool_t m, V yes, V no) noexcept {
  return svsel(m, yes, no);
}
template <class V> HYLE_FORCE_INLINE V IfThenElseZero(svbool_t m, V yes) noexcept {
  using T = TFromV<V>;
  return svsel(m, yes, Set(DFromV<V>(), static_cast<T>(0)));
}
template <class V> HYLE_FORCE_INLINE V IfThenZeroElse(svbool_t m, V no) noexcept {
  using T = TFromV<V>;
  return svsel(m, Set(DFromV<V>(), static_cast<T>(0)), no);
}
// MaskFromVec (Highway contract: lane active where the MSB / sign bit is set). Test < 0 on the
// signed view. For SVE we compare the vector < 0; for unsigned lanes the "MSB set" is >= half,
// but Highway's MaskFromVec is defined on the bit pattern's MSB, which for unsigned == top bit.
// Reinterpret to signed same-width and test < 0.
template <class V> HYLE_FORCE_INLINE svbool_t MaskFromVec(V v) noexcept {
  using T = TFromV<V>;
  return svcmplt(detail::ptrue<T>(), v, Set(DFromV<V>(), static_cast<T>(0)));
}
// VecFromMask(d, m): all-ones (T(-1)) where active, else 0.
template <class D> HYLE_FORCE_INLINE VFromD<D> VecFromMask(D d, svbool_t m) noexcept {
  using T = TFromD<D>;
  return svsel(m, Set(d, static_cast<T>(-1)), Zero(d));
}

// ---- mask reductions ----------------------------------------------------------------
// AllTrue/AllFalse: test the mask against the lane-width all-true predicate. CountTrue: svcntp
// under the lane-width predicate counts ACTIVE ELEMENTS (matching the lane count, not bytes).
template <class D> HYLE_FORCE_INLINE bool AllTrue(D, svbool_t m) noexcept {
  const svbool_t pg = detail::ptrue<TFromD<D>>();
  return !svptest_any(pg, svnot_z(pg, m));  // no lane is false
}
template <class D> HYLE_FORCE_INLINE bool AllFalse(D, svbool_t m) noexcept {
  return !svptest_any(detail::ptrue<TFromD<D>>(), m);
}
template <class D> HYLE_FORCE_INLINE std::size_t CountTrue(D, svbool_t m) noexcept {
  // svcntp needs an explicit element-width suffix (not overloaded). Count active elements under
  // the lane-width all-true predicate so the result is in LANES, not bytes.
  using T = TFromD<D>;
  if constexpr (sizeof(T) == 1) return static_cast<std::size_t>(svcntp_b8(svptrue_b8(), m));
  else if constexpr (sizeof(T) == 2) return static_cast<std::size_t>(svcntp_b16(svptrue_b16(), m));
  else if constexpr (sizeof(T) == 4) return static_cast<std::size_t>(svcntp_b32(svptrue_b32(), m));
  else return static_cast<std::size_t>(svcntp_b64(svptrue_b64(), m));
}

// ---- GetLane: lane 0 ----------------------------------------------------------------
template <class V> HYLE_FORCE_INLINE TFromV<V> GetLane(V v) noexcept {
  return svlasta(svpfalse_b(), v);  // lasta with empty pred = element 0
}

}  // namespace hyle::simd
