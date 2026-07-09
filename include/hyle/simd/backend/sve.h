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
#include <limits>
#include <type_traits>

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

// Splat an unsigned shift count into the same-width unsigned vector SVE shifts require.
template <typename T> HYLE_FORCE_INLINE auto dup_ucount(unsigned k) noexcept {
  if constexpr (sizeof(T) == 1) return svdup_n_u8(static_cast<std::uint8_t>(k));
  else if constexpr (sizeof(T) == 2) return svdup_n_u16(static_cast<std::uint16_t>(k));
  else if constexpr (sizeof(T) == 4) return svdup_n_u32(static_cast<std::uint32_t>(k));
  else return svdup_n_u64(static_cast<std::uint64_t>(k));
}
// Reinterpret a signed/unsigned vector to the same-width UNSIGNED vector (SVE shift amount type).
HYLE_FORCE_INLINE svuint8_t to_ucount(svint8_t v) noexcept { return svreinterpret_u8(v); }
HYLE_FORCE_INLINE svuint8_t to_ucount(svuint8_t v) noexcept { return v; }
HYLE_FORCE_INLINE svuint16_t to_ucount(svint16_t v) noexcept { return svreinterpret_u16(v); }
HYLE_FORCE_INLINE svuint16_t to_ucount(svuint16_t v) noexcept { return v; }
HYLE_FORCE_INLINE svuint32_t to_ucount(svint32_t v) noexcept { return svreinterpret_u32(v); }
HYLE_FORCE_INLINE svuint32_t to_ucount(svuint32_t v) noexcept { return v; }
HYLE_FORCE_INLINE svuint64_t to_ucount(svint64_t v) noexcept { return svreinterpret_u64(v); }
HYLE_FORCE_INLINE svuint64_t to_ucount(svuint64_t v) noexcept { return v; }

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
// FindFirstTrue: index of the first active lane, or -1 if none. svbrkb_z sets the lanes BEFORE the
// first true (exclusive); counting them gives the first-true index. FindLastTrue: reverse the mask
// (svrev_b*), find first, map back to (Lanes-1 - idx).
namespace detail {
template <typename T> HYLE_FORCE_INLINE svbool_t brkb_before_first(svbool_t m) noexcept {
  return svbrkb_z(ptrue<T>(), m);
}
template <typename T> HYLE_FORCE_INLINE svbool_t rev_mask(svbool_t m) noexcept {
  if constexpr (sizeof(T) == 1) return svrev_b8(m);
  else if constexpr (sizeof(T) == 2) return svrev_b16(m);
  else if constexpr (sizeof(T) == 4) return svrev_b32(m);
  else return svrev_b64(m);
}
}  // namespace detail
template <class D> HYLE_FORCE_INLINE std::ptrdiff_t FindFirstTrue(D d, svbool_t m) noexcept {
  using T = TFromD<D>;
  if (!svptest_any(detail::ptrue<T>(), m)) return -1;
  return static_cast<std::ptrdiff_t>(CountTrue(d, detail::brkb_before_first<T>(m)));
}
template <class D> HYLE_FORCE_INLINE std::ptrdiff_t FindLastTrue(D d, svbool_t m) noexcept {
  using T = TFromD<D>;
  if (!svptest_any(detail::ptrue<T>(), m)) return -1;
  const std::ptrdiff_t rev_first =
      static_cast<std::ptrdiff_t>(CountTrue(d, detail::brkb_before_first<T>(detail::rev_mask<T>(m))));
  return static_cast<std::ptrdiff_t>(Lanes(d)) - 1 - rev_first;
}


// ---- GetLane: lane 0 ----------------------------------------------------------------
template <class V> HYLE_FORCE_INLINE TFromV<V> GetLane(V v) noexcept {
  return svlasta(svpfalse_b(), v);  // lasta with empty pred = element 0
}

// ---- value reductions (whole-vector -> scalar) --------------------------------------
// svaddv/sv{min,max}v/sv{and,orr,eor}v reduce across all active lanes to a scalar. Under the
// lane-width all-true predicate this is the full-vector reduction. Signature matches Highway:
// Reduce*(D, V) -> T.
template <class D> HYLE_FORCE_INLINE TFromD<D> ReduceSum(D, VFromD<D> v) noexcept {
  return svaddv(detail::ptrue<TFromD<D>>(), v);
}
template <class D> HYLE_FORCE_INLINE TFromD<D> ReduceMin(D, VFromD<D> v) noexcept {
  return svminv(detail::ptrue<TFromD<D>>(), v);
}
template <class D> HYLE_FORCE_INLINE TFromD<D> ReduceMax(D, VFromD<D> v) noexcept {
  return svmaxv(detail::ptrue<TFromD<D>>(), v);
}
template <class D> HYLE_FORCE_INLINE TFromD<D> ReduceAnd(D, VFromD<D> v) noexcept {
  return svandv(detail::ptrue<TFromD<D>>(), v);
}
template <class D> HYLE_FORCE_INLINE TFromD<D> ReduceOr(D, VFromD<D> v) noexcept {
  return svorv(detail::ptrue<TFromD<D>>(), v);
}

// ---- shifts -------------------------------------------------------------------------
// ShiftLeft<k>/ShiftRight<k>: compile-time immediate. SVE svlsl/svlsr/svasr take a vector or
// scalar shift amount; for the immediate form we pass a splatted count. Signed ShiftRight is
// arithmetic (svasr), unsigned is logical (svlsr).
// SVE shift-by-immediate takes an unsigned splat of the count. svlsl (logical left), svlsr
// (logical right, unsigned), svasr (arithmetic right, signed).
template <int kBits, class V> HYLE_FORCE_INLINE V ShiftLeft(V a) noexcept {
  using T = TFromV<V>;
  return svlsl_x(detail::ptrue<T>(), a, detail::dup_ucount<T>(kBits));
}
template <int kBits, class V> HYLE_FORCE_INLINE V ShiftRight(V a) noexcept {
  using T = TFromV<V>;
  if constexpr (std::is_signed_v<T>) return svasr_x(detail::ptrue<T>(), a, detail::dup_ucount<T>(kBits));
  else return svlsr_x(detail::ptrue<T>(), a, detail::dup_ucount<T>(kBits));
}
// Shl/Shr: per-lane variable shift by b's lanes (as unsigned same-width). Highway: Shr signed = asr.
template <class V> HYLE_FORCE_INLINE V Shl(V a, V b) noexcept {
  using T = TFromV<V>;
  return svlsl_x(detail::ptrue<T>(), a, detail::to_ucount(b));
}
template <class V> HYLE_FORCE_INLINE V Shr(V a, V b) noexcept {
  using T = TFromV<V>;
  if constexpr (std::is_signed_v<T>) return svasr_x(detail::ptrue<T>(), a, detail::to_ucount(b));
  else return svlsr_x(detail::ptrue<T>(), a, detail::to_ucount(b));
}

// ---- Reverse (whole-vector element reversal) ----------------------------------------
template <class D> HYLE_FORCE_INLINE VFromD<D> Reverse(D, VFromD<D> v) noexcept {
  return svrev(v);
}

// ---- tail handling: FirstN / LoadN / StoreN (SVE NATIVE, not store+reload) ----------
// FirstN(d, n): predicate with the first n lanes active. svwhilelt_b*(0, n) is exactly this and
// is SVE's core tail primitive (no emulation). Returns svbool_t (our Mask type).
template <class D> HYLE_FORCE_INLINE svbool_t FirstN(D, std::size_t n) noexcept {
  using T = TFromD<D>;
  const std::uint64_t nn = static_cast<std::uint64_t>(n);
  if constexpr (sizeof(T) == 1) return svwhilelt_b8(std::uint64_t{0}, nn);
  else if constexpr (sizeof(T) == 2) return svwhilelt_b16(std::uint64_t{0}, nn);
  else if constexpr (sizeof(T) == 4) return svwhilelt_b32(std::uint64_t{0}, nn);
  else return svwhilelt_b64(std::uint64_t{0}, nn);
}
// LoadN: load exactly n lanes (rest zero) -- predicated svld1, fault-safe (only touches n lanes).
template <class D> HYLE_FORCE_INLINE VFromD<D> LoadN(D d, const TFromD<D>* p, std::size_t n) noexcept {
  return svld1(FirstN(d, n), p);
}
// StoreN: store exactly n lanes -- predicated svst1 (only writes n lanes).
template <class D> HYLE_FORCE_INLINE void StoreN(VFromD<D> v, D d, TFromD<D>* p, std::size_t n) noexcept {
  svst1(FirstN(d, n), p, v);
}

// ---- compress (SVE NATIVE svcompact, not store+reload) ------------------------------
// Compress(v, m): pack the active lanes to the low end. SVE svcompact is exactly this for 32/64-bit
// lanes (the widths that have it). Highway contract: inactive lanes' contents are unspecified.
template <class V> HYLE_FORCE_INLINE V Compress(V v, svbool_t m) noexcept {
  return svcompact(m, v);
}

// ---- same-width conversions (f32<->i32, the (de)quantize core) ----------------------
// ConvertTo: static_cast-per-lane into the dst lane type (truncating for f->i). Same lane COUNT
// (dst tag has the same N). SVE svcvt does exactly this at matching width.
template <typename DstT, std::size_t N, class V>
HYLE_FORCE_INLINE VFromD<Simd<DstT, N, 0>> ConvertTo(Simd<DstT, N, 0>, V v) noexcept {
  using SrcT = TFromV<V>;
  const svbool_t pg = detail::ptrue<SrcT>();
  if constexpr (std::is_same_v<SrcT, float> && std::is_same_v<DstT, std::int32_t>)
    return svcvt_s32_x(pg, v);  // truncate toward zero
  else if constexpr (std::is_same_v<SrcT, float> && std::is_same_v<DstT, std::uint32_t>)
    return svcvt_u32_x(pg, v);
  else if constexpr (std::is_same_v<SrcT, std::int32_t> && std::is_same_v<DstT, float>)
    return svcvt_f32_x(pg, v);
  else if constexpr (std::is_same_v<SrcT, std::uint32_t> && std::is_same_v<DstT, float>)
    return svcvt_f32_x(pg, v);
  else if constexpr (std::is_same_v<SrcT, double> && std::is_same_v<DstT, std::int64_t>)
    return svcvt_s64_x(pg, v);
  else if constexpr (std::is_same_v<SrcT, std::int64_t> && std::is_same_v<DstT, double>)
    return svcvt_f64_x(pg, v);
  else
    return svcvt_f64_x(pg, v);  // remaining f64<->u64 etc. (kept minimal for the quantize core)
}
// RoundToInt: round-to-nearest-even THEN convert (svrintn rounds, svcvt truncates the now-integral
// value). float->int only.
template <typename DstT, std::size_t N, class V>
HYLE_FORCE_INLINE VFromD<Simd<DstT, N, 0>> RoundToInt(Simd<DstT, N, 0>, V v) noexcept {
  using SrcT = TFromV<V>;
  const svbool_t pg = detail::ptrue<SrcT>();
  if constexpr (std::is_same_v<SrcT, float>)
    return svcvt_s32_x(pg, svrintn_x(pg, v));
  else
    return svcvt_s64_x(pg, svrintn_x(pg, v));
}
// Round/Floor/Ceil/Trunc: float->float rounding to an integral value (result stays the float type).
// SVE native: svrintn (nearest-even) / svrintm (toward -inf) / svrintp (toward +inf) / svrintz
// (toward zero). Float/double only.
template <class V> HYLE_FORCE_INLINE V Round(V v) noexcept { return svrintn_x(detail::ptrue<TFromV<V>>(), v); }
template <class V> HYLE_FORCE_INLINE V Floor(V v) noexcept { return svrintm_x(detail::ptrue<TFromV<V>>(), v); }
template <class V> HYLE_FORCE_INLINE V Ceil(V v) noexcept { return svrintp_x(detail::ptrue<TFromV<V>>(), v); }
template <class V> HYLE_FORCE_INLINE V Trunc(V v) noexcept { return svrintz_x(detail::ptrue<TFromV<V>>(), v); }

// ---- gather (SVE NATIVE svld1_gather, not store+reload) -----------------------------
// GatherIndex(d, base, idx): dst[i] = base[idx[i]]. SVE's gather takes the index vector as the
// same-width unsigned/signed lanes; svld1_gather_index scales by element size (a true index, not
// byte offset). 32/64-bit lanes only (SVE gather is defined there).
template <class D, class VI>
HYLE_FORCE_INLINE VFromD<D> GatherIndex(D, const TFromD<D>* base, VI idx) noexcept {
  return svld1_gather_index(detail::ptrue<TFromD<D>>(), base, idx);
}

// ---- swizzle (WHOLE-VECTOR scalable-native; SEE CONTRACT NOTE) ----------------------
// SCALABLE CONTRACT: on a scalable backend these are WHOLE-VECTOR, not per-128-bit-block like the
// fixed-width x86 backends. SVE's svzip1/svuzp1/svtrn1 operate across the entire VL. This matches
// the mathematical (scalar-oracle) definition and NEON's single-128-bit-block behavior; it differs
// from AVX2's 256-bit "two blocks" only for the block-relative ops, which are NOT provided here
// (ConcatLowerLower etc. have no scalable meaning). InterleaveLower(a,b) = [a0,b0,a1,b1,...].
template <class V> HYLE_FORCE_INLINE V InterleaveLower(V a, V b) noexcept { return svzip1(a, b); }
template <class V> HYLE_FORCE_INLINE V InterleaveUpper(V a, V b) noexcept { return svzip2(a, b); }
template <class D, class V> HYLE_FORCE_INLINE V InterleaveLower(D, V a, V b) noexcept { return svzip1(a, b); }
template <class D, class V> HYLE_FORCE_INLINE V InterleaveUpper(D, V a, V b) noexcept { return svzip2(a, b); }
template <class V> HYLE_FORCE_INLINE V ZipLower(V a, V b) noexcept { return svzip1(a, b); }
template <class V> HYLE_FORCE_INLINE V ZipUpper(V a, V b) noexcept { return svzip2(a, b); }
// ConcatEven/Odd: gather the even/odd lanes of the concatenation [b:a] (Highway's ConcatEven(d,hi,
// lo) = even lanes of lo then even lanes of hi). svuzp1/2 deinterleave across the whole vector.
template <class D, class V> HYLE_FORCE_INLINE V ConcatEven(D, V hi, V lo) noexcept { return svuzp1(lo, hi); }
template <class D, class V> HYLE_FORCE_INLINE V ConcatOdd(D, V hi, V lo) noexcept { return svuzp2(lo, hi); }
// OddEven(odd, even): even-position lanes from `even`, odd-position lanes from `odd`. Select on an
// alternating "(lane index & 1)==0" predicate built from the width-matched iota.
namespace detail {
template <typename T> HYLE_FORCE_INLINE svbool_t even_lane_mask() noexcept {
  const svbool_t pg = ptrue<T>();
  if constexpr (sizeof(T) == 1) return svcmpeq(pg, svand_x(pg, svindex_u8(0, 1), svdup_n_u8(1)), svdup_n_u8(0));
  else if constexpr (sizeof(T) == 2) return svcmpeq(pg, svand_x(pg, svindex_u16(0, 1), svdup_n_u16(1)), svdup_n_u16(0));
  else if constexpr (sizeof(T) == 4) return svcmpeq(pg, svand_x(pg, svindex_u32(0, 1), svdup_n_u32(1)), svdup_n_u32(0));
  else return svcmpeq(pg, svand_x(pg, svindex_u64(0, 1), svdup_n_u64(1)), svdup_n_u64(0));
}
}  // namespace detail
template <class V> HYLE_FORCE_INLINE V OddEven(V odd, V even) noexcept {
  return svsel(detail::even_lane_mask<TFromV<V>>(), even, odd);
}

// ---- bit counting: population / leading-zero count ----------------------------------
// svcnt_x/svclz_x return the UNSIGNED same-width vector; Highway keeps the input lane type, so
// reinterpret the unsigned result back to V's type via a shared helper.
namespace detail {
template <class V, class U> HYLE_FORCE_INLINE V as_lane(U u) noexcept {
  using T = TFromV<V>;
  if constexpr (!std::is_signed_v<T>) return u;
  else if constexpr (sizeof(T) == 1) return svreinterpret_s8(u);
  else if constexpr (sizeof(T) == 2) return svreinterpret_s16(u);
  else if constexpr (sizeof(T) == 4) return svreinterpret_s32(u);
  else return svreinterpret_s64(u);
}
}  // namespace detail
template <class V> HYLE_FORCE_INLINE V PopulationCount(V v) noexcept {
  return detail::as_lane<V>(svcnt_x(detail::ptrue<TFromV<V>>(), v));
}
template <class V> HYLE_FORCE_INLINE V LeadingZeroCount(V v) noexcept {
  return detail::as_lane<V>(svclz_x(detail::ptrue<TFromV<V>>(), v));
}

// ---- BitCast: reinterpret the bits to another lane type (same total width) ----------
template <class D, class V>
HYLE_FORCE_INLINE VFromD<D> BitCast(D, V v) noexcept {
  using T = TFromD<D>;
  if constexpr (std::is_same_v<T, float>) return svreinterpret_f32(v);
  else if constexpr (std::is_same_v<T, double>) return svreinterpret_f64(v);
  else if constexpr (std::is_same_v<T, std::int8_t>) return svreinterpret_s8(v);
  else if constexpr (std::is_same_v<T, std::uint8_t>) return svreinterpret_u8(v);
  else if constexpr (std::is_same_v<T, std::int16_t>) return svreinterpret_s16(v);
  else if constexpr (std::is_same_v<T, std::uint16_t>) return svreinterpret_u16(v);
  else if constexpr (std::is_same_v<T, std::int32_t>) return svreinterpret_s32(v);
  else if constexpr (std::is_same_v<T, std::uint32_t>) return svreinterpret_u32(v);
  else if constexpr (std::is_same_v<T, std::int64_t>) return svreinterpret_s64(v);
  else return svreinterpret_u64(v);
}

// ---- saturating arithmetic (integer) ------------------------------------------------
template <class V> HYLE_FORCE_INLINE V SaturatedAdd(V a, V b) noexcept { return svqadd(a, b); }
template <class V> HYLE_FORCE_INLINE V SaturatedSub(V a, V b) noexcept { return svqsub(a, b); }

// ---- misc lane math -----------------------------------------------------------------
template <class V> HYLE_FORCE_INLINE V AbsDiff(V a, V b) noexcept {
  return svabd_x(detail::ptrue<TFromV<V>>(), a, b);
}
template <class V> HYLE_FORCE_INLINE V MulHigh(V a, V b) noexcept {  // high half of the widening mul
  return svmulh_x(detail::ptrue<TFromV<V>>(), a, b);
}
// Broadcast<i>(v): splat lane i across all lanes. svdup_lane broadcasts a single lane.
template <int kLane, class V> HYLE_FORCE_INLINE V Broadcast(V v) noexcept {
  return svdup_lane(v, static_cast<std::uint64_t>(kLane));
}
template <class V> HYLE_FORCE_INLINE V Clamp(V v, V lo, V hi) noexcept {
  return Min(Max(v, lo), hi);
}
// CopySign(mag, sign): |mag| with sign's sign bit. Float only.
template <class V> HYLE_FORCE_INLINE V CopySign(V mag, V sign) noexcept {
  using T = TFromV<V>;
  const auto d = DFromV<V>();
  // clear mag's sign, OR in sign's sign bit (via integer view)
  if constexpr (std::is_same_v<T, float>) {
    const svuint32_t sbit = svdup_n_u32(0x80000000u);
    const svuint32_t mi = svbic_x(detail::ptrue<T>(), svreinterpret_u32(mag), sbit);
    const svuint32_t si = svand_x(detail::ptrue<T>(), svreinterpret_u32(sign), sbit);
    return svreinterpret_f32(svorr_x(detail::ptrue<T>(), mi, si));
  } else {
    const svuint64_t sbit = svdup_n_u64(0x8000000000000000ull);
    const svuint64_t mi = svbic_x(detail::ptrue<T>(), svreinterpret_u64(mag), sbit);
    const svuint64_t si = svand_x(detail::ptrue<T>(), svreinterpret_u64(sign), sbit);
    return svreinterpret_f64(svorr_x(detail::ptrue<T>(), mi, si));
  }
  (void)d;
}

// ---- FP classification (svcmpuo = unordered => NaN) ---------------------------------
template <class V> HYLE_FORCE_INLINE svbool_t IsNaN(V v) noexcept {
  return svcmpuo(detail::ptrue<TFromV<V>>(), v, v);
}
template <class V> HYLE_FORCE_INLINE svbool_t IsInf(V v) noexcept {
  using T = TFromV<V>;
  const V av = Abs(v);
  return Eq(av, Set(DFromV<V>(), std::numeric_limits<T>::infinity()));
}
template <class V> HYLE_FORCE_INLINE svbool_t IsFinite(V v) noexcept {
  using T = TFromV<V>;
  const V av = Abs(v);
  return Lt(av, Set(DFromV<V>(), std::numeric_limits<T>::infinity()));  // NaN also fails (unordered)
}

// ---- sign-predicated select ---------------------------------------------------------
template <class V> HYLE_FORCE_INLINE V IfNegativeThenElse(V v, V yes, V no) noexcept {
  return svsel(MaskFromVec(v), yes, no);
}
template <class V> HYLE_FORCE_INLINE V ZeroIfNegative(V v) noexcept {
  return svsel(MaskFromVec(v), Zero(DFromV<V>()), v);
}

// ---- IEEE Max/MinNumber (NaN-quieting: return the non-NaN operand) ------------------
template <class V> HYLE_FORCE_INLINE V MaxNumber(V a, V b) noexcept {
  return svmaxnm_x(detail::ptrue<TFromV<V>>(), a, b);
}
template <class V> HYLE_FORCE_INLINE V MinNumber(V a, V b) noexcept {
  return svminnm_x(detail::ptrue<TFromV<V>>(), a, b);
}

// ---- masked load / blended store (SVE native predicated, fault-safe) ----------------
template <class D> HYLE_FORCE_INLINE VFromD<D> MaskedLoad(svbool_t m, D, const TFromD<D>* p) noexcept {
  return svld1(m, p);  // inactive lanes = 0, only active lanes touch memory (fault-safe)
}
template <class D>
HYLE_FORCE_INLINE VFromD<D> MaskedLoadOr(VFromD<D> no, svbool_t m, D, const TFromD<D>* p) noexcept {
  return svsel(m, svld1(m, p), no);
}
template <class D> HYLE_FORCE_INLINE void BlendedStore(VFromD<D> v, svbool_t m, D, TFromD<D>* p) noexcept {
  svst1(m, p, v);  // only active lanes written (fault-safe)
}

// ---- scatter (SVE native svst1_scatter) + CompressStore -----------------------------
template <class D, class VI>
HYLE_FORCE_INLINE void ScatterIndex(VFromD<D> v, D, TFromD<D>* base, VI idx) noexcept {
  svst1_scatter_index(detail::ptrue<TFromD<D>>(), base, idx, v);
}
// CompressStore(v, m, d, p): pack active lanes to the front, store CountTrue(m) of them, return
// that count. svcompact packs; then a whilelt-predicated store writes exactly the active count.
template <class D>
HYLE_FORCE_INLINE std::size_t CompressStore(VFromD<D> v, svbool_t m, D d, TFromD<D>* p) noexcept {
  const std::size_t cnt = CountTrue(d, m);
  svst1(FirstN(d, cnt), p, svcompact(m, v));
  return cnt;
}

}  // namespace hyle::simd
