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
#include "hyle/simd/backend/crypto_soft.h"  // shared AES/CLMul software fallback (Crypto op face)

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
template <typename T> HYLE_FORCE_INLINE svbool_t even_lane_mask() noexcept;

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

// Forward declarations for generic compositions defined before the swizzle/bitcast sections.
template <class V> HYLE_FORCE_INLINE V OddEven(V odd, V even) noexcept;
template <class D, class V> HYLE_FORCE_INLINE VFromD<D> BitCast(D d, V v) noexcept;
template <class D> HYLE_FORCE_INLINE svbool_t FirstN(D d, std::size_t n) noexcept;

// ---- Lanes(d): RUNTIME vector length (the scalable hook; HAVE_SCALABLE=1 disables the
// constexpr generic Lanes in vec.h so this one is THE Lanes). svcnt{b,h,w,d} return the
// element count for the current VL at that width. ----
template <class D>
HYLE_FORCE_INLINE std::size_t Lanes(D) noexcept {
  using T = TFromD<D>;
  std::size_t full;
  if constexpr (sizeof(T) == 1) full = svcntb();
  else if constexpr (sizeof(T) == 2) full = svcnth();
  else if constexpr (sizeof(T) == 4) full = svcntw();
  else full = svcntd();
  constexpr int pow2 = D{}.Pow2();
  static_assert(pow2 <= 0, "SVE grouped vectors are not represented by one native value");
  const std::size_t scaled = full >> static_cast<unsigned>(-pow2);
  return scaled < D::kMaxLanes ? scaled : D::kMaxLanes;
}

namespace detail {
template <class D> HYLE_FORCE_INLINE svbool_t active(D d) noexcept {
  using T = TFromD<D>;
  const std::uint64_t n = static_cast<std::uint64_t>(Lanes(d));
  if constexpr (sizeof(T) == 1) return svwhilelt_b8(std::uint64_t{0}, n);
  else if constexpr (sizeof(T) == 2) return svwhilelt_b16(std::uint64_t{0}, n);
  else if constexpr (sizeof(T) == 4) return svwhilelt_b32(std::uint64_t{0}, n);
  else return svwhilelt_b64(std::uint64_t{0}, n);
}
}  // namespace detail

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
HYLE_FORCE_INLINE VFromD<D> LoadU(D d, const TFromD<D>* p) noexcept {
  return svld1(detail::active(d), p);
}
template <class D>
HYLE_FORCE_INLINE VFromD<D> Load(D d, const TFromD<D>* p) noexcept {
  return LoadU(d, p);
}
template <class D>
HYLE_FORCE_INLINE void StoreU(VFromD<D> v, D d, TFromD<D>* p) noexcept {
  svst1(detail::active(d), p, v);
}
template <class D>
HYLE_FORCE_INLINE void Store(VFromD<D> v, D d, TFromD<D>* p) noexcept {
  StoreU(v, d, p);
}

// ---- interleaved load / store (native SVE ld2/3/4 + st2/3/4) ------------------------
template <class D>
HYLE_FORCE_INLINE void LoadInterleaved2(D d, const TFromD<D>* p, VFromD<D>& v0,
                                        VFromD<D>& v1) noexcept {
  const auto tuple = svld2(detail::active(d), p);
  v0 = svget2(tuple, 0); v1 = svget2(tuple, 1);
}
template <class D>
HYLE_FORCE_INLINE void LoadInterleaved3(D d, const TFromD<D>* p, VFromD<D>& v0,
                                        VFromD<D>& v1, VFromD<D>& v2) noexcept {
  const auto tuple = svld3(detail::active(d), p);
  v0 = svget3(tuple, 0); v1 = svget3(tuple, 1); v2 = svget3(tuple, 2);
}
template <class D>
HYLE_FORCE_INLINE void LoadInterleaved4(D d, const TFromD<D>* p, VFromD<D>& v0,
                                        VFromD<D>& v1, VFromD<D>& v2,
                                        VFromD<D>& v3) noexcept {
  const auto tuple = svld4(detail::active(d), p);
  v0 = svget4(tuple, 0); v1 = svget4(tuple, 1); v2 = svget4(tuple, 2); v3 = svget4(tuple, 3);
}
template <class D>
HYLE_FORCE_INLINE void StoreInterleaved2(VFromD<D> v0, VFromD<D> v1, D d,
                                         TFromD<D>* p) noexcept {
  svst2(detail::active(d), p, svcreate2(v0, v1));
}
template <class D>
HYLE_FORCE_INLINE void StoreInterleaved3(VFromD<D> v0, VFromD<D> v1, VFromD<D> v2, D d,
                                         TFromD<D>* p) noexcept {
  svst3(detail::active(d), p, svcreate3(v0, v1, v2));
}
template <class D>
HYLE_FORCE_INLINE void StoreInterleaved4(VFromD<D> v0, VFromD<D> v1, VFromD<D> v2,
                                         VFromD<D> v3, D d, TFromD<D>* p) noexcept {
  svst4(detail::active(d), p, svcreate4(v0, v1, v2, v3));
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
// AddSub(a,b): a-b in even lanes, a+b in odd lanes. Base SVE has no integer addsub;
// selecting between the two native arithmetic results preserves the contract for every lane type.
template <class V> HYLE_FORCE_INLINE V AddSub(V a, V b) noexcept {
  return OddEven(Add(a, b), Sub(a, b));
}
// Raw hardware estimates. ReciprocalNewton/ReciprocalSqrtNewton in vec.h refine these once.
template <class V> HYLE_FORCE_INLINE V ApproximateReciprocal(V v) noexcept {
  static_assert(std::is_floating_point_v<TFromV<V>>, "ApproximateReciprocal is float only");
  return svrecpe(v);
}
template <class V> HYLE_FORCE_INLINE V ApproximateReciprocalSqrt(V v) noexcept {
  static_assert(std::is_floating_point_v<TFromV<V>>, "ApproximateReciprocalSqrt is float only");
  return svrsqrte(v);
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
// MaskFromVec: active where the lane's raw MSB is set. Numeric `v < 0` is wrong for unsigned
// lanes and misses floating -0/negative-NaN, so compare a same-width signed bit view against zero.
template <class V> HYLE_FORCE_INLINE svbool_t MaskFromVec(V v) noexcept {
  using T = TFromV<V>;
  if constexpr (sizeof(T) == 1) return svcmplt(detail::ptrue<T>(), svreinterpret_s8(v), std::int8_t{0});
  else if constexpr (sizeof(T) == 2) return svcmplt(detail::ptrue<T>(), svreinterpret_s16(v), std::int16_t{0});
  else if constexpr (sizeof(T) == 4) return svcmplt(detail::ptrue<T>(), svreinterpret_s32(v), std::int32_t{0});
  else return svcmplt(detail::ptrue<T>(), svreinterpret_s64(v), std::int64_t{0});
}
// MaskFromNormalizedVec: caller guarantees v is already canonical all-ones/all-zero per lane. SVE
// masks are predicate registers, so the MSB compare (MaskFromVec) IS the natural conversion -- there
// is no vector-domain sign-broadcast to skip (that was the avx2-only redundancy). Forward.
template <class V> HYLE_FORCE_INLINE svbool_t MaskFromNormalizedVec(V v) noexcept {
  return MaskFromVec(v);
}
// VecFromMask(d, m): all-ones (T(-1)) where active, else 0.
template <class D> HYLE_FORCE_INLINE VFromD<D> VecFromMask(D d, svbool_t m) noexcept {
  using T = TFromD<D>;
  return svsel(m, Set(d, static_cast<T>(-1)), Zero(d));
}

// ---- mask reductions ----------------------------------------------------------------
// AllTrue/AllFalse: test the mask against the lane-width all-true predicate. CountTrue: svcntp
// under the lane-width predicate counts ACTIVE ELEMENTS (matching the lane count, not bytes).
template <class D> HYLE_FORCE_INLINE bool AllTrue(D d, svbool_t m) noexcept {
  const svbool_t pg = detail::active(d);
  return !svptest_any(pg, svnot_z(pg, m));
}
template <class D> HYLE_FORCE_INLINE bool AllFalse(D d, svbool_t m) noexcept {
  return !svptest_any(detail::active(d), m);
}
template <class D> HYLE_FORCE_INLINE std::size_t CountTrue(D d, svbool_t m) noexcept {
  using T = TFromD<D>;
  const svbool_t pg = detail::active(d);
  if constexpr (sizeof(T) == 1) return static_cast<std::size_t>(svcntp_b8(pg, m));
  else if constexpr (sizeof(T) == 2) return static_cast<std::size_t>(svcntp_b16(pg, m));
  else if constexpr (sizeof(T) == 4) return static_cast<std::size_t>(svcntp_b32(pg, m));
  else return static_cast<std::size_t>(svcntp_b64(pg, m));
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
  if (AllFalse(d, m)) return -1;
  return static_cast<std::ptrdiff_t>(CountTrue(d, detail::brkb_before_first<T>(m)));
}
template <class D> HYLE_FORCE_INLINE std::ptrdiff_t FindLastTrue(D d, svbool_t m) noexcept {
  using T = TFromD<D>;
  if (AllFalse(d, m)) return -1;
  const svbool_t pg = detail::active(d);
  if constexpr (sizeof(T) == 1) return static_cast<std::ptrdiff_t>(svmaxv(pg, svsel(m, svindex_u8(0, 1), svdup_n_u8(0))));
  else if constexpr (sizeof(T) == 2) return static_cast<std::ptrdiff_t>(svmaxv(pg, svsel(m, svindex_u16(0, 1), svdup_n_u16(0))));
  else if constexpr (sizeof(T) == 4) return static_cast<std::ptrdiff_t>(svmaxv(pg, svsel(m, svindex_u32(0, 1), svdup_n_u32(0))));
  else return static_cast<std::ptrdiff_t>(svmaxv(pg, svsel(m, svindex_u64(0, 1), svdup_n_u64(0))));
}


// ---- GetLane: lane 0 ----------------------------------------------------------------
template <class V> HYLE_FORCE_INLINE TFromV<V> GetLane(V v) noexcept {
  return svlasta(svpfalse_b(), v);  // lasta with empty pred = element 0
}

// ---- value reductions (whole-vector -> scalar) --------------------------------------
// svaddv/sv{min,max}v/sv{and,orr,eor}v reduce across all active lanes to a scalar. Under the
// lane-width all-true predicate this is the full-vector reduction. Signature matches Highway:
// Reduce*(D, V) -> T.
template <class D> HYLE_FORCE_INLINE TFromD<D> ReduceSum(D d, VFromD<D> v) noexcept {
  return svaddv(detail::active(d), v);
}
template <class D> HYLE_FORCE_INLINE TFromD<D> ReduceMin(D d, VFromD<D> v) noexcept {
  return svminv(detail::active(d), v);
}
template <class D> HYLE_FORCE_INLINE TFromD<D> ReduceMax(D d, VFromD<D> v) noexcept {
  return svmaxv(detail::active(d), v);
}
template <class D> HYLE_FORCE_INLINE TFromD<D> ReduceAnd(D d, VFromD<D> v) noexcept {
  return svandv(detail::active(d), v);
}
template <class D> HYLE_FORCE_INLINE TFromD<D> ReduceOr(D d, VFromD<D> v) noexcept {
  return svorv(detail::active(d), v);
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
template <int kBits, class V> HYLE_FORCE_INLINE V RotateRight(V v) noexcept {
  using T = TFromV<V>;
  static_assert(std::is_integral_v<T>, "RotateRight is integer-lane only");
  constexpr int kSizeBits = static_cast<int>(sizeof(T) * 8);
  static_assert(kBits >= 0 && kBits < kSizeBits, "rotate count in [0,bits)");
  if constexpr (kBits == 0) {
    return v;
  } else {
    const RebindToUnsigned<DFromV<V>> du;
    const auto uv = BitCast(du, v);
    return BitCast(DFromV<V>(), Or(ShiftRight<kBits>(uv), ShiftLeft<kSizeBits - kBits>(uv)));
  }
}
template <int kBits, class V> HYLE_FORCE_INLINE V RotateLeft(V v) noexcept {
  constexpr int kSizeBits = static_cast<int>(sizeof(TFromV<V>) * 8);
  return RotateRight<(kSizeBits - kBits) % kSizeBits>(v);
}

// ---- whole-vector slides -------------------------------------------------------------
template <class D> HYLE_FORCE_INLINE VFromD<D> SlideUpLanes(D d, VFromD<D> v,
                                                            std::size_t amt) noexcept {
  using T = TFromD<D>; using U = MakeUnsigned<T>;
  const std::size_t n = Lanes(d);
  HYLE_FATAL_ASSERT(amt <= n);
  if (amt == n) return Zero(d);
  if constexpr (sizeof(T) == 1)
    return svtbl(v, svsub_x(svptrue_b8(), svindex_u8(0, 1), static_cast<U>(amt)));
  else if constexpr (sizeof(T) == 2)
    return svtbl(v, svsub_x(svptrue_b16(), svindex_u16(0, 1), static_cast<U>(amt)));
  else if constexpr (sizeof(T) == 4)
    return svtbl(v, svsub_x(svptrue_b32(), svindex_u32(0, 1), static_cast<U>(amt)));
  else
    return svtbl(v, svsub_x(svptrue_b64(), svindex_u64(0, 1), static_cast<U>(amt)));
}
template <class D> HYLE_FORCE_INLINE VFromD<D> SlideDownLanes(D d, VFromD<D> v,
                                                              std::size_t amt) noexcept {
  using T = TFromD<D>; using U = MakeUnsigned<T>;
  const std::size_t n = Lanes(d);
  HYLE_FATAL_ASSERT(amt <= n);
  if (amt == n) return Zero(d);
  VFromD<D> shifted;
  if constexpr (sizeof(T) == 1)
    shifted = svtbl(v, svadd_x(svptrue_b8(), svindex_u8(0, 1), static_cast<U>(amt)));
  else if constexpr (sizeof(T) == 2)
    shifted = svtbl(v, svadd_x(svptrue_b16(), svindex_u16(0, 1), static_cast<U>(amt)));
  else if constexpr (sizeof(T) == 4)
    shifted = svtbl(v, svadd_x(svptrue_b32(), svindex_u32(0, 1), static_cast<U>(amt)));
  else
    shifted = svtbl(v, svadd_x(svptrue_b64(), svindex_u64(0, 1), static_cast<U>(amt)));
  return IfThenElse(FirstN(d, n - amt), shifted, Zero(d));
}
template <class D> HYLE_FORCE_INLINE VFromD<D> Slide1Up(D d, VFromD<D> v) noexcept {
  return SlideUpLanes(d, v, 1);
}
template <class D> HYLE_FORCE_INLINE VFromD<D> Slide1Down(D d, VFromD<D> v) noexcept {
  return SlideDownLanes(d, v, 1);
}

// ---- Reverse (whole-vector element reversal) ----------------------------------------
template <class D> HYLE_FORCE_INLINE VFromD<D> Reverse(D d, VFromD<D> v) noexcept {
  using T = TFromD<D>; using U = MakeUnsigned<T>;
  const U last = static_cast<U>(Lanes(d) - 1);
  if constexpr (sizeof(T) == 1) return svtbl(v, svsub_x(svptrue_b8(), svdup_n_u8(last), svindex_u8(0, 1)));
  else if constexpr (sizeof(T) == 2) return svtbl(v, svsub_x(svptrue_b16(), svdup_n_u16(last), svindex_u16(0, 1)));
  else if constexpr (sizeof(T) == 4) return svtbl(v, svsub_x(svptrue_b32(), svdup_n_u32(last), svindex_u32(0, 1)));
  else return svtbl(v, svsub_x(svptrue_b64(), svdup_n_u64(last), svindex_u64(0, 1)));
}

// ---- tail handling: FirstN / LoadN / StoreN (SVE NATIVE, not store+reload) ----------
// FirstN(d, n): predicate with the first n lanes active. svwhilelt_b*(0, n) is exactly this and
// is SVE's core tail primitive (no emulation). Returns svbool_t (our Mask type).
template <class D> HYLE_FORCE_INLINE svbool_t FirstN(D d, std::size_t n) noexcept {
  using T = TFromD<D>;
  const std::uint64_t nn = static_cast<std::uint64_t>(n < Lanes(d) ? n : Lanes(d));
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

// ---- width-changing convert (Promote/Demote) — VL-agnostic via svunpk / clamp+uzp1 -----------
// SVE's svunpklo/hi widen the LOWER/UPPER half of the source's active lanes to double width, which
// is exactly the half-at-a-time HYLE contract (dst has half the lanes) AND is naturally VL-correct
// (operates on the runtime active-lane set, no compile-time lane count baked in -> no OOB risk that
// motivated the earlier deferral). Integer only for unpk; f32->f64 uses svcvt on the unpacked idx.
// PromoteLowerTo(dtag, v): widen v's lower half. NSrc(compile-time upper bound) = 2*NDst. Integer
// only (int8->16->32->64); float f32->f64 promote is deferred (svcvt_f64_f32 needs lane-position
// juggling, not the quantization hot path — documented boundary).
template <typename DstT, std::size_t NDst, class V>
HYLE_FORCE_INLINE VFromD<Simd<DstT, NDst, 0>> PromoteLowerTo(Simd<DstT, NDst, 0>, V v) noexcept {
  using SrcT = TFromV<V>;
  static_assert(sizeof(DstT) > sizeof(SrcT) && std::is_integral_v<DstT>, "PromoteLowerTo widens (integer)");
  if constexpr (std::is_signed_v<DstT>) {
    if constexpr (sizeof(DstT) == 2) return svunpklo_s16(v);
    else if constexpr (sizeof(DstT) == 4) return svunpklo_s32(v);
    else return svunpklo_s64(v);
  } else {
    if constexpr (sizeof(DstT) == 2) return svunpklo_u16(v);
    else if constexpr (sizeof(DstT) == 4) return svunpklo_u32(v);
    else return svunpklo_u64(v);
  }
}
template <typename DstT, std::size_t NDst, class V>
HYLE_FORCE_INLINE VFromD<Simd<DstT, NDst, 0>> PromoteUpperTo(Simd<DstT, NDst, 0> dd, V v) noexcept {
  using SrcT = TFromV<V>;
  static_assert(sizeof(DstT) > sizeof(SrcT) && std::is_integral_v<DstT>, "PromoteUpperTo widens (integer)");
  if constexpr (NDst < kNativeVectorBytes / sizeof(DstT)) {
    const Simd<SrcT, 2 * NDst, 0> ds;
    return PromoteLowerTo(dd, SlideDownLanes(ds, v, Lanes(dd)));
  } else if constexpr (std::is_signed_v<DstT>) {
    if constexpr (sizeof(DstT) == 2) return svunpkhi_s16(v);
    else if constexpr (sizeof(DstT) == 4) return svunpkhi_s32(v);
    else return svunpkhi_s64(v);
  } else {
    if constexpr (sizeof(DstT) == 2) return svunpkhi_u16(v);
    else if constexpr (sizeof(DstT) == 4) return svunpkhi_u32(v);
    else return svunpkhi_u64(v);
  }
}
// DemoteTo(dtag, a, b): saturate two wide vectors down to one narrow (dst has 2x lanes). Base SVE
// has no svqxtn (SVE2); clamp each wide lane to the narrow range, then svuzp1 the low halves (the
// clamped value fits in the low half of each wide lane -> even narrow lanes carry it in order).
template <typename DstT, std::size_t NDst, class V>
HYLE_FORCE_INLINE VFromD<Simd<DstT, NDst, 0>> DemoteTo(Simd<DstT, NDst, 0> dd, V a, V b) noexcept {
  using SrcT = TFromV<V>;
  static_assert(sizeof(DstT) < sizeof(SrcT), "DemoteTo narrows");
  static_assert(NDst % 2 == 0, "DemoteTo requires two equal source halves");
  const svbool_t pg = detail::ptrue<SrcT>();
  const SrcT lo = std::is_unsigned_v<SrcT> ? SrcT{0}
                                                : static_cast<SrcT>(std::numeric_limits<DstT>::min());
  const SrcT hi = static_cast<SrcT>(std::numeric_limits<DstT>::max());
  const V vlo = Set(Simd<SrcT, kNativeVectorBytes / sizeof(SrcT), 0>{}, lo);
  const V vhi = Set(Simd<SrcT, kNativeVectorBytes / sizeof(SrcT), 0>{}, hi);
  const V ca = svmax_x(pg, svmin_x(pg, a, vhi), vlo);
  const V cb = svmax_x(pg, svmin_x(pg, b, vhi), vlo);
  if constexpr (std::is_signed_v<DstT>) {
    const auto ba = [&]() {
      if constexpr (sizeof(DstT) == 1) return svreinterpret_s8(ca);
      else if constexpr (sizeof(DstT) == 2) return svreinterpret_s16(ca);
      else return svreinterpret_s32(ca);
    }();
    const auto bb = [&]() {
      if constexpr (sizeof(DstT) == 1) return svreinterpret_s8(cb);
      else if constexpr (sizeof(DstT) == 2) return svreinterpret_s16(cb);
      else return svreinterpret_s32(cb);
    }();
    if constexpr (NDst == kNativeVectorBytes / sizeof(DstT)) return svuzp1(ba, bb);
    const auto na = svuzp1(ba, ba), nb = svuzp1(bb, bb);
    const std::size_t half = Lanes(dd) / 2;
    return IfThenElse(FirstN(dd, half), na, SlideUpLanes(dd, nb, half));
  } else {
    const auto ba = [&]() {
      if constexpr (sizeof(DstT) == 1) return svreinterpret_u8(ca);
      else if constexpr (sizeof(DstT) == 2) return svreinterpret_u16(ca);
      else return svreinterpret_u32(ca);
    }();
    const auto bb = [&]() {
      if constexpr (sizeof(DstT) == 1) return svreinterpret_u8(cb);
      else if constexpr (sizeof(DstT) == 2) return svreinterpret_u16(cb);
      else return svreinterpret_u32(cb);
    }();
    if constexpr (NDst == kNativeVectorBytes / sizeof(DstT)) return svuzp1(ba, bb);
    const auto na = svuzp1(ba, ba), nb = svuzp1(bb, bb);
    const std::size_t half = Lanes(dd) / 2;
    return IfThenElse(FirstN(dd, half), na, SlideUpLanes(dd, nb, half));
  }
}
// WidenMulPairwiseAdd(d32, a_i16, b_i16): r[k] = a[2k]*b[2k] + a[2k+1]*b[2k+1] (the fixed-point dot
// core, matches x86 madd_epi16). Base SVE has no madd; view each adjacent i16 pair as one i32 lane,
// extract even (low 16, sign-extend via <<16 >>16) and odd (high 16, arith >>16), multiply as i32,
// add. VL-agnostic. i16 -> i32 only (the quantization pairwise-dot shape).
template <std::size_t NDst, class V>
HYLE_FORCE_INLINE VFromD<Simd<std::int32_t, NDst, 0>> WidenMulPairwiseAdd(Simd<std::int32_t, NDst, 0>, V a, V b) noexcept {
  static_assert(std::is_same_v<TFromV<V>, std::int16_t>, "WidenMulPairwiseAdd: i16 -> i32 pairwise");
  const svbool_t p32 = detail::ptrue<std::int32_t>();
  const svint32_t a32 = svreinterpret_s32_s16(a);
  const svint32_t b32 = svreinterpret_s32_s16(b);
  const svint32_t ae = svasr_x(p32, svlsl_x(p32, a32, 16), 16);  // even lane (low 16, sign-ext)
  const svint32_t ao = svasr_x(p32, a32, 16);                    // odd lane (high 16, arith)
  const svint32_t be = svasr_x(p32, svlsl_x(p32, b32, 16), 16);
  const svint32_t bo = svasr_x(p32, b32, 16);
  return svadd_x(p32, svmul_x(p32, ae, be), svmul_x(p32, ao, bo));
}

// ---- gather (SVE NATIVE svld1_gather, not store+reload) -----------------------------
// GatherIndex(d, base, idx): dst[i] = base[idx[i]]. SVE's gather takes the index vector as the
// same-width unsigned/signed lanes; svld1_gather_index scales by element size (a true index, not
// byte offset). 32/64-bit lanes only (SVE gather is defined there).
template <class D, class VI>
HYLE_FORCE_INLINE VFromD<D> GatherIndex(D d, const TFromD<D>* base, VI idx) noexcept {
  return svld1_gather_index(detail::active(d), base, idx);
}

// ---- swizzle (WHOLE-VECTOR scalable-native; SEE CONTRACT NOTE) ----------------------
// SCALABLE CONTRACT: on a scalable backend these are WHOLE-VECTOR, not per-128-bit-block like the
// fixed-width x86 backends. SVE's svzip1/svuzp1/svtrn1 operate across the entire VL. This matches
// the mathematical (scalar-oracle) definition and NEON's single-128-bit-block behavior; it differs
// from AVX2's 256-bit "two blocks" only for the block-relative ops. Concat* halves use Lanes(d)/2
// (whole-vector half-concat; Highway arm_sve implements the same -- "no scalable meaning" was wrong).
// InterleaveLower(a,b) = [a0,b0,a1,b1,...].
template <class V> HYLE_FORCE_INLINE V InterleaveLower(V a, V b) noexcept { return svzip1(a, b); }
template <class V> HYLE_FORCE_INLINE V InterleaveUpper(V a, V b) noexcept { return svzip2(a, b); }
template <class D, class V> HYLE_FORCE_INLINE V InterleaveLower(D, V a, V b) noexcept { return svzip1(a, b); }
template <class D, class V> HYLE_FORCE_INLINE V InterleaveUpper(D d, V a, V b) noexcept {
  const std::size_t half = Lanes(d) / 2;
  return svzip1(SlideDownLanes(d, a, half), SlideDownLanes(d, b, half));
}
template <class V> HYLE_FORCE_INLINE V ZipLower(V a, V b) noexcept { return svzip1(a, b); }
template <class V> HYLE_FORCE_INLINE V ZipUpper(V a, V b) noexcept { return svzip2(a, b); }

// Concat{Lower,Upper}{Lower,Upper}(d,hi,lo): whole-vector half-concat (Highway scalable).
// h = Lanes(d)/2. Result layout:
//   ConcatLowerLower = [lo[0..h), hi[0..h)]
//   ConcatUpperUpper = [lo[h..n), hi[h..n)]
//   ConcatLowerUpper = [lo[h..n), hi[0..h)]
//   ConcatUpperLower = [lo[0..h), hi[h..n)]
// Built from SlideUp/Down + FirstN select (HYLE has no Splice; SlideDown zero-fill is covered by mask).
template <class D, class V>
HYLE_FORCE_INLINE V ConcatLowerLower(D d, V hi, V lo) noexcept {
  const std::size_t h = Lanes(d) / 2;
  return IfThenElse(FirstN(d, h), lo, SlideUpLanes(d, hi, h));
}
template <class D, class V>
HYLE_FORCE_INLINE V ConcatUpperUpper(D d, V hi, V lo) noexcept {
  const std::size_t h = Lanes(d) / 2;
  return IfThenElse(FirstN(d, h), SlideDownLanes(d, lo, h), hi);
}
template <class D, class V>
HYLE_FORCE_INLINE V ConcatLowerUpper(D d, V hi, V lo) noexcept {
  const std::size_t h = Lanes(d) / 2;
  return IfThenElse(FirstN(d, h), SlideDownLanes(d, lo, h), SlideUpLanes(d, hi, h));
}
template <class D, class V>
HYLE_FORCE_INLINE V ConcatUpperLower(D d, V hi, V lo) noexcept {
  const std::size_t h = Lanes(d) / 2;
  return IfThenElse(FirstN(d, h), lo, hi);
}

// ConcatEven/Odd: gather the even/odd lanes of the logical concatenation [lo, hi]. For capped tags,
// compact each source first; full-vector svuzp would put too many `lo` lanes before `hi`.
template <class D, class V> HYLE_FORCE_INLINE V ConcatEven(D d, V hi, V lo) noexcept {
  const std::size_t half = Lanes(d) / 2;
  const auto m = detail::even_lane_mask<TFromV<V>>();
  const V lo_packed = svcompact(m, lo), hi_packed = svcompact(m, hi);
  return IfThenElse(FirstN(d, half), lo_packed, SlideUpLanes(d, hi_packed, half));
}
template <class D, class V> HYLE_FORCE_INLINE V ConcatOdd(D d, V hi, V lo) noexcept {
  using T = TFromV<V>;
  const std::size_t half = Lanes(d) / 2;
  const svbool_t pg = detail::ptrue<T>();
  const auto m = svnot_z(pg, detail::even_lane_mask<T>());
  const V lo_packed = svcompact(m, lo), hi_packed = svcompact(m, hi);
  return IfThenElse(FirstN(d, half), lo_packed, SlideUpLanes(d, hi_packed, half));
}
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
template <class V> HYLE_FORCE_INLINE V DupEven(V v) noexcept { return svtrn1(v, v); }
template <class V> HYLE_FORCE_INLINE V DupOdd(V v) noexcept { return svtrn2(v, v); }
template <class V> HYLE_FORCE_INLINE V InterleaveEven(V a, V b) noexcept { return svtrn1(a, b); }
template <class V> HYLE_FORCE_INLINE V InterleaveOdd(V a, V b) noexcept { return svtrn2(a, b); }
template <class D> HYLE_FORCE_INLINE VFromD<D> Reverse2(D, VFromD<D> v) noexcept {
  using T = TFromD<D>;
  const svbool_t pg = detail::ptrue<T>();
  if constexpr (sizeof(T) == 1)
    return svtbl(v, sveor_x(pg, svindex_u8(0, 1), svdup_n_u8(1)));
  else if constexpr (sizeof(T) == 2)
    return svtbl(v, sveor_x(pg, svindex_u16(0, 1), svdup_n_u16(1)));
  else if constexpr (sizeof(T) == 4)
    return svtbl(v, sveor_x(pg, svindex_u32(0, 1), svdup_n_u32(1)));
  else
    return svtbl(v, sveor_x(pg, svindex_u64(0, 1), svdup_n_u64(1)));
}
template <class D> HYLE_FORCE_INLINE VFromD<D> Reverse4(D d, VFromD<D> v) noexcept {
  using T = TFromD<D>; HYLE_FATAL_ASSERT(Lanes(d) >= 4 && Lanes(d) % 4 == 0);
  if constexpr (sizeof(T) == 1) return svtbl(v, sveor_x(svptrue_b8(), svindex_u8(0, 1), std::uint8_t{3}));
  else if constexpr (sizeof(T) == 2) return svtbl(v, sveor_x(svptrue_b16(), svindex_u16(0, 1), std::uint16_t{3}));
  else if constexpr (sizeof(T) == 4) return svtbl(v, sveor_x(svptrue_b32(), svindex_u32(0, 1), std::uint32_t{3}));
  else return svtbl(v, sveor_x(svptrue_b64(), svindex_u64(0, 1), std::uint64_t{3}));
}
template <class D> HYLE_FORCE_INLINE VFromD<D> Reverse8(D d, VFromD<D> v) noexcept {
  using T = TFromD<D>; HYLE_FATAL_ASSERT(Lanes(d) >= 8 && Lanes(d) % 8 == 0);
  if constexpr (sizeof(T) == 1) return svtbl(v, sveor_x(svptrue_b8(), svindex_u8(0, 1), std::uint8_t{7}));
  else if constexpr (sizeof(T) == 2) return svtbl(v, sveor_x(svptrue_b16(), svindex_u16(0, 1), std::uint16_t{7}));
  else if constexpr (sizeof(T) == 4) return svtbl(v, sveor_x(svptrue_b32(), svindex_u32(0, 1), std::uint32_t{7}));
  else return svtbl(v, sveor_x(svptrue_b64(), svindex_u64(0, 1), std::uint64_t{7}));
}
template <class V> HYLE_FORCE_INLINE V ReverseLaneBytes(V v) noexcept {
  using T = TFromV<V>;
  static_assert(std::is_integral_v<T>, "ReverseLaneBytes is integer-only");
  if constexpr (sizeof(T) == 1) return v;
  else return BitCast(DFromV<V>(), svtbl(svreinterpret_u8(v),
      sveor_x(svptrue_b8(), svindex_u8(0, 1), static_cast<std::uint8_t>(sizeof(T) - 1))));
}

// Indices are native unsigned scalable vectors (sizeless types cannot be wrapped in a struct).
template <class D, class VI>
HYLE_FORCE_INLINE VFromD<RebindToUnsigned<D>> IndicesFromVec(D, VI indices) noexcept {
  static_assert(sizeof(TFromD<D>) == sizeof(TFromV<VI>), "Index/lane size mismatch");
  return BitCast(RebindToUnsigned<D>(), indices);
}
template <class V, class VI> HYLE_FORCE_INLINE V TableLookupLanes(V v, VI indices) noexcept {
  return svtbl(v, indices);
}
template <class V, class VI> HYLE_FORCE_INLINE VI TableLookupBytes(V table, VI indices) noexcept {
  const svbool_t pg = svptrue_b8();
  const svuint8_t idx = svreinterpret_u8(indices);
  const svuint8_t local = svand_x(pg, idx, std::uint8_t{15});
  const svuint8_t block = svbic_x(pg, svindex_u8(0, 1), std::uint8_t{15});
  const svuint8_t out = svtbl(svreinterpret_u8(table), svadd_x(pg, block, local));
  return BitCast(DFromV<VI>(), out);
}
template <class V, class VI> HYLE_FORCE_INLINE VI TableLookupBytesInRange(V table, VI indices) noexcept {
  return TableLookupBytes(table, indices);  // masked form is valid on the in-range domain
}
template <class V, class VI> HYLE_FORCE_INLINE VI TableLookupBytesOr0(V table, VI indices) noexcept {
  const svuint8_t idx = svreinterpret_u8(indices);
  const svbool_t msb = svcmplt(svptrue_b8(), svreinterpret_s8(idx), std::int8_t{0});
  const VI lookup = TableLookupBytes(table, indices);
  return svsel(msb, Zero(DFromV<VI>()), lookup);
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
template <class D> HYLE_FORCE_INLINE VFromD<D> MaskedLoad(svbool_t m, D d, const TFromD<D>* p) noexcept {
  const svbool_t active = detail::active(d);
  const svbool_t masked = svand_z(active, active, m);
  return svld1(masked, p);
}
template <class D>
HYLE_FORCE_INLINE VFromD<D> MaskedLoadOr(VFromD<D> no, svbool_t m, D d, const TFromD<D>* p) noexcept {
  const svbool_t active = detail::active(d);
  const svbool_t masked = svand_z(active, active, m);
  return svsel(masked, svld1(masked, p), no);
}
template <class D> HYLE_FORCE_INLINE void BlendedStore(VFromD<D> v, svbool_t m, D d, TFromD<D>* p) noexcept {
  const svbool_t active = detail::active(d);
  svst1(svand_z(active, active, m), p, v);
}

// ---- scatter (SVE native svst1_scatter) + CompressStore -----------------------------
template <class D, class VI>
HYLE_FORCE_INLINE void ScatterIndex(VFromD<D> v, D d, TFromD<D>* base, VI idx) noexcept {
  svst1_scatter_index(detail::active(d), base, idx, v);
}
// CompressStore(v, m, d, p): pack active lanes to the front, store CountTrue(m) of them, return
// that count. svcompact packs; then a whilelt-predicated store writes exactly the active count.
template <class D>
HYLE_FORCE_INLINE std::size_t CompressStore(VFromD<D> v, svbool_t m, D d, TFromD<D>* p) noexcept {
  const svbool_t active = detail::active(d);
  const svbool_t masked = svand_z(active, active, m);
  const std::size_t cnt = CountTrue(d, masked);
  svst1(FirstN(d, cnt), p, svcompact(masked, v));
  return cnt;
}

// ---- P2 widen / quantize / reduction completion -------------------------------------
template <typename DstT, std::size_t NDst, class V>
HYLE_FORCE_INLINE VFromD<Simd<DstT, NDst, 0>> PromoteEvenTo(Simd<DstT, NDst, 0> dd,
                                                            V v) noexcept {
  using SrcT = TFromV<V>;
  static_assert(sizeof(DstT) == 2 * sizeof(SrcT), "PromoteEvenTo requires adjacent widths");
  if constexpr (std::is_same_v<SrcT, float> && std::is_same_v<DstT, double>) {
    return svcvt_f64_f32_x(detail::ptrue<float>(), v);
  } else {
    static_assert(std::is_integral_v<SrcT> && std::is_integral_v<DstT>,
                  "PromoteEvenTo supports integers and f32->f64");
    if constexpr (std::is_signed_v<SrcT>) {
      const Simd<MakeSigned<DstT>, NDst, 0> ds;
      const auto bits = BitCast(ds, v);
      const auto wide = [&]() {
        if constexpr (sizeof(DstT) == 2) return svextb_s16_x(detail::active(ds), bits);
        else if constexpr (sizeof(DstT) == 4) return svexth_s32_x(detail::active(ds), bits);
        else return svextw_s64_x(detail::active(ds), bits);
      }();
      if constexpr (std::is_signed_v<DstT>) return wide;
      else return BitCast(dd, wide);
    } else {
      const auto bits = BitCast(dd, v);
      return And(bits, Set(dd, static_cast<DstT>(std::numeric_limits<MakeUnsigned<SrcT>>::max())));
    }
  }
}
template <typename DstT, std::size_t NDst, class V>
HYLE_FORCE_INLINE VFromD<Simd<DstT, NDst, 0>> PromoteOddTo(Simd<DstT, NDst, 0> dd,
                                                           V v) noexcept {
  return PromoteEvenTo(dd, DupOdd(v));
}

template <typename DstT, std::size_t N, int kPow2, class V>
HYLE_FORCE_INLINE VFromD<Simd<DstT, N, kPow2>> TruncateTo(Simd<DstT, N, kPow2> dd, V v) noexcept {
  using SrcT = TFromV<V>;
  static_assert(std::is_integral_v<DstT> && std::is_integral_v<SrcT> && sizeof(DstT) < sizeof(SrcT),
                "TruncateTo is modular integer narrowing");
  constexpr std::size_t ratio = sizeof(SrcT) / sizeof(DstT);
  static_assert(ratio == 2 || ratio == 4 || ratio == 8);
  auto out = BitCast(dd, v);
  if constexpr (ratio >= 2) out = svuzp1(out, out);
  if constexpr (ratio >= 4) out = svuzp1(out, out);
  if constexpr (ratio >= 8) out = svuzp1(out, out);
  return out;
}
template <class V>
HYLE_FORCE_INLINE svuint8_t U8FromU32(V v) noexcept {
  static_assert(std::is_same_v<TFromV<V>, std::uint32_t>, "U8FromU32 requires u32 lanes");
  using DS = DFromV<V>;
  return TruncateTo(Rebind<std::uint8_t, DS>{}, v);
}

template <typename DstT, std::size_t NDst, class V>
HYLE_FORCE_INLINE VFromD<Simd<DstT, NDst, 0>> ReorderDemote2To(Simd<DstT, NDst, 0> d,
                                                               V a, V b) noexcept {
  return DemoteTo(d, a, b);
}
template <std::size_t NDst>
HYLE_FORCE_INLINE svint16_t SatWidenMulPairwiseAdd(Simd<std::int16_t, NDst, 0> di16,
                                                   svuint8_t a, svint8_t b) noexcept {
  const Simd<std::uint16_t, NDst, 0> du16;
  const auto ae = BitCast(di16, PromoteEvenTo(du16, a));
  const auto ao = BitCast(di16, PromoteOddTo(du16, a));
  return SaturatedAdd(Mul(ae, PromoteEvenTo(di16, b)), Mul(ao, PromoteOddTo(di16, b)));
}
template <std::size_t NDst>
HYLE_FORCE_INLINE svint32_t ReorderWidenMulAccumulate(Simd<std::int32_t, NDst, 0> d32,
                                                       svint16_t a, svint16_t b,
                                                       svint32_t sum0,
                                                       svint32_t& sum1) noexcept {
  (void)sum1;
  return Add(sum0, WidenMulPairwiseAdd(d32, a, b));
}
template <class V> HYLE_FORCE_INLINE V RearrangeToOddPlusEven(V sum0, V sum1) noexcept {
  return Add(sum0, sum1);
}
template <std::size_t NDst>
HYLE_FORCE_INLINE svint32_t SumOfMulQuadAccumulate(Simd<std::int32_t, NDst, 0>,
                                                   svint8_t a, svint8_t b,
                                                   svint32_t sum) noexcept {
  return svdot_s32(sum, a, b);
}

HYLE_FORCE_INLINE svuint16_t SumsOf2(svuint8_t v) noexcept {
  const DFromV<svuint16_t> d;
  return Add(PromoteEvenTo(d, v), PromoteOddTo(d, v));
}
HYLE_FORCE_INLINE svuint32_t SumsOf4(svuint8_t v) noexcept {
  return svdot_n_u32(Zero(DFromV<svuint32_t>{}), v, 1);
}
HYLE_FORCE_INLINE svuint64_t SumsOf8(svuint8_t v) noexcept {
  const DFromV<svuint64_t> d;
  const auto s4 = SumsOf4(v);
  return Add(PromoteEvenTo(d, s4), PromoteOddTo(d, s4));
}

template <class D> HYLE_FORCE_INLINE TFromD<D> ReduceMul(D d, VFromD<D> v) noexcept {
  const std::size_t n = Lanes(d);
  VFromD<D> acc = v;
  for (std::size_t step = 1; step < n; step *= 2) {
    const auto shifted = SlideDownLanes(d, acc, step);
    acc = Mul(acc, IfThenElse(FirstN(d, n - step), shifted, Set(d, TFromD<D>{1})));
  }
  return GetLane(acc);
}
HYLE_FORCE_INLINE svint16_t MulFixedPoint15(svint16_t a, svint16_t b) noexcept {
  const DFromV<svint16_t> d;
  const RebindToUnsigned<decltype(d)> du;
  const svuint16_t lo = BitCast(du, Mul(a, b));
  const svint16_t hi = MulHigh(a, b);
  const svuint16_t rounding = ShiftRight<1>(Add(ShiftRight<14>(lo), Set(du, 1)));
  return Add(Add(hi, hi), BitCast(d, rounding));
}
template <class V> HYLE_FORCE_INLINE V SaturatedNeg(V v) noexcept {
  static_assert(std::is_integral_v<TFromV<V>> && std::is_signed_v<TFromV<V>>,
                "SaturatedNeg is signed-int");
  return SaturatedSub(Zero(DFromV<V>()), v);
}
template <class V> HYLE_FORCE_INLINE V SaturatedAbs(V v) noexcept {
  static_assert(std::is_integral_v<TFromV<V>> && std::is_signed_v<TFromV<V>>,
                "SaturatedAbs is signed-int");
  return Max(v, SaturatedNeg(v));
}

// ---- P5 trivial synthesis + mask prefix ops -----------------------------------------
// Fixed-width backends already have these; keep scalable backends at parity. Mask ops that need
// a lane-width governing predicate take D (svbool_t itself is untyped).
template <class V> HYLE_FORCE_INLINE V BroadcastSignBit(V v) noexcept {
  using T = TFromV<V>;
  static_assert(std::is_integral_v<T> && std::is_signed_v<T>, "BroadcastSignBit is signed-int only");
  return ShiftRight<static_cast<int>(sizeof(T) * 8) - 1>(v);
}
template <class D>
HYLE_FORCE_INLINE svbool_t ExclusiveNeither(D d, svbool_t a, svbool_t b) noexcept {
  return svnor_z(detail::active(d), a, b);  // ~(a|b) under D's active lanes
}
template <class V> HYLE_FORCE_INLINE svbool_t IsEitherNaN(V a, V b) noexcept {
  static_assert(std::is_floating_point_v<TFromV<V>>, "IsEitherNaN is float only");
  return svcmpuo(detail::ptrue<TFromV<V>>(), a, b);  // unordered compare
}
template <class V> HYLE_FORCE_INLINE V MulRound(V a, V b) noexcept {
  static_assert(std::is_floating_point_v<TFromV<V>>, "MulRound is float only");
  return Round(Mul(a, b));
}
template <class V> HYLE_FORCE_INLINE V GetExponent(V v) noexcept {
  using T = TFromV<V>;
  static_assert(std::is_floating_point_v<T>, "GetExponent is float only");
  const DFromV<V> d;
  const RebindToUnsigned<DFromV<V>> du;
  const RebindToSigned<DFromV<V>> di;
  if constexpr (std::is_same_v<T, float>) {
    const auto bits = BitCast(du, v);
    const auto biased = And(ShiftRight<23>(bits), Set(du, std::uint32_t{0xFFu}));
    const auto unbiased = Sub(biased, Set(du, std::uint32_t{127}));
    return ConvertTo(d, BitCast(di, unbiased));
  } else {
    const auto bits = BitCast(du, v);
    const auto biased = And(ShiftRight<52>(bits), Set(du, std::uint64_t{0x7FFull}));
    const auto unbiased = Sub(biased, Set(du, std::uint64_t{1023}));
    return ConvertTo(d, BitCast(di, unbiased));
  }
}
template <int kLane, class V> HYLE_FORCE_INLINE V BroadcastLane(V v) noexcept {
  return Broadcast<kLane>(v);
}

// Set*First family: SVE break instructions (svbrkb = before, svbrka = at-or-before).
// no-true: Before/AtOrBefore -> all true; Only/AtOrAfter -> all false (native break behavior).
template <class D> HYLE_FORCE_INLINE svbool_t SetBeforeFirst(D, svbool_t m) noexcept {
  using T = TFromD<D>;
  return svbrkb_z(detail::ptrue<T>(), m);
}
template <class D> HYLE_FORCE_INLINE svbool_t SetAtOrBeforeFirst(D, svbool_t m) noexcept {
  using T = TFromD<D>;
  return svbrka_z(detail::ptrue<T>(), m);
}
template <class D> HYLE_FORCE_INLINE svbool_t SetOnlyFirst(D, svbool_t m) noexcept {
  using T = TFromD<D>;
  const svbool_t pg = detail::ptrue<T>();
  return svbic_z(pg, svbrka_z(pg, m), svbrkb_z(pg, m));  // AtOrBefore AND NOT Before
}
template <class D> HYLE_FORCE_INLINE svbool_t SetAtOrAfterFirst(D, svbool_t m) noexcept {
  using T = TFromD<D>;
  const svbool_t pg = detail::ptrue<T>();
  return svnot_z(pg, svbrkb_z(pg, m));  // ~Before
}

// FindKnown*: same as FindFirst/LastTrue but caller guarantees a true lane (no -1 path).
template <class D>
HYLE_FORCE_INLINE std::size_t FindKnownFirstTrue(D d, svbool_t m) noexcept {
  using T = TFromD<D>;
  return CountTrue(d, detail::brkb_before_first<T>(m));
}
template <class D>
HYLE_FORCE_INLINE std::size_t FindKnownLastTrue(D d, svbool_t m) noexcept {
  using T = TFromD<D>;
  const svbool_t pg = detail::active(d);
  if constexpr (sizeof(T) == 1)
    return static_cast<std::size_t>(svmaxv(pg, svsel(m, svindex_u8(0, 1), svdup_n_u8(0))));
  else if constexpr (sizeof(T) == 2)
    return static_cast<std::size_t>(svmaxv(pg, svsel(m, svindex_u16(0, 1), svdup_n_u16(0))));
  else if constexpr (sizeof(T) == 4)
    return static_cast<std::size_t>(svmaxv(pg, svsel(m, svindex_u32(0, 1), svdup_n_u32(0))));
  else
    return static_cast<std::size_t>(svmaxv(pg, svsel(m, svindex_u64(0, 1), svdup_n_u64(0))));
}

// ---- P5 batch C: MulByPow2 / Expand / TrailingZeroCount ------------------------------
// MulByPow2(a, b) = a * 2^b (std::ldexp). SVE has native FSCALE (svscale), matching IEEE ldexp for
// all inputs (overflow -> inf, underflow -> subnormal/0). b is the matching-width signed int vector.
template <class V, class VI> HYLE_FORCE_INLINE V MulByPow2(V a, VI b) noexcept {
  static_assert(std::is_floating_point_v<TFromV<V>>, "MulByPow2 is float only");
  return svscale_x(detail::ptrue<TFromV<V>>(), a, b);
}
// Expand(v, m): scatter v's low CountTrue(m) lanes out to the m-true positions, 0 elsewhere (the
// inverse of Compress). SVE base has no native expand instruction (Highway builds an 8-lane block
// table); store+reload the scalar expand is the correct VL-agnostic form -- same as the fixed-width
// AVX2/SSE4/NEON backends (no hardware there either). Scratch is sized to kMaxLanes (widest VL).
template <class V> HYLE_FORCE_INLINE V Expand(V v, svbool_t m) noexcept {
  using T = TFromV<V>;
  const DFromV<V> d;
  const RebindToUnsigned<DFromV<V>> du;
  constexpr std::size_t kMax = DFromV<V>::kMaxLanes;
  alignas(64) T src[kMax];
  alignas(64) T dst[kMax];
  alignas(64) MakeUnsigned<T> mv[kMax];
  const std::size_t n = Lanes(d);
  Store(v, d, src);
  Store(VecFromMask(du, m), du, mv);  // all-ones (nonzero) where active, else 0
  std::size_t k = 0;
  for (std::size_t i = 0; i < n; ++i) dst[i] = mv[i] ? src[k++] : T{0};
  return Load(d, dst);
}
// TrailingZeroCount(v) = LeadingZeroCount(ReverseBits(v)) (Highway's arm_sve identity). svrbit is
// the native per-lane bit reversal; svclz (LeadingZeroCount) then counts from the (now high) end.
template <class V> HYLE_FORCE_INLINE V TrailingZeroCount(V v) noexcept {
  static_assert(std::is_integral_v<TFromV<V>>, "TrailingZeroCount is integer-lane only");
  return LeadingZeroCount(svrbit_x(detail::ptrue<TFromV<V>>(), v));
}

// ---- Crypto: AES + CLMul (simd-16 reclaim; native SVE2-AES, else software per-128-block) --------
// SCALABLE CONTRACT: like the swizzle ops, these are WHOLE-VECTOR — SVE2 AES/PMULL operate on EACH
// 128-bit segment of the VL-wide vector (Arm defines them per-quadword), so a VL=512 vector processes
// 4 independent AES blocks / 4 CLMul lane-pairs at once. The software fallback (base variant, no
// +sve2-aes) loops the shared AES-NI-byte-order reference per 128-bit block → bit-identical to x86/
// NEON. `svaese` self-includes AddRoundKey, so a full round = svaesmc(svaese(s,0)) ^ key (same trick
// as NEON vaeseq). Gated on __ARM_FEATURE_SVE2_AES.
namespace detail {
// store the whole vector to a byte buffer, run a per-128-block software op, reload.
template <class V, class Fn>
HYLE_FORCE_INLINE V sve_crypto_soft_blocks(V state, V key, Fn per_block) noexcept {
  using T = TFromV<V>;
  const DFromV<V> d;
  constexpr std::size_t kMaxBytes = DFromV<V>::kMaxLanes * sizeof(T);
  alignas(64) std::uint8_t s[kMaxBytes], k[kMaxBytes];
  Store(state, d, reinterpret_cast<T*>(s));
  Store(key, d, reinterpret_cast<T*>(k));
  const std::size_t bytes = Lanes(d) * sizeof(T);
  for (std::size_t b = 0; b + 16 <= bytes; b += 16) per_block(s + b, k + b);
  return Load(d, reinterpret_cast<T*>(s));
}
}  // namespace detail

#if defined(__ARM_FEATURE_SVE2_AES)
namespace detail {
// V <-> svuint8_t. SVE Vec IS the native sizeless type (no .raw wrapper); svreinterpret_u8 is the
// single-arg C++ overload, and svreinterpret back to V's element type via BitCast(DFromV<V>, r).
template <class V> HYLE_FORCE_INLINE svuint8_t sve_as_u8(V v) noexcept { return svreinterpret_u8(v); }
template <class V> HYLE_FORCE_INLINE V sve_from_u8(svuint8_t r) noexcept {
  return BitCast(DFromV<V>(), r);
}
}  // namespace detail
#endif

template <class V> HYLE_FORCE_INLINE V AESRound(V state, V key) noexcept {
  static_assert(std::is_integral_v<TFromV<V>>, "AES op is integer-lane only");
#if defined(__ARM_FEATURE_SVE2_AES)
  const svuint8_t z = svdup_n_u8(0);
  const svuint8_t mc = svaesmc_u8(svaese_u8(detail::sve_as_u8(state), z));  // AddRK(0)+SubB+ShiftR+MixC
  return detail::sve_from_u8<V>(sveor_u8_x(svptrue_b8(), mc, detail::sve_as_u8(key)));  // ^ roundkey
#else
  return detail::sve_crypto_soft_blocks(state, key, detail::aes_round_soft);
#endif
}
template <class V> HYLE_FORCE_INLINE V AESLastRound(V state, V key) noexcept {
  static_assert(std::is_integral_v<TFromV<V>>, "AES op is integer-lane only");
#if defined(__ARM_FEATURE_SVE2_AES)
  const svuint8_t z = svdup_n_u8(0);
  return detail::sve_from_u8<V>(
      sveor_u8_x(svptrue_b8(), svaese_u8(detail::sve_as_u8(state), z), detail::sve_as_u8(key)));
#else
  return detail::sve_crypto_soft_blocks(state, key, detail::aes_last_round_soft);
#endif
}
template <class V> HYLE_FORCE_INLINE V AESRoundInv(V state, V key) noexcept {
  static_assert(std::is_integral_v<TFromV<V>>, "AES op is integer-lane only");
#if defined(__ARM_FEATURE_SVE2_AES)
  const svuint8_t z = svdup_n_u8(0);
  const svuint8_t mc = svaesimc_u8(svaesd_u8(detail::sve_as_u8(state), z));
  return detail::sve_from_u8<V>(sveor_u8_x(svptrue_b8(), mc, detail::sve_as_u8(key)));
#else
  return detail::sve_crypto_soft_blocks(state, key, detail::aes_round_inv_soft);
#endif
}
template <class V> HYLE_FORCE_INLINE V AESLastRoundInv(V state, V key) noexcept {
  static_assert(std::is_integral_v<TFromV<V>>, "AES op is integer-lane only");
#if defined(__ARM_FEATURE_SVE2_AES)
  const svuint8_t z = svdup_n_u8(0);
  return detail::sve_from_u8<V>(
      sveor_u8_x(svptrue_b8(), svaesd_u8(detail::sve_as_u8(state), z), detail::sve_as_u8(key)));
#else
  return detail::sve_crypto_soft_blocks(state, key, detail::aes_last_round_inv_soft);
#endif
}
template <class V> HYLE_FORCE_INLINE V AESInvMixColumns(V state) noexcept {
  static_assert(std::is_integral_v<TFromV<V>>, "AES op is integer-lane only");
#if defined(__ARM_FEATURE_SVE2_AES)
  return detail::sve_from_u8<V>(svaesimc_u8(detail::sve_as_u8(state)));
#else
  const DFromV<V> d;
  using T = TFromV<V>;
  constexpr std::size_t kMaxBytes = DFromV<V>::kMaxLanes * sizeof(T);
  alignas(64) std::uint8_t s[kMaxBytes];
  Store(state, d, reinterpret_cast<T*>(s));
  const std::size_t bytes = Lanes(d) * sizeof(T);
  for (std::size_t b = 0; b + 16 <= bytes; b += 16) detail::aes_inv_mix_columns(s + b);
  return Load(d, reinterpret_cast<T*>(s));
#endif
}
// AESKeyGenAssist<kRcon>: no SVE2 instruction (like NEON -- ARM key schedule differs from x86's);
// software per-128-block core matches _mm_aeskeygenassist_si128 exactly. Not a hot path.
template <std::uint8_t kRcon, class V> HYLE_FORCE_INLINE V AESKeyGenAssist(V v) noexcept {
  static_assert(std::is_integral_v<TFromV<V>>, "AES op is integer-lane only");
  const DFromV<V> d;
  using T = TFromV<V>;
  constexpr std::size_t kMaxBytes = DFromV<V>::kMaxLanes * sizeof(T);
  alignas(64) std::uint8_t s[kMaxBytes];
  Store(v, d, reinterpret_cast<T*>(s));
  const std::size_t bytes = Lanes(d) * sizeof(T);
  for (std::size_t b = 0; b + 16 <= bytes; b += 16) detail::aes_keygen_assist_soft(s + b, kRcon);
  return Load(d, reinterpret_cast<T*>(s));
}
// CLMulLower/Upper: carryless multiply of the low/high 64-bit half of each 128-bit block.
// SVE2 svpmullb_pair (even elems, "bottom") / svpmullt_pair (odd elems, "top") over u64 give the
// per-128-block low/high CLMul. Software fallback loops clmul64 per block.
template <class V> HYLE_FORCE_INLINE V CLMulLower(V a, V b) noexcept {
  static_assert(std::is_integral_v<TFromV<V>>, "CLMul is integer-lane only");
#if defined(__ARM_FEATURE_SVE2_AES)
  const svuint64_t r = svpmullb_pair(svreinterpret_u64(a), svreinterpret_u64(b));
  return detail::sve_from_u8<V>(svreinterpret_u8(r));
#else
  const DFromV<V> d;
  using T = TFromV<V>;
  constexpr std::size_t kMaxBytes = DFromV<V>::kMaxLanes * sizeof(T);
  alignas(64) std::uint8_t sa[kMaxBytes], sb[kMaxBytes];
  Store(a, d, reinterpret_cast<T*>(sa));
  Store(b, d, reinterpret_cast<T*>(sb));
  const std::size_t bytes = Lanes(d) * sizeof(T);
  for (std::size_t bo = 0; bo + 16 <= bytes; bo += 16) {
    std::uint64_t av[2], bv[2], o[2];
    std::memcpy(av, sa + bo, 16); std::memcpy(bv, sb + bo, 16);
    detail::clmul64(av[0], bv[0], o);
    std::memcpy(sa + bo, o, 16);
  }
  return Load(d, reinterpret_cast<T*>(sa));
#endif
}
template <class V> HYLE_FORCE_INLINE V CLMulUpper(V a, V b) noexcept {
  static_assert(std::is_integral_v<TFromV<V>>, "CLMul is integer-lane only");
#if defined(__ARM_FEATURE_SVE2_AES)
  const svuint64_t r = svpmullt_pair(svreinterpret_u64(a), svreinterpret_u64(b));
  return detail::sve_from_u8<V>(svreinterpret_u8(r));
#else
  const DFromV<V> d;
  using T = TFromV<V>;
  constexpr std::size_t kMaxBytes = DFromV<V>::kMaxLanes * sizeof(T);
  alignas(64) std::uint8_t sa[kMaxBytes], sb[kMaxBytes];
  Store(a, d, reinterpret_cast<T*>(sa));
  Store(b, d, reinterpret_cast<T*>(sb));
  const std::size_t bytes = Lanes(d) * sizeof(T);
  for (std::size_t bo = 0; bo + 16 <= bytes; bo += 16) {
    std::uint64_t av[2], bv[2], o[2];
    std::memcpy(av, sa + bo, 16); std::memcpy(bv, sb + bo, 16);
    detail::clmul64(av[1], bv[1], o);
    std::memcpy(sa + bo, o, 16);
  }
  return Load(d, reinterpret_cast<T*>(sa));
#endif
}

// ---- mask <-> packed bits (BYTE-BUFFER form; the fixed-width backends return uint64 but a scalable
// VL can reach 2048-bit = 256 i8 lanes > 64, so the portable API is a byte buffer, 1 bit/lane, like
// Highway's SVE StoreMaskBits/LoadMaskBits). Implemented VL-agnostically via VecFromMask store +
// scalar bit-pack over the runtime Lanes(d) -- no dependence on a fixed VL. ----
// StoreMaskBits(d, m, bits): write ceil(Lanes/8) bytes, lane i -> bit i of byte i/8. Returns #bytes.
template <class D>
HYLE_FORCE_INLINE std::size_t StoreMaskBits(D d, svbool_t m, std::uint8_t* bits) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t kMax = D::kMaxLanes;
  alignas(64) T lanes[kMax];
  Store(VecFromMask(d, m), d, lanes);  // all-ones (T(-1)) where active, else 0
  const std::size_t n = Lanes(d);
  const std::size_t nbytes = (n + 7) / 8;
  for (std::size_t b = 0; b < nbytes; ++b) bits[b] = 0;
  for (std::size_t i = 0; i < n; ++i)
    if (lanes[i] != T{0}) bits[i >> 3] |= static_cast<std::uint8_t>(1u << (i & 7));
  return nbytes;
}
// LoadMaskBits(d, bits): inverse -- lane i active iff bit i of the byte buffer is set.
template <class D>
HYLE_FORCE_INLINE svbool_t LoadMaskBits(D d, const std::uint8_t* bits) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t kMax = D::kMaxLanes;
  alignas(64) T lanes[kMax];
  const std::size_t n = Lanes(d);
  for (std::size_t i = 0; i < n; ++i)
    lanes[i] = (bits[i >> 3] >> (i & 7)) & 1u ? static_cast<T>(~T{0}) : T{0};
  return MaskFromVec(Load(d, lanes));  // MSB-set lane -> active (all-ones qualifies)
}
// BitsFromMask(d, m): uint64 convenience for the common case Lanes(d) <= 64 (fits a word). Wider
// masks (i8 at VL>=512-bit) MUST use the byte-buffer StoreMaskBits -- guarded by assert.
template <class D>
HYLE_FORCE_INLINE std::uint64_t BitsFromMask(D d, svbool_t m) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t kMax = D::kMaxLanes;
  alignas(64) T lanes[kMax];
  Store(VecFromMask(d, m), d, lanes);
  const std::size_t n = Lanes(d);
  HYLE_FATAL_ASSERT(n <= 64);  // Lanes>64 (wide i8 at large VL) must use the StoreMaskBits byte buffer
  std::uint64_t bits = 0;
  for (std::size_t i = 0; i < n; ++i)
    if (lanes[i] != T{0}) bits |= (std::uint64_t{1} << i);
  return bits;
}

}  // namespace hyle::simd
