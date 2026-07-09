#pragma once

// backend/rvv.h -- RISC-V Vector (RVV 1.0) backend. The SECOND scalable-width backend (after SVE);
// it reuses the SAME scalable machinery vec.h grew for SVE (HYLE_SIMD_HAVE_SCALABLE=1 -> runtime
// Lanes, kNativeVectorBytes as an upper bound). RVV's vector length is a runtime property (VLEN,
// 128..65536 bit implementation-defined); the true element count comes from vsetvl at run time.
//
// Two RVV specifics beyond SVE's sizeless-type story (both handled entirely here):
//   1. SIZELESS types, same as SVE: vfloat32m1_t etc. cannot be struct/array members or sizeof'd.
//      -> VFromD<D> maps the lane type to the native LMUL=1 type via a traits table; DFromV<V>
//      reverses it via a specialization table. (Identical approach to sve.h.)
//   2. EXPLICIT vl: unlike SVE (implicit governing predicate), EVERY RVV intrinsic takes an
//      element-count argument `vl`. We compute vl once from the runtime Lanes(d) and thread it
//      through each op. Masks are vbool<RATIO>_t where RATIO = SEW/LMUL (f32,LMUL1 -> vbool32_t).
//
// FIRST-VERSION SCOPE: LMUL=1 only (kPow2=0). Fractional/grouped LMUL (m2/mf2...) is deferred (the
// same fractional-N encoding vec.h dropped for v1). This covers the load-bearing op face; matches
// the SVE landing.

#include <riscv_vector.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "hyle/macros.h"

namespace hyle::simd {

namespace detail {

// ---- lane type -> native LMUL=1 sizeless vector type --------------------------------
template <typename T> struct RvvVecT;
template <> struct RvvVecT<float> { using type = vfloat32m1_t; };
template <> struct RvvVecT<double> { using type = vfloat64m1_t; };
template <> struct RvvVecT<std::int8_t> { using type = vint8m1_t; };
template <> struct RvvVecT<std::uint8_t> { using type = vuint8m1_t; };
template <> struct RvvVecT<std::int16_t> { using type = vint16m1_t; };
template <> struct RvvVecT<std::uint16_t> { using type = vuint16m1_t; };
template <> struct RvvVecT<std::int32_t> { using type = vint32m1_t; };
template <> struct RvvVecT<std::uint32_t> { using type = vuint32m1_t; };
template <> struct RvvVecT<std::int64_t> { using type = vint64m1_t; };
template <> struct RvvVecT<std::uint64_t> { using type = vuint64m1_t; };

// ---- native vector type -> lane type (reverse map for DFromV) -----------------------
template <typename V> struct DFromV_rvv;
template <> struct DFromV_rvv<vfloat32m1_t> { using T = float; };
template <> struct DFromV_rvv<vfloat64m1_t> { using T = double; };
template <> struct DFromV_rvv<vint8m1_t> { using T = std::int8_t; };
template <> struct DFromV_rvv<vuint8m1_t> { using T = std::uint8_t; };
template <> struct DFromV_rvv<vint16m1_t> { using T = std::int16_t; };
template <> struct DFromV_rvv<vuint16m1_t> { using T = std::uint16_t; };
template <> struct DFromV_rvv<vint32m1_t> { using T = std::int32_t; };
template <> struct DFromV_rvv<vuint32m1_t> { using T = std::uint32_t; };
template <> struct DFromV_rvv<vint64m1_t> { using T = std::int64_t; };
template <> struct DFromV_rvv<vuint64m1_t> { using T = std::uint64_t; };

// vsetvlmax for a given lane type (LMUL=1). The maximum element count for the current VLEN.
template <typename T> HYLE_FORCE_INLINE std::size_t vlmax() noexcept {
  if constexpr (sizeof(T) == 1) return __riscv_vsetvlmax_e8m1();
  else if constexpr (sizeof(T) == 2) return __riscv_vsetvlmax_e16m1();
  else if constexpr (sizeof(T) == 4) return __riscv_vsetvlmax_e32m1();
  else return __riscv_vsetvlmax_e64m1();
}

}  // namespace detail

// ---- VFromD / DFromV / TFromV -------------------------------------------------------
template <class D>
using VFromD = typename detail::RvvVecT<TFromD<D>>::type;
template <class V>
using DFromV = Simd<typename detail::DFromV_rvv<V>::T,
                    kNativeVectorBytes / sizeof(typename detail::DFromV_rvv<V>::T), 0>;
template <class V>
using TFromV = typename detail::DFromV_rvv<V>::T;

// ---- Lanes(d): RUNTIME vector length (vsetvlmax at LMUL=1) ---------------------------
template <class D>
HYLE_FORCE_INLINE std::size_t Lanes(D) noexcept {
  return detail::vlmax<TFromD<D>>();
}

// ---- initialization -----------------------------------------------------------------
template <class D, typename T2>
HYLE_FORCE_INLINE VFromD<D> Set(D d, T2 t) noexcept {
  using T = TFromD<D>;
  const std::size_t vl = Lanes(d);
  if constexpr (std::is_same_v<T, float>) return __riscv_vfmv_v_f_f32m1(static_cast<float>(t), vl);
  else if constexpr (std::is_same_v<T, double>) return __riscv_vfmv_v_f_f64m1(static_cast<double>(t), vl);
  else if constexpr (std::is_same_v<T, std::int8_t>) return __riscv_vmv_v_x_i8m1(static_cast<std::int8_t>(t), vl);
  else if constexpr (std::is_same_v<T, std::uint8_t>) return __riscv_vmv_v_x_u8m1(static_cast<std::uint8_t>(t), vl);
  else if constexpr (std::is_same_v<T, std::int16_t>) return __riscv_vmv_v_x_i16m1(static_cast<std::int16_t>(t), vl);
  else if constexpr (std::is_same_v<T, std::uint16_t>) return __riscv_vmv_v_x_u16m1(static_cast<std::uint16_t>(t), vl);
  else if constexpr (std::is_same_v<T, std::int32_t>) return __riscv_vmv_v_x_i32m1(static_cast<std::int32_t>(t), vl);
  else if constexpr (std::is_same_v<T, std::uint32_t>) return __riscv_vmv_v_x_u32m1(static_cast<std::uint32_t>(t), vl);
  else if constexpr (std::is_same_v<T, std::int64_t>) return __riscv_vmv_v_x_i64m1(static_cast<std::int64_t>(t), vl);
  else return __riscv_vmv_v_x_u64m1(static_cast<std::uint64_t>(t), vl);
}

template <class D>
HYLE_FORCE_INLINE VFromD<D> Zero(D d) noexcept {
  return Set(d, static_cast<TFromD<D>>(0));
}
// NOTE: Undefined / Iota / SignBit etc. are defined GENERICALLY in vec.h's shared layer.

// ---- load / store (vle/vse at the runtime vl) ---------------------------------------
template <class D>
HYLE_FORCE_INLINE VFromD<D> LoadU(D d, const TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  const std::size_t vl = Lanes(d);
  if constexpr (std::is_same_v<T, float>) return __riscv_vle32_v_f32m1(p, vl);
  else if constexpr (std::is_same_v<T, double>) return __riscv_vle64_v_f64m1(p, vl);
  else if constexpr (sizeof(T) == 1 && std::is_signed_v<T>) return __riscv_vle8_v_i8m1(reinterpret_cast<const std::int8_t*>(p), vl);
  else if constexpr (sizeof(T) == 1) return __riscv_vle8_v_u8m1(reinterpret_cast<const std::uint8_t*>(p), vl);
  else if constexpr (sizeof(T) == 2 && std::is_signed_v<T>) return __riscv_vle16_v_i16m1(reinterpret_cast<const std::int16_t*>(p), vl);
  else if constexpr (sizeof(T) == 2) return __riscv_vle16_v_u16m1(reinterpret_cast<const std::uint16_t*>(p), vl);
  else if constexpr (sizeof(T) == 4 && std::is_signed_v<T>) return __riscv_vle32_v_i32m1(reinterpret_cast<const std::int32_t*>(p), vl);
  else if constexpr (sizeof(T) == 4) return __riscv_vle32_v_u32m1(reinterpret_cast<const std::uint32_t*>(p), vl);
  else if constexpr (std::is_signed_v<T>) return __riscv_vle64_v_i64m1(reinterpret_cast<const std::int64_t*>(p), vl);
  else return __riscv_vle64_v_u64m1(reinterpret_cast<const std::uint64_t*>(p), vl);
}
template <class D>
HYLE_FORCE_INLINE VFromD<D> Load(D d, const TFromD<D>* p) noexcept { return LoadU(d, p); }

template <class D>
HYLE_FORCE_INLINE void StoreU(VFromD<D> v, D d, TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  const std::size_t vl = Lanes(d);
  if constexpr (std::is_same_v<T, float>) __riscv_vse32_v_f32m1(p, v, vl);
  else if constexpr (std::is_same_v<T, double>) __riscv_vse64_v_f64m1(p, v, vl);
  else if constexpr (sizeof(T) == 1 && std::is_signed_v<T>) __riscv_vse8_v_i8m1(reinterpret_cast<std::int8_t*>(p), v, vl);
  else if constexpr (sizeof(T) == 1) __riscv_vse8_v_u8m1(reinterpret_cast<std::uint8_t*>(p), v, vl);
  else if constexpr (sizeof(T) == 2 && std::is_signed_v<T>) __riscv_vse16_v_i16m1(reinterpret_cast<std::int16_t*>(p), v, vl);
  else if constexpr (sizeof(T) == 2) __riscv_vse16_v_u16m1(reinterpret_cast<std::uint16_t*>(p), v, vl);
  else if constexpr (sizeof(T) == 4 && std::is_signed_v<T>) __riscv_vse32_v_i32m1(reinterpret_cast<std::int32_t*>(p), v, vl);
  else if constexpr (sizeof(T) == 4) __riscv_vse32_v_u32m1(reinterpret_cast<std::uint32_t*>(p), v, vl);
  else if constexpr (std::is_signed_v<T>) __riscv_vse64_v_i64m1(reinterpret_cast<std::int64_t*>(p), v, vl);
  else __riscv_vse64_v_u64m1(reinterpret_cast<std::uint64_t*>(p), v, vl);
}
template <class D>
HYLE_FORCE_INLINE void Store(VFromD<D> v, D d, TFromD<D>* p) noexcept { StoreU(v, d, p); }

// ---- arithmetic ---------------------------------------------------------------------
// RVV intrinsics are heavily type-suffixed AND take an explicit vl. To avoid a 10-way if-constexpr
// per op, a macro generates the per-lane-type dispatch; each op computes vl once from Lanes(d).
// (T-generic overloads exist for the arithmetic verbs, so the suffix-free __riscv_vadd etc. work
// across int lane types; float uses vfadd. We branch only int-vs-float where the mnemonic differs.)
#define HYLE_RVV_BIN_INT(NAME, IOP)                                                  \
  template <class V> HYLE_FORCE_INLINE V NAME(V a, V b) noexcept {                   \
    const std::size_t vl = detail::vlmax<TFromV<V>>();                               \
    return IOP(a, b, vl);                                                            \
  }
// Add/Sub/Mul/Min/Max: int + float share a generic verb via the type-generic intrinsics.
template <class V> HYLE_FORCE_INLINE V Add(V a, V b) noexcept {
  const std::size_t vl = detail::vlmax<TFromV<V>>();
  if constexpr (std::is_floating_point_v<TFromV<V>>) return __riscv_vfadd(a, b, vl);
  else return __riscv_vadd(a, b, vl);
}
template <class V> HYLE_FORCE_INLINE V Sub(V a, V b) noexcept {
  const std::size_t vl = detail::vlmax<TFromV<V>>();
  if constexpr (std::is_floating_point_v<TFromV<V>>) return __riscv_vfsub(a, b, vl);
  else return __riscv_vsub(a, b, vl);
}
template <class V> HYLE_FORCE_INLINE V Mul(V a, V b) noexcept {
  const std::size_t vl = detail::vlmax<TFromV<V>>();
  if constexpr (std::is_floating_point_v<TFromV<V>>) return __riscv_vfmul(a, b, vl);
  else return __riscv_vmul(a, b, vl);
}
template <class V> HYLE_FORCE_INLINE V Min(V a, V b) noexcept {
  using T = TFromV<V>;
  const std::size_t vl = detail::vlmax<T>();
  if constexpr (std::is_floating_point_v<T>) return __riscv_vfmin(a, b, vl);
  else if constexpr (std::is_signed_v<T>) return __riscv_vmin(a, b, vl);
  else return __riscv_vminu(a, b, vl);  // unsigned uses the 'u' mnemonic
}
template <class V> HYLE_FORCE_INLINE V Max(V a, V b) noexcept {
  using T = TFromV<V>;
  const std::size_t vl = detail::vlmax<T>();
  if constexpr (std::is_floating_point_v<T>) return __riscv_vfmax(a, b, vl);
  else if constexpr (std::is_signed_v<T>) return __riscv_vmax(a, b, vl);
  else return __riscv_vmaxu(a, b, vl);
}
template <class V> HYLE_FORCE_INLINE V Div(V a, V b) noexcept {  // float only
  return __riscv_vfdiv(a, b, detail::vlmax<TFromV<V>>());
}
template <class V> HYLE_FORCE_INLINE V Sqrt(V a) noexcept {
  return __riscv_vfsqrt(a, detail::vlmax<TFromV<V>>());
}
template <class V> HYLE_FORCE_INLINE V Neg(V a) noexcept {
  const std::size_t vl = detail::vlmax<TFromV<V>>();
  if constexpr (std::is_floating_point_v<TFromV<V>>) return __riscv_vfneg(a, vl);
  else return __riscv_vneg(a, vl);
}
// MulAdd(a,b,c)=a*b+c ; NegMulAdd(a,b,c)=c-a*b. vfmacc(acc,a,b)=acc+a*b (float); ints via mul+add.
template <class V> HYLE_FORCE_INLINE V MulAdd(V a, V b, V c) noexcept {
  const std::size_t vl = detail::vlmax<TFromV<V>>();
  if constexpr (std::is_floating_point_v<TFromV<V>>) return __riscv_vfmadd(a, b, c, vl);  // a*b+c
  else return __riscv_vmadd(a, b, c, vl);
}
template <class V> HYLE_FORCE_INLINE V NegMulAdd(V a, V b, V c) noexcept {
  const std::size_t vl = detail::vlmax<TFromV<V>>();
  if constexpr (std::is_floating_point_v<TFromV<V>>) return __riscv_vfnmsac(c, a, b, vl);  // c-a*b
  else return __riscv_vnmsac(c, a, b, vl);
}

// ---- bitwise (integer) --------------------------------------------------------------
template <class V> HYLE_FORCE_INLINE V And(V a, V b) noexcept { return __riscv_vand(a, b, detail::vlmax<TFromV<V>>()); }
template <class V> HYLE_FORCE_INLINE V Or(V a, V b) noexcept { return __riscv_vor(a, b, detail::vlmax<TFromV<V>>()); }
template <class V> HYLE_FORCE_INLINE V Xor(V a, V b) noexcept { return __riscv_vxor(a, b, detail::vlmax<TFromV<V>>()); }
template <class V> HYLE_FORCE_INLINE V Not(V a) noexcept { return __riscv_vnot(a, detail::vlmax<TFromV<V>>()); }
template <class V> HYLE_FORCE_INLINE V AndNot(V a, V b) noexcept {  // ~a & b (Highway order)
  const std::size_t vl = detail::vlmax<TFromV<V>>();
  return __riscv_vand(__riscv_vnot(a, vl), b, vl);
}

// ---- comparisons -> vbool mask ------------------------------------------------------
// RVV masks are vbool<ratio>_t. The type-generic vmseq/vmslt/... deduce the mask type from the
// operand vector type, so we can stay verb-generic.
template <class V> HYLE_FORCE_INLINE auto Eq(V a, V b) noexcept {
  const std::size_t vl = detail::vlmax<TFromV<V>>();
  if constexpr (std::is_floating_point_v<TFromV<V>>) return __riscv_vmfeq(a, b, vl);
  else return __riscv_vmseq(a, b, vl);
}
template <class V> HYLE_FORCE_INLINE auto Ne(V a, V b) noexcept {
  const std::size_t vl = detail::vlmax<TFromV<V>>();
  if constexpr (std::is_floating_point_v<TFromV<V>>) return __riscv_vmfne(a, b, vl);
  else return __riscv_vmsne(a, b, vl);
}
template <class V> HYLE_FORCE_INLINE auto Lt(V a, V b) noexcept {
  using T = TFromV<V>;
  const std::size_t vl = detail::vlmax<T>();
  if constexpr (std::is_floating_point_v<T>) return __riscv_vmflt(a, b, vl);
  else if constexpr (std::is_signed_v<T>) return __riscv_vmslt(a, b, vl);
  else return __riscv_vmsltu(a, b, vl);
}
template <class V> HYLE_FORCE_INLINE auto Gt(V a, V b) noexcept {
  using T = TFromV<V>;
  const std::size_t vl = detail::vlmax<T>();
  if constexpr (std::is_floating_point_v<T>) return __riscv_vmfgt(a, b, vl);
  else if constexpr (std::is_signed_v<T>) return __riscv_vmsgt(a, b, vl);
  else return __riscv_vmsgtu(a, b, vl);
}
template <class V> HYLE_FORCE_INLINE auto Le(V a, V b) noexcept {
  using T = TFromV<V>;
  const std::size_t vl = detail::vlmax<T>();
  if constexpr (std::is_floating_point_v<T>) return __riscv_vmfle(a, b, vl);
  else if constexpr (std::is_signed_v<T>) return __riscv_vmsle(a, b, vl);
  else return __riscv_vmsleu(a, b, vl);
}
template <class V> HYLE_FORCE_INLINE auto Ge(V a, V b) noexcept {
  using T = TFromV<V>;
  const std::size_t vl = detail::vlmax<T>();
  if constexpr (std::is_floating_point_v<T>) return __riscv_vmfge(a, b, vl);
  else if constexpr (std::is_signed_v<T>) return __riscv_vmsge(a, b, vl);
  else return __riscv_vmsgeu(a, b, vl);
}

// ---- select -------------------------------------------------------------------------
// IfThenElse(m, yes, no): vmerge picks yes where m, else no.
template <class M, class V> HYLE_FORCE_INLINE V IfThenElse(M m, V yes, V no) noexcept {
  return __riscv_vmerge(no, yes, m, detail::vlmax<TFromV<V>>());
}

// ---- mask reductions ----------------------------------------------------------------
template <class D, class M> HYLE_FORCE_INLINE bool AllTrue(D d, M m) noexcept {
  const std::size_t vl = Lanes(d);
  return __riscv_vcpop(m, vl) == static_cast<long>(vl);
}
template <class D, class M> HYLE_FORCE_INLINE bool AllFalse(D d, M m) noexcept {
  return __riscv_vcpop(m, Lanes(d)) == 0;
}
template <class D, class M> HYLE_FORCE_INLINE std::size_t CountTrue(D d, M m) noexcept {
  return static_cast<std::size_t>(__riscv_vcpop(m, Lanes(d)));
}
// FindFirstTrue: index of first active lane or -1 (RVV vfirst does exactly this). FindLastTrue: no
// direct primitive -> select each active lane's index into an iota, inactive lanes get -1, ReduceMax.
template <class D, class M> HYLE_FORCE_INLINE std::ptrdiff_t FindFirstTrue(D d, M m) noexcept {
  return static_cast<std::ptrdiff_t>(__riscv_vfirst(m, Lanes(d)));
}
template <class D, class M> HYLE_FORCE_INLINE std::ptrdiff_t FindLastTrue(D d, M m) noexcept {
  using T = TFromD<D>;
  const std::size_t vl = Lanes(d);
  if (__riscv_vfirst(m, vl) < 0) return -1;  // no active lane
  // signed same-width iota; inactive lanes -> -1 via masked merge; ReduceMax picks the last index.
  if constexpr (sizeof(T) == 1) { auto id = __riscv_vreinterpret_i8m1(__riscv_vid_v_u8m1(vl)); auto sel = __riscv_vmerge(__riscv_vmv_v_x_i8m1(-1, vl), id, m, vl); return (std::ptrdiff_t)__riscv_vmv_x(__riscv_vredmax(sel, sel, vl)); }
  else if constexpr (sizeof(T) == 2) { auto id = __riscv_vreinterpret_i16m1(__riscv_vid_v_u16m1(vl)); auto sel = __riscv_vmerge(__riscv_vmv_v_x_i16m1(-1, vl), id, m, vl); return (std::ptrdiff_t)__riscv_vmv_x(__riscv_vredmax(sel, sel, vl)); }
  else if constexpr (sizeof(T) == 4) { auto id = __riscv_vreinterpret_i32m1(__riscv_vid_v_u32m1(vl)); auto sel = __riscv_vmerge(__riscv_vmv_v_x_i32m1(-1, vl), id, m, vl); return (std::ptrdiff_t)__riscv_vmv_x(__riscv_vredmax(sel, sel, vl)); }
  else { auto id = __riscv_vreinterpret_i64m1(__riscv_vid_v_u64m1(vl)); auto sel = __riscv_vmerge(__riscv_vmv_v_x_i64m1(-1, vl), id, m, vl); return (std::ptrdiff_t)__riscv_vmv_x(__riscv_vredmax(sel, sel, vl)); }
}

// ---- GetLane: lane 0 ----------------------------------------------------------------
template <class V> HYLE_FORCE_INLINE TFromV<V> GetLane(V v) noexcept {
  if constexpr (std::is_floating_point_v<TFromV<V>>) return __riscv_vfmv_f(v);
  else return __riscv_vmv_x(v);
}

// ---- value reductions ---------------------------------------------------------------
// RVV reductions take a scalar-init vector and return a vector whose lane 0 is the result.
template <class D> HYLE_FORCE_INLINE TFromD<D> ReduceSum(D d, VFromD<D> v) noexcept {
  const std::size_t vl = Lanes(d);
  auto z = Zero(d);
  if constexpr (std::is_floating_point_v<TFromD<D>>) return __riscv_vfmv_f(__riscv_vfredusum(v, z, vl));
  else return __riscv_vmv_x(__riscv_vredsum(v, z, vl));
}
template <class D> HYLE_FORCE_INLINE TFromD<D> ReduceMin(D d, VFromD<D> v) noexcept {
  const std::size_t vl = Lanes(d);
  if constexpr (std::is_floating_point_v<TFromD<D>>) return __riscv_vfmv_f(__riscv_vfredmin(v, v, vl));
  else return __riscv_vmv_x(__riscv_vredmin(v, v, vl));
}
template <class D> HYLE_FORCE_INLINE TFromD<D> ReduceMax(D d, VFromD<D> v) noexcept {
  const std::size_t vl = Lanes(d);
  if constexpr (std::is_floating_point_v<TFromD<D>>) return __riscv_vfmv_f(__riscv_vfredmax(v, v, vl));
  else return __riscv_vmv_x(__riscv_vredmax(v, v, vl));
}
// ReduceAnd/ReduceOr (integer lanes): vredand/vredor with an identity-seeded scalar vector.
template <class D> HYLE_FORCE_INLINE TFromD<D> ReduceAnd(D d, VFromD<D> v) noexcept {
  const std::size_t vl = Lanes(d);
  return __riscv_vmv_x(__riscv_vredand(v, Set(d, static_cast<TFromD<D>>(~TFromD<D>{0})), vl));  // seed all-ones
}
template <class D> HYLE_FORCE_INLINE TFromD<D> ReduceOr(D d, VFromD<D> v) noexcept {
  const std::size_t vl = Lanes(d);
  return __riscv_vmv_x(__riscv_vredor(v, Zero(d), vl));  // seed 0
}

// ---- shifts -------------------------------------------------------------------------
// Immediate: splat the count. Variable: shift by b's lanes (as unsigned same-width). Signed
// right shift = arithmetic (vsra), unsigned = logical (vsrl).
template <int kBits, class V> HYLE_FORCE_INLINE V ShiftLeft(V a) noexcept {
  return __riscv_vsll(a, static_cast<std::size_t>(kBits), detail::vlmax<TFromV<V>>());
}
template <int kBits, class V> HYLE_FORCE_INLINE V ShiftRight(V a) noexcept {
  const std::size_t vl = detail::vlmax<TFromV<V>>();
  if constexpr (std::is_signed_v<TFromV<V>>) return __riscv_vsra(a, static_cast<std::size_t>(kBits), vl);
  else return __riscv_vsrl(a, static_cast<std::size_t>(kBits), vl);
}
// Variable per-lane shift: Shl(a,b)=a<<b, Shr(a,b)=a>>b (arithmetic if signed). RVV's vsll/vsra/vsrl
// take the shift-amount as an UNSIGNED same-width vector; reinterpret b's bits to unsigned.
namespace detail {
template <class V> HYLE_FORCE_INLINE auto to_ucount_rvv(V b, std::size_t vl) noexcept {
  using T = TFromV<V>;
  if constexpr (std::is_signed_v<T>) {
    if constexpr (sizeof(T) == 1) return __riscv_vreinterpret_u8m1(b);
    else if constexpr (sizeof(T) == 2) return __riscv_vreinterpret_u16m1(b);
    else if constexpr (sizeof(T) == 4) return __riscv_vreinterpret_u32m1(b);
    else return __riscv_vreinterpret_u64m1(b);
  } else { (void)vl; return b; }
}
}  // namespace detail
template <class V> HYLE_FORCE_INLINE V Shl(V a, V b) noexcept {
  const std::size_t vl = detail::vlmax<TFromV<V>>();
  return __riscv_vsll(a, detail::to_ucount_rvv(b, vl), vl);
}
template <class V> HYLE_FORCE_INLINE V Shr(V a, V b) noexcept {
  const std::size_t vl = detail::vlmax<TFromV<V>>();
  if constexpr (std::is_signed_v<TFromV<V>>) return __riscv_vsra(a, detail::to_ucount_rvv(b, vl), vl);
  else return __riscv_vsrl(a, detail::to_ucount_rvv(b, vl), vl);
}

// ---- Neg / Abs ----------------------------------------------------------------------
template <class V> HYLE_FORCE_INLINE V Abs(V a) noexcept {
  const std::size_t vl = detail::vlmax<TFromV<V>>();
  if constexpr (std::is_floating_point_v<TFromV<V>>) return __riscv_vfabs(a, vl);
  else return __riscv_vmax(a, __riscv_vneg(a, vl), vl);  // int abs = max(a,-a)
}

// ---- saturating arithmetic (integer; unsigned uses the 'u' mnemonic) ----------------
template <class V> HYLE_FORCE_INLINE V SaturatedAdd(V a, V b) noexcept {
  const std::size_t vl = detail::vlmax<TFromV<V>>();
  if constexpr (std::is_signed_v<TFromV<V>>) return __riscv_vsadd(a, b, vl);
  else return __riscv_vsaddu(a, b, vl);
}
template <class V> HYLE_FORCE_INLINE V SaturatedSub(V a, V b) noexcept {
  const std::size_t vl = detail::vlmax<TFromV<V>>();
  if constexpr (std::is_signed_v<TFromV<V>>) return __riscv_vssub(a, b, vl);
  else return __riscv_vssubu(a, b, vl);
}
template <class V> HYLE_FORCE_INLINE V MulHigh(V a, V b) noexcept {  // high half of widening mul
  const std::size_t vl = detail::vlmax<TFromV<V>>();
  if constexpr (std::is_signed_v<TFromV<V>>) return __riscv_vmulh(a, b, vl);
  else return __riscv_vmulhu(a, b, vl);
}
template <class V> HYLE_FORCE_INLINE V AbsDiff(V a, V b) noexcept {
  return Sub(Max(a, b), Min(a, b));  // |a-b| = max-min (works for both signs in range)
}
template <class V> HYLE_FORCE_INLINE V Clamp(V v, V lo, V hi) noexcept {
  return Min(Max(v, lo), hi);
}

// ---- BitCast ------------------------------------------------------------------------
template <class D, class V>
HYLE_FORCE_INLINE VFromD<D> BitCast(D, V v) noexcept {
  using T = TFromD<D>;
  if constexpr (std::is_same_v<T, float>) return __riscv_vreinterpret_f32m1(v);
  else if constexpr (std::is_same_v<T, double>) return __riscv_vreinterpret_f64m1(v);
  else if constexpr (std::is_same_v<T, std::int8_t>) return __riscv_vreinterpret_i8m1(v);
  else if constexpr (std::is_same_v<T, std::uint8_t>) return __riscv_vreinterpret_u8m1(v);
  else if constexpr (std::is_same_v<T, std::int16_t>) return __riscv_vreinterpret_i16m1(v);
  else if constexpr (std::is_same_v<T, std::uint16_t>) return __riscv_vreinterpret_u16m1(v);
  else if constexpr (std::is_same_v<T, std::int32_t>) return __riscv_vreinterpret_i32m1(v);
  else if constexpr (std::is_same_v<T, std::uint32_t>) return __riscv_vreinterpret_u32m1(v);
  else if constexpr (std::is_same_v<T, std::int64_t>) return __riscv_vreinterpret_i64m1(v);
  else return __riscv_vreinterpret_u64m1(v);
}

// ---- same-width conversions (f32<->i32, f64<->i64) ----------------------------------
template <typename DstT, std::size_t N, class V>
HYLE_FORCE_INLINE VFromD<Simd<DstT, N, 0>> ConvertTo(Simd<DstT, N, 0> dd, V v) noexcept {
  using SrcT = TFromV<V>;
  const std::size_t vl = Lanes(dd);
  if constexpr (std::is_same_v<SrcT, float> && std::is_same_v<DstT, std::int32_t>)
    return __riscv_vfcvt_rtz_x_f_v_i32m1(v, vl);  // truncate toward zero
  else if constexpr (std::is_same_v<SrcT, float> && std::is_same_v<DstT, std::uint32_t>)
    return __riscv_vfcvt_rtz_xu_f_v_u32m1(v, vl);
  else if constexpr (std::is_same_v<SrcT, std::int32_t> && std::is_same_v<DstT, float>)
    return __riscv_vfcvt_f_x_v_f32m1(v, vl);
  else if constexpr (std::is_same_v<SrcT, std::uint32_t> && std::is_same_v<DstT, float>)
    return __riscv_vfcvt_f_xu_v_f32m1(v, vl);
  else if constexpr (std::is_same_v<SrcT, double> && std::is_same_v<DstT, std::int64_t>)
    return __riscv_vfcvt_rtz_x_f_v_i64m1(v, vl);
  else  // int64 -> double
    return __riscv_vfcvt_f_x_v_f64m1(v, vl);
}
// RoundToInt: float->int, round-to-nearest-even (vfcvt with default rounding mode = RNE).
template <typename DstT, std::size_t N, class V>
HYLE_FORCE_INLINE VFromD<Simd<DstT, N, 0>> RoundToInt(Simd<DstT, N, 0> dd, V v) noexcept {
  using SrcT = TFromV<V>;
  const std::size_t vl = Lanes(dd);
  if constexpr (std::is_same_v<SrcT, float>) return __riscv_vfcvt_x_f_v_i32m1(v, vl);
  else return __riscv_vfcvt_x_f_v_i64m1(v, vl);
}
// Round/Floor/Ceil/Trunc: float->float rounding to an integral value (result stays float). RVV base
// V has no vrint; use round-mode fcvt (vfcvt_x_f_..._rm) to int then vfcvt_f back. Values with
// |v| >= 2^(mantissa-bits) are ALREADY integral and would overflow the int convert, so keep the
// original there (select on |v| < 2^23 f32 / 2^52 f64). NaN/inf are >= the threshold -> passthrough.
namespace detail {
template <int kFrm, class V> HYLE_FORCE_INLINE V rvv_round(V v) noexcept {
  using T = TFromV<V>;
  const std::size_t vl = detail::vlmax<T>();
  if constexpr (std::is_same_v<T, float>) {
    const vint32m1_t i = __riscv_vfcvt_x_f_v_i32m1_rm(v, kFrm, vl);
    const vfloat32m1_t r = __riscv_vfcvt_f_x_v_f32m1(i, vl);
    const auto small = __riscv_vmflt(__riscv_vfabs(v, vl), 8388608.0f, vl);  // |v| < 2^23
    return __riscv_vmerge(v, r, small, vl);
  } else {
    const vint64m1_t i = __riscv_vfcvt_x_f_v_i64m1_rm(v, kFrm, vl);
    const vfloat64m1_t r = __riscv_vfcvt_f_x_v_f64m1(i, vl);
    const auto small = __riscv_vmflt(__riscv_vfabs(v, vl), 4503599627370496.0, vl);  // |v| < 2^52
    return __riscv_vmerge(v, r, small, vl);
  }
}
}  // namespace detail
template <class V> HYLE_FORCE_INLINE V Round(V v) noexcept { return detail::rvv_round<__RISCV_FRM_RNE>(v); }
template <class V> HYLE_FORCE_INLINE V Floor(V v) noexcept { return detail::rvv_round<__RISCV_FRM_RDN>(v); }
template <class V> HYLE_FORCE_INLINE V Ceil(V v) noexcept { return detail::rvv_round<__RISCV_FRM_RUP>(v); }
template <class V> HYLE_FORCE_INLINE V Trunc(V v) noexcept { return detail::rvv_round<__RISCV_FRM_RTZ>(v); }

// ---- width-changing convert (Promote/Demote), 2x integer, LMUL=1 -----------------------------
// RVV widening produces LMUL*2, but our type system is LMUL=1. PromoteLowerTo: view the low mf2
// portion (vlmul_trunc) then vsext/vzext_vf2 to a full m1 of double width. PromoteUpperTo: slide
// the upper half down first. DemoteTo: pack the two m1 sources into an m2 (vlmul_ext + vset), then
// vnclip (saturating narrow). Integer only (float f32<->f64 promote/demote deferred, not the
// quantization hot path). 2x steps only (i8<->i16, i16<->i32, i32<->i64).
template <typename DstT, std::size_t NDst, class V>
HYLE_FORCE_INLINE VFromD<Simd<DstT, NDst, 0>> PromoteLowerTo(Simd<DstT, NDst, 0> dd, V v) noexcept {
  using SrcT = TFromV<V>;
  static_assert(sizeof(DstT) == 2 * sizeof(SrcT) && std::is_integral_v<DstT>, "PromoteLowerTo: 2x integer widen");
  const std::size_t vl = Lanes(dd);
  if constexpr (std::is_signed_v<DstT>) {
    if constexpr (sizeof(SrcT) == 1) return __riscv_vsext_vf2_i16m1(__riscv_vlmul_trunc_v_i8m1_i8mf2(v), vl);
    else if constexpr (sizeof(SrcT) == 2) return __riscv_vsext_vf2_i32m1(__riscv_vlmul_trunc_v_i16m1_i16mf2(v), vl);
    else return __riscv_vsext_vf2_i64m1(__riscv_vlmul_trunc_v_i32m1_i32mf2(v), vl);
  } else {
    if constexpr (sizeof(SrcT) == 1) return __riscv_vzext_vf2_u16m1(__riscv_vlmul_trunc_v_u8m1_u8mf2(v), vl);
    else if constexpr (sizeof(SrcT) == 2) return __riscv_vzext_vf2_u32m1(__riscv_vlmul_trunc_v_u16m1_u16mf2(v), vl);
    else return __riscv_vzext_vf2_u64m1(__riscv_vlmul_trunc_v_u32m1_u32mf2(v), vl);
  }
}
template <typename DstT, std::size_t NDst, class V>
HYLE_FORCE_INLINE VFromD<Simd<DstT, NDst, 0>> PromoteUpperTo(Simd<DstT, NDst, 0> dd, V v) noexcept {
  using SrcT = TFromV<V>;
  static_assert(sizeof(DstT) == 2 * sizeof(SrcT) && std::is_integral_v<DstT>, "PromoteUpperTo: 2x integer widen");
  const std::size_t vlsrc = detail::vlmax<SrcT>();
  const V up = __riscv_vslidedown(v, vlsrc / 2, vlsrc);  // bring the upper half to the front
  return PromoteLowerTo(dd, up);
}
template <typename DstT, std::size_t NDst, class V>
HYLE_FORCE_INLINE VFromD<Simd<DstT, NDst, 0>> DemoteTo(Simd<DstT, NDst, 0> dd, V a, V b) noexcept {
  using SrcT = TFromV<V>;
  static_assert(sizeof(DstT) * 2 == sizeof(SrcT) && std::is_integral_v<DstT>, "DemoteTo: 2x integer narrow");
  const std::size_t vlsrc = detail::vlmax<SrcT>();
  const std::size_t vldst = Lanes(dd);
  if constexpr (std::is_signed_v<DstT>) {
    if constexpr (sizeof(SrcT) == 2) { auto m2 = __riscv_vset_v_i16m1_i16m2(__riscv_vlmul_ext_v_i16m1_i16m2(a), 1, b); return __riscv_vnclip_wx_i8m1(m2, 0, __RISCV_VXRM_RNU, vldst); (void)vlsrc; }
    else if constexpr (sizeof(SrcT) == 4) { auto m2 = __riscv_vset_v_i32m1_i32m2(__riscv_vlmul_ext_v_i32m1_i32m2(a), 1, b); return __riscv_vnclip_wx_i16m1(m2, 0, __RISCV_VXRM_RNU, vldst); }
    else { auto m2 = __riscv_vset_v_i64m1_i64m2(__riscv_vlmul_ext_v_i64m1_i64m2(a), 1, b); return __riscv_vnclip_wx_i32m1(m2, 0, __RISCV_VXRM_RNU, vldst); }
  } else {
    if constexpr (sizeof(SrcT) == 2) { auto m2 = __riscv_vset_v_u16m1_u16m2(__riscv_vlmul_ext_v_u16m1_u16m2(a), 1, b); return __riscv_vnclipu_wx_u8m1(m2, 0, __RISCV_VXRM_RNU, vldst); }
    else if constexpr (sizeof(SrcT) == 4) { auto m2 = __riscv_vset_v_u32m1_u32m2(__riscv_vlmul_ext_v_u32m1_u32m2(a), 1, b); return __riscv_vnclipu_wx_u16m1(m2, 0, __RISCV_VXRM_RNU, vldst); }
    else { auto m2 = __riscv_vset_v_u64m1_u64m2(__riscv_vlmul_ext_v_u64m1_u64m2(a), 1, b); return __riscv_vnclipu_wx_u32m1(m2, 0, __RISCV_VXRM_RNU, vldst); }
  }
}

// ---- FP classification --------------------------------------------------------------
template <class V> HYLE_FORCE_INLINE auto IsNaN(V v) noexcept {
  return __riscv_vmfne(v, v, detail::vlmax<TFromV<V>>());  // NaN != NaN
}
template <class V> HYLE_FORCE_INLINE auto IsInf(V v) noexcept {
  using T = TFromV<V>;
  return Eq(Abs(v), Set(DFromV<V>(), std::numeric_limits<T>::infinity()));
}
template <class V> HYLE_FORCE_INLINE auto IsFinite(V v) noexcept {
  using T = TFromV<V>;
  return Lt(Abs(v), Set(DFromV<V>(), std::numeric_limits<T>::infinity()));  // NaN also fails
}

// ---- IEEE Max/MinNumber (NaN-quieting) ----------------------------------------------
template <class V> HYLE_FORCE_INLINE V MaxNumber(V a, V b) noexcept {
  return __riscv_vfmax(a, b, detail::vlmax<TFromV<V>>());  // vfmax already NaN-quieting per IEEE
}
template <class V> HYLE_FORCE_INLINE V MinNumber(V a, V b) noexcept {
  return __riscv_vfmin(a, b, detail::vlmax<TFromV<V>>());
}

// ---- CopySign (float) ---------------------------------------------------------------
template <class V> HYLE_FORCE_INLINE V CopySign(V mag, V sign) noexcept {
  return __riscv_vfsgnj(mag, sign, detail::vlmax<TFromV<V>>());  // sign-inject: |mag| with sign's sign
}

// ---- mask <-> vec + sign-predicated select ------------------------------------------
// MaskFromVec: active where the sign bit is set (Highway MSB contract). Test < 0 on signed view.
template <class V> HYLE_FORCE_INLINE auto MaskFromVec(V v) noexcept {
  return Lt(v, Zero(DFromV<V>()));
}
template <class D, class M> HYLE_FORCE_INLINE VFromD<D> VecFromMask(D d, M m) noexcept {
  using T = TFromD<D>;
  return IfThenElse(m, Set(d, static_cast<T>(-1)), Zero(d));
}
template <class M, class V> HYLE_FORCE_INLINE V IfThenElseZero(M m, V yes) noexcept {
  return IfThenElse(m, yes, Zero(DFromV<V>()));
}
template <class M, class V> HYLE_FORCE_INLINE V IfThenZeroElse(M m, V no) noexcept {
  return IfThenElse(m, Zero(DFromV<V>()), no);
}
template <class V> HYLE_FORCE_INLINE V IfNegativeThenElse(V v, V yes, V no) noexcept {
  return IfThenElse(MaskFromVec(v), yes, no);
}
template <class V> HYLE_FORCE_INLINE V ZeroIfNegative(V v) noexcept {
  return IfThenElse(MaskFromVec(v), Zero(DFromV<V>()), v);
}

// ---- Reverse (whole-vector) via vrgather with (vl-1 - iota) indices ------------------
template <class D> HYLE_FORCE_INLINE VFromD<D> Reverse(D d, VFromD<D> v) noexcept {
  using T = TFromD<D>;
  const std::size_t vl = Lanes(d);
  // build reversed index vector in the unsigned same-width lane type
  if constexpr (sizeof(T) == 1) { auto id = __riscv_vid_v_u8m1(vl); return __riscv_vrgather(v, __riscv_vrsub(id, static_cast<std::uint8_t>(vl - 1), vl), vl); }
  else if constexpr (sizeof(T) == 2) { auto id = __riscv_vid_v_u16m1(vl); return __riscv_vrgather(v, __riscv_vrsub(id, static_cast<std::uint16_t>(vl - 1), vl), vl); }
  else if constexpr (sizeof(T) == 4) { auto id = __riscv_vid_v_u32m1(vl); return __riscv_vrgather(v, __riscv_vrsub(id, static_cast<std::uint32_t>(vl - 1), vl), vl); }
  else { auto id = __riscv_vid_v_u64m1(vl); return __riscv_vrgather(v, __riscv_vrsub(id, static_cast<std::uint64_t>(vl - 1), vl), vl); }
}

// ---- tail: FirstN / LoadN / StoreN (RVV NATIVE via vsetvl(n)-predicated ops) ---------
// LoadN/StoreN set the active vl to n, so only n elements are touched (fault-safe). FirstN builds
// a mask with the first n lanes active (compare iota < n).
template <class D> HYLE_FORCE_INLINE auto FirstN(D d, std::size_t n) noexcept {
  using T = TFromD<D>;
  const std::size_t vl = Lanes(d);
  if constexpr (sizeof(T) == 1) return __riscv_vmsltu(__riscv_vid_v_u8m1(vl), static_cast<std::uint8_t>(n), vl);
  else if constexpr (sizeof(T) == 2) return __riscv_vmsltu(__riscv_vid_v_u16m1(vl), static_cast<std::uint16_t>(n), vl);
  else if constexpr (sizeof(T) == 4) return __riscv_vmsltu(__riscv_vid_v_u32m1(vl), static_cast<std::uint32_t>(n), vl);
  else return __riscv_vmsltu(__riscv_vid_v_u64m1(vl), static_cast<std::uint64_t>(n), vl);
}
template <class D> HYLE_FORCE_INLINE VFromD<D> LoadN(D d, const TFromD<D>* p, std::size_t n) noexcept {
  using T = TFromD<D>;
  (void)d;
  if constexpr (std::is_same_v<T, float>) return __riscv_vle32_v_f32m1(p, n);
  else if constexpr (std::is_same_v<T, double>) return __riscv_vle64_v_f64m1(p, n);
  else if constexpr (sizeof(T) == 1 && std::is_signed_v<T>) return __riscv_vle8_v_i8m1(reinterpret_cast<const std::int8_t*>(p), n);
  else if constexpr (sizeof(T) == 1) return __riscv_vle8_v_u8m1(reinterpret_cast<const std::uint8_t*>(p), n);
  else if constexpr (sizeof(T) == 2 && std::is_signed_v<T>) return __riscv_vle16_v_i16m1(reinterpret_cast<const std::int16_t*>(p), n);
  else if constexpr (sizeof(T) == 2) return __riscv_vle16_v_u16m1(reinterpret_cast<const std::uint16_t*>(p), n);
  else if constexpr (sizeof(T) == 4 && std::is_signed_v<T>) return __riscv_vle32_v_i32m1(reinterpret_cast<const std::int32_t*>(p), n);
  else if constexpr (sizeof(T) == 4) return __riscv_vle32_v_u32m1(reinterpret_cast<const std::uint32_t*>(p), n);
  else if constexpr (std::is_signed_v<T>) return __riscv_vle64_v_i64m1(reinterpret_cast<const std::int64_t*>(p), n);
  else return __riscv_vle64_v_u64m1(reinterpret_cast<const std::uint64_t*>(p), n);
}
template <class D> HYLE_FORCE_INLINE void StoreN(VFromD<D> v, D d, TFromD<D>* p, std::size_t n) noexcept {
  using T = TFromD<D>;
  (void)d;
  if constexpr (std::is_same_v<T, float>) __riscv_vse32_v_f32m1(p, v, n);
  else if constexpr (std::is_same_v<T, double>) __riscv_vse64_v_f64m1(p, v, n);
  else if constexpr (sizeof(T) == 1 && std::is_signed_v<T>) __riscv_vse8_v_i8m1(reinterpret_cast<std::int8_t*>(p), v, n);
  else if constexpr (sizeof(T) == 1) __riscv_vse8_v_u8m1(reinterpret_cast<std::uint8_t*>(p), v, n);
  else if constexpr (sizeof(T) == 2 && std::is_signed_v<T>) __riscv_vse16_v_i16m1(reinterpret_cast<std::int16_t*>(p), v, n);
  else if constexpr (sizeof(T) == 2) __riscv_vse16_v_u16m1(reinterpret_cast<std::uint16_t*>(p), v, n);
  else if constexpr (sizeof(T) == 4 && std::is_signed_v<T>) __riscv_vse32_v_i32m1(reinterpret_cast<std::int32_t*>(p), v, n);
  else if constexpr (sizeof(T) == 4) __riscv_vse32_v_u32m1(reinterpret_cast<std::uint32_t*>(p), v, n);
  else if constexpr (std::is_signed_v<T>) __riscv_vse64_v_i64m1(reinterpret_cast<std::int64_t*>(p), v, n);
  else __riscv_vse64_v_u64m1(reinterpret_cast<std::uint64_t*>(p), v, n);
}

// ---- masked load / blended store (predicated, fault-safe) ---------------------------
template <class M, class D> HYLE_FORCE_INLINE VFromD<D> MaskedLoad(M m, D d, const TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  const std::size_t vl = Lanes(d);
  if constexpr (std::is_same_v<T, float>) return __riscv_vle32_v_f32m1_m(m, p, vl);
  else if constexpr (std::is_same_v<T, double>) return __riscv_vle64_v_f64m1_m(m, p, vl);
  else if constexpr (sizeof(T) == 4 && std::is_signed_v<T>) return __riscv_vle32_v_i32m1_m(m, reinterpret_cast<const std::int32_t*>(p), vl);
  else if constexpr (sizeof(T) == 4) return __riscv_vle32_v_u32m1_m(m, reinterpret_cast<const std::uint32_t*>(p), vl);
  else if constexpr (sizeof(T) == 8 && std::is_signed_v<T>) return __riscv_vle64_v_i64m1_m(m, reinterpret_cast<const std::int64_t*>(p), vl);
  else if constexpr (sizeof(T) == 8) return __riscv_vle64_v_u64m1_m(m, reinterpret_cast<const std::uint64_t*>(p), vl);
  else if constexpr (sizeof(T) == 2 && std::is_signed_v<T>) return __riscv_vle16_v_i16m1_m(m, reinterpret_cast<const std::int16_t*>(p), vl);
  else if constexpr (sizeof(T) == 2) return __riscv_vle16_v_u16m1_m(m, reinterpret_cast<const std::uint16_t*>(p), vl);
  else if constexpr (std::is_signed_v<T>) return __riscv_vle8_v_i8m1_m(m, reinterpret_cast<const std::int8_t*>(p), vl);
  else return __riscv_vle8_v_u8m1_m(m, reinterpret_cast<const std::uint8_t*>(p), vl);
}
template <class D, class M> HYLE_FORCE_INLINE void BlendedStore(VFromD<D> v, M m, D d, TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  const std::size_t vl = Lanes(d);
  if constexpr (std::is_same_v<T, float>) __riscv_vse32_v_f32m1_m(m, p, v, vl);
  else if constexpr (std::is_same_v<T, double>) __riscv_vse64_v_f64m1_m(m, p, v, vl);
  else if constexpr (sizeof(T) == 4 && std::is_signed_v<T>) __riscv_vse32_v_i32m1_m(m, reinterpret_cast<std::int32_t*>(p), v, vl);
  else if constexpr (sizeof(T) == 4) __riscv_vse32_v_u32m1_m(m, reinterpret_cast<std::uint32_t*>(p), v, vl);
  else if constexpr (sizeof(T) == 8 && std::is_signed_v<T>) __riscv_vse64_v_i64m1_m(m, reinterpret_cast<std::int64_t*>(p), v, vl);
  else if constexpr (sizeof(T) == 8) __riscv_vse64_v_u64m1_m(m, reinterpret_cast<std::uint64_t*>(p), v, vl);
  else if constexpr (sizeof(T) == 2 && std::is_signed_v<T>) __riscv_vse16_v_i16m1_m(m, reinterpret_cast<std::int16_t*>(p), v, vl);
  else if constexpr (sizeof(T) == 2) __riscv_vse16_v_u16m1_m(m, reinterpret_cast<std::uint16_t*>(p), v, vl);
  else if constexpr (std::is_signed_v<T>) __riscv_vse8_v_i8m1_m(m, reinterpret_cast<std::int8_t*>(p), v, vl);
  else __riscv_vse8_v_u8m1_m(m, reinterpret_cast<std::uint8_t*>(p), v, vl);
}

// ---- gather / scatter (RVV NATIVE vluxei/vsuxei; indices are BYTE offsets) -----------
// HYLE's GatherIndex takes an ELEMENT index, RVV's indexed load takes a BYTE offset -> scale by
// sizeof(T). idx is the unsigned same-width lane type.
template <class D, class VI>
HYLE_FORCE_INLINE VFromD<D> GatherIndex(D d, const TFromD<D>* base, VI idx) noexcept {
  using T = TFromD<D>;
  const std::size_t vl = Lanes(d);
  const auto boff = __riscv_vmul(idx, static_cast<TFromV<VI>>(sizeof(T)), vl);  // element->byte
  if constexpr (std::is_same_v<T, float>) return __riscv_vluxei32_v_f32m1(base, boff, vl);
  else if constexpr (sizeof(T) == 4 && std::is_signed_v<T>) return __riscv_vluxei32_v_i32m1(reinterpret_cast<const std::int32_t*>(base), boff, vl);
  else if constexpr (sizeof(T) == 4) return __riscv_vluxei32_v_u32m1(reinterpret_cast<const std::uint32_t*>(base), boff, vl);
  else if constexpr (std::is_same_v<T, double>) return __riscv_vluxei64_v_f64m1(base, boff, vl);
  else if constexpr (sizeof(T) == 8 && std::is_signed_v<T>) return __riscv_vluxei64_v_i64m1(reinterpret_cast<const std::int64_t*>(base), boff, vl);
  else return __riscv_vluxei64_v_u64m1(reinterpret_cast<const std::uint64_t*>(base), boff, vl);
}
template <class D, class VI>
HYLE_FORCE_INLINE void ScatterIndex(VFromD<D> v, D d, TFromD<D>* base, VI idx) noexcept {
  using T = TFromD<D>;
  const std::size_t vl = Lanes(d);
  const auto boff = __riscv_vmul(idx, static_cast<TFromV<VI>>(sizeof(T)), vl);
  if constexpr (std::is_same_v<T, float>) __riscv_vsuxei32_v_f32m1(base, boff, v, vl);
  else if constexpr (sizeof(T) == 4 && std::is_signed_v<T>) __riscv_vsuxei32_v_i32m1(reinterpret_cast<std::int32_t*>(base), boff, v, vl);
  else if constexpr (sizeof(T) == 4) __riscv_vsuxei32_v_u32m1(reinterpret_cast<std::uint32_t*>(base), boff, v, vl);
  else if constexpr (std::is_same_v<T, double>) __riscv_vsuxei64_v_f64m1(base, boff, v, vl);
  else if constexpr (sizeof(T) == 8 && std::is_signed_v<T>) __riscv_vsuxei64_v_i64m1(reinterpret_cast<std::int64_t*>(base), boff, v, vl);
  else __riscv_vsuxei64_v_u64m1(reinterpret_cast<std::uint64_t*>(base), boff, v, vl);
}

// ---- compress (RVV NATIVE vcompress) + CompressStore --------------------------------
template <class V, class M> HYLE_FORCE_INLINE V Compress(V v, M m) noexcept {
  return __riscv_vcompress(v, m, detail::vlmax<TFromV<V>>());
}
template <class D, class M>
HYLE_FORCE_INLINE std::size_t CompressStore(VFromD<D> v, M m, D d, TFromD<D>* p) noexcept {
  const std::size_t cnt = CountTrue(d, m);
  StoreN(Compress(v, m), d, p, cnt);
  return cnt;
}

// NOTE: element-wise PopulationCount / LeadingZeroCount are the Zvbb extension (__riscv_vcpop_v /
// __riscv_vclz_v), NOT in the rv64gcv base V. Deferred until a Zvbb tier is added (documented
// boundary, same as SVE's deferred block-swizzle/width-changers).

// ---- swizzle (whole-vector; RVV has no zip1/zip2 primitive -> build via vrgather + masks) --------
// RVV lacks SVE's svzip/svuzp. We synthesize the whole-vector interleave/deinterleave with vrgather
// index vectors (built from vid) and even-lane masks. All at the runtime vl.
namespace detail {
// even-lane mask: lane i active iff i is even. (iota & 1) == 0.
template <class D> HYLE_FORCE_INLINE auto even_mask_rvv(D d) noexcept {
  using T = TFromD<D>;
  const std::size_t vl = Lanes(d);
  if constexpr (sizeof(T) == 1) return __riscv_vmseq(__riscv_vand(__riscv_vid_v_u8m1(vl), (std::uint8_t)1, vl), (std::uint8_t)0, vl);
  else if constexpr (sizeof(T) == 2) return __riscv_vmseq(__riscv_vand(__riscv_vid_v_u16m1(vl), (std::uint16_t)1, vl), (std::uint16_t)0, vl);
  else if constexpr (sizeof(T) == 4) return __riscv_vmseq(__riscv_vand(__riscv_vid_v_u32m1(vl), (std::uint32_t)1, vl), (std::uint32_t)0, vl);
  else return __riscv_vmseq(__riscv_vand(__riscv_vid_v_u64m1(vl), (std::uint64_t)1, vl), (std::uint64_t)0, vl);
}
// index vector iota (unsigned same-width) for vrgather.
template <class T> HYLE_FORCE_INLINE auto iota_u_rvv(std::size_t vl) noexcept {
  if constexpr (sizeof(T) == 1) return __riscv_vid_v_u8m1(vl);
  else if constexpr (sizeof(T) == 2) return __riscv_vid_v_u16m1(vl);
  else if constexpr (sizeof(T) == 4) return __riscv_vid_v_u32m1(vl);
  else return __riscv_vid_v_u64m1(vl);
}
}  // namespace detail
// InterleaveLower(a,b) = [a0,b0,a1,b1,...] (lower half of each, interleaved). result[i] = (i even)?
// a[i/2] : b[i/2]. gather a and b by index i/2, then select even<-a, odd<-b.
template <class V> HYLE_FORCE_INLINE V InterleaveLower(V a, V b) noexcept {
  using T = TFromV<V>;
  const std::size_t vl = detail::vlmax<T>();
  const auto id = detail::iota_u_rvv<T>(vl);
  const auto half = __riscv_vsrl(id, (typename std::make_unsigned_t<T>)1, vl);  // i/2 (unsigned idx)
  const V ga = __riscv_vrgather(a, half, vl);
  const V gb = __riscv_vrgather(b, half, vl);
  return __riscv_vmerge(gb, ga, detail::even_mask_rvv(DFromV<V>()), vl);  // even<-a, odd<-b
}
// InterleaveUpper = [a_{h},b_{h},...] with h starting at vl/2. result[i] = base+i/2 where base=vl/2.
template <class V> HYLE_FORCE_INLINE V InterleaveUpper(V a, V b) noexcept {
  using T = TFromV<V>;
  using U = std::make_unsigned_t<T>;
  const std::size_t vl = detail::vlmax<T>();
  const auto id = detail::iota_u_rvv<T>(vl);
  const auto idx = __riscv_vadd(__riscv_vsrl(id, (U)1, vl), static_cast<U>(vl / 2), vl);  // vl/2 + i/2
  const V ga = __riscv_vrgather(a, idx, vl);
  const V gb = __riscv_vrgather(b, idx, vl);
  return __riscv_vmerge(gb, ga, detail::even_mask_rvv(DFromV<V>()), vl);
}
template <class V> HYLE_FORCE_INLINE V ZipLower(V a, V b) noexcept { return InterleaveLower(a, b); }
template <class V> HYLE_FORCE_INLINE V ZipUpper(V a, V b) noexcept { return InterleaveUpper(a, b); }
template <class D, class V> HYLE_FORCE_INLINE V InterleaveLower(D, V a, V b) noexcept { return InterleaveLower(a, b); }
template <class D, class V> HYLE_FORCE_INLINE V InterleaveUpper(D, V a, V b) noexcept { return InterleaveUpper(a, b); }
// ConcatEven(hi,lo) = [lo even lanes, hi even lanes]; ConcatOdd = odd lanes. Build by compressing
// each source with the even/odd mask (packs active lanes to front), then splice hi's packed half in
// after lo's (vl/2 each). Use a slideup to place hi's compressed result at offset vl/2.
template <class D, class V> HYLE_FORCE_INLINE V ConcatEven(D d, V hi, V lo) noexcept {
  using T = TFromV<V>;
  const std::size_t vl = Lanes(d);
  const auto em = detail::even_mask_rvv(d);
  const V lo_packed = __riscv_vcompress(lo, em, vl);  // lo's even lanes at [0, vl/2)
  const V hi_packed = __riscv_vcompress(hi, em, vl);
  return __riscv_vslideup(lo_packed, hi_packed, vl / 2, vl);  // hi's evens at [vl/2, vl)
}
template <class D, class V> HYLE_FORCE_INLINE V ConcatOdd(D d, V hi, V lo) noexcept {
  using T = TFromV<V>;
  const std::size_t vl = Lanes(d);
  const auto em = detail::even_mask_rvv(d);
  const auto om = __riscv_vmnot(em, vl);  // odd-lane mask
  const V lo_packed = __riscv_vcompress(lo, om, vl);
  const V hi_packed = __riscv_vcompress(hi, om, vl);
  return __riscv_vslideup(lo_packed, hi_packed, vl / 2, vl);
}
// OddEven(odd, even): even-position lanes from `even`, odd-position lanes from `odd`.
template <class V> HYLE_FORCE_INLINE V OddEven(V odd, V even) noexcept {
  const std::size_t vl = detail::vlmax<TFromV<V>>();
  return __riscv_vmerge(odd, even, detail::even_mask_rvv(DFromV<V>()), vl);  // even mask picks `even`
}
// Broadcast<kLane>(v): splat lane kLane across all lanes (vrgather by a constant index).
template <int kLane, class V> HYLE_FORCE_INLINE V Broadcast(V v) noexcept {
  using T = TFromV<V>;
  using U = std::make_unsigned_t<T>;
  const std::size_t vl = detail::vlmax<T>();
  return __riscv_vrgather(v, Set(Simd<U, kNativeVectorBytes / sizeof(U), 0>(), static_cast<U>(kLane)), vl);
}
// MaskedLoadOr(no, m, d, p): load active lanes from p, inactive lanes take `no`.
template <class M, class D> HYLE_FORCE_INLINE VFromD<D> MaskedLoadOr(VFromD<D> no, M m, D d, const TFromD<D>* p) noexcept {
  return IfThenElse(m, MaskedLoad(m, d, p), no);
}

}  // namespace hyle::simd
