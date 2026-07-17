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

#include <cmath>    // std::ldexp (MulByPow2 store+reload; RVV base V has no fscale)
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

// vsetvlmax for a given lane type (LMUL=1). HYLE's scalable tag reserves a 2048-bit scratch
// upper bound; RVV permits wider implementations, so reject those loudly instead of overrunning it.
template <typename T> HYLE_FORCE_INLINE std::size_t vlmax() noexcept {
  std::size_t vl;
  if constexpr (sizeof(T) == 1) vl = __riscv_vsetvlmax_e8m1();
  else if constexpr (sizeof(T) == 2) vl = __riscv_vsetvlmax_e16m1();
  else if constexpr (sizeof(T) == 4) vl = __riscv_vsetvlmax_e32m1();
  else vl = __riscv_vsetvlmax_e64m1();
  HYLE_FATAL_ASSERT(vl * sizeof(T) <= kNativeVectorBytes);
  return vl;
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

// Forward declarations for generic compositions defined before the swizzle/bitcast sections.
template <class V> HYLE_FORCE_INLINE V OddEven(V odd, V even) noexcept;
template <class D, class V> HYLE_FORCE_INLINE VFromD<D> BitCast(D d, V v) noexcept;

namespace detail {
// GCC's RVV reinterpret overloads do not accept every signed/float -> different-SEW pair directly.
// Normalize to same-width unsigned first, then change SEW; this keeps the full LMUL=1 register bits.
template <class V> HYLE_FORCE_INLINE auto unsigned_bits_rvv(V v) noexcept {
  using T = TFromV<V>;
  if constexpr (std::is_unsigned_v<T>) return v;
  else if constexpr (sizeof(T) == 1) return __riscv_vreinterpret_u8m1(v);
  else if constexpr (sizeof(T) == 2) return __riscv_vreinterpret_u16m1(v);
  else if constexpr (sizeof(T) == 4) return __riscv_vreinterpret_u32m1(v);
  else return __riscv_vreinterpret_u64m1(v);
}
template <class V> HYLE_FORCE_INLINE vuint8m1_t bits_u8_rvv(V v) noexcept {
  auto u = unsigned_bits_rvv(v);
  if constexpr (sizeof(TFromV<V>) == 1) return u;
  else return __riscv_vreinterpret_u8m1(u);
}
template <class V> HYLE_FORCE_INLINE vuint16m1_t bits_u16_rvv(V v) noexcept {
  auto u = unsigned_bits_rvv(v);
  if constexpr (sizeof(TFromV<V>) == 2) return u;
  else return __riscv_vreinterpret_u16m1(u);
}
template <class V> HYLE_FORCE_INLINE vuint32m1_t bits_u32_rvv(V v) noexcept {
  auto u = unsigned_bits_rvv(v);
  if constexpr (sizeof(TFromV<V>) == 4) return u;
  else return __riscv_vreinterpret_u32m1(u);
}
template <class V> HYLE_FORCE_INLINE vuint64m1_t bits_u64_rvv(V v) noexcept {
  auto u = unsigned_bits_rvv(v);
  if constexpr (sizeof(TFromV<V>) == 8) return u;
  else return __riscv_vreinterpret_u64m1(u);
}
}  // namespace detail

// ---- Lanes(d): RUNTIME vector length (vsetvlmax at LMUL=1) ---------------------------
template <class D>
HYLE_FORCE_INLINE std::size_t Lanes(D) noexcept {
  using T = TFromD<D>;
  constexpr int pow2 = D{}.Pow2();
  static_assert(pow2 <= 0, "RVV grouped LMUL vectors are not represented by the LMUL=1 backend");
  const std::size_t scaled = detail::vlmax<T>() >> static_cast<unsigned>(-pow2);
  return scaled < D::kMaxLanes ? scaled : D::kMaxLanes;
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

// ---- interleaved load / store (native RVV segmented instructions) -------------------
namespace detail {
#define HYLE_RVV_INTERLEAVED(VBASE, TOKEN, CTYPE, SEW)                                           \
  HYLE_FORCE_INLINE void load_interleaved2(const CTYPE* p, VBASE##m1_t& v0,            \
                                            VBASE##m1_t& v1, std::size_t vl) noexcept { \
    const VBASE##m1x2_t t = __riscv_vlseg2e##SEW##_v_##TOKEN##m1x2(p, vl);             \
    v0 = __riscv_vget_v_##TOKEN##m1x2_##TOKEN##m1(t, 0);                                  \
    v1 = __riscv_vget_v_##TOKEN##m1x2_##TOKEN##m1(t, 1);                                  \
  }                                                                                        \
  HYLE_FORCE_INLINE void load_interleaved3(const CTYPE* p, VBASE##m1_t& v0,            \
                                            VBASE##m1_t& v1, VBASE##m1_t& v2,       \
                                            std::size_t vl) noexcept {                     \
    const VBASE##m1x3_t t = __riscv_vlseg3e##SEW##_v_##TOKEN##m1x3(p, vl);             \
    v0 = __riscv_vget_v_##TOKEN##m1x3_##TOKEN##m1(t, 0);                                  \
    v1 = __riscv_vget_v_##TOKEN##m1x3_##TOKEN##m1(t, 1);                                  \
    v2 = __riscv_vget_v_##TOKEN##m1x3_##TOKEN##m1(t, 2);                                  \
  }                                                                                        \
  HYLE_FORCE_INLINE void load_interleaved4(const CTYPE* p, VBASE##m1_t& v0,            \
                                            VBASE##m1_t& v1, VBASE##m1_t& v2,       \
                                            VBASE##m1_t& v3, std::size_t vl) noexcept { \
    const VBASE##m1x4_t t = __riscv_vlseg4e##SEW##_v_##TOKEN##m1x4(p, vl);             \
    v0 = __riscv_vget_v_##TOKEN##m1x4_##TOKEN##m1(t, 0);                                  \
    v1 = __riscv_vget_v_##TOKEN##m1x4_##TOKEN##m1(t, 1);                                  \
    v2 = __riscv_vget_v_##TOKEN##m1x4_##TOKEN##m1(t, 2);                                  \
    v3 = __riscv_vget_v_##TOKEN##m1x4_##TOKEN##m1(t, 3);                                  \
  }                                                                                        \
  HYLE_FORCE_INLINE void store_interleaved2(VBASE##m1_t v0, VBASE##m1_t v1,        \
                                             CTYPE* p, std::size_t vl) noexcept {          \
    VBASE##m1x2_t t{};                                                                  \
    t = __riscv_vset_v_##TOKEN##m1_##TOKEN##m1x2(t, 0, v0);                               \
    t = __riscv_vset_v_##TOKEN##m1_##TOKEN##m1x2(t, 1, v1);                               \
    __riscv_vsseg2e##SEW##_v_##TOKEN##m1x2(p, t, vl);                                     \
  }                                                                                        \
  HYLE_FORCE_INLINE void store_interleaved3(VBASE##m1_t v0, VBASE##m1_t v1,        \
                                             VBASE##m1_t v2, CTYPE* p,                 \
                                             std::size_t vl) noexcept {                    \
    VBASE##m1x3_t t{};                                                                  \
    t = __riscv_vset_v_##TOKEN##m1_##TOKEN##m1x3(t, 0, v0);                               \
    t = __riscv_vset_v_##TOKEN##m1_##TOKEN##m1x3(t, 1, v1);                               \
    t = __riscv_vset_v_##TOKEN##m1_##TOKEN##m1x3(t, 2, v2);                               \
    __riscv_vsseg3e##SEW##_v_##TOKEN##m1x3(p, t, vl);                                     \
  }                                                                                        \
  HYLE_FORCE_INLINE void store_interleaved4(VBASE##m1_t v0, VBASE##m1_t v1,        \
                                             VBASE##m1_t v2, VBASE##m1_t v3,        \
                                             CTYPE* p, std::size_t vl) noexcept {          \
    VBASE##m1x4_t t{};                                                                  \
    t = __riscv_vset_v_##TOKEN##m1_##TOKEN##m1x4(t, 0, v0);                               \
    t = __riscv_vset_v_##TOKEN##m1_##TOKEN##m1x4(t, 1, v1);                               \
    t = __riscv_vset_v_##TOKEN##m1_##TOKEN##m1x4(t, 2, v2);                               \
    t = __riscv_vset_v_##TOKEN##m1_##TOKEN##m1x4(t, 3, v3);                               \
    __riscv_vsseg4e##SEW##_v_##TOKEN##m1x4(p, t, vl);                                     \
  }
HYLE_RVV_INTERLEAVED(vint8, i8, std::int8_t, 8)
HYLE_RVV_INTERLEAVED(vuint8, u8, std::uint8_t, 8)
HYLE_RVV_INTERLEAVED(vint16, i16, std::int16_t, 16)
HYLE_RVV_INTERLEAVED(vuint16, u16, std::uint16_t, 16)
HYLE_RVV_INTERLEAVED(vint32, i32, std::int32_t, 32)
HYLE_RVV_INTERLEAVED(vuint32, u32, std::uint32_t, 32)
HYLE_RVV_INTERLEAVED(vint64, i64, std::int64_t, 64)
HYLE_RVV_INTERLEAVED(vuint64, u64, std::uint64_t, 64)
HYLE_RVV_INTERLEAVED(vfloat32, f32, float, 32)
HYLE_RVV_INTERLEAVED(vfloat64, f64, double, 64)
#undef HYLE_RVV_INTERLEAVED
}  // namespace detail

template <class D>
HYLE_FORCE_INLINE void LoadInterleaved2(D d, const TFromD<D>* p, VFromD<D>& v0,
                                        VFromD<D>& v1) noexcept {
  detail::load_interleaved2(p, v0, v1, Lanes(d));
}
template <class D>
HYLE_FORCE_INLINE void LoadInterleaved3(D d, const TFromD<D>* p, VFromD<D>& v0,
                                        VFromD<D>& v1, VFromD<D>& v2) noexcept {
  detail::load_interleaved3(p, v0, v1, v2, Lanes(d));
}
template <class D>
HYLE_FORCE_INLINE void LoadInterleaved4(D d, const TFromD<D>* p, VFromD<D>& v0,
                                        VFromD<D>& v1, VFromD<D>& v2,
                                        VFromD<D>& v3) noexcept {
  detail::load_interleaved4(p, v0, v1, v2, v3, Lanes(d));
}
template <class D>
HYLE_FORCE_INLINE void StoreInterleaved2(VFromD<D> v0, VFromD<D> v1, D d,
                                         TFromD<D>* p) noexcept {
  detail::store_interleaved2(v0, v1, p, Lanes(d));
}
template <class D>
HYLE_FORCE_INLINE void StoreInterleaved3(VFromD<D> v0, VFromD<D> v1, VFromD<D> v2, D d,
                                         TFromD<D>* p) noexcept {
  detail::store_interleaved3(v0, v1, v2, p, Lanes(d));
}
template <class D>
HYLE_FORCE_INLINE void StoreInterleaved4(VFromD<D> v0, VFromD<D> v1, VFromD<D> v2,
                                         VFromD<D> v3, D d, TFromD<D>* p) noexcept {
  detail::store_interleaved4(v0, v1, v2, v3, p, Lanes(d));
}

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
// AddSub(a,b): a-b in even lanes, a+b in odd lanes. Base V has no addsub instruction.
template <class V> HYLE_FORCE_INLINE V AddSub(V a, V b) noexcept {
  return OddEven(Add(a, b), Sub(a, b));
}
// RVV 1.0's rec7/rsqrt7 are raw seven-bit estimates; vec.h applies one Newton refinement.
template <class V> HYLE_FORCE_INLINE V ApproximateReciprocal(V v) noexcept {
  static_assert(std::is_floating_point_v<TFromV<V>>, "ApproximateReciprocal is float only");
  return __riscv_vfrec7(v, detail::vlmax<TFromV<V>>());
}
template <class V> HYLE_FORCE_INLINE V ApproximateReciprocalSqrt(V v) noexcept {
  static_assert(std::is_floating_point_v<TFromV<V>>, "ApproximateReciprocalSqrt is float only");
  return __riscv_vfrsqrt7(v, detail::vlmax<TFromV<V>>());
}

// ---- bitwise ------------------------------------------------------------------------
// RVV bitwise instructions are integer-only; float overloads preserve bits via unsigned vectors.
template <class V> HYLE_FORCE_INLINE V And(V a, V b) noexcept {
  if constexpr (std::is_floating_point_v<TFromV<V>>) {
    const DFromV<V> d;
    const RebindToUnsigned<decltype(d)> du;
    return BitCast(d, And(BitCast(du, a), BitCast(du, b)));
  } else {
    return __riscv_vand(a, b, detail::vlmax<TFromV<V>>());
  }
}
template <class V> HYLE_FORCE_INLINE V Or(V a, V b) noexcept {
  if constexpr (std::is_floating_point_v<TFromV<V>>) {
    const DFromV<V> d;
    const RebindToUnsigned<decltype(d)> du;
    return BitCast(d, Or(BitCast(du, a), BitCast(du, b)));
  } else {
    return __riscv_vor(a, b, detail::vlmax<TFromV<V>>());
  }
}
template <class V> HYLE_FORCE_INLINE V Xor(V a, V b) noexcept {
  if constexpr (std::is_floating_point_v<TFromV<V>>) {
    const DFromV<V> d;
    const RebindToUnsigned<decltype(d)> du;
    return BitCast(d, Xor(BitCast(du, a), BitCast(du, b)));
  } else {
    return __riscv_vxor(a, b, detail::vlmax<TFromV<V>>());
  }
}
template <class V> HYLE_FORCE_INLINE V Not(V a) noexcept {
  if constexpr (std::is_floating_point_v<TFromV<V>>) {
    const DFromV<V> d;
    const RebindToUnsigned<decltype(d)> du;
    return BitCast(d, Not(BitCast(du, a)));
  } else {
    return __riscv_vnot(a, detail::vlmax<TFromV<V>>());
  }
}
template <class V> HYLE_FORCE_INLINE V AndNot(V a, V b) noexcept {  // ~a & b (Highway order)
  if constexpr (std::is_floating_point_v<TFromV<V>>) {
    const DFromV<V> d;
    const RebindToUnsigned<decltype(d)> du;
    return BitCast(d, AndNot(BitCast(du, a), BitCast(du, b)));
  } else {
    const std::size_t vl = detail::vlmax<TFromV<V>>();
    return __riscv_vand(__riscv_vnot(a, vl), b, vl);
  }
}

// LMUL=1 mask types by element width. Exact overloads keep predicate logic out of the vector
// TFromV path (mask types are sizeless and deliberately have no DFromV mapping).
#define HYLE_RVV_MASK_LOGICAL(M, T)                                                \
  HYLE_FORCE_INLINE M And(M a, M b) noexcept {                                    \
    return __riscv_vmand(a, b, detail::vlmax<T>());                               \
  }                                                                                \
  HYLE_FORCE_INLINE M Or(M a, M b) noexcept {                                     \
    return __riscv_vmor(a, b, detail::vlmax<T>());                                \
  }                                                                                \
  HYLE_FORCE_INLINE M Xor(M a, M b) noexcept {                                    \
    return __riscv_vmxor(a, b, detail::vlmax<T>());                               \
  }                                                                                \
  HYLE_FORCE_INLINE M Not(M a) noexcept {                                         \
    return __riscv_vmnot(a, detail::vlmax<T>());                                  \
  }                                                                                \
  HYLE_FORCE_INLINE M AndNot(M a, M b) noexcept {                                 \
    return __riscv_vmand(__riscv_vmnot(a, detail::vlmax<T>()), b,                 \
                         detail::vlmax<T>());                                      \
  }
HYLE_RVV_MASK_LOGICAL(vbool8_t, std::uint8_t)
HYLE_RVV_MASK_LOGICAL(vbool16_t, std::uint16_t)
HYLE_RVV_MASK_LOGICAL(vbool32_t, std::uint32_t)
HYLE_RVV_MASK_LOGICAL(vbool64_t, std::uint64_t)
#undef HYLE_RVV_MASK_LOGICAL

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
  if (__riscv_vfirst(m, vl) < 0) return -1;
  if constexpr (sizeof(T) == 1) { auto id = __riscv_vid_v_u8m1(vl); auto sel = __riscv_vmerge(__riscv_vmv_v_x_u8m1(0, vl), id, m, vl); return static_cast<std::ptrdiff_t>(__riscv_vmv_x(__riscv_vredmaxu(sel, sel, vl))); }
  else if constexpr (sizeof(T) == 2) { auto id = __riscv_vid_v_u16m1(vl); auto sel = __riscv_vmerge(__riscv_vmv_v_x_u16m1(0, vl), id, m, vl); return static_cast<std::ptrdiff_t>(__riscv_vmv_x(__riscv_vredmaxu(sel, sel, vl))); }
  else if constexpr (sizeof(T) == 4) { auto id = __riscv_vid_v_u32m1(vl); auto sel = __riscv_vmerge(__riscv_vmv_v_x_u32m1(0, vl), id, m, vl); return static_cast<std::ptrdiff_t>(__riscv_vmv_x(__riscv_vredmaxu(sel, sel, vl))); }
  else { auto id = __riscv_vid_v_u64m1(vl); auto sel = __riscv_vmerge(__riscv_vmv_v_x_u64m1(0, vl), id, m, vl); return static_cast<std::ptrdiff_t>(__riscv_vmv_x(__riscv_vredmaxu(sel, sel, vl))); }
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
  else if constexpr (std::is_signed_v<TFromD<D>>) return __riscv_vmv_x(__riscv_vredmin(v, v, vl));
  else return __riscv_vmv_x(__riscv_vredminu(v, v, vl));
}
template <class D> HYLE_FORCE_INLINE TFromD<D> ReduceMax(D d, VFromD<D> v) noexcept {
  const std::size_t vl = Lanes(d);
  if constexpr (std::is_floating_point_v<TFromD<D>>) return __riscv_vfmv_f(__riscv_vfredmax(v, v, vl));
  else if constexpr (std::is_signed_v<TFromD<D>>) return __riscv_vmv_x(__riscv_vredmax(v, v, vl));
  else return __riscv_vmv_x(__riscv_vredmaxu(v, v, vl));
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
template <int kBits, class V> HYLE_FORCE_INLINE V RotateRight(V v) noexcept {
  using T = TFromV<V>;
  static_assert(std::is_integral_v<T>, "RotateRight is integer-lane only");
  constexpr int kSizeBits = static_cast<int>(sizeof(T) * 8);
  static_assert(kBits >= 0 && kBits < kSizeBits, "rotate count in [0,bits)");
  if constexpr (kBits == 0) {
    return v;
  } else {
#if defined(__riscv_zvbb)
    // Zvbb native vror: one instruction instead of shift-or.
    const std::size_t vl = detail::vlmax<T>();
    const RebindToUnsigned<DFromV<V>> du;
    const auto uv = BitCast(du, v);
    if constexpr (sizeof(T) == 1) return BitCast(DFromV<V>(), __riscv_vror_vx_u8m1(uv, static_cast<std::size_t>(kBits), vl));
    else if constexpr (sizeof(T) == 2) return BitCast(DFromV<V>(), __riscv_vror_vx_u16m1(uv, static_cast<std::size_t>(kBits), vl));
    else if constexpr (sizeof(T) == 4) return BitCast(DFromV<V>(), __riscv_vror_vx_u32m1(uv, static_cast<std::size_t>(kBits), vl));
    else return BitCast(DFromV<V>(), __riscv_vror_vx_u64m1(uv, static_cast<std::size_t>(kBits), vl));
#else
    const RebindToUnsigned<DFromV<V>> du;
    const auto uv = BitCast(du, v);
    return BitCast(DFromV<V>(), Or(ShiftRight<kBits>(uv), ShiftLeft<kSizeBits - kBits>(uv)));
#endif
  }
}
template <int kBits, class V> HYLE_FORCE_INLINE V RotateLeft(V v) noexcept {
  using T = TFromV<V>;
  static_assert(std::is_integral_v<T>, "RotateLeft is integer-lane only");
  constexpr int kSizeBits = static_cast<int>(sizeof(T) * 8);
  static_assert(kBits >= 0 && kBits < kSizeBits, "rotate count in [0,bits)");
  if constexpr (kBits == 0) {
    return v;
  } else {
#if defined(__riscv_zvbb)
    const std::size_t vl = detail::vlmax<T>();
    const RebindToUnsigned<DFromV<V>> du;
    const auto uv = BitCast(du, v);
    if constexpr (sizeof(T) == 1) return BitCast(DFromV<V>(), __riscv_vrol_vx_u8m1(uv, static_cast<std::size_t>(kBits), vl));
    else if constexpr (sizeof(T) == 2) return BitCast(DFromV<V>(), __riscv_vrol_vx_u16m1(uv, static_cast<std::size_t>(kBits), vl));
    else if constexpr (sizeof(T) == 4) return BitCast(DFromV<V>(), __riscv_vrol_vx_u32m1(uv, static_cast<std::size_t>(kBits), vl));
    else return BitCast(DFromV<V>(), __riscv_vrol_vx_u64m1(uv, static_cast<std::size_t>(kBits), vl));
#else
    return RotateRight<(kSizeBits - kBits) % kSizeBits>(v);
#endif
  }
}

// ---- whole-vector slides -------------------------------------------------------------
template <class D> HYLE_FORCE_INLINE VFromD<D> SlideUpLanes(D d, VFromD<D> v,
                                                            std::size_t amt) noexcept {
  const std::size_t vl = Lanes(d); HYLE_FATAL_ASSERT(amt <= vl);
  if (amt == vl) return Zero(d);
  return __riscv_vslideup(Zero(d), v, amt, vl);
}
template <class D> HYLE_FORCE_INLINE VFromD<D> SlideDownLanes(D d, VFromD<D> v,
                                                              std::size_t amt) noexcept {
  using T = TFromD<D>;
  const std::size_t vl = Lanes(d); HYLE_FATAL_ASSERT(amt <= vl);
  if (amt == vl) return Zero(d);
  const auto shifted = __riscv_vslidedown(v, amt, vl);
  if constexpr (sizeof(T) == 1) return IfThenElse(__riscv_vmsltu(__riscv_vid_v_u8m1(vl), static_cast<std::uint8_t>(vl - amt), vl), shifted, Zero(d));
  else if constexpr (sizeof(T) == 2) return IfThenElse(__riscv_vmsltu(__riscv_vid_v_u16m1(vl), static_cast<std::uint16_t>(vl - amt), vl), shifted, Zero(d));
  else if constexpr (sizeof(T) == 4) return IfThenElse(__riscv_vmsltu(__riscv_vid_v_u32m1(vl), static_cast<std::uint32_t>(vl - amt), vl), shifted, Zero(d));
  else return IfThenElse(__riscv_vmsltu(__riscv_vid_v_u64m1(vl), static_cast<std::uint64_t>(vl - amt), vl), shifted, Zero(d));
}
template <class D> HYLE_FORCE_INLINE VFromD<D> Slide1Up(D d, VFromD<D> v) noexcept {
  return SlideUpLanes(d, v, 1);
}
template <class D> HYLE_FORCE_INLINE VFromD<D> Slide1Down(D d, VFromD<D> v) noexcept {
  return SlideDownLanes(d, v, 1);
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
  if constexpr (std::is_same_v<T, TFromV<V>>) return v;
  else if constexpr (std::is_same_v<T, float>) return __riscv_vreinterpret_f32m1(detail::bits_u32_rvv(v));
  else if constexpr (std::is_same_v<T, double>) return __riscv_vreinterpret_f64m1(detail::bits_u64_rvv(v));
  else if constexpr (std::is_same_v<T, std::int8_t>) return __riscv_vreinterpret_i8m1(detail::bits_u8_rvv(v));
  else if constexpr (std::is_same_v<T, std::uint8_t>) return detail::bits_u8_rvv(v);
  else if constexpr (std::is_same_v<T, std::int16_t>) return __riscv_vreinterpret_i16m1(detail::bits_u16_rvv(v));
  else if constexpr (std::is_same_v<T, std::uint16_t>) return detail::bits_u16_rvv(v);
  else if constexpr (std::is_same_v<T, std::int32_t>) return __riscv_vreinterpret_i32m1(detail::bits_u32_rvv(v));
  else if constexpr (std::is_same_v<T, std::uint32_t>) return detail::bits_u32_rvv(v);
  else if constexpr (std::is_same_v<T, std::int64_t>) return __riscv_vreinterpret_i64m1(detail::bits_u64_rvv(v));
  else return detail::bits_u64_rvv(v);
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
template <typename DstT, std::size_t N, class V>
HYLE_FORCE_INLINE VFromD<Simd<DstT, N, 0>> ConvertInRangeTo(
    Simd<DstT, N, 0> d, V v) noexcept {
  return ConvertTo(d, v);
}
// vfncvt writes one i32 per active f64 lane into an mf2 result. HYLE deliberately keeps native
// RVV values in LMUL=1 carriers even when the logical tag is a half vector, so extend the
// fractional result without changing its low-lane order. Only the first f64-lane-count elements
// are active; callers retain that count through Rebind's kPow2=-1 logical tag.
template <std::size_t N>
HYLE_FORCE_INLINE vint32m1_t DemoteInRangeTo(Simd<std::int32_t, N, 0>,
                                              vfloat64m1_t v) noexcept {
  const std::size_t vlmax = detail::vlmax<double>();
  const std::size_t vl = vlmax < N ? vlmax : N;
  const vint32mf2_t narrow = __riscv_vfncvt_rtz_x_f_w_i32mf2(v, vl);
  return __riscv_vlmul_ext_v_i32mf2_i32m1(narrow);
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
  const std::size_t dst_vl = Lanes(dd);
  const std::size_t src_vl = 2 * dst_vl;
  const V up = __riscv_vslidedown(v, dst_vl, src_vl);
  return PromoteLowerTo(dd, up);
}
template <typename DstT, std::size_t NDst, class V>
HYLE_FORCE_INLINE VFromD<Simd<DstT, NDst, 0>> DemoteTo(Simd<DstT, NDst, 0> dd, V a, V b) noexcept {
  using SrcT = TFromV<V>;
  static_assert(sizeof(DstT) * 2 == sizeof(SrcT) && std::is_integral_v<DstT>, "DemoteTo: 2x integer narrow");
  const std::size_t vldst = Lanes(dd);
  if constexpr (std::is_signed_v<SrcT> != std::is_signed_v<DstT>) {
    if constexpr (std::is_signed_v<SrcT>) {
      const auto src_d = DFromV<V>{};
      const auto zero = Zero(src_d);
      const auto au = BitCast(RebindToUnsigned<decltype(src_d)>{}, Max(a, zero));
      const auto bu = BitCast(RebindToUnsigned<decltype(src_d)>{}, Max(b, zero));
      return DemoteTo(dd, au, bu);
    } else {
      const auto src_d = DFromV<V>{};
      const auto hi = Set(src_d, static_cast<SrcT>(std::numeric_limits<DstT>::max()));
      const auto as = BitCast(RebindToSigned<decltype(src_d)>{}, Min(a, hi));
      const auto bs = BitCast(RebindToSigned<decltype(src_d)>{}, Min(b, hi));
      return DemoteTo(dd, as, bs);
    }
  } else if constexpr (std::is_signed_v<DstT>) {
    if constexpr (NDst < kNativeVectorBytes / sizeof(DstT)) {
      const V packed = __riscv_vslideup(a, b, vldst / 2, vldst);
      if constexpr (sizeof(SrcT) == 2) return __riscv_vnclip_wx_i8m1(__riscv_vlmul_ext_v_i16m1_i16m2(packed), 0, __RISCV_VXRM_RNU, vldst);
      else if constexpr (sizeof(SrcT) == 4) return __riscv_vnclip_wx_i16m1(__riscv_vlmul_ext_v_i32m1_i32m2(packed), 0, __RISCV_VXRM_RNU, vldst);
      else return __riscv_vnclip_wx_i32m1(__riscv_vlmul_ext_v_i64m1_i64m2(packed), 0, __RISCV_VXRM_RNU, vldst);
    } else {
      if constexpr (sizeof(SrcT) == 2) { auto m2 = __riscv_vset_v_i16m1_i16m2(__riscv_vlmul_ext_v_i16m1_i16m2(a), 1, b); return __riscv_vnclip_wx_i8m1(m2, 0, __RISCV_VXRM_RNU, vldst); }
      else if constexpr (sizeof(SrcT) == 4) { auto m2 = __riscv_vset_v_i32m1_i32m2(__riscv_vlmul_ext_v_i32m1_i32m2(a), 1, b); return __riscv_vnclip_wx_i16m1(m2, 0, __RISCV_VXRM_RNU, vldst); }
      else { auto m2 = __riscv_vset_v_i64m1_i64m2(__riscv_vlmul_ext_v_i64m1_i64m2(a), 1, b); return __riscv_vnclip_wx_i32m1(m2, 0, __RISCV_VXRM_RNU, vldst); }
    }
  } else {
    if constexpr (NDst < kNativeVectorBytes / sizeof(DstT)) {
      const V packed = __riscv_vslideup(a, b, vldst / 2, vldst);
      if constexpr (sizeof(SrcT) == 2) return __riscv_vnclipu_wx_u8m1(__riscv_vlmul_ext_v_u16m1_u16m2(packed), 0, __RISCV_VXRM_RNU, vldst);
      else if constexpr (sizeof(SrcT) == 4) return __riscv_vnclipu_wx_u16m1(__riscv_vlmul_ext_v_u32m1_u32m2(packed), 0, __RISCV_VXRM_RNU, vldst);
      else return __riscv_vnclipu_wx_u32m1(__riscv_vlmul_ext_v_u64m1_u64m2(packed), 0, __RISCV_VXRM_RNU, vldst);
    } else {
      if constexpr (sizeof(SrcT) == 2) { auto m2 = __riscv_vset_v_u16m1_u16m2(__riscv_vlmul_ext_v_u16m1_u16m2(a), 1, b); return __riscv_vnclipu_wx_u8m1(m2, 0, __RISCV_VXRM_RNU, vldst); }
      else if constexpr (sizeof(SrcT) == 4) { auto m2 = __riscv_vset_v_u32m1_u32m2(__riscv_vlmul_ext_v_u32m1_u32m2(a), 1, b); return __riscv_vnclipu_wx_u16m1(m2, 0, __RISCV_VXRM_RNU, vldst); }
      else { auto m2 = __riscv_vset_v_u64m1_u64m2(__riscv_vlmul_ext_v_u64m1_u64m2(a), 1, b); return __riscv_vnclipu_wx_u32m1(m2, 0, __RISCV_VXRM_RNU, vldst); }
    }
  }
}
// WidenMulPairwiseAdd(d32, a_i16, b_i16): r[k] = a[2k]*b[2k] + a[2k+1]*b[2k+1] (fixed-point dot
// core). Same bit-slice trick as SVE: view each adjacent i16 pair as one i32 lane, extract even
// (low 16, sign-ext via sll<<16 then sra>>16) and odd (sra>>16), multiply as i32, add. i16->i32.
template <std::size_t NDst, class V>
HYLE_FORCE_INLINE VFromD<Simd<std::int32_t, NDst, 0>> WidenMulPairwiseAdd(Simd<std::int32_t, NDst, 0> dd, V a, V b) noexcept {
  static_assert(std::is_same_v<TFromV<V>, std::int16_t>, "WidenMulPairwiseAdd: i16 -> i32 pairwise");
  const std::size_t vl = Lanes(dd);
  const vint32m1_t a32 = __riscv_vreinterpret_i32m1(a);
  const vint32m1_t b32 = __riscv_vreinterpret_i32m1(b);
  const vint32m1_t ae = __riscv_vsra(__riscv_vsll(a32, 16u, vl), 16u, vl);  // even (low 16, sign-ext)
  const vint32m1_t ao = __riscv_vsra(a32, 16u, vl);                         // odd  (high 16, arith)
  const vint32m1_t be = __riscv_vsra(__riscv_vsll(b32, 16u, vl), 16u, vl);
  const vint32m1_t bo = __riscv_vsra(b32, 16u, vl);
  return __riscv_vadd(__riscv_vmul(ae, be, vl), __riscv_vmul(ao, bo, vl), vl);
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
// MaskFromVec tests the raw MSB. Numeric `v < 0` is wrong for unsigned lanes and misses floating
// -0/negative-NaN, so reinterpret to a same-width signed vector before comparing with zero.
template <class V> HYLE_FORCE_INLINE auto MaskFromVec(V v) noexcept {
  using T = TFromV<V>;
  const std::size_t vl = detail::vlmax<T>();
  if constexpr (sizeof(T) == 1) {
    if constexpr (std::is_same_v<T, std::int8_t>) return __riscv_vmslt(v, std::int8_t{0}, vl);
    else return __riscv_vmslt(__riscv_vreinterpret_i8m1(v), std::int8_t{0}, vl);
  } else if constexpr (sizeof(T) == 2) {
    if constexpr (std::is_same_v<T, std::int16_t>) return __riscv_vmslt(v, std::int16_t{0}, vl);
    else return __riscv_vmslt(__riscv_vreinterpret_i16m1(v), std::int16_t{0}, vl);
  } else if constexpr (sizeof(T) == 4) {
    if constexpr (std::is_same_v<T, std::int32_t>) return __riscv_vmslt(v, std::int32_t{0}, vl);
    else return __riscv_vmslt(__riscv_vreinterpret_i32m1(v), std::int32_t{0}, vl);
  } else {
    if constexpr (std::is_same_v<T, std::int64_t>) return __riscv_vmslt(v, std::int64_t{0}, vl);
    else return __riscv_vmslt(__riscv_vreinterpret_i64m1(v), std::int64_t{0}, vl);
  }
}
// MaskFromNormalizedVec: caller guarantees v is already canonical all-ones/all-zero per lane. RVV
// masks are mask registers, so the MSB compare (MaskFromVec) IS the natural conversion -- no
// vector-domain sign-broadcast to skip (avx2-only redundancy). Forward.
template <class V> HYLE_FORCE_INLINE auto MaskFromNormalizedVec(V v) noexcept {
  return MaskFromVec(v);
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
  if constexpr (sizeof(T) == 1) { const auto id = __riscv_vid_v_u8m1(vl); return n >= vl ? __riscv_vmseq(id, id, vl) : __riscv_vmsltu(id, static_cast<std::uint8_t>(n), vl); }
  else if constexpr (sizeof(T) == 2) { const auto id = __riscv_vid_v_u16m1(vl); return n >= vl ? __riscv_vmseq(id, id, vl) : __riscv_vmsltu(id, static_cast<std::uint16_t>(n), vl); }
  else if constexpr (sizeof(T) == 4) { const auto id = __riscv_vid_v_u32m1(vl); return n >= vl ? __riscv_vmseq(id, id, vl) : __riscv_vmsltu(id, static_cast<std::uint32_t>(n), vl); }
  else { const auto id = __riscv_vid_v_u64m1(vl); return n >= vl ? __riscv_vmseq(id, id, vl) : __riscv_vmsltu(id, static_cast<std::uint64_t>(n), vl); }
}
template <class D> HYLE_FORCE_INLINE VFromD<D> LoadN(D d, const TFromD<D>* p, std::size_t n) noexcept {
  using T = TFromD<D>;
  const std::size_t vl = n < Lanes(d) ? n : Lanes(d);
  const auto zero = Zero(d);
  if constexpr (std::is_same_v<T, float>) return __riscv_vle32_v_f32m1_tu(zero, p, vl);
  else if constexpr (std::is_same_v<T, double>) return __riscv_vle64_v_f64m1_tu(zero, p, vl);
  else if constexpr (sizeof(T) == 1 && std::is_signed_v<T>) return __riscv_vle8_v_i8m1_tu(zero, reinterpret_cast<const std::int8_t*>(p), vl);
  else if constexpr (sizeof(T) == 1) return __riscv_vle8_v_u8m1_tu(zero, reinterpret_cast<const std::uint8_t*>(p), vl);
  else if constexpr (sizeof(T) == 2 && std::is_signed_v<T>) return __riscv_vle16_v_i16m1_tu(zero, reinterpret_cast<const std::int16_t*>(p), vl);
  else if constexpr (sizeof(T) == 2) return __riscv_vle16_v_u16m1_tu(zero, reinterpret_cast<const std::uint16_t*>(p), vl);
  else if constexpr (sizeof(T) == 4 && std::is_signed_v<T>) return __riscv_vle32_v_i32m1_tu(zero, reinterpret_cast<const std::int32_t*>(p), vl);
  else if constexpr (sizeof(T) == 4) return __riscv_vle32_v_u32m1_tu(zero, reinterpret_cast<const std::uint32_t*>(p), vl);
  else if constexpr (std::is_signed_v<T>) return __riscv_vle64_v_i64m1_tu(zero, reinterpret_cast<const std::int64_t*>(p), vl);
  else return __riscv_vle64_v_u64m1_tu(zero, reinterpret_cast<const std::uint64_t*>(p), vl);
}
template <class D> HYLE_FORCE_INLINE void StoreN(VFromD<D> v, D d, TFromD<D>* p, std::size_t n) noexcept {
  using T = TFromD<D>;
  const std::size_t vl = n < Lanes(d) ? n : Lanes(d);
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

// ---- masked load / blended store (predicated, fault-safe) ---------------------------
template <class M, class D> HYLE_FORCE_INLINE VFromD<D> MaskedLoad(M m, D d, const TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  const std::size_t vl = Lanes(d);
  const auto zero = Zero(d);
  if constexpr (std::is_same_v<T, float>) return __riscv_vle32_v_f32m1_mu(m, zero, p, vl);
  else if constexpr (std::is_same_v<T, double>) return __riscv_vle64_v_f64m1_mu(m, zero, p, vl);
  else if constexpr (sizeof(T) == 4 && std::is_signed_v<T>) return __riscv_vle32_v_i32m1_mu(m, zero, reinterpret_cast<const std::int32_t*>(p), vl);
  else if constexpr (sizeof(T) == 4) return __riscv_vle32_v_u32m1_mu(m, zero, reinterpret_cast<const std::uint32_t*>(p), vl);
  else if constexpr (sizeof(T) == 8 && std::is_signed_v<T>) return __riscv_vle64_v_i64m1_mu(m, zero, reinterpret_cast<const std::int64_t*>(p), vl);
  else if constexpr (sizeof(T) == 8) return __riscv_vle64_v_u64m1_mu(m, zero, reinterpret_cast<const std::uint64_t*>(p), vl);
  else if constexpr (sizeof(T) == 2 && std::is_signed_v<T>) return __riscv_vle16_v_i16m1_mu(m, zero, reinterpret_cast<const std::int16_t*>(p), vl);
  else if constexpr (sizeof(T) == 2) return __riscv_vle16_v_u16m1_mu(m, zero, reinterpret_cast<const std::uint16_t*>(p), vl);
  else if constexpr (std::is_signed_v<T>) return __riscv_vle8_v_i8m1_mu(m, zero, reinterpret_cast<const std::int8_t*>(p), vl);
  else return __riscv_vle8_v_u8m1_mu(m, zero, reinterpret_cast<const std::uint8_t*>(p), vl);
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
  static_assert(sizeof(TFromV<VI>) == sizeof(T), "RVV gather index width must match lane width");
  const auto scaled = __riscv_vmul(idx, static_cast<TFromV<VI>>(sizeof(T)), vl);
  if constexpr (sizeof(T) == 4) {
    const vuint32m1_t boff = detail::bits_u32_rvv(scaled);
    if constexpr (std::is_same_v<T, float>) return __riscv_vluxei32_v_f32m1(base, boff, vl);
    else if constexpr (std::is_signed_v<T>) return __riscv_vluxei32_v_i32m1(reinterpret_cast<const std::int32_t*>(base), boff, vl);
    else return __riscv_vluxei32_v_u32m1(reinterpret_cast<const std::uint32_t*>(base), boff, vl);
  } else {
    const vuint64m1_t boff = detail::bits_u64_rvv(scaled);
    if constexpr (std::is_same_v<T, double>) return __riscv_vluxei64_v_f64m1(base, boff, vl);
    else if constexpr (std::is_signed_v<T>) return __riscv_vluxei64_v_i64m1(reinterpret_cast<const std::int64_t*>(base), boff, vl);
    else return __riscv_vluxei64_v_u64m1(reinterpret_cast<const std::uint64_t*>(base), boff, vl);
  }
}
template <class D, class VI>
HYLE_FORCE_INLINE void ScatterIndex(VFromD<D> v, D d, TFromD<D>* base, VI idx) noexcept {
  using T = TFromD<D>;
  const std::size_t vl = Lanes(d);
  static_assert(sizeof(TFromV<VI>) == sizeof(T), "RVV scatter index width must match lane width");
  const auto scaled = __riscv_vmul(idx, static_cast<TFromV<VI>>(sizeof(T)), vl);
  if constexpr (sizeof(T) == 4) {
    const vuint32m1_t boff = detail::bits_u32_rvv(scaled);
    if constexpr (std::is_same_v<T, float>) __riscv_vsuxei32_v_f32m1(base, boff, v, vl);
    else if constexpr (std::is_signed_v<T>) __riscv_vsuxei32_v_i32m1(reinterpret_cast<std::int32_t*>(base), boff, v, vl);
    else __riscv_vsuxei32_v_u32m1(reinterpret_cast<std::uint32_t*>(base), boff, v, vl);
  } else {
    const vuint64m1_t boff = detail::bits_u64_rvv(scaled);
    if constexpr (std::is_same_v<T, double>) __riscv_vsuxei64_v_f64m1(base, boff, v, vl);
    else if constexpr (std::is_signed_v<T>) __riscv_vsuxei64_v_i64m1(reinterpret_cast<std::int64_t*>(base), boff, v, vl);
    else __riscv_vsuxei64_v_u64m1(reinterpret_cast<std::uint64_t*>(base), boff, v, vl);
  }
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

// ---- Zvbb bit counts / reverse (simd-16 reclaim) ------------------------------------
// Base rv64gcv has no per-lane clz/ctz/brev/popcount -- all four are Zvbb. clz/ctz/brev have GCC-14
// C wrappers (vclz/vctz/vbrev); PopulationCount uses inline-asm vcpop.v (GCC-14 lacks that wrapper).
#if defined(__riscv_zvbb)
template <class V> HYLE_FORCE_INLINE V LeadingZeroCount(V v) noexcept {
  using T = TFromV<V>;
  static_assert(std::is_integral_v<T>, "LeadingZeroCount is integer-lane only");
  const std::size_t vl = detail::vlmax<T>();
  const RebindToUnsigned<DFromV<V>> du;
  const auto uv = BitCast(du, v);
  if constexpr (sizeof(T) == 1) return BitCast(DFromV<V>(), __riscv_vclz_v_u8m1(uv, vl));
  else if constexpr (sizeof(T) == 2) return BitCast(DFromV<V>(), __riscv_vclz_v_u16m1(uv, vl));
  else if constexpr (sizeof(T) == 4) return BitCast(DFromV<V>(), __riscv_vclz_v_u32m1(uv, vl));
  else return BitCast(DFromV<V>(), __riscv_vclz_v_u64m1(uv, vl));
}
template <class V> HYLE_FORCE_INLINE V TrailingZeroCount(V v) noexcept {
  using T = TFromV<V>;
  static_assert(std::is_integral_v<T>, "TrailingZeroCount is integer-lane only");
  const std::size_t vl = detail::vlmax<T>();
  const RebindToUnsigned<DFromV<V>> du;
  const auto uv = BitCast(du, v);
  if constexpr (sizeof(T) == 1) return BitCast(DFromV<V>(), __riscv_vctz_v_u8m1(uv, vl));
  else if constexpr (sizeof(T) == 2) return BitCast(DFromV<V>(), __riscv_vctz_v_u16m1(uv, vl));
  else if constexpr (sizeof(T) == 4) return BitCast(DFromV<V>(), __riscv_vctz_v_u32m1(uv, vl));
  else return BitCast(DFromV<V>(), __riscv_vctz_v_u64m1(uv, vl));
}
template <class V> HYLE_FORCE_INLINE V ReverseBits(V v) noexcept {
  using T = TFromV<V>;
  static_assert(std::is_integral_v<T>, "ReverseBits is integer-only");
  const std::size_t vl = detail::vlmax<T>();
  const RebindToUnsigned<DFromV<V>> du;
  const auto uv = BitCast(du, v);
  if constexpr (sizeof(T) == 1) return BitCast(DFromV<V>(), __riscv_vbrev_v_u8m1(uv, vl));
  else if constexpr (sizeof(T) == 2) return BitCast(DFromV<V>(), __riscv_vbrev_v_u16m1(uv, vl));
  else if constexpr (sizeof(T) == 4) return BitCast(DFromV<V>(), __riscv_vbrev_v_u32m1(uv, vl));
  else return BitCast(DFromV<V>(), __riscv_vbrev_v_u64m1(uv, vl));
}
// PopulationCount: Zvbb HAS the per-lane vcpop.v instruction, but GCC-14's riscv_vector.h ships no
// C wrapper (`__riscv_vcpop_v_*` absent; the short `vcpop` name is taken by mask-popcount). Emit the
// instruction directly via inline-asm -- verified under qemu (vse32.v readback = correct popcounts).
// vsetvli sets e<width>/m1 for the vcpop.v; the vl is vlmax for the lane type. `+zvbb` arch on the
// asm block so the assembler accepts vcpop.v regardless of the TU's -march order.
namespace detail {
template <class VU>
HYLE_FORCE_INLINE VU rvv_vcpop_u(VU uv, std::size_t vl, int sew) noexcept {
  VU r = uv;
  switch (sew) {
    case 8:
      asm volatile("vsetvli zero,%1,e8,m1,ta,ma\n.option arch,+zvbb\nvcpop.v %0,%2"
                   : "=vr"(r) : "r"(vl), "vr"(uv));
      break;
    case 16:
      asm volatile("vsetvli zero,%1,e16,m1,ta,ma\n.option arch,+zvbb\nvcpop.v %0,%2"
                   : "=vr"(r) : "r"(vl), "vr"(uv));
      break;
    case 32:
      asm volatile("vsetvli zero,%1,e32,m1,ta,ma\n.option arch,+zvbb\nvcpop.v %0,%2"
                   : "=vr"(r) : "r"(vl), "vr"(uv));
      break;
    default:
      asm volatile("vsetvli zero,%1,e64,m1,ta,ma\n.option arch,+zvbb\nvcpop.v %0,%2"
                   : "=vr"(r) : "r"(vl), "vr"(uv));
      break;
  }
  return r;
}
}  // namespace detail
template <class V> HYLE_FORCE_INLINE V PopulationCount(V v) noexcept {
  using T = TFromV<V>;
  static_assert(std::is_integral_v<T>, "PopulationCount is integer-lane only");
  const std::size_t vl = detail::vlmax<T>();
  const RebindToUnsigned<DFromV<V>> du;
  const auto uv = BitCast(du, v);
  return BitCast(DFromV<V>(), detail::rvv_vcpop_u(uv, vl, static_cast<int>(sizeof(T) * 8)));
}
#endif  // __riscv_zvbb

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
// Pair-local swizzles. vrgather indices i&~1 / i|1 duplicate adjacent even/odd lanes;
// i^1 swaps each pair without assuming a fixed VLEN.
template <class V> HYLE_FORCE_INLINE V DupEven(V v) noexcept {
  using T = TFromV<V>;
  using U = MakeUnsigned<T>;
  const std::size_t vl = detail::vlmax<T>();
  return __riscv_vrgather(v, __riscv_vand(detail::iota_u_rvv<T>(vl), static_cast<U>(~U{1}), vl), vl);
}
template <class V> HYLE_FORCE_INLINE V DupOdd(V v) noexcept {
  using T = TFromV<V>;
  using U = MakeUnsigned<T>;
  const std::size_t vl = detail::vlmax<T>();
  return __riscv_vrgather(v, __riscv_vor(detail::iota_u_rvv<T>(vl), U{1}, vl), vl);
}
template <class D> HYLE_FORCE_INLINE VFromD<D> Reverse2(D d, VFromD<D> v) noexcept {
  using T = TFromD<D>; using U = MakeUnsigned<T>;
  const std::size_t vl = Lanes(d);
  return __riscv_vrgather(v, __riscv_vxor(detail::iota_u_rvv<T>(vl), U{1}, vl), vl);
}
template <class D> HYLE_FORCE_INLINE VFromD<D> Reverse4(D d, VFromD<D> v) noexcept {
  using T = TFromD<D>; using U = MakeUnsigned<T>;
  const std::size_t vl = Lanes(d); HYLE_FATAL_ASSERT(vl >= 4 && vl % 4 == 0);
  return __riscv_vrgather(v, __riscv_vxor(detail::iota_u_rvv<T>(vl), U{3}, vl), vl);
}
template <class D> HYLE_FORCE_INLINE VFromD<D> Reverse8(D d, VFromD<D> v) noexcept {
  using T = TFromD<D>; using U = MakeUnsigned<T>;
  const std::size_t vl = Lanes(d); HYLE_FATAL_ASSERT(vl >= 8 && vl % 8 == 0);
  return __riscv_vrgather(v, __riscv_vxor(detail::iota_u_rvv<T>(vl), U{7}, vl), vl);
}
template <class V> HYLE_FORCE_INLINE V ReverseLaneBytes(V v) noexcept {
  using T = TFromV<V>;
  static_assert(std::is_integral_v<T>, "ReverseLaneBytes is integer-only");
  if constexpr (sizeof(T) == 1) return v;
  else {
    const std::size_t vl = detail::vlmax<std::uint8_t>();
    const auto idx = __riscv_vxor(__riscv_vid_v_u8m1(vl),
                                  static_cast<std::uint8_t>(sizeof(T) - 1), vl);
    return BitCast(DFromV<V>(), __riscv_vrgather(detail::bits_u8_rvv(v), idx, vl));
  }
}
template <class V> HYLE_FORCE_INLINE V InterleaveEven(V a, V b) noexcept {
  return OddEven(Slide1Up(DFromV<V>(), b), a);
}
template <class V> HYLE_FORCE_INLINE V InterleaveOdd(V a, V b) noexcept {
  return OddEven(b, Slide1Down(DFromV<V>(), a));
}

// Indices are native unsigned scalable vectors (sizeless types cannot be wrapped in a struct).
template <class D, class VI>
HYLE_FORCE_INLINE VFromD<RebindToUnsigned<D>> IndicesFromVec(D, VI indices) noexcept {
  static_assert(sizeof(TFromD<D>) == sizeof(TFromV<VI>), "Index/lane size mismatch");
  return BitCast(RebindToUnsigned<D>(), indices);
}
template <class V, class VI> HYLE_FORCE_INLINE V TableLookupLanes(V v, VI indices) noexcept {
  using U = TFromV<VI>;
  const std::size_t vl = detail::vlmax<TFromV<V>>();
  const auto wrapped = __riscv_vand(indices, static_cast<U>(vl - 1), vl);
  return __riscv_vrgather(v, wrapped, vl);
}
template <class V, class VI> HYLE_FORCE_INLINE VI TableLookupBytes(V table, VI indices) noexcept {
  const std::size_t vl = detail::vlmax<std::uint8_t>();
  const vuint8m1_t idx = detail::bits_u8_rvv(indices);
  const vuint8m1_t local = __riscv_vand(idx, std::uint8_t{15}, vl);
  const vuint8m1_t block = __riscv_vand(__riscv_vid_v_u8m1(vl), std::uint8_t{0xF0}, vl);
  const vuint8m1_t out = __riscv_vrgather(detail::bits_u8_rvv(table),
                                          __riscv_vadd(block, local, vl), vl);
  return BitCast(DFromV<VI>(), out);
}
template <class V, class VI> HYLE_FORCE_INLINE VI TableLookupBytesInRange(V table, VI indices) noexcept {
  return TableLookupBytes(table, indices);  // masked form is valid on the in-range domain
}
template <class V, class VI> HYLE_FORCE_INLINE VI TableLookupBytesOr0(V table, VI indices) noexcept {
  const std::size_t vl = detail::vlmax<std::uint8_t>();
  const vint8m1_t signed_idx = __riscv_vreinterpret_i8m1(detail::bits_u8_rvv(indices));
  const auto msb = __riscv_vmslt(signed_idx, std::int8_t{0}, vl);
  return IfThenElse(msb, Zero(DFromV<VI>()), TableLookupBytes(table, indices));
}
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
  using U = MakeUnsigned<T>;
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
template <class D, class V> HYLE_FORCE_INLINE V InterleaveUpper(D d, V a, V b) noexcept {
  const std::size_t half = Lanes(d) / 2;
  return InterleaveLower(SlideDownLanes(d, a, half), SlideDownLanes(d, b, half));
}

// Concat{Lower,Upper}{Lower,Upper}(d,hi,lo): whole-vector half-concat (Highway scalable).
// Same slide+FirstN+IfThenElse construction as SVE; RVV LMUL=1 so no Get/Set segment ops needed.
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

// LMUL=1 sizeless vectors keep the same native type for logical halves; D controls active lanes.
template <class D, class V>
HYLE_FORCE_INLINE VFromD<D> Combine(D d, V hi, V lo) noexcept {
  return ConcatLowerLower(d, hi, lo);
}
template <class D, class V>
HYLE_FORCE_INLINE VFromD<D> ZeroExtendVector(D d, V lo) noexcept {
  return Combine(d, Zero(typename D::Half{}), lo);
}
template <class DH, class V>
HYLE_FORCE_INLINE V LowerHalf(DH, V v) noexcept {
  return v;
}
template <class DH, class V>
HYLE_FORCE_INLINE V UpperHalf(DH dh, V v) noexcept {
  const typename DH::Twice d;
  return SlideDownLanes(d, v, Lanes(dh));
}

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
  using U = MakeUnsigned<T>;
  const std::size_t vl = detail::vlmax<T>();
  return __riscv_vrgather(v, Set(Simd<U, kNativeVectorBytes / sizeof(U), 0>(), static_cast<U>(kLane)), vl);
}
// MaskedLoadOr(no, m, d, p): load active lanes from p, inactive lanes take `no`.
template <class M, class D> HYLE_FORCE_INLINE VFromD<D> MaskedLoadOr(VFromD<D> no, M m, D d, const TFromD<D>* p) noexcept {
  return IfThenElse(m, MaskedLoad(m, d, p), no);
}

// ---- P2 widen / quantize / reduction completion -------------------------------------
template <typename DstT, std::size_t NDst, class V>
HYLE_FORCE_INLINE VFromD<Simd<DstT, NDst, 0>> PromoteEvenTo(Simd<DstT, NDst, 0> dd,
                                                            V v) noexcept {
  using SrcT = TFromV<V>;
  static_assert(sizeof(DstT) == 2 * sizeof(SrcT), "PromoteEvenTo requires adjacent widths");
  const std::size_t src_vl = detail::vlmax<SrcT>();
  const auto idx = __riscv_vsll(detail::iota_u_rvv<SrcT>(src_vl), 1u, src_vl);
  const V even = __riscv_vrgather(v, idx, src_vl);
  if constexpr (std::is_same_v<SrcT, float> && std::is_same_v<DstT, double>) {
    const auto half = __riscv_vlmul_trunc_v_f32m1_f32mf2(even);
    return __riscv_vfwcvt_f_f_v_f64m1(half, Lanes(dd));
  } else {
    static_assert(std::is_integral_v<SrcT> && std::is_integral_v<DstT>,
                  "PromoteEvenTo supports integers and f32->f64");
    if constexpr (std::is_signed_v<SrcT>) {
      const Simd<MakeSigned<DstT>, NDst, 0> ds;
      return BitCast(dd, PromoteLowerTo(ds, even));
    } else {
      const Simd<MakeUnsigned<DstT>, NDst, 0> du;
      return BitCast(dd, PromoteLowerTo(du, even));
    }
  }
}
template <typename DstT, std::size_t NDst, class V>
HYLE_FORCE_INLINE VFromD<Simd<DstT, NDst, 0>> PromoteOddTo(Simd<DstT, NDst, 0> dd,
                                                           V v) noexcept {
  using SrcT = TFromV<V>; using U = MakeUnsigned<SrcT>;
  const std::size_t src_vl = detail::vlmax<SrcT>();
  auto idx = __riscv_vsll(detail::iota_u_rvv<SrcT>(src_vl), 1u, src_vl);
  idx = __riscv_vadd(idx, U{1}, src_vl);
  const V odd = __riscv_vrgather(v, idx, src_vl);
  if constexpr (std::is_same_v<SrcT, float> && std::is_same_v<DstT, double>) {
    const auto half = __riscv_vlmul_trunc_v_f32m1_f32mf2(odd);
    return __riscv_vfwcvt_f_f_v_f64m1(half, Lanes(dd));
  } else if constexpr (std::is_signed_v<SrcT>) {
    const Simd<MakeSigned<DstT>, NDst, 0> ds;
    return BitCast(dd, PromoteLowerTo(ds, odd));
  } else {
    const Simd<MakeUnsigned<DstT>, NDst, 0> du;
    return BitCast(dd, PromoteLowerTo(du, odd));
  }
}

template <typename DstT, std::size_t N, int kPow2, class V>
HYLE_FORCE_INLINE VFromD<Simd<DstT, N, kPow2>> TruncateTo(Simd<DstT, N, kPow2> dd, V v) noexcept {
  using SrcT = TFromV<V>;
  static_assert(std::is_integral_v<DstT> && std::is_integral_v<SrcT> && sizeof(DstT) < sizeof(SrcT),
                "TruncateTo is modular integer narrowing");
  using U = MakeUnsigned<DstT>;
  constexpr U ratio = static_cast<U>(sizeof(SrcT) / sizeof(DstT));
  const std::size_t vl = Lanes(dd);
  const auto idx = __riscv_vmul(detail::iota_u_rvv<DstT>(vl), ratio, vl);
  return __riscv_vrgather(BitCast(dd, v), idx, vl);
}
HYLE_FORCE_INLINE vuint8m1_t U8FromU32(vuint32m1_t v) noexcept {
  using DS = DFromV<vuint32m1_t>;
  return TruncateTo(Rebind<std::uint8_t, DS>{}, v);
}

template <typename DstT, std::size_t NDst, class V>
HYLE_FORCE_INLINE VFromD<Simd<DstT, NDst, 0>> ReorderDemote2To(Simd<DstT, NDst, 0> d,
                                                               V a, V b) noexcept {
  return DemoteTo(d, a, b);
}
template <std::size_t NDst>
HYLE_FORCE_INLINE vint16m1_t SatWidenMulPairwiseAdd(Simd<std::int16_t, NDst, 0> di16,
                                                   vuint8m1_t a, vint8m1_t b) noexcept {
  const Simd<std::uint16_t, NDst, 0> du16;
  const auto ae = BitCast(di16, PromoteEvenTo(du16, a));
  const auto ao = BitCast(di16, PromoteOddTo(du16, a));
  return SaturatedAdd(Mul(ae, PromoteEvenTo(di16, b)), Mul(ao, PromoteOddTo(di16, b)));
}
template <std::size_t NDst>
HYLE_FORCE_INLINE vint32m1_t ReorderWidenMulAccumulate(Simd<std::int32_t, NDst, 0> d32,
                                                       vint16m1_t a, vint16m1_t b,
                                                       vint32m1_t sum0,
                                                       vint32m1_t& sum1) noexcept {
  (void)sum1;
  return Add(sum0, WidenMulPairwiseAdd(d32, a, b));
}
template <class V> HYLE_FORCE_INLINE V RearrangeToOddPlusEven(V sum0, V sum1) noexcept {
  return Add(sum0, sum1);
}
template <std::size_t NDst>
HYLE_FORCE_INLINE vint32m1_t SumOfMulQuadAccumulate(Simd<std::int32_t, NDst, 0> d32,
                                                   vint8m1_t a, vint8m1_t b,
                                                   vint32m1_t sum) noexcept {
  const Simd<std::int16_t, NDst * 2, 0> d16;
  const auto ae = PromoteEvenTo(d16, a), ao = PromoteOddTo(d16, a);
  const auto be = PromoteEvenTo(d16, b), bo = PromoteOddTo(d16, b);
  return Add(sum, Add(WidenMulPairwiseAdd(d32, ae, be), WidenMulPairwiseAdd(d32, ao, bo)));
}

HYLE_FORCE_INLINE vuint16m1_t SumsOf2(vuint8m1_t v) noexcept {
  const DFromV<vuint16m1_t> d;
  return Add(PromoteEvenTo(d, v), PromoteOddTo(d, v));
}
HYLE_FORCE_INLINE vuint32m1_t SumsOf4(vuint8m1_t v) noexcept {
  const DFromV<vuint32m1_t> d;
  const auto s2 = SumsOf2(v);
  return Add(PromoteEvenTo(d, s2), PromoteOddTo(d, s2));
}
HYLE_FORCE_INLINE vuint64m1_t SumsOf8(vuint8m1_t v) noexcept {
  const DFromV<vuint64m1_t> d;
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
HYLE_FORCE_INLINE vint16m1_t MulFixedPoint15(vint16m1_t a, vint16m1_t b) noexcept {
  const DFromV<vint16m1_t> d;
  const RebindToUnsigned<decltype(d)> du;
  const vuint16m1_t lo = BitCast(du, Mul(a, b));
  const vint16m1_t hi = MulHigh(a, b);
  const vuint16m1_t rounding = ShiftRight<1>(Add(ShiftRight<14>(lo), Set(du, 1)));
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
template <class V> HYLE_FORCE_INLINE V BroadcastSignBit(V v) noexcept {
  using T = TFromV<V>;
  static_assert(std::is_integral_v<T> && std::is_signed_v<T>, "BroadcastSignBit is signed-int only");
  return ShiftRight<static_cast<int>(sizeof(T) * 8) - 1>(v);
}
template <class D, class M>
HYLE_FORCE_INLINE auto ExclusiveNeither(D d, M a, M b) noexcept {
  return __riscv_vmnor(a, b, Lanes(d));  // ~(a|b)
}
template <class V> HYLE_FORCE_INLINE auto IsEitherNaN(V a, V b) noexcept {
  static_assert(std::is_floating_point_v<TFromV<V>>, "IsEitherNaN is float only");
  const std::size_t vl = detail::vlmax<TFromV<V>>();
  return __riscv_vmor(__riscv_vmfne(a, a, vl), __riscv_vmfne(b, b, vl), vl);  // NaN | NaN
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
template <class V> HYLE_FORCE_INLINE auto GetBiasedExponent(V v) noexcept {
  using T = TFromV<V>;
  static_assert(std::is_floating_point_v<T>, "GetBiasedExponent is float only");
  const RebindToUnsigned<DFromV<V>> du;
  const auto bits = BitCast(du, v);
  if constexpr (std::is_same_v<T, float>)
    return And(ShiftRight<23>(bits), Set(du, std::uint32_t{0xFFu}));
  else
    return And(ShiftRight<52>(bits), Set(du, std::uint64_t{0x7FFull}));
}
template <int kLane, class V> HYLE_FORCE_INLINE V BroadcastLane(V v) noexcept {
  return Broadcast<kLane>(v);
}

// Set*First family via vfirst + iota compare (RVV has no break instructions).
// no-true (vfirst < 0): Before/AtOrBefore -> all true; Only/AtOrAfter -> all false.
namespace detail {
template <class D, class M>
HYLE_FORCE_INLINE auto set_prefix_mask(D d, M m, int mode) noexcept {
  // mode: 0=Before, 1=AtOrBefore, 2=Only, 3=AtOrAfter
  using T = TFromD<D>;
  const std::size_t vl = Lanes(d);
  const std::ptrdiff_t f = __riscv_vfirst(m, vl);
  if constexpr (sizeof(T) == 1) {
    const auto id = __riscv_vid_v_u8m1(vl);
    if (f < 0) {
      if (mode == 0 || mode == 1) return __riscv_vmseq(id, id, vl);  // all true
      return __riscv_vmsne(id, id, vl);  // all false
    }
    const auto fu = static_cast<std::uint8_t>(f);
    if (mode == 0) return __riscv_vmsltu(id, fu, vl);
    if (mode == 1) return __riscv_vmsleu(id, fu, vl);
    if (mode == 2) return __riscv_vmseq(id, fu, vl);
    return __riscv_vmsgeu(id, fu, vl);
  } else if constexpr (sizeof(T) == 2) {
    const auto id = __riscv_vid_v_u16m1(vl);
    if (f < 0) {
      if (mode == 0 || mode == 1) return __riscv_vmseq(id, id, vl);
      return __riscv_vmsne(id, id, vl);
    }
    const auto fu = static_cast<std::uint16_t>(f);
    if (mode == 0) return __riscv_vmsltu(id, fu, vl);
    if (mode == 1) return __riscv_vmsleu(id, fu, vl);
    if (mode == 2) return __riscv_vmseq(id, fu, vl);
    return __riscv_vmsgeu(id, fu, vl);
  } else if constexpr (sizeof(T) == 4) {
    const auto id = __riscv_vid_v_u32m1(vl);
    if (f < 0) {
      if (mode == 0 || mode == 1) return __riscv_vmseq(id, id, vl);
      return __riscv_vmsne(id, id, vl);
    }
    const auto fu = static_cast<std::uint32_t>(f);
    if (mode == 0) return __riscv_vmsltu(id, fu, vl);
    if (mode == 1) return __riscv_vmsleu(id, fu, vl);
    if (mode == 2) return __riscv_vmseq(id, fu, vl);
    return __riscv_vmsgeu(id, fu, vl);
  } else {
    const auto id = __riscv_vid_v_u64m1(vl);
    if (f < 0) {
      if (mode == 0 || mode == 1) return __riscv_vmseq(id, id, vl);
      return __riscv_vmsne(id, id, vl);
    }
    const auto fu = static_cast<std::uint64_t>(f);
    if (mode == 0) return __riscv_vmsltu(id, fu, vl);
    if (mode == 1) return __riscv_vmsleu(id, fu, vl);
    if (mode == 2) return __riscv_vmseq(id, fu, vl);
    return __riscv_vmsgeu(id, fu, vl);
  }
}
}  // namespace detail
template <class D, class M> HYLE_FORCE_INLINE auto SetBeforeFirst(D d, M m) noexcept {
  return detail::set_prefix_mask(d, m, 0);
}
template <class D, class M> HYLE_FORCE_INLINE auto SetAtOrBeforeFirst(D d, M m) noexcept {
  return detail::set_prefix_mask(d, m, 1);
}
template <class D, class M> HYLE_FORCE_INLINE auto SetOnlyFirst(D d, M m) noexcept {
  return detail::set_prefix_mask(d, m, 2);
}
template <class D, class M> HYLE_FORCE_INLINE auto SetAtOrAfterFirst(D d, M m) noexcept {
  return detail::set_prefix_mask(d, m, 3);
}

template <class D, class M>
HYLE_FORCE_INLINE std::size_t FindKnownFirstTrue(D d, M m) noexcept {
  return static_cast<std::size_t>(__riscv_vfirst(m, Lanes(d)));
}
template <class D, class M>
HYLE_FORCE_INLINE std::size_t FindKnownLastTrue(D d, M m) noexcept {
  using T = TFromD<D>;
  const std::size_t vl = Lanes(d);
  // No -1 check: caller guarantees a true lane exists.
  if constexpr (sizeof(T) == 1) {
    auto id = __riscv_vid_v_u8m1(vl);
    auto sel = __riscv_vmerge(__riscv_vmv_v_x_u8m1(0, vl), id, m, vl);
    return static_cast<std::size_t>(__riscv_vmv_x(__riscv_vredmaxu(sel, sel, vl)));
  } else if constexpr (sizeof(T) == 2) {
    auto id = __riscv_vid_v_u16m1(vl);
    auto sel = __riscv_vmerge(__riscv_vmv_v_x_u16m1(0, vl), id, m, vl);
    return static_cast<std::size_t>(__riscv_vmv_x(__riscv_vredmaxu(sel, sel, vl)));
  } else if constexpr (sizeof(T) == 4) {
    auto id = __riscv_vid_v_u32m1(vl);
    auto sel = __riscv_vmerge(__riscv_vmv_v_x_u32m1(0, vl), id, m, vl);
    return static_cast<std::size_t>(__riscv_vmv_x(__riscv_vredmaxu(sel, sel, vl)));
  } else {
    auto id = __riscv_vid_v_u64m1(vl);
    auto sel = __riscv_vmerge(__riscv_vmv_v_x_u64m1(0, vl), id, m, vl);
    return static_cast<std::size_t>(__riscv_vmv_x(__riscv_vredmaxu(sel, sel, vl)));
  }
}

// ---- P5 batch C: MulByPow2 / Expand -------------------------------------------------
// MulByPow2(a, b) = a * 2^b. PORTED from Highway generic_ops (see avx2.h for full rationale):
// 3-part clamped exponent split, each part built as a 2^eK float factor via exponent-bit packing,
// then multiplied -- entirely in-vector, no store+scalar-ldexp+reload. RVV has native vmin/vmax
// (Clamp) and vsll for every SEW, so this stays fully vectorized on the scalable backend too.
// Correctly rounded through denormal/overflow. b is the matching-width signed int vector.
template <class V, class VI> HYLE_FORCE_INLINE V MulByPow2(V a, VI b) noexcept {
  using T = TFromV<V>;
  static_assert(std::is_floating_point_v<T>, "MulByPow2 is float only");
  using IT = TFromV<VI>;
  static_assert(std::is_integral_v<IT> && std::is_signed_v<IT> && sizeof(IT) == sizeof(T),
                "MulByPow2 exponent is a signed int lane matching the float width");
  const DFromV<V> df;
  const DFromV<VI> di;
  constexpr int kMantBits = (sizeof(T) == 4) ? 23 : 52;
  constexpr IT kBias = (sizeof(T) == 4) ? IT{127} : IT{1023};
  const VI clamped = Clamp(b, Set(di, static_cast<IT>(3 - 3 * kBias)), Set(di, static_cast<IT>(3 * kBias)));
  const VI lo = Set(di, static_cast<IT>(1 - kBias));
  const VI hi = Set(di, kBias);
  const VI e3 = Clamp(clamped, lo, hi);
  const VI e2 = Clamp(Sub(clamped, e3), lo, hi);
  const VI e1 = Sub(Sub(clamped, e3), e2);
  const VI biasv = Set(di, kBias);
  const V f1 = BitCast(df, ShiftLeft<kMantBits>(Add(e1, biasv)));
  const V f2 = BitCast(df, ShiftLeft<kMantBits>(Add(e2, biasv)));
  const V f3 = BitCast(df, ShiftLeft<kMantBits>(Add(e3, biasv)));
  return Mul(Mul(Mul(a, f1), f2), f3);
}
// Expand(v, m): scatter v's low CountTrue(m) lanes to the m-true positions, 0 elsewhere (inverse of
// Compress). RVV is clean: viota(m) gives each lane the EXCLUSIVE prefix-count of active lanes below
// it (the source index for a true lane); vrgather pulls v[iota[i]], then merge 0 into inactive lanes.
template <class V, class M> HYLE_FORCE_INLINE V Expand(V v, M m) noexcept {
  using T = TFromV<V>;
  const std::size_t vl = detail::vlmax<T>();
  // viota(m)[i] = number of active lanes strictly below i = the source index a true lane pulls from.
  // (No overloaded short name in GCC-14's header -> per-SEW full name.) vrgather then 0-merges.
  const V z = Zero(DFromV<V>());
  if constexpr (sizeof(T) == 1) {
    const auto idx = __riscv_viota_m_u8m1(m, vl);
    return __riscv_vmerge(z, __riscv_vrgather(v, idx, vl), m, vl);
  } else if constexpr (sizeof(T) == 2) {
    const auto idx = __riscv_viota_m_u16m1(m, vl);
    return __riscv_vmerge(z, __riscv_vrgather(v, idx, vl), m, vl);
  } else if constexpr (sizeof(T) == 4) {
    const auto idx = __riscv_viota_m_u32m1(m, vl);
    return __riscv_vmerge(z, __riscv_vrgather(v, idx, vl), m, vl);
  } else {
    const auto idx = __riscv_viota_m_u64m1(m, vl);
    return __riscv_vmerge(z, __riscv_vrgather(v, idx, vl), m, vl);
  }
}

// ---- Zvbc carryless multiply (simd-16 reclaim) --------------------------------------
// HYLE/x86 CLMul operates on 128-bit blocks: Lower multiplies even u64 lanes, Upper the odd
// ones; the 128-bit product is written back into that block as [lo, hi]. RVV vclmul/vclmulh are
// per-u64-lane, so DupEven/DupOdd broadcast the selected operand, then OddEven interleaves
// lo/hi into the block layout.
#if defined(__riscv_zvbc)
template <class V> HYLE_FORCE_INLINE V CLMulLower(V a, V b) noexcept {
  using T = TFromV<V>;
  static_assert(std::is_integral_v<T> && sizeof(T) == 8, "CLMul is 64-bit-lane only");
  const std::size_t vl = detail::vlmax<T>();
  const RebindToUnsigned<DFromV<V>> du;
  const auto au = BitCast(du, a);
  const auto bu = BitCast(du, b);
  const auto ae = DupEven(au);
  const auto be = DupEven(bu);
  const auto lo = __riscv_vclmul_vv_u64m1(ae, be, vl);
  const auto hi = __riscv_vclmulh_vv_u64m1(ae, be, vl);
  return BitCast(DFromV<V>(), OddEven(hi, lo));  // even<-lo, odd<-hi
}
template <class V> HYLE_FORCE_INLINE V CLMulUpper(V a, V b) noexcept {
  using T = TFromV<V>;
  static_assert(std::is_integral_v<T> && sizeof(T) == 8, "CLMul is 64-bit-lane only");
  const std::size_t vl = detail::vlmax<T>();
  const RebindToUnsigned<DFromV<V>> du;
  const auto au = BitCast(du, a);
  const auto bu = BitCast(du, b);
  const auto ao = DupOdd(au);
  const auto bo = DupOdd(bu);
  const auto lo = __riscv_vclmul_vv_u64m1(ao, bo, vl);
  const auto hi = __riscv_vclmulh_vv_u64m1(ao, bo, vl);
  return BitCast(DFromV<V>(), OddEven(hi, lo));
}
#endif  // __riscv_zvbc

// ---- mask <-> packed bits (BYTE-BUFFER form). RVV VLEN can reach 65536-bit (RV spec max) so a
// mask never fits a fixed word in general; the portable API is a byte buffer, 1 bit/lane (like
// Highway). VL-agnostic via VecFromMask store + scalar bit-pack over runtime Lanes(d). ----
template <class D, class M>
HYLE_FORCE_INLINE std::size_t StoreMaskBits(D d, M m, std::uint8_t* bits) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t kMax = D::kMaxLanes;
  alignas(64) T lanes[kMax];
  Store(VecFromMask(d, m), d, lanes);  // all-ones where active, else 0
  const std::size_t n = Lanes(d);
  const std::size_t nbytes = (n + 7) / 8;
  for (std::size_t b = 0; b < nbytes; ++b) bits[b] = 0;
  for (std::size_t i = 0; i < n; ++i)
    if (lanes[i] != T{0}) bits[i >> 3] |= static_cast<std::uint8_t>(1u << (i & 7));
  return nbytes;
}
template <class D>
HYLE_FORCE_INLINE auto LoadMaskBits(D d, const std::uint8_t* bits) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t kMax = D::kMaxLanes;
  alignas(64) T lanes[kMax];
  const std::size_t n = Lanes(d);
  for (std::size_t i = 0; i < n; ++i)
    lanes[i] = (bits[i >> 3] >> (i & 7)) & 1u ? static_cast<T>(~T{0}) : T{0};
  return MaskFromVec(Load(d, lanes));
}
template <class D, class M>
HYLE_FORCE_INLINE std::uint64_t BitsFromMask(D d, M m) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t kMax = D::kMaxLanes;
  alignas(64) T lanes[kMax];
  Store(VecFromMask(d, m), d, lanes);
  const std::size_t n = Lanes(d);
  HYLE_FATAL_ASSERT(n <= 64);  // wider VL must use the StoreMaskBits byte buffer
  std::uint64_t bits = 0;
  for (std::size_t i = 0; i < n; ++i)
    if (lanes[i] != T{0}) bits |= (std::uint64_t{1} << i);
  return bits;
}

}  // namespace hyle::simd
