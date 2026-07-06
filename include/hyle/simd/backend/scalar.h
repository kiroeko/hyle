#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <type_traits>

#include "hyle/macros.h"

// backend/scalar.h -- the portable scalar FLOOR backend (simd-design landing list #1).
//
// Every op is a plain per-lane C++ loop: no intrinsics, compiles everywhere, always
// available. Two jobs: (1) make `simd/` run on any CPU even with zero SIMD; (2) be the
// REFERENCE the SSE2/AVX2/AVX512/NEON backends pair against (simd-design landing list
// #6: "each backend pairs against scalar"). Correctness here is by inspection -- it's
// just the definition of each op -- so a wide backend that disagrees lane-for-lane with
// scalar is the wide backend's bug.
//
// Vec<T,N> mirrors Highway's emu128 Vec128: a small struct wrapping T raw[N]. Mask<T,N>
// is a bool-per-lane array (Highway packs bits; a bool array is clearer for the floor
// and IfThenElse-equivalence is what the cross-backend tests actually compare, not the
// mask's internal representation).

namespace hyle::simd {

// ---- vector + mask storage ----------------------------------------------------------
template <typename T, std::size_t N>
struct Vec {
  T raw[N] = {};
  using PrivateT = T;
  static constexpr std::size_t kPrivateN = N;
};

template <typename T, std::size_t N>
struct Mask {
  bool raw[N] = {};
  using PrivateT = T;
  static constexpr std::size_t kPrivateN = N;
};

// Recover the tag / lane type from a vector (Highway DFromV/TFromV).
template <class V>
using DFromV = Simd<typename V::PrivateT, V::kPrivateN, 0>;
template <class V>
using TFromV = typename V::PrivateT;

// ---- initialization -----------------------------------------------------------------

// All lanes = t.
template <class D, typename T2>
HYLE_FORCE_INLINE Vec<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> Set(D, T2 t) noexcept {
  using T = TFromD<D>;
  Vec<T, HYLE_SIMD_MAX_LANES_D(D)> v;
  for (std::size_t i = 0; i < HYLE_SIMD_MAX_LANES_D(D); ++i)
    v.raw[i] = static_cast<T>(t);
  return v;
}

// All lanes = 0.
template <class D>
HYLE_FORCE_INLINE Vec<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> Zero(D d) noexcept {
  return Set(d, static_cast<TFromD<D>>(0));
}

// VFromD<D> is the vector type Zero(D) yields (Highway's definition). Defined after Zero
// for the same reason Highway defines it after Zero: it depends on Zero's return type.
template <class D>
using VFromD = decltype(Zero(D()));

// ---- load / store -------------------------------------------------------------------
// Aligned (Load/Store) vs unaligned (LoadU/StoreU) carry the same contract as Highway:
// Load assumes the pointer is Lanes()*sizeof(T)-aligned (caller's job; mem/ gives 64B).
// The scalar floor copies the same either way -- the distinction matters only once a
// real backend emits aligned vs unaligned move instructions.
template <class D>
HYLE_FORCE_INLINE VFromD<D> LoadU(D, const TFromD<D>* p) noexcept {
  VFromD<D> v;
  std::memcpy(v.raw, p, HYLE_SIMD_MAX_LANES_D(D) * sizeof(TFromD<D>));
  return v;
}
template <class D>
HYLE_FORCE_INLINE VFromD<D> Load(D d, const TFromD<D>* p) noexcept {
  return LoadU(d, p);
}
template <class D>
HYLE_FORCE_INLINE void StoreU(VFromD<D> v, D, TFromD<D>* p) noexcept {
  std::memcpy(p, v.raw, HYLE_SIMD_MAX_LANES_D(D) * sizeof(TFromD<D>));
}
template <class D>
HYLE_FORCE_INLINE void Store(VFromD<D> v, D d, TFromD<D>* p) noexcept {
  StoreU(v, d, p);
}

// ---- arithmetic ---------------------------------------------------------------------
#define HYLE_SIMD_SCALAR_BINOP(NAME, EXPR)                                  \
  template <typename T, std::size_t N>                                      \
  HYLE_FORCE_INLINE Vec<T, N> NAME(Vec<T, N> a, Vec<T, N> b) noexcept {     \
    Vec<T, N> r;                                                            \
    for (std::size_t i = 0; i < N; ++i) {                                   \
      const T x = a.raw[i];                                                 \
      const T y = b.raw[i];                                                 \
      r.raw[i] = (EXPR);                                                    \
    }                                                                       \
    return r;                                                               \
  }

HYLE_SIMD_SCALAR_BINOP(Add, static_cast<T>(x + y))
HYLE_SIMD_SCALAR_BINOP(Sub, static_cast<T>(x - y))
HYLE_SIMD_SCALAR_BINOP(Mul, static_cast<T>(x * y))
HYLE_SIMD_SCALAR_BINOP(Div, static_cast<T>(x / y))
HYLE_SIMD_SCALAR_BINOP(Min, (x < y ? x : y))
HYLE_SIMD_SCALAR_BINOP(Max, (x > y ? x : y))
#undef HYLE_SIMD_SCALAR_BINOP

template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> Neg(Vec<T, N> a) noexcept {
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i) r.raw[i] = static_cast<T>(-a.raw[i]);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> Abs(Vec<T, N> a) noexcept {
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i) {
    const T x = a.raw[i];
    r.raw[i] = x < T{0} ? static_cast<T>(-x) : x;
  }
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> Sqrt(Vec<T, N> a) noexcept {
  static_assert(std::is_floating_point_v<T>, "Sqrt is float/double only");
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i) r.raw[i] = std::sqrt(a.raw[i]);
  return r;
}

// Float rounding to an integral value, result stays float: Round=nearest-even,
// Floor=toward -inf, Ceil=toward +inf, Trunc=toward zero. The scalar floor is the libm
// call the wide backends' _mm256_round_ps / vrndnq etc. must match lane-for-lane.
#define HYLE_SIMD_SCALAR_ROUND(NAME, FN)                                  \
  template <typename T, std::size_t N>                                    \
  HYLE_FORCE_INLINE Vec<T, N> NAME(Vec<T, N> a) noexcept {                \
    static_assert(std::is_floating_point_v<T>, #NAME " is float/double only"); \
    Vec<T, N> r;                                                          \
    for (std::size_t i = 0; i < N; ++i) r.raw[i] = FN(a.raw[i]);          \
    return r;                                                             \
  }
HYLE_SIMD_SCALAR_ROUND(Round, std::nearbyint)  // round-to-nearest-even (default mode)
HYLE_SIMD_SCALAR_ROUND(Floor, std::floor)
HYLE_SIMD_SCALAR_ROUND(Ceil, std::ceil)
HYLE_SIMD_SCALAR_ROUND(Trunc, std::trunc)
#undef HYLE_SIMD_SCALAR_ROUND


// Fused multiply-add family (a*b (+/-) c). Scalar uses the plain expression; a real
// backend uses an FMA instruction (single rounding). Tests tolerate the 1-ulp gap for
// float by comparing with a small epsilon, not bit-equality.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> MulAdd(Vec<T, N> a, Vec<T, N> b, Vec<T, N> c) noexcept {
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i)
    r.raw[i] = static_cast<T>(a.raw[i] * b.raw[i] + c.raw[i]);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> NegMulAdd(Vec<T, N> a, Vec<T, N> b, Vec<T, N> c) noexcept {
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i)
    r.raw[i] = static_cast<T>(-(a.raw[i] * b.raw[i]) + c.raw[i]);
  return r;
}

// ---- bitwise (operate on the bit pattern; works for float via memcpy) ---------------
namespace detail {
template <typename T>
HYLE_FORCE_INLINE T bit_binop(T a, T b, int op) noexcept {
  using U = std::make_unsigned_t<
      std::conditional_t<sizeof(T) == 4, std::uint32_t,
          std::conditional_t<sizeof(T) == 8, std::uint64_t,
              std::conditional_t<sizeof(T) == 2, std::uint16_t, std::uint8_t>>>>;
  U ua, ub;
  std::memcpy(&ua, &a, sizeof(T));
  std::memcpy(&ub, &b, sizeof(T));
  U ur = op == 0 ? (ua & ub) : op == 1 ? (ua | ub) : op == 2 ? (ua ^ ub) : (~ua & ub);
  T r;
  std::memcpy(&r, &ur, sizeof(T));
  return r;
}
}  // namespace detail

#define HYLE_SIMD_SCALAR_BITOP(NAME, OP)                                \
  template <typename T, std::size_t N>                                  \
  HYLE_FORCE_INLINE Vec<T, N> NAME(Vec<T, N> a, Vec<T, N> b) noexcept { \
    Vec<T, N> r;                                                        \
    for (std::size_t i = 0; i < N; ++i)                                 \
      r.raw[i] = detail::bit_binop(a.raw[i], b.raw[i], OP);             \
    return r;                                                           \
  }
HYLE_SIMD_SCALAR_BITOP(And, 0)
HYLE_SIMD_SCALAR_BITOP(Or, 1)
HYLE_SIMD_SCALAR_BITOP(Xor, 2)
HYLE_SIMD_SCALAR_BITOP(AndNot, 3)  // AndNot(a,b) = ~a & b (Highway order)
#undef HYLE_SIMD_SCALAR_BITOP

// ---- shifts (integer lanes only) ----------------------------------------------------
// ShiftLeft<k>/ShiftRight<k>: compile-time immediate count (Highway names). ShiftRight on
// a SIGNED lane is ARITHMETIC (sign-extending, matches C++ >> on signed); on UNSIGNED it
// is LOGICAL (zero-fill). k must be in [0, bits-of-T). The scalar floor is just the C++
// operator on the (already-signed-or-unsigned) lane; a real backend uses _mm_slli/srli/
// srai / vshlq. Shl/Shr (per-lane VARIABLE count from a vector) mirror the same semantics.
template <int kBits, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> ShiftLeft(Vec<T, N> a) noexcept {
  static_assert(std::is_integral_v<T>, "shifts are integer-lane only");
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i)
    r.raw[i] = static_cast<T>(static_cast<std::make_unsigned_t<T>>(a.raw[i]) << kBits);
  return r;
}
template <int kBits, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> ShiftRight(Vec<T, N> a) noexcept {
  static_assert(std::is_integral_v<T>, "shifts are integer-lane only");
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i) r.raw[i] = static_cast<T>(a.raw[i] >> kBits);  // arith if signed
  return r;
}

// ---- saturated arithmetic + rounding average (8/16-bit integer lanes) ---------------
// SaturatedAdd/Sub clamp to [min,max] of T instead of wrapping -- the quantization
// primitive (add i8 activations without overflow). AverageRound = (a+b+1)>>1 unsigned
// rounded average (pooling / blend). Native on x86 (adds/subs/avg_epu*) and NEON
// (vqadd/vqsub/vrhadd); the scalar floor computes in a wider type then clamps/rounds.
namespace detail {
template <typename T>
HYLE_FORCE_INLINE T sat_clamp(long long x) noexcept {
  constexpr long long lo = static_cast<long long>(std::numeric_limits<T>::min());
  constexpr long long hi = static_cast<long long>(std::numeric_limits<T>::max());
  return static_cast<T>(x < lo ? lo : (x > hi ? hi : x));
}
}  // namespace detail
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> SaturatedAdd(Vec<T, N> a, Vec<T, N> b) noexcept {
  static_assert(std::is_integral_v<T> && sizeof(T) <= 2, "SaturatedAdd is 8/16-bit int only");
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i)
    r.raw[i] = detail::sat_clamp<T>(static_cast<long long>(a.raw[i]) + b.raw[i]);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> SaturatedSub(Vec<T, N> a, Vec<T, N> b) noexcept {
  static_assert(std::is_integral_v<T> && sizeof(T) <= 2, "SaturatedSub is 8/16-bit int only");
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i)
    r.raw[i] = detail::sat_clamp<T>(static_cast<long long>(a.raw[i]) - b.raw[i]);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> AverageRound(Vec<T, N> a, Vec<T, N> b) noexcept {
  static_assert(std::is_unsigned_v<T> && sizeof(T) <= 2, "AverageRound is u8/u16 only");
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i)
    r.raw[i] = static_cast<T>((static_cast<unsigned>(a.raw[i]) + b.raw[i] + 1u) >> 1);
  return r;
}

// ---- type conversion (same lane count, different lane type) -------------------------
// ConvertTo(dst_tag, v): float<->int, TRUNCATING toward zero (C++ cast semantics), same N.
// RoundToInt(dst_tag, v): f32/f64 -> int, round to NEAREST (ties to even). PromoteTo/
// DemoteTo change lane WIDTH (i8->i16 etc); a real backend uses cvtepi/packs -- the scalar
// floor is a per-lane static_cast into the destination lane type. The dst tag Simd<U,N>
// names the target type + (same) lane count. Quantization core: f32<->i32 (de)quantize.
template <typename DstT, std::size_t N, typename SrcT>
HYLE_FORCE_INLINE Vec<DstT, N> ConvertTo(Simd<DstT, N, 0>, Vec<SrcT, N> v) noexcept {
  Vec<DstT, N> r;
  for (std::size_t i = 0; i < N; ++i) r.raw[i] = static_cast<DstT>(v.raw[i]);  // trunc if f->i
  return r;
}
template <typename DstT, std::size_t N, typename SrcT>
HYLE_FORCE_INLINE Vec<DstT, N> RoundToInt(Simd<DstT, N, 0>, Vec<SrcT, N> v) noexcept {
  static_assert(std::is_floating_point_v<SrcT> && std::is_integral_v<DstT>,
                "RoundToInt is float->int");
  Vec<DstT, N> r;
  for (std::size_t i = 0; i < N; ++i)
    r.raw[i] = static_cast<DstT>(std::nearbyint(v.raw[i]));  // round-to-nearest-even
  return r;
}

// NOTE: PromoteTo/DemoteTo (i8<->i16<->i32 width-changing) are DEFERRED: on a fixed-
// native-width backend they need PARTIAL / multi-register vectors (promote 16 i8 lanes in
// a 128-bit half to 16 i16 lanes in a full 256-bit reg), which the v1 backends don't model
// (vec.h drops Highway's fractional-N until a scalable backend lands). Same-width
// float<->int (ConvertTo/RoundToInt above) needs no such support and is the quantization
// (de)quant core. Width-changing promote/demote is a separate effort (needs vec.h
// partial-vector support first). Registered in project_simd_int_lanes memory.

// ---- comparison -> mask -------------------------------------------------------------
#define HYLE_SIMD_SCALAR_CMP(NAME, REL)                                  \
  template <typename T, std::size_t N>                                   \
  HYLE_FORCE_INLINE Mask<T, N> NAME(Vec<T, N> a, Vec<T, N> b) noexcept { \
    Mask<T, N> m;                                                        \
    for (std::size_t i = 0; i < N; ++i) m.raw[i] = (a.raw[i] REL b.raw[i]); \
    return m;                                                            \
  }
HYLE_SIMD_SCALAR_CMP(Eq, ==)
HYLE_SIMD_SCALAR_CMP(Ne, !=)
HYLE_SIMD_SCALAR_CMP(Lt, <)
HYLE_SIMD_SCALAR_CMP(Le, <=)
HYLE_SIMD_SCALAR_CMP(Gt, >)
HYLE_SIMD_SCALAR_CMP(Ge, >=)
#undef HYLE_SIMD_SCALAR_CMP

// ---- mask ops -----------------------------------------------------------------------
// Blend: per lane, mask ? yes : no.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> IfThenElse(Mask<T, N> m, Vec<T, N> yes,
                                       Vec<T, N> no) noexcept {
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i) r.raw[i] = m.raw[i] ? yes.raw[i] : no.raw[i];
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> IfThenElseZero(Mask<T, N> m, Vec<T, N> yes) noexcept {
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i) r.raw[i] = m.raw[i] ? yes.raw[i] : T{0};
  return r;
}

template <typename T, std::size_t N>
HYLE_FORCE_INLINE bool AllTrue(DFromV<Vec<T, N>>, Mask<T, N> m) noexcept {
  for (std::size_t i = 0; i < N; ++i)
    if (!m.raw[i]) return false;
  return true;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE bool AllFalse(DFromV<Vec<T, N>>, Mask<T, N> m) noexcept {
  for (std::size_t i = 0; i < N; ++i)
    if (m.raw[i]) return false;
  return true;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE std::size_t CountTrue(DFromV<Vec<T, N>>, Mask<T, N> m) noexcept {
  std::size_t n = 0;
  for (std::size_t i = 0; i < N; ++i) n += m.raw[i] ? 1 : 0;
  return n;
}
// Lane index of the first set lane, or -1 if none (Highway FindFirstTrue).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE std::ptrdiff_t FindFirstTrue(DFromV<Vec<T, N>>,
                                               Mask<T, N> m) noexcept {
  for (std::size_t i = 0; i < N; ++i)
    if (m.raw[i]) return static_cast<std::ptrdiff_t>(i);
  return -1;
}

// ---- reductions (return the scalar, Highway's GetLane(SumOfLanes(...)) shorthand) ---
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceSum(DFromV<Vec<T, N>>, Vec<T, N> v) noexcept {
  T s = T{0};
  for (std::size_t i = 0; i < N; ++i) s = static_cast<T>(s + v.raw[i]);
  return s;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceMin(DFromV<Vec<T, N>>, Vec<T, N> v) noexcept {
  T m = v.raw[0];
  for (std::size_t i = 1; i < N; ++i) m = v.raw[i] < m ? v.raw[i] : m;
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceMax(DFromV<Vec<T, N>>, Vec<T, N> v) noexcept {
  T m = v.raw[0];
  for (std::size_t i = 1; i < N; ++i) m = v.raw[i] > m ? v.raw[i] : m;
  return m;
}
// ReduceMul (product), ReduceAnd/Or (bitwise, integer lanes) -- complete the reduction
// family (Sum/Min/Max above). Highway names GetLane(SumOfLanes/... ). Horizontal reduce
// is not a hot path, so even the wide backends fold via a stack array (see their notes).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceMul(DFromV<Vec<T, N>>, Vec<T, N> v) noexcept {
  T p = v.raw[0];
  for (std::size_t i = 1; i < N; ++i) p = static_cast<T>(p * v.raw[i]);
  return p;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceAnd(DFromV<Vec<T, N>>, Vec<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "ReduceAnd is integer-lane only");
  T a = v.raw[0];
  for (std::size_t i = 1; i < N; ++i) a = static_cast<T>(a & v.raw[i]);
  return a;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceOr(DFromV<Vec<T, N>>, Vec<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "ReduceOr is integer-lane only");
  T o = v.raw[0];
  for (std::size_t i = 1; i < N; ++i) o = static_cast<T>(o | v.raw[i]);
  return o;
}

// ---- swizzle: Broadcast<i> (splat lane i across all lanes; Highway name) ------------
// The one swizzle v1 needs (matrix-row / bias splat). Index-vector-dependent swizzles
// (TableLookup, Gather/Scatter, Compress, Interleave/Zip) are DEFERRED -- see the note at
// the top of the reductions region and project_simd_int_lanes memory: they need a portable
// index-vector type + per-ISA permute/gather units, a separate effort.
template <int kLane, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> Broadcast(Vec<T, N> v) noexcept {
  static_assert(kLane >= 0 && static_cast<std::size_t>(kLane) < N, "Broadcast lane out of range");
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i) r.raw[i] = v.raw[kLane];
  return r;
}

// Lane 0 (Highway GetLane).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T GetLane(Vec<T, N> v) noexcept {
  return v.raw[0];
}

// ---- masked / partial load-store (tail handling) ------------------------------------
// The vectorized-loop remainder toolkit (Highway names). FirstN(d,n) builds a mask with
// the first min(n,N) lanes true; MaskedLoad zeroes the off lanes; BlendedStore writes
// only the on lanes; LoadN/StoreN take a runtime count directly. The scalar floor is the
// per-lane reference the wide backends' vmaskmov / k-mask load-store must match.
template <class D>
HYLE_FORCE_INLINE Mask<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> FirstN(D, std::size_t n) noexcept {
  Mask<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m;
  for (std::size_t i = 0; i < HYLE_SIMD_MAX_LANES_D(D); ++i) m.raw[i] = (i < n);
  return m;
}
template <class D>
HYLE_FORCE_INLINE VFromD<D> MaskedLoad(Mask<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m, D,
                                       const TFromD<D>* p) noexcept {
  VFromD<D> v;
  for (std::size_t i = 0; i < HYLE_SIMD_MAX_LANES_D(D); ++i)
    v.raw[i] = m.raw[i] ? p[i] : TFromD<D>{0};
  return v;
}
template <class D>
HYLE_FORCE_INLINE VFromD<D> MaskedLoadOr(VFromD<D> no,
                                         Mask<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m, D,
                                         const TFromD<D>* p) noexcept {
  VFromD<D> v;
  for (std::size_t i = 0; i < HYLE_SIMD_MAX_LANES_D(D); ++i)
    v.raw[i] = m.raw[i] ? p[i] : no.raw[i];
  return v;
}
template <class D>
HYLE_FORCE_INLINE void BlendedStore(VFromD<D> v, Mask<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m,
                                    D, TFromD<D>* p) noexcept {
  for (std::size_t i = 0; i < HYLE_SIMD_MAX_LANES_D(D); ++i)
    if (m.raw[i]) p[i] = v.raw[i];
}
template <class D>
HYLE_FORCE_INLINE VFromD<D> LoadN(D d, const TFromD<D>* p, std::size_t n) noexcept {
  const std::size_t k = n < HYLE_SIMD_MAX_LANES_D(D) ? n : HYLE_SIMD_MAX_LANES_D(D);
  VFromD<D> v = Zero(d);
  for (std::size_t i = 0; i < k; ++i) v.raw[i] = p[i];
  return v;
}
template <class D>
HYLE_FORCE_INLINE void StoreN(VFromD<D> v, D, TFromD<D>* p, std::size_t n) noexcept {
  const std::size_t k = n < HYLE_SIMD_MAX_LANES_D(D) ? n : HYLE_SIMD_MAX_LANES_D(D);
  for (std::size_t i = 0; i < k; ++i) p[i] = v.raw[i];
}

}  // namespace hyle::simd
