#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <bit>
#include <cmath>
#include <limits>
#include <type_traits>

#include "hyle/core/compiler.h"
#include "hyle/simd/backend/crypto_soft.h"  // shared AES/CLMul software reference (Crypto op face)

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

// ---- interleaved load / store (AoS <-> SoA) -----------------------------------------
// LoadInterleaved2/3/4(d, p, v0..): DEINTERLEAVE K consecutive-in-memory streams into K
// vectors: v0[i]=p[K*i+0], v1[i]=p[K*i+1], ... (Highway: v0[0]==p[0], v1[0]==p[1]). The
// classic RGB(A) / complex / vertex-attribute AoS->SoA split. StoreInterleaved does the
// inverse SoA->AoS. Native on NEON (vld2/3/4q + vst2/3/4q); the scalar floor + x86 do the
// strided copy (NEON's own fallback is also memory-based for wide types). N = Lanes.
template <class D>
HYLE_FORCE_INLINE void LoadInterleaved2(D, const TFromD<D>* p, VFromD<D>& v0, VFromD<D>& v1) noexcept {
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  for (std::size_t i = 0; i < N; ++i) { v0.raw[i] = p[2 * i]; v1.raw[i] = p[2 * i + 1]; }
}
template <class D>
HYLE_FORCE_INLINE void LoadInterleaved3(D, const TFromD<D>* p, VFromD<D>& v0, VFromD<D>& v1,
                                        VFromD<D>& v2) noexcept {
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  for (std::size_t i = 0; i < N; ++i) { v0.raw[i] = p[3*i]; v1.raw[i] = p[3*i+1]; v2.raw[i] = p[3*i+2]; }
}
template <class D>
HYLE_FORCE_INLINE void LoadInterleaved4(D, const TFromD<D>* p, VFromD<D>& v0, VFromD<D>& v1,
                                        VFromD<D>& v2, VFromD<D>& v3) noexcept {
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  for (std::size_t i = 0; i < N; ++i) { v0.raw[i]=p[4*i]; v1.raw[i]=p[4*i+1]; v2.raw[i]=p[4*i+2]; v3.raw[i]=p[4*i+3]; }
}
template <class D>
HYLE_FORCE_INLINE void StoreInterleaved2(VFromD<D> v0, VFromD<D> v1, D, TFromD<D>* p) noexcept {
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  for (std::size_t i = 0; i < N; ++i) { p[2*i] = v0.raw[i]; p[2*i+1] = v1.raw[i]; }
}
template <class D>
HYLE_FORCE_INLINE void StoreInterleaved3(VFromD<D> v0, VFromD<D> v1, VFromD<D> v2, D,
                                         TFromD<D>* p) noexcept {
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  for (std::size_t i = 0; i < N; ++i) { p[3*i]=v0.raw[i]; p[3*i+1]=v1.raw[i]; p[3*i+2]=v2.raw[i]; }
}
template <class D>
HYLE_FORCE_INLINE void StoreInterleaved4(VFromD<D> v0, VFromD<D> v1, VFromD<D> v2, VFromD<D> v3, D,
                                         TFromD<D>* p) noexcept {
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  for (std::size_t i = 0; i < N; ++i) { p[4*i]=v0.raw[i]; p[4*i+1]=v1.raw[i]; p[4*i+2]=v2.raw[i]; p[4*i+3]=v3.raw[i]; }
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

// ---- math / arith extensions (Highway) ----------------------------------------------
// ApproximateReciprocal(a) ~= 1/a, ApproximateReciprocalSqrt(a) ~= 1/sqrt(a): the scalar
// floor is EXACT (a real backend uses rcpps/rsqrtps 12-bit approximations + a Newton step;
// the tests must therefore tolerate ~2^-12 error on the wide backends). Newton-iteration seed.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> ApproximateReciprocal(Vec<T, N> a) noexcept {
  static_assert(std::is_floating_point_v<T>, "float only");
  Vec<T, N> r; for (std::size_t i = 0; i < N; ++i) r.raw[i] = T(1) / a.raw[i]; return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> ApproximateReciprocalSqrt(Vec<T, N> a) noexcept {
  static_assert(std::is_floating_point_v<T>, "float only");
  Vec<T, N> r; for (std::size_t i = 0; i < N; ++i) r.raw[i] = T(1) / std::sqrt(a.raw[i]); return r;
}
// GetExponent(v) = floor(log2(|v|)) as a float (std::logb). Feeds range reduction.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> GetExponent(Vec<T, N> v) noexcept {
  static_assert(std::is_floating_point_v<T>, "float only");
  Vec<T, N> r; for (std::size_t i = 0; i < N; ++i) r.raw[i] = std::logb(v.raw[i]); return r;
}
// MulByPow2(a, b): a * 2^b (std::ldexp), b an integer-lane vector of matching width.
template <typename T, std::size_t N, typename IT>
HYLE_FORCE_INLINE Vec<T, N> MulByPow2(Vec<T, N> a, Vec<IT, N> b) noexcept {
  static_assert(std::is_floating_point_v<T>, "float only");
  Vec<T, N> r; for (std::size_t i = 0; i < N; ++i) r.raw[i] = std::ldexp(a.raw[i], static_cast<int>(b.raw[i])); return r;
}
// NaN-aware min/max (Highway MinNumber/MaxNumber): if one operand is NaN, return the other.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> MinNumber(Vec<T, N> a, Vec<T, N> b) noexcept {
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i) {
    if constexpr (std::is_floating_point_v<T>) {
      r.raw[i] = std::isnan(a.raw[i]) ? b.raw[i] : (std::isnan(b.raw[i]) ? a.raw[i] : (a.raw[i] < b.raw[i] ? a.raw[i] : b.raw[i]));
    } else r.raw[i] = a.raw[i] < b.raw[i] ? a.raw[i] : b.raw[i];
  }
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> MaxNumber(Vec<T, N> a, Vec<T, N> b) noexcept {
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i) {
    if constexpr (std::is_floating_point_v<T>) {
      r.raw[i] = std::isnan(a.raw[i]) ? b.raw[i] : (std::isnan(b.raw[i]) ? a.raw[i] : (a.raw[i] > b.raw[i] ? a.raw[i] : b.raw[i]));
    } else r.raw[i] = a.raw[i] > b.raw[i] ? a.raw[i] : b.raw[i];
  }
  return r;
}
// AddSub(a,b): a-b in EVEN lanes, a+b in ODD lanes (complex-arith / FFT butterfly helper).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> AddSub(Vec<T, N> a, Vec<T, N> b) noexcept {
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i) r.raw[i] = (i & 1) ? static_cast<T>(a.raw[i] + b.raw[i]) : static_cast<T>(a.raw[i] - b.raw[i]);
  return r;
}
// MulRound(a,b): round(a*b) to nearest-even, result stays float (f only).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> MulRound(Vec<T, N> a, Vec<T, N> b) noexcept {
  static_assert(std::is_floating_point_v<T>, "MulRound is float only");
  Vec<T, N> r; for (std::size_t i = 0; i < N; ++i) r.raw[i] = std::nearbyint(a.raw[i] * b.raw[i]); return r;
}
// MulFixedPoint15(a,b): Q1.15 * Q1.15 -> Q1.15 = (2*a*b) >> 16 with rounding (i16, x86
// _mm*_mulhrs_epi16). The int16 fixed-point multiply for DSP / quantized activations.
// CONTRACT: the MIN*MIN corner (both -1.0 -> product +1.0, unrepresentable in Q1.15) is
// IMPL-DEFINED -- scalar/x86 wrap to MIN, NEON saturates to MAX. Callers must not depend on it.
template <std::size_t N>
HYLE_FORCE_INLINE Vec<std::int16_t, N> MulFixedPoint15(Vec<std::int16_t, N> a, Vec<std::int16_t, N> b) noexcept {
  Vec<std::int16_t, N> r;
  for (std::size_t i = 0; i < N; ++i) {
    const int prod = static_cast<int>(a.raw[i]) * b.raw[i];
    r.raw[i] = static_cast<std::int16_t>((prod + 0x4000) >> 15);  // round + take upper (mulhrs)
  }
  return r;
}
// SaturatedNeg(a) = (a==MIN) ? MAX : -a. SaturatedAbs(a) = (a==MIN) ? MAX : |a|. Signed int.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> SaturatedNeg(Vec<T, N> a) noexcept {
  static_assert(std::is_integral_v<T> && std::is_signed_v<T>, "SaturatedNeg is signed-int");
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i)
    r.raw[i] = (a.raw[i] == std::numeric_limits<T>::min()) ? std::numeric_limits<T>::max() : static_cast<T>(-a.raw[i]);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> SaturatedAbs(Vec<T, N> a) noexcept {
  static_assert(std::is_integral_v<T> && std::is_signed_v<T>, "SaturatedAbs is signed-int");
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i)
    r.raw[i] = (a.raw[i] == std::numeric_limits<T>::min()) ? std::numeric_limits<T>::max()
                                                           : static_cast<T>(a.raw[i] < 0 ? -a.raw[i] : a.raw[i]);
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

// Shl/Shr: per-lane VARIABLE shift count from a second vector (Highway names). Same
// signed=arithmetic / unsigned=logical semantics as ShiftLeft<k>/ShiftRight<k>, but each
// lane shifts by its own count. Wide backends use sllv/srlv/srav (x86) / vshlq-with-negated-
// count (NEON); the scalar floor is the C++ operator per lane.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> Shl(Vec<T, N> a, Vec<T, N> b) noexcept {
  static_assert(std::is_integral_v<T>, "Shl is integer-lane only");
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i)
    r.raw[i] = static_cast<T>(static_cast<std::make_unsigned_t<T>>(a.raw[i]) << b.raw[i]);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> Shr(Vec<T, N> a, Vec<T, N> b) noexcept {
  static_assert(std::is_integral_v<T>, "Shr is integer-lane only");
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i) r.raw[i] = static_cast<T>(a.raw[i] >> b.raw[i]);  // arith if signed
  return r;
}

// Fixed-width native hook used by vec.h's all-integer MulHigh contract.
namespace detail {
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> MulHighLow8Native(Vec<T, N> a, Vec<T, N> b,
                                               Vec<T, N>& high) noexcept {
  static_assert(std::is_integral_v<T> && sizeof(T) == 1, "MulHighLow8Native is 8-bit int only");
  using U = std::make_unsigned_t<T>;
  using W = std::conditional_t<std::is_signed_v<T>, std::int16_t, std::uint16_t>;
  Vec<T, N> low;
  for (std::size_t i = 0; i < N; ++i) {
    const W product = static_cast<W>(a.raw[i]) * static_cast<W>(b.raw[i]);
    low.raw[i] = std::bit_cast<T>(static_cast<U>(product));
    high.raw[i] = static_cast<T>(product >> 8);
  }
  return low;
}

template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> MulHigh8Native(Vec<T, N> a, Vec<T, N> b) noexcept {
  static_assert(std::is_integral_v<T> && sizeof(T) == 1, "MulHigh8Native is 8-bit int only");
  using W = std::conditional_t<std::is_signed_v<T>, std::int16_t, std::uint16_t>;
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i)
    r.raw[i] = static_cast<T>((static_cast<W>(a.raw[i]) * static_cast<W>(b.raw[i])) >> 8);
  return r;
}

template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> MulHigh16Native(Vec<T, N> a, Vec<T, N> b) noexcept {
  static_assert(std::is_integral_v<T> && sizeof(T) == 2, "MulHigh16Native is 16-bit int only");
  using W = std::conditional_t<std::is_signed_v<T>, std::int32_t, std::uint32_t>;
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i)
    r.raw[i] = static_cast<T>((static_cast<W>(a.raw[i]) * static_cast<W>(b.raw[i])) >> 16);
  return r;
}

template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> MulHigh32Native(Vec<T, N> a, Vec<T, N> b) noexcept {
  static_assert(std::is_integral_v<T> && sizeof(T) == 4, "MulHigh32Native is 32-bit int only");
  using W = std::conditional_t<std::is_signed_v<T>, std::int64_t, std::uint64_t>;
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i)
    r.raw[i] = static_cast<T>((static_cast<W>(a.raw[i]) * static_cast<W>(b.raw[i])) >> 32);
  return r;
}
}  // namespace detail

// WidenMulPairwiseAdd(d32, a_i16, b_i16): multiply i16 lanes, sum ADJACENT PAIRS into i32
// lanes -- the fixed-point dot-product core (x86 _mm*_madd_epi16, NEON vmull+pairwise). The
// dst tag names i32 with HALF the lane count of the i16 inputs (2 i16 -> 1 i32). a/b are the
// FULL-width i16 vectors (2N lanes); result is the N-lane i32 vector of pairwise products.
template <std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec<std::int32_t, NDst> WidenMulPairwiseAdd(Simd<std::int32_t, NDst, 0>,
                                                             Vec<SrcT, NSrc> a,
                                                             Vec<SrcT, NSrc> b) noexcept {
  static_assert(std::is_same_v<SrcT, std::int16_t> && NSrc == 2 * NDst,
                "WidenMulPairwiseAdd: i16 -> i32 pairwise (dst has half the lanes)");
  Vec<std::int32_t, NDst> r;
  for (std::size_t i = 0; i < NDst; ++i) {
    // int64 intermediate + saturate: matches the x86 pmaddwd hardware contract and avoids the
    // signed-int32 overflow UB the naive int32 sum hit at the 4x(-32768) corner (which GCC
    // folded to a wrap the other backends' hardware saturates instead -- caught by the SDE
    // vs scalar D2 divergence, simd-18 four-check).
    const std::int64_t s = static_cast<std::int64_t>(a.raw[2 * i]) * b.raw[2 * i] +
                           static_cast<std::int64_t>(a.raw[2 * i + 1]) * b.raw[2 * i + 1];
    const std::int64_t clamped =
        s > 2147483647LL ? 2147483647LL : (s < -2147483648LL ? -2147483648LL : s);
    r.raw[i] = static_cast<std::int32_t>(clamped);
  }
  return r;
}

// SatWidenMulPairwiseAdd(di16, a_u8, b_i8): widen u8 a * i8 b, sum ADJACENT PAIRS into i16,
// SATURATED (x86 _mm*_maddubs_epi16). The int8 GEMM inner core (llama.cpp Q8 dot). dst i16
// has half the lane count of the u8/i8 inputs. a is UNSIGNED bytes, b is SIGNED bytes.
template <std::size_t NDst, std::size_t NSrc>
HYLE_FORCE_INLINE Vec<std::int16_t, NDst> SatWidenMulPairwiseAdd(
    Simd<std::int16_t, NDst, 0>, Vec<std::uint8_t, NSrc> a, Vec<std::int8_t, NSrc> b) noexcept {
  static_assert(NSrc == 2 * NDst, "SatWidenMulPairwiseAdd: u8/i8 -> i16 pairwise");
  Vec<std::int16_t, NDst> r;
  for (std::size_t i = 0; i < NDst; ++i) {
    const int p = static_cast<int>(a.raw[2 * i]) * b.raw[2 * i] +
                  static_cast<int>(a.raw[2 * i + 1]) * b.raw[2 * i + 1];
    r.raw[i] = static_cast<std::int16_t>(p < -32768 ? -32768 : (p > 32767 ? 32767 : p));
  }
  return r;
}

// ReorderWidenMulAccumulate(dw, a, b, sum0, sum1): widen i16 a*b and add each product to
// EITHER sum0 (returned) or sum1 (out-param), under an implementation-defined permutation P;
// the ONLY guarantee is SumOfLanes(sum0+sum1) increases by sum(a[i]*b[i]). The scalar floor
// picks P = pairwise-into-sum0 (identity for the pair sum), leaving sum1 untouched -- callers
// MUST init sum1 to zero and finalize with RearrangeToOddPlusEven. This is the widen-dot
// ACCUMULATE loop primitive (call repeatedly, one Rearrange at the end).
template <std::size_t NW, std::size_t NSrc>
HYLE_FORCE_INLINE Vec<std::int32_t, NW> ReorderWidenMulAccumulate(
    Simd<std::int32_t, NW, 0>, Vec<std::int16_t, NSrc> a, Vec<std::int16_t, NSrc> b,
    Vec<std::int32_t, NW> sum0, Vec<std::int32_t, NW>& /*sum1*/) noexcept {
  static_assert(NSrc == 2 * NW, "ReorderWidenMulAccumulate: i16 -> i32 (dst half lanes)");
  for (std::size_t i = 0; i < NW; ++i)
    sum0.raw[i] += static_cast<std::int32_t>(a.raw[2 * i]) * b.raw[2 * i] +
                   static_cast<std::int32_t>(a.raw[2 * i + 1]) * b.raw[2 * i + 1];
  return sum0;  // scalar P keeps all products in sum0; sum1 stays as the caller zeroed it
}
// RearrangeToOddPlusEven(sum0, sum1): finalize the accumulate so each i32 lane = the summed
// widened products for that lane. Scalar P already put everything in sum0, so this is sum0+sum1.
template <std::size_t NW>
HYLE_FORCE_INLINE Vec<std::int32_t, NW> RearrangeToOddPlusEven(Vec<std::int32_t, NW> sum0,
                                                              Vec<std::int32_t, NW> sum1) noexcept {
  Vec<std::int32_t, NW> r;
  for (std::size_t i = 0; i < NW; ++i) r.raw[i] = sum0.raw[i] + sum1.raw[i];
  return r;
}

// SumOfMulQuadAccumulate(di32, a_i8, b_i8, sum): sum[i] += a[4i+0..3]*b[4i+0..3] (4-way widen
// dot into i32, x86 VNNI vpdpbusd / ARM sdot). dst i32 has a QUARTER the lane count of the i8
// inputs. The int8 dot-product accelerator primitive.
template <std::size_t NDst, std::size_t NSrc>
HYLE_FORCE_INLINE Vec<std::int32_t, NDst> SumOfMulQuadAccumulate(
    Simd<std::int32_t, NDst, 0>, Vec<std::int8_t, NSrc> a, Vec<std::int8_t, NSrc> b,
    Vec<std::int32_t, NDst> sum) noexcept {
  static_assert(NSrc == 4 * NDst, "SumOfMulQuadAccumulate: i8 -> i32 (dst quarter lanes)");
  for (std::size_t i = 0; i < NDst; ++i)
    for (std::size_t k = 0; k < 4; ++k)
      sum.raw[i] += static_cast<std::int32_t>(a.raw[4 * i + k]) * b.raw[4 * i + k];
  return sum;
}

// AbsDiff: |a - b| per lane (no overflow for the intended small-int/f32 use). Highway builds
// this generically as Sub(Max,Min) for ints; the scalar floor is the direct definition.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> AbsDiff(Vec<T, N> a, Vec<T, N> b) noexcept {
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i) {
    const T x = a.raw[i], y = b.raw[i];
    r.raw[i] = x > y ? static_cast<T>(x - y) : static_cast<T>(y - x);
  }
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

// ---- bit-cast (reinterpret the bit pattern as another lane type; same TOTAL bytes) --
// BitCast(dst_tag, v): reinterpret v's bytes as vector of DstT. The byte width is
// preserved (sizeof(SrcT)*NSrc == sizeof(DstT)*NDst), so the lane COUNT changes when the
// lane widths differ (e.g. Vec<i8,16> <-> Vec<i32,4>). Pure reinterpret -- no value
// conversion (unlike ConvertTo). The floor copies bytes; a real backend is a zero-cost
// register cast (_mm_castps_si128 / vreinterpretq). This underpins gather index math,
// mask<->vector bridging, and FP bit tricks (exponent extraction, sign masks).
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec<DstT, NDst> BitCast(Simd<DstT, NDst, 0>, Vec<SrcT, NSrc> v) noexcept {
  static_assert(sizeof(DstT) * NDst == sizeof(SrcT) * NSrc,
                "BitCast preserves total byte width");
  Vec<DstT, NDst> r;
  std::memcpy(r.raw, v.raw, sizeof(SrcT) * NSrc);
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

// ---- width-changing promote / demote (Highway's fixed-width-friendly forms) ---------
// The fixed-native-width model can't do Highway's SAME-lane-count PromoteTo (16 i8 lanes
// -> 16 i16 needs a 2x-wide reg). But the HALF-at-a-time forms fit perfectly and ARE the
// quantization path: PromoteLowerTo/PromoteUpperTo take a FULL narrow vector (2N lanes)
// and widen its lower/upper N lanes to a FULL wide vector (N lanes); DemoteTo(dn,a,b) packs
// two FULL wide vectors (N lanes each) into one FULL narrow (2N lanes). The dst tag names
// the target lane type + its full lane count. Scalar floor is per-lane static_cast; wide
// backends use cvtepi/packs (x86) / vmovl/vqmovn (NEON).
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec<DstT, NDst> PromoteLowerTo(Simd<DstT, NDst, 0>, Vec<SrcT, NSrc> v) noexcept {
  static_assert(NSrc == 2 * NDst, "PromoteLowerTo: narrow src has 2x the wide dst's lanes");
  static_assert(sizeof(DstT) > sizeof(SrcT), "PromoteLowerTo widens the lane type");
  Vec<DstT, NDst> r;
  for (std::size_t i = 0; i < NDst; ++i) r.raw[i] = static_cast<DstT>(v.raw[i]);  // lower half
  return r;
}
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec<DstT, NDst> PromoteUpperTo(Simd<DstT, NDst, 0>, Vec<SrcT, NSrc> v) noexcept {
  static_assert(NSrc == 2 * NDst, "PromoteUpperTo: narrow src has 2x the wide dst's lanes");
  static_assert(sizeof(DstT) > sizeof(SrcT), "PromoteUpperTo widens the lane type");
  Vec<DstT, NDst> r;
  for (std::size_t i = 0; i < NDst; ++i) r.raw[i] = static_cast<DstT>(v.raw[NDst + i]);  // upper half
  return r;
}
// DemoteTo(dn, a, b): two wide vectors -> one narrow (a's lanes then b's lanes). Integer
// demotion SATURATES to the narrow type's range (Highway's packs/packus contract); f64->f32
// is the ordinary narrowing cast. Ordered result [a.lanes..., b.lanes...].
namespace detail {
template <typename DstT>
HYLE_FORCE_INLINE DstT demote_sat(long long x) noexcept {
  constexpr long long lo = static_cast<long long>(std::numeric_limits<DstT>::min());
  constexpr long long hi = static_cast<long long>(std::numeric_limits<DstT>::max());
  return static_cast<DstT>(x < lo ? lo : (x > hi ? hi : x));
}
}  // namespace detail
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec<DstT, NDst> DemoteTo(Simd<DstT, NDst, 0>, Vec<SrcT, NSrc> a,
                                           Vec<SrcT, NSrc> b) noexcept {
  static_assert(NDst == 2 * NSrc, "DemoteTo: narrow dst has 2x each wide src's lanes");
  Vec<DstT, NDst> r;
  if constexpr (std::is_integral_v<SrcT>) {
    for (std::size_t i = 0; i < NSrc; ++i) r.raw[i] = detail::demote_sat<DstT>(static_cast<long long>(a.raw[i]));
    for (std::size_t i = 0; i < NSrc; ++i) r.raw[NSrc + i] = detail::demote_sat<DstT>(static_cast<long long>(b.raw[i]));
  } else {  // f64 -> f32: ordinary narrowing
    for (std::size_t i = 0; i < NSrc; ++i) r.raw[i] = static_cast<DstT>(a.raw[i]);
    for (std::size_t i = 0; i < NSrc; ++i) r.raw[NSrc + i] = static_cast<DstT>(b.raw[i]);
  }
  return r;
}
// ReorderDemote2To: LANE ORDER IS UNSPECIFIED BY CONTRACT (Highway's ReorderDemote2To). The
// scalar floor happens to emit the ordered [a...,b...] (== DemoteTo, nothing to reorder), and
// NEON too (vcombine is ordered); but the wide x86 backends emit the cheaper block-interleaved
// permutation [a.lo,b.lo,a.hi,b.hi]. Callers MUST treat the result as a multiset, NOT rely on
// positions -- use DemoteTo when order matters. (Same values, different arrangement per backend.)
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec<DstT, NDst> ReorderDemote2To(Simd<DstT, NDst, 0> dn, Vec<SrcT, NSrc> a,
                                                   Vec<SrcT, NSrc> b) noexcept {
  return DemoteTo(dn, a, b);
}
// simd-18 C1: OrderedDemote2To == DemoteTo on scalar (already ordered [a...,b...]).
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec<DstT, NDst> OrderedDemote2To(Simd<DstT, NDst, 0> dn, Vec<SrcT, NSrc> a,
                                                   Vec<SrcT, NSrc> b) noexcept {
  return DemoteTo(dn, a, b);
}
// simd-18 C3: ConvertInRangeTo -- alias ConvertTo (no overflow fixup on scalar either).
template <typename DstT, std::size_t N, typename SrcT>
HYLE_FORCE_INLINE Vec<DstT, N> ConvertInRangeTo(Simd<DstT, N, 0> d, Vec<SrcT, N> v) noexcept {
  return ConvertTo(d, v);
}
// simd-18 C3: DemoteInRangeTo f64->i32 -- per-lane truncate (caller guarantees in-range).
template <std::size_t N>
HYLE_FORCE_INLINE Vec<std::int32_t, N> DemoteInRangeTo(Simd<std::int32_t, N, 0>,
                                                       Vec<double, N> v) noexcept {
  Vec<std::int32_t, N> r;
  for (std::size_t i = 0; i < N; ++i) r.raw[i] = static_cast<std::int32_t>(v.raw[i]);
  return r;
}

// TruncateTo(d, v): narrow each lane by keeping the LOW bits (modular truncation, NOT
// saturation -- distinct from DemoteTo). Same lane count, narrower type. E.g. u32 0x1FF -> u8
// 0xFF. Highway's wrap-around narrowing for hashing / bit-packing.
template <typename DstT, std::size_t N, typename SrcT>
HYLE_FORCE_INLINE Vec<DstT, N> TruncateTo(Simd<DstT, N, 0>, Vec<SrcT, N> v) noexcept {
  static_assert(sizeof(DstT) < sizeof(SrcT), "TruncateTo narrows the lane type");
  Vec<DstT, N> r;
  for (std::size_t i = 0; i < N; ++i) r.raw[i] = static_cast<DstT>(v.raw[i]);  // C++ narrowing = low bits
  return r;
}
// PromoteEvenTo/PromoteOddTo(d, v): widen the EVEN (or ODD) lanes of a 2N-lane narrow vector
// to a full N-lane wide vector. Complements PromoteLowerTo/UpperTo (which take contiguous
// halves) -- these take strided lanes, the (de)interleave-free widen for planar data.
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec<DstT, NDst> PromoteEvenTo(Simd<DstT, NDst, 0>, Vec<SrcT, NSrc> v) noexcept {
  static_assert(NSrc == 2 * NDst && sizeof(DstT) > sizeof(SrcT), "PromoteEvenTo: even lanes narrow->wide");
  Vec<DstT, NDst> r;
  for (std::size_t i = 0; i < NDst; ++i) r.raw[i] = static_cast<DstT>(v.raw[2 * i]);
  return r;
}
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec<DstT, NDst> PromoteOddTo(Simd<DstT, NDst, 0>, Vec<SrcT, NSrc> v) noexcept {
  static_assert(NSrc == 2 * NDst && sizeof(DstT) > sizeof(SrcT), "PromoteOddTo: odd lanes narrow->wide");
  Vec<DstT, NDst> r;
  for (std::size_t i = 0; i < NDst; ++i) r.raw[i] = static_cast<DstT>(v.raw[2 * i + 1]);
  return r;
}
// U8FromU32(v): special-case u32 -> u8 when all lanes are already in [0,256) (Highway). Same
// lane count; the scalar floor is a plain narrowing cast (caller guarantees the range).
template <std::size_t N>
HYLE_FORCE_INLINE Vec<std::uint8_t, N> U8FromU32(Vec<std::uint32_t, N> v) noexcept {
  Vec<std::uint8_t, N> r;
  for (std::size_t i = 0; i < N; ++i) r.raw[i] = static_cast<std::uint8_t>(v.raw[i]);
  return r;
}

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

// ---- floating-point classification -> mask (Highway) --------------------------------
// IsNaN(v): lane true iff NaN (the only value != itself). IsInf: true iff +/-infinity.
// IsFinite: true iff neither NaN nor Inf (a real number). IsEitherNaN(a,b): true iff a or b
// is NaN (the "unordered" predicate a compare needs). Float lanes only. Numerical solvers use
// these to detect divergence (NaN/Inf) per lane and mask off the bad elements.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask<T, N> IsNaN(Vec<T, N> v) noexcept {
  static_assert(std::is_floating_point_v<T>, "IsNaN is float-only");
  Mask<T, N> m; for (std::size_t i = 0; i < N; ++i) m.raw[i] = std::isnan(v.raw[i]); return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask<T, N> IsInf(Vec<T, N> v) noexcept {
  static_assert(std::is_floating_point_v<T>, "IsInf is float-only");
  Mask<T, N> m; for (std::size_t i = 0; i < N; ++i) m.raw[i] = std::isinf(v.raw[i]); return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask<T, N> IsFinite(Vec<T, N> v) noexcept {
  static_assert(std::is_floating_point_v<T>, "IsFinite is float-only");
  Mask<T, N> m; for (std::size_t i = 0; i < N; ++i) m.raw[i] = std::isfinite(v.raw[i]); return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask<T, N> IsEitherNaN(Vec<T, N> a, Vec<T, N> b) noexcept {
  static_assert(std::is_floating_point_v<T>, "IsEitherNaN is float-only");
  Mask<T, N> m; for (std::size_t i = 0; i < N; ++i) m.raw[i] = std::isnan(a.raw[i]) || std::isnan(b.raw[i]); return m;
}

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
// Lane index of the LAST set lane, or -1 if none (Highway FindLastTrue). The reverse-scan
// primitive rfind needs: highest matching lane in a block.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE std::ptrdiff_t FindLastTrue(DFromV<Vec<T, N>>, Mask<T, N> m) noexcept {
  for (std::size_t i = N; i-- > 0;)
    if (m.raw[i]) return static_cast<std::ptrdiff_t>(i);
  return -1;
}
// VecFromMask(d, m): all-ones bit pattern where m is true, else zero (Highway). The inverse
// of a compare's mask->vector; feeds SumsOf8 for a masked byte-count.
template <class D>
HYLE_FORCE_INLINE VFromD<D> VecFromMask(D, Mask<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m) noexcept {
  using T = TFromD<D>;
  VFromD<D> r;
  for (std::size_t i = 0; i < HYLE_SIMD_MAX_LANES_D(D); ++i) {
    if (m.raw[i]) {
      using U = std::make_unsigned_t<std::conditional_t<std::is_integral_v<T>, T, std::uint64_t>>;
      if constexpr (std::is_integral_v<T>) r.raw[i] = static_cast<T>(~U{0});
      else { U u = ~U{0}; std::memcpy(&r.raw[i], &u, sizeof(T)); }
    } else {
      r.raw[i] = T{0};
    }
  }
  return r;
}
// MaskFromVec(v): a lane is true iff its most-significant bit is set (Highway). The inverse
// of VecFromMask; used to bridge a hand-built vector predicate back into the mask domain.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask<T, N> MaskFromVec(Vec<T, N> v) noexcept {
  Mask<T, N> m;
  for (std::size_t i = 0; i < N; ++i) {
    // MSB of the lane's SAME-WIDTH bit pattern. Float must use a same-width unsigned (u32 for
    // f32, u64 for f64) -- an over-wide U (e.g. u64 for f32) would read bit 63 of a
    // partially-initialized word and misclassify every float lane (simd-18 batch A bug).
    using U = MakeUnsigned<T>;
    U u;
    if constexpr (std::is_integral_v<T>) u = static_cast<U>(v.raw[i]);
    else std::memcpy(&u, &v.raw[i], sizeof(T));
    m.raw[i] = (u >> (sizeof(U) * 8 - 1)) & 1;  // MSB
  }
  return m;
}
// MaskFromNormalizedVec(v): caller guarantees v is already a canonical all-ones/all-zero lane
// pattern. The scalar mask is a per-lane bool, so the MSB test (MaskFromVec) is the natural
// conversion -- there is no vector-domain sign-broadcast to skip (that was the avx2 optimization).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask<T, N> MaskFromNormalizedVec(Vec<T, N> v) noexcept {
  return MaskFromVec(v);
}

// ---- mask-domain logic (combine predicates for vectorized-loop control flow) --------
// Boolean ops directly on masks (Highway And/Or/Xor/AndNot/Not/ExclusiveNeither for masks).
// These let a loop compose conditions (e.g. `in_range = Ge(x,lo) & Lt(x,hi)`) without
// round-tripping through VecFromMask. The scalar floor is a per-lane bool op.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask<T, N> And(Mask<T, N> a, Mask<T, N> b) noexcept {
  Mask<T, N> m; for (std::size_t i = 0; i < N; ++i) m.raw[i] = a.raw[i] && b.raw[i]; return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask<T, N> Or(Mask<T, N> a, Mask<T, N> b) noexcept {
  Mask<T, N> m; for (std::size_t i = 0; i < N; ++i) m.raw[i] = a.raw[i] || b.raw[i]; return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask<T, N> Xor(Mask<T, N> a, Mask<T, N> b) noexcept {
  Mask<T, N> m; for (std::size_t i = 0; i < N; ++i) m.raw[i] = a.raw[i] != b.raw[i]; return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask<T, N> AndNot(Mask<T, N> a, Mask<T, N> b) noexcept {
  Mask<T, N> m; for (std::size_t i = 0; i < N; ++i) m.raw[i] = !a.raw[i] && b.raw[i]; return m;  // ~a & b
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask<T, N> Not(Mask<T, N> a) noexcept {
  Mask<T, N> m; for (std::size_t i = 0; i < N; ++i) m.raw[i] = !a.raw[i]; return m;
}
// Neither a nor b (Highway ExclusiveNeither = ~(a|b), i.e. NOR).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask<T, N> ExclusiveNeither(Mask<T, N> a, Mask<T, N> b) noexcept {
  Mask<T, N> m; for (std::size_t i = 0; i < N; ++i) m.raw[i] = !a.raw[i] && !b.raw[i]; return m;
}

// IfThenZeroElse(m, no): zero where m true, else `no` (the complement of IfThenElseZero).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> IfThenZeroElse(Mask<T, N> m, Vec<T, N> no) noexcept {
  Vec<T, N> r; for (std::size_t i = 0; i < N; ++i) r.raw[i] = m.raw[i] ? T{0} : no.raw[i]; return r;
}

// ---- mask <-> bits (serialize a mask to a bitfield, and back) -----------------------
// BitsFromMask(d, m): pack lane predicates into an integer bitfield (lane i -> bit i),
// Highway. StoreMaskBits writes those bytes; LoadMaskBits rebuilds a mask from them. Used
// to save/branch on a whole mask cheaply (e.g. `if (BitsFromMask(d,m) == 0) skip`).
template <class D>
HYLE_FORCE_INLINE std::uint64_t BitsFromMask(D, Mask<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m) noexcept {
  std::uint64_t bits = 0;
  for (std::size_t i = 0; i < HYLE_SIMD_MAX_LANES_D(D); ++i)
    if (m.raw[i]) bits |= (std::uint64_t{1} << i);
  return bits;
}
template <class D>
HYLE_FORCE_INLINE Mask<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> LoadMaskBits(D, std::uint64_t bits) noexcept {
  Mask<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m;
  for (std::size_t i = 0; i < HYLE_SIMD_MAX_LANES_D(D); ++i) m.raw[i] = (bits >> i) & 1;
  return m;
}
// StoreMaskBits(d, m, p): write ceil(N/8) bytes of the packed bitfield to p; returns bytes.
template <class D>
HYLE_FORCE_INLINE std::size_t StoreMaskBits(D d, Mask<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m,
                                            std::uint8_t* p) noexcept {
  const std::uint64_t bits = BitsFromMask(d, m);
  const std::size_t nbytes = (HYLE_SIMD_MAX_LANES_D(D) + 7) / 8;
  for (std::size_t b = 0; b < nbytes; ++b) p[b] = static_cast<std::uint8_t>(bits >> (b * 8));
  return nbytes;
}

// ---- prefix-scan masks (leading-run helpers for parsing/tokenizing) -----------------
// Highway SetBeforeFirst/SetAtOrBeforeFirst/SetOnlyFirst/SetAtOrAfterFirst: from a mask,
// derive a run relative to the FIRST true lane. E.g. SetBeforeFirst = lanes strictly before
// the first true (a "prefix until match" selector, used in string scans / run-length).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask<T, N> SetBeforeFirst(Mask<T, N> m) noexcept {
  Mask<T, N> r; std::size_t f = N;
  for (std::size_t i = 0; i < N; ++i) if (m.raw[i]) { f = i; break; }
  for (std::size_t i = 0; i < N; ++i) r.raw[i] = (i < f);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask<T, N> SetAtOrBeforeFirst(Mask<T, N> m) noexcept {
  // = Or(SetBeforeFirst, SetOnlyFirst) (Highway). None-case (f==N): SetBeforeFirst is all-true, so
  // this is all-true too -- `i <= f` with f==N is i<=N, true for every i in [0,N). Found-case: i<=f.
  // (The earlier `&& f < N` guard wrongly forced the none-case all-false, contradicting Highway.)
  Mask<T, N> r; std::size_t f = N;
  for (std::size_t i = 0; i < N; ++i) if (m.raw[i]) { f = i; break; }
  for (std::size_t i = 0; i < N; ++i) r.raw[i] = (i <= f);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask<T, N> SetOnlyFirst(Mask<T, N> m) noexcept {
  Mask<T, N> r{}; for (std::size_t i = 0; i < N; ++i) if (m.raw[i]) { r.raw[i] = true; break; }
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask<T, N> SetAtOrAfterFirst(Mask<T, N> m) noexcept {
  // Complement of SetBeforeFirst: lanes at or after the first true.
  Mask<T, N> r; std::size_t f = N;
  for (std::size_t i = 0; i < N; ++i) if (m.raw[i]) { f = i; break; }
  for (std::size_t i = 0; i < N; ++i) r.raw[i] = (i >= f);
  return r;
}
// FindKnownFirstTrue/LastTrue: like FindFirstTrue but the caller guarantees >=1 true lane,
// so the result is a plain size_t (no -1 sentinel). Highway's known-nonempty fast path.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE std::size_t FindKnownFirstTrue(DFromV<Vec<T, N>>, Mask<T, N> m) noexcept {
  for (std::size_t i = 0; i < N; ++i) if (m.raw[i]) return i;
  return 0;  // contract: caller guarantees a true lane exists
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE std::size_t FindKnownLastTrue(DFromV<Vec<T, N>>, Mask<T, N> m) noexcept {
  for (std::size_t i = N; i-- > 0;) if (m.raw[i]) return i;
  return 0;
}

// SumsOf8(v): sum each consecutive group of 8 u8 lanes into a u64 lane (Highway; x86 psadbw,
// NEON vpaddl chain). The count core: SumsOf8(VecFromMask(Eq(...))) sums 255*matches per group.
template <std::size_t N>
HYLE_FORCE_INLINE Vec<std::uint64_t, N / 8> SumsOf8(Vec<std::uint8_t, N> v) noexcept {
  static_assert(N % 8 == 0, "SumsOf8 needs a lane count that is a multiple of 8");
  Vec<std::uint64_t, N / 8> r;
  for (std::size_t g = 0; g < N / 8; ++g) {
    std::uint64_t s = 0;
    for (std::size_t j = 0; j < 8; ++j) s += v.raw[g * 8 + j];
    r.raw[g] = s;
  }
  return r;
}
// simd-17 batch5: SumsOf8AbsDiff(a,b) = sum |a-b| per group of 8 u8 lanes into u64 (SAD).
// Unary form vs zero equals SumsOf8 (bytes are non-negative).
template <std::size_t N>
HYLE_FORCE_INLINE Vec<std::uint64_t, N / 8> SumsOf8AbsDiff(Vec<std::uint8_t, N> a,
                                                           Vec<std::uint8_t, N> b) noexcept {
  static_assert(N % 8 == 0, "SumsOf8AbsDiff needs a lane count that is a multiple of 8");
  Vec<std::uint64_t, N / 8> r;
  for (std::size_t g = 0; g < N / 8; ++g) {
    std::uint64_t s = 0;
    for (std::size_t j = 0; j < 8; ++j) {
      const int d = static_cast<int>(a.raw[g * 8 + j]) - static_cast<int>(b.raw[g * 8 + j]);
      s += static_cast<std::uint64_t>(d < 0 ? -d : d);
    }
    r.raw[g] = s;
  }
  return r;
}
template <std::size_t N>
HYLE_FORCE_INLINE Vec<std::uint64_t, N / 8> SumsOf8AbsDiff(Vec<std::uint8_t, N> v) noexcept {
  return SumsOf8(v);  // vs zero
}

// ---- simd-18 批F: SAD video quad (scalar oracle) ------------------------------------
template <int kAOffset, int kBOffset, std::size_t N>
HYLE_FORCE_INLINE Vec<std::uint16_t, N / 2> SumsOfAdjQuadAbsDiff(Vec<std::uint8_t, N> a,
                                                                  Vec<std::uint8_t, N> b) noexcept {
  static_assert(kAOffset == 0 || kAOffset == 1, "kAOffset in {0,1}");
  static_assert(kBOffset >= 0 && kBOffset <= 3, "kBOffset in {0..3}");
  static_assert(N % 16 == 0, "SumsOfAdjQuadAbsDiff needs multiple of 16 u8 lanes");
  Vec<std::uint16_t, N / 2> r{};
  for (std::size_t blk = 0; blk < N; blk += 16) {
    for (int i = 0; i < 8; ++i) {
      unsigned sum = 0;
      for (int j = 0; j < 4; ++j) {
        const int da = static_cast<int>(a.raw[blk + static_cast<std::size_t>(kAOffset * 4 + i + j)]);
        const int db = static_cast<int>(b.raw[blk + static_cast<std::size_t>(kBOffset * 4 + j)]);
        sum += static_cast<unsigned>(da >= db ? da - db : db - da);
      }
      r.raw[blk / 2 + static_cast<std::size_t>(i)] = static_cast<std::uint16_t>(sum);
    }
  }
  return r;
}
template <int i3, int i2, int i1, int i0, std::size_t N>
HYLE_FORCE_INLINE Vec<std::uint16_t, N / 2> SumsOfShuffledQuadAbsDiff(Vec<std::uint8_t, N> a,
                                                                       Vec<std::uint8_t, N> b) noexcept {
  static_assert(N % 16 == 0, "SumsOfShuffledQuadAbsDiff needs multiple of 16 u8");
  Vec<std::uint16_t, N / 2> r{};
  constexpr int sel[4] = {i0, i1, i2, i3};
  for (std::size_t blk = 0; blk < N; blk += 16) {
    std::uint8_t sh[16];
    for (int q = 0; q < 4; ++q)
      for (int bi = 0; bi < 4; ++bi)
        sh[q * 4 + bi] = a.raw[blk + static_cast<std::size_t>(sel[q] * 4 + bi)];
    constexpr int kSBase[4] = {0, 2, 8, 10};  // per-pair start byte in shuffled a (Highway)
    for (int pair = 0; pair < 4; ++pair) {
      for (int odd = 0; odd < 2; ++odd) {
        unsigned sum = 0;
        for (int j = 0; j < 4; ++j) {
          const int da = static_cast<int>(sh[kSBase[pair] + odd + j]);
          const int db = static_cast<int>(b.raw[blk + static_cast<std::size_t>(pair * 4 + j)]);
          sum += static_cast<unsigned>(da >= db ? da - db : db - da);
        }
        r.raw[blk / 2 + static_cast<std::size_t>(pair * 2 + odd)] = static_cast<std::uint16_t>(sum);
      }
    }
  }
  return r;
}

// simd-17 batch5: BitShuffle / MultiRotateRight -- per-u64-lane bit gather / byte rotate. Data
// lanes are u64, index lanes are u8 (8 per data lane), so both take same-total-bytes vectors and
// return the u64-data vector (Highway's generic contract, portable across ALL targets -- the
// native AVX3_DL _mm512_bitshuffle / _mm512_multishift is only a fast path). Scalar is the oracle.
//
// BitShuffle(v, idx): for each u64 lane q, output bit j (j in [0,8)) = bit (idx[q*8+j] & 63) of
// v[q]; the 8 selected bits pack into the low byte of the output lane (bit0 = LSB), high 56 bits 0.
template <typename T, std::size_t N, typename TI, std::size_t NI>
HYLE_FORCE_INLINE Vec<T, N> BitShuffle(Vec<T, N> v, Vec<TI, NI> idx) noexcept {
  static_assert(sizeof(T) == 8, "BitShuffle data lanes are 64-bit");
  static_assert(sizeof(TI) == 1 && NI == N * 8, "BitShuffle index is u8, 8 per data lane");
  Vec<T, N> r;
  for (std::size_t q = 0; q < N; ++q) {
    const std::uint64_t src = static_cast<std::uint64_t>(v.raw[q]);
    std::uint64_t out = 0;
    for (std::size_t j = 0; j < 8; ++j) {
      const unsigned bit = static_cast<unsigned>(idx.raw[q * 8 + j]) & 63u;
      out |= ((src >> bit) & 1u) << j;
    }
    r.raw[q] = static_cast<T>(out);
  }
  return r;
}
// MultiRotateRight(v, idx): for each output byte i (i in [0, 8N)), take data lane q = i/8, rotate
// v[q] right by (idx[i] & 63) bits, keep the low 8 bits. Eight independently-rotated byte views of
// each u64 lane (x86 VPMULTISHIFTQB semantics).
template <typename T, std::size_t N, typename TI, std::size_t NI>
HYLE_FORCE_INLINE Vec<T, N> MultiRotateRight(Vec<T, N> v, Vec<TI, NI> idx) noexcept {
  static_assert(sizeof(T) == 8, "MultiRotateRight data lanes are 64-bit");
  static_assert(sizeof(TI) == 1 && NI == N * 8, "MultiRotateRight index is u8, 8 per data lane");
  Vec<T, N> r;
  for (std::size_t q = 0; q < N; ++q) {
    const std::uint64_t src = static_cast<std::uint64_t>(v.raw[q]);
    std::uint64_t out = 0;
    for (std::size_t j = 0; j < 8; ++j) {
      const unsigned rot = static_cast<unsigned>(idx.raw[q * 8 + j]) & 63u;
      const std::uint64_t rotated = rot == 0 ? src : ((src >> rot) | (src << (64 - rot)));
      out |= (rotated & 0xFFu) << (j * 8);
    }
    r.raw[q] = static_cast<T>(out);
  }
  return r;
}

// SumsOf2/SumsOf4: like SumsOf8 but sum consecutive groups of 2 / 4 u8 lanes, widening to
// u16 / u32 (Highway). The finer-grained horizontal byte sums (histogram / partial reduce).
template <std::size_t N>
HYLE_FORCE_INLINE Vec<std::uint16_t, N / 2> SumsOf2(Vec<std::uint8_t, N> v) noexcept {
  static_assert(N % 2 == 0, "SumsOf2 needs an even lane count");
  Vec<std::uint16_t, N / 2> r;
  for (std::size_t g = 0; g < N / 2; ++g) r.raw[g] = static_cast<std::uint16_t>(v.raw[2*g] + v.raw[2*g+1]);
  return r;
}
template <std::size_t N>
HYLE_FORCE_INLINE Vec<std::uint32_t, N / 4> SumsOf4(Vec<std::uint8_t, N> v) noexcept {
  static_assert(N % 4 == 0, "SumsOf4 needs a lane count that is a multiple of 4");
  Vec<std::uint32_t, N / 4> r;
  for (std::size_t g = 0; g < N / 4; ++g)
    r.raw[g] = static_cast<std::uint32_t>(v.raw[4*g] + v.raw[4*g+1] + v.raw[4*g+2] + v.raw[4*g+3]);
  return r;
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

// ---- swizzle (lane rearrangement) ---------------------------------------------------
// The portable-reference definitions the wide backends' shuffle/permute/unpack must match
// lane-for-lane. Highway names/semantics throughout.

// OddEven(a,b): even lanes from b, odd lanes from a (Highway's convention).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> OddEven(Vec<T, N> a, Vec<T, N> b) noexcept {
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i) r.raw[i] = (i & 1) ? a.raw[i] : b.raw[i];
  return r;
}
// Reverse: lane i <- lane N-1-i. Reverse2: swap within each adjacent pair.
template <class D>
HYLE_FORCE_INLINE VFromD<D> Reverse(D, VFromD<D> v) noexcept {
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  VFromD<D> r;
  for (std::size_t i = 0; i < N; ++i) r.raw[i] = v.raw[N - 1 - i];
  return r;
}
template <class D>
HYLE_FORCE_INLINE VFromD<D> Reverse2(D, VFromD<D> v) noexcept {
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  VFromD<D> r;
  for (std::size_t i = 0; i < N; ++i) r.raw[i] = v.raw[i ^ 1];
  return r;
}
// Reverse4/Reverse8: reverse within each group of 4 / 8 lanes (out[i]=a[i^3] / a[i^7]).
template <class D>
HYLE_FORCE_INLINE VFromD<D> Reverse4(D, VFromD<D> v) noexcept {
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  VFromD<D> r; for (std::size_t i = 0; i < N; ++i) r.raw[i] = v.raw[(i & ~std::size_t{3}) | ((i & 3) ^ 3)];
  return r;
}
template <class D>
HYLE_FORCE_INLINE VFromD<D> Reverse8(D, VFromD<D> v) noexcept {
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  VFromD<D> r; for (std::size_t i = 0; i < N; ++i) r.raw[i] = v.raw[(i & ~std::size_t{7}) | ((i & 7) ^ 7)];
  return r;
}
// ReverseBits: reverse the bits of each lane. ReverseLaneBytes: reverse the bytes of each lane
// (endian swap per lane). Integer lanes.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> ReverseBits(Vec<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "ReverseBits is integer-only");
  using U = std::make_unsigned_t<T>;
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i) {
    U x = static_cast<U>(v.raw[i]), y = 0;
    for (std::size_t b = 0; b < sizeof(T) * 8; ++b) { y = static_cast<U>((y << 1) | (x & 1)); x >>= 1; }
    r.raw[i] = static_cast<T>(y);
  }
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> ReverseLaneBytes(Vec<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "ReverseLaneBytes is integer-only");
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i) {
    auto* d = reinterpret_cast<std::uint8_t*>(&r.raw[i]);
    const auto* s = reinterpret_cast<const std::uint8_t*>(&v.raw[i]);
    for (std::size_t b = 0; b < sizeof(T); ++b) d[b] = s[sizeof(T) - 1 - b];
  }
  return r;
}
// InterleaveEven/Odd: alternate the EVEN (or ODD) lanes of a and b: out = a[0],b[0],a[2],b[2]..
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> InterleaveEven(Vec<T, N> a, Vec<T, N> b) noexcept {
  Vec<T, N> r; for (std::size_t i = 0; i < N / 2; ++i) { r.raw[2*i] = a.raw[2*i]; r.raw[2*i+1] = b.raw[2*i]; }
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> InterleaveOdd(Vec<T, N> a, Vec<T, N> b) noexcept {
  Vec<T, N> r; for (std::size_t i = 0; i < N / 2; ++i) { r.raw[2*i] = a.raw[2*i+1]; r.raw[2*i+1] = b.raw[2*i+1]; }
  return r;
}
// Slide1Up/Down: shift the whole vector up/down by one lane, zero-filling the vacated end.
template <class D>
HYLE_FORCE_INLINE VFromD<D> Slide1Up(D, VFromD<D> v) noexcept {
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  VFromD<D> r; r.raw[0] = TFromD<D>{0};
  for (std::size_t i = 1; i < N; ++i) r.raw[i] = v.raw[i - 1];
  return r;
}
template <class D>
HYLE_FORCE_INLINE VFromD<D> Slide1Down(D, VFromD<D> v) noexcept {
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  VFromD<D> r; if (N) r.raw[N - 1] = TFromD<D>{0};
  for (std::size_t i = 0; i + 1 < N; ++i) r.raw[i] = v.raw[i + 1];
  return r;
}
// SlideUpLanes/DownLanes: shift by a RUNTIME count `amt`, zero-filling vacated lanes.
template <class D>
HYLE_FORCE_INLINE VFromD<D> SlideUpLanes(D, VFromD<D> v, std::size_t amt) noexcept {
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  VFromD<D> r{};
  for (std::size_t i = amt; i < N; ++i) r.raw[i] = v.raw[i - amt];
  return r;
}
template <class D>
HYLE_FORCE_INLINE VFromD<D> SlideDownLanes(D, VFromD<D> v, std::size_t amt) noexcept {
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  VFromD<D> r{};
  for (std::size_t i = 0; i + amt < N; ++i) r.raw[i] = v.raw[i + amt];
  return r;
}
// BroadcastLane<kLane>: splat lane kLane across all lanes (Broadcast is the same but I keep
// the Highway name too). ZipLower/Upper: same bits as InterleaveLower/Upper (double-width
// repartition is a type-only reinterpret the callers do; the scalar floor keeps the lane type).
template <int kLane, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> BroadcastLane(Vec<T, N> v) noexcept {
  static_assert(kLane >= 0 && static_cast<std::size_t>(kLane) < N, "BroadcastLane out of range");
  Vec<T, N> r; for (std::size_t i = 0; i < N; ++i) r.raw[i] = v.raw[kLane];
  return r;
}

// InterleaveLower/Upper: zip lanes from the lower / upper half of a and b.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> InterleaveLower(Vec<T, N> a, Vec<T, N> b) noexcept {
  Vec<T, N> r;
  for (std::size_t i = 0; i < N / 2; ++i) { r.raw[2 * i] = a.raw[i]; r.raw[2 * i + 1] = b.raw[i]; }
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> InterleaveUpper(Vec<T, N> a, Vec<T, N> b) noexcept {
  Vec<T, N> r;
  for (std::size_t i = 0; i < N / 2; ++i) { r.raw[2 * i] = a.raw[N / 2 + i]; r.raw[2 * i + 1] = b.raw[N / 2 + i]; }
  return r;
}
// ZipLower/Upper: same bit pattern as InterleaveLower/Upper (Highway repartitions into
// double-width lanes; the scalar floor keeps the lane type -- the widen is a caller reinterpret).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> ZipLower(Vec<T, N> a, Vec<T, N> b) noexcept { return InterleaveLower(a, b); }
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> ZipUpper(Vec<T, N> a, Vec<T, N> b) noexcept { return InterleaveUpper(a, b); }
// Concat{Lower,Upper}{Lower,Upper}(hi,lo): take one half from each of hi/lo. The arg order
// is Highway's (hi first): ConcatLowerLower = [lo.lower, hi.lower], etc.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> ConcatLowerLower(Vec<T, N> hi, Vec<T, N> lo) noexcept {
  Vec<T, N> r;
  for (std::size_t i = 0; i < N / 2; ++i) { r.raw[i] = lo.raw[i]; r.raw[N / 2 + i] = hi.raw[i]; }
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> ConcatUpperUpper(Vec<T, N> hi, Vec<T, N> lo) noexcept {
  Vec<T, N> r;
  for (std::size_t i = 0; i < N / 2; ++i) { r.raw[i] = lo.raw[N / 2 + i]; r.raw[N / 2 + i] = hi.raw[N / 2 + i]; }
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> ConcatLowerUpper(Vec<T, N> hi, Vec<T, N> lo) noexcept {
  Vec<T, N> r;  // [lo.upper, hi.lower]
  for (std::size_t i = 0; i < N / 2; ++i) { r.raw[i] = lo.raw[N / 2 + i]; r.raw[N / 2 + i] = hi.raw[i]; }
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> ConcatUpperLower(Vec<T, N> hi, Vec<T, N> lo) noexcept {
  Vec<T, N> r;  // [lo.lower, hi.upper]
  for (std::size_t i = 0; i < N / 2; ++i) { r.raw[i] = lo.raw[i]; r.raw[N / 2 + i] = hi.raw[N / 2 + i]; }
  return r;
}
// ConcatOdd/Even: gather the odd / even lanes of lo then hi into one vector.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> ConcatEven(Vec<T, N> hi, Vec<T, N> lo) noexcept {
  Vec<T, N> r;
  for (std::size_t i = 0; i < N / 2; ++i) { r.raw[i] = lo.raw[2 * i]; r.raw[N / 2 + i] = hi.raw[2 * i]; }
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> ConcatOdd(Vec<T, N> hi, Vec<T, N> lo) noexcept {
  Vec<T, N> r;
  for (std::size_t i = 0; i < N / 2; ++i) { r.raw[i] = lo.raw[2 * i + 1]; r.raw[N / 2 + i] = hi.raw[2 * i + 1]; }
  return r;
}
// DupEven/DupOdd: duplicate the even / odd lane of each adjacent pair over both.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> DupEven(Vec<T, N> v) noexcept {
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i) r.raw[i] = v.raw[i & ~std::size_t{1}];
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> DupOdd(Vec<T, N> v) noexcept {
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i) r.raw[i] = v.raw[(i & ~std::size_t{1}) | 1];
  return r;
}

// ---- simd-17 batch1: fixed-pattern Shuffle + byte/lane shifts (per-128-bit-block) ----
// Highway contract: 256/512-bit backends apply these ops independently inside each 16-byte
// block. The scalar oracle models the same block-local rule so every backend agrees.
namespace detail {
template <typename T, std::size_t N, class Map4>
HYLE_FORCE_INLINE Vec<T, N> shuffle32_blocks(Vec<T, N> v, Map4 map) noexcept {
  static_assert(sizeof(T) == 4, "32-bit shuffle needs sizeof(T)==4");
  constexpr std::size_t kBlock = 4;  // 16/sizeof(T)
  Vec<T, N> r;
  for (std::size_t base = 0; base < N; base += kBlock) {
    for (std::size_t i = 0; i < kBlock && base + i < N; ++i)
      r.raw[base + i] = v.raw[base + map(i)];
  }
  return r;
}
}  // namespace detail

// Shuffle2301: [1,0,3,2] adjacent pair swap. Shuffle1032: [2,3,0,1] swap 64-bit halves.
// Shuffle0321: [1,2,3,0] rotate right 1. Shuffle2103: [3,0,1,2] rotate left 1.
// Shuffle0123: [3,2,1,0] reverse 4. Shuffle01 (64-bit): [1,0] swap the two 64-lanes.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> Shuffle2301(Vec<T, N> v) noexcept {
  return detail::shuffle32_blocks(v, [](std::size_t i) { return i ^ 1; });
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> Shuffle1032(Vec<T, N> v) noexcept {
  return detail::shuffle32_blocks(v, [](std::size_t i) { return i ^ 2; });
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> Shuffle0321(Vec<T, N> v) noexcept {
  return detail::shuffle32_blocks(v, [](std::size_t i) { return (i + 1) & 3; });
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> Shuffle2103(Vec<T, N> v) noexcept {
  return detail::shuffle32_blocks(v, [](std::size_t i) { return (i + 3) & 3; });
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> Shuffle0123(Vec<T, N> v) noexcept {
  return detail::shuffle32_blocks(v, [](std::size_t i) { return i ^ 3; });
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> Shuffle01(Vec<T, N> v) noexcept {
  static_assert(sizeof(T) == 8, "Shuffle01 needs sizeof(T)==8");
  Vec<T, N> r;
  for (std::size_t base = 0; base < N; base += 2) {
    r.raw[base + 0] = v.raw[base + 1];
    r.raw[base + 1] = v.raw[base + 0];
  }
  return r;
}

// ShiftLeftBytes<k>/ShiftRightBytes<k>: per 16-byte block, shift k bytes, zero-fill.
// CombineShiftRightBytes<k>(hi,lo): per block, take bytes [k..k+15] from the 32-byte window
// [lo_block | hi_block] (lo at low addresses). Lanes wrappers = k*sizeof(T) bytes.
template <int k, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> ShiftLeftBytes(Vec<T, N> v) noexcept {
  static_assert(k >= 0 && k <= 16, "ShiftLeftBytes count in [0,16]");
  Vec<T, N> r{};
  auto* rb = reinterpret_cast<std::uint8_t*>(r.raw);
  const auto* vb = reinterpret_cast<const std::uint8_t*>(v.raw);
  constexpr std::size_t kBytes = sizeof(T) * N;
  for (std::size_t base = 0; base < kBytes; base += 16) {
    for (std::size_t i = 0; i < 16 && base + i < kBytes; ++i)
      rb[base + i] = (i >= static_cast<std::size_t>(k)) ? vb[base + i - static_cast<std::size_t>(k)] : std::uint8_t{0};
  }
  return r;
}
template <int k, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> ShiftRightBytes(Vec<T, N> v) noexcept {
  static_assert(k >= 0 && k <= 16, "ShiftRightBytes count in [0,16]");
  Vec<T, N> r{};
  auto* rb = reinterpret_cast<std::uint8_t*>(r.raw);
  const auto* vb = reinterpret_cast<const std::uint8_t*>(v.raw);
  constexpr std::size_t kBytes = sizeof(T) * N;
  for (std::size_t base = 0; base < kBytes; base += 16) {
    for (std::size_t i = 0; i < 16 && base + i < kBytes; ++i)
      rb[base + i] = (i + static_cast<std::size_t>(k) < 16) ? vb[base + i + static_cast<std::size_t>(k)] : std::uint8_t{0};
  }
  return r;
}
template <int k, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> ShiftLeftLanes(Vec<T, N> v) noexcept {
  return ShiftLeftBytes<k * static_cast<int>(sizeof(T))>(v);
}
template <int k, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> ShiftRightLanes(Vec<T, N> v) noexcept {
  return ShiftRightBytes<k * static_cast<int>(sizeof(T))>(v);
}
template <int k, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> CombineShiftRightBytes(Vec<T, N> hi, Vec<T, N> lo) noexcept {
  static_assert(k > 0 && k < 16, "CombineShiftRightBytes count in (0,16)");
  Vec<T, N> r{};
  auto* rb = reinterpret_cast<std::uint8_t*>(r.raw);
  const auto* hb = reinterpret_cast<const std::uint8_t*>(hi.raw);
  const auto* lb = reinterpret_cast<const std::uint8_t*>(lo.raw);
  constexpr std::size_t kBytes = sizeof(T) * N;
  for (std::size_t base = 0; base < kBytes; base += 16) {
    for (std::size_t i = 0; i < 16 && base + i < kBytes; ++i) {
      const std::size_t src = static_cast<std::size_t>(k) + i;
      rb[base + i] = (src < 16) ? lb[base + src] : hb[base + src - 16];
    }
  }
  return r;
}
template <int k, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> CombineShiftRightLanes(Vec<T, N> hi, Vec<T, N> lo) noexcept {
  return CombineShiftRightBytes<k * static_cast<int>(sizeof(T))>(hi, lo);
}

// ---- simd-17 batch2: runtime *Same shifts, Pairwise, Stream, LoadDup128, exp domain ----
// ShiftLeftSame/RightSame: every lane shifts by the same runtime bit count.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> ShiftLeftSame(Vec<T, N> v, int bits) noexcept {
  static_assert(std::is_integral_v<T>, "ShiftLeftSame is integer-lane only");
  Vec<T, N> r;
  using U = std::make_unsigned_t<T>;
  for (std::size_t i = 0; i < N; ++i)
    r.raw[i] = static_cast<T>(static_cast<U>(v.raw[i]) << bits);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> ShiftRightSame(Vec<T, N> v, int bits) noexcept {
  static_assert(std::is_integral_v<T>, "ShiftRightSame is integer-lane only");
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i) r.raw[i] = static_cast<T>(v.raw[i] >> bits);  // arith if signed
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> RotateRightSame(Vec<T, N> v, int bits) noexcept {
  static_assert(std::is_integral_v<T>, "RotateRightSame is integer-lane only");
  constexpr int W = static_cast<int>(sizeof(T) * 8);
  bits &= (W - 1);
  if (bits == 0) return v;
  using U = std::make_unsigned_t<T>;
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i) {
    const U u = static_cast<U>(v.raw[i]);
    r.raw[i] = static_cast<T>((u >> bits) | (u << (W - bits)));
  }
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> RotateLeftSame(Vec<T, N> v, int bits) noexcept {
  static_assert(std::is_integral_v<T>, "RotateLeftSame is integer-lane only");
  constexpr int W = static_cast<int>(sizeof(T) * 8);
  bits &= (W - 1);
  if (bits == 0) return v;
  return RotateRightSame(v, W - bits);
}
// RoundingShiftRightSame: (v + (1<<(s-1))) >> s; s==0 returns v.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> RoundingShiftRightSame(Vec<T, N> v, int s) noexcept {
  static_assert(std::is_integral_v<T>, "RoundingShiftRightSame is integer-lane only");
  if (s <= 0) return v;
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i) {
    const auto half = static_cast<T>(T{1} << (s - 1));
    r.raw[i] = static_cast<T>((v.raw[i] + half) >> s);
  }
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> RoundingShr(Vec<T, N> v, Vec<T, N> amounts) noexcept {
  static_assert(std::is_integral_v<T>, "RoundingShr is integer-lane only");
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i) {
    const int s = static_cast<int>(amounts.raw[i]);
    if (s <= 0) r.raw[i] = v.raw[i];
    else {
      const auto half = static_cast<T>(T{1} << (s - 1));
      r.raw[i] = static_cast<T>((v.raw[i] + half) >> s);
    }
  }
  return r;
}

// PairwiseAdd/Sub: full-vector [a-pair-sums..., b-pair-sums...].
// PairwiseAdd128/Sub128: same layout but independent inside each 16-byte block.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> PairwiseAdd(Vec<T, N> a, Vec<T, N> b) noexcept {
  Vec<T, N> r;
  for (std::size_t i = 0; i < N / 2; ++i) {
    r.raw[i] = static_cast<T>(a.raw[2 * i] + a.raw[2 * i + 1]);
    r.raw[N / 2 + i] = static_cast<T>(b.raw[2 * i] + b.raw[2 * i + 1]);
  }
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> PairwiseSub(Vec<T, N> a, Vec<T, N> b) noexcept {
  Vec<T, N> r;
  for (std::size_t i = 0; i < N / 2; ++i) {
    r.raw[i] = static_cast<T>(a.raw[2 * i] - a.raw[2 * i + 1]);
    r.raw[N / 2 + i] = static_cast<T>(b.raw[2 * i] - b.raw[2 * i + 1]);
  }
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> PairwiseAdd128(Vec<T, N> a, Vec<T, N> b) noexcept {
  constexpr std::size_t kBL = 16 / sizeof(T);  // lanes per 128-bit block
  Vec<T, N> r;
  for (std::size_t base = 0; base < N; base += kBL) {
    const std::size_t h = kBL / 2;
    for (std::size_t i = 0; i < h; ++i) {
      r.raw[base + i] = static_cast<T>(a.raw[base + 2 * i] + a.raw[base + 2 * i + 1]);
      r.raw[base + h + i] = static_cast<T>(b.raw[base + 2 * i] + b.raw[base + 2 * i + 1]);
    }
  }
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> PairwiseSub128(Vec<T, N> a, Vec<T, N> b) noexcept {
  constexpr std::size_t kBL = 16 / sizeof(T);
  Vec<T, N> r;
  for (std::size_t base = 0; base < N; base += kBL) {
    const std::size_t h = kBL / 2;
    for (std::size_t i = 0; i < h; ++i) {
      r.raw[base + i] = static_cast<T>(a.raw[base + 2 * i] - a.raw[base + 2 * i + 1]);
      r.raw[base + h + i] = static_cast<T>(b.raw[base + 2 * i] - b.raw[base + 2 * i + 1]);
    }
  }
  return r;
}

// Stream: non-temporal aligned store (scalar = ordinary Store).
template <class D>
HYLE_FORCE_INLINE void Stream(VFromD<D> v, D d, TFromD<D>* aligned) noexcept {
  Store(v, d, aligned);
}
// LoadDup128: load 16 bytes and repeat into every 128-bit block of the full vector.
template <class D>
HYLE_FORCE_INLINE VFromD<D> LoadDup128(D, const TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  constexpr std::size_t kBL = 16 / sizeof(T);
  VFromD<D> r;
  for (std::size_t i = 0; i < N; ++i) r.raw[i] = p[i % kBL];
  return r;
}

// GetBiasedExponent: IEEE biased exponent as unsigned same-width vector (no unbias).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE auto GetBiasedExponent(Vec<T, N> v) noexcept {
  static_assert(std::is_floating_point_v<T>, "GetBiasedExponent is float only");
  using U = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
  Vec<U, N> r;
  for (std::size_t i = 0; i < N; ++i) {
    U bits;
    std::memcpy(&bits, &v.raw[i], sizeof(T));
    if constexpr (sizeof(T) == 4) r.raw[i] = (bits >> 23) & 0xFFu;
    else r.raw[i] = (bits >> 52) & 0x7FFull;
  }
  return r;
}
// MulByFloorPow2(v, exp): v * 2^floor(exp) with float exp vector.
namespace detail {
// floor(exp) clamped into int range so the int-cast is never UB; ldexp then saturates
// (+inf on huge positive, 0 on huge negative), matching v * 2^floor(exp) at the extremes.
template <typename T>
HYLE_FORCE_INLINE int floor_exp_to_int(T e) noexcept {
  const T fe = std::floor(e);
  if (fe >= static_cast<T>(std::numeric_limits<int>::max())) return std::numeric_limits<int>::max();
  if (fe <= static_cast<T>(std::numeric_limits<int>::min())) return std::numeric_limits<int>::min();
  return static_cast<int>(fe);
}
}  // namespace detail
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> MulByFloorPow2(Vec<T, N> v, Vec<T, N> exp) noexcept {
  static_assert(std::is_floating_point_v<T>, "MulByFloorPow2 is float only");
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i) {
    if (std::isnan(exp.raw[i]) ||
        (std::isinf(exp.raw[i]) && ((v.raw[i] == T{0} && exp.raw[i] > 0) ||
                                    (std::isinf(v.raw[i]) && exp.raw[i] < 0)))) {
      r.raw[i] = std::numeric_limits<T>::quiet_NaN();
    } else {
      r.raw[i] = std::ldexp(v.raw[i], detail::floor_exp_to_int(exp.raw[i]));
    }
  }
  return r;
}

// ---- simd-17 batch4: half-vector family (real half-width entities) ------------------
// LowerHalf/UpperHalf return N/2-lane vectors; Combine/ZeroExtendVector/InsertIntoUpper
// reassemble a full-width vector. ResizeBitCast reinterprets across widths (min bytes
// copied; destination excess zeroed). Parameter order matches Highway: Combine(d, hi, lo)
// yields [lo in lower half, hi in upper half].
template <typename T, std::size_t NH, int kPow2, std::size_t N>
HYLE_FORCE_INLINE Vec<T, NH> LowerHalf(Simd<T, NH, kPow2>, Vec<T, N> v) noexcept {
  static_assert(NH == N / 2 || (N == 1 && NH == 1), "LowerHalf: NH must be N/2");
  Vec<T, NH> r;
  for (std::size_t i = 0; i < NH; ++i) r.raw[i] = v.raw[i];
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, (N > 1 ? N / 2 : 1)> LowerHalf(Vec<T, N> v) noexcept {
  return LowerHalf(Simd<T, (N > 1 ? N / 2 : 1), 0>{}, v);
}
template <typename T, std::size_t NH, int kPow2, std::size_t N>
HYLE_FORCE_INLINE Vec<T, NH> UpperHalf(Simd<T, NH, kPow2>, Vec<T, N> v) noexcept {
  static_assert(NH == N / 2, "UpperHalf: NH must be N/2 (N>=2)");
  Vec<T, NH> r;
  for (std::size_t i = 0; i < NH; ++i) r.raw[i] = v.raw[NH + i];
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N / 2> UpperHalf(Vec<T, N> v) noexcept {
  static_assert(N >= 2, "UpperHalf needs at least 2 lanes");
  return UpperHalf(Simd<T, N / 2, 0>{}, v);
}
template <typename T, std::size_t N, int kPow2>
HYLE_FORCE_INLINE Vec<T, N> Combine(Simd<T, N, kPow2>, Vec<T, N / 2> hi, Vec<T, N / 2> lo) noexcept {
  static_assert(N >= 2, "Combine needs N>=2");
  Vec<T, N> r;
  for (std::size_t i = 0; i < N / 2; ++i) {
    r.raw[i] = lo.raw[i];
    r.raw[N / 2 + i] = hi.raw[i];
  }
  return r;
}
template <typename T, std::size_t N, int kPow2>
HYLE_FORCE_INLINE Vec<T, N> ZeroExtendVector(Simd<T, N, kPow2>, Vec<T, N / 2> lo) noexcept {
  static_assert(N >= 2, "ZeroExtendVector needs N>=2");
  Vec<T, N> r{};
  for (std::size_t i = 0; i < N / 2; ++i) r.raw[i] = lo.raw[i];
  return r;
}
template <typename T, std::size_t N, int kPow2>
HYLE_FORCE_INLINE Vec<T, N> InsertIntoUpper(Simd<T, N, kPow2>, Vec<T, N / 2> hi, Vec<T, N> v) noexcept {
  static_assert(N >= 2, "InsertIntoUpper needs N>=2");
  Vec<T, N> r = v;
  for (std::size_t i = 0; i < N / 2; ++i) r.raw[N / 2 + i] = hi.raw[i];
  return r;
}
// ResizeBitCast(d, v): bit-reinterpret v into D's width. Copies min(src,dst) bytes; excess
// destination bytes are zeroed. Same-width is pure BitCast; widen/narrow change lane count.
template <typename DstT, std::size_t NDst, int kPow2, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec<DstT, NDst> ResizeBitCast(Simd<DstT, NDst, kPow2>, Vec<SrcT, NSrc> v) noexcept {
  Vec<DstT, NDst> r{};
  const std::size_t n = (sizeof(SrcT) * NSrc < sizeof(DstT) * NDst) ? sizeof(SrcT) * NSrc
                                                                    : sizeof(DstT) * NDst;
  std::memcpy(r.raw, v.raw, n);
  return r;
}

// TableLookupLanes: permute v by a per-lane index vector (IndicesFromVec wraps the index
// vector; the scalar floor stores indices in a plain Vec of the same lane type). Highway's
// permute -- index i selects source lane idx.raw[i].
template <typename T, std::size_t N>
struct Indices { Vec<T, N> idx; };
template <class D>
HYLE_FORCE_INLINE Indices<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> IndicesFromVec(
    D, Vec<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> iv) noexcept {
  return Indices<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)>{iv};
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> TableLookupLanes(Vec<T, N> v, Indices<T, N> ind) noexcept {
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i)
    r.raw[i] = v.raw[static_cast<std::size_t>(ind.idx.raw[i]) & (N - 1)];  // mask like Highway
  return r;
}

// ---- byte table lookup (PSHUFB-style, per-128-bit-block) ----------------------------
// TableLookupBytes(bytes, idx): result byte i = bytes[idx_byte_i], operating on BYTE lanes
// regardless of the vectors' declared lane type. CRITICAL (Highway contract): the lookup is
// WITHIN each 128-bit block -- x86 PSHUFB and NEON's per-16 tbl index only the low 16 bytes
// of their block, so byte i in block b selects byte (b*16 + (idx_i & 15)) of `bytes`. The
// scalar floor models exactly this block-local semantics so every backend agrees. Used for
// int4/int8 dequant LUTs, base64, and byte-shuffle string kernels. TableLookupBytesOr0
// additionally zeroes the output byte where the index's high bit (0x80) is set (PSHUFB does
// this natively). Index lane type may differ from the data's; we view both as byte arrays.
template <typename T, std::size_t N, typename TI, std::size_t NI>
HYLE_FORCE_INLINE Vec<TI, NI> TableLookupBytes(Vec<T, N> bytes, Vec<TI, NI> idx) noexcept {
  static_assert(sizeof(T) * N == sizeof(TI) * NI, "TableLookupBytes: same total byte width");
  Vec<TI, NI> r;
  auto* rb = reinterpret_cast<std::uint8_t*>(r.raw);
  const auto* bb = reinterpret_cast<const std::uint8_t*>(bytes.raw);
  const auto* ib = reinterpret_cast<const std::uint8_t*>(idx.raw);
  constexpr std::size_t kBytes = sizeof(TI) * NI;
  for (std::size_t i = 0; i < kBytes; ++i) {
    const std::size_t block = (i / 16) * 16;          // start of this 128-bit block
    rb[i] = bb[block + (ib[i] & 15)];                 // block-local index (low 4 bits)
  }
  return r;
}
// TableLookupBytesInRange: index bytes in [0,15] guaranteed by caller. The masked form is a valid
// (and memory-safe) implementation on that domain -- the reference backend keeps it.
template <typename T, std::size_t N, typename TI, std::size_t NI>
HYLE_FORCE_INLINE Vec<TI, NI> TableLookupBytesInRange(Vec<T, N> bytes, Vec<TI, NI> idx) noexcept {
  return TableLookupBytes(bytes, idx);
}
template <typename T, std::size_t N, typename TI, std::size_t NI>
HYLE_FORCE_INLINE Vec<TI, NI> TableLookupBytesOr0(Vec<T, N> bytes, Vec<TI, NI> idx) noexcept {
  static_assert(sizeof(T) * N == sizeof(TI) * NI, "TableLookupBytesOr0: same total byte width");
  Vec<TI, NI> r;
  auto* rb = reinterpret_cast<std::uint8_t*>(r.raw);
  const auto* bb = reinterpret_cast<const std::uint8_t*>(bytes.raw);
  const auto* ib = reinterpret_cast<const std::uint8_t*>(idx.raw);
  constexpr std::size_t kBytes = sizeof(TI) * NI;
  for (std::size_t i = 0; i < kBytes; ++i) {
    const std::size_t block = (i / 16) * 16;
    rb[i] = (ib[i] & 0x80) ? std::uint8_t{0} : bb[block + (ib[i] & 15)];  // high bit -> 0
  }
  return r;
}

// ---- gather / scatter (indexed memory access) ---------------------------------------
// GatherIndex(d, base, idx): r[i] = base[idx[i]]; ScatterIndex is the inverse. The index
// vector is a same-lane-count vector of the matching-width integer type. Native on AVX2
// (gather) / AVX-512 (both); scalar + SSE4 + NEON loop (Highway's own fallback for those).
// The scalar floor is the reference loop.
template <class D, typename IdxT>
HYLE_FORCE_INLINE VFromD<D> GatherIndex(D, const TFromD<D>* base, Vec<IdxT, HYLE_SIMD_MAX_LANES_D(D)> idx) noexcept {
  VFromD<D> r;
  for (std::size_t i = 0; i < HYLE_SIMD_MAX_LANES_D(D); ++i)
    r.raw[i] = base[static_cast<std::ptrdiff_t>(idx.raw[i])];
  return r;
}
template <class D, typename IdxT>
HYLE_FORCE_INLINE void ScatterIndex(VFromD<D> v, D, TFromD<D>* base, Vec<IdxT, HYLE_SIMD_MAX_LANES_D(D)> idx) noexcept {
  for (std::size_t i = 0; i < HYLE_SIMD_MAX_LANES_D(D); ++i)
    base[static_cast<std::ptrdiff_t>(idx.raw[i])] = v.raw[i];
}

// ---- compress / expand (mask-driven pack / unpack) ----------------------------------
// Compress(v, m): left-pack the lanes where m is true (order preserved), remaining lanes
// undefined-but-typically-zeroed. CompressStore writes only the true lanes to p and returns
// the count. Expand is the inverse (scatter true-mask lanes back out). Native vpcompress/
// vpexpand on AVX-512; the scalar floor + SSE4/AVX2/NEON use a store+left-pack loop.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> Compress(Vec<T, N> v, Mask<T, N> m) noexcept {
  Vec<T, N> r;
  std::size_t k = 0;
  for (std::size_t i = 0; i < N; ++i)
    if (m.raw[i]) r.raw[k++] = v.raw[i];
  for (; k < N; ++k) r.raw[k] = T{0};
  return r;
}
template <class D>
HYLE_FORCE_INLINE std::size_t CompressStore(VFromD<D> v, Mask<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m,
                                            D, TFromD<D>* p) noexcept {
  std::size_t k = 0;
  for (std::size_t i = 0; i < HYLE_SIMD_MAX_LANES_D(D); ++i)
    if (m.raw[i]) p[k++] = v.raw[i];
  return k;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> Expand(Vec<T, N> v, Mask<T, N> m) noexcept {
  Vec<T, N> r;
  std::size_t k = 0;
  for (std::size_t i = 0; i < N; ++i) r.raw[i] = m.raw[i] ? v.raw[k++] : T{0};
  return r;
}

// ---- bit counts / sign helpers / misc (mostly built on primitives) ------------------
// PopulationCount / LeadingZeroCount / TrailingZeroCount: per-lane bit counts on integer
// lanes (std::popcount / countl_zero / countr_zero on the unsigned lane). Native vpopcnt/
// vplzcnt on AVX-512, PSHUFB nibble-LUT / float-exponent tricks on the others; scalar is std.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> PopulationCount(Vec<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "PopulationCount is integer-lane only");
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i)
    r.raw[i] = static_cast<T>(std::popcount(static_cast<std::make_unsigned_t<T>>(v.raw[i])));
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> LeadingZeroCount(Vec<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "LeadingZeroCount is integer-lane only");
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i)
    r.raw[i] = static_cast<T>(std::countl_zero(static_cast<std::make_unsigned_t<T>>(v.raw[i])));
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> TrailingZeroCount(Vec<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "TrailingZeroCount is integer-lane only");
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i)
    r.raw[i] = static_cast<T>(std::countr_zero(static_cast<std::make_unsigned_t<T>>(v.raw[i])));
  return r;
}

// Not: bitwise complement. Clamp(v,lo,hi) = Min(Max(v,lo),hi). RotateLeft/Right<k>: bit
// rotate within each integer lane. These compose from existing ops on every backend, so the
// definitions are identical -- the scalar floor is the reference.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> Not(Vec<T, N> v) noexcept {
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i) {
    using U = std::make_unsigned_t<std::conditional_t<std::is_integral_v<T>, T, std::uint64_t>>;
    if constexpr (std::is_integral_v<T>) r.raw[i] = static_cast<T>(~static_cast<U>(v.raw[i]));
    else { U u; std::memcpy(&u, &v.raw[i], sizeof(T)); u = ~u; std::memcpy(&r.raw[i], &u, sizeof(T)); }
  }
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> Clamp(Vec<T, N> v, Vec<T, N> lo, Vec<T, N> hi) noexcept {
  return Min(Max(v, lo), hi);
}
template <int kBits, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> RotateRight(Vec<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "RotateRight is integer-lane only");
  constexpr int kSizeBits = static_cast<int>(sizeof(T) * 8);
  static_assert(kBits >= 0 && kBits < kSizeBits, "rotate count in [0,bits)");
  Vec<T, N> r;
  using U = std::make_unsigned_t<T>;
  for (std::size_t i = 0; i < N; ++i) {
    const U u = static_cast<U>(v.raw[i]);
    r.raw[i] = static_cast<T>(kBits == 0 ? u : ((u >> kBits) | (u << (kSizeBits - kBits))));
  }
  return r;
}
template <int kBits, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> RotateLeft(Vec<T, N> v) noexcept {
  constexpr int kSizeBits = static_cast<int>(sizeof(T) * 8);
  return RotateRight<(kSizeBits - kBits) % kSizeBits>(v);
}

// Sign helpers (float lanes). CopySign(mag,sign): magnitude of mag, sign of sign. Broadcast
// SignBit: each lane -> all-ones if negative else 0 (as the same lane type's bit pattern).
// IfNegativeThenElse(v,yes,no): per lane, v<0 ? yes : no. ZeroIfNegative(v): v<0 ? 0 : v.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> CopySign(Vec<T, N> mag, Vec<T, N> sign) noexcept {
  static_assert(std::is_floating_point_v<T>, "CopySign is float-lane only");
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i) r.raw[i] = std::copysign(mag.raw[i], sign.raw[i]);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> BroadcastSignBit(Vec<T, N> v) noexcept {
  static_assert(std::is_integral_v<T> && std::is_signed_v<T>, "BroadcastSignBit is signed-int only");
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i) r.raw[i] = static_cast<T>(v.raw[i] < 0 ? ~T{0} : T{0});
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> IfNegativeThenElse(Vec<T, N> v, Vec<T, N> yes, Vec<T, N> no) noexcept {
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i) r.raw[i] = (v.raw[i] < T{0}) ? yes.raw[i] : no.raw[i];
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> ZeroIfNegative(Vec<T, N> v) noexcept {
  Vec<T, N> r;
  for (std::size_t i = 0; i < N; ++i) r.raw[i] = (v.raw[i] < T{0}) ? T{0} : v.raw[i];
  return r;
}

// ---- AES + carryless multiply (simd-08 C, software reference) ------------------------
// The scalar floor is the SOFTWARE reference the hardware AES-NI / PCLMULQDQ / ARM-crypto
// backends must match. AES operates on a 128-bit block = 16 bytes (lane type irrelevant, viewed
// as bytes). Semantics match x86 AES-NI exactly (Highway contract): AESRound =
// MixColumns(SubBytes(ShiftRows(state))) ^ round_key.
// Software-only floor: wrap the shared byte-level cores (detail::aes_*_soft / clmul64 from
// backend/crypto_soft.h) around the Vec<->16-byte state pun. The hardware backends override
// these with AES-NI / PMULL when the ISA extension is present; the byte order is identical so
// results are bit-for-bit equal. All AES ops need a 128-bit vector (sizeof(T)*N == 16).
namespace detail {
template <typename T, std::size_t N>
HYLE_FORCE_INLINE void aes_bytes(const Vec<T, N>& v, std::uint8_t s[16]) noexcept {
  static_assert(sizeof(T) * N == 16, "AES op needs a 128-bit vector");
  std::memcpy(s, v.raw, 16);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> aes_from(const std::uint8_t s[16]) noexcept {
  Vec<T, N> r; std::memcpy(r.raw, s, 16); return r;
}
}  // namespace detail

template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> AESRound(Vec<T, N> state, Vec<T, N> key) noexcept {
  std::uint8_t s[16], k[16]; detail::aes_bytes(state, s); detail::aes_bytes(key, k);
  detail::aes_round_soft(s, k);
  return detail::aes_from<T, N>(s);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> AESLastRound(Vec<T, N> state, Vec<T, N> key) noexcept {
  std::uint8_t s[16], k[16]; detail::aes_bytes(state, s); detail::aes_bytes(key, k);
  detail::aes_last_round_soft(s, k);
  return detail::aes_from<T, N>(s);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> AESRoundInv(Vec<T, N> state, Vec<T, N> key) noexcept {
  std::uint8_t s[16], k[16]; detail::aes_bytes(state, s); detail::aes_bytes(key, k);
  detail::aes_round_inv_soft(s, k);
  return detail::aes_from<T, N>(s);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> AESLastRoundInv(Vec<T, N> state, Vec<T, N> key) noexcept {
  std::uint8_t s[16], k[16]; detail::aes_bytes(state, s); detail::aes_bytes(key, k);
  detail::aes_last_round_inv_soft(s, k);
  return detail::aes_from<T, N>(s);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> AESInvMixColumns(Vec<T, N> state) noexcept {
  std::uint8_t s[16]; detail::aes_bytes(state, s); detail::aes_inv_mix_columns(s);
  return detail::aes_from<T, N>(s);
}
// AESKeyGenAssist<kRcon>(v): key-schedule helper, matches _mm_aeskeygenassist_si128(v, kRcon).
template <std::uint8_t kRcon, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> AESKeyGenAssist(Vec<T, N> v) noexcept {
  std::uint8_t s[16]; detail::aes_bytes(v, s); detail::aes_keygen_assist_soft(s, kRcon);
  return detail::aes_from<T, N>(s);
}
// CLMulLower/Upper(a, b): carryless multiply of the low / high 64 bits of the 128-bit block
// into a 128-bit product (x86 PCLMULQDQ, ARM PMULL). Result lanes viewed as the input type.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> CLMulLower(Vec<T, N> a, Vec<T, N> b) noexcept {
  static_assert(sizeof(T) * N == 16, "CLMul needs a 128-bit vector");
  std::uint64_t av[2], bv[2], o[2];
  std::memcpy(av, a.raw, 16); std::memcpy(bv, b.raw, 16);
  detail::clmul64(av[0], bv[0], o);
  Vec<T, N> r; std::memcpy(r.raw, o, 16); return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> CLMulUpper(Vec<T, N> a, Vec<T, N> b) noexcept {
  static_assert(sizeof(T) * N == 16, "CLMul needs a 128-bit vector");
  std::uint64_t av[2], bv[2], o[2];
  std::memcpy(av, a.raw, 16); std::memcpy(bv, b.raw, 16);
  detail::clmul64(av[1], bv[1], o);
  Vec<T, N> r; std::memcpy(r.raw, o, 16); return r;
}


// ---- simd-18 批E: 128-bit block ops (single block = identity / Interleave* alias) ----
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> SwapAdjacentBlocks(Vec<T, N> v) noexcept { return v; }
template <class D, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> ReverseBlocks(D, Vec<T, N> v) noexcept { return v; }
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> OddEvenBlocks(Vec<T, N> /*odd*/, Vec<T, N> even) noexcept { return even; }
template <int kBlock, class D, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> InsertBlock(D, Vec<T, N> /*v*/, Vec<T, N> blk) noexcept {
  static_assert(kBlock == 0, "single-block InsertBlock: kBlock==0 only");
  return blk;
}
template <int kB, class D, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> SlideUpBlocks(D, Vec<T, N> v) noexcept {
  static_assert(kB == 0, "single-block SlideUpBlocks: kB==0 only");
  return v;
}
template <int kB, class D, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> SlideDownBlocks(D, Vec<T, N> v) noexcept {
  static_assert(kB == 0, "single-block SlideDownBlocks: kB==0 only");
  return v;
}
template <class D, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> InterleaveEvenBlocks(D, Vec<T, N> a, Vec<T, N> /*b*/) noexcept { return a; }
template <class D, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> InterleaveOddBlocks(D, Vec<T, N> a, Vec<T, N> /*b*/) noexcept { return a; }
template <class D, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> InterleaveLowerBlocks(D, Vec<T, N> a, Vec<T, N> /*b*/) noexcept { return a; }
template <class D, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> InterleaveUpperBlocks(D, Vec<T, N> a, Vec<T, N> /*b*/) noexcept { return a; }
template <class D, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> InterleaveWholeLower(D, Vec<T, N> a, Vec<T, N> b) noexcept {
  return InterleaveLower(a, b);
}
template <class D, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> InterleaveWholeUpper(D, Vec<T, N> a, Vec<T, N> b) noexcept {
  return InterleaveUpper(a, b);
}
// E9 Per4LaneBlockShuffle: 32-bit via store+reload oracle (single 128-block = whole vector for N=4).
template <int i3, int i2, int i1, int i0, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> Per4LaneBlockShuffle(Vec<T, N> v) noexcept {
  static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
                "Per4LaneBlockShuffle: 1/2/4/8-byte lanes");
  if constexpr (i3 == 3 && i2 == 2 && i1 == 1 && i0 == 0) {
    return v;  // identity: no-op (see avx2.h)
  } else {
    alignas(16) T s[N], o[N];
    Store(v, Simd<T, N, 0>(), s);
    if constexpr (sizeof(T) <= 2) {
      // 8/16-bit: kLanesPer128 lanes per 128-block, in groups of 4; out[base+group*4+j]=src idx[j].
      constexpr int sel[4] = {i0, i1, i2, i3};
      constexpr std::size_t kLanesPer128 = 16 / sizeof(T);
      for (std::size_t base = 0; base < N; base += kLanesPer128)
        for (std::size_t g = 0; g < kLanesPer128 && base + g < N; g += 4)
          for (std::size_t j = 0; j < 4 && base + g + j < N; ++j)
            o[base + g + j] = s[base + g + static_cast<std::size_t>(sel[j])];
    } else if constexpr (sizeof(T) == 4) {
      constexpr int sel[4] = {i0, i1, i2, i3};
      constexpr std::size_t blk = 4;  // 4 lanes per 128-bit block for 32-bit
      for (std::size_t base = 0; base < N; base += blk) {
        for (std::size_t j = 0; j < blk && base + j < N; ++j)
          o[base + j] = s[base + static_cast<std::size_t>(sel[j])];
      }
    } else {
      // 64-bit: 2 lanes per 128-block; map i0.. via pairs (Highway uses permute4x64 on wider)
      constexpr int sel[4] = {i0, i1, i2, i3};
      for (std::size_t i = 0; i < N; ++i) {
        const std::size_t base = (i / 2) * 2;
        const int local = static_cast<int>(i % 2);
        // within 2-lane block, only i0/i1 matter for 128; keep simple reverse if imm asks
        o[i] = s[base + (static_cast<std::size_t>(sel[local]) & 1u)];
      }
    }
    return Load(Simd<T, N, 0>(), o);
  }
}
// E11 TwoTablesLookupLanes: dual-source table lookup.
template <class D, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec<T, N> TwoTablesLookupLanes(D d, Vec<T, N> a, Vec<T, N> b,
                                                 Indices<T, N> idx) noexcept {
  alignas(16) T sa[N], sb[N], ix[N], out[N];
  Store(a, d, sa); Store(b, d, sb); Store(idx.idx, d, ix);
  for (std::size_t i = 0; i < N; ++i) {
    const auto k = static_cast<std::size_t>(ix[i]);
    out[i] = (k < N) ? sa[k] : sb[k - N];
  }
  return Load(d, out);
}

}  // namespace hyle::simd
