#pragma once

#include <cstddef>
#include <cstdint>
#include <bit>
#include <type_traits>

#include <immintrin.h>  // AVX/AVX2/FMA (and transitively SSE*)

#include "hyle/macros.h"

// backend/avx2.h -- 256-bit AVX2 backend (simd-design landing list #2).
//
// ONE backend per TU (variant-DLL model): pulled in by vec.h only when the TU is built
// with -DHYLE_SIMD_TARGET_AVX2 (+ -mavx2 -mfma -mf16c on GCC/Clang; /arch:AVX2 on MSVC).
// Same op contract as scalar.h / sse4.h, so the unified test/kernel source compiles
// unchanged. Lane types f32/f64/i32/u32; NATIVE FULL WIDTH only: one __m256-family
// register == 32 bytes == 8 f32 / 4 f64 / 8 i32.
//
// Differences from sse4.h (verified against Highway x86_256-inl.h):
//  - float compares use _mm256_cmp_ps/pd with a _CMP_*_OQ predicate (no _mm256_cmpgt_ps).
//  - MulAdd uses TRUE FMA3 (_mm256_fmadd_ps / _fnmadd_ps): an AVX2 build has FMA3, and
//    single-rounding FMA is both faster and more accurate than the sse4 mul+add fallback.
//  - Reductions must FIRST combine the two 128-bit halves (the one lane-crossing trap):
//    all other ops here are per-lane and the two 128-bit halves never interact.

namespace hyle::simd {

namespace detail {
template <typename T>
struct Raw256 {
  using type = __m256i;
};
template <>
struct Raw256<float> {
  using type = __m256;
};
template <>
struct Raw256<double> {
  using type = __m256d;
};
template <typename T>
using Raw256T = typename Raw256<T>::type;

template <typename T>
inline constexpr bool kSupported =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::uint32_t> ||
    std::is_same_v<T, std::int16_t> || std::is_same_v<T, std::uint16_t> ||
    std::is_same_v<T, std::int8_t> || std::is_same_v<T, std::uint8_t> ||
    std::is_same_v<T, std::int64_t> || std::is_same_v<T, std::uint64_t>;

// Integer-width helpers so the per-op constexpr chains stay readable. AVX2 integer ops
// pick the intrinsic by lane WIDTH (epi8/16/32/64), so we branch on sizeof(T) + signedness.
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
struct Vec256 {
  static_assert(detail::kSupported<T>, "avx2 backend lane type unsupported");
  typename detail::Raw256T<T> raw;
  using PrivateT = T;
  static constexpr std::size_t kPrivateN = N;
};
template <typename T, std::size_t N>
struct Mask256 {
  typename detail::Raw256T<T> raw;
  using PrivateT = T;
  static constexpr std::size_t kPrivateN = N;
};

template <class V>
using DFromV = Simd<typename V::PrivateT, V::kPrivateN, 0>;
template <class V>
using TFromV = typename V::PrivateT;

namespace detail {
template <class D>
using V256 = Vec256<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)>;
template <class D>
inline constexpr bool kNativeWidth = HYLE_SIMD_MAX_LANES_D(D) * sizeof(TFromD<D>) == 32;
}  // namespace detail

// ---- initialization -----------------------------------------------------------------
template <class D, typename T2>
HYLE_FORCE_INLINE detail::V256<D> Set(D, T2 t) noexcept {
  using T = TFromD<D>;
  static_assert(detail::kNativeWidth<D>, "avx2 backend is native-width-only (256-bit)");
  detail::V256<D> v;
  if constexpr (std::is_same_v<T, float>) v.raw = _mm256_set1_ps(static_cast<float>(t));
  else if constexpr (std::is_same_v<T, double>) v.raw = _mm256_set1_pd(static_cast<double>(t));
  else if constexpr (detail::kI8<T>) v.raw = _mm256_set1_epi8(static_cast<char>(t));
  else if constexpr (detail::kI16<T>) v.raw = _mm256_set1_epi16(static_cast<short>(t));
  else if constexpr (detail::kI64<T>) v.raw = _mm256_set1_epi64x(static_cast<long long>(t));
  else v.raw = _mm256_set1_epi32(static_cast<int>(t));  // i32/u32
  return v;
}
template <class D>
HYLE_FORCE_INLINE detail::V256<D> Zero(D) noexcept {
  using T = TFromD<D>;
  static_assert(detail::kNativeWidth<D>, "avx2 backend is native-width-only (256-bit)");
  detail::V256<D> v;
  if constexpr (std::is_same_v<T, float>) v.raw = _mm256_setzero_ps();
  else if constexpr (std::is_same_v<T, double>) v.raw = _mm256_setzero_pd();
  else v.raw = _mm256_setzero_si256();
  return v;
}
template <class D>
using VFromD = decltype(Zero(D()));

// ---- load / store -------------------------------------------------------------------
template <class D>
HYLE_FORCE_INLINE detail::V256<D> LoadU(D, const TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  detail::V256<D> v;
  if constexpr (std::is_same_v<T, float>) v.raw = _mm256_loadu_ps(p);
  else if constexpr (std::is_same_v<T, double>) v.raw = _mm256_loadu_pd(p);
  else v.raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
  return v;
}
template <class D>
HYLE_FORCE_INLINE detail::V256<D> Load(D, const TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  detail::V256<D> v;
  if constexpr (std::is_same_v<T, float>) v.raw = _mm256_load_ps(p);
  else if constexpr (std::is_same_v<T, double>) v.raw = _mm256_load_pd(p);
  else v.raw = _mm256_load_si256(reinterpret_cast<const __m256i*>(p));
  return v;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE void StoreU(Vec256<T, N> v, Simd<T, N, 0>, T* p) noexcept {
  if constexpr (std::is_same_v<T, float>) _mm256_storeu_ps(p, v.raw);
  else if constexpr (std::is_same_v<T, double>) _mm256_storeu_pd(p, v.raw);
  else _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v.raw);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE void Store(Vec256<T, N> v, Simd<T, N, 0>, T* p) noexcept {
  if constexpr (std::is_same_v<T, float>) _mm256_store_ps(p, v.raw);
  else if constexpr (std::is_same_v<T, double>) _mm256_store_pd(p, v.raw);
  else _mm256_store_si256(reinterpret_cast<__m256i*>(p), v.raw);
}

// ---- arithmetic ---------------------------------------------------------------------
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> Add(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_add_ps(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm256_add_pd(a.raw, b.raw);
  else if constexpr (detail::kI8<T>) r.raw = _mm256_add_epi8(a.raw, b.raw);
  else if constexpr (detail::kI16<T>) r.raw = _mm256_add_epi16(a.raw, b.raw);
  else if constexpr (detail::kI64<T>) r.raw = _mm256_add_epi64(a.raw, b.raw);
  else r.raw = _mm256_add_epi32(a.raw, b.raw);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> Sub(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_sub_ps(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm256_sub_pd(a.raw, b.raw);
  else if constexpr (detail::kI8<T>) r.raw = _mm256_sub_epi8(a.raw, b.raw);
  else if constexpr (detail::kI16<T>) r.raw = _mm256_sub_epi16(a.raw, b.raw);
  else if constexpr (detail::kI64<T>) r.raw = _mm256_sub_epi64(a.raw, b.raw);
  else r.raw = _mm256_sub_epi32(a.raw, b.raw);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> Mul(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_mul_ps(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm256_mul_pd(a.raw, b.raw);
  else if constexpr (detail::kI16<T>) r.raw = _mm256_mullo_epi16(a.raw, b.raw);
  else if constexpr (detail::kI32<T>) r.raw = _mm256_mullo_epi32(a.raw, b.raw);
  else if constexpr (detail::kI8<T>) {
    // No _mm256_mullo_epi8. Highway's Mul(i8): widen to epi16 in two halves, mullo_epi16,
    // then narrow back keeping the low byte of each 16-bit product (wraparound = the
    // 2's-complement low 8 bits, matching scalar T(x*y)).
    const __m256i lo_mask = _mm256_set1_epi16(0x00FF);
    const __m256i a_lo = _mm256_and_si256(a.raw, lo_mask);              // even bytes
    const __m256i b_lo = _mm256_and_si256(b.raw, lo_mask);
    const __m256i a_hi = _mm256_srli_epi16(a.raw, 8);                   // odd bytes
    const __m256i b_hi = _mm256_srli_epi16(b.raw, 8);
    const __m256i p_lo = _mm256_and_si256(_mm256_mullo_epi16(a_lo, b_lo), lo_mask);
    const __m256i p_hi = _mm256_slli_epi16(_mm256_mullo_epi16(a_hi, b_hi), 8);
    r.raw = _mm256_or_si256(p_lo, p_hi);
  } else {
    // i64/u64: no _mm256_mullo_epi64 on AVX2 (it's AVX-512-DQ). Highway emulates via
    // 32x32->64 partial products: lo*lo + ((hi_a*lo_b + lo_a*hi_b) << 32). Low 64 bits
    // are exact for both signed and unsigned (2's complement wraparound = scalar).
    const __m256i lo_a = a.raw, lo_b = b.raw;
    const __m256i hi_a = _mm256_srli_epi64(a.raw, 32);
    const __m256i hi_b = _mm256_srli_epi64(b.raw, 32);
    const __m256i ll = _mm256_mul_epu32(lo_a, lo_b);                    // lo_a*lo_b (64b)
    const __m256i cross = _mm256_add_epi64(_mm256_mul_epu32(hi_a, lo_b),
                                           _mm256_mul_epu32(lo_a, hi_b));
    r.raw = _mm256_add_epi64(ll, _mm256_slli_epi64(cross, 32));
  }
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> Div(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  static_assert(std::is_floating_point_v<T>, "Div is float/double only");
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_div_ps(a.raw, b.raw);
  else r.raw = _mm256_div_pd(a.raw, b.raw);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> Min(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_min_ps(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm256_min_pd(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int8_t>) r.raw = _mm256_min_epi8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint8_t>) r.raw = _mm256_min_epu8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) r.raw = _mm256_min_epi16(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint16_t>) r.raw = _mm256_min_epu16(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) r.raw = _mm256_min_epi32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint32_t>) r.raw = _mm256_min_epu32(a.raw, b.raw);
  else {  // i64/u64: no AVX2 native min -> select via Gt (Highway emulation)
    r = IfThenElse(Gt(a, b), b, a);  // min = a>b ? b : a
  }
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> Max(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_max_ps(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm256_max_pd(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int8_t>) r.raw = _mm256_max_epi8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint8_t>) r.raw = _mm256_max_epu8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) r.raw = _mm256_max_epi16(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint16_t>) r.raw = _mm256_max_epu16(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) r.raw = _mm256_max_epi32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint32_t>) r.raw = _mm256_max_epu32(a.raw, b.raw);
  else {  // i64/u64: no AVX2 native max -> select via Gt (Highway emulation)
    r = IfThenElse(Gt(a, b), a, b);  // max = a>b ? a : b
  }
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> Neg(Vec256<T, N> a) noexcept {
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_xor_ps(a.raw, _mm256_set1_ps(-0.0f));
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm256_xor_pd(a.raw, _mm256_set1_pd(-0.0));
  else if constexpr (detail::kI8<T>) r.raw = _mm256_sub_epi8(_mm256_setzero_si256(), a.raw);
  else if constexpr (detail::kI16<T>) r.raw = _mm256_sub_epi16(_mm256_setzero_si256(), a.raw);
  else if constexpr (detail::kI64<T>) r.raw = _mm256_sub_epi64(_mm256_setzero_si256(), a.raw);
  else r.raw = _mm256_sub_epi32(_mm256_setzero_si256(), a.raw);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> Abs(Vec256<T, N> a) noexcept {
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, float>)
    r.raw = _mm256_and_ps(a.raw, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
  else if constexpr (std::is_same_v<T, double>)
    r.raw = _mm256_and_pd(a.raw, _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)));
  else if constexpr (std::is_same_v<T, std::int8_t>) r.raw = _mm256_abs_epi8(a.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) r.raw = _mm256_abs_epi16(a.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) r.raw = _mm256_abs_epi32(a.raw);
  else if constexpr (std::is_same_v<T, std::int64_t>) {
    // No _mm256_abs_epi64 on AVX2 (AVX-512-VL). Highway: mask = a>>63 (arithmetic via
    // cmpgt 0), abs = (a ^ mask) - mask. Emulate arithmetic >>63 with cmpgt(0, a).
    const __m256i neg = _mm256_cmpgt_epi64(_mm256_setzero_si256(), a.raw);  // all-ones where a<0
    r.raw = _mm256_sub_epi64(_mm256_xor_si256(a.raw, neg), neg);
  } else r.raw = a.raw;  // unsigned: abs is identity
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> Sqrt(Vec256<T, N> a) noexcept {
  static_assert(std::is_floating_point_v<T>, "Sqrt is float/double only");
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_sqrt_ps(a.raw);
  else r.raw = _mm256_sqrt_pd(a.raw);
  return r;
}

// ---- float rounding to integral value (result stays float) --------------------------
// Round=nearest-even, Floor=toward -inf, Ceil=toward +inf, Trunc=toward zero. One
// _mm256_round_ps/pd with the matching _MM_FROUND flag; _MM_FROUND_NO_EXC suppresses the
// inexact exception (Highway does the same). Float/double only.
#define HYLE_SIMD_AVX2_ROUND(NAME, MODE)                                     \
  template <typename T, std::size_t N>                                       \
  HYLE_FORCE_INLINE Vec256<T, N> NAME(Vec256<T, N> a) noexcept {             \
    static_assert(std::is_floating_point_v<T>, #NAME " is float/double only"); \
    Vec256<T, N> r;                                                          \
    if constexpr (std::is_same_v<T, float>)                                  \
      r.raw = _mm256_round_ps(a.raw, (MODE) | _MM_FROUND_NO_EXC);            \
    else r.raw = _mm256_round_pd(a.raw, (MODE) | _MM_FROUND_NO_EXC);         \
    return r;                                                                \
  }
HYLE_SIMD_AVX2_ROUND(Round, _MM_FROUND_TO_NEAREST_INT)
HYLE_SIMD_AVX2_ROUND(Floor, _MM_FROUND_TO_NEG_INF)
HYLE_SIMD_AVX2_ROUND(Ceil, _MM_FROUND_TO_POS_INF)
HYLE_SIMD_AVX2_ROUND(Trunc, _MM_FROUND_TO_ZERO)
#undef HYLE_SIMD_AVX2_ROUND

// TRUE FMA3 (single rounding) -- the reason an AVX2 backend beats scalar on accuracy
// as well as speed. Integer falls back to mul+add (no integer FMA on x86).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> MulAdd(Vec256<T, N> a, Vec256<T, N> b,
                                      Vec256<T, N> c) noexcept {
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_fmadd_ps(a.raw, b.raw, c.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm256_fmadd_pd(a.raw, b.raw, c.raw);
  else r = Add(Mul(a, b), c);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> NegMulAdd(Vec256<T, N> a, Vec256<T, N> b,
                                         Vec256<T, N> c) noexcept {
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_fnmadd_ps(a.raw, b.raw, c.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm256_fnmadd_pd(a.raw, b.raw, c.raw);
  else r = Sub(c, Mul(a, b));
  return r;
}

// ---- bitwise ------------------------------------------------------------------------
#define HYLE_AVX2_BITOP(NAME, PS, PD, SI)                                  \
  template <typename T, std::size_t N>                                     \
  HYLE_FORCE_INLINE Vec256<T, N> NAME(Vec256<T, N> a, Vec256<T, N> b) noexcept { \
    Vec256<T, N> r;                                                        \
    if constexpr (std::is_same_v<T, float>) r.raw = PS(a.raw, b.raw);      \
    else if constexpr (std::is_same_v<T, double>) r.raw = PD(a.raw, b.raw);\
    else r.raw = SI(a.raw, b.raw);                                         \
    return r;                                                              \
  }
HYLE_AVX2_BITOP(And, _mm256_and_ps, _mm256_and_pd, _mm256_and_si256)
HYLE_AVX2_BITOP(Or, _mm256_or_ps, _mm256_or_pd, _mm256_or_si256)
HYLE_AVX2_BITOP(Xor, _mm256_xor_ps, _mm256_xor_pd, _mm256_xor_si256)
// AndNot(a,b) = ~a & b (Highway order): _mm256_andnot_*(x,y) computes ~x & y, same shape.
HYLE_AVX2_BITOP(AndNot, _mm256_andnot_ps, _mm256_andnot_pd, _mm256_andnot_si256)
#undef HYLE_AVX2_BITOP

// ---- shifts (integer lanes, compile-time immediate count) ---------------------------
// AVX2 has slli/srli_epi16/32/64 (logical) and srai_epi16/32 (arithmetic). GAPS emulated
// (mirroring Highway x86_256): epi8 has NO shift (do it as epi16 + mask off the bits that
// cross the byte boundary); srai_epi64 does not exist on AVX2 (arithmetic via srli + sign
// fixup). ShiftRight is arithmetic for signed T, logical for unsigned (matches scalar >>).
template <int kBits, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> ShiftLeft(Vec256<T, N> a) noexcept {
  static_assert(std::is_integral_v<T>, "shifts are integer-lane only");
  Vec256<T, N> r;
  if constexpr (detail::kI16<T>) r.raw = _mm256_slli_epi16(a.raw, kBits);
  else if constexpr (detail::kI32<T>) r.raw = _mm256_slli_epi32(a.raw, kBits);
  else if constexpr (detail::kI64<T>) r.raw = _mm256_slli_epi64(a.raw, kBits);
  else {  // i8/u8: shift as epi16 then clear the low kBits of each high byte that leaked in
    const __m256i shifted = _mm256_slli_epi16(a.raw, kBits);
    r.raw = _mm256_and_si256(shifted, _mm256_set1_epi8(static_cast<char>(0xFFu << kBits)));
  }
  return r;
}
template <int kBits, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> ShiftRight(Vec256<T, N> a) noexcept {
  static_assert(std::is_integral_v<T>, "shifts are integer-lane only");
  Vec256<T, N> r;
  if constexpr (detail::kI16<T>) {
    if constexpr (detail::kSigned<T>) r.raw = _mm256_srai_epi16(a.raw, kBits);
    else r.raw = _mm256_srli_epi16(a.raw, kBits);
  } else if constexpr (detail::kI32<T>) {
    if constexpr (detail::kSigned<T>) r.raw = _mm256_srai_epi32(a.raw, kBits);
    else r.raw = _mm256_srli_epi32(a.raw, kBits);
  } else if constexpr (detail::kI64<T>) {
    if constexpr (detail::kSigned<T>) {
      // No _mm256_srai_epi64 on AVX2. Logical shift, then OR in the sign bits:
      // sign = (a >> 63 arithmetic) replicated; arith_sr = (a>>>k) | (sign << (64-k)).
      const __m256i logical = _mm256_srli_epi64(a.raw, kBits);
      const __m256i sign = _mm256_cmpgt_epi64(_mm256_setzero_si256(), a.raw);  // all-ones if neg
      const __m256i sfill = _mm256_slli_epi64(sign, 64 - kBits);
      r.raw = (kBits == 0) ? a.raw : _mm256_or_si256(logical, sfill);
    } else r.raw = _mm256_srli_epi64(a.raw, kBits);
  } else {  // i8/u8
    if constexpr (detail::kSigned<T>) {
      // i8 arithmetic right shift by k (no native epi8 shift on AVX2). Highway idiom:
      //   logical = (a >> k) as bytes, cleared of the k high bits that leaked across bytes
      //   sign    = bytes that are negative (high bit set) get the top k bits filled with 1
      // Do the byte logical shift via epi16 shift + mask, then OR in sign-fill where a<0.
      const __m256i logical =
          _mm256_and_si256(_mm256_srli_epi16(a.raw, kBits),
                           _mm256_set1_epi8(static_cast<char>(0xFFu >> kBits)));
      // neg byte mask: a's sign bit broadcast to a full 0xFF byte where the byte is negative.
      const __m256i is_neg = _mm256_cmpgt_epi8(_mm256_setzero_si256(), a.raw);  // 0xFF if a<0
      const __m256i signfill = _mm256_and_si256(
          is_neg, _mm256_set1_epi8(static_cast<char>(static_cast<unsigned char>(0xFFu << (8 - kBits)))));
      r.raw = _mm256_or_si256(logical, signfill);
    } else {  // u8 logical
      const __m256i shifted = _mm256_srli_epi16(a.raw, kBits);
      r.raw = _mm256_and_si256(shifted, _mm256_set1_epi8(static_cast<char>(0xFFu >> kBits)));
    }
  }
  return r;
}

// ---- per-lane variable shifts (Shl/Shr) ---------------------------------------------
// AVX2 has sllv/srlv_epi32/64 and srav_epi32 native. Gaps: no 16-bit variable shift (do it
// per-lane via 32-bit widen halves is heavy -- instead promote-free trick: 16-bit uses the
// scalar-store fallback, rare); no srav_epi64 (logical + sign fixup, as in ShiftRight<k>).
// 8-bit variable shift is not provided (no native, and no HYLE consumer -- 16/32/64 cover
// the quantization/hash needs). Signed Shr = arithmetic, unsigned = logical (matches scalar).
namespace detail {
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> shift_var_scalar(Vec256<T, N> a, Vec256<T, N> b, bool left) noexcept {
  alignas(32) T va[N], vb[N];
  _mm256_store_si256(reinterpret_cast<__m256i*>(va), a.raw);
  _mm256_store_si256(reinterpret_cast<__m256i*>(vb), b.raw);
  for (std::size_t i = 0; i < N; ++i)
    va[i] = left ? static_cast<T>(static_cast<std::make_unsigned_t<T>>(va[i]) << vb[i])
                 : static_cast<T>(va[i] >> vb[i]);
  Vec256<T, N> r;
  r.raw = _mm256_load_si256(reinterpret_cast<const __m256i*>(va));
  return r;
}
}  // namespace detail
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> Shl(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  static_assert(std::is_integral_v<T>, "Shl is integer-lane only");
  Vec256<T, N> r;
  if constexpr (detail::kI32<T>) r.raw = _mm256_sllv_epi32(a.raw, b.raw);
  else if constexpr (detail::kI64<T>) r.raw = _mm256_sllv_epi64(a.raw, b.raw);
  else r = detail::shift_var_scalar<T, N>(a, b, true);  // i8/u8/i16/u16: no native sllv
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> Shr(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  static_assert(std::is_integral_v<T>, "Shr is integer-lane only");
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, std::uint32_t>) r.raw = _mm256_srlv_epi32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) r.raw = _mm256_srav_epi32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint64_t>) r.raw = _mm256_srlv_epi64(a.raw, b.raw);
  else r = detail::shift_var_scalar<T, N>(a, b, false);  // i64 (no srav_epi64) / 8 / 16
  return r;
}

// MulHigh: high 16 bits of the i16/u16 product (native mulhi).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> MulHigh(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  static_assert(std::is_integral_v<T> && sizeof(T) == 2, "MulHigh is 16-bit int only");
  Vec256<T, N> r;
  if constexpr (std::is_signed_v<T>) r.raw = _mm256_mulhi_epi16(a.raw, b.raw);
  else r.raw = _mm256_mulhi_epu16(a.raw, b.raw);
  return r;
}

// WidenMulPairwiseAdd(d32, a_i16, b_i16): _mm256_madd_epi16 multiplies i16 pairs and sums
// adjacent pairs into i32 -- the fixed-point dot core. a/b are FULL i16 (2N lanes), result
// is N i32 lanes.
template <std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec256<std::int32_t, NDst> WidenMulPairwiseAdd(Simd<std::int32_t, NDst, 0>,
                                                                Vec256<SrcT, NSrc> a,
                                                                Vec256<SrcT, NSrc> b) noexcept {
  static_assert(std::is_same_v<SrcT, std::int16_t> && NSrc == 2 * NDst,
                "WidenMulPairwiseAdd: i16 -> i32 pairwise");
  Vec256<std::int32_t, NDst> r;
  r.raw = _mm256_madd_epi16(a.raw, b.raw);
  return r;
}

// AbsDiff: |a-b| per lane. Integers via Sub(Max,Min) (Highway's generic); float via Abs(Sub).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> AbsDiff(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  if constexpr (std::is_floating_point_v<T>) return Abs(Sub(a, b));
  else return Sub(Max(a, b), Min(a, b));
}

// ---- saturated arithmetic + rounding average (8/16-bit integer lanes) ---------------
// Native AVX2: adds/subs_epi8/epu8/epi16/epu16 (saturating), avg_epu8/epu16 (rounded).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> SaturatedAdd(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  static_assert(std::is_integral_v<T> && sizeof(T) <= 2, "SaturatedAdd is 8/16-bit int only");
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, std::int8_t>) r.raw = _mm256_adds_epi8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint8_t>) r.raw = _mm256_adds_epu8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) r.raw = _mm256_adds_epi16(a.raw, b.raw);
  else r.raw = _mm256_adds_epu16(a.raw, b.raw);  // u16
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> SaturatedSub(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  static_assert(std::is_integral_v<T> && sizeof(T) <= 2, "SaturatedSub is 8/16-bit int only");
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, std::int8_t>) r.raw = _mm256_subs_epi8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint8_t>) r.raw = _mm256_subs_epu8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) r.raw = _mm256_subs_epi16(a.raw, b.raw);
  else r.raw = _mm256_subs_epu16(a.raw, b.raw);  // u16
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> AverageRound(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  static_assert(std::is_unsigned_v<T> && sizeof(T) <= 2, "AverageRound is u8/u16 only");
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, std::uint8_t>) r.raw = _mm256_avg_epu8(a.raw, b.raw);
  else r.raw = _mm256_avg_epu16(a.raw, b.raw);  // u16
  return r;
}

// ---- type conversion (same lane count: f32<->i32) -----------------------------------
// ConvertTo: float<->int truncating (cvtt = toward zero). RoundToInt: f32->i32 round to
// nearest-even (cvtps respects the default MXCSR rounding = RNE). 8 lanes both sides =
// one __m256 <-> one __m256i, so it fits the native-width model (no partial vectors).
// Width-changing PromoteLowerTo/PromoteUpperTo/DemoteTo (half-at-a-time forms) are below.
template <typename DstT, std::size_t N, typename SrcT>
HYLE_FORCE_INLINE Vec256<DstT, N> ConvertTo(Simd<DstT, N, 0>, Vec256<SrcT, N> v) noexcept {
  Vec256<DstT, N> r;
  if constexpr (std::is_same_v<SrcT, float> && std::is_same_v<DstT, std::int32_t>)
    r.raw = _mm256_cvttps_epi32(v.raw);        // truncate toward zero
  else if constexpr (std::is_same_v<SrcT, std::int32_t> && std::is_same_v<DstT, float>)
    r.raw = _mm256_cvtepi32_ps(v.raw);
  else static_assert(sizeof(DstT) == 0, "avx2 ConvertTo: only f32<->i32 (same width) supported");
  return r;
}
template <typename DstT, std::size_t N, typename SrcT>
HYLE_FORCE_INLINE Vec256<DstT, N> RoundToInt(Simd<DstT, N, 0>, Vec256<SrcT, N> v) noexcept {
  static_assert(std::is_same_v<SrcT, float> && std::is_same_v<DstT, std::int32_t>,
                "avx2 RoundToInt: f32->i32");
  Vec256<DstT, N> r;
  r.raw = _mm256_cvtps_epi32(v.raw);           // round to nearest even (default MXCSR)
  return r;
}

// ---- width-changing promote / demote (half-at-a-time, fixed-width friendly) ---------
// PromoteLowerTo/PromoteUpperTo: FULL narrow src (2N lanes in a 256b reg) -> FULL wide dst
// (N lanes). The lower/upper 128-bit half (N narrow lanes) feeds a cvtep* that widens it to
// a full 256b wide vector -- the AVX2 cvtep* intrinsics take a 128b input by design, so no
// partial vectors are needed. Signed src sign-extends (cvtepi), unsigned zero-extends
// (cvtepu). f32->f64 uses cvtps_pd on the 128b half.
namespace detail {
template <typename DstT, typename SrcT>
HYLE_FORCE_INLINE Raw256T<DstT> promote128(__m128i half) noexcept {
  if constexpr (std::is_same_v<SrcT, std::int8_t>) return _mm256_cvtepi8_epi16(half);
  else if constexpr (std::is_same_v<SrcT, std::uint8_t>) return _mm256_cvtepu8_epi16(half);
  else if constexpr (std::is_same_v<SrcT, std::int16_t>) return _mm256_cvtepi16_epi32(half);
  else if constexpr (std::is_same_v<SrcT, std::uint16_t>) return _mm256_cvtepu16_epi32(half);
  else if constexpr (std::is_same_v<SrcT, std::int32_t>) return _mm256_cvtepi32_epi64(half);
  else return _mm256_cvtepu32_epi64(half);  // u32 -> u64
}
}  // namespace detail
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec256<DstT, NDst> PromoteLowerTo(Simd<DstT, NDst, 0>, Vec256<SrcT, NSrc> v) noexcept {
  static_assert(NSrc == 2 * NDst && sizeof(DstT) > sizeof(SrcT), "PromoteLowerTo widens, src 2x lanes");
  Vec256<DstT, NDst> r;
  if constexpr (std::is_same_v<SrcT, float>) r.raw = _mm256_cvtps_pd(_mm256_castps256_ps128(v.raw));
  else r.raw = detail::promote128<DstT, SrcT>(_mm256_castsi256_si128(v.raw));  // lower 128
  return r;
}
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec256<DstT, NDst> PromoteUpperTo(Simd<DstT, NDst, 0>, Vec256<SrcT, NSrc> v) noexcept {
  static_assert(NSrc == 2 * NDst && sizeof(DstT) > sizeof(SrcT), "PromoteUpperTo widens, src 2x lanes");
  Vec256<DstT, NDst> r;
  if constexpr (std::is_same_v<SrcT, float>)
    r.raw = _mm256_cvtps_pd(_mm256_extractf128_ps(v.raw, 1));
  else r.raw = detail::promote128<DstT, SrcT>(_mm256_extracti128_si256(v.raw, 1));  // upper 128
  return r;
}
// DemoteTo(dn, a, b): two FULL wide vectors -> one FULL narrow, SATURATING (integer) with
// lanes ORDERED [a..., b...]. AVX2 packs/packus operate per-128-block, yielding
// [a.lo,b.lo,a.hi,b.hi]; a permute4x64(...,3,1,2,0) restores [a.lo,a.hi,b.lo,b.hi]. f64->f32
// uses cvtpd_ps on each and inserts the second into the high 128.
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec256<DstT, NDst> DemoteTo(Simd<DstT, NDst, 0>, Vec256<SrcT, NSrc> a,
                                              Vec256<SrcT, NSrc> b) noexcept {
  static_assert(NDst == 2 * NSrc, "DemoteTo: narrow dst has 2x each wide src's lanes");
  Vec256<DstT, NDst> r;
  if constexpr (std::is_same_v<SrcT, double>) {  // f64 -> f32
    const __m128 alo = _mm256_cvtpd_ps(a.raw), blo = _mm256_cvtpd_ps(b.raw);
    r.raw = _mm256_insertf128_ps(_mm256_castps128_ps256(alo), blo, 1);
  } else {
    __m256i packed;
    if constexpr (std::is_same_v<SrcT, std::int16_t> && std::is_same_v<DstT, std::int8_t>)
      packed = _mm256_packs_epi16(a.raw, b.raw);
    else if constexpr (std::is_same_v<SrcT, std::int16_t> && std::is_same_v<DstT, std::uint8_t>)
      packed = _mm256_packus_epi16(a.raw, b.raw);
    else if constexpr (std::is_same_v<SrcT, std::int32_t> && std::is_same_v<DstT, std::int16_t>)
      packed = _mm256_packs_epi32(a.raw, b.raw);
    else if constexpr (std::is_same_v<SrcT, std::int32_t> && std::is_same_v<DstT, std::uint16_t>)
      packed = _mm256_packus_epi32(a.raw, b.raw);
    else static_assert(sizeof(DstT) == 0, "avx2 DemoteTo: unsupported width pair");
    r.raw = _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));  // fix 128-block order
  }
  return r;
}
// ReorderDemote2To: the cheaper block-interleaved order [a.lo,b.lo,a.hi,b.hi] (skips the
// permute) -- for callers that immediately reduce/re-permute and don't care about order.
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec256<DstT, NDst> ReorderDemote2To(Simd<DstT, NDst, 0>, Vec256<SrcT, NSrc> a,
                                                      Vec256<SrcT, NSrc> b) noexcept {
  static_assert(NDst == 2 * NSrc && std::is_integral_v<SrcT>, "ReorderDemote2To: integer 2->1");
  Vec256<DstT, NDst> r;
  if constexpr (std::is_same_v<SrcT, std::int16_t> && std::is_same_v<DstT, std::int8_t>)
    r.raw = _mm256_packs_epi16(a.raw, b.raw);
  else if constexpr (std::is_same_v<SrcT, std::int16_t> && std::is_same_v<DstT, std::uint8_t>)
    r.raw = _mm256_packus_epi16(a.raw, b.raw);
  else if constexpr (std::is_same_v<SrcT, std::int32_t> && std::is_same_v<DstT, std::int16_t>)
    r.raw = _mm256_packs_epi32(a.raw, b.raw);
  else r.raw = _mm256_packus_epi32(a.raw, b.raw);  // i32 -> u16
  return r;
}

// ---- swizzle (lane rearrangement) ---------------------------------------------------
// AVX2's shuffle/unpack/blend operate PER 128-BIT BLOCK; only vpermd/vpermq/vperm2i128 cross
// the 128 boundary. Ops that stay within-block (unpack, blend) map to one intrinsic; ops that
// must cross blocks in a width the permute units don't directly serve use a store+indexed-
// reload fallback (correct + simple; these swizzles aren't a proven hot path -- Highway
// likewise falls back for awkward widths). The scalar backend is the lane-for-lane oracle.
namespace detail {
// store v, apply a per-lane gather from an index function, reload (works for any lane type
// via the backend's own Store/Load, which handle float vs int raw types).
template <typename T, std::size_t N, typename IdxFn>
HYLE_FORCE_INLINE Vec256<T, N> swizzle_via_mem(Vec256<T, N> v, IdxFn idx) noexcept {
  alignas(32) T src[N], dst[N];
  Store(v, Simd<T, N, 0>(), src);
  for (std::size_t i = 0; i < N; ++i) dst[i] = src[idx(i)];
  return Load(Simd<T, N, 0>(), dst);
}
}  // namespace detail

// OddEven(a,b): even lanes <- b, odd <- a. Native blends (per-128-block but the pattern is
// block-symmetric so it's correct full-width).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> OddEven(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_blend_ps(b.raw, a.raw, 0xAA);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm256_blend_pd(b.raw, a.raw, 0xA);
  else if constexpr (sizeof(T) == 4) r.raw = _mm256_blend_epi32(b.raw, a.raw, 0xAA);
  else if constexpr (sizeof(T) == 2) r.raw = _mm256_blend_epi16(b.raw, a.raw, 0xAA);
  else if constexpr (sizeof(T) == 8) r.raw = _mm256_blend_epi32(b.raw, a.raw, 0xCC);  // 64-bit: pairs of 32
  else {  // 8-bit: no blend_epi8 by imm -> byte mask blendv
    const __m256i m = _mm256_set1_epi16(static_cast<short>(0xFF00u));  // odd bytes = a
    r.raw = _mm256_blendv_epi8(b.raw, a.raw, m);
  }
  return r;
}

// InterleaveLower/Upper: unpack -- AVX2 unpack is PER-128-BLOCK, which is exactly Highway's
// InterleaveLower/Upper semantics (they interleave within each 128-bit block).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> InterleaveLower(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_unpacklo_ps(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm256_unpacklo_pd(a.raw, b.raw);
  else if constexpr (sizeof(T) == 1) r.raw = _mm256_unpacklo_epi8(a.raw, b.raw);
  else if constexpr (sizeof(T) == 2) r.raw = _mm256_unpacklo_epi16(a.raw, b.raw);
  else if constexpr (sizeof(T) == 4) r.raw = _mm256_unpacklo_epi32(a.raw, b.raw);
  else r.raw = _mm256_unpacklo_epi64(a.raw, b.raw);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> InterleaveUpper(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_unpackhi_ps(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm256_unpackhi_pd(a.raw, b.raw);
  else if constexpr (sizeof(T) == 1) r.raw = _mm256_unpackhi_epi8(a.raw, b.raw);
  else if constexpr (sizeof(T) == 2) r.raw = _mm256_unpackhi_epi16(a.raw, b.raw);
  else if constexpr (sizeof(T) == 4) r.raw = _mm256_unpackhi_epi32(a.raw, b.raw);
  else r.raw = _mm256_unpackhi_epi64(a.raw, b.raw);
  return r;
}

// Concat{Lower,Upper}{Lower,Upper}(hi,lo): permute2x128 selects 128-bit halves across the two
// inputs. permute2x128 imm: bits[1:0]=src for result-low-128, bits[5:4]=src for result-high;
// operands are (lo,hi) so index 0=lo.low,1=lo.high,2=hi.low,3=hi.high.
namespace detail {
template <int kImm, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> perm2x128(Vec256<T, N> lo, Vec256<T, N> hi) noexcept {
  Vec256<T, N> r;
  if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4) r.raw = _mm256_permute2f128_ps(lo.raw, hi.raw, kImm);
  else if constexpr (std::is_floating_point_v<T>) r.raw = _mm256_permute2f128_pd(lo.raw, hi.raw, kImm);
  else r.raw = _mm256_permute2x128_si256(lo.raw, hi.raw, kImm);
  return r;
}
}  // namespace detail
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> ConcatLowerLower(Vec256<T, N> hi, Vec256<T, N> lo) noexcept {
  return detail::perm2x128<0x20>(lo, hi);  // [lo.low, hi.low]
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> ConcatUpperUpper(Vec256<T, N> hi, Vec256<T, N> lo) noexcept {
  return detail::perm2x128<0x31>(lo, hi);  // [lo.high, hi.high]
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> ConcatLowerUpper(Vec256<T, N> hi, Vec256<T, N> lo) noexcept {
  return detail::perm2x128<0x21>(lo, hi);  // [lo.high, hi.low]
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> ConcatUpperLower(Vec256<T, N> hi, Vec256<T, N> lo) noexcept {
  // [lo.low, hi.high]: blend the two 128 halves (blend_epi32 mask keeps lo's low, hi's high).
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_blend_ps(hi.raw, lo.raw, 0x0F);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm256_blend_pd(hi.raw, lo.raw, 0x3);
  else r.raw = _mm256_blend_epi32(hi.raw, lo.raw, 0x0F);
  return r;
}

// The remaining swizzles (Reverse, Reverse2, ConcatOdd/Even, DupOdd/Even, TableLookupLanes)
// need lane-index arithmetic that crosses 128-blocks for some widths; use the store+reload
// gather (correct, oracle-matched). vpermd/vpermq specializations for 32/64-bit could be
// added later if profiling shows a hot path -- not proven yet.
template <class D>
HYLE_FORCE_INLINE detail::V256<D> Reverse(D, detail::V256<D> v) noexcept {
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  return detail::swizzle_via_mem(v, [](std::size_t i) { return N - 1 - i; });
}
template <class D>
HYLE_FORCE_INLINE detail::V256<D> Reverse2(D, detail::V256<D> v) noexcept {
  return detail::swizzle_via_mem(v, [](std::size_t i) { return i ^ 1; });
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> ConcatEven(Vec256<T, N> hi, Vec256<T, N> lo) noexcept {
  alignas(32) T sl[N], sh[N], dst[N];
  Store(lo, Simd<T, N, 0>(), sl); Store(hi, Simd<T, N, 0>(), sh);
  for (std::size_t i = 0; i < N / 2; ++i) { dst[i] = sl[2 * i]; dst[N / 2 + i] = sh[2 * i]; }
  return Load(Simd<T, N, 0>(), dst);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> ConcatOdd(Vec256<T, N> hi, Vec256<T, N> lo) noexcept {
  alignas(32) T sl[N], sh[N], dst[N];
  Store(lo, Simd<T, N, 0>(), sl); Store(hi, Simd<T, N, 0>(), sh);
  for (std::size_t i = 0; i < N / 2; ++i) { dst[i] = sl[2 * i + 1]; dst[N / 2 + i] = sh[2 * i + 1]; }
  return Load(Simd<T, N, 0>(), dst);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> DupEven(Vec256<T, N> v) noexcept {
  return detail::swizzle_via_mem(v, [](std::size_t i) { return i & ~std::size_t{1}; });
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> DupOdd(Vec256<T, N> v) noexcept {
  return detail::swizzle_via_mem(v, [](std::size_t i) { return (i & ~std::size_t{1}) | 1; });
}
// TableLookupLanes: runtime per-lane permute. index lanes hold integer VALUES (scalar-oracle
// contract: result[i] = v[idx[i]]). i32/u32 use native vpermd (index bits == value); other
// widths + float use store+reload gather reading the numeric index value.
template <typename T, std::size_t N>
struct Indices256 { Vec256<T, N> idx; };
template <class D>
HYLE_FORCE_INLINE Indices256<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> IndicesFromVec(
    D, detail::V256<D> iv) noexcept {
  return Indices256<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)>{iv};
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> TableLookupLanes(Vec256<T, N> v, Indices256<T, N> ind) noexcept {
  if constexpr (sizeof(T) == 4 && std::is_integral_v<T>)
    return Vec256<T, N>{_mm256_permutevar8x32_epi32(v.raw, ind.idx.raw)};  // index bits == value
  else {  // float / 8 / 16 / 64-bit: store+reload gather reading the numeric index
    alignas(32) T src[N], idx[N], dst[N];
    Store(v, Simd<T, N, 0>(), src); Store(ind.idx, Simd<T, N, 0>(), idx);
    for (std::size_t i = 0; i < N; ++i) dst[i] = src[static_cast<std::size_t>(idx[i]) & (N - 1)];
    return Load(Simd<T, N, 0>(), dst);
  }
}

// ---- comparison -> mask -------------------------------------------------------------

namespace detail {
// Signed cmpgt per width (AVX2 has epi8/16/32/64 native).
template <typename T>
HYLE_FORCE_INLINE __m256i icmpgt_signed(__m256i a, __m256i b) noexcept {
  if constexpr (kI8<T>) return _mm256_cmpgt_epi8(a, b);
  else if constexpr (kI16<T>) return _mm256_cmpgt_epi16(a, b);
  else if constexpr (kI64<T>) return _mm256_cmpgt_epi64(a, b);
  else return _mm256_cmpgt_epi32(a, b);
}
// Unsigned cmpgt: flip the sign bit of each lane, then signed cmpgt (Highway's trick,
// generalized from the existing u32 path to all widths).
template <typename T>
HYLE_FORCE_INLINE __m256i icmpgt_unsigned(__m256i a, __m256i b) noexcept {
  __m256i msb;
  if constexpr (kI8<T>) msb = _mm256_set1_epi8(static_cast<char>(0x80));
  else if constexpr (kI16<T>) msb = _mm256_set1_epi16(static_cast<short>(0x8000));
  else if constexpr (kI64<T>) msb = _mm256_set1_epi64x(static_cast<long long>(0x8000000000000000ULL));
  else msb = _mm256_set1_epi32(static_cast<int>(0x80000000u));
  return icmpgt_signed<std::make_signed_t<T>>(_mm256_xor_si256(a, msb), _mm256_xor_si256(b, msb));
}
// cmpeq per width.
template <typename T>
HYLE_FORCE_INLINE __m256i icmpeq(__m256i a, __m256i b) noexcept {
  if constexpr (kI8<T>) return _mm256_cmpeq_epi8(a, b);
  else if constexpr (kI16<T>) return _mm256_cmpeq_epi16(a, b);
  else if constexpr (kI64<T>) return _mm256_cmpeq_epi64(a, b);
  else return _mm256_cmpeq_epi32(a, b);
}
}  // namespace detail

template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask256<T, N> Eq(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  Mask256<T, N> m;
  if constexpr (std::is_same_v<T, float>) m.raw = _mm256_cmp_ps(a.raw, b.raw, _CMP_EQ_OQ);
  else if constexpr (std::is_same_v<T, double>) m.raw = _mm256_cmp_pd(a.raw, b.raw, _CMP_EQ_OQ);
  else m.raw = detail::icmpeq<T>(a.raw, b.raw);
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask256<T, N> Ne(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  Mask256<T, N> m;
  // _CMP_NEQ_UQ (unordered): NaN compares not-equal -> true, matching scalar (!=),
  // SSE4 (_mm_cmpneq_ps) and NEON (~Eq). _CMP_NEQ_OQ (ordered) wrongly returns false on NaN.
  if constexpr (std::is_same_v<T, float>) m.raw = _mm256_cmp_ps(a.raw, b.raw, _CMP_NEQ_UQ);
  else if constexpr (std::is_same_v<T, double>) m.raw = _mm256_cmp_pd(a.raw, b.raw, _CMP_NEQ_UQ);
  else m.raw = _mm256_xor_si256(detail::icmpeq<T>(a.raw, b.raw), _mm256_set1_epi32(-1));
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask256<T, N> Gt(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  Mask256<T, N> m;
  if constexpr (std::is_same_v<T, float>) m.raw = _mm256_cmp_ps(a.raw, b.raw, _CMP_GT_OQ);
  else if constexpr (std::is_same_v<T, double>) m.raw = _mm256_cmp_pd(a.raw, b.raw, _CMP_GT_OQ);
  else if constexpr (detail::kSigned<T>) m.raw = detail::icmpgt_signed<T>(a.raw, b.raw);
  else m.raw = detail::icmpgt_unsigned<T>(a.raw, b.raw);
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask256<T, N> Lt(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  return Gt(b, a);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask256<T, N> Ge(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  Mask256<T, N> m;
  if constexpr (std::is_same_v<T, float>) m.raw = _mm256_cmp_ps(a.raw, b.raw, _CMP_GE_OQ);
  else if constexpr (std::is_same_v<T, double>) m.raw = _mm256_cmp_pd(a.raw, b.raw, _CMP_GE_OQ);
  else m.raw = _mm256_xor_si256(Lt(a, b).raw, _mm256_set1_epi32(-1));  // Ge = ~Lt (int)
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask256<T, N> Le(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  return Ge(b, a);
}

// ---- mask ops -----------------------------------------------------------------------
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> IfThenElse(Mask256<T, N> m, Vec256<T, N> yes,
                                          Vec256<T, N> no) noexcept {
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_blendv_ps(no.raw, yes.raw, m.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm256_blendv_pd(no.raw, yes.raw, m.raw);
  else r.raw = _mm256_blendv_epi8(no.raw, yes.raw, m.raw);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> IfThenElseZero(Mask256<T, N> m, Vec256<T, N> yes) noexcept {
  return IfThenElse(m, yes, Zero(Simd<T, N, 0>()));
}

namespace detail {
// One bit per LANE, LSB = lane 0. Float/double use the native movemask (1 bit/lane).
// Integers use movemask_epi8 (1 bit/BYTE = 32 bits) then compress to 1 bit/lane by taking
// every sizeof(T)-th byte's bit -- a lane's bytes are all-ones or all-zero (our compares
// produce full-lane masks), so the lane's low byte bit represents the whole lane.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE unsigned mask_bits(Mask256<T, N> m) noexcept {
  if constexpr (std::is_same_v<T, double>)
    return static_cast<unsigned>(_mm256_movemask_pd(m.raw));
  else if constexpr (std::is_same_v<T, float>)
    return static_cast<unsigned>(_mm256_movemask_ps(m.raw));
  else {
    const unsigned bytes = static_cast<unsigned>(_mm256_movemask_epi8(m.raw));  // 32 bits
    if constexpr (sizeof(T) == 1) {
      return bytes;  // 32 lanes, 1 bit/byte already
    } else {
      // Keep every sizeof(T)-th bit (the low byte of each lane), compacting to N bits.
      unsigned out = 0;
      for (std::size_t i = 0; i < N; ++i)
        out |= ((bytes >> (i * sizeof(T))) & 1u) << i;
      return out;
    }
  }
}
}  // namespace detail

template <typename T, std::size_t N>
HYLE_FORCE_INLINE bool AllTrue(Simd<T, N, 0>, Mask256<T, N> m) noexcept {
  // N may be 32 (i8) -> (1u<<32) is UB; compute the full-N mask safely.
  const unsigned full = (N >= 32) ? 0xFFFFFFFFu : ((1u << N) - 1u);
  return detail::mask_bits(m) == full;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE bool AllFalse(Simd<T, N, 0>, Mask256<T, N> m) noexcept {
  return detail::mask_bits(m) == 0u;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE std::size_t CountTrue(Simd<T, N, 0>, Mask256<T, N> m) noexcept {
  return static_cast<std::size_t>(std::popcount(detail::mask_bits(m)));
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE std::ptrdiff_t FindFirstTrue(Simd<T, N, 0>, Mask256<T, N> m) noexcept {
  const unsigned b = detail::mask_bits(m);
  return b ? static_cast<std::ptrdiff_t>(std::countr_zero(b)) : -1;
}

// ---- reductions ---------------------------------------------------------------------
// THE lane-crossing step: AVX2 ops are per-128-lane, so a horizontal reduce must first
// combine the two 128-bit halves (extract hi, op with lo), THEN cascade within 128.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T GetLane(Vec256<T, N> v) noexcept {
  if constexpr (std::is_same_v<T, float>) return _mm_cvtss_f32(_mm256_castps256_ps128(v.raw));
  else if constexpr (std::is_same_v<T, double>) return _mm_cvtsd_f64(_mm256_castpd256_pd128(v.raw));
  else {
    const __m128i lo = _mm256_castsi256_si128(v.raw);
    if constexpr (sizeof(T) == 8) return static_cast<T>(_mm_cvtsi128_si64(lo));
    else return static_cast<T>(_mm_cvtsi128_si32(lo));  // low 32 bits; T-cast truncates for i8/i16
  }
}

namespace detail {
// 128-bit op for the reduction (add/min/max selected by Op tag), on raw __m128-family.
enum class RedOp { kAdd, kMin, kMax };

template <typename T, RedOp op>
HYLE_FORCE_INLINE auto op128(auto a, auto b) noexcept {
  if constexpr (std::is_same_v<T, float>) {
    if constexpr (op == RedOp::kAdd) return _mm_add_ps(a, b);
    else if constexpr (op == RedOp::kMin) return _mm_min_ps(a, b);
    else return _mm_max_ps(a, b);
  } else if constexpr (std::is_same_v<T, double>) {
    if constexpr (op == RedOp::kAdd) return _mm_add_pd(a, b);
    else if constexpr (op == RedOp::kMin) return _mm_min_pd(a, b);
    else return _mm_max_pd(a, b);
  } else {  // i32/u32
    if constexpr (op == RedOp::kAdd) return _mm_add_epi32(a, b);
    else if constexpr (op == RedOp::kMin) {
      if constexpr (std::is_same_v<T, std::int32_t>) return _mm_min_epi32(a, b);
      else return _mm_min_epu32(a, b);
    } else {
      if constexpr (std::is_same_v<T, std::int32_t>) return _mm_max_epi32(a, b);
      else return _mm_max_epu32(a, b);
    }
  }
}

template <typename T, std::size_t N, RedOp op>
HYLE_FORCE_INLINE T reduce256(Vec256<T, N> v) noexcept {
  if constexpr (std::is_same_v<T, double>) {  // 4 lanes -> combine halves -> 2-lane cascade
    __m128d lo = _mm256_castpd256_pd128(v.raw);
    __m128d hi = _mm256_extractf128_pd(v.raw, 1);
    __m128d s = op128<T, op>(lo, hi);                       // 2 lanes
    __m128d sw = _mm_unpackhi_pd(s, s);
    s = op128<T, op>(s, sw);
    return _mm_cvtsd_f64(s);
  } else if constexpr (std::is_same_v<T, float>) {  // 8 lanes
    __m128 lo = _mm256_castps256_ps128(v.raw);
    __m128 hi = _mm256_extractf128_ps(v.raw, 1);
    __m128 s = op128<T, op>(lo, hi);                        // 4 lanes
    s = op128<T, op>(s, _mm_shuffle_ps(s, s, _MM_SHUFFLE(2, 3, 0, 1)));
    s = op128<T, op>(s, _mm_shuffle_ps(s, s, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(s);
  } else {  // 8 lanes i32/u32
    __m128i lo = _mm256_castsi256_si128(v.raw);
    __m128i hi = _mm256_extracti128_si256(v.raw, 1);
    __m128i s = op128<T, op>(lo, hi);                       // 4 lanes
    s = op128<T, op>(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    s = op128<T, op>(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    return static_cast<T>(_mm_cvtsi128_si32(s));
  }
}
}  // namespace detail

// Reductions that lack a clean 256-bit lane-crossing path for narrow/64-bit int lanes
// (i8/u8/i16/u16/i64/u64): store to a stack array and scalar-reduce. Correct + simple;
// the cascade shuffles above are only defined for f32/f64/i32/u32. (Highway likewise
// falls back for the awkward widths -- a horizontal reduce isn't the hot path.)
template <typename T, std::size_t N, detail::RedOp op>
HYLE_FORCE_INLINE T reduce_scalar(Vec256<T, N> v) noexcept {
  alignas(32) T tmp[N];
  _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), v.raw);
  T acc = tmp[0];
  for (std::size_t i = 1; i < N; ++i) {
    if constexpr (op == detail::RedOp::kAdd) acc = static_cast<T>(acc + tmp[i]);
    else if constexpr (op == detail::RedOp::kMin) acc = tmp[i] < acc ? tmp[i] : acc;
    else acc = tmp[i] > acc ? tmp[i] : acc;
  }
  return acc;
}

template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceSum(Simd<T, N, 0>, Vec256<T, N> v) noexcept {
  if constexpr (detail::kI32<T> || std::is_floating_point_v<T>)
    return detail::reduce256<T, N, detail::RedOp::kAdd>(v);
  else return reduce_scalar<T, N, detail::RedOp::kAdd>(v);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceMin(Simd<T, N, 0>, Vec256<T, N> v) noexcept {
  if constexpr (detail::kI32<T> || std::is_floating_point_v<T>)
    return detail::reduce256<T, N, detail::RedOp::kMin>(v);
  else return reduce_scalar<T, N, detail::RedOp::kMin>(v);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceMax(Simd<T, N, 0>, Vec256<T, N> v) noexcept {
  if constexpr (detail::kI32<T> || std::is_floating_point_v<T>)
    return detail::reduce256<T, N, detail::RedOp::kMax>(v);
  else return reduce_scalar<T, N, detail::RedOp::kMax>(v);
}

// Reduction family completion: ReduceMul (product), ReduceAnd/Or (bitwise, integer lanes).
// No native horizontal product/and/or across 256 bits for arbitrary widths, and horizontal
// reduce is not the hot path (same rationale as reduce_scalar), so fold via a stack array.
namespace detail {
enum class RedOp2 { kMul, kAnd, kOr };
template <typename T, std::size_t N, RedOp2 op>
HYLE_FORCE_INLINE T reduce_scalar2(Vec256<T, N> v) noexcept {
  alignas(32) T tmp[N];
  Store(v, Simd<T, N, 0>(), tmp);
  T acc = tmp[0];
  for (std::size_t i = 1; i < N; ++i) {
    if constexpr (op == RedOp2::kMul) acc = static_cast<T>(acc * tmp[i]);
    else if constexpr (op == RedOp2::kAnd) acc = static_cast<T>(acc & tmp[i]);
    else acc = static_cast<T>(acc | tmp[i]);
  }
  return acc;
}
}  // namespace detail
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceMul(Simd<T, N, 0>, Vec256<T, N> v) noexcept {
  return detail::reduce_scalar2<T, N, detail::RedOp2::kMul>(v);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceAnd(Simd<T, N, 0>, Vec256<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "ReduceAnd is integer-lane only");
  return detail::reduce_scalar2<T, N, detail::RedOp2::kAnd>(v);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceOr(Simd<T, N, 0>, Vec256<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "ReduceOr is integer-lane only");
  return detail::reduce_scalar2<T, N, detail::RedOp2::kOr>(v);
}

// Broadcast<i>: splat lane i across all lanes (matrix-row / bias splat). AVX2 cannot cheaply
// permute an arbitrary lane across the 128-bit halves for every lane width, and this is not
// a v1 hot path, so we extract-and-Set uniformly (a future native-permute pass -- vpermps /
// vpbroadcast -- can specialize hot widths). Index-vector swizzles (TableLookup, Gather/
// Scatter, Compress, Interleave/Zip) are DEFERRED as a cohesive effort (see scalar.h note).
template <int kLane, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> Broadcast(Vec256<T, N> v) noexcept {
  static_assert(kLane >= 0 && static_cast<std::size_t>(kLane) < N, "Broadcast lane out of range");
  alignas(32) T tmp[N];
  Store(v, Simd<T, N, 0>(), tmp);
  return Set(Simd<T, N, 0>(), tmp[kLane]);
}

// ---- masked / partial load-store (tail handling) ------------------------------------
// FirstN(d,n) -> mask with the first min(n,N) lanes true. An all-ones lane bit pattern is
// a valid AVX2 blend/movemask selector for EVERY lane type (blendv keys off each lane's
// MSB, all-ones has it set), so we build it via a stack array + Load, uniform over T.
// LoadN/StoreN touch ONLY the first k lanes (fault-safe tail I/O); MaskedLoad mirrors
// Highway's AVX2 narrow path (IfThenElseZero + full LoadU -- caller ensures the vector is
// loadable), BlendedStore writes only on-lanes (fault-safe, mask-driven).
template <class D>
HYLE_FORCE_INLINE Mask256<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> FirstN(D d, std::size_t n) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  T ones;
  auto* ob = reinterpret_cast<unsigned char*>(&ones);
  for (std::size_t k = 0; k < sizeof(T); ++k) ob[k] = 0xFFu;  // all-ones lane pattern
  alignas(32) T tmp[kN];
  for (std::size_t i = 0; i < kN; ++i) tmp[i] = (i < n) ? ones : T{0};
  Mask256<T, kN> m;
  m.raw = Load(d, tmp).raw;
  return m;
}
template <class D>
HYLE_FORCE_INLINE detail::V256<D> MaskedLoad(Mask256<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m,
                                             D d, const TFromD<D>* p) noexcept {
  return IfThenElseZero(m, LoadU(d, p));
}
template <class D>
HYLE_FORCE_INLINE detail::V256<D> MaskedLoadOr(detail::V256<D> no,
                                               Mask256<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m,
                                               D d, const TFromD<D>* p) noexcept {
  return IfThenElse(m, LoadU(d, p), no);
}
template <class D>
HYLE_FORCE_INLINE void BlendedStore(detail::V256<D> v,
                                    Mask256<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m, D d,
                                    TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  alignas(32) T tmp[kN];
  Store(v, d, tmp);
  const unsigned bits = detail::mask_bits(m);
  for (std::size_t i = 0; i < kN; ++i)
    if ((bits >> i) & 1u) p[i] = tmp[i];  // write only on-lanes (fault-safe)
}
template <class D>
HYLE_FORCE_INLINE detail::V256<D> LoadN(D d, const TFromD<D>* p, std::size_t n) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  const std::size_t k = n < kN ? n : kN;
  alignas(32) T tmp[kN] = {};
  for (std::size_t i = 0; i < k; ++i) tmp[i] = p[i];  // read only k lanes (fault-safe)
  return Load(d, tmp);
}
template <class D>
HYLE_FORCE_INLINE void StoreN(detail::V256<D> v, D d, TFromD<D>* p, std::size_t n) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  const std::size_t k = n < kN ? n : kN;
  alignas(32) T tmp[kN];
  Store(v, d, tmp);
  for (std::size_t i = 0; i < k; ++i) p[i] = tmp[i];  // write only k lanes (fault-safe)
}

// ---- gather / scatter + compress / expand -------------------------------------------
// Provided via store+reload (the capability is what matters for an infra lib; native
// vpgatherdd / vpcompressd are perf optimizations gated behind profiling -- AVX2 gather is
// famously slow on many uarchs, and Compress needs a LUT). Store the index/data, do the
// indexed loop, reload -- correct, oracle-matched. The scalar backend is the reference.
template <class D, typename IdxT>
HYLE_FORCE_INLINE detail::V256<D> GatherIndex(D d, const TFromD<D>* base,
                                              Vec256<IdxT, HYLE_SIMD_MAX_LANES_D(D)> idx) noexcept {
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  alignas(32) IdxT ix[kN]; alignas(32) TFromD<D> dst[kN];
  Store(idx, Simd<IdxT, kN, 0>(), ix);
  for (std::size_t i = 0; i < kN; ++i) dst[i] = base[static_cast<std::ptrdiff_t>(ix[i])];
  return Load(d, dst);
}
template <class D, typename IdxT>
HYLE_FORCE_INLINE void ScatterIndex(detail::V256<D> v, D d, TFromD<D>* base,
                                    Vec256<IdxT, HYLE_SIMD_MAX_LANES_D(D)> idx) noexcept {
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  alignas(32) IdxT ix[kN]; alignas(32) TFromD<D> src[kN];
  Store(idx, Simd<IdxT, kN, 0>(), ix); Store(v, d, src);
  for (std::size_t i = 0; i < kN; ++i) base[static_cast<std::ptrdiff_t>(ix[i])] = src[i];
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> Compress(Vec256<T, N> v, Mask256<T, N> m) noexcept {
  alignas(32) T src[N], dst[N] = {};
  Store(v, Simd<T, N, 0>(), src);
  const unsigned bits = detail::mask_bits(m);
  std::size_t k = 0;
  for (std::size_t i = 0; i < N; ++i) if ((bits >> i) & 1u) dst[k++] = src[i];
  return Load(Simd<T, N, 0>(), dst);
}
template <class D>
HYLE_FORCE_INLINE std::size_t CompressStore(detail::V256<D> v,
                                            Mask256<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m,
                                            D d, TFromD<D>* p) noexcept {
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  alignas(32) TFromD<D> src[kN];
  Store(v, d, src);
  const unsigned bits = detail::mask_bits(m);
  std::size_t k = 0;
  for (std::size_t i = 0; i < kN; ++i) if ((bits >> i) & 1u) p[k++] = src[i];
  return k;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> Expand(Vec256<T, N> v, Mask256<T, N> m) noexcept {
  alignas(32) T src[N], dst[N];
  Store(v, Simd<T, N, 0>(), src);
  const unsigned bits = detail::mask_bits(m);
  std::size_t k = 0;
  for (std::size_t i = 0; i < N; ++i) dst[i] = ((bits >> i) & 1u) ? src[k++] : T{0};
  return Load(Simd<T, N, 0>(), dst);
}

// ---- bit counts / sign helpers / misc -----------------------------------------------
// PopulationCount/LeadingZeroCount/TrailingZeroCount via store+reload + std:: (native
// vpopcnt/vplzcnt are AVX-512; the PSHUFB nibble-LUT is a perf optimization gated on
// profiling). Not/Clamp/Rotate/CopySign/... compose from existing ops -- same on every
// backend. The scalar backend is the reference.
namespace detail {
template <typename T, std::size_t N, int kind>
HYLE_FORCE_INLINE Vec256<T, N> bitcount_via_mem(Vec256<T, N> v) noexcept {
  using U = std::make_unsigned_t<T>;
  alignas(32) T s[N], d[N];
  Store(v, Simd<T, N, 0>(), s);
  for (std::size_t i = 0; i < N; ++i) {
    const U u = static_cast<U>(s[i]);
    if constexpr (kind == 0) d[i] = static_cast<T>(std::popcount(u));
    else if constexpr (kind == 1) d[i] = static_cast<T>(std::countl_zero(u));
    else d[i] = static_cast<T>(std::countr_zero(u));
  }
  return Load(Simd<T, N, 0>(), d);
}
}  // namespace detail
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> PopulationCount(Vec256<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "PopulationCount is integer-lane only");
  return detail::bitcount_via_mem<T, N, 0>(v);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> LeadingZeroCount(Vec256<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "LeadingZeroCount is integer-lane only");
  return detail::bitcount_via_mem<T, N, 1>(v);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> TrailingZeroCount(Vec256<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "TrailingZeroCount is integer-lane only");
  return detail::bitcount_via_mem<T, N, 2>(v);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> Not(Vec256<T, N> v) noexcept {
  Vec256<T, N> ones;  // all-ones bit pattern (works for int + float raw)
  if constexpr (std::is_same_v<T, float>) ones.raw = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
  else if constexpr (std::is_same_v<T, double>) ones.raw = _mm256_castsi256_pd(_mm256_set1_epi32(-1));
  else ones.raw = _mm256_set1_epi32(-1);
  return Xor(v, ones);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> Clamp(Vec256<T, N> v, Vec256<T, N> lo, Vec256<T, N> hi) noexcept {
  return Min(Max(v, lo), hi);
}
template <int kBits, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> RotateRight(Vec256<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "RotateRight is integer-lane only");
  constexpr int kSizeBits = static_cast<int>(sizeof(T) * 8);
  static_assert(kBits >= 0 && kBits < kSizeBits, "rotate count in [0,bits)");
  if constexpr (kBits == 0) return v;
  else if constexpr (std::is_unsigned_v<T>)  // logical shifts compose directly
    return Or(ShiftRight<kBits>(v), ShiftLeft<kSizeBits - kBits>(v));
  else {  // signed: ShiftRight is ARITHMETIC (sign-extends) -> wrong for rotate. Do it on the
    // unsigned bit pattern via store+reload (rotate isn't a hot path; keeps semantics exact).
    using U = std::make_unsigned_t<T>;
    alignas(32) T s[N], d[N];
    Store(v, Simd<T, N, 0>(), s);
    for (std::size_t i = 0; i < N; ++i) {
      const U u = static_cast<U>(s[i]);
      d[i] = static_cast<T>((u >> kBits) | (u << (kSizeBits - kBits)));
    }
    return Load(Simd<T, N, 0>(), d);
  }
}
template <int kBits, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> RotateLeft(Vec256<T, N> v) noexcept {
  constexpr int kSizeBits = static_cast<int>(sizeof(T) * 8);
  return RotateRight<(kSizeBits - kBits) % kSizeBits>(v);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> CopySign(Vec256<T, N> mag, Vec256<T, N> sign) noexcept {
  static_assert(std::is_floating_point_v<T>, "CopySign is float-lane only");
  const auto sign_bit = Set(Simd<T, N, 0>(), static_cast<T>(-0.0));  // only the sign bit set
  return Or(AndNot(sign_bit, mag), And(sign_bit, sign));  // |mag| bits with sign's sign bit
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> BroadcastSignBit(Vec256<T, N> v) noexcept {
  static_assert(std::is_integral_v<T> && std::is_signed_v<T>, "BroadcastSignBit is signed-int only");
  return ShiftRight<static_cast<int>(sizeof(T) * 8) - 1>(v);  // arithmetic shift replicates sign
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> IfNegativeThenElse(Vec256<T, N> v, Vec256<T, N> yes, Vec256<T, N> no) noexcept {
  return IfThenElse(Lt(v, Zero(Simd<T, N, 0>())), yes, no);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> ZeroIfNegative(Vec256<T, N> v) noexcept {
  return IfThenElse(Lt(v, Zero(Simd<T, N, 0>())), Zero(Simd<T, N, 0>()), v);
}

}  // namespace hyle::simd
