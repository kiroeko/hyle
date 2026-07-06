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

}  // namespace hyle::simd
