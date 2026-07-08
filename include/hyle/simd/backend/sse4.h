#pragma once

#include <cstddef>
#include <cstdint>
#include <bit>
#include <type_traits>

#include <nmmintrin.h>  // SSE4.2 (transitively pulls SSE4.1/SSSE3/SSE3/SSE2)

#include "hyle/macros.h"

// backend/sse4.h -- 128-bit SSE4.2-baseline backend (simd-design landing list #2).
//
// ONE backend per TU (variant-DLL model): this header is pulled in by vec.h only when
// the TU is compiled with -DHYLE_SIMD_TARGET_SSE4 (+ -msse4.2 on GCC/Clang; MSVC x64
// has all SSE4.2 intrinsics unconditionally). It defines the SAME op names/signatures
// as backend/scalar.h, so a single source (test/kernel) written against the D-tag
// contract compiles unchanged against scalar OR this backend -- the whole point of the
// abstraction.
//
// Lane types f32/f64 + all integer widths i8/u8/i16/u16/i32/u32/i64/u64. NATIVE FULL
// WIDTH only: one __m128-family register == 16 bytes == 4 f32 / 2 f64 / 16 i8 / 8 i16 /
// 4 i32 / 2 i64. Partial/Half-width vectors deferred.
//
// Intrinsic choices follow Google Highway's x86_128-inl.h (verified, not from memory):
// per-width add/sub (epi8/16/32/64), mullo_epi16/epi32 (i8 via widen-to-epi16, i64 via
// 32x32 partial products), min|max_epi8/epu8/epi16/epu16/epi32/epu32 (SSE4.1) with
// i64/u64 emulated via Gt-select, abs_epi8/16/32 (SSSE3) with i64 via (a^mask)-mask,
// cmpgt_epi8/16/32 + cmpgt_epi64 (SSE4.2) + xor-MSB for unsigned compares, cmpeq_epi8/
// 16/32 + cmpeq_epi64 (SSE4.1), blendv for select. FMA3 is an AVX-era extension NOT
// implied by SSE4.2, so MulAdd here is the mul+add fallback (AVX2 uses true FMA).
// Masks are a vector register, all-1s/all-0s per lane (Highway's non-AVX-512 model).

namespace hyle::simd {

// ---- raw register type per lane type ------------------------------------------------
namespace detail {
template <typename T>
struct Raw128 {
  using type = __m128i;
};  // all integer lanes
template <>
struct Raw128<float> {
  using type = __m128;
};
template <>
struct Raw128<double> {
  using type = __m128d;
};
template <typename T>
using Raw128T = typename Raw128<T>::type;

template <typename T>
inline constexpr bool kSupported =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::uint32_t> ||
    std::is_same_v<T, std::int16_t> || std::is_same_v<T, std::uint16_t> ||
    std::is_same_v<T, std::int8_t> || std::is_same_v<T, std::uint8_t> ||
    std::is_same_v<T, std::int64_t> || std::is_same_v<T, std::uint64_t>;

// Integer-width helpers so the per-op constexpr chains stay readable. SSE integer ops
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

// ---- vector + mask storage (mask = vector of all-1s/all-0s lanes) -------------------
template <typename T, std::size_t N>
struct Vec128 {
  static_assert(detail::kSupported<T>, "sse4 backend lane type unsupported");
  typename detail::Raw128T<T> raw;
  using PrivateT = T;
  static constexpr std::size_t kPrivateN = N;
};

template <typename T, std::size_t N>
struct Mask128 {
  typename detail::Raw128T<T> raw;
  using PrivateT = T;
  static constexpr std::size_t kPrivateN = N;
};

template <class V>
using DFromV = Simd<typename V::PrivateT, V::kPrivateN, 0>;
template <class V>
using TFromV = typename V::PrivateT;

// The 128-bit backend's vector type for tag D. N comes from D (== 16/sizeof(T) for a
// native ScalableTag). A wrong-width tag is a compile error (static_assert below).
namespace detail {
template <class D>
using V128 = Vec128<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)>;
template <class D>
using M128 = Mask128<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)>;
template <class D>
inline constexpr bool kNativeWidth = HYLE_SIMD_MAX_LANES_D(D) * sizeof(TFromD<D>) == 16;
}  // namespace detail

// ---- initialization -----------------------------------------------------------------
template <class D, typename T2>
HYLE_FORCE_INLINE detail::V128<D> Set(D, T2 t) noexcept {
  using T = TFromD<D>;
  static_assert(detail::kNativeWidth<D>, "sse4 backend is native-width-only (128-bit)");
  detail::V128<D> v;
  if constexpr (std::is_same_v<T, float>)
    v.raw = _mm_set1_ps(static_cast<float>(t));
  else if constexpr (std::is_same_v<T, double>)
    v.raw = _mm_set1_pd(static_cast<double>(t));
  else if constexpr (detail::kI8<T>) v.raw = _mm_set1_epi8(static_cast<char>(t));
  else if constexpr (detail::kI16<T>) v.raw = _mm_set1_epi16(static_cast<short>(t));
  else if constexpr (detail::kI64<T>) v.raw = _mm_set1_epi64x(static_cast<long long>(t));
  else v.raw = _mm_set1_epi32(static_cast<int>(t));  // i32/u32
  return v;
}

template <class D>
HYLE_FORCE_INLINE detail::V128<D> Zero(D) noexcept {
  using T = TFromD<D>;
  static_assert(detail::kNativeWidth<D>, "sse4 backend is native-width-only (128-bit)");
  detail::V128<D> v;
  if constexpr (std::is_same_v<T, float>)
    v.raw = _mm_setzero_ps();
  else if constexpr (std::is_same_v<T, double>)
    v.raw = _mm_setzero_pd();
  else
    v.raw = _mm_setzero_si128();
  return v;
}

template <class D>
using VFromD = decltype(Zero(D()));

// ---- load / store -------------------------------------------------------------------
template <class D>
HYLE_FORCE_INLINE detail::V128<D> LoadU(D, const TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  detail::V128<D> v;
  if constexpr (std::is_same_v<T, float>)
    v.raw = _mm_loadu_ps(p);
  else if constexpr (std::is_same_v<T, double>)
    v.raw = _mm_loadu_pd(p);
  else
    v.raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
  return v;
}
template <class D>
HYLE_FORCE_INLINE detail::V128<D> Load(D, const TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  detail::V128<D> v;
  if constexpr (std::is_same_v<T, float>)
    v.raw = _mm_load_ps(p);
  else if constexpr (std::is_same_v<T, double>)
    v.raw = _mm_load_pd(p);
  else
    v.raw = _mm_load_si128(reinterpret_cast<const __m128i*>(p));
  return v;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE void StoreU(Vec128<T, N> v, Simd<T, N, 0>, T* p) noexcept {
  if constexpr (std::is_same_v<T, float>)
    _mm_storeu_ps(p, v.raw);
  else if constexpr (std::is_same_v<T, double>)
    _mm_storeu_pd(p, v.raw);
  else
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v.raw);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE void Store(Vec128<T, N> v, Simd<T, N, 0>, T* p) noexcept {
  if constexpr (std::is_same_v<T, float>)
    _mm_store_ps(p, v.raw);
  else if constexpr (std::is_same_v<T, double>)
    _mm_store_pd(p, v.raw);
  else
    _mm_store_si128(reinterpret_cast<__m128i*>(p), v.raw);
}

// ---- arithmetic ---------------------------------------------------------------------
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> Add(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  Vec128<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm_add_ps(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm_add_pd(a.raw, b.raw);
  else if constexpr (detail::kI8<T>) r.raw = _mm_add_epi8(a.raw, b.raw);
  else if constexpr (detail::kI16<T>) r.raw = _mm_add_epi16(a.raw, b.raw);
  else if constexpr (detail::kI64<T>) r.raw = _mm_add_epi64(a.raw, b.raw);
  else r.raw = _mm_add_epi32(a.raw, b.raw);  // i32/u32 same intrinsic (wraparound)
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> Sub(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  Vec128<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm_sub_ps(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm_sub_pd(a.raw, b.raw);
  else if constexpr (detail::kI8<T>) r.raw = _mm_sub_epi8(a.raw, b.raw);
  else if constexpr (detail::kI16<T>) r.raw = _mm_sub_epi16(a.raw, b.raw);
  else if constexpr (detail::kI64<T>) r.raw = _mm_sub_epi64(a.raw, b.raw);
  else r.raw = _mm_sub_epi32(a.raw, b.raw);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> Mul(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  Vec128<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm_mul_ps(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm_mul_pd(a.raw, b.raw);
  else if constexpr (detail::kI16<T>) r.raw = _mm_mullo_epi16(a.raw, b.raw);
  else if constexpr (detail::kI32<T>) r.raw = _mm_mullo_epi32(a.raw, b.raw);  // SSE4.1; low 32, i32/u32 same
  else if constexpr (detail::kI8<T>) {
    // No _mm_mullo_epi8. Highway's Mul(i8): widen to epi16 in two halves, mullo_epi16,
    // then narrow back keeping the low byte of each 16-bit product (wraparound = the
    // 2's-complement low 8 bits, matching scalar T(x*y)).
    const __m128i lo_mask = _mm_set1_epi16(0x00FF);
    const __m128i a_lo = _mm_and_si128(a.raw, lo_mask);              // even bytes
    const __m128i b_lo = _mm_and_si128(b.raw, lo_mask);
    const __m128i a_hi = _mm_srli_epi16(a.raw, 8);                   // odd bytes
    const __m128i b_hi = _mm_srli_epi16(b.raw, 8);
    const __m128i p_lo = _mm_and_si128(_mm_mullo_epi16(a_lo, b_lo), lo_mask);
    const __m128i p_hi = _mm_slli_epi16(_mm_mullo_epi16(a_hi, b_hi), 8);
    r.raw = _mm_or_si128(p_lo, p_hi);
  } else {
    // i64/u64: no _mm_mullo_epi64 on SSE4 (it's AVX-512-DQ). Highway emulates via
    // 32x32->64 partial products: lo*lo + ((hi_a*lo_b + lo_a*hi_b) << 32). Low 64 bits
    // are exact for both signed and unsigned (2's complement wraparound = scalar).
    const __m128i lo_a = a.raw, lo_b = b.raw;
    const __m128i hi_a = _mm_srli_epi64(a.raw, 32);
    const __m128i hi_b = _mm_srli_epi64(b.raw, 32);
    const __m128i ll = _mm_mul_epu32(lo_a, lo_b);                    // lo_a*lo_b (64b)
    const __m128i cross = _mm_add_epi64(_mm_mul_epu32(hi_a, lo_b),
                                        _mm_mul_epu32(lo_a, hi_b));
    r.raw = _mm_add_epi64(ll, _mm_slli_epi64(cross, 32));
  }
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> Div(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  static_assert(std::is_floating_point_v<T>, "Div is float/double only");
  Vec128<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm_div_ps(a.raw, b.raw);
  else r.raw = _mm_div_pd(a.raw, b.raw);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> Min(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  Vec128<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm_min_ps(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm_min_pd(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int8_t>) r.raw = _mm_min_epi8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint8_t>) r.raw = _mm_min_epu8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) r.raw = _mm_min_epi16(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint16_t>) r.raw = _mm_min_epu16(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) r.raw = _mm_min_epi32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint32_t>) r.raw = _mm_min_epu32(a.raw, b.raw);  // SSE4.1
  else {  // i64/u64: no SSE native min -> select via Gt (Highway emulation)
    r = IfThenElse(Gt(a, b), b, a);  // min = a>b ? b : a
  }
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> Max(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  Vec128<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm_max_ps(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm_max_pd(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int8_t>) r.raw = _mm_max_epi8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint8_t>) r.raw = _mm_max_epu8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) r.raw = _mm_max_epi16(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint16_t>) r.raw = _mm_max_epu16(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) r.raw = _mm_max_epi32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint32_t>) r.raw = _mm_max_epu32(a.raw, b.raw);
  else {  // i64/u64: no SSE native max -> select via Gt (Highway emulation)
    r = IfThenElse(Gt(a, b), a, b);  // max = a>b ? a : b
  }
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> Neg(Vec128<T, N> a) noexcept {
  Vec128<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm_xor_ps(a.raw, _mm_set1_ps(-0.0f));
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm_xor_pd(a.raw, _mm_set1_pd(-0.0));
  else if constexpr (detail::kI8<T>) r.raw = _mm_sub_epi8(_mm_setzero_si128(), a.raw);
  else if constexpr (detail::kI16<T>) r.raw = _mm_sub_epi16(_mm_setzero_si128(), a.raw);
  else if constexpr (detail::kI64<T>) r.raw = _mm_sub_epi64(_mm_setzero_si128(), a.raw);
  else r.raw = _mm_sub_epi32(_mm_setzero_si128(), a.raw);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> Abs(Vec128<T, N> a) noexcept {
  Vec128<T, N> r;
  if constexpr (std::is_same_v<T, float>)
    r.raw = _mm_and_ps(a.raw, _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)));
  else if constexpr (std::is_same_v<T, double>)
    r.raw = _mm_and_pd(a.raw, _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)));
  else if constexpr (std::is_same_v<T, std::int8_t>) r.raw = _mm_abs_epi8(a.raw);   // SSSE3
  else if constexpr (std::is_same_v<T, std::int16_t>) r.raw = _mm_abs_epi16(a.raw);  // SSSE3
  else if constexpr (std::is_same_v<T, std::int32_t>) r.raw = _mm_abs_epi32(a.raw);  // SSSE3
  else if constexpr (std::is_same_v<T, std::int64_t>) {
    // No _mm_abs_epi64 on SSE4 (AVX-512-VL). Highway: mask = a>>63 (arithmetic via
    // cmpgt 0), abs = (a ^ mask) - mask. Emulate arithmetic >>63 with cmpgt(0, a)
    // (_mm_cmpgt_epi64 is SSE4.2).
    const __m128i neg = _mm_cmpgt_epi64(_mm_setzero_si128(), a.raw);  // all-ones where a<0
    r.raw = _mm_sub_epi64(_mm_xor_si128(a.raw, neg), neg);
  } else r.raw = a.raw;  // unsigned: abs is identity
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> Sqrt(Vec128<T, N> a) noexcept {
  static_assert(std::is_floating_point_v<T>, "Sqrt is float/double only");
  Vec128<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm_sqrt_ps(a.raw);
  else r.raw = _mm_sqrt_pd(a.raw);
  return r;
}

// ---- float rounding to integral value (result stays float) --------------------------
// Round=nearest-even, Floor=toward -inf, Ceil=toward +inf, Trunc=toward zero. SSE4.1
// _mm_round_ps/pd with the matching _MM_FROUND flag; _MM_FROUND_NO_EXC suppresses the
// inexact exception (Highway does the same). Float/double only.
#define HYLE_SIMD_SSE4_ROUND(NAME, MODE)                                     \
  template <typename T, std::size_t N>                                       \
  HYLE_FORCE_INLINE Vec128<T, N> NAME(Vec128<T, N> a) noexcept {             \
    static_assert(std::is_floating_point_v<T>, #NAME " is float/double only"); \
    Vec128<T, N> r;                                                          \
    if constexpr (std::is_same_v<T, float>)                                  \
      r.raw = _mm_round_ps(a.raw, (MODE) | _MM_FROUND_NO_EXC);               \
    else r.raw = _mm_round_pd(a.raw, (MODE) | _MM_FROUND_NO_EXC);            \
    return r;                                                                \
  }
HYLE_SIMD_SSE4_ROUND(Round, _MM_FROUND_TO_NEAREST_INT)
HYLE_SIMD_SSE4_ROUND(Floor, _MM_FROUND_TO_NEG_INF)
HYLE_SIMD_SSE4_ROUND(Ceil, _MM_FROUND_TO_POS_INF)
HYLE_SIMD_SSE4_ROUND(Trunc, _MM_FROUND_TO_ZERO)
#undef HYLE_SIMD_SSE4_ROUND


// FMA: SSE4.2 tier has no FMA3 (it's AVX-era), so this is the mul+add fallback (two
// roundings). The AVX2 backend overrides with true single-rounding _mm256_fmadd_*.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> MulAdd(Vec128<T, N> a, Vec128<T, N> b,
                                      Vec128<T, N> c) noexcept {
  return Add(Mul(a, b), c);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> NegMulAdd(Vec128<T, N> a, Vec128<T, N> b,
                                         Vec128<T, N> c) noexcept {
  return Sub(c, Mul(a, b));
}

// ---- bitwise (float stays in FP domain via _ps/_pd to avoid bypass penalty) ---------
#define HYLE_SSE4_BITOP(NAME, PS, PD, SI)                                  \
  template <typename T, std::size_t N>                                     \
  HYLE_FORCE_INLINE Vec128<T, N> NAME(Vec128<T, N> a, Vec128<T, N> b) noexcept { \
    Vec128<T, N> r;                                                        \
    if constexpr (std::is_same_v<T, float>) r.raw = PS(a.raw, b.raw);      \
    else if constexpr (std::is_same_v<T, double>) r.raw = PD(a.raw, b.raw);\
    else r.raw = SI(a.raw, b.raw);                                         \
    return r;                                                              \
  }
HYLE_SSE4_BITOP(And, _mm_and_ps, _mm_and_pd, _mm_and_si128)
HYLE_SSE4_BITOP(Or, _mm_or_ps, _mm_or_pd, _mm_or_si128)
HYLE_SSE4_BITOP(Xor, _mm_xor_ps, _mm_xor_pd, _mm_xor_si128)
// AndNot(a,b) = ~a & b (Highway order): _mm_andnot_*(x,y) computes ~x & y, same shape.
HYLE_SSE4_BITOP(AndNot, _mm_andnot_ps, _mm_andnot_pd, _mm_andnot_si128)
#undef HYLE_SSE4_BITOP

// ---- shifts (integer lanes, compile-time immediate count) ---------------------------
// SSE has slli/srli_epi16/32/64 (logical) and srai_epi16/32 (arithmetic). GAPS emulated
// (mirroring Highway x86_128): epi8 has NO shift (do it as epi16 + mask off the bits that
// cross the byte boundary); srai_epi64 does not exist on SSE (arithmetic via srli + sign
// fixup; cmpgt_epi64 is SSE4.2). ShiftRight is arithmetic for signed T, logical for
// unsigned (matches scalar >>).
template <int kBits, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> ShiftLeft(Vec128<T, N> a) noexcept {
  static_assert(std::is_integral_v<T>, "shifts are integer-lane only");
  Vec128<T, N> r;
  if constexpr (detail::kI16<T>) r.raw = _mm_slli_epi16(a.raw, kBits);
  else if constexpr (detail::kI32<T>) r.raw = _mm_slli_epi32(a.raw, kBits);
  else if constexpr (detail::kI64<T>) r.raw = _mm_slli_epi64(a.raw, kBits);
  else {  // i8/u8: shift as epi16 then clear the low kBits of each high byte that leaked in
    const __m128i shifted = _mm_slli_epi16(a.raw, kBits);
    r.raw = _mm_and_si128(shifted, _mm_set1_epi8(static_cast<char>(0xFFu << kBits)));
  }
  return r;
}
template <int kBits, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> ShiftRight(Vec128<T, N> a) noexcept {
  static_assert(std::is_integral_v<T>, "shifts are integer-lane only");
  Vec128<T, N> r;
  if constexpr (detail::kI16<T>) {
    if constexpr (detail::kSigned<T>) r.raw = _mm_srai_epi16(a.raw, kBits);
    else r.raw = _mm_srli_epi16(a.raw, kBits);
  } else if constexpr (detail::kI32<T>) {
    if constexpr (detail::kSigned<T>) r.raw = _mm_srai_epi32(a.raw, kBits);
    else r.raw = _mm_srli_epi32(a.raw, kBits);
  } else if constexpr (detail::kI64<T>) {
    if constexpr (detail::kSigned<T>) {
      // No _mm_srai_epi64 on SSE. Logical shift, then OR in the sign bits:
      // sign = (a >> 63 arithmetic) replicated; arith_sr = (a>>>k) | (sign << (64-k)).
      const __m128i logical = _mm_srli_epi64(a.raw, kBits);
      const __m128i sign = _mm_cmpgt_epi64(_mm_setzero_si128(), a.raw);  // all-ones if neg (SSE4.2)
      const __m128i sfill = _mm_slli_epi64(sign, 64 - kBits);
      r.raw = (kBits == 0) ? a.raw : _mm_or_si128(logical, sfill);
    } else r.raw = _mm_srli_epi64(a.raw, kBits);
  } else {  // i8/u8
    if constexpr (detail::kSigned<T>) {
      // i8 arithmetic right shift by k (no native epi8 shift on SSE). Highway idiom:
      //   logical = (a >> k) as bytes, cleared of the k high bits that leaked across bytes
      //   sign    = bytes that are negative (high bit set) get the top k bits filled with 1
      // Do the byte logical shift via epi16 shift + mask, then OR in sign-fill where a<0.
      const __m128i logical =
          _mm_and_si128(_mm_srli_epi16(a.raw, kBits),
                        _mm_set1_epi8(static_cast<char>(0xFFu >> kBits)));
      // neg byte mask: a's sign bit broadcast to a full 0xFF byte where the byte is negative.
      const __m128i is_neg = _mm_cmpgt_epi8(_mm_setzero_si128(), a.raw);  // 0xFF if a<0
      const __m128i signfill = _mm_and_si128(
          is_neg, _mm_set1_epi8(static_cast<char>(static_cast<unsigned char>(0xFFu << (8 - kBits)))));
      r.raw = _mm_or_si128(logical, signfill);
    } else {  // u8 logical
      const __m128i shifted = _mm_srli_epi16(a.raw, kBits);
      r.raw = _mm_and_si128(shifted, _mm_set1_epi8(static_cast<char>(0xFFu >> kBits)));
    }
  }
  return r;
}

// ---- per-lane variable shifts (Shl/Shr) ---------------------------------------------
// SSE4.2 has NO variable-shift intrinsic for ANY width (sllv/srlv/srav are AVX2, not SSE4),
// so EVERY width falls back to the scalar-store path: store to a 16-byte-aligned stack array,
// per-lane C++ shift, reload. (AVX2 gets the native sllv/srlv/srav for 32/64-bit.) Signed
// Shr = arithmetic, unsigned = logical (matches scalar semantics).
namespace detail {
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> shift_var_scalar(Vec128<T, N> a, Vec128<T, N> b, bool left) noexcept {
  alignas(16) T va[N], vb[N];
  _mm_store_si128(reinterpret_cast<__m128i*>(va), a.raw);
  _mm_store_si128(reinterpret_cast<__m128i*>(vb), b.raw);
  for (std::size_t i = 0; i < N; ++i)
    va[i] = left ? static_cast<T>(static_cast<std::make_unsigned_t<T>>(va[i]) << vb[i])
                 : static_cast<T>(va[i] >> vb[i]);
  Vec128<T, N> r;
  r.raw = _mm_load_si128(reinterpret_cast<const __m128i*>(va));
  return r;
}
}  // namespace detail
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> Shl(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  static_assert(std::is_integral_v<T>, "Shl is integer-lane only");
  return detail::shift_var_scalar<T, N>(a, b, true);  // no native variable shift on SSE4
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> Shr(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  static_assert(std::is_integral_v<T>, "Shr is integer-lane only");
  return detail::shift_var_scalar<T, N>(a, b, false);  // no native variable shift on SSE4
}

// MulHigh: high 16 bits of the i16/u16 product (native mulhi).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> MulHigh(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  static_assert(std::is_integral_v<T> && sizeof(T) == 2, "MulHigh is 16-bit int only");
  Vec128<T, N> r;
  if constexpr (std::is_signed_v<T>) r.raw = _mm_mulhi_epi16(a.raw, b.raw);
  else r.raw = _mm_mulhi_epu16(a.raw, b.raw);
  return r;
}

// WidenMulPairwiseAdd(d32, a_i16, b_i16): _mm_madd_epi16 multiplies i16 pairs and sums
// adjacent pairs into i32 -- the fixed-point dot core. a/b are FULL i16 (2N lanes), result
// is N i32 lanes.
template <std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec128<std::int32_t, NDst> WidenMulPairwiseAdd(Simd<std::int32_t, NDst, 0>,
                                                                Vec128<SrcT, NSrc> a,
                                                                Vec128<SrcT, NSrc> b) noexcept {
  static_assert(std::is_same_v<SrcT, std::int16_t> && NSrc == 2 * NDst,
                "WidenMulPairwiseAdd: i16 -> i32 pairwise");
  Vec128<std::int32_t, NDst> r;
  r.raw = _mm_madd_epi16(a.raw, b.raw);
  return r;
}

// AbsDiff: |a-b| per lane. Integers via Sub(Max,Min) (Highway's generic); float via Abs(Sub).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> AbsDiff(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  if constexpr (std::is_floating_point_v<T>) return Abs(Sub(a, b));
  else return Sub(Max(a, b), Min(a, b));
}

// ---- saturated arithmetic + rounding average (8/16-bit integer lanes) ---------------
// Native SSE2: adds/subs_epi8/epu8/epi16/epu16 (saturating), avg_epu8/epu16 (rounded).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> SaturatedAdd(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  static_assert(std::is_integral_v<T> && sizeof(T) <= 2, "SaturatedAdd is 8/16-bit int only");
  Vec128<T, N> r;
  if constexpr (std::is_same_v<T, std::int8_t>) r.raw = _mm_adds_epi8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint8_t>) r.raw = _mm_adds_epu8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) r.raw = _mm_adds_epi16(a.raw, b.raw);
  else r.raw = _mm_adds_epu16(a.raw, b.raw);  // u16
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> SaturatedSub(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  static_assert(std::is_integral_v<T> && sizeof(T) <= 2, "SaturatedSub is 8/16-bit int only");
  Vec128<T, N> r;
  if constexpr (std::is_same_v<T, std::int8_t>) r.raw = _mm_subs_epi8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint8_t>) r.raw = _mm_subs_epu8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) r.raw = _mm_subs_epi16(a.raw, b.raw);
  else r.raw = _mm_subs_epu16(a.raw, b.raw);  // u16
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> AverageRound(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  static_assert(std::is_unsigned_v<T> && sizeof(T) <= 2, "AverageRound is u8/u16 only");
  Vec128<T, N> r;
  if constexpr (std::is_same_v<T, std::uint8_t>) r.raw = _mm_avg_epu8(a.raw, b.raw);
  else r.raw = _mm_avg_epu16(a.raw, b.raw);  // u16
  return r;
}

// ---- bit-cast (reinterpret bytes as another lane type; zero-cost register cast) -----
// BitCast(dst_tag, v): reinterpret the 128 bits as Vec128<DstT>. Same total bytes (always
// 16 here), so lane count changes with lane width. Between the three raw register classes
// (__m128 / __m128i / __m128d) x86 has free cast intrinsics (_mm_cast*); same-class (any
// integer width -> any integer width, all __m128i) is a plain move. Underpins gather index
// math, mask<->vector bridging, FP bit tricks.
namespace detail {
// raw -> raw reinterpret between __m128 / __m128i / __m128d (free; no instruction emitted).
template <typename DstRaw, typename SrcRaw>
HYLE_FORCE_INLINE DstRaw bitcast_raw_sse(SrcRaw s) noexcept {
  if constexpr (std::is_same_v<DstRaw, SrcRaw>) return s;
  // Route through __m128i as the common integer form.
  else if constexpr (std::is_same_v<SrcRaw, __m128> && std::is_same_v<DstRaw, __m128i>)
    return _mm_castps_si128(s);
  else if constexpr (std::is_same_v<SrcRaw, __m128d> && std::is_same_v<DstRaw, __m128i>)
    return _mm_castpd_si128(s);
  else if constexpr (std::is_same_v<SrcRaw, __m128i> && std::is_same_v<DstRaw, __m128>)
    return _mm_castsi128_ps(s);
  else if constexpr (std::is_same_v<SrcRaw, __m128i> && std::is_same_v<DstRaw, __m128d>)
    return _mm_castsi128_pd(s);
  else if constexpr (std::is_same_v<SrcRaw, __m128> && std::is_same_v<DstRaw, __m128d>)
    return _mm_castps_pd(s);
  else if constexpr (std::is_same_v<SrcRaw, __m128d> && std::is_same_v<DstRaw, __m128>)
    return _mm_castpd_ps(s);
  else static_assert(sizeof(DstRaw) == 0, "sse4 bitcast_raw: unhandled raw pair");
}
}  // namespace detail
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec128<DstT, NDst> BitCast(Simd<DstT, NDst, 0>, Vec128<SrcT, NSrc> v) noexcept {
  static_assert(sizeof(DstT) * NDst == sizeof(SrcT) * NSrc, "BitCast preserves total bytes");
  Vec128<DstT, NDst> r;
  r.raw = detail::bitcast_raw_sse<detail::Raw128T<DstT>>(v.raw);
  return r;
}

// ---- type conversion (same lane count: f32<->i32) -----------------------------------
// ConvertTo: float<->int truncating (cvtt = toward zero). RoundToInt: f32->i32 round to
// nearest-even (cvtps respects the default MXCSR rounding = RNE). 4 lanes both sides =
// one __m128 <-> one __m128i, so it fits the native-width model (no partial vectors).
// Width-changing PromoteTo/DemoteTo are deferred (need partial vectors) -- see scalar.h.
template <typename DstT, std::size_t N, typename SrcT>
HYLE_FORCE_INLINE Vec128<DstT, N> ConvertTo(Simd<DstT, N, 0>, Vec128<SrcT, N> v) noexcept {
  Vec128<DstT, N> r;
  if constexpr (std::is_same_v<SrcT, float> && std::is_same_v<DstT, std::int32_t>)
    r.raw = _mm_cvttps_epi32(v.raw);           // truncate toward zero
  else if constexpr (std::is_same_v<SrcT, std::int32_t> && std::is_same_v<DstT, float>)
    r.raw = _mm_cvtepi32_ps(v.raw);
  else static_assert(sizeof(DstT) == 0, "sse4 ConvertTo: only f32<->i32 (same width) supported");
  return r;
}
template <typename DstT, std::size_t N, typename SrcT>
HYLE_FORCE_INLINE Vec128<DstT, N> RoundToInt(Simd<DstT, N, 0>, Vec128<SrcT, N> v) noexcept {
  static_assert(std::is_same_v<SrcT, float> && std::is_same_v<DstT, std::int32_t>,
                "sse4 RoundToInt: f32->i32");
  Vec128<DstT, N> r;
  r.raw = _mm_cvtps_epi32(v.raw);              // round to nearest even (default MXCSR)
  return r;
}

// ---- width-changing promote / demote (half-at-a-time, fixed-width friendly) ---------
// PromoteLowerTo/PromoteUpperTo: FULL narrow src (2N lanes in a 128b reg) -> FULL wide dst
// (N lanes in a 128b reg). The SSE4.1 cvtep* intrinsics widen the LOW N narrow lanes of a
// __m128i to a full 128b wide vector. So the LOWER half feeds cvtep* directly; the UPPER
// half is first brought down to the low bytes via _mm_srli_si128(v, 8) (shift right 8 BYTES
// = high 64 bits -> low 64 bits) -- for i8->i16 (upper 8 lanes at bytes 8..15), i16->i32
// (upper 4 at bytes 8..15) and i32->i64 (upper 2 at bytes 8..15) that single shift is
// correct for all integer widths. Signed src sign-extends (cvtepi), unsigned zero-extends
// (cvtepu). f32->f64 uses cvtps_pd (low 2 f32); the upper form moves the high 2 f32 down
// with movehl first.
namespace detail {
template <typename DstT, typename SrcT>
HYLE_FORCE_INLINE Raw128T<DstT> promote128_sse(__m128i half) noexcept {
  if constexpr (std::is_same_v<SrcT, std::int8_t>) return _mm_cvtepi8_epi16(half);
  else if constexpr (std::is_same_v<SrcT, std::uint8_t>) return _mm_cvtepu8_epi16(half);
  else if constexpr (std::is_same_v<SrcT, std::int16_t>) return _mm_cvtepi16_epi32(half);
  else if constexpr (std::is_same_v<SrcT, std::uint16_t>) return _mm_cvtepu16_epi32(half);
  else if constexpr (std::is_same_v<SrcT, std::int32_t>) return _mm_cvtepi32_epi64(half);
  else return _mm_cvtepu32_epi64(half);  // u32 -> u64
}
}  // namespace detail
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec128<DstT, NDst> PromoteLowerTo(Simd<DstT, NDst, 0>, Vec128<SrcT, NSrc> v) noexcept {
  static_assert(NSrc == 2 * NDst && sizeof(DstT) > sizeof(SrcT), "PromoteLowerTo widens, src 2x lanes");
  Vec128<DstT, NDst> r;
  if constexpr (std::is_same_v<SrcT, float>) r.raw = _mm_cvtps_pd(v.raw);  // low 2 f32 -> 2 f64
  else r.raw = detail::promote128_sse<DstT, SrcT>(v.raw);  // cvtep* reads low half-lanes
  return r;
}
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec128<DstT, NDst> PromoteUpperTo(Simd<DstT, NDst, 0>, Vec128<SrcT, NSrc> v) noexcept {
  static_assert(NSrc == 2 * NDst && sizeof(DstT) > sizeof(SrcT), "PromoteUpperTo widens, src 2x lanes");
  Vec128<DstT, NDst> r;
  if constexpr (std::is_same_v<SrcT, float>)
    r.raw = _mm_cvtps_pd(_mm_movehl_ps(v.raw, v.raw));  // high 2 f32 -> low, then -> 2 f64
  else r.raw = detail::promote128_sse<DstT, SrcT>(_mm_srli_si128(v.raw, 8));  // high 64b -> low
  return r;
}
// DemoteTo(dn, a, b): two FULL wide vectors -> one FULL narrow, SATURATING (integer) with
// lanes ORDERED [a..., b...]. Unlike AVX2, a 128b register IS a single block, so packs/packus
// already emit [a-lanes, b-lanes] in order -- NO block-reorder permute is needed. f64->f32
// uses cvtpd_ps on each (2 f32 in the low half) and movelh to concatenate [a.lo2, b.lo2].
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec128<DstT, NDst> DemoteTo(Simd<DstT, NDst, 0>, Vec128<SrcT, NSrc> a,
                                              Vec128<SrcT, NSrc> b) noexcept {
  static_assert(NDst == 2 * NSrc, "DemoteTo: narrow dst has 2x each wide src's lanes");
  Vec128<DstT, NDst> r;
  if constexpr (std::is_same_v<SrcT, double>) {  // f64 -> f32
    r.raw = _mm_movelh_ps(_mm_cvtpd_ps(a.raw), _mm_cvtpd_ps(b.raw));  // [a.lo2, b.lo2]
  } else if constexpr (std::is_same_v<SrcT, std::int16_t> && std::is_same_v<DstT, std::int8_t>)
    r.raw = _mm_packs_epi16(a.raw, b.raw);
  else if constexpr (std::is_same_v<SrcT, std::int16_t> && std::is_same_v<DstT, std::uint8_t>)
    r.raw = _mm_packus_epi16(a.raw, b.raw);
  else if constexpr (std::is_same_v<SrcT, std::int32_t> && std::is_same_v<DstT, std::int16_t>)
    r.raw = _mm_packs_epi32(a.raw, b.raw);
  else if constexpr (std::is_same_v<SrcT, std::int32_t> && std::is_same_v<DstT, std::uint16_t>)
    r.raw = _mm_packus_epi32(a.raw, b.raw);  // SSE4.1
  else static_assert(sizeof(DstT) == 0, "sse4 DemoteTo: unsupported width pair");
  return r;
}
// ReorderDemote2To: on SSE4 a 128b reg is a single block, so there is no reorder permute to
// skip -- the "reordered" form is byte-identical to DemoteTo. Just forward.
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec128<DstT, NDst> ReorderDemote2To(Simd<DstT, NDst, 0> dn, Vec128<SrcT, NSrc> a,
                                                      Vec128<SrcT, NSrc> b) noexcept {
  static_assert(NDst == 2 * NSrc && std::is_integral_v<SrcT>, "ReorderDemote2To: integer 2->1");
  return DemoteTo(dn, a, b);
}

// ---- swizzle (lane rearrangement) ---------------------------------------------------
// SSE4 packs the whole vector in a single 128-bit register (like NEON, unlike AVX2's two
// 128-bit halves). So there is no lane-crossing hazard: InterleaveLower/Upper operate on the
// whole vector, and Concat*/Reverse span all lanes. These are non-hot-path ops, so a simple
// store+reload through an alignas(16) scratch buffer is correct and clear -- the scalar
// backend is the oracle and this per-lane logic matches it (as validated on NEON).
namespace detail {
template <typename T, std::size_t N, typename IdxFn>
HYLE_FORCE_INLINE Vec128<T, N> swizzle_via_mem(Vec128<T, N> v, IdxFn idx) noexcept {
  alignas(16) T src[N], dst[N];
  Store(v, Simd<T, N, 0>(), src);
  for (std::size_t i = 0; i < N; ++i) dst[i] = src[idx(i)];
  return Load(Simd<T, N, 0>(), dst);
}
}  // namespace detail

// OddEven(a,b): even lanes <- b, odd <- a.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> OddEven(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  alignas(16) T sa[N], sb[N], dst[N];
  Store(a, Simd<T, N, 0>(), sa); Store(b, Simd<T, N, 0>(), sb);
  for (std::size_t i = 0; i < N; ++i) dst[i] = (i & 1) ? sa[i] : sb[i];
  return Load(Simd<T, N, 0>(), dst);
}
template <class D>
HYLE_FORCE_INLINE detail::V128<D> Reverse(D, detail::V128<D> v) noexcept {
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  return detail::swizzle_via_mem(v, [](std::size_t i) { return N - 1 - i; });
}
template <class D>
HYLE_FORCE_INLINE detail::V128<D> Reverse2(D, detail::V128<D> v) noexcept {
  return detail::swizzle_via_mem(v, [](std::size_t i) { return i ^ 1; });
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> InterleaveLower(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  alignas(16) T sa[N], sb[N], dst[N];
  Store(a, Simd<T, N, 0>(), sa); Store(b, Simd<T, N, 0>(), sb);
  for (std::size_t i = 0; i < N / 2; ++i) { dst[2 * i] = sa[i]; dst[2 * i + 1] = sb[i]; }
  return Load(Simd<T, N, 0>(), dst);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> InterleaveUpper(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  alignas(16) T sa[N], sb[N], dst[N];
  Store(a, Simd<T, N, 0>(), sa); Store(b, Simd<T, N, 0>(), sb);
  for (std::size_t i = 0; i < N / 2; ++i) { dst[2 * i] = sa[N / 2 + i]; dst[2 * i + 1] = sb[N / 2 + i]; }
  return Load(Simd<T, N, 0>(), dst);
}
namespace detail {
template <int loHalf, int hiHalf, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> concat_halves(Vec128<T, N> hi, Vec128<T, N> lo) noexcept {
  alignas(16) T sh[N], sl[N], dst[N];
  Store(hi, Simd<T, N, 0>(), sh); Store(lo, Simd<T, N, 0>(), sl);
  for (std::size_t i = 0; i < N / 2; ++i) {
    dst[i] = sl[loHalf * (N / 2) + i];
    dst[N / 2 + i] = sh[hiHalf * (N / 2) + i];
  }
  return Load(Simd<T, N, 0>(), dst);
}
}  // namespace detail
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> ConcatLowerLower(Vec128<T, N> hi, Vec128<T, N> lo) noexcept { return detail::concat_halves<0, 0>(hi, lo); }
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> ConcatUpperUpper(Vec128<T, N> hi, Vec128<T, N> lo) noexcept { return detail::concat_halves<1, 1>(hi, lo); }
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> ConcatLowerUpper(Vec128<T, N> hi, Vec128<T, N> lo) noexcept { return detail::concat_halves<1, 0>(hi, lo); }
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> ConcatUpperLower(Vec128<T, N> hi, Vec128<T, N> lo) noexcept { return detail::concat_halves<0, 1>(hi, lo); }
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> ConcatEven(Vec128<T, N> hi, Vec128<T, N> lo) noexcept {
  alignas(16) T sh[N], sl[N], dst[N];
  Store(hi, Simd<T, N, 0>(), sh); Store(lo, Simd<T, N, 0>(), sl);
  for (std::size_t i = 0; i < N / 2; ++i) { dst[i] = sl[2 * i]; dst[N / 2 + i] = sh[2 * i]; }
  return Load(Simd<T, N, 0>(), dst);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> ConcatOdd(Vec128<T, N> hi, Vec128<T, N> lo) noexcept {
  alignas(16) T sh[N], sl[N], dst[N];
  Store(hi, Simd<T, N, 0>(), sh); Store(lo, Simd<T, N, 0>(), sl);
  for (std::size_t i = 0; i < N / 2; ++i) { dst[i] = sl[2 * i + 1]; dst[N / 2 + i] = sh[2 * i + 1]; }
  return Load(Simd<T, N, 0>(), dst);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> DupEven(Vec128<T, N> v) noexcept {
  return detail::swizzle_via_mem(v, [](std::size_t i) { return i & ~std::size_t{1}; });
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> DupOdd(Vec128<T, N> v) noexcept {
  return detail::swizzle_via_mem(v, [](std::size_t i) { return (i & ~std::size_t{1}) | 1; });
}
template <typename T, std::size_t N>
struct Indices128 { Vec128<T, N> idx; };
template <class D>
HYLE_FORCE_INLINE Indices128<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> IndicesFromVec(D, detail::V128<D> iv) noexcept {
  return Indices128<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)>{iv};
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> TableLookupLanes(Vec128<T, N> v, Indices128<T, N> ind) noexcept {
  alignas(16) T src[N], idx[N], dst[N];
  Store(v, Simd<T, N, 0>(), src); Store(ind.idx, Simd<T, N, 0>(), idx);
  for (std::size_t i = 0; i < N; ++i) dst[i] = src[static_cast<std::size_t>(idx[i]) & (N - 1)];
  return Load(Simd<T, N, 0>(), dst);
}

// ---- comparison -> mask -------------------------------------------------------------
namespace detail {
// Signed cmpgt per width (SSE4.2 has epi8/16/32/64 native; epi64 is the SSE4.2 add).
template <typename T>
HYLE_FORCE_INLINE __m128i icmpgt_signed(__m128i a, __m128i b) noexcept {
  if constexpr (kI8<T>) return _mm_cmpgt_epi8(a, b);
  else if constexpr (kI16<T>) return _mm_cmpgt_epi16(a, b);
  else if constexpr (kI64<T>) return _mm_cmpgt_epi64(a, b);  // SSE4.2
  else return _mm_cmpgt_epi32(a, b);
}
// Unsigned cmpgt: flip the sign bit of each lane, then signed cmpgt (Highway's trick,
// generalized from the existing u32 path to all widths).
template <typename T>
HYLE_FORCE_INLINE __m128i icmpgt_unsigned(__m128i a, __m128i b) noexcept {
  __m128i msb;
  if constexpr (kI8<T>) msb = _mm_set1_epi8(static_cast<char>(0x80));
  else if constexpr (kI16<T>) msb = _mm_set1_epi16(static_cast<short>(0x8000));
  else if constexpr (kI64<T>) msb = _mm_set1_epi64x(static_cast<long long>(0x8000000000000000ULL));
  else msb = _mm_set1_epi32(static_cast<int>(0x80000000u));
  return icmpgt_signed<std::make_signed_t<T>>(_mm_xor_si128(a, msb), _mm_xor_si128(b, msb));
}
// cmpeq per width (epi64 is SSE4.1).
template <typename T>
HYLE_FORCE_INLINE __m128i icmpeq(__m128i a, __m128i b) noexcept {
  if constexpr (kI8<T>) return _mm_cmpeq_epi8(a, b);
  else if constexpr (kI16<T>) return _mm_cmpeq_epi16(a, b);
  else if constexpr (kI64<T>) return _mm_cmpeq_epi64(a, b);  // SSE4.1
  else return _mm_cmpeq_epi32(a, b);
}
}  // namespace detail

template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask128<T, N> Eq(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  Mask128<T, N> m;
  if constexpr (std::is_same_v<T, float>) m.raw = _mm_cmpeq_ps(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) m.raw = _mm_cmpeq_pd(a.raw, b.raw);
  else m.raw = detail::icmpeq<T>(a.raw, b.raw);
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask128<T, N> Ne(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  Mask128<T, N> m;
  if constexpr (std::is_same_v<T, float>) m.raw = _mm_cmpneq_ps(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) m.raw = _mm_cmpneq_pd(a.raw, b.raw);
  else m.raw = _mm_xor_si128(detail::icmpeq<T>(a.raw, b.raw), _mm_set1_epi32(-1));  // ~Eq
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask128<T, N> Gt(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  Mask128<T, N> m;
  if constexpr (std::is_same_v<T, float>) m.raw = _mm_cmpgt_ps(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) m.raw = _mm_cmpgt_pd(a.raw, b.raw);
  else if constexpr (detail::kSigned<T>) m.raw = detail::icmpgt_signed<T>(a.raw, b.raw);
  else m.raw = detail::icmpgt_unsigned<T>(a.raw, b.raw);  // xor-MSB trick
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask128<T, N> Lt(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  return Gt(b, a);  // a<b == b>a
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask128<T, N> Ge(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  Mask128<T, N> m;
  if constexpr (std::is_same_v<T, float>) m.raw = _mm_cmpge_ps(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) m.raw = _mm_cmpge_pd(a.raw, b.raw);
  else {  // ~(a<b) == ~Gt(b,a)
    const __m128i lt = Lt(a, b).raw;
    m.raw = _mm_xor_si128(lt, _mm_set1_epi32(-1));
  }
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask128<T, N> Le(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  return Ge(b, a);  // a<=b == b>=a
}

// ---- mask ops -----------------------------------------------------------------------
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> IfThenElse(Mask128<T, N> m, Vec128<T, N> yes,
                                          Vec128<T, N> no) noexcept {
  Vec128<T, N> r;  // blendv selects 'yes' where the mask lane's top bit is set
  if constexpr (std::is_same_v<T, float>) r.raw = _mm_blendv_ps(no.raw, yes.raw, m.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm_blendv_pd(no.raw, yes.raw, m.raw);
  else r.raw = _mm_blendv_epi8(no.raw, yes.raw, m.raw);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> IfThenElseZero(Mask128<T, N> m, Vec128<T, N> yes) noexcept {
  return IfThenElse(m, yes, Zero(Simd<T, N, 0>()));
}

namespace detail {
// One bit per LANE, LSB = lane 0. Float/double use the native movemask (1 bit/lane).
// Integers use movemask_epi8 (1 bit/BYTE = 16 bits) then compress to 1 bit/lane by taking
// every sizeof(T)-th byte's bit -- a lane's bytes are all-ones or all-zero (our compares
// produce full-lane masks), so the lane's low byte bit represents the whole lane.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE unsigned mask_bits(Mask128<T, N> m) noexcept {
  if constexpr (std::is_same_v<T, double>)
    return static_cast<unsigned>(_mm_movemask_pd(m.raw));
  else if constexpr (std::is_same_v<T, float>)
    return static_cast<unsigned>(_mm_movemask_ps(m.raw));
  else {
    const unsigned bytes = static_cast<unsigned>(_mm_movemask_epi8(m.raw));  // 16 bits
    if constexpr (sizeof(T) == 1) {
      return bytes;  // 16 lanes, 1 bit/byte already
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
HYLE_FORCE_INLINE bool AllTrue(Simd<T, N, 0>, Mask128<T, N> m) noexcept {
  // N max is 16 (i8 = 16 lanes at 128-bit), so (1u<<N)-1 is safe; compute the full-N
  // mask the same safe way as the AVX2 backend anyway (1<<16 is fine in unsigned).
  const unsigned full = (N >= 32) ? 0xFFFFFFFFu : ((1u << N) - 1u);
  return detail::mask_bits(m) == full;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE bool AllFalse(Simd<T, N, 0>, Mask128<T, N> m) noexcept {
  return detail::mask_bits(m) == 0u;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE std::size_t CountTrue(Simd<T, N, 0>, Mask128<T, N> m) noexcept {
  return static_cast<std::size_t>(std::popcount(detail::mask_bits(m)));
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE std::ptrdiff_t FindFirstTrue(Simd<T, N, 0>, Mask128<T, N> m) noexcept {
  const unsigned b = detail::mask_bits(m);
  return b ? static_cast<std::ptrdiff_t>(std::countr_zero(b)) : -1;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE std::ptrdiff_t FindLastTrue(Simd<T, N, 0>, Mask128<T, N> m) noexcept {
  const unsigned b = detail::mask_bits(m);
  return b ? static_cast<std::ptrdiff_t>(31 - std::countl_zero(b)) : -1;
}
// VecFromMask: the mask's per-lane 0xFF../0 pattern as a vector (compares yield full-lane).
template <class D>
HYLE_FORCE_INLINE detail::V128<D> VecFromMask(D, Mask128<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m) noexcept {
  using T = TFromD<D>;
  detail::V128<D> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm_castsi128_ps(m.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm_castsi128_pd(m.raw);
  else r.raw = m.raw;
  return r;
}
// SumsOf8(v): 16 u8 lanes -> 2 u64 (native _mm_sad_epu8 vs zero). The count core.
template <std::size_t N>
HYLE_FORCE_INLINE Vec128<std::uint64_t, N / 8> SumsOf8(Vec128<std::uint8_t, N> v) noexcept {
  static_assert(N == 16, "sse4 SumsOf8: full 16xu8 -> 2xu64");
  Vec128<std::uint64_t, N / 8> r;
  r.raw = _mm_sad_epu8(v.raw, _mm_setzero_si128());
  return r;
}

// ---- reductions (shuffle-and-combine cascade, Highway generic_ops pattern) ----------
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T GetLane(Vec128<T, N> v) noexcept {
  if constexpr (std::is_same_v<T, float>) return _mm_cvtss_f32(v.raw);
  else if constexpr (std::is_same_v<T, double>) return _mm_cvtsd_f64(v.raw);
  else if constexpr (sizeof(T) == 8) return static_cast<T>(_mm_cvtsi128_si64(v.raw));
  else return static_cast<T>(_mm_cvtsi128_si32(v.raw));  // low 32 bits; T-cast truncates for i8/i16
}

namespace detail {
// Combine all lanes of a 128-bit register with op f (Add/Min/Max), result broadcast to
// lane 0. 4-lane (f32/i32/u32): swap pairs then halves. 2-lane (f64): swap the pair.
template <typename T, std::size_t N, class F>
HYLE_FORCE_INLINE T reduce128(Vec128<T, N> v, F f) noexcept {
  if constexpr (std::is_same_v<T, double>) {  // 2 lanes
    Vec128<T, N> hi;
    hi.raw = _mm_unpackhi_pd(v.raw, v.raw);
    return GetLane(f(v, hi));
  } else if constexpr (std::is_same_v<T, float>) {  // 4 lanes
    Vec128<T, N> t1;
    t1.raw = _mm_shuffle_ps(v.raw, v.raw, _MM_SHUFFLE(2, 3, 0, 1));
    Vec128<T, N> s1 = f(v, t1);
    Vec128<T, N> t2;
    t2.raw = _mm_shuffle_ps(s1.raw, s1.raw, _MM_SHUFFLE(1, 0, 3, 2));
    return GetLane(f(s1, t2));
  } else {  // 4 lanes i32/u32
    Vec128<T, N> t1;
    t1.raw = _mm_shuffle_epi32(v.raw, _MM_SHUFFLE(2, 3, 0, 1));
    Vec128<T, N> s1 = f(v, t1);
    Vec128<T, N> t2;
    t2.raw = _mm_shuffle_epi32(s1.raw, _MM_SHUFFLE(1, 0, 3, 2));
    return GetLane(f(s1, t2));
  }
}

// Reductions that lack a clean cascade for narrow/64-bit int lanes (i8/u8/i16/u16/i64/
// u64): store to a stack array and scalar-reduce. Correct + simple; the cascade shuffles
// above are only defined for f32/f64/i32/u32. (Highway likewise falls back for the
// awkward widths -- a horizontal reduce isn't the hot path.)
enum class RedOp { kAdd, kMin, kMax };
template <typename T, std::size_t N, RedOp op>
HYLE_FORCE_INLINE T reduce_scalar(Vec128<T, N> v) noexcept {
  alignas(16) T tmp[N];
  _mm_store_si128(reinterpret_cast<__m128i*>(tmp), v.raw);
  T acc = tmp[0];
  for (std::size_t i = 1; i < N; ++i) {
    if constexpr (op == RedOp::kAdd) acc = static_cast<T>(acc + tmp[i]);
    else if constexpr (op == RedOp::kMin) acc = tmp[i] < acc ? tmp[i] : acc;
    else acc = tmp[i] > acc ? tmp[i] : acc;
  }
  return acc;
}
}  // namespace detail

template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceSum(Simd<T, N, 0>, Vec128<T, N> v) noexcept {
  if constexpr (detail::kI32<T> || std::is_floating_point_v<T>)
    return detail::reduce128(v, [](Vec128<T, N> x, Vec128<T, N> y) { return Add(x, y); });
  else return detail::reduce_scalar<T, N, detail::RedOp::kAdd>(v);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceMin(Simd<T, N, 0>, Vec128<T, N> v) noexcept {
  if constexpr (detail::kI32<T> || std::is_floating_point_v<T>)
    return detail::reduce128(v, [](Vec128<T, N> x, Vec128<T, N> y) { return Min(x, y); });
  else return detail::reduce_scalar<T, N, detail::RedOp::kMin>(v);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceMax(Simd<T, N, 0>, Vec128<T, N> v) noexcept {
  if constexpr (detail::kI32<T> || std::is_floating_point_v<T>)
    return detail::reduce128(v, [](Vec128<T, N> x, Vec128<T, N> y) { return Max(x, y); });
  else return detail::reduce_scalar<T, N, detail::RedOp::kMax>(v);
}

// Reduction family completion: ReduceMul (product), ReduceAnd/Or (bitwise, integer lanes).
// Fold via a stack array (no native horizontal product/and/or; horizontal reduce is cold).
namespace detail {
enum class RedOp2 { kMul, kAnd, kOr };
template <typename T, std::size_t N, RedOp2 op>
HYLE_FORCE_INLINE T reduce_scalar2(Vec128<T, N> v) noexcept {
  alignas(16) T tmp[N];
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
HYLE_FORCE_INLINE T ReduceMul(Simd<T, N, 0>, Vec128<T, N> v) noexcept {
  return detail::reduce_scalar2<T, N, detail::RedOp2::kMul>(v);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceAnd(Simd<T, N, 0>, Vec128<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "ReduceAnd is integer-lane only");
  return detail::reduce_scalar2<T, N, detail::RedOp2::kAnd>(v);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceOr(Simd<T, N, 0>, Vec128<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "ReduceOr is integer-lane only");
  return detail::reduce_scalar2<T, N, detail::RedOp2::kOr>(v);
}

// Broadcast<i>: splat lane i across all lanes (extract-and-Set, uniform over lane width).
// Index-vector swizzles (TableLookup/Gather/Scatter/Compress/Interleave) DEFERRED -- see
// scalar.h note.
template <int kLane, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> Broadcast(Vec128<T, N> v) noexcept {
  static_assert(kLane >= 0 && static_cast<std::size_t>(kLane) < N, "Broadcast lane out of range");
  alignas(16) T tmp[N];
  Store(v, Simd<T, N, 0>(), tmp);
  return Set(Simd<T, N, 0>(), tmp[kLane]);
}

// ---- masked / partial load-store (tail handling) ------------------------------------
// 128-bit mirror of the avx2 tail toolkit (same rationale): FirstN builds an all-ones
// lane pattern via stack array + Load (a valid blendv/movemask selector for every lane
// type); LoadN/StoreN touch only the first k lanes (fault-safe); MaskedLoad = IfThenElse
// Zero + full LoadU (Highway's non-AVX3 narrow path); BlendedStore writes only on-lanes.
template <class D>
HYLE_FORCE_INLINE Mask128<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> FirstN(D d, std::size_t n) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  T ones;
  auto* ob = reinterpret_cast<unsigned char*>(&ones);
  for (std::size_t k = 0; k < sizeof(T); ++k) ob[k] = 0xFFu;
  alignas(16) T tmp[kN];
  for (std::size_t i = 0; i < kN; ++i) tmp[i] = (i < n) ? ones : T{0};
  Mask128<T, kN> m;
  m.raw = Load(d, tmp).raw;
  return m;
}
template <class D>
HYLE_FORCE_INLINE detail::V128<D> MaskedLoad(Mask128<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m,
                                             D d, const TFromD<D>* p) noexcept {
  return IfThenElseZero(m, LoadU(d, p));
}
template <class D>
HYLE_FORCE_INLINE detail::V128<D> MaskedLoadOr(detail::V128<D> no,
                                               Mask128<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m,
                                               D d, const TFromD<D>* p) noexcept {
  return IfThenElse(m, LoadU(d, p), no);
}
template <class D>
HYLE_FORCE_INLINE void BlendedStore(detail::V128<D> v,
                                    Mask128<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m, D d,
                                    TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  alignas(16) T tmp[kN];
  Store(v, d, tmp);
  const unsigned bits = detail::mask_bits(m);
  for (std::size_t i = 0; i < kN; ++i)
    if ((bits >> i) & 1u) p[i] = tmp[i];
}
template <class D>
HYLE_FORCE_INLINE detail::V128<D> LoadN(D d, const TFromD<D>* p, std::size_t n) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  const std::size_t k = n < kN ? n : kN;
  alignas(16) T tmp[kN] = {};
  for (std::size_t i = 0; i < k; ++i) tmp[i] = p[i];
  return Load(d, tmp);
}
template <class D>
HYLE_FORCE_INLINE void StoreN(detail::V128<D> v, D d, TFromD<D>* p, std::size_t n) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  const std::size_t k = n < kN ? n : kN;
  alignas(16) T tmp[kN];
  Store(v, d, tmp);
  for (std::size_t i = 0; i < k; ++i) p[i] = tmp[i];
}

// ---- gather / scatter + compress / expand -------------------------------------------
// Store+reload (capability-complete; SSE4 has no gather/scatter/compress hardware, so this
// IS Highway's SSE4 approach). The scalar backend is the reference.
template <class D, typename IdxT>
HYLE_FORCE_INLINE detail::V128<D> GatherIndex(D d, const TFromD<D>* base,
                                              Vec128<IdxT, HYLE_SIMD_MAX_LANES_D(D)> idx) noexcept {
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  alignas(16) IdxT ix[kN]; alignas(16) TFromD<D> dst[kN];
  Store(idx, Simd<IdxT, kN, 0>(), ix);
  for (std::size_t i = 0; i < kN; ++i) dst[i] = base[static_cast<std::ptrdiff_t>(ix[i])];
  return Load(d, dst);
}
template <class D, typename IdxT>
HYLE_FORCE_INLINE void ScatterIndex(detail::V128<D> v, D d, TFromD<D>* base,
                                    Vec128<IdxT, HYLE_SIMD_MAX_LANES_D(D)> idx) noexcept {
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  alignas(16) IdxT ix[kN]; alignas(16) TFromD<D> src[kN];
  Store(idx, Simd<IdxT, kN, 0>(), ix); Store(v, d, src);
  for (std::size_t i = 0; i < kN; ++i) base[static_cast<std::ptrdiff_t>(ix[i])] = src[i];
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> Compress(Vec128<T, N> v, Mask128<T, N> m) noexcept {
  alignas(16) T src[N], dst[N] = {};
  Store(v, Simd<T, N, 0>(), src);
  const unsigned bits = detail::mask_bits(m);
  std::size_t k = 0;
  for (std::size_t i = 0; i < N; ++i) if ((bits >> i) & 1u) dst[k++] = src[i];
  return Load(Simd<T, N, 0>(), dst);
}
template <class D>
HYLE_FORCE_INLINE std::size_t CompressStore(detail::V128<D> v,
                                            Mask128<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m,
                                            D d, TFromD<D>* p) noexcept {
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  alignas(16) TFromD<D> src[kN];
  Store(v, d, src);
  const unsigned bits = detail::mask_bits(m);
  std::size_t k = 0;
  for (std::size_t i = 0; i < kN; ++i) if ((bits >> i) & 1u) p[k++] = src[i];
  return k;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> Expand(Vec128<T, N> v, Mask128<T, N> m) noexcept {
  alignas(16) T src[N], dst[N];
  Store(v, Simd<T, N, 0>(), src);
  const unsigned bits = detail::mask_bits(m);
  std::size_t k = 0;
  for (std::size_t i = 0; i < N; ++i) dst[i] = ((bits >> i) & 1u) ? src[k++] : T{0};
  return Load(Simd<T, N, 0>(), dst);
}

// ---- bit counts / sign helpers / misc -----------------------------------------------
// PopulationCount/LeadingZeroCount/TrailingZeroCount via store+reload + std:: (native
// vpopcnt is AVX-512; PSHUFB nibble-LUT is a profiling-gated perf opt). Not/Clamp/Rotate/
// CopySign/... compose from existing ops -- same on every backend. Scalar is the reference.
namespace detail {
template <typename T, std::size_t N, int kind>
HYLE_FORCE_INLINE Vec128<T, N> bitcount_via_mem(Vec128<T, N> v) noexcept {
  using U = std::make_unsigned_t<T>;
  alignas(16) T s[N], d[N];
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
HYLE_FORCE_INLINE Vec128<T, N> PopulationCount(Vec128<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "PopulationCount is integer-lane only");
  return detail::bitcount_via_mem<T, N, 0>(v);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> LeadingZeroCount(Vec128<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "LeadingZeroCount is integer-lane only");
  return detail::bitcount_via_mem<T, N, 1>(v);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> TrailingZeroCount(Vec128<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "TrailingZeroCount is integer-lane only");
  return detail::bitcount_via_mem<T, N, 2>(v);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> Not(Vec128<T, N> v) noexcept {
  Vec128<T, N> ones;
  if constexpr (std::is_same_v<T, float>) ones.raw = _mm_castsi128_ps(_mm_set1_epi32(-1));
  else if constexpr (std::is_same_v<T, double>) ones.raw = _mm_castsi128_pd(_mm_set1_epi32(-1));
  else ones.raw = _mm_set1_epi32(-1);
  return Xor(v, ones);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> Clamp(Vec128<T, N> v, Vec128<T, N> lo, Vec128<T, N> hi) noexcept {
  return Min(Max(v, lo), hi);
}
template <int kBits, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> RotateRight(Vec128<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "RotateRight is integer-lane only");
  constexpr int kSizeBits = static_cast<int>(sizeof(T) * 8);
  static_assert(kBits >= 0 && kBits < kSizeBits, "rotate count in [0,bits)");
  if constexpr (kBits == 0) return v;
  else if constexpr (std::is_unsigned_v<T>)
    return Or(ShiftRight<kBits>(v), ShiftLeft<kSizeBits - kBits>(v));
  else {  // signed ShiftRight is arithmetic -> rotate on the unsigned pattern (store+reload)
    using U = std::make_unsigned_t<T>;
    alignas(16) T s[N], d[N];
    Store(v, Simd<T, N, 0>(), s);
    for (std::size_t i = 0; i < N; ++i) { const U u = static_cast<U>(s[i]); d[i] = static_cast<T>((u >> kBits) | (u << (kSizeBits - kBits))); }
    return Load(Simd<T, N, 0>(), d);
  }
}
template <int kBits, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> RotateLeft(Vec128<T, N> v) noexcept {
  constexpr int kSizeBits = static_cast<int>(sizeof(T) * 8);
  return RotateRight<(kSizeBits - kBits) % kSizeBits>(v);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> CopySign(Vec128<T, N> mag, Vec128<T, N> sign) noexcept {
  static_assert(std::is_floating_point_v<T>, "CopySign is float-lane only");
  const auto sign_bit = Set(Simd<T, N, 0>(), static_cast<T>(-0.0));
  return Or(AndNot(sign_bit, mag), And(sign_bit, sign));  // AndNot(a,b)=~a&b
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> BroadcastSignBit(Vec128<T, N> v) noexcept {
  static_assert(std::is_integral_v<T> && std::is_signed_v<T>, "BroadcastSignBit is signed-int only");
  return ShiftRight<static_cast<int>(sizeof(T) * 8) - 1>(v);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> IfNegativeThenElse(Vec128<T, N> v, Vec128<T, N> yes, Vec128<T, N> no) noexcept {
  return IfThenElse(Lt(v, Zero(Simd<T, N, 0>())), yes, no);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> ZeroIfNegative(Vec128<T, N> v) noexcept {
  return IfThenElse(Lt(v, Zero(Simd<T, N, 0>())), Zero(Simd<T, N, 0>()), v);
}

}  // namespace hyle::simd
