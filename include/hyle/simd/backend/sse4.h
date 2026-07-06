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

}  // namespace hyle::simd
