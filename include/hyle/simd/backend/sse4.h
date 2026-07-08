#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>  // std::memcpy (LoadMaskBits lane pun)
#include <bit>
#include <cmath>   // std::logb / std::ldexp (GetExponent / MulByPow2 store+reload)
#include <limits>  // std::numeric_limits<T>::infinity() (FP classification splat)
#include <type_traits>

#include <nmmintrin.h>  // SSE4.2 (transitively pulls SSE4.1/SSSE3/SSE3/SSE2)

#include "hyle/macros.h"
#include "hyle/simd/backend/crypto_x86.h"  // shared 128-bit AES-NI/PCLMUL workers (Crypto op face)

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

// ---- interleaved load / store (AoS <-> SoA) -----------------------------------------
// correctness-first strided copy; native deinterleave shuffles are a profiling upgrade.
template <class D>
HYLE_FORCE_INLINE void LoadInterleaved2(D d, const TFromD<D>* p, VFromD<D>& v0,
                                        VFromD<D>& v1) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(16) T t0[N], t1[N];
  for (std::size_t i = 0; i < N; ++i) { t0[i] = p[2 * i]; t1[i] = p[2 * i + 1]; }
  v0 = Load(d, t0); v1 = Load(d, t1);
}
template <class D>
HYLE_FORCE_INLINE void LoadInterleaved3(D d, const TFromD<D>* p, VFromD<D>& v0, VFromD<D>& v1,
                                        VFromD<D>& v2) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(16) T t0[N], t1[N], t2[N];
  for (std::size_t i = 0; i < N; ++i) { t0[i] = p[3*i]; t1[i] = p[3*i+1]; t2[i] = p[3*i+2]; }
  v0 = Load(d, t0); v1 = Load(d, t1); v2 = Load(d, t2);
}
template <class D>
HYLE_FORCE_INLINE void LoadInterleaved4(D d, const TFromD<D>* p, VFromD<D>& v0, VFromD<D>& v1,
                                        VFromD<D>& v2, VFromD<D>& v3) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(16) T t0[N], t1[N], t2[N], t3[N];
  for (std::size_t i = 0; i < N; ++i) { t0[i]=p[4*i]; t1[i]=p[4*i+1]; t2[i]=p[4*i+2]; t3[i]=p[4*i+3]; }
  v0 = Load(d, t0); v1 = Load(d, t1); v2 = Load(d, t2); v3 = Load(d, t3);
}
template <class D>
HYLE_FORCE_INLINE void StoreInterleaved2(VFromD<D> v0, VFromD<D> v1, D d, TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(16) T t0[N], t1[N];
  Store(v0, d, t0); Store(v1, d, t1);
  for (std::size_t i = 0; i < N; ++i) { p[2*i] = t0[i]; p[2*i+1] = t1[i]; }
}
template <class D>
HYLE_FORCE_INLINE void StoreInterleaved3(VFromD<D> v0, VFromD<D> v1, VFromD<D> v2, D d,
                                         TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(16) T t0[N], t1[N], t2[N];
  Store(v0, d, t0); Store(v1, d, t1); Store(v2, d, t2);
  for (std::size_t i = 0; i < N; ++i) { p[3*i]=t0[i]; p[3*i+1]=t1[i]; p[3*i+2]=t2[i]; }
}
template <class D>
HYLE_FORCE_INLINE void StoreInterleaved4(VFromD<D> v0, VFromD<D> v1, VFromD<D> v2, VFromD<D> v3, D d,
                                         TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(16) T t0[N], t1[N], t2[N], t3[N];
  Store(v0, d, t0); Store(v1, d, t1); Store(v2, d, t2); Store(v3, d, t3);
  for (std::size_t i = 0; i < N; ++i) { p[4*i]=t0[i]; p[4*i+1]=t1[i]; p[4*i+2]=t2[i]; p[4*i+3]=t3[i]; }
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

// ---- math / arith extensions (Highway) ----------------------------------------------
// ApproximateReciprocal(a) ~= 1/a: f32 native _mm_rcp_ps (12-bit); f64 has NO native rcp,
// fall back to the exact 1/a = Div(Set(1),a) (reusing Div/Set).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> ApproximateReciprocal(Vec128<T, N> a) noexcept {
  static_assert(std::is_floating_point_v<T>, "float only");
  Vec128<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm_rcp_ps(a.raw);
  else r = Div(Set(Simd<T, N, 0>(), 1.0), a);  // no f64 rcp intrinsic: exact 1/a
  return r;
}
// ApproximateReciprocalSqrt(a) ~= 1/sqrt(a): f32 native _mm_rsqrt_ps (12-bit); f64 has no
// native rsqrt, fall back to Div(Set(1), Sqrt(a)) (reusing Sqrt/Div/Set).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> ApproximateReciprocalSqrt(Vec128<T, N> a) noexcept {
  static_assert(std::is_floating_point_v<T>, "float only");
  Vec128<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm_rsqrt_ps(a.raw);
  else r = Div(Set(Simd<T, N, 0>(), 1.0), Sqrt(a));  // no f64 rsqrt intrinsic
  return r;
}
// GetExponent(v) = floor(log2(|v|)) as float (std::logb). store+reload std::logb; native
// getexp is AVX512 (vgetexpps) -- SSE4/AVX2 lack it, so correctness-first per-lane libm.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> GetExponent(Vec128<T, N> v) noexcept {
  static_assert(std::is_floating_point_v<T>, "float only");
  alignas(16) T s[N];
  Store(v, Simd<T, N, 0>(), s);
  for (std::size_t i = 0; i < N; ++i) s[i] = std::logb(s[i]);
  return Load(Simd<T, N, 0>(), s);
}
// MulByPow2(a, b) = a * 2^b (std::ldexp), b an integer-lane vector of matching width.
// store+reload std::ldexp; no clean SSE4 native (AVX512 vscalefps is the hardware form).
template <typename T, std::size_t N, typename IT>
HYLE_FORCE_INLINE Vec128<T, N> MulByPow2(Vec128<T, N> a, Vec128<IT, N> b) noexcept {
  static_assert(std::is_floating_point_v<T>, "float only");
  alignas(16) T sa[N];
  alignas(16) IT sb[N];
  Store(a, Simd<T, N, 0>(), sa);
  Store(b, Simd<IT, N, 0>(), sb);
  for (std::size_t i = 0; i < N; ++i) sa[i] = std::ldexp(sa[i], static_cast<int>(sb[i]));
  return Load(Simd<T, N, 0>(), sa);
}
// NaN-aware min/max (Highway MinNumber/MaxNumber): if one operand is NaN, return the OTHER.
// x86 _mm_min_ps/_mm_max_ps return the second operand on any-NaN, so we can't map directly;
// reuse IsNaN + IfThenElse + Min/Max. Integer lanes have no NaN, so they are just Min/Max.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> MinNumber(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  if constexpr (std::is_floating_point_v<T>)
    return IfThenElse(IsNaN(a), b, IfThenElse(IsNaN(b), a, Min(a, b)));
  else return Min(a, b);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> MaxNumber(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  if constexpr (std::is_floating_point_v<T>)
    return IfThenElse(IsNaN(a), b, IfThenElse(IsNaN(b), a, Max(a, b)));
  else return Max(a, b);
}
// AddSub(a,b): a-b in EVEN lanes, a+b in ODD lanes. Native SSE3 _mm_addsub_ps/pd (float).
// Integer has no addsub -> blend Sub(even)/Add(odd) via OddEven (even<-b_arg, odd<-a_arg).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> AddSub(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  Vec128<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm_addsub_ps(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm_addsub_pd(a.raw, b.raw);
  else r = OddEven(Add(a, b), Sub(a, b));  // native addsub is float-only
  return r;
}
// MulRound(a,b) = round(a*b) to nearest-even, result stays float. Reuse Round + Mul.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> MulRound(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  static_assert(std::is_floating_point_v<T>, "MulRound is float only");
  return Round(Mul(a, b));
}
// MulFixedPoint15(a,b): Q1.15 * Q1.15 -> Q1.15 = (2*a*b + 0x8000) >> 16. Native mulhrs (i16).
template <std::size_t N>
HYLE_FORCE_INLINE Vec128<std::int16_t, N> MulFixedPoint15(Vec128<std::int16_t, N> a,
                                                          Vec128<std::int16_t, N> b) noexcept {
  Vec128<std::int16_t, N> r;
  r.raw = _mm_mulhrs_epi16(a.raw, b.raw);  // native Q1.15 rounding multiply-high
  return r;
}
// SaturatedNeg(a) = (a==MIN) ? MAX : -a. i8/i16 reuse native SaturatedSub(0, a) (0-MIN
// saturates to MAX); i32/i64 have no native satsub -> store+reload the scalar formula.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> SaturatedNeg(Vec128<T, N> a) noexcept {
  static_assert(std::is_integral_v<T> && std::is_signed_v<T>, "SaturatedNeg is signed-int");
  if constexpr (sizeof(T) <= 2) {
    return SaturatedSub(Zero(Simd<T, N, 0>()), a);  // 0 - a saturating
  } else {  // i32/i64: no native satsub
    alignas(16) T s[N];
    Store(a, Simd<T, N, 0>(), s);
    for (std::size_t i = 0; i < N; ++i)
      s[i] = (s[i] == std::numeric_limits<T>::min()) ? std::numeric_limits<T>::max()
                                                     : static_cast<T>(-s[i]);
    return Load(Simd<T, N, 0>(), s);
  }
}
// SaturatedAbs(a) = (a==MIN) ? MAX : |a|. |a| for a<0 is exactly SaturatedNeg(a), so reuse
// Lt/IfThenElse + SaturatedNeg (correct for every signed width, MIN maps to MAX).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> SaturatedAbs(Vec128<T, N> a) noexcept {
  static_assert(std::is_integral_v<T> && std::is_signed_v<T>, "SaturatedAbs is signed-int");
  return IfThenElse(Lt(a, Zero(Simd<T, N, 0>())), SaturatedNeg(a), a);
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

// SatWidenMulPairwiseAdd(di16, a_u8, b_i8): pairwise u8*i8 summed into SATURATED i16 -- the
// int8 GEMM inner core (llama.cpp Q8 dot). Native _mm_maddubs_epi16: unsigned a * signed b,
// saturating adjacent pairs. a is UNSIGNED bytes, b is SIGNED bytes; dst has half the lanes.
template <std::size_t NDst, std::size_t NSrc>
HYLE_FORCE_INLINE Vec128<std::int16_t, NDst> SatWidenMulPairwiseAdd(
    Simd<std::int16_t, NDst, 0>, Vec128<std::uint8_t, NSrc> a,
    Vec128<std::int8_t, NSrc> b) noexcept {
  static_assert(NSrc == 2 * NDst, "SatWidenMulPairwiseAdd: u8/i8 -> i16 pairwise");
  Vec128<std::int16_t, NDst> r;
  r.raw = _mm_maddubs_epi16(a.raw, b.raw);
  return r;
}

// ReorderWidenMulAccumulate(dw, a, b, sum0, sum1): pairwise i16 a*b added into sum0 (returned),
// sum1 left untouched -- matches the scalar P (all products into sum0). Callers zero sum1 and
// finalize with RearrangeToOddPlusEven. _mm_madd_epi16 (pairwise i16*i16 -> i32) + _mm_add_epi32.
template <std::size_t NW, std::size_t NSrc>
HYLE_FORCE_INLINE Vec128<std::int32_t, NW> ReorderWidenMulAccumulate(
    Simd<std::int32_t, NW, 0>, Vec128<std::int16_t, NSrc> a, Vec128<std::int16_t, NSrc> b,
    Vec128<std::int32_t, NW> sum0, Vec128<std::int32_t, NW>& /*sum1*/) noexcept {
  static_assert(NSrc == 2 * NW, "ReorderWidenMulAccumulate: i16 -> i32 (dst half lanes)");
  Vec128<std::int32_t, NW> r;
  r.raw = _mm_add_epi32(sum0.raw, _mm_madd_epi16(a.raw, b.raw));
  return r;  // all products land in sum0; sum1 stays as the caller zeroed it
}

// RearrangeToOddPlusEven(sum0, sum1): finalize the accumulate -- Add(sum0, sum1). Our Reorder
// kept everything in sum0 (sum1==0), so each i32 lane = the pairwise widened dot, matching scalar.
template <std::size_t NW>
HYLE_FORCE_INLINE Vec128<std::int32_t, NW> RearrangeToOddPlusEven(
    Vec128<std::int32_t, NW> sum0, Vec128<std::int32_t, NW> sum1) noexcept {
  Vec128<std::int32_t, NW> r;
  r.raw = _mm_add_epi32(sum0.raw, sum1.raw);
  return r;
}

// SumOfMulQuadAccumulate(di32, a_i8, b_i8, sum): sum[i] += 4-way i8 dot a[4i+0..3]*b[4i+0..3].
// emulation: no SSE4/AVX2 VNNI; native vpdpbusd is an AVX512-VNNI profiling upgrade. Store the
// operands to the stack, run the scalar 4-way loop, reload -- correctness over speed here.
template <std::size_t NDst, std::size_t NSrc>
HYLE_FORCE_INLINE Vec128<std::int32_t, NDst> SumOfMulQuadAccumulate(
    Simd<std::int32_t, NDst, 0>, Vec128<std::int8_t, NSrc> a, Vec128<std::int8_t, NSrc> b,
    Vec128<std::int32_t, NDst> sum) noexcept {
  static_assert(NSrc == 4 * NDst, "SumOfMulQuadAccumulate: i8 -> i32 (dst quarter lanes)");
  alignas(16) std::int8_t ab[16];
  alignas(16) std::int8_t bb[16];
  alignas(16) std::int32_t sb[4];
  _mm_store_si128(reinterpret_cast<__m128i*>(ab), a.raw);
  _mm_store_si128(reinterpret_cast<__m128i*>(bb), b.raw);
  _mm_store_si128(reinterpret_cast<__m128i*>(sb), sum.raw);
  for (std::size_t i = 0; i < NDst; ++i)
    for (std::size_t k = 0; k < 4; ++k)
      sb[i] += static_cast<std::int32_t>(ab[4 * i + k]) * bb[4 * i + k];
  Vec128<std::int32_t, NDst> r;
  r.raw = _mm_load_si128(reinterpret_cast<const __m128i*>(sb));
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
// TruncateTo/PromoteEvenTo/PromoteOddTo/U8FromU32: strided/modular width changers.
// correctness-first store+reload; native pack/cvt narrowing is a profiling upgrade.
template <typename DstT, std::size_t N, typename SrcT>
HYLE_FORCE_INLINE Vec128<DstT, N> TruncateTo(Simd<DstT, N, 0>, Vec128<SrcT, N> v) noexcept {
  static_assert(sizeof(DstT) < sizeof(SrcT), "TruncateTo narrows the lane type");
  alignas(16) SrcT src[N];
  alignas(16) DstT dst[N];
  Store(v, Simd<SrcT, N, 0>(), src);
  for (std::size_t i = 0; i < N; ++i) dst[i] = static_cast<DstT>(src[i]);
  return Load(Simd<DstT, N, 0>(), dst);
}
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec128<DstT, NDst> PromoteEvenTo(Simd<DstT, NDst, 0>, Vec128<SrcT, NSrc> v) noexcept {
  static_assert(NSrc == 2 * NDst && sizeof(DstT) > sizeof(SrcT), "PromoteEvenTo: even lanes narrow->wide");
  alignas(16) SrcT src[NSrc];
  alignas(16) DstT dst[NDst];
  Store(v, Simd<SrcT, NSrc, 0>(), src);
  for (std::size_t i = 0; i < NDst; ++i) dst[i] = static_cast<DstT>(src[2 * i]);
  return Load(Simd<DstT, NDst, 0>(), dst);
}
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec128<DstT, NDst> PromoteOddTo(Simd<DstT, NDst, 0>, Vec128<SrcT, NSrc> v) noexcept {
  static_assert(NSrc == 2 * NDst && sizeof(DstT) > sizeof(SrcT), "PromoteOddTo: odd lanes narrow->wide");
  alignas(16) SrcT src[NSrc];
  alignas(16) DstT dst[NDst];
  Store(v, Simd<SrcT, NSrc, 0>(), src);
  for (std::size_t i = 0; i < NDst; ++i) dst[i] = static_cast<DstT>(src[2 * i + 1]);
  return Load(Simd<DstT, NDst, 0>(), dst);
}
template <std::size_t N>
HYLE_FORCE_INLINE Vec128<std::uint8_t, N> U8FromU32(Vec128<std::uint32_t, N> v) noexcept {
  alignas(16) std::uint32_t src[N];
  alignas(16) std::uint8_t dst[N];
  Store(v, Simd<std::uint32_t, N, 0>(), src);
  for (std::size_t i = 0; i < N; ++i) dst[i] = static_cast<std::uint8_t>(src[i]);
  return Load(Simd<std::uint8_t, N, 0>(), dst);
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

// ---- byte table lookup (PSHUFB-style, per-128-bit-block) ----------------------------
// Maps directly to PSHUFB (_mm_shuffle_epi8): within the 128-bit register, result byte i =
// bytes[idx_i & 15], and idx_i's high bit (0x80) zeroes the output -- exactly the scalar
// oracle's block-local contract. Lane types may be float; view both raws as __m128i via the
// free bitcast, shuffle, then cast the result raw back to the index type Raw128T<TI>.
template <typename T, std::size_t N, typename TI, std::size_t NI>
HYLE_FORCE_INLINE Vec128<TI, NI> TableLookupBytes(Vec128<T, N> bytes, Vec128<TI, NI> idx) noexcept {
  static_assert(sizeof(T) * N == sizeof(TI) * NI, "TableLookupBytes: same total byte width");
  const __m128i br = detail::bitcast_raw_sse<__m128i>(bytes.raw);
  const __m128i ir = detail::bitcast_raw_sse<__m128i>(idx.raw);
  // Mask to low 4 bits: clears bit7 (no lane zeroed) and forces block-local index in-range.
  const __m128i sh = _mm_shuffle_epi8(br, _mm_and_si128(ir, _mm_set1_epi8(0x0F)));
  Vec128<TI, NI> r;
  r.raw = detail::bitcast_raw_sse<detail::Raw128T<TI>>(sh);
  return r;
}
template <typename T, std::size_t N, typename TI, std::size_t NI>
HYLE_FORCE_INLINE Vec128<TI, NI> TableLookupBytesOr0(Vec128<T, N> bytes, Vec128<TI, NI> idx) noexcept {
  static_assert(sizeof(T) * N == sizeof(TI) * NI, "TableLookupBytesOr0: same total byte width");
  const __m128i br = detail::bitcast_raw_sse<__m128i>(bytes.raw);
  const __m128i ir = detail::bitcast_raw_sse<__m128i>(idx.raw);
  // Raw PSHUFB: zeroes the output byte natively where the index byte's high bit is set.
  const __m128i sh = _mm_shuffle_epi8(br, ir);
  Vec128<TI, NI> r;
  r.raw = detail::bitcast_raw_sse<detail::Raw128T<TI>>(sh);
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

// ---- floating-point classification -> mask -----------------------------------------
// IsNaN(v): lane true iff NaN. IsInf: true iff +/-infinity. IsFinite: true iff neither.
// IsEitherNaN(a,b): true iff a or b is NaN (the "unordered" predicate). Float lanes only.
// IsNaN/IsEitherNaN use an unordered self/pair-compare (all-ones exactly where NaN). IsInf
// and IsFinite reuse Abs + Eq/Lt against a +inf splat: |v|==inf isolates inf, |v|<inf keeps
// only finite (NaN fails ordered <, inf fails <), matching scalar isinf/isfinite bit-for-bit.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask128<T, N> IsNaN(Vec128<T, N> v) noexcept {
  static_assert(std::is_floating_point_v<T>, "IsNaN is float-only");
  Mask128<T, N> m;
  if constexpr (std::is_same_v<T, float>) m.raw = _mm_cmpunord_ps(v.raw, v.raw);
  else m.raw = _mm_cmpunord_pd(v.raw, v.raw);
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask128<T, N> IsEitherNaN(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  static_assert(std::is_floating_point_v<T>, "IsEitherNaN is float-only");
  Mask128<T, N> m;
  if constexpr (std::is_same_v<T, float>) m.raw = _mm_cmpunord_ps(a.raw, b.raw);
  else m.raw = _mm_cmpunord_pd(a.raw, b.raw);
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask128<T, N> IsInf(Vec128<T, N> v) noexcept {
  static_assert(std::is_floating_point_v<T>, "IsInf is float-only");
  const auto inf = Set(Simd<T, N, 0>(), std::numeric_limits<T>::infinity());
  return Eq(Abs(v), inf);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask128<T, N> IsFinite(Vec128<T, N> v) noexcept {
  static_assert(std::is_floating_point_v<T>, "IsFinite is float-only");
  const auto inf = Set(Simd<T, N, 0>(), std::numeric_limits<T>::infinity());
  return Lt(Abs(v), inf);  // ordered <: NaN and inf both fail, only finite lanes true
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
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> IfThenZeroElse(Mask128<T, N> m, Vec128<T, N> no) noexcept {
  return IfThenElse(m, Zero(Simd<T, N, 0>()), no);
}

// ---- mask-domain logic (bitwise on the all-ones/all-zero lane mask) -----------------
// sse4 masks are lane vectors (compare results: 0xFF..pattern), so mask And/Or/Xor/AndNot/
// Not are the integer bitwise ops on the mask raw. Float/double mask raw is __m128/__m128d;
// route through si128 so one path serves all lane types. all-ones = Eq(x,x) for Not.
namespace detail {
template <typename T> HYLE_FORCE_INLINE __m128i mask_to_i(typename Raw128<T>::type m) noexcept {
  if constexpr (std::is_same_v<T, float>) return _mm_castps_si128(m);
  else if constexpr (std::is_same_v<T, double>) return _mm_castpd_si128(m);
  else return m;
}
template <typename T> HYLE_FORCE_INLINE typename Raw128<T>::type mask_from_i(__m128i m) noexcept {
  if constexpr (std::is_same_v<T, float>) return _mm_castsi128_ps(m);
  else if constexpr (std::is_same_v<T, double>) return _mm_castsi128_pd(m);
  else return m;
}
}  // namespace detail
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask128<T, N> And(Mask128<T, N> a, Mask128<T, N> b) noexcept {
  Mask128<T, N> m; m.raw = detail::mask_from_i<T>(_mm_and_si128(detail::mask_to_i<T>(a.raw), detail::mask_to_i<T>(b.raw))); return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask128<T, N> Or(Mask128<T, N> a, Mask128<T, N> b) noexcept {
  Mask128<T, N> m; m.raw = detail::mask_from_i<T>(_mm_or_si128(detail::mask_to_i<T>(a.raw), detail::mask_to_i<T>(b.raw))); return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask128<T, N> Xor(Mask128<T, N> a, Mask128<T, N> b) noexcept {
  Mask128<T, N> m; m.raw = detail::mask_from_i<T>(_mm_xor_si128(detail::mask_to_i<T>(a.raw), detail::mask_to_i<T>(b.raw))); return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask128<T, N> AndNot(Mask128<T, N> a, Mask128<T, N> b) noexcept {
  // ~a & b (Highway order): _mm_andnot(a,b) computes (~a)&b.
  Mask128<T, N> m; m.raw = detail::mask_from_i<T>(_mm_andnot_si128(detail::mask_to_i<T>(a.raw), detail::mask_to_i<T>(b.raw))); return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask128<T, N> Not(Mask128<T, N> a) noexcept {
  const __m128i ones = _mm_set1_epi32(-1);
  Mask128<T, N> m; m.raw = detail::mask_from_i<T>(_mm_xor_si128(detail::mask_to_i<T>(a.raw), ones)); return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask128<T, N> ExclusiveNeither(Mask128<T, N> a, Mask128<T, N> b) noexcept {
  return Not(Or(a, b));  // ~(a|b)
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
// Mask128<T>::raw is Raw128T<T> -- for float/double it is ALREADY __m128/__m128d (compares
// like _mm_cmpunord_ps and MaskFromVec produce FP-domain masks), so no cast is needed; the
// integer path is the identity too. (A cast here would double-convert and mistype.)
template <class D>
HYLE_FORCE_INLINE detail::V128<D> VecFromMask(D, Mask128<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m) noexcept {
  detail::V128<D> r;
  r.raw = m.raw;
  return r;
}
// MaskFromVec: lane true iff its MSB set (Highway). Route via the per-lane movemask so the
// resulting Mask128 raw is a proper all-ones/all-zero lane vector (what the other mask ops
// expect). Build it by comparing the sign: arithmetic compare (v < 0) reproduces full lanes.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask128<T, N> MaskFromVec(Vec128<T, N> v) noexcept {
  Mask128<T, N> m;
  // A lane's MSB -> all-ones lane: for int use signed compare vs zero; for float use the sign.
  if constexpr (std::is_same_v<T, float>)
    m.raw = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(v.raw), 31));
  else if constexpr (std::is_same_v<T, double>) {
    __m128i hi = _mm_srai_epi32(_mm_castpd_si128(v.raw), 31);          // sign to all 32 hi bits
    m.raw = _mm_castsi128_pd(_mm_shuffle_epi32(hi, _MM_SHUFFLE(3, 3, 1, 1)));  // broadcast to 64b
  } else if constexpr (sizeof(T) == 8) {
    __m128i hi = _mm_srai_epi32(v.raw, 31);
    m.raw = _mm_shuffle_epi32(hi, _MM_SHUFFLE(3, 3, 1, 1));
  } else if constexpr (sizeof(T) == 4) m.raw = _mm_srai_epi32(v.raw, 31);
  else if constexpr (sizeof(T) == 2) m.raw = _mm_srai_epi16(v.raw, 15);
  else m.raw = _mm_cmpgt_epi8(_mm_setzero_si128(), v.raw);  // i8: 0 > v  <=> v's sign set
  return m;
}

// mask <-> bits (reuse the mask_bits movemask helper).
template <class D>
HYLE_FORCE_INLINE std::uint64_t BitsFromMask(D, Mask128<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m) noexcept {
  return detail::mask_bits(m);
}
template <class D>
HYLE_FORCE_INLINE Mask128<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> LoadMaskBits(D d, std::uint64_t bits) noexcept {
  // Rebuild: lane i true iff bit i set. FirstN-style via an index compare against the bitset.
  using T = TFromD<D>;
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  alignas(16) T tmp[kN];
  for (std::size_t i = 0; i < kN; ++i) {
    using U = std::make_unsigned_t<std::conditional_t<std::is_integral_v<T>, T, std::uint64_t>>;
    U u = ((bits >> i) & 1) ? ~U{0} : U{0};
    std::memcpy(&tmp[i], &u, sizeof(T));
  }
  Mask128<T, kN> out;
  out.raw = detail::mask_from_i<T>(_mm_load_si128(reinterpret_cast<const __m128i*>(tmp)));
  return out;
}
template <class D>
HYLE_FORCE_INLINE std::size_t StoreMaskBits(D d, Mask128<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m,
                                            std::uint8_t* p) noexcept {
  const std::uint64_t bits = detail::mask_bits(m);
  const std::size_t nbytes = (HYLE_SIMD_MAX_LANES_D(D) + 7) / 8;
  for (std::size_t b = 0; b < nbytes; ++b) p[b] = static_cast<std::uint8_t>(bits >> (b * 8));
  return nbytes;
}

// prefix-scan masks (leading run relative to first true) -- computed on the bitfield then
// rebuilt via LoadMaskBits. N<=16 so a 32-bit bitfield holds it.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask128<T, N> SetBeforeFirst(Mask128<T, N> m) noexcept {
  const unsigned b = detail::mask_bits(m);
  const unsigned before = b ? ((1u << std::countr_zero(b)) - 1u) : ((N >= 32 ? 0xFFFFFFFFu : (1u << N) - 1u));
  return LoadMaskBits(Simd<T, N, 0>(), before);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask128<T, N> SetAtOrBeforeFirst(Mask128<T, N> m) noexcept {
  const unsigned b = detail::mask_bits(m);
  const unsigned upto = b ? ((1u << (std::countr_zero(b) + 1)) - 1u) : ((N >= 32 ? 0xFFFFFFFFu : (1u << N) - 1u));
  return LoadMaskBits(Simd<T, N, 0>(), upto);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask128<T, N> SetOnlyFirst(Mask128<T, N> m) noexcept {
  const unsigned b = detail::mask_bits(m);
  return LoadMaskBits(Simd<T, N, 0>(), b ? (1u << std::countr_zero(b)) : 0u);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask128<T, N> SetAtOrAfterFirst(Mask128<T, N> m) noexcept {
  const unsigned b = detail::mask_bits(m);
  const unsigned full = (N >= 32) ? 0xFFFFFFFFu : ((1u << N) - 1u);
  const unsigned before = b ? ((1u << std::countr_zero(b)) - 1u) : full;
  return LoadMaskBits(Simd<T, N, 0>(), full & ~before);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE std::size_t FindKnownFirstTrue(Simd<T, N, 0>, Mask128<T, N> m) noexcept {
  return static_cast<std::size_t>(std::countr_zero(detail::mask_bits(m)));
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE std::size_t FindKnownLastTrue(Simd<T, N, 0>, Mask128<T, N> m) noexcept {
  return static_cast<std::size_t>(31 - std::countl_zero(detail::mask_bits(m)));
}

// SumsOf8(v): 16 u8 lanes -> 2 u64 (native _mm_sad_epu8 vs zero). The count core.
template <std::size_t N>
HYLE_FORCE_INLINE Vec128<std::uint64_t, N / 8> SumsOf8(Vec128<std::uint8_t, N> v) noexcept {
  static_assert(N == 16, "sse4 SumsOf8: full 16xu8 -> 2xu64");
  Vec128<std::uint64_t, N / 8> r;
  r.raw = _mm_sad_epu8(v.raw, _mm_setzero_si128());
  return r;
}

// SumsOf2/SumsOf4: sum consecutive pairs/quads of u8 lanes, widening to u16/u32.
// correctness-first store+reload; native (pmaddubsw with 1s / vpaddl chains) is a profiling upgrade.
template <std::size_t N>
HYLE_FORCE_INLINE Vec128<std::uint16_t, N / 2> SumsOf2(Vec128<std::uint8_t, N> v) noexcept {
  static_assert(N % 2 == 0, "sse4 SumsOf2 needs an even lane count");
  alignas(16) std::uint8_t src[N];
  alignas(16) std::uint16_t dst[N / 2];
  _mm_store_si128(reinterpret_cast<__m128i*>(src), v.raw);
  for (std::size_t g = 0; g < N / 2; ++g)
    dst[g] = static_cast<std::uint16_t>(src[2 * g] + src[2 * g + 1]);
  return Load(Simd<std::uint16_t, N / 2, 0>(), dst);
}
template <std::size_t N>
HYLE_FORCE_INLINE Vec128<std::uint32_t, N / 4> SumsOf4(Vec128<std::uint8_t, N> v) noexcept {
  static_assert(N % 4 == 0, "sse4 SumsOf4 needs a lane count that is a multiple of 4");
  alignas(16) std::uint8_t src[N];
  alignas(16) std::uint32_t dst[N / 4];
  _mm_store_si128(reinterpret_cast<__m128i*>(src), v.raw);
  for (std::size_t g = 0; g < N / 4; ++g)
    dst[g] = static_cast<std::uint32_t>(src[4 * g] + src[4 * g + 1] + src[4 * g + 2] + src[4 * g + 3]);
  return Load(Simd<std::uint32_t, N / 4, 0>(), dst);
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

// ---- swizzle: B-2 completions (Reverse4/8, ReverseBits/LaneBytes, InterleaveEven/Odd,
// Slide1Up/Down, SlideUp/DownLanes, BroadcastLane, Zip). ------------------------------
// correctness-first store+reload; native shuffles are a profiling upgrade.
template <class D>
HYLE_FORCE_INLINE detail::V128<D> Reverse4(D, detail::V128<D> v) noexcept {
  return detail::swizzle_via_mem(v, [](std::size_t i) { return (i & ~std::size_t{3}) | ((i & 3) ^ 3); });
}
template <class D>
HYLE_FORCE_INLINE detail::V128<D> Reverse8(D, detail::V128<D> v) noexcept {
  return detail::swizzle_via_mem(v, [](std::size_t i) { return (i & ~std::size_t{7}) | ((i & 7) ^ 7); });
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> ReverseBits(Vec128<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "ReverseBits is integer-only");
  using U = std::make_unsigned_t<T>;
  alignas(16) T s[N], d[N];
  Store(v, Simd<T, N, 0>(), s);
  for (std::size_t i = 0; i < N; ++i) {
    U x = static_cast<U>(s[i]), y = 0;
    for (std::size_t b = 0; b < sizeof(T) * 8; ++b) { y = static_cast<U>((y << 1) | (x & 1)); x >>= 1; }
    d[i] = static_cast<T>(y);
  }
  return Load(Simd<T, N, 0>(), d);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> ReverseLaneBytes(Vec128<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "ReverseLaneBytes is integer-only");
  alignas(16) T s[N], d[N];
  Store(v, Simd<T, N, 0>(), s);
  for (std::size_t i = 0; i < N; ++i) {
    auto* db = reinterpret_cast<std::uint8_t*>(&d[i]);
    const auto* sb = reinterpret_cast<const std::uint8_t*>(&s[i]);
    for (std::size_t b = 0; b < sizeof(T); ++b) db[b] = sb[sizeof(T) - 1 - b];
  }
  return Load(Simd<T, N, 0>(), d);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> InterleaveEven(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  alignas(16) T sa[N], sb[N], d[N];
  Store(a, Simd<T, N, 0>(), sa); Store(b, Simd<T, N, 0>(), sb);
  for (std::size_t i = 0; i < N / 2; ++i) { d[2 * i] = sa[2 * i]; d[2 * i + 1] = sb[2 * i]; }
  return Load(Simd<T, N, 0>(), d);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> InterleaveOdd(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  alignas(16) T sa[N], sb[N], d[N];
  Store(a, Simd<T, N, 0>(), sa); Store(b, Simd<T, N, 0>(), sb);
  for (std::size_t i = 0; i < N / 2; ++i) { d[2 * i] = sa[2 * i + 1]; d[2 * i + 1] = sb[2 * i + 1]; }
  return Load(Simd<T, N, 0>(), d);
}
template <class D>
HYLE_FORCE_INLINE detail::V128<D> Slide1Up(D d, detail::V128<D> v) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(16) T s[N]; alignas(16) T dst[N];
  Store(v, d, s);
  dst[0] = T{0};
  for (std::size_t i = 1; i < N; ++i) dst[i] = s[i - 1];
  return Load(d, dst);
}
template <class D>
HYLE_FORCE_INLINE detail::V128<D> Slide1Down(D d, detail::V128<D> v) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(16) T s[N]; alignas(16) T dst[N];
  Store(v, d, s);
  dst[N - 1] = T{0};
  for (std::size_t i = 0; i + 1 < N; ++i) dst[i] = s[i + 1];
  return Load(d, dst);
}
template <class D>
HYLE_FORCE_INLINE detail::V128<D> SlideUpLanes(D d, detail::V128<D> v, std::size_t amt) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(16) T s[N]; alignas(16) T dst[N] = {};
  Store(v, d, s);
  for (std::size_t i = amt; i < N; ++i) dst[i] = s[i - amt];
  return Load(d, dst);
}
template <class D>
HYLE_FORCE_INLINE detail::V128<D> SlideDownLanes(D d, detail::V128<D> v, std::size_t amt) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(16) T s[N]; alignas(16) T dst[N] = {};
  Store(v, d, s);
  for (std::size_t i = 0; i + amt < N; ++i) dst[i] = s[i + amt];
  return Load(d, dst);
}
template <int kLane, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> BroadcastLane(Vec128<T, N> v) noexcept { return Broadcast<kLane>(v); }
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> ZipLower(Vec128<T, N> a, Vec128<T, N> b) noexcept { return InterleaveLower(a, b); }
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> ZipUpper(Vec128<T, N> a, Vec128<T, N> b) noexcept { return InterleaveUpper(a, b); }

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

// ---- crypto: AES round primitives + carryless multiply (simd-08 档C) -----------------
// One __m128i == one 128-bit AES block, so each op is a single AES-NI / PCLMULQDQ call
// (or the shared software fallback when -maes/-mpclmul are absent). The lane type is only
// a view: the contract is a 128-bit vector (sizeof(T)*N == 16). Integer lanes store raw as
// __m128i directly; FP lanes would need a cast, but AES on f32/f64 is nonsensical and the
// static_assert below rejects the non-128-bit case (matches Highway's uint8_t/uint64_t-only
// crypto surface, generalized to any 128-bit integer view).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> AESRound(Vec128<T, N> state, Vec128<T, N> key) noexcept {
  static_assert(sizeof(T) * N == 16, "AES op needs a 128-bit vector");
  return Vec128<T, N>{detail::aes_round_block(state.raw, key.raw)};
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> AESLastRound(Vec128<T, N> state, Vec128<T, N> key) noexcept {
  static_assert(sizeof(T) * N == 16, "AES op needs a 128-bit vector");
  return Vec128<T, N>{detail::aes_last_round_block(state.raw, key.raw)};
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> AESRoundInv(Vec128<T, N> state, Vec128<T, N> key) noexcept {
  static_assert(sizeof(T) * N == 16, "AES op needs a 128-bit vector");
  return Vec128<T, N>{detail::aes_round_inv_block(state.raw, key.raw)};
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> AESLastRoundInv(Vec128<T, N> state, Vec128<T, N> key) noexcept {
  static_assert(sizeof(T) * N == 16, "AES op needs a 128-bit vector");
  return Vec128<T, N>{detail::aes_last_round_inv_block(state.raw, key.raw)};
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> AESInvMixColumns(Vec128<T, N> state) noexcept {
  static_assert(sizeof(T) * N == 16, "AES op needs a 128-bit vector");
  return Vec128<T, N>{detail::aes_inv_mix_block(state.raw)};
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> CLMulLower(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  static_assert(sizeof(T) * N == 16, "CLMul needs a 128-bit vector");
  return Vec128<T, N>{detail::clmul_lower_block(a.raw, b.raw)};
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> CLMulUpper(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  static_assert(sizeof(T) * N == 16, "CLMul needs a 128-bit vector");
  return Vec128<T, N>{detail::clmul_upper_block(a.raw, b.raw)};
}

}  // namespace hyle::simd
