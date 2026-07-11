#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>  // std::memcpy (LoadMaskBits lane pun)
#include <bit>
#include <cmath>   // std::logb / std::ldexp (GetExponent / MulByPow2 store+reload)
#include <limits>  // std::numeric_limits<T>::infinity() (FP classification splat)
#include <type_traits>

#include <immintrin.h>  // AVX/AVX2/FMA (and transitively SSE*)

#include "hyle/macros.h"
#include "hyle/simd/backend/crypto_x86.h"  // shared 128-bit AES-NI/PCLMUL workers (Crypto op face)
#include "hyle/simd/backend/x86_detail.h"  // shared lane-type predicates (kSupported/kI*/kSigned)
// Layered include (simd-17 batch0): pull in the 128-bit backend so Vec128 coexists with Vec256
// in this TU. Legal because AVX2 hardware executes SSE instructions (VEX-128, zero penalty).
// Enables the half-vector family (LowerHalf(Vec256)->Vec128) + the partial-tag Load/Store used
// by TruncateTo/PromoteEvenTo (a sub-128 tag resolves to sse4's width-banded overload).
#include "hyle/simd/backend/sse4.h"

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

// kSupported / kI8 / kI16 / kI32 / kI64 / kSigned are lane-type-only predicates shared by all
// x86 backends -- defined once in backend/x86_detail.h (included above).
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

// Guarded so the layered include (sse4.h, pulled in above) doesn't clash -- MSVC C2386 otherwise.
#ifndef HYLE_SIMD_X86_VT_ALIASES
#define HYLE_SIMD_X86_VT_ALIASES
template <class V>
using DFromV = Simd<typename V::PrivateT, V::kPrivateN, 0>;
template <class V>
using TFromV = typename V::PrivateT;
#endif  // HYLE_SIMD_X86_VT_ALIASES

namespace detail {
template <class D>
using V256 = Vec256<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)>;
}  // namespace detail

// ---- initialization -----------------------------------------------------------------
template <class D, typename T2> requires HYLE_SIMD_W(D, 16, 32)
HYLE_FORCE_INLINE detail::V256<D> Set(D, T2 t) noexcept {
  using T = TFromD<D>;
  detail::V256<D> v;
  if constexpr (std::is_same_v<T, float>) v.raw = _mm256_set1_ps(static_cast<float>(t));
  else if constexpr (std::is_same_v<T, double>) v.raw = _mm256_set1_pd(static_cast<double>(t));
  else if constexpr (detail::kI8<T>) v.raw = _mm256_set1_epi8(static_cast<char>(t));
  else if constexpr (detail::kI16<T>) v.raw = _mm256_set1_epi16(static_cast<short>(t));
  else if constexpr (detail::kI64<T>) v.raw = _mm256_set1_epi64x(static_cast<long long>(t));
  else v.raw = _mm256_set1_epi32(static_cast<int>(t));  // i32/u32
  return v;
}
template <class D> requires HYLE_SIMD_W(D, 16, 32)
HYLE_FORCE_INLINE detail::V256<D> Zero(D) noexcept {
  using T = TFromD<D>;
  detail::V256<D> v;
  if constexpr (std::is_same_v<T, float>) v.raw = _mm256_setzero_ps();
  else if constexpr (std::is_same_v<T, double>) v.raw = _mm256_setzero_pd();
  else v.raw = _mm256_setzero_si256();
  return v;
}
// VFromD = decltype(Zero(D())) is a DEPENDENT alias: it resolves Zero at instantiation, when all
// layered width-band overloads are visible, so one definition serves every width. Guarded so the
// layered includes don't redefine it (MSVC C2386; sse4.h defines it first).
#ifndef HYLE_SIMD_X86_VFROMD
#define HYLE_SIMD_X86_VFROMD
template <class D>
using VFromD = decltype(Zero(D()));
#endif  // HYLE_SIMD_X86_VFROMD

// ---- load / store -------------------------------------------------------------------
template <class D> requires HYLE_SIMD_W(D, 16, 32)
HYLE_FORCE_INLINE detail::V256<D> LoadU(D, const TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  detail::V256<D> v;
  if constexpr (std::is_same_v<T, float>) v.raw = _mm256_loadu_ps(p);
  else if constexpr (std::is_same_v<T, double>) v.raw = _mm256_loadu_pd(p);
  else v.raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
  return v;
}
template <class D> requires HYLE_SIMD_W(D, 16, 32)
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

// ---- interleaved load / store (AoS <-> SoA) -----------------------------------------
// correctness-first strided copy; native deinterleave shuffles are a profiling upgrade.
template <class D> requires HYLE_SIMD_W(D, 16, 32)
HYLE_FORCE_INLINE void LoadInterleaved2(D d, const TFromD<D>* p, VFromD<D>& v0,
                                        VFromD<D>& v1) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(32) T t0[N], t1[N];
  for (std::size_t i = 0; i < N; ++i) { t0[i] = p[2 * i]; t1[i] = p[2 * i + 1]; }
  v0 = Load(d, t0); v1 = Load(d, t1);
}
template <class D> requires HYLE_SIMD_W(D, 16, 32)
HYLE_FORCE_INLINE void LoadInterleaved3(D d, const TFromD<D>* p, VFromD<D>& v0, VFromD<D>& v1,
                                        VFromD<D>& v2) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(32) T t0[N], t1[N], t2[N];
  for (std::size_t i = 0; i < N; ++i) { t0[i] = p[3*i]; t1[i] = p[3*i+1]; t2[i] = p[3*i+2]; }
  v0 = Load(d, t0); v1 = Load(d, t1); v2 = Load(d, t2);
}
template <class D> requires HYLE_SIMD_W(D, 16, 32)
HYLE_FORCE_INLINE void LoadInterleaved4(D d, const TFromD<D>* p, VFromD<D>& v0, VFromD<D>& v1,
                                        VFromD<D>& v2, VFromD<D>& v3) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(32) T t0[N], t1[N], t2[N], t3[N];
  for (std::size_t i = 0; i < N; ++i) { t0[i]=p[4*i]; t1[i]=p[4*i+1]; t2[i]=p[4*i+2]; t3[i]=p[4*i+3]; }
  v0 = Load(d, t0); v1 = Load(d, t1); v2 = Load(d, t2); v3 = Load(d, t3);
}
template <class D> requires HYLE_SIMD_W(D, 16, 32)
HYLE_FORCE_INLINE void StoreInterleaved2(VFromD<D> v0, VFromD<D> v1, D d, TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(32) T t0[N], t1[N];
  Store(v0, d, t0); Store(v1, d, t1);
  for (std::size_t i = 0; i < N; ++i) { p[2*i] = t0[i]; p[2*i+1] = t1[i]; }
}
template <class D> requires HYLE_SIMD_W(D, 16, 32)
HYLE_FORCE_INLINE void StoreInterleaved3(VFromD<D> v0, VFromD<D> v1, VFromD<D> v2, D d,
                                         TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(32) T t0[N], t1[N], t2[N];
  Store(v0, d, t0); Store(v1, d, t1); Store(v2, d, t2);
  for (std::size_t i = 0; i < N; ++i) { p[3*i]=t0[i]; p[3*i+1]=t1[i]; p[3*i+2]=t2[i]; }
}
template <class D> requires HYLE_SIMD_W(D, 16, 32)
HYLE_FORCE_INLINE void StoreInterleaved4(VFromD<D> v0, VFromD<D> v1, VFromD<D> v2, VFromD<D> v3, D d,
                                         TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(32) T t0[N], t1[N], t2[N], t3[N];
  Store(v0, d, t0); Store(v1, d, t1); Store(v2, d, t2); Store(v3, d, t3);
  for (std::size_t i = 0; i < N; ++i) { p[4*i]=t0[i]; p[4*i+1]=t1[i]; p[4*i+2]=t2[i]; p[4*i+3]=t3[i]; }
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

// ---- math / arith extensions (Highway) ----------------------------------------------
// ApproximateReciprocal(a) ~= 1/a: f32 native _mm256_rcp_ps (12-bit); f64 has NO native
// rcp, fall back to the exact 1/a = Div(Set(1),a) (reusing Div/Set).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> ApproximateReciprocal(Vec256<T, N> a) noexcept {
  static_assert(std::is_floating_point_v<T>, "float only");
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_rcp_ps(a.raw);
  else r = Div(Set(Simd<T, N, 0>(), 1.0), a);  // no f64 rcp intrinsic: exact 1/a
  return r;
}
// ApproximateReciprocalSqrt(a) ~= 1/sqrt(a): f32 native _mm256_rsqrt_ps (12-bit); f64 has
// no native rsqrt, fall back to Div(Set(1), Sqrt(a)) (reusing Sqrt/Div/Set).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> ApproximateReciprocalSqrt(Vec256<T, N> a) noexcept {
  static_assert(std::is_floating_point_v<T>, "float only");
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_rsqrt_ps(a.raw);
  else r = Div(Set(Simd<T, N, 0>(), 1.0), Sqrt(a));  // no f64 rsqrt intrinsic
  return r;
}
// GetExponent(v) = floor(log2(|v|)) as float (== std::logb for normal inputs). NATIVE bit-extract
// (Highway's approach): pull the biased exponent field out of the float bits and unbias, all in
// vector integer ops -- no store+reload/libm (op_parity: the old std::logb path was 30x slower).
// Single-instruction vgetexpps is AVX-512 only; this AVX2 bit-math is the portable equivalent.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> GetExponent(Vec256<T, N> v) noexcept {
  static_assert(std::is_floating_point_v<T>, "float only");
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, float>) {
    const __m256i bits = _mm256_castps_si256(v.raw);
    const __m256i biased = _mm256_and_si256(_mm256_srli_epi32(bits, 23), _mm256_set1_epi32(0xFF));
    const __m256i unbiased = _mm256_sub_epi32(biased, _mm256_set1_epi32(127));
    r.raw = _mm256_cvtepi32_ps(unbiased);
  } else {  // double: 11-bit exponent, bias 1023
    const __m256i bits = _mm256_castpd_si256(v.raw);
    const __m256i biased = _mm256_and_si256(_mm256_srli_epi64(bits, 52), _mm256_set1_epi64x(0x7FF));
    const __m256i unbiased = _mm256_sub_epi64(biased, _mm256_set1_epi64x(1023));
    // i64 -> f64: extract the two lanes (no direct AVX2 cvtepi64_pd); small, exact for |exp|<2^52.
    alignas(32) std::int64_t e[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(e), unbiased);
    r.raw = _mm256_set_pd(static_cast<double>(e[3]), static_cast<double>(e[2]),
                          static_cast<double>(e[1]), static_cast<double>(e[0]));
  }
  return r;
}
// MulByPow2(a, b) = a * 2^b (std::ldexp), b an integer-lane vector of matching width.
// store+reload std::ldexp; no clean AVX2 native (AVX512 vscalefps is the hardware form).
template <typename T, std::size_t N, typename IT>
HYLE_FORCE_INLINE Vec256<T, N> MulByPow2(Vec256<T, N> a, Vec256<IT, N> b) noexcept {
  static_assert(std::is_floating_point_v<T>, "float only");
  alignas(32) T sa[N];
  alignas(32) IT sb[N];
  Store(a, Simd<T, N, 0>(), sa);
  Store(b, Simd<IT, N, 0>(), sb);
  for (std::size_t i = 0; i < N; ++i) sa[i] = std::ldexp(sa[i], static_cast<int>(sb[i]));
  return Load(Simd<T, N, 0>(), sa);
}
// NaN-aware min/max (Highway MinNumber/MaxNumber): if one operand is NaN, return the OTHER.
// x86 _mm256_min_ps/_mm256_max_ps return the second operand on any-NaN, so we can't map
// directly; reuse IsNaN + IfThenElse + Min/Max. Integer lanes have no NaN -> just Min/Max.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> MinNumber(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  if constexpr (std::is_floating_point_v<T>)
    return IfThenElse(IsNaN(a), b, IfThenElse(IsNaN(b), a, Min(a, b)));
  else return Min(a, b);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> MaxNumber(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  if constexpr (std::is_floating_point_v<T>)
    return IfThenElse(IsNaN(a), b, IfThenElse(IsNaN(b), a, Max(a, b)));
  else return Max(a, b);
}
// AddSub(a,b): a-b in EVEN lanes, a+b in ODD lanes. Native _mm256_addsub_ps/pd (float).
// Integer has no addsub -> blend Sub(even)/Add(odd) via OddEven (even<-b_arg, odd<-a_arg).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> AddSub(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_addsub_ps(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm256_addsub_pd(a.raw, b.raw);
  else r = OddEven(Add(a, b), Sub(a, b));  // native addsub is float-only
  return r;
}
// MulRound(a,b) = round(a*b) to nearest-even, result stays float. Reuse Round + Mul.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> MulRound(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  static_assert(std::is_floating_point_v<T>, "MulRound is float only");
  return Round(Mul(a, b));
}
// MulFixedPoint15(a,b): Q1.15 * Q1.15 -> Q1.15 = (2*a*b + 0x8000) >> 16. Native mulhrs (i16).
template <std::size_t N>
HYLE_FORCE_INLINE Vec256<std::int16_t, N> MulFixedPoint15(Vec256<std::int16_t, N> a,
                                                          Vec256<std::int16_t, N> b) noexcept {
  Vec256<std::int16_t, N> r;
  r.raw = _mm256_mulhrs_epi16(a.raw, b.raw);  // native Q1.15 rounding multiply-high
  return r;
}
// SaturatedNeg(a) = (a==MIN) ? MAX : -a. i8/i16 reuse native SaturatedSub(0, a) (0-MIN
// saturates to MAX); i32/i64 have no native satsub -> store+reload the scalar formula.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> SaturatedNeg(Vec256<T, N> a) noexcept {
  static_assert(std::is_integral_v<T> && std::is_signed_v<T>, "SaturatedNeg is signed-int");
  if constexpr (sizeof(T) <= 2) {
    return SaturatedSub(Zero(Simd<T, N, 0>()), a);  // 0 - a saturating
  } else {  // i32/i64: no native satsub
    alignas(32) T s[N];
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
HYLE_FORCE_INLINE Vec256<T, N> SaturatedAbs(Vec256<T, N> a) noexcept {
  static_assert(std::is_integral_v<T> && std::is_signed_v<T>, "SaturatedAbs is signed-int");
  return IfThenElse(Lt(a, Zero(Simd<T, N, 0>())), SaturatedNeg(a), a);
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

// Uniform run-time shifts use the scalar-count form instead of materializing a per-lane count.
template <std::size_t N>
HYLE_FORCE_INLINE Vec256<std::uint32_t, N> ShiftRightUniform(Vec256<std::uint32_t, N> a,
                                                              int bits) noexcept {
  static_assert(N == 8, "ShiftRightUniform u32 requires a native AVX2 vector");
  return Vec256<std::uint32_t, N>{_mm256_srl_epi32(a.raw, _mm_cvtsi32_si128(bits))};
}
template <std::size_t N>
HYLE_FORCE_INLINE Vec256<std::int32_t, N> ShiftRightUniform(Vec256<std::int32_t, N> a,
                                                             int bits) noexcept {
  static_assert(N == 8, "ShiftRightUniform i32 requires a native AVX2 vector");
  return Vec256<std::int32_t, N>{_mm256_sra_epi32(a.raw, _mm_cvtsi32_si128(bits))};
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

// SatWidenMulPairwiseAdd(di16, a_u8, b_i8): pairwise u8*i8 summed into SATURATED i16 -- the
// int8 GEMM inner core (llama.cpp Q8 dot). Native _mm256_maddubs_epi16: unsigned a * signed b,
// saturating adjacent pairs. a is UNSIGNED bytes, b is SIGNED bytes; dst has half the lanes.
template <std::size_t NDst, std::size_t NSrc>
HYLE_FORCE_INLINE Vec256<std::int16_t, NDst> SatWidenMulPairwiseAdd(
    Simd<std::int16_t, NDst, 0>, Vec256<std::uint8_t, NSrc> a,
    Vec256<std::int8_t, NSrc> b) noexcept {
  static_assert(NSrc == 2 * NDst, "SatWidenMulPairwiseAdd: u8/i8 -> i16 pairwise");
  Vec256<std::int16_t, NDst> r;
  r.raw = _mm256_maddubs_epi16(a.raw, b.raw);
  return r;
}

// ReorderWidenMulAccumulate(dw, a, b, sum0, sum1): pairwise i16 a*b added into sum0 (returned),
// sum1 left untouched -- matches the scalar P (all products into sum0). Callers zero sum1 and
// finalize with RearrangeToOddPlusEven. _mm256_madd_epi16 (pairwise i16*i16) + _mm256_add_epi32.
template <std::size_t NW, std::size_t NSrc>
HYLE_FORCE_INLINE Vec256<std::int32_t, NW> ReorderWidenMulAccumulate(
    Simd<std::int32_t, NW, 0>, Vec256<std::int16_t, NSrc> a, Vec256<std::int16_t, NSrc> b,
    Vec256<std::int32_t, NW> sum0, Vec256<std::int32_t, NW>& /*sum1*/) noexcept {
  static_assert(NSrc == 2 * NW, "ReorderWidenMulAccumulate: i16 -> i32 (dst half lanes)");
  Vec256<std::int32_t, NW> r;
  r.raw = _mm256_add_epi32(sum0.raw, _mm256_madd_epi16(a.raw, b.raw));
  return r;  // all products land in sum0; sum1 stays as the caller zeroed it
}

// RearrangeToOddPlusEven(sum0, sum1): finalize the accumulate -- Add(sum0, sum1). Our Reorder
// kept everything in sum0 (sum1==0), so each i32 lane = the pairwise widened dot, matching scalar.
template <std::size_t NW>
HYLE_FORCE_INLINE Vec256<std::int32_t, NW> RearrangeToOddPlusEven(
    Vec256<std::int32_t, NW> sum0, Vec256<std::int32_t, NW> sum1) noexcept {
  Vec256<std::int32_t, NW> r;
  r.raw = _mm256_add_epi32(sum0.raw, sum1.raw);
  return r;
}

// SumOfMulQuadAccumulate(di32, a_i8, b_i8, sum): sum[i] += 4-way i8 dot a[4i+0..3]*b[4i+0..3].
// emulation: no SSE4/AVX2 VNNI; native vpdpbusd is an AVX512-VNNI profiling upgrade. Store the
// operands to the stack, run the scalar 4-way loop, reload -- correctness over speed here.
template <std::size_t NDst, std::size_t NSrc>
HYLE_FORCE_INLINE Vec256<std::int32_t, NDst> SumOfMulQuadAccumulate(
    Simd<std::int32_t, NDst, 0>, Vec256<std::int8_t, NSrc> a, Vec256<std::int8_t, NSrc> b,
    Vec256<std::int32_t, NDst> sum) noexcept {
  static_assert(NSrc == 4 * NDst, "SumOfMulQuadAccumulate: i8 -> i32 (dst quarter lanes)");
  alignas(32) std::int8_t ab[32];
  alignas(32) std::int8_t bb[32];
  alignas(32) std::int32_t sb[8];
  _mm256_store_si256(reinterpret_cast<__m256i*>(ab), a.raw);
  _mm256_store_si256(reinterpret_cast<__m256i*>(bb), b.raw);
  _mm256_store_si256(reinterpret_cast<__m256i*>(sb), sum.raw);
  for (std::size_t i = 0; i < NDst; ++i)
    for (std::size_t k = 0; k < 4; ++k)
      sb[i] += static_cast<std::int32_t>(ab[4 * i + k]) * bb[4 * i + k];
  Vec256<std::int32_t, NDst> r;
  r.raw = _mm256_load_si256(reinterpret_cast<const __m256i*>(sb));
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

// ---- bit-cast (reinterpret 256 bits as another lane type; zero-cost register cast) --
// Same as sse4's BitCast but 256-bit: free _mm256_cast* between __m256/__m256i/__m256d,
// plain move within integer widths. Total bytes preserved (32) so lane count tracks width.
namespace detail {
template <typename DstRaw, typename SrcRaw>
HYLE_FORCE_INLINE DstRaw bitcast_raw_avx2(SrcRaw s) noexcept {
  if constexpr (std::is_same_v<DstRaw, SrcRaw>) return s;
  else if constexpr (std::is_same_v<SrcRaw, __m256> && std::is_same_v<DstRaw, __m256i>)
    return _mm256_castps_si256(s);
  else if constexpr (std::is_same_v<SrcRaw, __m256d> && std::is_same_v<DstRaw, __m256i>)
    return _mm256_castpd_si256(s);
  else if constexpr (std::is_same_v<SrcRaw, __m256i> && std::is_same_v<DstRaw, __m256>)
    return _mm256_castsi256_ps(s);
  else if constexpr (std::is_same_v<SrcRaw, __m256i> && std::is_same_v<DstRaw, __m256d>)
    return _mm256_castsi256_pd(s);
  else if constexpr (std::is_same_v<SrcRaw, __m256> && std::is_same_v<DstRaw, __m256d>)
    return _mm256_castps_pd(s);
  else if constexpr (std::is_same_v<SrcRaw, __m256d> && std::is_same_v<DstRaw, __m256>)
    return _mm256_castpd_ps(s);
  else static_assert(sizeof(DstRaw) == 0, "avx2 bitcast_raw: unhandled raw pair");
}
}  // namespace detail
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec256<DstT, NDst> BitCast(Simd<DstT, NDst, 0>, Vec256<SrcT, NSrc> v) noexcept {
  static_assert(sizeof(DstT) * NDst == sizeof(SrcT) * NSrc, "BitCast preserves total bytes");
  Vec256<DstT, NDst> r;
  r.raw = detail::bitcast_raw_avx2<detail::Raw256T<DstT>>(v.raw);
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
// TruncateTo/PromoteEvenTo/PromoteOddTo/U8FromU32: strided/modular width changers.
// correctness-first store+reload; native pack/cvt narrowing is a profiling upgrade.
// Store()/Load() move a FULL 256-bit register (_mm256_store_si256 / _mm256_load_si256), so
// the scratch buffers MUST be a whole register wide (32 B) regardless of logical lane count:
// a narrow partial result still reloads via _mm256_load_si256 reading 32 B, so a lane-count-
// sized `DstT dst[N]` would be an out-of-bounds stack read. Size every buffer to 32 B.
// (Found via GCC segfault on Vec.TruncateAndStridedPromote; MSVC happened not to fault.)
// Narrowing result is N DstT lanes = N*sizeof(DstT) bytes, which may be < 32 (e.g. u32->u8,
// N=8 -> 8 bytes). Post-layering that reloads through the width-banded Load, so the natural
// return type is VFromD<Simd<DstT,N,0>> (a Vec128 for sub-256 results) -- not a Vec256 shell.
template <typename DstT, std::size_t N, typename SrcT>
HYLE_FORCE_INLINE VFromD<Simd<DstT, N, 0>> TruncateTo(Simd<DstT, N, 0>, Vec256<SrcT, N> v) noexcept {
  static_assert(sizeof(DstT) < sizeof(SrcT), "TruncateTo narrows the lane type");
  alignas(32) SrcT src[32 / sizeof(SrcT)];
  alignas(32) DstT dst[32 / sizeof(DstT)] = {};
  Store(v, Simd<SrcT, N, 0>(), src);
  for (std::size_t i = 0; i < N; ++i) dst[i] = static_cast<DstT>(src[i]);
  return Load(Simd<DstT, N, 0>(), dst);
}
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec256<DstT, NDst> PromoteEvenTo(Simd<DstT, NDst, 0>, Vec256<SrcT, NSrc> v) noexcept {
  static_assert(NSrc == 2 * NDst && sizeof(DstT) > sizeof(SrcT), "PromoteEvenTo: even lanes narrow->wide");
  alignas(32) SrcT src[32 / sizeof(SrcT)];
  alignas(32) DstT dst[32 / sizeof(DstT)] = {};
  Store(v, Simd<SrcT, NSrc, 0>(), src);
  for (std::size_t i = 0; i < NDst; ++i) dst[i] = static_cast<DstT>(src[2 * i]);
  return Load(Simd<DstT, NDst, 0>(), dst);
}
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec256<DstT, NDst> PromoteOddTo(Simd<DstT, NDst, 0>, Vec256<SrcT, NSrc> v) noexcept {
  static_assert(NSrc == 2 * NDst && sizeof(DstT) > sizeof(SrcT), "PromoteOddTo: odd lanes narrow->wide");
  alignas(32) SrcT src[32 / sizeof(SrcT)];
  alignas(32) DstT dst[32 / sizeof(DstT)] = {};
  Store(v, Simd<SrcT, NSrc, 0>(), src);
  for (std::size_t i = 0; i < NDst; ++i) dst[i] = static_cast<DstT>(src[2 * i + 1]);
  return Load(Simd<DstT, NDst, 0>(), dst);
}
template <std::size_t N>
HYLE_FORCE_INLINE VFromD<Simd<std::uint8_t, N, 0>> U8FromU32(Vec256<std::uint32_t, N> v) noexcept {
  alignas(32) std::uint32_t src[32 / sizeof(std::uint32_t)];
  alignas(32) std::uint8_t dst[32] = {};
  Store(v, Simd<std::uint32_t, N, 0>(), src);
  for (std::size_t i = 0; i < N; ++i) dst[i] = static_cast<std::uint8_t>(src[i]);
  return Load(Simd<std::uint8_t, N, 0>(), dst);
}

// P4: AVX2 has native dword->qword products on the even dword of every qword.
template <std::size_t N>
HYLE_FORCE_INLINE Vec256<std::int64_t, N / 2> MulEven(Vec256<std::int32_t, N> a,
                                                       Vec256<std::int32_t, N> b) noexcept {
  static_assert(N == 8, "MulEven i32 requires a native AVX2 vector");
  return Vec256<std::int64_t, N / 2>{_mm256_mul_epi32(a.raw, b.raw)};
}
template <std::size_t N>
HYLE_FORCE_INLINE Vec256<std::uint64_t, N / 2> MulEven(Vec256<std::uint32_t, N> a,
                                                        Vec256<std::uint32_t, N> b) noexcept {
  static_assert(N == 8, "MulEven u32 requires a native AVX2 vector");
  return Vec256<std::uint64_t, N / 2>{_mm256_mul_epu32(a.raw, b.raw)};
}
template <std::size_t N>
HYLE_FORCE_INLINE Vec256<std::int64_t, N / 2> MulOdd(Vec256<std::int32_t, N> a,
                                                      Vec256<std::int32_t, N> b) noexcept {
  static_assert(N == 8, "MulOdd i32 requires a native AVX2 vector");
  return Vec256<std::int64_t, N / 2>{_mm256_mul_epi32(_mm256_srli_epi64(a.raw, 32),
                                                       _mm256_srli_epi64(b.raw, 32))};
}
template <std::size_t N>
HYLE_FORCE_INLINE Vec256<std::uint64_t, N / 2> MulOdd(Vec256<std::uint32_t, N> a,
                                                       Vec256<std::uint32_t, N> b) noexcept {
  static_assert(N == 8, "MulOdd u32 requires a native AVX2 vector");
  return Vec256<std::uint64_t, N / 2>{_mm256_mul_epu32(_mm256_srli_epi64(a.raw, 32),
                                                        _mm256_srli_epi64(b.raw, 32))};
}

template <std::size_t N>
HYLE_FORCE_INLINE Vec256<std::uint32_t, N> MulHigh32(Vec256<std::uint32_t, N> a,
                                                      Vec256<std::uint32_t, N> b) noexcept {
  static_assert(N == 8, "MulHigh32 u32 requires a native AVX2 vector");
  const __m256i even = _mm256_srli_epi64(_mm256_mul_epu32(a.raw, b.raw), 32);
  const __m256i odd = _mm256_srli_epi64(
      _mm256_mul_epu32(_mm256_srli_epi64(a.raw, 32), _mm256_srli_epi64(b.raw, 32)), 32);
  return Vec256<std::uint32_t, N>{_mm256_or_si256(even, _mm256_slli_epi64(odd, 32))};
}
template <std::size_t N>
HYLE_FORCE_INLINE Vec256<std::int32_t, N> MulHigh32(Vec256<std::int32_t, N> a,
                                                     Vec256<std::int32_t, N> b) noexcept {
  static_assert(N == 8, "MulHigh32 i32 requires a native AVX2 vector");
  const __m256i even = _mm256_mul_epi32(a.raw, b.raw);
  const __m256i odd = _mm256_mul_epi32(_mm256_shuffle_epi32(a.raw, _MM_SHUFFLE(3, 3, 1, 1)),
                                       _mm256_shuffle_epi32(b.raw, _MM_SHUFFLE(3, 3, 1, 1)));
  const __m256i packed = _mm256_castps_si256(
      _mm256_shuffle_ps(_mm256_castsi256_ps(even), _mm256_castsi256_ps(odd),
                         _MM_SHUFFLE(3, 1, 3, 1)));
  return Vec256<std::int32_t, N>{_mm256_shuffle_epi32(packed, _MM_SHUFFLE(3, 1, 2, 0))};
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
  // [lo even lanes, hi even lanes]. NATIVE for 32-bit (8 lanes): vpermd gathers even lanes of each
  // (lo->low half, hi->high half) then blend. op_parity: store+reload was 1.7x slower.
  if constexpr (sizeof(T) == 4) {
    const __m256i el = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);  // lo's evens -> lanes 0..3
    const __m256i eh = _mm256_setr_epi32(0, 0, 0, 0, 0, 2, 4, 6);  // hi's evens -> lanes 4..7
    Vec256<T, N> r;
    if constexpr (std::is_same_v<T, float>) {
      const __m256 pl = _mm256_permutevar8x32_ps(lo.raw, el);
      const __m256 ph = _mm256_permutevar8x32_ps(hi.raw, eh);
      r.raw = _mm256_blend_ps(pl, ph, 0xF0);
    } else {
      const __m256i pl = _mm256_permutevar8x32_epi32(lo.raw, el);
      const __m256i ph = _mm256_permutevar8x32_epi32(hi.raw, eh);
      r.raw = _mm256_blend_epi32(pl, ph, 0xF0);
    }
    return r;
  }
  alignas(32) T sl[N], sh[N], dst[N];
  Store(lo, Simd<T, N, 0>(), sl); Store(hi, Simd<T, N, 0>(), sh);
  for (std::size_t i = 0; i < N / 2; ++i) { dst[i] = sl[2 * i]; dst[N / 2 + i] = sh[2 * i]; }
  return Load(Simd<T, N, 0>(), dst);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> ConcatOdd(Vec256<T, N> hi, Vec256<T, N> lo) noexcept {
  // [lo odd lanes, hi odd lanes]. NATIVE for 32-bit (vpermd odd-index gather + blend).
  if constexpr (sizeof(T) == 4) {
    const __m256i ol = _mm256_setr_epi32(1, 3, 5, 7, 0, 0, 0, 0);
    const __m256i oh = _mm256_setr_epi32(0, 0, 0, 0, 1, 3, 5, 7);
    Vec256<T, N> r;
    if constexpr (std::is_same_v<T, float>) {
      const __m256 pl = _mm256_permutevar8x32_ps(lo.raw, ol);
      const __m256 ph = _mm256_permutevar8x32_ps(hi.raw, oh);
      r.raw = _mm256_blend_ps(pl, ph, 0xF0);
    } else {
      const __m256i pl = _mm256_permutevar8x32_epi32(lo.raw, ol);
      const __m256i ph = _mm256_permutevar8x32_epi32(hi.raw, oh);
      r.raw = _mm256_blend_epi32(pl, ph, 0xF0);
    }
    return r;
  }
  alignas(32) T sl[N], sh[N], dst[N];
  Store(lo, Simd<T, N, 0>(), sl); Store(hi, Simd<T, N, 0>(), sh);
  for (std::size_t i = 0; i < N / 2; ++i) { dst[i] = sl[2 * i + 1]; dst[N / 2 + i] = sh[2 * i + 1]; }
  return Load(Simd<T, N, 0>(), dst);
}
// DupEven/DupOdd: within each 128-bit block, broadcast the even/odd element of each adjacent pair
// (Highway contract). NATIVE for 32/64-bit lanes (the hot widths -- op_parity showed the old
// store+reload was 10x slower than Highway); narrow 8/16-bit keep the mem swizzle (no clean single
// native, and not a proven hot path). f32 has dedicated moveldup/movehdup.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> DupEven(Vec256<T, N> v) noexcept {
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_moveldup_ps(v.raw);          // [0,0,2,2,...]
  else if constexpr (detail::kI32<T>) r.raw = _mm256_shuffle_epi32(v.raw, _MM_SHUFFLE(2, 2, 0, 0));
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm256_permute_pd(v.raw, 0);   // [0,0] per 128b
  else if constexpr (detail::kI64<T>)
    r.raw = _mm256_castpd_si256(_mm256_permute_pd(_mm256_castsi256_pd(v.raw), 0));
  else return detail::swizzle_via_mem(v, [](std::size_t i) { return i & ~std::size_t{1}; });
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> DupOdd(Vec256<T, N> v) noexcept {
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_movehdup_ps(v.raw);           // [1,1,3,3,...]
  else if constexpr (detail::kI32<T>) r.raw = _mm256_shuffle_epi32(v.raw, _MM_SHUFFLE(3, 3, 1, 1));
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm256_permute_pd(v.raw, 0xF); // [1,1] per 128b
  else if constexpr (detail::kI64<T>)
    r.raw = _mm256_castpd_si256(_mm256_permute_pd(_mm256_castsi256_pd(v.raw), 0xF));
  else return detail::swizzle_via_mem(v, [](std::size_t i) { return (i & ~std::size_t{1}) | 1; });
  return r;
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

// ---- byte table lookup (PSHUFB-style, per-128-bit-block) ----------------------------
// _mm256_shuffle_epi8 runs PSHUFB independently PER 128-bit lane, which exactly matches the
// scalar oracle's per-16-byte-block semantics (block = (i/16)*16). Same contract as sse4 but
// 256-bit: idx_i & 15 selects within its own block, high bit (0x80) zeroes the output. Lane
// types may be float; view both raws as __m256i via the free bitcast, shuffle, cast back.
template <typename T, std::size_t N, typename TI, std::size_t NI>
HYLE_FORCE_INLINE Vec256<TI, NI> TableLookupBytes(Vec256<T, N> bytes, Vec256<TI, NI> idx) noexcept {
  static_assert(sizeof(T) * N == sizeof(TI) * NI, "TableLookupBytes: same total byte width");
  const __m256i br = detail::bitcast_raw_avx2<__m256i>(bytes.raw);
  const __m256i ir = detail::bitcast_raw_avx2<__m256i>(idx.raw);
  // Mask to low 4 bits: clears bit7 (no lane zeroed) and forces block-local index in-range.
  const __m256i sh = _mm256_shuffle_epi8(br, _mm256_and_si256(ir, _mm256_set1_epi8(0x0F)));
  Vec256<TI, NI> r;
  r.raw = detail::bitcast_raw_avx2<detail::Raw256T<TI>>(sh);
  return r;
}
template <typename T, std::size_t N, typename TI, std::size_t NI>
HYLE_FORCE_INLINE Vec256<TI, NI> TableLookupBytesOr0(Vec256<T, N> bytes, Vec256<TI, NI> idx) noexcept {
  static_assert(sizeof(T) * N == sizeof(TI) * NI, "TableLookupBytesOr0: same total byte width");
  const __m256i br = detail::bitcast_raw_avx2<__m256i>(bytes.raw);
  const __m256i ir = detail::bitcast_raw_avx2<__m256i>(idx.raw);
  // Raw PSHUFB: zeroes the output byte natively where the index byte's high bit is set.
  const __m256i sh = _mm256_shuffle_epi8(br, ir);
  Vec256<TI, NI> r;
  r.raw = detail::bitcast_raw_avx2<detail::Raw256T<TI>>(sh);
  return r;
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

// ---- floating-point classification -> mask -----------------------------------------
// IsNaN(v): lane true iff NaN. IsInf: true iff +/-infinity. IsFinite: true iff neither.
// IsEitherNaN(a,b): true iff a or b is NaN (the "unordered" predicate). Float lanes only.
// IsNaN/IsEitherNaN use an unordered self/pair-compare (_CMP_UNORD_Q -> all-ones exactly
// where NaN). IsInf and IsFinite reuse Abs + Eq/Lt against a +inf splat: |v|==inf isolates
// inf, |v|<inf (ordered) keeps only finite (NaN and inf both fail), matching scalar isinf/
// isfinite bit-for-bit.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask256<T, N> IsNaN(Vec256<T, N> v) noexcept {
  static_assert(std::is_floating_point_v<T>, "IsNaN is float-only");
  Mask256<T, N> m;
  if constexpr (std::is_same_v<T, float>) m.raw = _mm256_cmp_ps(v.raw, v.raw, _CMP_UNORD_Q);
  else m.raw = _mm256_cmp_pd(v.raw, v.raw, _CMP_UNORD_Q);
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask256<T, N> IsEitherNaN(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  static_assert(std::is_floating_point_v<T>, "IsEitherNaN is float-only");
  Mask256<T, N> m;
  if constexpr (std::is_same_v<T, float>) m.raw = _mm256_cmp_ps(a.raw, b.raw, _CMP_UNORD_Q);
  else m.raw = _mm256_cmp_pd(a.raw, b.raw, _CMP_UNORD_Q);
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask256<T, N> IsInf(Vec256<T, N> v) noexcept {
  static_assert(std::is_floating_point_v<T>, "IsInf is float-only");
  const auto inf = Set(Simd<T, N, 0>(), std::numeric_limits<T>::infinity());
  return Eq(Abs(v), inf);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask256<T, N> IsFinite(Vec256<T, N> v) noexcept {
  static_assert(std::is_floating_point_v<T>, "IsFinite is float-only");
  const auto inf = Set(Simd<T, N, 0>(), std::numeric_limits<T>::infinity());
  return Lt(Abs(v), inf);  // ordered <: NaN and inf both fail, only finite lanes true
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
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> IfThenZeroElse(Mask256<T, N> m, Vec256<T, N> no) noexcept {
  return IfThenElse(m, Zero(Simd<T, N, 0>()), no);
}

// ---- mask-domain logic (bitwise on the all-ones/all-zero lane mask) -----------------
// avx2 masks are lane vectors (compare results: 0xFF..pattern), so mask And/Or/Xor/AndNot/
// Not are the integer bitwise ops on the mask raw. Float/double mask raw is __m256/__m256d;
// route through si256 so one path serves all lane types. all-ones = set1_epi32(-1) for Not.
namespace detail {
template <typename T> HYLE_FORCE_INLINE __m256i mask_to_i(typename Raw256<T>::type m) noexcept {
  if constexpr (std::is_same_v<T, float>) return _mm256_castps_si256(m);
  else if constexpr (std::is_same_v<T, double>) return _mm256_castpd_si256(m);
  else return m;
}
template <typename T> HYLE_FORCE_INLINE typename Raw256<T>::type mask_from_i(__m256i m) noexcept {
  if constexpr (std::is_same_v<T, float>) return _mm256_castsi256_ps(m);
  else if constexpr (std::is_same_v<T, double>) return _mm256_castsi256_pd(m);
  else return m;
}
}  // namespace detail
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask256<T, N> And(Mask256<T, N> a, Mask256<T, N> b) noexcept {
  Mask256<T, N> m; m.raw = detail::mask_from_i<T>(_mm256_and_si256(detail::mask_to_i<T>(a.raw), detail::mask_to_i<T>(b.raw))); return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask256<T, N> Or(Mask256<T, N> a, Mask256<T, N> b) noexcept {
  Mask256<T, N> m; m.raw = detail::mask_from_i<T>(_mm256_or_si256(detail::mask_to_i<T>(a.raw), detail::mask_to_i<T>(b.raw))); return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask256<T, N> Xor(Mask256<T, N> a, Mask256<T, N> b) noexcept {
  Mask256<T, N> m; m.raw = detail::mask_from_i<T>(_mm256_xor_si256(detail::mask_to_i<T>(a.raw), detail::mask_to_i<T>(b.raw))); return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask256<T, N> AndNot(Mask256<T, N> a, Mask256<T, N> b) noexcept {
  // ~a & b (Highway order): _mm256_andnot(a,b) computes (~a)&b.
  Mask256<T, N> m; m.raw = detail::mask_from_i<T>(_mm256_andnot_si256(detail::mask_to_i<T>(a.raw), detail::mask_to_i<T>(b.raw))); return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask256<T, N> Not(Mask256<T, N> a) noexcept {
  const __m256i ones = _mm256_set1_epi32(-1);
  Mask256<T, N> m; m.raw = detail::mask_from_i<T>(_mm256_xor_si256(detail::mask_to_i<T>(a.raw), ones)); return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask256<T, N> ExclusiveNeither(Mask256<T, N> a, Mask256<T, N> b) noexcept {
  return Not(Or(a, b));  // ~(a|b)
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
template <typename T, std::size_t N>
HYLE_FORCE_INLINE std::ptrdiff_t FindLastTrue(Simd<T, N, 0>, Mask256<T, N> m) noexcept {
  const unsigned b = detail::mask_bits(m);
  // highest set lane index = (bit-width-1) - clz; mask_bits packs 1 bit/lane (LSB=lane0).
  return b ? static_cast<std::ptrdiff_t>(31 - std::countl_zero(b)) : -1;
}
// VecFromMask(d, m): the mask's per-lane all-ones/all-zero bit pattern AS a vector. On AVX2
// the Mask256 raw IS that pattern already (compares yield full-lane 0xFF..), so it's a
// reinterpret for int lanes / a cast for float. Feeds SumsOf8 for a masked byte count.
template <class D>
HYLE_FORCE_INLINE detail::V256<D> VecFromMask(D, Mask256<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m) noexcept {
  detail::V256<D> r;
  // Mask256<T>::raw is Raw256T<T>: float/double masks are ALREADY __m256/__m256d (FP-domain
  // compares + MaskFromVec produce them), int masks are the __m256i 0xFF../0 vector -- either
  // way it's the identity, no cast (a cast would mistype the FP path).
  r.raw = m.raw;
  return r;
}
// MaskFromVec: lane true iff its MSB set (Highway). The inverse of VecFromMask. Build a
// proper all-ones/all-zero lane vector (what the other mask ops expect) by arithmetic sign
// broadcast: srai for i16/i32, cmpgt(0,v) for i8, and a 32->64 sign broadcast for 64-bit.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask256<T, N> MaskFromVec(Vec256<T, N> v) noexcept {
  Mask256<T, N> m;
  if constexpr (std::is_same_v<T, float>)
    m.raw = _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_castps_si256(v.raw), 31));
  else if constexpr (std::is_same_v<T, double>) {
    __m256i hi = _mm256_srai_epi32(_mm256_castpd_si256(v.raw), 31);          // sign to all 32 hi bits
    m.raw = _mm256_castsi256_pd(_mm256_shuffle_epi32(hi, _MM_SHUFFLE(3, 3, 1, 1)));  // broadcast to 64b
  } else if constexpr (sizeof(T) == 8) {
    __m256i hi = _mm256_srai_epi32(v.raw, 31);
    m.raw = _mm256_shuffle_epi32(hi, _MM_SHUFFLE(3, 3, 1, 1));
  } else if constexpr (sizeof(T) == 4) m.raw = _mm256_srai_epi32(v.raw, 31);
  else if constexpr (sizeof(T) == 2) m.raw = _mm256_srai_epi16(v.raw, 15);
  else m.raw = _mm256_cmpgt_epi8(_mm256_setzero_si256(), v.raw);  // i8: 0 > v  <=> v's sign set
  return m;
}

// mask <-> bits (reuse the mask_bits movemask helper, 1 bit/lane).
template <class D>
HYLE_FORCE_INLINE std::uint64_t BitsFromMask(D, Mask256<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m) noexcept {
  return detail::mask_bits(m);
}
template <class D> requires HYLE_SIMD_W(D, 16, 32)
HYLE_FORCE_INLINE Mask256<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> LoadMaskBits(D d, std::uint64_t bits) noexcept {
  // Rebuild: lane i true iff bit i set, via a stack array of all-ones/all-zero lanes.
  using T = TFromD<D>;
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  alignas(32) T tmp[kN];
  for (std::size_t i = 0; i < kN; ++i) {
    using U = std::make_unsigned_t<std::conditional_t<std::is_integral_v<T>, T, std::uint64_t>>;
    U u = ((bits >> i) & 1) ? ~U{0} : U{0};
    std::memcpy(&tmp[i], &u, sizeof(T));
  }
  Mask256<T, kN> out;
  out.raw = detail::mask_from_i<T>(_mm256_load_si256(reinterpret_cast<const __m256i*>(tmp)));
  return out;
}
template <class D>
HYLE_FORCE_INLINE std::size_t StoreMaskBits(D d, Mask256<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m,
                                            std::uint8_t* p) noexcept {
  const std::uint64_t bits = detail::mask_bits(m);
  const std::size_t nbytes = (HYLE_SIMD_MAX_LANES_D(D) + 7) / 8;
  for (std::size_t b = 0; b < nbytes; ++b) p[b] = static_cast<std::uint8_t>(bits >> (b * 8));
  return nbytes;
}

// prefix-scan masks (leading run relative to first true) -- computed on the bitfield then
// rebuilt via LoadMaskBits. N may be 32 (i8) so the bitfield is 32-bit; the "full" mask uses
// the (N>=32 ? 0xFFFFFFFFu : (1u<<N)-1u) guard to avoid the UB of (1u<<32).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask256<T, N> SetBeforeFirst(Mask256<T, N> m) noexcept {
  const unsigned b = detail::mask_bits(m);
  const unsigned before = b ? ((1u << std::countr_zero(b)) - 1u) : ((N >= 32 ? 0xFFFFFFFFu : (1u << N) - 1u));
  return LoadMaskBits(Simd<T, N, 0>(), before);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask256<T, N> SetAtOrBeforeFirst(Mask256<T, N> m) noexcept {
  const unsigned b = detail::mask_bits(m);
  // countr_zero(b)+1 can reach 32 when the first true is lane 31 (N=32, i8/u8): 1u<<32 is shift
  // UB (x86 masks it to 1u<<0 -> wrong result). Saturate to the all-ones mask in that case.
  const unsigned tz1 = b ? (std::countr_zero(b) + 1) : 0;
  const unsigned upto = b ? (tz1 >= 32 ? 0xFFFFFFFFu : (1u << tz1) - 1u)
                          : ((N >= 32 ? 0xFFFFFFFFu : (1u << N) - 1u));
  return LoadMaskBits(Simd<T, N, 0>(), upto);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask256<T, N> SetOnlyFirst(Mask256<T, N> m) noexcept {
  const unsigned b = detail::mask_bits(m);
  return LoadMaskBits(Simd<T, N, 0>(), b ? (1u << std::countr_zero(b)) : 0u);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask256<T, N> SetAtOrAfterFirst(Mask256<T, N> m) noexcept {
  const unsigned b = detail::mask_bits(m);
  const unsigned full = (N >= 32) ? 0xFFFFFFFFu : ((1u << N) - 1u);
  const unsigned before = b ? ((1u << std::countr_zero(b)) - 1u) : full;
  return LoadMaskBits(Simd<T, N, 0>(), full & ~before);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE std::size_t FindKnownFirstTrue(Simd<T, N, 0>, Mask256<T, N> m) noexcept {
  return static_cast<std::size_t>(std::countr_zero(detail::mask_bits(m)));
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE std::size_t FindKnownLastTrue(Simd<T, N, 0>, Mask256<T, N> m) noexcept {
  return static_cast<std::size_t>(31 - std::countl_zero(detail::mask_bits(m)));
}

// SumsOf8(v): sum each group of 8 u8 lanes into a u64 lane -- native psadbw against zero
// (v vs 0 = sum of absolute bytes = sum of bytes for u8). 32 u8 lanes -> 4 u64 lanes.
template <std::size_t N>
HYLE_FORCE_INLINE Vec256<std::uint64_t, N / 8> SumsOf8(Vec256<std::uint8_t, N> v) noexcept {
  static_assert(N == 32, "avx2 SumsOf8: full 32xu8 -> 4xu64");
  Vec256<std::uint64_t, N / 8> r;
  r.raw = _mm256_sad_epu8(v.raw, _mm256_setzero_si256());
  return r;
}
// simd-17 batch5: SumsOf8AbsDiff(a,b) = native _mm256_sad_epu8.
template <std::size_t N>
HYLE_FORCE_INLINE Vec256<std::uint64_t, N / 8> SumsOf8AbsDiff(Vec256<std::uint8_t, N> a,
                                                              Vec256<std::uint8_t, N> b) noexcept {
  static_assert(N == 32, "avx2 SumsOf8AbsDiff: full 32xu8 -> 4xu64");
  Vec256<std::uint64_t, N / 8> r;
  r.raw = _mm256_sad_epu8(a.raw, b.raw);
  return r;
}
template <std::size_t N>
HYLE_FORCE_INLINE Vec256<std::uint64_t, N / 8> SumsOf8AbsDiff(Vec256<std::uint8_t, N> v) noexcept {
  return SumsOf8(v);
}
// simd-17 batch5: BitShuffle / MultiRotateRight -- portable store+reload (matches scalar oracle
// + Highway's generic contract). Native VPSHUFBITQMB / VPMULTISHIFTQB live on the AVX3_DL avx512
// path only; the avx2 backend uses the correctness-first gather (a profiling upgrade could add
// GFNI/BITALG native here, but those need ISA flags this variant does not carry).
template <typename T, std::size_t N, typename TI, std::size_t NI>
HYLE_FORCE_INLINE Vec256<T, N> BitShuffle(Vec256<T, N> v, Vec256<TI, NI> idx) noexcept {
  static_assert(sizeof(T) == 8, "BitShuffle data lanes are 64-bit");
  static_assert(sizeof(TI) == 1 && NI == N * 8, "BitShuffle index is u8, 8 per data lane");
  alignas(32) std::uint64_t sv[N];
  alignas(32) std::uint8_t si[NI];
  Store(v, Simd<T, N, 0>(), reinterpret_cast<T*>(sv));
  Store(idx, Simd<TI, NI, 0>(), reinterpret_cast<TI*>(si));
  alignas(32) std::uint64_t out[N];
  for (std::size_t q = 0; q < N; ++q) {
    std::uint64_t o = 0;
    for (std::size_t j = 0; j < 8; ++j) {
      const unsigned bit = static_cast<unsigned>(si[q * 8 + j]) & 63u;
      o |= ((sv[q] >> bit) & 1u) << j;
    }
    out[q] = o;
  }
  return Load(Simd<T, N, 0>(), reinterpret_cast<const T*>(out));
}
template <typename T, std::size_t N, typename TI, std::size_t NI>
HYLE_FORCE_INLINE Vec256<T, N> MultiRotateRight(Vec256<T, N> v, Vec256<TI, NI> idx) noexcept {
  static_assert(sizeof(T) == 8, "MultiRotateRight data lanes are 64-bit");
  static_assert(sizeof(TI) == 1 && NI == N * 8, "MultiRotateRight index is u8, 8 per data lane");
  alignas(32) std::uint64_t sv[N];
  alignas(32) std::uint8_t si[NI];
  Store(v, Simd<T, N, 0>(), reinterpret_cast<T*>(sv));
  Store(idx, Simd<TI, NI, 0>(), reinterpret_cast<TI*>(si));
  alignas(32) std::uint64_t out[N];
  for (std::size_t q = 0; q < N; ++q) {
    std::uint64_t o = 0;
    for (std::size_t j = 0; j < 8; ++j) {
      const unsigned rot = static_cast<unsigned>(si[q * 8 + j]) & 63u;
      const std::uint64_t rr = rot == 0 ? sv[q] : ((sv[q] >> rot) | (sv[q] << (64 - rot)));
      o |= (rr & 0xFFu) << (j * 8);
    }
    out[q] = o;
  }
  return Load(Simd<T, N, 0>(), reinterpret_cast<const T*>(out));
}



// SumsOf2/SumsOf4: sum consecutive pairs/quads of u8 lanes, widening to u16/u32.
// correctness-first store+reload; native (pmaddubsw with 1s / vpaddl chains) is a profiling upgrade.
// scratch sized to the FULL 32 B register (not the logical lane count): _mm256_store/load_si256
// move 32 B, so a partial-width src would be a stack OOB with src[N] -- same class as TruncateTo.
template <std::size_t N>
HYLE_FORCE_INLINE Vec256<std::uint16_t, N / 2> SumsOf2(Vec256<std::uint8_t, N> v) noexcept {
  static_assert(N % 2 == 0, "avx2 SumsOf2 needs an even lane count");
  alignas(32) std::uint8_t src[32];
  alignas(32) std::uint16_t dst[32 / sizeof(std::uint16_t)] = {};
  _mm256_store_si256(reinterpret_cast<__m256i*>(src), v.raw);
  for (std::size_t g = 0; g < N / 2; ++g)
    dst[g] = static_cast<std::uint16_t>(src[2 * g] + src[2 * g + 1]);
  return Load(Simd<std::uint16_t, N / 2, 0>(), dst);
}
template <std::size_t N>
HYLE_FORCE_INLINE Vec256<std::uint32_t, N / 4> SumsOf4(Vec256<std::uint8_t, N> v) noexcept {
  static_assert(N % 4 == 0, "avx2 SumsOf4 needs a lane count that is a multiple of 4");
  alignas(32) std::uint8_t src[32];
  alignas(32) std::uint32_t dst[32 / sizeof(std::uint32_t)] = {};
  _mm256_store_si256(reinterpret_cast<__m256i*>(src), v.raw);
  for (std::size_t g = 0; g < N / 4; ++g)
    dst[g] = static_cast<std::uint32_t>(src[4 * g] + src[4 * g + 1] + src[4 * g + 2] + src[4 * g + 3]);
  return Load(Simd<std::uint32_t, N / 4, 0>(), dst);
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
// RedOp defined in x86_detail.h (shared across layered x86 backends).

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
// RedOp2 defined in x86_detail.h (shared across layered x86 backends).
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

// ---- swizzle: B-2 completions (Reverse4/8, ReverseBits/LaneBytes, InterleaveEven/Odd,
// Slide1Up/Down, SlideUp/DownLanes, BroadcastLane, Zip). ------------------------------
// correctness-first store+reload; native shuffles are a profiling upgrade.
template <class D>
HYLE_FORCE_INLINE detail::V256<D> Reverse4(D, detail::V256<D> v) noexcept {
  return detail::swizzle_via_mem(v, [](std::size_t i) { return (i & ~std::size_t{3}) | ((i & 3) ^ 3); });
}
template <class D>
HYLE_FORCE_INLINE detail::V256<D> Reverse8(D, detail::V256<D> v) noexcept {
  return detail::swizzle_via_mem(v, [](std::size_t i) { return (i & ~std::size_t{7}) | ((i & 7) ^ 7); });
}
// ReverseBits: reverse the bit order within each lane. NATIVE (Highway method): (1) reverse bits
// WITHIN each byte via two vpshufb nibble-LUTs (low nibble reversed into high, high into low),
// (2) reverse the BYTE order within each lane via a per-width vpshufb index. Replaces the O(bits)
// scalar loop (op_parity: 9x slower). 8/16/32/64-bit lanes.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> ReverseBits(Vec256<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "ReverseBits is integer-only");
  // per-nibble bit-reversal LUT: reverse(0000)=0000, reverse(0001)=1000=8, ... (4-bit mirror)
  const __m256i lut = _mm256_setr_epi8(
      0x0,0x8,0x4,0xC,0x2,0xA,0x6,0xE,0x1,0x9,0x5,0xD,0x3,0xB,0x7,0xF,
      0x0,0x8,0x4,0xC,0x2,0xA,0x6,0xE,0x1,0x9,0x5,0xD,0x3,0xB,0x7,0xF);
  const __m256i lo_mask = _mm256_set1_epi8(0x0F);
  const __m256i lo = _mm256_and_si256(v.raw, lo_mask);
  const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v.raw, 4), lo_mask);
  // byte with bits reversed: reversed-low-nibble goes to high half, reversed-high to low half
  __m256i rb = _mm256_or_si256(_mm256_slli_epi16(_mm256_shuffle_epi8(lut, lo), 4),
                               _mm256_shuffle_epi8(lut, hi));
  Vec256<T, N> r;
  if constexpr (sizeof(T) == 1) { r.raw = rb; return r; }
  // reverse byte order within each lane via a per-128-block vpshufb index (mirrors ReverseLaneBytes)
  __m256i idx;
  if constexpr (sizeof(T) == 2)
    idx = _mm256_setr_epi8(1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14, 1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14);
  else if constexpr (sizeof(T) == 4)
    idx = _mm256_setr_epi8(3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12, 3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12);
  else
    idx = _mm256_setr_epi8(7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8, 7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8);
  r.raw = _mm256_shuffle_epi8(rb, idx);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> ReverseLaneBytes(Vec256<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "ReverseLaneBytes is integer-only");
  alignas(32) T s[N], d[N];
  Store(v, Simd<T, N, 0>(), s);
  for (std::size_t i = 0; i < N; ++i) {
    auto* db = reinterpret_cast<std::uint8_t*>(&d[i]);
    const auto* sb = reinterpret_cast<const std::uint8_t*>(&s[i]);
    for (std::size_t b = 0; b < sizeof(T); ++b) db[b] = sb[sizeof(T) - 1 - b];
  }
  return Load(Simd<T, N, 0>(), d);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> InterleaveEven(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  alignas(32) T sa[N], sb[N], d[N];
  Store(a, Simd<T, N, 0>(), sa); Store(b, Simd<T, N, 0>(), sb);
  for (std::size_t i = 0; i < N / 2; ++i) { d[2 * i] = sa[2 * i]; d[2 * i + 1] = sb[2 * i]; }
  return Load(Simd<T, N, 0>(), d);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> InterleaveOdd(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  alignas(32) T sa[N], sb[N], d[N];
  Store(a, Simd<T, N, 0>(), sa); Store(b, Simd<T, N, 0>(), sb);
  for (std::size_t i = 0; i < N / 2; ++i) { d[2 * i] = sa[2 * i + 1]; d[2 * i + 1] = sb[2 * i + 1]; }
  return Load(Simd<T, N, 0>(), d);
}
template <class D>
HYLE_FORCE_INLINE detail::V256<D> Slide1Up(D d, detail::V256<D> v) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(32) T s[N]; alignas(32) T dst[N];
  Store(v, d, s);
  dst[0] = T{0};
  for (std::size_t i = 1; i < N; ++i) dst[i] = s[i - 1];
  return Load(d, dst);
}
template <class D>
HYLE_FORCE_INLINE detail::V256<D> Slide1Down(D d, detail::V256<D> v) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(32) T s[N]; alignas(32) T dst[N];
  Store(v, d, s);
  dst[N - 1] = T{0};
  for (std::size_t i = 0; i + 1 < N; ++i) dst[i] = s[i + 1];
  return Load(d, dst);
}
template <class D>
HYLE_FORCE_INLINE detail::V256<D> SlideUpLanes(D d, detail::V256<D> v, std::size_t amt) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(32) T s[N]; alignas(32) T dst[N] = {};
  Store(v, d, s);
  for (std::size_t i = amt; i < N; ++i) dst[i] = s[i - amt];
  return Load(d, dst);
}
template <class D>
HYLE_FORCE_INLINE detail::V256<D> SlideDownLanes(D d, detail::V256<D> v, std::size_t amt) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(32) T s[N]; alignas(32) T dst[N] = {};
  Store(v, d, s);
  for (std::size_t i = 0; i + amt < N; ++i) dst[i] = s[i + amt];
  return Load(d, dst);
}
template <int kLane, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> BroadcastLane(Vec256<T, N> v) noexcept { return Broadcast<kLane>(v); }
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> ZipLower(Vec256<T, N> a, Vec256<T, N> b) noexcept { return InterleaveLower(a, b); }
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> ZipUpper(Vec256<T, N> a, Vec256<T, N> b) noexcept { return InterleaveUpper(a, b); }

// ---- masked / partial load-store (tail handling) ------------------------------------
// FirstN(d,n) -> mask with the first min(n,N) lanes true. An all-ones lane bit pattern is
// a valid AVX2 blend/movemask selector for EVERY lane type (blendv keys off each lane's
// MSB, all-ones has it set), so we build it via a stack array + Load, uniform over T.
// LoadN/StoreN touch ONLY the first k lanes (fault-safe tail I/O); MaskedLoad mirrors
// Highway's AVX2 narrow path (IfThenElseZero + full LoadU -- caller ensures the vector is
// loadable), BlendedStore writes only on-lanes (fault-safe, mask-driven).
template <class D> requires HYLE_SIMD_W(D, 16, 32)
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
template <class D> requires HYLE_SIMD_W(D, 16, 32)
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
// NATIVE where AVX2 has it (vpgatherd for 32-bit lane+index -- the common i32/u32/f32 case),
// store+reload otherwise (8/16/64-bit, mismatched index width -- correct + oracle-matched).
// NOTE (measured, not assumed): AVX2 vpgather is famously slow on pre-Skylake uarchs -- the
// native path is NOT unconditionally faster. gather_bench measures store+reload vs native at
// the same ISA; the native path ships because it wins or ties on this dev box (Zen/Skylake+),
// but the store+reload fallback stays the correctness reference. AVX-512 gather/compress have
// real hardware (vpcompressd) and win more clearly.
template <class D, typename IdxT>
HYLE_FORCE_INLINE detail::V256<D> GatherIndex(D d, const TFromD<D>* base,
                                              Vec256<IdxT, HYLE_SIMD_MAX_LANES_D(D)> idx) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  if constexpr (sizeof(T) == 4 && sizeof(IdxT) == 4) {
    // vpgatherdd/dps: 8 lanes, scale 4 (index is element count -> *4 bytes). Signed i32 index.
    if constexpr (std::is_same_v<T, float>)
      return detail::V256<D>{_mm256_i32gather_ps(base, idx.raw, 4)};
    else {
      detail::V256<D> r;
      r.raw = _mm256_i32gather_epi32(reinterpret_cast<const int*>(base), idx.raw, 4);
      return r;
    }
  } else {
    alignas(32) IdxT ix[kN]; alignas(32) T dst[kN];
    Store(idx, Simd<IdxT, kN, 0>(), ix);
    for (std::size_t i = 0; i < kN; ++i) dst[i] = base[static_cast<std::ptrdiff_t>(ix[i])];
    return Load(d, dst);
  }
}
// Store+reload reference kept accessible for the bench (native-vs-emulation A/B) + as the
// fallback body above. Same result, different codegen -- gather_bench races the two.
template <class D, typename IdxT>
HYLE_FORCE_INLINE detail::V256<D> GatherIndexEmulated(D d, const TFromD<D>* base,
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
// Compress index table for 8x i32: for each 8-bit mask, the 8 vpermd indices that gather the
// active lanes to the front. PACKED to 1 KiB (Highway's method): each entry is a u32 holding the
// eight 3-bit(-ish) indices in nibbles [0:4,4:8,...,28:32]; broadcast + per-lane shift by
// {0,4,...,28} + vpermd (which ignores bits>2). 256*4 B = 1 KiB (vs the naive 256*8 u32 = 8 KiB,
// which cost ~2.4x from L1 pressure). constexpr-built (no hand-typed table).
namespace detail {
struct CompressIdx32Packed {
  std::uint32_t packed[256] = {};
  constexpr CompressIdx32Packed() {
    for (unsigned bits = 0; bits < 256; ++bits) {
      unsigned k = 0;
      std::uint32_t word = 0;
      for (unsigned i = 0; i < 8; ++i)
        if ((bits >> i) & 1u) { word |= static_cast<std::uint32_t>(i) << (4 * k); ++k; }
      // inactive tail nibbles stay 0 -> lane 0 (harmless: the tail is unspecified per contract).
      packed[bits] = word;
    }
  }
};
inline constexpr CompressIdx32Packed kCompressIdx32Tbl{};
inline constexpr const std::uint32_t* kCompressIdx32Packed = kCompressIdx32Tbl.packed;
}  // namespace detail
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> Compress(Vec256<T, N> v, Mask256<T, N> m) noexcept {
  // NATIVE for 32-bit lanes (8 lanes): a per-mask permute-index table + vpermd packs the active
  // lanes to the front in ONE shuffle (Highway's method; op_parity: store+reload was 6x slower).
  // 1 KiB packed table (each u32 holds the 8 indices as nibbles) + srlv-expand. CONTRACT: only the
  // first CountTrue(m) lanes are defined; the tail is UNSPECIFIED (matches Highway/xsimd -- so no
  // zero-tail mask, which cost ~2x on AVX2 where there is no zeroing-compress instruction). Safe
  // use = CompressStore (writes exactly CountTrue) or read only the first CountTrue lanes.
  // Other widths keep store+reload (no 8/16/64-bit vpermd on AVX2); that path happens to zero its
  // tail (a store+reload byproduct, still within the "unspecified" contract).
  if constexpr (sizeof(T) == 4) {
    // 8-bit mask straight from movemask_ps (one instruction) -- NOT detail::mask_bits, whose i32
    // path scalar-compacts movemask_epi8's 32 bits (an ~8-iteration loop that dominated the op).
    unsigned mbits;
    if constexpr (std::is_same_v<T, float>) mbits = static_cast<unsigned>(_mm256_movemask_ps(m.raw));
    else mbits = static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(m.raw)));
    // broadcast the packed u32, then shift each of the 8 copies right by {0,4,...,28} so lane i
    // holds its 3-bit index in the low bits (vpermd ignores bits 3..31).
    const __m256i packed = _mm256_set1_epi32(static_cast<int>(detail::kCompressIdx32Packed[mbits]));
    const __m256i shifts = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
    const __m256i idx = _mm256_srlv_epi32(packed, shifts);
    Vec256<T, N> r;
    if constexpr (std::is_same_v<T, float>) r.raw = _mm256_permutevar8x32_ps(v.raw, idx);
    else r.raw = _mm256_permutevar8x32_epi32(v.raw, idx);
    return r;
  } else {
    alignas(32) T src[N], dst[N] = {};
    Store(v, Simd<T, N, 0>(), src);
    const unsigned bits = detail::mask_bits(m);
    std::size_t k = 0;
    for (std::size_t i = 0; i < N; ++i) if ((bits >> i) & 1u) dst[k++] = src[i];
    return Load(Simd<T, N, 0>(), dst);
  }
}
// AVX2 has no native compress (VBMI2's vpcompress is AVX-512 only), so Compress IS the store+
// reload form -- CompressEmulated aliases it so gather_bench compiles the same A/B on both x86
// tiers (on avx2 the two rows are identical; the meaningful compress A/B is the avx512 exe).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> CompressEmulated(Vec256<T, N> v, Mask256<T, N> m) noexcept {
  return Compress(v, m);
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
namespace detail {
// Native byte-wise popcount via the vpshufb nibble-LUT (Highway/Muła's method): a 16-entry table
// maps each nibble to its bit count; shuffle it twice (low + high nibble) and add. Returns the
// per-BYTE popcount (0..8) in each byte lane. The per-lane-width reducers below sum these bytes.
HYLE_FORCE_INLINE __m256i popcnt_bytes_avx2(__m256i v) noexcept {
  const __m256i lut = _mm256_setr_epi8(
      0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4, 0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
  const __m256i lo_mask = _mm256_set1_epi8(0x0F);
  const __m256i lo = _mm256_and_si256(v, lo_mask);
  const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), lo_mask);
  return _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo), _mm256_shuffle_epi8(lut, hi));
}
}  // namespace detail
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> PopulationCount(Vec256<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "PopulationCount is integer-lane only");
  Vec256<T, N> r;
  const __m256i bytes = detail::popcnt_bytes_avx2(v.raw);  // per-byte counts
  if constexpr (sizeof(T) == 1) {
    r.raw = bytes;
  } else if constexpr (sizeof(T) == 8) {
    // sum 8 bytes per 64-bit lane: vpsadbw against zero gives the byte-sum in each 64-bit lane.
    r.raw = _mm256_sad_epu8(bytes, _mm256_setzero_si256());
  } else if constexpr (sizeof(T) == 4) {
    // sum 4 bytes per 32-bit lane: sad gives per-64 sums of 8 bytes; instead fold within 32b.
    const __m256i s16 = _mm256_maddubs_epi16(bytes, _mm256_set1_epi8(1));   // pairs of bytes -> u16
    const __m256i s32 = _mm256_madd_epi16(s16, _mm256_set1_epi16(1));       // pairs of u16 -> u32
    r.raw = s32;
  } else {  // 16-bit: fold byte pairs into each 16-bit lane
    r.raw = _mm256_maddubs_epi16(bytes, _mm256_set1_epi8(1));
  }
  return r;
}
// LeadingZeroCount / TrailingZeroCount: NATIVE via the fast PopulationCount above (op_parity: the
// old std::count[lr]_zero store+reload was 5-8x slower). Identities (per unsigned lane):
//   TZCnt(x) = PopCount((x & -x) - 1)                 [x==0 -> all-ones -> PopCount = bits]
//   LZCnt(x) = bits - PopCount(smear_right(x))        [smear MSB down so set bits = MSB..0]
// Needs per-width shifts; 8-bit lanes have no _mm256_srli_epi8, so they keep the mem fallback.
namespace detail {
template <typename T> HYLE_FORCE_INLINE __m256i srli_w(__m256i x, int s) noexcept {
  if constexpr (sizeof(T) == 2) return _mm256_srli_epi16(x, s);
  else if constexpr (sizeof(T) == 4) return _mm256_srli_epi32(x, s);
  else return _mm256_srli_epi64(x, s);
}
// float-exponent trick (Highway generic_ops UIntToF32BiasedExp): the biased exponent of the f32
// nearest to a u32 gives its bit-length, so LZCnt/TZCnt fall out of a single int->float convert +
// exponent extract -- far cheaper than the smear+popcount chain. For 32-bit lanes: convert via
// SIGNED cvtepi32_ps (1 instr; the top-bit-set case is folded by the internal min-with-158 that
// the LZCnt formula's overall Min(.,32) subsumes), bitcast, >>23. Returns the biased exponent.
HYLE_FORCE_INLINE __m256i u32_to_f32_biased_exp(__m256i v) noexcept {
  // zero out bit (HighestSetBit-24) for v>=2^24 so the inexact convert rounds DOWN (never bumps
  // the exponent up a notch): v & ~(v>>24). Exact for v<2^24 (no-op there).
  const __m256i norm = _mm256_andnot_si256(_mm256_srli_epi32(v, 24), v);
  const __m256 f = _mm256_cvtepi32_ps(norm);  // signed convert; norm<2^31 always (top 8 bits clear)
  const __m256i bits = _mm256_castps_si256(f);
  return _mm256_srli_epi32(bits, 23);  // biased exponent in [0,158]
}
}  // namespace detail
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> LeadingZeroCount(Vec256<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "LeadingZeroCount is integer-lane only");
  if constexpr (sizeof(T) == 4) {
    // LZCnt(x) = min( (32+126) - biased_exp(x), 32 ). biased_exp(0)=0 -> 158-0=158, clamped to 32.
    const __m256i be = detail::u32_to_f32_biased_exp(v.raw);
    const __m256i lz = _mm256_sub_epi32(_mm256_set1_epi32(32 + 126), be);
    Vec256<T, N> r; r.raw = _mm256_min_epu32(lz, _mm256_set1_epi32(32));
    return r;
  } else if constexpr (sizeof(T) == 1) {
    return detail::bitcount_via_mem<T, N, 1>(v);
  } else {  // 16/64-bit: smear the top set bit down, then bits - popcount(smeared)
    __m256i x = v.raw;
    x = _mm256_or_si256(x, detail::srli_w<T>(x, 1));
    x = _mm256_or_si256(x, detail::srli_w<T>(x, 2));
    x = _mm256_or_si256(x, detail::srli_w<T>(x, 4));
    x = _mm256_or_si256(x, detail::srli_w<T>(x, 8));
    if constexpr (sizeof(T) == 8) { x = _mm256_or_si256(x, detail::srli_w<T>(x, 16)); x = _mm256_or_si256(x, detail::srli_w<T>(x, 32)); }
    Vec256<T, N> sm; sm.raw = x;
    const Vec256<T, N> pc = PopulationCount(sm);
    return Sub(Set(Simd<T, N, 0>(), static_cast<T>(sizeof(T) * 8)), pc);
  }
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> TrailingZeroCount(Vec256<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "TrailingZeroCount is integer-lane only");
  if constexpr (sizeof(T) == 4) {
    // TZCnt(x) = min( biased_exp(x & -x) - 127, 32 ). x&-x isolates the lowest set bit; its f32
    // exponent is exactly its bit index. x==0 -> lowest=0 -> exp=0 -> 0-127 underflows huge ->
    // unsigned-min with 32 gives 32 (= bit width). Correct for all inputs.
    const __m256i neg = _mm256_sub_epi32(_mm256_setzero_si256(), v.raw);
    const __m256i lowest = _mm256_and_si256(v.raw, neg);
    const __m256i be = detail::u32_to_f32_biased_exp(lowest);
    const __m256i tz = _mm256_sub_epi32(be, _mm256_set1_epi32(127));
    Vec256<T, N> r; r.raw = _mm256_min_epu32(tz, _mm256_set1_epi32(32));
    return r;
  } else if constexpr (sizeof(T) == 1) {
    return detail::bitcount_via_mem<T, N, 2>(v);
  } else {  // 16/64-bit: PopCount((x & -x) - 1)
    const Vec256<T, N> neg = Neg(v);
    Vec256<T, N> lowest; lowest.raw = _mm256_and_si256(v.raw, neg.raw);
    const Vec256<T, N> mask = Sub(lowest, Set(Simd<T, N, 0>(), static_cast<T>(1)));
    return PopulationCount(mask);  // x==0: lowest=0, mask=all-ones -> PopCount = bits (correct)
  }
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


// ---- simd-17 batch2: runtime *Same + Pairwise + Stream + LoadDup128 + exp domain ----
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> ShiftLeftSame(Vec256<T, N> v, int bits) noexcept {
  static_assert(std::is_integral_v<T>, "ShiftLeftSame is integer-lane only");
  Vec256<T, N> r;
  const __m128i c = _mm_cvtsi32_si128(bits);
  if constexpr (detail::kI16<T>) r.raw = _mm256_sll_epi16(v.raw, c);
  else if constexpr (detail::kI32<T>) r.raw = _mm256_sll_epi32(v.raw, c);
  else if constexpr (detail::kI64<T>) r.raw = _mm256_sll_epi64(v.raw, c);
  else {
    const __m256i sh = _mm256_sll_epi16(v.raw, c);
    const unsigned char m = static_cast<unsigned char>(0xFFu << (bits & 7));
    r.raw = (bits >= 8) ? _mm256_setzero_si256() : _mm256_and_si256(sh, _mm256_set1_epi8(static_cast<char>(m)));
  }
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> ShiftRightSame(Vec256<T, N> v, int bits) noexcept {
  static_assert(std::is_integral_v<T>, "ShiftRightSame is integer-lane only");
  Vec256<T, N> r;
  const __m128i c = _mm_cvtsi32_si128(bits);
  if constexpr (detail::kI16<T>) r.raw = detail::kSigned<T> ? _mm256_sra_epi16(v.raw, c) : _mm256_srl_epi16(v.raw, c);
  else if constexpr (detail::kI32<T>) r.raw = detail::kSigned<T> ? _mm256_sra_epi32(v.raw, c) : _mm256_srl_epi32(v.raw, c);
  else if constexpr (detail::kI64<T>) {
    if constexpr (detail::kSigned<T>) {
      alignas(32) T s[N]; Store(v, Simd<T, N, 0>(), s);
      for (std::size_t i = 0; i < N; ++i) s[i] = static_cast<T>(s[i] >> bits);
      return Load(Simd<T, N, 0>(), s);
    } else r.raw = _mm256_srl_epi64(v.raw, c);
  } else {
    alignas(32) T s[N]; Store(v, Simd<T, N, 0>(), s);
    for (std::size_t i = 0; i < N; ++i) s[i] = static_cast<T>(s[i] >> bits);
    return Load(Simd<T, N, 0>(), s);
  }
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> RotateRightSame(Vec256<T, N> v, int bits) noexcept {
  static_assert(std::is_integral_v<T>, "RotateRightSame is integer-lane only");
  constexpr int W = static_cast<int>(sizeof(T) * 8);
  bits &= (W - 1);
  if (bits == 0) return v;
  return Or(ShiftRightSame(v, bits), ShiftLeftSame(v, W - bits));
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> RotateLeftSame(Vec256<T, N> v, int bits) noexcept {
  constexpr int W = static_cast<int>(sizeof(T) * 8);
  bits &= (W - 1);
  if (bits == 0) return v;
  return RotateRightSame(v, W - bits);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> RoundingShiftRightSame(Vec256<T, N> v, int s) noexcept {
  static_assert(std::is_integral_v<T>, "RoundingShiftRightSame is integer-lane only");
  if (s <= 0) return v;
  return ShiftRightSame(Add(v, Set(Simd<T, N, 0>(), static_cast<T>(T{1} << (s - 1)))), s);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> RoundingShr(Vec256<T, N> v, Vec256<T, N> amounts) noexcept {
  static_assert(std::is_integral_v<T>, "RoundingShr is integer-lane only");
  alignas(32) T sv[N], sa[N], so[N];
  Store(v, Simd<T, N, 0>(), sv); Store(amounts, Simd<T, N, 0>(), sa);
  for (std::size_t i = 0; i < N; ++i) {
    const int s = static_cast<int>(sa[i]);
    if (s <= 0) so[i] = sv[i];
    else so[i] = static_cast<T>((sv[i] + static_cast<T>(T{1} << (s - 1))) >> s);
  }
  return Load(Simd<T, N, 0>(), so);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> PairwiseAdd128(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  alignas(32) T sa[N], sb[N], so[N];
  Store(a, Simd<T, N, 0>(), sa); Store(b, Simd<T, N, 0>(), sb);
  constexpr std::size_t kBL = 16 / sizeof(T);
  for (std::size_t base = 0; base < N; base += kBL) {
    const std::size_t h = kBL / 2;
    for (std::size_t i = 0; i < h; ++i) {
      so[base + i] = static_cast<T>(sa[base + 2 * i] + sa[base + 2 * i + 1]);
      so[base + h + i] = static_cast<T>(sb[base + 2 * i] + sb[base + 2 * i + 1]);
    }
  }
  return Load(Simd<T, N, 0>(), so);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> PairwiseSub128(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  alignas(32) T sa[N], sb[N], so[N];
  Store(a, Simd<T, N, 0>(), sa); Store(b, Simd<T, N, 0>(), sb);
  constexpr std::size_t kBL = 16 / sizeof(T);
  for (std::size_t base = 0; base < N; base += kBL) {
    const std::size_t h = kBL / 2;
    for (std::size_t i = 0; i < h; ++i) {
      so[base + i] = static_cast<T>(sa[base + 2 * i] - sa[base + 2 * i + 1]);
      so[base + h + i] = static_cast<T>(sb[base + 2 * i] - sb[base + 2 * i + 1]);
    }
  }
  return Load(Simd<T, N, 0>(), so);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> PairwiseAdd(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  alignas(32) T sa[N], sb[N], so[N];
  Store(a, Simd<T, N, 0>(), sa); Store(b, Simd<T, N, 0>(), sb);
  for (std::size_t i = 0; i < N / 2; ++i) {
    so[i] = static_cast<T>(sa[2 * i] + sa[2 * i + 1]);
    so[N / 2 + i] = static_cast<T>(sb[2 * i] + sb[2 * i + 1]);
  }
  return Load(Simd<T, N, 0>(), so);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> PairwiseSub(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  alignas(32) T sa[N], sb[N], so[N];
  Store(a, Simd<T, N, 0>(), sa); Store(b, Simd<T, N, 0>(), sb);
  for (std::size_t i = 0; i < N / 2; ++i) {
    so[i] = static_cast<T>(sa[2 * i] - sa[2 * i + 1]);
    so[N / 2 + i] = static_cast<T>(sb[2 * i] - sb[2 * i + 1]);
  }
  return Load(Simd<T, N, 0>(), so);
}
template <class D>
HYLE_FORCE_INLINE void Stream(detail::V256<D> v, D, TFromD<D>* aligned) noexcept {
  using T = TFromD<D>;
  if constexpr (std::is_same_v<T, float>) _mm256_stream_ps(aligned, v.raw);
  else if constexpr (std::is_same_v<T, double>) _mm256_stream_pd(aligned, v.raw);
  else _mm256_stream_si256(reinterpret_cast<__m256i*>(aligned), v.raw);
}
template <class D> requires HYLE_SIMD_W(D, 16, 32)
HYLE_FORCE_INLINE detail::V256<D> LoadDup128(D d, const TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  detail::V256<D> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_broadcast_ps(reinterpret_cast<const __m128*>(p));
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm256_broadcast_pd(reinterpret_cast<const __m128d*>(p));
  else r.raw = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE auto GetBiasedExponent(Vec256<T, N> v) noexcept {
  static_assert(std::is_floating_point_v<T>, "GetBiasedExponent is float only");
  using U = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
  const auto bits = BitCast(Simd<U, N, 0>(), v);
  if constexpr (sizeof(T) == 4) return And(ShiftRight<23>(bits), Set(Simd<U, N, 0>(), U{0xFF}));
  else return And(ShiftRight<52>(bits), Set(Simd<U, N, 0>(), U{0x7FF}));
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> MulByFloorPow2(Vec256<T, N> v, Vec256<T, N> exp) noexcept {
  static_assert(std::is_floating_point_v<T>, "MulByFloorPow2 is float only");
  // Store+reload scalar ldexp with Highway's NaN special-cases: NaN exp, (0,+inf), (inf,-inf)
  // -> NaN. floor(exp) is clamped into int range first so the cast is never UB (a raw
  // static_cast<int>(floor(+inf)) is UB and diverged across backends). Matches the scalar oracle.
  alignas(32) T sv[N], se[N], ro[N];
  Store(v, Simd<T, N, 0>(), sv); Store(exp, Simd<T, N, 0>(), se);
  for (std::size_t i = 0; i < N; ++i) {
    if (std::isnan(se[i]) ||
        (std::isinf(se[i]) && ((sv[i] == T{0} && se[i] > 0) || (std::isinf(sv[i]) && se[i] < 0)))) {
      ro[i] = std::numeric_limits<T>::quiet_NaN();
    } else {
      const T fe = std::floor(se[i]);
      const int ie = (fe >= static_cast<T>(std::numeric_limits<int>::max())) ? std::numeric_limits<int>::max()
                   : (fe <= static_cast<T>(std::numeric_limits<int>::min())) ? std::numeric_limits<int>::min()
                   : static_cast<int>(fe);
      ro[i] = std::ldexp(sv[i], ie);
    }
  }
  return Load(Simd<T, N, 0>(), ro);
}

// ---- simd-17 batch4: half-vector family (256 -> 128 real Vec128 entity) -------------
// Batch0 layered includes make Vec128 coexist; LowerHalf returns a true Vec128 via cast
// (zero-cost "forget high 128"). UpperHalf uses extract. ZeroExtendVector uses zext*
// (MSVC-safe zero of high half). Combine = ZeroExtend(lo) then insert hi at lane 1.
// Parameter order: Combine(d, hi, lo) = [lo in low 128, hi in high 128].
template <typename T, std::size_t NH, int kPow2, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, NH> LowerHalf(Simd<T, NH, kPow2>, Vec256<T, N> v) noexcept {
  static_assert(NH == N / 2 || (N == 1 && NH == 1), "LowerHalf: NH must be N/2");
  Vec128<T, NH> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_castps256_ps128(v.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm256_castpd256_pd128(v.raw);
  else r.raw = _mm256_castsi256_si128(v.raw);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, (N > 1 ? N / 2 : 1)> LowerHalf(Vec256<T, N> v) noexcept {
  return LowerHalf(Simd<T, (N > 1 ? N / 2 : 1), 0>{}, v);
}
template <typename T, std::size_t NH, int kPow2, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, NH> UpperHalf(Simd<T, NH, kPow2>, Vec256<T, N> v) noexcept {
  static_assert(NH == N / 2, "UpperHalf: NH must be N/2");
  Vec128<T, NH> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_extractf128_ps(v.raw, 1);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm256_extractf128_pd(v.raw, 1);
  else r.raw = _mm256_extracti128_si256(v.raw, 1);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N / 2> UpperHalf(Vec256<T, N> v) noexcept {
  static_assert(N >= 2, "UpperHalf needs N>=2");
  return UpperHalf(Simd<T, N / 2, 0>{}, v);
}
template <class D> requires HYLE_SIMD_W(D, 16, 32)
HYLE_FORCE_INLINE detail::V256<D> ZeroExtendVector(
    D /*d*/, Vec128<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D) / 2> lo) noexcept {
  using T = TFromD<D>;
  static_assert(HYLE_SIMD_MAX_LANES_D(D) >= 2, "ZeroExtendVector needs N>=2");
  detail::V256<D> r;
  // zext* zeros the high 128 (MSVC-safe; plain cast leaves high undefined).
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_zextps128_ps256(lo.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm256_zextpd128_pd256(lo.raw);
  else r.raw = _mm256_zextsi128_si256(lo.raw);
  return r;
}
template <class D> requires HYLE_SIMD_W(D, 16, 32)
HYLE_FORCE_INLINE detail::V256<D> Combine(D d,
    Vec128<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D) / 2> hi,
    Vec128<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D) / 2> lo) noexcept {
  using T = TFromD<D>;
  static_assert(HYLE_SIMD_MAX_LANES_D(D) >= 2, "Combine needs N>=2");
  // Zero-extend lo, then insert hi into the high 128.
  auto r = ZeroExtendVector(d, lo);
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_insertf128_ps(r.raw, hi.raw, 1);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm256_insertf128_pd(r.raw, hi.raw, 1);
  else r.raw = _mm256_inserti128_si256(r.raw, hi.raw, 1);
  return r;
}
template <class D> requires HYLE_SIMD_W(D, 16, 32)
HYLE_FORCE_INLINE detail::V256<D> InsertIntoUpper(
    D /*d*/, Vec128<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D) / 2> hi, detail::V256<D> v) noexcept {
  using T = TFromD<D>;
  static_assert(HYLE_SIMD_MAX_LANES_D(D) >= 2, "InsertIntoUpper needs N>=2");
  detail::V256<D> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_insertf128_ps(v.raw, hi.raw, 1);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm256_insertf128_pd(v.raw, hi.raw, 1);
  else r.raw = _mm256_inserti128_si256(v.raw, hi.raw, 1);
  return r;
}
// ResizeBitCast: Vec256 -> avx2-band D (same/narrow within 256); Vec128 -> avx2-band D (widen);
// Vec256 -> sse4-band D (narrow to 128). Excess destination bytes zeroed.
template <class D, typename SrcT, std::size_t NSrc> requires HYLE_SIMD_W(D, 16, 32)
HYLE_FORCE_INLINE detail::V256<D> ResizeBitCast(D d, Vec256<SrcT, NSrc> v) noexcept {
  using DstT = TFromD<D>;
  constexpr std::size_t NDst = HYLE_SIMD_MAX_LANES_D(D);
  alignas(32) std::uint8_t buf[32] = {};
  alignas(32) SrcT src[NSrc];
  Store(v, Simd<SrcT, NSrc, 0>(), src);
  const std::size_t n = (sizeof(SrcT) * NSrc < sizeof(DstT) * NDst) ? sizeof(SrcT) * NSrc
                                                                    : sizeof(DstT) * NDst;
  std::memcpy(buf, src, n);
  return Load(d, reinterpret_cast<const DstT*>(buf));
}
template <class D, typename SrcT, std::size_t NSrc> requires HYLE_SIMD_W(D, 16, 32)
HYLE_FORCE_INLINE detail::V256<D> ResizeBitCast(D d, Vec128<SrcT, NSrc> v) noexcept {
  // Widen: zero-extend 128 -> 256 then bit-view as Dst (same as ZeroExtend when types match).
  using DstT = TFromD<D>;
  constexpr std::size_t NDst = HYLE_SIMD_MAX_LANES_D(D);
  alignas(32) std::uint8_t buf[32] = {};
  alignas(16) SrcT src[NSrc];
  Store(v, Simd<SrcT, NSrc, 0>(), src);
  const std::size_t n = (sizeof(SrcT) * NSrc < sizeof(DstT) * NDst) ? sizeof(SrcT) * NSrc
                                                                    : sizeof(DstT) * NDst;
  std::memcpy(buf, src, n);
  return Load(d, reinterpret_cast<const DstT*>(buf));
}
template <class D, typename SrcT, std::size_t NSrc> requires HYLE_SIMD_W(D, 0, 16)
HYLE_FORCE_INLINE detail::V128<D> ResizeBitCast(D d, Vec256<SrcT, NSrc> v) noexcept {
  // Narrow 256 -> 128: take low min-bytes.
  using DstT = TFromD<D>;
  constexpr std::size_t NDst = HYLE_SIMD_MAX_LANES_D(D);
  alignas(16) std::uint8_t buf[16] = {};
  alignas(32) SrcT src[NSrc];
  Store(v, Simd<SrcT, NSrc, 0>(), src);
  const std::size_t n = (sizeof(SrcT) * NSrc < sizeof(DstT) * NDst) ? sizeof(SrcT) * NSrc
                                                                    : sizeof(DstT) * NDst;
  std::memcpy(buf, src, n);
  return Load(d, reinterpret_cast<const DstT*>(buf));
}

// ---- simd-17 batch5: 128-bit block extract/broadcast (avx2 = 2 blocks) ---------------
// ExtractBlock<k>(v): return the k-th 128-bit block as a real Vec128.
// BroadcastBlock<k>(v): splat the k-th 128-bit block across both halves of the 256-bit vector.
template <int k, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N / 2> ExtractBlock(Vec256<T, N> v) noexcept {
  static_assert(k == 0 || k == 1, "avx2 ExtractBlock: k in {0,1}");
  static_assert(sizeof(T) * N == 32, "ExtractBlock needs full 256-bit vector");
  Vec128<T, N / 2> r;
  if constexpr (k == 0) {
    if constexpr (std::is_same_v<T, float>) r.raw = _mm256_castps256_ps128(v.raw);
    else if constexpr (std::is_same_v<T, double>) r.raw = _mm256_castpd256_pd128(v.raw);
    else r.raw = _mm256_castsi256_si128(v.raw);
  } else {
    if constexpr (std::is_same_v<T, float>) r.raw = _mm256_extractf128_ps(v.raw, 1);
    else if constexpr (std::is_same_v<T, double>) r.raw = _mm256_extractf128_pd(v.raw, 1);
    else r.raw = _mm256_extracti128_si256(v.raw, 1);
  }
  return r;
}
template <int k, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> BroadcastBlock(Vec256<T, N> v) noexcept {
  static_assert(k == 0 || k == 1, "avx2 BroadcastBlock: k in {0,1}");
  static_assert(sizeof(T) * N == 32, "BroadcastBlock needs full 256-bit vector");
  // permute2x128 imm: both result halves select source half k of the first operand.
  // k=0 -> 0x00 (lo,lo); k=1 -> 0x11 (hi,hi). Second operand unused.
  constexpr int imm = (k == 0) ? 0x00 : 0x11;
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_permute2f128_ps(v.raw, v.raw, imm);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm256_permute2f128_pd(v.raw, v.raw, imm);
  else r.raw = _mm256_permute2x128_si256(v.raw, v.raw, imm);
  return r;
}

// ---- simd-17 batch1: fixed-pattern Shuffle + byte/lane shifts (per-128-bit-block) ----
// _mm256_shuffle_epi32 / slli_si256 / alignr_epi8 all act independently on each 128-bit half,
// which is exactly the Highway per-block contract.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> Shuffle2301(Vec256<T, N> v) noexcept {
  static_assert(sizeof(T) == 4, "Shuffle2301 needs sizeof(T)==4");
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_shuffle_ps(v.raw, v.raw, 0xB1);
  else r.raw = _mm256_shuffle_epi32(v.raw, 0xB1);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> Shuffle1032(Vec256<T, N> v) noexcept {
  static_assert(sizeof(T) == 4, "Shuffle1032 needs sizeof(T)==4");
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_shuffle_ps(v.raw, v.raw, 0x4E);
  else r.raw = _mm256_shuffle_epi32(v.raw, 0x4E);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> Shuffle0321(Vec256<T, N> v) noexcept {
  static_assert(sizeof(T) == 4, "Shuffle0321 needs sizeof(T)==4");
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_shuffle_ps(v.raw, v.raw, 0x39);
  else r.raw = _mm256_shuffle_epi32(v.raw, 0x39);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> Shuffle2103(Vec256<T, N> v) noexcept {
  static_assert(sizeof(T) == 4, "Shuffle2103 needs sizeof(T)==4");
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_shuffle_ps(v.raw, v.raw, 0x93);
  else r.raw = _mm256_shuffle_epi32(v.raw, 0x93);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> Shuffle0123(Vec256<T, N> v) noexcept {
  static_assert(sizeof(T) == 4, "Shuffle0123 needs sizeof(T)==4");
  Vec256<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm256_shuffle_ps(v.raw, v.raw, 0x1B);
  else r.raw = _mm256_shuffle_epi32(v.raw, 0x1B);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> Shuffle01(Vec256<T, N> v) noexcept {
  static_assert(sizeof(T) == 8, "Shuffle01 needs sizeof(T)==8");
  Vec256<T, N> r;
  // 0x5 = 0b0101: swap the two 64-bit lanes inside each 128-bit half.
  if constexpr (std::is_same_v<T, double>) r.raw = _mm256_shuffle_pd(v.raw, v.raw, 0x5);
  else r.raw = _mm256_shuffle_epi32(v.raw, 0x4E);
  return r;
}

template <int k, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> ShiftLeftBytes(Vec256<T, N> v) noexcept {
  static_assert(k >= 0 && k <= 16, "ShiftLeftBytes count in [0,16]");
  const __m256i i = detail::bitcast_raw_avx2<__m256i>(v.raw);
  const __m256i s = _mm256_slli_si256(i, k);  // per-128-half
  Vec256<T, N> r;
  r.raw = detail::bitcast_raw_avx2<detail::Raw256T<T>>(s);
  return r;
}
template <int k, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> ShiftRightBytes(Vec256<T, N> v) noexcept {
  static_assert(k >= 0 && k <= 16, "ShiftRightBytes count in [0,16]");
  const __m256i i = detail::bitcast_raw_avx2<__m256i>(v.raw);
  const __m256i s = _mm256_srli_si256(i, k);
  Vec256<T, N> r;
  r.raw = detail::bitcast_raw_avx2<detail::Raw256T<T>>(s);
  return r;
}
template <int k, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> ShiftLeftLanes(Vec256<T, N> v) noexcept {
  return ShiftLeftBytes<k * static_cast<int>(sizeof(T))>(v);
}
template <int k, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> ShiftRightLanes(Vec256<T, N> v) noexcept {
  return ShiftRightBytes<k * static_cast<int>(sizeof(T))>(v);
}
template <int k, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> CombineShiftRightBytes(Vec256<T, N> hi, Vec256<T, N> lo) noexcept {
  static_assert(k > 0 && k < 16, "CombineShiftRightBytes count in (0,16)");
  const __m256i hi_i = detail::bitcast_raw_avx2<__m256i>(hi.raw);
  const __m256i lo_i = detail::bitcast_raw_avx2<__m256i>(lo.raw);
  const __m256i s = _mm256_alignr_epi8(hi_i, lo_i, k);  // per-128-half
  Vec256<T, N> r;
  r.raw = detail::bitcast_raw_avx2<detail::Raw256T<T>>(s);
  return r;
}
template <int k, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> CombineShiftRightLanes(Vec256<T, N> hi, Vec256<T, N> lo) noexcept {
  return CombineShiftRightBytes<k * static_cast<int>(sizeof(T))>(hi, lo);
}

// ---- crypto: AES round primitives + carryless multiply (simd-08 档C) -----------------
// One __m256i == TWO independent 128-bit AES blocks. AES-NI operates on a 128-bit register,
// so we split lo/hi (cast / extracti128) and call the shared per-block worker on each, then
// recombine (set_m128i). This is EXACTLY Highway's x86_256 AES: it builds the 256-bit op from
// AES-NI per-128 rather than requiring VAES (_mm256_aesenc_epi128, Ice Lake+, needs -mvaes).
// Promoting to VAES is a pure profiling upgrade. Contract: a 256-bit vector = 32 bytes = two
// 128-bit AES states (sizeof(T)*N == 32). Software fallback lives inside the block worker.
namespace detail {
template <typename BlockOp>
HYLE_FORCE_INLINE __m256i aes256_map1(__m256i s, BlockOp op) noexcept {
  const __m128i lo = op(_mm256_castsi256_si128(s));
  const __m128i hi = op(_mm256_extracti128_si256(s, 1));
  return _mm256_set_m128i(hi, lo);
}
template <typename BlockOp>
HYLE_FORCE_INLINE __m256i aes256_map2(__m256i s, __m256i k, BlockOp op) noexcept {
  const __m128i lo = op(_mm256_castsi256_si128(s), _mm256_castsi256_si128(k));
  const __m128i hi = op(_mm256_extracti128_si256(s, 1), _mm256_extracti128_si256(k, 1));
  return _mm256_set_m128i(hi, lo);
}
}  // namespace detail
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> AESRound(Vec256<T, N> state, Vec256<T, N> key) noexcept {
  static_assert(sizeof(T) * N == 32, "AES op needs a 256-bit vector (two 128-bit blocks)");
  return Vec256<T, N>{detail::aes256_map2(state.raw, key.raw, detail::aes_round_block)};
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> AESLastRound(Vec256<T, N> state, Vec256<T, N> key) noexcept {
  static_assert(sizeof(T) * N == 32, "AES op needs a 256-bit vector (two 128-bit blocks)");
  return Vec256<T, N>{detail::aes256_map2(state.raw, key.raw, detail::aes_last_round_block)};
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> AESRoundInv(Vec256<T, N> state, Vec256<T, N> key) noexcept {
  static_assert(sizeof(T) * N == 32, "AES op needs a 256-bit vector (two 128-bit blocks)");
  return Vec256<T, N>{detail::aes256_map2(state.raw, key.raw, detail::aes_round_inv_block)};
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> AESLastRoundInv(Vec256<T, N> state, Vec256<T, N> key) noexcept {
  static_assert(sizeof(T) * N == 32, "AES op needs a 256-bit vector (two 128-bit blocks)");
  return Vec256<T, N>{detail::aes256_map2(state.raw, key.raw, detail::aes_last_round_inv_block)};
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> AESInvMixColumns(Vec256<T, N> state) noexcept {
  static_assert(sizeof(T) * N == 32, "AES op needs a 256-bit vector (two 128-bit blocks)");
  return Vec256<T, N>{detail::aes256_map1(state.raw, detail::aes_inv_mix_block)};
}
template <std::uint8_t kRcon, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> AESKeyGenAssist(Vec256<T, N> v) noexcept {
  static_assert(sizeof(T) * N == 32, "AES op needs a 256-bit vector (two 128-bit blocks)");
  // kRcon is a compile-time immediate, so the per-block worker is templated (can't route
  // through aes256_map1's plain fn-ptr) -- split lo/hi explicitly.
  const __m128i lo = detail::aes_keygen_block<kRcon>(_mm256_castsi256_si128(v.raw));
  const __m128i hi = detail::aes_keygen_block<kRcon>(_mm256_extracti128_si256(v.raw, 1));
  return Vec256<T, N>{_mm256_set_m128i(hi, lo)};
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> CLMulLower(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  static_assert(sizeof(T) * N == 32, "CLMul needs a 256-bit vector (two 128-bit blocks)");
  return Vec256<T, N>{detail::aes256_map2(a.raw, b.raw, detail::clmul_lower_block)};
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec256<T, N> CLMulUpper(Vec256<T, N> a, Vec256<T, N> b) noexcept {
  static_assert(sizeof(T) * N == 32, "CLMul needs a 256-bit vector (two 128-bit blocks)");
  return Vec256<T, N>{detail::aes256_map2(a.raw, b.raw, detail::clmul_upper_block)};
}

}  // namespace hyle::simd
