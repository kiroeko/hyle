#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>  // std::memcpy (LoadMaskBits lane pun)
#include <bit>
#include <cmath>  // std::ldexp (MulByPow2 store+reload)
#include <limits>  // std::numeric_limits<T>::infinity (IsInf/IsFinite splat)
#include <type_traits>

#include <immintrin.h>

#include "hyle/macros.h"
#include "hyle/simd/backend/crypto_x86.h"  // shared 128-bit AES-NI/PCLMUL workers (Crypto op face)

// backend/avx512.h -- 512-bit AVX-512 backend (simd-design landing list #2, AVX512 half).
//
// ONE backend per TU: pulled in by vec.h only when -DHYLE_SIMD_TARGET_AVX512. Same op
// contract as scalar/sse4/avx2/neon, so the unified test/kernel source compiles unchanged.
// Lane types f32/f64 + i8/u8/i16/u16/i32/u32/i64/u64; NATIVE FULL WIDTH (512-bit = 64 bytes =
// 64 i8 / 32 i16 / 16 i32 / 8 i64 / 16 f32 / 8 f64). Tier = AVX-512 F+BW+VL+DQ, so BW gives
// 8/16-bit ops, DQ gives mullo_epi64 + 64-bit min/max, F gives abs_epi64 + reduce_*.
//
// STATUS (honest, per user directive "没机器也要做,写完存疑"): written + NATIVELY compiled
// (-mavx512f -mavx512bw -mavx512vl -mavx512dq) + objdump-verified to emit real zmm/k-reg
// instructions, but NOT executed (no AVX-512 hardware / no Intel SDE here). Any x86-64 can
// COMPILE this; correctness-by-execution is PENDING (an AVX-512 machine or SDE). Marked
// 存疑 until run. Implementing it now validates the DESIGN -- see the mask note below.
//
// THE MASK-MODEL POINT (this is why writing it matters): AVX-512 masks are __mmask8/16/32/64
// "k registers" (1 bit per lane, width == lane count), NOT the all-1s/all-0s vector that SSE/AVX2/NEON use. This
// backend defines its OWN Mask512 wrapping the k-reg. Because vec.h's Mask type is already
// PER-BACKEND (scalar uses bool[], not a vector), this needs ZERO change to vec.h or the
// tests -- the op signatures match, and nothing outside inspects a mask's internals. So
// AVX-512 fits the existing abstraction cleanly; the earlier "needs a mask redesign" worry
// was wrong, and writing this is what proved it.

namespace hyle::simd {

namespace detail {
// Every integer lane type shares __m512i (width-agnostic register); only float/double differ.
template <typename T> struct Raw512 { using type = __m512i; };
template <> struct Raw512<float> { using type = __m512; };
template <> struct Raw512<double> { using type = __m512d; };
template <typename T> using Raw512T = typename Raw512<T>::type;

// k-register mask width == lane count: __mmask64 for i8/u8 (64 lanes), __mmask32 for i16/u16
// (32), __mmask16 for f32/i32/u32 (16), __mmask8 for f64/i64/u64 (8). The k-reg is 1 bit/lane.
template <typename T> struct KMask { using type = __mmask16; };  // f32/i32/u32
template <> struct KMask<double> { using type = __mmask8; };
template <> struct KMask<std::int64_t> { using type = __mmask8; };
template <> struct KMask<std::uint64_t> { using type = __mmask8; };
template <> struct KMask<std::int16_t> { using type = __mmask32; };
template <> struct KMask<std::uint16_t> { using type = __mmask32; };
template <> struct KMask<std::int8_t> { using type = __mmask64; };
template <> struct KMask<std::uint8_t> { using type = __mmask64; };
template <typename T> using KMaskT = typename KMask<T>::type;

template <typename T>
inline constexpr bool kSupported =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::uint32_t> ||
    std::is_same_v<T, std::int16_t> || std::is_same_v<T, std::uint16_t> ||
    std::is_same_v<T, std::int8_t> || std::is_same_v<T, std::uint8_t> ||
    std::is_same_v<T, std::int64_t> || std::is_same_v<T, std::uint64_t>;

// Integer-width helpers so the per-op constexpr chains stay readable (mirror avx2.h). AVX-512
// integer ops pick the intrinsic by lane WIDTH (epi8/16/32/64), so we branch on sizeof(T).
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
struct Vec512 {
  static_assert(detail::kSupported<T>, "avx512 backend lane type unsupported");
  typename detail::Raw512T<T> raw;
  using PrivateT = T;
  static constexpr std::size_t kPrivateN = N;
};
// Mask = k register (NOT a vector). This is the model difference from SSE/AVX2/NEON.
template <typename T, std::size_t N>
struct Mask512 {
  typename detail::KMaskT<T> raw;
  using PrivateT = T;
  static constexpr std::size_t kPrivateN = N;
};

template <class V>
using DFromV = Simd<typename V::PrivateT, V::kPrivateN, 0>;
template <class V>
using TFromV = typename V::PrivateT;

namespace detail {
template <class D>
using V512 = Vec512<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)>;
template <class D>
inline constexpr bool kNativeWidth = HYLE_SIMD_MAX_LANES_D(D) * sizeof(TFromD<D>) == 64;
}  // namespace detail

// ---- initialization -----------------------------------------------------------------
template <class D, typename T2>
HYLE_FORCE_INLINE detail::V512<D> Set(D, T2 t) noexcept {
  using T = TFromD<D>;
  static_assert(detail::kNativeWidth<D>, "avx512 backend is native-width-only (512-bit)");
  detail::V512<D> v;
  if constexpr (std::is_same_v<T, float>) v.raw = _mm512_set1_ps(static_cast<float>(t));
  else if constexpr (std::is_same_v<T, double>) v.raw = _mm512_set1_pd(static_cast<double>(t));
  else if constexpr (detail::kI8<T>) v.raw = _mm512_set1_epi8(static_cast<char>(t));
  else if constexpr (detail::kI16<T>) v.raw = _mm512_set1_epi16(static_cast<short>(t));
  else if constexpr (detail::kI64<T>) v.raw = _mm512_set1_epi64(static_cast<long long>(t));
  else v.raw = _mm512_set1_epi32(static_cast<int>(t));  // i32/u32
  return v;
}
template <class D>
HYLE_FORCE_INLINE detail::V512<D> Zero(D) noexcept {
  using T = TFromD<D>;
  static_assert(detail::kNativeWidth<D>, "avx512 backend is native-width-only (512-bit)");
  detail::V512<D> v;
  if constexpr (std::is_same_v<T, float>) v.raw = _mm512_setzero_ps();
  else if constexpr (std::is_same_v<T, double>) v.raw = _mm512_setzero_pd();
  else v.raw = _mm512_setzero_si512();
  return v;
}
template <class D>
using VFromD = decltype(Zero(D()));

// ---- load / store -------------------------------------------------------------------
template <class D>
HYLE_FORCE_INLINE detail::V512<D> LoadU(D, const TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  detail::V512<D> v;
  if constexpr (std::is_same_v<T, float>) v.raw = _mm512_loadu_ps(p);
  else if constexpr (std::is_same_v<T, double>) v.raw = _mm512_loadu_pd(p);
  else v.raw = _mm512_loadu_si512(p);
  return v;
}
template <class D>
HYLE_FORCE_INLINE detail::V512<D> Load(D, const TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  detail::V512<D> v;
  if constexpr (std::is_same_v<T, float>) v.raw = _mm512_load_ps(p);
  else if constexpr (std::is_same_v<T, double>) v.raw = _mm512_load_pd(p);
  else v.raw = _mm512_load_si512(p);
  return v;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE void StoreU(Vec512<T, N> v, Simd<T, N, 0>, T* p) noexcept {
  if constexpr (std::is_same_v<T, float>) _mm512_storeu_ps(p, v.raw);
  else if constexpr (std::is_same_v<T, double>) _mm512_storeu_pd(p, v.raw);
  else _mm512_storeu_si512(p, v.raw);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE void Store(Vec512<T, N> v, Simd<T, N, 0>, T* p) noexcept {
  if constexpr (std::is_same_v<T, float>) _mm512_store_ps(p, v.raw);
  else if constexpr (std::is_same_v<T, double>) _mm512_store_pd(p, v.raw);
  else _mm512_store_si512(p, v.raw);
}

// ---- interleaved load / store (AoS <-> SoA) -----------------------------------------
// correctness-first strided copy; native deinterleave shuffles are a profiling upgrade.
template <class D>
HYLE_FORCE_INLINE void LoadInterleaved2(D d, const TFromD<D>* p, VFromD<D>& v0,
                                        VFromD<D>& v1) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(64) T t0[N], t1[N];
  for (std::size_t i = 0; i < N; ++i) { t0[i] = p[2 * i]; t1[i] = p[2 * i + 1]; }
  v0 = Load(d, t0); v1 = Load(d, t1);
}
template <class D>
HYLE_FORCE_INLINE void LoadInterleaved3(D d, const TFromD<D>* p, VFromD<D>& v0, VFromD<D>& v1,
                                        VFromD<D>& v2) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(64) T t0[N], t1[N], t2[N];
  for (std::size_t i = 0; i < N; ++i) { t0[i] = p[3*i]; t1[i] = p[3*i+1]; t2[i] = p[3*i+2]; }
  v0 = Load(d, t0); v1 = Load(d, t1); v2 = Load(d, t2);
}
template <class D>
HYLE_FORCE_INLINE void LoadInterleaved4(D d, const TFromD<D>* p, VFromD<D>& v0, VFromD<D>& v1,
                                        VFromD<D>& v2, VFromD<D>& v3) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(64) T t0[N], t1[N], t2[N], t3[N];
  for (std::size_t i = 0; i < N; ++i) { t0[i]=p[4*i]; t1[i]=p[4*i+1]; t2[i]=p[4*i+2]; t3[i]=p[4*i+3]; }
  v0 = Load(d, t0); v1 = Load(d, t1); v2 = Load(d, t2); v3 = Load(d, t3);
}
template <class D>
HYLE_FORCE_INLINE void StoreInterleaved2(VFromD<D> v0, VFromD<D> v1, D d, TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(64) T t0[N], t1[N];
  Store(v0, d, t0); Store(v1, d, t1);
  for (std::size_t i = 0; i < N; ++i) { p[2*i] = t0[i]; p[2*i+1] = t1[i]; }
}
template <class D>
HYLE_FORCE_INLINE void StoreInterleaved3(VFromD<D> v0, VFromD<D> v1, VFromD<D> v2, D d,
                                         TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(64) T t0[N], t1[N], t2[N];
  Store(v0, d, t0); Store(v1, d, t1); Store(v2, d, t2);
  for (std::size_t i = 0; i < N; ++i) { p[3*i]=t0[i]; p[3*i+1]=t1[i]; p[3*i+2]=t2[i]; }
}
template <class D>
HYLE_FORCE_INLINE void StoreInterleaved4(VFromD<D> v0, VFromD<D> v1, VFromD<D> v2, VFromD<D> v3, D d,
                                         TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(64) T t0[N], t1[N], t2[N], t3[N];
  Store(v0, d, t0); Store(v1, d, t1); Store(v2, d, t2); Store(v3, d, t3);
  for (std::size_t i = 0; i < N; ++i) { p[4*i]=t0[i]; p[4*i+1]=t1[i]; p[4*i+2]=t2[i]; p[4*i+3]=t3[i]; }
}

// ---- arithmetic ---------------------------------------------------------------------
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> Add(Vec512<T, N> a, Vec512<T, N> b) noexcept {
  Vec512<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm512_add_ps(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm512_add_pd(a.raw, b.raw);
  else if constexpr (detail::kI8<T>) r.raw = _mm512_add_epi8(a.raw, b.raw);   // AVX512BW
  else if constexpr (detail::kI16<T>) r.raw = _mm512_add_epi16(a.raw, b.raw); // AVX512BW
  else if constexpr (detail::kI64<T>) r.raw = _mm512_add_epi64(a.raw, b.raw);
  else r.raw = _mm512_add_epi32(a.raw, b.raw);  // i32/u32
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> Sub(Vec512<T, N> a, Vec512<T, N> b) noexcept {
  Vec512<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm512_sub_ps(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm512_sub_pd(a.raw, b.raw);
  else if constexpr (detail::kI8<T>) r.raw = _mm512_sub_epi8(a.raw, b.raw);   // AVX512BW
  else if constexpr (detail::kI16<T>) r.raw = _mm512_sub_epi16(a.raw, b.raw); // AVX512BW
  else if constexpr (detail::kI64<T>) r.raw = _mm512_sub_epi64(a.raw, b.raw);
  else r.raw = _mm512_sub_epi32(a.raw, b.raw);  // i32/u32
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> Mul(Vec512<T, N> a, Vec512<T, N> b) noexcept {
  Vec512<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm512_mul_ps(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm512_mul_pd(a.raw, b.raw);
  else if constexpr (detail::kI16<T>) r.raw = _mm512_mullo_epi16(a.raw, b.raw);  // AVX512BW
  else if constexpr (detail::kI32<T>) r.raw = _mm512_mullo_epi32(a.raw, b.raw);  // AVX512F
  else if constexpr (detail::kI64<T>) r.raw = _mm512_mullo_epi64(a.raw, b.raw);  // AVX512DQ (native!)
  else if constexpr (detail::kI8<T>) {
    // No _mm512_mullo_epi8. Same widen-to-epi16 scheme as avx2 (BW ops): mask even bytes /
    // shift out odd bytes, mullo_epi16 each half, keep the low byte of each 16-bit product
    // (wraparound = 2's-complement low 8 bits, matching scalar T(x*y)).
    const __m512i lo_mask = _mm512_set1_epi16(0x00FF);
    const __m512i a_lo = _mm512_and_si512(a.raw, lo_mask);              // even bytes
    const __m512i b_lo = _mm512_and_si512(b.raw, lo_mask);
    const __m512i a_hi = _mm512_srli_epi16(a.raw, 8);                   // odd bytes
    const __m512i b_hi = _mm512_srli_epi16(b.raw, 8);
    const __m512i p_lo = _mm512_and_si512(_mm512_mullo_epi16(a_lo, b_lo), lo_mask);
    const __m512i p_hi = _mm512_slli_epi16(_mm512_mullo_epi16(a_hi, b_hi), 8);
    r.raw = _mm512_or_si512(p_lo, p_hi);
  } else r.raw = _mm512_mullo_epi32(a.raw, b.raw);  // unreachable (all int widths covered)
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> Div(Vec512<T, N> a, Vec512<T, N> b) noexcept {
  static_assert(std::is_floating_point_v<T>, "Div is float/double only");
  Vec512<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm512_div_ps(a.raw, b.raw);
  else r.raw = _mm512_div_pd(a.raw, b.raw);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> Min(Vec512<T, N> a, Vec512<T, N> b) noexcept {
  Vec512<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm512_min_ps(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm512_min_pd(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int8_t>) r.raw = _mm512_min_epi8(a.raw, b.raw);   // BW
  else if constexpr (std::is_same_v<T, std::uint8_t>) r.raw = _mm512_min_epu8(a.raw, b.raw);  // BW
  else if constexpr (std::is_same_v<T, std::int16_t>) r.raw = _mm512_min_epi16(a.raw, b.raw); // BW
  else if constexpr (std::is_same_v<T, std::uint16_t>) r.raw = _mm512_min_epu16(a.raw, b.raw);// BW
  else if constexpr (std::is_same_v<T, std::int32_t>) r.raw = _mm512_min_epi32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint32_t>) r.raw = _mm512_min_epu32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int64_t>) r.raw = _mm512_min_epi64(a.raw, b.raw); // F (native!)
  else r.raw = _mm512_min_epu64(a.raw, b.raw);  // u64, AVX512F (native!)
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> Max(Vec512<T, N> a, Vec512<T, N> b) noexcept {
  Vec512<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm512_max_ps(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm512_max_pd(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int8_t>) r.raw = _mm512_max_epi8(a.raw, b.raw);   // BW
  else if constexpr (std::is_same_v<T, std::uint8_t>) r.raw = _mm512_max_epu8(a.raw, b.raw);  // BW
  else if constexpr (std::is_same_v<T, std::int16_t>) r.raw = _mm512_max_epi16(a.raw, b.raw); // BW
  else if constexpr (std::is_same_v<T, std::uint16_t>) r.raw = _mm512_max_epu16(a.raw, b.raw);// BW
  else if constexpr (std::is_same_v<T, std::int32_t>) r.raw = _mm512_max_epi32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint32_t>) r.raw = _mm512_max_epu32(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int64_t>) r.raw = _mm512_max_epi64(a.raw, b.raw); // F (native!)
  else r.raw = _mm512_max_epu64(a.raw, b.raw);  // u64, AVX512F (native!)
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> Neg(Vec512<T, N> a) noexcept {
  Vec512<T, N> r;
  // xor sign bit (not 0-a): Neg(+0.0) must be -0.0, matching SSE4/AVX2. 0-a yields +0.0.
  if constexpr (std::is_same_v<T, float>) r.raw = _mm512_xor_ps(a.raw, _mm512_set1_ps(-0.0f));
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm512_xor_pd(a.raw, _mm512_set1_pd(-0.0));
  else if constexpr (detail::kI8<T>) r.raw = _mm512_sub_epi8(_mm512_setzero_si512(), a.raw);   // BW
  else if constexpr (detail::kI16<T>) r.raw = _mm512_sub_epi16(_mm512_setzero_si512(), a.raw); // BW
  else if constexpr (detail::kI64<T>) r.raw = _mm512_sub_epi64(_mm512_setzero_si512(), a.raw);
  else r.raw = _mm512_sub_epi32(_mm512_setzero_si512(), a.raw);  // i32/u32
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> Abs(Vec512<T, N> a) noexcept {
  Vec512<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm512_abs_ps(a.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm512_abs_pd(a.raw);
  else if constexpr (std::is_same_v<T, std::int8_t>) r.raw = _mm512_abs_epi8(a.raw);   // BW
  else if constexpr (std::is_same_v<T, std::int16_t>) r.raw = _mm512_abs_epi16(a.raw); // BW
  else if constexpr (std::is_same_v<T, std::int32_t>) r.raw = _mm512_abs_epi32(a.raw);
  else if constexpr (std::is_same_v<T, std::int64_t>) r.raw = _mm512_abs_epi64(a.raw); // F (native!)
  else r.raw = a.raw;  // unsigned: abs is identity
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> Sqrt(Vec512<T, N> a) noexcept {
  static_assert(std::is_floating_point_v<T>, "Sqrt is float/double only");
  Vec512<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm512_sqrt_ps(a.raw);
  else r.raw = _mm512_sqrt_pd(a.raw);
  return r;
}

// ---- math / arith extensions (Highway) ----------------------------------------------
// ApproximateReciprocal/Sqrt: AVX-512 has NATIVE 14-bit estimates for BOTH ps AND pd
// (rcp14/rsqrt14) -- SSE/AVX2 only had the f32 rcp/rsqrt. ~2^-14 relative error (tests tolerate).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> ApproximateReciprocal(Vec512<T, N> a) noexcept {
  static_assert(std::is_floating_point_v<T>, "float only");
  Vec512<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm512_rcp14_ps(a.raw);
  else r.raw = _mm512_rcp14_pd(a.raw);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> ApproximateReciprocalSqrt(Vec512<T, N> a) noexcept {
  static_assert(std::is_floating_point_v<T>, "float only");
  Vec512<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm512_rsqrt14_ps(a.raw);
  else r.raw = _mm512_rsqrt14_pd(a.raw);
  return r;
}
// GetExponent(v) = floor(log2(|v|)) as float. AVX-512 has a NATIVE vgetexpps/pd that extracts
// exactly the unbiased exponent -- one instruction, no store+reload (unlike NEON's logb path).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> GetExponent(Vec512<T, N> v) noexcept {
  static_assert(std::is_floating_point_v<T>, "float only");
  Vec512<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm512_getexp_ps(v.raw);
  else r.raw = _mm512_getexp_pd(v.raw);
  return r;
}
// MulByPow2(a, b) = a * 2^b (b an integer-lane vector). store+reload std::ldexp: vscalef does
// a*2^floor(b) with b taken as FLOAT (needs an int->float convert + exactness argument), so the
// ldexp path is correctness-first and exact for the integer exponent contract.
template <typename T, std::size_t N, typename IT>
HYLE_FORCE_INLINE Vec512<T, N> MulByPow2(Vec512<T, N> a, Vec512<IT, N> b) noexcept {
  static_assert(std::is_floating_point_v<T>, "float only");
  alignas(64) T ta[N], tr[N];
  alignas(64) IT tb[N];
  Store(a, Simd<T, N, 0>(), ta);
  Store(b, Simd<IT, N, 0>(), tb);
  for (std::size_t i = 0; i < N; ++i) tr[i] = std::ldexp(ta[i], static_cast<int>(tb[i]));
  return Load(Simd<T, N, 0>(), tr);
}
// NaN-aware min/max (Highway MinNumber/MaxNumber): if one operand is NaN, return the other.
// Portable IsNaN/IfThenElse blend -- unambiguously oracle-correct for every lane type (AVX-512's
// native _mm512_min_ps is IEEE minNum-like, but the explicit form leaves no doubt). Int = Min/Max.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> MinNumber(Vec512<T, N> a, Vec512<T, N> b) noexcept {
  if constexpr (std::is_floating_point_v<T>)
    return IfThenElse(IsNaN(a), b, IfThenElse(IsNaN(b), a, Min(a, b)));
  else return Min(a, b);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> MaxNumber(Vec512<T, N> a, Vec512<T, N> b) noexcept {
  if constexpr (std::is_floating_point_v<T>)
    return IfThenElse(IsNaN(a), b, IfThenElse(IsNaN(b), a, Max(a, b)));
  else return Max(a, b);
}
// AddSub(a,b): a-b in EVEN lanes, a+b in ODD lanes (complex-arith / FFT butterfly). No single
// AVX-512 addsub across all widths; store+reload by lane parity (correctness-first, all types).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> AddSub(Vec512<T, N> a, Vec512<T, N> b) noexcept {
  alignas(64) T ta[N], tb[N], tr[N];
  Store(a, Simd<T, N, 0>(), ta);
  Store(b, Simd<T, N, 0>(), tb);
  for (std::size_t i = 0; i < N; ++i)
    tr[i] = (i & 1) ? static_cast<T>(ta[i] + tb[i]) : static_cast<T>(ta[i] - tb[i]);
  return Load(Simd<T, N, 0>(), tr);
}
// MulRound(a,b) = round(a*b) to nearest-even, result stays float (reuse Round + Mul).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> MulRound(Vec512<T, N> a, Vec512<T, N> b) noexcept {
  static_assert(std::is_floating_point_v<T>, "MulRound is float only");
  return Round(Mul(a, b));
}
// MulFixedPoint15(a,b): Q1.15 fixed-point multiply with rounding (i16). AVX-512-BW native
// _mm512_mulhrs_epi16 = (2*a*b + 0x8000) >> 16 -- exactly the oracle's rounded high half.
template <std::size_t N>
HYLE_FORCE_INLINE Vec512<std::int16_t, N> MulFixedPoint15(Vec512<std::int16_t, N> a,
                                                          Vec512<std::int16_t, N> b) noexcept {
  Vec512<std::int16_t, N> r;
  r.raw = _mm512_mulhrs_epi16(a.raw, b.raw);  // BW
  return r;
}
// SaturatedNeg(a) = (a==MIN)?MAX:-a; SaturatedAbs(a) = (a==MIN)?MAX:|a| (signed int). i8/i16
// reuse the native saturating SaturatedSub(0,a); i32/i64 have no saturating sub here, so guard
// the sole overflowing input (MIN) with IfThenElse. SaturatedAbs selects |a| = SaturatedNeg on a<0.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> SaturatedNeg(Vec512<T, N> a) noexcept {
  static_assert(std::is_integral_v<T> && std::is_signed_v<T>, "SaturatedNeg is signed-int");
  if constexpr (sizeof(T) <= 2) return SaturatedSub(Zero(Simd<T, N, 0>()), a);  // BW
  else {
    const auto lo = Set(Simd<T, N, 0>(), std::numeric_limits<T>::min());
    const auto hi = Set(Simd<T, N, 0>(), std::numeric_limits<T>::max());
    return IfThenElse(Eq(a, lo), hi, Neg(a));
  }
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> SaturatedAbs(Vec512<T, N> a) noexcept {
  static_assert(std::is_integral_v<T> && std::is_signed_v<T>, "SaturatedAbs is signed-int");
  return IfThenElse(Lt(a, Zero(Simd<T, N, 0>())), SaturatedNeg(a), a);
}

// ---- float rounding to integral value (result stays float) --------------------------
// Round=nearest-even, Floor=toward -inf, Ceil=toward +inf, Trunc=toward zero. AVX-512
// _mm512_roundscale_ps/pd with scale M=0: its imm[3:0] (mode + suppress-precision) matches
// the _MM_FROUND flags, so the same MODE | _MM_FROUND_NO_EXC constant works (like Highway).
#define HYLE_SIMD_AVX512_ROUND(NAME, MODE)                                   \
  template <typename T, std::size_t N>                                       \
  HYLE_FORCE_INLINE Vec512<T, N> NAME(Vec512<T, N> a) noexcept {             \
    static_assert(std::is_floating_point_v<T>, #NAME " is float/double only"); \
    Vec512<T, N> r;                                                          \
    if constexpr (std::is_same_v<T, float>)                                  \
      r.raw = _mm512_roundscale_ps(a.raw, (MODE) | _MM_FROUND_NO_EXC);       \
    else r.raw = _mm512_roundscale_pd(a.raw, (MODE) | _MM_FROUND_NO_EXC);    \
    return r;                                                                \
  }
HYLE_SIMD_AVX512_ROUND(Round, _MM_FROUND_TO_NEAREST_INT)
HYLE_SIMD_AVX512_ROUND(Floor, _MM_FROUND_TO_NEG_INF)
HYLE_SIMD_AVX512_ROUND(Ceil, _MM_FROUND_TO_POS_INF)
HYLE_SIMD_AVX512_ROUND(Trunc, _MM_FROUND_TO_ZERO)
#undef HYLE_SIMD_AVX512_ROUND

// AVX-512 has native FMA (it implies it). Integer falls back to mul+add.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> MulAdd(Vec512<T, N> a, Vec512<T, N> b, Vec512<T, N> c) noexcept {
  Vec512<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm512_fmadd_ps(a.raw, b.raw, c.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm512_fmadd_pd(a.raw, b.raw, c.raw);
  else r = Add(Mul(a, b), c);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> NegMulAdd(Vec512<T, N> a, Vec512<T, N> b, Vec512<T, N> c) noexcept {
  Vec512<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm512_fnmadd_ps(a.raw, b.raw, c.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm512_fnmadd_pd(a.raw, b.raw, c.raw);
  else r = Sub(c, Mul(a, b));
  return r;
}

// ---- bitwise (AVX512DQ has _ps/_pd logical forms; int via _si512) -------------------
#define HYLE_AVX512_BITOP(NAME, PS, PD, SI)                                \
  template <typename T, std::size_t N>                                     \
  HYLE_FORCE_INLINE Vec512<T, N> NAME(Vec512<T, N> a, Vec512<T, N> b) noexcept { \
    Vec512<T, N> r;                                                        \
    if constexpr (std::is_same_v<T, float>) r.raw = PS(a.raw, b.raw);      \
    else if constexpr (std::is_same_v<T, double>) r.raw = PD(a.raw, b.raw);\
    else r.raw = SI(a.raw, b.raw);                                         \
    return r;                                                              \
  }
HYLE_AVX512_BITOP(And, _mm512_and_ps, _mm512_and_pd, _mm512_and_si512)
HYLE_AVX512_BITOP(Or, _mm512_or_ps, _mm512_or_pd, _mm512_or_si512)
HYLE_AVX512_BITOP(Xor, _mm512_xor_ps, _mm512_xor_pd, _mm512_xor_si512)
HYLE_AVX512_BITOP(AndNot, _mm512_andnot_ps, _mm512_andnot_pd, _mm512_andnot_si512)  // ~a&b
#undef HYLE_AVX512_BITOP

// ---- shifts (integer lanes, compile-time immediate count) ---------------------------
// AVX-512 has slli/srli_epi16/32/64 (logical) AND srai_epi16/32/64 (arithmetic) — note
// _mm512_srai_epi64 EXISTS on AVX512F (unlike AVX2), so i64 arithmetic right is NATIVE,
// no sign-fixup emulation. GAP: epi8 still has NO shift (mirror avx2: epi16 shift + mask
// off bits that cross the byte boundary; i8 arithmetic = logical-byte-shift + sign-fill).
// AVX-512 compares yield k-masks, so recover a 0xFF/0x00 byte vector via _mm512_movm_epi8
// (BW) for the sign-fill AND. ShiftRight is arithmetic for signed T, logical for unsigned.
template <int kBits, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> ShiftLeft(Vec512<T, N> a) noexcept {
  static_assert(std::is_integral_v<T>, "shifts are integer-lane only");
  Vec512<T, N> r;
  if constexpr (detail::kI16<T>) r.raw = _mm512_slli_epi16(a.raw, kBits);        // BW
  else if constexpr (detail::kI32<T>) r.raw = _mm512_slli_epi32(a.raw, kBits);   // F
  else if constexpr (detail::kI64<T>) r.raw = _mm512_slli_epi64(a.raw, kBits);   // F
  else {  // i8/u8: shift as epi16 then clear the low kBits of each high byte that leaked in
    const __m512i shifted = _mm512_slli_epi16(a.raw, kBits);
    r.raw = _mm512_and_si512(shifted, _mm512_set1_epi8(static_cast<char>(0xFFu << kBits)));
  }
  return r;
}
template <int kBits, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> ShiftRight(Vec512<T, N> a) noexcept {
  static_assert(std::is_integral_v<T>, "shifts are integer-lane only");
  Vec512<T, N> r;
  if constexpr (detail::kI16<T>) {
    if constexpr (detail::kSigned<T>) r.raw = _mm512_srai_epi16(a.raw, kBits);   // BW
    else r.raw = _mm512_srli_epi16(a.raw, kBits);
  } else if constexpr (detail::kI32<T>) {
    if constexpr (detail::kSigned<T>) r.raw = _mm512_srai_epi32(a.raw, kBits);   // F
    else r.raw = _mm512_srli_epi32(a.raw, kBits);
  } else if constexpr (detail::kI64<T>) {
    // Unlike AVX2, _mm512_srai_epi64 IS native on AVX512F — no sign-fixup emulation needed.
    if constexpr (detail::kSigned<T>) r.raw = _mm512_srai_epi64(a.raw, kBits);   // F (native!)
    else r.raw = _mm512_srli_epi64(a.raw, kBits);
  } else {  // i8/u8 (no native epi8 shift on AVX-512 either)
    if constexpr (detail::kSigned<T>) {
      // i8 arithmetic right shift by k. Highway idiom: logical byte shift (epi16 shift +
      // mask off the k high bits that leaked across bytes), then OR in sign-fill where a<0.
      const __m512i logical =
          _mm512_and_si512(_mm512_srli_epi16(a.raw, kBits),
                           _mm512_set1_epi8(static_cast<char>(0xFFu >> kBits)));
      // AVX-512 cmp yields a k-mask; recover a 0xFF/0x00 byte vector via movm_epi8 (BW).
      const __mmask64 neg = _mm512_cmplt_epi8_mask(a.raw, _mm512_setzero_si512());  // a<0
      const __m512i is_neg = _mm512_movm_epi8(neg);                                 // 0xFF if a<0
      const __m512i signfill = _mm512_and_si512(
          is_neg, _mm512_set1_epi8(static_cast<char>(static_cast<unsigned char>(0xFFu << (8 - kBits)))));
      r.raw = _mm512_or_si512(logical, signfill);
    } else {  // u8 logical
      const __m512i shifted = _mm512_srli_epi16(a.raw, kBits);
      r.raw = _mm512_and_si512(shifted, _mm512_set1_epi8(static_cast<char>(0xFFu >> kBits)));
    }
  }
  return r;
}

// ---- per-lane variable shifts (Shl/Shr) ---------------------------------------------
// AVX-512 has FULL native variable shifts for 16/32/64 -- two wins over AVX2: (1) native
// 16-bit sllv/srlv/srav_epi16 (AVX512BW), which AVX2 lacked entirely (forced a scalar-store
// fallback); (2) native _mm512_srav_epi64 (AVX512F), which AVX2 also lacked (forced logical
// + sign fixup). So no scalar fallback here -- just static_assert-guard out 8-bit (no native
// sllv/srlv_epi8 on AVX-512 either, and no HYLE consumer). Signed Shr = arithmetic (srav),
// unsigned = logical (srlv), matching scalar.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> Shl(Vec512<T, N> a, Vec512<T, N> b) noexcept {
  static_assert(std::is_integral_v<T> && sizeof(T) >= 2,
                "avx512 Shl: variable shift is 16/32/64-bit only (no native sllv_epi8)");
  Vec512<T, N> r;
  if constexpr (detail::kI16<T>) r.raw = _mm512_sllv_epi16(a.raw, b.raw);        // BW (AVX2 lacked!)
  else if constexpr (detail::kI32<T>) r.raw = _mm512_sllv_epi32(a.raw, b.raw);   // F
  else r.raw = _mm512_sllv_epi64(a.raw, b.raw);                                  // F (i64)
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> Shr(Vec512<T, N> a, Vec512<T, N> b) noexcept {
  static_assert(std::is_integral_v<T> && sizeof(T) >= 2,
                "avx512 Shr: variable shift is 16/32/64-bit only (no native srlv/srav_epi8)");
  Vec512<T, N> r;
  if constexpr (std::is_same_v<T, std::uint16_t>) r.raw = _mm512_srlv_epi16(a.raw, b.raw);      // BW
  else if constexpr (std::is_same_v<T, std::int16_t>) r.raw = _mm512_srav_epi16(a.raw, b.raw);  // BW (AVX2 lacked!)
  else if constexpr (std::is_same_v<T, std::uint32_t>) r.raw = _mm512_srlv_epi32(a.raw, b.raw); // F
  else if constexpr (std::is_same_v<T, std::int32_t>) r.raw = _mm512_srav_epi32(a.raw, b.raw);  // F
  else if constexpr (std::is_same_v<T, std::uint64_t>) r.raw = _mm512_srlv_epi64(a.raw, b.raw); // F
  else r.raw = _mm512_srav_epi64(a.raw, b.raw);  // i64: native on AVX512F, unlike AVX2!
  return r;
}

// MulHigh: high 16 bits of the i16/u16 product (native mulhi, BW).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> MulHigh(Vec512<T, N> a, Vec512<T, N> b) noexcept {
  static_assert(std::is_integral_v<T> && sizeof(T) == 2, "MulHigh is 16-bit int only");
  Vec512<T, N> r;
  if constexpr (std::is_signed_v<T>) r.raw = _mm512_mulhi_epi16(a.raw, b.raw);
  else r.raw = _mm512_mulhi_epu16(a.raw, b.raw);
  return r;
}

// WidenMulPairwiseAdd(d32, a_i16, b_i16): _mm512_madd_epi16 (BW) multiplies i16 pairs and sums
// adjacent pairs into i32 -- the fixed-point dot core. a/b are FULL i16 (2N lanes), result N i32.
template <std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec512<std::int32_t, NDst> WidenMulPairwiseAdd(Simd<std::int32_t, NDst, 0>,
                                                                Vec512<SrcT, NSrc> a,
                                                                Vec512<SrcT, NSrc> b) noexcept {
  static_assert(std::is_same_v<SrcT, std::int16_t> && NSrc == 2 * NDst,
                "WidenMulPairwiseAdd: i16 -> i32 pairwise");
  Vec512<std::int32_t, NDst> r;
  r.raw = _mm512_madd_epi16(a.raw, b.raw);
  return r;
}

// SatWidenMulPairwiseAdd(di16, a_u8, b_i8): _mm512_maddubs_epi16 (BW) -- widen u8 a * i8 b,
// sum ADJACENT PAIRS into SATURATED i16. The int8 GEMM inner core (llama.cpp Q8 dot). a is
// UNSIGNED bytes, b is SIGNED bytes; dst i16 has half the lane count of the u8/i8 inputs.
template <std::size_t NDst, std::size_t NSrc>
HYLE_FORCE_INLINE Vec512<std::int16_t, NDst> SatWidenMulPairwiseAdd(
    Simd<std::int16_t, NDst, 0>, Vec512<std::uint8_t, NSrc> a,
    Vec512<std::int8_t, NSrc> b) noexcept {
  static_assert(NSrc == 2 * NDst, "SatWidenMulPairwiseAdd: u8/i8 -> i16 pairwise");
  Vec512<std::int16_t, NDst> r;
  r.raw = _mm512_maddubs_epi16(a.raw, b.raw);  // native u8*i8 -> saturated i16 pairwise
  return r;
}

// ReorderWidenMulAccumulate(dw, a, b, sum0, sum1): widen i16 a*b pairwise and add all products
// into sum0 (returned); sum1 is left untouched (matches scalar P). madd_epi16 already folds the
// adjacent pair, so the pairwise sum lands in sum0. Callers zero sum1 + finalize with Rearrange.
template <std::size_t NW, std::size_t NSrc>
HYLE_FORCE_INLINE Vec512<std::int32_t, NW> ReorderWidenMulAccumulate(
    Simd<std::int32_t, NW, 0>, Vec512<std::int16_t, NSrc> a, Vec512<std::int16_t, NSrc> b,
    Vec512<std::int32_t, NW> sum0, Vec512<std::int32_t, NW>& /*sum1*/) noexcept {
  static_assert(NSrc == 2 * NW, "ReorderWidenMulAccumulate: i16 -> i32 (dst half lanes)");
  sum0.raw = _mm512_add_epi32(sum0.raw, _mm512_madd_epi16(a.raw, b.raw));
  return sum0;  // all products in sum0; sum1 stays as the caller zeroed it
}

// RearrangeToOddPlusEven(sum0, sum1): finalize the accumulate. Our P put everything in sum0,
// so this is sum0 + sum1.
template <std::size_t NW>
HYLE_FORCE_INLINE Vec512<std::int32_t, NW> RearrangeToOddPlusEven(
    Vec512<std::int32_t, NW> sum0, Vec512<std::int32_t, NW> sum1) noexcept {
  Vec512<std::int32_t, NW> r;
  r.raw = _mm512_add_epi32(sum0.raw, sum1.raw);
  return r;
}

// SumOfMulQuadAccumulate(di32, a_i8, b_i8, sum): sum[i] += 4-way i8*i8 dot. Native vpdpbusd needs
// VNNI + is u8*i8 (wrong signedness for i8*i8); correctness-first store+reload scalar loop.
template <std::size_t NDst, std::size_t NSrc>
HYLE_FORCE_INLINE Vec512<std::int32_t, NDst> SumOfMulQuadAccumulate(
    Simd<std::int32_t, NDst, 0>, Vec512<std::int8_t, NSrc> a, Vec512<std::int8_t, NSrc> b,
    Vec512<std::int32_t, NDst> sum) noexcept {
  static_assert(NSrc == 4 * NDst, "SumOfMulQuadAccumulate: i8 -> i32 (dst quarter lanes)");
  alignas(64) std::int8_t ta[NSrc], tb[NSrc];
  alignas(64) std::int32_t ts[NDst];
  _mm512_store_si512(ta, a.raw);
  _mm512_store_si512(tb, b.raw);
  _mm512_store_si512(ts, sum.raw);
  for (std::size_t i = 0; i < NDst; ++i)
    for (std::size_t k = 0; k < 4; ++k)
      ts[i] += static_cast<std::int32_t>(ta[4 * i + k]) * tb[4 * i + k];
  sum.raw = _mm512_load_si512(ts);
  return sum;
}

// AbsDiff: |a-b| per lane. Integers via Sub(Max,Min) (Highway's generic); float via Abs(Sub).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> AbsDiff(Vec512<T, N> a, Vec512<T, N> b) noexcept {
  if constexpr (std::is_floating_point_v<T>) return Abs(Sub(a, b));
  else return Sub(Max(a, b), Min(a, b));
}

// ---- saturated arithmetic + rounding average (8/16-bit integer lanes) ---------------
// Native AVX-512-BW: adds/subs_epi8/epu8/epi16/epu16 (saturating), avg_epu8/epu16 (rounded).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> SaturatedAdd(Vec512<T, N> a, Vec512<T, N> b) noexcept {
  static_assert(std::is_integral_v<T> && sizeof(T) <= 2, "SaturatedAdd is 8/16-bit int only");
  Vec512<T, N> r;
  if constexpr (std::is_same_v<T, std::int8_t>) r.raw = _mm512_adds_epi8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint8_t>) r.raw = _mm512_adds_epu8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) r.raw = _mm512_adds_epi16(a.raw, b.raw);
  else r.raw = _mm512_adds_epu16(a.raw, b.raw);  // u16
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> SaturatedSub(Vec512<T, N> a, Vec512<T, N> b) noexcept {
  static_assert(std::is_integral_v<T> && sizeof(T) <= 2, "SaturatedSub is 8/16-bit int only");
  Vec512<T, N> r;
  if constexpr (std::is_same_v<T, std::int8_t>) r.raw = _mm512_subs_epi8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::uint8_t>) r.raw = _mm512_subs_epu8(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, std::int16_t>) r.raw = _mm512_subs_epi16(a.raw, b.raw);
  else r.raw = _mm512_subs_epu16(a.raw, b.raw);  // u16
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> AverageRound(Vec512<T, N> a, Vec512<T, N> b) noexcept {
  static_assert(std::is_unsigned_v<T> && sizeof(T) <= 2, "AverageRound is u8/u16 only");
  Vec512<T, N> r;
  if constexpr (std::is_same_v<T, std::uint8_t>) r.raw = _mm512_avg_epu8(a.raw, b.raw);
  else r.raw = _mm512_avg_epu16(a.raw, b.raw);  // u16
  return r;
}

// ---- bit-cast (reinterpret 512 bits as another lane type; zero-cost register cast) --
// Free _mm512_cast* between __m512/__m512i/__m512d, plain move within integer widths.
// Total bytes preserved (64) so lane count tracks width.
namespace detail {
template <typename DstRaw, typename SrcRaw>
HYLE_FORCE_INLINE DstRaw bitcast_raw_avx512(SrcRaw s) noexcept {
  if constexpr (std::is_same_v<DstRaw, SrcRaw>) return s;
  else if constexpr (std::is_same_v<SrcRaw, __m512> && std::is_same_v<DstRaw, __m512i>)
    return _mm512_castps_si512(s);
  else if constexpr (std::is_same_v<SrcRaw, __m512d> && std::is_same_v<DstRaw, __m512i>)
    return _mm512_castpd_si512(s);
  else if constexpr (std::is_same_v<SrcRaw, __m512i> && std::is_same_v<DstRaw, __m512>)
    return _mm512_castsi512_ps(s);
  else if constexpr (std::is_same_v<SrcRaw, __m512i> && std::is_same_v<DstRaw, __m512d>)
    return _mm512_castsi512_pd(s);
  else if constexpr (std::is_same_v<SrcRaw, __m512> && std::is_same_v<DstRaw, __m512d>)
    return _mm512_castps_pd(s);
  else if constexpr (std::is_same_v<SrcRaw, __m512d> && std::is_same_v<DstRaw, __m512>)
    return _mm512_castpd_ps(s);
  else static_assert(sizeof(DstRaw) == 0, "avx512 bitcast_raw: unhandled raw pair");
}
}  // namespace detail
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec512<DstT, NDst> BitCast(Simd<DstT, NDst, 0>, Vec512<SrcT, NSrc> v) noexcept {
  static_assert(sizeof(DstT) * NDst == sizeof(SrcT) * NSrc, "BitCast preserves total bytes");
  Vec512<DstT, NDst> r;
  r.raw = detail::bitcast_raw_avx512<detail::Raw512T<DstT>>(v.raw);
  return r;
}

// ---- type conversion (same lane count: f32<->i32) -----------------------------------
// ConvertTo: float<->int truncating (cvtt = toward zero). RoundToInt: f32->i32 round to
// nearest-even (cvtps respects the default MXCSR rounding = RNE). 16 lanes both sides =
// one __m512 <-> one __m512i, so it fits the native-width model (no partial vectors).
// Width-changing PromoteTo/DemoteTo are deferred (need partial vectors) -- see scalar.h.
template <typename DstT, std::size_t N, typename SrcT>
HYLE_FORCE_INLINE Vec512<DstT, N> ConvertTo(Simd<DstT, N, 0>, Vec512<SrcT, N> v) noexcept {
  Vec512<DstT, N> r;
  if constexpr (std::is_same_v<SrcT, float> && std::is_same_v<DstT, std::int32_t>)
    r.raw = _mm512_cvttps_epi32(v.raw);        // truncate toward zero
  else if constexpr (std::is_same_v<SrcT, std::int32_t> && std::is_same_v<DstT, float>)
    r.raw = _mm512_cvtepi32_ps(v.raw);
  else static_assert(sizeof(DstT) == 0, "avx512 ConvertTo: only f32<->i32 (same width) supported");
  return r;
}
template <typename DstT, std::size_t N, typename SrcT>
HYLE_FORCE_INLINE Vec512<DstT, N> RoundToInt(Simd<DstT, N, 0>, Vec512<SrcT, N> v) noexcept {
  static_assert(std::is_same_v<SrcT, float> && std::is_same_v<DstT, std::int32_t>,
                "avx512 RoundToInt: f32->i32");
  Vec512<DstT, N> r;
  r.raw = _mm512_cvtps_epi32(v.raw);           // round to nearest even (default MXCSR)
  return r;
}

// ---- width-changing promote / demote (half-at-a-time, fixed-width friendly) ---------
// PromoteLowerTo/PromoteUpperTo: FULL narrow src (2N lanes in a 512b reg) -> FULL wide dst
// (N lanes). The AVX-512 cvtep* intrinsics take a HALF-WIDTH 256b input and widen it to a
// full 512b wide vector, so we feed the lower/upper 256b half (N narrow lanes). Signed src
// sign-extends (cvtepi), unsigned zero-extends (cvtepu). f32->f64 uses cvtps_pd on a 256b
// half. Lower half = cast512->256; upper half = extracti64x4 (int) / extractf32x8 (float,DQ).
namespace detail {
template <typename DstT, typename SrcT>
HYLE_FORCE_INLINE Raw512T<DstT> promote256(__m256i half) noexcept {
  if constexpr (std::is_same_v<SrcT, std::int8_t>) return _mm512_cvtepi8_epi16(half);    // BW
  else if constexpr (std::is_same_v<SrcT, std::uint8_t>) return _mm512_cvtepu8_epi16(half);  // BW
  else if constexpr (std::is_same_v<SrcT, std::int16_t>) return _mm512_cvtepi16_epi32(half);  // F
  else if constexpr (std::is_same_v<SrcT, std::uint16_t>) return _mm512_cvtepu16_epi32(half);  // F
  else if constexpr (std::is_same_v<SrcT, std::int32_t>) return _mm512_cvtepi32_epi64(half);  // F
  else return _mm512_cvtepu32_epi64(half);  // u32 -> u64  (F)
}
}  // namespace detail
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec512<DstT, NDst> PromoteLowerTo(Simd<DstT, NDst, 0>, Vec512<SrcT, NSrc> v) noexcept {
  static_assert(NSrc == 2 * NDst && sizeof(DstT) > sizeof(SrcT), "PromoteLowerTo widens, src 2x lanes");
  Vec512<DstT, NDst> r;
  if constexpr (std::is_same_v<SrcT, float>) r.raw = _mm512_cvtps_pd(_mm512_castps512_ps256(v.raw));
  else r.raw = detail::promote256<DstT, SrcT>(_mm512_castsi512_si256(v.raw));  // lower 256
  return r;
}
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec512<DstT, NDst> PromoteUpperTo(Simd<DstT, NDst, 0>, Vec512<SrcT, NSrc> v) noexcept {
  static_assert(NSrc == 2 * NDst && sizeof(DstT) > sizeof(SrcT), "PromoteUpperTo widens, src 2x lanes");
  Vec512<DstT, NDst> r;
  if constexpr (std::is_same_v<SrcT, float>)
    r.raw = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v.raw, 1));  // upper 256 (DQ)
  else r.raw = detail::promote256<DstT, SrcT>(_mm512_extracti64x4_epi64(v.raw, 1));  // upper 256 (F)
  return r;
}
// DemoteTo(dn, a, b): two FULL wide vectors -> one FULL narrow, SATURATING (integer) with
// lanes ORDERED [a..., b...]. THE 512-bit CATCH: AVX-512 packs/packus operate WITHIN EACH of
// the FOUR independent 128-bit lanes, NOT across the whole register. In 128-lane L the pack
// puts a_L's results in the low 64 bits and b_L's in the high 64 bits, so the eight 64-bit
// groups of `packed` come out interleaved a0,b0,a1,b1,a2,b2,a3,b3. To recover the flat
// [a0,a1,a2,a3, b0,b1,b2,b3] order we permute the 64-bit groups by idx=(0,2,4,6,1,3,5,7) via
// permutexvar_epi64 (F). Same idx for every integer width -- the interleave pattern is
// identical whether the pack was epi16 or epi32. f64->f32 needs NO permute: cvtpd_ps yields
// a naturally-contiguous 256b half per src, and insertf32x8 (DQ) stacks b above a.
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec512<DstT, NDst> DemoteTo(Simd<DstT, NDst, 0>, Vec512<SrcT, NSrc> a,
                                              Vec512<SrcT, NSrc> b) noexcept {
  static_assert(NDst == 2 * NSrc, "DemoteTo: narrow dst has 2x each wide src's lanes");
  Vec512<DstT, NDst> r;
  if constexpr (std::is_same_v<SrcT, double>) {  // f64 -> f32 (contiguous per src, no permute)
    r.raw = _mm512_insertf32x8(_mm512_castps256_ps512(_mm512_cvtpd_ps(a.raw)),
                               _mm512_cvtpd_ps(b.raw), 1);  // DQ
  } else {
    __m512i packed;
    if constexpr (std::is_same_v<SrcT, std::int16_t> && std::is_same_v<DstT, std::int8_t>)
      packed = _mm512_packs_epi16(a.raw, b.raw);   // BW
    else if constexpr (std::is_same_v<SrcT, std::int16_t> && std::is_same_v<DstT, std::uint8_t>)
      packed = _mm512_packus_epi16(a.raw, b.raw);  // BW
    else if constexpr (std::is_same_v<SrcT, std::int32_t> && std::is_same_v<DstT, std::int16_t>)
      packed = _mm512_packs_epi32(a.raw, b.raw);   // F
    else if constexpr (std::is_same_v<SrcT, std::int32_t> && std::is_same_v<DstT, std::uint16_t>)
      packed = _mm512_packus_epi32(a.raw, b.raw);  // F
    else static_assert(sizeof(DstT) == 0, "avx512 DemoteTo: unsupported width pair");
    // undo the 4-way-128-lane interleave: 64-bit groups a0,b0,a1,b1,a2,b2,a3,b3 -> a0..a3,b0..b3
    const __m512i idx = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    r.raw = _mm512_permutexvar_epi64(idx, packed);  // F
  }
  return r;
}
// ReorderDemote2To: the cheaper block-interleaved order [a0,b0,a1,b1,a2,b2,a3,b3] straight out
// of the packs (skips the permutexvar) -- for callers that immediately reduce/re-permute and
// don't care about order. Integer only.
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec512<DstT, NDst> ReorderDemote2To(Simd<DstT, NDst, 0>, Vec512<SrcT, NSrc> a,
                                                      Vec512<SrcT, NSrc> b) noexcept {
  static_assert(NDst == 2 * NSrc && std::is_integral_v<SrcT>, "ReorderDemote2To: integer 2->1");
  Vec512<DstT, NDst> r;
  if constexpr (std::is_same_v<SrcT, std::int16_t> && std::is_same_v<DstT, std::int8_t>)
    r.raw = _mm512_packs_epi16(a.raw, b.raw);
  else if constexpr (std::is_same_v<SrcT, std::int16_t> && std::is_same_v<DstT, std::uint8_t>)
    r.raw = _mm512_packus_epi16(a.raw, b.raw);
  else if constexpr (std::is_same_v<SrcT, std::int32_t> && std::is_same_v<DstT, std::int16_t>)
    r.raw = _mm512_packs_epi32(a.raw, b.raw);
  else r.raw = _mm512_packus_epi32(a.raw, b.raw);  // i32 -> u16
  return r;
}
// TruncateTo/PromoteEvenTo/PromoteOddTo/U8FromU32: strided/modular width changers.
// correctness-first store+reload; native pack/cvt narrowing is a profiling upgrade.
// Store()/Load() move a FULL 512-bit register (_mm512_store_si512 / _mm512_load_si512), so
// the scratch buffers MUST be a whole register wide (64 B) regardless of logical lane count:
// a narrow partial result still reloads 64 B via _mm512_load_si512, so a lane-count-sized
// `DstT dst[N]` would be an out-of-bounds stack read. Size every buffer to 64 B. (Same class
// of bug the x86-128/256 backends segfaulted on -- Vec.TruncateAndStridedPromote under GCC.)
template <typename DstT, std::size_t N, typename SrcT>
HYLE_FORCE_INLINE Vec512<DstT, N> TruncateTo(Simd<DstT, N, 0>, Vec512<SrcT, N> v) noexcept {
  static_assert(sizeof(DstT) < sizeof(SrcT), "TruncateTo narrows the lane type");
  alignas(64) SrcT src[64 / sizeof(SrcT)];
  alignas(64) DstT dst[64 / sizeof(DstT)] = {};
  Store(v, Simd<SrcT, N, 0>(), src);
  for (std::size_t i = 0; i < N; ++i) dst[i] = static_cast<DstT>(src[i]);
  return Load(Simd<DstT, N, 0>(), dst);
}
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec512<DstT, NDst> PromoteEvenTo(Simd<DstT, NDst, 0>, Vec512<SrcT, NSrc> v) noexcept {
  static_assert(NSrc == 2 * NDst && sizeof(DstT) > sizeof(SrcT), "PromoteEvenTo: even lanes narrow->wide");
  alignas(64) SrcT src[64 / sizeof(SrcT)];
  alignas(64) DstT dst[64 / sizeof(DstT)] = {};
  Store(v, Simd<SrcT, NSrc, 0>(), src);
  for (std::size_t i = 0; i < NDst; ++i) dst[i] = static_cast<DstT>(src[2 * i]);
  return Load(Simd<DstT, NDst, 0>(), dst);
}
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec512<DstT, NDst> PromoteOddTo(Simd<DstT, NDst, 0>, Vec512<SrcT, NSrc> v) noexcept {
  static_assert(NSrc == 2 * NDst && sizeof(DstT) > sizeof(SrcT), "PromoteOddTo: odd lanes narrow->wide");
  alignas(64) SrcT src[64 / sizeof(SrcT)];
  alignas(64) DstT dst[64 / sizeof(DstT)] = {};
  Store(v, Simd<SrcT, NSrc, 0>(), src);
  for (std::size_t i = 0; i < NDst; ++i) dst[i] = static_cast<DstT>(src[2 * i + 1]);
  return Load(Simd<DstT, NDst, 0>(), dst);
}
template <std::size_t N>
HYLE_FORCE_INLINE Vec512<std::uint8_t, N> U8FromU32(Vec512<std::uint32_t, N> v) noexcept {
  alignas(64) std::uint32_t src[64 / sizeof(std::uint32_t)];
  alignas(64) std::uint8_t dst[64] = {};
  Store(v, Simd<std::uint32_t, N, 0>(), src);
  for (std::size_t i = 0; i < N; ++i) dst[i] = static_cast<std::uint8_t>(src[i]);
  return Load(Simd<std::uint8_t, N, 0>(), dst);
}

// ---- swizzle (lane rearrangement) ---------------------------------------------------
// AVX-512 store+reload swizzles (correct + simple; not a proven hot path, and native 512b
// permutes are unvalidated without hardware). InterleaveLower/Upper are PER-128-BIT-BLOCK
// (Highway contract): 4 blocks in a 512b reg, interleave within each. Others are whole-vector.
// The scalar backend is the lane-for-lane oracle.
namespace detail {
template <typename T, std::size_t N, typename IdxFn>
HYLE_FORCE_INLINE Vec512<T, N> swizzle_via_mem(Vec512<T, N> v, IdxFn idx) noexcept {
  alignas(64) T src[N], dst[N];
  Store(v, Simd<T, N, 0>(), src);
  for (std::size_t i = 0; i < N; ++i) dst[i] = src[idx(i)];
  return Load(Simd<T, N, 0>(), dst);
}
}  // namespace detail
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> OddEven(Vec512<T, N> a, Vec512<T, N> b) noexcept {
  alignas(64) T sa[N], sb[N], dst[N];
  Store(a, Simd<T, N, 0>(), sa); Store(b, Simd<T, N, 0>(), sb);
  for (std::size_t i = 0; i < N; ++i) dst[i] = (i & 1) ? sa[i] : sb[i];
  return Load(Simd<T, N, 0>(), dst);
}
template <class D>
HYLE_FORCE_INLINE detail::V512<D> Reverse(D, detail::V512<D> v) noexcept {
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  return detail::swizzle_via_mem(v, [](std::size_t i) { return N - 1 - i; });
}
template <class D>
HYLE_FORCE_INLINE detail::V512<D> Reverse2(D, detail::V512<D> v) noexcept {
  return detail::swizzle_via_mem(v, [](std::size_t i) { return i ^ 1; });
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> InterleaveLower(Vec512<T, N> a, Vec512<T, N> b) noexcept {
  alignas(64) T sa[N], sb[N], dst[N];
  Store(a, Simd<T, N, 0>(), sa); Store(b, Simd<T, N, 0>(), sb);
  const std::size_t blk = 16 / sizeof(T);
  for (std::size_t i = 0; i < N; ++i) {
    std::size_t base = (i / blk) * blk, o = i % blk;
    dst[i] = (o & 1) ? sb[base + o / 2] : sa[base + o / 2];
  }
  return Load(Simd<T, N, 0>(), dst);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> InterleaveUpper(Vec512<T, N> a, Vec512<T, N> b) noexcept {
  alignas(64) T sa[N], sb[N], dst[N];
  Store(a, Simd<T, N, 0>(), sa); Store(b, Simd<T, N, 0>(), sb);
  const std::size_t blk = 16 / sizeof(T);
  for (std::size_t i = 0; i < N; ++i) {
    std::size_t base = (i / blk) * blk, o = i % blk;
    dst[i] = (o & 1) ? sb[base + blk / 2 + o / 2] : sa[base + blk / 2 + o / 2];
  }
  return Load(Simd<T, N, 0>(), dst);
}
// Concat{Lower,Upper}{Lower,Upper}(hi,lo): whole-vector halves (Highway arg order hi,lo).
// result-lower <- a chosen half of lo; result-upper <- a chosen half of hi.
namespace detail {
template <int loHalf, int hiHalf, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> concat_halves(Vec512<T, N> hi, Vec512<T, N> lo) noexcept {
  alignas(64) T sh[N], sl[N], dst[N];
  Store(hi, Simd<T, N, 0>(), sh); Store(lo, Simd<T, N, 0>(), sl);
  for (std::size_t i = 0; i < N / 2; ++i) {
    dst[i] = sl[loHalf * (N / 2) + i];
    dst[N / 2 + i] = sh[hiHalf * (N / 2) + i];
  }
  return Load(Simd<T, N, 0>(), dst);
}
}  // namespace detail
template <typename T, std::size_t N> HYLE_FORCE_INLINE Vec512<T, N> ConcatLowerLower(Vec512<T, N> hi, Vec512<T, N> lo) noexcept { return detail::concat_halves<0,0>(hi, lo); }
template <typename T, std::size_t N> HYLE_FORCE_INLINE Vec512<T, N> ConcatUpperUpper(Vec512<T, N> hi, Vec512<T, N> lo) noexcept { return detail::concat_halves<1,1>(hi, lo); }
template <typename T, std::size_t N> HYLE_FORCE_INLINE Vec512<T, N> ConcatLowerUpper(Vec512<T, N> hi, Vec512<T, N> lo) noexcept { return detail::concat_halves<1,0>(hi, lo); }
template <typename T, std::size_t N> HYLE_FORCE_INLINE Vec512<T, N> ConcatUpperLower(Vec512<T, N> hi, Vec512<T, N> lo) noexcept { return detail::concat_halves<0,1>(hi, lo); }
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> ConcatEven(Vec512<T, N> hi, Vec512<T, N> lo) noexcept {
  alignas(64) T sh[N], sl[N], dst[N];
  Store(hi, Simd<T, N, 0>(), sh); Store(lo, Simd<T, N, 0>(), sl);
  for (std::size_t i = 0; i < N / 2; ++i) { dst[i] = sl[2 * i]; dst[N / 2 + i] = sh[2 * i]; }
  return Load(Simd<T, N, 0>(), dst);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> ConcatOdd(Vec512<T, N> hi, Vec512<T, N> lo) noexcept {
  alignas(64) T sh[N], sl[N], dst[N];
  Store(hi, Simd<T, N, 0>(), sh); Store(lo, Simd<T, N, 0>(), sl);
  for (std::size_t i = 0; i < N / 2; ++i) { dst[i] = sl[2 * i + 1]; dst[N / 2 + i] = sh[2 * i + 1]; }
  return Load(Simd<T, N, 0>(), dst);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> DupEven(Vec512<T, N> v) noexcept { return detail::swizzle_via_mem(v, [](std::size_t i){ return i & ~std::size_t{1}; }); }
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> DupOdd(Vec512<T, N> v) noexcept { return detail::swizzle_via_mem(v, [](std::size_t i){ return (i & ~std::size_t{1}) | 1; }); }
template <typename T, std::size_t N>
struct Indices512 { Vec512<T, N> idx; };
template <class D>
HYLE_FORCE_INLINE Indices512<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> IndicesFromVec(D, detail::V512<D> iv) noexcept {
  return Indices512<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)>{iv};
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> TableLookupLanes(Vec512<T, N> v, Indices512<T, N> ind) noexcept {
  alignas(64) T src[N], idx[N], dst[N];
  Store(v, Simd<T, N, 0>(), src); Store(ind.idx, Simd<T, N, 0>(), idx);
  for (std::size_t i = 0; i < N; ++i) dst[i] = src[static_cast<std::size_t>(idx[i]) & (N - 1)];
  return Load(Simd<T, N, 0>(), dst);
}

// ---- byte table lookup (PSHUFB-style, per-128-bit block) ----------------------------
// _mm512_shuffle_epi8 is PSHUFB replicated across all four 128-bit lanes, so output byte i
// selects bytes[(i/16)*16 + (idx_i & 15)] WITHIN its own lane -- exactly the scalar block-local
// contract. PSHUFB natively zeroes the output byte when the index's bit7 (0x80) is set, which
// IS the Or0 variant. TableLookupBytes (no zeroing) first masks the index to its low 4 bits,
// clearing bit7 and forcing a block-local index. Index and data lane types may differ; both are
// viewed as __m512i byte arrays and the result is cast back to the index's lane type. (AVX512BW.)
template <typename T, std::size_t N, typename TI, std::size_t NI>
HYLE_FORCE_INLINE Vec512<TI, NI> TableLookupBytes(Vec512<T, N> bytes, Vec512<TI, NI> idx) noexcept {
  static_assert(sizeof(T) * N == sizeof(TI) * NI, "TableLookupBytes: same total byte width");
  const __m512i b = detail::bitcast_raw_avx512<__m512i>(bytes.raw);
  const __m512i lo = _mm512_and_si512(detail::bitcast_raw_avx512<__m512i>(idx.raw),
                                      _mm512_set1_epi8(0x0F));  // clear bit7 + force block-local
  Vec512<TI, NI> r;
  r.raw = detail::bitcast_raw_avx512<detail::Raw512T<TI>>(_mm512_shuffle_epi8(b, lo));
  return r;
}
template <typename T, std::size_t N, typename TI, std::size_t NI>
HYLE_FORCE_INLINE Vec512<TI, NI> TableLookupBytesOr0(Vec512<T, N> bytes, Vec512<TI, NI> idx) noexcept {
  static_assert(sizeof(T) * N == sizeof(TI) * NI, "TableLookupBytesOr0: same total byte width");
  const __m512i b = detail::bitcast_raw_avx512<__m512i>(bytes.raw);
  const __m512i i = detail::bitcast_raw_avx512<__m512i>(idx.raw);
  Vec512<TI, NI> r;
  r.raw = detail::bitcast_raw_avx512<detail::Raw512T<TI>>(_mm512_shuffle_epi8(b, i));  // bit7 -> 0
  return r;
}

// ---- comparison -> k-register mask --------------------------------------------------
// AVX-512 compares WRITE A K REGISTER directly (vcmpps/vpcmpd ... -> __mmask). No xor-MSB
// for unsigned: there are native epu8/16/32/64 mask compares. Per-width detail helpers pick
// the intrinsic by lane width + signedness (mirroring avx2.h's icmp* helpers, but the result
// is a k-reg, not a vector). BW gives the 8/16-bit mask compares; F/DQ the 64-bit ones.
namespace detail {
template <typename T>
HYLE_FORCE_INLINE KMaskT<T> icmpeq(Raw512T<T> a, Raw512T<T> b) noexcept {
  if constexpr (kI8<T>) return _mm512_cmpeq_epi8_mask(a, b);
  else if constexpr (kI16<T>) return _mm512_cmpeq_epi16_mask(a, b);
  else if constexpr (kI64<T>) return _mm512_cmpeq_epi64_mask(a, b);
  else return _mm512_cmpeq_epi32_mask(a, b);
}
template <typename T>
HYLE_FORCE_INLINE KMaskT<T> icmpne(Raw512T<T> a, Raw512T<T> b) noexcept {
  if constexpr (kI8<T>) return _mm512_cmpneq_epi8_mask(a, b);
  else if constexpr (kI16<T>) return _mm512_cmpneq_epi16_mask(a, b);
  else if constexpr (kI64<T>) return _mm512_cmpneq_epi64_mask(a, b);
  else return _mm512_cmpneq_epi32_mask(a, b);
}
// Gt: native signed epi* and unsigned epu* mask compares (no sign-flip trick needed).
template <typename T>
HYLE_FORCE_INLINE KMaskT<T> icmpgt(Raw512T<T> a, Raw512T<T> b) noexcept {
  if constexpr (kSigned<T>) {
    if constexpr (kI8<T>) return _mm512_cmpgt_epi8_mask(a, b);
    else if constexpr (kI16<T>) return _mm512_cmpgt_epi16_mask(a, b);
    else if constexpr (kI64<T>) return _mm512_cmpgt_epi64_mask(a, b);
    else return _mm512_cmpgt_epi32_mask(a, b);
  } else {
    if constexpr (kI8<T>) return _mm512_cmpgt_epu8_mask(a, b);
    else if constexpr (kI16<T>) return _mm512_cmpgt_epu16_mask(a, b);
    else if constexpr (kI64<T>) return _mm512_cmpgt_epu64_mask(a, b);
    else return _mm512_cmpgt_epu32_mask(a, b);
  }
}
template <typename T>
HYLE_FORCE_INLINE KMaskT<T> icmpge(Raw512T<T> a, Raw512T<T> b) noexcept {
  if constexpr (kSigned<T>) {
    if constexpr (kI8<T>) return _mm512_cmpge_epi8_mask(a, b);
    else if constexpr (kI16<T>) return _mm512_cmpge_epi16_mask(a, b);
    else if constexpr (kI64<T>) return _mm512_cmpge_epi64_mask(a, b);
    else return _mm512_cmpge_epi32_mask(a, b);
  } else {
    if constexpr (kI8<T>) return _mm512_cmpge_epu8_mask(a, b);
    else if constexpr (kI16<T>) return _mm512_cmpge_epu16_mask(a, b);
    else if constexpr (kI64<T>) return _mm512_cmpge_epu64_mask(a, b);
    else return _mm512_cmpge_epu32_mask(a, b);
  }
}
}  // namespace detail

template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask512<T, N> Eq(Vec512<T, N> a, Vec512<T, N> b) noexcept {
  Mask512<T, N> m;
  if constexpr (std::is_same_v<T, float>) m.raw = _mm512_cmp_ps_mask(a.raw, b.raw, _CMP_EQ_OQ);
  else if constexpr (std::is_same_v<T, double>) m.raw = _mm512_cmp_pd_mask(a.raw, b.raw, _CMP_EQ_OQ);
  else m.raw = detail::icmpeq<T>(a.raw, b.raw);
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask512<T, N> Ne(Vec512<T, N> a, Vec512<T, N> b) noexcept {
  Mask512<T, N> m;
  // _CMP_NEQ_UQ (unordered): NaN compares not-equal -> true, matching scalar (!=),
  // SSE4 (_mm_cmpneq_ps) and NEON (~Eq). _CMP_NEQ_OQ (ordered) wrongly returns false on NaN.
  if constexpr (std::is_same_v<T, float>) m.raw = _mm512_cmp_ps_mask(a.raw, b.raw, _CMP_NEQ_UQ);
  else if constexpr (std::is_same_v<T, double>) m.raw = _mm512_cmp_pd_mask(a.raw, b.raw, _CMP_NEQ_UQ);
  else m.raw = detail::icmpne<T>(a.raw, b.raw);
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask512<T, N> Gt(Vec512<T, N> a, Vec512<T, N> b) noexcept {
  Mask512<T, N> m;
  if constexpr (std::is_same_v<T, float>) m.raw = _mm512_cmp_ps_mask(a.raw, b.raw, _CMP_GT_OQ);
  else if constexpr (std::is_same_v<T, double>) m.raw = _mm512_cmp_pd_mask(a.raw, b.raw, _CMP_GT_OQ);
  else m.raw = detail::icmpgt<T>(a.raw, b.raw);
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask512<T, N> Lt(Vec512<T, N> a, Vec512<T, N> b) noexcept {
  Mask512<T, N> m;
  if constexpr (std::is_same_v<T, float>) m.raw = _mm512_cmp_ps_mask(a.raw, b.raw, _CMP_LT_OQ);
  else if constexpr (std::is_same_v<T, double>) m.raw = _mm512_cmp_pd_mask(a.raw, b.raw, _CMP_LT_OQ);
  else m.raw = detail::icmpgt<T>(b.raw, a.raw);  // Lt(a,b) = Gt(b,a)
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask512<T, N> Ge(Vec512<T, N> a, Vec512<T, N> b) noexcept {
  Mask512<T, N> m;
  if constexpr (std::is_same_v<T, float>) m.raw = _mm512_cmp_ps_mask(a.raw, b.raw, _CMP_GE_OQ);
  else if constexpr (std::is_same_v<T, double>) m.raw = _mm512_cmp_pd_mask(a.raw, b.raw, _CMP_GE_OQ);
  else m.raw = detail::icmpge<T>(a.raw, b.raw);
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask512<T, N> Le(Vec512<T, N> a, Vec512<T, N> b) noexcept {
  Mask512<T, N> m;
  if constexpr (std::is_same_v<T, float>) m.raw = _mm512_cmp_ps_mask(a.raw, b.raw, _CMP_LE_OQ);
  else if constexpr (std::is_same_v<T, double>) m.raw = _mm512_cmp_pd_mask(a.raw, b.raw, _CMP_LE_OQ);
  else m.raw = detail::icmpge<T>(b.raw, a.raw);  // Le(a,b) = Ge(b,a)
  return m;
}

// ---- floating-point classification -> mask (k-register) -----------------------------
// Portable cmp_mask form (not _mm512_fpclass_*): the ordered/unordered compare predicates
// are unambiguously correct and avoid fpclass category-bit pitfalls. IsNaN via _CMP_UNORD_Q
// (v vs v: true iff NaN). IsInf: |v| == +inf (_CMP_EQ_OQ, ordered so NaN->false). IsFinite:
// |v| < +inf (_CMP_LT_OQ: NaN->false, inf->false, finite->true). IsEitherNaN: a vs b
// _CMP_UNORD_Q (true iff a or b is NaN) -- the native unordered predicate. Result __mmask
// wrapped in Mask512 exactly as Eq/Lt do. f32 -> __mmask16, f64 -> __mmask8 (KMaskT).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask512<T, N> IsNaN(Vec512<T, N> v) noexcept {
  static_assert(std::is_floating_point_v<T>, "IsNaN is float-only");
  Mask512<T, N> m;
  if constexpr (std::is_same_v<T, float>) m.raw = _mm512_cmp_ps_mask(v.raw, v.raw, _CMP_UNORD_Q);
  else m.raw = _mm512_cmp_pd_mask(v.raw, v.raw, _CMP_UNORD_Q);
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask512<T, N> IsInf(Vec512<T, N> v) noexcept {
  static_assert(std::is_floating_point_v<T>, "IsInf is float-only");
  Mask512<T, N> m;
  const T inf = std::numeric_limits<T>::infinity();
  if constexpr (std::is_same_v<T, float>)
    m.raw = _mm512_cmp_ps_mask(_mm512_abs_ps(v.raw), _mm512_set1_ps(inf), _CMP_EQ_OQ);
  else
    m.raw = _mm512_cmp_pd_mask(_mm512_abs_pd(v.raw), _mm512_set1_pd(inf), _CMP_EQ_OQ);
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask512<T, N> IsFinite(Vec512<T, N> v) noexcept {
  static_assert(std::is_floating_point_v<T>, "IsFinite is float-only");
  Mask512<T, N> m;
  const T inf = std::numeric_limits<T>::infinity();
  if constexpr (std::is_same_v<T, float>)
    m.raw = _mm512_cmp_ps_mask(_mm512_abs_ps(v.raw), _mm512_set1_ps(inf), _CMP_LT_OQ);
  else
    m.raw = _mm512_cmp_pd_mask(_mm512_abs_pd(v.raw), _mm512_set1_pd(inf), _CMP_LT_OQ);
  return m;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask512<T, N> IsEitherNaN(Vec512<T, N> a, Vec512<T, N> b) noexcept {
  static_assert(std::is_floating_point_v<T>, "IsEitherNaN is float-only");
  Mask512<T, N> m;
  if constexpr (std::is_same_v<T, float>) m.raw = _mm512_cmp_ps_mask(a.raw, b.raw, _CMP_UNORD_Q);
  else m.raw = _mm512_cmp_pd_mask(a.raw, b.raw, _CMP_UNORD_Q);
  return m;
}

// ---- mask ops (k-register; vblendmps selects on mask bits) --------------------------
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> IfThenElse(Mask512<T, N> m, Vec512<T, N> yes, Vec512<T, N> no) noexcept {
  Vec512<T, N> r;  // mask_blend(k, no, yes): lanes with k bit set take 'yes'
  if constexpr (std::is_same_v<T, float>) r.raw = _mm512_mask_blend_ps(m.raw, no.raw, yes.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm512_mask_blend_pd(m.raw, no.raw, yes.raw);
  else if constexpr (detail::kI8<T>) r.raw = _mm512_mask_blend_epi8(m.raw, no.raw, yes.raw);   // BW
  else if constexpr (detail::kI16<T>) r.raw = _mm512_mask_blend_epi16(m.raw, no.raw, yes.raw); // BW
  else if constexpr (detail::kI64<T>) r.raw = _mm512_mask_blend_epi64(m.raw, no.raw, yes.raw);
  else r.raw = _mm512_mask_blend_epi32(m.raw, no.raw, yes.raw);  // i32/u32
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> IfThenElseZero(Mask512<T, N> m, Vec512<T, N> yes) noexcept {
  return IfThenElse(m, yes, Zero(Simd<T, N, 0>()));
}

// k-register -> integer bits (the mask IS already a bitmask, just widen to uint64_t so an
// i8 mask's 64 lanes all fit -- __mmask64 needs the full 64 bits, unlike f32/i32's 16).
namespace detail {
template <typename T, std::size_t N>
HYLE_FORCE_INLINE std::uint64_t kbits(Mask512<T, N> m) noexcept {
  return static_cast<std::uint64_t>(m.raw);  // __mmask8/16/32/64 already a packed bitmask
}
}  // namespace detail
template <typename T, std::size_t N>
HYLE_FORCE_INLINE bool AllTrue(Simd<T, N, 0>, Mask512<T, N> m) noexcept {
  // N may be 64 (i8) -> (1ull<<64) is UB; compute the full-N mask safely (mirror avx2's guard).
  const std::uint64_t full = (N >= 64) ? ~0ull : ((std::uint64_t{1} << N) - 1u);
  return detail::kbits(m) == full;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE bool AllFalse(Simd<T, N, 0>, Mask512<T, N> m) noexcept {
  return detail::kbits(m) == 0u;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE std::size_t CountTrue(Simd<T, N, 0>, Mask512<T, N> m) noexcept {
  return static_cast<std::size_t>(std::popcount(detail::kbits(m)));
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE std::ptrdiff_t FindFirstTrue(Simd<T, N, 0>, Mask512<T, N> m) noexcept {
  const std::uint64_t b = detail::kbits(m);
  return b ? static_cast<std::ptrdiff_t>(std::countr_zero(b)) : -1;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE std::ptrdiff_t FindLastTrue(Simd<T, N, 0>, Mask512<T, N> m) noexcept {
  const std::uint64_t b = detail::kbits(m);
  return b ? static_cast<std::ptrdiff_t>(63 - std::countl_zero(b)) : -1;  // highest set lane
}
// VecFromMask: expand the k-register mask to a per-lane 0xFF../0 vector (vpmovm2b/w/d/q).
template <class D>
HYLE_FORCE_INLINE detail::V512<D> VecFromMask(D, Mask512<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m) noexcept {
  using T = TFromD<D>;
  detail::V512<D> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm512_castsi512_ps(_mm512_movm_epi32(m.raw));
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm512_castsi512_pd(_mm512_movm_epi64(m.raw));
  else if constexpr (detail::kI8<T>) r.raw = _mm512_movm_epi8(m.raw);    // BW
  else if constexpr (detail::kI16<T>) r.raw = _mm512_movm_epi16(m.raw);  // BW
  else if constexpr (detail::kI64<T>) r.raw = _mm512_movm_epi64(m.raw);  // DQ
  else r.raw = _mm512_movm_epi32(m.raw);  // i32/u32
  return r;
}

// ---- mask-domain family (simd-08 档A-2) -- AVX-512 k-register form -----------------------
// On AVX-512 a Mask512 IS a k-register bitmask (1 bit/lane), so these are just bitwise ops on
// the packed bits -- cleaner than the vector-mask backends (which pun through movemask). N may
// be up to 64 (i8), so full-mask math uses (N>=64 ? ~0ull : (1<<N)-1) to dodge the 1<<64 UB.
// These were missing from avx512.h (it froze at ~simd-07; the mask family landed in simd-08 A-2
// but avx512 had no gtest variant to expose the gap -- caught 2026-07-09 wiring the SDE run).
namespace detail {
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask512<T, N> mask_from_kbits(std::uint64_t bits) noexcept {
  Mask512<T, N> m;
  m.raw = static_cast<typename KMaskT<T>>(bits);
  return m;
}
template <std::size_t N>
HYLE_FORCE_INLINE std::uint64_t full_kmask() noexcept {
  return (N >= 64) ? ~0ull : ((std::uint64_t{1} << N) - 1ull);
}
}  // namespace detail

// Mask-domain boolean logic: bitwise on the k-register (mask AND out-of-range bits stay 0 since
// compares only ever set in-range lanes; Not/AndNot re-clamp to the N-lane window).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask512<T, N> And(Mask512<T, N> a, Mask512<T, N> b) noexcept {
  return detail::mask_from_kbits<T, N>(detail::kbits(a) & detail::kbits(b));
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask512<T, N> Or(Mask512<T, N> a, Mask512<T, N> b) noexcept {
  return detail::mask_from_kbits<T, N>(detail::kbits(a) | detail::kbits(b));
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask512<T, N> Xor(Mask512<T, N> a, Mask512<T, N> b) noexcept {
  return detail::mask_from_kbits<T, N>(detail::kbits(a) ^ detail::kbits(b));
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask512<T, N> AndNot(Mask512<T, N> a, Mask512<T, N> b) noexcept {
  return detail::mask_from_kbits<T, N>(~detail::kbits(a) & detail::kbits(b));  // Highway: ~a & b
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask512<T, N> Not(Mask512<T, N> a) noexcept {
  return detail::mask_from_kbits<T, N>(~detail::kbits(a) & detail::full_kmask<N>());
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask512<T, N> ExclusiveNeither(Mask512<T, N> a, Mask512<T, N> b) noexcept {
  return detail::mask_from_kbits<T, N>(~(detail::kbits(a) | detail::kbits(b)) & detail::full_kmask<N>());
}
// MaskFromVec: lane true iff its MSB set. Route via signed<0 compare on the reinterpreted vector.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask512<T, N> MaskFromVec(Vec512<T, N> v) noexcept {
  Mask512<T, N> m;
  if constexpr (std::is_same_v<T, float>)
    m.raw = _mm512_movepi32_mask(_mm512_castps_si512(v.raw));
  else if constexpr (std::is_same_v<T, double>)
    m.raw = _mm512_movepi64_mask(_mm512_castpd_si512(v.raw));
  else if constexpr (detail::kI8<T>) m.raw = _mm512_movepi8_mask(v.raw);
  else if constexpr (detail::kI16<T>) m.raw = _mm512_movepi16_mask(v.raw);
  else if constexpr (detail::kI64<T>) m.raw = _mm512_movepi64_mask(v.raw);
  else m.raw = _mm512_movepi32_mask(v.raw);
  return m;
}
// LoadMaskBits / BitsFromMask / StoreMaskBits: the k-register IS the packed bitfield already.
template <class D>
HYLE_FORCE_INLINE Mask512<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> LoadMaskBits(D, std::uint64_t bits) noexcept {
  return detail::mask_from_kbits<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)>(
      bits & detail::full_kmask<HYLE_SIMD_MAX_LANES_D(D)>());
}
template <class D>
HYLE_FORCE_INLINE std::uint64_t BitsFromMask(D, Mask512<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m) noexcept {
  return detail::kbits(m);
}
template <class D>
HYLE_FORCE_INLINE std::size_t StoreMaskBits(D, Mask512<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m,
                                            std::uint8_t* p) noexcept {
  const std::uint64_t bits = detail::kbits(m);
  const std::size_t nbytes = (HYLE_SIMD_MAX_LANES_D(D) + 7) / 8;
  for (std::size_t b = 0; b < nbytes; ++b) p[b] = static_cast<std::uint8_t>(bits >> (b * 8));
  return nbytes;
}
// IfThenZeroElse(m, no): zero where m true, else `no` (complement of IfThenElseZero). Masked-mov.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> IfThenZeroElse(Mask512<T, N> m, Vec512<T, N> no) noexcept {
  return IfThenElseZero(Not(m), no);
}
// prefix-scan masks (leading run relative to the first true lane). Bitfield math on kbits.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask512<T, N> SetBeforeFirst(Mask512<T, N> m) noexcept {
  const std::uint64_t b = detail::kbits(m);
  const std::uint64_t before = b ? ((std::uint64_t{1} << std::countr_zero(b)) - 1ull) : detail::full_kmask<N>();
  return detail::mask_from_kbits<T, N>(before);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask512<T, N> SetAtOrBeforeFirst(Mask512<T, N> m) noexcept {
  const std::uint64_t b = detail::kbits(m);
  // tz+1 can reach 64 (first true at lane 63) -> 1<<64 UB; saturate to the full N-lane mask.
  std::uint64_t upto;
  if (b) {
    const int tz1 = std::countr_zero(b) + 1;
    upto = (tz1 >= 64) ? ~0ull : ((std::uint64_t{1} << tz1) - 1ull);
  } else {
    upto = detail::full_kmask<N>();
  }
  return detail::mask_from_kbits<T, N>(upto & detail::full_kmask<N>());
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask512<T, N> SetOnlyFirst(Mask512<T, N> m) noexcept {
  const std::uint64_t b = detail::kbits(m);
  return detail::mask_from_kbits<T, N>(b ? (std::uint64_t{1} << std::countr_zero(b)) : 0ull);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask512<T, N> SetAtOrAfterFirst(Mask512<T, N> m) noexcept {
  const std::uint64_t b = detail::kbits(m);
  const std::uint64_t full = detail::full_kmask<N>();
  const std::uint64_t before = b ? ((std::uint64_t{1} << std::countr_zero(b)) - 1ull) : full;
  return detail::mask_from_kbits<T, N>(full & ~before);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE std::size_t FindKnownFirstTrue(Simd<T, N, 0>, Mask512<T, N> m) noexcept {
  return static_cast<std::size_t>(std::countr_zero(detail::kbits(m)));  // precondition: m != 0
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE std::size_t FindKnownLastTrue(Simd<T, N, 0>, Mask512<T, N> m) noexcept {
  return static_cast<std::size_t>(63 - std::countl_zero(detail::kbits(m)));
}

// SumsOf8(v): 64 u8 lanes -> 8 u64 (native _mm512_sad_epu8 vs zero). The count core.
template <std::size_t N>
HYLE_FORCE_INLINE Vec512<std::uint64_t, N / 8> SumsOf8(Vec512<std::uint8_t, N> v) noexcept {
  static_assert(N == 64, "avx512 SumsOf8: full 64xu8 -> 8xu64");
  Vec512<std::uint64_t, N / 8> r;
  r.raw = _mm512_sad_epu8(v.raw, _mm512_setzero_si512());
  return r;
}

// SumsOf2/SumsOf4: sum consecutive pairs/quads of u8 lanes, widening to u16/u32.
// correctness-first store+reload; native (pmaddubsw with 1s / vpaddl chains) is a profiling upgrade.
// scratch sized to the FULL 64 B register (not the logical lane count): _mm512_store/load_si512
// move 64 B, so a partial-width src would be a stack OOB with src[N] -- same class as TruncateTo.
template <std::size_t N>
HYLE_FORCE_INLINE Vec512<std::uint16_t, N / 2> SumsOf2(Vec512<std::uint8_t, N> v) noexcept {
  static_assert(N % 2 == 0, "avx512 SumsOf2 needs an even lane count");
  alignas(64) std::uint8_t src[64];
  alignas(64) std::uint16_t dst[64 / sizeof(std::uint16_t)] = {};
  _mm512_store_si512(src, v.raw);
  for (std::size_t g = 0; g < N / 2; ++g)
    dst[g] = static_cast<std::uint16_t>(src[2 * g] + src[2 * g + 1]);
  return Load(Simd<std::uint16_t, N / 2, 0>(), dst);
}
template <std::size_t N>
HYLE_FORCE_INLINE Vec512<std::uint32_t, N / 4> SumsOf4(Vec512<std::uint8_t, N> v) noexcept {
  static_assert(N % 4 == 0, "avx512 SumsOf4 needs a lane count that is a multiple of 4");
  alignas(64) std::uint8_t src[64];
  alignas(64) std::uint32_t dst[64 / sizeof(std::uint32_t)] = {};
  _mm512_store_si512(src, v.raw);
  for (std::size_t g = 0; g < N / 4; ++g)
    dst[g] = static_cast<std::uint32_t>(src[4 * g] + src[4 * g + 1] + src[4 * g + 2] + src[4 * g + 3]);
  return Load(Simd<std::uint32_t, N / 4, 0>(), dst);
}

// ---- reductions (AVX-512 has _mm512_reduce_* for 32/64-bit) --------------------------
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T GetLane(Vec512<T, N> v) noexcept {
  if constexpr (std::is_same_v<T, float>) return _mm512_cvtss_f32(v.raw);
  else if constexpr (std::is_same_v<T, double>) return _mm512_cvtsd_f64(v.raw);
  else {
    const __m128i lo = _mm512_castsi512_si128(v.raw);
    if constexpr (sizeof(T) == 8) return static_cast<T>(_mm_cvtsi128_si64(lo));
    else return static_cast<T>(_mm_cvtsi128_si32(lo));  // low 32 bits; T-cast truncates for i8/i16
  }
}

// No _mm512_reduce_*_epi8/16 intrinsics -> store to a stack array and scalar-reduce for the
// 8/16-bit widths (mirrors avx2's reduce_scalar; horizontal reduce isn't the hot path). The
// 32/64-bit widths use the native _mm512_reduce_* intrinsics below.
namespace detail {
enum class RedOp { kAdd, kMin, kMax };
template <typename T, std::size_t N, RedOp op>
HYLE_FORCE_INLINE T reduce_scalar(Vec512<T, N> v) noexcept {
  alignas(64) T tmp[N];
  _mm512_store_si512(tmp, v.raw);
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
HYLE_FORCE_INLINE T ReduceSum(Simd<T, N, 0>, Vec512<T, N> v) noexcept {
  if constexpr (std::is_same_v<T, float>) return _mm512_reduce_add_ps(v.raw);
  else if constexpr (std::is_same_v<T, double>) return _mm512_reduce_add_pd(v.raw);
  else if constexpr (detail::kI32<T>) return static_cast<T>(_mm512_reduce_add_epi32(v.raw));
  else if constexpr (detail::kI64<T>) return static_cast<T>(_mm512_reduce_add_epi64(v.raw));
  else return detail::reduce_scalar<T, N, detail::RedOp::kAdd>(v);  // i8/u8/i16/u16
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceMin(Simd<T, N, 0>, Vec512<T, N> v) noexcept {
  if constexpr (std::is_same_v<T, float>) return _mm512_reduce_min_ps(v.raw);
  else if constexpr (std::is_same_v<T, double>) return _mm512_reduce_min_pd(v.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) return _mm512_reduce_min_epi32(v.raw);
  else if constexpr (std::is_same_v<T, std::uint32_t>) return _mm512_reduce_min_epu32(v.raw);
  else if constexpr (std::is_same_v<T, std::int64_t>) return static_cast<T>(_mm512_reduce_min_epi64(v.raw));
  else if constexpr (std::is_same_v<T, std::uint64_t>) return static_cast<T>(_mm512_reduce_min_epu64(v.raw));
  else return detail::reduce_scalar<T, N, detail::RedOp::kMin>(v);  // i8/u8/i16/u16
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceMax(Simd<T, N, 0>, Vec512<T, N> v) noexcept {
  if constexpr (std::is_same_v<T, float>) return _mm512_reduce_max_ps(v.raw);
  else if constexpr (std::is_same_v<T, double>) return _mm512_reduce_max_pd(v.raw);
  else if constexpr (std::is_same_v<T, std::int32_t>) return _mm512_reduce_max_epi32(v.raw);
  else if constexpr (std::is_same_v<T, std::uint32_t>) return _mm512_reduce_max_epu32(v.raw);
  else if constexpr (std::is_same_v<T, std::int64_t>) return static_cast<T>(_mm512_reduce_max_epi64(v.raw));
  else if constexpr (std::is_same_v<T, std::uint64_t>) return static_cast<T>(_mm512_reduce_max_epu64(v.raw));
  else return detail::reduce_scalar<T, N, detail::RedOp::kMax>(v);  // i8/u8/i16/u16
}

// Reduction family completion: ReduceMul / ReduceAnd / ReduceOr. AVX-512 has native
// _mm512_reduce_mul/and/or for 32/64-bit lanes (F); i8/u8/i16/u16 have no such intrinsic,
// so fold via a stack array (mirrors reduce_scalar; horizontal reduce isn't the hot path).
namespace detail {
enum class RedOp2 { kMul, kAnd, kOr };
template <typename T, std::size_t N, RedOp2 op>
HYLE_FORCE_INLINE T reduce_scalar2(Vec512<T, N> v) noexcept {
  alignas(64) T tmp[N];
  _mm512_store_si512(tmp, v.raw);
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
HYLE_FORCE_INLINE T ReduceMul(Simd<T, N, 0>, Vec512<T, N> v) noexcept {
  if constexpr (std::is_same_v<T, float>) return _mm512_reduce_mul_ps(v.raw);
  else if constexpr (std::is_same_v<T, double>) return _mm512_reduce_mul_pd(v.raw);
  else if constexpr (detail::kI32<T>) return static_cast<T>(_mm512_reduce_mul_epi32(v.raw));
  else if constexpr (detail::kI64<T>) return static_cast<T>(_mm512_reduce_mul_epi64(v.raw));
  else return detail::reduce_scalar2<T, N, detail::RedOp2::kMul>(v);  // i8/u8/i16/u16
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceAnd(Simd<T, N, 0>, Vec512<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "ReduceAnd is integer-lane only");
  if constexpr (detail::kI32<T>) return static_cast<T>(_mm512_reduce_and_epi32(v.raw));
  else if constexpr (detail::kI64<T>) return static_cast<T>(_mm512_reduce_and_epi64(v.raw));
  else return detail::reduce_scalar2<T, N, detail::RedOp2::kAnd>(v);  // i8/u8/i16/u16
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceOr(Simd<T, N, 0>, Vec512<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "ReduceOr is integer-lane only");
  if constexpr (detail::kI32<T>) return static_cast<T>(_mm512_reduce_or_epi32(v.raw));
  else if constexpr (detail::kI64<T>) return static_cast<T>(_mm512_reduce_or_epi64(v.raw));
  else return detail::reduce_scalar2<T, N, detail::RedOp2::kOr>(v);  // i8/u8/i16/u16
}

// Broadcast<i>: splat lane i across all lanes (extract-and-Set, uniform over lane width).
// Index-vector swizzles (TableLookup/Gather/Scatter/Compress/Interleave) DEFERRED -- see
// scalar.h note.
template <int kLane, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> Broadcast(Vec512<T, N> v) noexcept {
  static_assert(kLane >= 0 && static_cast<std::size_t>(kLane) < N, "Broadcast lane out of range");
  alignas(64) T tmp[N];
  if constexpr (std::is_same_v<T, float>) _mm512_store_ps(tmp, v.raw);
  else if constexpr (std::is_same_v<T, double>) _mm512_store_pd(tmp, v.raw);
  else _mm512_store_si512(tmp, v.raw);
  return Set(Simd<T, N, 0>(), tmp[kLane]);
}

// ---- swizzle: B-2 completions (Reverse4/8, ReverseBits/LaneBytes, InterleaveEven/Odd,
// Slide1Up/Down, SlideUp/DownLanes, BroadcastLane, Zip). ------------------------------
// correctness-first store+reload; native shuffles are a profiling upgrade.
template <class D>
HYLE_FORCE_INLINE detail::V512<D> Reverse4(D, detail::V512<D> v) noexcept {
  return detail::swizzle_via_mem(v, [](std::size_t i) { return (i & ~std::size_t{3}) | ((i & 3) ^ 3); });
}
template <class D>
HYLE_FORCE_INLINE detail::V512<D> Reverse8(D, detail::V512<D> v) noexcept {
  return detail::swizzle_via_mem(v, [](std::size_t i) { return (i & ~std::size_t{7}) | ((i & 7) ^ 7); });
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> ReverseBits(Vec512<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "ReverseBits is integer-only");
  using U = std::make_unsigned_t<T>;
  alignas(64) T s[N], d[N];
  Store(v, Simd<T, N, 0>(), s);
  for (std::size_t i = 0; i < N; ++i) {
    U x = static_cast<U>(s[i]), y = 0;
    for (std::size_t b = 0; b < sizeof(T) * 8; ++b) { y = static_cast<U>((y << 1) | (x & 1)); x >>= 1; }
    d[i] = static_cast<T>(y);
  }
  return Load(Simd<T, N, 0>(), d);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> ReverseLaneBytes(Vec512<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "ReverseLaneBytes is integer-only");
  alignas(64) T s[N], d[N];
  Store(v, Simd<T, N, 0>(), s);
  for (std::size_t i = 0; i < N; ++i) {
    auto* db = reinterpret_cast<std::uint8_t*>(&d[i]);
    const auto* sb = reinterpret_cast<const std::uint8_t*>(&s[i]);
    for (std::size_t b = 0; b < sizeof(T); ++b) db[b] = sb[sizeof(T) - 1 - b];
  }
  return Load(Simd<T, N, 0>(), d);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> InterleaveEven(Vec512<T, N> a, Vec512<T, N> b) noexcept {
  alignas(64) T sa[N], sb[N], d[N];
  Store(a, Simd<T, N, 0>(), sa); Store(b, Simd<T, N, 0>(), sb);
  for (std::size_t i = 0; i < N / 2; ++i) { d[2 * i] = sa[2 * i]; d[2 * i + 1] = sb[2 * i]; }
  return Load(Simd<T, N, 0>(), d);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> InterleaveOdd(Vec512<T, N> a, Vec512<T, N> b) noexcept {
  alignas(64) T sa[N], sb[N], d[N];
  Store(a, Simd<T, N, 0>(), sa); Store(b, Simd<T, N, 0>(), sb);
  for (std::size_t i = 0; i < N / 2; ++i) { d[2 * i] = sa[2 * i + 1]; d[2 * i + 1] = sb[2 * i + 1]; }
  return Load(Simd<T, N, 0>(), d);
}
template <class D>
HYLE_FORCE_INLINE detail::V512<D> Slide1Up(D dd, detail::V512<D> v) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(64) T s[N]; alignas(64) T dst[N];
  Store(v, dd, s);
  dst[0] = T{0};
  for (std::size_t i = 1; i < N; ++i) dst[i] = s[i - 1];
  return Load(dd, dst);
}
template <class D>
HYLE_FORCE_INLINE detail::V512<D> Slide1Down(D dd, detail::V512<D> v) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(64) T s[N]; alignas(64) T dst[N];
  Store(v, dd, s);
  dst[N - 1] = T{0};
  for (std::size_t i = 0; i + 1 < N; ++i) dst[i] = s[i + 1];
  return Load(dd, dst);
}
template <class D>
HYLE_FORCE_INLINE detail::V512<D> SlideUpLanes(D dd, detail::V512<D> v, std::size_t amt) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(64) T s[N]; alignas(64) T dst[N] = {};
  Store(v, dd, s);
  for (std::size_t i = amt; i < N; ++i) dst[i] = s[i - amt];
  return Load(dd, dst);
}
template <class D>
HYLE_FORCE_INLINE detail::V512<D> SlideDownLanes(D dd, detail::V512<D> v, std::size_t amt) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(64) T s[N]; alignas(64) T dst[N] = {};
  Store(v, dd, s);
  for (std::size_t i = 0; i + amt < N; ++i) dst[i] = s[i + amt];
  return Load(dd, dst);
}
template <int kLane, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> BroadcastLane(Vec512<T, N> v) noexcept { return Broadcast<kLane>(v); }
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> ZipLower(Vec512<T, N> a, Vec512<T, N> b) noexcept { return InterleaveLower(a, b); }
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> ZipUpper(Vec512<T, N> a, Vec512<T, N> b) noexcept { return InterleaveUpper(a, b); }

// ---- masked / partial load-store (tail handling) ------------------------------------
// AVX-512's k-register mask model makes these NATIVE and clean (unlike the SSE/AVX2/NEON
// memory-staging): FirstN(d,n) is literally the k register with the low min(n,N) bits set;
// MaskedLoad = maskz_loadu (off lanes zeroed by hardware), MaskedLoadOr = mask_loadu (off
// lanes keep 'no'), BlendedStore = mask_storeu (writes only on-lanes, no fault on off).
// LoadN/StoreN compose via FirstN (Highway does the same). Load/Store are per-width epi8/
// 16/32/64 (BW for 8/16) + ps/pd; all touch memory only for on-lanes (hardware-fault-safe).
template <class D>
HYLE_FORCE_INLINE Mask512<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> FirstN(D, std::size_t n) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  const std::uint64_t full = (kN >= 64) ? ~0ull : ((1ull << kN) - 1ull);
  const std::uint64_t bits = (n >= kN) ? full : ((1ull << n) - 1ull);
  Mask512<T, kN> m;
  m.raw = static_cast<typename detail::KMaskT<T>>(bits);
  return m;
}
template <class D>
HYLE_FORCE_INLINE detail::V512<D> MaskedLoad(Mask512<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m,
                                             D, const TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  detail::V512<D> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm512_maskz_loadu_ps(m.raw, p);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm512_maskz_loadu_pd(m.raw, p);
  else if constexpr (detail::kI8<T>) r.raw = _mm512_maskz_loadu_epi8(m.raw, p);
  else if constexpr (detail::kI16<T>) r.raw = _mm512_maskz_loadu_epi16(m.raw, p);
  else if constexpr (detail::kI64<T>) r.raw = _mm512_maskz_loadu_epi64(m.raw, p);
  else r.raw = _mm512_maskz_loadu_epi32(m.raw, p);  // i32/u32
  return r;
}
template <class D>
HYLE_FORCE_INLINE detail::V512<D> MaskedLoadOr(detail::V512<D> no,
                                               Mask512<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m,
                                               D, const TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  detail::V512<D> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm512_mask_loadu_ps(no.raw, m.raw, p);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm512_mask_loadu_pd(no.raw, m.raw, p);
  else if constexpr (detail::kI8<T>) r.raw = _mm512_mask_loadu_epi8(no.raw, m.raw, p);
  else if constexpr (detail::kI16<T>) r.raw = _mm512_mask_loadu_epi16(no.raw, m.raw, p);
  else if constexpr (detail::kI64<T>) r.raw = _mm512_mask_loadu_epi64(no.raw, m.raw, p);
  else r.raw = _mm512_mask_loadu_epi32(no.raw, m.raw, p);
  return r;
}
template <class D>
HYLE_FORCE_INLINE void BlendedStore(detail::V512<D> v,
                                    Mask512<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m, D,
                                    TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  if constexpr (std::is_same_v<T, float>) _mm512_mask_storeu_ps(p, m.raw, v.raw);
  else if constexpr (std::is_same_v<T, double>) _mm512_mask_storeu_pd(p, m.raw, v.raw);
  else if constexpr (detail::kI8<T>) _mm512_mask_storeu_epi8(p, m.raw, v.raw);
  else if constexpr (detail::kI16<T>) _mm512_mask_storeu_epi16(p, m.raw, v.raw);
  else if constexpr (detail::kI64<T>) _mm512_mask_storeu_epi64(p, m.raw, v.raw);
  else _mm512_mask_storeu_epi32(p, m.raw, v.raw);
}
template <class D>
HYLE_FORCE_INLINE detail::V512<D> LoadN(D d, const TFromD<D>* p, std::size_t n) noexcept {
  return MaskedLoad(FirstN(d, n), d, p);
}
template <class D>
HYLE_FORCE_INLINE void StoreN(detail::V512<D> v, D d, TFromD<D>* p, std::size_t n) noexcept {
  BlendedStore(v, FirstN(d, n), d, p);
}

// ---- gather / scatter + compress / expand -------------------------------------------
// Store+reload (capability-complete). AVX-512 HAS native vpgatherdd/vpscatterdd + vpcompress/
// vpexpand which would be faster, but they're perf optimizations gated behind profiling +
// real-hardware validation (none here) -- the store+reload form is correct and oracle-matched
// now. kbits() yields the uint64_t lane bitmask. The scalar backend is the reference.
template <class D, typename IdxT>
HYLE_FORCE_INLINE detail::V512<D> GatherIndex(D d, const TFromD<D>* base,
                                              Vec512<IdxT, HYLE_SIMD_MAX_LANES_D(D)> idx) noexcept {
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  alignas(64) IdxT ix[kN]; alignas(64) TFromD<D> dst[kN];
  Store(idx, Simd<IdxT, kN, 0>(), ix);
  for (std::size_t i = 0; i < kN; ++i) dst[i] = base[static_cast<std::ptrdiff_t>(ix[i])];
  return Load(d, dst);
}
template <class D, typename IdxT>
HYLE_FORCE_INLINE void ScatterIndex(detail::V512<D> v, D d, TFromD<D>* base,
                                    Vec512<IdxT, HYLE_SIMD_MAX_LANES_D(D)> idx) noexcept {
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  alignas(64) IdxT ix[kN]; alignas(64) TFromD<D> src[kN];
  Store(idx, Simd<IdxT, kN, 0>(), ix); Store(v, d, src);
  for (std::size_t i = 0; i < kN; ++i) base[static_cast<std::ptrdiff_t>(ix[i])] = src[i];
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> Compress(Vec512<T, N> v, Mask512<T, N> m) noexcept {
  alignas(64) T src[N], dst[N] = {};
  Store(v, Simd<T, N, 0>(), src);
  const std::uint64_t bits = detail::kbits(m);
  std::size_t k = 0;
  for (std::size_t i = 0; i < N; ++i) if ((bits >> i) & 1ull) dst[k++] = src[i];
  return Load(Simd<T, N, 0>(), dst);
}
template <class D>
HYLE_FORCE_INLINE std::size_t CompressStore(detail::V512<D> v,
                                            Mask512<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m,
                                            D d, TFromD<D>* p) noexcept {
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  alignas(64) TFromD<D> src[kN];
  Store(v, d, src);
  const std::uint64_t bits = detail::kbits(m);
  std::size_t k = 0;
  for (std::size_t i = 0; i < kN; ++i) if ((bits >> i) & 1ull) p[k++] = src[i];
  return k;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> Expand(Vec512<T, N> v, Mask512<T, N> m) noexcept {
  alignas(64) T src[N], dst[N];
  Store(v, Simd<T, N, 0>(), src);
  const std::uint64_t bits = detail::kbits(m);
  std::size_t k = 0;
  for (std::size_t i = 0; i < N; ++i) dst[i] = ((bits >> i) & 1ull) ? src[k++] : T{0};
  return Load(Simd<T, N, 0>(), dst);
}

// ---- bit counts / sign helpers / misc -----------------------------------------------
// PopulationCount/LeadingZeroCount/TrailingZeroCount via store+reload + std:: (AVX-512 HAS
// native vpopcntd (VPOPCNTDQ) + vplzcntd, but they're perf opts gated on real-hw validation;
// store+reload is correct now). Not/Clamp/Rotate/CopySign/... compose from existing ops.
namespace detail {
template <typename T, std::size_t N, int kind>
HYLE_FORCE_INLINE Vec512<T, N> bitcount_via_mem(Vec512<T, N> v) noexcept {
  using U = std::make_unsigned_t<T>;
  alignas(64) T s[N], d[N];
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
HYLE_FORCE_INLINE Vec512<T, N> PopulationCount(Vec512<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "PopulationCount is integer-lane only");
  return detail::bitcount_via_mem<T, N, 0>(v);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> LeadingZeroCount(Vec512<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "LeadingZeroCount is integer-lane only");
  return detail::bitcount_via_mem<T, N, 1>(v);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> TrailingZeroCount(Vec512<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "TrailingZeroCount is integer-lane only");
  return detail::bitcount_via_mem<T, N, 2>(v);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> Not(Vec512<T, N> v) noexcept {
  Vec512<T, N> ones;
  if constexpr (std::is_same_v<T, float>) ones.raw = _mm512_castsi512_ps(_mm512_set1_epi32(-1));
  else if constexpr (std::is_same_v<T, double>) ones.raw = _mm512_castsi512_pd(_mm512_set1_epi32(-1));
  else ones.raw = _mm512_set1_epi32(-1);
  return Xor(v, ones);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> Clamp(Vec512<T, N> v, Vec512<T, N> lo, Vec512<T, N> hi) noexcept {
  return Min(Max(v, lo), hi);
}
template <int kBits, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> RotateRight(Vec512<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "RotateRight is integer-lane only");
  constexpr int kSizeBits = static_cast<int>(sizeof(T) * 8);
  static_assert(kBits >= 0 && kBits < kSizeBits, "rotate count in [0,bits)");
  if constexpr (kBits == 0) return v;
  else if constexpr (std::is_unsigned_v<T>)
    return Or(ShiftRight<kBits>(v), ShiftLeft<kSizeBits - kBits>(v));
  else {
    using U = std::make_unsigned_t<T>;
    alignas(64) T s[N], d[N];
    Store(v, Simd<T, N, 0>(), s);
    for (std::size_t i = 0; i < N; ++i) { const U u = static_cast<U>(s[i]); d[i] = static_cast<T>((u >> kBits) | (u << (kSizeBits - kBits))); }
    return Load(Simd<T, N, 0>(), d);
  }
}
template <int kBits, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> RotateLeft(Vec512<T, N> v) noexcept {
  constexpr int kSizeBits = static_cast<int>(sizeof(T) * 8);
  return RotateRight<(kSizeBits - kBits) % kSizeBits>(v);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> CopySign(Vec512<T, N> mag, Vec512<T, N> sign) noexcept {
  static_assert(std::is_floating_point_v<T>, "CopySign is float-lane only");
  const auto sign_bit = Set(Simd<T, N, 0>(), static_cast<T>(-0.0));
  return Or(AndNot(sign_bit, mag), And(sign_bit, sign));
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> BroadcastSignBit(Vec512<T, N> v) noexcept {
  static_assert(std::is_integral_v<T> && std::is_signed_v<T>, "BroadcastSignBit is signed-int only");
  return ShiftRight<static_cast<int>(sizeof(T) * 8) - 1>(v);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> IfNegativeThenElse(Vec512<T, N> v, Vec512<T, N> yes, Vec512<T, N> no) noexcept {
  return IfThenElse(Lt(v, Zero(Simd<T, N, 0>())), yes, no);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> ZeroIfNegative(Vec512<T, N> v) noexcept {
  return IfThenElse(Lt(v, Zero(Simd<T, N, 0>())), Zero(Simd<T, N, 0>()), v);
}

// ---- crypto: AES round primitives + carryless multiply (simd-08 档C) -----------------
// One __m512i == FOUR independent 128-bit AES blocks. Same model as avx2: split into four
// 128-bit lanes (extracti32x4 / inserti32x4), run the shared per-block AES-NI/PCLMUL worker
// on each, recombine. Highway's x86_512 builds AES from AES-NI per-128 exactly this way
// unless VAES (_mm512_aesenc_epi128, needs -mvaes) is available -- promoting to VAES is a
// profiling upgrade only. Contract: 512-bit vector = 64 bytes = four 128-bit AES states.
namespace detail {
template <typename BlockOp>
HYLE_FORCE_INLINE __m512i aes512_map1(__m512i s, BlockOp op) noexcept {
  __m512i r = s;
  r = _mm512_inserti32x4(r, op(_mm512_extracti32x4_epi32(s, 0)), 0);
  r = _mm512_inserti32x4(r, op(_mm512_extracti32x4_epi32(s, 1)), 1);
  r = _mm512_inserti32x4(r, op(_mm512_extracti32x4_epi32(s, 2)), 2);
  r = _mm512_inserti32x4(r, op(_mm512_extracti32x4_epi32(s, 3)), 3);
  return r;
}
template <typename BlockOp>
HYLE_FORCE_INLINE __m512i aes512_map2(__m512i s, __m512i k, BlockOp op) noexcept {
  __m512i r = s;
  r = _mm512_inserti32x4(r, op(_mm512_extracti32x4_epi32(s, 0), _mm512_extracti32x4_epi32(k, 0)), 0);
  r = _mm512_inserti32x4(r, op(_mm512_extracti32x4_epi32(s, 1), _mm512_extracti32x4_epi32(k, 1)), 1);
  r = _mm512_inserti32x4(r, op(_mm512_extracti32x4_epi32(s, 2), _mm512_extracti32x4_epi32(k, 2)), 2);
  r = _mm512_inserti32x4(r, op(_mm512_extracti32x4_epi32(s, 3), _mm512_extracti32x4_epi32(k, 3)), 3);
  return r;
}
}  // namespace detail
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> AESRound(Vec512<T, N> state, Vec512<T, N> key) noexcept {
  static_assert(sizeof(T) * N == 64, "AES op needs a 512-bit vector (four 128-bit blocks)");
  return Vec512<T, N>{detail::aes512_map2(state.raw, key.raw, detail::aes_round_block)};
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> AESLastRound(Vec512<T, N> state, Vec512<T, N> key) noexcept {
  static_assert(sizeof(T) * N == 64, "AES op needs a 512-bit vector (four 128-bit blocks)");
  return Vec512<T, N>{detail::aes512_map2(state.raw, key.raw, detail::aes_last_round_block)};
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> AESRoundInv(Vec512<T, N> state, Vec512<T, N> key) noexcept {
  static_assert(sizeof(T) * N == 64, "AES op needs a 512-bit vector (four 128-bit blocks)");
  return Vec512<T, N>{detail::aes512_map2(state.raw, key.raw, detail::aes_round_inv_block)};
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> AESLastRoundInv(Vec512<T, N> state, Vec512<T, N> key) noexcept {
  static_assert(sizeof(T) * N == 64, "AES op needs a 512-bit vector (four 128-bit blocks)");
  return Vec512<T, N>{detail::aes512_map2(state.raw, key.raw, detail::aes_last_round_inv_block)};
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> AESInvMixColumns(Vec512<T, N> state) noexcept {
  static_assert(sizeof(T) * N == 64, "AES op needs a 512-bit vector (four 128-bit blocks)");
  return Vec512<T, N>{detail::aes512_map1(state.raw, detail::aes_inv_mix_block)};
}
template <std::uint8_t kRcon, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> AESKeyGenAssist(Vec512<T, N> v) noexcept {
  static_assert(sizeof(T) * N == 64, "AES op needs a 512-bit vector (four 128-bit blocks)");
  // kRcon compile-time immediate -> templated per-block worker, split explicitly across 4 lanes.
  __m512i r = v.raw;
  r = _mm512_inserti32x4(r, detail::aes_keygen_block<kRcon>(_mm512_extracti32x4_epi32(v.raw, 0)), 0);
  r = _mm512_inserti32x4(r, detail::aes_keygen_block<kRcon>(_mm512_extracti32x4_epi32(v.raw, 1)), 1);
  r = _mm512_inserti32x4(r, detail::aes_keygen_block<kRcon>(_mm512_extracti32x4_epi32(v.raw, 2)), 2);
  r = _mm512_inserti32x4(r, detail::aes_keygen_block<kRcon>(_mm512_extracti32x4_epi32(v.raw, 3)), 3);
  return Vec512<T, N>{r};
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> CLMulLower(Vec512<T, N> a, Vec512<T, N> b) noexcept {
  static_assert(sizeof(T) * N == 64, "CLMul needs a 512-bit vector (four 128-bit blocks)");
  return Vec512<T, N>{detail::aes512_map2(a.raw, b.raw, detail::clmul_lower_block)};
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec512<T, N> CLMulUpper(Vec512<T, N> a, Vec512<T, N> b) noexcept {
  static_assert(sizeof(T) * N == 64, "CLMul needs a 512-bit vector (four 128-bit blocks)");
  return Vec512<T, N>{detail::aes512_map2(a.raw, b.raw, detail::clmul_upper_block)};
}

}  // namespace hyle::simd
