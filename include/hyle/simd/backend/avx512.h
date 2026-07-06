#pragma once

#include <cstddef>
#include <cstdint>
#include <bit>
#include <type_traits>

#include <immintrin.h>

#include "hyle/macros.h"

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

}  // namespace hyle::simd
