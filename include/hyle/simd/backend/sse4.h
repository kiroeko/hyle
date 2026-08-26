#pragma once

#include <array>    // std::array (TruncateTo PSHUFB byte-index table)
#include <cstddef>
#include <cstdint>
#include <cstring>  // std::memcpy (LoadMaskBits lane pun)
#include <bit>
#include <cmath>   // std::logb / std::ldexp (GetExponent / MulByPow2 store+reload)
#include <limits>  // std::numeric_limits<T>::infinity() (FP classification splat)
#include <type_traits>

#include <nmmintrin.h>  // SSE4.2 (transitively pulls SSE4.1/SSSE3/SSE3/SSE2)

#include "hyle/core/compiler.h"
#include "hyle/simd/backend/crypto_x86.h"  // shared 128-bit AES-NI/PCLMUL workers (Crypto op face)
#include "hyle/simd/backend/x86_detail.h"  // shared lane-type predicates (kSupported/kI*/kSigned)

// backend/sse4.h -- 128-bit SSE4.2-baseline backend (simd-design landing list #2).
//
// ONE backend per TU (variant-DLL model): this header is pulled in by vec.h only when
// the TU is compiled with -DHYLE_SIMD_TARGET_SSE4 (+ -msse4.2 on GCC/Clang; MSVC x64
// has all SSE4.2 intrinsics unconditionally). It defines the SAME op names/signatures
// as backend/scalar.h, so a single source (test/kernel) written against the D-tag
// contract compiles unchanged against scalar OR this backend -- the whole point of the
// abstraction.
//
// Lane types f32/f64 + all integer widths i8/u8/i16/u16/i32/u32/i64/u64. One
// __m128-family register == 16 bytes == 4 f32 / 2 f64 / 16 i8 / 8 i16 / 4 i32 / 2 i64.
// Sub-128 partials remain physically XMM (x86 has no useful D-register equivalent),
// while Vec128<T,N>, masks, exact-byte LoadU/StoreU and N-aware half/combine preserve
// their logical lane count.
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

// kSupported / kI8 / kI16 / kI32 / kI64 / kSigned are lane-type-only predicates shared by all
// x86 backends -- defined once in backend/x86_detail.h (included above).
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

// Width-agnostic vector<->tag aliases. IDENTICAL across the layered x86 backends (sse4/avx2/
// avx512), so a guard defines them ONCE per TU -- sse4.h is always included first (avx2.h pulls
// it, avx512.h pulls avx2.h), so it wins. Without the guard MSVC rejects the second alias-template
// definition as C2386 (GCC tolerates the identical redefinition; the layered-include refactor
// exposed the divergence). DFromV/TFromV are pure; VFromD is dependent (see below).
#ifndef HYLE_SIMD_X86_VT_ALIASES
#define HYLE_SIMD_X86_VT_ALIASES
template <class V>
using DFromV = Simd<typename V::PrivateT, V::kPrivateN, 0>;
template <class V>
using TFromV = typename V::PrivateT;
#endif  // HYLE_SIMD_X86_VT_ALIASES

// The 128-bit backend's vector type for tag D. N comes from D (== 16/sizeof(T) for a
// native ScalableTag). A wrong-width tag is a compile error (static_assert below).
namespace detail {
template <class D>
using V128 = Vec128<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)>;
template <class D>
using M128 = Mask128<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)>;
}  // namespace detail

// ---- initialization -----------------------------------------------------------------
template <class D, typename T2> requires HYLE_SIMD_W(D, 0, 16)
HYLE_FORCE_INLINE detail::V128<D> Set(D, T2 t) noexcept {
  using T = TFromD<D>;
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

template <class D> requires HYLE_SIMD_W(D, 0, 16)
HYLE_FORCE_INLINE detail::V128<D> Zero(D) noexcept {
  using T = TFromD<D>;
  detail::V128<D> v;
  if constexpr (std::is_same_v<T, float>)
    v.raw = _mm_setzero_ps();
  else if constexpr (std::is_same_v<T, double>)
    v.raw = _mm_setzero_pd();
  else
    v.raw = _mm_setzero_si128();
  return v;
}

// Dependent alias (resolves Zero at instantiation); guarded so layered includes define it once.
#ifndef HYLE_SIMD_X86_VFROMD
#define HYLE_SIMD_X86_VFROMD
template <class D>
using VFromD = decltype(Zero(D()));
#endif  // HYLE_SIMD_X86_VFROMD

// ---- load / store -------------------------------------------------------------------
// Partial tags (CappedTag/FixedTag with N*sizeof(T) < 16) must touch ONLY the logical
// bytes -- same contract as Highway's Vec64/Vec32 specializations. Full-width LoadU/StoreU
// would over-read/write past small arrays (vqsort Sort8Rows kKeysPerRow=1 on a 6-float
// buffer → heap corruption). Scratch buffers that feed Load/Store are sized to a full
// 16 B register (see TruncateTo / LoadN comments); partial Load/Store still only use the
// low logical lanes of that scratch.
template <class D> requires HYLE_SIMD_W(D, 0, 16)
HYLE_FORCE_INLINE detail::V128<D> LoadU(D, const TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t kBytes = HYLE_SIMD_MAX_LANES_D(D) * sizeof(T);
  detail::V128<D> v;
  if constexpr (kBytes == 16) {
    if constexpr (std::is_same_v<T, float>)
      v.raw = _mm_loadu_ps(p);
    else if constexpr (std::is_same_v<T, double>)
      v.raw = _mm_loadu_pd(p);
    else
      v.raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
  } else if constexpr (kBytes == 8) {
    // Avoid __m64 (removed on MSVC x64): load low 64 via integer, cast.
    const __m128i lo = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
    if constexpr (std::is_same_v<T, float>)
      v.raw = _mm_castsi128_ps(lo);
    else if constexpr (std::is_same_v<T, double>)
      v.raw = _mm_castsi128_pd(lo);
    else
      v.raw = lo;
  } else if constexpr (kBytes == 4) {
    if constexpr (std::is_same_v<T, float>)
      v.raw = _mm_load_ss(p);
    else {
      std::uint32_t bits = 0;
      std::memcpy(&bits, p, 4);
      v.raw = _mm_cvtsi32_si128(static_cast<int>(bits));
    }
  } else {
    // 1/2-byte lanes or other sub-8 partials: zero + exact-byte copy (Highway safe path).
    if constexpr (std::is_same_v<T, float>)
      v.raw = _mm_setzero_ps();
    else if constexpr (std::is_same_v<T, double>)
      v.raw = _mm_setzero_pd();
    else
      v.raw = _mm_setzero_si128();
    std::memcpy(&v.raw, p, kBytes);
  }
  return v;
}
template <class D> requires HYLE_SIMD_W(D, 0, 16)
HYLE_FORCE_INLINE detail::V128<D> Load(D d, const TFromD<D>* p) noexcept {
  // Partial tags have no stricter alignment contract than LoadU (Highway: LoadU==Load for <128).
  return LoadU(d, p);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE void StoreU(Vec128<T, N> v, Simd<T, N, 0>, T* p) noexcept {
  constexpr std::size_t kBytes = N * sizeof(T);
  if constexpr (kBytes == 16) {
    if constexpr (std::is_same_v<T, float>)
      _mm_storeu_ps(p, v.raw);
    else if constexpr (std::is_same_v<T, double>)
      _mm_storeu_pd(p, v.raw);
    else
      _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v.raw);
  } else if constexpr (kBytes == 8) {
    __m128i lo;
    if constexpr (std::is_same_v<T, float>)
      lo = _mm_castps_si128(v.raw);
    else if constexpr (std::is_same_v<T, double>)
      lo = _mm_castpd_si128(v.raw);
    else
      lo = v.raw;
    _mm_storel_epi64(reinterpret_cast<__m128i*>(p), lo);
  } else if constexpr (kBytes == 4) {
    if constexpr (std::is_same_v<T, float>)
      _mm_store_ss(p, v.raw);
    else {
      const std::uint32_t bits = static_cast<std::uint32_t>(_mm_cvtsi128_si32(v.raw));
      std::memcpy(p, &bits, 4);
    }
  } else {
    std::memcpy(p, &v.raw, kBytes);
  }
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE void Store(Vec128<T, N> v, Simd<T, N, 0> d, T* p) noexcept {
  StoreU(v, d, p);
}

// ---- interleaved load / store (AoS <-> SoA) -----------------------------------------
// 32/64-bit lanes: in-register transpose (Highway recipe). 128-bit = ONE block, so the block
// transpose is trivial (raw loads/stores); only the per-block shuffle/unpack stage runs. 8/16-bit
// keep the store+reload fallback (Highway's pshufb-table port deferred).
namespace detail {
template <int kImm, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> shuffle_two_ps(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  __m128 af, bf;
  if constexpr (std::is_same_v<T, float>) { af = a.raw; bf = b.raw; }
  else { af = _mm_castsi128_ps(a.raw); bf = _mm_castsi128_ps(b.raw); }
  const __m128 s = _mm_shuffle_ps(af, bf, kImm);
  Vec128<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = s;
  else r.raw = _mm_castps_si128(s);
  return r;
}
}  // namespace detail

template <class D> requires HYLE_SIMD_W(D, 0, 16)
HYLE_FORCE_INLINE void LoadInterleaved2(D d, const TFromD<D>* p, VFromD<D>& v0,
                                        VFromD<D>& v1) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  if constexpr (sizeof(T) >= 4) {
    const auto A = LoadU(d, p), B = LoadU(d, p + N);
    v0 = ConcatEven(B, A); v1 = ConcatOdd(B, A);
  } else {
    alignas(16) T t0[16 / sizeof(T)], t1[16 / sizeof(T)];
    for (std::size_t i = 0; i < N; ++i) { t0[i] = p[2 * i]; t1[i] = p[2 * i + 1]; }
    v0 = Load(d, t0); v1 = Load(d, t1);
  }
}
template <class D> requires HYLE_SIMD_W(D, 0, 16)
HYLE_FORCE_INLINE void LoadInterleaved3(D d, const TFromD<D>* p, VFromD<D>& v0, VFromD<D>& v1,
                                        VFromD<D>& v2) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  if constexpr (sizeof(T) == 4 || sizeof(T) == 8) {
    const auto A = LoadU(d, p + 0 * N), B = LoadU(d, p + 1 * N), C = LoadU(d, p + 2 * N);
    if constexpr (sizeof(T) == 4) {
      const auto x02 = OddEven(C, B);
      v0 = detail::shuffle_two_ps<_MM_SHUFFLE(1, 2, 3, 0)>(A, x02);
      const auto x10 = OddEven(A, B), x12 = OddEven(B, C);
      v1 = detail::shuffle_two_ps<_MM_SHUFFLE(2, 3, 0, 1)>(x10, x12);
      const auto x20 = OddEven(B, A);
      v2 = detail::shuffle_two_ps<_MM_SHUFFLE(3, 0, 1, 2)>(x20, C);
    } else {  // 64-bit (N=2)
      v0 = OddEven(B, A);
      v1 = CombineShiftRightBytes<sizeof(T)>(C, A);
      v2 = OddEven(C, B);
    }
  } else {  // 8/16-bit: 1 block => raw loads are the transpose; byte-table gather
    const auto A = LoadU(d, p + 0 * N), B = LoadU(d, p + 1 * N), C = LoadU(d, p + 2 * N);
    if constexpr (sizeof(T) == 1) detail::Interleave3To_u8(d, A, B, C, v0, v1, v2);
    else detail::Interleave3To_u16(d, A, B, C, v0, v1, v2);
  }
}
template <class D> requires HYLE_SIMD_W(D, 0, 16)
HYLE_FORCE_INLINE void LoadInterleaved4(D d, const TFromD<D>* p, VFromD<D>& v0, VFromD<D>& v1,
                                        VFromD<D>& v2, VFromD<D>& v3) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  if constexpr (sizeof(T) == 4 || sizeof(T) == 8) {
    const auto A = LoadU(d, p + 0 * N), B = LoadU(d, p + 1 * N),
               C = LoadU(d, p + 2 * N), Dd = LoadU(d, p + 3 * N);
    if constexpr (sizeof(T) == 4) {
      const auto e0 = InterleaveLower(A, C), o0 = InterleaveLower(B, Dd);
      const auto e2 = InterleaveUpper(A, C), o2 = InterleaveUpper(B, Dd);
      v0 = InterleaveLower(e0, o0); v1 = InterleaveUpper(e0, o0);
      v2 = InterleaveLower(e2, o2); v3 = InterleaveUpper(e2, o2);
    } else {  // 64-bit (N=2)
      v0 = InterleaveLower(A, C); v1 = InterleaveUpper(A, C);
      v2 = InterleaveLower(B, Dd); v3 = InterleaveUpper(B, Dd);
    }
  } else {  // 8/16-bit: 1 block => raw loads are the transpose; unpack tree
    const auto A = LoadU(d, p + 0 * N), B = LoadU(d, p + 1 * N),
               C = LoadU(d, p + 2 * N), Dd = LoadU(d, p + 3 * N);
    if constexpr (sizeof(T) == 1) detail::Interleave4To_u8(d, A, B, C, Dd, v0, v1, v2, v3);
    else detail::Interleave4To_u16(d, A, B, C, Dd, v0, v1, v2, v3);
  }
}
template <class D> requires HYLE_SIMD_W(D, 0, 16)
HYLE_FORCE_INLINE void StoreInterleaved2(VFromD<D> v0, VFromD<D> v1, D d, TFromD<D>* p) noexcept {
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  // width-independent: InterleaveLower/Upper native for all lane widths (1 block, raw store).
  StoreU(InterleaveLower(v0, v1), d, p + 0 * N);
  StoreU(InterleaveUpper(v0, v1), d, p + 1 * N);
}
template <class D> requires HYLE_SIMD_W(D, 0, 16)
HYLE_FORCE_INLINE void StoreInterleaved3(VFromD<D> v0, VFromD<D> v1, VFromD<D> v2, D d,
                                         TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  if constexpr (sizeof(T) == 4 || sizeof(T) == 8) {
    VFromD<D> A, B, C;
    if constexpr (sizeof(T) == 4) {
      const Simd<std::uint64_t, N / 2, 0> dw;
      const auto v10 = InterleaveLower(v0, v1), v01_20 = OddEven(v0, v2);
      A = BitCast(d, InterleaveLower(BitCast(dw, v10), BitCast(dw, v01_20)));
      const auto v1_321 = ShiftRightLanes<1>(v1), v0_32 = ShiftRightLanes<2>(v0);
      const auto v21_11 = OddEven(v2, v1_321), v12_02 = OddEven(v1_321, v0_32);
      B = BitCast(d, InterleaveLower(BitCast(dw, v21_11), BitCast(dw, v12_02)));
      const auto v23_13 = OddEven(v2, v1_321), v03_22 = OddEven(v0, v2);
      C = BitCast(d, InterleaveUpper(BitCast(dw, v03_22), BitCast(dw, v23_13)));
    } else {  // 64-bit (N=2)
      A = InterleaveLower(v0, v1); B = OddEven(v0, v2); C = InterleaveUpper(v1, v2);
    }
    StoreU(A, d, p + 0 * N); StoreU(B, d, p + 1 * N); StoreU(C, d, p + 2 * N);
  } else {  // 8/16-bit: byte-table scatter (1 block => raw store)
    VFromD<D> A, B, C;
    if constexpr (sizeof(T) == 1) detail::Interleave3From_u8(d, v0, v1, v2, A, B, C);
    else detail::Interleave3From_u16(d, v0, v1, v2, A, B, C);
    StoreU(A, d, p + 0 * N); StoreU(B, d, p + 1 * N); StoreU(C, d, p + 2 * N);
  }
}
template <class D> requires HYLE_SIMD_W(D, 0, 16)
HYLE_FORCE_INLINE void StoreInterleaved4(VFromD<D> v0, VFromD<D> v1, VFromD<D> v2, VFromD<D> v3, D d,
                                         TFromD<D>* p) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  if constexpr (sizeof(T) == 4 || sizeof(T) == 8) {
    VFromD<D> A, B, C, Dd;
    if constexpr (sizeof(T) == 4) {
      const Simd<std::uint64_t, N / 2, 0> dw;
      const auto v10L = BitCast(dw, InterleaveLower(v0, v1)), v32L = BitCast(dw, InterleaveLower(v2, v3));
      const auto v10U = BitCast(dw, InterleaveUpper(v0, v1)), v32U = BitCast(dw, InterleaveUpper(v2, v3));
      A = BitCast(d, InterleaveLower(v10L, v32L)); B = BitCast(d, InterleaveUpper(v10L, v32L));
      C = BitCast(d, InterleaveLower(v10U, v32U)); Dd = BitCast(d, InterleaveUpper(v10U, v32U));
    } else {  // 64-bit (N=2)
      A = InterleaveLower(v0, v1); C = InterleaveUpper(v0, v1);
      B = InterleaveLower(v2, v3); Dd = InterleaveUpper(v2, v3);
    }
    StoreU(A, d, p + 0 * N); StoreU(B, d, p + 1 * N); StoreU(C, d, p + 2 * N); StoreU(Dd, d, p + 3 * N);
  } else {  // 8/16-bit: widening-zip (1 block => raw store)
    VFromD<D> A, B, C, Dd;
    detail::Interleave4From_narrow(d, v0, v1, v2, v3, A, B, C, Dd);
    StoreU(A, d, p + 0 * N); StoreU(B, d, p + 1 * N); StoreU(C, d, p + 2 * N); StoreU(Dd, d, p + 3 * N);
  }
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
// GetExponent(v) = floor(log2(|v|)) as float. Extract the IEEE-754 exponent field via integer
// vector ops -- no store+reload/libm (op_parity: the old std::logb path was 30x slower on avx2).
// Single-instruction vgetexpps is AVX-512 only; this bit-math is the portable SSE4 equivalent.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> GetExponent(Vec128<T, N> v) noexcept {
  static_assert(std::is_floating_point_v<T>, "float only");
  Vec128<T, N> r;
  if constexpr (std::is_same_v<T, float>) {
    const __m128i bits = _mm_castps_si128(v.raw);
    const __m128i biased = _mm_and_si128(_mm_srli_epi32(bits, 23), _mm_set1_epi32(0xFF));
    r.raw = _mm_cvtepi32_ps(_mm_sub_epi32(biased, _mm_set1_epi32(127)));
  } else {  // double: 11-bit exponent, bias 1023
    const __m128i bits = _mm_castpd_si128(v.raw);
    const __m128i biased = _mm_and_si128(_mm_srli_epi64(bits, 52), _mm_set1_epi64x(0x7FF));
    const __m128i unbiased = _mm_sub_epi64(biased, _mm_set1_epi64x(1023));
    // i64 -> f64: extract the two lanes (no SSE cvtepi64_pd); exact for |exp| < 2^52.
    alignas(16) std::int64_t e[2];
    _mm_store_si128(reinterpret_cast<__m128i*>(e), unbiased);
    r.raw = _mm_set_pd(static_cast<double>(e[1]), static_cast<double>(e[0]));
  }
  return r;
}
// MulByPow2(a, b) = a * 2^b, b an integer-lane vector. PORTED from Highway generic_ops (see avx2.h
// for the full rationale): 3-part clamped exponent split, each part built as a 2^eK float via
// exponent-bit packing, then multiplied -- entirely in the vector unit, no store+scalar-ldexp+reload
// (that was the ~19x op_parity loss). Correctly rounded through denormal/overflow.
template <typename T, std::size_t N, typename IT>
HYLE_FORCE_INLINE Vec128<T, N> MulByPow2(Vec128<T, N> a, Vec128<IT, N> b) noexcept {
  static_assert(std::is_floating_point_v<T>, "float only");
  static_assert(std::is_integral_v<IT> && std::is_signed_v<IT> && sizeof(IT) == sizeof(T),
                "MulByPow2 exponent is a signed int lane matching the float width");
  const Simd<T, N, 0> df;
  const Simd<IT, N, 0> di;
  constexpr int kMantBits = (sizeof(T) == 4) ? 23 : 52;
  constexpr IT kBias = (sizeof(T) == 4) ? IT{127} : IT{1023};
  const auto clamped = Clamp(b, Set(di, static_cast<IT>(3 - 3 * kBias)), Set(di, static_cast<IT>(3 * kBias)));
  const auto lo = Set(di, static_cast<IT>(1 - kBias));
  const auto hi = Set(di, kBias);
  const auto e3 = Clamp(clamped, lo, hi);
  const auto e2 = Clamp(Sub(clamped, e3), lo, hi);
  const auto e1 = Sub(Sub(clamped, e3), e2);
  const auto biasv = Set(di, kBias);
  const auto f1 = BitCast(df, ShiftLeft<kMantBits>(Add(e1, biasv)));
  const auto f2 = BitCast(df, ShiftLeft<kMantBits>(Add(e2, biasv)));
  const auto f3 = BitCast(df, ShiftLeft<kMantBits>(Add(e3, biasv)));
  return Mul(Mul(Mul(a, f1), f2), f3);
}
// NaN-aware min/max (Highway MinNumber/MaxNumber): if one operand is NaN, return the OTHER.
// x86 _mm_min_ps/_mm_max_ps return the second operand on any-NaN, so we can't map directly;
// reuse IsNaN + IfThenElse + Min/Max. Integer lanes have no NaN, so they are just Min/Max.
// One-sided NaN fixup (see avx2.h): _mm_min/max_ps return the 2nd operand on any-NaN, so
// IfThenElse(IsNaN(b), a, Min(a,b)) suffices (3 ops vs the old double-check's 5). Equivalence over
// all 64 NaN/value combos verified.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> MinNumber(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  if constexpr (std::is_floating_point_v<T>)
    return IfThenElse(IsNaN(b), a, Min(a, b));
  else return Min(a, b);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> MaxNumber(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  if constexpr (std::is_floating_point_v<T>)
    return IfThenElse(IsNaN(b), a, Max(a, b));
  else return Max(a, b);
}
// AddSub(a,b): a-b in EVEN lanes, a+b in ODD lanes. Native SSE3 _mm_addsub_ps/pd (float).
// Integer has no addsub -> Add(a, OddEven(b, -b)) (Highway's generic shape): isolates the a-independent
// OddEven(b,-b) so it hoists out of a loop with invariant b, vs OddEven(Add(a,b),Sub(a,b)) which
// recomputes both a+b and a-b (both a-dependent) each call. Same win as the avx2 path.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> AddSub(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  Vec128<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm_addsub_ps(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm_addsub_pd(a.raw, b.raw);
  else r = Add(a, OddEven(b, Sub(Zero(Simd<T, N, 0>()), b)));  // native addsub is float-only
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
  } else {  // i32/i64: no native satsub. Highway branchless bit trick: ~v - (v>MIN ? -1 : 0) = -v
    // for v>MIN, MAX at v==MIN (~MIN==MAX). 3 ops vs old Eq+Set+Neg+IfThenElse (see avx2.h).
    const Simd<T, N, 0> d;
    return Sub(Not(a), VecFromMask(d, Gt(a, Set(d, std::numeric_limits<T>::min()))));
  }
}
// SaturatedAbs(a) = (a==MIN) ? MAX : |a|. |a| for a<0 is exactly SaturatedNeg(a), so reuse
// Lt/IfThenElse + SaturatedNeg (correct for every signed width, MIN maps to MAX).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> SaturatedAbs(Vec128<T, N> a) noexcept {
  static_assert(std::is_integral_v<T> && std::is_signed_v<T>, "SaturatedAbs is signed-int");
  if constexpr (sizeof(T) == 4) {  // Highway x86 i32: Min_u32(|v|, INT_MAX) -- |MIN| overflows to
    // 0x80000000 (largest u32) so the unsigned Min clamps it. 2 ops (vpabsd+vpminud) vs 4 (see avx2).
    const Simd<T, N, 0> d;
    const Simd<std::uint32_t, N, 0> du;
    return BitCast(d, Min(BitCast(du, Abs(a)), Set(du, static_cast<std::uint32_t>(0x7FFFFFFF))));
  } else {  // i8/i16 (native satsub) / i64
    return Max(a, SaturatedNeg(a));
  }
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
  alignas(16) T va[16 / sizeof(T)], vb[16 / sizeof(T)];
  _mm_store_si128(reinterpret_cast<__m128i*>(va), a.raw);
  _mm_store_si128(reinterpret_cast<__m128i*>(vb), b.raw);
  for (std::size_t i = 0; i < N; ++i)
    va[i] = left ? static_cast<T>(static_cast<std::make_unsigned_t<T>>(va[i]) << vb[i])
                 : static_cast<T>(va[i] >> vb[i]);
  Vec128<T, N> r;
  r.raw = _mm_load_si128(reinterpret_cast<const __m128i*>(va));
  return r;
}
// Pow2_u32(bits) = 2^bits per 32-bit lane, VECTORIZED (Highway x86_128 Pow2). SSE4 has no
// per-lane variable shift (_mm_sllv/_mm_srlv are AVX2), but "v << bits" == "v * 2^bits" and
// "v >>u bits" == "MulHigh(v, 2^(32-bits))" -- both need 2^k as a vector, which we build by
// stuffing k into an f32 exponent field (k in [0,31] -> 2^k is an EXACT power of two an f32
// represents precisely) and converting back to int. This replaces the store+scalar-loop+reload
// (op_parity -n15: sse4 Shl ~2.0x / RoundingShr ~1.7x / MaskedShl ~2.2x behind Highway).
HYLE_FORCE_INLINE __m128i Pow2_u32(__m128i bits) noexcept {
  const __m128i exp = _mm_slli_epi32(bits, 23);                     // k into the f32 exponent slot
  const __m128i fbits = _mm_add_epi32(exp, _mm_set1_epi32(0x3F800000));  // + 1.0f bit pattern
  const __m128 f = _mm_castsi128_ps(fbits);
  // (int)2^k. k in [0,31] -> 2^k is exact in f32; 2^31/2^32 overflow int32 and cvttps2dq yields
  // 0x80000000 (= 2^31 as u32), which is EXACTLY the multiplier we want. But GCC/Clang, when they
  // can see the operand at compile time, constant-fold (int)float-overflow to SATURATION
  // (0x7FFFFFFF) instead -- silently off-by-one (caught by VarShiftUint32). Force the real
  // cvttps2dq via inline asm (same reason Highway wraps it); MSVC doesn't do this fold.
#if defined(__GNUC__) || defined(__clang__)
  __m128i r;
  __asm__("cvttps2dq %1, %0" : "=x"(r) : "x"(f));
  return r;
#else
  return _mm_cvttps_epi32(f);
#endif
}
}  // namespace detail
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> Shl(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  static_assert(std::is_integral_v<T>, "Shl is integer-lane only");
  if constexpr (sizeof(T) == 4) {  // i32/u32: v << b == v * 2^b, one mullo (Highway手法)
    Vec128<T, N> r;
    r.raw = _mm_mullo_epi32(a.raw, detail::Pow2_u32(b.raw));
    return r;
  } else {  // TODO(i8/i16/i64): each needs a different width-banded mul; not bench losers yet.
    return detail::shift_var_scalar<T, N>(a, b, true);
  }
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> Shr(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  static_assert(std::is_integral_v<T>, "Shr is integer-lane only");
  if constexpr (sizeof(T) == 4) {
    // u32: v >>u b == high 32 of (v * 2^(32-b)); do it on even+odd lanes via _mm_mul_epu32, then
    // recombine. b==0 would need 2^32 (overflows) so patch those lanes back to v. (Highway手法.)
    const __m128i b32 = _mm_sub_epi32(_mm_set1_epi32(32), b.raw);
    const __m128i mul = detail::Pow2_u32(b32);
    const __m128i in31 = _mm_shuffle_epi32(a.raw, 0x31);   // odd lanes -> even slots
    const __m128i mul31 = _mm_shuffle_epi32(mul, 0x31);
    const __m128i out20 = _mm_srli_epi64(_mm_mul_epu32(a.raw, mul), 32);  // z 2 z 0
    const __m128i out31 = _mm_mul_epu32(in31, mul31);                     // 3 ? 1 ? (already hi)
    // OddEven(out31, out20): even lanes from out20, odd lanes from out31.
    const __m128i out = _mm_castps_si128(_mm_blend_ps(
        _mm_castsi128_ps(out20), _mm_castsi128_ps(out31), 0b1010));
    const __m128i zero = _mm_cmpeq_epi32(b.raw, _mm_setzero_si128());     // b==0 mask
    const __m128i ru = _mm_or_si128(_mm_and_si128(zero, a.raw), _mm_andnot_si128(zero, out));
    Vec128<T, N> r;
    if constexpr (std::is_signed_v<T>) {
      // arithmetic: clear sign, logical-shift, restore sign (Highway SignedShr).
      const __m128i sign = _mm_srai_epi32(a.raw, 31);
      const __m128i absx = _mm_xor_si128(a.raw, sign);
      const __m128i b32s = _mm_sub_epi32(_mm_set1_epi32(32), b.raw);
      const __m128i muls = detail::Pow2_u32(b32s);
      const __m128i s31 = _mm_shuffle_epi32(absx, 0x31);
      const __m128i m31 = _mm_shuffle_epi32(muls, 0x31);
      const __m128i o20 = _mm_srli_epi64(_mm_mul_epu32(absx, muls), 32);
      const __m128i o31 = _mm_mul_epu32(s31, m31);
      const __m128i o = _mm_castps_si128(_mm_blend_ps(_mm_castsi128_ps(o20), _mm_castsi128_ps(o31), 0b1010));
      const __m128i z = _mm_cmpeq_epi32(b.raw, _mm_setzero_si128());
      const __m128i shifted = _mm_or_si128(_mm_and_si128(z, absx), _mm_andnot_si128(z, o));
      r.raw = _mm_xor_si128(shifted, sign);
    } else {
      r.raw = ru;
    }
    return r;
  } else {  // TODO(i8/i16/i64): width-banded; not bench losers yet.
    return detail::shift_var_scalar<T, N>(a, b, false);
  }
}

// Uniform run-time shifts are native even though SSE4 lacks per-lane variable shifts.
template <std::size_t N>
HYLE_FORCE_INLINE Vec128<std::uint32_t, N> ShiftRightUniform(Vec128<std::uint32_t, N> a,
                                                              int bits) noexcept {
  static_assert(N == 4, "ShiftRightUniform u32 requires a native SSE4 vector");
  return Vec128<std::uint32_t, N>{_mm_srl_epi32(a.raw, _mm_cvtsi32_si128(bits))};
}
template <std::size_t N>
HYLE_FORCE_INLINE Vec128<std::int32_t, N> ShiftRightUniform(Vec128<std::int32_t, N> a,
                                                             int bits) noexcept {
  static_assert(N == 4, "ShiftRightUniform i32 requires a native SSE4 vector");
  return Vec128<std::int32_t, N>{_mm_sra_epi32(a.raw, _mm_cvtsi32_si128(bits))};
}

// Fixed-width native hook used by vec.h's all-integer MulHigh contract.
namespace detail {
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> MulHighLow8Native(Vec128<T, N> a, Vec128<T, N> b,
                                                  Vec128<T, N>& high) noexcept {
  static_assert(std::is_integral_v<T> && sizeof(T) == 1, "MulHighLow8Native is 8-bit int only");
  static_assert(N == 16, "MulHighLow8Native requires a native SSE4 vector");
  const __m128i even_a =
      std::is_signed_v<T> ? _mm_srai_epi16(_mm_slli_epi16(a.raw, 8), 8)
                          : _mm_and_si128(a.raw, _mm_set1_epi16(0x00FF));
  const __m128i even_b =
      std::is_signed_v<T> ? _mm_srai_epi16(_mm_slli_epi16(b.raw, 8), 8)
                          : _mm_and_si128(b.raw, _mm_set1_epi16(0x00FF));
  const __m128i odd_a = std::is_signed_v<T> ? _mm_srai_epi16(a.raw, 8)
                                             : _mm_srli_epi16(a.raw, 8);
  const __m128i odd_b = std::is_signed_v<T> ? _mm_srai_epi16(b.raw, 8)
                                             : _mm_srli_epi16(b.raw, 8);
  const __m128i even = _mm_mullo_epi16(even_a, even_b);
  const __m128i odd = _mm_mullo_epi16(odd_a, odd_b);
  high.raw = _mm_blendv_epi8(odd, _mm_srli_epi16(even, 8), _mm_set1_epi16(0x00FF));
  return Vec128<T, N>{
      _mm_blendv_epi8(even, _mm_slli_epi16(odd, 8),
                      _mm_set1_epi16(static_cast<short>(0xFF00)))};
}

template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> MulHigh8Native(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  static_assert(std::is_integral_v<T> && sizeof(T) == 1, "MulHigh8Native is 8-bit int only");
  static_assert(N == 16, "MulHigh8Native requires a native SSE4 vector");
  const __m128i even_a =
      std::is_signed_v<T> ? _mm_srai_epi16(_mm_slli_epi16(a.raw, 8), 8)
                          : _mm_and_si128(a.raw, _mm_set1_epi16(0x00FF));
  const __m128i even_b =
      std::is_signed_v<T> ? _mm_srai_epi16(_mm_slli_epi16(b.raw, 8), 8)
                          : _mm_and_si128(b.raw, _mm_set1_epi16(0x00FF));
  const __m128i odd_a = std::is_signed_v<T> ? _mm_srai_epi16(a.raw, 8)
                                             : _mm_srli_epi16(a.raw, 8);
  const __m128i odd_b = std::is_signed_v<T> ? _mm_srai_epi16(b.raw, 8)
                                             : _mm_srli_epi16(b.raw, 8);
  const __m128i even = _mm_mullo_epi16(even_a, even_b);
  const __m128i odd = _mm_mullo_epi16(odd_a, odd_b);
  return Vec128<T, N>{
      _mm_blendv_epi8(odd, _mm_srli_epi16(even, 8), _mm_set1_epi16(0x00FF))};
}

template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> MulHigh16Native(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  static_assert(std::is_integral_v<T> && sizeof(T) == 2, "MulHigh16Native is 16-bit int only");
  Vec128<T, N> r;
  if constexpr (std::is_signed_v<T>) r.raw = _mm_mulhi_epi16(a.raw, b.raw);
  else r.raw = _mm_mulhi_epu16(a.raw, b.raw);
  return r;
}
}  // namespace detail

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
// SumOfMulQuadAccumulate is defined AFTER PromoteEvenTo/PromoteOddTo (below) so it can use the
// vectorized i8->i16 split; see its definition following ConcatOdd.

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
// Overload selection avoids feeding GCC vector types through std::is_same,
// which discards their target attributes and warns under -Werror. All nine
// raw-class pairs still lower to a move or a zero-instruction cast intrinsic.
HYLE_FORCE_INLINE __m128 bitcast_raw_sse_impl(__m128 s, __m128*) noexcept { return s; }
HYLE_FORCE_INLINE __m128i bitcast_raw_sse_impl(__m128i s, __m128i*) noexcept { return s; }
HYLE_FORCE_INLINE __m128d bitcast_raw_sse_impl(__m128d s, __m128d*) noexcept { return s; }
HYLE_FORCE_INLINE __m128i bitcast_raw_sse_impl(__m128 s, __m128i*) noexcept {
  return _mm_castps_si128(s);
}
HYLE_FORCE_INLINE __m128i bitcast_raw_sse_impl(__m128d s, __m128i*) noexcept {
  return _mm_castpd_si128(s);
}
HYLE_FORCE_INLINE __m128 bitcast_raw_sse_impl(__m128i s, __m128*) noexcept {
  return _mm_castsi128_ps(s);
}
HYLE_FORCE_INLINE __m128d bitcast_raw_sse_impl(__m128i s, __m128d*) noexcept {
  return _mm_castsi128_pd(s);
}
HYLE_FORCE_INLINE __m128d bitcast_raw_sse_impl(__m128 s, __m128d*) noexcept {
  return _mm_castps_pd(s);
}
HYLE_FORCE_INLINE __m128 bitcast_raw_sse_impl(__m128d s, __m128*) noexcept {
  return _mm_castpd_ps(s);
}
template <typename DstRaw, typename SrcRaw>
HYLE_FORCE_INLINE DstRaw bitcast_raw_sse(SrcRaw s) noexcept {
  return bitcast_raw_sse_impl(s, static_cast<DstRaw*>(nullptr));
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
// simd-18 C1: OrderedDemote2To == Reorder (single 128-bit block is already ordered).
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec128<DstT, NDst> OrderedDemote2To(Simd<DstT, NDst, 0> dn, Vec128<SrcT, NSrc> a,
                                                      Vec128<SrcT, NSrc> b) noexcept {
  return ReorderDemote2To(dn, a, b);
}
// simd-18 C3: ConvertInRangeTo -- raw cvtt alias.
template <typename DstT, std::size_t N, typename SrcT>
HYLE_FORCE_INLINE Vec128<DstT, N> ConvertInRangeTo(Simd<DstT, N, 0> d, Vec128<SrcT, N> v) noexcept {
  return ConvertTo(d, v);
}
// simd-18 C3: DemoteInRangeTo f64->i32 -- raw cvttpd (2xf64 -> 2xi32 in low half of __m128i).
template <std::size_t NDst, std::size_t NSrc>
HYLE_FORCE_INLINE Vec128<std::int32_t, NDst> DemoteInRangeTo(Simd<std::int32_t, NDst, 0>,
                                                              Vec128<double, NSrc> v) noexcept {
  static_assert(NDst == NSrc, "DemoteInRangeTo: same lane count f64->i32");
  Vec128<std::int32_t, NDst> r;
  r.raw = _mm_cvttpd_epi32(v.raw);
  return r;
}
// TruncateTo/PromoteEvenTo/PromoteOddTo/U8FromU32: strided/modular width changers.
// correctness-first store+reload; native pack/cvt narrowing is a profiling upgrade.
// Store()/Load() move a FULL 128-bit register (_mm_store_si128 / _mm_load_si128), so the
// scratch buffers MUST be a whole register wide (16 B) regardless of the logical lane count:
// a narrow partial result (e.g. Vec128<u8,4>) still reloads via _mm_load_si128 reading 16 B,
// so a `DstT dst[4]` (4 B) would be an out-of-bounds stack read. Size every buffer to 16 B.
// (Found via GCC segfault on Vec.TruncateAndStridedPromote; MSVC happened not to fault.)
template <typename DstT, std::size_t N, typename SrcT>
HYLE_FORCE_INLINE Vec128<DstT, N> TruncateTo(Simd<DstT, N, 0>, Vec128<SrcT, N> v) noexcept {
  static_assert(sizeof(DstT) < sizeof(SrcT), "TruncateTo narrows the lane type");
  if constexpr (std::is_integral_v<SrcT> && std::is_integral_v<DstT>) {
    // Modular narrowing = keep the low sizeof(DstT) bytes of each source lane, packed into the
    // low N*sizeof(DstT) bytes of the result. One PSHUFB byte-gather does this in-register (the op
    // is per-128-lane, and everything is one 128-bit lane here) -- no store+scalar-loop+reload,
    // which the old code did and which -O3 did NOT fold to a pack (op_parity -n15 was ~2x Highway).
    constexpr int sb = static_cast<int>(sizeof(SrcT)), db = static_cast<int>(sizeof(DstT));
    alignas(16) constexpr auto idx = [] {
      std::array<char, 16> t{};
      for (int j = 0; j < 16; ++j) t[j] = static_cast<char>(0x80);  // default: emit 0
      for (std::size_t i = 0; i < N; ++i)
        for (int k = 0; k < db; ++k)
          t[i * db + k] = static_cast<char>(i * sb + k);  // dst lane i byte k <- src lane i byte k
      return t;
    }();
    const __m128i shuf = _mm_loadu_si128(reinterpret_cast<const __m128i*>(idx.data()));
    Vec128<DstT, N> r;
    r.raw = _mm_shuffle_epi8(v.raw, shuf);
    return r;
  } else {
    alignas(16) SrcT src[16 / sizeof(SrcT)];
    alignas(16) DstT dst[16 / sizeof(DstT)] = {};
    Store(v, Simd<SrcT, N, 0>(), src);
    for (std::size_t i = 0; i < N; ++i) dst[i] = static_cast<DstT>(src[i]);
    return Load(Simd<DstT, N, 0>(), dst);
  }
}
// PromoteEvenTo/PromoteOddTo: exact-2x integer widenings use a register-in shift/mask path
// (mirrors avx2 with 128-bit intrinsics). In a source pair viewed as one destination lane, the
// EVEN source lane sits in the LOW half and the ODD in the HIGH half. Signed = arithmetic shift
// to sign-fill; unsigned = logical shift / mask. SSE has no _mm_srai_epi64, so signed i32->i64
// rebuilds sign fill from _mm_srai_epi32(.,31). Non-integer / non-2x / multi-level combos keep
// the correctness-first store+reload fallback.
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec128<DstT, NDst> PromoteEvenTo(Simd<DstT, NDst, 0>, Vec128<SrcT, NSrc> v) noexcept {
  static_assert(NSrc == 2 * NDst && sizeof(DstT) > sizeof(SrcT), "PromoteEvenTo: even lanes narrow->wide");
  if constexpr (std::is_integral_v<SrcT> && std::is_integral_v<DstT> && sizeof(DstT) == 2 * sizeof(SrcT)) {
    Vec128<DstT, NDst> r;
    if constexpr (sizeof(SrcT) == 1) {  // i8/u8 -> i16/u16: even byte is the low byte of each i16
      if constexpr (std::is_signed_v<SrcT>)
        r.raw = _mm_srai_epi16(_mm_slli_epi16(v.raw, 8), 8);  // shift up to hi byte, sign-fill back
      else
        r.raw = _mm_and_si128(v.raw, _mm_set1_epi16(static_cast<short>(0x00FF)));  // zero-extend
    } else if constexpr (sizeof(SrcT) == 2) {  // i16/u16 -> i32/u32: even word is the low word of each i32
      if constexpr (std::is_signed_v<SrcT>)
        r.raw = _mm_srai_epi32(_mm_slli_epi32(v.raw, 16), 16);
      else
        r.raw = _mm_and_si128(v.raw, _mm_set1_epi32(0x0000FFFF));
    } else {  // sizeof 4: i32/u32 -> i64/u64; even dword is the low dword of each i64
      if constexpr (std::is_signed_v<SrcT>) {
        // No srai_epi64: low = even dword, high = sign(even dword) (its 32-bit sign shifted up).
        const __m128i lo = _mm_and_si128(v.raw, _mm_set1_epi64x(0xFFFFFFFFLL));
        const __m128i hi = _mm_slli_epi64(_mm_srai_epi32(v.raw, 31), 32);
        r.raw = _mm_or_si128(lo, hi);
      } else {
        r.raw = _mm_and_si128(v.raw, _mm_set1_epi64x(0xFFFFFFFFLL));
      }
    }
    return r;
  } else {
    alignas(16) SrcT src[16 / sizeof(SrcT)];
    alignas(16) DstT dst[16 / sizeof(DstT)] = {};
    Store(v, Simd<SrcT, NSrc, 0>(), src);
    for (std::size_t i = 0; i < NDst; ++i) dst[i] = static_cast<DstT>(src[2 * i]);
    return Load(Simd<DstT, NDst, 0>(), dst);
  }
}
template <typename DstT, std::size_t NDst, typename SrcT, std::size_t NSrc>
HYLE_FORCE_INLINE Vec128<DstT, NDst> PromoteOddTo(Simd<DstT, NDst, 0>, Vec128<SrcT, NSrc> v) noexcept {
  static_assert(NSrc == 2 * NDst && sizeof(DstT) > sizeof(SrcT), "PromoteOddTo: odd lanes narrow->wide");
  if constexpr (std::is_integral_v<SrcT> && std::is_integral_v<DstT> && sizeof(DstT) == 2 * sizeof(SrcT)) {
    Vec128<DstT, NDst> r;
    if constexpr (sizeof(SrcT) == 1) {  // i8/u8 -> i16/u16: odd byte already in the high byte of each i16
      if constexpr (std::is_signed_v<SrcT>)
        r.raw = _mm_srai_epi16(v.raw, 8);  // arithmetic: sign-fill from the hi byte
      else
        r.raw = _mm_srli_epi16(v.raw, 8);  // logical: zero-extend from the hi byte
    } else if constexpr (sizeof(SrcT) == 2) {  // i16/u16 -> i32/u32: odd word in the high word of each i32
      if constexpr (std::is_signed_v<SrcT>)
        r.raw = _mm_srai_epi32(v.raw, 16);
      else
        r.raw = _mm_srli_epi32(v.raw, 16);
    } else {  // sizeof 4: i32/u32 -> i64/u64; odd dword in the high dword of each i64
      if constexpr (std::is_signed_v<SrcT>) {
        // No srai_epi64: low = odd dword (logical down), high = sign of the odd dword (kept in place).
        const __m128i lo = _mm_srli_epi64(v.raw, 32);
        const __m128i hi = _mm_and_si128(_mm_srai_epi32(v.raw, 31),
                                         _mm_set1_epi64x(static_cast<long long>(0xFFFFFFFF00000000ULL)));
        r.raw = _mm_or_si128(lo, hi);
      } else {
        r.raw = _mm_srli_epi64(v.raw, 32);
      }
    }
    return r;
  } else {
    alignas(16) SrcT src[16 / sizeof(SrcT)];
    alignas(16) DstT dst[16 / sizeof(DstT)] = {};
    Store(v, Simd<SrcT, NSrc, 0>(), src);
    for (std::size_t i = 0; i < NDst; ++i) dst[i] = static_cast<DstT>(src[2 * i + 1]);
    return Load(Simd<DstT, NDst, 0>(), dst);
  }
}
template <std::size_t N>
HYLE_FORCE_INLINE Vec128<std::uint8_t, N> U8FromU32(Vec128<std::uint32_t, N> v) noexcept {
  alignas(16) std::uint32_t src[16 / sizeof(std::uint32_t)];
  alignas(16) std::uint8_t dst[16] = {};
  Store(v, Simd<std::uint32_t, N, 0>(), src);
  for (std::size_t i = 0; i < N; ++i) dst[i] = static_cast<std::uint8_t>(src[i]);
  return Load(Simd<std::uint8_t, N, 0>(), dst);
}

// P4: SSE4 dword->qword multiply consumes the even dword of each qword.
template <std::size_t N>
HYLE_FORCE_INLINE Vec128<std::int64_t, N / 2> MulEven(Vec128<std::int32_t, N> a,
                                                       Vec128<std::int32_t, N> b) noexcept {
  static_assert(N == 4, "MulEven i32 requires a native SSE4 vector");
  return Vec128<std::int64_t, N / 2>{_mm_mul_epi32(a.raw, b.raw)};
}
template <std::size_t N>
HYLE_FORCE_INLINE Vec128<std::uint64_t, N / 2> MulEven(Vec128<std::uint32_t, N> a,
                                                        Vec128<std::uint32_t, N> b) noexcept {
  static_assert(N == 4, "MulEven u32 requires a native SSE4 vector");
  return Vec128<std::uint64_t, N / 2>{_mm_mul_epu32(a.raw, b.raw)};
}
template <std::size_t N>
HYLE_FORCE_INLINE Vec128<std::int64_t, N / 2> MulOdd(Vec128<std::int32_t, N> a,
                                                      Vec128<std::int32_t, N> b) noexcept {
  static_assert(N == 4, "MulOdd i32 requires a native SSE4 vector");
  return Vec128<std::int64_t, N / 2>{_mm_mul_epi32(_mm_srli_epi64(a.raw, 32),
                                                    _mm_srli_epi64(b.raw, 32))};
}
template <std::size_t N>
HYLE_FORCE_INLINE Vec128<std::uint64_t, N / 2> MulOdd(Vec128<std::uint32_t, N> a,
                                                       Vec128<std::uint32_t, N> b) noexcept {
  static_assert(N == 4, "MulOdd u32 requires a native SSE4 vector");
  return Vec128<std::uint64_t, N / 2>{_mm_mul_epu32(_mm_srli_epi64(a.raw, 32),
                                                     _mm_srli_epi64(b.raw, 32))};
}

namespace detail {
template <std::size_t N>
HYLE_FORCE_INLINE Vec128<std::uint32_t, N> MulHigh32Native(Vec128<std::uint32_t, N> a,
                                                            Vec128<std::uint32_t, N> b) noexcept {
  static_assert(N == 4, "MulHigh32 u32 requires a native SSE4 vector");
  const __m128i even = _mm_srli_epi64(_mm_mul_epu32(a.raw, b.raw), 32);
  const __m128i odd = _mm_srli_epi64(
      _mm_mul_epu32(_mm_srli_epi64(a.raw, 32), _mm_srli_epi64(b.raw, 32)), 32);
  return Vec128<std::uint32_t, N>{_mm_or_si128(even, _mm_slli_epi64(odd, 32))};
}
template <std::size_t N>
HYLE_FORCE_INLINE Vec128<std::int32_t, N> MulHigh32Native(Vec128<std::int32_t, N> a,
                                                           Vec128<std::int32_t, N> b) noexcept {
  static_assert(N == 4, "MulHigh32 i32 requires a native SSE4 vector");
  const __m128i even = _mm_mul_epi32(a.raw, b.raw);
  const __m128i odd = _mm_mul_epi32(_mm_shuffle_epi32(a.raw, _MM_SHUFFLE(3, 3, 1, 1)),
                                    _mm_shuffle_epi32(b.raw, _MM_SHUFFLE(3, 3, 1, 1)));
  const __m128i packed = _mm_castps_si128(
      _mm_shuffle_ps(_mm_castsi128_ps(even), _mm_castsi128_ps(odd), _MM_SHUFFLE(3, 1, 3, 1)));
  return Vec128<std::int32_t, N>{_mm_shuffle_epi32(packed, _MM_SHUFFLE(3, 1, 2, 0))};
}
}  // namespace detail

// ---- swizzle (lane rearrangement) ---------------------------------------------------
// SSE4 packs the whole vector in a single 128-bit register (like NEON, unlike AVX2's two
// 128-bit halves). So there is no lane-crossing hazard: InterleaveLower/Upper operate on the
// whole vector, and Concat*/Reverse span all lanes. These are non-hot-path ops, so a simple
// store+reload through an alignas(16) scratch buffer is correct and clear -- the scalar
// backend is the oracle and this per-lane logic matches it (as validated on NEON).
namespace detail {
template <typename T, std::size_t N, typename IdxFn>
HYLE_FORCE_INLINE Vec128<T, N> swizzle_via_mem(Vec128<T, N> v, IdxFn idx) noexcept {
  alignas(16) T src[16 / sizeof(T)], dst[16 / sizeof(T)];
  Store(v, Simd<T, N, 0>(), src);
  for (std::size_t i = 0; i < N; ++i) dst[i] = src[idx(i)];
  return Load(Simd<T, N, 0>(), dst);
}
}  // namespace detail

// OddEven(a,b): even lanes <- b, odd <- a.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> OddEven(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  alignas(16) T sa[16 / sizeof(T)], sb[16 / sizeof(T)], dst[16 / sizeof(T)];
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
  alignas(16) T sa[16 / sizeof(T)], sb[16 / sizeof(T)], dst[16 / sizeof(T)];
  Store(a, Simd<T, N, 0>(), sa); Store(b, Simd<T, N, 0>(), sb);
  for (std::size_t i = 0; i < N / 2; ++i) { dst[2 * i] = sa[i]; dst[2 * i + 1] = sb[i]; }
  return Load(Simd<T, N, 0>(), dst);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> InterleaveUpper(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  alignas(16) T sa[16 / sizeof(T)], sb[16 / sizeof(T)], dst[16 / sizeof(T)];
  Store(a, Simd<T, N, 0>(), sa); Store(b, Simd<T, N, 0>(), sb);
  for (std::size_t i = 0; i < N / 2; ++i) { dst[2 * i] = sa[N / 2 + i]; dst[2 * i + 1] = sb[N / 2 + i]; }
  return Load(Simd<T, N, 0>(), dst);
}
namespace detail {
template <int loHalf, int hiHalf, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> concat_halves(Vec128<T, N> hi, Vec128<T, N> lo) noexcept {
  alignas(16) T sh[16 / sizeof(T)], sl[16 / sizeof(T)], dst[16 / sizeof(T)];
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
  // [lo even lanes, hi even lanes]. NATIVE for 32-bit (4 lanes): one shuffle_ps picks lanes
  // {0,2} of lo into the low half and {0,2} of hi into the high half (op_parity: avx2 store+reload
  // was 1.7x slower). Narrower widths keep the mem gather.
  if constexpr (sizeof(T) == 4) {
    Vec128<T, N> r;
    if constexpr (std::is_same_v<T, float>)
      r.raw = _mm_shuffle_ps(lo.raw, hi.raw, _MM_SHUFFLE(2, 0, 2, 0));
    else
      r.raw = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(lo.raw), _mm_castsi128_ps(hi.raw),
                                              _MM_SHUFFLE(2, 0, 2, 0)));
    return r;
  } else {
    alignas(16) T sh[16 / sizeof(T)], sl[16 / sizeof(T)], dst[16 / sizeof(T)];
    Store(hi, Simd<T, N, 0>(), sh); Store(lo, Simd<T, N, 0>(), sl);
    for (std::size_t i = 0; i < N / 2; ++i) { dst[i] = sl[2 * i]; dst[N / 2 + i] = sh[2 * i]; }
    return Load(Simd<T, N, 0>(), dst);
  }
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> ConcatOdd(Vec128<T, N> hi, Vec128<T, N> lo) noexcept {
  // [lo odd lanes, hi odd lanes]. NATIVE for 32-bit: shuffle_ps picks lanes {1,3}.
  if constexpr (sizeof(T) == 4) {
    Vec128<T, N> r;
    if constexpr (std::is_same_v<T, float>)
      r.raw = _mm_shuffle_ps(lo.raw, hi.raw, _MM_SHUFFLE(3, 1, 3, 1));
    else
      r.raw = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(lo.raw), _mm_castsi128_ps(hi.raw),
                                              _MM_SHUFFLE(3, 1, 3, 1)));
    return r;
  } else {
    alignas(16) T sh[16 / sizeof(T)], sl[16 / sizeof(T)], dst[16 / sizeof(T)];
    Store(hi, Simd<T, N, 0>(), sh); Store(lo, Simd<T, N, 0>(), sl);
    for (std::size_t i = 0; i < N / 2; ++i) { dst[i] = sl[2 * i + 1]; dst[N / 2 + i] = sh[2 * i + 1]; }
    return Load(Simd<T, N, 0>(), dst);
  }
}
// SumOfMulQuadAccumulate(di32, a_i8, b_i8, sum): sum[i] += 4-way i8 dot. PORTED from Highway generic
// (see avx2.h): split each i8 operand into even/odd i16 halves, two WidenMulPairwiseAdd (pmaddwd)
// sum the pairs, add over even+odd = the quad dot. Fully vectorized (defined here, after
// PromoteEven/OddTo, so those are visible). Replaces the store + scalar 4-way loop + reload.
template <std::size_t NDst, std::size_t NSrc>
HYLE_FORCE_INLINE Vec128<std::int32_t, NDst> SumOfMulQuadAccumulate(
    Simd<std::int32_t, NDst, 0> di32, Vec128<std::int8_t, NSrc> a, Vec128<std::int8_t, NSrc> b,
    Vec128<std::int32_t, NDst> sum) noexcept {
  static_assert(NSrc == 4 * NDst, "SumOfMulQuadAccumulate: i8 -> i32 (dst quarter lanes)");
  const Simd<std::int16_t, NSrc / 2, 0> di16;
  const auto a0 = PromoteEvenTo(di16, a);
  const auto b0 = PromoteEvenTo(di16, b);
  const auto a1 = PromoteOddTo(di16, a);
  const auto b1 = PromoteOddTo(di16, b);
  return Add(sum, Add(WidenMulPairwiseAdd(di32, a0, b0), WidenMulPairwiseAdd(di32, a1, b1)));
}
// DupEven/DupOdd: broadcast the even/odd element of each adjacent pair. NATIVE for 32/64-bit lanes
// (op_parity: avx2 store+reload was 10x slower); narrow 8/16-bit keep the mem swizzle.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> DupEven(Vec128<T, N> v) noexcept {
  Vec128<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm_moveldup_ps(v.raw);            // [0,0,2,2]
  else if constexpr (detail::kI32<T>) r.raw = _mm_shuffle_epi32(v.raw, _MM_SHUFFLE(2, 2, 0, 0));
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm_shuffle_pd(v.raw, v.raw, 0);  // [0,0]
  else if constexpr (detail::kI64<T>)
    r.raw = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(v.raw), _mm_castsi128_pd(v.raw), 0));
  else r = detail::swizzle_via_mem(v, [](std::size_t i) { return i & ~std::size_t{1}; });
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> DupOdd(Vec128<T, N> v) noexcept {
  Vec128<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm_movehdup_ps(v.raw);            // [1,1,3,3]
  else if constexpr (detail::kI32<T>) r.raw = _mm_shuffle_epi32(v.raw, _MM_SHUFFLE(3, 3, 1, 1));
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm_shuffle_pd(v.raw, v.raw, 3);  // [1,1]
  else if constexpr (detail::kI64<T>)
    r.raw = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(v.raw), _mm_castsi128_pd(v.raw), 3));
  else r = detail::swizzle_via_mem(v, [](std::size_t i) { return (i & ~std::size_t{1}) | 1; });
  return r;
}
template <typename T, std::size_t N>
struct Indices128 { Vec128<T, N> idx; };
template <class D>
HYLE_FORCE_INLINE Indices128<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> IndicesFromVec(D, detail::V128<D> iv) noexcept {
  return Indices128<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)>{iv};
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> TableLookupLanes(Vec128<T, N> v, Indices128<T, N> ind) noexcept {
  alignas(16) T src[16 / sizeof(T)], idx[16 / sizeof(T)], dst[16 / sizeof(T)];
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
// TableLookupBytesInRange: index bytes in [0,15] guaranteed by caller -> raw pshufb (see avx2.h).
template <typename T, std::size_t N, typename TI, std::size_t NI>
HYLE_FORCE_INLINE Vec128<TI, NI> TableLookupBytesInRange(Vec128<T, N> bytes, Vec128<TI, NI> idx) noexcept {
  static_assert(sizeof(T) * N == sizeof(TI) * NI, "TableLookupBytesInRange: same total byte width");
  const __m128i br = detail::bitcast_raw_sse<__m128i>(bytes.raw);
  const __m128i ir = detail::bitcast_raw_sse<__m128i>(idx.raw);
  Vec128<TI, NI> r;
  r.raw = detail::bitcast_raw_sse<detail::Raw128T<TI>>(_mm_shuffle_epi8(br, ir));
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
// mask ? yes : 0  and  mask ? 0 : no. SSE4 mask raw IS the all-ones/all-zero lane pattern, so
// zeroing is a bitwise AND / ANDNOT -- one pand/andps (1 cyc) instead of blendv (2 cyc, port-5).
// Highway's exact SSE4 path (yes & VecFromMask). Feeds every generic Masked op.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> IfThenElseZero(Mask128<T, N> m, Vec128<T, N> yes) noexcept {
  Vec128<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm_and_ps(yes.raw, m.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm_and_pd(yes.raw, m.raw);
  else r.raw = _mm_and_si128(yes.raw, m.raw);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> IfThenZeroElse(Mask128<T, N> m, Vec128<T, N> no) noexcept {
  // ~mask & no
  Vec128<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm_andnot_ps(m.raw, no.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm_andnot_pd(m.raw, no.raw);
  else r.raw = _mm_andnot_si128(m.raw, no.raw);
  return r;
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
  // Width-matched movemask so EVERY lane width is ONE instruction, not movemask_epi8 + a scalar
  // bit-compaction loop (mirrors the avx2 fix: 4-byte -> movemask_ps, 8-byte -> movemask_pd). A mask
  // lane is all-ones/all-zero so its SIGN bit represents it. Only 2-byte lanes lack a native
  // movemask and keep the byte-mask compaction. Removes the ~N-iteration scalar tail that dominated
  // every sse4 CountTrue/FindFirstTrue/Compress/mask-op (op_parity: was ~2.5x slower than Highway).
  if constexpr (std::is_same_v<T, double>) {
    return static_cast<unsigned>(_mm_movemask_pd(m.raw));
  } else if constexpr (std::is_same_v<T, float>) {
    return static_cast<unsigned>(_mm_movemask_ps(m.raw));
  } else if constexpr (sizeof(T) == 8) {  // i64/u64 raw is __m128i -> reinterpret to pd
    return static_cast<unsigned>(_mm_movemask_pd(_mm_castsi128_pd(m.raw)));
  } else if constexpr (sizeof(T) == 4) {  // i32/u32 raw is __m128i -> reinterpret to ps
    return static_cast<unsigned>(_mm_movemask_ps(_mm_castsi128_ps(m.raw)));
  } else if constexpr (sizeof(T) == 1) {
    return static_cast<unsigned>(_mm_movemask_epi8(m.raw));  // 16 lanes, 1 bit/byte already
  } else {  // 2-byte lanes: no native 16-bit movemask. Pack i16 lanes down to i8 (saturating: a mask
    // lane -1 stays -1, 0 stays 0), THEN movemask_epi8 gives exactly 1 bit/lane -- Highway's x86_128
    // recipe. Replaces the ~N-iteration scalar bit-compaction (which GCC vectorized into a
    // vpsrlvd/vpsllvd/vpor fold, the i16 BitsFromMask bottleneck behind DemoteMaskTo's ~0.33 vs HWY).
    const __m128i packed = _mm_packs_epi16(m.raw, _mm_setzero_si128());  // 8xi16 -> low 8xi8
    const unsigned bits = static_cast<unsigned>(_mm_movemask_epi8(packed));
    return (N >= 32) ? bits : (bits & ((1u << N) - 1u));  // keep only the N valid lane bits
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
  // ptrdiff_t-domain subtract -- avoids a movslq before 64-bit consumers (see avx2.h FindLastTrue).
  return b ? std::ptrdiff_t{31} - std::countl_zero(b) : std::ptrdiff_t{-1};
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
// MaskFromNormalizedVec: caller GUARANTEES v is already canonical all-ones/all-zero (e.g. PromoteTo/
// OrderedDemote2To of a mask vector). Skips MaskFromVec's redundant srai. Mask128::raw IS the vec raw.
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask128<T, N> MaskFromNormalizedVec(Vec128<T, N> v) noexcept {
  Mask128<T, N> m;
  m.raw = v.raw;
  return m;
}

// mask <-> bits (reuse the mask_bits movemask helper).
template <class D>
HYLE_FORCE_INLINE std::uint64_t BitsFromMask(D, Mask128<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m) noexcept {
  return detail::mask_bits(m);
}
template <class D> requires HYLE_SIMD_W(D, 0, 16)
HYLE_FORCE_INLINE Mask128<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> LoadMaskBits(D, std::uint64_t bits) noexcept {
  // Rebuild: lane i true iff bit i set. FirstN-style via an index compare against the bitset.
  // _mm_load_si128 reads a FULL 16 B regardless of logical lane count, so the scratch MUST be a
  // whole register wide (16 B): a partial tag (e.g. Simd<i16,4,-1> from Rebind, kN*sizeof=8 B) would
  // otherwise be a stack OOB read -- same trap as TruncateTo. Extra lanes past kN stay zero (unread).
  using T = TFromD<D>;
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  alignas(16) T tmp[16 / sizeof(T)] = {};
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
HYLE_FORCE_INLINE std::size_t StoreMaskBits(D, Mask128<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m,
                                            std::uint8_t* p) noexcept {
  const std::uint64_t bits = detail::mask_bits(m);
  const std::size_t nbytes = (HYLE_SIMD_MAX_LANES_D(D) + 7) / 8;
  for (std::size_t b = 0; b < nbytes; ++b) p[b] = static_cast<std::uint8_t>(bits >> (b * 8));
  return nbytes;
}

// prefix-scan masks (leading run relative to first true). For 32-bit lanes (the common i32/f32
// case, op_parity -n15: sse4 bitfield round-trip was ~2.2-2.57x behind Highway) these are computed
// PURELY IN THE VECTOR DOMAIN via Highway's x86_128 bit tricks -- no mask_bits/LoadMaskBits scalar
// round-trip. A mask lane is all-ones/all-zero; viewing the 128-bit vector as two i64 blocks:
//   * "x | -x" fills from the lowest set bit up to the MSB within each i64 block,
//   * the low block's sign bit is propagated to the high block via shuffle_ps + arithmetic-shift
//     broadcast (so a first-true in the low 64 bits forces the whole high 64 bits on).
// Non-32-bit lanes keep the scalar bitfield fallback (not bench losers; would need per-width work).
namespace detail {
// Extract a mask's 128-bit lane pattern as __m128i regardless of T (float mask raw is __m128).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE __m128i mask_raw_i(Mask128<T, N> m) noexcept {
  if constexpr (std::is_same_v<T, float>) return _mm_castps_si128(m.raw);
  else if constexpr (std::is_same_v<T, double>) return _mm_castpd_si128(m.raw);
  else return m.raw;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask128<T, N> mask_from_i(__m128i v) noexcept {
  Mask128<T, N> m;
  if constexpr (std::is_same_v<T, float>) m.raw = _mm_castsi128_ps(v);
  else if constexpr (std::is_same_v<T, double>) m.raw = _mm_castsi128_pd(v);
  else m.raw = v;
  return m;
}
// SetAtOrAfterFirst core on the raw lane vector (32-bit lanes, full 128-bit / N==4).
HYLE_FORCE_INLINE __m128i set_at_or_after_first_i32(__m128i vm) noexcept {
  const __m128i negx = _mm_sub_epi64(_mm_setzero_si128(), vm);  // -x per i64 block
  const __m128i or1 = _mm_or_si128(vm, negx);                   // lowest-set..MSB filled per block
  // Copy low block's sign (its high i32's MSB) into the high block: shuffle_ps(1,1,0,0) puts the
  // low block's high dword into both hi-block dwords, then srai broadcasts that sign across.
  const __m128 sh = _mm_shuffle_ps(_mm_setzero_ps(), _mm_castsi128_ps(or1), _MM_SHUFFLE(1, 1, 0, 0));
  const __m128i vm2 = _mm_srai_epi32(_mm_castps_si128(sh), 31);
  return _mm_or_si128(or1, vm2);
}
// SetOnlyFirst core (32-bit, full 128): isolate the lowest set bit across the whole vector.
HYLE_FORCE_INLINE __m128i set_only_first_i32(__m128i vm) noexcept {
  // only_first = -(x & -x) per i64 block gives an all-ones block iff that block held the lowest bit
  // (turns the isolated bit into a full-width lane run); then mask off the high block if the low
  // block was non-zero (vmask2: high block kept only when low block == 0).
  const __m128i negx = _mm_sub_epi64(_mm_setzero_si128(), vm);
  const __m128i lowest = _mm_and_si128(vm, negx);              // x & -x per i64 (isolate low set bit)
  const __m128i only = _mm_sub_epi32(_mm_setzero_si128(), lowest);  // Neg in i32 domain: only the
                                                                    // i32 lane owning that bit -> all-ones
  // vmask2: for each i64, was the *lower* i64 block zero? InterleaveLower(zero, vm) puts vm's low
  // block into the high 64 bits; ==zero tests it. Highway builds it via cmpeq; here directly.
  const __m128i inter = _mm_unpacklo_epi64(_mm_setzero_si128(), vm);  // [0 : low-block-of-vm]
  const __m128i is_zero = _mm_cmpeq_epi64(inter, _mm_setzero_si128());
  return _mm_and_si128(only, is_zero);
}
}  // namespace detail
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask128<T, N> SetAtOrAfterFirst(Mask128<T, N> m) noexcept {
  if constexpr (sizeof(T) == 4 && N == 4) {
    return detail::mask_from_i<T, N>(detail::set_at_or_after_first_i32(detail::mask_raw_i(m)));
  } else {
    const unsigned b = detail::mask_bits(m);
    const unsigned full = (N >= 32) ? 0xFFFFFFFFu : ((1u << N) - 1u);
    const unsigned before = b ? ((1u << std::countr_zero(b)) - 1u) : full;
    return LoadMaskBits(Simd<T, N, 0>(), full & ~before);
  }
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask128<T, N> SetBeforeFirst(Mask128<T, N> m) noexcept {
  if constexpr (sizeof(T) == 4 && N == 4) {
    const __m128i aoa = detail::set_at_or_after_first_i32(detail::mask_raw_i(m));
    return detail::mask_from_i<T, N>(_mm_xor_si128(aoa, _mm_set1_epi32(-1)));  // Not(AtOrAfter)
  } else {
    const unsigned b = detail::mask_bits(m);
    const unsigned before = b ? ((1u << std::countr_zero(b)) - 1u) : ((N >= 32 ? 0xFFFFFFFFu : (1u << N) - 1u));
    return LoadMaskBits(Simd<T, N, 0>(), before);
  }
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask128<T, N> SetOnlyFirst(Mask128<T, N> m) noexcept {
  if constexpr (sizeof(T) == 4 && N == 4) {
    return detail::mask_from_i<T, N>(detail::set_only_first_i32(detail::mask_raw_i(m)));
  } else {
    const unsigned b = detail::mask_bits(m);
    return LoadMaskBits(Simd<T, N, 0>(), b ? (1u << std::countr_zero(b)) : 0u);
  }
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Mask128<T, N> SetAtOrBeforeFirst(Mask128<T, N> m) noexcept {
  if constexpr (sizeof(T) == 4 && N == 4) {
    // AtOrBeforeFirst(m) == BeforeFirst(m shifted up one lane): shifting the mask toward higher
    // lanes moves its first-true to lane+1, so "strictly before" that == "at-or-before" the
    // original first-true. ShiftLeftLanes<1> for i32 = byte shift by 4; then Not(AtOrAfterFirst).
    const __m128i shifted = _mm_slli_si128(detail::mask_raw_i(m), 4);
    const __m128i aoa = detail::set_at_or_after_first_i32(shifted);
    return detail::mask_from_i<T, N>(_mm_xor_si128(aoa, _mm_set1_epi32(-1)));
  } else {
    const unsigned b = detail::mask_bits(m);
    const unsigned upto = b ? ((1u << (std::countr_zero(b) + 1)) - 1u) : ((N >= 32 ? 0xFFFFFFFFu : (1u << N) - 1u));
    return LoadMaskBits(Simd<T, N, 0>(), upto);
  }
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
// simd-17 batch5: SumsOf8AbsDiff(a,b) = native _mm_sad_epu8 (SAD per 8-byte group -> u64).
template <std::size_t N>
HYLE_FORCE_INLINE Vec128<std::uint64_t, N / 8> SumsOf8AbsDiff(Vec128<std::uint8_t, N> a,
                                                              Vec128<std::uint8_t, N> b) noexcept {
  static_assert(N == 16, "sse4 SumsOf8AbsDiff: full 16xu8 -> 2xu64");
  Vec128<std::uint64_t, N / 8> r;
  r.raw = _mm_sad_epu8(a.raw, b.raw);
  return r;
}
template <std::size_t N>
HYLE_FORCE_INLINE Vec128<std::uint64_t, N / 8> SumsOf8AbsDiff(Vec128<std::uint8_t, N> v) noexcept {
  return SumsOf8(v);
}
// ---- simd-18 批F: SAD video quad (sse4 native mpsadbw for F1) -----------------------
template <int kAOffset, int kBOffset, std::size_t N>
HYLE_FORCE_INLINE Vec128<std::uint16_t, N / 2> SumsOfAdjQuadAbsDiff(Vec128<std::uint8_t, N> a,
                                                                     Vec128<std::uint8_t, N> b) noexcept {
  static_assert(N == 16, "sse4 SumsOfAdjQuadAbsDiff: full 16xu8");
  static_assert(kAOffset == 0 || kAOffset == 1, "kAOffset in {0,1}");
  static_assert(kBOffset >= 0 && kBOffset <= 3, "kBOffset in {0..3}");
  constexpr int imm = (kAOffset << 2) | kBOffset;
  Vec128<std::uint16_t, N / 2> r;
  r.raw = _mm_mpsadbw_epu8(a.raw, b.raw, imm);
  return r;
}
template <int i3, int i2, int i1, int i0, std::size_t N>
HYLE_FORCE_INLINE Vec128<std::uint16_t, N / 2> SumsOfShuffledQuadAbsDiff(Vec128<std::uint8_t, N> a,
                                                                          Vec128<std::uint8_t, N> b) noexcept {
  static_assert(N == 16, "sse4 SumsOfShuffledQuadAbsDiff: full 16xu8");
  static_assert(0 <= i0 && i0 <= 3 && 0 <= i1 && i1 <= 3 &&
                0 <= i2 && i2 <= 3 && 0 <= i3 && i3 <= 3,
                "SumsOfShuffledQuadAbsDiff indices are in [0,3]");

  // SSE4 has no DB-SAD instruction, but the same register-only recipe used by Highway and the
  // AVX2 backend is available. This replaces the old store/scalar/reload correctness oracle.
  const __m128i shuf = _mm_shuffle_epi32(a.raw, _MM_SHUFFLE(i3, i2, i1, i0));
  const __m128i a0123_2345 = _mm_blend_epi16(shuf, _mm_slli_si128(shuf, 2), 0xCC);
  const __m128i a1234_3456 =
      _mm_blend_epi16(_mm_srli_si128(shuf, 1), _mm_slli_si128(shuf, 1), 0xCC);
  const auto absdiff = [](__m128i x, __m128i y) noexcept {
    return _mm_sub_epi8(_mm_max_epu8(x, y), _mm_min_epu8(x, y));
  };
  const __m128i ones8 = _mm_set1_epi8(1);
  const __m128i ones16 = _mm_set1_epi16(1);
  const auto sums4 = [&](const __m128i diff) noexcept {
    return _mm_madd_epi16(_mm_maddubs_epi16(diff, ones8), ones16);
  };
  const __m128i even = sums4(absdiff(a0123_2345, b.raw));
  const __m128i odd = _mm_slli_epi32(sums4(absdiff(a1234_3456, b.raw)), 16);
  Vec128<std::uint16_t, N / 2> r;
  r.raw = _mm_blend_epi16(even, odd, 0xAA);
  return r;
}

// simd-17 batch5: BitShuffle / MultiRotateRight -- portable store+reload (matches scalar oracle
// + Highway's generic contract). Native VPSHUFBITQMB / VPMULTISHIFTQB live on the AVX3_DL avx512
// path only; the sse4 backend uses the correctness-first gather (a profiling upgrade could add
// GFNI/BITALG native here, but those need ISA flags this variant does not carry).
template <typename T, std::size_t N, typename TI, std::size_t NI>
HYLE_FORCE_INLINE Vec128<T, N> BitShuffle(Vec128<T, N> v, Vec128<TI, NI> idx) noexcept {
  static_assert(sizeof(T) == 8, "BitShuffle data lanes are 64-bit");
  static_assert(sizeof(TI) == 1 && NI == N * 8, "BitShuffle index is u8, 8 per data lane");
  alignas(16) std::uint64_t sv[N];
  alignas(16) std::uint8_t si[NI];
  Store(v, Simd<T, N, 0>(), reinterpret_cast<T*>(sv));
  Store(idx, Simd<TI, NI, 0>(), reinterpret_cast<TI*>(si));
  alignas(16) std::uint64_t out[N];
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
HYLE_FORCE_INLINE Vec128<T, N> MultiRotateRight(Vec128<T, N> v, Vec128<TI, NI> idx) noexcept {
  static_assert(sizeof(T) == 8, "MultiRotateRight data lanes are 64-bit");
  static_assert(sizeof(TI) == 1 && NI == N * 8, "MultiRotateRight index is u8, 8 per data lane");
  alignas(16) std::uint64_t sv[N];
  alignas(16) std::uint8_t si[NI];
  Store(v, Simd<T, N, 0>(), reinterpret_cast<T*>(sv));
  Store(idx, Simd<TI, NI, 0>(), reinterpret_cast<TI*>(si));
  alignas(16) std::uint64_t out[N];
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
// scratch sized to the FULL 16 B register (not the logical lane count): _mm_store/load_si128
// move 16 B, so a partial-width src (e.g. Vec128<u8,8>) would be a stack OOB with src[N] --
// same class as the TruncateTo fix above.
// SumsOf2/4(u8): native pairwise widen-sum (Highway generic form) -- Add(PromoteEvenTo,
// PromoteOddTo), SumsOf4 = the same applied twice. Was store+reload+scalar loop (op_parity: ~9x
// slower than Highway on avx2, same shape here). No overflow: 255+255 fits u16, quad-sum fits u32.
template <std::size_t N>
HYLE_FORCE_INLINE Vec128<std::uint16_t, N / 2> SumsOf2(Vec128<std::uint8_t, N> v) noexcept {
  static_assert(N % 2 == 0, "sse4 SumsOf2 needs an even lane count");
  const Simd<std::uint16_t, N / 2, 0> dw;
  return Add(PromoteEvenTo(dw, v), PromoteOddTo(dw, v));
}
template <std::size_t N>
HYLE_FORCE_INLINE Vec128<std::uint32_t, N / 4> SumsOf4(Vec128<std::uint8_t, N> v) noexcept {
  static_assert(N % 4 == 0, "sse4 SumsOf4 needs a lane count that is a multiple of 4");
  const auto s2 = SumsOf2(v);
  const Simd<std::uint32_t, N / 4, 0> dq;
  return Add(PromoteEvenTo(dq, s2), PromoteOddTo(dq, s2));
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
// RedOp defined in x86_detail.h (shared across layered x86 backends).
template <typename T, std::size_t N, RedOp op>
HYLE_FORCE_INLINE T reduce_scalar(Vec128<T, N> v) noexcept {
  // Full 16 B scratch: _mm_store_si128 always writes a whole register.
  alignas(16) T tmp[16 / sizeof(T)];
  StoreU(v, Simd<T, N, 0>(), tmp);
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
  if constexpr (N < 16 / sizeof(T))
    return detail::reduce_scalar<T, N, detail::RedOp::kMin>(v);
  else if constexpr (detail::kI32<T> || std::is_floating_point_v<T>)
    return detail::reduce128(v, [](Vec128<T, N> x, Vec128<T, N> y) { return Min(x, y); });
  else return detail::reduce_scalar<T, N, detail::RedOp::kMin>(v);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE T ReduceMax(Simd<T, N, 0>, Vec128<T, N> v) noexcept {
  if constexpr (N < 16 / sizeof(T))
    return detail::reduce_scalar<T, N, detail::RedOp::kMax>(v);
  else if constexpr (detail::kI32<T> || std::is_floating_point_v<T>)
    return detail::reduce128(v, [](Vec128<T, N> x, Vec128<T, N> y) { return Max(x, y); });
  else return detail::reduce_scalar<T, N, detail::RedOp::kMax>(v);
}

// Reduction family completion: ReduceMul (product), ReduceAnd/Or (bitwise, integer lanes).
// Fold via a stack array (no native horizontal product/and/or; horizontal reduce is cold).
namespace detail {
// RedOp2 defined in x86_detail.h (shared across layered x86 backends).
template <typename T, std::size_t N, RedOp2 op>
HYLE_FORCE_INLINE T reduce_scalar2(Vec128<T, N> v) noexcept {
  // Full 16 B scratch: Store always writes a whole register (partial-tag safe).
  alignas(16) T tmp[16 / sizeof(T)];
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
  // Full 16 B scratch: Store always writes a whole register (partial-tag safe).
  alignas(16) T tmp[16 / sizeof(T)];
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
  // NATIVE (op_parity: the avx2 O(bits) scalar loop was 9x slower): (1) reverse bits within each
  // byte via two pshufb nibble-LUTs, (2) reverse the byte order within each lane via a per-width
  // pshufb index. per-nibble bit-reversal LUT: reverse(0001)=1000=8, ... (4-bit mirror).
  const __m128i lut = _mm_setr_epi8(0x0,0x8,0x4,0xC,0x2,0xA,0x6,0xE,0x1,0x9,0x5,0xD,0x3,0xB,0x7,0xF);
  const __m128i lo_mask = _mm_set1_epi8(0x0F);
  const __m128i lo = _mm_and_si128(v.raw, lo_mask);
  const __m128i hi = _mm_and_si128(_mm_srli_epi16(v.raw, 4), lo_mask);
  __m128i rb = _mm_or_si128(_mm_slli_epi16(_mm_shuffle_epi8(lut, lo), 4), _mm_shuffle_epi8(lut, hi));
  Vec128<T, N> r;
  if constexpr (sizeof(T) == 1) { r.raw = rb; return r; }
  __m128i idx;
  if constexpr (sizeof(T) == 2)
    idx = _mm_setr_epi8(1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14);
  else if constexpr (sizeof(T) == 4)
    idx = _mm_setr_epi8(3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12);
  else
    idx = _mm_setr_epi8(7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8);
  r.raw = _mm_shuffle_epi8(rb, idx);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> ReverseLaneBytes(Vec128<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "ReverseLaneBytes is integer-only");
  alignas(16) T s[16 / sizeof(T)], d[16 / sizeof(T)];
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
  alignas(16) T sa[16 / sizeof(T)], sb[16 / sizeof(T)], d[16 / sizeof(T)];
  Store(a, Simd<T, N, 0>(), sa); Store(b, Simd<T, N, 0>(), sb);
  for (std::size_t i = 0; i < N / 2; ++i) { d[2 * i] = sa[2 * i]; d[2 * i + 1] = sb[2 * i]; }
  return Load(Simd<T, N, 0>(), d);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> InterleaveOdd(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  alignas(16) T sa[16 / sizeof(T)], sb[16 / sizeof(T)], d[16 / sizeof(T)];
  Store(a, Simd<T, N, 0>(), sa); Store(b, Simd<T, N, 0>(), sb);
  for (std::size_t i = 0; i < N / 2; ++i) { d[2 * i] = sa[2 * i + 1]; d[2 * i + 1] = sb[2 * i + 1]; }
  return Load(Simd<T, N, 0>(), d);
}
template <class D>
HYLE_FORCE_INLINE detail::V128<D> Slide1Up(D d, detail::V128<D> v) noexcept {
  // Single 128-bit block: shift up one lane (dst[0]=0) = palignr(v, Zero, 16-sizeof(T)). Pure
  // vector, replaces the old store + scalar copy + reload.
  using T = TFromD<D>;
  return CombineShiftRightBytes<16 - static_cast<int>(sizeof(T))>(v, Zero(d));
}
template <class D>
HYLE_FORCE_INLINE detail::V128<D> Slide1Down(D d, detail::V128<D> v) noexcept {
  // Shift down one lane (dst[N-1]=0) = palignr(Zero, v, sizeof(T)). Pure vector.
  using T = TFromD<D>;
  return CombineShiftRightBytes<static_cast<int>(sizeof(T))>(Zero(d), v);
}
template <class D>
HYLE_FORCE_INLINE detail::V128<D> SlideUpLanes(D d, detail::V128<D> v, std::size_t amt) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(16) T s[16 / sizeof(T)]; alignas(16) T dst[16 / sizeof(T)] = {};
  Store(v, d, s);
  for (std::size_t i = amt; i < N; ++i) dst[i] = s[i - amt];
  return Load(d, dst);
}
template <class D>
HYLE_FORCE_INLINE detail::V128<D> SlideDownLanes(D d, detail::V128<D> v, std::size_t amt) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t N = HYLE_SIMD_MAX_LANES_D(D);
  alignas(16) T s[16 / sizeof(T)]; alignas(16) T dst[16 / sizeof(T)] = {};
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
// Load/Store always move a FULL 16 B register (_mm_load/store_si128), so every scratch that
// feeds them MUST be whole-register wide -- same trap as TruncateTo / LoadMaskBits. Partial
// tags (e.g. CappedTag<u32,1> from vqsort Sort2To2) have kN*sizeof(T) < 16; sizing tmp[kN]
// is a stack OOB (GCC SIGSEGV on Sort.U32; MSVC often silent). Extra lanes past kN stay 0.
template <class D> requires HYLE_SIMD_W(D, 0, 16)
HYLE_FORCE_INLINE Mask128<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> FirstN(D d, std::size_t n) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  T ones;
  auto* ob = reinterpret_cast<unsigned char*>(&ones);
  for (std::size_t k = 0; k < sizeof(T); ++k) ob[k] = 0xFFu;
  alignas(16) T tmp[16 / sizeof(T)] = {};
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
  alignas(16) T tmp[16 / sizeof(T)];
  Store(v, d, tmp);
  const unsigned bits = detail::mask_bits(m);
  for (std::size_t i = 0; i < kN; ++i)
    if ((bits >> i) & 1u) p[i] = tmp[i];
}
template <class D> requires HYLE_SIMD_W(D, 0, 16)
HYLE_FORCE_INLINE detail::V128<D> LoadN(D d, const TFromD<D>* p, std::size_t n) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  const std::size_t k = n < kN ? n : kN;
  alignas(16) T tmp[16 / sizeof(T)] = {};
  for (std::size_t i = 0; i < k; ++i) tmp[i] = p[i];
  return Load(d, tmp);
}
template <class D>
HYLE_FORCE_INLINE void StoreN(detail::V128<D> v, D d, TFromD<D>* p, std::size_t n) noexcept {
  using T = TFromD<D>;
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  const std::size_t k = n < kN ? n : kN;
  alignas(16) T tmp[16 / sizeof(T)];
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
  // Full 16 B scratch: Store/Load always touch a whole register (partial-tag safe).
  alignas(16) IdxT ix[16 / sizeof(IdxT)];
  alignas(16) TFromD<D> dst[16 / sizeof(TFromD<D>)] = {};
  Store(idx, Simd<IdxT, kN, 0>(), ix);
  for (std::size_t i = 0; i < kN; ++i) dst[i] = base[static_cast<std::ptrdiff_t>(ix[i])];
  return Load(d, dst);
}
template <class D, typename IdxT>
HYLE_FORCE_INLINE void ScatterIndex(detail::V128<D> v, D d, TFromD<D>* base,
                                    Vec128<IdxT, HYLE_SIMD_MAX_LANES_D(D)> idx) noexcept {
  constexpr std::size_t kN = HYLE_SIMD_MAX_LANES_D(D);
  // Full 16 B scratch: Store always writes a whole register (partial-tag safe).
  alignas(16) IdxT ix[16 / sizeof(IdxT)];
  alignas(16) TFromD<D> src[16 / sizeof(TFromD<D>)];
  Store(idx, Simd<IdxT, kN, 0>(), ix); Store(v, d, src);
  for (std::size_t i = 0; i < kN; ++i) base[static_cast<std::ptrdiff_t>(ix[i])] = src[i];
}
// Compress byte-index table for 4x i32 (Highway's method): only 16 mask combos, each a 16-byte
// pshufb index that gathers active 32-bit lanes to the front. 16*16 = 256 B. constexpr-built.
namespace detail {
// _mm_load_si128 below is part of the hot single-shuffle path. Make the table's
// alignment a language-level guarantee instead of relying on how each compiler
// happens to place an inline constexpr byte array (clang-cl only gave it byte
// alignment and correctly faulted on the aligned load).
struct alignas(16) CompressIdx32x4 {
  std::uint8_t idx[16 * 16] = {};
  constexpr CompressIdx32x4() {
    for (unsigned bits = 0; bits < 16; ++bits) {
      unsigned k = 0;
      unsigned pos = 0;
      for (unsigned lane = 0; lane < 4; ++lane)
        if ((bits >> lane) & 1u) { for (unsigned b = 0; b < 4; ++b) idx[bits * 16 + (pos++)] = static_cast<std::uint8_t>(lane * 4 + b); ++k; }
      // fill the inactive tail with identity bytes so pshufb yields defined (unspecified-tail) data.
      for (unsigned lane = 0; lane < 4; ++lane)
        if (!((bits >> lane) & 1u)) for (unsigned b = 0; b < 4; ++b) idx[bits * 16 + (pos++)] = static_cast<std::uint8_t>(lane * 4 + b);
      (void)k;
    }
  }
};
inline constexpr CompressIdx32x4 kCompressIdx32x4Tbl{};
}  // namespace detail
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> Compress(Vec128<T, N> v, Mask128<T, N> m) noexcept {
  // NATIVE for 32-bit lanes (4 lanes, 16 mask combos): a 16-byte pshufb index (loaded from a 256 B
  // table) packs the active lanes to the front in ONE shuffle (Highway's method; op_parity: the
  // store+reload was ~5x slower). CONTRACT: only the first CountTrue(m) lanes are defined; the tail
  // is UNSPECIFIED (matches Highway/xsimd). Other widths keep store+reload.
  if constexpr (sizeof(T) == 4) {
    unsigned bits;
    if constexpr (std::is_same_v<T, float>) bits = static_cast<unsigned>(_mm_movemask_ps(m.raw));
    else bits = static_cast<unsigned>(_mm_movemask_ps(_mm_castsi128_ps(m.raw)));
    const __m128i byte_idx = _mm_load_si128(reinterpret_cast<const __m128i*>(detail::kCompressIdx32x4Tbl.idx + bits * 16));
    Vec128<T, N> r;
    if constexpr (std::is_same_v<T, float>)
      r.raw = _mm_castsi128_ps(_mm_shuffle_epi8(_mm_castps_si128(v.raw), byte_idx));
    else r.raw = _mm_shuffle_epi8(v.raw, byte_idx);
    return r;
  } else {
    alignas(16) T src[16 / sizeof(T)], dst[16 / sizeof(T)] = {};
    Store(v, Simd<T, N, 0>(), src);
    const unsigned bits = detail::mask_bits(m);
    std::size_t k = 0;
    for (std::size_t i = 0; i < N; ++i) if ((bits >> i) & 1u) dst[k++] = src[i];
    return Load(Simd<T, N, 0>(), dst);
  }
}
template <class D>
HYLE_FORCE_INLINE std::size_t CompressStore(detail::V128<D> v,
                                            Mask128<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D)> m,
                                            D d, TFromD<D>* p) noexcept {
  // CONTRACT (aligned to Highway/xsimd): first CountTrue lanes = active elements in order, tail
  // UNSPECIFIED. Compress is native pshufb (32-bit), then one full StoreU + return CountTrue. The
  // old exact tail-untouched contract forced a scalar pack (no consumer) -- dropped, same as avx2.
  StoreU(Compress(v, m), d, p);
  return CountTrue(d, m);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> Expand(Vec128<T, N> v, Mask128<T, N> m) noexcept {
  alignas(16) T src[16 / sizeof(T)], dst[16 / sizeof(T)];
  Store(v, Simd<T, N, 0>(), src);
  const unsigned bits = detail::mask_bits(m);
  std::size_t k = 0;
  for (std::size_t i = 0; i < N; ++i) dst[i] = ((bits >> i) & 1u) ? src[k++] : T{0};
  return Load(Simd<T, N, 0>(), dst);
}

// ---- bit counts / sign helpers / misc -----------------------------------------------
// PopulationCount/LeadingZeroCount/TrailingZeroCount: NATIVE via the pshufb nibble-LUT (op_parity:
// the old std::count[lr]_zero store+reload was 5-8x slower on avx2). Not/Clamp/Rotate/CopySign/...
// compose from existing ops -- same on every backend. Scalar is the reference oracle.
namespace detail {
template <typename T, std::size_t N, int kind>
HYLE_FORCE_INLINE Vec128<T, N> bitcount_via_mem(Vec128<T, N> v) noexcept {
  using U = std::make_unsigned_t<T>;
  alignas(16) T s[16 / sizeof(T)], d[16 / sizeof(T)];
  Store(v, Simd<T, N, 0>(), s);
  for (std::size_t i = 0; i < N; ++i) {
    const U u = static_cast<U>(s[i]);
    if constexpr (kind == 0) d[i] = static_cast<T>(std::popcount(u));
    else if constexpr (kind == 1) d[i] = static_cast<T>(std::countl_zero(u));
    else d[i] = static_cast<T>(std::countr_zero(u));
  }
  return Load(Simd<T, N, 0>(), d);
}
// 128-bit byte-wise popcount via pshufb nibble-LUT (Muła's method): shuffle a 16-entry nibble->count
// table twice (low + high nibble) and add; returns per-BYTE popcount (0..8).
HYLE_FORCE_INLINE __m128i popcnt_bytes_sse4(__m128i v) noexcept {
  const __m128i lut = _mm_setr_epi8(0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
  const __m128i lo_mask = _mm_set1_epi8(0x0F);
  const __m128i lo = _mm_and_si128(v, lo_mask);
  const __m128i hi = _mm_and_si128(_mm_srli_epi16(v, 4), lo_mask);
  return _mm_add_epi8(_mm_shuffle_epi8(lut, lo), _mm_shuffle_epi8(lut, hi));
}
template <typename T> HYLE_FORCE_INLINE __m128i srli_w128(__m128i x, int s) noexcept {
  if constexpr (sizeof(T) == 2) return _mm_srli_epi16(x, s);
  else if constexpr (sizeof(T) == 4) return _mm_srli_epi32(x, s);
  else return _mm_srli_epi64(x, s);
}
// float-exponent trick for 32-bit lanes (Highway generic_ops), same as avx2/avx512: the biased
// exponent of the f32 nearest to a u32 gives its bit-length. Returns the biased exponent [0,158].
HYLE_FORCE_INLINE __m128i u32_to_f32_biased_exp128(__m128i v) noexcept {
  const __m128i norm = _mm_andnot_si128(_mm_srli_epi32(v, 24), v);  // round-down guard for v>=2^24
  const __m128 f = _mm_cvtepi32_ps(norm);
  const __m128i be = _mm_srli_epi32(_mm_castps_si128(f), 23);
  // bit-31-set inputs read negative through the signed convert -> sign bit leaks past >>23 -> bogus
  // exp. Highway clamps with Min(.,158) (biased exp of 2^31), pinning LZCnt=0 / TZCnt(2^31)=31.
  return _mm_min_epu32(be, _mm_set1_epi32(158));
}
}  // namespace detail
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> PopulationCount(Vec128<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "PopulationCount is integer-lane only");
  Vec128<T, N> r;
  const __m128i bytes = detail::popcnt_bytes_sse4(v.raw);
  if constexpr (sizeof(T) == 1) r.raw = bytes;
  else if constexpr (sizeof(T) == 8) r.raw = _mm_sad_epu8(bytes, _mm_setzero_si128());
  else if constexpr (sizeof(T) == 4)
    r.raw = _mm_madd_epi16(_mm_maddubs_epi16(bytes, _mm_set1_epi8(1)), _mm_set1_epi16(1));
  else r.raw = _mm_maddubs_epi16(bytes, _mm_set1_epi8(1));  // 16-bit
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> LeadingZeroCount(Vec128<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "LeadingZeroCount is integer-lane only");
  if constexpr (sizeof(T) == 4) {
    // LZCnt(x) = min( 158 - biased_exp(x), 32 ). biased_exp(0)=0 -> 158, clamped to 32.
    const __m128i be = detail::u32_to_f32_biased_exp128(v.raw);
    const __m128i lz = _mm_sub_epi32(_mm_set1_epi32(32 + 126), be);
    Vec128<T, N> r; r.raw = _mm_min_epu32(lz, _mm_set1_epi32(32));  // SSE4.1
    return r;
  } else if constexpr (sizeof(T) == 1) {
    return detail::bitcount_via_mem<T, N, 1>(v);  // no _mm_srli_epi8
  } else {  // 16/64-bit: smear the top set bit down, then bits - popcount
    __m128i x = v.raw;
    x = _mm_or_si128(x, detail::srli_w128<T>(x, 1));
    x = _mm_or_si128(x, detail::srli_w128<T>(x, 2));
    x = _mm_or_si128(x, detail::srli_w128<T>(x, 4));
    x = _mm_or_si128(x, detail::srli_w128<T>(x, 8));
    if constexpr (sizeof(T) == 8) { x = _mm_or_si128(x, detail::srli_w128<T>(x, 16)); x = _mm_or_si128(x, detail::srli_w128<T>(x, 32)); }
    Vec128<T, N> sm; sm.raw = x;
    return Sub(Set(Simd<T, N, 0>(), static_cast<T>(sizeof(T) * 8)), PopulationCount(sm));
  }
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> TrailingZeroCount(Vec128<T, N> v) noexcept {
  static_assert(std::is_integral_v<T>, "TrailingZeroCount is integer-lane only");
  if constexpr (sizeof(T) == 4) {
    // TZCnt(x) = min( biased_exp(x & -x) - 127, 32 ). unsigned-min saturates x==0 to 32.
    const __m128i neg = _mm_sub_epi32(_mm_setzero_si128(), v.raw);
    const __m128i lowest = _mm_and_si128(v.raw, neg);
    const __m128i be = detail::u32_to_f32_biased_exp128(lowest);
    const __m128i tz = _mm_sub_epi32(be, _mm_set1_epi32(127));
    Vec128<T, N> r; r.raw = _mm_min_epu32(tz, _mm_set1_epi32(32));  // SSE4.1
    return r;
  } else if constexpr (sizeof(T) == 1) {
    return detail::bitcount_via_mem<T, N, 2>(v);
  } else {  // 16/64-bit: PopCount((x & -x) - 1)
    Vec128<T, N> lowest; lowest.raw = _mm_and_si128(v.raw, Neg(v).raw);
    return PopulationCount(Sub(lowest, Set(Simd<T, N, 0>(), static_cast<T>(1))));
  }
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
    alignas(16) T s[16 / sizeof(T)], d[16 / sizeof(T)];
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
  Vec128<T, N> r;
  if constexpr (sizeof(T) == 1)
    r.raw = _mm_cmpgt_epi8(_mm_setzero_si128(), v.raw);
  else if constexpr (sizeof(T) == 2)
    r.raw = _mm_srai_epi16(v.raw, 15);
  else if constexpr (sizeof(T) == 4)
    r.raw = _mm_srai_epi32(v.raw, 31);
  else
    r.raw = _mm_cmpgt_epi64(_mm_setzero_si128(), v.raw);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> IfNegativeThenElse(Vec128<T, N> v, Vec128<T, N> yes, Vec128<T, N> no) noexcept {
  return IfThenElse(Lt(v, Zero(Simd<T, N, 0>())), yes, no);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> ZeroIfNegative(Vec128<T, N> v) noexcept {
  return IfThenElse(Lt(v, Zero(Simd<T, N, 0>())), Zero(Simd<T, N, 0>()), v);
}


// ---- simd-17 batch2: runtime *Same + Pairwise + Stream + LoadDup128 + exp domain ----
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> ShiftLeftSame(Vec128<T, N> v, int bits) noexcept {
  static_assert(std::is_integral_v<T>, "ShiftLeftSame is integer-lane only");
  Vec128<T, N> r;
  const __m128i c = _mm_cvtsi32_si128(bits);
  if constexpr (detail::kI16<T>) r.raw = _mm_sll_epi16(v.raw, c);
  else if constexpr (detail::kI32<T>) r.raw = _mm_sll_epi32(v.raw, c);
  else if constexpr (detail::kI64<T>) r.raw = _mm_sll_epi64(v.raw, c);
  else {
    const __m128i sh = _mm_sll_epi16(v.raw, c);
    const unsigned char m = static_cast<unsigned char>(0xFFu << (bits & 7));
    r.raw = (bits >= 8) ? _mm_setzero_si128() : _mm_and_si128(sh, _mm_set1_epi8(static_cast<char>(m)));
  }
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> ShiftRightSame(Vec128<T, N> v, int bits) noexcept {
  static_assert(std::is_integral_v<T>, "ShiftRightSame is integer-lane only");
  Vec128<T, N> r;
  const __m128i c = _mm_cvtsi32_si128(bits);
  if constexpr (detail::kI16<T>) {
    r.raw = detail::kSigned<T> ? _mm_sra_epi16(v.raw, c) : _mm_srl_epi16(v.raw, c);
  } else if constexpr (detail::kI32<T>) {
    r.raw = detail::kSigned<T> ? _mm_sra_epi32(v.raw, c) : _mm_srl_epi32(v.raw, c);
  } else if constexpr (detail::kI64<T>) {
    if constexpr (detail::kSigned<T>) {
      alignas(16) T s[16 / sizeof(T)]; Store(v, Simd<T, N, 0>(), s);
      for (std::size_t i = 0; i < N; ++i) s[i] = static_cast<T>(s[i] >> bits);
      return Load(Simd<T, N, 0>(), s);
    } else r.raw = _mm_srl_epi64(v.raw, c);
  } else {
    alignas(16) T s[16 / sizeof(T)]; Store(v, Simd<T, N, 0>(), s);
    for (std::size_t i = 0; i < N; ++i) s[i] = static_cast<T>(s[i] >> bits);
    return Load(Simd<T, N, 0>(), s);
  }
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> RotateRightSame(Vec128<T, N> v, int bits) noexcept {
  static_assert(std::is_integral_v<T>, "RotateRightSame is integer-lane only");
  constexpr int W = static_cast<int>(sizeof(T) * 8);
  bits &= (W - 1);
  if (bits == 0) return v;
  return Or(ShiftRightSame(v, bits), ShiftLeftSame(v, W - bits));
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> RotateLeftSame(Vec128<T, N> v, int bits) noexcept {
  constexpr int W = static_cast<int>(sizeof(T) * 8);
  bits &= (W - 1);
  if (bits == 0) return v;
  return RotateRightSame(v, W - bits);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> RoundingShiftRightSame(Vec128<T, N> v, int s) noexcept {
  static_assert(std::is_integral_v<T>, "RoundingShiftRightSame is integer-lane only");
  if (s <= 0) return v;
  return ShiftRightSame(Add(v, Set(Simd<T, N, 0>(), static_cast<T>(T{1} << (s - 1)))), s);
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> RoundingShr(Vec128<T, N> v, Vec128<T, N> amounts) noexcept {
  static_assert(std::is_integral_v<T>, "RoundingShr is integer-lane only");
  if constexpr (sizeof(T) == 4) {
    // (v + 2^(s-1)) >> s for s>0, else v. Now that Shr is vectorized (v>>b via mul-high, above),
    // build the round bias 2^(s-1) with the same Pow2 helper and select s<=0 lanes back to v --
    // no more store+scalar-loop+reload (op_parity -n15: sse4 RoundingShr ~1.7x behind Highway).
    const Simd<T, N, 0> d;
    const auto spos = Gt(amounts, Set(d, T{0}));                 // s > 0 lanes
    const __m128i sm1 = _mm_sub_epi32(amounts.raw, _mm_set1_epi32(1));  // s-1 (only used where s>0)
    Vec128<T, N> biasp; biasp.raw = detail::Pow2_u32(sm1);       // 2^(s-1)
    const auto bias = IfThenElse(spos, biasp, Set(d, T{0}));     // 0 where s<=0
    const auto shifted = Shr(Add(v, bias), amounts);            // arithmetic for i32, logical for u32
    return IfThenElse(spos, shifted, v);                        // keep v where s<=0
  } else {
    alignas(16) T sv[16 / sizeof(T)], sa[16 / sizeof(T)], so[16 / sizeof(T)];
    Store(v, Simd<T, N, 0>(), sv); Store(amounts, Simd<T, N, 0>(), sa);
    for (std::size_t i = 0; i < N; ++i) {
      const int s = static_cast<int>(sa[i]);
      if (s <= 0) so[i] = sv[i];
      else so[i] = static_cast<T>((sv[i] + static_cast<T>(T{1} << (s - 1))) >> s);
    }
    return Load(Simd<T, N, 0>(), so);
  }
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> PairwiseAdd128(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  Vec128<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm_hadd_ps(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm_hadd_pd(a.raw, b.raw);
  else if constexpr (detail::kI32<T>) r.raw = _mm_hadd_epi32(a.raw, b.raw);
  else if constexpr (detail::kI16<T>) r.raw = _mm_hadd_epi16(a.raw, b.raw);
  else {
    alignas(16) T sa[16 / sizeof(T)], sb[16 / sizeof(T)], so[16 / sizeof(T)];
    Store(a, Simd<T, N, 0>(), sa); Store(b, Simd<T, N, 0>(), sb);
    for (std::size_t i = 0; i < N / 2; ++i) {
      so[i] = static_cast<T>(sa[2 * i] + sa[2 * i + 1]);
      so[N / 2 + i] = static_cast<T>(sb[2 * i] + sb[2 * i + 1]);
    }
    return Load(Simd<T, N, 0>(), so);
  }
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> PairwiseSub128(Vec128<T, N> a, Vec128<T, N> b) noexcept {
  Vec128<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm_hsub_ps(a.raw, b.raw);
  else if constexpr (std::is_same_v<T, double>) r.raw = _mm_hsub_pd(a.raw, b.raw);
  else if constexpr (detail::kI32<T>) r.raw = _mm_hsub_epi32(a.raw, b.raw);
  else if constexpr (detail::kI16<T>) r.raw = _mm_hsub_epi16(a.raw, b.raw);
  else {
    alignas(16) T sa[16 / sizeof(T)], sb[16 / sizeof(T)], so[16 / sizeof(T)];
    Store(a, Simd<T, N, 0>(), sa); Store(b, Simd<T, N, 0>(), sb);
    for (std::size_t i = 0; i < N / 2; ++i) {
      so[i] = static_cast<T>(sa[2 * i] - sa[2 * i + 1]);
      so[N / 2 + i] = static_cast<T>(sb[2 * i] - sb[2 * i + 1]);
    }
    return Load(Simd<T, N, 0>(), so);
  }
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> PairwiseAdd(Vec128<T, N> a, Vec128<T, N> b) noexcept { return PairwiseAdd128(a, b); }
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> PairwiseSub(Vec128<T, N> a, Vec128<T, N> b) noexcept { return PairwiseSub128(a, b); }
template <class D>
HYLE_FORCE_INLINE void Stream(detail::V128<D> v, D, TFromD<D>* aligned) noexcept {
  using T = TFromD<D>;
  if constexpr (std::is_same_v<T, float>) _mm_stream_ps(aligned, v.raw);
  else if constexpr (std::is_same_v<T, double>) _mm_stream_pd(aligned, v.raw);
  else _mm_stream_si128(reinterpret_cast<__m128i*>(aligned), v.raw);
}
template <class D> requires HYLE_SIMD_W(D, 0, 16)
HYLE_FORCE_INLINE detail::V128<D> LoadDup128(D d, const TFromD<D>* p) noexcept { return LoadU(d, p); }
template <typename T, std::size_t N>
HYLE_FORCE_INLINE auto GetBiasedExponent(Vec128<T, N> v) noexcept {
  static_assert(std::is_floating_point_v<T>, "GetBiasedExponent is float only");
  using U = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
  const auto bits = BitCast(Simd<U, N, 0>(), v);
  if constexpr (sizeof(T) == 4) return And(ShiftRight<23>(bits), Set(Simd<U, N, 0>(), U{0xFF}));
  else return And(ShiftRight<52>(bits), Set(Simd<U, N, 0>(), U{0x7FF}));
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> MulByFloorPow2(Vec128<T, N> v, Vec128<T, N> exp) noexcept {
  static_assert(std::is_floating_point_v<T>, "MulByFloorPow2 is float only");
  // Vectorized (see avx2.h): v * 2^floor(exp) = MulByPow2(v, FloorInt(exp)) with the IEEE NaN
  // special-cases blended (NaN exp / (0,+inf) / (inf,-inf) -> NaN). Float floor clamped into int
  // range before the (otherwise-UB for inf) cast. No store+reload.
  using IT = std::conditional_t<sizeof(T) == 4, std::int32_t, std::int64_t>;
  const Simd<T, N, 0> df;
  const auto is_nan_result =
      Or(IsNaN(exp), And(Eq(Abs(v), IfNegativeThenElseZero(exp, Inf(df))), IsInf(exp)));
  // Clamp to +/-4096 before the cast: an out-of-range float->int (cvttps2dq) returns INT_MIN even
  // for +inf, which would flip +inf exp to underflow. 4096 > 3*1023 so no reachable exp is lost.
  const T kCap = static_cast<T>(4096);
  const auto fe = Max(Min(Floor(exp), Set(df, kCap)), Set(df, -kCap));
  const auto ie = ConvertTo(Simd<IT, N, 0>(), fe);
  return IfThenElse(is_nan_result, NaN(df), MulByPow2(v, ie));
}

// ---- simd-17 batch4: half-vector family (128 -> logical half) -----------------------
// Half of Vec128 is Vec128<T,N/2> in the same __m128 raw. For a full XMM, the split is
// 64 bits; for sub-XMM fixed tags it is N*sizeof(T)/2 bytes. Every consumer below uses
// that logical split and clears padding instead of assuming a physical 64-bit half.
// Parameter order: Combine(d, hi, lo) = [lo low-half, hi low-half].
template <typename T, std::size_t NH, int kPow2, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, NH> LowerHalf(Simd<T, NH, kPow2>, Vec128<T, N> v) noexcept {
  static_assert(NH == N / 2 || (N == 1 && NH == 1), "LowerHalf: NH must be N/2");
  return Vec128<T, NH>{v.raw};  // low N/2 lanes already in place
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, (N > 1 ? N / 2 : 1)> LowerHalf(Vec128<T, N> v) noexcept {
  return LowerHalf(Simd<T, (N > 1 ? N / 2 : 1), 0>{}, v);
}
template <typename T, std::size_t NH, int kPow2, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, NH> UpperHalf(Simd<T, NH, kPow2>, Vec128<T, N> v) noexcept {
  static_assert(NH == N / 2, "UpperHalf: NH must be N/2");
  Vec128<T, NH> r;
  constexpr int kHalfBytes = static_cast<int>(NH * sizeof(T));
  static_assert(kHalfBytes > 0 && kHalfBytes <= 8, "UpperHalf shift in [1,8]");
  r.raw = detail::mask_from_i<T>(
      _mm_srli_si128(detail::mask_to_i<T>(v.raw), kHalfBytes));
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N / 2> UpperHalf(Vec128<T, N> v) noexcept {
  static_assert(N >= 2, "UpperHalf needs N>=2");
  return UpperHalf(Simd<T, N / 2, 0>{}, v);
}
template <class D> requires HYLE_SIMD_W(D, 0, 16)
HYLE_FORCE_INLINE detail::V128<D> Combine(D /*d*/,
    Vec128<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D) / 2> hi,
    Vec128<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D) / 2> lo) noexcept {
  using T = TFromD<D>;
  static_assert(HYLE_SIMD_MAX_LANES_D(D) >= 2, "Combine needs N>=2");
  constexpr int kBytes = static_cast<int>(HYLE_SIMD_BYTES_D(D));
  constexpr int kHalfBytes = kBytes / 2;
  detail::V128<D> r;
  if constexpr (kBytes == 16) {
    r.raw = detail::mask_from_i<T>(_mm_unpacklo_epi64(
        detail::mask_to_i<T>(lo.raw), detail::mask_to_i<T>(hi.raw)));
  } else {
    const __m128i lo_clean = _mm_srli_si128(
        _mm_slli_si128(detail::mask_to_i<T>(lo.raw), 16 - kHalfBytes),
        16 - kHalfBytes);
    const __m128i joined = _mm_or_si128(
        lo_clean, _mm_slli_si128(detail::mask_to_i<T>(hi.raw), kHalfBytes));
    r.raw = detail::mask_from_i<T>(
        _mm_srli_si128(_mm_slli_si128(joined, 16 - kBytes), 16 - kBytes));
  }
  return r;
}
template <class D> requires HYLE_SIMD_W(D, 0, 16)
HYLE_FORCE_INLINE detail::V128<D> ZeroExtendVector(
    D /*d*/, Vec128<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D) / 2> lo) noexcept {
  using T = TFromD<D>;
  static_assert(HYLE_SIMD_MAX_LANES_D(D) >= 2, "ZeroExtendVector needs N>=2");
  constexpr int kBytes = static_cast<int>(HYLE_SIMD_BYTES_D(D));
  constexpr int kHalfBytes = kBytes / 2;
  detail::V128<D> r;
  const __m128i lo_clean = _mm_srli_si128(
      _mm_slli_si128(detail::mask_to_i<T>(lo.raw), 16 - kHalfBytes),
      16 - kHalfBytes);
  r.raw = detail::mask_from_i<T>(lo_clean);
  return r;
}
template <class D> requires HYLE_SIMD_W(D, 0, 16)
HYLE_FORCE_INLINE detail::V128<D> InsertIntoUpper(
    D /*d*/, Vec128<TFromD<D>, HYLE_SIMD_MAX_LANES_D(D) / 2> hi, detail::V128<D> v) noexcept {
  using T = TFromD<D>;
  static_assert(HYLE_SIMD_MAX_LANES_D(D) >= 2, "InsertIntoUpper needs N>=2");
  constexpr int kBytes = static_cast<int>(HYLE_SIMD_BYTES_D(D));
  constexpr int kHalfBytes = kBytes / 2;
  detail::V128<D> r;
  if constexpr (kBytes == 16) {
    r.raw = detail::mask_from_i<T>(_mm_unpacklo_epi64(
        detail::mask_to_i<T>(v.raw), detail::mask_to_i<T>(hi.raw)));
  } else {
    const __m128i lo_clean = _mm_srli_si128(
        _mm_slli_si128(detail::mask_to_i<T>(v.raw), 16 - kHalfBytes),
        16 - kHalfBytes);
    const __m128i joined = _mm_or_si128(
        lo_clean, _mm_slli_si128(detail::mask_to_i<T>(hi.raw), kHalfBytes));
    r.raw = detail::mask_from_i<T>(
        _mm_srli_si128(_mm_slli_si128(joined, 16 - kBytes), 16 - kBytes));
  }
  return r;
}
// ResizeBitCast within the sse4 band: min-bytes copy, excess zeroed.
template <class D, typename SrcT, std::size_t NSrc> requires HYLE_SIMD_W(D, 0, 16)
HYLE_FORCE_INLINE detail::V128<D> ResizeBitCast(D d, Vec128<SrcT, NSrc> v) noexcept {
  using DstT = TFromD<D>;
  constexpr std::size_t NDst = HYLE_SIMD_MAX_LANES_D(D);
  alignas(16) std::uint8_t buf[16] = {};
  alignas(16) std::uint8_t src[16] = {};
  if constexpr (std::is_same_v<SrcT, float>)
    _mm_store_si128(reinterpret_cast<__m128i*>(src), _mm_castps_si128(v.raw));
  else if constexpr (std::is_same_v<SrcT, double>)
    _mm_store_si128(reinterpret_cast<__m128i*>(src), _mm_castpd_si128(v.raw));
  else
    _mm_store_si128(reinterpret_cast<__m128i*>(src), v.raw);
  const std::size_t n = (sizeof(SrcT) * NSrc < sizeof(DstT) * NDst) ? sizeof(SrcT) * NSrc
                                                                    : sizeof(DstT) * NDst;
  std::memcpy(buf, src, n);
  return Load(d, reinterpret_cast<const DstT*>(buf));
}

// ---- simd-17 batch1: fixed-pattern Shuffle + byte/lane shifts (single 128-bit block) ----
// Immediate constants match Highway / Intel SHUFPS encoding. Float uses _mm_shuffle_ps;
// integer uses _mm_shuffle_epi32. Byte shifts are native SSSE3/SSE2 (_mm_slli/srli_si128,
// _mm_alignr_epi8).
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> Shuffle2301(Vec128<T, N> v) noexcept {
  static_assert(sizeof(T) == 4, "Shuffle2301 needs sizeof(T)==4");
  Vec128<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm_shuffle_ps(v.raw, v.raw, 0xB1);
  else r.raw = _mm_shuffle_epi32(v.raw, 0xB1);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> Shuffle1032(Vec128<T, N> v) noexcept {
  static_assert(sizeof(T) == 4, "Shuffle1032 needs sizeof(T)==4");
  Vec128<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm_shuffle_ps(v.raw, v.raw, 0x4E);
  else r.raw = _mm_shuffle_epi32(v.raw, 0x4E);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> Shuffle0321(Vec128<T, N> v) noexcept {
  static_assert(sizeof(T) == 4, "Shuffle0321 needs sizeof(T)==4");
  Vec128<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm_shuffle_ps(v.raw, v.raw, 0x39);
  else r.raw = _mm_shuffle_epi32(v.raw, 0x39);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> Shuffle2103(Vec128<T, N> v) noexcept {
  static_assert(sizeof(T) == 4, "Shuffle2103 needs sizeof(T)==4");
  Vec128<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm_shuffle_ps(v.raw, v.raw, 0x93);
  else r.raw = _mm_shuffle_epi32(v.raw, 0x93);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> Shuffle0123(Vec128<T, N> v) noexcept {
  static_assert(sizeof(T) == 4, "Shuffle0123 needs sizeof(T)==4");
  Vec128<T, N> r;
  if constexpr (std::is_same_v<T, float>) r.raw = _mm_shuffle_ps(v.raw, v.raw, 0x1B);
  else r.raw = _mm_shuffle_epi32(v.raw, 0x1B);
  return r;
}
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> Shuffle01(Vec128<T, N> v) noexcept {
  static_assert(sizeof(T) == 8, "Shuffle01 needs sizeof(T)==8");
  Vec128<T, N> r;
  if constexpr (std::is_same_v<T, double>) r.raw = _mm_shuffle_pd(v.raw, v.raw, 1);
  else r.raw = _mm_shuffle_epi32(v.raw, 0x4E);
  return r;
}

template <int k, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> ShiftLeftBytes(Vec128<T, N> v) noexcept {
  static_assert(k >= 0 && k <= 16, "ShiftLeftBytes count in [0,16]");
  const __m128i i = detail::bitcast_raw_sse<__m128i>(v.raw);
  const __m128i s = _mm_slli_si128(i, k);
  Vec128<T, N> r;
  r.raw = detail::bitcast_raw_sse<detail::Raw128T<T>>(s);
  return r;
}
template <int k, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> ShiftRightBytes(Vec128<T, N> v) noexcept {
  static_assert(k >= 0 && k <= 16, "ShiftRightBytes count in [0,16]");
  const __m128i i = detail::bitcast_raw_sse<__m128i>(v.raw);
  const __m128i s = _mm_srli_si128(i, k);
  Vec128<T, N> r;
  r.raw = detail::bitcast_raw_sse<detail::Raw128T<T>>(s);
  return r;
}
template <int k, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> ShiftLeftLanes(Vec128<T, N> v) noexcept {
  return ShiftLeftBytes<k * static_cast<int>(sizeof(T))>(v);
}
template <int k, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> ShiftRightLanes(Vec128<T, N> v) noexcept {
  return ShiftRightBytes<k * static_cast<int>(sizeof(T))>(v);
}
template <int k, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> CombineShiftRightBytes(Vec128<T, N> hi, Vec128<T, N> lo) noexcept {
  static_assert(k > 0 && k < 16, "CombineShiftRightBytes count in (0,16)");
  const __m128i hi_i = detail::bitcast_raw_sse<__m128i>(hi.raw);
  const __m128i lo_i = detail::bitcast_raw_sse<__m128i>(lo.raw);
  const __m128i s = _mm_alignr_epi8(hi_i, lo_i, k);  // SSSE3
  Vec128<T, N> r;
  r.raw = detail::bitcast_raw_sse<detail::Raw128T<T>>(s);
  return r;
}
template <int k, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> CombineShiftRightLanes(Vec128<T, N> hi, Vec128<T, N> lo) noexcept {
  return CombineShiftRightBytes<k * static_cast<int>(sizeof(T))>(hi, lo);
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
template <std::uint8_t kRcon, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> AESKeyGenAssist(Vec128<T, N> v) noexcept {
  static_assert(sizeof(T) * N == 16, "AES op needs a 128-bit vector");
  return Vec128<T, N>{detail::aes_keygen_block<kRcon>(v.raw)};
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


// ---- simd-18 批E: 128-bit block ops (single block = identity / Interleave* alias) ----
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> SwapAdjacentBlocks(Vec128<T, N> v) noexcept { return v; }
template <class D, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> ReverseBlocks(D, Vec128<T, N> v) noexcept { return v; }
template <typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> OddEvenBlocks(Vec128<T, N> /*odd*/, Vec128<T, N> even) noexcept { return even; }
template <int kBlock, class D, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> InsertBlock(D, Vec128<T, N> /*v*/, Vec128<T, N> blk) noexcept {
  static_assert(kBlock == 0, "single-block InsertBlock: kBlock==0 only");
  return blk;
}
template <int kB, class D, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> SlideUpBlocks(D, Vec128<T, N> v) noexcept {
  static_assert(kB == 0, "single-block SlideUpBlocks: kB==0 only");
  return v;
}
template <int kB, class D, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> SlideDownBlocks(D, Vec128<T, N> v) noexcept {
  static_assert(kB == 0, "single-block SlideDownBlocks: kB==0 only");
  return v;
}
template <class D, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> InterleaveEvenBlocks(D, Vec128<T, N> a, Vec128<T, N> /*b*/) noexcept { return a; }
template <class D, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> InterleaveOddBlocks(D, Vec128<T, N> a, Vec128<T, N> /*b*/) noexcept { return a; }
template <class D, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> InterleaveLowerBlocks(D, Vec128<T, N> a, Vec128<T, N> /*b*/) noexcept { return a; }
template <class D, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> InterleaveUpperBlocks(D, Vec128<T, N> a, Vec128<T, N> /*b*/) noexcept { return a; }
template <class D, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> InterleaveWholeLower(D, Vec128<T, N> a, Vec128<T, N> b) noexcept {
  return InterleaveLower(a, b);
}
template <class D, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> InterleaveWholeUpper(D, Vec128<T, N> a, Vec128<T, N> b) noexcept {
  return InterleaveUpper(a, b);
}
// E9 Per4LaneBlockShuffle: 32-bit via store+reload oracle (single 128-block = whole vector for N=4).
template <int i3, int i2, int i1, int i0, typename T, std::size_t N>
HYLE_FORCE_INLINE Vec128<T, N> Per4LaneBlockShuffle(Vec128<T, N> v) noexcept {
  static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
                "Per4LaneBlockShuffle: 1/2/4/8-byte lanes");
  if constexpr (i3 == 3 && i2 == 2 && i1 == 1 && i0 == 0) {
    return v;  // identity: no-op (see avx2.h)
  } else {
    alignas(16) T s[16 / sizeof(T)], o[16 / sizeof(T)];
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
HYLE_FORCE_INLINE Vec128<T, N> TwoTablesLookupLanes(D d, Vec128<T, N> a, Vec128<T, N> b,
                                                 Indices128<T, N> idx) noexcept {
  alignas(16) T sa[16 / sizeof(T)], sb[16 / sizeof(T)], ix[16 / sizeof(T)], out[16 / sizeof(T)];
  Store(a, d, sa); Store(b, d, sb); Store(idx.idx, d, ix);
  for (std::size_t i = 0; i < N; ++i) {
    const auto k = static_cast<std::size_t>(ix[i]);
    out[i] = (k < N) ? sa[k] : sb[k - N];
  }
  return Load(d, out);
}

}  // namespace hyle::simd
