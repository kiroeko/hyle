#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>  // std::memcpy (software fallback: __m128i <-> 16 bytes)

#include <emmintrin.h>  // __m128i (SSE2, always available on x86-64)
// AES-NI (_mm_aesenc_si128 ...) + PCLMULQDQ (_mm_clmulepi64_si128) live in <wmmintrin.h>.
// GCC/Clang (including clang-cl) only expose them under -maes / -mpclmul
// (__AES__ / __PCLMUL__); cl.exe x64 exposes them unconditionally. Guard the
// include so a base variant (no -maes) still compiles and falls back to the
// software floor below. _MSC_VER alone is not a cl.exe test: clang-cl defines it
// for ABI compatibility while still enforcing Clang's target-feature contract.
#if (defined(_MSC_VER) && !defined(__clang__)) || defined(__AES__) || defined(__PCLMUL__)
#  include <wmmintrin.h>
#endif

#include "hyle/core/compiler.h"
#include "hyle/simd/backend/crypto_soft.h"  // shared AES/CLMul software reference

// backend/crypto_x86.h -- ONE-per-128-bit-block crypto workers shared by the sse4 / avx2 /
// avx512 x86 backends. AES-NI + PCLMULQDQ operate on a 128-bit register regardless of the
// surrounding tier, so the wider backends (avx2 = 2 blocks, avx512 = 4 blocks) split their
// native-width register into 128-bit lanes and call these per lane -- exactly Highway's
// x86_256 / x86_512 AES model (and the VAES _mm256/512_aesenc_epi128 semantics), just built
// from the always-present AES-NI rather than requiring the optional VAES extension (which
// would need -mvaes and Ice Lake+). Promoting to VAES is a pure profiling upgrade.
//
// When AES-NI / PCLMULQDQ are compiled out (base variant, no -maes/-mpclmul), every worker
// falls back to the shared software reference (detail::aes_*_soft / clmul64), which uses the
// identical AES-NI byte order, so hardware and software paths are bit-for-bit equal -- the
// crypto test asserts exactly that by running the same source with and without -maes.

namespace hyle::simd::detail {

// ---- AES round primitives on a single 128-bit block --------------------------------------
HYLE_FORCE_INLINE __m128i aes_round_block(__m128i s, __m128i k) noexcept {
#if (defined(_MSC_VER) && !defined(__clang__)) || defined(__AES__)
  return _mm_aesenc_si128(s, k);
#else
  std::uint8_t sb[16], kb[16];
  std::memcpy(sb, &s, 16); std::memcpy(kb, &k, 16);
  aes_round_soft(sb, kb);
  __m128i r; std::memcpy(&r, sb, 16); return r;
#endif
}
HYLE_FORCE_INLINE __m128i aes_last_round_block(__m128i s, __m128i k) noexcept {
#if (defined(_MSC_VER) && !defined(__clang__)) || defined(__AES__)
  return _mm_aesenclast_si128(s, k);
#else
  std::uint8_t sb[16], kb[16];
  std::memcpy(sb, &s, 16); std::memcpy(kb, &k, 16);
  aes_last_round_soft(sb, kb);
  __m128i r; std::memcpy(&r, sb, 16); return r;
#endif
}
HYLE_FORCE_INLINE __m128i aes_round_inv_block(__m128i s, __m128i k) noexcept {
#if (defined(_MSC_VER) && !defined(__clang__)) || defined(__AES__)
  return _mm_aesdec_si128(s, k);
#else
  std::uint8_t sb[16], kb[16];
  std::memcpy(sb, &s, 16); std::memcpy(kb, &k, 16);
  aes_round_inv_soft(sb, kb);
  __m128i r; std::memcpy(&r, sb, 16); return r;
#endif
}
HYLE_FORCE_INLINE __m128i aes_last_round_inv_block(__m128i s, __m128i k) noexcept {
#if (defined(_MSC_VER) && !defined(__clang__)) || defined(__AES__)
  return _mm_aesdeclast_si128(s, k);
#else
  std::uint8_t sb[16], kb[16];
  std::memcpy(sb, &s, 16); std::memcpy(kb, &k, 16);
  aes_last_round_inv_soft(sb, kb);
  __m128i r; std::memcpy(&r, sb, 16); return r;
#endif
}
HYLE_FORCE_INLINE __m128i aes_inv_mix_block(__m128i s) noexcept {
#if (defined(_MSC_VER) && !defined(__clang__)) || defined(__AES__)
  return _mm_aesimc_si128(s);
#else
  std::uint8_t sb[16];
  std::memcpy(sb, &s, 16);
  aes_inv_mix_columns(sb);
  __m128i r; std::memcpy(&r, sb, 16); return r;
#endif
}

// AESKeyGenAssist needs a compile-time rcon immediate, so the worker is templated on kRcon.
template <std::uint8_t kRcon>
HYLE_FORCE_INLINE __m128i aes_keygen_block(__m128i v) noexcept {
#if (defined(_MSC_VER) && !defined(__clang__)) || defined(__AES__)
  return _mm_aeskeygenassist_si128(v, kRcon);
#else
  std::uint8_t s[16];
  std::memcpy(s, &v, 16);
  aes_keygen_assist_soft(s, kRcon);
  __m128i r; std::memcpy(&r, s, 16); return r;
#endif
}

// ---- carryless multiply on a single 128-bit block ----------------------------------------
HYLE_FORCE_INLINE __m128i clmul_lower_block(__m128i a, __m128i b) noexcept {
#if (defined(_MSC_VER) && !defined(__clang__)) || defined(__PCLMUL__)
  return _mm_clmulepi64_si128(a, b, 0x00);
#else
  std::uint64_t av[2], bv[2], o[2];
  std::memcpy(av, &a, 16); std::memcpy(bv, &b, 16);
  clmul64(av[0], bv[0], o);
  __m128i r; std::memcpy(&r, o, 16); return r;
#endif
}
HYLE_FORCE_INLINE __m128i clmul_upper_block(__m128i a, __m128i b) noexcept {
#if (defined(_MSC_VER) && !defined(__clang__)) || defined(__PCLMUL__)
  return _mm_clmulepi64_si128(a, b, 0x11);
#else
  std::uint64_t av[2], bv[2], o[2];
  std::memcpy(av, &a, 16); std::memcpy(bv, &b, 16);
  clmul64(av[1], bv[1], o);
  __m128i r; std::memcpy(&r, o, 16); return r;
#endif
}

}  // namespace hyle::simd::detail
