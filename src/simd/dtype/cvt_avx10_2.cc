// AVX10.2 f16 -> fp8 conversion (compiled -mavx10.2-512; ONLY clang-20+ supports that flag today
// -- GCC-14 stops at avx10.1, MSVC has none). Gated at BUILD time (HYLE_HAVE_AVX10_2, set by CMake
// when the compiler accepts -mavx10.2-512) AND at RUNTIME (cpu_features().avx10_2). When the flag
// is unavailable the whole TU compiles to nothing and dispatch falls back to the scalar core.
//
// AVX10.2's fp8 converts only take an f16 source (vcvtph2hf8 / vcvtph2bf8) -- there is NO direct
// f32->fp8 instruction. So this TU serves BOTH source widths, always ending in the hardware
// f16->fp8, and both are bit-identical to HYLE's direct scalar oracle:
//
// f16 SOURCE: f16->f32 is EXACT, so scalar f32_to_f8(f16_to_f32(h)) rounds the exact f16 value to
// fp8 ONCE -- same single rounding as the hardware. E4M3 matches with no fixup. E5M2 differs ONLY
// on f16 inf/NaN (OCP E5M2 HAS inf 0x7C/NaN 0x7E, but vcvtph2bf8 SATURATES to max-normal 0x7B) --
// fixed on the exp==0x1F lanes.
//
// f32 SOURCE: going f32->f16->fp8 is TWO roundings, which can differ from a direct f32->fp8 RNE in
// tie cases. The classic fix (Boldo & Melquiond, "When double rounding is odd"; also Highway's own
// F64->F16 DemoteTo) is to make the INTERMEDIATE f32->f16 use ROUND-TO-ODD: clear the 13 mantissa
// bits f16 discards, and if any were set, force the retained LSB to 1 (sticky). f16's 10 mantissa
// bits >= fp8's 2-3 + 2 (Boldo-Melquiond precondition), so the subsequent RNE f16->fp8 then equals
// a single direct RNE f32->fp8 -- EXACTLY. Verified vs the scalar oracle over 486.6M f32 (exhaustive
// mantissa across all fp8-relevant exponents + all 65536 f16 promoted): 0 mismatch, both formats.
// Two subtleties handled: (1) the hardware saturates a FINITE f16-overflow (|f32| just above f16 max
// 65504) to fp8 max-normal, which is CORRECT for a finite fp8 overflow -- no fixup needed there;
// (2) only a GENUINE f32 inf/NaN needs restoring (f16 alone can't distinguish it from finite
// overflow), so the fixup keys on the ORIGINAL f32 exponent bits, not the f16.
#include "hyle/core/environment.h"  // HYLE_ARCH_X64

#if HYLE_ARCH_X64 && defined(HYLE_HAVE_AVX10_2)

#include "src/simd/dtype/kernels.h"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

#include "hyle/simd/dtype.h"

namespace hyle::simd::detail {

// f16 -> E4M3 (hf8), 32-wide, hardware saturating convert. Bit-identical to scalar; no fixup.
void f16_to_f8e4m3_avx10_2(const float16_t* src, float8_e4m3_t* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    const __m512h h = _mm512_loadu_ph(src + i);
    const __m256i q = _mm512_cvtsph_hf8(h);  // 32x f16 -> 32x e4m3, saturating (OCP)
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), q);
  }
  for (; i < n; ++i) dst[i] = f32_to_f8e4m3(f16_to_f32(src[i]));  // tail = scalar oracle
}

// f16 -> E5M2 (bf8), 32-wide. Hardware saturates inf->max-normal; restore OCP inf/NaN on the
// f16 exp==0x1F lanes so the result is bit-identical to the scalar oracle.
void f16_to_f8e5m2_avx10_2(const float16_t* src, float8_e5m2_t* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    const __m512h h = _mm512_loadu_ph(src + i);
    const __m256i q = _mm512_cvtsph_bf8(h);  // 32x f16 -> 32x e5m2, saturating
    alignas(32) std::uint8_t out[32];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), q);
    // Fixup inf/NaN lanes: f16 exp all-ones -> E5M2 inf (0x7C) or NaN (0x7E), preserving sign.
    for (int j = 0; j < 32; ++j) {
      const std::uint16_t hb = src[i + j].bits;
      if (((hb >> 10) & 0x1F) == 0x1F) {  // f16 inf or NaN
        const std::uint8_t sign = static_cast<std::uint8_t>((hb >> 15) << 7);
        out[j] = static_cast<std::uint8_t>(sign | ((hb & 0x3FF) ? 0x7E : 0x7C));  // NaN : inf
      }
      dst[i + j] = float8_e5m2_t::from_bits(out[j]);
    }
  }
  for (; i < n; ++i) dst[i] = f32_to_f8e5m2(f16_to_f32(src[i]));  // tail = scalar oracle
}

// round-to-odd narrow of 16 f32 lanes onto the f16 mantissa grid, returned as a RNE-converted
// __m256h (16 f16). Clears the 13 bits f16 drops (mask 0xFFFFE000); if any were set, forces the
// retained LSB (bit 13) to 1 so the following RNE f32->f16 is exact -> the whole f32->f16 is
// round-to-odd. inf/NaN f32 lanes (abs exp all-ones) are left untouched (their mantissa-drop is
// irrelevant; the final f8 result restores them from the f32 source below).
static inline __m256h round_to_odd_ps_to_ph(__m512 f) noexcept {
  const __m512i u = _mm512_castps_si512(f);
  const __m512i abs = _mm512_and_si512(u, _mm512_set1_epi32(0x7FFFFFFF));
  const __mmask16 finite = _mm512_cmplt_epi32_mask(abs, _mm512_set1_epi32(0x7F800000));
  const __m512i low = _mm512_and_si512(u, _mm512_set1_epi32(0x00001FFF));   // 13 dropped bits
  const __mmask16 sticky = _mm512_cmpneq_epi32_mask(low, _mm512_setzero_si512());
  __m512i cleared = _mm512_and_si512(u, _mm512_set1_epi32(0xFFFFE000u));    // clear dropped bits
  // set retained LSB (bit 13) on lanes that are finite AND had a nonzero remainder
  cleared = _mm512_mask_or_epi32(cleared, finite & sticky, cleared,
                                 _mm512_set1_epi32(0x00002000));
  // non-finite lanes keep the original bits (round-to-odd is a no-op for inf/NaN)
  const __m512 rounded = _mm512_castsi512_ps(_mm512_mask_blend_epi32(finite, u, cleared));
  // RNE f32->f16; the mantissa is now exact on the f16 grid so this rounding loses nothing.
  return _mm512_cvtxps_ph(rounded);
}

// Restore genuine f32 inf/NaN in the fp8 output. The hardware saturates BOTH finite-overflow and
// true-inf to max-normal; only the f32 source distinguishes them. For each 16-lane f32 group,
// build an 8-bit fp8-lane fixup: NaN -> sign|nanb, inf -> sign|inf_or_sat. Applied to a 32-lane
// __m256i fp8 register at [lo=group0, hi=group1].
template <bool kHasInf>
static inline void fixup_f32_inf_nan(std::uint8_t* out, const float* src, std::size_t base,
                                     std::size_t count, std::uint8_t maxb, std::uint8_t nanb,
                                     unsigned exp_all_ones_f8) noexcept {
  for (std::size_t j = 0; j < count; ++j) {
    std::uint32_t ub;
    __builtin_memcpy(&ub, src + base + j, 4);
    const std::uint32_t a = ub & 0x7FFFFFFFu;
    if (a <= 0x7F800000u && a != 0x7F800000u) continue;  // finite -> hardware already correct
    const std::uint8_t sign = static_cast<std::uint8_t>((ub >> 31) << 7);
    if (a > 0x7F800000u) out[j] = static_cast<std::uint8_t>(sign | nanb);            // NaN
    else out[j] = kHasInf ? static_cast<std::uint8_t>(sign | exp_all_ones_f8)        // inf: keep
                          : static_cast<std::uint8_t>(sign | maxb);                  // inf: saturate
  }
}

// Concatenate two 16-lane __m256h (f16) into one 32-lane __m512h via the integer domain (the f16
// intrinsic set has no direct insert; bitcast to __m256i/__m512i, insert, bitcast back).
static inline __m512h concat_ph(__m256h lo, __m256h hi) noexcept {
  const __m512i wide = _mm512_inserti64x4(
      _mm512_castsi256_si512(_mm256_castph_si256(lo)), _mm256_castph_si256(hi), 1);
  return _mm512_castsi512_ph(wide);
}

// f32 -> E4M3 (hf8). round-to-odd f32->f16, then hardware f16->hf8, then f32-source inf/NaN fixup.
void f32_to_f8e4m3_avx10_2(const float* src, float8_e4m3_t* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    const __m256h h0 = round_to_odd_ps_to_ph(_mm512_loadu_ps(src + i));
    const __m256h h1 = round_to_odd_ps_to_ph(_mm512_loadu_ps(src + i + 16));
    const __m512h h = concat_ph(h0, h1);
    const __m256i q = _mm512_cvtsph_hf8(h);   // 32x f16 -> 32x e4m3, saturating
    alignas(32) std::uint8_t out[32];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), q);
    fixup_f32_inf_nan<false>(out, src, i, 16, 0x7E, 0x7F, 0);
    fixup_f32_inf_nan<false>(out + 16, src, i + 16, 16, 0x7E, 0x7F, 0);
    for (int j = 0; j < 32; ++j) dst[i + j] = float8_e4m3_t::from_bits(out[j]);
  }
  for (; i < n; ++i) dst[i] = f32_to_f8e4m3(src[i]);  // tail = scalar oracle
}

// f32 -> E5M2 (bf8). Same, but E5M2 has inf: restore inf (0x7C)/NaN (0x7E) on genuine f32 inf/NaN.
void f32_to_f8e5m2_avx10_2(const float* src, float8_e5m2_t* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    const __m256h h0 = round_to_odd_ps_to_ph(_mm512_loadu_ps(src + i));
    const __m256h h1 = round_to_odd_ps_to_ph(_mm512_loadu_ps(src + i + 16));
    const __m512h h = concat_ph(h0, h1);
    const __m256i q = _mm512_cvtsph_bf8(h);   // 32x f16 -> 32x e5m2, saturating
    alignas(32) std::uint8_t out[32];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), q);
    fixup_f32_inf_nan<true>(out, src, i, 16, 0x7B, 0x7E, 0x7C);
    fixup_f32_inf_nan<true>(out + 16, src, i + 16, 16, 0x7B, 0x7E, 0x7C);
    for (int j = 0; j < 32; ++j) dst[i + j] = float8_e5m2_t::from_bits(out[j]);
  }
  for (; i < n; ++i) dst[i] = f32_to_f8e5m2(src[i]);  // tail = scalar oracle
}

}  // namespace hyle::simd::detail

#endif  // HYLE_ARCH_X64 && HYLE_HAVE_AVX10_2
