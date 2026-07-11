// AVX10.2 f16 -> fp8 conversion (compiled -mavx10.2-512; ONLY clang-20+ supports that flag today
// -- GCC-14 stops at avx10.1, MSVC has none). Gated at BUILD time (HYLE_HAVE_AVX10_2, set by CMake
// when the compiler accepts -mavx10.2-512) AND at RUNTIME (cpu_features().avx10_2). When the flag
// is unavailable the whole TU compiles to nothing and dispatch falls back to the scalar core.
//
// WHY f16 SOURCE (not f32): AVX10.2's fp8 converts only take an f16 source (vcvtph2hf8 / vcvtph2bf8).
// f16->f32 is EXACT, so HYLE's scalar f32_to_f8(f16_to_f32(h)) rounds the exact f16 value to fp8
// ONCE -- same single rounding as the hardware -> bit-identical (verified exhaustively over all
// 65536 f16). E4M3 matches with no fixup. E5M2 differs ONLY on f16 inf/NaN: OCP E5M2 HAS inf so
// HYLE keeps it (0x7C) / NaN (0x7E), but Intel's vcvtph2bf8 SATURATES inf to max-normal (0x7B).
// We restore HYLE's inf/NaN semantics on the exp==0x1F lanes so the hardware path stays exactly
// bit-identical to the scalar oracle -- a transparent dispatch target (unlike the f32-source path).
#include "hyle/platform.h"  // HYLE_ARCH_X64

#if defined(HYLE_ARCH_X64) && defined(HYLE_HAVE_AVX10_2)

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

}  // namespace hyle::simd::detail

#endif  // HYLE_ARCH_X64 && HYLE_HAVE_AVX10_2
