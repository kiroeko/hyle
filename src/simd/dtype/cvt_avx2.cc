// AVX2 bf16 batch conversion (compiled -mavx2 / MSVC /arch:AVX2). Only REACHED when
// cpu_features().avx2 is true (the dispatcher gates it), so no AVX2 instruction executes
// on a CPU lacking it. 8-wide; scalar tail via dtype.h inline so it matches the scalar
// oracle EXACTLY (bf16<->f32 is pure integer bit-math, so the vector path is bit-identical
// to scalar, NaN payloads included -- unlike the F16C fp16 path).
//
// Why this exists (the bf16 batch was scalar-only): bf16 needs no AVX-512-BF16 instruction
// to vectorize -- bf16 IS the top 16 bits of f32, so the convert is a shift (bf16->f32) or
// a round-to-nearest-even bias + shift (f32->bf16), both expressible in plain AVX2 integer
// ops. Highway does exactly this; HYLE had deferred it, leaving the batch op losing to an
// inlined scalar loop. This closes that.

#include "src/simd/dtype/kernels.h"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

#include "hyle/simd/dtype.h"

namespace hyle::simd::detail {

// bf16 -> f32: zero-extend each u16 to u32, shift left 16, reinterpret as f32. Exact.
void bf16_to_f32_avx2(const bfloat16_t* src, float* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
    const __m256i w = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
    _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(w));
  }
  for (; i < n; ++i) dst[i] = bf16_to_f32(src[i]);  // tail = scalar oracle
}

// f32 -> bf16, round-to-nearest-even. Mirrors the scalar f32_to_bf16 bit-for-bit on 8
// lanes: incr = finite ? 0x7FFF + (bit16) : 0;  snan = (abs > inf) ? (1<<6) : 0;
// out = (snan | ((u + incr) >> 16)).
void f32_to_bf16_avx2(const float* src, bfloat16_t* dst, std::size_t n) noexcept {
  const __m256i k7fffffff = _mm256_set1_epi32(0x7FFFFFFF);
  const __m256i kInf = _mm256_set1_epi32(0x7F800000);
  const __m256i k7fff = _mm256_set1_epi32(0x7FFF);
  const __m256i kOne = _mm256_set1_epi32(1);
  const __m256i kSnanBit = _mm256_set1_epi32(1 << 6);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256i u = _mm256_castps_si256(_mm256_loadu_ps(src + i));
    const __m256i absv = _mm256_and_si256(u, k7fffffff);
    const __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(u, 16), kOne);
    // finite = abs < inf (abs has the sign bit cleared -> nonneg, signed cmp is valid)
    const __m256i finite = _mm256_cmpgt_epi32(kInf, absv);
    const __m256i incr = _mm256_and_si256(_mm256_add_epi32(k7fff, lsb), finite);
    // snan = abs > inf
    const __m256i snan = _mm256_and_si256(_mm256_cmpgt_epi32(absv, kInf), kSnanBit);
    const __m256i shifted = _mm256_srli_epi32(_mm256_add_epi32(u, incr), 16);
    const __m256i out32 = _mm256_or_si256(snan, shifted);  // each in [0, 0xFFFF]
    // Pack 8x u32 (low 16 bits each) -> 8x u16. packus is per-128-lane: gives
    // [A A B B] qwords (A=lanes0-3, B=lanes4-7); permute4x64 reorders to [A B ..].
    const __m256i packed = _mm256_packus_epi32(out32, out32);
    const __m256i ord = _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i),
                     _mm256_castsi256_si128(ord));
  }
  for (; i < n; ++i) dst[i] = f32_to_bf16(src[i]);  // tail = scalar oracle
}

}  // namespace hyle::simd::detail
