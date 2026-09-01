#include "hyle/core/environment.h"

#if HYLE_ARCH_X64

#include "src/simd/ops/kernels.h"

#include <immintrin.h>

#include <cstddef>

#include "hyle/simd/dtype.h"

namespace hyle::simd::detail {

// AVX+F16C, deliberately below AVX2/FMA so Ivy Bridge can use the hardware
// conversion path. The runtime reaches this only after F16C+OS-YMM gating.
float dot_f16_f16_f16c(const float16_t* x, const float16_t* y,
                        std::size_t n) noexcept {
  __m256 sum0 = _mm256_setzero_ps();
  __m256 sum1 = _mm256_setzero_ps();
  __m256 sum2 = _mm256_setzero_ps();
  __m256 sum3 = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    const __m256 x0 = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i)));
    const __m256 y0 = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(y + i)));
    const __m256 x1 = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i + 8)));
    const __m256 y1 = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(y + i + 8)));
    const __m256 x2 = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i + 16)));
    const __m256 y2 = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(y + i + 16)));
    const __m256 x3 = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i + 24)));
    const __m256 y3 = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(y + i + 24)));
    sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(x0, y0));
    sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(x1, y1));
    sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(x2, y2));
    sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(x3, y3));
  }
  for (; i + 8 <= n; i += 8) {
    const __m256 vx = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i)));
    const __m256 vy = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(y + i)));
    sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(vx, vy));
  }
  const __m256 total = _mm256_add_ps(_mm256_add_ps(sum0, sum1),
                                     _mm256_add_ps(sum2, sum3));
  __m128 reduced = _mm_add_ps(_mm256_castps256_ps128(total),
                              _mm256_extractf128_ps(total, 1));
  reduced = _mm_hadd_ps(reduced, reduced);
  reduced = _mm_hadd_ps(reduced, reduced);
  float sum = _mm_cvtss_f32(reduced);
  for (; i < n; ++i) sum += f16_to_f32(x[i]) * f16_to_f32(y[i]);
  return sum;
}

}  // namespace hyle::simd::detail

#endif  // HYLE_ARCH_X64
