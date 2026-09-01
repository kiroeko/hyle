#include "hyle/core/environment.h"

#if HYLE_ARCH_X64

#include "src/simd/ops/kernels.h"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

#include "hyle/simd/dtype.h"

namespace hyle::simd::detail {
namespace {
HYLE_FORCE_INLINE __m512bh as_bf16(__m512i v) noexcept {
#if defined(_MSC_VER)
  return v;  // MSVC typedefs __m512bh to __m512i.
#else
  return reinterpret_cast<__m512bh>(v);
#endif
}
}  // namespace

// AVX512-BF16 pairwise dot: 32 packed bf16 inputs become 16 f32 partial
// sums per instruction. The runtime reaches this only after BF16+OS-ZMM gating.
float dot_bf16_bf16_avx512bf16(const bfloat16_t* x, const bfloat16_t* y,
                               std::size_t n) noexcept {
  __m512 sum0 = _mm512_setzero_ps();
  __m512 sum1 = _mm512_setzero_ps();
  __m512 sum2 = _mm512_setzero_ps();
  __m512 sum3 = _mm512_setzero_ps();
  std::size_t i = 0;
  for (; i + 128 <= n; i += 128) {
    const __m512i x0 = _mm512_loadu_si512(x + i);
    const __m512i y0 = _mm512_loadu_si512(y + i);
    const __m512i x1 = _mm512_loadu_si512(x + i + 32);
    const __m512i y1 = _mm512_loadu_si512(y + i + 32);
    const __m512i x2 = _mm512_loadu_si512(x + i + 64);
    const __m512i y2 = _mm512_loadu_si512(y + i + 64);
    const __m512i x3 = _mm512_loadu_si512(x + i + 96);
    const __m512i y3 = _mm512_loadu_si512(y + i + 96);
    sum0 = _mm512_dpbf16_ps(sum0, as_bf16(x0), as_bf16(y0));
    sum1 = _mm512_dpbf16_ps(sum1, as_bf16(x1), as_bf16(y1));
    sum2 = _mm512_dpbf16_ps(sum2, as_bf16(x2), as_bf16(y2));
    sum3 = _mm512_dpbf16_ps(sum3, as_bf16(x3), as_bf16(y3));
  }
  for (; i + 32 <= n; i += 32) {
    const __m512i vx = _mm512_loadu_si512(x + i);
    const __m512i vy = _mm512_loadu_si512(y + i);
    sum0 = _mm512_dpbf16_ps(sum0, as_bf16(vx), as_bf16(vy));
  }
  const __m512 total = _mm512_add_ps(_mm512_add_ps(sum0, sum1),
                                     _mm512_add_ps(sum2, sum3));
  alignas(64) float lanes[16];
  _mm512_store_ps(lanes, total);
  float sum = 0.0f;
  for (float lane : lanes) sum += lane;
  for (; i < n; ++i) sum += bf16_to_f32(x[i]) * bf16_to_f32(y[i]);
  return sum;
}

}  // namespace hyle::simd::detail

#endif  // HYLE_ARCH_X64
