// F16C hardware fp16 batch conversion (compiled -mf16c -mavx2 / MSVC /arch:AVX2). Only
// REACHED when cpu_features().f16c is true (the dispatcher gates it), so vcvtph2ps /
// vcvtps2ph never execute on a CPU lacking F16C. 8-wide; scalar tail via dtype.h inline
// (so the tail matches the scalar oracle exactly).
//
// Rounding: _mm256_cvtps_ph takes _MM_FROUND_NO_EXC (==8) alone -- bits[1:0]=00 already
// selects round-to-nearest-even. Hardware chooses a different NaN payload, so the cold
// exceptional lanes are restored through HYLE's scalar oracle after the vector store.

#include "src/simd/dtype/kernels.h"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

#include "hyle/simd/dtype.h"

namespace hyle::simd::detail {

void f16_to_f32_f16c(const float16_t* src, float* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
    _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(h));
  }
  for (; i < n; ++i) dst[i] = f16_to_f32(src[i]);  // tail = scalar oracle
}

void f32_to_f16_f16c(const float* src, float16_t* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 f = _mm256_loadu_ps(src + i);
    // _MM_FROUND_NO_EXC (8) is a VALID immediate selecting round-to-nearest-even with no
    // exceptions; MSVC's C4556 ("value out of 0-7") is a false positive (Highway documents
    // and suppresses the identical case). Keep the correct immediate, mute the bogus warn.
#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable : 4556)
#endif
    const __m128i h = _mm256_cvtps_ph(f, _MM_FROUND_NO_EXC);
#if defined(_MSC_VER)
#  pragma warning(pop)
#endif
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), h);
    const int nan_mask =
        _mm256_movemask_ps(_mm256_cmp_ps(f, f, _CMP_UNORD_Q));
    if (nan_mask != 0) {
      for (std::size_t lane = 0; lane < 8; ++lane) {
        if ((nan_mask & (1 << lane)) != 0)
          dst[i + lane] = f32_to_f16(src[i + lane]);
      }
    }
  }
  for (; i < n; ++i) dst[i] = f32_to_f16(src[i]);  // tail = scalar oracle
}

}  // namespace hyle::simd::detail
