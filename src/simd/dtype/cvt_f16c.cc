// F16C hardware fp16 batch conversion (compiled -mf16c -mavx2 / MSVC /arch:AVX2). Only
// REACHED when cpu_features().f16c is true (the dispatcher gates it), so vcvtph2ps /
// vcvtps2ph never execute on a CPU lacking F16C. 8-wide; scalar tail via dtype.h inline
// (so the tail matches the scalar oracle exactly).
//
// Rounding: _mm256_cvtps_ph takes _MM_FROUND_NO_EXC (==8) alone -- bits[1:0]=00 already
// selects round-to-nearest-even, matching the scalar F16FromF32 (verified vs Highway,
// which passes the same immediate). NaN payloads MAY differ from the scalar path (hardware
// keeps truncated mantissa bits; scalar forces 0x3FF) -- finite values are bit-identical;
// the cross-check test compares NaN by is-NaN, not bits.

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
  }
  for (; i < n; ++i) dst[i] = f32_to_f16(src[i]);  // tail = scalar oracle
}

}  // namespace hyle::simd::detail
