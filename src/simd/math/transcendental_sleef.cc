// SLEEF-accelerated AVX2 transcendentals (8-wide, _u10 = 1 ULP). Built ONLY when
// HYLE_SIMD_SLEEF is defined (CMake gates this TU on -DHYLE_SIMD_SLEEF=ON + a prebuilt
// SLEEF). BORROWS SLEEF's verified polynomial cores -- HYLE does not hand-roll these.
//
// Uses SLEEF's PORTABLE declaration header <sleef.h> (extern decls only; the bodies live
// in the linked libsleef), NOT the GCC-only inline header -- so this is dual-platform
// (MSVC builds libsleef via winbuild-msvc.bat). Reached only when best_isa() >= AVX2, so
// the -mavx2 codegen never runs on a non-AVX2 CPU. 8-wide loop + scalar tail (libm).
#if defined(HYLE_SIMD_SLEEF)

#include "src/simd/math/kernels.h"

#include <cmath>
#include <cstddef>
#include <immintrin.h>

#include <sleef.h>  // portable extern declarations (Sleef_expf8_u10avx2 ...)

namespace hyle::simd::detail {

void exp_f32_sleef_avx2(const float* src, float* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8)
    _mm256_storeu_ps(dst + i, Sleef_expf8_u10avx2(_mm256_loadu_ps(src + i)));
  for (; i < n; ++i) dst[i] = std::exp(src[i]);  // tail = libm (matches scalar oracle)
}
void log_f32_sleef_avx2(const float* src, float* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8)
    _mm256_storeu_ps(dst + i, Sleef_logf8_u10avx2(_mm256_loadu_ps(src + i)));
  for (; i < n; ++i) dst[i] = std::log(src[i]);
}
void tanh_f32_sleef_avx2(const float* src, float* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8)
    _mm256_storeu_ps(dst + i, Sleef_tanhf8_u10avx2(_mm256_loadu_ps(src + i)));
  for (; i < n; ++i) dst[i] = std::tanh(src[i]);
}

}  // namespace hyle::simd::detail

#endif  // HYLE_SIMD_SLEEF
