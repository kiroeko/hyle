// xsimd-accelerated AVX2 transcendentals (8-wide). BORROWS xsimd's verified math cores
// (xsimd::exp/tanh) -- HYLE does not hand-roll these (charter: 借验证核不手搓). The same-ISA
// bench (math_bench) found xsimd's exp/tanh polynomials fastest at the ~3.5-ULP class
// (Highway wins log), so per-op best-of-breed (math_dispatch.cc) uses xsimd for EXP/TANH
// ONLY -- hence log is not defined here (it would be dead code: Highway serves log).
//
// x86-only: xsimd::batch<float> lowers to the 8-wide AVX2 register here, so the TU compiles
// to nothing on non-x86 (math_dispatch falls to the scalar libm floor there). Header-only +
// vendored in third_party, so it can't be missing -- no opt-in define gates it, only the
// architecture does. Compiled at -mavx2; reached only when cpu_features().avx2.
#include "hyle/platform.h"  // HYLE_ARCH_X64 -- MUST precede the guard below

#if defined(HYLE_ARCH_X64)

#include "src/simd/math/kernels.h"

#include <cmath>
#include <cstddef>

#include "xsimd/xsimd.hpp"

namespace hyle::simd::detail {

namespace {
using batch = xsimd::batch<float>;
constexpr std::size_t kW = batch::size;  // 8 at AVX2
}  // namespace

void exp_f32_xsimd_avx2(const float* src, float* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + kW <= n; i += kW)
    xsimd::exp(batch::load_unaligned(src + i)).store_unaligned(dst + i);
  for (; i < n; ++i) dst[i] = std::exp(src[i]);  // tail = libm (scalar oracle)
}
void tanh_f32_xsimd_avx2(const float* src, float* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + kW <= n; i += kW)
    xsimd::tanh(batch::load_unaligned(src + i)).store_unaligned(dst + i);
  for (; i < n; ++i) dst[i] = std::tanh(src[i]);
}

// simd-09: xsimd wins exp2/erf at the same ISA (bench-decided; Highway serves the rest).
void exp2_f32_xsimd_avx2(const float* src, float* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + kW <= n; i += kW)
    xsimd::exp2(batch::load_unaligned(src + i)).store_unaligned(dst + i);
  for (; i < n; ++i) dst[i] = std::exp2(src[i]);
}
void erf_f32_xsimd_avx2(const float* src, float* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + kW <= n; i += kW)
    xsimd::erf(batch::load_unaligned(src + i)).store_unaligned(dst + i);
  for (; i < n; ++i) dst[i] = std::erf(src[i]);
}

// xsimd-ONLY borrow: Highway's contrib/math has no gamma, so there is no best-of race -- xsimd
// is the single source. Only TGAMMA is borrowed: xsimd's vectorized LGAMMA has a lane-mixing bug
// (a lane in [1.25,1.5) gets the wrong argument-reduction kernel when it shares a vector with a
// lane in [0.75,1.25); reproduced GCC-x86 + MSVC-x86 + GCC-NEON => algorithm, not codegen). So
// lgamma stays on the scalar libm reference. See docs/debug/xsimd-lgamma-lane-mixing-hunt.md.
void tgamma_f32_xsimd_avx2(const float* src, float* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + kW <= n; i += kW)
    xsimd::tgamma(batch::load_unaligned(src + i)).store_unaligned(dst + i);
  for (; i < n; ++i) dst[i] = std::tgamma(src[i]);
}

}  // namespace hyle::simd::detail

#endif  // HYLE_ARCH_X64
