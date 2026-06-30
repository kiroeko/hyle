// xsimd-accelerated AVX2 transcendentals (8-wide). Built ONLY when HYLE_SIMD_XSIMD_MATH
// is defined (CMake gates this TU on -DHYLE_SIMD_XSIMD_MATH=ON + -DHYLE_XSIMD_DIR=<xsimd>).
// BORROWS xsimd's verified math cores (xsimd::exp/log/tanh) -- HYLE does not hand-roll
// these (charter: 借验证核不手搓).
//
// WHY a THIRD vectorized-math backend (next to Highway + SLEEF): the same-ISA bench
// (math_bench) found NO single library wins all three ops at the same ~3.5-ULP class --
// xsimd's exp/tanh polynomials are fastest, Highway's log is fastest. So the dispatcher
// does PER-OP best-of-breed: exp/tanh -> xsimd, log -> Highway (math_dispatch.cc). This TU
// supplies the xsimd exp/tanh (and log, as a fallback when Highway isn't wired).
//
// Static single-target: compiled at -mavx2, so xsimd::batch<float> resolves to the 8-wide
// AVX2 register. Reached only when best_isa() >= AVX2 (gated in math_dispatch), so the
// AVX2 codegen never runs on an older CPU. 8-wide loop + scalar tail (libm).
#if defined(HYLE_SIMD_XSIMD_MATH)

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
void log_f32_xsimd_avx2(const float* src, float* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + kW <= n; i += kW)
    xsimd::log(batch::load_unaligned(src + i)).store_unaligned(dst + i);
  for (; i < n; ++i) dst[i] = std::log(src[i]);
}
void tanh_f32_xsimd_avx2(const float* src, float* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + kW <= n; i += kW)
    xsimd::tanh(batch::load_unaligned(src + i)).store_unaligned(dst + i);
  for (; i < n; ++i) dst[i] = std::tanh(src[i]);
}

}  // namespace hyle::simd::detail

#endif  // HYLE_SIMD_XSIMD_MATH
