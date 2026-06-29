#include "hyle/simd/math.h"

#include <cstddef>

#include "hyle/platform.h"
#include "hyle/simd/dispatch.h"
#include "src/simd/math/kernels.h"

// math_dispatch.cc -- baseline dispatcher for the transcendental batch ops. Picks the
// SLEEF AVX2 kernel when (a) HYLE was built with SLEEF wired in AND (b) the CPU has AVX2;
// else the scalar libm kernel. Baseline + LTO-off keeps any AVX2 codegen confined to the
// SLEEF TU (same isolation rule as ops/dtype).

namespace hyle::simd {
namespace {

bool use_sleef_avx2() noexcept {
#if defined(HYLE_SIMD_SLEEF) && defined(HYLE_ARCH_X64)
  return cpu_features().avx2;  // SLEEF avx2 cores need AVX2 (+FMA, implied by our tier)
#else
  return false;
#endif
}

}  // namespace

void exp_f32_batch(const float* src, float* dst, std::size_t n) noexcept {
#if defined(HYLE_SIMD_SLEEF) && defined(HYLE_ARCH_X64)
  static const bool hw = use_sleef_avx2();
  if (hw) { detail::exp_f32_sleef_avx2(src, dst, n); return; }
#endif
  detail::exp_f32_scalar(src, dst, n);
}
void log_f32_batch(const float* src, float* dst, std::size_t n) noexcept {
#if defined(HYLE_SIMD_SLEEF) && defined(HYLE_ARCH_X64)
  static const bool hw = use_sleef_avx2();
  if (hw) { detail::log_f32_sleef_avx2(src, dst, n); return; }
#endif
  detail::log_f32_scalar(src, dst, n);
}
void tanh_f32_batch(const float* src, float* dst, std::size_t n) noexcept {
#if defined(HYLE_SIMD_SLEEF) && defined(HYLE_ARCH_X64)
  static const bool hw = use_sleef_avx2();
  if (hw) { detail::tanh_f32_sleef_avx2(src, dst, n); return; }
#endif
  detail::tanh_f32_scalar(src, dst, n);
}

Isa active_math_variant() noexcept {
  static const Isa tier = use_sleef_avx2() ? Isa::kAvx2 : Isa::kScalar;
  return tier;
}

}  // namespace hyle::simd
