#include "hyle/simd/math.h"

#include <cstddef>

#include "hyle/platform.h"
#include "hyle/simd/dispatch.h"
#include "src/simd/math/kernels.h"

// math_dispatch.cc -- baseline dispatcher for the transcendental batch ops, PER-OP
// best-of-breed across the wired vectorized backends. The same-ISA bench found NO single
// library wins all three ops at the same ~3.5-ULP class: xsimd's exp/tanh polynomials are
// fastest, Highway's log is fastest. So:
//   exp, tanh : xsimd > Highway > SLEEF > scalar
//   log       : Highway > xsimd > SLEEF > scalar
// All vectorized tiers need AVX2; all are header-only verified cores (借核不手搓). Baseline
// + LTO-off keeps any AVX2 codegen confined to the kernel TUs (same isolation as ops/dtype).

namespace hyle::simd {
namespace {

// AVX2 host? (every vectorized tier requires it). Compiled to false when none wired.
bool have_avx2() noexcept {
#if (defined(HYLE_SIMD_XSIMD_MATH) || defined(HYLE_SIMD_HIGHWAY_MATH) || \
     defined(HYLE_SIMD_SLEEF)) && defined(HYLE_ARCH_X64)
  return cpu_features().avx2;
#else
  return false;
#endif
}

}  // namespace

void exp_f32_batch(const float* src, float* dst, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
  static const bool hw = have_avx2();
  if (hw) {
#  if defined(HYLE_SIMD_XSIMD_MATH)
    detail::exp_f32_xsimd_avx2(src, dst, n); return;       // exp: xsimd fastest
#  elif defined(HYLE_SIMD_HIGHWAY_MATH)
    detail::exp_f32_highway_avx2(src, dst, n); return;
#  elif defined(HYLE_SIMD_SLEEF)
    detail::exp_f32_sleef_avx2(src, dst, n); return;
#  endif
  }
#endif
  detail::exp_f32_scalar(src, dst, n);
}
void log_f32_batch(const float* src, float* dst, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
  static const bool hw = have_avx2();
  if (hw) {
#  if defined(HYLE_SIMD_HIGHWAY_MATH)
    detail::log_f32_highway_avx2(src, dst, n); return;     // log: Highway fastest
#  elif defined(HYLE_SIMD_XSIMD_MATH)
    detail::log_f32_xsimd_avx2(src, dst, n); return;
#  elif defined(HYLE_SIMD_SLEEF)
    detail::log_f32_sleef_avx2(src, dst, n); return;
#  endif
  }
#endif
  detail::log_f32_scalar(src, dst, n);
}
void tanh_f32_batch(const float* src, float* dst, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
  static const bool hw = have_avx2();
  if (hw) {
#  if defined(HYLE_SIMD_XSIMD_MATH)
    detail::tanh_f32_xsimd_avx2(src, dst, n); return;      // tanh: xsimd fastest
#  elif defined(HYLE_SIMD_HIGHWAY_MATH)
    detail::tanh_f32_highway_avx2(src, dst, n); return;
#  elif defined(HYLE_SIMD_SLEEF)
    detail::tanh_f32_sleef_avx2(src, dst, n); return;
#  endif
  }
#endif
  detail::tanh_f32_scalar(src, dst, n);
}

Isa active_math_variant() noexcept {
  static const Isa tier = have_avx2() ? Isa::kAvx2 : Isa::kScalar;
  return tier;
}

}  // namespace hyle::simd
