#include "hyle/simd/math.h"

#include <cstddef>

#include "hyle/platform.h"
#include "hyle/simd/dispatch.h"
#include "src/simd/math/kernels.h"

// math_dispatch.cc -- baseline dispatcher for the transcendental batch ops. Tier order:
// Highway-math AVX2 (if wired) > SLEEF AVX2 (if wired) > scalar libm. Both vectorized
// tiers need AVX2; Highway is PREFERRED when both are present (same-ISA bench: Highway
// log ~1.85x faster than SLEEF _u35, ties exp/tanh, and is header-only). Baseline +
// LTO-off keeps any AVX2 codegen confined to the kernel TUs (same isolation as ops/dtype).

namespace hyle::simd {
namespace {

// AVX2 host? (both vectorized tiers require it). Compiled to false when neither tier wired.
bool have_avx2() noexcept {
#if (defined(HYLE_SIMD_HIGHWAY_MATH) || defined(HYLE_SIMD_SLEEF)) && defined(HYLE_ARCH_X64)
  return cpu_features().avx2;
#else
  return false;
#endif
}

}  // namespace

void exp_f32_batch(const float* src, float* dst, std::size_t n) noexcept {
#if defined(HYLE_SIMD_HIGHWAY_MATH) && defined(HYLE_ARCH_X64)
  static const bool hw = have_avx2();
  if (hw) { detail::exp_f32_highway_avx2(src, dst, n); return; }
#elif defined(HYLE_SIMD_SLEEF) && defined(HYLE_ARCH_X64)
  static const bool hw = have_avx2();
  if (hw) { detail::exp_f32_sleef_avx2(src, dst, n); return; }
#endif
  detail::exp_f32_scalar(src, dst, n);
}
void log_f32_batch(const float* src, float* dst, std::size_t n) noexcept {
#if defined(HYLE_SIMD_HIGHWAY_MATH) && defined(HYLE_ARCH_X64)
  static const bool hw = have_avx2();
  if (hw) { detail::log_f32_highway_avx2(src, dst, n); return; }
#elif defined(HYLE_SIMD_SLEEF) && defined(HYLE_ARCH_X64)
  static const bool hw = have_avx2();
  if (hw) { detail::log_f32_sleef_avx2(src, dst, n); return; }
#endif
  detail::log_f32_scalar(src, dst, n);
}
void tanh_f32_batch(const float* src, float* dst, std::size_t n) noexcept {
#if defined(HYLE_SIMD_HIGHWAY_MATH) && defined(HYLE_ARCH_X64)
  static const bool hw = have_avx2();
  if (hw) { detail::tanh_f32_highway_avx2(src, dst, n); return; }
#elif defined(HYLE_SIMD_SLEEF) && defined(HYLE_ARCH_X64)
  static const bool hw = have_avx2();
  if (hw) { detail::tanh_f32_sleef_avx2(src, dst, n); return; }
#endif
  detail::tanh_f32_scalar(src, dst, n);
}

Isa active_math_variant() noexcept {
  static const Isa tier = have_avx2() ? Isa::kAvx2 : Isa::kScalar;
  return tier;
}

}  // namespace hyle::simd
