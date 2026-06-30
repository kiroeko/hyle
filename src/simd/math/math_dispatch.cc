#include "hyle/simd/math.h"

#include <cstddef>

#include "hyle/platform.h"
#include "hyle/simd/dispatch.h"
#include "src/simd/math/kernels.h"

// math_dispatch.cc -- baseline dispatcher for the transcendental batch ops, PER-OP
// best-of-breed across the two vendored header-only backends. The same-ISA bench found NO
// single library wins all three ops at the same ~3.5-ULP class: xsimd's exp/tanh
// polynomials are fastest, Highway's log is fastest. So:
//   exp, tanh : xsimd > Highway > scalar
//   log       : Highway > xsimd > scalar
// Both backends are vendored + wired UNCONDITIONALLY on x86 (CMake). AVX2 codegen is
// confined to the kernel TUs (baseline + LTO-off, same isolation as ops/dtype); this
// dispatcher only enters them when cpu_features().avx2 -> old CPU falls to scalar (no SIGILL).

namespace hyle::simd {
namespace {

// AVX2 host? (the vectorized backends require it). On x86 the backends are always wired.
bool have_avx2() noexcept {
#if (defined(HYLE_SIMD_XSIMD_MATH) || defined(HYLE_SIMD_HIGHWAY_MATH)) && defined(HYLE_ARCH_X64)
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
