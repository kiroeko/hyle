#include "hyle/simd/math.h"

#include <cstddef>

#include "hyle/platform.h"
#include "hyle/simd/dispatch.h"
#include "src/simd/math/kernels.h"

// math_dispatch.cc -- baseline dispatcher for the transcendental batch ops, PER-OP
// best-of-breed across the two vendored header-only backends. The same-ISA bench found NO
// single library wins all three ops at the same ~3.5-ULP class: xsimd's exp/tanh
// polynomials are fastest, Highway's log is fastest. So (fixed at compile time, not
// machine-dependent):
//   exp, tanh : xsimd  (fastest)
//   log       : Highway (fastest)
// Both backends are vendored in-tree (can't be missing) + always compiled on x86; the AVX2
// codegen is confined to the kernel TUs (baseline + LTO-off, same isolation as ops/dtype).
// This dispatcher only enters them when HYLE_ARCH_X64 && cpu_features().avx2 -> a non-x86
// build or an old x86 CPU falls to the scalar libm floor (no SIGILL).

namespace hyle::simd {
namespace {

// AVX2 x86 host? (the borrow cores are x86 AVX2-only). On non-x86 this is always false.
bool have_avx2() noexcept {
#if defined(HYLE_ARCH_X64)
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
    detail::exp_f32_xsimd_avx2(src, dst, n); return;       // exp: xsimd fastest
  }
#endif
  detail::exp_f32_scalar(src, dst, n);
}
void log_f32_batch(const float* src, float* dst, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
  static const bool hw = have_avx2();
  if (hw) {
    detail::log_f32_highway_avx2(src, dst, n); return;     // log: Highway fastest
  }
#endif
  detail::log_f32_scalar(src, dst, n);
}
void tanh_f32_batch(const float* src, float* dst, std::size_t n) noexcept {
#if defined(HYLE_ARCH_X64)
  static const bool hw = have_avx2();
  if (hw) {
    detail::tanh_f32_xsimd_avx2(src, dst, n); return;      // tanh: xsimd fastest
  }
#endif
  detail::tanh_f32_scalar(src, dst, n);
}

Isa active_math_variant() noexcept {
  static const Isa tier = have_avx2() ? Isa::kAvx2 : Isa::kScalar;
  return tier;
}

}  // namespace hyle::simd
