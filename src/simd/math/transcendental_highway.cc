// Highway-accelerated AVX2 transcendentals (8-wide). Built ONLY when
// HYLE_SIMD_HIGHWAY_MATH is defined (CMake gates this TU on -DHYLE_SIMD_HIGHWAY_MATH=ON
// + -DHYLE_HIGHWAY_DIR=<highway>). BORROWS Highway's verified contrib/math cores
// (hn::Exp/Log/Tanh) -- HYLE does not hand-roll these (charter: 借验证核不手搓).
//
// WHY a second vectorized-math backend next to SLEEF: the same-ISA bench (math_bench)
// measured Highway's log ~1.85x FASTER than SLEEF _u35 at the SAME ~3.5-ULP accuracy
// class, and a tie on exp/tanh -- Highway's polynomial is simply better-tuned for this
// op set. Highway contrib/math is HEADER-ONLY (lighter than SLEEF's prebuilt lib), so
// when wired in the dispatcher PREFERS it (math_dispatch.cc: Highway > SLEEF > scalar).
//
// Static single-target dispatch: this TU is compiled at -mavx2, so HWY_STATIC_TARGET
// resolves to AVX2 and hn:: lowers to AVX2 intrinsics. Reached only when best_isa()
// >= AVX2 (gated in math_dispatch), so the AVX2 codegen never runs on an older CPU.
#if defined(HYLE_SIMD_HIGHWAY_MATH)

#include "src/simd/math/kernels.h"

#include <cmath>
#include <cstddef>

#include "hwy/highway.h"
#include "hwy/contrib/math/math-inl.h"

namespace hn = hwy::HWY_NAMESPACE;

namespace hyle::simd::detail {

void exp_f32_highway_avx2(const float* src, float* dst, std::size_t n) noexcept {
  const hn::ScalableTag<float> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w)
    hn::StoreU(hn::Exp(d, hn::LoadU(d, src + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = std::exp(src[i]);  // tail = libm (scalar oracle)
}
void log_f32_highway_avx2(const float* src, float* dst, std::size_t n) noexcept {
  const hn::ScalableTag<float> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w)
    hn::StoreU(hn::Log(d, hn::LoadU(d, src + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = std::log(src[i]);
}
void tanh_f32_highway_avx2(const float* src, float* dst, std::size_t n) noexcept {
  const hn::ScalableTag<float> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w)
    hn::StoreU(hn::Tanh(d, hn::LoadU(d, src + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = std::tanh(src[i]);
}

}  // namespace hyle::simd::detail

#endif  // HYLE_SIMD_HIGHWAY_MATH
