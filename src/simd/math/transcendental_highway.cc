// Highway-accelerated AVX2 transcendental (8-wide). BORROWS Highway's verified contrib/math
// core (hn::Log) -- HYLE does not hand-roll these (charter: 借验证核不手搓). The same-ISA
// bench (math_bench) found Highway's log fastest at the ~3.5-ULP class (xsimd wins exp/tanh),
// so per-op best-of-breed (math_dispatch.cc) uses Highway for LOG ONLY -- hence only log is
// defined here (exp/tanh would be dead code: xsimd serves those).
//
// x86-only: highway lowers to AVX2 intrinsics here, so the TU compiles to nothing on non-x86
// (math_dispatch falls to the scalar libm floor there). Header-only + vendored in third_party,
// so it can't be missing -- no opt-in define gates it, only the architecture does.
//
// Static single-target dispatch: this TU is compiled at -mavx2 (+ full Highway baseline via
// hyle_set_highway_isa, else HWY_STATIC_TARGET silently downgrades to 128-bit SSSE3), so hn::
// lowers to 256-bit AVX2. Reached only when cpu_features().avx2 (gated in math_dispatch), so
// the AVX2 codegen never runs on an older CPU.
#include "hyle/platform.h"  // HYLE_ARCH_X64 -- MUST precede the guard below

#if defined(HYLE_ARCH_X64)

#include "src/simd/math/kernels.h"

#include <cmath>
#include <cstddef>

#include "hwy/highway.h"
#include "hwy/contrib/math/math-inl.h"

namespace hn = hwy::HWY_NAMESPACE;

namespace hyle::simd::detail {

void log_f32_highway_avx2(const float* src, float* dst, std::size_t n) noexcept {
  const hn::ScalableTag<float> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w)
    hn::StoreU(hn::Log(d, hn::LoadU(d, src + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = std::log(src[i]);  // tail = libm (scalar oracle)
}

// simd-09: Highway wins sin/cos/log2/atan/cbrt/expm1/log1p/asinh/acosh at the same ISA
// (bench-decided, GCC+MSVC cross-checked; xsimd serves exp2/erf). Same borrow pattern as log.
#define HYLE_HWY_MATH_AVX2(name, hwyfn, libmfn)                                  \
  void name##_f32_highway_avx2(const float* src, float* dst, std::size_t n) noexcept { \
    const hn::ScalableTag<float> d;                                             \
    const std::size_t w = hn::Lanes(d);                                         \
    std::size_t i = 0;                                                          \
    for (; i + w <= n; i += w)                                                  \
      hn::StoreU(hn::hwyfn(d, hn::LoadU(d, src + i)), d, dst + i);               \
    for (; i < n; ++i) dst[i] = libmfn(src[i]);                                 \
  }
HYLE_HWY_MATH_AVX2(sin, Sin, std::sin)
HYLE_HWY_MATH_AVX2(cos, Cos, std::cos)
HYLE_HWY_MATH_AVX2(log2, Log2, std::log2)
HYLE_HWY_MATH_AVX2(atan, Atan, std::atan)
HYLE_HWY_MATH_AVX2(cbrt, Cbrt, std::cbrt)
HYLE_HWY_MATH_AVX2(expm1, Expm1, std::expm1)
HYLE_HWY_MATH_AVX2(log1p, Log1p, std::log1p)
HYLE_HWY_MATH_AVX2(asinh, Asinh, std::asinh)
HYLE_HWY_MATH_AVX2(acosh, Acosh, std::acosh)
#undef HYLE_HWY_MATH_AVX2

}  // namespace hyle::simd::detail

#endif  // HYLE_ARCH_X64
