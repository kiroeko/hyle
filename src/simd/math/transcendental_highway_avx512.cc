// Highway-accelerated AVX-512 transcendental (16-wide). The 512-bit sibling of
// transcendental_highway.cc -- SAME borrowed Highway contrib/math core (hn::Log), the ONLY
// difference is this TU is compiled with the AVX-512 baseline so HWY_STATIC_TARGET resolves
// to AVX3 (zmm, Lanes=16) instead of AVX2 (ymm, Lanes=8). Per-op best-of-breed is unchanged
// (Highway serves log; xsimd serves exp/tanh), this just adds the wider tier.
//
// This is item ③ of the perf "遗留优化方向" table (the log half): the 256-bit log already
// ties Highway's own core; AVX-512 doubles the lane count so throughput should ~2x on a
// 512-capable server. The dispatcher (math_dispatch.cc) enters this only when the host has
// the AVX-512 group (best_isa() >= kAvx512), else it stays on the AVX2 core.
//
// STATUS (honest, same bar as the other AVX-512 TUs): written + compiled (GCC14
// objdump-verified to emit zmm: 299 zmm ops in a standalone probe) + NOT executed -- this
// dev box is AVX2-only (no AVX-512 hardware, no Intel SDE here), so execution-correctness is
// PENDING (存疑 until an AVX-512 machine or SDE runs it). math_test cross-checks against the
// scalar libm reference, so on a real AVX-512 box that test validates this path, no new code.
//
// Static single-target dispatch: like transcendental_highway.cc, Highway's HWY_STATIC_TARGET
// must see the FULL baseline macro chain to pick AVX3: the AVX2 baseline (SSE4=PCLMUL+AES +
// BMI2+FMA+F16C) PLUS the AVX-512 group (f/bw/vl/dq/cd). The hyle_set_avx512_math_isa CMake
// flag set supplies all of it, so hn:: lowers to 512-bit on GCC/Clang (objdump: 299 zmm).
//
// !!! MSVC EXCLUSION (probe-found, not a wiring bug): Highway marks its ENTIRE AVX3 target
// HWY_BROKEN on MSVC (detect_targets.h:192 HWY_BROKEN_MSVC -- MSVC's AVX-512 codegen has a
// known compiler bug, https://github.com/Mysticial/Flops/issues/16). So under /arch:AVX512
// Highway SILENTLY stays at AVX2 and hn:: emits ymm, not zmm (dumpbin: 250 ymm / 0 zmm).
// There is therefore NO 512-bit log win on MSVC -- the avx2 highway log core is already the
// widest Highway can give there. We exclude this TU on MSVC (the symbol is not defined) and
// math_dispatch keeps MSVC's AVX-512 log on the avx2 core, rather than route to a fake-512
// path. xsimd (exp/tanh) is NOT affected -- its 512 batch is plain AVX-512F, MSVC emits zmm.
#include "hyle/platform.h"  // HYLE_ARCH_X64 -- MUST precede the guard below

#if defined(HYLE_ARCH_X64) && !defined(_MSC_VER)

#include "src/simd/math/kernels.h"

#include <cmath>
#include <cstddef>

#include "hwy/highway.h"
#include "hwy/contrib/math/math-inl.h"

namespace hn = hwy::HWY_NAMESPACE;

namespace hyle::simd::detail {

void log_f32_highway_avx512(const float* src, float* dst, std::size_t n) noexcept {
  const hn::ScalableTag<float> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w)
    hn::StoreU(hn::Log(d, hn::LoadU(d, src + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = std::log(src[i]);  // tail = libm (scalar oracle)
}

// simd-09 16-wide Highway siblings (GCC/Clang only; MSVC excluded above). Same MSVC-AVX3-broken
// caveat as log. objdump-verified zmm PENDING execution on real AVX-512 (math_test validates).
#define HYLE_HWY_MATH_512(name, hwyfn, libmfn)                                  \
  void name##_f32_highway_avx512(const float* src, float* dst, std::size_t n) noexcept { \
    const hn::ScalableTag<float> d;                                            \
    const std::size_t w = hn::Lanes(d);                                        \
    std::size_t i = 0;                                                         \
    for (; i + w <= n; i += w)                                                 \
      hn::StoreU(hn::hwyfn(d, hn::LoadU(d, src + i)), d, dst + i);              \
    for (; i < n; ++i) dst[i] = libmfn(src[i]);                                \
  }
HYLE_HWY_MATH_512(sin, Sin, std::sin)
HYLE_HWY_MATH_512(cos, Cos, std::cos)
HYLE_HWY_MATH_512(log2, Log2, std::log2)
HYLE_HWY_MATH_512(atan, Atan, std::atan)
HYLE_HWY_MATH_512(cbrt, Cbrt, std::cbrt)
HYLE_HWY_MATH_512(expm1, Expm1, std::expm1)
HYLE_HWY_MATH_512(log1p, Log1p, std::log1p)
HYLE_HWY_MATH_512(asinh, Asinh, std::asinh)
HYLE_HWY_MATH_512(acosh, Acosh, std::acosh)
#undef HYLE_HWY_MATH_512

}  // namespace hyle::simd::detail

#endif  // HYLE_ARCH_X64
