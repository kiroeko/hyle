// xsimd-accelerated AVX-512 transcendentals (16-wide). The 512-bit sibling of
// transcendental_xsimd.cc -- SAME borrowed xsimd cores (xsimd::exp/tanh), the ONLY
// difference is this TU is compiled with the AVX-512 flag set so xsimd::batch<float>
// lowers to the 16-lane zmm register instead of the 8-lane ymm. Per-op best-of-breed is
// unchanged (xsimd serves exp/tanh; Highway serves log), this just adds the wider tier.
//
// This is item ③ of the perf "遗留优化方向" table: the 256-bit math already ties the two
// vendor SOTA libs; AVX-512 doubles the lane count (8->16) so throughput should ~2x on a
// 512-capable server. The dispatcher (math_dispatch.cc) enters this only when the host has
// the AVX-512 group (best_isa() >= kAvx512), else it stays on the AVX2 cores.
//
// STATUS (honest, same bar as the other AVX-512 TUs): written + compiled (GCC14
// objdump-verified to emit zmm: 354 zmm ops in a standalone probe) + NOT executed -- this
// dev box is AVX2-only (no AVX-512 hardware, no Intel SDE here), so execution-correctness is
// PENDING (存疑 until an AVX-512 machine or SDE runs it). The dispatcher routes an AVX-512
// host here by default, so a real such CPU runs this unvalidated path -- flagged in
// docs/architecture/simd-design.md. math_test cross-checks the result against the scalar
// libm reference, so on a real AVX-512 box that test validates this path with no new code.
//
// xsimd needs -mavx512cd alongside f/bw/vl/dq (its avx512dq kernels assume cd); the
// hyle_set_avx512_math_isa CMake flag set provides the full AVX-512 group + baseline.
#include "hyle/platform.h"  // HYLE_ARCH_X64 -- MUST precede the guard below

#if defined(HYLE_ARCH_X64)

#include "src/simd/math/kernels.h"

#include <cmath>
#include <cstddef>

#include "xsimd/xsimd.hpp"

namespace hyle::simd::detail {

namespace {
using batch = xsimd::batch<float>;
constexpr std::size_t kW = batch::size;  // 16 at AVX-512
}  // namespace

void exp_f32_xsimd_avx512(const float* src, float* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + kW <= n; i += kW)
    xsimd::exp(batch::load_unaligned(src + i)).store_unaligned(dst + i);
  for (; i < n; ++i) dst[i] = std::exp(src[i]);  // tail = libm (scalar oracle)
}
void tanh_f32_xsimd_avx512(const float* src, float* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + kW <= n; i += kW)
    xsimd::tanh(batch::load_unaligned(src + i)).store_unaligned(dst + i);
  for (; i < n; ++i) dst[i] = std::tanh(src[i]);
}

// simd-09 16-wide siblings (xsimd wins exp2/erf).
void exp2_f32_xsimd_avx512(const float* src, float* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + kW <= n; i += kW)
    xsimd::exp2(batch::load_unaligned(src + i)).store_unaligned(dst + i);
  for (; i < n; ++i) dst[i] = std::exp2(src[i]);
}
void erf_f32_xsimd_avx512(const float* src, float* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + kW <= n; i += kW)
    xsimd::erf(batch::load_unaligned(src + i)).store_unaligned(dst + i);
  for (; i < n; ++i) dst[i] = std::erf(src[i]);
}

// xsimd-only borrow, 16-wide sibling (tgamma only; lgamma stays scalar -- vendor lane-mixing bug).
void tgamma_f32_xsimd_avx512(const float* src, float* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + kW <= n; i += kW)
    xsimd::tgamma(batch::load_unaligned(src + i)).store_unaligned(dst + i);
  for (; i < n; ++i) dst[i] = std::tgamma(src[i]);
}

}  // namespace hyle::simd::detail

#endif  // HYLE_ARCH_X64
