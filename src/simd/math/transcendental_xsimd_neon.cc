// xsimd-accelerated NEON transcendentals (aarch64, 4-wide 128-bit) -- the FIRST xsimd core on
// arm64. The other NEON math ops borrow Highway's contrib/math (transcendental_highway_neon.cc,
// one verified core for all of exp/log/sin/...); but Highway has NO gamma function, so tgamma
// can't use that path. xsimd DOES have it, and xsimd::batch<float> lowers to NEON here (verified
// under qemu-aarch64: batch::size=4, tgamma matches libm). So tgamma borrows xsimd on arm64 --
// not a best-of pick (no Highway rival exists), just the one library that provides gamma.
//
// LGAMMA is NOT here: xsimd's vectorized lgamma has a lane-mixing bug (reproduced on NEON too via
// qemu), so it stays on the scalar libm reference. tgamma is unaffected. See
// docs/debug/xsimd-lgamma-lane-mixing-hunt.md.
//
// NEON is the ARMv8 baseline (no -march, no runtime CPU gate): math_dispatch enters this
// unconditionally on aarch64. scalar libm stays the tail (matches the oracle exactly).
#include "hyle/platform.h"  // HYLE_ARCH_ARM64 -- MUST precede the guard below

#if defined(HYLE_ARCH_ARM64)

#include "src/simd/math/kernels.h"

#include <cmath>
#include <cstddef>

#include "xsimd/xsimd.hpp"
#include "src/simd/math/complex_hyle-inl.h"

namespace hyle::simd::detail {

namespace {
using batch = xsimd::batch<float>;
constexpr std::size_t kW = batch::size;  // 4 at NEON
}  // namespace

void tgamma_f32_xsimd_neon(const float* src, float* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + kW <= n; i += kW)
    xsimd::tgamma(batch::load_unaligned(src + i)).store_unaligned(dst + i);
  for (; i < n; ++i) dst[i] = std::tgamma(src[i]);
}
void hypot_fast_f32_xsimd_neon(const float* a, const float* b, float* dst,
                               std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + kW <= n; i += kW)
    xsimd::hypot(batch::load_unaligned(a + i), batch::load_unaligned(b + i))
        .store_unaligned(dst + i);
  for (; i < n; ++i) dst[i] = std::sqrt(a[i] * a[i] + b[i] * b[i]);
}
void csin_f32_hyle_neon(const float* sr, const float* si, float* dr, float* di,
                         std::size_t n) noexcept {
  complex_sincos_hyle<false>(sr, si, dr, di, n);
}
void ccos_f32_hyle_neon(const float* sr, const float* si, float* dr, float* di,
                         std::size_t n) noexcept {
  complex_sincos_hyle<true>(sr, si, dr, di, n);
}
void ctanh_f32_hyle_neon(const float* sr, const float* si, float* dr, float* di,
                          std::size_t n) noexcept {
  complex_ctanh_hyle(sr, si, dr, di, n);
}
void cpow_f32_hyle_neon(const float* zr, const float* zi, const float* wr, const float* wi,
                         float* dr, float* di, std::size_t n) noexcept {
  complex_cpow_hyle(zr, zi, wr, wi, dr, di, n);
}
}  // namespace hyle::simd::detail

#endif  // HYLE_ARCH_ARM64
