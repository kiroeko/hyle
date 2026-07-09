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

}  // namespace hyle::simd::detail

#endif  // HYLE_ARCH_ARM64
