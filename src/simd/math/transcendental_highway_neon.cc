// Highway-accelerated NEON transcendentals (aarch64, 4-wide 128-bit). BORROWS Highway's
// verified contrib/math cores (hn::Exp/Log/Tanh) -- HYLE does not hand-roll these (charter:
// 借验证核不手搓). Highway lowers hn:: to NEON here (probed on Apple Silicon: target=NEON,
// Lanes=4, all three ops match libm at Highway's ~3.5-ULP class).
//
// Unlike the x86 side (which splits xsimd for exp/tanh + Highway for log, a per-op SPEED
// pick from the same-ISA bench), arm64 uses Highway for ALL THREE: one verified core, no
// second library pulled in unproven. A NEON xsimd core could be added later IF a NEON bench
// shows a per-op win -- but Highway-all-three is the honest, validated starting point.
//
// NEON is the ARMv8 baseline (no -march needed, no runtime CPU gate): math_dispatch enters
// this unconditionally on aarch64. scalar libm stays the tail (matches the oracle exactly).
#include "hyle/platform.h"  // HYLE_ARCH_ARM64 -- MUST precede the guard below

#if defined(HYLE_ARCH_ARM64)

#include "src/simd/math/kernels.h"

#include <cmath>
#include <cstddef>

#include "hwy/highway.h"
#include "hwy/contrib/math/math-inl.h"

namespace hn = hwy::HWY_NAMESPACE;

namespace hyle::simd::detail {

void exp_f32_highway_neon(const float* src, float* dst, std::size_t n) noexcept {
  const hn::ScalableTag<float> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w)
    hn::StoreU(hn::Exp(d, hn::LoadU(d, src + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = std::exp(src[i]);  // tail = libm (scalar oracle)
}

void log_f32_highway_neon(const float* src, float* dst, std::size_t n) noexcept {
  const hn::ScalableTag<float> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w)
    hn::StoreU(hn::Log(d, hn::LoadU(d, src + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = std::log(src[i]);  // tail = libm (scalar oracle)
}

void tanh_f32_highway_neon(const float* src, float* dst, std::size_t n) noexcept {
  const hn::ScalableTag<float> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w)
    hn::StoreU(hn::Tanh(d, hn::LoadU(d, src + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = std::tanh(src[i]);  // tail = libm (scalar oracle)
}

}  // namespace hyle::simd::detail

#endif  // HYLE_ARCH_ARM64
