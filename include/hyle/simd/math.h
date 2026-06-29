#pragma once

#include <cstddef>

#include "hyle/simd/dispatch.h"  // Isa, HYLE_SIMD_API

// math.h -- vectorized transcendentals (exp/log/tanh) for f32 (simd-06). The AI hot path
// (softmax/gelu/attention) is dominated by these; hand-rolling polynomial approximations
// is error-prone, so the accelerated path BORROWS SLEEF's verified cores (simd-design:
// "借 sleef 不手搓") rather than reimplementing them.
//
// TWO TIERS (same dispatch spirit as ops.h / dtype_ops.h):
//   - scalar libm reference: std::exp/log/tanh per element. ALWAYS compiled, correct on
//     every platform, the ULP oracle. NOT a stub -- it's the floor + the fallback.
//   - SLEEF AVX2 (8-wide): opt-in. Compiled only when the build is configured with
//     -DHYLE_SIMD_SLEEF=ON + -DHYLE_SLEEF_DIR=<prebuilt sleef> (links libsleef, includes
//     the PORTABLE declaration header sleef.h -- dual-platform, unlike SLEEF's GCC-only
//     INLINE headers). When present + the CPU has AVX2, the batch calls run SLEEF's
//     Sleef_expf8_u10avx2 etc. (1-ULP "_u10" accuracy).
//
// WHY opt-in (honest): vectorized transcendentals require a SLEEF build. Mandating it would
// force every consumer to vendor+build SLEEF; instead, like the absl/highway bench deps, it
// is enabled explicitly and HYLE consumes the prebuilt lib. Without it, math is correct via
// libm, just not vectorized. active_math_variant() reports which tier is live.

namespace hyle::simd {

HYLE_SIMD_API void exp_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void log_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void tanh_f32_batch(const float* src, float* dst, std::size_t n) noexcept;

// The tier the math batch ops selected: kAvx2 if SLEEF-accelerated AVX2 is live, else
// kScalar (libm). For tests/diagnostics.
HYLE_SIMD_API Isa active_math_variant() noexcept;

}  // namespace hyle::simd
