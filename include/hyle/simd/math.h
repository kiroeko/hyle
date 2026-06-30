#pragma once

#include <cstddef>

#include "hyle/simd/dispatch.h"  // Isa, HYLE_SIMD_API

// math.h -- vectorized transcendentals (exp/log/tanh) for f32 (simd-06). The AI hot path
// (softmax/gelu/attention) is dominated by these; hand-rolling polynomial approximations
// is error-prone, so the accelerated paths BORROW verified vendor cores (simd-design:
// "借核不手搓") rather than reimplementing them.
//
// TIERS + PER-OP BEST-OF-BREED dispatch (math_dispatch.cc):
//   - scalar libm reference: std::exp/log/tanh per element. ALWAYS compiled, the ULP
//     oracle + fallback. NOT a stub.
//   - vectorized AVX2 cores (8-wide, all ~3.5-ULP class, all opt-in, all header-only
//     except SLEEF): Highway contrib/math (-DHYLE_SIMD_HIGHWAY_MATH), xsimd
//     (-DHYLE_SIMD_XSIMD_MATH), SLEEF _u35 (-DHYLE_SIMD_SLEEF, prebuilt lib).
//   - the same-ISA bench found NO single library wins all three ops: xsimd's exp/tanh
//     polynomials are fastest, Highway's log is fastest. So the dispatcher picks PER OP:
//     exp/tanh -> xsimd > Highway > SLEEF; log -> Highway > xsimd > SLEEF (then scalar).
//
// BEST-OF-BREED needs BOTH Highway+xsimd wired (each op then routes to the faster of the
// two). Wiring only one is a SUBOPTIMAL trap: that lib serves all three ops and loses on
// the ones it isn't fastest at. Default build (none wired) = scalar libm. active_math_
// variant() reports whether a vectorized tier is live (kAvx2) or scalar.

namespace hyle::simd {

HYLE_SIMD_API void exp_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void log_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void tanh_f32_batch(const float* src, float* dst, std::size_t n) noexcept;

// The tier the math batch ops selected: kAvx2 if SLEEF-accelerated AVX2 is live, else
// kScalar (libm). For tests/diagnostics.
HYLE_SIMD_API Isa active_math_variant() noexcept;

}  // namespace hyle::simd
