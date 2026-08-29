#pragma once

#include <cstddef>

#include "hyle/simd/dispatch.h"  // Isa, HYLE_SIMD_API

// Whole-buffer real/complex mathematics and model-independent activations.
//
// Ordinary callers use pointer/count operations and never depend on a vendor vector type.
// n may be any size, including zero. Exact src == dst aliasing is supported for unary
// operations; otherwise input and output ranges must be disjoint unless a declaration below
// says more. The ordinary functions preserve the corresponding std::<function> domain and
// special-value semantics.
//
// The accelerated implementation directly uses the centrally locked Highway math core and
// retains only HYLE-specific compositions or capabilities that Highway does not provide.
// Operations outside an upstream-qualified domain fall back to scalar libm by lane/block so
// the ordinary contract does not become a vendor precondition. Runtime selection chooses a
// compiled AVX2/AVX-512, NEON/SVE or RVV tier once; scalar libm is the always-available floor.
// runtime().math.isa reports the active broad tier. Backend choice is an implementation detail,
// not a stable ABI or a promise to preserve a historical benchmark winner.

namespace hyle::simd {

HYLE_SIMD_API void exp_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void log_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void tanh_f32_batch(const float* src, float* dst, std::size_t n) noexcept;

// Explicit narrow-domain fast paths: every exp lane must be finite in [-80, 80]; every log lane
// must be finite, normal and positive (>= FLT_MIN). They skip ordinary-domain recovery. Exact
// src == dst aliasing is supported.
HYLE_SIMD_API void exp_in_range_f32_batch(const float* src, float* dst,
                                          std::size_t n) noexcept;
HYLE_SIMD_API void log_in_range_f32_batch(const float* src, float* dst,
                                          std::size_t n) noexcept;

// f64 counterparts serve double-precision solver, geometry and numeric workloads.
HYLE_SIMD_API void exp_f64_batch(const double* src, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void log_f64_batch(const double* src, double* dst, std::size_t n) noexcept;
// Explicit f64 narrow-domain fast paths: every exp lane must be finite in [-700, 700]; every log
// lane must be finite, normal and positive (>= DBL_MIN). They omit ordinary-domain recovery and
// therefore must not be used on untrusted/general values. Exact src == dst aliasing is supported.
HYLE_SIMD_API void exp_in_range_f64_batch(const double* src, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void log_in_range_f64_batch(const double* src, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void tanh_f64_batch(const double* src, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void sin_f64_batch(const double* src, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void cos_f64_batch(const double* src, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void exp2_f64_batch(const double* src, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void log2_f64_batch(const double* src, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void erf_f64_batch(const double* src, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void atan_f64_batch(const double* src, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void cbrt_f64_batch(const double* src, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void expm1_f64_batch(const double* src, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void log1p_f64_batch(const double* src, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void asinh_f64_batch(const double* src, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void acosh_f64_batch(const double* src, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void acos_f64_batch(const double* src, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void asin_f64_batch(const double* src, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void sinh_f64_batch(const double* src, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void cosh_f64_batch(const double* src, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void tan_f64_batch(const double* src, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void atanh_f64_batch(const double* src, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void log10_f64_batch(const double* src, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void atan2_f64_batch(const double* a, const double* b, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void pow_f64_batch(const double* a, const double* b, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void sincos_f64_batch(const double* src, double* s, double* c, std::size_t n) noexcept;
HYLE_SIMD_API void tgamma_f64_batch(const double* src, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void lgamma_f64_batch(const double* src, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void hypot_f64_batch(const double* a, const double* b, double* dst, std::size_t n) noexcept;

// Additional f32 real functions use the same ordinary contract and target selection.
HYLE_SIMD_API void sin_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void cos_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void exp2_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void log2_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void erf_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void atan_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void cbrt_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void expm1_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void log1p_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void asinh_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void acosh_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void acos_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void asin_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void sinh_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void cosh_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void tan_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void atanh_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void log10_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void atan2_f32_batch(const float* a, const float* b, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void pow_f32_batch(const float* a, const float* b, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void sincos_f32_batch(const float* src, float* s, float* c, std::size_t n) noexcept;
HYLE_SIMD_API void erfc_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void exp10_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void fmod_f32_batch(const float* a, const float* b, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void remainder_f32_batch(const float* a, const float* b, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void nextafter_f32_batch(const float* a, const float* b, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void fdim_f32_batch(const float* a, const float* b, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void erfc_f64_batch(const double* src, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void exp10_f64_batch(const double* src, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void fmod_f64_batch(const double* a, const double* b, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void remainder_f64_batch(const double* a, const double* b, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void nextafter_f64_batch(const double* a, const double* b, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void fdim_f64_batch(const double* a, const double* b, double* dst, std::size_t n) noexcept;

// Gamma functions use the locked Highway implementation on its qualified domain and scalar
// libm recovery elsewhere.
HYLE_SIMD_API void lgamma_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void tgamma_f32_batch(const float* src, float* dst, std::size_t n) noexcept;

// Two explicit hypot contracts; neither silently substitutes for the other:
// - hypot_f32_batch: full-range std::hypot semantics via exponent scaling.
// - hypot_in_range_f32_batch: direct square/add/sqrt (FMA may be fused), ~2x faster, but callers must
//   keep the intermediate sum representable and accept loss of underflow protection. No per-lane
//   checks. dst may exactly alias a or b; other overlap is outside the contract. See usage-guide.md.
//   ("InRange" == the caller-guarantees-narrower-domain fast-path convention, same as Highway's
//   DemoteToNearestIntInRange / HYLE's Convert/Demote/PromoteInRangeTo.)
HYLE_SIMD_API void hypot_f32_batch(const float* a, const float* b, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void hypot_in_range_f32_batch(const float* a, const float* b, float* dst, std::size_t n) noexcept;

// AI activation COMBINED cores (simd-design "AI 数学函数"). Unlike the per-op transcendentals
// above (each borrows one vendor polynomial), these COMPOSE the vectorized cores into the shapes
// the AI hot path needs -- HYLE-specific, not a single vendor op. They call the batch ops above
// (which self-dispatch to the best tier), so no new per-tier wiring.
//   gelu(x)      = 0.5 x (1 + erf(x/sqrt2))            -- exact Gaussian Error Linear Unit
//   gelu_tanh(x) = 0.5 x (1 + tanh(sqrt(2/pi) x (1 + 0.044715 x^2)))  -- the tanh approximation
//                  (matches ggml_gelu_f32; slightly cheaper, ~1e-3 off exact -- the common LLM form)
//   sigmoid(x)   = 1/(1 + exp(-x))                     -- logistic
//   silu(x)      = x * sigmoid(x)                      -- SiLU/swish (Llama FFN gate)
// All element-wise: dst[i] = f(src[i]). n arbitrary.
HYLE_SIMD_API void gelu_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void gelu_tanh_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void sigmoid_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void silu_f32_batch(const float* src, float* dst, std::size_t n) noexcept;

// softmax(x)[i] = exp(x[i] - max) / sum_j exp(x[j] - max)  -- IN PLACE.
// Fused multipass: max → one vectorized exp (sum accumulated same chunks) → scale.
// n == 0 is a no-op.
HYLE_SIMD_API void softmax_f32(float* data, std::size_t n) noexcept;
// log_softmax(x)[i] = (x[i] - max) - log(sum_j exp(x[j] - max))  -- in place, numerically stable.
// max + sum-exp for logZ, then single subtract of (m + log s).
HYLE_SIMD_API void log_softmax_f32(float* data, std::size_t n) noexcept;

// ---- COMPLEX transcendentals --------------------------------------------------------------
// Highway has no complex math surface. HYLE composes the locked Highway real primitives on
// AVX2/AVX-512, NEON/SVE and RVV; shared real work removes redundant operations. Their
// operation-specific exceptional lanes use std::complex; ctanh/cpow specifically isolate axes,
// subnormal, non-finite, branch-cut and overflow-risk lanes without scalarizing regular neighbors.
//
// STORAGE = SoA (planar): real and imaginary parts live in separate arrays. Exact in-place use
// (dst_re==src_re and dst_im==src_im) is supported; arbitrary offset overlap is not. n arbitrary.
//   cexp/clog/csqrt : the workhorses (FFT twiddle, transfer functions, roots)
//   csin/ccos/ctan + csinh/ccosh/ctanh : complex trig and hyperbolic families
//   cpow : z^w (both complex)
HYLE_SIMD_API void cexp_f32_batch(const float* src_re, const float* src_im, float* dst_re, float* dst_im, std::size_t n) noexcept;
HYLE_SIMD_API void clog_f32_batch(const float* src_re, const float* src_im, float* dst_re, float* dst_im, std::size_t n) noexcept;
HYLE_SIMD_API void csqrt_f32_batch(const float* src_re, const float* src_im, float* dst_re, float* dst_im, std::size_t n) noexcept;
HYLE_SIMD_API void csin_f32_batch(const float* src_re, const float* src_im, float* dst_re, float* dst_im, std::size_t n) noexcept;
HYLE_SIMD_API void ccos_f32_batch(const float* src_re, const float* src_im, float* dst_re, float* dst_im, std::size_t n) noexcept;
HYLE_SIMD_API void ctan_f32_batch(const float* src_re, const float* src_im, float* dst_re, float* dst_im, std::size_t n) noexcept;
HYLE_SIMD_API void csinh_f32_batch(const float* src_re, const float* src_im, float* dst_re, float* dst_im, std::size_t n) noexcept;
HYLE_SIMD_API void ccosh_f32_batch(const float* src_re, const float* src_im, float* dst_re, float* dst_im, std::size_t n) noexcept;
HYLE_SIMD_API void ctanh_f32_batch(const float* src_re, const float* src_im, float* dst_re, float* dst_im, std::size_t n) noexcept;
// cpow(z, w): both operands complex. Four input planes, two output planes.
HYLE_SIMD_API void cpow_f32_batch(const float* z_re, const float* z_im, const float* w_re, const float* w_im, float* dst_re, float* dst_im, std::size_t n) noexcept;

HYLE_SIMD_API void cexp_f64_batch(const double* src_re, const double* src_im, double* dst_re, double* dst_im, std::size_t n) noexcept;
HYLE_SIMD_API void clog_f64_batch(const double* src_re, const double* src_im, double* dst_re, double* dst_im, std::size_t n) noexcept;
HYLE_SIMD_API void csqrt_f64_batch(const double* src_re, const double* src_im, double* dst_re, double* dst_im, std::size_t n) noexcept;
HYLE_SIMD_API void csin_f64_batch(const double* src_re, const double* src_im, double* dst_re, double* dst_im, std::size_t n) noexcept;
HYLE_SIMD_API void ccos_f64_batch(const double* src_re, const double* src_im, double* dst_re, double* dst_im, std::size_t n) noexcept;
HYLE_SIMD_API void ctan_f64_batch(const double* src_re, const double* src_im, double* dst_re, double* dst_im, std::size_t n) noexcept;
HYLE_SIMD_API void csinh_f64_batch(const double* src_re, const double* src_im, double* dst_re, double* dst_im, std::size_t n) noexcept;
HYLE_SIMD_API void ccosh_f64_batch(const double* src_re, const double* src_im, double* dst_re, double* dst_im, std::size_t n) noexcept;
HYLE_SIMD_API void ctanh_f64_batch(const double* src_re, const double* src_im, double* dst_re, double* dst_im, std::size_t n) noexcept;
HYLE_SIMD_API void cpow_f64_batch(const double* z_re, const double* z_im, const double* w_re, const double* w_im, double* dst_re, double* dst_im, std::size_t n) noexcept;

}  // namespace hyle::simd
