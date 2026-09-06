#pragma once

#include <cstddef>

#include "hyle/simd/types.h"
#include "hyle/simd/visibility.h"

// Whole-buffer real/complex mathematics and model-independent activations.
//
// Ordinary callers use pointer/count operations and never depend on a vendor
// vector type. n may be any size, including zero. Exact src == dst aliasing is
// supported for unary operations; otherwise input and output ranges must be
// disjoint unless a declaration below says more. The ordinary functions
// preserve the corresponding std::<function> domain and special-value
// semantics.
//
// The accelerated implementation directly uses the centrally locked Highway
// math core and retains only HYLE-specific compositions or capabilities that
// Highway does not provide. Operations outside an upstream-qualified domain
// fall back to scalar libm by lane/block so the ordinary contract does not
// become a vendor precondition. Runtime selection chooses a compiled
// AVX2/AVX-512, NEON/SVE or RVV tier once; scalar libm is the always-available
// floor. selected_target(KernelFamily::kMath) reports the active broad tier.
// Backend choice is an implementation detail, not a stable ABI or a promise to
// preserve a historical benchmark winner.

namespace hyle::simd {

HYLE_SIMD_API void exp_f32_batch(const float* src, float* dst,
                                 std::size_t n) noexcept;
HYLE_SIMD_API void log_f32_batch(const float* src, float* dst,
                                 std::size_t n) noexcept;
HYLE_SIMD_API void tanh_f32_batch(const float* src, float* dst,
                                  std::size_t n) noexcept;

// Explicit narrow-domain fast paths: every exp lane must be finite in [-80,
// 80]; every log lane must be finite, normal and positive (>= FLT_MIN). They
// skip ordinary-domain recovery. Exact src == dst aliasing is supported.
HYLE_SIMD_API void exp_in_range_f32_batch(const float* src, float* dst,
                                          std::size_t n) noexcept;
HYLE_SIMD_API void log_in_range_f32_batch(const float* src, float* dst,
                                          std::size_t n) noexcept;

// f64 counterparts serve double-precision solver, geometry and numeric
// workloads.
HYLE_SIMD_API void exp_f64_batch(const double* src, double* dst,
                                 std::size_t n) noexcept;
HYLE_SIMD_API void log_f64_batch(const double* src, double* dst,
                                 std::size_t n) noexcept;
// Explicit f64 narrow-domain fast paths: every exp lane must be finite in
// [-700, 700]; every log lane must be finite, normal and positive (>= DBL_MIN).
// They omit ordinary-domain recovery and therefore must not be used on
// untrusted/general values. Exact src == dst aliasing is supported.
HYLE_SIMD_API void exp_in_range_f64_batch(const double* src, double* dst,
                                          std::size_t n) noexcept;
HYLE_SIMD_API void log_in_range_f64_batch(const double* src, double* dst,
                                          std::size_t n) noexcept;
HYLE_SIMD_API void tanh_f64_batch(const double* src, double* dst,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void sin_f64_batch(const double* src, double* dst,
                                 std::size_t n) noexcept;
HYLE_SIMD_API void cos_f64_batch(const double* src, double* dst,
                                 std::size_t n) noexcept;
HYLE_SIMD_API void exp2_f64_batch(const double* src, double* dst,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void log2_f64_batch(const double* src, double* dst,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void erf_f64_batch(const double* src, double* dst,
                                 std::size_t n) noexcept;
HYLE_SIMD_API void atan_f64_batch(const double* src, double* dst,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void cbrt_f64_batch(const double* src, double* dst,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void expm1_f64_batch(const double* src, double* dst,
                                   std::size_t n) noexcept;
HYLE_SIMD_API void log1p_f64_batch(const double* src, double* dst,
                                   std::size_t n) noexcept;
HYLE_SIMD_API void asinh_f64_batch(const double* src, double* dst,
                                   std::size_t n) noexcept;
HYLE_SIMD_API void acosh_f64_batch(const double* src, double* dst,
                                   std::size_t n) noexcept;
HYLE_SIMD_API void acos_f64_batch(const double* src, double* dst,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void asin_f64_batch(const double* src, double* dst,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void sinh_f64_batch(const double* src, double* dst,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void cosh_f64_batch(const double* src, double* dst,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void tan_f64_batch(const double* src, double* dst,
                                 std::size_t n) noexcept;
HYLE_SIMD_API void atanh_f64_batch(const double* src, double* dst,
                                   std::size_t n) noexcept;
HYLE_SIMD_API void log10_f64_batch(const double* src, double* dst,
                                   std::size_t n) noexcept;
HYLE_SIMD_API void atan2_f64_batch(const double* a, const double* b,
                                   double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void pow_f64_batch(const double* a, const double* b, double* dst,
                                 std::size_t n) noexcept;
HYLE_SIMD_API void sincos_f64_batch(const double* src, double* s, double* c,
                                    std::size_t n) noexcept;
HYLE_SIMD_API void tgamma_f64_batch(const double* src, double* dst,
                                    std::size_t n) noexcept;
HYLE_SIMD_API void lgamma_f64_batch(const double* src, double* dst,
                                    std::size_t n) noexcept;
HYLE_SIMD_API void hypot_f64_batch(const double* a, const double* b,
                                   double* dst, std::size_t n) noexcept;

// Additional f32 real functions use the same ordinary contract and target
// selection.
HYLE_SIMD_API void sin_f32_batch(const float* src, float* dst,
                                 std::size_t n) noexcept;
HYLE_SIMD_API void cos_f32_batch(const float* src, float* dst,
                                 std::size_t n) noexcept;
HYLE_SIMD_API void exp2_f32_batch(const float* src, float* dst,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void log2_f32_batch(const float* src, float* dst,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void erf_f32_batch(const float* src, float* dst,
                                 std::size_t n) noexcept;
HYLE_SIMD_API void atan_f32_batch(const float* src, float* dst,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void cbrt_f32_batch(const float* src, float* dst,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void expm1_f32_batch(const float* src, float* dst,
                                   std::size_t n) noexcept;
HYLE_SIMD_API void log1p_f32_batch(const float* src, float* dst,
                                   std::size_t n) noexcept;
HYLE_SIMD_API void asinh_f32_batch(const float* src, float* dst,
                                   std::size_t n) noexcept;
HYLE_SIMD_API void acosh_f32_batch(const float* src, float* dst,
                                   std::size_t n) noexcept;
HYLE_SIMD_API void acos_f32_batch(const float* src, float* dst,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void asin_f32_batch(const float* src, float* dst,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void sinh_f32_batch(const float* src, float* dst,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void cosh_f32_batch(const float* src, float* dst,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void tan_f32_batch(const float* src, float* dst,
                                 std::size_t n) noexcept;
HYLE_SIMD_API void atanh_f32_batch(const float* src, float* dst,
                                   std::size_t n) noexcept;
HYLE_SIMD_API void log10_f32_batch(const float* src, float* dst,
                                   std::size_t n) noexcept;
HYLE_SIMD_API void atan2_f32_batch(const float* a, const float* b, float* dst,
                                   std::size_t n) noexcept;
HYLE_SIMD_API void pow_f32_batch(const float* a, const float* b, float* dst,
                                 std::size_t n) noexcept;
HYLE_SIMD_API void sincos_f32_batch(const float* src, float* s, float* c,
                                    std::size_t n) noexcept;
HYLE_SIMD_API void erfc_f32_batch(const float* src, float* dst,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void exp10_f32_batch(const float* src, float* dst,
                                   std::size_t n) noexcept;
HYLE_SIMD_API void fmod_f32_batch(const float* a, const float* b, float* dst,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void remainder_f32_batch(const float* a, const float* b,
                                       float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void nextafter_f32_batch(const float* a, const float* b,
                                       float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void fdim_f32_batch(const float* a, const float* b, float* dst,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void erfc_f64_batch(const double* src, double* dst,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void exp10_f64_batch(const double* src, double* dst,
                                   std::size_t n) noexcept;
HYLE_SIMD_API void fmod_f64_batch(const double* a, const double* b, double* dst,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void remainder_f64_batch(const double* a, const double* b,
                                       double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void nextafter_f64_batch(const double* a, const double* b,
                                       double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void fdim_f64_batch(const double* a, const double* b, double* dst,
                                  std::size_t n) noexcept;

// Gamma functions use the locked Highway implementation on its qualified domain
// and scalar libm recovery elsewhere.
HYLE_SIMD_API void lgamma_f32_batch(const float* src, float* dst,
                                    std::size_t n) noexcept;
HYLE_SIMD_API void tgamma_f32_batch(const float* src, float* dst,
                                    std::size_t n) noexcept;

// Two explicit hypot contracts; neither silently substitutes for the other:
// - hypot_f32_batch: full-range std::hypot semantics via exponent scaling.
// - hypot_in_range_f32_batch: direct square/add/sqrt (FMA may be fused), ~2x
// faster, but callers must
//   keep the intermediate sum representable and accept loss of underflow
//   protection. No per-lane checks. dst may exactly alias a or b; other overlap
//   is outside the contract. See usage-guide.md.
//   ("InRange" == the caller-guarantees-narrower-domain fast-path convention,
//   same as Highway's DemoteToNearestIntInRange / HYLE's
//   Convert/Demote/PromoteInRangeTo.)
HYLE_SIMD_API void hypot_f32_batch(const float* a, const float* b, float* dst,
                                   std::size_t n) noexcept;
HYLE_SIMD_API void hypot_in_range_f32_batch(const float* a, const float* b,
                                            float* dst, std::size_t n) noexcept;

// Model-independent f32 activation rows. Each entry selects one target-bound
// kernel and composes the locked Highway math/register primitives inside that
// kernel; there is no hidden allocation, full-buffer temporary, nested public
// dispatch, lock, or per-lane callback. All are O(n), O(1) extra storage,
// support exact src == dst, and otherwise require disjoint ranges. n == 0
// accepts null pointers. NaNs propagate. ReLU maps both signed zeros to +0; the
// other piecewise functions preserve the zero selected by their mathematical
// branch.
//   gelu(x)      = 0.5 x (1 + erf(x/sqrt2))            -- exact Gaussian Error
//   Linear Unit gelu_tanh(x) = 0.5 x (1 + tanh(sqrt(2/pi) x (1 + 0.044715
//   x^2)))  -- the tanh approximation
//                  (the common ggml/LLM form; it is a distinct approximation
//                  contract)
//   sigmoid(x)   = 1/(1 + exp(-x))                     -- logistic
//   silu(x)      = x * sigmoid(x)                      -- SiLU/swish (Llama FFN
//   gate)
HYLE_SIMD_API void gelu_f32_batch(const float* src, float* dst,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void gelu_tanh_f32_batch(const float* src, float* dst,
                                       std::size_t n) noexcept;
HYLE_SIMD_API void sigmoid_f32_batch(const float* src, float* dst,
                                     std::size_t n) noexcept;
HYLE_SIMD_API void silu_f32_batch(const float* src, float* dst,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void relu_f32_batch(const float* src, float* dst,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void leaky_relu_f32_batch(const float* src, float negative_slope,
                                        float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void elu_f32_batch(const float* src, float alpha, float* dst,
                                 std::size_t n) noexcept;
// hard-sigmoid = clamp((x + 3) / 6, 0, 1); hard-swish = x * hard-sigmoid(x).
HYLE_SIMD_API void hard_sigmoid_f32_batch(const float* src, float* dst,
                                          std::size_t n) noexcept;
HYLE_SIMD_API void hard_swish_f32_batch(const float* src, float* dst,
                                        std::size_t n) noexcept;

// Inference activation rows use f32 arithmetic even when their storage is
// f16/bf16.  The result is rounded once when stored.  `gelu_quick` is the
// widely deployed x * sigmoid(1.702 * x) approximation; it is deliberately
// distinct from exact-erf GELU and tanh-GELU.  Narrow rows support exact
// in-place operation and otherwise require disjoint ranges.
HYLE_SIMD_API void gelu_quick_f32_batch(const float* src, float* dst,
                                        std::size_t n) noexcept;
HYLE_SIMD_API void gelu_f16_batch(const float16_t* src, float16_t* dst,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void gelu_tanh_f16_batch(const float16_t* src, float16_t* dst,
                                       std::size_t n) noexcept;
HYLE_SIMD_API void gelu_quick_f16_batch(const float16_t* src, float16_t* dst,
                                        std::size_t n) noexcept;
HYLE_SIMD_API void silu_f16_batch(const float16_t* src, float16_t* dst,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void gelu_bf16_batch(const bfloat16_t* src, bfloat16_t* dst,
                                   std::size_t n) noexcept;
HYLE_SIMD_API void gelu_tanh_bf16_batch(const bfloat16_t* src, bfloat16_t* dst,
                                        std::size_t n) noexcept;
HYLE_SIMD_API void gelu_quick_bf16_batch(const bfloat16_t* src, bfloat16_t* dst,
                                         std::size_t n) noexcept;
HYLE_SIMD_API void silu_bf16_batch(const bfloat16_t* src, bfloat16_t* dst,
                                   std::size_t n) noexcept;

// Split-input gated rows compute activation(gate[i]) * value[i].  dst may
// exactly alias gate, value, or both; all other overlap is outside the
// contract.  The five names keep exact-erf, tanh and quick GeGLU semantics
// visible rather than hiding an approximation behind `geglu`.
#define HYLE_SIMD_DECLARE_GATED_ACTIVATIONS(T, tag)                            \
  HYLE_SIMD_API void reglu_##tag##_batch(const T* gate, const T* value,        \
                                         T* dst, std::size_t n) noexcept;      \
  HYLE_SIMD_API void geglu_##tag##_batch(const T* gate, const T* value,        \
                                         T* dst, std::size_t n) noexcept;      \
  HYLE_SIMD_API void geglu_tanh_##tag##_batch(const T* gate, const T* value,   \
                                              T* dst, std::size_t n) noexcept; \
  HYLE_SIMD_API void geglu_quick_##tag##_batch(                                \
      const T* gate, const T* value, T* dst, std::size_t n) noexcept;          \
  HYLE_SIMD_API void swiglu_##tag##_batch(const T* gate, const T* value,       \
                                          T* dst, std::size_t n) noexcept

HYLE_SIMD_DECLARE_GATED_ACTIVATIONS(float, f32);
HYLE_SIMD_DECLARE_GATED_ACTIVATIONS(float16_t, f16);
HYLE_SIMD_DECLARE_GATED_ACTIVATIONS(bfloat16_t, bf16);

#undef HYLE_SIMD_DECLARE_GATED_ACTIVATIONS

// One contiguous row of RMSNorm or LayerNorm.  The overload without learned
// parameters is non-affine; weighted RMSNorm applies `weight`, and affine
// LayerNorm applies `scale` then `bias`.  Narrow storage is promoted for f32
// accumulation/arithmetic and rounded once on store. n == 0 is a no-op.  For
// non-empty rows epsilon must be finite and non-negative.  dst may exactly
// alias src; learned parameter ranges must be disjoint from dst and arbitrary
// partial overlap is outside the contract.  All forms are O(n), O(1),
// allocation-free and lock-free.
#define HYLE_SIMD_DECLARE_NORMALIZATION(T, tag)                             \
  HYLE_SIMD_API void rms_norm_##tag(const T* src, T* dst, std::size_t n,    \
                                    float epsilon) noexcept;                \
  HYLE_SIMD_API void rms_norm_##tag(const T* src, const T* weight, T* dst,  \
                                    std::size_t n, float epsilon) noexcept; \
  HYLE_SIMD_API void layer_norm_##tag(const T* src, T* dst, std::size_t n,  \
                                      float epsilon) noexcept;              \
  HYLE_SIMD_API void layer_norm_##tag(const T* src, const T* scale,         \
                                      const T* bias, T* dst, std::size_t n, \
                                      float epsilon) noexcept

HYLE_SIMD_DECLARE_NORMALIZATION(float, f32);
HYLE_SIMD_DECLARE_NORMALIZATION(float16_t, f16);
HYLE_SIMD_DECLARE_NORMALIZATION(bfloat16_t, bf16);

#undef HYLE_SIMD_DECLARE_NORMALIZATION

// Stable row reductions. The core is out-of-place and also supports exact src
// == dst; arbitrary partial overlap is undefined. At least one finite lane is
// required. -inf mask lanes are valid when another lane is finite. A row
// containing NaN or +inf, or an all--inf row, has no supported normalizer and
// deterministically writes quiet NaN to every destination lane.
HYLE_SIMD_API void softmax_f32(const float* src, float* dst,
                               std::size_t n) noexcept;
HYLE_SIMD_API void log_softmax_f32(const float* src, float* dst,
                                   std::size_t n) noexcept;

// Natural in-place conveniences; both route to the same exported out-of-place
// core.
inline void softmax_f32(float* data, std::size_t n) noexcept {
  softmax_f32(data, data, n);
}
inline void log_softmax_f32(float* data, std::size_t n) noexcept {
  log_softmax_f32(data, data, n);
}

// ---- COMPLEX transcendentals
// -------------------------------------------------------------- Highway has no
// complex math surface. HYLE composes the locked Highway real primitives on
// AVX2/AVX-512, NEON/SVE and RVV; shared real work removes redundant
// operations. Their operation-specific exceptional lanes use std::complex;
// ctanh/cpow specifically isolate axes, subnormal, non-finite, branch-cut and
// overflow-risk lanes without scalarizing regular neighbors.
//
// STORAGE = SoA (planar): real and imaginary parts live in separate arrays.
// Exact in-place use (dst_re==src_re and dst_im==src_im) is supported;
// arbitrary offset overlap is not. n arbitrary.
//   cexp/clog/csqrt : the workhorses (FFT twiddle, transfer functions, roots)
//   csin/ccos/ctan + csinh/ccosh/ctanh : complex trig and hyperbolic families
//   cpow : z^w (both complex)
HYLE_SIMD_API void cexp_f32_batch(const float* src_re, const float* src_im,
                                  float* dst_re, float* dst_im,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void clog_f32_batch(const float* src_re, const float* src_im,
                                  float* dst_re, float* dst_im,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void csqrt_f32_batch(const float* src_re, const float* src_im,
                                   float* dst_re, float* dst_im,
                                   std::size_t n) noexcept;
HYLE_SIMD_API void csin_f32_batch(const float* src_re, const float* src_im,
                                  float* dst_re, float* dst_im,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void ccos_f32_batch(const float* src_re, const float* src_im,
                                  float* dst_re, float* dst_im,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void ctan_f32_batch(const float* src_re, const float* src_im,
                                  float* dst_re, float* dst_im,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void csinh_f32_batch(const float* src_re, const float* src_im,
                                   float* dst_re, float* dst_im,
                                   std::size_t n) noexcept;
HYLE_SIMD_API void ccosh_f32_batch(const float* src_re, const float* src_im,
                                   float* dst_re, float* dst_im,
                                   std::size_t n) noexcept;
HYLE_SIMD_API void ctanh_f32_batch(const float* src_re, const float* src_im,
                                   float* dst_re, float* dst_im,
                                   std::size_t n) noexcept;
// cpow(z, w): both operands complex. Four input planes, two output planes.
HYLE_SIMD_API void cpow_f32_batch(const float* z_re, const float* z_im,
                                  const float* w_re, const float* w_im,
                                  float* dst_re, float* dst_im,
                                  std::size_t n) noexcept;

HYLE_SIMD_API void cexp_f64_batch(const double* src_re, const double* src_im,
                                  double* dst_re, double* dst_im,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void clog_f64_batch(const double* src_re, const double* src_im,
                                  double* dst_re, double* dst_im,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void csqrt_f64_batch(const double* src_re, const double* src_im,
                                   double* dst_re, double* dst_im,
                                   std::size_t n) noexcept;
HYLE_SIMD_API void csin_f64_batch(const double* src_re, const double* src_im,
                                  double* dst_re, double* dst_im,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void ccos_f64_batch(const double* src_re, const double* src_im,
                                  double* dst_re, double* dst_im,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void ctan_f64_batch(const double* src_re, const double* src_im,
                                  double* dst_re, double* dst_im,
                                  std::size_t n) noexcept;
HYLE_SIMD_API void csinh_f64_batch(const double* src_re, const double* src_im,
                                   double* dst_re, double* dst_im,
                                   std::size_t n) noexcept;
HYLE_SIMD_API void ccosh_f64_batch(const double* src_re, const double* src_im,
                                   double* dst_re, double* dst_im,
                                   std::size_t n) noexcept;
HYLE_SIMD_API void ctanh_f64_batch(const double* src_re, const double* src_im,
                                   double* dst_re, double* dst_im,
                                   std::size_t n) noexcept;
HYLE_SIMD_API void cpow_f64_batch(const double* z_re, const double* z_im,
                                  const double* w_re, const double* w_im,
                                  double* dst_re, double* dst_im,
                                  std::size_t n) noexcept;

}  // namespace hyle::simd
