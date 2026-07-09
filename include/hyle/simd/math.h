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
//     oracle + the non-x86 / non-AVX2 fallback. NOT a stub.
//   - two vendored header-only AVX2 cores (8-wide, ~3.5-ULP class), wired UNCONDITIONALLY
//     on x86 (third_party/highway + third_party/xsimd; no opt-in gate -- a perf library
//     ships its fastest path by default). The same-ISA bench found NO single library wins
//     all three ops: xsimd's exp/tanh polynomials are fastest, Highway's log is fastest.
//     So the dispatcher picks PER OP: exp/tanh -> xsimd > Highway; log -> Highway > xsimd.
//
// At runtime exp/log/tanh route to the best core when cpu_features().avx2, else scalar
// libm (so an old CPU never SIGILLs). active_math_variant() reports kAvx2 vs kScalar.
// (SLEEF was dropped: not header-only, and ranked third on every op -> never selected.)

namespace hyle::simd {

HYLE_SIMD_API void exp_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void log_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void tanh_f32_batch(const float* src, float* dst, std::size_t n) noexcept;

// simd-09: best-of-breed expansion. Per-op winner from the same-ISA bench (math_bench,
// GCC+MSVC cross-checked): Highway wins sin/cos/log2/atan/cbrt/expm1/log1p/asinh/acosh;
// xsimd wins exp2/erf. (Contrast simd-06's exp/tanh->xsimd, log->Highway -- the winner does
// NOT extrapolate across ops, which is why every op is benched.) All borrow a verified vendor
// core (charter: 借核不手搓); scalar libm is the always-present ULP oracle + fallback.
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

// softmax(x)[i] = exp(x[i] - max) / sum_j exp(x[j] - max)  -- IN PLACE over the whole array
// (a REDUCTION, not element-wise: needs the array max for numerical stability + the exp sum).
// dst may alias src. n == 0 is a no-op. Returns nothing; writes the normalized distribution.
HYLE_SIMD_API void softmax_f32(float* data, std::size_t n) noexcept;
// log_softmax(x)[i] = (x[i] - max) - log(sum_j exp(x[j] - max))  -- in place, numerically stable.
HYLE_SIMD_API void log_softmax_f32(float* data, std::size_t n) noexcept;

// The tier the math batch ops selected: kAvx2 if a vectorized AVX2 core is live, else
// kScalar (libm). For tests/diagnostics.
HYLE_SIMD_API Isa active_math_variant() noexcept;

}  // namespace hyle::simd
