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

// f64 (double) transcendentals -- the CAE / solver / geometry-kernel path where double precision
// is the working type. Same vendored cores as f32, but the best-of winner is bench-picked PER
// PRECISION: at f64 Highway wins exp/log/tanh (the reverse of f32, where xsimd wins exp/tanh) --
// see math_dispatch.cc. avx2 (4-wide) + scalar floor; avx512/neon f64 siblings are follow-on.
HYLE_SIMD_API void exp_f64_batch(const double* src, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void log_f64_batch(const double* src, double* dst, std::size_t n) noexcept;
HYLE_SIMD_API void tanh_f64_batch(const double* src, double* dst, std::size_t n) noexcept;

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

// xsimd-ONLY borrows (Highway's contrib/math has no equivalent, so these are a single-source
// borrow, NOT a best-of pick -- there is no second library to race). lgamma/tgamma are the
// log-gamma and gamma functions (Bayesian / statistical / combinatorial hot paths). Borrowed
// from xsimd on EVERY arch that has a vector backend -- x86 (avx2/avx512) AND arm64 (xsimd-NEON,
// the first xsimd core on arm64: the NEON math TU is otherwise Highway-only, but Highway can't
// serve gamma). scalar libm is the always-present oracle + fallback.
HYLE_SIMD_API void lgamma_f32_batch(const float* src, float* dst, std::size_t n) noexcept;
HYLE_SIMD_API void tgamma_f32_batch(const float* src, float* dst, std::size_t n) noexcept;

// hypot(a,b) = sqrt(a*a + b*b), overflow-safe -- a BINARY op (two input arrays -> one output),
// unlike every unary transcendental above. BOTH xsimd and Highway provide it. math_bench
// (MSVC+GCC) found xsimd ~2.2x FASTER (0.14 vs 0.30 ns/op) -- BUT xsimd's hypot is NOT
// overflow-safe: it computes sqrt(a*a+b*b) naively and returns inf once a^2/b^2 exceed FLT_MAX,
// whereas Highway's Hypot does exponent scaling to match std::hypot across the full range.
// Overflow-safety is the whole reason hypot exists over a naive sqrt, so CORRECTNESS beats the
// 2.2x: HYLE borrows HIGHWAY here (the best-of winner isn't always the fastest -- a bounded,
// wrong-for-large-inputs result is not a valid substitute). scalar std::hypot is oracle + fallback.
HYLE_SIMD_API void hypot_f32_batch(const float* a, const float* b, float* dst, std::size_t n) noexcept;

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

// ---- COMPLEX transcendentals (xsimd-ONLY: Highway's contrib/math has no complex support) -------
// Borrowed from xsimd's batch<std::complex<float>> cores -- HYLE does not hand-roll complex
// polynomial approximations (charter: 借验证核不手搓). Highway offers NO complex path, so these are
// a single-source borrow (no best-of race), the complex analogue of the lgamma/tgamma gamma borrow.
//
// STORAGE = SoA (planar): real and imaginary parts in SEPARATE arrays (re[i], im[i]). This matches
// xsimd's batch<complex>::load_unaligned(re, im) constructor exactly (zero interleave overhead) and
// is the natural AI/DSP layout (FFT, signal). dst_re/dst_im may alias src_re/src_im. n arbitrary.
// scalar std::complex<float> is the always-present oracle + fallback (x86 AVX2 core, else scalar).
//   cexp/clog/csqrt : the workhorses (FFT twiddle, transfer functions, roots)
//   csin/ccos/ctan + csinh/ccosh/ctanh : trig / hyperbolic
//   cpow : z^w (both complex)
HYLE_SIMD_API void cexp_f32_batch(const float* src_re, const float* src_im, float* dst_re, float* dst_im, std::size_t n) noexcept;
HYLE_SIMD_API void clog_f32_batch(const float* src_re, const float* src_im, float* dst_re, float* dst_im, std::size_t n) noexcept;
HYLE_SIMD_API void csqrt_f32_batch(const float* src_re, const float* src_im, float* dst_re, float* dst_im, std::size_t n) noexcept;
HYLE_SIMD_API void csin_f32_batch(const float* src_re, const float* src_im, float* dst_re, float* dst_im, std::size_t n) noexcept;
HYLE_SIMD_API void ccos_f32_batch(const float* src_re, const float* src_im, float* dst_re, float* dst_im, std::size_t n) noexcept;
HYLE_SIMD_API void ctanh_f32_batch(const float* src_re, const float* src_im, float* dst_re, float* dst_im, std::size_t n) noexcept;
// cpow(z, w): both operands complex. Four input planes, two output planes.
HYLE_SIMD_API void cpow_f32_batch(const float* z_re, const float* z_im, const float* w_re, const float* w_im, float* dst_re, float* dst_im, std::size_t n) noexcept;

// The tier the math batch ops selected: kAvx2 if a vectorized AVX2 core is live, else
// kScalar (libm). For tests/diagnostics.
HYLE_SIMD_API Isa active_math_variant() noexcept;

}  // namespace hyle::simd
