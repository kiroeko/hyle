// AI activation combined cores (simd-design "AI 数学函数"). These COMPOSE the vectorized
// transcendental batch ops (exp/tanh/erf -- each self-dispatches to its best tier) into the
// shapes the AI hot path needs. HYLE-specific combinations, not single vendor ops, so there is
// no per-tier borrow wiring here: this TU is plain baseline C++ that calls the *_f32_batch entry
// points (which internally route to xsimd/Highway/scalar per cpu_features). The elementwise
// combine math (the 0.5x(1+...), the 1/(1+e), the divide) runs scalar -- it is cheap relative to
// the transcendental it wraps, and staying baseline keeps this TU SIGILL-safe with no -march.
//
// softmax/log_softmax: online max+sum-of-exp (Milakov–Gimelshein) — one streaming read for
// (m,s), then one write pass. Avoids the old 4~5 full-array sweeps (max / sub / exp / sum / scale).
#include "hyle/simd/math.h"

#include <cmath>
#include <cstddef>
#include <vector>

namespace hyle::simd {

namespace {
constexpr float kSqrt2Inv = 0.70710678118654752440f;      // 1/sqrt(2)  (erf-gelu)
constexpr float kSqrt2OverPi = 0.79788456080286535588f;   // sqrt(2/pi) (tanh-gelu)
constexpr float kGeluCoefA = 0.044715f;                   // tanh-gelu cubic coef (ggml)

// Online (m, s): s = Σ exp(x_i - m) with m = max x. Single forward pass, double sum.
// n >= 1. Uses std::exp in double for the running scale (same stability class as the old sum fold).
inline void online_max_sum_exp(const float* data, std::size_t n, float& m_out,
                               double& s_out) noexcept {
  float m = data[0];
  double s = 1.0;  // exp(data[0] - m) == 1
  for (std::size_t i = 1; i < n; ++i) {
    const float x = data[i];
    if (x <= m) {
      s += std::exp(static_cast<double>(x - m));
    } else {
      s = s * std::exp(static_cast<double>(m - x)) + 1.0;
      m = x;
    }
  }
  m_out = m;
  s_out = s;
}

// Chunk size for pass-2 exp+scale: keep the three logical ops in cache (~16 KiB floats).
constexpr std::size_t kSoftmaxChunk = 4096;
}  // namespace

// gelu(x) = 0.5 x (1 + erf(x / sqrt2)). Vectorize the erf pass, combine scalar.
void gelu_f32_batch(const float* src, float* dst, std::size_t n) noexcept {
  if (n == 0) return;
  std::vector<float> e(n);
  for (std::size_t i = 0; i < n; ++i) e[i] = src[i] * kSqrt2Inv;  // x/sqrt2
  erf_f32_batch(e.data(), e.data(), n);                          // erf(x/sqrt2) (vectorized)
  for (std::size_t i = 0; i < n; ++i) dst[i] = 0.5f * src[i] * (1.0f + e[i]);
}

// gelu_tanh(x) = 0.5 x (1 + tanh(sqrt(2/pi) x (1 + 0.044715 x^2))). The ggml/LLM approximation.
void gelu_tanh_f32_batch(const float* src, float* dst, std::size_t n) noexcept {
  if (n == 0) return;
  std::vector<float> t(n);
  for (std::size_t i = 0; i < n; ++i) {
    const float x = src[i];
    t[i] = kSqrt2OverPi * x * (1.0f + kGeluCoefA * x * x);  // inner arg
  }
  tanh_f32_batch(t.data(), t.data(), n);                    // tanh(...) (vectorized)
  for (std::size_t i = 0; i < n; ++i) dst[i] = 0.5f * src[i] * (1.0f + t[i]);
}

// sigmoid(x) = 1/(1 + exp(-x)). Vectorize exp(-x), combine scalar.
void sigmoid_f32_batch(const float* src, float* dst, std::size_t n) noexcept {
  if (n == 0) return;
  std::vector<float> e(n);
  for (std::size_t i = 0; i < n; ++i) e[i] = -src[i];
  exp_f32_batch(e.data(), e.data(), n);                     // exp(-x)
  for (std::size_t i = 0; i < n; ++i) dst[i] = 1.0f / (1.0f + e[i]);
}

// silu(x) = x * sigmoid(x) = x / (1 + exp(-x)). (SwiGLU/Llama FFN gate.)
void silu_f32_batch(const float* src, float* dst, std::size_t n) noexcept {
  if (n == 0) return;
  std::vector<float> e(n);
  for (std::size_t i = 0; i < n; ++i) e[i] = -src[i];
  exp_f32_batch(e.data(), e.data(), n);
  for (std::size_t i = 0; i < n; ++i) dst[i] = src[i] / (1.0f + e[i]);
}

// softmax in place (online): pass1 = (m,s) in one read; pass2 = chunked (x-m) → exp → /s.
// Memory traffic: ~2n words streaming vs old 4~5 full sweeps (vocab ~128k is bandwidth-bound).
void softmax_f32(float* data, std::size_t n) noexcept {
  if (n == 0) return;
  float m = 0.f;
  double s = 0.0;
  online_max_sum_exp(data, n, m, s);
  const float inv = static_cast<float>(1.0 / s);
  for (std::size_t base = 0; base < n; base += kSoftmaxChunk) {
    const std::size_t c = (n - base < kSoftmaxChunk) ? (n - base) : kSoftmaxChunk;
    float* p = data + base;
    for (std::size_t i = 0; i < c; ++i) p[i] -= m;  // x - max (<= 0)
    exp_f32_batch(p, p, c);                         // vectorized exp
    for (std::size_t i = 0; i < c; ++i) p[i] *= inv;
  }
}

// log_softmax in place (online): pass1 = (m,s); pass2 = x - (m + log s). True 2-pass, no exp write.
// More stable than log(softmax); matches old contract.
void log_softmax_f32(float* data, std::size_t n) noexcept {
  if (n == 0) return;
  float m = 0.f;
  double s = 0.0;
  online_max_sum_exp(data, n, m, s);
  const float shift = m + static_cast<float>(std::log(s));
  for (std::size_t i = 0; i < n; ++i) data[i] -= shift;
}

}  // namespace hyle::simd
