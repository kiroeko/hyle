// AI activation combined cores (simd-design "AI 数学函数"). These COMPOSE the vectorized
// transcendental batch ops (exp/tanh/erf -- each self-dispatches to its best tier) into the
// shapes the AI hot path needs. HYLE-specific combinations, not single vendor ops, so there is
// no per-tier borrow wiring here: this TU is plain baseline C++ that calls the *_f32_batch entry
// points (which internally route to xsimd/Highway/scalar per cpu_features). The elementwise
// combine math (the 0.5x(1+...), the 1/(1+e), the divide) runs scalar -- it is cheap relative to
// the transcendental it wraps, and staying baseline keeps this TU SIGILL-safe with no -march.
//
// softmax/log_softmax are REDUCTIONS (max + sum-of-exp), done in place: the exp pass is the
// vectorized cost (exp_f32_batch), the max/sum are simple scalar folds (memory-bound anyway).
#include "hyle/simd/math.h"

#include <cmath>
#include <cstddef>
#include <vector>

namespace hyle::simd {

namespace {
constexpr float kSqrt2Inv = 0.70710678118654752440f;      // 1/sqrt(2)  (erf-gelu)
constexpr float kSqrt2OverPi = 0.79788456080286535588f;   // sqrt(2/pi) (tanh-gelu)
constexpr float kGeluCoefA = 0.044715f;                   // tanh-gelu cubic coef (ggml)
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

// softmax in place: subtract the max (numerical stability), exp, normalize by the sum.
void softmax_f32(float* data, std::size_t n) noexcept {
  if (n == 0) return;
  float mx = data[0];
  for (std::size_t i = 1; i < n; ++i) mx = data[i] > mx ? data[i] : mx;
  for (std::size_t i = 0; i < n; ++i) data[i] -= mx;   // x - max (<= 0, exp in (0,1])
  exp_f32_batch(data, data, n);                        // exp(x - max) (vectorized)
  double sum = 0.0;                                    // double accum: n can be large (vocab ~128k)
  for (std::size_t i = 0; i < n; ++i) sum += static_cast<double>(data[i]);
  const float inv = static_cast<float>(1.0 / sum);
  for (std::size_t i = 0; i < n; ++i) data[i] *= inv;
}

// log_softmax in place: (x - max) - log(sum exp(x - max)). Keeps the shifted logits, then
// subtracts logZ. More stable than log(softmax) (no exp underflow -> log(0)).
void log_softmax_f32(float* data, std::size_t n) noexcept {
  if (n == 0) return;
  float mx = data[0];
  for (std::size_t i = 1; i < n; ++i) mx = data[i] > mx ? data[i] : mx;
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    data[i] -= mx;                                     // shifted logit (kept)
    sum += std::exp(static_cast<double>(data[i]));     // sum exp(x - max)
  }
  const float log_z = static_cast<float>(std::log(sum));
  for (std::size_t i = 0; i < n; ++i) data[i] -= log_z;
}

}  // namespace hyle::simd
