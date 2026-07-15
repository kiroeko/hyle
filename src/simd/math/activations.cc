// AI activation combined cores (simd-design "AI 数学函数"). These COMPOSE the vectorized
// transcendental batch ops (exp/tanh/erf -- each self-dispatches to its best tier) into the
// shapes the AI hot path needs. HYLE-specific combinations, not single vendor ops, so there is
// no per-tier borrow wiring here: this TU is plain baseline C++ that calls the *_f32_batch entry
// points (which internally route to xsimd/Highway/scalar per cpu_features). The elementwise
// combine math (the 0.5x(1+...), the 1/(1+e), the divide) runs scalar -- it is cheap relative to
// the transcendental it wraps, and staying baseline keeps this TU SIGILL-safe with no -march.
//
// softmax/log_softmax: single-exp fused multipass (max → exp+sum → scale). One vectorized exp
// only — online max+sum-exp would re-exp on the write pass and is slower on CPU.
#include "hyle/simd/math.h"

#include <cmath>
#include <cstddef>
#include <vector>

namespace hyle::simd {

namespace {
constexpr float kSqrt2Inv = 0.70710678118654752440f;      // 1/sqrt(2)  (erf-gelu)
constexpr float kSqrt2OverPi = 0.79788456080286535588f;   // sqrt(2/pi) (tanh-gelu)
constexpr float kGeluCoefA = 0.044715f;                   // tanh-gelu cubic coef (ggml)

// Keep exp+sum+scale working set hot (~16 KiB floats).
constexpr std::size_t kSoftmaxChunk = 4096;

inline float array_max(const float* data, std::size_t n) noexcept {
  float m = data[0];
  for (std::size_t i = 1; i < n; ++i) m = data[i] > m ? data[i] : m;
  return m;
}
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

// softmax in place: max → chunked (x-m, exp, sum) → scale. Exactly one vectorized exp.
void softmax_f32(float* data, std::size_t n) noexcept {
  if (n == 0) return;
  const float m = array_max(data, n);
  double sum = 0.0;
  for (std::size_t base = 0; base < n; base += kSoftmaxChunk) {
    const std::size_t c = (n - base < kSoftmaxChunk) ? (n - base) : kSoftmaxChunk;
    float* p = data + base;
    for (std::size_t i = 0; i < c; ++i) p[i] -= m;
    exp_f32_batch(p, p, c);
    for (std::size_t i = 0; i < c; ++i) sum += static_cast<double>(p[i]);
  }
  const float inv = static_cast<float>(1.0 / sum);
  for (std::size_t i = 0; i < n; ++i) data[i] *= inv;
}

// log_softmax in place: max + sum exp(x-m) in one read (scalar exp for s only), then
// x -= (m + log s). Output path has no second full exp of the array.
// (s still needs n scalar exps — cheaper than vector exp+write when only log-sum is needed.)
void log_softmax_f32(float* data, std::size_t n) noexcept {
  if (n == 0) return;
  const float m = array_max(data, n);
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) sum += std::exp(static_cast<double>(data[i] - m));
  const float shift = m + static_cast<float>(std::log(sum));
  for (std::size_t i = 0; i < n; ++i) data[i] -= shift;
}

}  // namespace hyle::simd
