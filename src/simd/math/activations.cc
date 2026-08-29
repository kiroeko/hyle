// Scalar semantic floor for model-independent f32 activations and row
// softmax. Wide implementations live beside the direct Highway math kernels;
// this TU deliberately contains no allocation, target dispatch, or temporary
// row storage.
#include "src/simd/math/kernels.h"

#include <cmath>
#include <cstddef>
#include <limits>

namespace hyle::simd::detail {
namespace {

constexpr float kSqrt2Inv = 0.70710678118654752440f;
constexpr float kSqrt2OverPi = 0.79788456080286535588f;
constexpr float kGeluCoefficient = 0.044715f;

float sigmoid(float x) noexcept {
  if (std::isnan(x)) return x;
  const float e = std::exp(-std::abs(x));
  return x < 0.0f ? e / (1.0f + e) : 1.0f / (1.0f + e);
}

float gelu(float x) noexcept {
  if (x == -std::numeric_limits<float>::infinity()) return 0.0f;
  return 0.5f * x * (1.0f + std::erf(x * kSqrt2Inv));
}

float gelu_tanh(float x) noexcept {
  if (x == -std::numeric_limits<float>::infinity()) return 0.0f;
  const float inner = kSqrt2OverPi * x *
                      std::fma(kGeluCoefficient, x * x, 1.0f);
  return 0.5f * x * (1.0f + std::tanh(inner));
}

float silu(float x) noexcept {
  if (x == -std::numeric_limits<float>::infinity()) return 0.0f;
  return x * sigmoid(x);
}

float relu(float x) noexcept {
  if (std::isnan(x)) return x;
  return x > 0.0f ? x : 0.0f;
}

float leaky_relu(float x, float negative_slope) noexcept {
  if (std::isnan(x)) return x;
  return x >= 0.0f ? x : negative_slope * x;
}

float elu(float x, float alpha) noexcept {
  if (std::isnan(x)) return x;
  return x >= 0.0f ? x : alpha * std::expm1(x);
}

float hard_sigmoid(float x) noexcept {
  if (std::isnan(x)) return x;
  const float scaled = (x + 3.0f) / 6.0f;
  if (scaled <= 0.0f) return 0.0f;
  return scaled >= 1.0f ? 1.0f : scaled;
}

float hard_swish(float x) noexcept {
  if (std::isnan(x)) return x;
  if (x <= -3.0f) return 0.0f;
  if (x >= 3.0f) return x;
  return x * ((x + 3.0f) / 6.0f);
}

template <class Fn>
void unary(const float* src, float* dst, std::size_t n, Fn&& fn) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = fn(src[i]);
}

bool finite_row_max(const float* src, std::size_t n,
                    float& maximum) noexcept {
  maximum = -std::numeric_limits<float>::infinity();
  for (std::size_t i = 0; i < n; ++i) {
    const float x = src[i];
    if (std::isnan(x) || x == std::numeric_limits<float>::infinity()) {
      return false;
    }
    if (x > maximum) maximum = x;
  }
  return std::isfinite(maximum);
}

void fill_invalid_row(float* dst, std::size_t n) noexcept {
  const float value = std::numeric_limits<float>::quiet_NaN();
  for (std::size_t i = 0; i < n; ++i) dst[i] = value;
}

}  // namespace

void gelu_f32_scalar(const float* src, float* dst, std::size_t n) noexcept {
  unary(src, dst, n, gelu);
}

void gelu_tanh_f32_scalar(const float* src, float* dst,
                          std::size_t n) noexcept {
  unary(src, dst, n, gelu_tanh);
}

void sigmoid_f32_scalar(const float* src, float* dst,
                        std::size_t n) noexcept {
  unary(src, dst, n, sigmoid);
}

void silu_f32_scalar(const float* src, float* dst, std::size_t n) noexcept {
  unary(src, dst, n, silu);
}

void relu_f32_scalar(const float* src, float* dst, std::size_t n) noexcept {
  unary(src, dst, n, relu);
}

void leaky_relu_f32_scalar(const float* src, float negative_slope, float* dst,
                           std::size_t n) noexcept {
  unary(src, dst, n,
        [negative_slope](float x) { return leaky_relu(x, negative_slope); });
}

void elu_f32_scalar(const float* src, float alpha, float* dst,
                    std::size_t n) noexcept {
  unary(src, dst, n, [alpha](float x) { return elu(x, alpha); });
}

void hard_sigmoid_f32_scalar(const float* src, float* dst,
                             std::size_t n) noexcept {
  unary(src, dst, n, hard_sigmoid);
}

void hard_swish_f32_scalar(const float* src, float* dst,
                           std::size_t n) noexcept {
  unary(src, dst, n, hard_swish);
}

void softmax_f32_scalar(const float* src, float* dst,
                        std::size_t n) noexcept {
  if (n == 0) return;
  float maximum = 0.0f;
  if (!finite_row_max(src, n, maximum)) {
    fill_invalid_row(dst, n);
    return;
  }

  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const float value = std::exp(src[i] - maximum);
    dst[i] = value;
    sum += static_cast<double>(value);
  }
  const float inverse = static_cast<float>(1.0 / sum);
  for (std::size_t i = 0; i < n; ++i) dst[i] *= inverse;
}

void log_softmax_f32_scalar(const float* src, float* dst,
                            std::size_t n) noexcept {
  if (n == 0) return;
  float maximum = 0.0f;
  if (!finite_row_max(src, n, maximum)) {
    fill_invalid_row(dst, n);
    return;
  }

  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    sum += static_cast<double>(std::exp(src[i] - maximum));
  }
  const float shift = maximum + static_cast<float>(std::log(sum));
  for (std::size_t i = 0; i < n; ++i) dst[i] = src[i] - shift;
}

}  // namespace hyle::simd::detail
