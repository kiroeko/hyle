// Scalar semantic floor for model-independent f32 activations and row
// softmax. Wide implementations live beside the direct Highway math kernels;
// this TU deliberately contains no allocation, target dispatch, or temporary
// row storage.
#include "src/simd/math/kernels.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "hyle/simd/dtype.h"

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
  const float inner =
      kSqrt2OverPi * x * std::fma(kGeluCoefficient, x * x, 1.0f);
  return 0.5f * x * (1.0f + std::tanh(inner));
}

float gelu_quick(float x) noexcept {
  if (x == -std::numeric_limits<float>::infinity()) return 0.0f;
  return x * sigmoid(1.702f * x);
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

template <typename T>
float storage_to_f32(T value) noexcept {
  if constexpr (std::is_same_v<T, float>) {
    return value;
  } else if constexpr (std::is_same_v<T, float16_t>) {
    return f16_to_f32(value);
  } else {
    static_assert(std::is_same_v<T, bfloat16_t>);
    return bf16_to_f32(value);
  }
}

template <typename T>
T f32_to_storage(float value) noexcept {
  if constexpr (std::is_same_v<T, float>) {
    return value;
  } else if constexpr (std::is_same_v<T, float16_t>) {
    return f32_to_f16(value);
  } else {
    static_assert(std::is_same_v<T, bfloat16_t>);
    return f32_to_bf16(value);
  }
}

template <typename T, class Fn>
void unary_storage(const T* src, T* dst, std::size_t n, Fn&& fn) noexcept {
  for (std::size_t i = 0; i < n; ++i) {
    dst[i] = f32_to_storage<T>(fn(storage_to_f32(src[i])));
  }
}

template <typename T, class Fn>
void gated_storage(const T* gate, const T* value, T* dst, std::size_t n,
                   Fn&& activation) noexcept {
  for (std::size_t i = 0; i < n; ++i) {
    const float gate_value = storage_to_f32(gate[i]);
    const float linear_value = storage_to_f32(value[i]);
    dst[i] = f32_to_storage<T>(activation(gate_value) * linear_value);
  }
}

template <typename T>
void rms_norm_storage(const T* src, const T* weight, T* dst, std::size_t n,
                      float epsilon) noexcept {
  if (n == 0) return;
  float sum_squares = 0.0f;
  for (std::size_t i = 0; i < n; ++i) {
    const float value = storage_to_f32(src[i]);
    sum_squares = std::fma(value, value, sum_squares);
  }
  const float inverse_rms =
      1.0f / std::sqrt(sum_squares / static_cast<float>(n) + epsilon);
  if (weight == nullptr) {
    for (std::size_t i = 0; i < n; ++i) {
      dst[i] = f32_to_storage<T>(storage_to_f32(src[i]) * inverse_rms);
    }
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      const float value = storage_to_f32(src[i]);
      const float learned_scale = storage_to_f32(weight[i]);
      dst[i] = f32_to_storage<T>(value * inverse_rms * learned_scale);
    }
  }
}

template <typename T>
void layer_norm_storage(const T* src, const T* scale, const T* bias, T* dst,
                        std::size_t n, float epsilon) noexcept {
  if (n == 0) return;
  float sum = 0.0f;
  for (std::size_t i = 0; i < n; ++i) sum += storage_to_f32(src[i]);
  const float mean = sum / static_cast<float>(n);

  float sum_squares = 0.0f;
  for (std::size_t i = 0; i < n; ++i) {
    const float centered = storage_to_f32(src[i]) - mean;
    sum_squares = std::fma(centered, centered, sum_squares);
  }
  const float inverse_stddev =
      1.0f / std::sqrt(sum_squares / static_cast<float>(n) + epsilon);
  if (scale == nullptr) {
    for (std::size_t i = 0; i < n; ++i) {
      dst[i] =
          f32_to_storage<T>((storage_to_f32(src[i]) - mean) * inverse_stddev);
    }
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      const float normalized = (storage_to_f32(src[i]) - mean) * inverse_stddev;
      dst[i] = f32_to_storage<T>(std::fma(normalized, storage_to_f32(scale[i]),
                                          storage_to_f32(bias[i])));
    }
  }
}

bool finite_row_max(const float* src, std::size_t n, float& maximum) noexcept {
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

void gelu_quick_f32_scalar(const float* src, float* dst,
                           std::size_t n) noexcept {
  unary(src, dst, n, gelu_quick);
}

void sigmoid_f32_scalar(const float* src, float* dst, std::size_t n) noexcept {
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

#define HYLE_DEFINE_NARROW_ACTIVATIONS(T, tag)                             \
  void gelu_##tag##_scalar(const T* src, T* dst, std::size_t n) noexcept { \
    unary_storage(src, dst, n, gelu);                                      \
  }                                                                        \
  void gelu_tanh_##tag##_scalar(const T* src, T* dst,                      \
                                std::size_t n) noexcept {                  \
    unary_storage(src, dst, n, gelu_tanh);                                 \
  }                                                                        \
  void gelu_quick_##tag##_scalar(const T* src, T* dst,                     \
                                 std::size_t n) noexcept {                 \
    unary_storage(src, dst, n, gelu_quick);                                \
  }                                                                        \
  void silu_##tag##_scalar(const T* src, T* dst, std::size_t n) noexcept { \
    unary_storage(src, dst, n, silu);                                      \
  }

HYLE_DEFINE_NARROW_ACTIVATIONS(float16_t, f16)
HYLE_DEFINE_NARROW_ACTIVATIONS(bfloat16_t, bf16)

#undef HYLE_DEFINE_NARROW_ACTIVATIONS

#define HYLE_DEFINE_GATED(T, tag, name, activation)                 \
  void name##_##tag##_scalar(const T* gate, const T* value, T* dst, \
                             std::size_t n) noexcept {              \
    gated_storage(gate, value, dst, n, activation);                 \
  }

#define HYLE_DEFINE_GATED_SET(T, tag)                \
  HYLE_DEFINE_GATED(T, tag, reglu, relu)             \
  HYLE_DEFINE_GATED(T, tag, geglu, gelu)             \
  HYLE_DEFINE_GATED(T, tag, geglu_tanh, gelu_tanh)   \
  HYLE_DEFINE_GATED(T, tag, geglu_quick, gelu_quick) \
  HYLE_DEFINE_GATED(T, tag, swiglu, silu)

HYLE_DEFINE_GATED_SET(float, f32)
HYLE_DEFINE_GATED_SET(float16_t, f16)
HYLE_DEFINE_GATED_SET(bfloat16_t, bf16)

#undef HYLE_DEFINE_GATED_SET
#undef HYLE_DEFINE_GATED

#define HYLE_DEFINE_NORMALIZATION(T, tag)                                     \
  void rms_norm_##tag##_scalar(const T* src, const T* weight, T* dst,         \
                               std::size_t n, float epsilon) noexcept {       \
    rms_norm_storage(src, weight, dst, n, epsilon);                           \
  }                                                                           \
  void layer_norm_##tag##_scalar(const T* src, const T* scale, const T* bias, \
                                 T* dst, std::size_t n,                       \
                                 float epsilon) noexcept {                    \
    layer_norm_storage(src, scale, bias, dst, n, epsilon);                    \
  }

HYLE_DEFINE_NORMALIZATION(float, f32)
HYLE_DEFINE_NORMALIZATION(float16_t, f16)
HYLE_DEFINE_NORMALIZATION(bfloat16_t, bf16)

#undef HYLE_DEFINE_NORMALIZATION

void softmax_f32_scalar(const float* src, float* dst, std::size_t n) noexcept {
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
