#pragma once

#include <cstddef>

#include "hyle/simd/types.h"

// Internal pointer/count kernels. Highway types never cross this boundary.
namespace hyle::simd::detail {

#define HYLE_DECLARE_UNARY(symbol, type) \
  void symbol(const type* src, type* dst, std::size_t n) noexcept;
#define HYLE_DECLARE_BINARY(symbol, type) \
  void symbol(const type* a, const type* b, type* dst, std::size_t n) noexcept;
#define HYLE_DECLARE_SINCOS(symbol, type)                \
  void symbol(const type* src, type* sine, type* cosine, \
              std::size_t n) noexcept;
#define HYLE_DECLARE_COMPLEX_UNARY(symbol, type)                    \
  void symbol(const type* src_re, const type* src_im, type* dst_re, \
              type* dst_im, std::size_t n) noexcept;
#define HYLE_DECLARE_COMPLEX_POW(symbol, type)                      \
  void symbol(const type* z_re, const type* z_im, const type* w_re, \
              const type* w_im, type* dst_re, type* dst_im,         \
              std::size_t n) noexcept;
#define HYLE_DECLARE_PARAM_UNARY(symbol)                     \
  void symbol(const float* src, float parameter, float* dst, \
              std::size_t n) noexcept;
#define HYLE_DECLARE_ROW_REDUCTION(symbol) \
  void symbol(const float* src, float* dst, std::size_t n) noexcept;
#define HYLE_DECLARE_GATED(symbol, type)                      \
  void symbol(const type* gate, const type* value, type* dst, \
              std::size_t n) noexcept;
#define HYLE_DECLARE_NORM(symbol, type)                                      \
  void symbol(const type* src, const type* weight, type* dst, std::size_t n, \
              float epsilon) noexcept;
#define HYLE_DECLARE_LAYER_NORM(symbol, type)                                  \
  void symbol(const type* src, const type* scale, const type* bias, type* dst, \
              std::size_t n, float epsilon) noexcept;

#define HYLE_REAL_UNARY_LIST_WITHOUT_TANH(M, suffix, type, tag) \
  M(exp_##tag##_##suffix, type)                                 \
  M(log_##tag##_##suffix, type)                                 \
  M(sin_##tag##_##suffix, type)                                 \
  M(cos_##tag##_##suffix, type)                                 \
  M(exp2_##tag##_##suffix, type)                                \
  M(log2_##tag##_##suffix, type)                                \
  M(erf_##tag##_##suffix, type)                                 \
  M(atan_##tag##_##suffix, type)                                \
  M(cbrt_##tag##_##suffix, type)                                \
  M(expm1_##tag##_##suffix, type)                               \
  M(log1p_##tag##_##suffix, type)                               \
  M(asinh_##tag##_##suffix, type)                               \
  M(acosh_##tag##_##suffix, type)                               \
  M(acos_##tag##_##suffix, type)                                \
  M(asin_##tag##_##suffix, type)                                \
  M(sinh_##tag##_##suffix, type)                                \
  M(cosh_##tag##_##suffix, type)                                \
  M(tan_##tag##_##suffix, type)                                 \
  M(atanh_##tag##_##suffix, type)                               \
  M(log10_##tag##_##suffix, type)                               \
  M(tgamma_##tag##_##suffix, type)                              \
  M(lgamma_##tag##_##suffix, type)

#define HYLE_REAL_UNARY_LIST(M, suffix, type, tag)        \
  HYLE_REAL_UNARY_LIST_WITHOUT_TANH(M, suffix, type, tag) \
  M(tanh_##tag##_##suffix, type)

#define HYLE_REAL_BINARY_LIST(M, suffix, type, tag) \
  M(atan2_##tag##_##suffix, type)                   \
  M(pow_##tag##_##suffix, type)                     \
  M(hypot_##tag##_##suffix, type)

#define HYLE_COMPLEX_UNARY_LIST(M, suffix, type, tag) \
  M(cexp_##tag##_##suffix, type)                      \
  M(clog_##tag##_##suffix, type)                      \
  M(csqrt_##tag##_##suffix, type)                     \
  M(csin_##tag##_##suffix, type)                      \
  M(ccos_##tag##_##suffix, type)                      \
  M(ctan_##tag##_##suffix, type)                      \
  M(csinh_##tag##_##suffix, type)                     \
  M(ccosh_##tag##_##suffix, type)                     \
  M(ctanh_##tag##_##suffix, type)

// Always-present scalar libm/std::complex floor.
HYLE_REAL_UNARY_LIST(HYLE_DECLARE_UNARY, scalar, float, f32)
HYLE_REAL_UNARY_LIST(HYLE_DECLARE_UNARY, scalar, double, f64)
HYLE_REAL_BINARY_LIST(HYLE_DECLARE_BINARY, scalar, float, f32)
HYLE_REAL_BINARY_LIST(HYLE_DECLARE_BINARY, scalar, double, f64)
HYLE_DECLARE_SINCOS(sincos_f32_scalar, float)
HYLE_DECLARE_SINCOS(sincos_f64_scalar, double)
HYLE_DECLARE_BINARY(hypot_in_range_f32_scalar, float)
HYLE_DECLARE_UNARY(exp_in_range_f32_scalar, float)
HYLE_DECLARE_UNARY(log_in_range_f32_scalar, float)
HYLE_DECLARE_UNARY(exp_in_range_f64_scalar, double)
HYLE_DECLARE_UNARY(log_in_range_f64_scalar, double)

#define HYLE_DECLARE_SCALAR_GAPS(type, tag)           \
  HYLE_DECLARE_UNARY(erfc_##tag##_scalar, type)       \
  HYLE_DECLARE_UNARY(exp10_##tag##_scalar, type)      \
  HYLE_DECLARE_BINARY(fmod_##tag##_scalar, type)      \
  HYLE_DECLARE_BINARY(remainder_##tag##_scalar, type) \
  HYLE_DECLARE_BINARY(nextafter_##tag##_scalar, type) \
  HYLE_DECLARE_BINARY(fdim_##tag##_scalar, type)
HYLE_DECLARE_SCALAR_GAPS(float, f32)
HYLE_DECLARE_SCALAR_GAPS(double, f64)
#undef HYLE_DECLARE_SCALAR_GAPS

HYLE_COMPLEX_UNARY_LIST(HYLE_DECLARE_COMPLEX_UNARY, scalar, float, f32)
HYLE_COMPLEX_UNARY_LIST(HYLE_DECLARE_COMPLEX_UNARY, scalar, double, f64)
HYLE_DECLARE_COMPLEX_POW(cpow_f32_scalar, float)
HYLE_DECLARE_COMPLEX_POW(cpow_f64_scalar, double)

// Scalar semantic floor for model-independent f32 activations.
HYLE_DECLARE_UNARY(gelu_f32_scalar, float)
HYLE_DECLARE_UNARY(gelu_tanh_f32_scalar, float)
HYLE_DECLARE_UNARY(sigmoid_f32_scalar, float)
HYLE_DECLARE_UNARY(silu_f32_scalar, float)
HYLE_DECLARE_UNARY(relu_f32_scalar, float)
HYLE_DECLARE_PARAM_UNARY(leaky_relu_f32_scalar)
HYLE_DECLARE_PARAM_UNARY(elu_f32_scalar)
HYLE_DECLARE_UNARY(hard_sigmoid_f32_scalar, float)
HYLE_DECLARE_UNARY(hard_swish_f32_scalar, float)
HYLE_DECLARE_UNARY(gelu_quick_f32_scalar, float)
HYLE_DECLARE_ROW_REDUCTION(softmax_f32_scalar)
HYLE_DECLARE_ROW_REDUCTION(log_softmax_f32_scalar)

#define HYLE_DECLARE_INFERENCE_SCALAR(type, tag)       \
  HYLE_DECLARE_UNARY(gelu_##tag##_scalar, type)        \
  HYLE_DECLARE_UNARY(gelu_tanh_##tag##_scalar, type)   \
  HYLE_DECLARE_UNARY(gelu_quick_##tag##_scalar, type)  \
  HYLE_DECLARE_UNARY(silu_##tag##_scalar, type)        \
  HYLE_DECLARE_GATED(reglu_##tag##_scalar, type)       \
  HYLE_DECLARE_GATED(geglu_##tag##_scalar, type)       \
  HYLE_DECLARE_GATED(geglu_tanh_##tag##_scalar, type)  \
  HYLE_DECLARE_GATED(geglu_quick_##tag##_scalar, type) \
  HYLE_DECLARE_GATED(swiglu_##tag##_scalar, type)      \
  HYLE_DECLARE_NORM(rms_norm_##tag##_scalar, type)     \
  HYLE_DECLARE_LAYER_NORM(layer_norm_##tag##_scalar, type)

HYLE_DECLARE_INFERENCE_SCALAR(float16_t, f16)
HYLE_DECLARE_INFERENCE_SCALAR(bfloat16_t, bf16)
HYLE_DECLARE_GATED(reglu_f32_scalar, float)
HYLE_DECLARE_GATED(geglu_f32_scalar, float)
HYLE_DECLARE_GATED(geglu_tanh_f32_scalar, float)
HYLE_DECLARE_GATED(geglu_quick_f32_scalar, float)
HYLE_DECLARE_GATED(swiglu_f32_scalar, float)
HYLE_DECLARE_NORM(rms_norm_f32_scalar, float)
HYLE_DECLARE_LAYER_NORM(layer_norm_f32_scalar, float)

#undef HYLE_DECLARE_INFERENCE_SCALAR

// Direct Highway kernels and the small HYLE semantic/composition layer for a
// fixed target. Declarations are harmless on architectures whose TUs are not
// built; the baseline dispatcher only references compiled variants.
#define HYLE_DECLARE_VARIANT(suffix)                                         \
  HYLE_REAL_UNARY_LIST_WITHOUT_TANH(HYLE_DECLARE_UNARY, highway_##suffix,    \
                                    float, f32)                              \
  HYLE_REAL_UNARY_LIST(HYLE_DECLARE_UNARY, highway_##suffix, double, f64)    \
  HYLE_DECLARE_UNARY(tanh_f32_route_##suffix, float)                         \
  HYLE_REAL_BINARY_LIST(HYLE_DECLARE_BINARY, highway_##suffix, float, f32)   \
  HYLE_REAL_BINARY_LIST(HYLE_DECLARE_BINARY, highway_##suffix, double, f64)  \
  HYLE_DECLARE_SINCOS(sincos_f32_highway_##suffix, float)                    \
  HYLE_DECLARE_SINCOS(sincos_f64_highway_##suffix, double)                   \
  HYLE_DECLARE_UNARY(exp_in_range_f32_highway_##suffix, float)               \
  HYLE_DECLARE_UNARY(log_in_range_f32_highway_##suffix, float)               \
  HYLE_DECLARE_UNARY(exp_in_range_f64_highway_##suffix, double)              \
  HYLE_DECLARE_UNARY(log_in_range_f64_highway_##suffix, double)              \
  HYLE_DECLARE_BINARY(hypot_in_range_f32_hyle_##suffix, float)               \
  HYLE_DECLARE_UNARY(erfc_f32_hyle_##suffix, float)                          \
  HYLE_DECLARE_UNARY(erfc_f64_hyle_##suffix, double)                         \
  HYLE_DECLARE_UNARY(exp10_f32_hyle_##suffix, float)                         \
  HYLE_DECLARE_UNARY(exp10_f64_hyle_##suffix, double)                        \
  HYLE_DECLARE_BINARY(fdim_f32_hyle_##suffix, float)                         \
  HYLE_DECLARE_BINARY(fdim_f64_hyle_##suffix, double)                        \
  HYLE_COMPLEX_UNARY_LIST(HYLE_DECLARE_COMPLEX_UNARY, hyle_##suffix, float,  \
                          f32)                                               \
  HYLE_COMPLEX_UNARY_LIST(HYLE_DECLARE_COMPLEX_UNARY, hyle_##suffix, double, \
                          f64)                                               \
  HYLE_DECLARE_COMPLEX_POW(cpow_f32_hyle_##suffix, float)                    \
  HYLE_DECLARE_COMPLEX_POW(cpow_f64_hyle_##suffix, double)

#define HYLE_DECLARE_ACTIVATION_VARIANT(suffix)                  \
  HYLE_DECLARE_UNARY(gelu_f32_hyle_##suffix, float)              \
  HYLE_DECLARE_UNARY(gelu_tanh_f32_route_##suffix, float)        \
  HYLE_DECLARE_UNARY(sigmoid_f32_hyle_##suffix, float)           \
  HYLE_DECLARE_UNARY(silu_f32_hyle_##suffix, float)              \
  HYLE_DECLARE_UNARY(relu_f32_hyle_##suffix, float)              \
  HYLE_DECLARE_PARAM_UNARY(leaky_relu_f32_hyle_##suffix)         \
  HYLE_DECLARE_PARAM_UNARY(elu_f32_hyle_##suffix)                \
  HYLE_DECLARE_UNARY(hard_sigmoid_f32_hyle_##suffix, float)      \
  HYLE_DECLARE_UNARY(hard_swish_f32_hyle_##suffix, float)        \
  HYLE_DECLARE_UNARY(gelu_quick_f32_hyle_##suffix, float)        \
  HYLE_DECLARE_ROW_REDUCTION(softmax_f32_hyle_##suffix)          \
  HYLE_DECLARE_ROW_REDUCTION(log_softmax_f32_hyle_##suffix)      \
  HYLE_DECLARE_INFERENCE_VARIANT(float, f32, suffix)             \
  HYLE_DECLARE_INFERENCE_VARIANT(float16_t, f16, suffix)         \
  HYLE_DECLARE_INFERENCE_VARIANT(bfloat16_t, bf16, suffix)       \
  HYLE_DECLARE_NARROW_ACTIVATION_VARIANT(float16_t, f16, suffix) \
  HYLE_DECLARE_NARROW_ACTIVATION_VARIANT(bfloat16_t, bf16, suffix)

#define HYLE_DECLARE_INFERENCE_VARIANT(type, tag, suffix)     \
  HYLE_DECLARE_GATED(reglu_##tag##_hyle_##suffix, type)       \
  HYLE_DECLARE_GATED(geglu_##tag##_hyle_##suffix, type)       \
  HYLE_DECLARE_GATED(geglu_tanh_##tag##_hyle_##suffix, type)  \
  HYLE_DECLARE_GATED(geglu_quick_##tag##_hyle_##suffix, type) \
  HYLE_DECLARE_GATED(swiglu_##tag##_hyle_##suffix, type)      \
  HYLE_DECLARE_NORM(rms_norm_##tag##_hyle_##suffix, type)     \
  HYLE_DECLARE_LAYER_NORM(layer_norm_##tag##_hyle_##suffix, type)

#define HYLE_DECLARE_NARROW_ACTIVATION_VARIANT(type, tag, suffix) \
  HYLE_DECLARE_UNARY(gelu_##tag##_hyle_##suffix, type)            \
  HYLE_DECLARE_UNARY(gelu_tanh_##tag##_hyle_##suffix, type)       \
  HYLE_DECLARE_UNARY(gelu_quick_##tag##_hyle_##suffix, type)      \
  HYLE_DECLARE_UNARY(silu_##tag##_hyle_##suffix, type)

HYLE_DECLARE_VARIANT(avx2)
HYLE_DECLARE_VARIANT(avx512)
HYLE_DECLARE_VARIANT(neon)
HYLE_DECLARE_VARIANT(sve)
HYLE_DECLARE_VARIANT(rvv)
HYLE_DECLARE_ACTIVATION_VARIANT(avx2)
HYLE_DECLARE_ACTIVATION_VARIANT(avx512)
HYLE_DECLARE_ACTIVATION_VARIANT(neon)
HYLE_DECLARE_ACTIVATION_VARIANT(sve)
HYLE_DECLARE_ACTIVATION_VARIANT(rvv)

#undef HYLE_DECLARE_NARROW_ACTIVATION_VARIANT
#undef HYLE_DECLARE_INFERENCE_VARIANT
#undef HYLE_DECLARE_ACTIVATION_VARIANT
#undef HYLE_DECLARE_VARIANT
#undef HYLE_COMPLEX_UNARY_LIST
#undef HYLE_REAL_BINARY_LIST
#undef HYLE_REAL_UNARY_LIST
#undef HYLE_REAL_UNARY_LIST_WITHOUT_TANH
#undef HYLE_DECLARE_COMPLEX_POW
#undef HYLE_DECLARE_COMPLEX_UNARY
#undef HYLE_DECLARE_SINCOS
#undef HYLE_DECLARE_BINARY
#undef HYLE_DECLARE_UNARY
#undef HYLE_DECLARE_ROW_REDUCTION
#undef HYLE_DECLARE_PARAM_UNARY
#undef HYLE_DECLARE_LAYER_NORM
#undef HYLE_DECLARE_NORM
#undef HYLE_DECLARE_GATED

}  // namespace hyle::simd::detail
