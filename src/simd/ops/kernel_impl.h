// Per-target exported wrappers around the public target-bound implementation. This header is
// intentionally included once by each generated variant TU.

#include <cstddef>

#include "hyle/simd/native/ops.h"
#include "src/simd/ops/kernels.h"

#ifndef HYLE_SIMD_VARIANT_SUFFIX
#error "define HYLE_SIMD_VARIANT_SUFFIX before including ops/kernel_impl.h"
#endif

#define HYLE_SIMD_OPS_CAT_(a, b) a##b
#define HYLE_SIMD_OPS_CAT(a, b) HYLE_SIMD_OPS_CAT_(a, b)
#define HYLE_SIMD_OPS_FN(name) HYLE_SIMD_OPS_CAT(name, HYLE_SIMD_VARIANT_SUFFIX)

namespace hyle::simd::detail {

float HYLE_SIMD_OPS_FN(dot_f32_)(const float* x, const float* y,
                                 std::size_t n) noexcept {
  return native::dot(x, y, n);
}
double HYLE_SIMD_OPS_FN(dot_f64_)(const double* x, const double* y,
                                  std::size_t n) noexcept {
  return native::dot(x, y, n);
}
float HYLE_SIMD_OPS_FN(dot_bf16_bf16_)(const bfloat16_t* x,
                                       const bfloat16_t* y,
                                       std::size_t n) noexcept {
  return native::dot(x, y, n);
}
float HYLE_SIMD_OPS_FN(dot_f32_bf16_)(const float* x, const bfloat16_t* y,
                                      std::size_t n) noexcept {
  return native::dot(x, y, n);
}
float HYLE_SIMD_OPS_FN(dot_f16_f16_)(const float16_t* x, const float16_t* y,
                                     std::size_t n) noexcept {
  return native::dot(x, y, n);
}
std::int64_t HYLE_SIMD_OPS_FN(dot_i16_i16_)(const std::int16_t* x,
                                            const std::int16_t* y,
                                            std::size_t n) noexcept {
  return native::dot(x, y, n);
}
std::int64_t HYLE_SIMD_OPS_FN(dot_i8_i8_)(const std::int8_t* x,
                                          const std::int8_t* y,
                                          std::size_t n) noexcept {
  return native::dot(x, y, n);
}
void HYLE_SIMD_OPS_FN(saxpy_f32_)(float a, const float* x, float* y,
                                  std::size_t n) noexcept {
  native::saxpy(a, x, y, n);
}
void HYLE_SIMD_OPS_FN(saxpy_f64_)(double a, const double* x, double* y,
                                  std::size_t n) noexcept {
  native::saxpy(a, x, y, n);
}
void HYLE_SIMD_OPS_FN(add_f32_)(const float* x, const float* y, float* out,
                                std::size_t n) noexcept {
  native::add(x, y, out, n);
}
void HYLE_SIMD_OPS_FN(add_f64_)(const double* x, const double* y, double* out,
                                std::size_t n) noexcept {
  native::add(x, y, out, n);
}

}  // namespace hyle::simd::detail

#undef HYLE_SIMD_OPS_FN
#undef HYLE_SIMD_OPS_CAT
#undef HYLE_SIMD_OPS_CAT_
