#include "hyle/simd/ops.h"

#include "src/simd/runtime_internal.h"
#include "src/simd/dispatch_internal.h"
#include "src/simd/ops/kernels.h"

namespace hyle::simd {

float dot(const float* x, const float* y, std::size_t n) noexcept {
#if defined(HYLE_SIMD_EXACT)
  return HYLE_SIMD_EXACT_SYMBOL(dot_f32)(x, y, n);
#else
  return runtime().ops.dot_f32(x, y, n);
#endif
}
double dot(const double* x, const double* y, std::size_t n) noexcept {
#if defined(HYLE_SIMD_EXACT)
  return HYLE_SIMD_EXACT_SYMBOL(dot_f64)(x, y, n);
#else
  return runtime().ops.dot_f64(x, y, n);
#endif
}
float dot(const bfloat16_t* x, const bfloat16_t* y, std::size_t n) noexcept {
#if defined(HYLE_SIMD_EXACT)
  return HYLE_SIMD_EXACT_SYMBOL(dot_bf16_bf16)(x, y, n);
#else
  return runtime().ops.dot_bf16_bf16(x, y, n);
#endif
}
float dot(const float* x, const bfloat16_t* y, std::size_t n) noexcept {
#if defined(HYLE_SIMD_EXACT)
  return HYLE_SIMD_EXACT_SYMBOL(dot_f32_bf16)(x, y, n);
#else
  return runtime().ops.dot_f32_bf16(x, y, n);
#endif
}
float dot(const bfloat16_t* x, const float* y, std::size_t n) noexcept {
  return dot(y, x, n);
}
float dot(const float16_t* x, const float16_t* y, std::size_t n) noexcept {
#if defined(HYLE_SIMD_EXACT)
  return HYLE_SIMD_EXACT_SYMBOL(dot_f16_f16)(x, y, n);
#else
  return runtime().ops.dot_f16_f16(x, y, n);
#endif
}
std::int64_t dot(const std::int16_t* x, const std::int16_t* y,
                 std::size_t n) noexcept {
#if defined(HYLE_SIMD_EXACT)
  return HYLE_SIMD_EXACT_SYMBOL(dot_i16_i16)(x, y, n);
#else
  return runtime().ops.dot_i16_i16(x, y, n);
#endif
}
std::int64_t dot(const std::int8_t* x, const std::int8_t* y,
                 std::size_t n) noexcept {
#if defined(HYLE_SIMD_EXACT)
  return HYLE_SIMD_EXACT_SYMBOL(dot_i8_i8)(x, y, n);
#else
  return runtime().ops.dot_i8_i8(x, y, n);
#endif
}
void saxpy(float a, const float* x, float* y, std::size_t n) noexcept {
#if defined(HYLE_SIMD_EXACT)
  HYLE_SIMD_EXACT_SYMBOL(saxpy_f32)(a, x, y, n);
#else
  runtime().ops.saxpy_f32(a, x, y, n);
#endif
}
void saxpy(double a, const double* x, double* y, std::size_t n) noexcept {
#if defined(HYLE_SIMD_EXACT)
  HYLE_SIMD_EXACT_SYMBOL(saxpy_f64)(a, x, y, n);
#else
  runtime().ops.saxpy_f64(a, x, y, n);
#endif
}
void add(const float* x, const float* y, float* out, std::size_t n) noexcept {
#if defined(HYLE_SIMD_EXACT)
  HYLE_SIMD_EXACT_SYMBOL(add_f32)(x, y, out, n);
#else
  runtime().ops.add_f32(x, y, out, n);
#endif
}
void add(const double* x, const double* y, double* out, std::size_t n) noexcept {
#if defined(HYLE_SIMD_EXACT)
  HYLE_SIMD_EXACT_SYMBOL(add_f64)(x, y, out, n);
#else
  runtime().ops.add_f64(x, y, out, n);
#endif
}

}  // namespace hyle::simd
