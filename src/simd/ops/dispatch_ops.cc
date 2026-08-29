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

#if defined(HYLE_SIMD_EXACT)
#define HYLE_SIMD_ROW_CALL(name, ...) HYLE_SIMD_EXACT_SYMBOL(name)(__VA_ARGS__)
#else
#define HYLE_SIMD_ROW_CALL(name, ...) runtime().ops.name(__VA_ARGS__)
#endif

#define HYLE_SIMD_DEFINE_ROW_API(T, tag)                                       \
  void subtract(const T* x, const T* y, T* out, std::size_t n) noexcept {      \
    HYLE_SIMD_ROW_CALL(sub_##tag, x, y, out, n);                              \
  }                                                                            \
  void multiply(const T* x, const T* y, T* out, std::size_t n) noexcept {      \
    HYLE_SIMD_ROW_CALL(mul_##tag, x, y, out, n);                              \
  }                                                                            \
  void divide(const T* x, const T* y, T* out, std::size_t n) noexcept {        \
    HYLE_SIMD_ROW_CALL(div_##tag, x, y, out, n);                              \
  }                                                                            \
  void scale(const T* x, T factor, T* out, std::size_t n) noexcept {           \
    HYLE_SIMD_ROW_CALL(scale_##tag, x, factor, out, n);                       \
  }                                                                            \
  void add_scalar(const T* x, T value, T* out, std::size_t n) noexcept {       \
    HYLE_SIMD_ROW_CALL(add_scalar_##tag, x, value, out, n);                   \
  }                                                                            \
  void affine(const T* x, T factor, T bias, T* out, std::size_t n) noexcept {  \
    HYLE_SIMD_ROW_CALL(affine_##tag, x, factor, bias, out, n);                \
  }                                                                            \
  T sum(const T* x, std::size_t n) noexcept {                                  \
    return HYLE_SIMD_ROW_CALL(sum_##tag, x, n);                               \
  }                                                                            \
  void abs(const T* x, T* out, std::size_t n) noexcept {                       \
    HYLE_SIMD_ROW_CALL(abs_##tag, x, out, n);                                 \
  }                                                                            \
  void negate(const T* x, T* out, std::size_t n) noexcept {                    \
    HYLE_SIMD_ROW_CALL(neg_##tag, x, out, n);                                 \
  }                                                                            \
  void square(const T* x, T* out, std::size_t n) noexcept {                    \
    HYLE_SIMD_ROW_CALL(square_##tag, x, out, n);                              \
  }                                                                            \
  void sqrt(const T* x, T* out, std::size_t n) noexcept {                      \
    HYLE_SIMD_ROW_CALL(sqrt_##tag, x, out, n);                                \
  }

HYLE_SIMD_DEFINE_ROW_API(float, f32)
HYLE_SIMD_DEFINE_ROW_API(double, f64)

#undef HYLE_SIMD_DEFINE_ROW_API
#undef HYLE_SIMD_ROW_CALL

}  // namespace hyle::simd
