#include "hyle/simd/ops.h"

#include "hyle/simd/runtime.h"
#include "src/simd/ops/kernels.h"

#if defined(HYLE_SIMD_EXACT_TARGET_AVX512)
#define HYLE_OP(name) detail::name##_avx512
#elif defined(HYLE_SIMD_EXACT_TARGET_AVX2)
#define HYLE_OP(name) detail::name##_avx2
#elif defined(HYLE_SIMD_EXACT_TARGET_SSE4)
#define HYLE_OP(name) detail::name##_sse4
#elif defined(HYLE_SIMD_EXACT_TARGET_NEON)
#define HYLE_OP(name) detail::name##_neon
#elif defined(HYLE_SIMD_EXACT_TARGET_SVE)
#define HYLE_OP(name) detail::name##_sve
#elif defined(HYLE_SIMD_EXACT_TARGET_RVV)
#define HYLE_OP(name) detail::name##_rvv
#elif defined(HYLE_SIMD_EXACT_TARGET_SCALAR)
#define HYLE_OP(name) detail::name##_scalar
#endif

namespace hyle::simd {

float dot(const float* x, const float* y, std::size_t n) noexcept {
#if defined(HYLE_SIMD_EXACT)
  return HYLE_OP(dot_f32)(x, y, n);
#else
  return runtime().ops.dot_f32(x, y, n);
#endif
}
double dot(const double* x, const double* y, std::size_t n) noexcept {
#if defined(HYLE_SIMD_EXACT)
  return HYLE_OP(dot_f64)(x, y, n);
#else
  return runtime().ops.dot_f64(x, y, n);
#endif
}
void saxpy(float a, const float* x, float* y, std::size_t n) noexcept {
#if defined(HYLE_SIMD_EXACT)
  HYLE_OP(saxpy_f32)(a, x, y, n);
#else
  runtime().ops.saxpy_f32(a, x, y, n);
#endif
}
void saxpy(double a, const double* x, double* y, std::size_t n) noexcept {
#if defined(HYLE_SIMD_EXACT)
  HYLE_OP(saxpy_f64)(a, x, y, n);
#else
  runtime().ops.saxpy_f64(a, x, y, n);
#endif
}
void add(const float* x, const float* y, float* out, std::size_t n) noexcept {
#if defined(HYLE_SIMD_EXACT)
  HYLE_OP(add_f32)(x, y, out, n);
#else
  runtime().ops.add_f32(x, y, out, n);
#endif
}
void add(const double* x, const double* y, double* out, std::size_t n) noexcept {
#if defined(HYLE_SIMD_EXACT)
  HYLE_OP(add_f64)(x, y, out, n);
#else
  runtime().ops.add_f64(x, y, out, n);
#endif
}

}  // namespace hyle::simd

#ifdef HYLE_OP
#undef HYLE_OP
#endif
