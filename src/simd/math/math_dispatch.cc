#include "hyle/simd/math.h"

#include <cstddef>

#include "hyle/platform.h"
#include "hyle/simd/runtime.h"
#include "src/simd/math/kernels.h"
#include "src/simd/target_manifest.h"

// Baseline-only dispatcher. Fixed-target TUs contain all Highway vector types
// and wide instructions; this file only selects a pointer/count kernel.
namespace hyle::simd {
namespace {

#if defined(HYLE_ARCH_X64) &&                                      \
    (defined(HYLE_SIMD_COMPILED_AVX2) ||                           \
     defined(HYLE_SIMD_COMPILED_AVX512))
bool have_avx2_math() noexcept {
#if defined(HYLE_SIMD_EXACT_TARGET_AVX2) || \
    defined(HYLE_SIMD_EXACT_TARGET_AVX512)
  return true;
#elif defined(HYLE_SIMD_EXACT)
  return false;
#else
  return runtime().math.isa == Isa::kAvx2 ||
         runtime().math.isa == Isa::kAvx512;
#endif
}
#endif

#if defined(HYLE_ARCH_X64) && defined(HYLE_SIMD_COMPILED_AVX512) && \
    !defined(_MSC_VER)
bool have_avx512_math() noexcept {
#if defined(HYLE_SIMD_EXACT_TARGET_AVX512)
  return true;
#elif defined(HYLE_SIMD_EXACT)
  return false;
#else
  return runtime().math.isa == Isa::kAvx512;
#endif
}
#endif

#if defined(HYLE_ARCH_ARM64) && defined(HYLE_SIMD_COMPILED_SVE)
bool have_sve_math() noexcept {
#if defined(HYLE_SIMD_EXACT_TARGET_SVE)
  return true;
#elif defined(HYLE_SIMD_EXACT)
  return false;
#else
  return runtime().math.isa == Isa::kSve;
#endif
}
#endif

#if defined(HYLE_ARCH_RISCV64) && defined(HYLE_SIMD_COMPILED_RVV)
bool have_rvv_math() noexcept {
#if defined(HYLE_SIMD_EXACT_TARGET_RVV)
  return true;
#elif defined(HYLE_SIMD_EXACT)
  return false;
#else
  return runtime().math.isa == Isa::kRvv;
#endif
}
#endif

}  // namespace

#define HYLE_VARIANT_KERNEL_I(name, tag, owner, suffix) \
  detail::name##_##tag##_##owner##_##suffix
#define HYLE_VARIANT_KERNEL_II(name, tag, owner, suffix) \
  HYLE_VARIANT_KERNEL_I(name, tag, owner, suffix)
#define HYLE_SCALAR_KERNEL_I(name, tag) detail::name##_##tag##_scalar
#define HYLE_SCALAR_KERNEL(name, tag) HYLE_SCALAR_KERNEL_I(name, tag)

#if defined(HYLE_ARCH_X64) && defined(HYLE_SIMD_COMPILED_AVX512) && \
    !defined(_MSC_VER)
#define HYLE_TRY_AVX512(name, tag, owner, ...)                              \
  if (have_avx512_math()) {                                                 \
    HYLE_VARIANT_KERNEL_II(name, tag, owner, avx512)(__VA_ARGS__);          \
    return;                                                                 \
  }
#else
#define HYLE_TRY_AVX512(name, tag, owner, ...)
#endif

#if defined(HYLE_ARCH_X64) &&                                      \
    (defined(HYLE_SIMD_COMPILED_AVX2) ||                           \
     (defined(HYLE_SIMD_COMPILED_AVX512) && defined(_MSC_VER)))
#define HYLE_TRY_AVX2(name, tag, owner, ...)                                 \
  if (have_avx2_math()) {                                                    \
    HYLE_VARIANT_KERNEL_II(name, tag, owner, avx2)(__VA_ARGS__);             \
    return;                                                                  \
  }
#else
#define HYLE_TRY_AVX2(name, tag, owner, ...)
#endif

#if defined(HYLE_ARCH_ARM64) && defined(HYLE_SIMD_COMPILED_SVE)
#define HYLE_TRY_SVE(name, tag, owner, ...)                                  \
  if (have_sve_math()) {                                                     \
    HYLE_VARIANT_KERNEL_II(name, tag, owner, sve)(__VA_ARGS__);              \
    return;                                                                  \
  }
#else
#define HYLE_TRY_SVE(name, tag, owner, ...)
#endif

#if defined(HYLE_ARCH_ARM64) && defined(HYLE_SIMD_COMPILED_NEON)
#define HYLE_TRY_NEON(name, tag, owner, ...)                                 \
  HYLE_VARIANT_KERNEL_II(name, tag, owner, neon)(__VA_ARGS__);               \
  return;
#else
#define HYLE_TRY_NEON(name, tag, owner, ...)
#endif

#if defined(HYLE_ARCH_RISCV64) && defined(HYLE_SIMD_COMPILED_RVV)
#define HYLE_TRY_RVV(name, tag, owner, ...)                                  \
  if (have_rvv_math()) {                                                     \
    HYLE_VARIANT_KERNEL_II(name, tag, owner, rvv)(__VA_ARGS__);              \
    return;                                                                  \
  }
#else
#define HYLE_TRY_RVV(name, tag, owner, ...)
#endif

#define HYLE_DISPATCH(name, tag, owner, ...)                                 \
  do {                                                                       \
    HYLE_TRY_AVX512(name, tag, owner, __VA_ARGS__)                           \
    HYLE_TRY_AVX2(name, tag, owner, __VA_ARGS__)                             \
    HYLE_TRY_SVE(name, tag, owner, __VA_ARGS__)                              \
    HYLE_TRY_NEON(name, tag, owner, __VA_ARGS__)                             \
    HYLE_TRY_RVV(name, tag, owner, __VA_ARGS__)                              \
    HYLE_SCALAR_KERNEL(name, tag)(__VA_ARGS__);                              \
  } while (false)

#define HYLE_DEFINE_PUBLIC_UNARY(type, tag, name)                            \
  void name##_##tag##_batch(const type* src, type* dst,                      \
                            std::size_t n) noexcept {                         \
    HYLE_DISPATCH(name, tag, highway, src, dst, n);                          \
  }

#define HYLE_DEFINE_PUBLIC_UNARY_SET(type, tag) \
  HYLE_DEFINE_PUBLIC_UNARY(type, tag, exp)        \
  HYLE_DEFINE_PUBLIC_UNARY(type, tag, log)        \
  HYLE_DEFINE_PUBLIC_UNARY(type, tag, tanh)       \
  HYLE_DEFINE_PUBLIC_UNARY(type, tag, sin)        \
  HYLE_DEFINE_PUBLIC_UNARY(type, tag, cos)        \
  HYLE_DEFINE_PUBLIC_UNARY(type, tag, exp2)       \
  HYLE_DEFINE_PUBLIC_UNARY(type, tag, log2)       \
  HYLE_DEFINE_PUBLIC_UNARY(type, tag, erf)        \
  HYLE_DEFINE_PUBLIC_UNARY(type, tag, atan)       \
  HYLE_DEFINE_PUBLIC_UNARY(type, tag, cbrt)       \
  HYLE_DEFINE_PUBLIC_UNARY(type, tag, expm1)      \
  HYLE_DEFINE_PUBLIC_UNARY(type, tag, log1p)      \
  HYLE_DEFINE_PUBLIC_UNARY(type, tag, asinh)      \
  HYLE_DEFINE_PUBLIC_UNARY(type, tag, acosh)      \
  HYLE_DEFINE_PUBLIC_UNARY(type, tag, acos)       \
  HYLE_DEFINE_PUBLIC_UNARY(type, tag, asin)       \
  HYLE_DEFINE_PUBLIC_UNARY(type, tag, sinh)       \
  HYLE_DEFINE_PUBLIC_UNARY(type, tag, cosh)       \
  HYLE_DEFINE_PUBLIC_UNARY(type, tag, tan)        \
  HYLE_DEFINE_PUBLIC_UNARY(type, tag, atanh)      \
  HYLE_DEFINE_PUBLIC_UNARY(type, tag, log10)      \
  HYLE_DEFINE_PUBLIC_UNARY(type, tag, tgamma)     \
  HYLE_DEFINE_PUBLIC_UNARY(type, tag, lgamma)

HYLE_DEFINE_PUBLIC_UNARY_SET(float, f32)
HYLE_DEFINE_PUBLIC_UNARY_SET(double, f64)
#undef HYLE_DEFINE_PUBLIC_UNARY_SET
#undef HYLE_DEFINE_PUBLIC_UNARY

#define HYLE_DEFINE_PUBLIC_BINARY(type, tag, name)                           \
  void name##_##tag##_batch(const type* a, const type* b, type* dst,          \
                            std::size_t n) noexcept {                         \
    HYLE_DISPATCH(name, tag, highway, a, b, dst, n);                         \
  }

#define HYLE_DEFINE_PUBLIC_BINARY_SET(type, tag) \
  HYLE_DEFINE_PUBLIC_BINARY(type, tag, atan2)      \
  HYLE_DEFINE_PUBLIC_BINARY(type, tag, pow)        \
  HYLE_DEFINE_PUBLIC_BINARY(type, tag, hypot)
HYLE_DEFINE_PUBLIC_BINARY_SET(float, f32)
HYLE_DEFINE_PUBLIC_BINARY_SET(double, f64)
#undef HYLE_DEFINE_PUBLIC_BINARY_SET
#undef HYLE_DEFINE_PUBLIC_BINARY

void sincos_f32_batch(const float* src, float* sine, float* cosine,
                      std::size_t n) noexcept {
  HYLE_DISPATCH(sincos, f32, highway, src, sine, cosine, n);
}
void sincos_f64_batch(const double* src, double* sine, double* cosine,
                      std::size_t n) noexcept {
  HYLE_DISPATCH(sincos, f64, highway, src, sine, cosine, n);
}

void exp_in_range_f32_batch(const float* src, float* dst,
                            std::size_t n) noexcept {
  HYLE_DISPATCH(exp_in_range, f32, highway, src, dst, n);
}
void log_in_range_f32_batch(const float* src, float* dst,
                            std::size_t n) noexcept {
  HYLE_DISPATCH(log_in_range, f32, highway, src, dst, n);
}
void exp_in_range_f64_batch(const double* src, double* dst,
                            std::size_t n) noexcept {
  HYLE_DISPATCH(exp_in_range, f64, highway, src, dst, n);
}
void log_in_range_f64_batch(const double* src, double* dst,
                            std::size_t n) noexcept {
  HYLE_DISPATCH(log_in_range, f64, highway, src, dst, n);
}

void hypot_in_range_f32_batch(const float* a, const float* b, float* dst,
                              std::size_t n) noexcept {
  HYLE_DISPATCH(hypot_in_range, f32, hyle, a, b, dst, n);
}

#define HYLE_DEFINE_PUBLIC_GAP_UNARY(type, tag, name)                        \
  void name##_##tag##_batch(const type* src, type* dst,                     \
                            std::size_t n) noexcept {                        \
    HYLE_DISPATCH(name, tag, hyle, src, dst, n);                            \
  }
#define HYLE_DEFINE_PUBLIC_GAP_BINARY(type, tag, name)                       \
  void name##_##tag##_batch(const type* a, const type* b, type* dst,         \
                            std::size_t n) noexcept {                        \
    HYLE_DISPATCH(name, tag, hyle, a, b, dst, n);                           \
  }
HYLE_DEFINE_PUBLIC_GAP_UNARY(float, f32, erfc)
HYLE_DEFINE_PUBLIC_GAP_UNARY(double, f64, erfc)
HYLE_DEFINE_PUBLIC_GAP_UNARY(float, f32, exp10)
HYLE_DEFINE_PUBLIC_GAP_UNARY(double, f64, exp10)
HYLE_DEFINE_PUBLIC_GAP_BINARY(float, f32, fdim)
HYLE_DEFINE_PUBLIC_GAP_BINARY(double, f64, fdim)
#undef HYLE_DEFINE_PUBLIC_GAP_BINARY
#undef HYLE_DEFINE_PUBLIC_GAP_UNARY

// These operations have exact or cancellation-sensitive semantics for which
// the current vendor surface has no equivalent qualified vector operation.
// Keep the public capability and use the scalar oracle rather than preserving
// an unproven copied port.
#define HYLE_DEFINE_SCALAR_BINARY(type, tag, name)                            \
  void name##_##tag##_batch(const type* a, const type* b, type* dst,          \
                            std::size_t n) noexcept {                         \
    HYLE_SCALAR_KERNEL(name, tag)(a, b, dst, n);                             \
  }
HYLE_DEFINE_SCALAR_BINARY(float, f32, fmod)
HYLE_DEFINE_SCALAR_BINARY(double, f64, fmod)
HYLE_DEFINE_SCALAR_BINARY(float, f32, remainder)
HYLE_DEFINE_SCALAR_BINARY(double, f64, remainder)
HYLE_DEFINE_SCALAR_BINARY(float, f32, nextafter)
HYLE_DEFINE_SCALAR_BINARY(double, f64, nextafter)
#undef HYLE_DEFINE_SCALAR_BINARY

#define HYLE_DEFINE_PUBLIC_COMPLEX_UNARY(type, tag, name)                    \
  void name##_##tag##_batch(const type* src_re, const type* src_im,           \
                            type* dst_re, type* dst_im,                       \
                            std::size_t n) noexcept {                         \
    HYLE_DISPATCH(name, tag, hyle, src_re, src_im, dst_re, dst_im, n);        \
  }

#define HYLE_DEFINE_PUBLIC_COMPLEX_SET(type, tag) \
  HYLE_DEFINE_PUBLIC_COMPLEX_UNARY(type, tag, cexp)  \
  HYLE_DEFINE_PUBLIC_COMPLEX_UNARY(type, tag, clog)  \
  HYLE_DEFINE_PUBLIC_COMPLEX_UNARY(type, tag, csqrt) \
  HYLE_DEFINE_PUBLIC_COMPLEX_UNARY(type, tag, csin)  \
  HYLE_DEFINE_PUBLIC_COMPLEX_UNARY(type, tag, ccos)  \
  HYLE_DEFINE_PUBLIC_COMPLEX_UNARY(type, tag, ctan)  \
  HYLE_DEFINE_PUBLIC_COMPLEX_UNARY(type, tag, csinh) \
  HYLE_DEFINE_PUBLIC_COMPLEX_UNARY(type, tag, ccosh) \
  HYLE_DEFINE_PUBLIC_COMPLEX_UNARY(type, tag, ctanh)

HYLE_DEFINE_PUBLIC_COMPLEX_SET(float, f32)
HYLE_DEFINE_PUBLIC_COMPLEX_SET(double, f64)
#undef HYLE_DEFINE_PUBLIC_COMPLEX_SET
#undef HYLE_DEFINE_PUBLIC_COMPLEX_UNARY

void cpow_f32_batch(const float* z_re, const float* z_im, const float* w_re,
                    const float* w_im, float* dst_re, float* dst_im,
                    std::size_t n) noexcept {
  HYLE_DISPATCH(cpow, f32, hyle, z_re, z_im, w_re, w_im, dst_re, dst_im, n);
}
void cpow_f64_batch(const double* z_re, const double* z_im,
                    const double* w_re, const double* w_im, double* dst_re,
                    double* dst_im, std::size_t n) noexcept {
  HYLE_DISPATCH(cpow, f64, hyle, z_re, z_im, w_re, w_im, dst_re, dst_im, n);
}

#undef HYLE_DISPATCH
#undef HYLE_TRY_RVV
#undef HYLE_TRY_NEON
#undef HYLE_TRY_SVE
#undef HYLE_TRY_AVX2
#undef HYLE_TRY_AVX512
#undef HYLE_SCALAR_KERNEL
#undef HYLE_SCALAR_KERNEL_I
#undef HYLE_VARIANT_KERNEL_II
#undef HYLE_VARIANT_KERNEL_I

}  // namespace hyle::simd
