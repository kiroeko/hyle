// Scalar libm transcendentals -- the always-available reference + fallback + ULP oracle.
// Plain std:: math per element. Compiled everywhere without target-specific ISA flags.
#include "src/simd/math/kernels.h"

#include <cmath>
#include <complex>
#include <cstddef>

namespace hyle::simd::detail {

void exp_f32_scalar(const float* src, float* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = std::exp(src[i]);
}
void log_f32_scalar(const float* src, float* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = std::log(src[i]);
}
void tanh_f32_scalar(const float* src, float* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = std::tanh(src[i]);
}
void exp_in_range_f32_scalar(const float* src, float* dst,
                             std::size_t n) noexcept {
  exp_f32_scalar(src, dst, n);
}
void log_in_range_f32_scalar(const float* src, float* dst,
                             std::size_t n) noexcept {
  log_f32_scalar(src, dst, n);
}

// f64 scalar references (libm oracle + fallback for the double transcendental path -- CAE/solver).
void exp_f64_scalar(const double* src, double* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = std::exp(src[i]);
}
void log_f64_scalar(const double* src, double* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = std::log(src[i]);
}
void exp_in_range_f64_scalar(const double* src, double* dst,
                             std::size_t n) noexcept {
  exp_f64_scalar(src, dst, n);
}
void log_in_range_f64_scalar(const double* src, double* dst,
                             std::size_t n) noexcept {
  log_f64_scalar(src, dst, n);
}
void tanh_f64_scalar(const double* src, double* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = std::tanh(src[i]);
}
// f64 scalar references for the rest of the double transcendentals (oracle + fallback floor).
#define HYLE_F64_SCALAR(name, libmfn)                                            \
  void name##_f64_scalar(const double* src, double* dst, std::size_t n) noexcept { \
    for (std::size_t i = 0; i < n; ++i) dst[i] = libmfn(src[i]);                  \
  }
HYLE_F64_SCALAR(sin, std::sin)
HYLE_F64_SCALAR(cos, std::cos)
HYLE_F64_SCALAR(exp2, std::exp2)
HYLE_F64_SCALAR(log2, std::log2)
HYLE_F64_SCALAR(erf, std::erf)
HYLE_F64_SCALAR(atan, std::atan)
HYLE_F64_SCALAR(cbrt, std::cbrt)
HYLE_F64_SCALAR(expm1, std::expm1)
HYLE_F64_SCALAR(log1p, std::log1p)
HYLE_F64_SCALAR(asinh, std::asinh)
HYLE_F64_SCALAR(acosh, std::acosh)
HYLE_F64_SCALAR(tgamma, std::tgamma)
HYLE_F64_SCALAR(lgamma, std::lgamma)
#undef HYLE_F64_SCALAR
void hypot_f64_scalar(const double* a, const double* b, double* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = std::hypot(a[i], b[i]);
}

// Scalar references for the remaining real functions.
void sin_f32_scalar(const float* src, float* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = std::sin(src[i]);
}
void cos_f32_scalar(const float* src, float* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = std::cos(src[i]);
}
void exp2_f32_scalar(const float* src, float* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = std::exp2(src[i]);
}
void log2_f32_scalar(const float* src, float* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = std::log2(src[i]);
}
void erf_f32_scalar(const float* src, float* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = std::erf(src[i]);
}
void atan_f32_scalar(const float* src, float* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = std::atan(src[i]);
}
void cbrt_f32_scalar(const float* src, float* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = std::cbrt(src[i]);
}
void expm1_f32_scalar(const float* src, float* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = std::expm1(src[i]);
}
void log1p_f32_scalar(const float* src, float* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = std::log1p(src[i]);
}
void asinh_f32_scalar(const float* src, float* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = std::asinh(src[i]);
}
void acosh_f32_scalar(const float* src, float* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = std::acosh(src[i]);
}
// Scalar semantic floor for operations that lack an equivalent qualified vector primitive.
void lgamma_f32_scalar(const float* src, float* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = std::lgamma(src[i]);
}
void tgamma_f32_scalar(const float* src, float* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = std::tgamma(src[i]);
}
// hypot: BINARY (two inputs). scalar libm oracle + fallback. std::hypot is overflow-safe.
void hypot_f32_scalar(const float* a, const float* b, float* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = std::hypot(a[i], b[i]);
}
void hypot_in_range_f32_scalar(const float* a, const float* b, float* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = std::sqrt(a[i] * a[i] + b[i] * b[i]);
}

// complex transcendentals: std::complex per element -- the always-present oracle + fallback.
// SoA planar in/out (re[i], im[i]).
namespace {
template <typename T, class Fn>
void cunary(const T* sr, const T* si, T* dr, T* di, std::size_t n, Fn&& fn) noexcept {
  for (std::size_t i = 0; i < n; ++i) {
    const std::complex<T> r = fn(std::complex<T>(sr[i], si[i]));
    dr[i] = r.real();
    di[i] = r.imag();
  }
}
template <typename T>
void cpow_scalar(const T* zr, const T* zi, const T* wr, const T* wi, T* dr, T* di,
                 std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) {
    const std::complex<T> r =
        std::pow(std::complex<T>(zr[i], zi[i]), std::complex<T>(wr[i], wi[i]));
    dr[i] = r.real();
    di[i] = r.imag();
  }
}
}  // namespace
#define HYLE_DEFINE_COMPLEX_SCALAR_UNARY(type, tag, name, fn)                       \
  void name##_##tag##_scalar(const type* sr, const type* si, type* dr, type* di,    \
                             std::size_t n) noexcept {                               \
    cunary(sr, si, dr, di, n, [](std::complex<type> z) { return std::fn(z); });      \
  }
#define HYLE_DEFINE_COMPLEX_SCALAR_SET(type, tag)       \
  HYLE_DEFINE_COMPLEX_SCALAR_UNARY(type, tag, cexp, exp)   \
  HYLE_DEFINE_COMPLEX_SCALAR_UNARY(type, tag, clog, log)   \
  HYLE_DEFINE_COMPLEX_SCALAR_UNARY(type, tag, csqrt, sqrt) \
  HYLE_DEFINE_COMPLEX_SCALAR_UNARY(type, tag, csin, sin)   \
  HYLE_DEFINE_COMPLEX_SCALAR_UNARY(type, tag, ccos, cos)   \
  HYLE_DEFINE_COMPLEX_SCALAR_UNARY(type, tag, ctan, tan)   \
  HYLE_DEFINE_COMPLEX_SCALAR_UNARY(type, tag, csinh, sinh) \
  HYLE_DEFINE_COMPLEX_SCALAR_UNARY(type, tag, ccosh, cosh) \
  HYLE_DEFINE_COMPLEX_SCALAR_UNARY(type, tag, ctanh, tanh)
HYLE_DEFINE_COMPLEX_SCALAR_SET(float, f32)
HYLE_DEFINE_COMPLEX_SCALAR_SET(double, f64)
#undef HYLE_DEFINE_COMPLEX_SCALAR_SET
#undef HYLE_DEFINE_COMPLEX_SCALAR_UNARY

void cpow_f32_scalar(const float* zr, const float* zi, const float* wr, const float* wi,
                     float* dr, float* di, std::size_t n) noexcept {
  cpow_scalar(zr, zi, wr, wi, dr, di, n);
}
void cpow_f64_scalar(const double* zr, const double* zi, const double* wr, const double* wi,
                     double* dr, double* di, std::size_t n) noexcept {
  cpow_scalar(zr, zi, wr, wi, dr, di, n);
}


// simd-18 批G scalar oracles
void acos_f32_scalar(const float* src, float* dst, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) dst[i]=std::acos(src[i]); }
void asin_f32_scalar(const float* src, float* dst, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) dst[i]=std::asin(src[i]); }
void sinh_f32_scalar(const float* src, float* dst, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) dst[i]=std::sinh(src[i]); }
void cosh_f32_scalar(const float* src, float* dst, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) dst[i]=std::cosh(src[i]); }
void tan_f32_scalar(const float* src, float* dst, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) dst[i]=std::tan(src[i]); }
void atanh_f32_scalar(const float* src, float* dst, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) dst[i]=std::atanh(src[i]); }
void log10_f32_scalar(const float* src, float* dst, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) dst[i]=std::log10(src[i]); }
void atan2_f32_scalar(const float* a, const float* b, float* dst, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) dst[i]=std::atan2(a[i], b[i]); }
void pow_f32_scalar(const float* a, const float* b, float* dst, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) dst[i]=std::pow(a[i], b[i]); }
void sincos_f32_scalar(const float* src, float* s, float* c, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) { s[i]=std::sin(src[i]); c[i]=std::cos(src[i]); } }
void erfc_f32_scalar(const float* src, float* dst, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) dst[i]=std::erfc(src[i]); }
void exp10_f32_scalar(const float* src, float* dst, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) dst[i]=std::pow(10.f, src[i]); }
void fmod_f32_scalar(const float* a, const float* b, float* dst, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) dst[i]=std::fmod(a[i], b[i]); }
void remainder_f32_scalar(const float* a, const float* b, float* dst, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) dst[i]=std::remainder(a[i], b[i]); }
void nextafter_f32_scalar(const float* a, const float* b, float* dst, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) dst[i]=std::nextafter(a[i], b[i]); }
void fdim_f32_scalar(const float* a, const float* b, float* dst, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) dst[i]=std::fdim(a[i], b[i]); }
void acos_f64_scalar(const double* src, double* dst, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) dst[i]=std::acos(src[i]); }
void asin_f64_scalar(const double* src, double* dst, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) dst[i]=std::asin(src[i]); }
void sinh_f64_scalar(const double* src, double* dst, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) dst[i]=std::sinh(src[i]); }
void cosh_f64_scalar(const double* src, double* dst, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) dst[i]=std::cosh(src[i]); }
void tan_f64_scalar(const double* src, double* dst, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) dst[i]=std::tan(src[i]); }
void atanh_f64_scalar(const double* src, double* dst, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) dst[i]=std::atanh(src[i]); }
void log10_f64_scalar(const double* src, double* dst, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) dst[i]=std::log10(src[i]); }
void atan2_f64_scalar(const double* a, const double* b, double* dst, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) dst[i]=std::atan2(a[i], b[i]); }
void pow_f64_scalar(const double* a, const double* b, double* dst, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) dst[i]=std::pow(a[i], b[i]); }
void sincos_f64_scalar(const double* src, double* s, double* c, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) { s[i]=std::sin(src[i]); c[i]=std::cos(src[i]); } }
void erfc_f64_scalar(const double* src, double* dst, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) dst[i]=std::erfc(src[i]); }
void exp10_f64_scalar(const double* src, double* dst, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) dst[i]=std::pow(10.0, src[i]); }
void fmod_f64_scalar(const double* a, const double* b, double* dst, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) dst[i]=std::fmod(a[i], b[i]); }
void remainder_f64_scalar(const double* a, const double* b, double* dst, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) dst[i]=std::remainder(a[i], b[i]); }
void nextafter_f64_scalar(const double* a, const double* b, double* dst, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) dst[i]=std::nextafter(a[i], b[i]); }
void fdim_f64_scalar(const double* a, const double* b, double* dst, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) dst[i]=std::fdim(a[i], b[i]); }


}  // namespace hyle::simd::detail
