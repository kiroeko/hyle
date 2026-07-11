// Scalar libm transcendentals -- the always-available reference + fallback + ULP oracle.
// Plain std::exp/log/tanh per element. Compiled everywhere (no -march, no SLEEF).
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

// f64 scalar references (libm oracle + fallback for the double transcendental path -- CAE/solver).
void exp_f64_scalar(const double* src, double* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = std::exp(src[i]);
}
void log_f64_scalar(const double* src, double* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = std::log(src[i]);
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

// simd-09 scalar references (libm oracle + fallback for every new best-of function).
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
// xsimd-only borrows: scalar libm oracle + fallback (std::lgamma/tgamma per element).
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

// complex transcendentals: std::complex<float> per element -- the always-present oracle + fallback
// for the xsimd complex borrow cores. SoA planar in/out (re[i], im[i]).
namespace {
using cf = std::complex<float>;
template <cf (*Fn)(const cf&)>
void cunary(const float* sr, const float* si, float* dr, float* di, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) { cf r = Fn(cf(sr[i], si[i])); dr[i] = r.real(); di[i] = r.imag(); }
}
cf c_exp(const cf& z) { return std::exp(z); }
cf c_log(const cf& z) { return std::log(z); }
cf c_sqrt(const cf& z) { return std::sqrt(z); }
cf c_sin(const cf& z) { return std::sin(z); }
cf c_cos(const cf& z) { return std::cos(z); }
cf c_tanh(const cf& z) { return std::tanh(z); }
}  // namespace
void cexp_f32_scalar(const float* sr, const float* si, float* dr, float* di, std::size_t n) noexcept { cunary<c_exp>(sr, si, dr, di, n); }
void clog_f32_scalar(const float* sr, const float* si, float* dr, float* di, std::size_t n) noexcept { cunary<c_log>(sr, si, dr, di, n); }
void csqrt_f32_scalar(const float* sr, const float* si, float* dr, float* di, std::size_t n) noexcept { cunary<c_sqrt>(sr, si, dr, di, n); }
void csin_f32_scalar(const float* sr, const float* si, float* dr, float* di, std::size_t n) noexcept { cunary<c_sin>(sr, si, dr, di, n); }
void ccos_f32_scalar(const float* sr, const float* si, float* dr, float* di, std::size_t n) noexcept { cunary<c_cos>(sr, si, dr, di, n); }
void ctanh_f32_scalar(const float* sr, const float* si, float* dr, float* di, std::size_t n) noexcept { cunary<c_tanh>(sr, si, dr, di, n); }
void cpow_f32_scalar(const float* zr, const float* zi, const float* wr, const float* wi, float* dr, float* di, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) { cf r = std::pow(cf(zr[i], zi[i]), cf(wr[i], wi[i])); dr[i] = r.real(); di[i] = r.imag(); }
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
