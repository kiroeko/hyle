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

}  // namespace hyle::simd::detail
