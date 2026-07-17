// NEON (aarch64, 4-wide) HYLE-ported xsimd product kernels + complex SoA.
// Historical role: tgamma/hypot_in_range + complex (Highway has no gamma/complex).
#include "hyle/platform.h"

#if defined(HYLE_ARCH_ARM64)

#include "src/simd/math/kernels.h"

#include <cmath>
#include <complex>
#include <cstddef>

#define HYLE_SIMD_TARGET_NEON
#include "hyle/simd/vec.h"
#include "src/simd/math/math_xsimd_hyle-inl.h"
#include "src/simd/math/math_xsimd_body.inc"
#include "src/simd/math/complex_hyle-inl.h"
#include "src/simd/math/complex_typed_hyle-inl.h"

namespace hyle::simd::detail {
using namespace hyle::simd;
using namespace xsimd_hyle;
using xsimd_body::unary_loop;
using xsimd_body::binary_loop;

void tgamma_f32_xsimd_neon(const float* src, float* dst, std::size_t n) noexcept {
  unary_loop(src, dst, n, [](auto d, auto v) { return xsimd_hyle::Tgamma(d, v); },
             [](float x) { return std::tgamma(x); });
}
void hypot_in_range_f32_xsimd_neon(const float* a, const float* b, float* dst,
                                   std::size_t n) noexcept {
  binary_loop(a, b, dst, n, [](auto d, auto x, auto y) { return xsimd_hyle::HypotInRange(d, x, y); },
              [](float x, float y) { return std::sqrt(x * x + y * y); });
}

void csin_f32_hyle_neon(const float* sr, const float* si, float* dr, float* di,
                        std::size_t n) noexcept {
  complex_sincos_hyle<false>(sr, si, dr, di, n);
}
void ccos_f32_hyle_neon(const float* sr, const float* si, float* dr, float* di,
                        std::size_t n) noexcept {
  complex_sincos_hyle<true>(sr, si, dr, di, n);
}
void ctan_f32_hyle_neon(const float* sr, const float* si, float* dr, float* di,
                        std::size_t n) noexcept {
  complex_tan_typed_hyle(sr, si, dr, di, n);
}
void csinh_f32_hyle_neon(const float* sr, const float* si, float* dr, float* di,
                         std::size_t n) noexcept {
  complex_sinhcosh_typed_hyle<float, false>(sr, si, dr, di, n);
}
void ccosh_f32_hyle_neon(const float* sr, const float* si, float* dr, float* di,
                         std::size_t n) noexcept {
  complex_sinhcosh_typed_hyle<float, true>(sr, si, dr, di, n);
}
void ctanh_f32_hyle_neon(const float* sr, const float* si, float* dr, float* di,
                         std::size_t n) noexcept {
  complex_ctanh_hyle(sr, si, dr, di, n);
}
void cpow_f32_hyle_neon(const float* zr, const float* zi, const float* wr, const float* wi,
                        float* dr, float* di, std::size_t n) noexcept {
  complex_cpow_hyle(zr, zi, wr, wi, dr, di, n);
}

namespace {
using ccf = std::complex<float>;

void cexp_loop(const float* sr, const float* si, float* dr, float* di, std::size_t n) noexcept {
  const ScalableTag<float> d;
  const std::size_t w = Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w) {
    const auto re = LoadU(d, sr + i);
    const auto im = LoadU(d, si + i);
    const auto mag = xsimd_hyle::Exp(d, re);
    auto s = Zero(d), c = Zero(d);
    SinCos(d, im, s, c);
    StoreU(Mul(mag, c), d, dr + i);
    StoreU(Mul(mag, s), d, di + i);
  }
  for (; i < n; ++i) {
    const ccf r = std::exp(ccf(sr[i], si[i]));
    dr[i] = r.real();
    di[i] = r.imag();
  }
}
void clog_loop(const float* sr, const float* si, float* dr, float* di, std::size_t n) noexcept {
  const ScalableTag<float> d;
  const std::size_t w = Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w) {
    const auto re = LoadU(d, sr + i);
    const auto im = LoadU(d, si + i);
    StoreU(Log(d, xsimd_hyle::HypotInRange(d, re, im)), d, dr + i);
    StoreU(Atan2(d, im, re), d, di + i);
  }
  for (; i < n; ++i) {
    const ccf r = std::log(ccf(sr[i], si[i]));
    dr[i] = r.real();
    di[i] = r.imag();
  }
}
void csqrt_loop(const float* sr, const float* si, float* dr, float* di, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) {
    const ccf r = std::sqrt(ccf(sr[i], si[i]));
    dr[i] = r.real();
    di[i] = r.imag();
  }
}
}  // namespace

void cexp_f32_hyle_neon(const float* sr, const float* si, float* dr, float* di,
                        std::size_t n) noexcept {
  cexp_loop(sr, si, dr, di, n);
}
void clog_f32_hyle_neon(const float* sr, const float* si, float* dr, float* di,
                        std::size_t n) noexcept {
  clog_loop(sr, si, dr, di, n);
}
void csqrt_f32_hyle_neon(const float* sr, const float* si, float* dr, float* di,
                         std::size_t n) noexcept {
  csqrt_loop(sr, si, dr, di, n);
}

#define HYLE_DEFINE_COMPLEX_F64_NEON(name, body)                                   \
  void name##_f64_hyle_neon(const double* sr, const double* si, double* dr,         \
                             double* di, std::size_t n) noexcept {                   \
    body;                                                                           \
  }
HYLE_DEFINE_COMPLEX_F64_NEON(cexp, complex_exp_typed_hyle(sr, si, dr, di, n))
HYLE_DEFINE_COMPLEX_F64_NEON(clog, complex_log_typed_hyle(sr, si, dr, di, n))
HYLE_DEFINE_COMPLEX_F64_NEON(csqrt, complex_sqrt_typed_hyle(sr, si, dr, di, n))
HYLE_DEFINE_COMPLEX_F64_NEON(
    csin, (complex_sincos_typed_hyle<double, false>(sr, si, dr, di, n)))
HYLE_DEFINE_COMPLEX_F64_NEON(
    ccos, (complex_sincos_typed_hyle<double, true>(sr, si, dr, di, n)))
HYLE_DEFINE_COMPLEX_F64_NEON(ctan, complex_tan_typed_hyle(sr, si, dr, di, n))
HYLE_DEFINE_COMPLEX_F64_NEON(
    csinh, (complex_sinhcosh_typed_hyle<double, false>(sr, si, dr, di, n)))
HYLE_DEFINE_COMPLEX_F64_NEON(
    ccosh, (complex_sinhcosh_typed_hyle<double, true>(sr, si, dr, di, n)))
HYLE_DEFINE_COMPLEX_F64_NEON(ctanh, complex_tanh_typed_hyle(sr, si, dr, di, n))
#undef HYLE_DEFINE_COMPLEX_F64_NEON

void cpow_f64_hyle_neon(const double* zr, const double* zi, const double* wr,
                        const double* wi, double* dr, double* di, std::size_t n) noexcept {
  complex_pow_typed_hyle(zr, zi, wr, wi, dr, di, n);
}

}  // namespace hyle::simd::detail

#endif  // HYLE_ARCH_ARM64
