// xsimd-algorithm AVX2 transcendentals (8-wide), now HYLE D-tag product kernels
// (math_xsimd_hyle-inl.h). Symbol names keep the historical *_xsimd_avx2 suffix used by
// math_dispatch / benches. Algorithms/constants from xsimd SHA 5d2490f.
#include "hyle/platform.h"  // HYLE_ARCH_X64 -- MUST precede the guard below

#if defined(HYLE_ARCH_X64)

#include "src/simd/math/kernels.h"

#include <cmath>
#include <complex>
#include <cstddef>

#define HYLE_SIMD_TARGET_AVX2
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

void exp_f32_xsimd_avx2(const float* src, float* dst, std::size_t n) noexcept {
  unary_loop(src, dst, n, [](auto d, auto v) { return xsimd_hyle::Exp(d, v); },
             [](float x) { return std::exp(x); });
}
void tanh_f32_xsimd_avx2(const float* src, float* dst, std::size_t n) noexcept {
  unary_loop(src, dst, n, [](auto d, auto v) { return xsimd_hyle::Tanh(d, v); },
             [](float x) { return std::tanh(x); });
}

void exp_f64_xsimd_avx2(const double* src, double* dst, std::size_t n) noexcept {
  unary_loop(src, dst, n, [](auto d, auto v) { return xsimd_hyle::Exp(d, v); },
             [](double x) { return std::exp(x); });
}
void log_f64_xsimd_avx2(const double* src, double* dst, std::size_t n) noexcept {
  // Log remains the Highway product winner; keep the symbol for dispatch/benches.
  unary_loop(src, dst, n, [](auto d, auto v) { return Log(d, v); },
             [](double x) { return std::log(x); });
}
void tanh_f64_xsimd_avx2(const double* src, double* dst, std::size_t n) noexcept {
  unary_loop(src, dst, n, [](auto d, auto v) { return xsimd_hyle::Tanh(d, v); },
             [](double x) { return std::tanh(x); });
}
void erf_f64_xsimd_avx2(const double* src, double* dst, std::size_t n) noexcept {
  unary_loop(src, dst, n, [](auto d, auto v) { return xsimd_hyle::Erf(d, v); },
             [](double x) { return std::erf(x); });
}
void tgamma_f64_xsimd_avx2(const double* src, double* dst, std::size_t n) noexcept {
  unary_loop(src, dst, n, [](auto d, auto v) { return xsimd_hyle::Tgamma(d, v); },
             [](double x) { return std::tgamma(x); });
}

void exp2_f32_xsimd_avx2(const float* src, float* dst, std::size_t n) noexcept {
  unary_loop(src, dst, n, [](auto d, auto v) { return xsimd_hyle::Exp2(d, v); },
             [](float x) { return std::exp2(x); });
}
void erf_f32_xsimd_avx2(const float* src, float* dst, std::size_t n) noexcept {
  unary_loop(src, dst, n, [](auto d, auto v) { return xsimd_hyle::Erf(d, v); },
             [](float x) { return std::erf(x); });
}
void tgamma_f32_xsimd_avx2(const float* src, float* dst, std::size_t n) noexcept {
  unary_loop(src, dst, n, [](auto d, auto v) { return xsimd_hyle::Tgamma(d, v); },
             [](float x) { return std::tgamma(x); });
}

void hypot_in_range_f32_xsimd_avx2(const float* a, const float* b, float* dst,
                                   std::size_t n) noexcept {
  binary_loop(a, b, dst, n, [](auto d, auto x, auto y) { return xsimd_hyle::HypotInRange(d, x, y); },
              [](float x, float y) { return std::sqrt(x * x + y * y); });
}

void erfc_f32_xsimd_avx2(const float* src, float* dst, std::size_t n) noexcept {
  unary_loop(src, dst, n, [](auto d, auto v) { return xsimd_hyle::Erfc(d, v); },
             [](float x) { return std::erfc(x); });
}
void exp10_f32_xsimd_avx2(const float* src, float* dst, std::size_t n) noexcept {
  unary_loop(src, dst, n, [](auto d, auto v) { return xsimd_hyle::Exp10(d, v); },
             [](float x) { return std::pow(10.0f, x); });
}
void fmod_f32_xsimd_avx2(const float* a, const float* b, float* dst, std::size_t n) noexcept {
  binary_loop(a, b, dst, n, [](auto d, auto x, auto y) { return xsimd_hyle::Fmod(d, x, y); },
              [](float x, float y) { return std::fmod(x, y); });
}
void remainder_f32_xsimd_avx2(const float* a, const float* b, float* dst, std::size_t n) noexcept {
  // simd-20: remainder stays scalar (not an xsimd product port target).
  for (std::size_t i = 0; i < n; ++i) dst[i] = std::remainder(a[i], b[i]);
}
void nextafter_f32_xsimd_avx2(const float* a, const float* b, float* dst, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = std::nextafter(a[i], b[i]);
}
void fdim_f32_xsimd_avx2(const float* a, const float* b, float* dst, std::size_t n) noexcept {
  binary_loop(a, b, dst, n, [](auto d, auto x, auto y) { return xsimd_hyle::Fdim(d, x, y); },
              [](float x, float y) { return std::fdim(x, y); });
}
void erfc_f64_xsimd_avx2(const double* src, double* dst, std::size_t n) noexcept {
  unary_loop(src, dst, n, [](auto d, auto v) { return xsimd_hyle::Erfc(d, v); },
             [](double x) { return std::erfc(x); });
}
void exp10_f64_xsimd_avx2(const double* src, double* dst, std::size_t n) noexcept {
  unary_loop(src, dst, n, [](auto d, auto v) { return xsimd_hyle::Exp10(d, v); },
             [](double x) { return std::pow(10.0, x); });
}
void fmod_f64_xsimd_avx2(const double* a, const double* b, double* dst, std::size_t n) noexcept {
  binary_loop(a, b, dst, n, [](auto d, auto x, auto y) { return xsimd_hyle::Fmod(d, x, y); },
              [](double x, double y) { return std::fmod(x, y); });
}
void remainder_f64_xsimd_avx2(const double* a, const double* b, double* dst,
                              std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = std::remainder(a[i], b[i]);
}
void nextafter_f64_xsimd_avx2(const double* a, const double* b, double* dst,
                              std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) dst[i] = std::nextafter(a[i], b[i]);
}
void fdim_f64_xsimd_avx2(const double* a, const double* b, double* dst, std::size_t n) noexcept {
  binary_loop(a, b, dst, n, [](auto d, auto x, auto y) { return xsimd_hyle::Fdim(d, x, y); },
              [](double x, double y) { return std::fdim(x, y); });
}

// ---- complex SoA ------------------------------------------------------------
// cexp/clog/csqrt: thin SoA loops using HYLE math (no xsimd batch<complex>).
// csin/ccos/ctanh/cpow: HYLE-owned complex_hyle-inl.h (also pure HYLE now).

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
  // Branch-cut-correct principal sqrt via magnitude + half-angle; exceptional lanes use libm.
  const ScalableTag<float> d;
  const std::size_t w = Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w) {
    const auto re = LoadU(d, sr + i);
    const auto im = LoadU(d, si + i);
    const auto rho = xsimd_hyle::HypotInRange(d, re, im);
    const auto half = Set(d, 0.5f);
    // t = sqrt((rho + |re|) / 2)
    const auto t = Sqrt(Mul(Add(rho, Abs(re)), half));
    // if re >= 0: (t, im/(2t)); else: (|im|/(2t), copysign(t, im))
    const auto two_t = Add(t, t);
    const auto safe_t = IfThenElse(Eq(t, Zero(d)), Set(d, 1.0f), t);
    const auto safe_two_t = IfThenElse(Eq(t, Zero(d)), Set(d, 1.0f), two_t);
    const auto pos = Ge(re, Zero(d));
    const auto out_re = IfThenElse(pos, t, Div(Abs(im), safe_two_t));
    const auto out_im = IfThenElse(pos, Div(im, safe_two_t), CopySign(t, im));
    // zero/inf/nan lanes: scalar fixup below if any exceptional
    const auto exceptional =
        Or(Or(IsNaN(re), IsNaN(im)), Or(IsInf(re), IsInf(im)));
    StoreU(out_re, d, dr + i);
    StoreU(out_im, d, di + i);
    if (!AllFalse(d, exceptional) || !AllFalse(d, Eq(t, Zero(d)))) {
      for (std::size_t lane = 0; lane < w; ++lane) {
        const ccf r = std::sqrt(ccf(sr[i + lane], si[i + lane]));
        // Only rewrite lanes that may be wrong under the fast formula.
        if (!std::isfinite(sr[i + lane]) || !std::isfinite(si[i + lane]) ||
            (sr[i + lane] == 0.0f && si[i + lane] == 0.0f)) {
          dr[i + lane] = r.real();
          di[i + lane] = r.imag();
        }
      }
    }
    (void)safe_t;
  }
  for (; i < n; ++i) {
    const ccf r = std::sqrt(ccf(sr[i], si[i]));
    dr[i] = r.real();
    di[i] = r.imag();
  }
}
}  // namespace

void cexp_f32_xsimd_avx2(const float* sr, const float* si, float* dr, float* di,
                         std::size_t n) noexcept {
  cexp_loop(sr, si, dr, di, n);
}
void clog_f32_xsimd_avx2(const float* sr, const float* si, float* dr, float* di,
                         std::size_t n) noexcept {
  clog_loop(sr, si, dr, di, n);
}
void csqrt_f32_xsimd_avx2(const float* sr, const float* si, float* dr, float* di,
                          std::size_t n) noexcept {
  csqrt_loop(sr, si, dr, di, n);
}
void csin_f32_hyle_avx2(const float* sr, const float* si, float* dr, float* di,
                        std::size_t n) noexcept {
  complex_sincos_hyle<false>(sr, si, dr, di, n);
}
void ccos_f32_hyle_avx2(const float* sr, const float* si, float* dr, float* di,
                        std::size_t n) noexcept {
  complex_sincos_hyle<true>(sr, si, dr, di, n);
}
void ctan_f32_hyle_avx2(const float* sr, const float* si, float* dr, float* di,
                        std::size_t n) noexcept {
  complex_tan_typed_hyle(sr, si, dr, di, n);
}
void csinh_f32_hyle_avx2(const float* sr, const float* si, float* dr, float* di,
                         std::size_t n) noexcept {
  complex_sinhcosh_typed_hyle<float, false>(sr, si, dr, di, n);
}
void ccosh_f32_hyle_avx2(const float* sr, const float* si, float* dr, float* di,
                         std::size_t n) noexcept {
  complex_sinhcosh_typed_hyle<float, true>(sr, si, dr, di, n);
}
void ctanh_f32_hyle_avx2(const float* sr, const float* si, float* dr, float* di,
                         std::size_t n) noexcept {
  complex_ctanh_hyle(sr, si, dr, di, n);
}
void ctanh_f32_xsimd_avx2(const float* sr, const float* si, float* dr, float* di,
                          std::size_t n) noexcept {
  // Historical xsimd-control symbol: same HYLE SoA path (zero-xsimd product).
  complex_ctanh_hyle(sr, si, dr, di, n);
}
void cpow_f32_hyle_avx2(const float* zr, const float* zi, const float* wr, const float* wi,
                        float* dr, float* di, std::size_t n) noexcept {
  complex_cpow_hyle(zr, zi, wr, wi, dr, di, n);
}
void cpow_f32_xsimd_avx2(const float* zr, const float* zi, const float* wr, const float* wi,
                         float* dr, float* di, std::size_t n) noexcept {
  complex_cpow_hyle(zr, zi, wr, wi, dr, di, n);
}

#define HYLE_DEFINE_COMPLEX_F64_AVX2(name, body)                                    \
  void name##_f64_hyle_avx2(const double* sr, const double* si, double* dr,          \
                             double* di, std::size_t n) noexcept {                    \
    body;                                                                            \
  }
HYLE_DEFINE_COMPLEX_F64_AVX2(cexp, complex_exp_typed_hyle(sr, si, dr, di, n))
HYLE_DEFINE_COMPLEX_F64_AVX2(clog, complex_log_typed_hyle(sr, si, dr, di, n))
HYLE_DEFINE_COMPLEX_F64_AVX2(csqrt, complex_sqrt_typed_hyle(sr, si, dr, di, n))
HYLE_DEFINE_COMPLEX_F64_AVX2(csin,
                             (complex_sincos_typed_hyle<double, false>(sr, si, dr, di, n)))
HYLE_DEFINE_COMPLEX_F64_AVX2(ccos,
                             (complex_sincos_typed_hyle<double, true>(sr, si, dr, di, n)))
HYLE_DEFINE_COMPLEX_F64_AVX2(ctan, complex_tan_typed_hyle(sr, si, dr, di, n))
HYLE_DEFINE_COMPLEX_F64_AVX2(csinh,
                             (complex_sinhcosh_typed_hyle<double, false>(sr, si, dr, di, n)))
HYLE_DEFINE_COMPLEX_F64_AVX2(ccosh,
                             (complex_sinhcosh_typed_hyle<double, true>(sr, si, dr, di, n)))
HYLE_DEFINE_COMPLEX_F64_AVX2(ctanh, complex_tanh_typed_hyle(sr, si, dr, di, n))
#undef HYLE_DEFINE_COMPLEX_F64_AVX2

void cpow_f64_hyle_avx2(const double* zr, const double* zi, const double* wr,
                        const double* wi, double* dr, double* di, std::size_t n) noexcept {
  complex_pow_typed_hyle(zr, zi, wr, wi, dr, di, n);
}

}  // namespace hyle::simd::detail

#endif  // HYLE_ARCH_X64
