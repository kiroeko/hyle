// Length-agnostic SVE/RVV product kernels for the HYLE D-tag port of Highway contrib/math.
// CMake compiles this source once for the host scalable ISA. The baseline dispatcher calls its
// suffixed symbols only after Runtime selected that ISA. Lanes(d) remains a per-call query.

#include "hyle/platform.h"

#if (defined(HYLE_ARCH_ARM64) && defined(HYLE_SIMD_HAS_SVE_VARIANT)) || \
    (defined(HYLE_ARCH_RISCV64) && defined(HYLE_SIMD_HAS_RVV_VARIANT))

#include "src/simd/math/kernels.h"

#include <cmath>
#include <complex>
#include <cstddef>

#if defined(HYLE_ARCH_ARM64) && !defined(HYLE_SIMD_TARGET_SVE)
#define HYLE_SIMD_TARGET_SVE
#elif defined(HYLE_ARCH_RISCV64) && !defined(HYLE_SIMD_TARGET_RVV)
#define HYLE_SIMD_TARGET_RVV
#endif
#include "hyle/simd/vec.h"
#include "src/simd/math/math_hyle-inl.h"
#include "src/simd/math/math_xsimd_hyle-inl.h"
#include "src/simd/math/complex_hyle-inl.h"
#include "src/simd/math/complex_typed_hyle-inl.h"
#include "src/simd/math/table_f64.h"

namespace hyle::simd::detail {
using namespace hyle::simd;
namespace {

#if defined(HYLE_ARCH_ARM64)
#define HYLE_SCALABLE_MATH_SUFFIX sve
#else
#define HYLE_SCALABLE_MATH_SUFFIX rvv
#endif
#define HYLE_SCALABLE_MATH_NAME_I(name, suffix) name##suffix
#define HYLE_SCALABLE_MATH_NAME_II(name, suffix) HYLE_SCALABLE_MATH_NAME_I(name, suffix)
#define HYLE_SCALABLE_MATH_NAME(name) \
  HYLE_SCALABLE_MATH_NAME_II(name, HYLE_SCALABLE_MATH_SUFFIX)
#define HYLE_SCALABLE_MATH_TYPED_I(name, tag, suffix) name##_##tag##_highway_##suffix
#define HYLE_SCALABLE_MATH_TYPED_II(name, tag, suffix) \
  HYLE_SCALABLE_MATH_TYPED_I(name, tag, suffix)
#define HYLE_SCALABLE_MATH_TYPED(name, tag) \
  HYLE_SCALABLE_MATH_TYPED_II(name, tag, HYLE_SCALABLE_MATH_SUFFIX)
#define HYLE_SCALABLE_HYLE_TYPED_I(name, tag, suffix) name##_##tag##_hyle_##suffix
#define HYLE_SCALABLE_HYLE_TYPED_II(name, tag, suffix) \
  HYLE_SCALABLE_HYLE_TYPED_I(name, tag, suffix)
#define HYLE_SCALABLE_HYLE_TYPED(name, tag) \
  HYLE_SCALABLE_HYLE_TYPED_II(name, tag, HYLE_SCALABLE_MATH_SUFFIX)

template <class D, class V>
bool exp_f64_needs_scalar(D d, V x) noexcept {
  const auto exceptional = Or(Or(IsNaN(x), IsInf(x)),
                              Or(Lt(x, Set(d, -700.0)), Gt(x, Set(d, 700.0))));
  return !AllFalse(d, exceptional);
}

template <class D, class V>
bool log_f64_needs_scalar(D d, V x) noexcept {
  const auto exceptional = Or(Or(IsNaN(x), IsInf(x)),
                              Or(Le(x, Zero(d)), Lt(x, Set(d, 0x1p-1022))));
  return !AllFalse(d, exceptional);
}

template <class T, class VecFn, class ScalarFn>
void unary_loop(const T* src, T* dst, std::size_t n, VecFn vec_fn,
                ScalarFn scalar_fn) noexcept {
  const ScalableTag<T> d;
  const std::size_t w = Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w) StoreU(vec_fn(d, LoadU(d, src + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = scalar_fn(src[i]);
}

template <class T, class VecFn, class ScalarFn>
void binary_loop(const T* a, const T* b, T* dst, std::size_t n, VecFn vec_fn,
                 ScalarFn scalar_fn) noexcept {
  const ScalableTag<T> d;
  const std::size_t w = Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w)
    StoreU(vec_fn(d, LoadU(d, a + i), LoadU(d, b + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = scalar_fn(a[i], b[i]);
}

// Exact one-ULP nextafter for the common finite-normal case. Boundary chunks remain on libm so
// NaN payload/quieting, signed zero, overflow/underflow exceptions and errno retain the platform
// contract. Unlike the old xsimd formula, the integer bit pattern moves exactly once.
template <class D>
HYLE_FORCE_INLINE VFromD<D> nextafter_normal(D d, VFromD<D> x,
                                             VFromD<D> y) noexcept {
  const RebindToUnsigned<D> du;
  const auto bits = BitCast(du, x);
  const auto one = Set(du, 1);
  const auto z = Zero(d);
  const auto toward_larger_bits =
      Or(And(Lt(x, y), Gt(x, z)), And(Gt(x, y), Lt(x, z)));
  return BitCast(d, IfThenElse(toward_larger_bits, Add(bits, one), Sub(bits, one)));
}

template <class T>
void nextafter_exact_loop(const T* a, const T* b, T* dst, std::size_t n) noexcept {
  const ScalableTag<T> d;
  const std::size_t w = Lanes(d);
  const auto min_normal = Set(d, std::numeric_limits<T>::min());
  const auto max_finite = Set(d, std::numeric_limits<T>::max());
  std::size_t i = 0;
  for (; i + w <= n; i += w) {
    const auto x = LoadU(d, a + i);
    const auto y = LoadU(d, b + i);
    const auto ax = Abs(x);
    const auto exceptional =
        Or(Or(Or(IsNaN(x), IsNaN(y)), IsInf(x)),
           Or(Or(Eq(x, y), Le(ax, min_normal)), Eq(ax, max_finite)));
    if (!AllFalse(d, exceptional)) {
      for (std::size_t lane = 0; lane < w; ++lane)
        dst[i + lane] = std::nextafter(a[i + lane], b[i + lane]);
    } else {
      StoreU(nextafter_normal(d, x, y), d, dst + i);
    }
  }
  for (; i < n; ++i) dst[i] = std::nextafter(a[i], b[i]);
}

}  // namespace

#define HYLE_SCALABLE_UNARY(type, tag, name, vec_fn, scalar_fn)                  \
  void HYLE_SCALABLE_MATH_TYPED(name, tag)(const type* src, type* dst,            \
                                            std::size_t n) noexcept {              \
    unary_loop(src, dst, n,                                                      \
               [](auto d, auto x) { return vec_fn(d, x); },                      \
               [](type x) { return scalar_fn(x); });                             \
  }

HYLE_SCALABLE_UNARY(float, f32, exp, Exp, std::exp)
HYLE_SCALABLE_UNARY(float, f32, log, Log, std::log)
HYLE_SCALABLE_UNARY(float, f32, tanh, Tanh, std::tanh)
HYLE_SCALABLE_UNARY(float, f32, sin, Sin, std::sin)
HYLE_SCALABLE_UNARY(float, f32, cos, Cos, std::cos)
HYLE_SCALABLE_UNARY(float, f32, exp2, Exp2, std::exp2)
HYLE_SCALABLE_UNARY(float, f32, log2, Log2, std::log2)
HYLE_SCALABLE_UNARY(float, f32, erf, Erf, std::erf)
HYLE_SCALABLE_UNARY(float, f32, atan, Atan, std::atan)
HYLE_SCALABLE_UNARY(float, f32, cbrt, Cbrt, std::cbrt)
HYLE_SCALABLE_UNARY(float, f32, expm1, Expm1, std::expm1)
HYLE_SCALABLE_UNARY(float, f32, log1p, Log1p, std::log1p)
HYLE_SCALABLE_UNARY(float, f32, asinh, Asinh, std::asinh)
HYLE_SCALABLE_UNARY(float, f32, acosh, Acosh, std::acosh)
HYLE_SCALABLE_UNARY(float, f32, acos, Acos, std::acos)
HYLE_SCALABLE_UNARY(float, f32, asin, Asin, std::asin)
HYLE_SCALABLE_UNARY(float, f32, sinh, Sinh, std::sinh)
HYLE_SCALABLE_UNARY(float, f32, cosh, Cosh, std::cosh)
HYLE_SCALABLE_UNARY(float, f32, tan, Tan, std::tan)
HYLE_SCALABLE_UNARY(float, f32, atanh, Atanh, std::atanh)
HYLE_SCALABLE_UNARY(float, f32, log10, Log10, std::log10)

HYLE_SCALABLE_UNARY(double, f64, tanh, Tanh, std::tanh)
HYLE_SCALABLE_UNARY(double, f64, sin, Sin, std::sin)
HYLE_SCALABLE_UNARY(double, f64, cos, Cos, std::cos)
HYLE_SCALABLE_UNARY(double, f64, exp2, Exp2, std::exp2)
HYLE_SCALABLE_UNARY(double, f64, log2, Log2, std::log2)
HYLE_SCALABLE_UNARY(double, f64, erf, Erf, std::erf)
HYLE_SCALABLE_UNARY(double, f64, atan, Atan, std::atan)
HYLE_SCALABLE_UNARY(double, f64, cbrt, Cbrt, std::cbrt)
HYLE_SCALABLE_UNARY(double, f64, expm1, Expm1, std::expm1)
HYLE_SCALABLE_UNARY(double, f64, log1p, Log1p, std::log1p)
HYLE_SCALABLE_UNARY(double, f64, asinh, Asinh, std::asinh)
HYLE_SCALABLE_UNARY(double, f64, acosh, Acosh, std::acosh)
HYLE_SCALABLE_UNARY(double, f64, acos, Acos, std::acos)
HYLE_SCALABLE_UNARY(double, f64, asin, Asin, std::asin)
HYLE_SCALABLE_UNARY(double, f64, sinh, Sinh, std::sinh)
HYLE_SCALABLE_UNARY(double, f64, cosh, Cosh, std::cosh)
HYLE_SCALABLE_UNARY(double, f64, tan, Tan, std::tan)
HYLE_SCALABLE_UNARY(double, f64, atanh, Atanh, std::atanh)
HYLE_SCALABLE_UNARY(double, f64, log10, Log10, std::log10)
#undef HYLE_SCALABLE_UNARY

void HYLE_SCALABLE_HYLE_TYPED(nextafter, f32)(const float* a, const float* b,
                                              float* dst, std::size_t n) noexcept {
  nextafter_exact_loop(a, b, dst, n);
}
void HYLE_SCALABLE_HYLE_TYPED(nextafter, f64)(const double* a, const double* b,
                                              double* dst, std::size_t n) noexcept {
  nextafter_exact_loop(a, b, dst, n);
}

void HYLE_SCALABLE_MATH_NAME(exp_f64_highway_)(const double* src, double* dst,
                                                std::size_t n) noexcept {
  const ScalableTag<double> d;
  const std::size_t w = Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w) {
    const auto x = LoadU(d, src + i);
    if (exp_f64_needs_scalar(d, x)) {
      for (std::size_t lane = 0; lane < w; ++lane) dst[i + lane] = std::exp(src[i + lane]);
    } else {
      StoreU(table_f64::ExpNormal(d, x), d, dst + i);
    }
  }
  for (; i < n; ++i) dst[i] = std::exp(src[i]);
}

void HYLE_SCALABLE_MATH_NAME(log_f64_highway_)(const double* src, double* dst,
                                                std::size_t n) noexcept {
  const ScalableTag<double> d;
  const std::size_t w = Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w) {
    const auto x = LoadU(d, src + i);
    if (log_f64_needs_scalar(d, x)) {
      for (std::size_t lane = 0; lane < w; ++lane) dst[i + lane] = std::log(src[i + lane]);
    } else {
      StoreU(table_f64::LogNormal(d, x), d, dst + i);
    }
  }
  for (; i < n; ++i) dst[i] = std::log(src[i]);
}

void HYLE_SCALABLE_MATH_NAME(exp_in_range_f64_highway_)(
    const double* src, double* dst, std::size_t n) noexcept {
  unary_loop(src, dst, n,
             [](auto d, auto x) { return table_f64::ExpNormal(d, x); },
             [](double x) { return std::exp(x); });
}

void HYLE_SCALABLE_MATH_NAME(log_in_range_f64_highway_)(
    const double* src, double* dst, std::size_t n) noexcept {
  unary_loop(src, dst, n,
             [](auto d, auto x) { return table_f64::LogNormal(d, x); },
             [](double x) { return std::log(x); });
}

#define HYLE_SCALABLE_BINARY(type, tag, name, vec_fn, scalar_fn)                  \
  void HYLE_SCALABLE_MATH_TYPED(name, tag)(const type* a, const type* b,           \
                                            type* dst, std::size_t n) noexcept {    \
    binary_loop(a, b, dst, n,                                                     \
                [](auto d, auto x, auto y) { return vec_fn(d, x, y); },            \
                [](type x, type y) { return scalar_fn(x, y); });                   \
  }
HYLE_SCALABLE_BINARY(float, f32, hypot, Hypot, std::hypot)
HYLE_SCALABLE_BINARY(float, f32, atan2, Atan2, std::atan2)
HYLE_SCALABLE_BINARY(float, f32, pow, Pow, std::pow)
HYLE_SCALABLE_BINARY(double, f64, hypot, Hypot, std::hypot)
HYLE_SCALABLE_BINARY(double, f64, atan2, Atan2, std::atan2)
HYLE_SCALABLE_BINARY(double, f64, pow, Pow, std::pow)
#undef HYLE_SCALABLE_BINARY

#define HYLE_SCALABLE_SINCOS(type, tag)                                            \
  void HYLE_SCALABLE_MATH_TYPED(sincos, tag)(const type* src, type* s, type* c,    \
                                              std::size_t n) noexcept {              \
    const ScalableTag<type> d;                                                     \
    const std::size_t w = Lanes(d);                                                \
    std::size_t i = 0;                                                             \
    for (; i + w <= n; i += w) {                                                   \
      VFromD<decltype(d)> vs, vc;                                                  \
      SinCos(d, LoadU(d, src + i), vs, vc);                                        \
      StoreU(vs, d, s + i);                                                        \
      StoreU(vc, d, c + i);                                                        \
    }                                                                              \
    for (; i < n; ++i) {                                                           \
      s[i] = std::sin(src[i]);                                                     \
      c[i] = std::cos(src[i]);                                                     \
    }                                                                              \
  }
HYLE_SCALABLE_SINCOS(float, f32)
HYLE_SCALABLE_SINCOS(double, f64)
#undef HYLE_SCALABLE_SINCOS

// HYLE-owned xsimd-derived bodies. These are D-tag algorithms, so scalable targets should not
// fall back merely because their historical symbol lived in an x86/NEON TU.
#define HYLE_SCALABLE_HYLE_UNARY(type, tag, name, vec_fn, scalar_fn)               \
  void HYLE_SCALABLE_HYLE_TYPED(name, tag)(const type* src, type* dst,              \
                                            std::size_t n) noexcept {                \
    unary_loop(src, dst, n,                                                        \
               [](auto d, auto x) { return xsimd_hyle::vec_fn(d, x); },            \
               [](type x) { return scalar_fn(x); });                               \
  }
HYLE_SCALABLE_HYLE_UNARY(float, f32, erfc, Erfc, std::erfc)
HYLE_SCALABLE_HYLE_UNARY(double, f64, erfc, Erfc, std::erfc)
HYLE_SCALABLE_HYLE_UNARY(float, f32, exp10, Exp10, [](float x) { return std::pow(10.0f, x); })
HYLE_SCALABLE_HYLE_UNARY(double, f64, exp10, Exp10, [](double x) { return std::pow(10.0, x); })
HYLE_SCALABLE_HYLE_UNARY(float, f32, tgamma, Tgamma, std::tgamma)
HYLE_SCALABLE_HYLE_UNARY(double, f64, tgamma, Tgamma, std::tgamma)
#undef HYLE_SCALABLE_HYLE_UNARY

#define HYLE_SCALABLE_HYLE_BINARY(type, tag, name, vec_fn, scalar_fn)              \
  void HYLE_SCALABLE_HYLE_TYPED(name, tag)(const type* a, const type* b, type* dst,  \
                                            std::size_t n) noexcept {                \
    binary_loop(a, b, dst, n,                                                      \
                [](auto d, auto x, auto y) { return xsimd_hyle::vec_fn(d, x, y); }, \
                [](type x, type y) { return scalar_fn(x, y); });                    \
  }
HYLE_SCALABLE_HYLE_BINARY(float, f32, fmod, Fmod, std::fmod)
HYLE_SCALABLE_HYLE_BINARY(double, f64, fmod, Fmod, std::fmod)
HYLE_SCALABLE_HYLE_BINARY(float, f32, fdim, Fdim, std::fdim)
HYLE_SCALABLE_HYLE_BINARY(double, f64, fdim, Fdim, std::fdim)
HYLE_SCALABLE_HYLE_BINARY(float, f32, hypot_in_range, HypotInRange,
                          [](float x, float y) { return std::sqrt(x * x + y * y); })
#undef HYLE_SCALABLE_HYLE_BINARY

namespace {
using ccf = std::complex<float>;

void cexp_loop(const float* sr, const float* si, float* dr, float* di,
               std::size_t n) noexcept {
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

void clog_loop(const float* sr, const float* si, float* dr, float* di,
               std::size_t n) noexcept {
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

void csqrt_loop(const float* sr, const float* si, float* dr, float* di,
                std::size_t n) noexcept {
  const ScalableTag<float> d;
  const std::size_t w = Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w) {
    const auto re = LoadU(d, sr + i);
    const auto im = LoadU(d, si + i);
    const auto rho = xsimd_hyle::HypotInRange(d, re, im);
    const auto t = Sqrt(Mul(Add(rho, Abs(re)), Set(d, 0.5f)));
    const auto two_t = Add(t, t);
    const auto safe_two_t = IfThenElse(Eq(t, Zero(d)), Set(d, 1.0f), two_t);
    const auto pos = Ge(re, Zero(d));
    const auto out_re = IfThenElse(pos, t, Div(Abs(im), safe_two_t));
    const auto out_im = IfThenElse(pos, Div(im, safe_two_t), CopySign(t, im));
    StoreU(out_re, d, dr + i);
    StoreU(out_im, d, di + i);
    const auto exceptional = Or(Or(IsNaN(re), IsNaN(im)), Or(IsInf(re), IsInf(im)));
    if (!AllFalse(d, exceptional) || !AllFalse(d, Eq(t, Zero(d)))) {
      for (std::size_t lane = 0; lane < w; ++lane) {
        if (!std::isfinite(sr[i + lane]) || !std::isfinite(si[i + lane]) ||
            (sr[i + lane] == 0.0f && si[i + lane] == 0.0f)) {
          const ccf r = std::sqrt(ccf(sr[i + lane], si[i + lane]));
          dr[i + lane] = r.real();
          di[i + lane] = r.imag();
        }
      }
    }
  }
  for (; i < n; ++i) {
    const ccf r = std::sqrt(ccf(sr[i], si[i]));
    dr[i] = r.real();
    di[i] = r.imag();
  }
}
}  // namespace

void HYLE_SCALABLE_HYLE_TYPED(cexp, f32)(const float* sr, const float* si, float* dr,
                                         float* di, std::size_t n) noexcept {
  cexp_loop(sr, si, dr, di, n);
}
void HYLE_SCALABLE_HYLE_TYPED(clog, f32)(const float* sr, const float* si, float* dr,
                                         float* di, std::size_t n) noexcept {
  clog_loop(sr, si, dr, di, n);
}
void HYLE_SCALABLE_HYLE_TYPED(csqrt, f32)(const float* sr, const float* si, float* dr,
                                          float* di, std::size_t n) noexcept {
  csqrt_loop(sr, si, dr, di, n);
}
void HYLE_SCALABLE_HYLE_TYPED(csin, f32)(const float* sr, const float* si, float* dr,
                                         float* di, std::size_t n) noexcept {
  complex_sincos_hyle<false>(sr, si, dr, di, n);
}
void HYLE_SCALABLE_HYLE_TYPED(ccos, f32)(const float* sr, const float* si, float* dr,
                                         float* di, std::size_t n) noexcept {
  complex_sincos_hyle<true>(sr, si, dr, di, n);
}
void HYLE_SCALABLE_HYLE_TYPED(ctan, f32)(const float* sr, const float* si, float* dr,
                                         float* di, std::size_t n) noexcept {
  complex_tan_typed_hyle(sr, si, dr, di, n);
}
void HYLE_SCALABLE_HYLE_TYPED(csinh, f32)(const float* sr, const float* si, float* dr,
                                          float* di, std::size_t n) noexcept {
  complex_sinhcosh_typed_hyle<float, false>(sr, si, dr, di, n);
}
void HYLE_SCALABLE_HYLE_TYPED(ccosh, f32)(const float* sr, const float* si, float* dr,
                                          float* di, std::size_t n) noexcept {
  complex_sinhcosh_typed_hyle<float, true>(sr, si, dr, di, n);
}
void HYLE_SCALABLE_HYLE_TYPED(ctanh, f32)(const float* sr, const float* si, float* dr,
                                          float* di, std::size_t n) noexcept {
  complex_ctanh_hyle(sr, si, dr, di, n);
}
void HYLE_SCALABLE_HYLE_TYPED(cpow, f32)(const float* zr, const float* zi, const float* wr,
                                         const float* wi, float* dr, float* di,
                                         std::size_t n) noexcept {
  complex_cpow_hyle(zr, zi, wr, wi, dr, di, n);
}

#define HYLE_DEFINE_SCALABLE_COMPLEX_F64(name, body)                               \
  void HYLE_SCALABLE_HYLE_TYPED(name, f64)(const double* sr, const double* si,      \
                                            double* dr, double* di,                 \
                                            std::size_t n) noexcept {               \
    body;                                                                           \
  }
HYLE_DEFINE_SCALABLE_COMPLEX_F64(cexp, complex_exp_typed_hyle(sr, si, dr, di, n))
HYLE_DEFINE_SCALABLE_COMPLEX_F64(clog, complex_log_typed_hyle(sr, si, dr, di, n))
HYLE_DEFINE_SCALABLE_COMPLEX_F64(csqrt, complex_sqrt_typed_hyle(sr, si, dr, di, n))
HYLE_DEFINE_SCALABLE_COMPLEX_F64(
    csin, (complex_sincos_typed_hyle<double, false>(sr, si, dr, di, n)))
HYLE_DEFINE_SCALABLE_COMPLEX_F64(
    ccos, (complex_sincos_typed_hyle<double, true>(sr, si, dr, di, n)))
HYLE_DEFINE_SCALABLE_COMPLEX_F64(ctan, complex_tan_typed_hyle(sr, si, dr, di, n))
HYLE_DEFINE_SCALABLE_COMPLEX_F64(
    csinh, (complex_sinhcosh_typed_hyle<double, false>(sr, si, dr, di, n)))
HYLE_DEFINE_SCALABLE_COMPLEX_F64(
    ccosh, (complex_sinhcosh_typed_hyle<double, true>(sr, si, dr, di, n)))
HYLE_DEFINE_SCALABLE_COMPLEX_F64(ctanh, complex_tanh_typed_hyle(sr, si, dr, di, n))
#undef HYLE_DEFINE_SCALABLE_COMPLEX_F64

void HYLE_SCALABLE_HYLE_TYPED(cpow, f64)(
    const double* zr, const double* zi, const double* wr, const double* wi, double* dr,
    double* di, std::size_t n) noexcept {
  complex_pow_typed_hyle(zr, zi, wr, wi, dr, di, n);
}

#undef HYLE_SCALABLE_HYLE_TYPED
#undef HYLE_SCALABLE_HYLE_TYPED_II
#undef HYLE_SCALABLE_HYLE_TYPED_I
#undef HYLE_SCALABLE_MATH_TYPED
#undef HYLE_SCALABLE_MATH_TYPED_II
#undef HYLE_SCALABLE_MATH_TYPED_I
#undef HYLE_SCALABLE_MATH_NAME
#undef HYLE_SCALABLE_MATH_NAME_II
#undef HYLE_SCALABLE_MATH_NAME_I
#undef HYLE_SCALABLE_MATH_SUFFIX

}  // namespace hyle::simd::detail

#endif
