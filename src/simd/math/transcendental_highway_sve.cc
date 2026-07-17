// Length-agnostic SVE product kernels for the HYLE D-tag port of Highway contrib/math.
// This TU is compiled only for Linux ARM64 with +sve; the baseline dispatcher calls it only when
// runtime detection selected SVE. Lanes(d) remains a per-call query because SVE VL is per-thread.

#include "hyle/platform.h"

#if defined(HYLE_ARCH_ARM64) && defined(HYLE_SIMD_HAS_SVE_VARIANT)

#include "src/simd/math/kernels.h"

#include <cmath>
#include <cstddef>

#define HYLE_SIMD_TARGET_SVE
#include "hyle/simd/vec.h"
#include "src/simd/math/math_hyle-inl.h"
#include "src/simd/math/table_f64.h"

namespace hyle::simd::detail {
using namespace hyle::simd;
namespace {

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

}  // namespace

#define HYLE_SVE_UNARY(type, tag, name, vec_fn, scalar_fn)                       \
  void name##_##tag##_highway_sve(const type* src, type* dst,                    \
                                  std::size_t n) noexcept {                       \
    unary_loop(src, dst, n,                                                      \
               [](auto d, auto x) { return vec_fn(d, x); },                      \
               [](type x) { return scalar_fn(x); });                             \
  }

HYLE_SVE_UNARY(float, f32, exp, Exp, std::exp)
HYLE_SVE_UNARY(float, f32, log, Log, std::log)
HYLE_SVE_UNARY(float, f32, tanh, Tanh, std::tanh)
HYLE_SVE_UNARY(float, f32, sin, Sin, std::sin)
HYLE_SVE_UNARY(float, f32, cos, Cos, std::cos)
HYLE_SVE_UNARY(float, f32, exp2, Exp2, std::exp2)
HYLE_SVE_UNARY(float, f32, log2, Log2, std::log2)
HYLE_SVE_UNARY(float, f32, erf, Erf, std::erf)
HYLE_SVE_UNARY(float, f32, atan, Atan, std::atan)
HYLE_SVE_UNARY(float, f32, cbrt, Cbrt, std::cbrt)
HYLE_SVE_UNARY(float, f32, expm1, Expm1, std::expm1)
HYLE_SVE_UNARY(float, f32, log1p, Log1p, std::log1p)
HYLE_SVE_UNARY(float, f32, asinh, Asinh, std::asinh)
HYLE_SVE_UNARY(float, f32, acosh, Acosh, std::acosh)
HYLE_SVE_UNARY(float, f32, acos, Acos, std::acos)
HYLE_SVE_UNARY(float, f32, asin, Asin, std::asin)
HYLE_SVE_UNARY(float, f32, sinh, Sinh, std::sinh)
HYLE_SVE_UNARY(float, f32, cosh, Cosh, std::cosh)
HYLE_SVE_UNARY(float, f32, tan, Tan, std::tan)
HYLE_SVE_UNARY(float, f32, atanh, Atanh, std::atanh)
HYLE_SVE_UNARY(float, f32, log10, Log10, std::log10)

HYLE_SVE_UNARY(double, f64, tanh, Tanh, std::tanh)
HYLE_SVE_UNARY(double, f64, sin, Sin, std::sin)
HYLE_SVE_UNARY(double, f64, cos, Cos, std::cos)
HYLE_SVE_UNARY(double, f64, exp2, Exp2, std::exp2)
HYLE_SVE_UNARY(double, f64, log2, Log2, std::log2)
HYLE_SVE_UNARY(double, f64, erf, Erf, std::erf)
HYLE_SVE_UNARY(double, f64, atan, Atan, std::atan)
HYLE_SVE_UNARY(double, f64, cbrt, Cbrt, std::cbrt)
HYLE_SVE_UNARY(double, f64, expm1, Expm1, std::expm1)
HYLE_SVE_UNARY(double, f64, log1p, Log1p, std::log1p)
HYLE_SVE_UNARY(double, f64, asinh, Asinh, std::asinh)
HYLE_SVE_UNARY(double, f64, acosh, Acosh, std::acosh)
HYLE_SVE_UNARY(double, f64, acos, Acos, std::acos)
HYLE_SVE_UNARY(double, f64, asin, Asin, std::asin)
HYLE_SVE_UNARY(double, f64, sinh, Sinh, std::sinh)
HYLE_SVE_UNARY(double, f64, cosh, Cosh, std::cosh)
HYLE_SVE_UNARY(double, f64, tan, Tan, std::tan)
HYLE_SVE_UNARY(double, f64, atanh, Atanh, std::atanh)
HYLE_SVE_UNARY(double, f64, log10, Log10, std::log10)
#undef HYLE_SVE_UNARY

void exp_f64_highway_sve(const double* src, double* dst, std::size_t n) noexcept {
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

void log_f64_highway_sve(const double* src, double* dst, std::size_t n) noexcept {
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

void exp_in_range_f64_highway_sve(const double* src, double* dst,
                                  std::size_t n) noexcept {
  unary_loop(src, dst, n,
             [](auto d, auto x) { return table_f64::ExpNormal(d, x); },
             [](double x) { return std::exp(x); });
}

void log_in_range_f64_highway_sve(const double* src, double* dst,
                                  std::size_t n) noexcept {
  unary_loop(src, dst, n,
             [](auto d, auto x) { return table_f64::LogNormal(d, x); },
             [](double x) { return std::log(x); });
}

#define HYLE_SVE_BINARY(type, tag, name, vec_fn, scalar_fn)                       \
  void name##_##tag##_highway_sve(const type* a, const type* b, type* dst,         \
                                  std::size_t n) noexcept {                        \
    binary_loop(a, b, dst, n,                                                     \
                [](auto d, auto x, auto y) { return vec_fn(d, x, y); },            \
                [](type x, type y) { return scalar_fn(x, y); });                   \
  }
HYLE_SVE_BINARY(float, f32, hypot, Hypot, std::hypot)
HYLE_SVE_BINARY(float, f32, atan2, Atan2, std::atan2)
HYLE_SVE_BINARY(float, f32, pow, Pow, std::pow)
HYLE_SVE_BINARY(double, f64, hypot, Hypot, std::hypot)
HYLE_SVE_BINARY(double, f64, atan2, Atan2, std::atan2)
HYLE_SVE_BINARY(double, f64, pow, Pow, std::pow)
#undef HYLE_SVE_BINARY

#define HYLE_SVE_SINCOS(type, tag)                                                 \
  void sincos_##tag##_highway_sve(const type* src, type* s, type* c,              \
                                  std::size_t n) noexcept {                        \
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
HYLE_SVE_SINCOS(float, f32)
HYLE_SVE_SINCOS(double, f64)
#undef HYLE_SVE_SINCOS

}  // namespace hyle::simd::detail

#endif
