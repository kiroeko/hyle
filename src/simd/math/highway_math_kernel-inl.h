// HYLE batch/SoA façade over the centrally locked Highway mathematics core.
//
// Include this file from one fixed-target TU after defining
// HYLE_HIGHWAY_MATH_SUFFIX. The TU must be compiled with the matching ISA flags
// and must define HWY_COMPILE_ONLY_STATIC before any Highway header is parsed.
// Highway vector types stay inside this implementation unit; the public API is
// pointer/count based and therefore identical for source, static and shared use.

#include "src/simd/math/kernels.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <type_traits>

#include "hwy/contrib/math/math-inl.h"
#include "src/simd/math/complex_hyle-inl.h"
#include "src/simd/math/complex_typed_hyle-inl.h"

#ifndef HYLE_HIGHWAY_MATH_SUFFIX
#error "Define HYLE_HIGHWAY_MATH_SUFFIX before including highway_math_kernel-inl.h"
#endif

namespace hyle_highway_math_detail {
using namespace hwy::HWY_NAMESPACE;

enum class UnaryDomain {
  // The locked Highway implementation owns the complete lane contract,
  // including NaN and infinity handling. Do not add a redundant horizontal
  // qualification branch around these operations.
  kDirect,
  kFinite,
  kExp,
  kExp2,
  kPositive,
  kLog1p,
  kUnitClosed,
  kUnitOpen,
  kAcosh,
  kTrig,
  kSinhCosh,
  kTgamma,
  kLogGamma,
};

template <class D, class V>
HWY_INLINE bool all_finite(const D d, const V x) noexcept {
  return AllFalse(d, Or(IsNaN(x), IsInf(x)));
}

template <UnaryDomain kDomain, typename T, class D, class V>
HWY_INLINE bool unary_domain_accepts(const D d, const V x) noexcept {
  if constexpr (kDomain == UnaryDomain::kDirect) {
    return true;
  } else {
    if (!all_finite(d, x)) return false;

    if constexpr (kDomain == UnaryDomain::kFinite) {
      return true;
    } else if constexpr (kDomain == UnaryDomain::kExp) {
      constexpr T kMax = std::is_same_v<T, float> ? T{104} : T{706};
      return AllTrue(d, Le(x, Set(d, kMax)));
    } else if constexpr (kDomain == UnaryDomain::kExp2) {
      constexpr T kMax = std::is_same_v<T, float> ? T{128} : T{1024};
      return AllTrue(d, Le(x, Set(d, kMax)));
    } else if constexpr (kDomain == UnaryDomain::kPositive) {
      return AllTrue(d, Gt(x, Zero(d)));
    } else if constexpr (kDomain == UnaryDomain::kLog1p) {
      // Highway currently documents Log1p for x >= 0. Preserve the broader
      // std::log1p contract by sending negative blocks to the scalar oracle.
      return AllTrue(d, Ge(x, Zero(d)));
    } else if constexpr (kDomain == UnaryDomain::kUnitClosed) {
      return AllTrue(d, Le(Abs(x), Set(d, T{1})));
    } else if constexpr (kDomain == UnaryDomain::kUnitOpen) {
      return AllTrue(d, Lt(Abs(x), Set(d, T{1})));
    } else if constexpr (kDomain == UnaryDomain::kAcosh) {
      return AllTrue(d, Ge(x, Set(d, T{1})));
    } else if constexpr (kDomain == UnaryDomain::kTrig) {
      return AllTrue(d, Le(Abs(x), Set(d, T{39000})));
    } else if constexpr (kDomain == UnaryDomain::kSinhCosh) {
      constexpr T kMax =
          std::is_same_v<T, float> ? T{88.7228f} : T{709};
      return AllTrue(d, Le(Abs(x), Set(d, kMax)));
    } else if constexpr (kDomain == UnaryDomain::kTgamma) {
      constexpr T kMax =
          std::is_same_v<T, float> ? T{35} : T{171.6};
      return AllTrue(d, And(Gt(x, Zero(d)), Le(x, Set(d, kMax))));
    } else {
      static_assert(kDomain == UnaryDomain::kLogGamma);
      return AllTrue(d, Gt(x, Zero(d)));
    }
  }
}

template <typename T, class ScalarFn>
HWY_INLINE void scalar_unary_chunk(const T* src, T* dst, std::size_t first,
                                   std::size_t count,
                                   ScalarFn&& scalar_fn) noexcept {
  for (std::size_t lane = 0; lane < count; ++lane) {
    dst[first + lane] = scalar_fn(src[first + lane]);
  }
}

template <typename T, class ScalarFn>
HWY_INLINE void scalar_binary_chunk(const T* a, const T* b, T* dst,
                                    std::size_t first, std::size_t count,
                                    ScalarFn&& scalar_fn) noexcept {
  for (std::size_t lane = 0; lane < count; ++lane) {
    dst[first + lane] = scalar_fn(a[first + lane], b[first + lane]);
  }
}

template <UnaryDomain kDomain, typename T, class VectorFn, class ScalarFn>
void unary_loop(const T* src, T* dst, std::size_t n, VectorFn&& vector_fn,
                ScalarFn&& scalar_fn) noexcept {
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  std::size_t i = 0;
  for (; i + width <= n; i += width) {
    const auto x = LoadU(d, src + i);
    if (unary_domain_accepts<kDomain, T>(d, x)) {
      StoreU(vector_fn(d, x), d, dst + i);
    } else {
      scalar_unary_chunk(src, dst, i, width, scalar_fn);
    }
  }
  scalar_unary_chunk(src, dst, i, n - i, scalar_fn);
}

template <typename T, class VectorFn, class ScalarFn>
void binary_loop(const T* a, const T* b, T* dst, std::size_t n,
                 VectorFn&& vector_fn, ScalarFn&& scalar_fn) noexcept {
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  std::size_t i = 0;
  for (; i + width <= n; i += width) {
    const auto va = LoadU(d, a + i);
    const auto vb = LoadU(d, b + i);
    StoreU(vector_fn(d, va, vb), d, dst + i);
  }
  scalar_binary_chunk(a, b, dst, i, n - i, scalar_fn);
}

template <typename T, class VectorFn, class ScalarFn>
void finite_binary_loop(const T* a, const T* b, T* dst, std::size_t n,
                        VectorFn&& vector_fn,
                        ScalarFn&& scalar_fn) noexcept {
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  std::size_t i = 0;
  for (; i + width <= n; i += width) {
    const auto va = LoadU(d, a + i);
    const auto vb = LoadU(d, b + i);
    if (all_finite(d, va) && all_finite(d, vb)) {
      StoreU(vector_fn(d, va, vb), d, dst + i);
    } else {
      scalar_binary_chunk(a, b, dst, i, width, scalar_fn);
    }
  }
  scalar_binary_chunk(a, b, dst, i, n - i, scalar_fn);
}

template <typename T>
void sincos_loop(const T* src, T* sine, T* cosine, std::size_t n) noexcept {
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  std::size_t i = 0;
  for (; i + width <= n; i += width) {
    const auto x = LoadU(d, src + i);
    if (unary_domain_accepts<UnaryDomain::kTrig, T>(d, x)) {
      auto s = Zero(d);
      auto c = Zero(d);
      SinCos(d, x, s, c);
      StoreU(s, d, sine + i);
      StoreU(c, d, cosine + i);
    } else {
      for (std::size_t lane = 0; lane < width; ++lane) {
        sine[i + lane] = std::sin(src[i + lane]);
        cosine[i + lane] = std::cos(src[i + lane]);
      }
    }
  }
  for (; i < n; ++i) {
    sine[i] = std::sin(src[i]);
    cosine[i] = std::cos(src[i]);
  }
}

template <typename T>
void exp10_loop(const T* src, T* dst, std::size_t n) noexcept {
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  constexpr T kLn10 = static_cast<T>(2.3025850929940456840179914546843642L);
  constexpr T kMaxExp = std::is_same_v<T, float> ? T{104} : T{706};
  std::size_t i = 0;
  for (; i + width <= n; i += width) {
    const auto x = LoadU(d, src + i);
    const auto exponent = Mul(x, Set(d, kLn10));
    if (all_finite(d, x) && AllTrue(d, Le(exponent, Set(d, kMaxExp)))) {
      StoreU(Exp(d, exponent), d, dst + i);
    } else {
      scalar_unary_chunk(src, dst, i, width,
                         [](T value) { return std::pow(T{10}, value); });
    }
  }
  scalar_unary_chunk(src, dst, i, n - i,
                     [](T value) { return std::pow(T{10}, value); });
}

template <typename T>
void erfc_loop(const T* src, T* dst, std::size_t n) noexcept {
  // 1-erf(x) is accurate around the origin, but cancellation dominates the
  // positive tail. Use the direct Highway core only for the qualified center
  // and preserve std::erfc semantics elsewhere.
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  std::size_t i = 0;
  for (; i + width <= n; i += width) {
    const auto x = LoadU(d, src + i);
    if (all_finite(d, x) && AllTrue(d, Le(Abs(x), Set(d, T{1})))) {
      StoreU(Sub(Set(d, T{1}), Erf(d, x)), d, dst + i);
    } else {
      scalar_unary_chunk(src, dst, i, width,
                         [](T value) { return std::erfc(value); });
    }
  }
  scalar_unary_chunk(src, dst, i, n - i,
                     [](T value) { return std::erfc(value); });
}

template <typename T>
void fdim_loop(const T* a, const T* b, T* dst, std::size_t n) noexcept {
  finite_binary_loop(
      a, b, dst, n,
      [](auto d, auto x, auto y) { return Max(Sub(x, y), Zero(d)); },
      [](T x, T y) { return std::fdim(x, y); });
}

}  // namespace hyle_highway_math_detail

namespace hyle::simd::detail {
namespace hm = ::hyle_highway_math_detail;

#define HYLE_HWY_SYMBOL_I(name, tag, suffix) name##_##tag##_highway_##suffix
#define HYLE_HWY_SYMBOL_II(name, tag, suffix) HYLE_HWY_SYMBOL_I(name, tag, suffix)
#define HYLE_HWY_SYMBOL(name, tag) \
  HYLE_HWY_SYMBOL_II(name, tag, HYLE_HIGHWAY_MATH_SUFFIX)
#define HYLE_OWNED_SYMBOL_I(name, tag, suffix) name##_##tag##_hyle_##suffix
#define HYLE_OWNED_SYMBOL_II(name, tag, suffix) HYLE_OWNED_SYMBOL_I(name, tag, suffix)
#define HYLE_OWNED_SYMBOL(name, tag) \
  HYLE_OWNED_SYMBOL_II(name, tag, HYLE_HIGHWAY_MATH_SUFFIX)

#define HYLE_DEFINE_DIRECT_UNARY(type, tag, name, hwy_name, scalar_name, domain)       \
  void HYLE_HWY_SYMBOL(name, tag)(const type* src, type* dst,                         \
                                  std::size_t n) noexcept {                           \
    hm::unary_loop<hm::UnaryDomain::domain>(                                          \
        src, dst, n,                                                                  \
        [](auto d, auto x) { return hwy::HWY_NAMESPACE::hwy_name(d, x); },            \
        [](type x) { return std::scalar_name(x); });                                   \
  }

#define HYLE_DEFINE_DIRECT_UNARY_SET(type, tag)                                       \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, exp, Exp, exp, kExp)                            \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, log, Log, log, kPositive)                       \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, tanh, Tanh, tanh, kDirect)                      \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, sin, Sin, sin, kTrig)                           \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, cos, Cos, cos, kTrig)                           \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, exp2, Exp2, exp2, kExp2)                        \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, log2, Log2, log2, kPositive)                    \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, erf, Erf, erf, kDirect)                         \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, atan, Atan, atan, kDirect)                      \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, cbrt, Cbrt, cbrt, kDirect)                      \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, expm1, Expm1, expm1, kExp)                      \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, log1p, Log1p, log1p, kLog1p)                    \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, asinh, Asinh, asinh, kFinite)                   \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, acosh, Acosh, acosh, kAcosh)                    \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, acos, Acos, acos, kUnitClosed)                  \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, asin, Asin, asin, kUnitClosed)                  \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, sinh, Sinh, sinh, kSinhCosh)                    \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, cosh, Cosh, cosh, kSinhCosh)                    \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, tan, Tan, tan, kTrig)                           \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, atanh, Atanh, atanh, kUnitOpen)                 \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, log10, Log10, log10, kPositive)                 \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, tgamma, Tgamma, tgamma, kTgamma)                \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, lgamma, LogGamma, lgamma, kLogGamma)

HYLE_DEFINE_DIRECT_UNARY_SET(float, f32)
HYLE_DEFINE_DIRECT_UNARY_SET(double, f64)
#undef HYLE_DEFINE_DIRECT_UNARY_SET
#undef HYLE_DEFINE_DIRECT_UNARY

#define HYLE_DEFINE_DIRECT_BINARY(type, tag, name, hwy_name, scalar_name)             \
  void HYLE_HWY_SYMBOL(name, tag)(const type* a, const type* b, type* dst,             \
                                  std::size_t n) noexcept {                            \
    hm::binary_loop(                                                                  \
        a, b, dst, n,                                                                 \
        [](auto d, auto x, auto y) { return hwy::HWY_NAMESPACE::hwy_name(d, x, y); }, \
        [](type x, type y) { return std::scalar_name(x, y); });                        \
  }

#define HYLE_DEFINE_DIRECT_BINARY_SET(type, tag)                                     \
  HYLE_DEFINE_DIRECT_BINARY(type, tag, atan2, Atan2, atan2)                          \
  HYLE_DEFINE_DIRECT_BINARY(type, tag, pow, Pow, pow)                                \
  HYLE_DEFINE_DIRECT_BINARY(type, tag, hypot, Hypot, hypot)

HYLE_DEFINE_DIRECT_BINARY_SET(float, f32)
HYLE_DEFINE_DIRECT_BINARY_SET(double, f64)
#undef HYLE_DEFINE_DIRECT_BINARY_SET
#undef HYLE_DEFINE_DIRECT_BINARY

void HYLE_HWY_SYMBOL(sincos, f32)(const float* src, float* sine, float* cosine,
                                  std::size_t n) noexcept {
  hm::sincos_loop(src, sine, cosine, n);
}
void HYLE_HWY_SYMBOL(sincos, f64)(const double* src, double* sine, double* cosine,
                                  std::size_t n) noexcept {
  hm::sincos_loop(src, sine, cosine, n);
}

void HYLE_HWY_SYMBOL(exp_in_range, f32)(const float* src, float* dst,
                                        std::size_t n) noexcept {
  hm::unary_loop<hm::UnaryDomain::kDirect>(
      src, dst, n,
      [](auto d, auto x) { return hwy::HWY_NAMESPACE::Exp(d, x); },
      [](float x) { return std::exp(x); });
}
void HYLE_HWY_SYMBOL(log_in_range, f32)(const float* src, float* dst,
                                        std::size_t n) noexcept {
  hm::unary_loop<hm::UnaryDomain::kDirect>(
      src, dst, n,
      [](auto d, auto x) { return hwy::HWY_NAMESPACE::Log(d, x); },
      [](float x) { return std::log(x); });
}

void HYLE_HWY_SYMBOL(exp_in_range, f64)(const double* src, double* dst,
                                        std::size_t n) noexcept {
  hm::unary_loop<hm::UnaryDomain::kDirect>(
      src, dst, n,
      [](auto d, auto x) { return hwy::HWY_NAMESPACE::Exp(d, x); },
      [](double x) { return std::exp(x); });
}
void HYLE_HWY_SYMBOL(log_in_range, f64)(const double* src, double* dst,
                                        std::size_t n) noexcept {
  hm::unary_loop<hm::UnaryDomain::kDirect>(
      src, dst, n,
      [](auto d, auto x) { return hwy::HWY_NAMESPACE::Log(d, x); },
      [](double x) { return std::log(x); });
}

void HYLE_OWNED_SYMBOL(hypot_in_range, f32)(const float* a, const float* b,
                                            float* dst,
                                            std::size_t n) noexcept {
  hm::binary_loop(
      a, b, dst, n,
      [](auto, auto x, auto y) {
        return hwy::HWY_NAMESPACE::Sqrt(
            hwy::HWY_NAMESPACE::MulAdd(x, x, hwy::HWY_NAMESPACE::Mul(y, y)));
      },
      [](float x, float y) { return std::sqrt(std::fma(x, x, y * y)); });
}

#define HYLE_DEFINE_OWNED_GAPS(type, tag)                                         \
  void HYLE_OWNED_SYMBOL(erfc, tag)(const type* src, type* dst,                   \
                                    std::size_t n) noexcept {                     \
    hm::erfc_loop(src, dst, n);                                                   \
  }                                                                               \
  void HYLE_OWNED_SYMBOL(exp10, tag)(const type* src, type* dst,                  \
                                     std::size_t n) noexcept {                    \
    hm::exp10_loop(src, dst, n);                                                  \
  }                                                                               \
  void HYLE_OWNED_SYMBOL(fdim, tag)(const type* a, const type* b, type* dst,       \
                                    std::size_t n) noexcept {                     \
    hm::fdim_loop(a, b, dst, n);                                                  \
  }

HYLE_DEFINE_OWNED_GAPS(float, f32)
HYLE_DEFINE_OWNED_GAPS(double, f64)
#undef HYLE_DEFINE_OWNED_GAPS

#define HYLE_DEFINE_COMPLEX_UNARY(type, tag, name, body)                          \
  void HYLE_OWNED_SYMBOL(name, tag)(const type* src_re, const type* src_im,        \
                                    type* dst_re, type* dst_im,                    \
                                    std::size_t n) noexcept {                     \
    body;                                                                         \
  }

HYLE_DEFINE_COMPLEX_UNARY(float, f32, cexp,
                          hm::complex_exp_typed_hyle(src_re, src_im, dst_re, dst_im, n))
HYLE_DEFINE_COMPLEX_UNARY(float, f32, clog,
                          hm::complex_log_typed_hyle(src_re, src_im, dst_re, dst_im, n))
HYLE_DEFINE_COMPLEX_UNARY(float, f32, csqrt,
                          hm::complex_sqrt_typed_hyle(src_re, src_im, dst_re, dst_im, n))
HYLE_DEFINE_COMPLEX_UNARY(float, f32, csin,
                          hm::complex_sincos_hyle<false>(src_re, src_im, dst_re, dst_im, n))
HYLE_DEFINE_COMPLEX_UNARY(float, f32, ccos,
                          hm::complex_sincos_hyle<true>(src_re, src_im, dst_re, dst_im, n))
HYLE_DEFINE_COMPLEX_UNARY(float, f32, ctan,
                          hm::complex_tan_typed_hyle(src_re, src_im, dst_re, dst_im, n))
HYLE_DEFINE_COMPLEX_UNARY(float, f32, csinh,
                          (hm::complex_sinhcosh_typed_hyle<float, false>(
                              src_re, src_im, dst_re, dst_im, n)))
HYLE_DEFINE_COMPLEX_UNARY(float, f32, ccosh,
                          (hm::complex_sinhcosh_typed_hyle<float, true>(
                              src_re, src_im, dst_re, dst_im, n)))
HYLE_DEFINE_COMPLEX_UNARY(float, f32, ctanh,
                          hm::complex_ctanh_hyle(src_re, src_im, dst_re, dst_im, n))

HYLE_DEFINE_COMPLEX_UNARY(double, f64, cexp,
                          hm::complex_exp_typed_hyle(src_re, src_im, dst_re, dst_im, n))
HYLE_DEFINE_COMPLEX_UNARY(double, f64, clog,
                          hm::complex_log_typed_hyle(src_re, src_im, dst_re, dst_im, n))
HYLE_DEFINE_COMPLEX_UNARY(double, f64, csqrt,
                          hm::complex_sqrt_typed_hyle(src_re, src_im, dst_re, dst_im, n))
HYLE_DEFINE_COMPLEX_UNARY(double, f64, csin,
                          (hm::complex_sincos_typed_hyle<double, false>(
                              src_re, src_im, dst_re, dst_im, n)))
HYLE_DEFINE_COMPLEX_UNARY(double, f64, ccos,
                          (hm::complex_sincos_typed_hyle<double, true>(
                              src_re, src_im, dst_re, dst_im, n)))
HYLE_DEFINE_COMPLEX_UNARY(double, f64, ctan,
                          hm::complex_tan_typed_hyle(src_re, src_im, dst_re, dst_im, n))
HYLE_DEFINE_COMPLEX_UNARY(double, f64, csinh,
                          (hm::complex_sinhcosh_typed_hyle<double, false>(
                              src_re, src_im, dst_re, dst_im, n)))
HYLE_DEFINE_COMPLEX_UNARY(double, f64, ccosh,
                          (hm::complex_sinhcosh_typed_hyle<double, true>(
                              src_re, src_im, dst_re, dst_im, n)))
HYLE_DEFINE_COMPLEX_UNARY(double, f64, ctanh,
                          hm::complex_tanh_typed_hyle(src_re, src_im, dst_re, dst_im, n))
#undef HYLE_DEFINE_COMPLEX_UNARY

void HYLE_OWNED_SYMBOL(cpow, f32)(const float* z_re, const float* z_im,
                                  const float* w_re, const float* w_im,
                                  float* dst_re, float* dst_im,
                                  std::size_t n) noexcept {
  hm::complex_cpow_hyle(z_re, z_im, w_re, w_im, dst_re, dst_im, n);
}
void HYLE_OWNED_SYMBOL(cpow, f64)(const double* z_re, const double* z_im,
                                  const double* w_re, const double* w_im,
                                  double* dst_re, double* dst_im,
                                  std::size_t n) noexcept {
  hm::complex_pow_typed_hyle(z_re, z_im, w_re, w_im, dst_re, dst_im, n);
}

#undef HYLE_OWNED_SYMBOL
#undef HYLE_OWNED_SYMBOL_II
#undef HYLE_OWNED_SYMBOL_I
#undef HYLE_HWY_SYMBOL
#undef HYLE_HWY_SYMBOL_II
#undef HYLE_HWY_SYMBOL_I

}  // namespace hyle::simd::detail
