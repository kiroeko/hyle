// HYLE batch/SoA façade over the centrally locked Highway mathematics core.
//
// Include this file from one fixed-target TU after defining
// HYLE_HIGHWAY_MATH_SUFFIX. The TU must be compiled with the matching ISA flags
// and must define HWY_COMPILE_ONLY_STATIC before any Highway header is parsed.
// Highway vector types stay inside this implementation unit; the public API is
// pointer/count based and therefore identical for source, static and shared
// use.

#include "src/simd/math/kernels.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "hwy/contrib/math/math-inl.h"
#include "hyle/simd/dtype.h"
#include "hyle/simd/provider/route.h"
#include "src/simd/dtype/highway_narrow.h"

// Only the currently qualified AVX2/classic-MSVC family route needs xsimd in
// this body. Other target TUs do not parse or instantiate an unused vendor
// stack; a future route must deliberately widen this availability guard.
#if defined(HYLE_SIMD_TARGET_AVX2) && defined(_MSC_VER) && !defined(__clang__)
#include "xsimd/xsimd.hpp"
#define HYLE_SIMD_MATH_HAS_XSIMD_BODY 1
#endif

#ifndef HYLE_HIGHWAY_MATH_SUFFIX
#error \
    "Define HYLE_HIGHWAY_MATH_SUFFIX before including highway_math_kernel-inl.h"
#endif

#define HYLE_HWY_DETAIL_NAMESPACE_I(suffix) hyle_highway_math_detail_##suffix
#define HYLE_HWY_DETAIL_NAMESPACE_II(suffix) HYLE_HWY_DETAIL_NAMESPACE_I(suffix)
#define HYLE_HWY_DETAIL_NAMESPACE \
  HYLE_HWY_DETAIL_NAMESPACE_II(HYLE_HIGHWAY_MATH_SUFFIX)
#define HYLE_HIGHWAY_MATH_DETAIL_NAMESPACE HYLE_HWY_DETAIL_NAMESPACE

#include "src/simd/math/complex_hyle-inl.h"
#include "src/simd/math/complex_typed_hyle-inl.h"

// The detail namespace is target-qualified. This is not cosmetic: the same
// header is instantiated into several ISA TUs, and target-dependent template
// instantiations must never be COMDAT-folded across those TUs.
namespace HYLE_HIGHWAY_MATH_DETAIL_NAMESPACE {
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
      constexpr T kMax = std::is_same_v<T, float> ? T{88.7228f} : T{709};
      return AllTrue(d, Le(Abs(x), Set(d, kMax)));
    } else if constexpr (kDomain == UnaryDomain::kTgamma) {
      constexpr T kMax = std::is_same_v<T, float> ? T{35} : T{171.6};
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
                        VectorFn&& vector_fn, ScalarFn&& scalar_fn) noexcept {
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

constexpr float kActivationSqrt2Inv = 0.70710678118654752440f;
constexpr float kActivationSqrt2OverPi = 0.79788456080286535588f;
constexpr float kActivationGeluCoefficient = 0.044715f;
constexpr float kActivationGeluQuickScale = 1.702f;

HWY_INLINE float activation_sigmoid_scalar(float x) noexcept {
  if (std::isnan(x)) return x;
  const float e = std::exp(-std::abs(x));
  return x < 0.0f ? e / (1.0f + e) : 1.0f / (1.0f + e);
}

HWY_INLINE float activation_gelu_scalar(float x) noexcept {
  if (x == -std::numeric_limits<float>::infinity()) return 0.0f;
  return 0.5f * x * (1.0f + std::erf(x * kActivationSqrt2Inv));
}

HWY_INLINE float activation_gelu_tanh_scalar(float x) noexcept {
  if (x == -std::numeric_limits<float>::infinity()) return 0.0f;
  const float inner = kActivationSqrt2OverPi * x *
                      std::fma(kActivationGeluCoefficient, x * x, 1.0f);
  return 0.5f * x * (1.0f + std::tanh(inner));
}

HWY_INLINE float activation_gelu_quick_scalar(float x) noexcept {
  if (x == -std::numeric_limits<float>::infinity()) return 0.0f;
  return x * activation_sigmoid_scalar(kActivationGeluQuickScale * x);
}

HWY_INLINE float activation_silu_scalar(float x) noexcept {
  if (x == -std::numeric_limits<float>::infinity()) return 0.0f;
  return x * activation_sigmoid_scalar(x);
}

HWY_INLINE float activation_relu_scalar(float x) noexcept {
  if (std::isnan(x)) return x;
  return x > 0.0f ? x : 0.0f;
}

HWY_INLINE float activation_leaky_relu_scalar(float x,
                                              float negative_slope) noexcept {
  if (std::isnan(x)) return x;
  return x >= 0.0f ? x : negative_slope * x;
}

HWY_INLINE float activation_elu_scalar(float x, float alpha) noexcept {
  if (std::isnan(x)) return x;
  return x >= 0.0f ? x : alpha * std::expm1(x);
}

HWY_INLINE float activation_hard_sigmoid_scalar(float x) noexcept {
  if (std::isnan(x)) return x;
  const float scaled = (x + 3.0f) / 6.0f;
  if (scaled <= 0.0f) return 0.0f;
  return scaled >= 1.0f ? 1.0f : scaled;
}

HWY_INLINE float activation_hard_swish_scalar(float x) noexcept {
  if (std::isnan(x)) return x;
  if (x <= -3.0f) return 0.0f;
  if (x >= 3.0f) return x;
  return x * ((x + 3.0f) / 6.0f);
}

template <class D, class V>
HWY_INLINE V activation_gelu(const D d, const V x) noexcept {
  const auto result =
      Mul(Mul(Set(d, 0.5f), x),
          Add(Set(d, 1.0f), Erf(d, Mul(x, Set(d, kActivationSqrt2Inv)))));
  const auto negative_infinity = And(IsInf(x), Lt(x, Zero(d)));
  return IfThenElse(negative_infinity, Zero(d), result);
}

template <class D, class V>
HWY_INLINE V activation_gelu_tanh(const D d, const V x) noexcept {
  const auto one = Set(d, 1.0f);
  const auto inner =
      Mul(Mul(Set(d, kActivationSqrt2OverPi), x),
          MulAdd(Set(d, kActivationGeluCoefficient), Mul(x, x), one));
  const auto result = Mul(Mul(Set(d, 0.5f), x), Add(one, Tanh(d, inner)));
  const auto negative_infinity = And(IsInf(x), Lt(x, Zero(d)));
  return IfThenElse(negative_infinity, Zero(d), result);
}

template <class D, class V>
HWY_INLINE V activation_gelu_quick(const D d, const V x) noexcept {
  const auto scaled = Mul(Set(d, kActivationGeluQuickScale), x);
  const auto one = Set(d, 1.0f);
  const auto e = Exp(d, Neg(Abs(scaled)));
  const auto denominator = Add(one, e);
  const auto sigmoid = IfThenElse(Lt(scaled, Zero(d)), Div(e, denominator),
                                  Div(one, denominator));
  const auto result = Mul(x, sigmoid);
  const auto negative_infinity = And(IsInf(x), Lt(x, Zero(d)));
  return IfThenElse(negative_infinity, Zero(d), result);
}

template <class D, class V>
HWY_INLINE V activation_sigmoid(const D d, const V x) noexcept {
  const auto one = Set(d, 1.0f);
  const auto e = Exp(d, Neg(Abs(x)));
  const auto denominator = Add(one, e);
  return IfThenElse(Lt(x, Zero(d)), Div(e, denominator), Div(one, denominator));
}

template <class D, class V>
HWY_INLINE V activation_silu(const D d, const V x) noexcept {
  const auto result = Mul(x, activation_sigmoid(d, x));
  const auto negative_infinity = And(IsInf(x), Lt(x, Zero(d)));
  return IfThenElse(negative_infinity, Zero(d), result);
}

template <class D, class V>
HWY_INLINE V activation_relu(const D d, const V x) noexcept {
  const auto result = IfThenElse(Gt(x, Zero(d)), x, Zero(d));
  return IfThenElse(IsNaN(x), x, result);
}

template <class D, class V>
HWY_INLINE V activation_leaky_relu(const D d, const V x,
                                   float negative_slope) noexcept {
  const auto result =
      IfThenElse(Ge(x, Zero(d)), x, Mul(Set(d, negative_slope), x));
  return IfThenElse(IsNaN(x), x, result);
}

template <class D, class V>
HWY_INLINE V activation_elu(const D d, const V x, float alpha) noexcept {
  // Clamp the inactive lanes to zero before Expm1 so positive infinity never
  // enters a math approximation whose result will be discarded.
  const auto negative = Mul(Set(d, alpha), Expm1(d, Min(x, Zero(d))));
  const auto result = IfThenElse(Ge(x, Zero(d)), x, negative);
  return IfThenElse(IsNaN(x), x, result);
}

template <class D, class V>
HWY_INLINE V activation_hard_sigmoid(const D d, const V x) noexcept {
  const auto scaled = Mul(Add(x, Set(d, 3.0f)), Set(d, 1.0f / 6.0f));
  const auto result = Min(Set(d, 1.0f), Max(Zero(d), scaled));
  return IfThenElse(IsNaN(x), x, result);
}

template <class D, class V>
HWY_INLINE V activation_hard_swish(const D d, const V x) noexcept {
  const auto middle = Mul(x, Mul(Add(x, Set(d, 3.0f)), Set(d, 1.0f / 6.0f)));
  const auto piecewise = IfThenElse(Le(x, Set(d, -3.0f)), Zero(d),
                                    IfThenElse(Ge(x, Set(d, 3.0f)), x, middle));
  return IfThenElse(IsNaN(x), x, piecewise);
}

template <typename Storage>
HWY_INLINE float storage_to_f32(Storage value) noexcept {
  if constexpr (std::is_same_v<Storage, float>) {
    return value;
  } else if constexpr (std::is_same_v<Storage, hyle::simd::float16_t>) {
    return hyle::simd::f16_to_f32(value);
  } else {
    static_assert(std::is_same_v<Storage, hyle::simd::bfloat16_t>);
    return hyle::simd::bf16_to_f32(value);
  }
}

template <typename Storage>
HWY_INLINE Storage f32_to_storage(float value) noexcept {
  if constexpr (std::is_same_v<Storage, float>) {
    return value;
  } else if constexpr (std::is_same_v<Storage, hyle::simd::float16_t>) {
    return hyle::simd::f32_to_f16(value);
  } else {
    static_assert(std::is_same_v<Storage, hyle::simd::bfloat16_t>);
    return hyle::simd::f32_to_bf16(value);
  }
}

template <typename Storage, typename HNarrow, class D>
HWY_INLINE auto load_storage_as_f32(D d, const Storage* source) noexcept {
  if constexpr (std::is_same_v<Storage, float>) {
    return LoadU(d, source);
  } else {
    return hyle::simd::detail::highway_narrow::load_as_f32<HNarrow>(d, source);
  }
}

template <typename Storage, typename HNarrow, class D, class V>
HWY_INLINE void store_f32_as_storage(D d, V value,
                                     Storage* destination) noexcept {
  if constexpr (std::is_same_v<Storage, float>) {
    StoreU(value, d, destination);
  } else {
    hyle::simd::detail::highway_narrow::store_from_f32<HNarrow>(d, value,
                                                                destination);
  }
}

template <typename Storage, typename HNarrow, class VectorFn, class ScalarFn>
void activation_storage_loop(const Storage* src, Storage* dst, std::size_t n,
                             VectorFn&& vector_fn,
                             ScalarFn&& scalar_fn) noexcept {
  const ScalableTag<float> d;
  const std::size_t width = Lanes(d);
  std::size_t i = 0;
  for (; width <= n - i; i += width) {
    const auto x = load_storage_as_f32<Storage, HNarrow>(d, src + i);
    store_f32_as_storage<Storage, HNarrow>(d, vector_fn(d, x), dst + i);
  }
  for (; i < n; ++i) {
    dst[i] = f32_to_storage<Storage>(scalar_fn(storage_to_f32(src[i])));
  }
}

template <typename Storage, typename HNarrow, class VectorFn, class ScalarFn>
void gated_storage_loop(const Storage* gate, const Storage* value, Storage* dst,
                        std::size_t n, VectorFn&& vector_fn,
                        ScalarFn&& scalar_fn) noexcept {
  const ScalableTag<float> d;
  const std::size_t width = Lanes(d);
  std::size_t i = 0;
  for (; width <= n - i; i += width) {
    const auto gate_v = load_storage_as_f32<Storage, HNarrow>(d, gate + i);
    const auto value_v = load_storage_as_f32<Storage, HNarrow>(d, value + i);
    store_f32_as_storage<Storage, HNarrow>(
        d, Mul(vector_fn(d, gate_v), value_v), dst + i);
  }
  for (; i < n; ++i) {
    const float gate_value = storage_to_f32(gate[i]);
    const float linear_value = storage_to_f32(value[i]);
    dst[i] = f32_to_storage<Storage>(scalar_fn(gate_value) * linear_value);
  }
}

template <typename Storage, typename HNarrow>
void rms_norm_storage(const Storage* src, const Storage* weight, Storage* dst,
                      std::size_t n, float epsilon) noexcept {
  if (n == 0) return;
  const ScalableTag<float> d;
  const std::size_t width = Lanes(d);
  auto sum_squares_v = Zero(d);
  std::size_t i = 0;
  for (; width <= n - i; i += width) {
    const auto value = load_storage_as_f32<Storage, HNarrow>(d, src + i);
    sum_squares_v = MulAdd(value, value, sum_squares_v);
  }
  float sum_squares = ReduceSum(d, sum_squares_v);
  for (; i < n; ++i) {
    const float value = storage_to_f32(src[i]);
    sum_squares = std::fma(value, value, sum_squares);
  }
  const float inverse_rms =
      1.0f / std::sqrt(sum_squares / static_cast<float>(n) + epsilon);
  const auto inverse_rms_v = Set(d, inverse_rms);
  i = 0;
  if (weight == nullptr) {
    for (; width <= n - i; i += width) {
      const auto value = load_storage_as_f32<Storage, HNarrow>(d, src + i);
      store_f32_as_storage<Storage, HNarrow>(d, Mul(value, inverse_rms_v),
                                             dst + i);
    }
    for (; i < n; ++i) {
      dst[i] = f32_to_storage<Storage>(storage_to_f32(src[i]) * inverse_rms);
    }
  } else {
    for (; width <= n - i; i += width) {
      const auto value = load_storage_as_f32<Storage, HNarrow>(d, src + i);
      const auto learned = load_storage_as_f32<Storage, HNarrow>(d, weight + i);
      store_f32_as_storage<Storage, HNarrow>(
          d, Mul(Mul(value, inverse_rms_v), learned), dst + i);
    }
    for (; i < n; ++i) {
      dst[i] = f32_to_storage<Storage>(storage_to_f32(src[i]) * inverse_rms *
                                       storage_to_f32(weight[i]));
    }
  }
}

template <typename Storage, typename HNarrow>
void layer_norm_storage(const Storage* src, const Storage* scale,
                        const Storage* bias, Storage* dst, std::size_t n,
                        float epsilon) noexcept {
  if (n == 0) return;
  const ScalableTag<float> d;
  const std::size_t width = Lanes(d);
  auto sum_v = Zero(d);
  std::size_t i = 0;
  for (; width <= n - i; i += width) {
    sum_v = Add(sum_v, load_storage_as_f32<Storage, HNarrow>(d, src + i));
  }
  float sum = ReduceSum(d, sum_v);
  for (; i < n; ++i) sum += storage_to_f32(src[i]);
  const float mean = sum / static_cast<float>(n);
  const auto mean_v = Set(d, mean);

  auto sum_squares_v = Zero(d);
  i = 0;
  for (; width <= n - i; i += width) {
    const auto centered =
        Sub(load_storage_as_f32<Storage, HNarrow>(d, src + i), mean_v);
    sum_squares_v = MulAdd(centered, centered, sum_squares_v);
  }
  float sum_squares = ReduceSum(d, sum_squares_v);
  for (; i < n; ++i) {
    const float centered = storage_to_f32(src[i]) - mean;
    sum_squares = std::fma(centered, centered, sum_squares);
  }
  const float inverse_stddev =
      1.0f / std::sqrt(sum_squares / static_cast<float>(n) + epsilon);
  const auto inverse_stddev_v = Set(d, inverse_stddev);

  i = 0;
  if (scale == nullptr) {
    for (; width <= n - i; i += width) {
      const auto normalized =
          Mul(Sub(load_storage_as_f32<Storage, HNarrow>(d, src + i), mean_v),
              inverse_stddev_v);
      store_f32_as_storage<Storage, HNarrow>(d, normalized, dst + i);
    }
    for (; i < n; ++i) {
      dst[i] = f32_to_storage<Storage>((storage_to_f32(src[i]) - mean) *
                                       inverse_stddev);
    }
  } else {
    for (; width <= n - i; i += width) {
      const auto normalized =
          Mul(Sub(load_storage_as_f32<Storage, HNarrow>(d, src + i), mean_v),
              inverse_stddev_v);
      const auto learned_scale =
          load_storage_as_f32<Storage, HNarrow>(d, scale + i);
      const auto learned_bias =
          load_storage_as_f32<Storage, HNarrow>(d, bias + i);
      store_f32_as_storage<Storage, HNarrow>(
          d, MulAdd(normalized, learned_scale, learned_bias), dst + i);
    }
    for (; i < n; ++i) {
      const float normalized = (storage_to_f32(src[i]) - mean) * inverse_stddev;
      dst[i] = f32_to_storage<Storage>(std::fma(
          normalized, storage_to_f32(scale[i]), storage_to_f32(bias[i])));
    }
  }
}

HWY_INLINE void fill_invalid_row(float* dst, std::size_t n) noexcept {
  const float value = std::numeric_limits<float>::quiet_NaN();
  for (std::size_t i = 0; i < n; ++i) dst[i] = value;
}

HWY_INLINE bool activation_row_max(const float* src, std::size_t n,
                                   float& maximum) noexcept {
  const ScalableTag<float> d;
  const std::size_t width = Lanes(d);
  const auto positive_infinity = Set(d, std::numeric_limits<float>::infinity());
  auto maximum_v = Set(d, -std::numeric_limits<float>::infinity());
  bool invalid = false;
  std::size_t i = 0;
  for (; i + width <= n; i += width) {
    const auto x = LoadU(d, src + i);
    invalid = invalid || !AllFalse(d, Or(IsNaN(x), Eq(x, positive_infinity)));
    maximum_v = Max(maximum_v, x);
  }
  maximum = ReduceMax(d, maximum_v);
  for (; i < n; ++i) {
    const float x = src[i];
    if (std::isnan(x) || x == std::numeric_limits<float>::infinity()) {
      invalid = true;
    }
    if (x > maximum) maximum = x;
  }
  return !invalid && std::isfinite(maximum);
}

HWY_INLINE void activation_softmax(const float* src, float* dst,
                                   std::size_t n) noexcept {
  if (n == 0) return;
  float maximum = 0.0f;
  if (!activation_row_max(src, n, maximum)) {
    fill_invalid_row(dst, n);
    return;
  }

  const ScalableTag<float> d;
  const std::size_t width = Lanes(d);
  const auto maximum_v = Set(d, maximum);
  double sum = 0.0;
  std::size_t i = 0;
  for (; i + width <= n; i += width) {
    const auto value = Exp(d, Sub(LoadU(d, src + i), maximum_v));
    StoreU(value, d, dst + i);
    sum += static_cast<double>(ReduceSum(d, value));
  }
  for (; i < n; ++i) {
    const float value = std::exp(src[i] - maximum);
    dst[i] = value;
    sum += static_cast<double>(value);
  }

  const float inverse_scalar = static_cast<float>(1.0 / sum);
  const auto inverse = Set(d, inverse_scalar);
  i = 0;
  for (; i + width <= n; i += width) {
    StoreU(Mul(LoadU(d, dst + i), inverse), d, dst + i);
  }
  for (; i < n; ++i) dst[i] *= inverse_scalar;
}

HWY_INLINE void activation_log_softmax(const float* src, float* dst,
                                       std::size_t n) noexcept {
  if (n == 0) return;
  float maximum = 0.0f;
  if (!activation_row_max(src, n, maximum)) {
    fill_invalid_row(dst, n);
    return;
  }

  const ScalableTag<float> d;
  const std::size_t width = Lanes(d);
  const auto maximum_v = Set(d, maximum);
  double sum = 0.0;
  std::size_t i = 0;
  for (; i + width <= n; i += width) {
    sum += static_cast<double>(
        ReduceSum(d, Exp(d, Sub(LoadU(d, src + i), maximum_v))));
  }
  for (; i < n; ++i) {
    sum += static_cast<double>(std::exp(src[i] - maximum));
  }

  const float shift_scalar = maximum + static_cast<float>(std::log(sum));
  const auto shift = Set(d, shift_scalar);
  i = 0;
  for (; i + width <= n; i += width) {
    StoreU(Sub(LoadU(d, src + i), shift), d, dst + i);
  }
  for (; i < n; ++i) dst[i] = src[i] - shift_scalar;
}

}  // namespace HYLE_HIGHWAY_MATH_DETAIL_NAMESPACE

namespace hyle::simd::detail {
namespace hm = ::HYLE_HIGHWAY_MATH_DETAIL_NAMESPACE;

#define HYLE_HWY_SYMBOL_I(name, tag, suffix) name##_##tag##_highway_##suffix
#define HYLE_HWY_SYMBOL_II(name, tag, suffix) \
  HYLE_HWY_SYMBOL_I(name, tag, suffix)
#define HYLE_HWY_SYMBOL(name, tag) \
  HYLE_HWY_SYMBOL_II(name, tag, HYLE_HIGHWAY_MATH_SUFFIX)
#define HYLE_OWNED_SYMBOL_I(name, tag, suffix) name##_##tag##_hyle_##suffix
#define HYLE_OWNED_SYMBOL_II(name, tag, suffix) \
  HYLE_OWNED_SYMBOL_I(name, tag, suffix)
#define HYLE_OWNED_SYMBOL(name, tag) \
  HYLE_OWNED_SYMBOL_II(name, tag, HYLE_HIGHWAY_MATH_SUFFIX)
#define HYLE_ROUTE_SYMBOL_I(name, tag, suffix) name##_##tag##_route_##suffix
#define HYLE_ROUTE_SYMBOL_II(name, tag, suffix) \
  HYLE_ROUTE_SYMBOL_I(name, tag, suffix)
#define HYLE_ROUTE_SYMBOL(name, tag) \
  HYLE_ROUTE_SYMBOL_II(name, tag, HYLE_HIGHWAY_MATH_SUFFIX)

#define HYLE_DEFINE_DIRECT_UNARY(type, tag, name, hwy_name, scalar_name,   \
                                 domain)                                   \
  void HYLE_HWY_SYMBOL(name, tag)(const type* src, type* dst,              \
                                  std::size_t n) noexcept {                \
    hm::unary_loop<hm::UnaryDomain::domain>(                               \
        src, dst, n,                                                       \
        [](auto d, auto x) { return hwy::HWY_NAMESPACE::hwy_name(d, x); }, \
        [](type x) { return std::scalar_name(x); });                       \
  }

#define HYLE_DEFINE_DIRECT_UNARY_SET(type, tag)                        \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, exp, Exp, exp, kExp)             \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, log, Log, log, kPositive)        \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, sin, Sin, sin, kTrig)            \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, cos, Cos, cos, kTrig)            \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, exp2, Exp2, exp2, kExp2)         \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, log2, Log2, log2, kPositive)     \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, erf, Erf, erf, kDirect)          \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, atan, Atan, atan, kDirect)       \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, cbrt, Cbrt, cbrt, kDirect)       \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, expm1, Expm1, expm1, kExp)       \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, log1p, Log1p, log1p, kLog1p)     \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, asinh, Asinh, asinh, kFinite)    \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, acosh, Acosh, acosh, kAcosh)     \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, acos, Acos, acos, kUnitClosed)   \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, asin, Asin, asin, kUnitClosed)   \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, sinh, Sinh, sinh, kSinhCosh)     \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, cosh, Cosh, cosh, kSinhCosh)     \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, tan, Tan, tan, kTrig)            \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, atanh, Atanh, atanh, kUnitOpen)  \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, log10, Log10, log10, kPositive)  \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, tgamma, Tgamma, tgamma, kTgamma) \
  HYLE_DEFINE_DIRECT_UNARY(type, tag, lgamma, LogGamma, lgamma, kLogGamma)

HYLE_DEFINE_DIRECT_UNARY_SET(float, f32)
HYLE_DEFINE_DIRECT_UNARY_SET(double, f64)
HYLE_DEFINE_DIRECT_UNARY(double, f64, tanh, Tanh, tanh, kDirect)

void HYLE_ROUTE_SYMBOL(tanh, f32)(const float* src, float* dst,
                                  std::size_t n) noexcept {
  constexpr auto selected =
      provider::kQualifiedProviderFor<provider::Family::kTanhF32Batch,
                                      provider::LaneVariant<float>>;
  static_assert(selected == provider::Provider::kHighway ||
                    selected == provider::Provider::kXsimd,
                "tanh f32 route lacks a provider-coherent body");
  if constexpr (selected == provider::Provider::kXsimd) {
#if defined(HYLE_SIMD_MATH_HAS_XSIMD_BODY)
    using Batch = xsimd::batch<float>;
    constexpr std::size_t width = Batch::size;
    std::size_t i = 0;
    for (; width <= n - i; i += width) {
      xsimd::tanh(Batch::load_unaligned(src + i)).store_unaligned(dst + i);
    }
    for (; i < n; ++i) dst[i] = std::tanh(src[i]);
#else
    static_assert(selected != provider::Provider::kXsimd,
                  "qualified xsimd route has no target body");
#endif
  } else {
    hm::unary_loop<hm::UnaryDomain::kDirect>(
        src, dst, n,
        [](auto d, auto x) { return hwy::HWY_NAMESPACE::Tanh(d, x); },
        [](float x) { return std::tanh(x); });
  }
}
#undef HYLE_DEFINE_DIRECT_UNARY_SET
#undef HYLE_DEFINE_DIRECT_UNARY

#define HYLE_DEFINE_DIRECT_BINARY(type, tag, name, hwy_name, scalar_name)  \
  void HYLE_HWY_SYMBOL(name, tag)(const type* a, const type* b, type* dst, \
                                  std::size_t n) noexcept {                \
    hm::binary_loop(                                                       \
        a, b, dst, n,                                                      \
        [](auto d, auto x, auto y) {                                       \
          return hwy::HWY_NAMESPACE::hwy_name(d, x, y);                    \
        },                                                                 \
        [](type x, type y) { return std::scalar_name(x, y); });            \
  }

#define HYLE_DEFINE_DIRECT_BINARY_SET(type, tag)            \
  HYLE_DEFINE_DIRECT_BINARY(type, tag, atan2, Atan2, atan2) \
  HYLE_DEFINE_DIRECT_BINARY(type, tag, pow, Pow, pow)       \
  HYLE_DEFINE_DIRECT_BINARY(type, tag, hypot, Hypot, hypot)

HYLE_DEFINE_DIRECT_BINARY_SET(float, f32)
HYLE_DEFINE_DIRECT_BINARY_SET(double, f64)
#undef HYLE_DEFINE_DIRECT_BINARY_SET
#undef HYLE_DEFINE_DIRECT_BINARY

void HYLE_HWY_SYMBOL(sincos, f32)(const float* src, float* sine, float* cosine,
                                  std::size_t n) noexcept {
  hm::sincos_loop(src, sine, cosine, n);
}
void HYLE_HWY_SYMBOL(sincos, f64)(const double* src, double* sine,
                                  double* cosine, std::size_t n) noexcept {
  hm::sincos_loop(src, sine, cosine, n);
}

void HYLE_HWY_SYMBOL(exp_in_range, f32)(const float* src, float* dst,
                                        std::size_t n) noexcept {
  hm::unary_loop<hm::UnaryDomain::kDirect>(
      src, dst, n, [](auto d, auto x) { return hwy::HWY_NAMESPACE::Exp(d, x); },
      [](float x) { return std::exp(x); });
}
void HYLE_HWY_SYMBOL(log_in_range, f32)(const float* src, float* dst,
                                        std::size_t n) noexcept {
  hm::unary_loop<hm::UnaryDomain::kDirect>(
      src, dst, n, [](auto d, auto x) { return hwy::HWY_NAMESPACE::Log(d, x); },
      [](float x) { return std::log(x); });
}

void HYLE_HWY_SYMBOL(exp_in_range, f64)(const double* src, double* dst,
                                        std::size_t n) noexcept {
  hm::unary_loop<hm::UnaryDomain::kDirect>(
      src, dst, n, [](auto d, auto x) { return hwy::HWY_NAMESPACE::Exp(d, x); },
      [](double x) { return std::exp(x); });
}
void HYLE_HWY_SYMBOL(log_in_range, f64)(const double* src, double* dst,
                                        std::size_t n) noexcept {
  hm::unary_loop<hm::UnaryDomain::kDirect>(
      src, dst, n, [](auto d, auto x) { return hwy::HWY_NAMESPACE::Log(d, x); },
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

#define HYLE_DEFINE_OWNED_GAPS(type, tag)                                    \
  void HYLE_OWNED_SYMBOL(erfc, tag)(const type* src, type* dst,              \
                                    std::size_t n) noexcept {                \
    hm::erfc_loop(src, dst, n);                                              \
  }                                                                          \
  void HYLE_OWNED_SYMBOL(exp10, tag)(const type* src, type* dst,             \
                                     std::size_t n) noexcept {               \
    hm::exp10_loop(src, dst, n);                                             \
  }                                                                          \
  void HYLE_OWNED_SYMBOL(fdim, tag)(const type* a, const type* b, type* dst, \
                                    std::size_t n) noexcept {                \
    hm::fdim_loop(a, b, dst, n);                                             \
  }

HYLE_DEFINE_OWNED_GAPS(float, f32)
HYLE_DEFINE_OWNED_GAPS(double, f64)
#undef HYLE_DEFINE_OWNED_GAPS

#define HYLE_DEFINE_ACTIVATION_UNARY(name, vector_fn, scalar_fn)         \
  void HYLE_OWNED_SYMBOL(name, f32)(const float* src, float* dst,        \
                                    std::size_t n) noexcept {            \
    hm::unary_loop<hm::UnaryDomain::kDirect>(                            \
        src, dst, n, [](auto d, auto x) { return hm::vector_fn(d, x); }, \
        [](float x) { return hm::scalar_fn(x); });                       \
  }

HYLE_DEFINE_ACTIVATION_UNARY(gelu, activation_gelu, activation_gelu_scalar)
HYLE_DEFINE_ACTIVATION_UNARY(sigmoid, activation_sigmoid,
                             activation_sigmoid_scalar)
HYLE_DEFINE_ACTIVATION_UNARY(silu, activation_silu, activation_silu_scalar)
HYLE_DEFINE_ACTIVATION_UNARY(relu, activation_relu, activation_relu_scalar)
HYLE_DEFINE_ACTIVATION_UNARY(hard_sigmoid, activation_hard_sigmoid,
                             activation_hard_sigmoid_scalar)
HYLE_DEFINE_ACTIVATION_UNARY(hard_swish, activation_hard_swish,
                             activation_hard_swish_scalar)
HYLE_DEFINE_ACTIVATION_UNARY(gelu_quick, activation_gelu_quick,
                             activation_gelu_quick_scalar)

void HYLE_ROUTE_SYMBOL(gelu_tanh, f32)(const float* src, float* dst,
                                       std::size_t n) noexcept {
  constexpr auto selected =
      provider::kQualifiedProviderFor<provider::Family::kGeluTanhF32Batch,
                                      provider::LaneVariant<float>>;
  static_assert(selected == provider::Provider::kHighway ||
                    selected == provider::Provider::kXsimd,
                "gelu_tanh f32 route lacks a provider-coherent body");
  if constexpr (selected == provider::Provider::kXsimd) {
#if defined(HYLE_SIMD_MATH_HAS_XSIMD_BODY)
    using Batch = xsimd::batch<float>;
    constexpr std::size_t width = Batch::size;
    const Batch zero(0.0f);
    const Batch one(1.0f);
    const Batch half(0.5f);
    const Batch scale(hm::kActivationSqrt2OverPi);
    const Batch cubic(hm::kActivationGeluCoefficient);
    std::size_t i = 0;
    for (; width <= n - i; i += width) {
      const Batch x = Batch::load_unaligned(src + i);
      const Batch inner = (scale * x) * xsimd::fma(cubic, x * x, one);
      const Batch result = (half * x) * (one + xsimd::tanh(inner));
      xsimd::select(xsimd::isinf(x) & (x < zero), zero, result)
          .store_unaligned(dst + i);
    }
    for (; i < n; ++i) {
      dst[i] = hm::activation_gelu_tanh_scalar(src[i]);
    }
#else
    static_assert(selected != provider::Provider::kXsimd,
                  "qualified xsimd route has no target body");
#endif
  } else {
    hm::unary_loop<hm::UnaryDomain::kDirect>(
        src, dst, n,
        [](auto d, auto x) { return hm::activation_gelu_tanh(d, x); },
        [](float x) { return hm::activation_gelu_tanh_scalar(x); });
  }
}
#undef HYLE_DEFINE_ACTIVATION_UNARY

#define HYLE_DEFINE_NARROW_ACTIVATION(type, tag, hwy_type, name, vector_fn, \
                                      scalar_fn)                            \
  void HYLE_OWNED_SYMBOL(name, tag)(const type* src, type* dst,             \
                                    std::size_t n) noexcept {               \
    hm::activation_storage_loop<type, hwy_type>(                            \
        src, dst, n, [](auto d, auto x) { return hm::vector_fn(d, x); },    \
        [](float x) { return hm::scalar_fn(x); });                          \
  }

#define HYLE_DEFINE_NARROW_ACTIVATION_SET(type, tag, hwy_type)              \
  HYLE_DEFINE_NARROW_ACTIVATION(type, tag, hwy_type, gelu, activation_gelu, \
                                activation_gelu_scalar)                     \
  HYLE_DEFINE_NARROW_ACTIVATION(type, tag, hwy_type, gelu_tanh,             \
                                activation_gelu_tanh,                       \
                                activation_gelu_tanh_scalar)                \
  HYLE_DEFINE_NARROW_ACTIVATION(type, tag, hwy_type, gelu_quick,            \
                                activation_gelu_quick,                      \
                                activation_gelu_quick_scalar)               \
  HYLE_DEFINE_NARROW_ACTIVATION(type, tag, hwy_type, silu, activation_silu, \
                                activation_silu_scalar)

HYLE_DEFINE_NARROW_ACTIVATION_SET(float16_t, f16, hwy::float16_t)
HYLE_DEFINE_NARROW_ACTIVATION_SET(bfloat16_t, bf16, hwy::bfloat16_t)

#undef HYLE_DEFINE_NARROW_ACTIVATION_SET
#undef HYLE_DEFINE_NARROW_ACTIVATION

#define HYLE_DEFINE_GATED(type, tag, hwy_type, name, vector_fn, scalar_fn) \
  void HYLE_OWNED_SYMBOL(name, tag)(const type* gate, const type* value,   \
                                    type* dst, std::size_t n) noexcept {   \
    hm::gated_storage_loop<type, hwy_type>(                                \
        gate, value, dst, n,                                               \
        [](auto d, auto x) { return hm::vector_fn(d, x); },                \
        [](float x) { return hm::scalar_fn(x); });                         \
  }

#define HYLE_DEFINE_GATED_SET(type, tag, hwy_type)                           \
  HYLE_DEFINE_GATED(type, tag, hwy_type, reglu, activation_relu,             \
                    activation_relu_scalar)                                  \
  HYLE_DEFINE_GATED(type, tag, hwy_type, geglu, activation_gelu,             \
                    activation_gelu_scalar)                                  \
  HYLE_DEFINE_GATED(type, tag, hwy_type, geglu_tanh, activation_gelu_tanh,   \
                    activation_gelu_tanh_scalar)                             \
  HYLE_DEFINE_GATED(type, tag, hwy_type, geglu_quick, activation_gelu_quick, \
                    activation_gelu_quick_scalar)                            \
  HYLE_DEFINE_GATED(type, tag, hwy_type, swiglu, activation_silu,            \
                    activation_silu_scalar)

HYLE_DEFINE_GATED_SET(float, f32, float)
HYLE_DEFINE_GATED_SET(float16_t, f16, hwy::float16_t)
HYLE_DEFINE_GATED_SET(bfloat16_t, bf16, hwy::bfloat16_t)

#undef HYLE_DEFINE_GATED_SET
#undef HYLE_DEFINE_GATED

#define HYLE_DEFINE_NORMALIZATION(type, tag, hwy_type)                         \
  void HYLE_OWNED_SYMBOL(rms_norm, tag)(const type* src, const type* weight,   \
                                        type* dst, std::size_t n,              \
                                        float epsilon) noexcept {              \
    hm::rms_norm_storage<type, hwy_type>(src, weight, dst, n, epsilon);        \
  }                                                                            \
  void HYLE_OWNED_SYMBOL(layer_norm, tag)(                                     \
      const type* src, const type* scale, const type* bias, type* dst,         \
      std::size_t n, float epsilon) noexcept {                                 \
    hm::layer_norm_storage<type, hwy_type>(src, scale, bias, dst, n, epsilon); \
  }

HYLE_DEFINE_NORMALIZATION(float, f32, float)
HYLE_DEFINE_NORMALIZATION(float16_t, f16, hwy::float16_t)
HYLE_DEFINE_NORMALIZATION(bfloat16_t, bf16, hwy::bfloat16_t)

#undef HYLE_DEFINE_NORMALIZATION

void HYLE_OWNED_SYMBOL(leaky_relu, f32)(const float* src, float negative_slope,
                                        float* dst, std::size_t n) noexcept {
  hm::unary_loop<hm::UnaryDomain::kDirect>(
      src, dst, n,
      [negative_slope](auto d, auto x) {
        return hm::activation_leaky_relu(d, x, negative_slope);
      },
      [negative_slope](float x) {
        return hm::activation_leaky_relu_scalar(x, negative_slope);
      });
}

void HYLE_OWNED_SYMBOL(elu, f32)(const float* src, float alpha, float* dst,
                                 std::size_t n) noexcept {
  hm::unary_loop<hm::UnaryDomain::kDirect>(
      src, dst, n,
      [alpha](auto d, auto x) { return hm::activation_elu(d, x, alpha); },
      [alpha](float x) { return hm::activation_elu_scalar(x, alpha); });
}

void HYLE_OWNED_SYMBOL(softmax, f32)(const float* src, float* dst,
                                     std::size_t n) noexcept {
  hm::activation_softmax(src, dst, n);
}

void HYLE_OWNED_SYMBOL(log_softmax, f32)(const float* src, float* dst,
                                         std::size_t n) noexcept {
  hm::activation_log_softmax(src, dst, n);
}

#define HYLE_DEFINE_COMPLEX_UNARY(type, tag, name, body)                    \
  void HYLE_OWNED_SYMBOL(name, tag)(const type* src_re, const type* src_im, \
                                    type* dst_re, type* dst_im,             \
                                    std::size_t n) noexcept {               \
    body;                                                                   \
  }

HYLE_DEFINE_COMPLEX_UNARY(float, f32, cexp,
                          hm::complex_exp_typed_hyle(src_re, src_im, dst_re,
                                                     dst_im, n))
HYLE_DEFINE_COMPLEX_UNARY(float, f32, clog,
                          hm::complex_log_typed_hyle(src_re, src_im, dst_re,
                                                     dst_im, n))
HYLE_DEFINE_COMPLEX_UNARY(float, f32, csqrt,
                          hm::complex_sqrt_typed_hyle(src_re, src_im, dst_re,
                                                      dst_im, n))
HYLE_DEFINE_COMPLEX_UNARY(float, f32, csin,
                          hm::complex_sincos_hyle<false>(src_re, src_im, dst_re,
                                                         dst_im, n))
HYLE_DEFINE_COMPLEX_UNARY(float, f32, ccos,
                          hm::complex_sincos_hyle<true>(src_re, src_im, dst_re,
                                                        dst_im, n))
HYLE_DEFINE_COMPLEX_UNARY(float, f32, ctan,
                          hm::complex_tan_typed_hyle(src_re, src_im, dst_re,
                                                     dst_im, n))
HYLE_DEFINE_COMPLEX_UNARY(float, f32, csinh,
                          (hm::complex_sinhcosh_typed_hyle<float, false>(
                              src_re, src_im, dst_re, dst_im, n)))
HYLE_DEFINE_COMPLEX_UNARY(float, f32, ccosh,
                          (hm::complex_sinhcosh_typed_hyle<float, true>(
                              src_re, src_im, dst_re, dst_im, n)))
HYLE_DEFINE_COMPLEX_UNARY(float, f32, ctanh,
                          hm::complex_ctanh_hyle(src_re, src_im, dst_re, dst_im,
                                                 n))

HYLE_DEFINE_COMPLEX_UNARY(double, f64, cexp,
                          hm::complex_exp_typed_hyle(src_re, src_im, dst_re,
                                                     dst_im, n))
HYLE_DEFINE_COMPLEX_UNARY(double, f64, clog,
                          hm::complex_log_typed_hyle(src_re, src_im, dst_re,
                                                     dst_im, n))
HYLE_DEFINE_COMPLEX_UNARY(double, f64, csqrt,
                          hm::complex_sqrt_typed_hyle(src_re, src_im, dst_re,
                                                      dst_im, n))
HYLE_DEFINE_COMPLEX_UNARY(double, f64, csin,
                          (hm::complex_sincos_typed_hyle<double, false>(
                              src_re, src_im, dst_re, dst_im, n)))
HYLE_DEFINE_COMPLEX_UNARY(double, f64, ccos,
                          (hm::complex_sincos_typed_hyle<double, true>(
                              src_re, src_im, dst_re, dst_im, n)))
HYLE_DEFINE_COMPLEX_UNARY(double, f64, ctan,
                          hm::complex_tan_typed_hyle(src_re, src_im, dst_re,
                                                     dst_im, n))
HYLE_DEFINE_COMPLEX_UNARY(double, f64, csinh,
                          (hm::complex_sinhcosh_typed_hyle<double, false>(
                              src_re, src_im, dst_re, dst_im, n)))
HYLE_DEFINE_COMPLEX_UNARY(double, f64, ccosh,
                          (hm::complex_sinhcosh_typed_hyle<double, true>(
                              src_re, src_im, dst_re, dst_im, n)))
HYLE_DEFINE_COMPLEX_UNARY(double, f64, ctanh,
                          hm::complex_tanh_typed_hyle(src_re, src_im, dst_re,
                                                      dst_im, n))
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
#undef HYLE_ROUTE_SYMBOL
#undef HYLE_ROUTE_SYMBOL_II
#undef HYLE_ROUTE_SYMBOL_I
#undef HYLE_HWY_SYMBOL
#undef HYLE_HWY_SYMBOL_II
#undef HYLE_HWY_SYMBOL_I

}  // namespace hyle::simd::detail

#undef HYLE_HIGHWAY_MATH_DETAIL_NAMESPACE
#undef HYLE_HWY_DETAIL_NAMESPACE
#undef HYLE_HWY_DETAIL_NAMESPACE_II
#undef HYLE_HWY_DETAIL_NAMESPACE_I

#if defined(HYLE_SIMD_MATH_HAS_XSIMD_BODY)
#undef HYLE_SIMD_MATH_HAS_XSIMD_BODY
#endif
