// Highway-accelerated AVX2 transcendental (8-wide). BORROWS Highway's verified contrib/math
// core (hn::Log) -- HYLE does not hand-roll these (charter: 借验证核不手搓). The same-ISA
// bench (math_bench) found Highway's log fastest at the ~3.5-ULP class (xsimd wins exp/tanh),
// so per-op best-of-breed (math_dispatch.cc) uses Highway for LOG ONLY -- hence only log is
// defined here (exp/tanh would be dead code: xsimd serves those).
//
// x86-only: highway lowers to AVX2 intrinsics here, so the TU compiles to nothing on non-x86
// (math_dispatch falls to the scalar libm floor there). Header-only + vendored in third_party,
// so it can't be missing -- no opt-in define gates it, only the architecture does.
//
// Static single-target dispatch: this TU is compiled at -mavx2 (+ full Highway baseline via
// hyle_set_highway_isa, else HWY_STATIC_TARGET silently downgrades to 128-bit SSSE3), so hn::
// lowers to 256-bit AVX2. Reached only when cpu_features().avx2 (gated in math_dispatch), so
// the AVX2 codegen never runs on an older CPU.
#include "hyle/platform.h"  // HYLE_ARCH_X64 -- MUST precede the guard below

#if defined(HYLE_ARCH_X64)

#include "src/simd/math/kernels.h"

#include <cmath>
#include <cstddef>

#include "hwy/highway.h"
#include "hwy/contrib/math/math-inl.h"
#include "src/simd/math/table_f64.h"

namespace hn = hwy::HWY_NAMESPACE;

namespace hyle::simd::detail {
namespace {

// Highway's polynomial kernels are accurate for normal finite lanes, but their internal float-to-
// int reduction deliberately does not define IEEE special values. Do not feed exceptional lanes
// into that reduction: batch APIs retain scalar-libm semantics for NaN/Inf/domain/edge ranges.
template <class D, class V>
bool exp_f64_needs_scalar(D d, V x) noexcept {
  const auto exceptional = hn::Or(
      hn::Or(hn::IsNaN(x), hn::IsInf(x)),
      hn::Or(hn::Lt(x, hn::Set(d, -700.0)), hn::Gt(x, hn::Set(d, 700.0))));
  return !hn::AllFalse(d, exceptional);
}

template <class D, class V>
bool log_f64_needs_scalar(D d, V x) noexcept {
  const auto exceptional = hn::Or(
      hn::Or(hn::IsNaN(x), hn::IsInf(x)),
      hn::Or(hn::Le(x, hn::Zero(d)), hn::Lt(x, hn::Set(d, 0x1p-1022))));
  return !hn::AllFalse(d, exceptional);
}

}  // namespace

void log_f32_highway_avx2(const float* src, float* dst, std::size_t n) noexcept {
  const hn::ScalableTag<float> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w)
    hn::StoreU(hn::Log(d, hn::LoadU(d, src + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = std::log(src[i]);  // tail = libm (scalar oracle)
}

// f64 (4-wide AVX2) Highway borrow cores. hn::ScalableTag<double> lowers to the 4-lane ymm. Same
// verified contrib/math polynomials as f32, double precision. Winner (Highway vs xsimd) bench-
// decided per op in math_dispatch -- these exist so best-of can pick them.
void exp_f64_highway_avx2(const double* src, double* dst, std::size_t n) noexcept {
  const hn::ScalableTag<double> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w) {
    const auto x = hn::LoadU(d, src + i);
    if (exp_f64_needs_scalar(d, x)) {
      for (std::size_t lane = 0; lane < w; ++lane) dst[i + lane] = std::exp(src[i + lane]);
    } else {
      hn::StoreU(table_f64::ExpNormal(d, x), d, dst + i);
    }
  }
  for (; i < n; ++i) dst[i] = std::exp(src[i]);
}
void log_f64_highway_avx2(const double* src, double* dst, std::size_t n) noexcept {
  const hn::ScalableTag<double> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w) {
    const auto x = hn::LoadU(d, src + i);
    if (log_f64_needs_scalar(d, x)) {
      for (std::size_t lane = 0; lane < w; ++lane) dst[i + lane] = std::log(src[i + lane]);
    } else {
      hn::StoreU(table_f64::LogNormal(d, x), d, dst + i);
    }
  }
  for (; i < n; ++i) dst[i] = std::log(src[i]);
}
void exp_in_range_f64_highway_avx2(const double* src, double* dst, std::size_t n) noexcept {
  const hn::ScalableTag<double> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w)
    hn::StoreU(table_f64::ExpNormal(d, hn::LoadU(d, src + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = std::exp(src[i]);
}
void log_in_range_f64_highway_avx2(const double* src, double* dst, std::size_t n) noexcept {
#if defined(HYLE_OS_LINUX) && defined(HYLE_COMPILER_GCC)
  // GCC AVX2 hoists every table constant across the gather loop, causing register spills and a
  // measured 25% regression (1.02–1.07 vs 0.77–0.85 ns/op). The full core's branch-local table
  // block avoids that allocation; keep it until a no-check GCC form actually wins.
  log_f64_highway_avx2(src, dst, n);
  return;
#endif
  const hn::ScalableTag<double> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w)
    hn::StoreU(table_f64::LogNormal(d, hn::LoadU(d, src + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = std::log(src[i]);
}
void tanh_f64_highway_avx2(const double* src, double* dst, std::size_t n) noexcept {
  const hn::ScalableTag<double> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w)
    hn::StoreU(hn::Tanh(d, hn::LoadU(d, src + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = std::tanh(src[i]);
}
// f64 unary batch: Highway wins 10 of 11 remaining transcendentals at f64 (bench: sin/cos/exp2/
// log2/atan/cbrt/expm1/log1p/asinh/acosh -- xsimd only wins erf). Same borrow macro as f32.
#define HYLE_HWY_MATH_F64_AVX2(name, hwyfn, libmfn)                              \
  void name##_f64_highway_avx2(const double* src, double* dst, std::size_t n) noexcept { \
    const hn::ScalableTag<double> d;                                            \
    const std::size_t w = hn::Lanes(d);                                         \
    std::size_t i = 0;                                                          \
    for (; i + w <= n; i += w)                                                  \
      hn::StoreU(hn::hwyfn(d, hn::LoadU(d, src + i)), d, dst + i);               \
    for (; i < n; ++i) dst[i] = libmfn(src[i]);                                 \
  }
HYLE_HWY_MATH_F64_AVX2(sin, Sin, std::sin)
HYLE_HWY_MATH_F64_AVX2(cos, Cos, std::cos)
HYLE_HWY_MATH_F64_AVX2(exp2, Exp2, std::exp2)
HYLE_HWY_MATH_F64_AVX2(log2, Log2, std::log2)
HYLE_HWY_MATH_F64_AVX2(atan, Atan, std::atan)
HYLE_HWY_MATH_F64_AVX2(cbrt, Cbrt, std::cbrt)
HYLE_HWY_MATH_F64_AVX2(expm1, Expm1, std::expm1)
HYLE_HWY_MATH_F64_AVX2(log1p, Log1p, std::log1p)
HYLE_HWY_MATH_F64_AVX2(asinh, Asinh, std::asinh)
HYLE_HWY_MATH_F64_AVX2(acosh, Acosh, std::acosh)
HYLE_HWY_MATH_F64_AVX2(acos, Acos, std::acos)
HYLE_HWY_MATH_F64_AVX2(asin, Asin, std::asin)
HYLE_HWY_MATH_F64_AVX2(sinh, Sinh, std::sinh)
HYLE_HWY_MATH_F64_AVX2(cosh, Cosh, std::cosh)
HYLE_HWY_MATH_F64_AVX2(tan, Tan, std::tan)
HYLE_HWY_MATH_F64_AVX2(atanh, Atanh, std::atanh)
HYLE_HWY_MATH_F64_AVX2(log10, Log10, std::log10)
#undef HYLE_HWY_MATH_F64_AVX2
// hypot f64 (binary, overflow-safe -> Highway; xsimd's naive sqrt overflows).
void hypot_f64_highway_avx2(const double* a, const double* b, double* dst, std::size_t n) noexcept {
  const hn::ScalableTag<double> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w)
    hn::StoreU(hn::Hypot(d, hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = std::hypot(a[i], b[i]);
}

// simd-09: Highway wins sin/cos/log2/atan/cbrt/expm1/log1p/asinh/acosh at the same ISA
// (bench-decided, GCC+MSVC cross-checked; xsimd serves exp2/erf). Same borrow pattern as log.
#define HYLE_HWY_MATH_AVX2(name, hwyfn, libmfn)                                  \
  void name##_f32_highway_avx2(const float* src, float* dst, std::size_t n) noexcept { \
    const hn::ScalableTag<float> d;                                             \
    const std::size_t w = hn::Lanes(d);                                         \
    std::size_t i = 0;                                                          \
    for (; i + w <= n; i += w)                                                  \
      hn::StoreU(hn::hwyfn(d, hn::LoadU(d, src + i)), d, dst + i);               \
    for (; i < n; ++i) dst[i] = libmfn(src[i]);                                 \
  }
HYLE_HWY_MATH_AVX2(sin, Sin, std::sin)
HYLE_HWY_MATH_AVX2(cos, Cos, std::cos)
HYLE_HWY_MATH_AVX2(log2, Log2, std::log2)
HYLE_HWY_MATH_AVX2(atan, Atan, std::atan)
HYLE_HWY_MATH_AVX2(cbrt, Cbrt, std::cbrt)
HYLE_HWY_MATH_AVX2(expm1, Expm1, std::expm1)
HYLE_HWY_MATH_AVX2(log1p, Log1p, std::log1p)
HYLE_HWY_MATH_AVX2(asinh, Asinh, std::asinh)
HYLE_HWY_MATH_AVX2(acosh, Acosh, std::acosh)
HYLE_HWY_MATH_AVX2(acos, Acos, std::acos)
HYLE_HWY_MATH_AVX2(asin, Asin, std::asin)
HYLE_HWY_MATH_AVX2(sinh, Sinh, std::sinh)
HYLE_HWY_MATH_AVX2(cosh, Cosh, std::cosh)
HYLE_HWY_MATH_AVX2(tan, Tan, std::tan)
HYLE_HWY_MATH_AVX2(atanh, Atanh, std::atanh)
HYLE_HWY_MATH_AVX2(log10, Log10, std::log10)
#undef HYLE_HWY_MATH_AVX2

// Full-range hypot core: Highway exponent scaling; scalar std::hypot tail. The explicit fast API
// uses xsimd's direct formula in the xsimd TU.
void hypot_f32_highway_avx2(const float* a, const float* b, float* dst, std::size_t n) noexcept {
  const hn::ScalableTag<float> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w)
    hn::StoreU(hn::Hypot(d, hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = std::hypot(a[i], b[i]);
}


// simd-18 批G binary + sincos (Highway)
void atan2_f32_highway_avx2(const float* a, const float* b, float* dst, std::size_t n) noexcept {
  const hn::ScalableTag<float> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w)
    hn::StoreU(hn::Atan2(d, hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = std::atan2(a[i], b[i]);
}
void pow_f32_highway_avx2(const float* a, const float* b, float* dst, std::size_t n) noexcept {
  const hn::ScalableTag<float> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w)
    hn::StoreU(hn::Pow(d, hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = std::pow(a[i], b[i]);
}
void sincos_f32_highway_avx2(const float* src, float* s, float* c, std::size_t n) noexcept {
  const hn::ScalableTag<float> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w) {
    hn::Vec<hn::ScalableTag<float>> sv, cv;
    hn::SinCos(d, hn::LoadU(d, src + i), sv, cv);
    hn::StoreU(sv, d, s + i);
    hn::StoreU(cv, d, c + i);
  }
  for (; i < n; ++i) { s[i] = std::sin(src[i]); c[i] = std::cos(src[i]); }
}
void atan2_f64_highway_avx2(const double* a, const double* b, double* dst, std::size_t n) noexcept {
  const hn::ScalableTag<double> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w)
    hn::StoreU(hn::Atan2(d, hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = std::atan2(a[i], b[i]);
}
void pow_f64_highway_avx2(const double* a, const double* b, double* dst, std::size_t n) noexcept {
  const hn::ScalableTag<double> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w)
    hn::StoreU(hn::Pow(d, hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = std::pow(a[i], b[i]);
}
void sincos_f64_highway_avx2(const double* src, double* s, double* c, std::size_t n) noexcept {
  const hn::ScalableTag<double> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w) {
    hn::Vec<hn::ScalableTag<double>> sv, cv;
    hn::SinCos(d, hn::LoadU(d, src + i), sv, cv);
    hn::StoreU(sv, d, s + i);
    hn::StoreU(cv, d, c + i);
  }
  for (; i < n; ++i) { s[i] = std::sin(src[i]); c[i] = std::cos(src[i]); }
}
}  // namespace hyle::simd::detail

#endif  // HYLE_ARCH_X64
