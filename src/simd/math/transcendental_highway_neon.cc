// Highway-accelerated NEON transcendentals (aarch64, 4-wide 128-bit). BORROWS Highway's
// verified contrib/math cores (hn::Exp/Log/Tanh) -- HYLE does not hand-roll these (charter:
// 借验证核不手搓). Highway lowers hn:: to NEON here (probed on Apple Silicon: target=NEON,
// Lanes=4, all three ops match libm at Highway's ~3.5-ULP class).
//
// Unlike the x86 side (which splits xsimd for exp/tanh + Highway for log, a per-op SPEED
// pick from the same-ISA bench), arm64 uses Highway for ALL THREE: one verified core, no
// second library pulled in unproven. A NEON xsimd core could be added later IF a NEON bench
// shows a per-op win -- but Highway-all-three is the honest, validated starting point.
//
// NEON is the ARMv8 baseline (no -march needed, no runtime CPU gate): math_dispatch enters
// this unconditionally on aarch64. scalar libm stays the tail (matches the oracle exactly).
#include "hyle/platform.h"  // HYLE_ARCH_ARM64 -- MUST precede the guard below

#if defined(HYLE_ARCH_ARM64)

#include "src/simd/math/kernels.h"

#include <cmath>
#include <cstddef>

#include "hwy/highway.h"
#include "hwy/contrib/math/math-inl.h"

namespace hn = hwy::HWY_NAMESPACE;

namespace hyle::simd::detail {

void exp_f32_highway_neon(const float* src, float* dst, std::size_t n) noexcept {
  const hn::ScalableTag<float> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w)
    hn::StoreU(hn::Exp(d, hn::LoadU(d, src + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = std::exp(src[i]);  // tail = libm (scalar oracle)
}

void log_f32_highway_neon(const float* src, float* dst, std::size_t n) noexcept {
  const hn::ScalableTag<float> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w)
    hn::StoreU(hn::Log(d, hn::LoadU(d, src + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = std::log(src[i]);  // tail = libm (scalar oracle)
}

void tanh_f32_highway_neon(const float* src, float* dst, std::size_t n) noexcept {
  const hn::ScalableTag<float> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w)
    hn::StoreU(hn::Tanh(d, hn::LoadU(d, src + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = std::tanh(src[i]);  // tail = libm (scalar oracle)
}

// simd-09: NEON uses Highway for ALL new ops (one verified core, consistent -- the x86 per-op
// xsimd/Highway split is an x86 SPEED pick; a NEON bench could later tune per op if it shows a
// win). exp2/erf also go through Highway here (whereas x86 borrows xsimd for those two).
#define HYLE_HWY_MATH_NEON(name, hwyfn, libmfn)                                 \
  void name##_f32_highway_neon(const float* src, float* dst, std::size_t n) noexcept { \
    const hn::ScalableTag<float> d;                                            \
    const std::size_t w = hn::Lanes(d);                                        \
    std::size_t i = 0;                                                         \
    for (; i + w <= n; i += w)                                                 \
      hn::StoreU(hn::hwyfn(d, hn::LoadU(d, src + i)), d, dst + i);              \
    for (; i < n; ++i) dst[i] = libmfn(src[i]);                                \
  }
HYLE_HWY_MATH_NEON(sin, Sin, std::sin)
HYLE_HWY_MATH_NEON(cos, Cos, std::cos)
HYLE_HWY_MATH_NEON(exp2, Exp2, std::exp2)
HYLE_HWY_MATH_NEON(log2, Log2, std::log2)
HYLE_HWY_MATH_NEON(erf, Erf, std::erf)
HYLE_HWY_MATH_NEON(atan, Atan, std::atan)
HYLE_HWY_MATH_NEON(cbrt, Cbrt, std::cbrt)
HYLE_HWY_MATH_NEON(expm1, Expm1, std::expm1)
HYLE_HWY_MATH_NEON(log1p, Log1p, std::log1p)
HYLE_HWY_MATH_NEON(asinh, Asinh, std::asinh)
HYLE_HWY_MATH_NEON(acosh, Acosh, std::acosh)
HYLE_HWY_MATH_NEON(acos, Acos, std::acos)
HYLE_HWY_MATH_NEON(asin, Asin, std::asin)
HYLE_HWY_MATH_NEON(sinh, Sinh, std::sinh)
HYLE_HWY_MATH_NEON(cosh, Cosh, std::cosh)
HYLE_HWY_MATH_NEON(tan, Tan, std::tan)
HYLE_HWY_MATH_NEON(atanh, Atanh, std::atanh)
HYLE_HWY_MATH_NEON(log10, Log10, std::log10)
#undef HYLE_HWY_MATH_NEON

// Full-range hypot core; the explicit fast API uses xsimd-NEON's direct formula.
void hypot_f32_highway_neon(const float* a, const float* b, float* dst, std::size_t n) noexcept {
  const hn::ScalableTag<float> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w)
    hn::StoreU(hn::Hypot(d, hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = std::hypot(a[i], b[i]);
}

// f64 (2-wide NEON) Highway siblings. arm64 uses Highway for ALL f64 ops (one verified core; the
// x86 per-op xsimd/Highway split is an x86 speed pick). hn::ScalableTag<double> = 2 lanes on NEON.
// erf also goes through Highway here (x86 borrows xsimd for erf-f64). exp/log/tanh + 11 unary + hypot.
void exp_f64_highway_neon(const double* src, double* dst, std::size_t n) noexcept {
  const hn::ScalableTag<double> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w) hn::StoreU(hn::Exp(d, hn::LoadU(d, src + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = std::exp(src[i]);
}
void log_f64_highway_neon(const double* src, double* dst, std::size_t n) noexcept {
  const hn::ScalableTag<double> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w) hn::StoreU(hn::Log(d, hn::LoadU(d, src + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = std::log(src[i]);
}
void tanh_f64_highway_neon(const double* src, double* dst, std::size_t n) noexcept {
  const hn::ScalableTag<double> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w) hn::StoreU(hn::Tanh(d, hn::LoadU(d, src + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = std::tanh(src[i]);
}
#define HYLE_HWY_MATH_F64_NEON(name, hwyfn, libmfn)                             \
  void name##_f64_highway_neon(const double* src, double* dst, std::size_t n) noexcept { \
    const hn::ScalableTag<double> d;                                           \
    const std::size_t w = hn::Lanes(d);                                        \
    std::size_t i = 0;                                                         \
    for (; i + w <= n; i += w) hn::StoreU(hn::hwyfn(d, hn::LoadU(d, src + i)), d, dst + i); \
    for (; i < n; ++i) dst[i] = libmfn(src[i]);                                \
  }
HYLE_HWY_MATH_F64_NEON(sin, Sin, std::sin)
HYLE_HWY_MATH_F64_NEON(cos, Cos, std::cos)
HYLE_HWY_MATH_F64_NEON(exp2, Exp2, std::exp2)
HYLE_HWY_MATH_F64_NEON(log2, Log2, std::log2)
HYLE_HWY_MATH_F64_NEON(erf, Erf, std::erf)
HYLE_HWY_MATH_F64_NEON(atan, Atan, std::atan)
HYLE_HWY_MATH_F64_NEON(cbrt, Cbrt, std::cbrt)
HYLE_HWY_MATH_F64_NEON(expm1, Expm1, std::expm1)
HYLE_HWY_MATH_F64_NEON(log1p, Log1p, std::log1p)
HYLE_HWY_MATH_F64_NEON(asinh, Asinh, std::asinh)
HYLE_HWY_MATH_F64_NEON(acosh, Acosh, std::acosh)
HYLE_HWY_MATH_F64_NEON(acos, Acos, std::acos)
HYLE_HWY_MATH_F64_NEON(asin, Asin, std::asin)
HYLE_HWY_MATH_F64_NEON(sinh, Sinh, std::sinh)
HYLE_HWY_MATH_F64_NEON(cosh, Cosh, std::cosh)
HYLE_HWY_MATH_F64_NEON(tan, Tan, std::tan)
HYLE_HWY_MATH_F64_NEON(atanh, Atanh, std::atanh)
HYLE_HWY_MATH_F64_NEON(log10, Log10, std::log10)
#undef HYLE_HWY_MATH_F64_NEON
void hypot_f64_highway_neon(const double* a, const double* b, double* dst, std::size_t n) noexcept {
  const hn::ScalableTag<double> d;
  const std::size_t w = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w)
    hn::StoreU(hn::Hypot(d, hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = std::hypot(a[i], b[i]);
}

// simd-18 批G binary + sincos (Highway NEON) -- MUST stay inside hyle::simd::detail (math_dispatch
// calls detail::pow_f32_highway_neon etc.); a stray namespace close here put them in the global ns
// and broke the NEON link (caught only when Mac finally re-linked batch-G).
void atan2_f32_highway_neon(const float* a, const float* b, float* dst, std::size_t n) noexcept {
  const hn::ScalableTag<float> d; const std::size_t w = hn::Lanes(d); std::size_t i = 0;
  for (; i + w <= n; i += w) hn::StoreU(hn::Atan2(d, hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = std::atan2(a[i], b[i]);
}
void pow_f32_highway_neon(const float* a, const float* b, float* dst, std::size_t n) noexcept {
  const hn::ScalableTag<float> d; const std::size_t w = hn::Lanes(d); std::size_t i = 0;
  for (; i + w <= n; i += w) hn::StoreU(hn::Pow(d, hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = std::pow(a[i], b[i]);
}
void sincos_f32_highway_neon(const float* src, float* s, float* c, std::size_t n) noexcept {
  const hn::ScalableTag<float> d; const std::size_t w = hn::Lanes(d); std::size_t i = 0;
  for (; i + w <= n; i += w) {
    hn::Vec<hn::ScalableTag<float>> sv, cv; hn::SinCos(d, hn::LoadU(d, src + i), sv, cv);
    hn::StoreU(sv, d, s + i); hn::StoreU(cv, d, c + i);
  }
  for (; i < n; ++i) { s[i] = std::sin(src[i]); c[i] = std::cos(src[i]); }
}
void atan2_f64_highway_neon(const double* a, const double* b, double* dst, std::size_t n) noexcept {
  const hn::ScalableTag<double> d; const std::size_t w = hn::Lanes(d); std::size_t i = 0;
  for (; i + w <= n; i += w) hn::StoreU(hn::Atan2(d, hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = std::atan2(a[i], b[i]);
}
void pow_f64_highway_neon(const double* a, const double* b, double* dst, std::size_t n) noexcept {
  const hn::ScalableTag<double> d; const std::size_t w = hn::Lanes(d); std::size_t i = 0;
  for (; i + w <= n; i += w) hn::StoreU(hn::Pow(d, hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, dst + i);
  for (; i < n; ++i) dst[i] = std::pow(a[i], b[i]);
}
void sincos_f64_highway_neon(const double* src, double* s, double* c, std::size_t n) noexcept {
  const hn::ScalableTag<double> d; const std::size_t w = hn::Lanes(d); std::size_t i = 0;
  for (; i + w <= n; i += w) {
    hn::Vec<hn::ScalableTag<double>> sv, cv; hn::SinCos(d, hn::LoadU(d, src + i), sv, cv);
    hn::StoreU(sv, d, s + i); hn::StoreU(cv, d, c + i);
  }
  for (; i < n; ++i) { s[i] = std::sin(src[i]); c[i] = std::cos(src[i]); }
}

}  // namespace hyle::simd::detail
#endif  // HYLE_ARCH_ARM64
