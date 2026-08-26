#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "hyle/core/compiler.h"
#include "hyle/simd/vec.h"
#include "src/simd/math/math_hyle-inl.h"
#include "src/simd/math/math_xsimd_hyle-inl.h"

// Type-generic HYLE SoA complex kernels. These are deliberately expressed only in the
// public D-tag vocabulary so the same body instantiates for AVX2, NEON, SVE and RVV.
// Exceptional vectors use std::complex before any output store, preserving exact in-place use.

namespace hyle::simd::detail {
using namespace hyle::simd;

template <typename T>
inline constexpr T complex_exp_limit_v = std::is_same_v<T, float> ? T{88} : T{709};

template <typename T>
inline constexpr T complex_magnitude_limit_v =
    [] {
      if constexpr (std::is_same_v<T, float>) {
        return T{1.0e18f};
      } else {
        return T{1.0e150};
      }
    }();

template <typename T>
inline constexpr T complex_magnitude_floor_v =
    std::is_same_v<T, float> ? static_cast<T>(1.0e-19f) : static_cast<T>(1.0e-154);

template <typename T, class Fn>
HYLE_FORCE_INLINE void complex_scalar_chunk(const T* sr, const T* si, T* dr, T* di,
                                            std::size_t first, std::size_t count,
                                            Fn&& fn) noexcept {
  for (std::size_t lane = 0; lane < count; ++lane) {
    const std::complex<T> value = fn(std::complex<T>(sr[first + lane], si[first + lane]));
    dr[first + lane] = value.real();
    di[first + lane] = value.imag();
  }
}

template <typename T>
HYLE_FORCE_INLINE void complex_exp_typed_hyle(const T* sr, const T* si, T* dr, T* di,
                                              std::size_t n) noexcept {
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  const auto limit = Set(d, complex_exp_limit_v<T>);
  std::size_t i = 0;
  for (; i + width <= n; i += width) {
    const auto re = LoadU(d, sr + i);
    const auto im = LoadU(d, si + i);
    if (!AllTrue(d, Le(Abs(re), limit)) ||
        !AllTrue(d, Le(Abs(im), Set(d, complex_magnitude_limit_v<T>)))) {
      complex_scalar_chunk(sr, si, dr, di, i, width,
                           [](std::complex<T> z) { return std::exp(z); });
      continue;
    }
    const auto magnitude = xsimd_hyle::Exp(d, re);
    auto sine = Zero(d), cosine = Zero(d);
    SinCos(d, im, sine, cosine);
    StoreU(Mul(magnitude, cosine), d, dr + i);
    StoreU(Mul(magnitude, sine), d, di + i);
  }
  complex_scalar_chunk(sr, si, dr, di, i, n - i,
                       [](std::complex<T> z) { return std::exp(z); });
}

template <typename T>
HYLE_FORCE_INLINE void complex_log_typed_hyle(const T* sr, const T* si, T* dr, T* di,
                                              std::size_t n) noexcept {
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  const auto limit = Set(d, complex_magnitude_limit_v<T>);
  std::size_t i = 0;
  for (; i + width <= n; i += width) {
    const auto re = LoadU(d, sr + i);
    const auto im = LoadU(d, si + i);
    const auto nonzero = Or(Ne(re, Zero(d)), Ne(im, Zero(d)));
    const auto direct_hypot_safe =
        Or(Ge(Abs(re), Set(d, complex_magnitude_floor_v<T>)),
           Ge(Abs(im), Set(d, complex_magnitude_floor_v<T>)));
    if (!AllTrue(d, Le(Abs(re), limit)) || !AllTrue(d, Le(Abs(im), limit)) ||
        !AllTrue(d, nonzero) || !AllTrue(d, direct_hypot_safe)) {
      complex_scalar_chunk(sr, si, dr, di, i, width,
                           [](std::complex<T> z) { return std::log(z); });
      continue;
    }
    StoreU(Log(d, xsimd_hyle::HypotInRange(d, re, im)), d, dr + i);
    StoreU(Atan2(d, im, re), d, di + i);
  }
  complex_scalar_chunk(sr, si, dr, di, i, n - i,
                       [](std::complex<T> z) { return std::log(z); });
}

template <typename T>
HYLE_FORCE_INLINE void complex_sqrt_typed_hyle(const T* sr, const T* si, T* dr, T* di,
                                               std::size_t n) noexcept {
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  const auto limit = Set(d, complex_magnitude_limit_v<T>);
  const auto half = Set(d, T{0.5});
  std::size_t i = 0;
  for (; i + width <= n; i += width) {
    const auto re = LoadU(d, sr + i);
    const auto im = LoadU(d, si + i);
    const auto nonzero = Or(Ne(re, Zero(d)), Ne(im, Zero(d)));
    const auto direct_hypot_safe =
        Or(Ge(Abs(re), Set(d, complex_magnitude_floor_v<T>)),
           Ge(Abs(im), Set(d, complex_magnitude_floor_v<T>)));
    if (!AllTrue(d, Le(Abs(re), limit)) || !AllTrue(d, Le(Abs(im), limit)) ||
        !AllTrue(d, nonzero) || !AllTrue(d, direct_hypot_safe)) {
      complex_scalar_chunk(sr, si, dr, di, i, width,
                           [](std::complex<T> z) { return std::sqrt(z); });
      continue;
    }
    const auto rho = xsimd_hyle::HypotInRange(d, re, im);
    const auto t = Sqrt(Mul(Add(rho, Abs(re)), half));
    const auto two_t = Add(t, t);
    const auto positive = Ge(re, Zero(d));
    StoreU(IfThenElse(positive, t, Div(Abs(im), two_t)), d, dr + i);
    StoreU(IfThenElse(positive, Div(im, two_t), CopySign(t, im)), d, di + i);
  }
  complex_scalar_chunk(sr, si, dr, di, i, n - i,
                       [](std::complex<T> z) { return std::sqrt(z); });
}

template <typename T, bool kCos>
HYLE_FORCE_INLINE void complex_sincos_typed_hyle(const T* sr, const T* si, T* dr, T* di,
                                                 std::size_t n) noexcept {
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  const auto one = Set(d, T{1});
  const auto half = Set(d, T{0.5});
  const auto two = Set(d, T{2});
  std::size_t i = 0;
  for (; i + width <= n; i += width) {
    const auto x = LoadU(d, sr + i);
    const auto y = LoadU(d, si + i);
    if (!AllTrue(d, Ne(x, Zero(d))) ||
        !AllTrue(d, Le(Abs(x), Set(d, complex_magnitude_limit_v<T>))) ||
        !AllTrue(d, Le(Abs(y), Set(d, complex_exp_limit_v<T>)))) {
      complex_scalar_chunk(sr, si, dr, di, i, width, [](std::complex<T> z) {
        if constexpr (kCos) {
          return std::cos(z);
        } else {
          return std::sin(z);
        }
      });
      continue;
    }
    auto sx = Zero(d), cx = Zero(d);
    SinCos(d, x, sx, cx);
    const auto t = Expm1(d, Abs(y));
    const auto e = Add(one, t);
    const auto ch = Add(one, Div(Mul(t, t), Mul(two, e)));
    const auto sh = CopySign(Div(Mul(Mul(half, t), Add(e, one)), e), y);
    if constexpr (kCos) {
      StoreU(Mul(cx, ch), d, dr + i);
      StoreU(Neg(Mul(sx, sh)), d, di + i);
    } else {
      StoreU(Mul(sx, ch), d, dr + i);
      StoreU(Mul(cx, sh), d, di + i);
    }
  }
  complex_scalar_chunk(sr, si, dr, di, i, n - i, [](std::complex<T> z) {
    if constexpr (kCos) {
      return std::cos(z);
    } else {
      return std::sin(z);
    }
  });
}

template <typename T, bool kCosh>
HYLE_FORCE_INLINE void complex_sinhcosh_typed_hyle(const T* sr, const T* si, T* dr, T* di,
                                                   std::size_t n) noexcept {
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  const auto one = Set(d, T{1});
  const auto half = Set(d, T{0.5});
  const auto two = Set(d, T{2});
  std::size_t i = 0;
  for (; i + width <= n; i += width) {
    const auto x = LoadU(d, sr + i);
    const auto y = LoadU(d, si + i);
    if (!AllTrue(d, Ne(y, Zero(d))) ||
        !AllTrue(d, Le(Abs(x), Set(d, complex_exp_limit_v<T>))) ||
        !AllTrue(d, Le(Abs(y), Set(d, complex_magnitude_limit_v<T>)))) {
      complex_scalar_chunk(sr, si, dr, di, i, width, [](std::complex<T> z) {
        if constexpr (kCosh) {
          return std::cosh(z);
        } else {
          return std::sinh(z);
        }
      });
      continue;
    }
    auto sy = Zero(d), cy = Zero(d);
    SinCos(d, y, sy, cy);
    const auto t = Expm1(d, Abs(x));
    const auto e = Add(one, t);
    const auto ch = Add(one, Div(Mul(t, t), Mul(two, e)));
    const auto sh = CopySign(Div(Mul(Mul(half, t), Add(e, one)), e), x);
    if constexpr (kCosh) {
      StoreU(Mul(ch, cy), d, dr + i);
      StoreU(Mul(sh, sy), d, di + i);
    } else {
      StoreU(Mul(sh, cy), d, dr + i);
      StoreU(Mul(ch, sy), d, di + i);
    }
  }
  complex_scalar_chunk(sr, si, dr, di, i, n - i, [](std::complex<T> z) {
    if constexpr (kCosh) {
      return std::cosh(z);
    } else {
      return std::sinh(z);
    }
  });
}

template <typename T>
HYLE_FORCE_INLINE void complex_tan_typed_hyle(const T* sr, const T* si, T* dr, T* di,
                                              std::size_t n) noexcept {
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  const auto one = Set(d, T{1});
  const auto half = Set(d, T{0.5});
  const auto two = Set(d, T{2});
  const auto epsilon =
      Set(d, std::is_same_v<T, float> ? static_cast<T>(1.0e-4f) : static_cast<T>(1.0e-10));
  std::size_t i = 0;
  for (; i + width <= n; i += width) {
    const auto x = LoadU(d, sr + i);
    const auto y = LoadU(d, si + i);
    if (!AllTrue(d, Le(Abs(x), Set(d, complex_magnitude_limit_v<T>))) ||
        !AllTrue(d, Lt(Abs(y), Set(d, complex_exp_limit_v<T> * T{0.5})))) {
      complex_scalar_chunk(sr, si, dr, di, i, width,
                           [](std::complex<T> z) { return std::tan(z); });
      continue;
    }
    auto sx = Zero(d), cx = Zero(d);
    SinCos(d, Mul(two, x), sx, cx);
    const auto t = Expm1(d, Mul(two, Abs(y)));
    const auto e = Add(one, t);
    const auto ch = Add(one, Div(Mul(t, t), Mul(two, e)));
    const auto sh = CopySign(Div(Mul(Mul(half, t), Add(e, one)), e), y);
    const auto denominator = Add(cx, ch);
    if (!AllTrue(d, Gt(Abs(denominator), epsilon))) {
      complex_scalar_chunk(sr, si, dr, di, i, width,
                           [](std::complex<T> z) { return std::tan(z); });
      continue;
    }
    StoreU(Div(sx, denominator), d, dr + i);
    StoreU(Div(sh, denominator), d, di + i);
  }
  complex_scalar_chunk(sr, si, dr, di, i, n - i,
                       [](std::complex<T> z) { return std::tan(z); });
}

template <typename T>
HYLE_FORCE_INLINE void complex_tanh_typed_hyle(const T* sr, const T* si, T* dr, T* di,
                                               std::size_t n) noexcept {
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  const auto one = Set(d, T{1});
  const auto half = Set(d, T{0.5});
  const auto two = Set(d, T{2});
  const T cutoff = std::is_same_v<T, float> ? static_cast<T>(20) : static_cast<T>(350);
  const auto epsilon =
      Set(d, std::is_same_v<T, float> ? static_cast<T>(0.01f) : static_cast<T>(1.0e-10));
  std::size_t i = 0;
  for (; i + width <= n; i += width) {
    const auto x = LoadU(d, sr + i);
    const auto y = LoadU(d, si + i);
    const auto abs_x = Abs(x);
    if (!AllTrue(d, Ne(x, Zero(d))) || !AllTrue(d, Ne(y, Zero(d))) ||
        !AllTrue(d, Le(abs_x, Set(d, complex_exp_limit_v<T>))) ||
        !AllTrue(d, Le(Abs(y), Set(d, complex_magnitude_limit_v<T>)))) {
      complex_scalar_chunk(sr, si, dr, di, i, width,
                           [](std::complex<T> z) { return std::tanh(z); });
      continue;
    }
    auto sy = Zero(d), cy = Zero(d);
    SinCos(d, Mul(two, y), sy, cy);
    const auto t = Expm1(d, Mul(two, Min(abs_x, Set(d, cutoff))));
    const auto e = Add(one, t);
    const auto ch = Add(one, Div(Mul(t, t), Mul(two, e)));
    const auto sh = CopySign(Div(Mul(Mul(half, t), Add(e, one)), e), x);
    const auto denominator = Add(ch, cy);
    if (!AllTrue(d, Gt(Abs(denominator), epsilon))) {
      complex_scalar_chunk(sr, si, dr, di, i, width,
                           [](std::complex<T> z) { return std::tanh(z); });
      continue;
    }
    StoreU(Div(sh, denominator), d, dr + i);
    StoreU(Div(sy, denominator), d, di + i);
  }
  complex_scalar_chunk(sr, si, dr, di, i, n - i,
                       [](std::complex<T> z) { return std::tanh(z); });
}

template <typename T>
HYLE_FORCE_INLINE void complex_pow_typed_hyle(const T* zr, const T* zi, const T* wr,
                                              const T* wi, T* dr, T* di,
                                              std::size_t n) noexcept {
  const ScalableTag<T> d;
  const std::size_t width = Lanes(d);
  const T exponent_limit = std::is_same_v<T, float> ? T{80} : T{700};
  std::size_t i = 0;
  for (; i + width <= n; i += width) {
    const auto zre = LoadU(d, zr + i);
    const auto zim = LoadU(d, zi + i);
    const auto wre = LoadU(d, wr + i);
    const auto wim = LoadU(d, wi + i);
    const auto rho = xsimd_hyle::HypotInRange(d, zre, zim);
    const auto log_rho = Log(d, rho);
    const auto phi = Atan2(d, zim, zre);
    const auto exponent = Sub(Mul(wre, log_rho), Mul(wim, phi));
    if (!AllTrue(d, Ge(Abs(zre), Set(d, complex_magnitude_floor_v<T>))) ||
        !AllTrue(d, Ge(Abs(zim), Set(d, complex_magnitude_floor_v<T>))) ||
        !AllTrue(d, Ne(wre, Zero(d))) || !AllTrue(d, Ne(wim, Zero(d))) ||
        !AllTrue(d, Le(Abs(zre), Set(d, complex_magnitude_limit_v<T>))) ||
        !AllTrue(d, Le(Abs(zim), Set(d, complex_magnitude_limit_v<T>))) ||
        !AllTrue(d, Gt(exponent, Set(d, -exponent_limit))) ||
        !AllTrue(d, Lt(exponent, Set(d, exponent_limit)))) {
      for (std::size_t lane = 0; lane < width; ++lane) {
        const std::complex<T> value =
            std::pow(std::complex<T>(zr[i + lane], zi[i + lane]),
                     std::complex<T>(wr[i + lane], wi[i + lane]));
        dr[i + lane] = value.real();
        di[i + lane] = value.imag();
      }
      continue;
    }
    const auto angle = Add(Mul(wre, phi), Mul(wim, log_rho));
    const auto magnitude = xsimd_hyle::Exp(d, exponent);
    auto sine = Zero(d), cosine = Zero(d);
    SinCos(d, angle, sine, cosine);
    StoreU(Mul(magnitude, cosine), d, dr + i);
    StoreU(Mul(magnitude, sine), d, di + i);
  }
  for (; i < n; ++i) {
    const std::complex<T> value =
        std::pow(std::complex<T>(zr[i], zi[i]), std::complex<T>(wr[i], wi[i]));
    dr[i] = value.real();
    di[i] = value.imag();
  }
}

}  // namespace hyle::simd::detail
