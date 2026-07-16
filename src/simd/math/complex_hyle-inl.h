#pragma once

#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>

#include "hyle/macros.h"
#include "hyle/simd/vec.h"
#include "src/simd/math/math_hyle-inl.h"
#include "src/simd/math/math_xsimd_hyle-inl.h"

// Pure HYLE SoA complex kernels. Requires HYLE_SIMD_TARGET_* defined before vec.h
// in the including TU (same contract as math_hyle / math_xsimd ports).

namespace hyle::simd::detail {
using namespace hyle::simd;
using namespace xsimd_hyle;

HYLE_FORCE_INLINE bool complex_sincos_regular_lane(float real, float imag) noexcept {
  const std::uint32_t real_bits = std::bit_cast<std::uint32_t>(real);
  const std::uint32_t imag_bits = std::bit_cast<std::uint32_t>(imag);
  if ((real_bits & 0x7FFFFFFFu) == 0 ||
      (real_bits & 0x7F800000u) == 0x7F800000u ||
      (imag_bits & 0x7F800000u) == 0x7F800000u) {
    return false;
  }
  return (imag_bits & 0x7FFFFFFFu) <= std::bit_cast<std::uint32_t>(88.0f);
}

// SoA complex sin/cos. One real sincos plus one expm1-derived sinh/cosh pair.
template <bool kCos>
HYLE_FORCE_INLINE void complex_sincos_hyle(const float* sr, const float* si, float* dr,
                                           float* di, std::size_t n) noexcept {
  const ScalableTag<float> d;
  const std::size_t width = Lanes(d);
  const auto half = Set(d, 0.5f);
  const auto one = Set(d, 1.0f);
  const auto two = Set(d, 2.0f);
  std::size_t i = 0;
  for (; i + width <= n; i += width) {
    const auto x = LoadU(d, sr + i);
    const auto y = LoadU(d, si + i);
    const bool regular = AllTrue(d, Ne(x, Zero(d))) &&
                         AllTrue(d, Le(Abs(x), Set(d, 3.402823466e38f))) &&
                         AllTrue(d, Le(Abs(y), Set(d, 88.0f)));
    auto sx = Zero(d), cx = Zero(d);
    SinCos(d, x, sx, cx);
    sx = IfThenElse(Eq(x, Zero(d)), x, sx);
    const auto t = Expm1(d, Abs(y));
    const auto e = Add(one, t);
    const auto ch = Add(one, Div(Mul(t, t), Mul(two, e)));
    auto sh = CopySign(Div(Mul(Mul(half, t), Add(e, one)), e), y);
    sh = IfThenElse(Eq(y, Zero(d)), y, sh);
    std::uint32_t exceptional = 0;
    float exceptional_re[64];
    float exceptional_im[64];
    if (!regular) {
      for (std::size_t lane = 0; lane < width; ++lane) {
        const float real = sr[i + lane];
        const float imag = si[i + lane];
        if (complex_sincos_regular_lane(real, imag)) continue;
        exceptional |= std::uint32_t{1} << lane;
        const std::complex<float> z(real, imag);
        const std::complex<float> value = kCos ? std::cos(z) : std::sin(z);
        exceptional_re[lane] = value.real();
        exceptional_im[lane] = value.imag();
      }
    }

    if constexpr (kCos) {
      StoreU(Mul(cx, ch), d, dr + i);
      StoreU(Neg(Mul(sx, sh)), d, di + i);
    } else {
      StoreU(Mul(sx, ch), d, dr + i);
      StoreU(Mul(cx, sh), d, di + i);
    }
    if (exceptional != 0) {
      for (std::size_t lane = 0; lane < width; ++lane) {
        if ((exceptional & (std::uint32_t{1} << lane)) == 0) continue;
        dr[i + lane] = exceptional_re[lane];
        di[i + lane] = exceptional_im[lane];
      }
    }
  }
  for (; i < n; ++i) {
    const std::complex<float> z(sr[i], si[i]);
    const std::complex<float> value = kCos ? std::cos(z) : std::sin(z);
    dr[i] = value.real();
    di[i] = value.imag();
  }
}

HYLE_FORCE_INLINE bool complex_ctanh_regular_lane(float real, float imag, float denominator) noexcept {
  constexpr std::uint32_t kAbs = 0x7FFFFFFFu;
  constexpr std::uint32_t kExponent = 0x7F800000u;
  constexpr std::uint32_t kMinNormal = 0x00800000u;
  constexpr std::uint32_t kMaxSafeImag = 0x7EFFffffu;
  const std::uint32_t real_bits = std::bit_cast<std::uint32_t>(real);
  const std::uint32_t imag_bits = std::bit_cast<std::uint32_t>(imag);
  const std::uint32_t abs_real = real_bits & kAbs;
  const std::uint32_t abs_imag = imag_bits & kAbs;
  if ((real_bits & kExponent) == kExponent || (imag_bits & kExponent) == kExponent ||
      abs_real < kMinNormal || abs_imag < kMinNormal || abs_imag > kMaxSafeImag) {
    return false;
  }
  if (abs_real > std::bit_cast<std::uint32_t>(20.0f)) return true;
  return std::isfinite(denominator) && std::abs(denominator) > 0.01f;
}

HYLE_FORCE_INLINE void complex_ctanh_hyle(const float* sr, const float* si, float* dr, float* di,
                                          std::size_t n) noexcept {
  const ScalableTag<float> d;
  const std::size_t width = Lanes(d);
  const auto one = Set(d, 1.0f);
  const auto half = Set(d, 0.5f);
  const auto two = Set(d, 2.0f);
  const auto min_normal = Set(d, 1.17549435e-38f);
  const auto max_safe_imag = Set(d, 1.701411733e38f);
  std::size_t i = 0;
  for (; i + width <= n; i += width) {
    const auto x = LoadU(d, sr + i);
    const auto y = LoadU(d, si + i);
    const auto abs_x = Abs(x);
    const auto abs_y = Abs(y);
    if (AllTrue(d, Gt(abs_x, Set(d, 20.0f))) &&
        AllTrue(d, Le(abs_x, Set(d, 3.402823466e38f))) &&
        AllTrue(d, Ge(abs_y, min_normal)) && AllTrue(d, Le(abs_y, max_safe_imag))) {
      auto s = Zero(d), c = Zero(d);
      SinCos(d, Mul(two, y), s, c);
      StoreU(CopySign(one, x), d, dr + i);
      StoreU(CopySign(Zero(d), s), d, di + i);
      continue;
    }
    auto s = Zero(d), c = Zero(d);
    SinCos(d, Mul(two, y), s, c);
    const auto t = Expm1(d, Mul(two, Min(abs_x, Set(d, 20.0f))));
    const auto e = Add(one, t);
    const auto ch = Add(one, Div(Mul(t, t), Mul(two, e)));
    const auto sh = CopySign(Div(Mul(Mul(half, t), Add(e, one)), e), x);
    const auto denominator = Add(ch, c);
    const auto large = Gt(abs_x, Set(d, 20.0f));
    auto out_re = Div(sh, denominator);
    auto out_im = Div(s, denominator);
    auto regular_denominator = Abs(denominator);
    if (!AllTrue(d, Le(abs_x, Set(d, 20.0f)))) {
      const auto q = xsimd_hyle::Exp(d, Neg(Mul(two, abs_x)));
      const auto q2 = Mul(q, q);
      const auto large_denominator = Add(Add(one, q2), Mul(Mul(two, q), c));
      const auto large_re = CopySign(Div(Sub(one, q2), large_denominator), x);
      const auto large_im = Div(Mul(Mul(two, q), s), large_denominator);
      out_re = IfThenElse(large, large_re, out_re);
      out_im = IfThenElse(large, large_im, out_im);
      regular_denominator = IfThenElse(large, one, regular_denominator);
    }
    const bool regular = AllTrue(d, Ge(abs_x, min_normal)) &&
                         AllTrue(d, Le(abs_x, Set(d, 3.402823466e38f))) &&
                         AllTrue(d, Ge(abs_y, min_normal)) &&
                         AllTrue(d, Le(abs_y, max_safe_imag)) &&
                         AllTrue(d, Gt(regular_denominator, Set(d, 0.01f)));
    std::uint32_t exceptional = 0;
    float exceptional_re[64];
    float exceptional_im[64];
    if (!regular) {
      float denominators[64];
      StoreU(denominator, d, denominators);
      for (std::size_t lane = 0; lane < width; ++lane) {
        const float real = sr[i + lane];
        const float imag = si[i + lane];
        if (complex_ctanh_regular_lane(real, imag, denominators[lane])) continue;
        exceptional |= std::uint32_t{1} << lane;
        const std::complex<float> value = std::tanh(std::complex<float>(real, imag));
        exceptional_re[lane] = value.real();
        exceptional_im[lane] = value.imag();
      }
    }
    StoreU(out_re, d, dr + i);
    StoreU(out_im, d, di + i);
    if (exceptional != 0) {
      for (std::size_t lane = 0; lane < width; ++lane) {
        if ((exceptional & (std::uint32_t{1} << lane)) == 0) continue;
        dr[i + lane] = exceptional_re[lane];
        di[i + lane] = exceptional_im[lane];
      }
    }
  }
  for (; i < n; ++i) {
    const std::complex<float> value = std::tanh(std::complex<float>(sr[i], si[i]));
    dr[i] = value.real();
    di[i] = value.imag();
  }
}

HYLE_FORCE_INLINE bool complex_cpow_regular_lane(float zr, float zi, float wr, float wi,
                                                  float rho, float exponent) noexcept {
  const std::uint32_t zr_bits = std::bit_cast<std::uint32_t>(zr);
  const std::uint32_t zi_bits = std::bit_cast<std::uint32_t>(zi);
  const std::uint32_t wr_bits = std::bit_cast<std::uint32_t>(wr);
  const std::uint32_t wi_bits = std::bit_cast<std::uint32_t>(wi);
  constexpr std::uint32_t kAbs = 0x7FFFFFFFu;
  constexpr std::uint32_t kExponent = 0x7F800000u;
  if ((zr_bits & kExponent) == kExponent || (zi_bits & kExponent) == kExponent ||
      (wr_bits & kExponent) == kExponent || (wi_bits & kExponent) == kExponent ||
      (zr_bits & kAbs) < 0x00800000u || (zi_bits & kAbs) < 0x00800000u ||
      (wr_bits & kAbs) < 0x00800000u || (wi_bits & kAbs) < 0x00800000u) {
    return false;
  }
  return std::abs(zr) <= 1.0e18f && std::abs(zi) <= 1.0e18f &&
         std::abs(wr) <= 1.0e4f && std::abs(wi) <= 1.0e4f && std::isfinite(rho) &&
         rho > 0.0f && std::isfinite(exponent) && exponent > -80.0f && exponent < 80.0f;
}

HYLE_FORCE_INLINE void complex_cpow_hyle(const float* zr, const float* zi, const float* wr,
                                         const float* wi, float* dr, float* di,
                                         std::size_t n) noexcept {
  const ScalableTag<float> d;
  const std::size_t width = Lanes(d);
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
    const auto angle = Add(Mul(wre, phi), Mul(wim, log_rho));
    const auto magnitude = xsimd_hyle::Exp(d, exponent);
    auto s = Zero(d), c = Zero(d);
    SinCos(d, angle, s, c);
    std::uint32_t exceptional = 0;
    float exceptional_re[64];
    float exceptional_im[64];
    const bool regular = AllTrue(d, Ge(Abs(zre), Set(d, 1.17549435e-38f))) &&
                         AllTrue(d, Ge(Abs(zim), Set(d, 1.17549435e-38f))) &&
                         AllTrue(d, Ge(Abs(wre), Set(d, 1.17549435e-38f))) &&
                         AllTrue(d, Ge(Abs(wim), Set(d, 1.17549435e-38f))) &&
                         AllTrue(d, Le(Abs(zre), Set(d, 1.0e18f))) &&
                         AllTrue(d, Le(Abs(zim), Set(d, 1.0e18f))) &&
                         AllTrue(d, Le(Abs(wre), Set(d, 1.0e4f))) &&
                         AllTrue(d, Le(Abs(wim), Set(d, 1.0e4f))) &&
                         AllTrue(d, Gt(exponent, Set(d, -80.0f))) &&
                         AllTrue(d, Lt(exponent, Set(d, 80.0f)));
    if (!regular) {
      float rhos[64];
      float exponents[64];
      StoreU(rho, d, rhos);
      StoreU(exponent, d, exponents);
      for (std::size_t lane = 0; lane < width; ++lane) {
        if (complex_cpow_regular_lane(zr[i + lane], zi[i + lane], wr[i + lane], wi[i + lane],
                                      rhos[lane], exponents[lane])) {
          continue;
        }
        exceptional |= std::uint32_t{1} << lane;
        const std::complex<float> value = std::pow(std::complex<float>(zr[i + lane], zi[i + lane]),
                                                    std::complex<float>(wr[i + lane], wi[i + lane]));
        exceptional_re[lane] = value.real();
        exceptional_im[lane] = value.imag();
      }
    }
    StoreU(Mul(magnitude, c), d, dr + i);
    StoreU(Mul(magnitude, s), d, di + i);
    if (exceptional != 0) {
      for (std::size_t lane = 0; lane < width; ++lane) {
        if ((exceptional & (std::uint32_t{1} << lane)) == 0) continue;
        dr[i + lane] = exceptional_re[lane];
        di[i + lane] = exceptional_im[lane];
      }
    }
  }
  for (; i < n; ++i) {
    const std::complex<float> value = std::pow(std::complex<float>(zr[i], zi[i]),
                                                std::complex<float>(wr[i], wi[i]));
    dr[i] = value.real();
    di[i] = value.imag();
  }
}

}  // namespace hyle::simd::detail
