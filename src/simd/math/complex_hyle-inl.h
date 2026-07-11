#pragma once

#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>

#include "hyle/macros.h"
#include "xsimd/xsimd.hpp"

namespace hyle::simd::detail {

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

// SoA complex sin/cos. xsimd's batch<complex> path independently computes sin+cos and
// sinh+cosh, duplicating both argument reduction and exp. One real sincos plus one exp/recriprocal
// yields both pairs while preserving signed zero on sinh via copysign.
template <bool kCos>
HYLE_FORCE_INLINE void complex_sincos_hyle(const float* sr, const float* si, float* dr,
                                           float* di, std::size_t n) noexcept {
  using Batch = xsimd::batch<float>;
  constexpr std::size_t width = Batch::size;
  const Batch half(0.5f);
  const Batch one(1.0f);
  std::size_t i = 0;
  for (; i + width <= n; i += width) {
    const Batch x = Batch::load_unaligned(sr + i);
    const Batch y = Batch::load_unaligned(si + i);
    const bool regular = xsimd::all(x != Batch(0.0f)) &&
                         xsimd::all(xsimd::abs(x) <= Batch(3.402823466e38f)) &&
                         xsimd::all(xsimd::abs(y) <= Batch(88.0f));
    const auto sc = xsimd::sincos(x);
    const Batch sx = xsimd::select(x == Batch(0.0f), x, sc.first);
    const Batch t = xsimd::expm1(xsimd::abs(y));
    const Batch e = one + t;
    const Batch ch = one + (t * t) / (Batch(2.0f) * e);
    Batch sh = xsimd::copysign(half * t * (e + one) / e, y);
    sh = xsimd::select(y == Batch(0.0f), y, sh);
    std::uint32_t exceptional = 0;
    float exceptional_re[width];
    float exceptional_im[width];
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
      (sc.second * ch).store_unaligned(dr + i);
      (-(sx * sh)).store_unaligned(di + i);
    } else {
      (sx * ch).store_unaligned(dr + i);
      (sc.second * sh).store_unaligned(di + i);
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
  constexpr std::uint32_t kMaxSafeImag = 0x7EFFffffu;  // max finite / 2: 2 * imag stays finite
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

// SoA complex tanh. xsimd's complex path computes sin/cos and sinh/cosh independently. This
// computes sincos(2y) once and uses expm1(|2x|) to derive both hyperbolic terms without losing
// small real parts. Large real parts instead use exp(-|2x|), avoiding overflow. Near poles and
// other exceptional lanes use std::complex lane-by-lane.
HYLE_FORCE_INLINE void complex_ctanh_hyle(const float* sr, const float* si, float* dr, float* di,
                                          std::size_t n) noexcept {
  using Batch = xsimd::batch<float>;
  constexpr std::size_t width = Batch::size;
  const Batch one(1.0f);
  const Batch half(0.5f);
  const Batch two(2.0f);
  const Batch min_normal(1.17549435e-38f);
  const Batch max_safe_imag(1.701411733e38f);
  std::size_t i = 0;
  for (; i + width <= n; i += width) {
    const Batch x = Batch::load_unaligned(sr + i);
    const Batch y = Batch::load_unaligned(si + i);
    const Batch abs_x = xsimd::abs(x);
    const Batch abs_y = xsimd::abs(y);
    if (xsimd::all(abs_x > Batch(20.0f)) &&
        xsimd::all(abs_x <= Batch(3.402823466e38f)) &&
        xsimd::all(abs_y >= min_normal) && xsimd::all(abs_y <= max_safe_imag)) {
      const auto large_sc = xsimd::sincos(two * y);
      xsimd::copysign(one, x).store_unaligned(dr + i);
      xsimd::copysign(Batch(0.0f), large_sc.first).store_unaligned(di + i);
      continue;
    }
    const auto sc = xsimd::sincos(two * y);
    const Batch t = xsimd::expm1(two * xsimd::min(abs_x, Batch(20.0f)));
    const Batch e = one + t;
    const Batch ch = one + (t * t) / (two * e);
    const Batch sh = xsimd::copysign(half * t * (e + one) / e, x);
    const Batch denominator = ch + sc.second;
    const auto large = abs_x > Batch(20.0f);
    Batch out_re = sh / denominator;
    Batch out_im = sc.first / denominator;
    Batch regular_denominator = xsimd::abs(denominator);
    if (!xsimd::all(abs_x <= Batch(20.0f))) {
      const Batch q = xsimd::exp(-(two * abs_x));
      const Batch q2 = q * q;
      const Batch large_denominator = one + q2 + two * q * sc.second;
      const Batch large_re = xsimd::copysign((one - q2) / large_denominator, x);
      const Batch large_im = (two * q * sc.first) / large_denominator;
      out_re = xsimd::select(large, large_re, out_re);
      out_im = xsimd::select(large, large_im, out_im);
      regular_denominator = xsimd::select(large, one, regular_denominator);
    }
    const bool regular = xsimd::all(abs_x >= min_normal) &&
                         xsimd::all(abs_x <= Batch(3.402823466e38f)) &&
                         xsimd::all(abs_y >= min_normal) &&
                         xsimd::all(abs_y <= max_safe_imag) &&
                         xsimd::all(regular_denominator > Batch(0.01f));
    std::uint32_t exceptional = 0;
    float exceptional_re[width];
    float exceptional_im[width];
    if (!regular) {
      float denominators[width];
      denominator.store_unaligned(denominators);
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
    out_re.store_unaligned(dr + i);
    out_im.store_unaligned(di + i);
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

// SoA complex power. xsimd's generic expression computes log(rho) twice and exp twice when the
// exponent has an imaginary component. Reuse one log(rho), fuse the two linear combinations, then
// evaluate one exp and one sincos. Singular, axis, subnormal and non-finite lanes use std::pow.
HYLE_FORCE_INLINE void complex_cpow_hyle(const float* zr, const float* zi, const float* wr,
                                         const float* wi, float* dr, float* di,
                                         std::size_t n) noexcept {
  using Batch = xsimd::batch<float>;
  constexpr std::size_t width = Batch::size;
  std::size_t i = 0;
  for (; i + width <= n; i += width) {
    const Batch zre = Batch::load_unaligned(zr + i);
    const Batch zim = Batch::load_unaligned(zi + i);
    const Batch wre = Batch::load_unaligned(wr + i);
    const Batch wim = Batch::load_unaligned(wi + i);
    const Batch rho = xsimd::hypot(zre, zim);
    const Batch log_rho = xsimd::log(rho);
    const Batch phi = xsimd::atan2(zim, zre);
    const Batch exponent = wre * log_rho - wim * phi;
    const Batch angle = wre * phi + wim * log_rho;
    const Batch magnitude = xsimd::exp(exponent);
    const auto sc = xsimd::sincos(angle);
    std::uint32_t exceptional = 0;
    float exceptional_re[width];
    float exceptional_im[width];
    const bool regular = xsimd::all(xsimd::abs(zre) >= Batch(1.17549435e-38f)) &&
                         xsimd::all(xsimd::abs(zim) >= Batch(1.17549435e-38f)) &&
                         xsimd::all(xsimd::abs(wre) >= Batch(1.17549435e-38f)) &&
                         xsimd::all(xsimd::abs(wim) >= Batch(1.17549435e-38f)) &&
                         xsimd::all(xsimd::abs(zre) <= Batch(1.0e18f)) &&
                         xsimd::all(xsimd::abs(zim) <= Batch(1.0e18f)) &&
                         xsimd::all(xsimd::abs(wre) <= Batch(1.0e4f)) &&
                         xsimd::all(xsimd::abs(wim) <= Batch(1.0e4f)) &&
                         xsimd::all(exponent > Batch(-80.0f)) &&
                         xsimd::all(exponent < Batch(80.0f));
    if (!regular) {
      float rhos[width];
      float exponents[width];
      rho.store_unaligned(rhos);
      exponent.store_unaligned(exponents);
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
    (magnitude * sc.second).store_unaligned(dr + i);
    (magnitude * sc.first).store_unaligned(di + i);
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
