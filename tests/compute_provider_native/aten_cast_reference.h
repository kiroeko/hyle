#pragma once
#include "hyle/compute/tensor.h"
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace cast_reference {
// Independent nearest-representable and IEEE NaN storage reference.
struct Format {
  hyle::compute::DataType type;
  int exponent, mantissa, bias, width;
  bool infinity;
};
constexpr std::array formats{
    Format{hyle::compute::DataType::kFloat32, 8, 23, 127, 4, true},
    Format{hyle::compute::DataType::kFloat16, 5, 10, 15, 2, true},
    Format{hyle::compute::DataType::kBFloat16, 8, 7, 127, 2, true},
    Format{hyle::compute::DataType::kFloat8E4M3FN, 4, 3, 7, 1, false},
    Format{hyle::compute::DataType::kFloat8E5M2, 5, 2, 15, 1, true}};
double decode(std::uint32_t bits, const Format &f) {
  if (f.width == 4)
    return std::bit_cast<float>(bits);
  const int sign = bits >> (f.exponent + f.mantissa);
  const auto mantissa = bits & ((1U << f.mantissa) - 1);
  const auto exponent = (bits >> f.mantissa) & ((1U << f.exponent) - 1);
  if (exponent == ((1U << f.exponent) - 1) &&
      (f.infinity || mantissa == ((1U << f.mantissa) - 1)))
    return mantissa == 0 && f.infinity
               ? (sign ? -INFINITY : INFINITY)
               : std::numeric_limits<double>::quiet_NaN();
  const auto magnitude =
      exponent == 0 ? std::ldexp(double(mantissa), 1 - f.bias - f.mantissa)
                    : std::ldexp(1 + std::ldexp(double(mantissa), -f.mantissa),
                                 int(exponent) - f.bias);
  return sign ? -magnitude : magnitude;
}
std::uint32_t encode(float value, const Format &f, bool negative_nan) {
  const auto sign =
      std::uint32_t(std::signbit(value) || (std::isnan(value) && negative_nan))
      << (f.exponent + f.mantissa);
  const std::uint32_t nan = f.width == 1       ? (f.infinity ? 0x7e : 0x7f)
                            : f.mantissa == 10 ? 0x7fff
                                               : 0x7fc0;
  if (std::isnan(value))
    return sign | nan;
  const auto infinity = ((1U << f.exponent) - 1) << f.mantissa;
  const auto maximum = f.infinity ? infinity - 1 : 0x7e;
  if (std::isinf(value))
    return sign | (f.infinity ? infinity : maximum);
  const double magnitude = std::abs(double(value));
  if (f.width == 2 &&
      magnitude >=
          std::ldexp(1., ((1 << f.exponent) - 2) - f.bias + 1) -
              std::ldexp(1., ((1 << f.exponent) - 2) - f.bias - f.mantissa - 1))
    return sign | infinity;
  // Independent nearest-representable search; ties select an even encoding.
  unsigned lo = 0, hi = maximum;
  while (lo < hi) {
    const auto mid = (lo + hi) / 2;
    if (decode(mid, f) < magnitude)
      lo = mid + 1;
    else
      hi = mid;
  }
  if (lo > 0) {
    const auto lower = decode(lo - 1, f), upper = decode(lo, f);
    if (magnitude - lower < upper - magnitude ||
        (magnitude - lower == upper - magnitude && ((lo - 1) & 1) == 0))
      --lo;
  }
  return sign | lo;
}
std::uint32_t float_bits(std::uint32_t source, const Format &f) {
  if (f.width == 4)
    return source;
  const auto value = static_cast<float>(decode(source, f));
  if (!std::isnan(value))
    return std::bit_cast<std::uint32_t>(value);
  const auto sign = (source >> (f.exponent + f.mantissa)) << 31;
  if (f.width == 2 && f.mantissa == 10)
    return sign | 0x7f800000 | ((source & 0x3ff) << 13);
  if (f.width == 2)
    return source << 16;
  return f.infinity ? sign | 0x7f800000 | ((source & 3) << 21)
                    : sign | 0x7fc00000;
}

} // namespace cast_reference
