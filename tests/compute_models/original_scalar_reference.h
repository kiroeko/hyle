#pragma once

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

// Test-only original-coefficient scalar arithmetic. No GTest or model IR.
namespace original_scalar_reference {
inline float widen(std::uint16_t bits) {
  return std::bit_cast<float>(std::uint32_t(bits) << 16U);
}
inline double upper(double value) {
  return std::nextafter(value, std::numeric_limits<double>::infinity());
}
inline double upper_add(double a, double b) { return upper(a + b); }
inline double upper_multiply(double a, double b) { return upper(a * b); }
inline double distance(double a, double b) {
  return a == b ? 0.0 : upper(std::abs(a - b));
}
inline double gamma(std::size_t count, double unit) {
  const double product = double(count) * unit;
  if (product >= 1.0) throw std::runtime_error("original indexer reduction bound is undefined");
  return upper(product / (1.0 - product));
}
struct Sum final {
  double plain = 0.0, correction = 0.0;
  void add(double term) {
    const double next = plain + term;
    correction += std::abs(plain) >= std::abs(term)
        ? (plain - next) + term : (term - next) + plain;
    plain = next;
  }
  double value() const { return plain + correction; }
};
inline void checked_bfloat(std::span<const float> values) {
  for (const float value : values)
    if (!std::isfinite(value) || (std::bit_cast<std::uint32_t>(value) & 0xffffU))
      throw std::runtime_error("original indexer needs actual widened BF16 operands");
}
inline void checked_bfloat(std::span<const std::uint16_t> values) {
  for (const auto value : values)
    if (!std::isfinite(widen(value)))
      throw std::runtime_error("original indexer coefficient or ring is nonfinite");
}
inline std::uint16_t bfloat(float value) {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  return std::uint16_t((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}
} // namespace original_scalar_reference
