#pragma once

#include <cmath>
#include <cstdint>

#include "hyle/time/clock.h"

namespace hyle::time::detail {

struct CpuidTscValues {
  std::uint32_t denominator;
  std::uint32_t numerator;
  std::uint32_t crystal_hz;
  std::uint32_t base_mhz;
};

// Pure CPUID.15/.16 derivation shared by production probing and injected tuple
// tests. A missing crystal is approximated as base_MHz * 1e6 * denom / numer.
// The wide multiply/divide prevents a malformed hypervisor tuple from wrapping
// before the result is validated against CPUID's 32-bit crystal field.
constexpr double derive_tsc_hz(CpuidTscValues values) noexcept {
  if (values.denominator == 0 || values.numerator == 0) return 0.0;

  std::uint32_t crystal_hz = values.crystal_hz;
  if (crystal_hz == 0) {
    if (values.base_mhz == 0) return 0.0;
    const std::uint64_t base_hz =
        static_cast<std::uint64_t>(values.base_mhz) * 1'000'000u;
    const WideDivision derived = multiply_divide(
        base_hz, values.denominator, values.numerator);
    if (derived.overflow || derived.quotient == 0 ||
        derived.quotient > UINT32_MAX) {
      return 0.0;
    }
    crystal_hz = static_cast<std::uint32_t>(derived.quotient);
  }

  return static_cast<double>(crystal_hz) *
         static_cast<double>(values.numerator) /
         static_cast<double>(values.denominator);
}

// Convert a measured/derived TSC frequency to the 32.32 counter multiplier.
// Checking against the exactly representable 2^64 boundary before the cast is
// stricter than comparing with double(UINT64_MAX), which rounds up to 2^64.
inline std::uint64_t tsc_scale_multiplier(double hz) noexcept {
  if (!std::isfinite(hz) || !(hz > 1.0)) return 0;
  constexpr double kTwoTo64 = 0x1p64;
  const double multiplier =
      (1e9 / hz) * static_cast<double>(std::uint64_t{1} << 32);
  if (!std::isfinite(multiplier) || multiplier < 1.0 ||
      multiplier >= kTwoTo64) {
    return 0;
  }
  return static_cast<std::uint64_t>(multiplier);
}

}  // namespace hyle::time::detail
