#include <cstdint>

#include "hyle/core/compiler.h"
#include "hyle/math/scalar.h"

extern "C" {

HYLE_NOINLINE std::uint64_t hyle_math_probe_ceil(
    std::uint64_t value, std::uint64_t divisor) noexcept {
  return hyle::math::ceil_div(value, divisor);
}

HYLE_NOINLINE std::uint64_t hyle_math_probe_round(
    std::uint64_t value, std::uint64_t multiple) noexcept {
  return hyle::math::round_up_to_multiple(value, multiple);
}

HYLE_NOINLINE bool hyle_math_probe_try_round(
    std::uint64_t value, std::uint64_t multiple,
    std::uint64_t* output) noexcept {
  const auto result = hyle::math::try_round_up_to_multiple(value, multiple);
  if (!result.has_value()) return false;
  *output = *result;
  return true;
}

}  // extern "C"
