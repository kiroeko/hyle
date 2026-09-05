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

#define HYLE_MATH_CHECKED_PROBE(name, type, operation)                       \
  HYLE_NOINLINE bool hyle_math_probe_##name(type lhs, type rhs,              \
                                             type* output) noexcept {         \
    const auto result = hyle::math::operation(lhs, rhs);                     \
    if (!result.has_value()) return false;                                   \
    *output = *result;                                                       \
    return true;                                                             \
  }

HYLE_MATH_CHECKED_PROBE(try_add_u64, std::uint64_t, try_add)
HYLE_MATH_CHECKED_PROBE(try_add_i64, std::int64_t, try_add)
HYLE_MATH_CHECKED_PROBE(try_subtract_u64, std::uint64_t, try_subtract)
HYLE_MATH_CHECKED_PROBE(try_subtract_i64, std::int64_t, try_subtract)
HYLE_MATH_CHECKED_PROBE(try_multiply_u64, std::uint64_t, try_multiply)
HYLE_MATH_CHECKED_PROBE(try_multiply_i64, std::int64_t, try_multiply)

#undef HYLE_MATH_CHECKED_PROBE

HYLE_NOINLINE bool hyle_math_probe_cast_i64_u32(
    std::int64_t value, std::uint32_t* output) noexcept {
  const auto result = hyle::math::try_integer_cast<std::uint32_t>(value);
  if (!result) return false;
  *output = *result;
  return true;
}

HYLE_NOINLINE bool hyle_math_probe_divide_i64(
    std::int64_t lhs, std::int64_t rhs, std::int64_t* output) noexcept {
  const auto result = hyle::math::try_divide(lhs, rhs);
  if (!result) return false;
  *output = *result;
  return true;
}

HYLE_NOINLINE std::int64_t hyle_math_probe_saturating_add_i64(
    std::int64_t lhs, std::int64_t rhs) noexcept {
  return hyle::math::saturating_add(lhs, rhs);
}

HYLE_NOINLINE std::int64_t hyle_math_probe_saturating_multiply_i64(
    std::int64_t lhs, std::int64_t rhs) noexcept {
  return hyle::math::saturating_multiply(lhs, rhs);
}

}  // extern "C"
