#pragma once

#include <concepts>
#include <limits>
#include <optional>
#include <type_traits>

#include "hyle/core/assert.h"

namespace hyle::math {

namespace detail {

template <class T>
concept NonBooleanIntegral =
    std::integral<T> && !std::same_as<std::remove_cv_t<T>, bool>;

template <NonBooleanIntegral T>
constexpr bool is_non_negative(T value) noexcept {
  if constexpr (std::signed_integral<T>) {
    return value >= T{0};
  } else {
    return true;
  }
}

}  // namespace detail

// Ceiling of value/divisor for a non-negative value and positive divisor.
// The checked form accepts untrusted shape/count input without division by zero.
// Requiring one integral type avoids implicit signed/unsigned promotion traps.
template <detail::NonBooleanIntegral T>
[[nodiscard]] constexpr std::optional<T> try_ceil_div(
    T value, T divisor) noexcept {
  if (!detail::is_non_negative(value) || divisor <= T{0}) {
    return std::nullopt;
  }
  return static_cast<T>(value / divisor +
                        (value % divisor != T{0} ? T{1} : T{0}));
}

template <detail::NonBooleanIntegral T>
[[nodiscard]] constexpr T ceil_div(T value, T divisor) noexcept {
  const auto result = try_ceil_div(value, divisor);
  HYLE_FATAL_ASSERT(result.has_value());
  return *result;
}

// Smallest representable multiple of `multiple` that is >= value. Unlike
// (value + multiple - 1), the checked path cannot overflow before detecting the
// boundary. Address alignment to a power of two remains hyle::mem's concern.
template <detail::NonBooleanIntegral T>
[[nodiscard]] constexpr std::optional<T> try_round_up_to_multiple(
    T value, T multiple) noexcept {
  if (!detail::is_non_negative(value) || multiple <= T{0}) {
    return std::nullopt;
  }
  const T remainder = static_cast<T>(value % multiple);
  if (remainder == T{0}) {
    return value;
  }
  const T increment = static_cast<T>(multiple - remainder);
  if (value > std::numeric_limits<T>::max() - increment) {
    return std::nullopt;
  }
  return static_cast<T>(value + increment);
}

template <detail::NonBooleanIntegral T>
[[nodiscard]] constexpr T round_up_to_multiple(
    T value, T multiple) noexcept {
  const auto result = try_round_up_to_multiple(value, multiple);
  HYLE_FATAL_ASSERT(result.has_value());
  return *result;
}

}  // namespace hyle::math
