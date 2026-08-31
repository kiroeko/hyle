#pragma once

#include <bit>
#include <concepts>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
#  include <intrin.h>
#endif

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

template <NonBooleanIntegral T>
using Unsigned = std::make_unsigned_t<T>;

template <NonBooleanIntegral T>
constexpr T from_unsigned_bits(Unsigned<T> value) noexcept {
  if constexpr (std::signed_integral<T>) {
    return std::bit_cast<T>(value);
  } else {
    return value;
  }
}

template <NonBooleanIntegral T>
constexpr std::optional<T> add_portable(T lhs, T rhs) noexcept {
  using U = Unsigned<T>;
  const U lhs_bits = static_cast<U>(lhs);
  const U rhs_bits = static_cast<U>(rhs);
  const U result_bits = static_cast<U>(lhs_bits + rhs_bits);
  if constexpr (std::unsigned_integral<T>) {
    if (result_bits < lhs_bits) return std::nullopt;
  } else {
    constexpr U kSignBit = U{1} << (std::numeric_limits<U>::digits - 1);
    const U overflow_bits = static_cast<U>(
        static_cast<U>(~static_cast<U>(lhs_bits ^ rhs_bits)) &
        static_cast<U>(lhs_bits ^ result_bits));
    if ((overflow_bits & kSignBit) != U{0}) return std::nullopt;
  }
  return from_unsigned_bits<T>(result_bits);
}

template <NonBooleanIntegral T>
constexpr std::optional<T> subtract_portable(T lhs, T rhs) noexcept {
  using U = Unsigned<T>;
  const U lhs_bits = static_cast<U>(lhs);
  const U rhs_bits = static_cast<U>(rhs);
  const U result_bits = static_cast<U>(lhs_bits - rhs_bits);
  if constexpr (std::unsigned_integral<T>) {
    if (lhs_bits < rhs_bits) return std::nullopt;
  } else {
    constexpr U kSignBit = U{1} << (std::numeric_limits<U>::digits - 1);
    const U overflow_bits = static_cast<U>(
        static_cast<U>(lhs_bits ^ rhs_bits) &
        static_cast<U>(lhs_bits ^ result_bits));
    if ((overflow_bits & kSignBit) != U{0}) return std::nullopt;
  }
  return from_unsigned_bits<T>(result_bits);
}

template <NonBooleanIntegral T>
constexpr std::optional<T> multiply_portable(T lhs, T rhs) noexcept {
  constexpr T kMin = (std::numeric_limits<T>::min)();
  constexpr T kMax = (std::numeric_limits<T>::max)();
  if constexpr (std::unsigned_integral<T>) {
    if (rhs != T{0} && lhs > static_cast<T>(kMax / rhs)) {
      return std::nullopt;
    }
  } else {
    if (lhs == T{0} || rhs == T{0}) return T{0};
    if (lhs > T{0}) {
      if ((rhs > T{0} && lhs > static_cast<T>(kMax / rhs)) ||
          (rhs < T{0} && rhs < static_cast<T>(kMin / lhs))) {
        return std::nullopt;
      }
    } else if ((rhs > T{0} && lhs < static_cast<T>(kMin / rhs)) ||
               (rhs < T{0} && lhs < static_cast<T>(kMax / rhs))) {
      return std::nullopt;
    }
  }
  return static_cast<T>(lhs * rhs);
}

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
template <NonBooleanIntegral T>
inline std::optional<T> multiply_msvc(T lhs, T rhs) noexcept {
  if constexpr (sizeof(T) < sizeof(std::uint64_t)) {
    if constexpr (std::signed_integral<T>) {
      const std::int64_t product = static_cast<std::int64_t>(lhs) *
                                   static_cast<std::int64_t>(rhs);
      if (product < static_cast<std::int64_t>((std::numeric_limits<T>::min)()) ||
          product > static_cast<std::int64_t>((std::numeric_limits<T>::max)())) {
        return std::nullopt;
      }
      return static_cast<T>(product);
    } else {
      const std::uint64_t product = static_cast<std::uint64_t>(lhs) *
                                    static_cast<std::uint64_t>(rhs);
      if (product > static_cast<std::uint64_t>((std::numeric_limits<T>::max)())) {
        return std::nullopt;
      }
    return static_cast<T>(product);
    }
  } else if constexpr (sizeof(T) == sizeof(std::uint64_t)) {
    if constexpr (std::signed_integral<T>) {
      std::int64_t high;
#  if defined(_M_X64)
      const std::int64_t low = _mul128(static_cast<std::int64_t>(lhs),
                                       static_cast<std::int64_t>(rhs), &high);
#  else
      const std::int64_t low = static_cast<std::int64_t>(
          static_cast<std::uint64_t>(lhs) * static_cast<std::uint64_t>(rhs));
      high = __mulh(static_cast<std::int64_t>(lhs),
                    static_cast<std::int64_t>(rhs));
#  endif
      if (high != (low < 0 ? std::int64_t{-1} : std::int64_t{0})) {
        return std::nullopt;
      }
      return static_cast<T>(low);
    } else {
      std::uint64_t high;
#  if defined(_M_X64)
      const std::uint64_t low = _umul128(static_cast<std::uint64_t>(lhs),
                                         static_cast<std::uint64_t>(rhs), &high);
#  else
      const std::uint64_t low = static_cast<std::uint64_t>(lhs) *
                                static_cast<std::uint64_t>(rhs);
      high = __umulh(static_cast<std::uint64_t>(lhs),
                     static_cast<std::uint64_t>(rhs));
#  endif
      if (high != 0) return std::nullopt;
      return static_cast<T>(low);
    }
  } else {
    return multiply_portable(lhs, rhs);
  }
}
#endif

}  // namespace detail

// Overflow-safe arithmetic for untrusted shape, stride, offset and byte-count
// inputs. Successful results are exact; failure returns nullopt rather than a
// wrapped or saturated value. Trusted ordinary arithmetic should keep using the
// language operators directly.
template <detail::NonBooleanIntegral T>
[[nodiscard]] constexpr std::optional<T> try_add(T lhs, T rhs) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  if (!std::is_constant_evaluated()) {
    T result{};
    if (__builtin_add_overflow(lhs, rhs, &result)) return std::nullopt;
    return result;
  }
#endif
  return detail::add_portable(lhs, rhs);
}

template <detail::NonBooleanIntegral T>
[[nodiscard]] constexpr std::optional<T> try_subtract(T lhs, T rhs) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  if (!std::is_constant_evaluated()) {
    T result{};
    if (__builtin_sub_overflow(lhs, rhs, &result)) return std::nullopt;
    return result;
  }
#endif
  return detail::subtract_portable(lhs, rhs);
}

template <detail::NonBooleanIntegral T>
[[nodiscard]] constexpr std::optional<T> try_multiply(T lhs, T rhs) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  if (!std::is_constant_evaluated()) {
    T result{};
    if (__builtin_mul_overflow(lhs, rhs, &result)) return std::nullopt;
    return result;
  }
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
  if (!std::is_constant_evaluated()) {
    return detail::multiply_msvc(lhs, rhs);
  }
#endif
  return detail::multiply_portable(lhs, rhs);
}

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
