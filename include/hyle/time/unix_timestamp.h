#pragma once

#include <chrono>
#include <compare>
#include <cstdint>
#include <ctime>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

namespace hyle::time {

// Deliberately timespec-shaped rather than a single nanosecond integer:
// pre-epoch values stay negative and the range is not limited to ~292 years.
// This is a C++ value type, not a frozen wire/FFI layout; serialization belongs
// at the consuming protocol boundary.
struct UnixTimestamp {
  std::int64_t seconds = 0;
  std::uint32_t nanoseconds = 0;

  constexpr auto operator<=>(const UnixTimestamp&) const noexcept = default;
};

static_assert(std::is_standard_layout_v<UnixTimestamp>);
static_assert(std::is_trivially_copyable_v<UnixTimestamp>);
using UnixTimePoint =
    std::chrono::time_point<std::chrono::system_clock,
                            std::chrono::nanoseconds>;

constexpr bool unix_timestamp_valid(UnixTimestamp value) noexcept {
  return value.nanoseconds < 1'000'000'000u;
}

constexpr UnixTimestamp unix_timestamp(std::int64_t seconds,
                                       std::uint32_t nanoseconds = 0) noexcept {
  return UnixTimestamp{seconds, nanoseconds};
}

namespace unix_timestamp_detail {

constexpr UnixTimestamp from_scaled_units(std::int64_t value,
                                          std::int64_t units_per_second,
                                          std::uint32_t nanos_per_unit) noexcept {
  std::int64_t seconds = value / units_per_second;
  std::int64_t remainder = value % units_per_second;
  if (remainder < 0) {
    --seconds;
    remainder += units_per_second;
  }
  return unix_timestamp(
      seconds,
      static_cast<std::uint32_t>(remainder) * nanos_per_unit);
}

constexpr std::optional<std::int64_t> to_scaled_units(
    UnixTimestamp value, std::int64_t units_per_second,
    std::uint32_t nanos_per_unit) noexcept {
  if (!unix_timestamp_valid(value)) return std::nullopt;

  constexpr std::int64_t kMax = (std::numeric_limits<std::int64_t>::max)();
  constexpr std::int64_t kMin = (std::numeric_limits<std::int64_t>::min)();
  const std::uint64_t scale = static_cast<std::uint64_t>(units_per_second);
  const std::uint64_t fraction = static_cast<std::uint64_t>(
      value.nanoseconds / nanos_per_unit);
  if (value.seconds >= 0) {
    const std::uint64_t seconds =
        static_cast<std::uint64_t>(value.seconds);
    const std::uint64_t max = static_cast<std::uint64_t>(kMax);
    if (seconds > (max - fraction) / scale) return std::nullopt;
    return static_cast<std::int64_t>(seconds * scale + fraction);
  }

  // Avoid an overflowing negative intermediate at INT64_MIN. A canonical
  // negative timestamp is -(abs(seconds + 1) * scale + (scale - fraction)).
  const std::uint64_t preceding_seconds =
      static_cast<std::uint64_t>(-(value.seconds + 1));
  const std::uint64_t tail = scale - fraction;
  const std::uint64_t max_magnitude =
      static_cast<std::uint64_t>(kMax) + 1u;
  if (preceding_seconds > (max_magnitude - tail) / scale) {
    return std::nullopt;
  }
  const std::uint64_t magnitude = preceding_seconds * scale + tail;
  if (magnitude == max_magnitude) return kMin;
  return -static_cast<std::int64_t>(magnitude);
}

}  // namespace unix_timestamp_detail

// Signed Unix-unit adapters keep common database, trace and protocol boundaries
// out of ad-hoc arithmetic. Lower-precision extraction floors toward negative
// infinity, matching the canonical seconds field: Unix -1ns is -1 in every
// integer unit rather than zero. The wider split representation accepts every
// int64 input; extraction can overflow and is therefore checked.
constexpr UnixTimestamp unix_timestamp_from_unix_seconds(
    std::int64_t value) noexcept {
  return unix_timestamp(value);
}

constexpr UnixTimestamp unix_timestamp_from_unix_milliseconds(
    std::int64_t value) noexcept {
  return unix_timestamp_detail::from_scaled_units(value, 1'000,
                                                   1'000'000u);
}

constexpr UnixTimestamp unix_timestamp_from_unix_microseconds(
    std::int64_t value) noexcept {
  return unix_timestamp_detail::from_scaled_units(value, 1'000'000, 1'000u);
}

constexpr UnixTimestamp unix_timestamp_from_unix_nanoseconds(
    std::int64_t value) noexcept {
  return unix_timestamp_detail::from_scaled_units(value, 1'000'000'000, 1u);
}

constexpr std::optional<std::int64_t> unix_timestamp_to_unix_seconds(
    UnixTimestamp value) noexcept {
  return unix_timestamp_detail::to_scaled_units(value, 1, 1'000'000'000u);
}

constexpr std::optional<std::int64_t> unix_timestamp_to_unix_milliseconds(
    UnixTimestamp value) noexcept {
  return unix_timestamp_detail::to_scaled_units(value, 1'000, 1'000'000u);
}

constexpr std::optional<std::int64_t> unix_timestamp_to_unix_microseconds(
    UnixTimestamp value) noexcept {
  return unix_timestamp_detail::to_scaled_units(value, 1'000'000, 1'000u);
}

constexpr std::optional<std::int64_t> unix_timestamp_to_unix_nanoseconds(
    UnixTimestamp value) noexcept {
  return unix_timestamp_detail::to_scaled_units(value, 1'000'000'000, 1u);
}

// Convert the finite range of std::chrono::nanoseconds to canonical floor
// seconds.  Thus -1ns becomes {-1, 999999999}, matching POSIX timespec and
// avoiding truncation-toward-zero surprises at the Unix epoch.
constexpr UnixTimestamp unix_timestamp_from_time_point(
    UnixTimePoint point) noexcept {
  constexpr std::int64_t kNanosPerSecond = 1'000'000'000;
  const std::int64_t total = point.time_since_epoch().count();
  std::int64_t seconds = total / kNanosPerSecond;
  std::int64_t nanoseconds = total % kNanosPerSecond;
  if (nanoseconds < 0) {
    --seconds;
    nanoseconds += kNanosPerSecond;
  }
  return unix_timestamp(seconds, static_cast<std::uint32_t>(nanoseconds));
}

// UnixTimestamp itself has a much wider range than a nanosecond chrono point.
// Refuse an unrepresentable conversion rather than saturating a file timestamp
// into a different instant.
constexpr std::optional<UnixTimePoint> unix_time_point(
    UnixTimestamp value) noexcept {
  if (!unix_timestamp_valid(value)) return std::nullopt;
  constexpr std::int64_t kNanosPerSecond = 1'000'000'000;
  constexpr std::int64_t kMax = (std::numeric_limits<std::int64_t>::max)();
  constexpr std::int64_t kMin = (std::numeric_limits<std::int64_t>::min)();
  constexpr std::int64_t kMaxSeconds = kMax / kNanosPerSecond;
  constexpr std::uint32_t kMaxNanoseconds =
      static_cast<std::uint32_t>(kMax % kNanosPerSecond);
  constexpr std::int64_t kMinSeconds = kMin / kNanosPerSecond - 1;
  constexpr std::uint32_t kMinNanoseconds =
      static_cast<std::uint32_t>(kNanosPerSecond + kMin % kNanosPerSecond);
  if (value.seconds > kMaxSeconds ||
      (value.seconds == kMaxSeconds && value.nanoseconds > kMaxNanoseconds) ||
      value.seconds < kMinSeconds ||
      (value.seconds == kMinSeconds && value.nanoseconds < kMinNanoseconds)) {
    return std::nullopt;
  }
  if (value.seconds == kMinSeconds) {
    return UnixTimePoint(std::chrono::nanoseconds(
        kMin + static_cast<std::int64_t>(value.nanoseconds - kMinNanoseconds)));
  }
  const std::int64_t whole = value.seconds * kNanosPerSecond;
  const std::int64_t fraction = static_cast<std::int64_t>(value.nanoseconds);
  return UnixTimePoint(std::chrono::nanoseconds(whole + fraction));
}

constexpr std::optional<UnixTimestamp> unix_timestamp_from_time_t(
    std::time_t value) noexcept {
  if (!std::in_range<std::int64_t>(value)) return std::nullopt;
  return unix_timestamp(static_cast<std::int64_t>(value));
}

// Fractional seconds are discarded toward negative infinity because seconds
// is the canonical floor field (-1ns therefore converts to time_t(-1)).
constexpr std::optional<std::time_t> unix_timestamp_to_time_t(
    UnixTimestamp value) noexcept {
  if (!unix_timestamp_valid(value) ||
      !std::in_range<std::time_t>(value.seconds)) {
    return std::nullopt;
  }
  return static_cast<std::time_t>(value.seconds);
}

constexpr std::optional<UnixTimestamp> unix_timestamp_from_timespec(
    const ::timespec& value) noexcept {
  if (value.tv_nsec < 0 || value.tv_nsec >= 1'000'000'000L ||
      !std::in_range<std::int64_t>(value.tv_sec)) {
    return std::nullopt;
  }
  return unix_timestamp(static_cast<std::int64_t>(value.tv_sec),
                        static_cast<std::uint32_t>(value.tv_nsec));
}

constexpr std::optional<::timespec> unix_timestamp_to_timespec(
    UnixTimestamp value) noexcept {
  if (!unix_timestamp_valid(value) ||
      !std::in_range<decltype(std::declval<::timespec>().tv_sec)>(
          value.seconds)) {
    return std::nullopt;
  }
  ::timespec result{};
  result.tv_sec = static_cast<decltype(result.tv_sec)>(value.seconds);
  result.tv_nsec = static_cast<decltype(result.tv_nsec)>(value.nanoseconds);
  return result;
}

// Windows FILETIME is an unsigned count of 100ns ticks since 1601.  Accept the
// raw count so this header does not pull windows.h into public users.  The
// conversion remains exact and canonical for timestamps before 1970.
constexpr UnixTimestamp unix_timestamp_from_windows_filetime_ticks(
    std::uint64_t ticks_100ns) noexcept {
  constexpr std::uint64_t kTicksPerSecond = 10'000'000;
  constexpr std::uint64_t kUnixEpochTicks = 116'444'736'000'000'000;
  if (ticks_100ns >= kUnixEpochTicks) {
    const std::uint64_t delta = ticks_100ns - kUnixEpochTicks;
    return unix_timestamp(
        static_cast<std::int64_t>(delta / kTicksPerSecond),
        static_cast<std::uint32_t>((delta % kTicksPerSecond) * 100));
  }

  const std::uint64_t delta = kUnixEpochTicks - ticks_100ns;
  const std::uint64_t whole = delta / kTicksPerSecond;
  const std::uint64_t remainder = delta % kTicksPerSecond;
  if (remainder == 0) {
    return unix_timestamp(-static_cast<std::int64_t>(whole));
  }
  return unix_timestamp(
      -static_cast<std::int64_t>(whole) - 1,
      static_cast<std::uint32_t>((kTicksPerSecond - remainder) * 100));
}

// Convert to Windows FILETIME's unsigned 100ns ticks since 1601 without
// exposing windows.h in the public header. FILETIME cannot represent instants
// before 1601 or beyond UINT64_MAX ticks; those return nullopt. Nanoseconds that
// are not 100ns-aligned are floored to the preceding FILETIME tick. In
// particular, Unix -1ns becomes the tick at Unix -100ns rather than being
// truncated toward the epoch. Every value produced by the forward conversion
// is aligned and therefore round-trips exactly.
constexpr std::optional<std::uint64_t>
unix_timestamp_to_windows_filetime_ticks(UnixTimestamp value) noexcept {
  if (!unix_timestamp_valid(value)) return std::nullopt;

  constexpr std::uint64_t kTicksPerSecond = 10'000'000;
  constexpr std::uint64_t kUnixEpochTicks = 116'444'736'000'000'000;
  const std::uint64_t fraction_ticks = value.nanoseconds / 100u;

  if (value.seconds >= 0) {
    const std::uint64_t seconds = static_cast<std::uint64_t>(value.seconds);
    if (seconds >
        (UINT64_MAX - kUnixEpochTicks - fraction_ticks) / kTicksPerSecond) {
      return std::nullopt;
    }
    return kUnixEpochTicks + seconds * kTicksPerSecond + fraction_ticks;
  }

  // -(seconds + 1) is representable even for INT64_MIN. The timestamp is
  // `-(whole_seconds + 1) + fraction`, so its distance before the Unix epoch is
  // `whole_seconds + (1 - fraction)` in 100ns ticks.
  const std::uint64_t whole_seconds =
      static_cast<std::uint64_t>(-(value.seconds + 1));
  const std::uint64_t partial_ticks = kTicksPerSecond - fraction_ticks;
  if (whole_seconds >
      (kUnixEpochTicks - partial_ticks) / kTicksPerSecond) {
    return std::nullopt;
  }
  const std::uint64_t before_epoch =
      whole_seconds * kTicksPerSecond + partial_ticks;
  return kUnixEpochTicks - before_epoch;
}

}  // namespace hyle::time
