#pragma once

#include <chrono>
#include <cstdint>
#include <ctime>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

namespace hyle::time {

// Deliberately timespec-shaped rather than a single nanosecond integer:
// pre-epoch values stay negative and the range is not limited to ~292 years.
// reserved must be zero so the representation can grow without silently
// accepting values that a newer implementation may interpret differently.
struct UnixTimestamp {
  std::int64_t seconds;
  std::uint32_t nanoseconds;
  std::uint32_t reserved;
};

static_assert(sizeof(UnixTimestamp) == 16);
static_assert(std::is_standard_layout_v<UnixTimestamp>);
static_assert(std::is_trivially_copyable_v<UnixTimestamp>);
using UnixTimePoint =
    std::chrono::time_point<std::chrono::system_clock,
                            std::chrono::nanoseconds>;

constexpr bool unix_timestamp_valid(UnixTimestamp value) noexcept {
  return value.nanoseconds < 1'000'000'000u && value.reserved == 0;
}

constexpr UnixTimestamp unix_timestamp(std::int64_t seconds,
                                       std::uint32_t nanoseconds = 0) noexcept {
  return UnixTimestamp{seconds, nanoseconds, 0};
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

}  // namespace hyle::time
