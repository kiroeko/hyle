#include "hyle/core/environment.h"

#if HYLE_OS_WINDOWS
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <winsock2.h>
#  include <windows.h>
#endif

#include "hyle/time/native_interop.h"

#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

namespace hyle::time {
namespace {

constexpr std::int64_t kMicrosPerSecond = 1'000'000;

bool assign_timeval_from_microseconds(std::int64_t total_microseconds,
                                      ::timeval& output) noexcept {
  std::int64_t seconds = total_microseconds / kMicrosPerSecond;
  std::int64_t microseconds = total_microseconds % kMicrosPerSecond;
  if (microseconds < 0) {
    --seconds;
    microseconds += kMicrosPerSecond;
  }

  using NativeSeconds = decltype(std::declval<::timeval>().tv_sec);
  using NativeMicros = decltype(std::declval<::timeval>().tv_usec);
  if (!std::in_range<NativeSeconds>(seconds) ||
      !std::in_range<NativeMicros>(microseconds)) {
    return false;
  }

  ::timeval converted{};
  converted.tv_sec = static_cast<NativeSeconds>(seconds);
  converted.tv_usec = static_cast<NativeMicros>(microseconds);
  output = converted;
  return true;
}

}  // namespace

std::optional<Duration> duration_from_timespec(
    const ::timespec& value) noexcept {
  const auto timestamp = unix_timestamp_from_timespec(value);
  if (!timestamp) return std::nullopt;
  const auto point = unix_time_point(*timestamp);
  if (!point) return std::nullopt;
  return point->time_since_epoch();
}

std::optional<::timespec> duration_to_timespec(Duration value) noexcept {
  return unix_timestamp_to_timespec(
      unix_timestamp_from_time_point(UnixTimePoint(value)));
}

std::optional<UnixTimestamp> unix_timestamp_from_timeval(
    const ::timeval& value) noexcept {
  if (!std::in_range<std::int64_t>(value.tv_sec) || value.tv_usec < 0 ||
      value.tv_usec >= kMicrosPerSecond) {
    return std::nullopt;
  }
  return unix_timestamp(
      static_cast<std::int64_t>(value.tv_sec),
      static_cast<std::uint32_t>(value.tv_usec) * 1'000u);
}

bool unix_timestamp_to_timeval(UnixTimestamp value,
                               ::timeval& output) noexcept {
  if (!unix_timestamp_valid(value)) return false;
  using NativeSeconds = decltype(std::declval<::timeval>().tv_sec);
  if (!std::in_range<NativeSeconds>(value.seconds)) return false;

  ::timeval converted{};
  converted.tv_sec = static_cast<NativeSeconds>(value.seconds);
  converted.tv_usec = static_cast<decltype(converted.tv_usec)>(
      value.nanoseconds / 1'000u);
  output = converted;
  return true;
}

std::optional<Duration> duration_from_timeval(
    const ::timeval& value) noexcept {
  const auto timestamp = unix_timestamp_from_timeval(value);
  if (!timestamp) return std::nullopt;
  const auto point = unix_time_point(*timestamp);
  if (!point) return std::nullopt;
  return point->time_since_epoch();
}

bool duration_to_timeval_trunc(Duration value, ::timeval& output) noexcept {
  const std::int64_t total_microseconds = value.count() / 1'000;
  return assign_timeval_from_microseconds(total_microseconds, output);
}

bool timeout_to_timeval(Duration value, ::timeval& output) noexcept {
  if (value <= Duration::zero()) {
    return assign_timeval_from_microseconds(0, output);
  }
  const std::int64_t nanoseconds = value.count();
  const std::int64_t total_microseconds =
      nanoseconds / 1'000 + (nanoseconds % 1'000 != 0 ? 1 : 0);
  return assign_timeval_from_microseconds(total_microseconds, output);
}

#if HYLE_OS_WINDOWS
UnixTimestamp unix_timestamp_from_windows_filetime(
    const ::_FILETIME& value) noexcept {
  const std::uint64_t ticks =
      static_cast<std::uint64_t>(value.dwLowDateTime) |
      (static_cast<std::uint64_t>(value.dwHighDateTime) << 32);
  return unix_timestamp_from_windows_filetime_ticks(ticks);
}

bool unix_timestamp_to_windows_filetime(UnixTimestamp value,
                                        ::_FILETIME& output) noexcept {
  const auto ticks = unix_timestamp_to_windows_filetime_ticks(value);
  if (!ticks) return false;

  ::_FILETIME converted{};
  converted.dwLowDateTime = static_cast<DWORD>(*ticks);
  converted.dwHighDateTime = static_cast<DWORD>(*ticks >> 32);
  output = converted;
  return true;
}
#endif

}  // namespace hyle::time
