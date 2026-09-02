#pragma once

#include <optional>

#include "hyle/core/environment.h"
#include "hyle/time/clock.h"

#if HYLE_OS_WINDOWS
struct timeval;
struct _FILETIME;
#else
#  include <sys/time.h>
#endif

namespace hyle::time {

// Native OS structs are process-local C++ interoperability shapes, not a
// stable wire format or Native FFI ABI. Input subsecond fields must already be
// canonical. Checked output functions leave the destination unchanged when
// the target platform's native seconds field cannot represent the value.

[[nodiscard]] HYLE_TIME_API std::optional<Duration> duration_from_timespec(
    const ::timespec& value) noexcept;
[[nodiscard]] HYLE_TIME_API std::optional<::timespec> duration_to_timespec(
    Duration value) noexcept;

[[nodiscard]] HYLE_TIME_API std::optional<UnixTimestamp>
unix_timestamp_from_timeval(const ::timeval& value) noexcept;
[[nodiscard]] HYLE_TIME_API bool unix_timestamp_to_timeval(
    UnixTimestamp value, ::timeval& output) noexcept;

[[nodiscard]] HYLE_TIME_API std::optional<Duration> duration_from_timeval(
    const ::timeval& value) noexcept;

// Convert after truncating sub-microsecond precision toward zero. This is the
// general signed-duration route; timeout_to_timeval has different semantics.
[[nodiscard]] HYLE_TIME_API bool duration_to_timeval_trunc(
    Duration value, ::timeval& output) noexcept;

// OS timeout boundaries conventionally treat nonpositive durations as a poll.
// Positive fractional microseconds round up so conversion cannot wake early.
[[nodiscard]] HYLE_TIME_API bool timeout_to_timeval(
    Duration value, ::timeval& output) noexcept;

#if HYLE_OS_WINDOWS
[[nodiscard]] HYLE_TIME_API UnixTimestamp
unix_timestamp_from_windows_filetime(const ::_FILETIME& value) noexcept;
[[nodiscard]] HYLE_TIME_API bool unix_timestamp_to_windows_filetime(
    UnixTimestamp value, ::_FILETIME& output) noexcept;
#endif

}  // namespace hyle::time
