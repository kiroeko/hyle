#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <source_location>
#include <utility>

#include "hyle/c/obs.h"
#include "hyle/error/status.h"
#include "hyle/macros.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_OBS_API
#elif defined(HYLE_OBS_BUILD)
#  define HYLE_OBS_API HYLE_EXPORT
#else
#  define HYLE_OBS_API HYLE_IMPORT
#endif

// Synchronous structured-log production. Aggregation, storage and alerting stay
// outside HYLE; this module owns the low-level record + fan-out seam.
namespace hyle::obs {

enum class LogLevel : int {
  Trace = 0,
  Debug = 1,
  Info = 2,
  Warn = 3,
  Error = 4,
  Off = 5,  // threshold-only sentinel; never a record level
};

using LogField = hyle_obs_log_field;

// Borrowed for the synchronous duration of log_record(). Strings are UTF-8,
// NUL-terminated and may be null only where explicitly marked optional.
struct LogRecord {
  LogLevel level = LogLevel::Info;
  const char* message = "";
  const LogField* fields = nullptr;
  std::size_t field_count = 0;
  const char* file = nullptr;      // optional source path
  const char* function = nullptr;  // optional source function
  std::uint32_t line = 0;
  std::uint64_t timestamp_ns = 0;  // optional Unix ns; 0 = not supplied
};

// A backend may be called concurrently. It must not throw. remove_sink() is a
// quiescent unregister: once it returns, no OTHER thread can still be inside
// this callback. A callback may remove itself without deadlocking; naturally,
// its state must remain alive until that callback returns. Removing a different
// sink from a callback is rejected because it may still be pending in the same
// dispatch snapshot; perform cross-removal outside all sink callbacks.
struct LogSink {
  void* state = nullptr;
  void (*write)(void* state, const LogRecord& record) noexcept = nullptr;
  // Optional. flush_sinks() invokes this synchronously; it may run concurrently
  // with write() and therefore follows the same thread-safety/noexcept contract.
  void (*flush)(void* state) noexcept = nullptr;
};

// Identity is {state, write}; flush is behavior, not identity. Registry changes
// return the shared HYLE Status taxonomy so callers can distinguish malformed,
// duplicate, absent, full and callback cross-removal cases.
HYLE_OBS_API Status add_sink(LogSink sink) noexcept;
HYLE_OBS_API Status remove_sink(LogSink sink) noexcept;
HYLE_OBS_API std::size_t sink_count() noexcept;
HYLE_OBS_API std::size_t sink_capacity() noexcept;
// Flush every registered sink that provides a callback. Returns the number
// invoked. Nested flush from a sink callback is suppressed, like nested log
// fan-out, to keep arbitrary sink implementations recursion-safe.
HYLE_OBS_API std::size_t flush_sinks() noexcept;

// Global threshold. Off is valid as a threshold and silences all records.
HYLE_OBS_API void set_level(LogLevel level) noexcept;
HYLE_OBS_API LogLevel level() noexcept;
HYLE_OBS_API bool should_log(LogLevel level) noexcept;

// Fan out synchronously. Returns true when at least one sink received the
// record. Contract violations (null message/field strings) fail fast.
HYLE_OBS_API bool log_record(const LogRecord& record) noexcept;
HYLE_OBS_API bool log_message(LogLevel level, const char* message) noexcept;

// Default stderr sink (colored only when stderr is a TTY). It is installed on
// first registry use and can be removed/re-added explicitly. Returned by value;
// identity remains stable.
HYLE_OBS_API LogSink default_sink() noexcept;
HYLE_OBS_API bool default_sink_enabled() noexcept;
// Idempotently reach the requested state. Failure uses the same Status contract
// as add/remove (full, retiring duplicate, or callback cross-removal).
HYLE_OBS_API Status set_default_sink_enabled(bool enabled) noexcept;

// Caller-side format convenience. The fixed 1 KiB buffer avoids allocation;
// oversized messages end in "...[truncated]" rather than being cut silently.
// The source location is attached as structured metadata and macros below
// capture the caller, not this template.
namespace detail {

template <class... Args>
void log_fmt_unchecked_at(LogLevel log_level, std::source_location source,
                          std::format_string<Args...> fmt, Args&&... args) {
  char buffer[1024];
  auto result = std::format_to_n(buffer, sizeof(buffer) - 1, fmt,
                                 std::forward<Args>(args)...);
  *result.out = '\0';
  if (result.size >= static_cast<decltype(result.size)>(sizeof(buffer))) {
    constexpr char kMarker[] = "...[truncated]";
    static_assert(sizeof(kMarker) <= sizeof(buffer));
    std::memcpy(buffer + sizeof(buffer) - sizeof(kMarker), kMarker,
                sizeof(kMarker));
  }
  LogRecord record{
      .level = log_level,
      .message = buffer,
      .file = source.file_name(),
      .function = source.function_name(),
      .line = source.line(),
  };
  (void)log_record(record);
}

}  // namespace detail

template <class... Args>
void log_fmt_at(LogLevel log_level, std::source_location source,
                std::format_string<Args...> fmt, Args&&... args) {
  if (!should_log(log_level)) return;
  detail::log_fmt_unchecked_at(log_level, source, fmt,
                               std::forward<Args>(args)...);
}

template <class... Args>
void log_fmt(LogLevel log_level, std::format_string<Args...> fmt,
             Args&&... args) {
  log_fmt_at(log_level, std::source_location{}, fmt,
             std::forward<Args>(args)...);
}

}  // namespace hyle::obs

#define HYLE_LOG_AT_(level, ...)                                               \
  do {                                                                         \
    if (::hyle::obs::should_log(level)) {                                       \
      ::hyle::obs::detail::log_fmt_unchecked_at(                               \
          level, std::source_location::current(), __VA_ARGS__);                \
    }                                                                          \
  } while (false)
#define HYLE_LOG_TRACE(...) \
  HYLE_LOG_AT_(::hyle::obs::LogLevel::Trace, __VA_ARGS__)
#define HYLE_LOG_DEBUG(...) \
  HYLE_LOG_AT_(::hyle::obs::LogLevel::Debug, __VA_ARGS__)
#define HYLE_LOG_INFO(...) \
  HYLE_LOG_AT_(::hyle::obs::LogLevel::Info, __VA_ARGS__)
#define HYLE_LOG_WARN(...) \
  HYLE_LOG_AT_(::hyle::obs::LogLevel::Warn, __VA_ARGS__)
#define HYLE_LOG_ERROR(...) \
  HYLE_LOG_AT_(::hyle::obs::LogLevel::Error, __VA_ARGS__)
