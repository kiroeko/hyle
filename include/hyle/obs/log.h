#pragma once

#include <cstddef>
#include <cstdint>
#include <format>
#include <source_location>
#include <utility>

#include "hyle/c/obs.h"
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
  void* state;
  void (*write)(void* state, const LogRecord& record) noexcept;
};

// Identity is {state, write}. add returns false for a null callback, duplicate,
// retiring duplicate or a full registry. remove returns false when absent or
// when a callback attempts to remove a different sink.
HYLE_OBS_API bool add_sink(LogSink sink) noexcept;
HYLE_OBS_API bool remove_sink(LogSink sink) noexcept;
HYLE_OBS_API std::size_t sink_count() noexcept;
HYLE_OBS_API std::size_t sink_capacity() noexcept;

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
// Idempotently reach the requested state. Returns false only when the registry
// cannot be changed (full, retiring duplicate, or cross-removal from a callback).
HYLE_OBS_API bool set_default_sink_enabled(bool enabled) noexcept;

// Caller-side format convenience. The fixed 1 KiB buffer avoids allocation;
// oversized messages are truncated. The source location is attached as
// structured metadata and macros below capture the caller, not this template.
namespace detail {

template <class... Args>
void log_fmt_unchecked_at(LogLevel log_level, std::source_location source,
                          std::format_string<Args...> fmt, Args&&... args) {
  char buffer[1024];
  auto result = std::format_to_n(buffer, sizeof(buffer) - 1, fmt,
                                 std::forward<Args>(args)...);
  *result.out = '\0';
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
