#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <source_location>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

#include "hyle/core/compiler.h"
#include "hyle/error/result.h"
#include "hyle/error/status.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_OBS_API
#elif defined(HYLE_OBS_BUILD)
#define HYLE_OBS_API HYLE_EXPORT
#else
#define HYLE_OBS_API HYLE_IMPORT
#endif

#ifndef HYLE_OBS_COMPILED_MIN_LEVEL
#define HYLE_OBS_COMPILED_MIN_LEVEL 0
#endif
#if HYLE_OBS_COMPILED_MIN_LEVEL < 0 || HYLE_OBS_COMPILED_MIN_LEVEL > 5
#error "HYLE_OBS_COMPILED_MIN_LEVEL must be in [0, 5]"
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
  Off = 5, // threshold-only sentinel; never a record level
};

struct LogField {
  std::string_view key{};
  std::string_view value{};
};

// Borrowed for the synchronous duration of log_record(). Message and field
// slices are exact bytes: they need not be NUL-terminated and may contain NUL.
struct LogRecord {
  LogLevel level = LogLevel::Info;
  std::string_view message{};
  std::span<const LogField> fields{};
  const char *file = nullptr;     // optional source path
  const char *function = nullptr; // optional source function
  std::uint32_t line = 0;
  std::uint64_t timestamp_ns = 0; // optional Unix ns; 0 = not supplied
};

template <class Backend>
concept LogSinkBackend = !std::is_const_v<Backend> &&
                         requires(Backend &backend, const LogRecord &record) {
                           {
                             backend.write(record)
                           } noexcept -> std::same_as<void>;
                         };

// A backend may be called concurrently. It must not throw. Each callback runs
// synchronously on its producer OS thread and must return on that same thread
// without yielding, suspending or migrating a stackful fiber. Copy/enqueue the
// record before returning when asynchronous work is needed. remove_sink() is a
// quiescent unregister: once it returns, no OTHER thread can still be inside
// this callback. A callback may remove itself without deadlocking; naturally,
// its state must remain alive until that callback returns. Removing a different
// sink from a callback is rejected because it may still be pending in the same
// dispatch snapshot; perform cross-removal outside all sink callbacks.
struct LogSink {
  void *state = nullptr;
  void (*write)(void *state, const LogRecord &record) noexcept = nullptr;
  // Optional. flush_sinks() invokes this synchronously; it may run concurrently
  // with write() and therefore follows the same thread-safety, noexcept and
  // no-suspension contract.
  void (*flush)(void *state) noexcept = nullptr;

  // Zero-allocation typed adapter for the normal C++ shape. Backend must expose
  // `void write(const LogRecord&) noexcept` and may expose
  // `void flush() noexcept`. The backend remains borrowed.
  template <LogSinkBackend Backend>
  [[nodiscard]] static LogSink from(Backend &backend) noexcept {
    void (*flush_callback)(void *) noexcept = nullptr;
    if constexpr (requires(Backend &value) { value.flush(); }) {
      static_assert(std::same_as<decltype(backend.flush()), void>,
                    "LogSink backend flush() must return void");
      static_assert(noexcept(backend.flush()),
                    "LogSink backend flush() must be noexcept");
      flush_callback =
          +[](void *state) noexcept { static_cast<Backend *>(state)->flush(); };
    }
    return LogSink{
        .state = &backend,
        .write =
            +[](void *state, const LogRecord &record) noexcept {
              static_cast<Backend *>(state)->write(record);
            },
        .flush = flush_callback,
    };
  }
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

// Move-only RAII ownership of one registry membership. create() registers the
// borrowed LogSink and returns its Status on failure; destruction unregisters
// quiescently. The sink state must therefore be declared before (and outlive)
// this object. Do not also remove the same sink through remove_sink().
class HYLE_OBS_API SinkRegistration final {
public:
  SinkRegistration() noexcept = default;
  ~SinkRegistration() noexcept;

  SinkRegistration(const SinkRegistration &) = delete;
  SinkRegistration &operator=(const SinkRegistration &) = delete;
  SinkRegistration(SinkRegistration &&other) noexcept;
  SinkRegistration &operator=(SinkRegistration &&other) noexcept;

  [[nodiscard]] static Result<SinkRegistration> create(LogSink sink) noexcept;
  template <LogSinkBackend Backend>
  [[nodiscard]] static Result<SinkRegistration>
  create(Backend &backend) noexcept {
    return create(LogSink::from(backend));
  }

  [[nodiscard]] bool registered() const noexcept { return registered_; }
  [[nodiscard]] LogSink sink() const noexcept { return sink_; }

  // Idempotent explicit unregister. On failure ownership is retained so the
  // caller may retry outside a sink callback. Destruction treats such a
  // failure as a fatal lifetime violation rather than leaving a dangling sink.
  Status reset() noexcept;

private:
  explicit SinkRegistration(LogSink sink) noexcept
      : sink_(sink), registered_(true) {}
  void reset_or_abort() noexcept;

  LogSink sink_{};
  bool registered_ = false;
};

// Global threshold. Off is valid as a threshold and silences all records.
HYLE_OBS_API void set_level(LogLevel level) noexcept;
HYLE_OBS_API LogLevel level() noexcept;
HYLE_OBS_API bool should_log(LogLevel level) noexcept;

// Fan out synchronously. Returns true when at least one sink received the
// record. Every borrowed view must remain valid through the synchronous call.
HYLE_OBS_API bool log_record(const LogRecord &record) noexcept;
HYLE_OBS_API bool log_message(LogLevel level,
                              std::string_view message) noexcept;
inline bool log_message(LogLevel level, const char *message) noexcept {
  HYLE_FATAL_ASSERT(message != nullptr);
  return log_message(level, std::string_view(message));
}

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

// The caller has already passed should_log(). This remains an implementation
// seam for the format helpers so the normal macro path pays exactly one level
// load; direct users should call log_record().
HYLE_OBS_API bool log_record_unchecked(const LogRecord &record) noexcept;

template <class... Args>
void log_fmt_unchecked_at(LogLevel log_level, std::source_location source,
                          std::format_string<Args...> fmt,
                          Args &&...args) noexcept {
  char buffer[1024];
  std::string_view message;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    auto result = std::format_to_n(buffer, sizeof(buffer) - 1, fmt,
                                   std::forward<Args>(args)...);
    *result.out = '\0';
    if (result.size >= static_cast<decltype(result.size)>(sizeof(buffer))) {
      constexpr char kMarker[] = "...[truncated]";
      static_assert(sizeof(kMarker) <= sizeof(buffer));
      std::memcpy(buffer + sizeof(buffer) - sizeof(kMarker), kMarker,
                  sizeof(kMarker));
    }
    message =
        std::string_view(buffer, static_cast<std::size_t>(result.out - buffer));
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    // Formatting is an adapter boundary for caller-defined formatters. Logging
    // must never turn a diagnostic failure into application control flow.
    message = "[log formatting failed]";
  }
#endif
  LogRecord record{
      .level = log_level,
      .message = message,
      .file = source.file_name(),
      .function = source.function_name(),
      .line = source.line(),
  };
  (void)log_record_unchecked(record);
}

} // namespace detail

template <class... Args>
void log_fmt_at(LogLevel log_level, std::source_location source,
                std::format_string<Args...> fmt, Args &&...args) noexcept {
  if (!should_log(log_level))
    return;
  detail::log_fmt_unchecked_at(log_level, source, fmt,
                               std::forward<Args>(args)...);
}

template <class... Args>
void log_fmt(LogLevel log_level, std::format_string<Args...> fmt,
             Args &&...args) noexcept {
  log_fmt_at(log_level, std::source_location{}, fmt,
             std::forward<Args>(args)...);
}

} // namespace hyle::obs

#define HYLE_LOG_AT_(level, ...)                                               \
  do {                                                                         \
    if (::hyle::obs::should_log(level)) {                                      \
      ::hyle::obs::detail::log_fmt_unchecked_at(                               \
          level, std::source_location::current(), __VA_ARGS__);                \
    }                                                                          \
  } while (false)
#if HYLE_OBS_COMPILED_MIN_LEVEL <= 0
#define HYLE_LOG_TRACE(...)                                                    \
  HYLE_LOG_AT_(::hyle::obs::LogLevel::Trace, __VA_ARGS__)
#else
#define HYLE_LOG_TRACE(...)                                                    \
  do {                                                                         \
  } while (false)
#endif
#if HYLE_OBS_COMPILED_MIN_LEVEL <= 1
#define HYLE_LOG_DEBUG(...)                                                    \
  HYLE_LOG_AT_(::hyle::obs::LogLevel::Debug, __VA_ARGS__)
#else
#define HYLE_LOG_DEBUG(...)                                                    \
  do {                                                                         \
  } while (false)
#endif
#if HYLE_OBS_COMPILED_MIN_LEVEL <= 2
#define HYLE_LOG_INFO(...)                                                     \
  HYLE_LOG_AT_(::hyle::obs::LogLevel::Info, __VA_ARGS__)
#else
#define HYLE_LOG_INFO(...)                                                     \
  do {                                                                         \
  } while (false)
#endif
#if HYLE_OBS_COMPILED_MIN_LEVEL <= 3
#define HYLE_LOG_WARN(...)                                                     \
  HYLE_LOG_AT_(::hyle::obs::LogLevel::Warn, __VA_ARGS__)
#else
#define HYLE_LOG_WARN(...)                                                     \
  do {                                                                         \
  } while (false)
#endif
#if HYLE_OBS_COMPILED_MIN_LEVEL <= 4
#define HYLE_LOG_ERROR(...)                                                    \
  HYLE_LOG_AT_(::hyle::obs::LogLevel::Error, __VA_ARGS__)
#else
#define HYLE_LOG_ERROR(...)                                                    \
  do {                                                                         \
  } while (false)
#endif
