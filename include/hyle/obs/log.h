#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <optional>
#include <source_location>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "hyle/core/compiler.h"
#include "hyle/error/result.h"
#include "hyle/error/status.h"
#include "hyle/obs/log_control.h"
#include "hyle/time/unix_timestamp.h"
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

enum class LogValueKind : std::uint8_t {
  String,
  SignedInteger,
  UnsignedInteger,
  FloatingPoint,
  Boolean,
};

// Allocation-free scalar vocabulary for logger-neutral structured events.
// String values are exact borrowed bytes; the numeric alternatives normalize
// common C++ scalar widths so sinks do not need a runtime type registry.
class LogValue final {
public:
  constexpr LogValue() noexcept = default;
  constexpr LogValue(std::string_view value) noexcept : storage_(value) {}

  template <std::size_t N>
  constexpr LogValue(const char (&value)[N]) noexcept
      : storage_(std::string_view(value, N != 0 && value[N - 1] == '\0' ? N - 1
                                                                        : N)) {}

  template <class Traits, class Allocator>
  constexpr LogValue(
      const std::basic_string<char, Traits, Allocator> &value) noexcept
      : storage_(std::string_view(value.data(), value.size())) {}
  template <class Traits, class Allocator>
  LogValue(std::basic_string<char, Traits, Allocator> &&) = delete;
  template <class Traits, class Allocator>
  LogValue(const std::basic_string<char, Traits, Allocator> &&) = delete;

  template <std::signed_integral Integer>
    requires(!std::same_as<std::remove_cv_t<Integer>, bool> &&
             sizeof(Integer) <= sizeof(std::int64_t))
  constexpr LogValue(Integer value) noexcept
      : kind_(LogValueKind::SignedInteger),
        storage_(static_cast<std::int64_t>(value)) {}

  template <std::unsigned_integral Integer>
    requires(!std::same_as<std::remove_cv_t<Integer>, bool> &&
             sizeof(Integer) <= sizeof(std::uint64_t))
  constexpr LogValue(Integer value) noexcept
      : kind_(LogValueKind::UnsignedInteger),
        storage_(static_cast<std::uint64_t>(value)) {}

  constexpr LogValue(float value) noexcept
      : kind_(LogValueKind::FloatingPoint),
        storage_(static_cast<double>(value)) {}
  constexpr LogValue(double value) noexcept
      : kind_(LogValueKind::FloatingPoint), storage_(value) {}
  constexpr LogValue(bool value) noexcept
      : kind_(LogValueKind::Boolean), storage_(value) {}

  LogValue(long double) = delete;
  LogValue(std::nullptr_t) = delete;
  template <class Enum>
    requires std::is_enum_v<Enum>
  LogValue(Enum) = delete;
  template <class Pointer>
    requires std::is_pointer_v<Pointer>
  LogValue(Pointer) = delete;

  [[nodiscard]] constexpr LogValueKind kind() const noexcept { return kind_; }

  [[nodiscard]] constexpr std::string_view as_string() const noexcept {
    HYLE_FATAL_ASSERT(kind_ == LogValueKind::String);
    return storage_.string;
  }
  [[nodiscard]] constexpr std::int64_t as_signed_integer() const noexcept {
    HYLE_FATAL_ASSERT(kind_ == LogValueKind::SignedInteger);
    return storage_.signed_integer;
  }
  [[nodiscard]] constexpr std::uint64_t as_unsigned_integer() const noexcept {
    HYLE_FATAL_ASSERT(kind_ == LogValueKind::UnsignedInteger);
    return storage_.unsigned_integer;
  }
  [[nodiscard]] constexpr double as_floating_point() const noexcept {
    HYLE_FATAL_ASSERT(kind_ == LogValueKind::FloatingPoint);
    return storage_.floating_point;
  }
  [[nodiscard]] constexpr bool as_boolean() const noexcept {
    HYLE_FATAL_ASSERT(kind_ == LogValueKind::Boolean);
    return storage_.boolean;
  }

private:
  union Storage {
    constexpr Storage() noexcept : string{} {}
    constexpr explicit Storage(std::string_view value) noexcept
        : string(value) {}
    constexpr explicit Storage(std::int64_t value) noexcept
        : signed_integer(value) {}
    constexpr explicit Storage(std::uint64_t value) noexcept
        : unsigned_integer(value) {}
    constexpr explicit Storage(double value) noexcept : floating_point(value) {}
    constexpr explicit Storage(bool value) noexcept : boolean(value) {}

    std::string_view string;
    std::int64_t signed_integer;
    std::uint64_t unsigned_integer;
    double floating_point;
    bool boolean;
  };

  LogValueKind kind_ = LogValueKind::String;
  Storage storage_{};
};

static_assert(std::is_standard_layout_v<LogValue>);
static_assert(std::is_trivially_copyable_v<LogValue>);

struct LogAttribute {
  std::string_view key{};
  LogValue value{};
};

static_assert(std::is_standard_layout_v<LogAttribute>);
static_assert(std::is_trivially_copyable_v<LogAttribute>);

// A separate value type keeps standard C++ call-site capture convenient while
// allowing bridges to supply source locations from another logger/runtime.
struct LogSourceLocation {
  std::string_view file{};
  std::string_view function{};
  std::uint32_t line = 0;
  std::uint32_t column = 0;

  [[nodiscard]] static constexpr LogSourceLocation
  from(std::source_location source) noexcept {
    return LogSourceLocation{
        .file = source.file_name(),
        .function = source.function_name(),
        .line = source.line(),
        .column = source.column(),
    };
  }
};

static_assert(std::is_standard_layout_v<LogSourceLocation>);
static_assert(std::is_trivially_copyable_v<LogSourceLocation>);

// Borrowed for the synchronous duration of log_record(). Every string and span
// is an exact byte view: it need not be NUL-terminated and may contain NUL.
// Attributes preserve order and duplicate/empty keys. Correlation attributes
// use the same scalar vocabulary but remain semantically distinct so a host
// adapter can index request/slot/trace identity without parsing ordinary data.
struct LogRecord {
  LogLevel level = LogLevel::Info;
  std::string_view category{};
  std::string_view event{};
  std::string_view message{};
  std::span<const LogAttribute> attributes{};
  std::span<const LogAttribute> correlation_attributes{};
  LogSourceLocation source{};
  std::optional<time::UnixTimestamp> timestamp{};
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
      .source = LogSourceLocation::from(source),
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
#define HYLE_LOG_IF_AT_(level, condition, ...)                                 \
  do {                                                                         \
    if (::hyle::obs::should_log(level) && static_cast<bool>(condition)) {      \
      ::hyle::obs::detail::log_fmt_unchecked_at(                               \
          level, std::source_location::current(), __VA_ARGS__);                \
    }                                                                          \
  } while (false)
#define HYLE_LOG_EVERY_N_AT_(level, n, ...)                                    \
  do {                                                                         \
    if (::hyle::obs::should_log(level)) {                                      \
      static ::hyle::obs::LogEveryNState hyle_log_control_state;               \
      if (hyle_log_control_state.should_log(n)) {                              \
        ::hyle::obs::detail::log_fmt_unchecked_at(                             \
            level, std::source_location::current(), __VA_ARGS__);              \
      }                                                                        \
    }                                                                          \
  } while (false)
#define HYLE_LOG_EVERY_N_EXACT_AT_(level, n, ...)                              \
  do {                                                                         \
    if (::hyle::obs::should_log(level)) {                                      \
      static ::hyle::obs::LogEveryNExactState hyle_log_control_state;          \
      if (hyle_log_control_state.should_log(n)) {                              \
        ::hyle::obs::detail::log_fmt_unchecked_at(                             \
            level, std::source_location::current(), __VA_ARGS__);              \
      }                                                                        \
    }                                                                          \
  } while (false)
#define HYLE_LOG_FIRST_N_AT_(level, n, ...)                                    \
  do {                                                                         \
    if (::hyle::obs::should_log(level)) {                                      \
      static ::hyle::obs::LogFirstNState hyle_log_control_state;               \
      if (hyle_log_control_state.should_log(n)) {                              \
        ::hyle::obs::detail::log_fmt_unchecked_at(                             \
            level, std::source_location::current(), __VA_ARGS__);              \
      }                                                                        \
    }                                                                          \
  } while (false)
#define HYLE_LOG_EVERY_POW_2_AT_(level, ...)                                   \
  do {                                                                         \
    if (::hyle::obs::should_log(level)) {                                      \
      static ::hyle::obs::LogEveryPow2State hyle_log_control_state;            \
      if (hyle_log_control_state.should_log()) {                               \
        ::hyle::obs::detail::log_fmt_unchecked_at(                             \
            level, std::source_location::current(), __VA_ARGS__);              \
      }                                                                        \
    }                                                                          \
  } while (false)
#define HYLE_LOG_EVERY_AT_(level, interval, ...)                               \
  do {                                                                         \
    if (::hyle::obs::should_log(level)) {                                      \
      static ::hyle::obs::LogIntervalLimiter hyle_log_control_state;           \
      if (hyle_log_control_state.should_log(interval)) {                       \
        ::hyle::obs::detail::log_fmt_unchecked_at(                             \
            level, std::source_location::current(), __VA_ARGS__);              \
      }                                                                        \
    }                                                                          \
  } while (false)
#if HYLE_OBS_COMPILED_MIN_LEVEL <= 0
#define HYLE_LOG_TRACE(...)                                                    \
  HYLE_LOG_AT_(::hyle::obs::LogLevel::Trace, __VA_ARGS__)
#define HYLE_LOG_TRACE_IF(condition, ...)                                      \
  HYLE_LOG_IF_AT_(::hyle::obs::LogLevel::Trace, condition, __VA_ARGS__)
#define HYLE_LOG_TRACE_EVERY_N(n, ...)                                         \
  HYLE_LOG_EVERY_N_AT_(::hyle::obs::LogLevel::Trace, n, __VA_ARGS__)
#define HYLE_LOG_TRACE_EVERY_N_EXACT(n, ...)                                   \
  HYLE_LOG_EVERY_N_EXACT_AT_(::hyle::obs::LogLevel::Trace, n, __VA_ARGS__)
#define HYLE_LOG_TRACE_FIRST_N(n, ...)                                         \
  HYLE_LOG_FIRST_N_AT_(::hyle::obs::LogLevel::Trace, n, __VA_ARGS__)
#define HYLE_LOG_TRACE_EVERY_POW_2(...)                                        \
  HYLE_LOG_EVERY_POW_2_AT_(::hyle::obs::LogLevel::Trace, __VA_ARGS__)
#define HYLE_LOG_TRACE_EVERY(interval, ...)                                    \
  HYLE_LOG_EVERY_AT_(::hyle::obs::LogLevel::Trace, interval, __VA_ARGS__)
#else
#define HYLE_LOG_TRACE(...)                                                    \
  do {                                                                         \
  } while (false)
#define HYLE_LOG_TRACE_IF(...) HYLE_LOG_TRACE()
#define HYLE_LOG_TRACE_EVERY_N(...) HYLE_LOG_TRACE()
#define HYLE_LOG_TRACE_EVERY_N_EXACT(...) HYLE_LOG_TRACE()
#define HYLE_LOG_TRACE_FIRST_N(...) HYLE_LOG_TRACE()
#define HYLE_LOG_TRACE_EVERY_POW_2(...) HYLE_LOG_TRACE()
#define HYLE_LOG_TRACE_EVERY(...) HYLE_LOG_TRACE()
#endif
#if HYLE_OBS_COMPILED_MIN_LEVEL <= 1
#define HYLE_LOG_DEBUG(...)                                                    \
  HYLE_LOG_AT_(::hyle::obs::LogLevel::Debug, __VA_ARGS__)
#define HYLE_LOG_DEBUG_IF(condition, ...)                                      \
  HYLE_LOG_IF_AT_(::hyle::obs::LogLevel::Debug, condition, __VA_ARGS__)
#define HYLE_LOG_DEBUG_EVERY_N(n, ...)                                         \
  HYLE_LOG_EVERY_N_AT_(::hyle::obs::LogLevel::Debug, n, __VA_ARGS__)
#define HYLE_LOG_DEBUG_EVERY_N_EXACT(n, ...)                                   \
  HYLE_LOG_EVERY_N_EXACT_AT_(::hyle::obs::LogLevel::Debug, n, __VA_ARGS__)
#define HYLE_LOG_DEBUG_FIRST_N(n, ...)                                         \
  HYLE_LOG_FIRST_N_AT_(::hyle::obs::LogLevel::Debug, n, __VA_ARGS__)
#define HYLE_LOG_DEBUG_EVERY_POW_2(...)                                        \
  HYLE_LOG_EVERY_POW_2_AT_(::hyle::obs::LogLevel::Debug, __VA_ARGS__)
#define HYLE_LOG_DEBUG_EVERY(interval, ...)                                    \
  HYLE_LOG_EVERY_AT_(::hyle::obs::LogLevel::Debug, interval, __VA_ARGS__)
#else
#define HYLE_LOG_DEBUG(...)                                                    \
  do {                                                                         \
  } while (false)
#define HYLE_LOG_DEBUG_IF(...) HYLE_LOG_DEBUG()
#define HYLE_LOG_DEBUG_EVERY_N(...) HYLE_LOG_DEBUG()
#define HYLE_LOG_DEBUG_EVERY_N_EXACT(...) HYLE_LOG_DEBUG()
#define HYLE_LOG_DEBUG_FIRST_N(...) HYLE_LOG_DEBUG()
#define HYLE_LOG_DEBUG_EVERY_POW_2(...) HYLE_LOG_DEBUG()
#define HYLE_LOG_DEBUG_EVERY(...) HYLE_LOG_DEBUG()
#endif
#if HYLE_OBS_COMPILED_MIN_LEVEL <= 2
#define HYLE_LOG_INFO(...)                                                     \
  HYLE_LOG_AT_(::hyle::obs::LogLevel::Info, __VA_ARGS__)
#define HYLE_LOG_INFO_IF(condition, ...)                                       \
  HYLE_LOG_IF_AT_(::hyle::obs::LogLevel::Info, condition, __VA_ARGS__)
#define HYLE_LOG_INFO_EVERY_N(n, ...)                                          \
  HYLE_LOG_EVERY_N_AT_(::hyle::obs::LogLevel::Info, n, __VA_ARGS__)
#define HYLE_LOG_INFO_EVERY_N_EXACT(n, ...)                                    \
  HYLE_LOG_EVERY_N_EXACT_AT_(::hyle::obs::LogLevel::Info, n, __VA_ARGS__)
#define HYLE_LOG_INFO_FIRST_N(n, ...)                                          \
  HYLE_LOG_FIRST_N_AT_(::hyle::obs::LogLevel::Info, n, __VA_ARGS__)
#define HYLE_LOG_INFO_EVERY_POW_2(...)                                         \
  HYLE_LOG_EVERY_POW_2_AT_(::hyle::obs::LogLevel::Info, __VA_ARGS__)
#define HYLE_LOG_INFO_EVERY(interval, ...)                                     \
  HYLE_LOG_EVERY_AT_(::hyle::obs::LogLevel::Info, interval, __VA_ARGS__)
#else
#define HYLE_LOG_INFO(...)                                                     \
  do {                                                                         \
  } while (false)
#define HYLE_LOG_INFO_IF(...) HYLE_LOG_INFO()
#define HYLE_LOG_INFO_EVERY_N(...) HYLE_LOG_INFO()
#define HYLE_LOG_INFO_EVERY_N_EXACT(...) HYLE_LOG_INFO()
#define HYLE_LOG_INFO_FIRST_N(...) HYLE_LOG_INFO()
#define HYLE_LOG_INFO_EVERY_POW_2(...) HYLE_LOG_INFO()
#define HYLE_LOG_INFO_EVERY(...) HYLE_LOG_INFO()
#endif
#if HYLE_OBS_COMPILED_MIN_LEVEL <= 3
#define HYLE_LOG_WARN(...)                                                     \
  HYLE_LOG_AT_(::hyle::obs::LogLevel::Warn, __VA_ARGS__)
#define HYLE_LOG_WARN_IF(condition, ...)                                       \
  HYLE_LOG_IF_AT_(::hyle::obs::LogLevel::Warn, condition, __VA_ARGS__)
#define HYLE_LOG_WARN_EVERY_N(n, ...)                                          \
  HYLE_LOG_EVERY_N_AT_(::hyle::obs::LogLevel::Warn, n, __VA_ARGS__)
#define HYLE_LOG_WARN_EVERY_N_EXACT(n, ...)                                    \
  HYLE_LOG_EVERY_N_EXACT_AT_(::hyle::obs::LogLevel::Warn, n, __VA_ARGS__)
#define HYLE_LOG_WARN_FIRST_N(n, ...)                                          \
  HYLE_LOG_FIRST_N_AT_(::hyle::obs::LogLevel::Warn, n, __VA_ARGS__)
#define HYLE_LOG_WARN_EVERY_POW_2(...)                                         \
  HYLE_LOG_EVERY_POW_2_AT_(::hyle::obs::LogLevel::Warn, __VA_ARGS__)
#define HYLE_LOG_WARN_EVERY(interval, ...)                                     \
  HYLE_LOG_EVERY_AT_(::hyle::obs::LogLevel::Warn, interval, __VA_ARGS__)
#else
#define HYLE_LOG_WARN(...)                                                     \
  do {                                                                         \
  } while (false)
#define HYLE_LOG_WARN_IF(...) HYLE_LOG_WARN()
#define HYLE_LOG_WARN_EVERY_N(...) HYLE_LOG_WARN()
#define HYLE_LOG_WARN_EVERY_N_EXACT(...) HYLE_LOG_WARN()
#define HYLE_LOG_WARN_FIRST_N(...) HYLE_LOG_WARN()
#define HYLE_LOG_WARN_EVERY_POW_2(...) HYLE_LOG_WARN()
#define HYLE_LOG_WARN_EVERY(...) HYLE_LOG_WARN()
#endif
#if HYLE_OBS_COMPILED_MIN_LEVEL <= 4
#define HYLE_LOG_ERROR(...)                                                    \
  HYLE_LOG_AT_(::hyle::obs::LogLevel::Error, __VA_ARGS__)
#define HYLE_LOG_ERROR_IF(condition, ...)                                      \
  HYLE_LOG_IF_AT_(::hyle::obs::LogLevel::Error, condition, __VA_ARGS__)
#define HYLE_LOG_ERROR_EVERY_N(n, ...)                                         \
  HYLE_LOG_EVERY_N_AT_(::hyle::obs::LogLevel::Error, n, __VA_ARGS__)
#define HYLE_LOG_ERROR_EVERY_N_EXACT(n, ...)                                   \
  HYLE_LOG_EVERY_N_EXACT_AT_(::hyle::obs::LogLevel::Error, n, __VA_ARGS__)
#define HYLE_LOG_ERROR_FIRST_N(n, ...)                                         \
  HYLE_LOG_FIRST_N_AT_(::hyle::obs::LogLevel::Error, n, __VA_ARGS__)
#define HYLE_LOG_ERROR_EVERY_POW_2(...)                                        \
  HYLE_LOG_EVERY_POW_2_AT_(::hyle::obs::LogLevel::Error, __VA_ARGS__)
#define HYLE_LOG_ERROR_EVERY(interval, ...)                                    \
  HYLE_LOG_EVERY_AT_(::hyle::obs::LogLevel::Error, interval, __VA_ARGS__)
#else
#define HYLE_LOG_ERROR(...)                                                    \
  do {                                                                         \
  } while (false)
#define HYLE_LOG_ERROR_IF(...) HYLE_LOG_ERROR()
#define HYLE_LOG_ERROR_EVERY_N(...) HYLE_LOG_ERROR()
#define HYLE_LOG_ERROR_EVERY_N_EXACT(...) HYLE_LOG_ERROR()
#define HYLE_LOG_ERROR_FIRST_N(...) HYLE_LOG_ERROR()
#define HYLE_LOG_ERROR_EVERY_POW_2(...) HYLE_LOG_ERROR()
#define HYLE_LOG_ERROR_EVERY(...) HYLE_LOG_ERROR()
#endif
